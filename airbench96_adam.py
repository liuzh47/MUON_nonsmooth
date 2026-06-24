# A variant of airbench optimized for time-to-96%.
# 27.3s runtime on an A100; 3.1 PFLOPs.
# Evidence: 96.00 average accuracy in n=200 runs.
#
# We recorded the above runtime on an NVIDIA A100-SXM4-40GB with the following nvidia-smi:
# NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7
# torch.__version__ == '2.4.0+cu121'

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

# Optimizer selection:
#   'muon_sgd'        - original: SGD for biases/linear params + Muon for conv filters
#   'adam_muon'       - Adam for biases/linear params + Muon for conv filters
#   'adamw_muon'      - AdamW for biases/linear params + Muon for conv filters
#   'adam_muon_soft'  - Adam for biases/linear params + MuonSoftclip for conv filters
#   'adamw_muon_soft' - AdamW for biases/linear params + MuonSoftclip for conv filters
#   'adam_muon_fw'    - Adam for biases/linear params + MuonFrankWolfe for conv filters
#   'adamw_muon_fw'   - AdamW for biases/linear params + MuonFrankWolfe for conv filters
#   'adam'            - Adam for all params (no Muon)
#   'adamw'           - AdamW for all params (no Muon)
OPTIMIZER = 'adam_muon_fw'

hyp = {
    'opt': {
        'train_epochs': 45.0,
        'batch_size': 1024,
        'batch_size_masked': 512,
        # --- SGD + Muon hyperparameters (used when OPTIMIZER == 'muon_sgd') ---
        'lr': 9.0,               # learning rate per 1024 examples (for the SGD-handled params)
        'momentum': 0.85,
        'weight_decay': 0.012,   # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,     # scales up learning rate (but not weight decay) for BatchNorm biases
        # --- Muon hyperparameters (used when OPTIMIZER contains 'muon') ---
        'muon_lr': 0.24,         # learning rate for the Muon-handled conv filter weights
        'muon_momentum': 0.6,
        'muon_softclip_D': 1.0,  # soft-clip singular value ceiling (MuonSoftclip only)
        'muon_softclip_steps': 5,# Newton-Schulz iterations for softclip (MuonSoftclip only)
        'muon_fw_D': 1.0,        # spectral norm constraint radius (MuonFrankWolfe only)
        'muon_fw_steps': 2,     # Frank-Wolfe outer iterations (MuonFrankWolfe only)
        'muon_fw_ns_steps': 10,  # Newton-Schulz inner iterations for matrix sign (MuonFrankWolfe only)
        # --- Adam / AdamW hyperparameters (used when OPTIMIZER contains 'adam') ---
        'adam_lr': 1e-3,         # base learning rate for non-norm params
        'adam_lr_biases': 1e-2,  # learning rate for BatchNorm bias params (usually higher)
        'adam_betas': (0.9, 0.999),
        'adam_eps': 1e-6,        # fp16-safe: default 1e-8 underflows in fp16 (min ~6e-8) → NaN
        'adam_weight_decay': 0.01,  # only meaningful for AdamW (decoupled weight decay)
        # --- shared ---
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3, # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 12,
    },
    'proxy': {
        'widths': {
            'block1': 32,
            'block2': 64,
            'block3': 64,
        },
        'depth': 2,
        'scaling_factor': 1/9,
    },
    'net': {
        'widths': {
            'block1': 128,
            'block2': 384,
            'block3': 512,
        },
        'depth': 3,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

def set_random_state(seed, state):
    if seed is None:
        # If we don't get a data seed, then make sure to randomize the state using independent generator, since
        # it might have already been set by the model seed.
        import random
        torch.manual_seed(random.randint(0, 2**63))
    else:
        seed1 = 1000 * seed + state # just don't do more than 1000 epochs or else there will be overlap
        torch.manual_seed(seed1)

class InfiniteCifarLoader:
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    The purpose is to support cross-epoch batches (in case the batch size does not divide the number of train examples),
    and support stochastic iteration counts in order to preserve perfect linearity/independence.
    """

    def __init__(self, path, train=True, batch_size=500, aug=None, altflip=True, subset_mask=None, aug_seed=None, order_seed=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location='cuda')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.altflip = altflip
        self.subset_mask = subset_mask if subset_mask is not None else torch.tensor([True]*len(self.images)).cuda()
        self.train = train
        self.aug_seed = aug_seed
        self.order_seed = order_seed

    def __iter__(self):

        # Preprocess
        images0 = self.normalize(self.images)
        # Pre-randomly flip images in order to do alternating flip later.
        if self.aug.get('flip', False) and self.altflip:
            set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        # Pre-pad images to save time when doing random translation
        pad = self.aug.get('translate', 0)
        if pad > 0:
            images0 = F.pad(images0, (pad,)*4, 'reflect')
        labels0 = self.labels

        # Iterate forever
        epoch = 0
        batch_size = self.batch_size

        # In the below while-loop, we will repeatedly build a batch and then yield it.
        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
        batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:

            # Assume we need to generate more data to add to the batch.
            assert len(batch_images) < batch_size

            # If we have already exhausted the current epoch, then begin a new one.
            if current_pointer >= num_examples:
                # If we already reached the end of the last epoch then we need to generate
                # a new augmented epoch of data (using random crop and alternating flip).
                epoch += 1

                set_random_state(self.aug_seed, epoch)
                if pad > 0:
                    images1 = batch_crop(images0, 32)
                if self.aug.get('flip', False):
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)
                if self.aug.get('cutout', 0) > 0:
                    images1 = batch_cutout(images1, self.aug['cutout'])

                set_random_state(self.order_seed, epoch)
                indices = (torch.randperm if self.train else torch.arange)(len(self.images), device=images0.device)

                # The effect of doing subsetting in this manner is as follows. If the permutation wants to show us
                # our four examples in order [3, 2, 0, 1], and the subset mask is [True, False, True, False],
                # then we will be shown the examples [2, 0]. It is the subset of the ordering.
                # The purpose is to minimize the interaction between the subset mask and the randomness.
                # So that the subset causes not only a subset of the total examples to be shown, but also a subset of
                # the actual sequence of examples which is shown during training.
                indices_subset = indices[self.subset_mask[indices]]
                current_pointer = 0

            # Now we are sure to have more data in this epoch remaining.
            # This epoch's remaining data is given by (images1[current_pointer:], labels0[current_pointer:])
            # We add more data to the batch, up to whatever is needed to make a full batch (but it might not be enough).
            remaining_size = batch_size - len(batch_images)

            # Given that we want `remaining_size` more training examples, we construct them here, using
            # the remaining available examples in the epoch.

            extra_indices = indices_subset[current_pointer:current_pointer+remaining_size]
            extra_images = images1[extra_indices]
            extra_labels = labels0[extra_indices]
            current_pointer += remaining_size
            batch_indices = torch.cat([batch_indices, extra_indices])
            batch_images = torch.cat([batch_images, extra_images])
            batch_labels = torch.cat([batch_labels, extra_labels])

            # If we have a full batch ready then yield it and reset.
            if len(batch_images) == batch_size:
                assert len(batch_images) == len(batch_labels)
                yield (batch_indices, batch_images, batch_labels)
                batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
                batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
                batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0
            x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net(hyp):
    widths = hyp['widths']
    scaling_factor = hyp['scaling_factor']
    depth = hyp['depth']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], depth),
        ConvGroup(widths['block1'], widths['block2'], depth),
        ConvGroup(widths['block2'], widths['block3'], depth),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(scaling_factor),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#               Muon Optimizer              #
############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Computes O ~= UV^T where G = USV^T, via Newton-Schulz iteration. Quintic-degree iteration
    designed for fast convergence in a few steps; coefficients are tuned, not exact, so this
    does NOT converge to UV^T for huge numbers of steps, only for a handful (intended use).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon: orthogonalizes the (momentum-accumulated) gradient of each parameter via Newton-Schulz
    iteration before taking a step. Intended only for the 2D-ish "hidden" weight matrices of a
    network (e.g. conv filters reshaped to 2D, linear layer weights) -- NOT for biases, norm
    parameters, embeddings, or the very first/last layers of a network, which should stay on a
    standard optimizer (SGD/AdamW). lr here lives on a different natural scale than a typical SGD
    lr, since the orthogonalized update has singular values ~O(1) regardless of the gradient's
    own magnitude -- so muon_lr generally needs its own separate tuning, not the SGD lr value.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=5).view(g.shape)
                p.data.add_(update, alpha=-lr)

@torch.compile
def softclip_via_newtonschulz(G, D=1.0, steps=5, eps=1e-7):
    """
    Soft-clips singular values of G to ≤ D via matrix-inverse-square-root.
    Unlike zeropower_via_newtonschulz5 (which maps all singular values to ~1),
    this preserves the relative scale of singular values while capping their
    maximum at D — hence "soft clip" rather than full orthogonalisation.
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    if X.size(0) <= X.size(1):
        M = X @ X.T        # (rows × rows) — smaller Gram matrix
        right_side = False
    else:
        M = X.T @ X        # (cols × cols) — smaller Gram matrix
        right_side = True
    I = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    M = M + D * D * I
    alpha = M.norm() + eps
    Y = M / alpha
    Z = I
    for _ in range(steps):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    invsqrt = Z / alpha.sqrt()
    if right_side:
        out = D * (X @ invsqrt)
    else:
        out = D * (invsqrt @ X)
    return out.to(G.dtype)


class MuonSoftclip(torch.optim.Optimizer):
    """
    Muon variant that replaces the Newton-Schulz orthogonalisation step with
    softclip_via_newtonschulz, which soft-clips singular values to ≤ D rather
    than mapping them all to ~1. This preserves gradient magnitude ratios while
    still bounding the maximum update scale, potentially giving smoother training.
    All other mechanics (momentum buffer, Nesterov, per-step lr) are identical
    to the standard Muon optimizer.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 softclip_D=1.0, softclip_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        softclip_D=softclip_D, softclip_steps=softclip_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            D = group['softclip_D']
            sc_steps = group['softclip_steps']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = softclip_via_newtonschulz(
                    g.reshape(len(g), -1), D=D, steps=sc_steps
                ).view(g.shape)
                p.data.add_(update, alpha=-lr)



@torch.compile
def _msign_exact_via_newtonschulz(G, steps=10, eps=1e-7):
    """
    Computes the matrix sign M ~= UV^T where G = USV^T via a cubic Newton-Schulz
    iteration. Unlike the quintic variant used by standard Muon, this uses the
    simpler X ← 1.5X − 0.5(XX^T)X recurrence, which is sufficient for the
    Frank-Wolfe linear-minimisation oracle that only needs the sign direction.
    """
    X = G.bfloat16()
    X /= (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 1.5 * X - 0.5 * (A @ X)
    if transposed:
        X = X.T
    return X


@torch.compile
def clip_via_frankwolfe(G, D, fw_steps=20, ns_steps=10, eps=1e-12):
    """
    Projects G onto the spectral-norm ball {X : σ_max(X) ≤ D} via Frank-Wolfe.

    At each iteration the linear minimisation oracle is solved exactly:
        s_k = argmin_{‖S‖_* ≤ D·mn}  〈S, X_k − G〉  =  −D · msign(X_k − G)
    where msign(·) = UV^T is the matrix sign computed via Newton-Schulz.
    The step size α_k is chosen by exact line-search on the quadratic surrogate.

    Frank-Wolfe avoids expensive SVD while staying within the spectral-norm
    constraint, making it attractive as a drop-in replacement for the
    orthogonalisation step in Muon.
    """
    assert len(G.shape) == 2
    A = G.bfloat16()
    X = torch.zeros_like(A)
    for k in range(fw_steps):
        Y = -D * _msign_exact_via_newtonschulz(X - A, steps=ns_steps)
        Z = Y - X
        denom = (Z * Z).sum()
        alpha = (((A - X) * Z).sum() / (denom + eps)).clamp(0.0, 1.0)
        X = X + alpha * Z
    return X.to(G.dtype)


class MuonFrankWolfe(torch.optim.Optimizer):
    """
    Muon variant that replaces the Newton-Schulz orthogonalisation step with a
    Frank-Wolfe projection onto the spectral-norm ball of radius D.

    Instead of mapping all singular values to ~1 (standard Muon) or soft-clipping
    them (MuonSoftclip), this projects the momentum-accumulated gradient onto
    {X : σ_max(X) ≤ D} by solving a sequence of cheap linear minimisation
    oracles, each requiring only a matrix-sign computation via Newton-Schulz.

    Hyperparameters:
      fw_D        – spectral norm constraint radius (analogous to muon_lr scale)
      fw_steps    – Frank-Wolfe outer iterations (more → closer to true projection)
      fw_ns_steps – Newton-Schulz inner iterations for the matrix-sign oracle
    All other mechanics (momentum buffer, Nesterov, per-step lr) are identical
    to the standard Muon optimizer.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 fw_D=1.0, fw_steps=20, fw_ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        fw_D=fw_D, fw_steps=fw_steps, fw_ns_steps=fw_ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr        = group['lr']
            momentum  = group['momentum']
            D         = group['fw_D']
            fw_steps  = group['fw_steps']
            ns_steps  = group['fw_ns_steps']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = clip_via_frankwolfe(
                    g.reshape(len(g), -1), D=D,
                    fw_steps=fw_steps, ns_steps=ns_steps,
                ).view(g.shape)
                p.data.add_(update, alpha=-lr)


############################################
#          fp32-grad Adam wrapper          #
############################################

class Fp32GradAdam:
    """
    Wrapper around Adam/AdamW that performs the update in fp32 to avoid NaN
    from fp16 second-moment underflow, while keeping model params in fp16 so
    torch.compile sees a consistent dtype throughout.

    Strategy (fp32 master-weight lite):
      - Maintain a fp32 shadow copy of each managed parameter.
      - On step(): copy fp16 grad → fp32 grad on the shadow param, run Adam
        on the fp32 shadow, then copy the updated fp32 value back to the
        original fp16 param.
    PyTorch requires p.grad.dtype == p.dtype, so we never touch p.grad on the
    original fp16 tensor — all gradient work happens on the fp32 shadow.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups  # kept in sync for schedulers etc.
        # Build fp32 shadow params and re-point the optimizer's param groups at them.
        self._fp16_params = []   # original fp16 tensors (model params)
        self._fp32_shadows = []  # fp32 copies owned by this wrapper
        for group in self.optimizer.param_groups:
            fp32_group = []
            for p in group['params']:
                shadow = p.detach().float().requires_grad_(p.requires_grad)
                fp32_group.append(shadow)
                self._fp16_params.append(p)
                self._fp32_shadows.append(shadow)
            group['params'] = fp32_group

    def zero_grad(self, set_to_none=False):
        # Zero grads on the original fp16 params (where backward writes them).
        for p in self._fp16_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def step(self):
        # 1. Copy fp16 grads → fp32 shadow grads.
        for p16, p32 in zip(self._fp16_params, self._fp32_shadows):
            if p16.grad is not None:
                p32.grad = p16.grad.float()
            else:
                p32.grad = None
        # 2. Run Adam entirely in fp32.
        self.optimizer.step()
        # 3. Write fp32 result back into the fp16 model param.
        for p16, p32 in zip(self._fp16_params, self._fp32_shadows):
            p16.data.copy_(p32.data)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry, log_writer):
    formatted = []
    all_vars = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        all_vars.append(var)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    log_writer.writerow(all_vars)
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#                Training                  #
############################################

def train_proxy(hyp, model, data_seed):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model)
    print('Proxy parameters:', sum(p.numel() for p in model.parameters()))
    current_steps = 0

    if OPTIMIZER == 'muon_sgd':
        norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
        param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                         dict(params=other_params, lr=lr, weight_decay=wd/lr)]
        optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
    elif OPTIMIZER in ('adam', 'adamw', 'adam_muon', 'adamw_muon',
                       'adam_muon_soft', 'adamw_muon_soft',
                       'adam_muon_fw',   'adamw_muon_fw'):
        adam_lr        = hyp['opt']['adam_lr']
        adam_lr_biases = hyp['opt']['adam_lr_biases']
        adam_betas     = hyp['opt']['adam_betas']
        adam_eps       = hyp['opt']['adam_eps']
        adam_wd        = hyp['opt']['adam_weight_decay'] if 'adamw' in OPTIMIZER else 0.0
        OptCls = torch.optim.AdamW if 'adamw' in OPTIMIZER else torch.optim.Adam
        norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
        param_configs = [dict(params=norm_biases,  lr=adam_lr_biases, weight_decay=0.0),
                         dict(params=other_params, lr=adam_lr,        weight_decay=adam_wd)]
        # Wrap in Fp32GradAdam so gradients are upcast to fp32 before each step.
        # This keeps model params in fp16 (torch.compile stays happy) while
        # preventing fp16 underflow in Adam's second-moment accumulator.
        optimizer = Fp32GradAdam(OptCls(param_configs, betas=adam_betas, eps=adam_eps))
    else:
        raise ValueError(f"Unknown OPTIMIZER: {OPTIMIZER!r}. Choose 'muon_sgd', 'adam_muon', 'adamw_muon', 'adam_muon_soft', 'adamw_muon_soft', 'adam_muon_fw', 'adamw_muon_fw', 'adam', or 'adamw'.")

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    # LambdaLR needs a real torch.optim.Optimizer; unwrap Fp32GradAdam if needed.
    inner_opt = optimizer.optimizer if isinstance(optimizer, Fp32GradAdam) else optimizer
    scheduler = torch.optim.lr_scheduler.LambdaLR(inner_opt, get_lr)

    # Initialize the whitening layer using training images
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model._orig_mod[0], train_images)

    masks = []

    for indices, inputs, labels in train_loader:

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            model.train()

        # Skip every other backward pass
        if current_steps % 4 == 0:
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            mask = torch.zeros(len(inputs)).cuda().bool()
            mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
            masks.append(mask)
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                mask = torch.zeros(len(inputs)).cuda().bool()
                mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
                masks.append(mask)

        scheduler.step()

        current_steps += 1
        if current_steps == total_train_steps:
            break

    return masks

def main(run, hyp, model_proxy, model_trainbias, model_freezebias, writer):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']
    muon_lr = hyp['opt']['muon_lr']

    set_random_state(None, 0)
    import random
    data_seed = random.randint(0, 2**50)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = InfiniteCifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model_trainbias)
    print('Main model parameters:', sum(p.numel() for p in model_trainbias.parameters()))
    current_steps = 0

    def make_optimizers(model):
        """
        Build the optimizer list for a given model according to the global OPTIMIZER flag.

        Returns a list of optimizers so the rest of the training loop (which calls
        `for opt in optimizers: opt.step()`) works unchanged.

        Param split for *_muon* modes:
          - norm params      → Adam/AdamW
          - 4-D conv weights → Muon / MuonSoftclip / MuonFrankWolfe
          - other params     → Adam/AdamW
        """
        if OPTIMIZER == 'muon_sgd':
            norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k]
            muon_params  = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
            other_params = [p for k, p in model.named_parameters()
                            if 'norm' not in k and len(p.shape) != 4 and p.requires_grad]
            param_configs = [dict(params=norm_biases,  lr=lr_biases, weight_decay=wd/lr_biases),
                             dict(params=other_params, lr=lr,        weight_decay=wd/lr)]
            sgd  = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
            muon = Muon(muon_params, lr=muon_lr, momentum=hyp['opt']['muon_momentum'], nesterov=True)
            return [sgd, muon]
        elif OPTIMIZER in ('adam_muon', 'adamw_muon',
                           'adam_muon_soft', 'adamw_muon_soft',
                           'adam_muon_fw',   'adamw_muon_fw'):
            adam_lr        = hyp['opt']['adam_lr']
            adam_lr_biases = hyp['opt']['adam_lr_biases']
            adam_betas     = hyp['opt']['adam_betas']
            adam_eps       = hyp['opt']['adam_eps']
            adam_wd        = hyp['opt']['adam_weight_decay'] if 'adamw' in OPTIMIZER else 0.0
            OptCls = torch.optim.AdamW if 'adamw' in OPTIMIZER else torch.optim.Adam
            norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
            muon_params  = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
            other_params = [p for k, p in model.named_parameters()
                            if 'norm' not in k and len(p.shape) != 4 and p.requires_grad]
            param_configs = [dict(params=norm_biases,  lr=adam_lr_biases, weight_decay=0.0),
                             dict(params=other_params, lr=adam_lr,        weight_decay=adam_wd)]
            adam = Fp32GradAdam(OptCls(param_configs, betas=adam_betas, eps=adam_eps))
            if 'soft' in OPTIMIZER:
                muon = MuonSoftclip(muon_params, lr=muon_lr,
                                    momentum=hyp['opt']['muon_momentum'], nesterov=True,
                                    softclip_D=hyp['opt']['muon_softclip_D'],
                                    softclip_steps=hyp['opt']['muon_softclip_steps'])
            elif '_fw' in OPTIMIZER:
                muon = MuonFrankWolfe(muon_params, lr=muon_lr,
                                      momentum=hyp['opt']['muon_momentum'], nesterov=True,
                                      fw_D=hyp['opt']['muon_fw_D'],
                                      fw_steps=hyp['opt']['muon_fw_steps'],
                                      fw_ns_steps=hyp['opt']['muon_fw_ns_steps'])
            else:
                muon = Muon(muon_params, lr=muon_lr, momentum=hyp['opt']['muon_momentum'], nesterov=True)
            return [adam, muon]
        elif OPTIMIZER in ('adam', 'adamw'):
            adam_lr        = hyp['opt']['adam_lr']
            adam_lr_biases = hyp['opt']['adam_lr_biases']
            adam_betas     = hyp['opt']['adam_betas']
            adam_eps       = hyp['opt']['adam_eps']
            adam_wd        = hyp['opt']['adam_weight_decay'] if OPTIMIZER == 'adamw' else 0.0
            OptCls = torch.optim.AdamW if OPTIMIZER == 'adamw' else torch.optim.Adam
            norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
            other_params = [p for k, p in model.named_parameters()
                            if 'norm' not in k and p.requires_grad]
            param_configs = [dict(params=norm_biases,  lr=adam_lr_biases, weight_decay=0.0),
                             dict(params=other_params, lr=adam_lr,        weight_decay=adam_wd)]
            return [Fp32GradAdam(OptCls(param_configs, betas=adam_betas, eps=adam_eps))]
        else:
            raise ValueError(f"Unknown OPTIMIZER: {OPTIMIZER!r}. "
                             "Choose 'muon_sgd', 'adam_muon', 'adamw_muon', "
                             "'adam_muon_soft', 'adamw_muon_soft', "
                             "'adam_muon_fw', 'adamw_muon_fw', 'adam', or 'adamw'.")

    optimizers_trainbias  = make_optimizers(model_trainbias)
    optimizers_freezebias = make_optimizers(model_freezebias)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    def unwrap(opt):
        # LambdaLR needs a real torch.optim.Optimizer; unwrap Fp32GradAdam if needed.
        return opt.optimizer if isinstance(opt, Fp32GradAdam) else opt
    schedulers_trainbias  = [torch.optim.lr_scheduler.LambdaLR(unwrap(opt), get_lr) for opt in optimizers_trainbias]
    schedulers_freezebias = [torch.optim.lr_scheduler.LambdaLR(unwrap(opt), get_lr) for opt in optimizers_freezebias]

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model_trainbias)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Do a small proxy run to collect masks for use in fullsize run
    print('Training small proxy...')
    starter.record()
    masks = iter(train_proxy(hyp, model_proxy, data_seed))
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for indices, inputs, labels in train_loader:

        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if current_steps == 0:
            model = model_trainbias
            optimizers = optimizers_trainbias
            schedulers = schedulers_trainbias
        elif epoch == hyp['opt']['whiten_bias_epochs'] * steps_per_epoch:
            model = model_freezebias
            optimizers = optimizers_freezebias
            schedulers = schedulers_freezebias
            model.load_state_dict(model_trainbias.state_dict())
            for opt, opt_trainbias in zip(optimizers, optimizers_trainbias):
                opt.load_state_dict(opt_trainbias.state_dict())
            for sch, sch_trainbias in zip(schedulers, schedulers_trainbias):
                sch.load_state_dict(sch_trainbias.state_dict())

        ####################
        #     Training     #
        ####################

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            starter.record()
            model.train()

        mask = next(masks)
        inputs = inputs[mask]
        labels = labels[mask]
        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()

        model.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        for sch in schedulers:
            sch.step()

        current_steps += 1
        if current_steps % 5 == 0:
            lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        if current_steps == total_train_steps:
            if lookahead_state is not None:
                lookahead_state.update(model, decay=1.0)

        if (current_steps % steps_per_epoch == 0) or (current_steps == total_train_steps):
            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)

            ####################
            #    Evaluation    #
            ####################

            # Save the accuracy and loss from the last training batch of the epoch
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)
            print_training_details(locals(), is_final_entry=False, log_writer=writer)
            run = None # Only print the run number once

        if current_steps == total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True, log_writer=writer)
    
    return tta_val_acc

if __name__ == "__main__":
    import csv

    logfile = open(OPTIMIZER+"_results.csv", "w", newline="")
    writer = csv.writer(logfile)

    writer.writerow(['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds'])
    
    with open(sys.argv[0]) as f:
        code = f.read()

    model_proxy = make_net(hyp['proxy'])
    model_proxy[0].bias.requires_grad = False
    model_trainbias = make_net(hyp['net'])
    model_freezebias = make_net(hyp['net'])
    model_freezebias[0].bias.requires_grad = False
    model_proxy = torch.compile(model_proxy, mode='max-autotune')
    model_trainbias = torch.compile(model_trainbias, mode='max-autotune')
    model_freezebias = torch.compile(model_freezebias, mode='max-autotune')

    print_columns(logging_columns_list, is_head=True)
    accs = torch.tensor([main(run, hyp, model_proxy, model_trainbias, model_freezebias, writer)
                         for run in range(30)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))
    logfile.close()


