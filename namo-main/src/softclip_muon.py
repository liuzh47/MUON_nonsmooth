"""
SoftClip Muon optimizer.

Instead of orthogonalizing (pushing every singular value toward 1 as in vanilla Muon),
this caps each singular value sigma at a threshold D via:

    sigma -> D * sigma / sqrt(D^2 + sigma^2)

computed via a coupled Newton-Schulz iteration for the matrix inverse square root.

Singular values well below D are barely changed; singular values near or above D are
smoothly pulled down toward D.  This is a drop-in replacement for standard Muon with
one extra hyperparameter: clip_D (the soft-clip threshold).

Reference: softclip_muon in the NorMuon / modded-nanogpt codebase.
"""

import math
import torch


# ── Orthogonalization helpers ──────────────────────────────────────────────────


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def zeropower_via_svd(G):
    """
    Compute the zeroth power / orthogonalization of G using exact SVD.
    This produces the exact orthogonal matrix UV^T where USV^T = G is the SVD.
    This is more accurate than Newton-Schulz but potentially slower.
    """
    assert len(G.shape) == 2
    original_dtype = G.dtype

    # Convert to float32 for better numerical stability in SVD
    G_float = G.float()

    # Perform SVD
    U, S, Vt = torch.linalg.svd(G_float, full_matrices=False)

    # Return UV^T (the orthogonal part)
    result = U @ Vt

    # Convert back to original dtype
    return result.to(original_dtype)


# ── Soft-clip spectral map ─────────────────────────────────────────────────────


@torch.compile
def softclip_invsqrt(G, clip_D, steps=8):
    """
    Soft spectral clip via coupled Newton-Schulz inverse square root iteration.

    Given G with SVD = U diag(sigma) V^T, computes:

        U diag( D * sigma / sqrt(D^2 + sigma^2) ) V^T

    This smoothly caps singular values at D without a hard threshold.

    The core idea: form M = G G^T + D^2 I, then compute M^{-1/2} via the coupled
    Newton-Schulz iteration for the matrix inverse square root, and finally multiply
    D * M^{-1/2} @ G.

    Arguments:
        G: 2D gradient / momentum tensor (m x n).
        clip_D: the soft-clip threshold (float > 0).
        steps: number of Newton-Schulz iterations (8 is usually sufficient).

    Returns:
        Tensor of the same shape as G with soft-clipped singular values.
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # X is now always wide (rows <= cols); use X @ X^T (the smaller Gram matrix)
    M = X @ X.T  # (m, m)
    n = M.size(0)
    I = torch.eye(n, device=X.device, dtype=X.dtype)
    M = M + (clip_D * clip_D) * I

    # Normalize so the iteration converges
    alpha = M.norm() + 1e-6
    Y = M / alpha
    Z = I.clone()

    # Coupled Newton-Schulz iteration for M^{-1/2}
    for _ in range(steps):
        ZY = Z @ Y
        T = 0.5 * (3 * I - ZY)
        Y = Y @ T
        Z = T @ Z

    invsqrt = Z / alpha.sqrt()

    # D * M^{-1/2} @ X  =>  sigma -> D * sigma / sqrt(sigma^2 + D^2)
    out = clip_D * (invsqrt @ X)

    if G.size(0) > G.size(1):
        out = out.T
    return out


# ── Optimizer ──────────────────────────────────────────────────────────────────


class SoftClipMuon(torch.optim.Optimizer):
    """
    SoftClip Muon - Muon with soft spectral clipping.

    Like standard Muon, this runs SGD-momentum internally and then replaces each
    2D parameter's update with a spectrally-processed matrix.  Instead of the
    Newton-Schulz orthogonalization (which pushes all singular values toward 1),
    SoftClip Muon applies a smooth cap:

        sigma -> D * sigma / sqrt(D^2 + sigma^2)

    Singular values well below D pass through nearly unchanged; those near or
    above D are softly pulled down toward D.  The parameter `clip_D` controls the
    threshold.  Setting clip_D very large recovers (approximately) vanilla Muon.

    Arguments:
        muon_params: The parameters to be optimized by SoftClip Muon.
        lr: The learning rate. (0.02 is a good default)
        wd: Weight decay coefficient. (0.1 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum. (recommended)
        clip_D: Soft-clip threshold for singular values. (1.0 is a reasonable default)
        ns_steps: Newton-Schulz iterations for the inverse sqrt. (8 is usually enough)
        use_muon_fallback: If True, use standard Muon orthogonalization instead of softclip.
                           Useful for A/B comparison. (default: False)
        use_exact_svd: If True, use exact SVD orthogonalization as the fallback
                       (only applies when use_muon_fallback=True). (default: False)
        muon_ns_steps: Newton-Schulz steps for the Muon fallback. (default: 5)
        adamw_params: The parameters to be optimized by AdamW (embeddings, heads, 1D params).
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        clip_D=1.0,
        ns_steps=8,
        use_muon_fallback=False,
        use_exact_svd=False,
        muon_ns_steps=5,
        adamw_params=None,
        adamw_wd=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        if adamw_wd is None:
            adamw_wd = wd

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            clip_D=clip_D,
            ns_steps=ns_steps,
            use_muon_fallback=use_muon_fallback,
            use_exact_svd=use_exact_svd,
            muon_ns_steps=muon_ns_steps,
            adamw_wd=adamw_wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use SoftClip Muon, and those for AdamW
        if muon_params is not None:
            for p in muon_params:
                assert p.ndim == 2 or p.ndim == 4, p.ndim
                self.state[p]["use_muon"] = True
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #     SoftClip Muon        #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # Momentum update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                if group["nesterov"]:
                    g = g.lerp(buf, momentum)
                else:
                    g = buf

                # Choose spectral processing method
                if group["use_muon_fallback"]:
                    # Standard Muon orthogonalization (for A/B comparison)
                    if group["use_exact_svd"]:
                        u = zeropower_via_svd(g).view_as(p)
                    else:
                        u = zeropower_via_newtonschulz5(g, steps=group["muon_ns_steps"]).view_as(p)
                else:
                    # Soft spectral clip
                    u = softclip_invsqrt(g, clip_D=group["clip_D"], steps=group["ns_steps"]).view_as(p)

                # Scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # Apply weight decay
                p.data.mul_(1 - lr * wd)

                # Apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            adamw_lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - adamw_lr * weight_decay)
                p.data.add_(g, alpha=-adamw_lr / scale)

        return loss
