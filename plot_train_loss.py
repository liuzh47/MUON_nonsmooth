import pandas as pd
import matplotlib.pyplot as plt


def load_and_summarize(path, exclude_first_run=True):
    """Load a results CSV, drop the per-run 'eval' summary row, exclude run 0 (which
    includes one-time compile/warmup overhead), and return epoch index, mean +/- std of
    train_loss, and mean total_time_seconds, grouped by epoch."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()       # header has stray whitespace, e.g. "run   "
    df['run'] = df['run'].ffill()              # 'run' is only written on each run's first row
    df = df[df['epoch'] != 'eval'].copy()      # drop the final TTA-eval summary row per run
    df['epoch'] = df['epoch'].astype(int)
    df['run'] = df['run'].astype(int)
    if exclude_first_run:
        df = df[df['run'] != df['run'].min()]

    grouped = df.groupby('epoch')['train_loss']
    mean_loss = grouped.mean().sort_index()
    std_loss = grouped.std().sort_index()
    mean_time = df.groupby('epoch')['total_time_seconds'].mean().sort_index()
    return mean_loss.index, mean_time, mean_loss, std_loss


results = {
    'Muon':           ('muon_results.csv', 'tab:blue'),
    'SGD':            ('sgd_results.csv', 'tab:orange'),  
    'Muon Clip FW':   ('muon_fw_results.csv', 'tab:purple'),
    'Muon Clip FW exact lr':   ('muon_fw_exact_lr_results.csv', 'tab:pink'),
    'Muon Soft Clip':   ('muon_softclip_results.csv', 'tab:olive'),
    'Adam Muon':        ('adam_muon_results.csv', 'tab:green'),
}

#'Muon SVD':       ('muon_svd_results.csv', 'tab:green'),
#'Muon Clip SVD':  ('muon_clip_svd_results.csv', 'tab:red'),
# --- Figure 1: train_loss vs total_time_seconds ---
plt.figure(figsize=(8, 5))
for label, (path, color) in results.items():
    epoch, mean_time, mean_loss, std_loss = load_and_summarize(path)
    plt.plot(mean_time, mean_loss, marker='o', markersize=3, color=color, label=f'{label} (mean)')
    plt.fill_between(mean_time, mean_loss - std_loss, mean_loss + std_loss,
                      alpha=0.25, color=color, label=f'{label} (±1 std)')
plt.xlabel('total_time_seconds (averaged across runs)')
plt.ylabel('train_loss')
plt.title('Train loss vs. training time (mean ± std, excluding each run 0)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('train_loss_vs_time_comparison.png', dpi=150)

# --- Figure 2: train_loss vs epoch ---
plt.figure(figsize=(8, 5))
for label, (path, color) in results.items():
    epoch, mean_time, mean_loss, std_loss = load_and_summarize(path)
    plt.plot(epoch, mean_loss, marker='o', markersize=3, color=color, label=f'{label} (mean)')
    plt.fill_between(epoch, mean_loss - std_loss, mean_loss + std_loss,
                      alpha=0.25, color=color, label=f'{label} (±1 std)')
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('Train loss vs. epoch (mean ± std, excluding each run 0)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('train_loss_vs_epoch_comparison.png', dpi=150)

