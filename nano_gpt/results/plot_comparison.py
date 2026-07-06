"""
Plot mean val_loss vs train_time (with ±1 std dev band) for two groups of runs:
  - muon_softclip_{1..5}.csv
  - muon_fw_ls_{1..5}.csv

Both groups share the same `step` grid across all 5 seeds, so we can stack the
per-seed CSVs directly and average row-wise without needing to interpolate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

UPLOAD_DIR = "./"
OUTPUT_PATH = "./val_loss_vs_train_time.png"


def load_group(prefix: str, n_runs: int = 5):
    """Load n_runs CSVs named '{prefix}_{i}.csv' and return mean/std arrays."""
    dfs = [pd.read_csv(f"{UPLOAD_DIR}/{prefix}_{i}.csv") for i in range(1, n_runs + 1)]

    val_loss = np.stack([df["val_loss"].values for df in dfs])        # (n_runs, n_steps)
    train_time = np.stack([df["train_time_ms"].values for df in dfs])  # (n_runs, n_steps)

    mean_loss = val_loss.mean(axis=0)
    std_loss = val_loss.std(axis=0)
    mean_time_min = train_time.mean(axis=0) / 1000 / 60  # ms -> minutes

    return mean_time_min, mean_loss, std_loss


# Define each group: prefix, display label, color
groups = [
    {"prefix": "muon_softclip", "label": "Muon Softclip (n=5)", "color": "tab:blue"},
    {"prefix": "muon_fw_ls", "label": "Muon FW-LS (n=5)", "color": "tab:orange"},
    {"prefix": "adamw", "label": "AdamW (n=5)", "color": "tab:green"},
    {"prefix": "adam", "label": "Adam (n=5)", "color": "tab:red"},
    {"prefix": "muon", "label": "Muon (n=5)", "color": "tab:brown"},
]

fig, ax = plt.subplots(figsize=(9, 6))

for g in groups:
    mean_time, mean_loss, std_loss = load_group(g["prefix"])

    ax.plot(mean_time, mean_loss, color=g["color"], linewidth=2, label=f"Mean — {g['label']}")
    ax.fill_between(
        mean_time, mean_loss - std_loss, mean_loss + std_loss,
        color=g["color"], alpha=0.25, label=f"±1 std dev — {g['label']}"
    )

ax.set_yscale("log")
ax.set_xlabel("Train time (minutes)", fontsize=12)
ax.set_ylabel("Validation loss (log scale)", fontsize=12)
ax.set_title("Validation Loss vs Train Time (mean ± 1 std dev, n=5 per group)", fontsize=13)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved plot to {OUTPUT_PATH}")
