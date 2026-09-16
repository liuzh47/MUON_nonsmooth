import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?|nan)"
)

PLOT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PLOT_DIR / "fineweb" / "small"

methods = [
    ("muon",     "Muon",      PLOT_DIR / "muon_1gpu_0.2.log"),
    ("muonclip", "MuonClip",  PLOT_DIR / "muonclip_1gpu_0.2.log"),
    ("spectra",  "SPECTRA",   PLOT_DIR / "spectra_1gpu_0.2.log"),
    ("softclip", "Soft Musec",PLOT_DIR / "softclip_1gpu_0.2.log"),
]

def parse_val_loss(path):
    steps, losses = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                val = m.group(3)
                loss = float("nan") if val.lower() == "nan" else float(val)
                steps.append(step)
                losses.append(loss)
    return steps, losses

fig, ax = plt.subplots(figsize=(8, 5.5))
markers = ["o", "s", "^", "D"]
colors = plt.get_cmap("tab10").colors

for i, (key, label, path) in enumerate(methods):
    steps, losses = parse_val_loss(path)
    ax.plot(
        steps, losses,
        marker=markers[i], color=colors[i],
        linewidth=2, markersize=5, markevery=5,
        label=label,
    )

ax.set_xlabel("Steps", fontsize=28)
ax.set_ylabel("Validation loss", fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=28)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=28, loc="best")

plt.tight_layout()
output_path = PLOT_DIR / "val_loss_vs_steps.pdf"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
