import re
import matplotlib.pyplot as plt
import numpy as np

VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?|nan)"
)

methods = [
    ("muon",     "Muon",      "/mnt/user-data/uploads/muon_1gpu_0_2.log"),
    ("muonclip", "MuonClip",  "/mnt/user-data/uploads/muonclip_1gpu_0_2.log"),
    ("spectra",  "SPECTRA",   "/mnt/user-data/uploads/spectra_1gpu_0_2.log"),
    ("softclip", "Soft Musec","/mnt/user-data/uploads/softclip_1gpu_0_2.log"),
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

ax.set_xlabel("Steps", fontsize=18)
ax.set_ylabel("Validation loss", fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=14, loc="best")

plt.tight_layout()
output_path = "/mnt/user-data/outputs/val_loss_vs_steps.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
