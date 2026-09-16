import re
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
log_dir = script_dir / "clip_ablation" / "lr_0.2"

files = {
    "D=0.05": log_dir / "softclip_1gpu_0.1_D=0.05.log",
    "D=0.25": log_dir / "softclip_1gpu_0.1_D=0.25.log",
    "D=0.75": log_dir / "softclip_1gpu_0.1_D=0.75.log",
    "D=1.0":  log_dir / "softclip_1gpu_0.1_D=1.log",
}

pattern = re.compile(r"step:(\d+)/\d+\s+val_loss:([\d.]+)")

plt.rcParams.update({
    "font.size": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})
fig, ax = plt.subplots(figsize=(8, 5.5))

for label, path in files.items():
    steps, losses = [], []
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    ax.plot(steps, losses, label=label, linewidth=2.5)

ax.set_xlabel("Step")
ax.set_ylabel("Val Loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(log_dir / "val_loss_plot.pdf", dpi=150)
print("Saved to val_loss_plot.pdf")