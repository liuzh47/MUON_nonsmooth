import re
from pathlib import Path
import matplotlib.pyplot as plt

from pathlib import Path


script_dir = Path(__file__).resolve().parent
log_dir = script_dir / "ns_ablation"

files = {
    "ns steps=3": log_dir / "ns=3" / "softclip_1gpu_0.1_ns=3.log",
    "ns steps=5": log_dir / "ns=5" / "softclip_1gpu_0.1_ns=5.log",
    "ns steps=8": log_dir / "ns=8" / "softclip_1gpu_0.1_ns=8.log",
}

pattern = re.compile(r"step:(\d+)/\d+\s+val_loss:([\d.]+)")

plt.rcParams.update({"font.size": 28})
fig, ax = plt.subplots(figsize=(8, 5.5))

for label, path in files.items():
    steps, losses = [], []
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    ax.plot(steps, losses, label=label, linewidth=2)

ax.set_xlabel("Step")
ax.set_ylabel("Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = Path(log_dir / "val_loss_plot_ns.pdf")
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
