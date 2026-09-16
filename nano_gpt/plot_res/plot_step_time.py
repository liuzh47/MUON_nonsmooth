"""
Plot average time per step for different optimizer methods.

Parses log files produced by a training run, extracts the final
`step_avg` value reported at the last training step, and produces
a bar chart comparing methods.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


script_dir = Path(__file__).resolve().parent
log_dir = script_dir / "fineweb" / "medium"


# ── Config ──────────────────────────────────────────────────────
LOG_FILES = {
    "Muon":     log_dir / "muon_1gpu_0.2.log",
    "MuonClip": log_dir / "muonclip_1gpu_0.2.log",
    "SPECTRA":  log_dir / "spectra_1gpu_0.2.log",
    "Soft Musec": log_dir / "softclip_1gpu_0.2.log",
}

# ── Parse ───────────────────────────────────────────────────────
pattern = re.compile(r"step:(\d+)/\d+.*?step_avg:([\d.]+)ms")

results = {}
for method, path in LOG_FILES.items():
    with open(path) as f:
        last_step, last_avg = -1, None
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                avg  = float(m.group(2))
                if step >= last_step:
                    last_step, last_avg = step, avg
        results[method] = last_avg

methods = list(results.keys())
avgs    = [results[m] for m in methods]

# ── Plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
bars = ax.bar(methods, avgs, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

# Value labels on each bar
for bar, v in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{v:.1f} ms", ha="center", va="bottom", fontsize=11, fontweight="600")

ax.set_ylabel("Average time(ms)", fontsize=20)
#ax.set_title("Training Step Time by Optimizer Method  (1 GPU, lr=0.2)", fontsize=13, fontweight="600")
ax.set_ylim(0, max(avgs) * 1.12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=19)
ax.tick_params(axis="y", labelsize=20)

plt.tight_layout()
plt.savefig(script_dir / "time" / "step_time_comparison.pdf", dpi=180)
plt.close()
print("Saved plot to step_time_comparison.png")
for m, v in zip(methods, avgs):
    print(f"  {m:10s}  {v:.2f} ms/step")
