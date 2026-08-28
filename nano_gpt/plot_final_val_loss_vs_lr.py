import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

method_dict = {"muon":"Muon", "muonclip":"MuonClip", "spectra":"SPECTRA", "softclip":"Soft Musec"}

import os

PLOT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PLOT_DIR / "c4" / "medium"

LOG_RE = re.compile(r"^(?P<method>.+)_1gpu_(?P<lr>\d+(?:\.\d+)?)\.log$")
VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?|nan)"
)


def discover_logs():
    """Return log paths grouped by method and learning rate."""
    logs = defaultdict(dict)
    for path in sorted(PLOT_DIR.glob("*_1gpu_*.log")):
        match = LOG_RE.match(path.name)
        if match:
            logs[match.group("method")][float(match.group("lr"))] = path
    return {method: paths for method, paths in sorted(logs.items())}


def final_val_loss(path):
    """Return (final_val_loss, last_step, total_steps)."""
    last = None
    last_step = None
    total_steps = None
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = VAL_RE.search(line)
            if match:
                last_step = int(match.group(1))
                total_steps = int(match.group(2))
                value = match.group(3)
                last = float("nan") if value.lower() == "nan" else float(value)
    return last, last_step, total_steps


logs = discover_logs()
if not logs:
    raise FileNotFoundError(f"No *_1gpu_*.log files found in {PLOT_DIR}")

results = defaultdict(list)
for method, paths in logs.items():
    for lr, path in sorted(paths.items()):
        value, last_step, total_steps = final_val_loss(path)
        if value is None:
            print(f"Skipping {path.name}: no validation-loss records found")
            continue
        results[method].append((lr, value, last_step, total_steps))
        print(
            f"{method} lr={lr:g}: final val_loss={value}, "
            f"at step {last_step}/{total_steps} ({path.name})"
        )

#results["muonclip"].pop()

if not results:
    raise ValueError("The discovered log files contain no validation-loss records")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5.5))
markers = ["o", "s", "^", "D", "v", "P", "X"]
colors = plt.get_cmap("tab10").colors

for index, (method, method_results) in enumerate(sorted(results.items())):
    color = colors[index % len(colors)]
    marker = markers[index % len(markers)]
    xs = [row[0] for row in method_results]
    ys = [row[1] for row in method_results]
    ax.plot(
        xs,
        ys,
        marker=marker,
        color=color,
        linewidth=2,
        markersize=9,
        label=method_dict[method],
    )

    y_offset = 8 if index % 2 == 0 else -14
    """
    for lr, value, _last_step, _total_steps in method_results:
        if np.isfinite(value):
            ax.annotate(
                f"{value:.3f}",
                (lr, value),
                textcoords="offset points",
                xytext=(6, y_offset),
                fontsize=8,
                color=color,
            )
    """

ax.set_xscale("log")
ax.set_xlabel("Learning rate", fontsize=24)
ax.set_ylabel("Final validation loss", fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=24, loc="best")

# 2. Extract current handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# 3. Define your desired order by index (e.g., Line C, then Line A, then Line B)
order = [0,1,3,2]

# 4. Pass the reordered lists to plt.legend()
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)


plt.tight_layout()
output_path = PLOT_DIR / "final_val_loss_vs_lr.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
