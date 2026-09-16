import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path(__file__).resolve().parent
LOG_RE = re.compile(r"^(?P<method>.+)_1gpu_(?P<lr>\d+(?:\.\d+)?)\.log$")
VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?|nan)\s+train_time:(\d+)ms"
)


def discover_logs():
    """Return log paths grouped by method and learning rate."""
    logs = defaultdict(dict)
    for path in sorted(PLOT_DIR.glob("*_1gpu_*.log")):
        match = LOG_RE.match(path.name)
        if match:
            logs[match.group("method")][float(match.group("lr"))] = path
    return {method: paths for method, paths in sorted(logs.items())}


def get_all_val_losses(path):
    """Return (train_time_ms, step, val_loss) records from one log."""
    entries = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = VAL_RE.search(line)
            if match:
                value = match.group(3)
                entries.append(
                    (
                        int(match.group(4)),
                        int(match.group(1)),
                        float("nan") if value.lower() == "nan" else float(value),
                    )
                )
    return entries


logs = discover_logs()
if not logs:
    raise FileNotFoundError(f"No *_1gpu_*.log files found in {PLOT_DIR}")

data = defaultdict(dict)
for method, paths in logs.items():
    for lr, path in sorted(paths.items()):
        entries = get_all_val_losses(path)
        if not entries:
            print(f"Skipping {path.name}: no validation-loss records found")
            continue
        data[method][lr] = entries
        last_time = entries[-1][0]
        print(
            f"{method} lr={lr:g}: {len(entries)} val checkpoints, "
            f"last train_time={last_time}ms ({last_time / 1000:.1f}s)"
        )

if not data:
    raise ValueError("The discovered log files contain no validation-loss records")

print("\n--- Aligned by train_time for each learning rate ---")
all_lrs = sorted({lr for method_data in data.values() for lr in method_data})
results = defaultdict(list)

for lr in all_lrs:
    available = {
        method: method_data[lr]
        for method, method_data in data.items()
        if lr in method_data
    }
    reference_time = min(entries[-1][0] for entries in available.values())

    for method, entries in sorted(available.items()):
        eligible = [entry for entry in entries if entry[0] <= reference_time]
        best = eligible[-1] if eligible else min(
            entries, key=lambda entry: abs(entry[0] - reference_time)
        )
        train_time, step, value = best
        results[method].append((lr, value, step, train_time, reference_time))
        print(
            f"{method} lr={lr:g}: val_loss={value:.4f} at step {step}, "
            f"train_time={train_time}ms (reference={reference_time}ms)"
        )

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
        label=method,
    )

    y_offset = 8 if index % 2 == 0 else -20
    for lr, value, _step, train_time, _reference_time in method_results:
        if np.isfinite(value):
            ax.annotate(
                f"{value:.3f}\n({train_time / 1000:.0f}s)",
                (lr, value),
                textcoords="offset points",
                xytext=(6, y_offset),
                fontsize=7,
                color=color,
            )

ax.set_xscale("log")
ax.set_xlabel("Learning rate", fontsize=12)
ax.set_ylabel("Validation loss", fontsize=12)
ax.set_title(
    "Validation loss vs. learning rate (aligned to earlier final train_time)",
    fontsize=12,
)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=10, loc="best")

plt.tight_layout()
output_path = PLOT_DIR / "val_loss_vs_lr.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
