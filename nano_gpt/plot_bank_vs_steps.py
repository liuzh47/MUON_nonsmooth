import re
from pathlib import Path

import matplotlib.pyplot as plt

BANK_RE = re.compile(
    r"step:(\d+)\s+sigma_top\s+"
    r"qk_bank:([\d.]+)\s+vo_bank:([\d.]+)\s+mlp_bank:([\d.]+)"
)

PLOT_DIR = Path(__file__).resolve().parent
PLOT_DIR = PLOT_DIR / "openweb" / "medium"

methods = [
    ("muon",     "Muon",       "muon_1gpu_0.2.log"),
    ("muonclip", "MuonClip",   "muonclip_1gpu_0.2.log"),
    ("spectra",  "SPECTRA",    "spectra_1gpu_0.2.log"),
    ("softclip", "Soft Musec", "softclip_1gpu_0.2.log"),
]


def parse_bank(path):
    steps, qk, vo, mlp = [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BANK_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                qk.append(float(m.group(2)))
                vo.append(float(m.group(3)))
                mlp.append(float(m.group(4)))
    return steps, qk, vo, mlp


data = {}
for key, label, fname in methods:
    steps, qk, vo, mlp = parse_bank(PLOT_DIR / fname)
    data[key] = {"label": label, "steps": steps, "qk": qk, "vo": vo, "mlp": mlp}

colors = plt.get_cmap("tab10").colors
markers = ["o", "s", "^", "D"]

for bank_key, ylabel in [("qk", "qk_bank"), ("vo", "vo_bank"), ("mlp", "mlp_bank")]:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (key, label, _) in enumerate(methods):
        d = data[key]
        ax.plot(
            d["steps"], d[bank_key],
            marker=markers[i], color=colors[i],
            linewidth=2, markersize=5, markevery=5,
            label=d["label"],
        )
    ax.set_xlabel("Steps", fontsize=28)
    ax.set_ylabel(bank_key.upper(), fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=28)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=24, loc="best")
    plt.tight_layout()
    output_path = PLOT_DIR / f"{bank_key}_bank_vs_steps.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
