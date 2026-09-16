import re
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Gets the folder where your script is located
script_dir = Path(__file__).resolve().parent
log_dir = script_dir / "effective_rank"


matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ---------- config ----------
files = {
    "Muon":     log_dir / "muon_1gpu_0.5_er.log",
    "MuonClip": log_dir / "muonclip_1gpu_0.5_er.log",
    "SPECTRA":  log_dir / "spectra_1gpu_0.5_er_D=1.log",
    "Soft Musec": log_dir / "softclip_1gpu_0.5_er_D=1.log",
}

# ---------- parse ----------
pattern = re.compile(
    r"step:(\d+)\s+erank\[.*?\]\s+"
    r"qk_bank:([\d.]+)\s+"
    r"vo_bank:([\d.]+)\s+"
    r"mlp_bank:([\d.]+)"
)

data = {}  # name -> {steps, qk, vo, mlp}
for name, path in files.items():
    steps, qk, vo, mlp = [], [], [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                qk.append(float(m.group(2)))
                vo.append(float(m.group(3)))
                mlp.append(float(m.group(4)))
    data[name] = {"steps": steps, "qk": qk, "vo": vo, "mlp": mlp}

# ---------- plot ----------
colors = {
    "Muon":     "#2563eb",
    "MuonClip": "#dc2626",
    "SPECTRA":  "#16a34a",
    "Soft Musec": "#9333ea",
}

colors = plt.get_cmap("tab10").colors

bank_labels = {
    "qk":  "QK",
    "vo":  "VO",
    "mlp": "MLP",
}

for bank_key in ["qk", "vo", "mlp"]:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ind = 0
    for name in files:
        d = data[name]
        ax.plot(d["steps"], d[bank_key],
                label=name, color=colors[ind], linewidth=1.8, alpha=0.9)
        ind += 1
    #ax.set_title(f"Effective Rank vs Steps — {bank_labels[bank_key]}", fontsize=15)
    ax.set_xlabel("Step", fontsize=28)
    ax.set_ylabel("erank", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=28)
    ax.legend(loc="best", fontsize=28)
    fig.tight_layout()
    out = f"effective_rank/erank_{bank_key}.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
