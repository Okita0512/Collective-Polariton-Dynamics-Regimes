import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def is_regular_font_path(path):
    name = os.path.basename(path).lower()
    rejected = (
        "bold", "black", "heavy", "italic", "oblique",
        "condensed", "narrow", "compressed",
    )
    return not any(token in name for token in rejected)


def set_helvetica():
    candidates = []
    direct_path = r"C:\Windows\Fonts\Helvetica.ttf"
    if os.path.exists(direct_path):
        candidates.append(direct_path)
    for font_entry in fm.fontManager.ttflist:
        if font_entry.name.lower().startswith("helvetica"):
            candidates.append(font_entry.fname)
    for path in fm.findSystemFonts():
        if "helvetica" in os.path.basename(path).lower():
            candidates.append(path)
    if not candidates:
        return
    def font_score(path):
        name = os.path.basename(path).lower()
        score = 0
        if "regular" in name or "roman" in name:
            score -= 2
        if "bold" in name or "black" in name:
            score += 2
        if "italic" in name or "oblique" in name:
            score += 1
        return (score, len(name))
    best_path = sorted(candidates, key=font_score)[0]
    fm.fontManager.addfont(best_path)
    font_name = fm.FontProperties(fname=best_path).get_name()
    plt.rcParams["font.family"] = font_name

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"
set_helvetica()

script_dir = Path(__file__).resolve().parent
mf_conv_root = script_dir / "MF_convergence"

rabi_period_fs = 2.0 * np.pi / (0.2 / 0.658211951)
t_max = 100.0

mf_conv_folders = ["N=1", "N=2", "N=4", "N=8", "N=10000"]
mf_conv_colors = {
    "N=1": "#7b2cbf",
    "N=2": "#1f77b4",
    "N=4": "#2ca02c",
    "N=8": "#d62728",
    "N=10000": "#000000",
}
mf_conv_labels = {
    "N=1": r"$N=1$",
    "N=2": r"$N=2$",
    "N=4": r"$N=4$",
    "N=8": r"$N=8$",
    "N=10000": r"$N\!\to\!\infty$",
}


def add_rabi_period_lines(ax, xmax):
    for i, x_value in enumerate(np.arange(rabi_period_fs, xmax + 1e-9, rabi_period_fs)):
        ax.axvline(
            x_value,
            color="black",
            ls="--",
            lw=1.5,
            label=f"Harmonic / SE Rabi period ({rabi_period_fs:.2f} fs)" if i == 0 else None,
        )


fig, ax = plt.subplots(figsize=(10, 4.8), dpi=128)

for folder in mf_conv_folders:
    path = mf_conv_root / folder / "Photon_number.dat"
    data = np.loadtxt(path)
    mask = data[:, 0] <= t_max
    ax.plot(
        data[mask, 0],
        data[mask, 1],
        lw=2.5,
        color=mf_conv_colors[folder],
        label=mf_conv_labels[folder],
    )

add_rabi_period_lines(ax, t_max)

ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Time (fs)", fontsize=22)
ax.set_ylabel("Photon Number", fontsize=22)
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=16, frameon=False)

ax.tick_params(which="major", length=8, width=2.0, labelsize=20, direction="in")
ax.tick_params(which="minor", length=4, width=1.5, direction="in")
ax.tick_params(axis="x", pad=10)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("Fig2.pdf", bbox_inches="tight")
fig.savefig("Fig2.png", bbox_inches="tight", dpi=150)
print("Saved Fig2.pdf / Fig2.png")
