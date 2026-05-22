import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


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

    regular_candidates = [
        path for path in candidates
        if os.path.basename(path).lower() in {"helvetica.ttf", "helvetica_0.ttf", "helvetica_1.ttf"}
    ]
    best_path = sorted(regular_candidates or candidates, key=font_score)[0]
    fm.fontManager.addfont(best_path)
    font_name = fm.FontProperties(fname=best_path).get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial"]
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


plt.rcParams["font.family"] = "Helvetica"
set_helvetica()

script_dir = Path(__file__).resolve().parent
se_root = script_dir / "SE_4states"
mf_2vib_root = script_dir / "MF_2vib"
t_vib = 1000.0

n_few = [1, 2, 4, 8]
n_few_colors = {1: "#7b2cbf", 2: "#1f77b4", 4: "#2ca02c", 8: "#d62728"}
n_few_labels = {1: r"$N=1$", 2: r"$N=2$", 4: r"$N=4$", 8: r"$N=8$"}

n_large = [10, 1000, 100000, 10000000]
n_large_colors = {
    10: "#7b2cbf",
    1000: "#ff7f0e",
    100000: "#2ca02c",
    10000000: "#000000",
}
n_large_alphas = {10: 0.3, 1000: 0.55, 100000: 0.8, 10000000: 1.0}
n_large_labels = {
    10: r"$N=10$",
    1000: r"$N=10^3$",
    100000: r"$N=10^5$",
    10000000: r"$N=10^7$",
}

n0_values = [1e-1, 1e-2, 1e-3, 1e-4]
n0_colors = {1e-1: "#7b2cbf", 1e-2: "#ff7f0e", 1e-3: "#2ca02c", 1e-4: "black"}
n0_alphas = {1e-1: 0.3, 1e-2: 0.55, 1e-3: 0.8, 1e-4: 1.0}
n0_files = {
    1e-1: "n0=1e-1/ceb_sq_2vib.dat",
    1e-2: "n0=1e-2/ceb_sq_2vib.dat",
    1e-3: "n0=1e-3/ceb_sq_2vib.dat",
    1e-4: "n0=1e-4/ceb_sq_2vib.dat",
}
n0_labels = {
    1e-1: r"$n_0=10^{-1}$",
    1e-2: r"$n_0=10^{-2}$",
    1e-3: r"$n_0=10^{-3}$",
    1e-4: r"$n_0=10^{-4}$",
}


def panel_label(ax, label, x_pos=-0.13, y_pos=1.10):
    ax.text(
        x_pos,
        y_pos,
        label,
        transform=ax.transAxes,
        fontsize=36,
        fontname="Helvetica",
        va="top",
        ha="left",
    )


def plot_se_vib_population(ax, n_values, colors, labels, alphas=None):
    for n_value in n_values:
        dat_b1 = np.loadtxt(se_root / f"N={n_value}" / "B1_vs_t.dat", comments="#")
        dat_c1 = np.loadtxt(se_root / f"N={n_value}" / "C1_vs_t.dat", comments="#")
        mask = dat_b1[:, 0] <= t_vib
        ax.plot(
            dat_b1[mask, 0],
            dat_b1[mask, 1] + dat_c1[mask, 1],
            lw=2.5,
            color=colors[n_value],
            alpha=1.0 if alphas is None else alphas[n_value],
            label=labels[n_value],
        )


fig, axes = plt.subplot_mosaic(
    [["a"], ["b"], ["c"]],
    figsize=(12, 12),
    dpi=128,
    gridspec_kw={"height_ratios": [1.0, 1.0, 1.05]},
)
ax_a = axes["a"]
ax_b = axes["b"]
ax_c = axes["c"]

plot_se_vib_population(ax_a, n_few, n_few_colors, n_few_labels)
ax_a.set_xlim(0, t_vib)
ax_a.set_ylim(0, 0.7)
ax_a.set_xlabel("Time (fs)", fontsize=22)
ax_a.set_ylabel(r"$P_{B_1}(t)+P_{C_1}(t)$     ", fontsize=22)
ax_a.xaxis.set_major_locator(MultipleLocator(200))
ax_a.xaxis.set_minor_locator(MultipleLocator(100))
ax_a.yaxis.set_major_locator(MultipleLocator(0.2))
ax_a.yaxis.set_minor_locator(MultipleLocator(0.05))
ax_a.legend(loc="upper center", ncol=4, fontsize=18, frameon=False)
panel_label(ax_a, "(a)", y_pos=1.05)

plot_se_vib_population(ax_b, n_large, n_large_colors, n_large_labels, n_large_alphas)
ax_b.set_xlim(0, t_vib)
ax_b.set_ylim(0, 0.06)
ax_b.set_xlabel("Time (fs)", fontsize=22)
ax_b.set_ylabel(r"$P_{B_1}(t)+P_{C_1}(t)$", fontsize=22)
ax_b.xaxis.set_major_locator(MultipleLocator(200))
ax_b.xaxis.set_minor_locator(MultipleLocator(100))
ax_b.yaxis.set_major_locator(MultipleLocator(0.02))
ax_b.yaxis.set_minor_locator(MultipleLocator(0.01))
ax_b.legend(loc="upper center", ncol=4, fontsize=18, frameon=False)
panel_label(ax_b, "(b)", y_pos=1.20)

dat_b1 = np.loadtxt(se_root / "N=10000000" / "B1_vs_t.dat", comments="#")
dat_c1 = np.loadtxt(se_root / "N=10000000" / "C1_vs_t.dat", comments="#")
mask_se = dat_b1[:, 0] <= t_vib
t_se = dat_b1[mask_se, 0]
y_se = dat_b1[mask_se, 1] + dat_c1[mask_se, 1]

ax_c_r = ax_c.twinx()
mf_lines = []
for n0 in n0_values:
    dat_mf = np.loadtxt(mf_2vib_root / n0_files[n0], comments="#")
    mask_mf = dat_mf[:, 0] <= t_vib
    line_mf, = ax_c.plot(
        dat_mf[mask_mf, 0],
        dat_mf[mask_mf, 1],
        lw=2.5,
        ls="-",
        color=n0_colors[n0],
        alpha=n0_alphas[n0],
        label=n0_labels[n0],
    )
    mf_lines.append(line_mf)

line_se, = ax_c_r.plot(
    t_se,
    y_se,
    color="#d62728",
    lw=3.0,
    ls="dotted",
    label=r"SE 4-state ($N\!\to\!\infty$)",
)

ax_c.set_xlim(0, t_vib)
ax_c.set_ylim(0, 0.06)
ax_c.set_xlabel("Time (fs)", fontsize=22)
ax_c.set_ylabel(r"$|c_{eb}(t)|^2/n_0$", fontsize=22)
ax_c.xaxis.set_major_locator(MultipleLocator(200))
ax_c.xaxis.set_minor_locator(MultipleLocator(100))
ax_c.yaxis.set_major_locator(MultipleLocator(0.02))
ax_c.yaxis.set_minor_locator(MultipleLocator(0.01))

ax_c_r.set_ylabel(r"$P_{B_1}(t)+P_{C_1}(t)$", fontsize=22, color="#d62728")
ax_c_r.set_ylim(0, 0.06)
ax_c_r.yaxis.set_major_locator(MultipleLocator(0.02))
ax_c_r.yaxis.set_minor_locator(MultipleLocator(0.01))

ax_c.legend(
    handles=mf_lines + [line_se],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    fontsize=15,
    columnspacing=1.1,
    handlelength=2.0,
    frameon=False,
)
panel_label(ax_c, "(c)", y_pos=1.20)

for ax in [ax_a, ax_b, ax_c]:
    ax.tick_params(which="major", length=8, width=2.0, labelsize=20, direction="in")
    ax.tick_params(which="minor", length=4, width=1.5, direction="in")
    ax.tick_params(axis="x", pad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

ax_c_r.tick_params(
    which="major",
    length=8,
    width=2.0,
    labelsize=20,
    direction="in",
    colors="#d62728",
)
ax_c_r.tick_params(which="minor", length=4, width=1.5, direction="in", colors="#d62728")
ax_c_r.spines["right"].set_linewidth(2.5)
ax_c_r.spines["right"].set_color("#d62728")

fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.5)
pos_b = ax_b.get_position()
ax_b.set_position([pos_b.x0, pos_b.y0 + 0.046, pos_b.width, pos_b.height])
pos_c = ax_c.get_position()
ax_c.set_position([pos_c.x0, pos_c.y0 + 0.026, pos_c.width, pos_c.height])
fig.savefig("Fig4.pdf", bbox_inches="tight")
fig.savefig("Fig4.png", bbox_inches="tight", dpi=150)
print("Saved Fig4.pdf / Fig4.png")
