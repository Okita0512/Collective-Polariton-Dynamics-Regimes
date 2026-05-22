"""Main Fig. 3.

Panel (a): valid N=1, N_exc=1 single-emitter CE benchmark.
Panels (b,c): many-molecule CE2 connected pair correlations.
"""
from pathlib import Path
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT # .parent / "Figures"
HBAR_eVfs = 0.658211951
G_COLLECTIVE_EV = 0.1


def load_xy(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


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


def apply_regular_font(fig, font_prop):
    for text in fig.findobj(Text):
        fontsize = text.get_fontsize()
        text.set_fontproperties(font_prop)
        text.set_fontsize(fontsize)
        text.set_fontweight("normal")


def draw_rabi_lines(ax, xmax):
    period = np.pi / (G_COLLECTIVE_EV / HBAR_eVfs)
    n_lines = int(xmax // period)
    for k in range(1, n_lines + 1):
        x = k * period
        if k == 1:
            ax.axvline(
                x,
                color="#6F6F6F",
                linestyle="--",
                linewidth=1.15,
                alpha=0.85,
                zorder=0,
                label=f"Harmonic / SE Rabi period ({period:.2f} fs)",
            )
        else:
            ax.axvline(x, color="#8A8A8A", linestyle="--", linewidth=0.9, alpha=0.45, zorder=0)


def fit_power_law(N, y, min_N=5.0):
    mask = (N >= min_N) & (y > 0.0)
    coeff = np.polyfit(np.log(N[mask]), np.log(y[mask]), 1)
    return np.exp(coeff[1]), coeff[0]


def main():
    plt.rcParams["font.family"] = "Helvetica"
    font_prop = set_helvetica()

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.2), dpi=170)

    # ---------- panel a: N=1, N_exc=1 ----------
    ax = axes[0]
    base = ROOT / "N1_Nexc1_CE2-CE5"
    styles = {
        "CE2": dict(color="#0072B2", lw=2.0, ls="-"),
        "CE3": dict(color="#D55E00", lw=1.9, ls="--"),
        "CE4": dict(color="#009E73", lw=2.0, ls="-"),
        "CE5": dict(color="#8B0000", lw=1.9, ls="--"),
        "MF": dict(color="#222222", lw=1.8, ls=":"),
    }
    draw_rabi_lines(ax, 200.0)
    for label in ["CE2", "CE3", "CE4", "CE5"]:
        t, y = load_xy(base / f"Photon_number_{label}.dat")
        ax.plot(t, y, label=label, **styles[label])
    t, y = load_xy(base / "Photon_number.dat")
    ax.plot(t, y, label="MF", **styles["MF"])
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(0.0, 1.20)
    ax.set_xlabel(r"$t$ (fs)", fontsize=22)
    ax.set_ylabel(r"$n(t)$", fontsize=22)
    ax.text(-0.20, 0.88, "(a)", transform=ax.transAxes, fontsize=36)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="major", labelsize=15, direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    # ---------- panels b/c: many-molecule CE2 connected correlations ----------
    data = np.load(ROOT / "N_Nexc1_CE2" / "correlation_sweep.npz")
    N = data["N"].astype(float)
    t = data["t"]
    P_conn = np.real(data["P_conn"])
    peak_P = data["peak_P"]
    pref, exponent = fit_power_law(N, peak_P, min_N=5.0)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(N)))

    ax = axes[1]
    for idx, N_value in enumerate(N.astype(int)):
        ax.plot(t, P_conn[idx], color=colors[idx], lw=1.8, label=rf"$N = {N_value}$")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(0.0, 0.35)
    ax.set_xlabel(r"$t$ (fs)", fontsize=22)
    ax.set_ylabel(r"$P_c(t)$", fontsize=22)
    ax.text(-0.20, 0.88, "(b)", transform=ax.transAxes, fontsize=36)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="major", labelsize=15, direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.legend(frameon=False, loc="upper center", ncol=6, fontsize=8)

    ax = axes[2]
    N_fine = np.linspace(N.min(), N.max(), 300)
    ax.loglog(N, peak_P, "o", color="#0072B2", ms=5.0, label=r"CE2")
    ax.loglog(
        N_fine,
        pref * N_fine**exponent,
        "--",
        color="#000000",
        lw=1.8,
        label="fit",
    )
    ax.set_xlabel(r"$N$", fontsize=22)
    ax.set_ylabel(r"$\max_t |P_c(t)|$", fontsize=22)
    ax.text(-0.20, 0.88, "(c)", transform=ax.transAxes, fontsize=36)
    ax.tick_params(which="major", labelsize=15, direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.legend(frameon=False, fontsize=16)
    ax.text(
        0.05, 0.08,
        rf"$\propto N^{{{exponent:.2f}}}$",
        transform=ax.transAxes,
        fontsize=32,
        va="bottom",
    )

    apply_regular_font(fig, font_prop)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "Fig3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig3.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("Saved Figures/Fig3.pdf / Figures/Fig3.png")
    print(f"many-molecule peak P_c exponent for N>=5: {exponent:.4f}")


if __name__ == "__main__":
    main()
