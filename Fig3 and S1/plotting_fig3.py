"""Main Fig. 3.

Panel (a): valid N=1, N_exc=1 single-emitter benchmark. Shows only the exact
SE/JC dynamics, CE2, and the MF coherent-state counterpart. The unprojected
CE3 and fourth-order diagnostic closures are shown instead in Fig. S1
(see plotting_figS1.py), where they are treated as diagnostics rather than a
controlled improvement sequence.
Panels (b,c): many-molecule CE2 connected pair correlations.
"""
from pathlib import Path
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "Figures"
HBAR_eVfs = 0.658211951
G_COLLECTIVE_EV = 0.1


def load_xy(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def _font_descriptor(path):
    parts = [os.path.basename(path)]
    try:
        font = FT2Font(path)
        parts.extend([font.family_name, font.style_name])
        postscript_name = getattr(font, "postscript_name", "")
        parts.append(postscript_name() if callable(postscript_name) else postscript_name)
    except Exception:
        pass
    return " ".join(str(part) for part in parts if part).lower()


def _normalized_font_score(path):
    descriptor = _font_descriptor(path)
    compact = descriptor.replace("-", "").replace("_", "").replace(" ", "")
    disallowed = (
        "bold",
        "black",
        "heavy",
        "semibold",
        "semib",
        "demibold",
        "demi",
        "medium",
        "italic",
        "oblique",
        "condensed",
        "narrow",
    )
    if any(term in compact for term in disallowed):
        return None

    family_order = (
        ("helvetica", 0),
        ("arial", 1),
        ("liberationsans", 2),
        ("nimbussans", 3),
    )
    family_score = 4
    for family, score in family_order:
        if family in compact:
            family_score = score
            break

    regular_bonus = -2 if any(term in compact for term in ("regular", "roman", "normal", "book")) else 0
    basename = os.path.basename(path).lower()
    if basename in {"helvetica.ttf", "arial.ttf"}:
        regular_bonus -= 1
    return (family_score, regular_bonus, len(basename), basename)


def set_helvetica():
    candidates = []

    def add_candidate(path):
        if os.path.exists(path):
            candidates.append(path)

    font_roots = (r"C:\Windows\Fonts", "/mnt/c/Windows/Fonts")
    common_names = (
        "Helvetica-Regular.ttf",
        "Helvetica.ttf",
        "HelveticaNeue-Regular.ttf",
        "HelveticaNeue.ttf",
        "HelveticaLTStd-Roman.otf",
        "HelveticaLTStd-Roman.ttf",
        "arial.ttf",
    )
    for font_root in font_roots:
        for font_name in common_names:
            add_candidate(os.path.join(font_root, font_name))

    for font_entry in fm.fontManager.ttflist:
        descriptor = f"{font_entry.name} {os.path.basename(font_entry.fname)}".lower()
        if any(name in descriptor for name in ("helvetica", "arial", "liberation sans", "nimbus sans")):
            add_candidate(font_entry.fname)
    for path in fm.findSystemFonts():
        descriptor = os.path.basename(path).lower()
        if any(name in descriptor for name in ("helvetica", "arial", "liberation", "nimbus")):
            add_candidate(path)

    unique_candidates = []
    seen = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(path)

    scored = []
    for path in unique_candidates:
        score = _normalized_font_score(path)
        if score is not None:
            scored.append((score, path))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if not scored:
        plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
        return

    best_path = sorted(scored, key=lambda item: item[0])[0][1]
    fm.fontManager.addfont(best_path)
    font_name = fm.FontProperties(fname=best_path).get_name()
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans"]
    return fm.FontProperties(fname=best_path, weight="normal", style="normal")

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"
set_helvetica()


def apply_regular_font(fig, font_prop):
    if font_prop is None:
        return
    for text in fig.findobj(Text):
        fontsize = text.get_fontsize()
        text.set_fontproperties(font_prop)
        text.set_fontsize(fontsize)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")


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
    base = ROOT / "N1_Nexc1_CE2-CE3"
    styles = {
        "CE2": dict(color="#0072B2", lw=2.0, ls="-"),
        "SE": dict(color="#009E73", lw=2.0, ls="-"),
        "MF": dict(color="#222222", lw=1.8, ls=":"),
    }
    draw_rabi_lines(ax, 200.0)
    t, y = load_xy(base / "Photon_number_CE2.dat")
    ax.plot(t, y, label="CE2", **styles["CE2"])
    t, y = load_xy(base / "Photon_number_JC.dat")
    ax.plot(t, y, label="exact SE/JC", **styles["SE"])
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
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.02))

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
