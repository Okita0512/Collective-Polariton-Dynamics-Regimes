"""Supporting Information Fig. S1.

Two-panel diagnostic figure for unprojected cumulant closures in the
N=1, N_exc=1 single-emitter benchmark.

Panel (a): unprojected CE2 and unprojected CE3 photon-number dynamics
compared against the exact SE/JC result. Both CE2 and CE3 are diagnostic
cumulant closures in the full photon-emitter algebra, not a controlled
accuracy ladder for this projected single-excitation benchmark.

Panel (b): the unprojected fourth-order diagnostic closure compared against
the projected (exact) single-excitation dynamics and the exact SE/JC result.
The unprojected fourth-order diagnostic can leave the physical N_exc=1
projected manifold, while the projected algebra reproduces the exact JC
dynamics.

Data are read from:
    Fig3 and S1/N1_Nexc1_CE2-CE3/       (panel a: CE2, CE3, exact JC)
    Fig3 and S1/N1_Nexc1_CE4_FigS1/     (panel b: unprojected 4th-order
                                          diagnostic, projected/exact dynamics)

To regenerate data:
    cd "Fig3 and S1/N1_Nexc1_CE2-CE3"
    python3 CE2.py
    python3 CE3.py
    cd ../N1_Nexc1_CE4_FigS1
    python3 CE4_naive.py
    python3 CE4_constrained.py
"""
from pathlib import Path
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parent
base_a = ROOT / "N1_Nexc1_CE2-CE3"
base_b = ROOT / "N1_Nexc1_CE4_FigS1"
OUT = ROOT.parent / "Figures"


def load_xy(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def symlog_power_formatter(value, _pos):
    if np.isclose(value, 0.0):
        return "0"
    sign = "-" if value < 0.0 else ""
    exponent = int(np.round(np.log10(abs(value))))
    return rf"${sign}10^{{{exponent}}}$"


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
        return None

    best_path = sorted(scored, key=lambda item: item[0])[0][1]
    fm.fontManager.addfont(best_path)
    font_name = fm.FontProperties(fname=best_path).get_name()
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans"]
    return fm.FontProperties(fname=best_path, weight="normal", style="normal")


def apply_regular_font(fig, font_prop):
    if font_prop is None:
        return
    for text in fig.findobj(Text):
        fontsize = text.get_fontsize()
        text.set_fontproperties(font_prop)
        text.set_fontsize(fontsize)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")


def draw_rabi_lines(ax, xmax, period):
    n_lines = int(xmax // period)
    for k in range(1, n_lines + 1):
        x = k * period
        if k == 1:
            ax.axvline(
                x, color="#6F6F6F", linestyle="--", linewidth=1.15, alpha=0.85, zorder=0,
                label=f"Harmonic / SE Rabi period ({period:.2f} fs)",
            )
        else:
            ax.axvline(x, color="#8A8A8A", linestyle="--", linewidth=0.9, alpha=0.45, zorder=0)


def main():
    font_prop = set_helvetica()
    plt.rcParams["axes.unicode_minus"] = False
    HBAR_eVfs = 0.658211951
    G_COLLECTIVE_EV = 0.1
    period = np.pi / (G_COLLECTIVE_EV / HBAR_eVfs)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.6), dpi=170)

    # ---------- panel (a): unprojected CE2 vs CE3 vs exact SE/JC ----------
    ax = axes[0]
    t_se, n_se = load_xy(base_a / "Photon_number_JC.dat")
    t_ce2, n_ce2 = load_xy(base_a / "Photon_number_CE2.dat")
    t_ce3, n_ce3 = load_xy(base_a / "Photon_number_CE3.dat")

    draw_rabi_lines(ax, 200.0, period)
    ax.plot(t_ce2, n_ce2, color="#0072B2", lw=2.0, ls="-", zorder=2, label="unprojected CE2")
    ax.plot(t_ce3, n_ce3, color="#D55E00", lw=1.9, ls="--", zorder=3, label="unprojected CE3")
    ax.plot(t_se, n_se, color="#009E73", lw=2.0, ls="-", zorder=4, label="exact SE/JC")

    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(0.0, 1.20)
    ax.set_xlabel(r"$t$ (fs)", fontsize=14)
    ax.set_ylabel(r"$n(t)$", fontsize=14)
    ax.text(-0.12, 0.93, "(a)", transform=ax.transAxes, fontsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="both", direction="in", labelsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    # ---------- panel (b): unprojected 4th-order diagnostic vs projected/exact ----------
    ax = axes[1]
    t_ex, n_ex = load_xy(base_b / "Photon_number_exact.dat")
    t_diag, n_diag = load_xy(base_b / "Photon_number_unprojected_4th_order_diagnostic.dat")
    t_proj, n_proj = load_xy(base_b / "Photon_number_projected_single_excitation.dat")

    ax.plot(t_ex, n_ex, color="#009E73", lw=2.0, ls="-", zorder=3, label="exact SE/JC")
    ax.plot(t_diag, n_diag, color="#D55E00", lw=1.9, ls="-", zorder=2, label="unprojected 4th-order diagnostic")
    ax.plot(t_proj, n_proj, color="#000D83", lw=2.0, ls="--", zorder=4, label="projected single-excitation algebra")

    ax.set_xlim(0.0, 200.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-2.0e5, 2.0)
    ax.set_xlabel(r"$t$ (fs)", fontsize=14)
    ax.set_ylabel(r"$n(t)$", fontsize=14)
    ax.text(-0.12, 0.93, "(b)", transform=ax.transAxes, fontsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(symlog_power_formatter))
    ax.tick_params(which="both", direction="in", labelsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    apply_regular_font(fig, font_prop)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "FigS1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigS1.png", bbox_inches="tight", dpi=170)
    plt.close(fig)

    print("Saved Figures/FigS1.pdf / Figures/FigS1.png")
    print(f"(a) Exact SE:            n range = [{np.min(n_se):.6f}, {np.max(n_se):.6f}]")
    print(f"(a) Unprojected CE2:     n range = [{np.min(n_ce2):.6f}, {np.max(n_ce2):.6f}]")
    print(f"(a) Unprojected CE3:     n range = [{np.min(n_ce3):.6f}, {np.max(n_ce3):.6f}]")
    print(f"(b) Exact JC:            n range = [{np.min(n_ex):.6f}, {np.max(n_ex):.6f}]")
    print(f"(b) Unprojected 4th-order diagnostic: n range = [{np.min(n_diag):.4g}, {np.max(n_diag):.4g}]")
    print(f"(b) Projected algebra:   n range = [{np.min(n_proj):.6f}, {np.max(n_proj):.6f}]")


if __name__ == "__main__":
    main()
