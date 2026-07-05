"""Supporting Information Fig. S2.

Detuning dependence of MF-SE agreement in the bright vibrational mode.

The vibrational frequency is fixed at nu = 0.20 eV in every calculation.  The
detuning is the cavity-exciton detuning:

    Delta = omega_c - omega_0.

For each Delta, the script compares the large-N SE four-state benchmark with a
weak-excitation MF two-vibrational-state calculation.  The plotted MF observable
is |c_eb(t)|^2 / n0, which is the weak-excitation counterpart of the SE
population P_B1(t) + P_C1(t).
"""
from pathlib import Path
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "Figures"
HBAR_EVFS = 0.658211951

OMEGA0 = 2.0
NU = 0.20
G_COLLECTIVE = 0.10
CV = 0.020
N_SE = 10_000_000
N0_MF = 1.0e-4
T_MAX = 1000.0
DT = 0.5
TIMES = np.arange(0.0, T_MAX + DT / 2.0, DT)

CASES = [
    {"tag": "delta_p000meV", "delta_meV": 0.0, "label": r"$\Delta=0$ meV"},
    {"tag": "delta_p050meV", "delta_meV": 50.0, "label": r"$\Delta=+50$ meV"},
    {"tag": "delta_m050meV", "delta_meV": -50.0, "label": r"$\Delta=-50$ meV"},
]


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
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def build_se_hamiltonian(delta_eV):
    omegac = OMEGA0 + delta_eV
    gc = G_COLLECTIVE / np.sqrt(N_SE)
    sqrt_n = np.sqrt(N_SE)
    return np.array(
        [
            [OMEGA0, CV, gc * sqrt_n, 0.0],
            [CV, OMEGA0 + NU, 0.0, gc],
            [gc * sqrt_n, 0.0, omegac, 0.0],
            [0.0, gc, 0.0, omegac + NU],
        ],
        dtype=float,
    )


def propagate_se(delta_eV):
    h = build_se_hamiltonian(delta_eV)
    eigvals, eigvecs = np.linalg.eigh(h)
    psi0 = np.array([0.0, 0.0, 1.0, 0.0])
    psi0_eig = eigvecs.T @ psi0
    phases = np.exp(-1j * np.outer(TIMES, eigvals) / HBAR_EVFS)
    psi_t = (phases * psi0_eig[np.newaxis, :]) @ eigvecs.T
    populations = np.abs(psi_t) ** 2
    return populations[:, 1] + populations[:, 3]


def mf_rhs(delta_eV):
    omegac = OMEGA0 + delta_eV
    gc = G_COLLECTIVE

    def rhs(_t, y):
        alpha = y[0] + 1j * y[1]
        c_g0 = y[2] + 1j * y[3]
        c_e0 = y[4] + 1j * y[5]
        c_g1 = y[6] + 1j * y[7]
        c_e1 = y[8] + 1j * y[9]

        sigma = np.conj(c_g0) * c_e0 + np.conj(c_g1) * c_e1
        d_alpha = (-1j / HBAR_EVFS) * (omegac * alpha + gc * sigma)

        d_c_g0 = (-1j / HBAR_EVFS) * (gc * np.conj(alpha) * c_e0)
        d_c_e0 = (-1j / HBAR_EVFS) * (OMEGA0 * c_e0 + gc * alpha * c_g0 + CV * c_e1)
        d_c_g1 = (-1j / HBAR_EVFS) * (NU * c_g1 + gc * np.conj(alpha) * c_e1)
        d_c_e1 = (-1j / HBAR_EVFS) * (
            (OMEGA0 + NU) * c_e1 + gc * alpha * c_g1 + CV * c_e0
        )

        return [
            d_alpha.real,
            d_alpha.imag,
            d_c_g0.real,
            d_c_g0.imag,
            d_c_e0.real,
            d_c_e0.imag,
            d_c_g1.real,
            d_c_g1.imag,
            d_c_e1.real,
            d_c_e1.imag,
        ]

    return rhs


def propagate_mf(delta_eV):
    alpha0 = np.sqrt(N0_MF)
    y0 = [
        alpha0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    sol = solve_ivp(
        mf_rhs(delta_eV),
        (0.0, T_MAX),
        y0,
        t_eval=TIMES,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"MF propagation failed: {sol.message}")
    c_e1 = sol.y[8] + 1j * sol.y[9]
    return np.abs(c_e1) ** 2 / N0_MF


def panel_label(ax, label):
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=18, va="top", ha="left")


def main():
    set_helvetica()
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.5), dpi=128, sharex=True)
    panel_ids = ["(a)", "(b)", "(c)"]
    summary_rows = []

    for ax, panel_id, case in zip(axes, panel_ids, CASES):
        delta_eV = case["delta_meV"] / 1000.0
        omegac = OMEGA0 + delta_eV
        se = propagate_se(delta_eV)
        mf = propagate_mf(delta_eV)
        diff = mf - se
        max_abs_diff = float(np.max(np.abs(diff)))
        rms_diff = float(np.sqrt(np.mean(diff**2)))

        data_dir = ROOT / case["tag"]
        data_dir.mkdir(exist_ok=True)
        np.savetxt(
            data_dir / "se_largeN.dat",
            np.column_stack([TIMES, se]),
            header=f"time(fs)  P_B1_plus_P_C1  N=1e7  Delta={case['delta_meV']}meV  nu={NU}eV",
            fmt="%.12e",
        )
        np.savetxt(
            data_dir / "mf_weak_excitation.dat",
            np.column_stack([TIMES, mf]),
            header=f"time(fs)  |c_eb|^2/n0  n0=1e-4  Delta={case['delta_meV']}meV  nu={NU}eV",
            fmt="%.12e",
        )
        summary_rows.append([case["delta_meV"], omegac, NU, max_abs_diff, rms_diff])

        ax_r = ax.twinx()
        mf_color = "#000000"
        se_color = "#d62728"
        ax.plot(
            TIMES,
            mf,
            color=mf_color,
            lw=2.0,
            ls="-",
        )
        ax_r.plot(TIMES, se, color=se_color, lw=2.2, ls="dotted")
        ax.set_xlim(0.0, T_MAX)
        y_max = max(0.012, 1.12 * max(np.max(se), np.max(mf)))
        ax.set_ylim(-0.002, y_max)
        ax_r.set_ylim(-0.002, y_max)
        ax.set_ylabel(r"$|c_{eb}(t)|^2/n_0$", fontsize=13, color=mf_color)
        ax_r.set_ylabel(r"$P_{B_1}(t)+P_{C_1}(t)$", fontsize=13, color=se_color)
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax_r.yaxis.set_major_locator(MultipleLocator(0.02))
        ax_r.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax.tick_params(
            which="major",
            length=6,
            width=1.4,
            labelsize=11,
            direction="in",
            colors=mf_color,
        )
        ax.tick_params(which="minor", length=3, width=1.0, direction="in", colors=mf_color)
        ax.tick_params(axis="x", colors="black")
        ax_r.tick_params(
            which="major",
            length=6,
            width=1.4,
            labelsize=11,
            direction="in",
            colors=se_color,
        )
        ax_r.tick_params(which="minor", length=3, width=1.0, direction="in", colors=se_color)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.spines["left"].set_color(mf_color)
        ax_r.spines["right"].set_color(se_color)
        ax_r.spines["right"].set_linewidth(1.4)
        panel_label(ax, panel_id)

    axes[-1].set_xlabel(r"$t$ (fs)", fontsize=14)
    axes[-1].xaxis.set_major_locator(MultipleLocator(200))
    axes[-1].xaxis.set_minor_locator(MultipleLocator(50))

    np.savetxt(
        ROOT / "agreement_summary.dat",
        np.array(summary_rows),
        header="Delta(meV)  omega_c(eV)  nu(eV)  max_abs_diff  rms_diff",
        fmt="%.12e",
    )
    fig.tight_layout(h_pad=0.25)
    fig.savefig(OUT / "FigS2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigS2.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("Saved Figures/FigS2.pdf / Figures/FigS2.png")


if __name__ == "__main__":
    main()
