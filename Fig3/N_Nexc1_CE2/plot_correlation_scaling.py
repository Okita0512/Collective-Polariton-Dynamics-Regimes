import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
import os


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
    if candidates:
        best_path = sorted(candidates, key=lambda path: ("bold" in os.path.basename(path).lower(), len(os.path.basename(path))))[0]
        fm.fontManager.addfont(best_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=best_path).get_name()
    else:
        plt.rcParams["font.family"] = "Helvetica"


def fit_power_law(N, y, min_N=1.0):
    mask = (y > 0.0) & (N >= min_N)
    coeff = np.polyfit(np.log(N[mask]), np.log(y[mask]), 1)
    exponent = coeff[0]
    prefactor = np.exp(coeff[1])
    return prefactor, exponent


def main():
    set_helvetica()

    data = np.load("correlation_sweep.npz")
    N = data["N"].astype(float)
    t = data["t"]
    P_conn = np.real(data["P_conn"])
    peak_P = data["peak_P"]

    fit_min_N = 5.0
    pref_P, exp_P = fit_power_law(N, peak_P, min_N=fit_min_N)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(N)))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), dpi=170, sharex=False)

    ax = axes[0]
    for idx, N_value in enumerate(N.astype(int)):
        ax.plot(t, P_conn[idx], lw=1.9, color=colors[idx], label=f"N={N_value}")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylabel(r"$P_c=\langle\sigma_i^+\sigma_j^-\rangle_c$")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(which="both", direction="in")
    ax.legend(frameon=False, ncol=4, fontsize=8)

    ax = axes[1]
    for idx, N_value in enumerate(N.astype(int)):
        ax.plot(t, N_value * P_conn[idx], lw=1.9, color=colors[idx], label=f"N={N_value}")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylabel(r"$N P_c$")
    ax.set_xlabel(r"$t$ (fs)")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(which="both", direction="in")

    ax = axes[2]
    N_fine = np.linspace(N.min(), N.max(), 300)
    ax.loglog(N, peak_P, "o", ms=5.0, color="#0072B2", label=rf"max $|P_c|$")
    ax.loglog(N_fine, pref_P * N_fine**exp_P, "-", lw=1.8, color="#0072B2")
    ax.loglog(N_fine, peak_P[0] * (N_fine / N[0]) ** -0.5, "--", lw=1.2, color="#777777", label=r"$N^{-1/2}$ guide")
    ax.loglog(N_fine, peak_P[0] * (N_fine / N[0]) ** -1.0, ":", lw=1.5, color="#444444", label=r"$N^{-1}$ guide")
    ax.text(
        0.05,
        0.08,
        rf"large-$N$ fit ($N\geq {fit_min_N:.0f}$): $N^{{{exp_P:.2f}}}$",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("peak connected correlation")
    ax.tick_params(which="both", direction="in")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("CE2_correlation_scaling.pdf", bbox_inches="tight")
    fig.savefig("CE2_correlation_scaling.png", bbox_inches="tight", dpi=170)
    print("Saved CE2_correlation_scaling.pdf / CE2_correlation_scaling.png")
    print(f"Peak P_c fit exponent for N >= {fit_min_N:.0f}: {exp_P:.4f}")


if __name__ == "__main__":
    main()
