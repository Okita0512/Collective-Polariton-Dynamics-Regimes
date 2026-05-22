import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def load_xy(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def main():
    t, n_ce2 = load_xy("Photon_number_CE2.dat")
    _, ex_ce2 = load_xy("Excited_population_CE2.dat")
    _, n_exact = load_xy("Photon_number_exact.dat")
    _, ex_exact = load_xy("Excited_population_exact.dat")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(t, n_ce2, color="#0072B2", lw=2.4, label="CE2 photons")
    ax.plot(t, ex_ce2, color="#D55E00", lw=2.4, label="CE2 excitons")
    ax.plot(t, n_exact, color="#0072B2", lw=1.5, ls="--", label="exact photons")
    ax.plot(t, ex_exact, color="#D55E00", lw=1.5, ls="--", label="exact excitons")
    ax.plot(t, n_ce2 + ex_ce2, color="#222222", lw=1.8, ls=":", label="CE2 total")

    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("t (fs)", fontsize=13)
    ax.set_ylabel("population", fontsize=13)
    ax.set_title("N=5 CE2, one-photon initial state", fontsize=13)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="both", direction="in")
    ax.legend(frameon=False, ncol=2, fontsize=10)

    fig.tight_layout()
    fig.savefig("CE2_N5_Nexc1.pdf", bbox_inches="tight")
    fig.savefig("CE2_N5_Nexc1.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved CE2_N5_Nexc1.pdf / CE2_N5_Nexc1.png")


if __name__ == "__main__":
    main()
