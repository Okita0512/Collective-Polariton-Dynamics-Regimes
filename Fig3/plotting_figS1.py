"""Supporting Information Fig. S1.

CE4 stability check for the N=1, N_exc=1 benchmark.  The comparison shows the
effect of imposing the exact single-excitation identities for the quartic
moments T1, T2, Cn, and Sd.
"""
from pathlib import Path
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
OUT = ROOT # .parent / "Figures"
HBAR_eVfs = 0.658211951
TICK_FONT = "Times New Roman"


def eV_to_rate(x):
    return x / HBAR_eVfs


omega_c = 2.0
omega_0 = 2.0
g_eV = 0.1
kappa = 0.0
gamma1 = 0.0
gamma_phi = 0.0
eta = 0.0 + 0.0j

Delta_c = eV_to_rate(omega_c - 2.0)
Delta_x = eV_to_rate(omega_0 - 2.0)
g = eV_to_rate(g_eV)
gamma_perp = gamma1 / 2.0 + gamma_phi
w_eq = -1.0
t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)


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
        best_path = sorted(
            candidates,
            key=lambda path: ("bold" in os.path.basename(path).lower(), len(os.path.basename(path))),
        )[0]
        fm.fontManager.addfont(best_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=best_path).get_name()
    else:
        plt.rcParams["font.family"] = "Helvetica"


def set_tick_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(TICK_FONT)


def pack(alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2):
    zc = np.array([alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na], dtype=np.complex128)
    return np.concatenate([zc.view(np.float64), np.array([w, n, Nw, n2], dtype=np.float64)])


def make_rhs(enforce_identities):
    def unpack(y):
        y = np.asarray(y, dtype=np.float64)
        zc = np.ascontiguousarray(y[:24]).view(np.complex128)
        alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na = zc
        w, n, Nw, n2 = y[24], y[25], y[26], y[27]
        if enforce_identities:
            T1 = 0.0 + 0.0j
            T2 = U
            Cn = 0.0 + 0.0j
            Sd = D
        return alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2

    def rhs(t, y):
        alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2 = unpack(y)
        a = alpha
        sm = s
        smC = np.conjugate(sm)

        dalpha = -(kappa / 2 + 1j * Delta_c) * a - 1j * g * sm + eta
        ds = -(gamma_perp + 1j * Delta_x) * sm + 1j * g * U
        dw = -gamma1 * (w - w_eq) + 2j * g * (D - C)
        dn = -kappa * n + 1j * g * (C - D) + 2.0 * np.real(np.conjugate(eta) * a)

        dC = (
            -(kappa / 2 + gamma_perp + 1j * (Delta_c - Delta_x)) * C
            - 1j * g * (1.0 - w) / 2.0
            - 1j * g * (Nw + w)
            + eta * smC
        )
        dD = (
            -(kappa / 2 + gamma_perp - 1j * (Delta_c - Delta_x)) * D
            + 1j * g * (1.0 + w) / 2.0
            + 1j * g * Nw
            + np.conjugate(eta) * sm
        )

        dU = -(kappa / 2 + 1j * Delta_c + gamma1) * U - 1j * g * sm + eta * w + 2j * g * (S1 + sm - Q)
        dS1 = -(kappa + gamma_perp + 1j * Delta_x) * S1 + 0.5j * g * (a + U) + 1j * g * T1
        dQ = -(kappa + gamma_perp - 1j * Delta_x + 2j * Delta_c) * Q - 1j * g * (a - U) + 2.0 * eta * C - 1j * g * T2
        dNw = -kappa * Nw - gamma1 * (Nw - w_eq * n) + 1j * g * (-C - D) + 2j * g * (Sd - Cn)

        Qn = n * Q
        n2_sigminus = n2 * sm + n * S1
        dT1 = (
            -(3 * kappa / 2 + gamma1 + 1j * Delta_c) * T1
            - 1j * g * Q
            - 2j * g * S1
            + 2j * g * (n2_sigminus - Qn)
            + gamma1 * w_eq * Na
        )

        an_sigminus = (n2 + 2 * n + 1.0) * sm
        an_sigplus = n * Q + Q
        dT2 = (
            -(3 * kappa / 2 + gamma1 + 1j * Delta_c) * T2
            - 1j * g * (Q + 2 * S1)
            + 2j * g * (an_sigminus - an_sigplus)
            + gamma1 * w_eq * Na
        )

        N2w = n2 * w
        dSd = (
            -(3 * kappa / 2 + gamma_perp + 1j * (Delta_x - Delta_c)) * Sd
            + 1j * g * (n + Nw + 0.5)
            + 1j * g * N2w
        )
        dCn = (
            -(3 * kappa / 2 + gamma_perp - 1j * (Delta_x - Delta_c)) * Cn
            - 1j * g * (n + Nw + 0.5)
            - 1j * g * N2w
        )
        dNa = -(3 * kappa / 2 + 1j * Delta_c) * Na + 1j * g * (Q - 2 * S1) + eta * n
        dn2 = -2.0 * kappa * n2 + 2j * g * (Cn - Sd) + 2.0 * np.real(np.conjugate(eta) * Na)

        if enforce_identities:
            dT1 = 0.0 + 0.0j
            dT2 = dU
            dCn = 0.0 + 0.0j
            dSd = dD

        return pack(
            dalpha,
            ds,
            dC,
            dD,
            dU,
            dS1,
            dQ,
            dT1,
            dT2,
            dSd,
            dCn,
            dNa,
            np.real(dw),
            np.real(dn),
            np.real(dNw),
            np.real(dn2),
        )

    return rhs


def initial_state():
    alpha0 = 0.0 + 0.0j
    n0 = 1.0
    s0 = 0.0 + 0.0j
    w0 = -1.0
    zero = 0.0 + 0.0j
    Nw0 = n0 * w0
    n20 = n0**2
    return pack(alpha0, s0, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, w0, n0, Nw0, n20)


def run_case(enforce_identities):
    return solve_ivp(
        make_rhs(enforce_identities),
        (t0, t1),
        initial_state(),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )


def extract_photon_number(sol):
    return sol.t, sol.y[25]


def main():
    set_helvetica()
    restricted = run_case(enforce_identities=True)
    unrestricted = run_case(enforce_identities=False)
    if not restricted.success:
        raise RuntimeError(f"restricted CE4 failed: {restricted.message}")

    t_r, n_r = extract_photon_number(restricted)
    t_u, n_u = extract_photon_number(unrestricted)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.35), dpi=170)
    ax.plot(t_r, n_r, color="#0072B2", lw=2.1, label="identity-constrained CE4")
    ax.plot(t_u, n_u, color="#D55E00", lw=1.9, ls="--", label="unconstrained CE4")

    ax.set_xlim(0.0, 200.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-2.0e5, 2.0)
    ax.set_xlabel(r"$t$ (fs)")
    ax.set_ylabel(r"$n(t)$")
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="both", direction="in")
    set_tick_font(ax)
    ax.legend(frameon=False, fontsize=12, loc="lower right")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "FigS1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigS1.png", bbox_inches="tight", dpi=170)
    plt.close(fig)

    print("Saved Figures/FigS1.pdf / Figures/FigS1.png")
    print(f"restricted n range=[{np.min(n_r):.6g}, {np.max(n_r):.6g}]")
    print(
        f"unconstrained n range=[{np.min(n_u):.6g}, {np.max(n_u):.6g}], "
        f"t_end={unrestricted.t[-1]:.3g} fs, success={unrestricted.success}"
    )


if __name__ == "__main__":
    main()
