"""Permutation-symmetric CE2 dynamics for an N-molecule TC model.

This is the many-molecule CE2 hierarchy described in Discussion.txt.  It
retains the intermolecular two-body moments M, P, R, and Z, which are absent
from the single-JC CE2 scripts in the n0_* folders.

Default run:
  N = 5 molecules
  N_exc = 1, initialized as |1 photon> x |g_1 ... g_N>
  resonant, lossless parameters matching Fig3/N1_Nexc1_CE2-CE5
  fixed collective coupling g_collective = 0.1 eV
"""
import numpy as np
from scipy.integrate import solve_ivp


HBAR_eVfs = 0.658211951


def eV_to_rate(x):
    return x / HBAR_eVfs


# ---------- parameters matching Fig3/N1_Nexc1_CE2-CE5 ----------
N = 5
N_exc = 1.0

omega_c = 2.0
omega_0 = 2.0
omega_d = 2.0

# Same collective Rabi scale as the N=1 CE2-CE5 panel.  For N=1 this is
# identical to g_eV = 0.1; for N>1 the per-molecule coupling is scaled.
g_collective_eV = 0.1
g_eV = g_collective_eV / np.sqrt(N)

kappa = 0.0
gamma1 = 0.0
gamma_phi = 0.0
eta = 0.0 + 0.0j

Delta_c = eV_to_rate(omega_c - omega_d)
Delta_x = eV_to_rate(omega_0 - omega_d)
g = eV_to_rate(g_eV)
gamma_perp = gamma1 / 2.0 + gamma_phi

t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)


# State order:
#   complex: alpha, m, s, V, C, D, U, M, P, R
#   real:    w, n, Z
# where i != j:
#   alpha = <a>, m = <a^2>, n = <a^dag a>
#   s = <sigma_i^->, w = <sigma_i^z>
#   V = <a sigma_i^->, C = <a sigma_i^+>, D = <a^dag sigma_i^->
#   U = <a sigma_i^z>
#   M = <sigma_i^- sigma_j^->, P = <sigma_i^+ sigma_j^->
#   R = <sigma_i^- sigma_j^z>, Z = <sigma_i^z sigma_j^z>
N_COMPLEX = 10


def pack(alpha, m, s, V, C, D, U, M, P, R, w, n, Z):
    zc = np.array([alpha, m, s, V, C, D, U, M, P, R], dtype=np.complex128)
    return np.concatenate([zc.view(np.float64), np.array([w, n, Z], dtype=np.float64)])


def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[: 2 * N_COMPLEX]).view(np.complex128)
    alpha, m, s, V, C, D, U, M, P, R = zc
    w, n, Z = y[2 * N_COMPLEX :]
    return alpha, m, s, V, C, D, U, M, P, R, float(w), float(n), float(Z)


def rhs(t, y):
    alpha, m, s, V, C, D, U, M, P, R, w, n, Z = unpack(y)

    alpha_c = np.conjugate(alpha)
    s_c = np.conjugate(s)
    U_c = np.conjugate(U)
    D_c = np.conjugate(D)
    R_c = np.conjugate(R)
    P_c = np.conjugate(P)

    abs_alpha2 = abs(alpha) ** 2
    abs_s2 = abs(s) ** 2

    # CE2 closures for field-molecule triplets.
    A_z = n * w + alpha * U_c + alpha_c * U - 2.0 * abs_alpha2 * w
    A_z_plus = (n + 1.0) * w + alpha * U_c + alpha_c * U - 2.0 * abs_alpha2 * w
    B_minus = (n + 1.0) * s + alpha_c * V + alpha * D - 2.0 * abs_alpha2 * s
    B_plus = m * s_c + 2.0 * alpha * C - 2.0 * alpha**2 * s_c
    A_2z = m * w + 2.0 * alpha * U - 2.0 * alpha**2 * w

    # CE2 closures for field-spin-spin triplets.
    X_R = alpha * R + U * s + V * w - 2.0 * alpha * w * s
    X_plus = alpha * R_c + C * w + U * s_c - 2.0 * alpha * w * s_c
    Y_minus = alpha_c * R + U_c * s + D * w - 2.0 * alpha_c * w * s
    X_Z = alpha * Z + 2.0 * w * U - 2.0 * alpha * w**2
    Y_M = alpha_c * M + 2.0 * D * s - 2.0 * alpha_c * s**2
    X_P = alpha * P_c + V * s_c + C * s - 2.0 * alpha * abs_s2

    # One-body moments.
    dalpha = -(kappa / 2.0 + 1j * Delta_c) * alpha - 1j * g * N * s + eta
    dm = -(kappa + 2j * Delta_c) * m - 2j * g * N * V + 2.0 * eta * alpha
    ds = -(gamma_perp + 1j * Delta_x) * s + 1j * g * U
    dw = -gamma1 * (w - (-1.0)) + 2j * g * (D - C)
    dn = -kappa * n + 1j * g * N * (C - D) + 2.0 * np.real(np.conjugate(eta) * alpha)

    # Field-molecule two-body moments.
    dV = (
        -(kappa / 2.0 + gamma_perp + 1j * (Delta_c + Delta_x)) * V
        - 1j * g * (N - 1) * M
        + 1j * g * A_2z
        + eta * s
    )
    dC = (
        -(kappa / 2.0 + gamma_perp + 1j * (Delta_c - Delta_x)) * C
        - 1j * g * (A_z_plus + (1.0 - w) / 2.0 + (N - 1) * P)
        + eta * s_c
    )
    dD = (
        -(kappa / 2.0 + gamma_perp - 1j * (Delta_c - Delta_x)) * D
        + 1j * g * (A_z + (1.0 + w) / 2.0 + (N - 1) * P_c)
        + np.conjugate(eta) * s
    )
    dU = (
        -(kappa / 2.0 + 1j * Delta_c + gamma1) * U
        - 1j * g * (s + (N - 1) * R)
        + 2j * g * B_minus
        - 2j * g * B_plus
        + eta * w
    )

    # Intermolecular two-body moments.  The gamma terms vanish for the present
    # lossless run, but keeping the linear parts makes the state layout explicit.
    dM = -2.0 * (gamma_perp + 1j * Delta_x) * M + 2j * g * X_R
    dP = -2.0 * gamma_perp * P + 1j * g * (X_plus - Y_minus)
    dR = (
        -(gamma_perp + gamma1 + 1j * Delta_x) * R
        + gamma1 * (-1.0) * s
        + 1j * g * X_Z
        + 2j * g * Y_M
        - 2j * g * X_P
    )
    dZ = -2.0 * gamma1 * (Z - (-1.0) * w) + 4j * g * (Y_minus - X_plus)

    return pack(
        dalpha,
        dm,
        ds,
        dV,
        dC,
        dD,
        dU,
        dM,
        dP,
        dR,
        dw.real,
        dn.real,
        dZ.real,
    )


def initial_state_fock_one_photon():
    alpha0 = 0.0 + 0.0j
    m0 = 0.0 + 0.0j
    n0 = N_exc

    s0 = 0.0 + 0.0j
    w0 = -1.0

    V0 = 0.0 + 0.0j
    C0 = 0.0 + 0.0j
    D0 = 0.0 + 0.0j
    U0 = 0.0 + 0.0j

    M0 = 0.0 + 0.0j
    P0 = 0.0 + 0.0j
    R0 = 0.0 + 0.0j
    Z0 = 1.0

    return pack(alpha0, m0, s0, V0, C0, D0, U0, M0, P0, R0, w0, n0, Z0)


def exact_single_excitation_reference(times):
    """Closed TC reference for |1 photon, all ground> in the one-excitation sector."""
    omega_rabi = g * np.sqrt(N)
    n_ph = np.cos(omega_rabi * times) ** 2
    n_ex_total = np.sin(omega_rabi * times) ** 2
    return n_ph, n_ex_total


def main():
    y0 = initial_state_fock_one_photon()
    sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)

    Y = sol.y.T.copy()
    Zc = Y[:, : 2 * N_COMPLEX].view(np.complex128)

    alpha = Zc[:, 0]
    m = Zc[:, 1]
    s = Zc[:, 2]
    V = Zc[:, 3]
    C = Zc[:, 4]
    D = Zc[:, 5]
    U = Zc[:, 6]
    M = Zc[:, 7]
    P = Zc[:, 8]
    R = Zc[:, 9]

    w = Y[:, 2 * N_COMPLEX]
    n = Y[:, 2 * N_COMPLEX + 1]
    Zspin = Y[:, 2 * N_COMPLEX + 2]

    excited_per_molecule = 0.5 * (1.0 + w)
    excited_total = N * excited_per_molecule
    total_excitation = n + excited_total

    exact_n_ph, exact_n_ex_total = exact_single_excitation_reference(sol.t)

    np.savetxt("Photon_number_CE2.dat", np.column_stack([sol.t, n]))
    np.savetxt("Excited_population_CE2.dat", np.column_stack([sol.t, excited_total]))
    np.savetxt(
        "Excited_population_per_molecule_CE2.dat",
        np.column_stack([sol.t, excited_per_molecule]),
    )
    np.savetxt("Total_excitation_CE2.dat", np.column_stack([sol.t, total_excitation]))
    np.savetxt("alpha_CE2.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
    np.savetxt("Photon_number_exact.dat", np.column_stack([sol.t, exact_n_ph]))
    np.savetxt("Excited_population_exact.dat", np.column_stack([sol.t, exact_n_ex_total]))
    np.savetxt(
        "CE2_observables.dat",
        np.column_stack(
            [
                sol.t,
                n,
                excited_total,
                excited_per_molecule,
                total_excitation,
                Zspin,
                np.real(P),
                np.imag(P),
            ]
        ),
        header="t_fs photon_number total_excited_population excited_per_molecule total_excitation Z ReP ImP",
    )
    np.savez(
        "CE2_moments.npz",
        t=sol.t,
        alpha=alpha,
        m=m,
        s=s,
        V=V,
        C=C,
        D=D,
        U=U,
        M=M,
        P=P,
        R=R,
        w=w,
        n=n,
        Z=Zspin,
    )

    max_exc_drift = np.max(np.abs(total_excitation - total_excitation[0]))
    max_exact_photon_deviation = np.max(np.abs(n - exact_n_ph))
    print("Saved CE2 many-molecule data for N=5, N_exc=1.")
    print(f"g_single_eV = {g_eV:.12g}, g_collective_eV = {g_collective_eV:.12g}")
    print(f"max |Delta N_exc| = {max_exc_drift:.3e}")
    print(f"max CE2 photon deviation from one-excitation TC reference = {max_exact_photon_deviation:.3e}")


if __name__ == "__main__":
    main()
