"""Run CE2 correlation dynamics for several molecule numbers.

The default initial condition is the same as CE2_many_molecules.py:
|1 photon> x |g_1 ... g_N>, with fixed collective coupling 0.1 eV.
"""
import numpy as np

import CE2_many_molecules as ce2


N_VALUES = np.array([2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200], dtype=int)


def run_for_N(N_value):
    ce2.N = int(N_value)
    ce2.g_eV = ce2.g_collective_eV / np.sqrt(ce2.N)
    ce2.g = ce2.eV_to_rate(ce2.g_eV)

    y0 = ce2.initial_state_fock_one_photon()
    sol = ce2.solve_ivp(
        ce2.rhs,
        (ce2.t0, ce2.t1),
        y0,
        t_eval=ce2.t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"N={N_value}: {sol.message}")

    Y = sol.y.T.copy()
    moments = Y[:, : 2 * ce2.N_COMPLEX].view(np.complex128)

    s = moments[:, 2]
    M = moments[:, 7]
    P = moments[:, 8]
    R = moments[:, 9]
    w = Y[:, 2 * ce2.N_COMPLEX]
    n_ph = Y[:, 2 * ce2.N_COMPLEX + 1]
    Z = Y[:, 2 * ce2.N_COMPLEX + 2]

    M_conn = M - s**2
    P_conn = P - np.abs(s) ** 2
    R_conn = R - s * w
    Z_conn = Z - w**2

    total_exc = n_ph + 0.5 * ce2.N * (1.0 + w)
    drift = np.max(np.abs(total_exc - total_exc[0]))

    return {
        "N": ce2.N,
        "g_single_eV": ce2.g_eV,
        "t": sol.t,
        "M": M,
        "P": P,
        "R": R,
        "Z": Z,
        "M_conn": M_conn,
        "P_conn": P_conn,
        "R_conn": R_conn,
        "Z_conn": Z_conn,
        "total_excitation_drift": drift,
    }


def main():
    runs = [run_for_N(N) for N in N_VALUES]
    t = runs[0]["t"]

    P_conn = np.vstack([run["P_conn"] for run in runs])
    Z_conn = np.vstack([run["Z_conn"] for run in runs])
    M_conn = np.vstack([run["M_conn"] for run in runs])
    R_conn = np.vstack([run["R_conn"] for run in runs])

    peak_P = np.max(np.abs(P_conn), axis=1)
    peak_Z = np.max(np.abs(Z_conn), axis=1)

    np.savez(
        "correlation_sweep.npz",
        N=N_VALUES,
        t=t,
        P_conn=P_conn,
        Z_conn=Z_conn,
        M_conn=M_conn,
        R_conn=R_conn,
        peak_P=peak_P,
        peak_Z=peak_Z,
        g_single_eV=np.array([run["g_single_eV"] for run in runs]),
        total_excitation_drift=np.array([run["total_excitation_drift"] for run in runs]),
    )
    np.savetxt(
        "correlation_scaling_summary.dat",
        np.column_stack([N_VALUES, peak_P, N_VALUES * peak_P, np.sqrt(N_VALUES) * peak_P, peak_Z]),
        header="N max_abs_P_conn N_times_max_abs_P_conn sqrtN_times_max_abs_P_conn max_abs_Z_conn",
    )

    print("Saved correlation_sweep.npz")
    for run, peak_p, peak_z in zip(runs, peak_P, peak_Z):
        print(
            f"N={run['N']:>3d}  max|P_c|={peak_p:.6e}  "
            f"max|Z_c|={peak_z:.6e}  "
            f"max|Delta N_exc|={run['total_excitation_drift']:.2e}"
        )


if __name__ == "__main__":
    main()
