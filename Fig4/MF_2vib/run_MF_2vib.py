"""
MF single-molecule model with NVIB=2 vibrational Fock states.

Single-molecule space: {|g,n>, |e,n>}  for  n = 0, 1
Total: 2 * NVIB = 4 states per molecule.

EOMs  (hbar = HBAR_eVfs, energies in eV, time in fs)
  d alpha   / dt = -(i/hbar) [omegac * alpha  +  gc*N * sigma]
  d c_{g,n} / dt = -(i/hbar) [n*nu * c_{g,n}  +  gc * alpha* * c_{e,n}]
  d c_{e,n} / dt = -(i/hbar) [(omega0 + n*nu) * c_{e,n}
                               + gc * alpha * c_{g,n}
                               + cv * (sqrt(n)   * c_{e,n-1}   [if n>0]
                                     + sqrt(n+1) * c_{e,n+1})  [if n<NVIB-1]]
where sigma = sum_n  conj(c_{g,n}) * c_{e,n}

y layout:  [Re(alpha), Im(alpha),
            Re(c_g0), Im(c_g0), Re(c_e0), Im(c_e0),
            Re(c_g1), Im(c_g1), Re(c_e1), Im(c_e1)]
  => 2 + 4*NVIB = 10 elements

Initial condition: alpha(0) = sqrt(n0),  c_g0 = 1,  all others = 0.

Saves |c_{e,1}|^2 / n0  (normalised excited v=1 population) to disk.
Output: ceb_sq_n0_<tag>_2vib.dat  in the same folder as this script.
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
HBAR_eVfs = 0.658211951
omega0    = 2.0
omegac    = 2.0
sqrtN_gc  = 0.10
cv        = 0.020
NU        = 0.199
N         = 1
gc        = sqrtN_gc / np.sqrt(N)

NVIB   = 2    # vibrational Fock states: n = 0, 1

T_MAX  = 1000.0
DT     = 0.5
t_eval = np.arange(0.0, T_MAX + DT / 2, DT)


# --------------------------------------------------------------------------
# RHS
# --------------------------------------------------------------------------
def make_rhs(nu):
    h  = HBAR_eVfs

    def rhs(_t, y):
        alpha = y[0] + 1j * y[1]
        c_g = np.array([y[2 + 4*n] + 1j * y[3 + 4*n] for n in range(NVIB)])
        c_e = np.array([y[4 + 4*n] + 1j * y[5 + 4*n] for n in range(NVIB)])

        sigma   = np.sum(np.conj(c_g) * c_e)
        d_alpha = (-1j / h) * (omegac * alpha + gc * N * sigma)

        d_c_g = np.array([
            (-1j / h) * (n * nu * c_g[n] + gc * np.conj(alpha) * c_e[n])
            for n in range(NVIB)
        ])

        d_c_e = np.empty(NVIB, dtype=complex)
        for n in range(NVIB):
            val = (omega0 + n * nu) * c_e[n] + gc * alpha * c_g[n]
            if n > 0:
                val += cv * np.sqrt(n)     * c_e[n - 1]
            if n < NVIB - 1:
                val += cv * np.sqrt(n + 1) * c_e[n + 1]
            d_c_e[n] = (-1j / h) * val

        out = [d_alpha.real, d_alpha.imag]
        for n in range(NVIB):
            out += [d_c_g[n].real, d_c_g[n].imag,
                    d_c_e[n].real, d_c_e[n].imag]
        return out

    return rhs


# --------------------------------------------------------------------------
# Run for each n0
# --------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# n0=1e-6 saved flat in this folder (used by Fig 2c)
# n0=1e-1..1e-4 each saved into their own subfolder (used by Fig S2)
n0_vals = {"1e-6": 1e-6, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2, "1e-1": 1e-1}

rhs = make_rhs(NU)

for tag, n0 in n0_vals.items():
    alpha0 = np.sqrt(n0)

    # initial condition: alpha=alpha0, c_g0=1, rest=0
    y0  = [alpha0, 0.0, 1.0, 0.0, 0.0, 0.0]   # alpha, c_g0, c_e0
    y0 += [0.0, 0.0, 0.0, 0.0]                 # c_g1=0, c_e1=0

    sol = solve_ivp(rhs, (0.0, T_MAX), y0,
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)

    # c_{e,1} indices: Re=y[8], Im=y[9]  (= 4+4*1, 5+4*1)
    c_e1   = sol.y[8] + 1j * sol.y[9]
    ceb_sq = np.abs(c_e1) ** 2 / n0   # normalised by n0

    if tag == "1e-6":
        # flat file for Fig 2c compatibility
        folder = script_dir
        fname  = folder / f"ceb_sq_n0_{tag}_2vib.dat"
    else:
        folder = script_dir / f"n0={tag}"
        folder.mkdir(exist_ok=True)
        fname  = folder / "ceb_sq_2vib.dat"

    np.savetxt(fname,
               np.column_stack([sol.t, ceb_sq]),
               header=f"time(fs)  |c_e1|^2/n0  (N=1, n0={tag}, NVIB=2)",
               fmt="%.12e")
    print(f"n0={tag}  |c_e1|^2/n0 max={ceb_sq.max():.4e}  -> {fname.relative_to(script_dir)}")

print("Done.")
