"""
SE 4-state model dynamics for Sec. II-B of the mean-field paper.

Basis: {|B0>, |B1>, |C0>, |C1>}  (Eq. 18 in main text)
  |B0> = 0-photon, bright exciton in vib-a state  (energy: omega0)
  |B1> = 0-photon, bright exciton in vib-b state  (energy: omega0 + nu)
  |C0> = 1-photon, ground electronic, all vib-a   (energy: omegac)
  |C1> = 1-photon, ground electronic, one vib-b   (energy: omegac + nu)

Hamiltonian (Eq. 18):
  H = [[ omega0,      cv,         gc*sqrtN,  0          ],
       [ cv,          omega0+nu,  0,         gc         ],
       [ gc*sqrtN,    0,          omegac,    0          ],
       [ 0,           gc,         0,         omegac+nu  ]]

Initial state: |C0> = (0, 0, 1, 0)

Parameters match Fig. 3 of the paper:
  N=4, sqrt(N)*gc = 0.10 eV, cv = 20 meV, omega0 = omegac = 2.0 eV
"""

import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------
# Constants and parameters
# --------------------------------------------------------------------------
HBAR_eVfs = 0.658211951          # ħ [eV·fs]

N         = 4
omega0    = 2.0                  # eV
omegac    = 2.0                  # eV
gc        = 0.10 / np.sqrt(N)   # eV  (so that sqrt(N)*gc = 0.10 eV)
cv        = 0.020                # eV  (20 meV vibronic coupling)

# ν values for time-trace panels (Fig. 3a)
NU_TRACES = [0.190, 0.195, 0.199]

# ν scan range for spectral panel (Fig. 3b)
NU_SCAN   = np.linspace(0.16, 0.24, 300)

# Time grid
T_MAX  = 2000.0                  # fs
DT     = 0.5                     # fs
times  = np.arange(0.0, T_MAX + DT / 2, DT)


# --------------------------------------------------------------------------
# 4-state Hamiltonian (Eq. 18)  -- real symmetric
# --------------------------------------------------------------------------
def build_H(nu):
    sqrtN = np.sqrt(N)
    return np.array(
        [
            [omega0,        cv,          gc * sqrtN,  0.0         ],
            [cv,            omega0 + nu, 0.0,         gc          ],
            [gc * sqrtN,    0.0,         omegac,      0.0         ],
            [0.0,           gc,          0.0,         omegac + nu ],
        ],
        dtype=float,
    )


# --------------------------------------------------------------------------
# Exact time evolution via diagonalisation
# Returns P_B1(t) = |<B1|psi(t)>|^2  for every t in `times`
# --------------------------------------------------------------------------
def propagate(nu, times):
    H = build_H(nu)
    eigvals, eigvecs = np.linalg.eigh(H)   # eigvecs[:,k] = k-th eigenvector (real)

    # Initial state |C0> = (0,0,1,0); project onto eigenbasis
    psi0     = np.array([0.0, 0.0, 1.0, 0.0])
    psi0_eig = eigvecs.T @ psi0            # shape (4,)

    # Vectorised propagation over all times
    # phases[t, k] = exp(-i E_k t / hbar)
    phases    = np.exp(-1j * np.outer(times, eigvals) / HBAR_eVfs)  # (T, 4)
    psi_eig_t = phases * psi0_eig[np.newaxis, :]                    # (T, 4)
    psi_t     = psi_eig_t @ eigvecs.T                               # (T, 4)

    P_B1 = np.abs(psi_t[:, 1]) ** 2       # population of |B1>
    return P_B1


# --------------------------------------------------------------------------
# 1. Time traces for selected ν values  →  SE_4states/nu_X.XXX/
# --------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

for nu in NU_TRACES:
    folder = script_dir / f"nu_{nu:.3f}"
    folder.mkdir(exist_ok=True)

    P_B1   = propagate(nu, times)
    Evib_E = nu * P_B1                    # vibrational energy on excited-state surface

    np.savetxt(folder / "Pe1_vs_t.dat",    np.column_stack([times, P_B1]),
               header="time(fs)  P_B1(t)",    fmt="%.12e")
    np.savetxt(folder / "Evib_E_vs_t.dat", np.column_stack([times, Evib_E]),
               header="time(fs)  Evib_E(eV)", fmt="%.12e")

    print(f"  nu = {nu:.3f} eV  |  P_B1 max = {P_B1.max():.4f}"
          f"  |  T_osc ~ {2.0 * np.pi * HBAR_eVfs / (cv / (2 * np.sqrt(N))):.0f} fs (est.)")

# --------------------------------------------------------------------------
# 2. Spectral response: time-averaged E_vib,E vs ν  →  SE_4states/
# --------------------------------------------------------------------------
Evib_avg = np.zeros(len(NU_SCAN))
for i, nu in enumerate(NU_SCAN):
    P_B1         = propagate(nu, times)
    Evib_avg[i]  = np.mean(nu * P_B1)

np.savetxt(script_dir / "Evib_vs_nu.dat",
           np.column_stack([NU_SCAN, Evib_avg]),
           header="nu(eV)  <Evib_E>(eV)", fmt="%.12e")

print("Done.  Data saved to SE_4states/")
