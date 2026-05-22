"""
SE 4-state model: fixed nu, varying N.

Observables saved per N folder
  C0_vs_t.dat        -- time [fs], P_{C0}(t) = |<C0|psi>|^2
  C1_vs_t.dat        -- time [fs], P_{C1}(t) = |<C1|psi>|^2
  Photon_number.dat  -- time [fs], P_{C0}+P_{C1}  (total photon number)

Parameters
  sqrt(N)*gc = 0.10 eV  (collective coupling fixed)
  nu         = 0.199 eV (fixed vibrational frequency, near Rabi resonance)
  cv         = 20  meV
  omega0 = omegac = 2.0 eV
  initial state: |C0> = (0,0,1,0) in {|B0>,|B1>,|C0>,|C1>}
"""

import numpy as np
from pathlib import Path

HBAR_eVfs  = 0.658211951          # ħ [eV·fs]
omega0     = 2.0                  # eV
omegac     = 2.0                  # eV
sqrtN_gc   = 0.10                 # eV (collective coupling, fixed)
cv         = 0.020                # eV (20 meV)
NU         = 0.199                # eV (fixed)

N_VALUES   = [1, 2, 4, 8, 10, 100, 1000, 10000, 100000, 10000000]

T_MAX  = 2000.0                   # fs
DT     = 0.5
times  = np.arange(0.0, T_MAX + DT / 2, DT)

psi0 = np.array([0.0, 0.0, 1.0, 0.0])   # |C0>


def build_H(N, nu):
    gc    = sqrtN_gc / np.sqrt(N)
    sqrtN = np.sqrt(N)
    return np.array(
        [
            [omega0,       cv,          gc * sqrtN,   0.0         ],
            [cv,           omega0 + nu, 0.0,           gc          ],
            [gc * sqrtN,   0.0,         omegac,        0.0         ],
            [0.0,          gc,          0.0,           omegac + nu ],
        ],
        dtype=float,
    )


def propagate(N, nu, times):
    H                  = build_H(N, nu)
    eigvals, eigvecs   = np.linalg.eigh(H)          # real eigvecs (H real symmetric)
    psi0_eig           = eigvecs.T @ psi0
    phases             = np.exp(-1j * np.outer(times, eigvals) / HBAR_eVfs)
    psi_t              = (phases * psi0_eig[np.newaxis, :]) @ eigvecs.T   # (T, 4)
    P                  = np.abs(psi_t) ** 2          # (T, 4)  columns: B0,B1,C0,C1
    return P                                         # P[:,2]=C0, P[:,3]=C1


script_dir = Path(__file__).resolve().parent

for N in N_VALUES:
    folder = script_dir / f"N={N}"
    folder.mkdir(exist_ok=True)

    P    = propagate(N, NU, times)
    P_B0 = P[:, 0]
    P_B1 = P[:, 1]
    P_C0 = P[:, 2]
    P_C1 = P[:, 3]
    phot = P_C0 + P_C1

    np.savetxt(folder / "B0_vs_t.dat",       np.column_stack([times, P_B0]),
               header="time(fs)  P_B0(t)",      fmt="%.12e")
    np.savetxt(folder / "B1_vs_t.dat",       np.column_stack([times, P_B1]),
               header="time(fs)  P_B1(t)",      fmt="%.12e")
    np.savetxt(folder / "C0_vs_t.dat",       np.column_stack([times, P_C0]),
               header="time(fs)  P_C0(t)",      fmt="%.12e")
    np.savetxt(folder / "C1_vs_t.dat",       np.column_stack([times, P_C1]),
               header="time(fs)  P_C1(t)",      fmt="%.12e")
    np.savetxt(folder / "Photon_number.dat",  np.column_stack([times, phot]),
               header="time(fs)  photon_number", fmt="%.12e")

    gc = sqrtN_gc / np.sqrt(N)
    print(f"N={N:5d}  gc={gc:.4f} eV"
          f"  P_C1_max={P_C1.max():.4e}")

print("Done.")
