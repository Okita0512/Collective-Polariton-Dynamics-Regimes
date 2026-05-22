import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------------- parameters ----------------
omega_c = 2.0   # eV  (not actually used in the rotating-frame equations below)
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa     = 0.0   # fs^-1
gamma1    = 0.0   # fs^-1
gamma_phi = 0.0   # fs^-1

# For this benchmark we assume:
#   - no losses: kappa=gamma1=gamma_phi=0
#   - resonant: omega_c = omega_0
#   - no coherent drive: eta = 0
# and we start from |1_ph, g>

g = eV_to_rate(g_eV)  # Rabi coupling in fs^-1

# ---------------- single-excitation JC amplitudes ----------------
# State restricted to |1,g> and |0,e>:
#   |ψ(t)> = c1(t) |1,g> + c0(t) |0,e>
#
# Exact equations of motion (rotating frame, resonance, no loss):
#   i d/dt c1 = g c0
#   i d/dt c0 = g c1

def rhs_amp(t, y):
    # y = [Re c1, Im c1, Re c0, Im c0]
    c1 = y[0] + 1j*y[1]
    c0 = y[2] + 1j*y[3]

    dc1 = -1j * g * c0
    dc0 = -1j * g * c1

    return np.array([dc1.real, dc1.imag, dc0.real, dc0.imag], dtype=float)

# initial condition: |ψ(0)> = |1,g> -> c1=1, c0=0
y0 = np.array([1.0, 0.0,   # c1(0)
               0.0, 0.0])  # c0(0)

t_span = (0.0, 200.0)    # fs
t_eval = np.linspace(t_span[0], t_span[1], 2001)

sol = solve_ivp(rhs_amp, t_span, y0, t_eval=t_eval,
                rtol=1e-9, atol=1e-12)

t = sol.t
c1 = sol.y[0] + 1j*sol.y[1]
c0 = sol.y[2] + 1j*sol.y[3]

# ---------------- map back to "CE" observables ----------------
# Photon and exciton populations
n_ph = np.abs(c1)**2
P_e  = np.abs(c0)**2

# TLS inversion w = P_e - P_g; here P_g = n_ph (only |1,g> present in ground)
w = P_e - n_ph

# For compatibility with your previous CE code:
alpha = np.zeros_like(c1)  # <a> = 0 in this manifold
# (If you want C, D, etc., here are the exact formulas:)
C  = c1 * np.conjugate(c0)     # <a σ_+>
D  = np.conjugate(C)           # <a† σ_->
Nw = -n_ph                     # <n σ_z>
n2 = n_ph                      # <n^2>

# ---------------- save in same format as old CE4 script ----------------
np.savetxt("Photon_number_JC.dat", np.column_stack([t, n_ph]))
np.savetxt("Excited_population_JC.dat", np.column_stack([t, P_e]))
np.savetxt("alpha_JC.dat",
           np.column_stack([t, alpha.real, alpha.imag]))

print("Saved: Photon_number_JC.dat, Excited_population_JC.dat, alpha_JC.dat")