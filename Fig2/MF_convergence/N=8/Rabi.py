import numpy as np
from numpy import kron
from scipy.integrate import solve_ivp

# ---------------- constants ----------------
HBAR_eVfs = 0.658211951  # ħ [eV·fs]
def eV_to_rate(e_eV):    # energy [eV] -> angular frequency [1/fs]
    return e_eV / HBAR_eVfs

# ---------------- user parameters ----------------
N           = 8          # number of identical, uncorrelated molecules
vib_trunc   = 5          # vibrational cutoff per molecule
omega_c_eV  = 2.000       # cavity frequency [eV]
omega_0_eV  = 2.000       # exciton frequency [eV]
omega_v_eV  = 0.200       # vibrational frequency [eV] (used if vib_trunc>=2)
g_eV        = 0.10 / np.sqrt(N)      # light–matter coupling [eV]
# c_v_eV      = 0.02       # Holstein coupling [eV], set 0 for clean Rabi
c_v_eV      = 0.0       # Holstein coupling [eV], set 0 for clean Rabi

# rotating frame frequency (pick equal to cavity/exciton for resonance)
omega_d_eV  = 2.000      # choose =omega_c_eV=omega_0_eV for Δ=0 (resonant Rabi)

# no dissipation, no external drive:
kappa_rate     = 0.0     # cavity loss [1/fs]
gamma1_rate    = 0.0     # |e>→|g| [1/fs]
gamma_phi_rate = 0.0     # pure dephasing [1/fs]
gamma_v_rate   = 0.0     # vib damping [1/fs]
n_th           = 0.0     # bath occupancy

eta            = 0.0 + 0.0j  # external cavity drive [1/fs]; keep 0
seed_alpha     = 1.0         # tiny seed to trigger Rabi in mean-field

# time window
t0, t1 = 0.0, 200.0     # fs
num_pts = 4001
t_eval = np.linspace(t0, t1, num_pts)

# ---------------- derived (rates in 1/fs) ----------------
Delta_c = eV_to_rate(omega_c_eV - omega_d_eV)
Delta_x = eV_to_rate(omega_0_eV - omega_d_eV)
omega_v = eV_to_rate(omega_v_eV)
g       = eV_to_rate(g_eV)
c_v     = eV_to_rate(c_v_eV)

# ---------------- one-molecule vibronic Hilbert space ----------------
d_v = vib_trunc
Iv  = np.eye(d_v, dtype=complex)
# vib operators (if d_v=1, these are zeros)
a = np.zeros((d_v, d_v), dtype=complex)
for n in range(1, d_v):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
n_op = adag @ a

# electronic operators
eg = np.array([[0,0],[0,1]], dtype=complex)  # |e><e|
gg = np.array([[1,0],[0,0]], dtype=complex)  # |g><g|
sig_minus = np.array([[0,1],[0,0]], dtype=complex)  # |g><e|
sig_plus  = sig_minus.conj().T
sig_z     = np.array([[-1,0],[0,1]], dtype=complex)

I_e  = np.eye(2, dtype=complex)
dim  = 2 * d_v

def E(op_e): return kron(op_e, Iv)   # electronic ⊗ I_v
def V(op_v): return kron(I_e, op_v)  # I_e ⊗ vibrational

SigmaMinus = E(sig_minus)
SigmaPlus  = E(sig_plus)
Proj_g     = E(gg)
Proj_e     = E(eg)
Nhat       = V(n_op)
B          = V(a)
Bd         = V(adag)

# projectors onto |g,n> and |e,n> vibrational states
P_gn = []
P_en = []  # NEW: projectors onto |e,n>
for n in range(d_v):
    n_proj = np.zeros((d_v, d_v), dtype=complex); n_proj[n, n] = 1.0
    P_gn.append(Proj_g @ V(n_proj))
    P_en.append(Proj_e @ V(n_proj))  # NEW

# Hamiltonian for one molecule in rotating frame at ω_d (ħ=1 in rate units)
# H = Δx |e><e| + ωv b†b + c_v |e><e| (b+b†) + g( α σ^+ + α* σ^- )
def H_mol(alpha):
    H = Delta_x * Proj_e + omega_v * Nhat
    if c_v != 0.0:
        H = H + c_v * Proj_e @ (B + Bd)
    H = H + g * (alpha * SigmaPlus + np.conj(alpha) * SigmaMinus)
    return H

# Lindblad superoperator (kept but with zero rates here)
def D(L, rho):
    return L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)

L_relax = np.sqrt(gamma1_rate)    * SigmaMinus
L_phi   = np.sqrt(0.5*gamma_phi_rate) * (E(sig_z))
L_vdown = np.sqrt(gamma_v_rate*(n_th+1.0)) * B
L_vup   = np.sqrt(gamma_v_rate*n_th) * Bd

collapses = []
for L in [L_relax, L_phi, L_vdown] + ([L_vup] if n_th>0 else []):
    if np.any(np.abs(L) > 0):
        collapses.append(L)

def expect(op, rho): return np.trace(op @ rho)

# ---------------- ODE: y = [alpha, vec(rho)] ----------------
def rhs(t, y):
    alpha = y[0]
    rho   = y[1:].reshape((dim, dim))

    # molecule master equation
    H  = H_mol(alpha)
    drho = -1j*(H @ rho - rho @ H)
    for L in collapses:
        drho += D(L, rho)

    # cavity mean-field amplitude
    sigma_minus = expect(SigmaMinus, rho)
    dalpha = -(kappa_rate/2 + 1j*Delta_c)*alpha - 1j * g * N * sigma_minus + eta

    return np.concatenate(([dalpha], drho.reshape(-1)))

# ---------------- initial condition: molecule in |g,0>, cavity tiny seed ----------------
ket_e = np.array([0,1], dtype=complex)
ket_g = np.array([1,0], dtype=complex)
ket_v0 = np.zeros((d_v,), dtype=complex); ket_v0[0] = 1.0
psie = kron(ket_e, ket_v0)            # |e,0>
psig = kron(ket_g, ket_v0)            # |g,0>

rho0 = np.outer(psig, psig.conj())

alpha0 = seed_alpha                   # tiny seed to kick-start MF dynamics
y0 = np.concatenate(([alpha0], rho0.reshape(-1)))

# ---------------- integrate ----------------
sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10, method="RK45")
alpha_t = sol.y[0, :]
rho_t   = sol.y[1:, :].T.reshape((-1, dim, dim))

# ---------------- observables & save ----------------
excited_pop = np.real([expect(Proj_e, rho) for rho in rho_t])   # ⟨e|ρ|e⟩
photon_num  = np.abs(alpha_t)**2                                # |α|^2
n_avg       = np.real([expect(Nhat, rho) for rho in rho_t])     # ⟨b†b⟩

# NEW: vibrational occupation split by electronic surface
n_g_avg = np.real([expect(Proj_g @ Nhat, rho) for rho in rho_t])  # ⟨N_hat in g⟩
n_e_avg = np.real([expect(Proj_e @ Nhat, rho) for rho in rho_t])  # ⟨N_hat in e⟩

# sanity: n_avg ≈ n_g_avg + n_e_avg (numerically)
# print(np.max(np.abs(n_avg - (n_g_avg + n_e_avg))))

# vibrational energies (per molecule, above zero-point), in eV
Evib_per_mol_eV    = (omega_v * n_avg)   * HBAR_eVfs
Evib_g_per_mol_eV  = (omega_v * n_g_avg) * HBAR_eVfs  # NEW
Evib_e_per_mol_eV  = (omega_v * n_e_avg) * HBAR_eVfs  # NEW

# total (ensemble) vibrational energies [eV]
Evib_total_eV      = N * Evib_per_mol_eV
Evib_g_total_eV    = N * Evib_g_per_mol_eV   # NEW
Evib_e_total_eV    = N * Evib_e_per_mol_eV   # NEW

Xg_eV     = np.real([expect((B + Bd) @ Proj_g, rho) for rho in rho_t])
Vg_eV     = np.real([expect((B + Bd) @ (B + Bd) @ Proj_g, rho) for rho in rho_t])

# vib populations on g and e
vib_pops_g = np.empty((len(rho_t), d_v))
vib_pops_e = np.empty((len(rho_t), d_v))   # NEW
for ti, rho in enumerate(rho_t):
    for n in range(d_v):
        vib_pops_g[ti, n] = np.real(expect(P_gn[n], rho))
        vib_pops_e[ti, n] = np.real(expect(P_en[n], rho))  # NEW

# ---------------- save to disk ----------------
np.savetxt("Excited_population.dat", np.column_stack([sol.t, excited_pop]))
np.savetxt("Photon_number.dat",      np.column_stack([sol.t, photon_num]))
np.savetxt("Xg.dat",                 np.column_stack([sol.t, Xg_eV]))
np.savetxt("Vg.dat",                 np.column_stack([sol.t, Vg_eV]))

np.savetxt("Vib_pops_g.dat", np.column_stack([sol.t, vib_pops_g]))
np.savetxt("Vib_pops_e.dat", np.column_stack([sol.t, vib_pops_e]))  # NEW

# NEW: total vibrational energy (total, g, e) vs time
# columns: t, E_vib_total, E_vib_g, E_vib_e
# np.savetxt(
#     "Evib_total.dat",
#     np.column_stack([sol.t, Evib_total_eV, Evib_g_total_eV, Evib_e_total_eV])
# )

# (Optional) keep per-molecule file if you still want it
np.savetxt(
    "Evib_per_molecule.dat",
    np.column_stack([sol.t, Evib_per_mol_eV, Evib_g_per_mol_eV, Evib_e_per_mol_eV])
)
