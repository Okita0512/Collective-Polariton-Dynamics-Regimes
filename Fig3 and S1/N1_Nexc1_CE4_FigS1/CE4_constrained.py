"""Exact projected single-excitation algebra for the N=1, N_exc=1 JC benchmark.

Within the N_exc=1 invariant subspace spanned by {|1,g>, |0,e>}, the Hilbert space
is two-dimensional and the Heisenberg equations close exactly at second-moment order.
All higher-moment identities hold exactly:

    alpha = s = U = 0
    w     = 1 - 2n
    Nw    = -n,   n2 = n
    S1 = Q = T1 = T2 = Cn = Na = 0
    Sd = D

The projected equations are:
    dn/dt = i g_c (C - D)
    dC/dt = -i g_c (1 - 2n)
    dD/dt = +i g_c (1 - 2n)

This is NOT a CE4 approximation; it is the exact result within the
single-excitation manifold. The comparison with CE4_naive.py diagnoses the
failure of an unconstrained fourth-order diagnostic closure to preserve the
invariant subspace.

Initial state: |1,g>  =>  n(0)=1, C(0)=0, D(0)=0.
Exact result:  n(t) = cos²(g_c t).
"""
import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

g_eV = 0.1
g    = eV_to_rate(g_eV)

t0, t1 = 0.0, 200.0
t_eval  = np.linspace(t0, t1, 4001)

# ---- state: [C_re, C_im, D_re, D_im, n] ----
def pack_c(C, D, n):
    zc = np.array([C, D], dtype=np.complex128)
    return np.concatenate([zc.view(np.float64), np.array([float(n)])])

def unpack_c(y):
    y  = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:4]).view(np.complex128)
    C, D = zc[0], zc[1]
    n    = y[4]
    return C, D, n

def rhs_constrained(t, y):
    C, D, n = unpack_c(y)
    dn = 1j*g*(C - D)
    dC = -1j*g*(1.0 - 2.0*n)
    dD = +1j*g*(1.0 - 2.0*n)
    zc = np.array([dC, dD], dtype=np.complex128)
    return np.concatenate([zc.view(np.float64), np.array([float(np.real(dn))])])

y0 = pack_c(0+0j, 0+0j, 1.0)

sol = solve_ivp(rhs_constrained, (t0, t1), y0, t_eval=t_eval,
                rtol=1e-10, atol=1e-12)

# ---- unpack solution ----
Y    = sol.y.T.copy()
zc   = np.ascontiguousarray(Y[:, :4]).view(np.complex128)
C_t  = zc[:, 0]; D_t = zc[:, 1]
n_t  = Y[:, 4]

# derived quantities from projected-subspace identities
w_t     = 1.0 - 2.0*n_t
P_e     = 1.0 - n_t            # = (1 + w)/2
alpha_t = np.zeros_like(n_t)   # <a> = 0 in single-excitation subspace

# ---- save outputs ----
np.savetxt("Photon_number_projected_single_excitation.dat",
           np.column_stack([sol.t, n_t]))
np.savetxt("Excited_population_projected_single_excitation.dat",
           np.column_stack([sol.t, P_e]))
np.savetxt("alpha_projected_single_excitation.dat",
           np.column_stack([sol.t, alpha_t, alpha_t]))   # Re=Im=0

# ---- diagnostics ----
n_exact      = np.cos(g * sol.t)**2
max_err      = np.max(np.abs(n_t - n_exact))
# N_exc = n + (1+w)/2 = n + 1 - n = 1  (exact by construction)
Nexc         = n_t + 0.5*(1.0 + w_t)
max_Nexc_err = np.max(np.abs(Nexc - 1.0))
max_D_Cconj  = np.max(np.abs(D_t - np.conjugate(C_t)))
has_nan_inf  = bool(np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)))

print("\n--- Diagnostics (exact projected single-excitation algebra, N=1, N_exc=1 benchmark) ---")
print(f"solve_ivp: success={sol.success}  {sol.message}")
print(f"NaN/Inf present:                    {has_nan_inf}")
print(f"max |n_constrained - n_exact|     = {max_err:.4e}")
print(f"max |N_exc(t) - 1|                = {max_Nexc_err:.4e}")
print(f"max |D - C*|                      = {max_D_Cconj:.4e}")
print(f"n range: [{np.min(n_t):.8f}, {np.max(n_t):.8f}]")

with open("diagnostics_projected_single_excitation.txt", "w") as f:
    f.write(f"solve_ivp_success: {sol.success}\n")
    f.write(f"nan_inf_present: {has_nan_inf}\n")
    f.write(f"max_abs_error_vs_exact: {max_err:.6e}\n")
    f.write(f"max_abs_Nexc_error: {max_Nexc_err:.6e}\n")
    f.write(f"max_abs_D_minus_C_conj: {max_D_Cconj:.6e}\n")
    f.write(f"min_n: {np.min(n_t):.6e}\n")
    f.write(f"max_n: {np.max(n_t):.6e}\n")

print("Saved: Photon_number_projected_single_excitation.dat,")
print("       Excited_population_projected_single_excitation.dat,")
print("       alpha_projected_single_excitation.dat, diagnostics_projected_single_excitation.txt")
