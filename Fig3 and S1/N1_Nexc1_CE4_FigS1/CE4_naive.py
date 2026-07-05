"""Unprojected fourth-order diagnostic closure for the N=1, N_exc=1 JC benchmark.

This is not the fourth-order member of a systematic cluster-expansion hierarchy:
it is a diagnostic closure that evolves a selected set of fourth-order moments
without imposing the projected-subspace identities that hold in the exact
single-excitation manifold (see CE4_constrained.py).

All retained variables (alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2)
evolve freely without imposing any projected-subspace identities.  The invariant-subspace
constraints (w=1-2n, n2=n, Sd=D, Cn=0, Nw=-n) are NOT enforced here; their violation
diagnoses the failure of this unprojected closure.

Hamiltonian (resonant rotating frame):
    H = g_c (a† σ⁻ + a σ⁺)
Parameters: omega_c = omega_0 = 2.0 eV, g_c = 0.1 eV, kappa = gamma1 = gamma_phi = 0.
Initial state: |1,g> (one photon, atom in ground state, N_exc = 1).
"""
import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---- parameters ----
omega_c   = 2.0           # eV
omega_0   = 2.0           # eV
g_eV      = 0.1           # eV
kappa     = 0.0           # fs^-1
gamma1    = 0.0           # fs^-1
gamma_phi = 0.0           # fs^-1
eta       = 0.0 + 0.0j    # coherent drive

Delta_c    = eV_to_rate(omega_c - 2.0)   # = 0 (on resonance)
Delta_x    = eV_to_rate(omega_0 - 2.0)   # = 0
g          = eV_to_rate(g_eV)
gamma_perp = gamma1 / 2.0 + gamma_phi
w_eq       = -1.0

t0, t1  = 0.0, 200.0
t_eval  = np.linspace(t0, t1, 4001)

# ---- state layout ----
# Complex (12 variables, stored as 24 floats):
#   alpha=<a>, s=<σ->, C=<a σ+>, D=<a† σ->, U=<a σz>,
#   S1=<n σ->, Q=<a² σ+>, T1=<n a σz>, T2=<a n σz>,
#   Sd=<n a† σ->, Cn=<n a σ+>, Na=<n a>
# Real (4 variables):
#   w=<σz>, n=<a†a>, Nw=<n σz>, n2=<n²>

def pack(alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2):
    zc = np.array([alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na], dtype=np.complex128)
    return np.concatenate([zc.view(np.float64), np.array([w, n, Nw, n2], dtype=np.float64)])

def unpack(y):
    y  = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:24]).view(np.complex128)
    alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na = zc
    w, n, Nw, n2 = y[24], y[25], y[26], y[27]
    # NO constraints applied — all variables evolve freely
    return alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2

def real_if_close(x, name, tol=1e-10):
    if abs(np.imag(x)) > tol:
        print(f"Warning: {name} has non-negligible imaginary part: {x}")
    return float(np.real(x))

# ---- unprojected fourth-order diagnostic RHS ----
def rhs(t, y):
    alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na, w, n, Nw, n2 = unpack(y)

    # ---------- 1st order (exact) ----------
    dalpha = -(kappa/2 + 1j*Delta_c)*alpha - 1j*g*s + eta
    ds     = -(gamma_perp + 1j*Delta_x)*s + 1j*g*U
    dw     = -gamma1*(w - w_eq) + 2j*g*(D - C)
    dn     = -kappa*n + 1j*g*(C - D) + 2.0*np.real(np.conjugate(eta)*alpha)

    # ---------- 2nd order (exact, using Nw as promoted) ----------
    # <a a† σz> = Nw + w  (exact identity: a a† = n + 1)
    # <a† a σz> = Nw      (exact identity)
    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 - 1j*g*(Nw + w) + eta*np.conjugate(s)
    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 + 1j*g*Nw + np.conjugate(eta)*s
    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U \
         - 1j*g*s + eta*w + 2j*g*(S1 + s - Q)

    # ---------- 3rd order (exact, using T1/T2 as promoted) ----------
    # S1 = <n σ->
    dS1 = -(kappa + gamma_perp + 1j*Delta_x)*S1 \
          + 0.5j*g*(alpha + U) + 1j*g*T1

    # Q = <a² σ+>
    # a² a† = a n + a  =>  <a² a† σz> = T2 + U  (not just T2)
    dQ  = -(kappa + gamma_perp - 1j*Delta_x + 2j*Delta_c)*Q \
          - 1j*g*(alpha - U) + 2.0*eta*C - 1j*g*(T2 + U)

    # Nw = <n σz>
    dNw = -kappa*Nw - gamma1*(Nw - w_eq*n) \
          + 1j*g*(-C - D) + 2j*g*(Sd - Cn)

    # ---------- 4th order diagnostic closures ----------
    # <n² σz>:  κ(n,n,σz) = 0 in this diagnostic closure
    N2w = n2*w + 2.0*n*Nw - 2.0*n**2*w
    # <n² σ->:  κ(n,n,σ-) = 0 in this diagnostic closure
    N2s = n2*s + 2.0*n*S1 - 2.0*n**2*s
    # <n a² σ+>_closure ≈ n Q  (factorize n out of Q)
    Qn  = n*Q

    # T1 = <n a σz>
    # origin of terms:
    #   (dn/dt) a σz  → -κ T1 - ig Q - ig S1 + η Nw
    #   n (da/dt) σz  → -(κ/2+iΔc) T1 - ig S1 + η Nw
    #   n a (dσz/dt)  → -γ1(T1 - w_eq Na) + 2ig(N2s + S1 - Qn)
    # combined (η=0, γ1=0):
    #   = -(3κ/2+iΔc) T1 - ig Q - 2ig S1 + 2ig(N2s + S1 - Qn)
    # equivalently: -(3κ/2+iΔc) T1 - ig Q + 2ig(N2s - Qn)
    dT1 = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T1 \
          - 1j*g*Q - 2j*g*S1 + 2j*g*(N2s + S1 - Qn) \
          + gamma1*w_eq*Na

    # T2 = <a n σz>
    # origin of terms:
    #   (da/dt) n σz  → -(κ/2+iΔc) T2 - ig S1 + η Nw
    #   a (dn/dt) σz  → -κ T2 - ig Q - ig S1 + η(Nw+w)
    #   a n (dσz/dt)  → -γ1(T2 - w_eq Na) + 2ig[(N2s + 2S1 + s) - (nQ + Q)]
    # The extra s in -(Q+2S1+s) and +(N2s+2S1+s) comes from commutator an = na + a
    dT2 = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T2 \
          - 1j*g*(Q + 2*S1 + s) \
          + 2j*g*((N2s + 2*S1 + s) - (n*Q + Q)) \
          + gamma1*w_eq*Na

    # Sd = <n a† σ->
    # n (da†/dt) σ-  → -(κ/2-iΔc) Sd + ig(n+Nw)/2
    # (dn/dt) a† σ-  → -κ Sd + ig(n+Nw+1)/2  [via n a† a† σ-σ- terms averaged]
    # n a† (dσ-/dt)  → -(γ⊥+iΔx) Sd + ig <n² σz> = ig N2w
    # combined: -(3κ/2+γ⊥+i(Δx-Δc)) Sd + ig(n + Nw + (1+w)/2) + ig N2w
    dSd = -(3*kappa/2 + gamma_perp + 1j*(Delta_x - Delta_c))*Sd \
          + 1j*g*(n + Nw + (1.0 + w)/2.0) + 1j*g*N2w

    # Cn = <n a σ+>  — NOT a mirror image of Sd
    # n (da/dt) σ+   → -(κ/2+iΔc) Cn - ig S1 + η Nw
    # (dn/dt) a σ+   → -κ Cn - ig n (?) [corrected derivation gives -ig(n-Nw)]
    # n a (dσ+/dt)   → -(γ⊥-iΔx) Cn - ig <n² σz> - ig Nw <n σz> terms
    # combined: -(3κ/2+γ⊥-i(Δx-Δc)) Cn - ig(n-Nw) - ig(N2w+Nw)
    dCn = -(3*kappa/2 + gamma_perp - 1j*(Delta_x - Delta_c))*Cn \
          - 1j*g*(n - Nw) - 1j*g*(N2w + Nw)

    # Na = <n a>
    dNa = -(3*kappa/2 + 1j*Delta_c)*Na + 1j*g*(Q - 2*S1) + eta*n

    # n2 = <n²>
    # d n2 = 2<n dn/dt> = -2κ n2 + 2ig(Cn-Sd) + 2 Re(η* Na)
    # +ig(C+D) arises from the non-commutator correction: <a†a a†σ-> - <a†a aσ+> terms
    dn2 = -2.0*kappa*n2 + 2j*g*(Cn - Sd) + 1j*g*(C + D) \
          + 2.0*np.real(np.conjugate(eta)*Na)

    return pack(dalpha, ds, dC, dD, dU, dS1, dQ, dT1, dT2, dSd, dCn, dNa,
                real_if_close(dw,  "dw"),
                real_if_close(dn,  "dn"),
                real_if_close(dNw, "dNw"),
                real_if_close(dn2, "dn2"))

# ---- initial conditions: |1,g>  (N_exc = 1) ----
# alpha=0, s=0, w=-1, n=1; all correlators zero except Nw=n*w=-1, n2=n^2=1
y0 = pack(0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j,
          0+0j, 0+0j, 0+0j, 0+0j, 0+0j,
          -1.0, 1.0, -1.0, 1.0)

sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# ---- unpack solution ----
Y      = sol.y.T.copy()
Z      = Y[:, :24].view(np.complex128)
alpha_t = Z[:, 0]
C_t    = Z[:, 2]; D_t = Z[:, 3]
w_t    = Y[:, 24]; n_t = Y[:, 25]; Nw_t = Y[:, 26]
P_e    = 0.5*(1.0 + w_t)
n_ph   = n_t

# ---- save outputs ----
np.savetxt("Photon_number_unprojected_4th_order_diagnostic.dat",
           np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_unprojected_4th_order_diagnostic.dat",
           np.column_stack([sol.t, P_e]))
np.savetxt("alpha_unprojected_4th_order_diagnostic.dat",
           np.column_stack([sol.t, alpha_t.real, alpha_t.imag]))

# Save the exact reference on the full requested grid even if this diagnostic
# integration becomes unstable and stops early.
n_exact_ref = np.cos(g * t_eval)**2
np.savetxt("Photon_number_exact.dat",
           np.column_stack([t_eval, n_exact_ref]))

# ---- diagnostics ----
n_exact       = np.cos(g * sol.t)**2
max_err       = np.max(np.abs(n_ph - n_exact))
Nexc          = n_ph + 0.5*(1.0 + w_t)
max_Nexc_err  = np.max(np.abs(Nexc - 1.0))
max_D_Cconj   = np.max(np.abs(D_t - np.conjugate(C_t)))
has_nan_inf   = bool(np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)))

print("\n--- Diagnostics (unprojected 4th-order diagnostic, N=1, N_exc=1 benchmark) ---")
print(f"solve_ivp: success={sol.success}  {sol.message}")
print(f"NaN/Inf present:                {has_nan_inf}")
print(f"max |n_diagnostic - n_exact| = {max_err:.4e}")
print(f"max |N_exc(t) - 1|            = {max_Nexc_err:.4e}")
print(f"max |D - C*|                  = {max_D_Cconj:.4e}")
print(f"n range: [{np.min(n_ph):.4g}, {np.max(n_ph):.4g}]")

with open("diagnostics_unprojected_4th_order_diagnostic.txt", "w") as f:
    f.write(f"solve_ivp_success: {sol.success}\n")
    f.write(f"nan_inf_present: {has_nan_inf}\n")
    f.write(f"max_abs_error_vs_exact: {max_err:.6e}\n")
    f.write(f"max_abs_Nexc_error: {max_Nexc_err:.6e}\n")
    f.write(f"max_abs_D_minus_C_conj: {max_D_Cconj:.6e}\n")
    f.write(f"min_n: {np.min(n_ph):.6e}\n")
    f.write(f"max_n: {np.max(n_ph):.6e}\n")

print("Saved: Photon_number_unprojected_4th_order_diagnostic.dat,")
print("       Excited_population_unprojected_4th_order_diagnostic.dat,")
print("       alpha_unprojected_4th_order_diagnostic.dat,")
print("       diagnostics_unprojected_4th_order_diagnostic.txt")
