import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------------- parameters ----------------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa     = 0.0           # fs^-1  (cavity loss)
gamma1    = 0.0           # fs^-1  (relaxation)
gamma_phi = 0.0           # fs^-1  (pure dephasing)
eta       = 0.0 + 0.0j   # coherent drive (keep 0 here)

# rotating frame chosen at omega_d = omega_c = omega_0 (on resonance)
Delta_c    = eV_to_rate(omega_c - 2.0)
Delta_x    = eV_to_rate(omega_0 - 2.0)
g          = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi
w_eq       = -1.0

# time grid
t0, t1 = 0.0, 200.0  # fs
t_eval = np.linspace(t0, t1, 4001)

# ------------- pack/unpack -------------
# complex vars: alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z  (13 complex = 26 floats)
# real vars:    w, n, Nw                                           (3 reals)
# total: 29 elements

def pack(alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z, w, n, Nw):
    zc = np.array([alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z], dtype=np.complex128)
    zf = zc.view(np.float64)   # 26 floats
    return np.concatenate([zf, np.array([float(w), float(n), float(Nw)], dtype=np.float64)])

def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:26]).view(np.complex128)
    alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z = zc
    w, n, Nw = float(y[26]), float(y[27]), float(y[28])
    return alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z, w, n, Nw

def real_if_close(x, name, tol=1e-10):
    if abs(np.imag(x)) > tol:
        print(f"Warning: {name} has non-negligible imaginary part: {x}")
    return float(np.real(x))

# ------------- CE3 RHS (unprojected CE3 diagnostic closure) -------------
def rhs(t, y):
    alpha, s, m, V, C, D, U, Na, a3, S1, R2, Q, A2z, w, n, Nw = unpack(y)

    a   = alpha
    ac  = np.conjugate(alpha)
    sc  = np.conjugate(s)
    mc  = np.conjugate(m)
    Vc  = np.conjugate(V)
    Uc  = np.conjugate(U)
    Nac = np.conjugate(Na)
    S1c = np.conjugate(S1)
    Qc  = np.conjugate(Q)
    abs2 = abs(a)**2

    # --- 1st/2nd order (promoted: use full third-order vars where exact) ---
    dalpha = -(kappa/2 + 1j*Delta_c)*a - 1j*g*s + eta

    ds = -(gamma_perp + 1j*Delta_x)*s + 1j*g*U

    dw = -gamma1*(w - w_eq) + 2j*g*(D - C)

    dn = -kappa*n + 1j*g*(C - D) + 2.0*np.real(np.conjugate(eta)*a)

    dm = -(kappa + 2j*Delta_c)*m - 2j*g*V + 2.0*eta*a

    dV = -(kappa/2 + gamma_perp + 1j*(Delta_c + Delta_x))*V \
         + 1j*g*A2z + eta*s

    # dC and dD: use promoted Nw (exact at CE3 level, <aa†σz> = Nw + w)
    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 \
         - 1j*g*(Nw + w) + eta*sc

    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 \
         + 1j*g*Nw + ac*eta*0 + np.conjugate(eta)*s

    # dU: use promoted S1 and Q (exact at CE3 level, <aa†σ-> = S1 + s)
    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*s + eta*w \
         + 2j*g*(S1 + s - Q)

    # --- full fourth-order CE3 closures ---
    # <n a sigma_z>
    T_naz = (
        w*Na + 2.0*a*Nw + ac*A2z + 2.0*n*U + m*Uc
        - 4.0*n*a*w - 2.0*(a**2)*Uc - 2.0*m*ac*w
        - 4.0*abs2*U + 6.0*abs2*a*w
    )

    # <n a sigma_->
    T_na_sm = (
        Na*s + 2.0*a*S1 + ac*R2 + 2.0*n*V + m*D
        - 4.0*n*a*s - 2.0*D*(a**2) - 2.0*m*ac*s
        - 4.0*abs2*V + 6.0*abs2*a*s
    )

    # <a^3 sigma+>
    T_a3_sp = (
        a3*sc + 3.0*a*Q + 3.0*m*C
        - 6.0*m*a*sc - 6.0*C*(a**2) + 6.0*(a**3)*sc
    )

    # <a^3 sigma_z>
    T_a3_z = (
        a3*w + 3.0*a*A2z + 3.0*m*U
        - 6.0*m*a*w - 6.0*U*(a**2) + 6.0*(a**3)*w
    )

    # <n a† sigma_->: use n a† = a†^2 a + a†, so <n a† sigma_-> = <a†^2 a sigma_-> + D
    F_adag2_a_sm = (
        Nac*s + Qc*a + 2.0*ac*S1 + mc*V + 2.0*n*D
        - 2.0*(mc*a*s + 2.0*n*ac*s + 2.0*D*abs2 + V*(ac**2))
        + 6.0*(ac**2)*a*s
    )
    T_nadag_sm = F_adag2_a_sm + D

    # <n a sigma+>
    T_na_sp = (
        Na*sc + 2.0*a*S1c + ac*Q + 2.0*n*C + m*Vc
        - 4.0*n*a*sc - 2.0*Vc*(a**2) - 2.0*m*ac*sc
        - 4.0*abs2*C + 6.0*abs2*a*sc
    )

    # --- 3rd order dynamics ---
    dNa = -(1.5*kappa + 1j*Delta_c)*Na \
          + 1j*g*(Q - 2.0*S1) \
          + np.conjugate(eta)*m + 2.0*eta*n

    da3 = -(1.5*kappa + 3j*Delta_c)*a3 \
          - 3j*g*R2 + 3.0*eta*m

    dS1 = -(kappa + gamma_perp + 1j*Delta_x)*S1 \
          + 1j*g*0.5*(a + U) \
          + 1j*g*T_naz \
          + np.conjugate(eta)*V + eta*D

    dR2 = -(kappa + gamma_perp + 1j*(2.0*Delta_c + Delta_x))*R2 \
          + 1j*g*T_a3_z + 2.0*eta*V

    dQ = -(kappa + gamma_perp + 1j*(2.0*Delta_c - Delta_x))*Q \
         - 1j*g*(a - U) \
         - 1j*g*(T_naz + 2.0*U) \
         + 2.0*eta*C

    dA2z = -(kappa + 2j*Delta_c + gamma1)*A2z \
           - 2j*g*V + 2.0*eta*U \
           + 2j*g*(T_na_sm + 2.0*V - T_a3_sp)

    dNw = -kappa*Nw - gamma1*(Nw - w_eq*n) \
          + 1j*g*(-C - D) \
          + 2j*g*(T_nadag_sm - T_na_sp) \
          + np.conjugate(eta)*U + eta*np.conjugate(U)

    return pack(dalpha, ds, dm, dV, dC, dD, dU, dNa, da3, dS1, dR2, dQ, dA2z,
                real_if_close(dw,  "dw"),
                real_if_close(dn,  "dn"),
                real_if_close(dNw, "dNw"))

# ------------- initial conditions: |1> x |g> -------------
alpha0 = 0.0 + 0.0j
s0     = 0.0 + 0.0j
m0     = 0.0 + 0.0j
V0     = 0.0 + 0.0j
C0     = 0.0 + 0.0j
D0     = 0.0 + 0.0j
U0     = 0.0 + 0.0j
Na0    = 0.0 + 0.0j
a30    = 0.0 + 0.0j
S10    = 0.0 + 0.0j
R20    = 0.0 + 0.0j
Q0     = 0.0 + 0.0j
A2z0   = 0.0 + 0.0j
w0     = -1.0
n0     = 1.0
Nw0    = -1.0

y0 = pack(alpha0, s0, m0, V0, C0, D0, U0, Na0, a30, S10, R20, Q0, A2z0, w0, n0, Nw0)

sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# ------------- vectorized unpack -------------
Y  = sol.y.T.copy()
Z  = Y[:, :26].view(np.complex128)
alpha = Z[:, 0];  s   = Z[:, 1];  m   = Z[:, 2];  V   = Z[:, 3]
C     = Z[:, 4];  D   = Z[:, 5];  U   = Z[:, 6];  Na  = Z[:, 7]
a3    = Z[:, 8];  S1  = Z[:, 9];  R2  = Z[:, 10]; Q   = Z[:, 11]
A2z   = Z[:, 12]
w  = Y[:, 26];  n  = Y[:, 27];  Nw = Y[:, 28]

P_e  = 0.5*(1.0 + w)
n_ph = n

np.savetxt("Photon_number_CE3.dat", np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_CE3.dat", np.column_stack([sol.t, P_e]))
np.savetxt("alpha_CE3.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
print("Saved: Photon_number_CE3.dat, Excited_population_CE3.dat, alpha_CE3.dat")

# ---------- diagnostics ----------
n_exact = np.cos(g * sol.t)**2
np.savetxt("Photon_number_exact.dat", np.column_stack([sol.t, n_exact]))

print("\n--- Diagnostics (unprojected CE3, N=1, N_exc=1 benchmark) ---")
print(f"solve_ivp: success={sol.success}  {sol.message}")

has_nan_inf = (np.any(np.isnan(n_ph)) or np.any(np.isinf(n_ph))
               or np.any(np.isnan(w)) or np.any(np.isinf(w))
               or np.any(np.isnan(Nw)) or np.any(np.isinf(Nw)))
print(f"NaN/Inf present: {has_nan_inf}")

max_n_err = np.max(np.abs(n_ph - n_exact))
print(f"max |n_CE3 - n_exact|  = {max_n_err:.4e}")

Nexc = n_ph + 0.5*(1.0 + w)
max_Nexc_err = np.max(np.abs(Nexc - 1.0))
print(f"max |N_exc(t) - 1|     = {max_Nexc_err:.4e}")

max_D_Cconj = np.max(np.abs(D - np.conjugate(C)))
print(f"max |D - C*|           = {max_D_Cconj:.4e}")
print("Saved: Photon_number_exact.dat")
