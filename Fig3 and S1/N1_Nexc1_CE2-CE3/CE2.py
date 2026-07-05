import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------- parameters ----------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa     = 0.0           # fs^-1
gamma1    = 0.0           # fs^-1
gamma_phi = 0.0           # fs^-1
eta       = 0.0 + 0.0j   # coherent drive (set 0 here)
w_eq      = -1.0

# rotating frame at omega_d = omega_c = omega_0 (on resonance)
Delta_c    = eV_to_rate(omega_c - 2.0)
Delta_x    = eV_to_rate(omega_0 - 2.0)
g          = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi

# time grid
t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)

# ---------- pack/unpack ----------
# complex vars: alpha, s, m, V, C, D, U  (7 complex = 14 floats)
# real vars:    w, n                      (2 reals)
# total: 16 elements

def pack(alpha, s, m, V, C, D, U, w, n):
    zc = np.array([alpha, s, m, V, C, D, U], dtype=np.complex128)
    zf = zc.view(np.float64)   # 14 floats
    return np.concatenate([zf, np.array([float(w), float(n)], dtype=np.float64)])

def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:14]).view(np.complex128)
    alpha, s, m, V, C, D, U = zc
    w, n = float(y[14]), float(y[15])
    return alpha, s, m, V, C, D, U, w, n

# ---------- RHS ----------
def rhs(t, y):
    alpha, s, m, V, C, D, U, w, n = unpack(y)

    abs2 = abs(alpha)**2

    # Full primitive CE2 closures (Pauli + bosonic identities applied before closure)
    Az      = n*w + alpha*np.conjugate(U) + np.conjugate(alpha)*U - 2.0*abs2*w
    Az_plus = (n + 1.0)*w + alpha*np.conjugate(U) + np.conjugate(alpha)*U - 2.0*abs2*w
    A2z     = m*w + 2.0*alpha*U - 2.0*(alpha**2)*w
    Bminus  = (n + 1.0)*s + np.conjugate(alpha)*V + alpha*D - 2.0*abs2*s
    Bplus   = m*np.conjugate(s) + 2.0*alpha*C - 2.0*(alpha**2)*np.conjugate(s)

    dalpha = -(kappa/2 + 1j*Delta_c)*alpha - 1j*g*s + eta

    ds = -(gamma_perp + 1j*Delta_x)*s + 1j*g*U

    dw = -gamma1*(w - w_eq) + 2j*g*(D - C)

    dn = -kappa*n + 1j*g*(C - D) + 2.0*np.real(np.conjugate(eta)*alpha)

    dm = -(kappa + 2j*Delta_c)*m - 2j*g*V + 2.0*eta*alpha

    dV = -(kappa/2 + gamma_perp + 1j*(Delta_c + Delta_x))*V \
         + 1j*g*A2z + eta*s

    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 \
         - 1j*g*Az_plus + eta*np.conjugate(s)

    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 \
         + 1j*g*Az + np.conjugate(eta)*s

    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*s + eta*w \
         + 2j*g*(Bminus - Bplus)

    return pack(dalpha, ds, dm, dV, dC, dD, dU, dw.real, dn.real)

# ---------- initial conditions: |1> x |g> ----------
alpha0 = 0.0 + 0.0j
s0     = 0.0 + 0.0j
m0     = 0.0 + 0.0j
V0     = 0.0 + 0.0j
C0     = 0.0 + 0.0j
D0     = 0.0 + 0.0j
U0     = 0.0 + 0.0j
w0     = -1.0
n0     = 1.0

y0 = pack(alpha0, s0, m0, V0, C0, D0, U0, w0, n0)

sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# ---------- vectorized unpack ----------
Y = sol.y.T.copy()
Z = Y[:, :14].view(np.complex128)
alpha = Z[:, 0]
s     = Z[:, 1]
m     = Z[:, 2]
V     = Z[:, 3]
C     = Z[:, 4]
D     = Z[:, 5]
U     = Z[:, 6]
w     = Y[:, 14]
n     = Y[:, 15]

P_e  = 0.5*(1.0 + w)
n_ph = n

np.savetxt("Photon_number_CE2.dat", np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_CE2.dat", np.column_stack([sol.t, P_e]))
np.savetxt("alpha_CE2.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
print("Saved: Photon_number_CE2.dat, Excited_population_CE2.dat, alpha_CE2.dat")

# ---------- diagnostics ----------
n_exact = np.cos(g * sol.t)**2
np.savetxt("Photon_number_exact.dat", np.column_stack([sol.t, n_exact]))

print("\n--- Diagnostics (unprojected CE2, N=1, N_exc=1 benchmark) ---")
print(f"solve_ivp: success={sol.success}  {sol.message}")

has_nan_inf = (np.any(np.isnan(n_ph)) or np.any(np.isinf(n_ph))
               or np.any(np.isnan(w)) or np.any(np.isinf(w)))
print(f"NaN/Inf present: {has_nan_inf}")

max_n_err = np.max(np.abs(n_ph - n_exact))
print(f"max |n_CE2 - n_exact|  = {max_n_err:.4e}")

Nexc = n_ph + 0.5*(1.0 + w)
max_Nexc_err = np.max(np.abs(Nexc - 1.0))
print(f"max |N_exc(t) - 1|     = {max_Nexc_err:.4e}")

max_D_Cconj = np.max(np.abs(D - np.conjugate(C)))
print(f"max |D - C*|           = {max_D_Cconj:.4e}")
print("Saved: Photon_number_exact.dat")
