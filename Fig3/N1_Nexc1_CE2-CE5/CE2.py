import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------- parameters ----------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa   = 0.0   # fs^-1
gamma1  = 0.0   # fs^-1
gamma_phi = 0.0 # fs^-1
eta     = 0.0 + 0.0j  # coherent drive (set 0 here)

# rotating frame at omega_d = omega_c = omega_0 (on resonance)
Delta_c = eV_to_rate(omega_c - 2.0)
Delta_x = eV_to_rate(omega_0 - 2.0)
g       = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi

# time grid
t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)

# ---------- pack/unpack complex state to real vector ----------
# state order: alpha, s, C, D, U (complex), then w, n (real)

def pack(alpha, s, C, D, U, w, n):
    # Make a contiguous 1-D complex128 array of length 5
    zc = np.array([alpha, s, C, D, U], dtype=np.complex128)
    # Reinterpret as float64: [Re(alpha), Im(alpha), ..., Re(U), Im(U)]
    zf = zc.view(np.float64)
    # Append the two real scalars w, n
    return np.concatenate([zf, np.array([float(w), float(n)], dtype=np.float64)])

def unpack(y):
    # First 10 floats -> 5 complex numbers
    zc = y[:-2].view(np.complex128)
    alpha, s, C, D, U = zc[0], zc[1], zc[2], zc[3], zc[4]
    w, n = float(y[-2]), float(y[-1])
    return alpha, s, C, D, U, w, n


# ---------- RHS ----------
def rhs(t, y):
    alpha, s, C, D, U, w, n = unpack(y)

    # eqs
    dalpha = -(kappa/2 + 1j*Delta_c)*alpha - 1j*g*s + eta
    ds     = -(gamma_perp + 1j*Delta_x)*s + 1j*g*U
    dw     = -gamma1*(w - (-1.0)) + 2j*g*(D - C)  # weq=-1

    dn     = -kappa*n + 1j*g*(C - D) + (eta.conjugate()*alpha + eta*alpha.conjugate()).real

    # closures for triple moments
    # <a a^\dagger sigma_-> ~ (n+1)s + alpha*D?  we use: (n+1)s + alpha*D? careful:
    # we derived: (n+1) s + alpha*? Wait D = <a^\dagger sigma_->; need alpha*<a sigma_->; approximate <a sigma_-> ~ alpha*s
    aa_dag_sig_minus = (n + 1.0)*s + alpha*D - (abs(alpha)**2)*s
    a2_sig_plus      = 2.0*alpha*C - (alpha**2)*np.conjugate(s)

    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 \
         - 1j*g*((n + 1.0)*w + np.conjugate(alpha)*U + alpha*np.conjugate(U) - 2.0*(abs(alpha)**2)*w) \
         + eta*np.conjugate(s)

    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 \
         + 1j*g*(n*w + np.conjugate(alpha)*U + alpha*np.conjugate(U) - 2.0*(abs(alpha)**2)*w) \
         + np.conjugate(eta)*s

    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*s + eta*w \
         + 2j*g*(aa_dag_sig_minus - a2_sig_plus)

    return pack(dalpha, ds, dC, dD, dU, dw.real, dn.real)

# ---------- initial conditions ----------
# Example 1: atom excited, empty cavity (will Rabi-exchange without seeding)
alpha0 = 0.0 + 0.0j
s0     = 0.0 + 0.0j
w0     = -1.0
n0     = 1.0
C0 = 0.0 + 0.0j
D0 = 0.0 + 0.0j
U0 = 0.0 + 0.0j

y0 = pack(alpha0, s0, C0, D0, U0, w0, n0)

# integrate
sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
Y = sol.y.T.copy()                 # (Nt, 12) and C-contiguous
Z = Y[:, :10].view(np.complex128)  # reinterpret first 10 floats as 5 complex vars

alpha = Z[:, 0]
s     = Z[:, 1]
C     = Z[:, 2]
D     = Z[:, 3]
U     = Z[:, 4]
w     = Y[:, 10]
n     = Y[:, 11]

# Observables
P_e  = 0.5*(1.0 + w)
n_ph = n

np.savetxt("Photon_number_CE2.dat", np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_CE2.dat", np.column_stack([sol.t, P_e]))
np.savetxt("alpha_CE2.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
print("Saved: Photon_number_CE2.dat, Excited_population_CE2.dat, alpha_CE2.dat")
