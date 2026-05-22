import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------------- parameters ----------------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa   = 0.0   # fs^-1  (cavity loss)
gamma1  = 0.0   # fs^-1  (relaxation)
gamma_phi = 0.0 # fs^-1  (pure dephasing)
eta     = 0.0 + 0.0j     # coherent drive (keep 0 here)

# rotating frame chosen at omega_d = omega_c = omega_0 (on resonance)
Delta_c = eV_to_rate(omega_c - 2.0)
Delta_x = eV_to_rate(omega_0 - 2.0)
g       = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi
w_eq = -1.0

# time grid
t0, t1 = 0.0, 200.0  # fs
t_eval = np.linspace(t0, t1, 4001)

# ------------- pack/unpack -------------
# State vector y packs complex vars as float pairs for solve_ivp.
# Order of complex vars: alpha, s, C, D, U, S1, Q  (7 complex = 14 floats)
# Then real vars: w, n, Nw  (3 reals)
def pack(alpha, s, C, D, U, S1, Q, w, n, Nw):
    zc = np.array([alpha, s, C, D, U, S1, Q], dtype=np.complex128)
    zf = zc.view(np.float64)        # 14 floats
    return np.concatenate([zf, np.array([float(w), float(n), float(Nw)], dtype=np.float64)])

def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:14]).view(np.complex128)
    alpha, s, C, D, U, S1, Q = zc
    w, n, Nw = float(y[14]), float(y[15]), float(y[16])
    return alpha, s, C, D, U, S1, Q, w, n, Nw

# ------------- CE3 RHS -------------
def rhs(t, y):
    alpha, s, C, D, U, S1, Q, w, n, Nw = unpack(y)
    a, sminus = alpha, s  # aliases

    # Useful shorthands
    a_conj = np.conjugate(a)
    s_conj = np.conjugate(sminus)
    U_conj = np.conjugate(U)
    C_conj = np.conjugate(C)
    D_conj = np.conjugate(D)
    S1_conj = np.conjugate(S1)
    Q_conj = np.conjugate(Q)

    # --- 1st/2nd order (exact) ---
    dalpha = -(kappa/2 + 1j*Delta_c)*a - 1j*g*sminus + eta
    ds     = -(gamma_perp + 1j*Delta_x)*sminus + 1j*g*U
    dw     = -gamma1*(w - w_eq) + 2j*g*(D - C)

    dn     = -kappa*n + 1j*g*(C - D) + 2.0*np.real(np.conjugate(eta)*a)

    # --- 2nd order with exact triple dependences ---
    # dC exact: -(k/2+g_perp+i(Δc-Δx)) C - i g (1-w)/2 - i g <a a† σz> + η s*
    #           with <a a† σz> = <(n+1) σz> = Nw + w
    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 \
         - 1j*g*(Nw + w) + eta*s_conj

    # dD exact: -(k/2+g_perp-i(Δc-Δx)) D + i g (1+w)/2 + i g <a† a σz> + η* s
    #           with <a† a σz> = Nw
    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 \
         + 1j*g*Nw + np.conjugate(eta)*sminus

    # dU exact: -(k/2+iΔc+γ1)U - i g s + η w + 2i g( <a a† σ-> - <a^2 σ+> )
    #           with <a a† σ-> = <(n+1) σ-> = S1 + s
    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*sminus + eta*w \
         + 2j*g*(S1 + sminus - Q)

    # --- 3rd order dynamics with CE3 closures ---

    # dS1 = d< n σ- > = <(dn) σ-> + < n dσ- >
    # (dn)σ- → -κ S1 + i g < a σ+ σ- >  (since <a† σ- σ->=0)
    #          = -κ S1 + i g * (1/2)*(a + U)
    term1 = -kappa*S1 + 1j*g*0.5*(a + U)
    # n dσ- → -(γ_perp+iΔx) S1 + i g < n a σz >
    # CE3 closure: < n a σz > ≈ n U + a Nw + w*<n a> - 2 n a w, with <n a> ≈ n a
    na_sigma_z = n*U + a*Nw + w*(n*a) - 2.0*n*a*w
    term2 = -(gamma_perp + 1j*Delta_x)*S1 + 1j*g*na_sigma_z
    dS1 = term1 + term2

    # dQ = d< a^2 σ+ > = <2 a da σ+> + < a^2 dσ+ >
    # 2 a da σ+ → -(κ + 2iΔc) Q - 2i g < a σ- σ+ > + 2 η C
    #            with <a σ- σ+> = < a (1 - σz)/2 > = 0.5*(a - U)
    termQ1 = -(kappa + 2j*Delta_c)*Q - 2j*g*0.5*(a - U) + 2.0*eta*C
    # a^2 dσ+ → -(γ_perp - iΔx) Q - i g < a^2 a† σz >
    # and  a^2 a† = a n + a  ⇒  < a^2 a† σz > = < a n σz > + U
    # CE3 closure: < a n σz > ≈ a Nw + n U + w*<a n> - 2 a n w, with <a n> ≈ a n
    an_sigma_z = a*Nw + n*U + w*(a*n) - 2.0*a*n*w
    termQ2 = -(gamma_perp - 1j*Delta_x)*Q - 1j*g*(an_sigma_z + U)
    dQ = termQ1 + termQ2

    # dNw = d< n σz > = < (dn) σz > + < n dσz >
    # (dn)σz → -κ Nw + i g ( < a σ+ σz > - < a† σ- σz > )
    #          = -κ Nw + i g ( -C - D )
    partA = -kappa*Nw + 1j*g*(-C - D)
    # n dσz → -γ1(Nw - w_eq n) + 2 i g ( < n a† σ- > - < n a σ+ > )
    # CE3 closure for triple:
    #   < n a† σ- > ≈ n D + a_conj*S1 + sminus*(n*a_conj) - 2 n a_conj sminus
    #   < n a  σ+ > ≈ n C + a*S1_conj + np.conjugate(sminus)*(n*a) - 2 n a np.conjugate(sminus)
    na_dag_sigma = n*D + a_conj*S1 + sminus*(n*a_conj) - 2.0*n*a_conj*sminus
    na_sigma_plus = n*C + a*S1_conj + np.conjugate(sminus)*(n*a) - 2.0*n*a*np.conjugate(sminus)
    partB = -gamma1*(Nw - w_eq*n) + 2j*g*(na_dag_sigma - na_sigma_plus)
    dNw = partA + partB

    # pack
    return pack(dalpha, ds, dC, dD, dU, dS1, dQ, dw, dn, dNw)

# ------------- initial conditions (|1>⊗|g>) -------------
alpha0 = 0.0 + 0.0j   # <a>
n0     = 1.0          # <a†a>
s0     = 0.0 + 0.0j   # <σ->
w0     = -1.0         # <σz>

C0 = 0.0 + 0.0j       # <a σ+>
D0 = 0.0 + 0.0j       # <a† σ->
U0 = 0.0 + 0.0j       # <a σz>
S10 = 0.0 + 0.0j      # <n σ->
Q0  = 0.0 + 0.0j      # <a^2 σ+>
Nw0 = n0*w0           # factorized start for <n σz> in a product state

y0 = pack(alpha0, s0, C0, D0, U0, S10, Q0, w0, n0, Nw0)

# integrate
sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# unpack all times (vectorized, contiguous)
Y = sol.y.T.copy()
Z = Y[:, :14].view(np.complex128)
alpha = Z[:, 0]; s = Z[:, 1]; C = Z[:, 2]; D = Z[:, 3]; U = Z[:, 4]; S1 = Z[:, 5]; Q = Z[:, 6]
w  = Y[:, 14]; n = Y[:, 15]; Nw = Y[:, 16]

# observables
P_e  = 0.5*(1.0 + w)     # excited population
n_ph = n                 # photon number

np.savetxt("Photon_number_CE3.dat", np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_CE3.dat", np.column_stack([sol.t, P_e]))
np.savetxt("alpha_CE3.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
print("Saved: Photon_number_CE3.dat, Excited_population_CE3.dat, alpha_CE3.dat")
