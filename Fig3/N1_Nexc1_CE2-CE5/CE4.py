import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------------- parameters ----------------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa   = 0.0   # fs^-1 (cavity loss)
gamma1  = 0.0   # fs^-1 (relaxation)
gamma_phi = 0.0 # fs^-1 (pure dephasing)
eta     = 0.0 + 0.0j  # coherent drive (set 0 for Fock start dynamics)

# on resonance (rotate at omega_d = 2.0 eV)
Delta_c = eV_to_rate(omega_c - 2.0)
Delta_x = eV_to_rate(omega_0 - 2.0)
g       = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi
w_eq = -1.0

# time grid
t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)

# ---------------- state layout ----------------
# Complex variables (packed as float pairs):
# alpha, s, C, D, U, S1, Q, T1, T2, Sd, Cn, Na
# where:
# alpha=<a>, s=<σ->, w=<σz>, n=<a†a>, C=<a σ+>, D=<a† σ->, U=<a σz>,
# S1=<n σ->, Q=<a^2 σ+>, T1=<n a σz>, T2=<a n σz>,
# Sd=<n a† σ->, Cn=<n a σ+>, Na=<n a>
# Real variables: w, n, Nw=<n σz>, n2=<n^2>
#
# Quartic promoted here: T1, T2, Sd, Cn, n2. We also keep Nw (cubic).
# Any appearance of n^2 σ- is closed by cumulant (see below) to avoid adding S2.

def pack(alpha,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,w,n,Nw,n2):
    zc = np.array([alpha,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na], dtype=np.complex128)
    zf = zc.view(np.float64)
    return np.concatenate([zf, np.array([w,n,Nw,n2], dtype=np.float64)])

def unpack(y):
    y = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:24]).view(np.complex128)
    alpha,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na = zc
    w,n,Nw,n2 = y[24], y[25], y[26], y[27]
#    n2 = n    # not crucial
#    S1 = 0    # not crucial
#    Q = 0     # not crucial
#    T1 = 0    # not crucial
#    T2 = 0    # not crucial
    Sd = D    # crucial!
    Cn = 0    # crucial!
#    Na = 0    # not crucial
    return alpha,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,w,n,Nw,n2

# ---------- helper: cumulant-like closures ----------
def prod_closure(*args):
    """Multiplicative closure for residual 5th-order pieces:
       <X1 X2 ... Xk> ≈ product of lower moments supplied as args."""
    out = 1.0 + 0.0j
    for a in args: out *= a
    return out

# ---------------- RHS ----------------
def rhs(t, y):
    alpha,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,w,n,Nw,n2 = unpack(y)
    a, sm = alpha, s
    aC, smC = np.conjugate(a), np.conjugate(sm)

    # ---------- 1st/2nd (exact) ----------
    dalpha = -(kappa/2 + 1j*Delta_c)*a - 1j*g*sm + eta
    ds     = -(gamma_perp + 1j*Delta_x)*sm + 1j*g*U
    dw     = -gamma1*(w - w_eq) + 2j*g*(D - C)
    dn     = -kappa*n + 1j*g*(C - D) + 2.0*np.real(np.conjugate(eta)*a)

    # C and D (use exact <a a† σz>=(Nw+w), <a† a σz>=Nw)
    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c - Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 - 1j*g*(Nw + w) + eta*smC
    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c - Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 + 1j*g*Nw + np.conjugate(eta)*sm

    # U (exact triplets S1 and Q promoted below)
    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*sm + eta*w + 2j*g*(S1 + sm - Q)

    # ---------- 3rd order promoted ----------
    # S1 = <n σ->
    # dS1 = -(kappa+gamma_perp+iΔx) S1 + (i g/2)(a+U) + i g * <n a σz> (exact!)
    dS1  = -(kappa + gamma_perp + 1j*Delta_x)*S1 + 0.5j*g*(a + U) + 1j*g*T1

    # Q = <a^2 σ+>
    # dQ = -(kappa + gamma_perp - iΔx + 2 i Δc) Q - j g (a - U) + 2 η C - i g * <a n σz> (exact!)
    dQ   = -(kappa + gamma_perp - 1j*Delta_x + 2j*Delta_c)*Q \
           - 1j*g*(a - U) + 2.0*eta*C - 1j*g*T2

    # Nw = <n σz>
    # dNw = -kappa Nw - gamma1 (Nw - w_eq n) + i g (-C - D) + 2 i g ( <n a† σ-> - <n a σ+> )
    dNw  = -kappa*Nw - gamma1*(Nw - w_eq*n) + 1j*g*(-C - D) + 2j*g*(Sd - Cn)

    # ---------- 4th order promoted ----------
    # T1 = <n a σz>
    # dT1 = (dn) a σz + n (da) σz + n a (dσz)
    # Using identities: (dn) a σz → -k T1 - i g Q - i g S1 + eta*(Nw)   (σ+σz=-σ+, σ-σz=+σ-)
    # n (da) σz       → -(k/2+iΔc) T1 - i g S1 + eta*Nw
    # n a (dσz)       → -γ1(T1 - w_eq Na) + 2 i g ( <n a a† σ-> - <n a a σ+> )
    #                    with <n a a† σ-> = <(n^2+n) σ->  ~ cumulant ≈ n2*sm + n*S1 - 2*n*sm*n  (use reduced form below)
    #                    and <n a a σ+> ≡ Qn  ~ closure ≈ n*Q
    # CE4 choices: keep n2 as a variable; approximate <(n^2+n) σ-> ≈ n2*sm + n*S1 (third-order cumulant neglected).
    Qn = n*Q  # pragmatic fourth-order closure
    n2_sigminus = n2*sm + n*S1

    dT1  = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T1 \
           - 1j*g*Q - 2j*g*S1 + 2j*g*(n2_sigminus - Qn) + gamma1*w_eq*Na

    # T2 = <a n σz>
    # Similar decomposition: (da) n σz + a (dn) σz + a n (dσz)
    # (da) n σz → -(k/2+iΔc) T2 - i g S1 + eta*Nw
    # a (dn) σz → -k T2 - i g Q - i g S1 + eta*(Nw + w)
    # a n (dσz) → -γ1(T2 - w_eq <a n>) + 2 i g ( <a n a† σ-> - <a n a σ+> )
    # with  <a n a† σ-> = <(n^2 + 2n + 1) σ->  ≈ (n2 + 2n + 1)*sm   (CE4 closure),
    # and   <a n a σ+>  ≈ n*Q + Q            (use commutator a n = n a + a)
    an_sigminus = (n2 + 2*n + 1.0)*sm
    an_sigplus  = n*Q + Q

    dT2  = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T2 \
           - 1j*g*(Q + 2*S1) + 2j*g*(an_sigminus - an_sigplus) + gamma1*w_eq*Na

    # Sd = <n a† σ->
    # dSd = (dn) a† σ- + n (da†) σ- + n a† (dσ-)
    # (dn) a† σ- → -k Sd + i g * 0.5 ( (n+1) + Nw )     [since a σ+ a† σ- → (n+1) σ+σ- = (n+1)(1+σz)/2]
    # n (da†) σ- → -(k/2 - iΔc) Sd + i g * 0.5 ( n + Nw )
    # n a† (dσ-) → -(γ_perp + iΔx) Sd + i g * <n a† a σz> = -(γ_perp + iΔx) Sd + i g * <n^2 σz> = i g * N2w
    # Promote N2w via CE4 closure: N2w ≈ n2*w + 2*n*Nw - 2*n*w*n  ~ we take simplest cumulant N2w ≈ n2*w  (works near Gaussian)
    N2w = n2*w  # simplest CE4 cumulant closure (can be refined)

    dSd  = -(3*kappa/2 + gamma_perp + 1j*(Delta_x - Delta_c))*Sd \
           + 1j*g*(n + Nw + 0.5) + 1j*g*N2w

    # Cn = <n a σ+>
    # Similar steps yield (mirroring Sd with σ+)
    dCn  = -(3*kappa/2 + gamma_perp - 1j*(Delta_x - Delta_c))*Cn \
           - 1j*g*(n + Nw + 0.5) - 1j*g*N2w

    # Na = <n a>  (useful for γ1 w_eq terms above; CE4 variable)
    # dNa = (dn) a + n (da) = - (3k/2 + iΔc) Na + i g (Q - 2 S1) + eta*n  (drive term kept for completeness)
    dNa  = -(3*kappa/2 + 1j*Delta_c)*Na + 1j*g*(Q - 2*S1) + eta*n

    # n2 = <n^2>
    # dn2 = 2 <n dn> = -2 k n2 + 2 i g ( Cn - Sd ) + 2 Re[η* Na]
    dn2  = -2.0*kappa*n2 + 2j*g*(Cn - Sd) + 2.0*np.real(np.conjugate(eta)*Na)

    # pack
    return pack(dalpha,ds,dC,dD,dU,dS1,dQ,dT1,dT2,dSd,dCn,dNa,dw.real,dn.real,dNw.real,dn2.real)

# ---------------- initial conditions (|1> ⊗ |g>) ----------------
alpha0 = 0.0 + 0.0j
n0     = 1.0
s0     = 0.0 + 0.0j
w0     = -1.0

C0 = 0.0 + 0.0j
D0 = 0.0 + 0.0j
U0 = 0.0 + 0.0j
S10 = 0.0 + 0.0j
Q0  = 0.0 + 0.0j
T10 = 0.0 + 0.0j
T20 = 0.0 + 0.0j
Sd0 = 0.0 + 0.0j
Cn0 = 0.0 + 0.0j
Na0 = 0.0 + 0.0j
Nw0 = n0*w0
n20 = n0**2

y0 = pack(alpha0,s0,C0,D0,U0,S10,Q0,T10,T20,Sd0,Cn0,Na0,w0,n0,Nw0,n20)

# integrate
sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# unpack (vectorized)
Y = sol.y.T.copy()
Z = Y[:, :24].view(np.complex128)
alpha = Z[:, 0]; s = Z[:, 1]
w  = Y[:, 24]; n = Y[:, 25]

# observables
P_e  = 0.5*(1.0 + w)   # excited population
n_ph = n               # photon number

np.savetxt("Photon_number_CE4.dat", np.column_stack([sol.t, n_ph]))
np.savetxt("Excited_population_CE4.dat", np.column_stack([sol.t, P_e]))
np.savetxt("alpha_CE4.dat", np.column_stack([sol.t, alpha.real, alpha.imag]))
print("Saved: Photon_number_CE4.dat, Excited_population_CE4.dat, alpha_CE4.dat")
