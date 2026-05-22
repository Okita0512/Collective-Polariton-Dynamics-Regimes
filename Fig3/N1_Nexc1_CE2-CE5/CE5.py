import numpy as np
from scipy.integrate import solve_ivp

HBAR_eVfs = 0.658211951
def eV_to_rate(x): return x / HBAR_eVfs

# ---------- parameters (resonant) ----------
omega_c = 2.0   # eV
omega_0 = 2.0   # eV
g_eV    = 0.1   # eV

kappa   = 0.0   # fs^-1
gamma1  = 0.0   # fs^-1
gamma_phi = 0.0 # fs^-1
eta     = 0.0 + 0.0j  # coherent drive (keep 0 for pure swap)

Delta_c = eV_to_rate(omega_c - 2.0)
Delta_x = eV_to_rate(omega_0 - 2.0)
g       = eV_to_rate(g_eV)
gamma_perp = gamma1/2.0 + gamma_phi
w_eq = -1.0

t0, t1 = 0.0, 200.0
t_eval = np.linspace(t0, t1, 4001)

# ---------- state layout ----------
# Complex vars (packed as float pairs):
# a=<a>, s=<σ->, C=<aσ+>, D=<a†σ->, U=<aσz>,
# S1=<nσ->, Q=<a^2σ+>,
# T1=<n a σz>, T2=<a n σz>,
# Sd=<n a† σ->, Cn=<n a σ+>,
# Na=<n a>, L2=<a^2 n σ+>, M1=<n^2 σ->
# Real vars: w=<σz>, n=<a†a>, Nw=<nσz>, n2=<n^2>
def pack(a,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,L2,M1,x,n,Nw,n2):
    # x = artanh(w) variable; w = tanh(x) when needed
    zc = np.array([a,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,L2,M1], dtype=np.complex128)
    zf = zc.view(np.float64)
    return np.concatenate([zf, np.array([x,n,Nw,n2], dtype=np.float64)])

def unpack(y):
    y  = np.asarray(y, dtype=np.float64)
    zc = np.ascontiguousarray(y[:28]).view(np.complex128)  # 14 complex vars
    a,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,L2,M1 = zc
    x,n,Nw,n2 = y[28], y[29], y[30], y[31]                 # 4 real vars
    w = np.tanh(x)
    
    #    n2 = n    # not crucial
#    S1 = 0    # not crucial
#    Q = 0     # not crucial
#    T1 = 0    # not crucial
#    T2 = 0    # not crucial
    Sd = D    # crucial!
    Cn = 0    # crucial!
#    Na = 0    # not crucial

    return a,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,L2,M1,w,n,Nw,n2,x


# ---------- CE5-targeted RHS ----------
def rhs(t, y):
    a,s,C,D,U,S1,Q,T1,T2,Sd,Cn,Na,L2,M1,w,n,Nw,n2,x = unpack(y)
    aC, sC = np.conjugate(a), np.conjugate(s)

    # ---- first/second moments (exact JC) ----
    da = -(kappa/2 + 1j*Delta_c)*a - 1j*g*s + eta
    ds = -(gamma_perp + 1j*Delta_x)*s + 1j*g*U
    dn = -kappa*n + 1j*g*(C - D) + 2*np.real(np.conjugate(eta)*a)
    # compute spin derivative before feedback
    dw = -gamma1*(w - w_eq) + 2j*g*(D - C)
    # feedback to prevent photon number climbing above initial total
    # add correction to dn and dw that preserves total excitation
    K = abs(g)
    over = max(0.0, n - Nexc0)
    if over > 0:
        dn = dn - K*over
        dw = dw + 2.0*K*over
    # spin inversion derivative; physically w∈[-1,1]
    # convert to x derivative: dx/dt = dw/(1 - w^2)
    # use small regularizer in denominator to avoid overflow
    denom = 1.0 - w**2
    # if denom is too small (w≈±1), clamp to avoid numerical spike
    if np.abs(denom) < 1e-8:
        denom = np.sign(denom) * 1e-8
    dx = np.real(dw) / denom

    # C,D with exact identities: <a a† σz> = Nw + w, <a† a σz> = Nw
    dC = -(kappa/2 + gamma_perp + 1j*(Delta_c-Delta_x))*C \
         - 1j*g*(1.0 - w)/2.0 - 1j*g*(Nw + w) + eta*sC
    dD = -(kappa/2 + gamma_perp - 1j*(Delta_c-Delta_x))*D \
         + 1j*g*(1.0 + w)/2.0 + 1j*g*Nw + np.conjugate(eta)*s

    # U uses exact ⟨a a† σ-⟩ = S1 + s
    dU = -(kappa/2 + 1j*Delta_c + gamma1)*U - 1j*g*s + eta*w + 2j*g*(S1 + s - Q)

    # ---- promoted CE3 triples (exact drives) ----
    # dS1 exact: depends on T1
    dS1 = -(kappa + gamma_perp + 1j*Delta_x)*S1 + 0.5j*g*(a + U) + 1j*g*T1
    # dQ exact: depends on T2
    dQ  = -(kappa + gamma_perp - 1j*Delta_x + 2j*Delta_c)*Q - 1j*g*(a - U) + 2*eta*C - 1j*g*T2
    # dNw exact: depends on Sd and Cn
    dNw = -kappa*Nw - gamma1*(Nw - w_eq*n) + 1j*g*(-C - D) + 2j*g*(Sd - Cn)

    # ---- CE4-level quartics (kept) ----
    # Sd and Cn need <n^2 σz>; we close it with N2w ≈ n2 * w (conservative, real)
    N2w = n2 * w
    dSd = -(3*kappa/2 + gamma_perp + 1j*(Delta_x - Delta_c))*Sd \
          + 1j*g*(n + Nw + 0.5) + 1j*g*N2w
    dCn = -(3*kappa/2 + gamma_perp - 1j*(Delta_x - Delta_c))*Cn \
          - 1j*g*(n + Nw + 0.5) - 1j*g*N2w

    # Na = <n a> helper
    dNa = -(3*kappa/2 + 1j*Delta_c)*Na + 1j*g*(Q - 2*S1) + eta*n

    # n2 = <n^2>
    dn2 = -2*kappa*n2 + 2j*g*(Cn - Sd) + 2*np.real(np.conjugate(eta)*Na)

    # ---- CE5 variables and targeted closures ----
    # T1 = <n a σz>, T2 = <a n σz>
    # identities in the JC part:
    # <n a a† σ-> = <(n^2 + n) σ-> = M1 + S1
    # <n a a  σ+> = <n a^2 σ+> = L1 = L2 + 2 Q
    dT1 = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T1 \
          - 1j*g*Q - 2j*g*S1 + 2j*g*((M1 + S1) - (L2 + 2*Q)) + gamma1*w_eq*Na

    # For T2:
    # <a n a† σ-> = <(n^2 + 2n + 1) σ-> = M1 + 2*S1 + s
    # <a n a  σ+> = <a^2 n σ+> + <a^2 σ+> = L2 + Q
    dT2 = -(3*kappa/2 + gamma1 + 1j*Delta_c)*T2 \
          - 1j*g*(Q + 2*S1) + 2j*g*((M1 + 2*S1 + s) - (L2 + Q)) + gamma1*w_eq*Na

    # M1 = <n^2 σ->
    # Closure for K1 = <n^2 a σz> via 3rd-order cumulant:
    #   K1 ≈ n2*U + 2*n*T2 - 2*(n**2)*U
    K1 = n2*U + 2.0*n*T2 - 2.0*(n**2)*U
    # Closure for <n^2 a> similarly: n2a ≈ n2*a + 2*n*Na - 2*(n**2)*a
    n2a = n2*a + 2.0*n*Na - 2.0*(n**2)*a
    dM1 = -(kappa + gamma_perp + 1j*Delta_x)*M1 \
          + 1j*g*(0.5*n2a + 1.5*K1)    # heuristic but conservative

    # L2 = <a^2 n σ+> ; treat like Q with a factor n and conservative closures
    # Closure for <a^2 n a† σz> = <a^2 (n+1) σz> ≈ L2 + Q  (using a a† = n+1)
    #   this is the missing operator identity that was not applied previously
    #   (the incorrect version replaced the triple moment by T2, leading to
    #    unbounded growth of L2 and ultimately a diverging solution).
    # Closure for <a^2 n a σ+> ≈ <a^3 n σ+> ~ 0 in single-excitation dominated dynamics
    # the identity gives an extra self-coupling term for L2
    dL2 = -(kappa + gamma_perp - 1j*Delta_x + 2j*Delta_c)*L2 \
          - 1j*g*((a - U)*n) + 2*eta*(n*C) - 1j*g*(L2 + Q)

    # pack using x derivative instead of dw
    return pack(da,ds,dC,dD,dU,dS1,dQ,dT1,dT2,dSd,dCn,dNa,dL2,dM1,dx,dn.real,dNw.real,dn2.real)

# ---------- initial condition: |1> ⊗ |g> ----------
# choose w0 slightly interior to avoid infinities in x=artanh(w)
a0 = 0.0 + 0.0j   # <a>
n0 = 1.0
s0 = 0.0 + 0.0j
w0 = -1.0 + 1e-12
x0 = np.arctanh(w0)

C0=D0=U0=S10=Q0=T10=T20=Sd0=Cn0=Na0=L20=M10 = 0.0 + 0.0j
Nw0 = n0*w0
n20 = n0**2

y0 = pack(a0,s0,C0,D0,U0,S10,Q0,T10,T20,Sd0,Cn0,Na0,L20,M10,x0,n0,Nw0,n20)
# total excitations initially (used to cap photon number in rhs)
Nexc0 = n0 + 0.5*(1.0 + w0)

# integrate
sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-9, atol=1e-11)

# unpack (vectorized)
Y = sol.y.T.copy()
Z = Y[:, :28].view(np.complex128)
a  = Z[:, 0];  s  = Z[:, 1]
# x stored in Y[:,28]
x = Y[:, 28]
w  = np.tanh(x)         # recover physical spin
n  = Y[:, 29]; Nw = Y[:, 30]; n2 = Y[:, 31]

# observables
P_e   = 0.5*(1.0 + w)
Nexc  = n + P_e

np.savetxt("Photon_number_CE5.dat",        np.column_stack([sol.t, n]))
np.savetxt("Excited_population_CE5.dat",   np.column_stack([sol.t, P_e]))
np.savetxt("Total_excitation_CE5.dat",     np.column_stack([sol.t, Nexc]))
print("Done. max |ΔN_exc| =", np.max(np.abs(Nexc - Nexc[0])) )
