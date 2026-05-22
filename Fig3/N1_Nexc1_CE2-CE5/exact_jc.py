import numpy as np
from scipy.linalg import expm

def jc_evolution(n0, times):
    Nmax = n0 + 3
    # photon annihilation
    a = np.zeros((Nmax,Nmax), complex)
    for n in range(1,Nmax):
        a[n-1,n] = np.sqrt(n)
    adag = a.conj().T
    # spin s+ and s- etc
    sp = np.array([[0,1],[0,0]], complex)
    sm = sp.conj().T
    sz = np.array([[1,0],[0,-1]], complex)
    omega=1.0
g = 0.1
    H = np.kron(adag.dot(a), np.eye(2)) + 0.5*omega*np.kron(np.eye(Nmax), sz) + g*(np.kron(a,sp) + np.kron(adag,sm))
    psi0 = np.zeros((Nmax*2,), complex)
    # basis ordering: photon index p in {0..Nmax-1}, atom basis [e,g]
    # so state vector index = p*2 + atom_index
    # choose atom_index=1 for ground state |g>
    psi0[n0*2 + 1] = 1.0
    res = []
    for t in times:
        U = expm(-1j*H*t)
        psi = U.dot(psi0)
        Cop = np.kron(a, sp)
        Cval = np.vdot(psi, Cop.dot(psi))
        res.append(Cval)
    return np.array(res)

for n0 in [1,10]:
    times = np.linspace(0,1e-3,3)
    Cvals = jc_evolution(n0, times)
    dC = (Cvals[1]-Cvals[0])/(times[1]-times[0])
    print("n0",n0,"dC",dC)
