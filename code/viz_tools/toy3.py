import numpy as np, sys, random
sys.path.insert(0, '/Users/zelaa/ICML27/akorn_sfprm/code')
from esfprm_pilot import make_puzzle
T=16; D=4
peers=np.zeros((T,T))
for t in range(T):
    for s in range(T):
        if s!=t and (s//4==t//4 or s%4==t%4 or ((s//4)//2==(t//4)//2 and (s%4)//2==(t%4)//2)): peers[t,s]=1
I=np.eye(D)
def norm(z): return z/np.linalg.norm(z,axis=1,keepdims=True)
def proj(z,v): return v-(v*z).sum(1,keepdims=True)*z
def expm(z,v):
    n=np.linalg.norm(v,axis=1,keepdims=True); return np.cos(n)*z+np.sinc(n/np.pi)*v
def parts(z,clues,k0,kA,bA,kM,bM):
    E=0; g=np.zeros_like(z)
    for t in range(T):
        if clues[t]:
            x=I[clues[t]-1]; c=np.clip(z[t]@x,-1+1e-9,1-1e-9); th=np.arccos(c)
            E+=k0*0.5*th**2; g[t]+= -k0*th/np.sqrt(1-c*c)*x   # euclidean grad of 1/2 th^2 = -th/sin th * x
    G=bA*(z@z.T); L=np.where(peers>0,G,-np.inf); m=L.max(1,keepdims=True); A=np.exp(L-m); Z=A.sum(1,keepdims=True); A/=Z
    E+=kA/bA*(np.log(Z[:,0])+m[:,0]).sum(); g+=kA*((A+A.T)@z)
    M=bM*z; mm=M.max(1,keepdims=True); p=np.exp(M-mm); Zp=p.sum(1,keepdims=True); p/=Zp
    E-=kM/bM*(np.log(Zp[:,0])+mm[:,0]).sum(); g-=kM*p
    return E,g
def solve(clues,rng,anneal,k0=1.0,kA=1.0,bA=6.0,kM=1.0,bM=12.0,rho=12,K=40,eta=0.25,iters=300,trace=False):
    z=norm(rng.normal(size=(T,D)))
    for i in range(iters):
        b=bM*(rho**(-(1-min(i/K,1))) if anneal else 1)
        E,g=parts(z,clues,k0,kA,bA,kM,b)
        z=norm(expm(z,-eta*proj(z,g)))
    E,_=parts(z,clues,k0,kA,bA,kM,bM)
    return z,z.argmax(1)+1,E
