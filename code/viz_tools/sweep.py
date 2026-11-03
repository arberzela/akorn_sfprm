import numpy as np, random, sys
sys.path.insert(0,'.')
from toy3 import solve, make_puzzle
import itertools
random.seed(11)
puzzles=[]
for _ in range(10):
    puz,sol,nc=make_puzzle(4,2,2,random.randint(5,6),random); puzzles.append((puz.reshape(-1),sol.reshape(-1)))
for kA,bA,kM,rho,K,eta in [(1,6,1,12,40,.25),(2,6,1,12,40,.2),(1.5,4,1,20,60,.2),(2,8,1,20,60,.15),(1,6,0.5,12,60,.25),(2,4,0.7,30,80,.2)]:
    for anneal in [True,False]:
        s=0;o=0;ep=0
        for i,(c,sol) in enumerate(puzzles):
            rng=np.random.default_rng(100+i)
            R=[solve(c,rng,anneal,kA=kA,bA=bA,kM=kM,rho=rho,K=K,eta=eta,iters=K+200) for _ in range(12)]
            corr=[(p==sol).all() for _,p,_ in R]; Es=[e for *_,e in R]
            s+=np.mean(corr); o+=any(corr); ep+=corr[int(np.argmin(Es))]
        print(kA,bA,kM,rho,K,eta,'anneal' if anneal else 'fixed ', round(s/10,2), ep/10, o/10, flush=True)
