import json, random, numpy as np, sys
sys.path.insert(0,'/Users/zelaa/ICML27/akorn_sfprm/code')
from esfprm_pilot import make_puzzle
R='/Users/zelaa/ICML27/akorn_sfprm/code/results/'
random.seed(2026)
puz=[]
while len(puz)<40:
    p,s,nc=make_puzzle(4,2,2,random.randint(5,6),random)
    puz.append([''.join(map(str,p.reshape(-1))), ''.join(map(str,s.reshape(-1)))])
def eig(name, k=0):
    d=json.load(open(R+name+'_eval.json'))
    e=d['test_ood']['jac_eigs'][k]
    return [[round(a,3),round(b,3)] for a,b in e]
def restarts(name):
    d=json.load(open(R+name+'_eval.json'))['test_ood']['restarts']
    return {k:[round(100*v[b],1) for b in sorted(v,key=int)] for k,v in d.items() if k!='first'}
def single(name, mode):
    d=json.load(open(R+name+'_eval.json'))
    return {s:round(100*d[s][mode+'/board_acc'],1) for s in ['test_id','test_ood']}
data=dict(puzzles=puz,
          eig_sfprm=eig('s4_sfprm'), eig_energy=eig('s4_ev2_norank'),
          r4_sfprm=restarts('s4_sfprm'), r4_code=restarts('s4_ev2_norank'), r4_first=restarts('s4_esfprm'),
          r6_sfprm=restarts('s6_sfprm'), r6_code=restarts('s6_ev2'), r6_first=restarts('s6_esfprm'),
          Bs=[1,2,4,8,16,32,64])
s=json.dumps(data,separators=(',',':'))
open('vizdata.json','w').write(s)
print(len(s), 'bytes;', len(data['eig_sfprm']), len(data['eig_energy']), 'eigs')
print({k:data[k] for k in ['r4_code','r6_code']})
