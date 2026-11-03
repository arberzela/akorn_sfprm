import sys, torch, math
sys.path.insert(0, '/Users/zelaa/ICML27/akorn_sfprm/code')
from esfprm_pilot import *
torch.manual_seed(0)
T, d, n_sym = 36, 64, 6
# 1) closed-form drive vs autograd of the energy (double precision)
m = EnergySFPRM(n_sym, T, d, 8, 256).double()
clues = torch.randint(0, n_sym + 1, (3, T))
x = m.embed(clues).detach()
z = random_sphere((3, T, d), 'cpu').double().requires_grad_(True)
E = m.energy(z, x).sum()
g, = torch.autograd.grad(E, z)
v_auto = -proj(z, g)
v_closed = m.drive(z, x)
print('drive closed-form vs autograd: max abs err', (v_auto - v_closed).abs().max().item())
# per-term norms at init (float32)
m = EnergySFPRM(n_sym, T, d, 8, 1024)
with torch.no_grad():
    z = random_sphere((64, T, d), 'cpu'); x = m.embed(torch.randint(0, n_sym + 1, (64, T)))
    c = (z * x).sum(-1, keepdim=True); f, th = angle_factor(c)
    inj = m.kappa0 * f * proj(z, x)
    q, k = m._qk(z)
    logits = torch.einsum('bhte,bhse->bhts', q, k) * m.betaA.view(1, -1, 1, 1)
    eye = torch.eye(T, dtype=torch.bool)
    a = logits.masked_fill(eye, float('-inf')).softmax(-1)
    ak = torch.einsum('bhts,bhse->bhte', a, k); aq = torch.einsum('bhst,bhse->bhte', a, q)
    att = ((torch.einsum('hed,bhte->bhtd', m.Wq, ak) + torch.einsum('hed,bhte->bhtd', m.Wk, aq)) * m.kappaA.view(1, -1, 1, 1)).sum(1)
    mm = torch.einsum('kd,btd->btk', m.Xi, z) * m.betaM
    mem = m.kappaM * torch.einsum('btk,kd->btd', mm.softmax(-1), m.Xi)
    print('init norms per token: inj %.3f att %.3f mem %.3f  total %.3f ; logits std %.3f; mem-logit std %.3f' % (
        inj.norm(dim=-1).mean(), proj(z, att).norm(dim=-1).mean(), proj(z, mem).norm(dim=-1).mean(),
        m.drive(z, x).norm(dim=-1).mean(), logits.std(), mm.std()))
    for eta in [1.0, 0.5, 0.25]:
        zz = z.clone(); inc = 0; ep = m.energy(zz, x)
        for i in range(16):
            zz, r, nv = m.step(zz, x, eta); e = m.energy(zz, x); inc += (e > ep).float().mean().item(); ep = e
        print(f'eta {eta}: energy-increase frac over 16 iters {inc/16:.3f}, final resid {r.mean():.2e}, final |v| {nv.mean():.3f}')
