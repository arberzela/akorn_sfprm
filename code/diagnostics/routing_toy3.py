import sys, torch, math, time, torch.nn.functional as F
sys.path.insert(0, '/Users/zelaa/ICML27/akorn_sfprm/code')
from esfprm_pilot import *
torch.manual_seed(0)
kind, dev, eta, K, kappaA, lr = sys.argv[1], torch.device(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
T, d, n_sym = 16, 64, 4
if kind == 'sfprm':
    m = SFPRM(n_sym, T, d, 4, 2, structured=(4,2,2))
elif kind == 'flat':
    m = EnergySFPRM(n_sym, T, d, 8, 1024, structured=(4,2,2), kappaA=kappaA)
else:
    m = EnergySFPRM(n_sym, T, d, 8, 1024, hidden=True, structured=(4,2,2), kappaA=kappaA)
m.to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.98))
t0 = time.time(); inc = tot = 0
for step in range(400):
    clues = torch.randint(1, n_sym + 1, (128, T), device=dev)
    target = clues.roll(-1, 1) - 1
    x = m.embed(clues)
    z = random_sphere((128, m.n_state, d), dev)
    with torch.no_grad():
        ep = m.energy(z, x) if m.is_energy else None
    for _ in range(K):
        z, _, nv = m.step(z, x, eta)
        if m.is_energy:
            with torch.no_grad():
                e = m.energy(z, x); inc += (e > ep + 1e-6).float().sum().item(); tot += e.numel(); ep = e
    logits = m.readout(m.visible(z))
    loss = F.cross_entropy(logits.reshape(-1, n_sym), target.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 50 == 0 or step == 399:
        acc = (logits.argmax(-1) == target).float().mean().item()
        extra = ' |v| %.3f' % nv.mean().item() if m.is_energy else ''
        print(sys.argv[1:], step, 'loss %.3f acc %.3f inc %.3f (%.0fs)' % (loss.item(), acc, inc / max(tot, 1), time.time() - t0) + extra, flush=True)
        inc = tot = 0
