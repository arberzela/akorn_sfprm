import sys, torch, math, time, torch.nn.functional as F
sys.path.insert(0, '/Users/zelaa/ICML27/akorn_sfprm/code')
from esfprm_pilot import *
torch.manual_seed(0)
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else 'cpu')
kind = sys.argv[1]
T, d, n_sym = 16, 64, 4
if kind == 'sfprm':
    m = SFPRM(n_sym, T, d, 4, 2, structured=(4,2,2))
elif kind == 'flat':
    m = EnergySFPRM(n_sym, T, d, 8, 1024, structured=(4,2,2))
else:
    m = EnergySFPRM(n_sym, T, d, 8, 1024, hidden=True, structured=(4,2,2))
m.to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.98))
t0 = time.time()
for step in range(300):
    clues = torch.randint(1, n_sym + 1, (128, T), device=dev)     # all cells have digits
    target = clues.roll(-1, 1) - 1                                  # digit of the right neighbour (cyclic)
    x = m.embed(clues)
    z = random_sphere((128, m.n_state, d), dev)
    for _ in range(8):
        z, _, _ = m.step(z, x, 1.0)
    logits = m.readout(m.visible(z))
    loss = F.cross_entropy(logits.reshape(-1, n_sym), target.reshape(-1))
    opt.zero_grad(); loss.backward()
    if step in (0, 100, 299):
        gn = {}
        for name, p in m.named_parameters():
            if p.grad is not None:
                key = name.split('.')[0]
                gn[key] = gn.get(key, 0.0) + p.grad.norm().item() ** 2
        print(kind, 'step', step, 'grad norms', {k: round(v ** 0.5, 5) for k, v in gn.items()})
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 50 == 0 or step == 299:
        acc = (logits.argmax(-1) == target).float().mean().item()
        print(kind, step, 'loss %.3f acc %.3f  (%.0fs)' % (loss.item(), acc, time.time() - t0), flush=True)
