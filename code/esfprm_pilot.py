"""Pilot experiment: SFPRM versus energy-based SFPRM (E-SFPRM) on small Sudoku.

Subcommands
  gen    generate n x n Sudoku puzzles with unique solutions (train / ID test / OOD test)
  train  train SFPRM or E-SFPRM with deep supervision and truncated BPTT
  eval   accuracy, iterations to halt, Jacobian spectrum at the halted state,
         and accuracy against the number of restarts B for energy / anchor-angle /
         majority selection

Everything is deliberately small so that it runs on a laptop CPU/MPS in minutes.
"""
import argparse, json, math, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Sudoku generation
# ----------------------------------------------------------------------------

def _ok(grid, n, br, bc, r, c, v):
    if v in grid[r, :] or v in grid[:, c]:
        return False
    r0, c0 = (r // br) * br, (c // bc) * bc
    return v not in grid[r0:r0 + br, c0:c0 + bc]


def random_full_grid(n, br, bc, rng):
    grid = np.zeros((n, n), dtype=np.int64)
    cells = [(r, c) for r in range(n) for c in range(n)]

    def fill(k):
        if k == n * n:
            return True
        r, c = cells[k]
        vals = list(range(1, n + 1))
        rng.shuffle(vals)
        for v in vals:
            if _ok(grid, n, br, bc, r, c, v):
                grid[r, c] = v
                if fill(k + 1):
                    return True
                grid[r, c] = 0
        return False

    assert fill(0)
    return grid


def count_solutions(puzzle, n, br, bc, limit=2):
    grid = puzzle.copy()
    empties = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0]
    count = 0

    def candidates(r, c):
        return [v for v in range(1, n + 1) if _ok(grid, n, br, bc, r, c, v)]

    def rec():
        nonlocal count
        if count >= limit:
            return
        # most-constrained-cell ordering
        best, best_c = None, None
        for (r, c) in empties:
            if grid[r, c] == 0:
                cs = candidates(r, c)
                if best is None or len(cs) < len(best_c):
                    best, best_c = (r, c), cs
                    if len(cs) <= 1:
                        break
        if best is None:
            count += 1
            return
        r, c = best
        for v in best_c:
            grid[r, c] = v
            rec()
            grid[r, c] = 0
            if count >= limit:
                return

    rec()
    return count


def make_puzzle(n, br, bc, target_clues, rng):
    """Remove cells from a random full grid in random order while the puzzle stays unique."""
    sol = random_full_grid(n, br, bc, rng)
    puz = sol.copy()
    order = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(order)
    clues = n * n
    for (r, c) in order:
        if clues <= target_clues:
            break
        v = puz[r, c]
        puz[r, c] = 0
        if count_solutions(puz, n, br, bc) == 1:
            clues -= 1
        else:
            puz[r, c] = v
    return puz, sol, clues


def _gen_worker(args):
    n, br, bc, lo, hi, seed, count = args
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        target = rng.randint(lo, hi)
        puz, sol, clues = make_puzzle(n, br, bc, target, rng)
        out.append((puz, sol, clues))
    return out


def cmd_gen(a):
    from multiprocessing import Pool
    n, br, bc = a.n, a.br, a.bc
    specs = {
        'train': (a.n_train, a.id_clues[0], a.id_clues[1], 1000),
        'test_id': (a.n_test, a.id_clues[0], a.id_clues[1], 2000),
        'test_ood': (a.n_test, a.ood_clues[0], a.ood_clues[1], 3000),
    }
    os.makedirs(a.out, exist_ok=True)
    for name, (count, lo, hi, seed0) in specs.items():
        per = max(1, count // a.workers)
        jobs = [(n, br, bc, lo, hi, seed0 + w, per) for w in range(a.workers)]
        with Pool(a.workers) as pool:
            res = pool.map(_gen_worker, jobs)
        items = [it for r in res for it in r][:count]
        puz = np.stack([it[0] for it in items])
        sol = np.stack([it[1] for it in items])
        clues = np.array([it[2] for it in items])
        np.savez(os.path.join(a.out, f'{name}.npz'), puzzles=puz, solutions=sol, clues=clues)
        print(f'{name}: {len(items)} puzzles, clues mean {clues.mean():.1f} '
              f'min {clues.min()} max {clues.max()}')


# ----------------------------------------------------------------------------
# Sphere utilities (token-wise, last dimension)
# ----------------------------------------------------------------------------

def normalize(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def proj(z, v):
    """Tangent projection P_z v = v - <v,z> z."""
    return v - (v * z).sum(-1, keepdim=True) * z


def _sinc(a):
    # sin(a)/a, safe at 0
    return torch.sinc(a / math.pi)


def exp_map(z, v):
    """Exp_z(v) for tangent v (token-wise)."""
    nv = v.norm(dim=-1, keepdim=True)
    return torch.cos(nv) * z + _sinc(nv) * v


def angle_factor(c):
    """theta/sin(theta) as a function of c = cos(theta), safe near c = 1."""
    c = c.clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(c)
    return th / torch.sqrt(1 - c * c), th


def log_map(u, v):
    """Log_u(v) = theta/sin(theta) P_u v, with the angle."""
    c = (u * v).sum(-1, keepdim=True)
    f, th = angle_factor(c)
    return f * proj(u, v), th


def slerp(u, v, eta):
    """Slerp_eta(u,v) = Exp_u(eta Log_u v); eta broadcastable to (..., 1)."""
    lg, _ = log_map(u, v)
    return exp_map(u, eta * lg)


def random_sphere(shape, device, generator=None):
    return normalize(torch.randn(*shape, device=device, generator=generator))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class Embedding(nn.Module):
    """x_t = Norm(E_digit[clue_t] + pos_t), pos_t learned freely or a learned linear map of
    row / column / box one-hot features (structured)."""

    def __init__(self, n_sym, T, d, structured=None):
        super().__init__()
        self.dig = nn.Embedding(n_sym + 1, d)
        self.blank = nn.Parameter(torch.randn(d) / math.sqrt(d))     # used when tied to a codebook
        if structured is None:
            self.pos = nn.Parameter(torch.randn(T, d) / math.sqrt(d))
            self.feat = None
        else:
            n, br, bc = structured
            feat = torch.zeros(T, 3 * n)
            for t in range(T):
                r, c = divmod(t, n)
                b = (r // br) * (n // bc) + (c // bc)
                feat[t, r] = 1; feat[t, n + c] = 1; feat[t, 2 * n + b] = 1
            self.register_buffer('feat', feat)
            self.pos_map = nn.Linear(3 * n, d, bias=False)

    def forward(self, clues, proto=None):
        pos = self.pos if self.feat is None else self.pos_map(self.feat)
        if proto is None:
            return normalize(self.dig(clues) + pos)
        table = torch.cat([self.blank.view(1, -1), proto], 0)       # digit k -> prototype xi_k
        return normalize(table[clues] + pos)


class Readout(nn.Module):
    def __init__(self, d, n_sym):
        super().__init__()
        self.W = nn.Linear(d, n_sym, bias=True)
        self.s = nn.Parameter(torch.tensor(float(math.sqrt(d))))

    def forward(self, z):
        return self.s * self.W(z)


class NGPTLayer(nn.Module):
    """One nGPT layer: attention and MLP sub-layers folded in by normalized residual steps."""

    def __init__(self, d, H, alpha_init=0.25, mlp='gelu', n_mem=None):
        super().__init__()
        self.d, self.H, self.dh = d, H, d // H
        self.mlp = mlp
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.sqk = nn.Parameter(torch.full((H,), math.sqrt(self.dh)))
        self.alpha_A = nn.Parameter(torch.full((d,), alpha_init))
        self.alpha_M = nn.Parameter(torch.full((d,), alpha_init))
        if mlp == 'gelu':
            self.W1 = nn.Linear(d, 4 * d)
            self.W2 = nn.Linear(4 * d, d)
        else:                                   # tied memory: M = Xi^T softmax(beta Xi h)
            self.Xi = nn.Parameter(torch.randn(n_mem or 4 * d, d) / math.sqrt(d))
            self.betaM_ = nn.Parameter(torch.tensor(math.log(math.expm1(math.sqrt(d)))))

    def attn(self, h):
        B, T, d = h.shape
        q = normalize(self.Wq(h).view(B, T, self.H, self.dh)).transpose(1, 2)
        k = normalize(self.Wk(h).view(B, T, self.H, self.dh)).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.H, self.dh).transpose(1, 2)
        logits = torch.einsum('bhtd,bhsd->bhts', q, k) * self.sqk.view(1, -1, 1, 1)
        a = logits.softmax(-1)
        o = torch.einsum('bhts,bhsd->bhtd', a, v).transpose(1, 2).reshape(B, T, d)
        return self.Wo(o)

    def forward(self, h):
        A = self.attn(h)
        h = normalize(h + self.alpha_A * (normalize(A) - h))
        if self.mlp == 'gelu':
            M = self.W2(F.gelu(self.W1(h)))
        else:
            m = torch.einsum('kd,btd->btk', self.Xi, h) * F.softplus(self.betaM_)
            M = torch.einsum('btk,kd->btd', m.softmax(-1), self.Xi)
        h = normalize(h + self.alpha_M * (normalize(M) - h))
        return h


class SFPRM(nn.Module):
    """z~ = f_theta(Slerp_lambda(z, x)), z <- Slerp_eta(z, z~)."""

    def __init__(self, n_sym, T, d, H, depth, lam=0.5, structured=None, mlp='gelu', n_mem=None):
        super().__init__()
        self.embed = Embedding(n_sym, T, d, structured)
        self.layers = nn.ModuleList([NGPTLayer(d, H, mlp=mlp, n_mem=n_mem) for _ in range(depth)])
        self.readout = Readout(d, n_sym)
        self.lam = lam
        self.is_energy = False
        self.T, self.n_state = T, T

    def visible(self, w):
        return w

    def proposal(self, z, x):
        h = slerp(z, x, self.lam)
        for layer in self.layers:
            h = layer(h)
        return h

    def step(self, z, x, eta):
        zt = self.proposal(z, x)
        resid = (1 - (z * zt).sum(-1)).amax(-1)      # chordal residual, max over tokens
        return slerp(z, zt, eta), resid, zt


class EnergySFPRM(nn.Module):
    """Energy-based SFPRM.

    flat   (hidden=False): state z (T tokens), E = kappa0 Phi_x(z) + E_A(z) + E_M(z)
    hidden (hidden=True) : state w = (z, u) (2T tokens): hidden tokens u query the visible tokens z,
                           E = kappa0 Phi_x(z) + E_A(u, z) - kappa_C sum_t <u_t, C z_t> + E_M(u)
    In both cases F_E(w) = Exp_w(-grad E) and w <- Exp_w(-eta grad E).
    """

    def __init__(self, n_sym, T, d, H, n_mem, dh=None, injection='geodesic',
                 kappa0=0.5, kappaA=0.5, kappaM=0.1, betaM=None, signed_heads=True, betaA=None,
                 hidden=False, kappaC=0.5, structured=None, w_sigma=None,
                 tied_readout=False, tie_embed=False, kappaAns=0.0, betaAns=None,
                 anneal_ratio=1.0, anneal_iters=0):
        super().__init__()
        self.embed_mod = Embedding(n_sym, T, d, structured)
        self.tied_readout, self.tie_embed = tied_readout, tie_embed
        if not tied_readout:
            self.readout_head = Readout(d, n_sym)
        # answer codebook: one unit-norm prototype per symbol (fixed norm, learned direction)
        Q, _ = torch.linalg.qr(torch.randn(d, n_sym))
        self.Xi_ans_ = nn.Parameter(Q.T.contiguous())
        self.s_read = nn.Parameter(torch.tensor(float(math.sqrt(d))))
        self.kappaAns_ = nn.Parameter(torch.tensor(math.log(math.expm1(kappaAns)) if kappaAns > 0 else -30.0))
        self.betaAns_ = nn.Parameter(torch.tensor(math.log(math.expm1(betaAns if betaAns else math.sqrt(d)))))
        self.anneal_ratio, self.anneal_iters = anneal_ratio, anneal_iters
        self.beta_mult = 1.0
        self.n_sym = n_sym
        self.d, self.H = d, H
        self.dh = dh or d // H
        self.injection = injection
        self.hidden = hidden
        self.T, self.n_state = T, (2 * T if hidden else T)
        if hidden:
            self.C = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            self.kappaC_ = nn.Parameter(torch.tensor(math.log(math.expm1(kappaC))))
        # small W (unit-scale messages) and a large logit scale beta (sharp attention): the energy
        # separates attention sharpness from message size, as QK-normalization does in nGPT
        sig = (1.0 / math.sqrt(d)) if w_sigma is None else w_sigma
        self.Wq = nn.Parameter(torch.randn(H, self.dh, d) * sig)
        self.Wk = nn.Parameter(torch.randn(H, self.dh, d) * sig)
        if betaA is None:
            betaA = d / math.sqrt(self.dh)          # unit-variance logits at initialization
        if betaM is None:
            betaM = math.sqrt(d)
        self.Xi = nn.Parameter(torch.randn(n_mem, d) / math.sqrt(d))
        inv = lambda v: math.log(math.expm1(v))  # softplus inverse
        self.kappa0_ = nn.Parameter(torch.tensor(inv(kappa0)))
        self.signed_heads = signed_heads
        if signed_heads:   # attractive (+) and repulsive (-) heads; sigma_k is a free real parameter
            self.kappaA_ = nn.Parameter(torch.tensor([kappaA * (1 if h % 2 == 0 else -1) for h in range(H)]))
        else:
            self.kappaA_ = nn.Parameter(torch.full((H,), inv(kappaA)))
        self.kappaM_ = nn.Parameter(torch.tensor(inv(kappaM)))
        self.betaA_ = nn.Parameter(torch.full((H,), inv(betaA)))
        self.betaM_ = nn.Parameter(torch.tensor(inv(betaM)))
        self.is_energy = True

    # positive parameters
    @property
    def kappa0(self): return F.softplus(self.kappa0_)
    @property
    def kappaA(self): return self.kappaA_ if self.signed_heads else F.softplus(self.kappaA_)
    @property
    def kappaM(self): return F.softplus(self.kappaM_)
    @property
    def betaA(self): return F.softplus(self.betaA_)
    @property
    def betaM(self): return F.softplus(self.betaM_)
    @property
    def kappaC(self): return F.softplus(self.kappaC_)
    @property
    def kappaAns(self): return F.softplus(self.kappaAns_)
    @property
    def betaAns(self): return F.softplus(self.betaAns_) * self.beta_mult
    @property
    def betaM_eff(self): return self.betaM * self.beta_mult
    @property
    def Xi_ans(self): return normalize(self.Xi_ans_)

    def set_iter(self, i):
        """beta schedule over iterations: beta_i = beta * ratio^{-(1 - min(i / K, 1))}."""
        if self.anneal_iters > 0 and self.anneal_ratio != 1.0:
            self.beta_mult = float(self.anneal_ratio ** (-(1.0 - min(i / self.anneal_iters, 1.0))))
        else:
            self.beta_mult = 1.0

    def embed(self, clues):
        return self.embed_mod(clues, proto=self.Xi_ans if self.tie_embed else None)

    def readout(self, z):
        if self.tied_readout:
            return self.s_read * torch.einsum('kd,btd->btk', self.Xi_ans, z)
        return self.readout_head(z)

    def commitment(self, z):
        """max_mu softmax(beta Xi_ans z_t)_mu per token (B, T)."""
        return (torch.einsum('kd,btd->btk', self.Xi_ans, z) * self.betaAns).softmax(-1).amax(-1)

    def visible(self, w):
        return w[:, :self.T]

    def split(self, w):
        return (w[:, :self.T], w[:, self.T:]) if self.hidden else (w, w)

    def _qk(self, zq, zk):
        q = torch.einsum('hed,btd->bhte', self.Wq, zq)
        k = torch.einsum('hed,btd->bhte', self.Wk, zk)
        return q, k

    def energy_terms(self, w, x):
        """Returns (Phi, E_A, E_M, E_C) per batch element (unweighted Phi; weighted E_A, E_M, E_C)."""
        z, u = self.split(w)
        B, T, d = z.shape
        c = (z * x).sum(-1)
        if self.injection == 'geodesic':
            _, th = angle_factor(c.unsqueeze(-1))
            phi = 0.5 * (th.squeeze(-1) ** 2).sum(-1)
        else:                                           # chordal: 1 - <z,x>
            phi = (1 - c).sum(-1)
        q, k = self._qk(u, z)                           # queries from u (= z in the flat model), keys from z
        logits = torch.einsum('bhte,bhse->bhts', q, k) * self.betaA.view(1, -1, 1, 1)
        eye = torch.eye(T, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, float('-inf'))
        lse = torch.logsumexp(logits, -1)                                   # (B,H,T)
        EA = -(self.kappaA.view(1, -1, 1) / self.betaA.view(1, -1, 1) * lse).sum((1, 2))
        m = torch.einsum('kd,btd->btk', self.Xi, u) * self.betaM_eff
        EM = -(self.kappaM / self.betaM_eff) * torch.logsumexp(m, -1).sum(-1)
        EC = -(self.kappaC * (u * torch.einsum('de,bte->btd', self.C, z)).sum((-1, -2))) if self.hidden \
            else torch.zeros_like(phi)
        ma = torch.einsum('kd,btd->btk', self.Xi_ans, z) * self.betaAns
        EAns = -(self.kappaAns / self.betaAns) * torch.logsumexp(ma, -1).sum(-1)
        return phi, EA, EM, EC, EAns

    def energy(self, w, x):
        phi, EA, EM, EC, EAns = self.energy_terms(w, x)
        return self.kappa0 * phi + EA + EM + EC + EAns

    def drive(self, w, x):
        """v = -grad E(w;x), token-wise tangent vector, in closed form."""
        z, u = self.split(w)
        B, T, d = z.shape
        c = (z * x).sum(-1, keepdim=True)
        if self.injection == 'geodesic':
            f, _ = angle_factor(c)
            inj = self.kappa0 * f * proj(z, x)                  # kappa0 Log_z x
        else:
            inj = self.kappa0 * proj(z, x)
        q, k = self._qk(u, z)
        logits = torch.einsum('bhte,bhse->bhts', q, k) * self.betaA.view(1, -1, 1, 1)
        eye = torch.eye(T, dtype=torch.bool, device=z.device)
        a = logits.masked_fill(eye, float('-inf')).softmax(-1)             # rows sum to 1
        ak = torch.einsum('bhts,bhse->bhte', a, k)                          # sum_s a_ts k_s   (query path)
        aq = torch.einsum('bhst,bhse->bhte', a, q)                          # sum_s a_st q_s   (key path)
        term_q = (torch.einsum('hed,bhte->bhtd', self.Wq, ak) * self.kappaA.view(1, -1, 1, 1)).sum(1)
        term_k = (torch.einsum('hed,bhte->bhtd', self.Wk, aq) * self.kappaA.view(1, -1, 1, 1)).sum(1)
        m = torch.einsum('kd,btd->btk', self.Xi, u) * self.betaM_eff
        mem = self.kappaM * torch.einsum('btk,kd->btd', m.softmax(-1), self.Xi)
        ma = torch.einsum('kd,btd->btk', self.Xi_ans, z) * self.betaAns
        mem_ans = self.kappaAns * torch.einsum('btk,kd->btd', ma.softmax(-1), self.Xi_ans)
        if not self.hidden:
            return inj + proj(z, term_q + term_k + mem + mem_ans)
        Cz = torch.einsum('de,bte->btd', self.C, z)
        Ctu = torch.einsum('de,btd->bte', self.C, u)
        vz = inj + proj(z, term_k + self.kappaC * Ctu + mem_ans)
        vu = proj(u, term_q + mem + self.kappaC * Cz)
        return torch.cat([vz, vu], 1)

    def proposal(self, w, x):
        return exp_map(w, self.drive(w, x))

    def step(self, w, x, eta):
        v = self.drive(w, x)
        nv = v.norm(dim=-1)
        resid = (1 - torch.cos(nv)).amax(-1)                # same chordal test as SFPRM
        return exp_map(w, eta * v), resid, nv

    def step_armijo(self, w, x, eta0=1.0, c=1e-4, max_halvings=4):
        """Gradient step with per-sample Armijo backtracking on E (eta treated as a constant)."""
        v = self.drive(w, x)
        g2 = (v * v).sum((-1, -2))
        eta = torch.full((w.shape[0], 1, 1), float(eta0), device=w.device, dtype=w.dtype)
        with torch.no_grad():
            e0 = self.energy(w, x)
            for _ in range(max_halvings):
                e1 = self.energy(exp_map(w, eta * v), x)
                bad = e1 > e0 - c * eta.view(-1) * g2
                if not bad.any():
                    break
                eta = torch.where(bad.view(-1, 1, 1), eta * 0.5, eta)
        nv = v.norm(dim=-1)
        resid = (1 - torch.cos(nv)).amax(-1)
        return exp_map(w, eta * v), resid, nv, eta


def build_model(cfg, T, n_sym):
    structured = (cfg['n'], cfg['br'], cfg['bc']) if cfg.get('structured') else None
    if cfg['model'] == 'sfprm':
        return SFPRM(n_sym, T, cfg['d'], cfg['H'], cfg['depth'], lam=cfg['lam'], structured=structured,
                     mlp=cfg.get('mlp', 'gelu'), n_mem=cfg.get('n_mem'))
    return EnergySFPRM(n_sym, T, cfg['d'], cfg['H'], cfg['n_mem'], dh=cfg.get('dh'),
                       injection=cfg.get('injection', 'geodesic'),
                       kappa0=cfg.get('kappa0', 0.5), kappaA=cfg.get('kappaA', 0.5),
                       kappaM=cfg.get('kappaM', 0.1), betaM=cfg.get('betaM'),
                       signed_heads=cfg.get('signed_heads', True), betaA=cfg.get('betaA'),
                       hidden=cfg.get('hidden', False), kappaC=cfg.get('kappaC', 0.5),
                       structured=structured, tied_readout=cfg.get('tied_readout', False),
                       tie_embed=cfg.get('tie_embed', False), kappaAns=cfg.get('kappaAns', 0.0),
                       betaAns=cfg.get('betaAns'), anneal_ratio=cfg.get('anneal_ratio', 1.0),
                       anneal_iters=cfg.get('anneal_iters', 0))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def load_split(path):
    z = np.load(path)
    return torch.from_numpy(z['puzzles']), torch.from_numpy(z['solutions']), torch.from_numpy(z['clues'])


def get_device(name):
    if name == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def run_iterations(model, z, x, n_iter, eta=1.0):
    for _ in range(n_iter):
        z, _, _ = model.step(z, x, eta)
    return z


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = get_device(a.device)
    puz, sol, _ = load_split(os.path.join(a.data, 'train.npz'))
    n = puz.shape[1]; T = n * n; n_sym = n
    puz, sol = puz.view(-1, T), sol.view(-1, T) - 1
    cfg = dict(model=a.model, d=a.d, H=a.H, depth=a.depth, n_mem=a.n_mem, lam=a.lam, dh=a.dh,
               injection=a.injection, n=n, br=a.br, bc=a.bc, kappa0=a.kappa0, kappaA=a.kappaA, kappaM=a.kappaM,
               betaM=a.betaM, signed_heads=not a.unsigned_heads, betaA=a.betaA, hidden=a.hidden,
               kappaC=a.kappaC, structured=a.structured, mlp=a.mlp, tied_readout=a.tied_readout,
               tie_embed=a.tie_embed, kappaAns=a.kappaAns, betaAns=a.betaAns,
               anneal_ratio=a.anneal_ratio, anneal_iters=a.anneal_iters)
    model = build_model(cfg, T, n_sym).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{a.model}: {n_params} parameters, device {dev}')
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or name.endswith('_')) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    N = puz.shape[0]
    log = []
    t0 = time.time()
    n_inc = n_tot = 0
    R = a.rank_restarts if (model.is_energy and a.rank_loss > 0) else 1
    rank_stat = []
    for step in range(a.steps):
        idx = torch.randint(0, N, (a.batch // R,)).repeat_interleave(R)
        clues = puz[idx].to(dev); target = sol[idx].to(dev)
        x = model.embed(clues)
        z = random_sphere((idx.shape[0], model.n_state, a.d), dev)
        loss_total = 0.0
        it = 0
        for w in range(a.windows):
            if model.is_energy and a.monitor_energy:
                with torch.no_grad():
                    e_prev = model.energy(z, x)
            for _ in range(a.K):
                if model.is_energy:
                    model.set_iter(it)
                it += 1
                if model.is_energy and a.armijo_train:
                    z, _, _, _ = model.step_armijo(z, x)
                else:
                    z, _, _ = model.step(z, x, 1.0)
                if model.is_energy and a.monitor_energy:
                    with torch.no_grad():
                        e_now = model.energy(z, x)
                        n_inc += (e_now > e_prev + 1e-6).sum().item(); n_tot += e_now.numel()
                        e_prev = e_now
            logits = model.readout(model.visible(z))
            loss = F.cross_entropy(logits.reshape(-1, n_sym), target.reshape(-1))
            loss_total = loss_total + loss / a.windows
            z = z.detach()
        if R > 1:
            # energy ranking over restarts: E(correct) + m <= E(wrong), energies at the final beta
            model.set_iter(10 ** 9)
            E_fin = model.energy(z, x).view(-1, R)
            with torch.no_grad():
                corr = (logits.argmax(-1) == target).all(-1).view(-1, R)
            pair = corr.unsqueeze(2) & ~corr.unsqueeze(1)                       # (b, R, R): i correct, j wrong
            if pair.any():
                diff = E_fin.unsqueeze(2) - E_fin.unsqueeze(1) + a.rank_margin
                loss_rank = (F.relu(diff) * pair).sum() / pair.sum()
                loss_total = loss_total + a.rank_loss * loss_rank
                rank_stat.append(((E_fin.unsqueeze(2) < E_fin.unsqueeze(1)) & pair).sum().item() / pair.sum().item())
        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            with torch.no_grad():
                acc_cell = (logits.argmax(-1) == target).float().mean().item()
                acc_board = (logits.argmax(-1) == target).all(-1).float().mean().item()
            rec = dict(step=step, loss=loss_total.item(), acc_cell=acc_cell, acc_board=acc_board,
                       time=time.time() - t0)
            if model.is_energy and a.monitor_energy and n_tot:
                rec['frac_energy_increase'] = n_inc / n_tot
                n_inc = n_tot = 0
            if rank_stat:
                rec['rank_correct_lower'] = float(np.mean(rank_stat)); rank_stat = []
            if model.is_energy and model.tied_readout:
                with torch.no_grad():
                    rec['commit_frac'] = (model.commitment(model.visible(z)) > 1 - a.commit_eps).float().mean().item()
            log.append(rec)
            print(json.dumps(rec), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save({'cfg': cfg, 'state': model.state_dict(), 'log': log, 'args': {k: v for k, v in vars(a).items() if k != 'fn'},
                'n_params': n_params}, a.out)
    print('saved', a.out)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def solve_batch(model, x, z0, max_iter, tau, patience, eta_min, mode='patience', record_energy=False,
                require_commit=False, commit_eps=0.1):
    """Run the damped iteration with halting; returns final state, iterations to halt,
    halted mask, per-step energies (optional) and the final damping."""
    B = x.shape[0]
    z = z0.clone()
    eta = torch.ones(B, 1, 1, device=x.device)
    best = torch.full((B,), float('inf'), device=x.device)
    since = torch.zeros(B, dtype=torch.long, device=x.device)
    halted = torch.zeros(B, dtype=torch.bool, device=x.device)
    n_halt = torch.full((B,), max_iter, dtype=torch.long, device=x.device)
    energies = []
    e_inc = torch.zeros(B, dtype=torch.long, device=x.device)
    e_prev = model.energy(z, x) if (model.is_energy and record_energy) else None
    for i in range(max_iter):
        if model.is_energy:
            model.set_iter(i)
        if mode == 'armijo' and model.is_energy:
            # energy backtracking: accept the step if E decreases, else halve eta (per sample)
            v = model.drive(z, x)
            e0 = model.energy(z, x)
            g2 = (v * v).sum((-1, -2))
            for _ in range(6):
                z_new = exp_map(z, eta * v)
                e1 = model.energy(z_new, x)
                bad = (e1 > e0 - 1e-4 * eta.view(-1) * g2) & ~halted
                if not bad.any():
                    break
                eta = torch.where(bad.view(-1, 1, 1), eta * 0.5, eta)
            resid = (1 - torch.cos(v.norm(dim=-1))).amax(-1)
        else:
            z_new, resid, _ = model.step(z, x, eta)
        newly = (resid < tau) & ~halted
        if require_commit:
            newly &= (model.commitment(model.visible(z)) > 1 - commit_eps).all(-1)
        n_halt = torch.where(newly, torch.full_like(n_halt, i), n_halt)
        halted |= newly
        # patience rule on the residual
        if mode == 'patience':
            improved = resid < best
            best = torch.minimum(best, resid)
            since = torch.where(improved, torch.zeros_like(since), since + 1)
            decay = (since >= patience) & ~halted
            eta = torch.where(decay.view(-1, 1, 1), (eta * 0.5).clamp_min(eta_min), eta)
            since = torch.where(decay, torch.zeros_like(since), since)
        z = torch.where(halted.view(-1, 1, 1), z, z_new)
        if model.is_energy and record_energy:
            e_now = model.energy(z, x)
            e_inc += ((e_now > e_prev + 1e-6) & ~halted).long()
            energies.append(e_now)
            e_prev = e_now
        if halted.all():
            break
    if model.is_energy:
        model.set_iter(10 ** 9)
    return z, n_halt, halted, (torch.stack(energies, 1) if energies else None), eta.view(-1), e_inc


def tangent_basis(z):
    """Orthonormal basis of T_z S^{d-1} for each token: (T, d, d-1)."""
    T, d = z.shape
    Qs = []
    for t in range(T):
        M = torch.cat([z[t].view(-1, 1), torch.randn(d, d - 1, dtype=z.dtype)], 1)
        Q, _ = torch.linalg.qr(M)
        Qs.append(Q[:, 1:])
    return torch.stack(Qs)


def jacobian_spectrum(model, x, z, device):
    """Eigenvalues of the tangent Jacobian of the proposal map F at z (double precision)."""
    import copy
    m = copy.deepcopy(model).to('cpu').double()
    if m.is_energy:
        m.set_iter(10 ** 9)
    x = x.to('cpu').double(); z = z.to('cpu').double()
    T, d = z.shape
    Q = tangent_basis(z)                                   # (T, d, d-1)

    Tx = x.shape[0]

    def fmap(zflat):
        return m.proposal(zflat.view(1, T, d), x.view(1, Tx, d)).reshape(-1)

    J = torch.func.jacrev(fmap)(z.reshape(-1))             # (Td, Td)
    J = J.view(T, d, T, d)
    Jt = torch.einsum('tda,tdse,seb->tasb', Q, J, Q).reshape(T * (d - 1), T * (d - 1))
    ev = np.linalg.eigvals(Jt.numpy())
    asym = (Jt - Jt.T).abs().max().item() / Jt.abs().max().item()
    return ev, asym


def polish(model, x, z, eta, n_steps, tol):
    """Continue the damped iteration (double precision, cpu) to reduce the residual below tol."""
    import copy
    m = copy.deepcopy(model).to('cpu').double()
    if m.is_energy:
        m.set_iter(10 ** 9)
    x = x.to('cpu').double(); z = z.to('cpu').double()
    eta_t = torch.full((z.shape[0], 1, 1), float(eta), dtype=torch.float64)
    with torch.no_grad():
        r = None
        for _ in range(n_steps):
            z_new, r, _ = m.step(z, x, eta_t)
            if r.max().item() < tol:
                z = z_new
                break
            z = z_new
    return z, (r.max().item() if r is not None else float('nan'))


def board_key(pred):
    return tuple(pred.tolist())


@torch.no_grad()
def cmd_eval(a):
    torch.manual_seed(a.seed)
    dev = get_device(a.device)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    n = cfg['n']; T = n * n; n_sym = n
    model = build_model(cfg, T, n_sym)
    state = ck['state']
    if cfg['model'] == 'esfprm':      # checkpoints from the first version of the code
        state = {k.replace('embed.', 'embed_mod.', 1) if k.startswith('embed.') else
                 (k.replace('readout.', 'readout_head.', 1) if k.startswith('readout.') else k): v
                 for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith(('Xi_ans_', 's_read', 'kappaAns_', 'betaAns_', 'embed_mod.blank')) for m in missing), missing
    model.to(dev).eval()
    results = dict(model=cfg['model'], n_params=ck['n_params'], ckpt=a.ckpt)
    for split in ['test_id', 'test_ood']:
        puz, sol, clues = load_split(os.path.join(a.data, f'{split}.npz'))
        puz, sol = puz.view(-1, T)[:a.n_eval], sol.view(-1, T)[:a.n_eval] - 1
        N = puz.shape[0]
        res = {}
        # ---------------- single restart: accuracy and iterations to halt ----------------
        modes = ['patience', 'armijo'] if model.is_energy else ['patience']
        if model.is_energy and model.tied_readout:
            modes.append('armijo+commit')
        for mode in modes:
            accs, boards, halts, halted_all, e_incs, commits = [], [], [], [], [], []
            for i0 in range(0, N, a.chunk):
                clues_b = puz[i0:i0 + a.chunk].to(dev); target = sol[i0:i0 + a.chunk].to(dev)
                x = model.embed(clues_b)
                z0 = random_sphere((clues_b.shape[0], model.n_state, cfg['d']), dev)
                z, n_halt, halted, _, _, e_inc = solve_batch(model, x, z0, a.max_iter, a.tau, a.patience,
                                                             a.eta_min, mode=mode.replace('+commit', ''),
                                                             record_energy=model.is_energy,
                                                             require_commit=mode.endswith('+commit'),
                                                             commit_eps=a.commit_eps)
                pred = model.readout(model.visible(z)).argmax(-1)
                if model.is_energy and model.tied_readout:
                    commits.append((model.commitment(model.visible(z)) > 1 - a.commit_eps).all(-1).cpu())
                accs.append((pred == target).float().mean(-1).cpu())
                boards.append((pred == target).all(-1).cpu())
                halts.append(n_halt.cpu()); halted_all.append(halted.cpu()); e_incs.append(e_inc.cpu())
            accs = torch.cat(accs); boards = torch.cat(boards); halts = torch.cat(halts).float()
            halted_all = torch.cat(halted_all); e_incs = torch.cat(e_incs).float()
            hb = halts[halted_all]
            res[f'{mode}/cell_acc'] = accs.mean().item()
            res[f'{mode}/board_acc'] = boards.float().mean().item()
            res[f'{mode}/halt_frac'] = halted_all.float().mean().item()
            res[f'{mode}/iters_mean'] = hb.mean().item() if len(hb) else float('nan')
            res[f'{mode}/iters_median'] = hb.median().item() if len(hb) else float('nan')
            res[f'{mode}/iters_p90'] = hb.quantile(0.9).item() if len(hb) else float('nan')
            res[f'{mode}/iters_list'] = [int(v) for v in halts.tolist()]
            res[f'{mode}/halted_list'] = [bool(v) for v in halted_all.tolist()]
            if commits:
                cm = torch.cat(commits)
                res[f'{mode}/committed_frac'] = cm.float().mean().item()
                res[f'{mode}/acc_committed'] = boards[cm].float().mean().item() if cm.any() else float('nan')
                res[f'{mode}/acc_uncommitted'] = boards[~cm].float().mean().item() if (~cm).any() else float('nan')
            if model.is_energy:
                res[f'{mode}/energy_increase_steps_mean'] = e_incs.mean().item()
        # ---------------- Jacobian spectrum at the halted state ----------------
        if a.n_jac > 0:
            spec = dict(n=0, complex_frac=[], max_imag=[], rho=[], asym=[], window=[], resid=[],
                        n_unstable=[], min_real_gap=[])
            raw_eigs = []
            clues_b = puz[:a.n_jac].to(dev); target = sol[:a.n_jac].to(dev)
            x = model.embed(clues_b)
            z0 = random_sphere((clues_b.shape[0], model.n_state, cfg['d']), dev)
            emode = 'armijo' if model.is_energy else 'patience'
            z, n_halt, halted, _, eta_f, _ = solve_batch(model, x, z0, a.max_iter, a.tau, a.patience, a.eta_min, mode=emode)
            for i in range(clues_b.shape[0]):
                if not halted[i]:
                    continue
                zi, r = polish(model, x[i:i + 1], z[i:i + 1], eta_f[i].item(), a.polish_steps, a.polish_tol)
                ev, asym = jacobian_spectrum(model, x[i], zi[0], dev)
                if len(raw_eigs) < 3:
                    raw_eigs.append([[float(e.real), float(e.imag)] for e in ev])
                im = np.abs(ev.imag); re = ev.real
                spec['n'] += 1
                spec['complex_frac'].append(float((im > a.imag_tol).mean()))
                spec['max_imag'].append(float(im.max()))
                spec['rho'].append(float(np.abs(ev).max()))
                spec['asym'].append(float(asym))
                spec['resid'].append(float(r))
                spec['n_unstable'].append(int((np.abs(ev) >= 1).sum()))
                # damping window of Cor. (window): min over modes of 2(1-Re)/|zeta-1|^2 (modes with Re<1)
                dz = ev - 1
                ok = re < 1 - 1e-12
                w = 2 * (1 - re[ok]) / (np.abs(dz[ok]) ** 2) if ok.any() else np.array([np.inf])
                spec['window'].append(float(w.min()))
                spec['min_real_gap'].append(float((1 - re).min()))
            res['jac'] = {k: (v if k == 'n' else dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                                      max=float(np.max(v)), min=float(np.min(v))))
                          for k, v in spec.items() if (k == 'n' or len(v))}
            res['jac_raw'] = {k: v for k, v in spec.items() if k != 'n'}
            res['jac_eigs'] = raw_eigs
        # ---------------- restarts and selection rules ----------------
        if a.B_max > 1:
            Bs = [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256] if b <= a.B_max]
            sel = {k: {b: 0 for b in Bs} for k in ['energy', 'energy_sum', 'anchor', 'majority', 'oracle', 'first']}
            n_done = 0; halted_frac_sum = 0.0
            per = max(1, a.chunk // a.B_max)
            for i0 in range(0, N, per):
                clues_b = puz[i0:i0 + per]; target = sol[i0:i0 + per]
                nb = clues_b.shape[0]
                clues_r = clues_b.repeat_interleave(a.B_max, 0).to(dev)
                target_r = target.repeat_interleave(a.B_max, 0).to(dev)
                x = model.embed(clues_r)
                z0 = random_sphere((nb * a.B_max, model.n_state, cfg['d']), dev)
                z, n_halt, halted, energies, _, _ = solve_batch(model, x, z0, a.max_iter, a.tau, a.patience,
                                                                a.eta_min, mode=('armijo' if model.is_energy else 'patience'),
                                                                record_energy=model.is_energy)
                pred = model.readout(model.visible(z)).argmax(-1)
                correct = (pred == target_r).all(-1).view(nb, a.B_max).cpu()
                halted_r = halted.view(nb, a.B_max).cpu()
                # anchor angle potential Phi_x at the halted state
                c = (model.visible(z) * x).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
                anchor = (0.5 * torch.acos(c) ** 2).sum(-1).view(nb, a.B_max).cpu()
                if model.is_energy:
                    efin = model.energy(z, x).view(nb, a.B_max).cpu()
                    esum = energies[:, :a.sum_horizon].sum(1).view(nb, a.B_max).cpu() if energies is not None else efin
                pred_c = pred.view(nb, a.B_max, T).cpu()
                for b in Bs:
                    cb = correct[:, :b]
                    sel['oracle'][b] += cb.any(1).sum().item()
                    sel['first'][b] += cb[:, 0].sum().item()
                    sel['anchor'][b] += cb[torch.arange(nb), anchor[:, :b].argmin(1)].sum().item()
                    if model.is_energy:
                        sel['energy'][b] += cb[torch.arange(nb), efin[:, :b].argmin(1)].sum().item()
                        sel['energy_sum'][b] += cb[torch.arange(nb), esum[:, :b].argmin(1)].sum().item()
                    for j in range(nb):
                        cnt = Counter(board_key(pred_c[j, r]) for r in range(b))
                        best_key = cnt.most_common(1)[0][0]
                        r_best = next(r for r in range(b) if board_key(pred_c[j, r]) == best_key)
                        sel['majority'][b] += int(cb[j, r_best].item())
                n_done += nb; halted_frac_sum += halted_r.float().mean(1).sum().item()
            res['restarts'] = {k: {str(b): v[b] / n_done for b in Bs} for k, v in sel.items()
                               if (model.is_energy or not k.startswith('energy'))}
            res['restarts_n'] = n_done
            res['restarts_halted_frac'] = halted_frac_sum / n_done
        results[split] = res
        print(split, json.dumps({k: v for k, v in res.items() if k not in ('jac_raw', 'jac_eigs') and not k.endswith('_list')}, indent=1), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', a.out)


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--n', type=int, default=6); g.add_argument('--br', type=int, default=2); g.add_argument('--bc', type=int, default=3)
    g.add_argument('--n_train', type=int, default=12000); g.add_argument('--n_test', type=int, default=1000)
    g.add_argument('--id_clues', type=int, nargs=2, default=[14, 22])
    g.add_argument('--ood_clues', type=int, nargs=2, default=[8, 13])
    g.add_argument('--workers', type=int, default=10); g.add_argument('--out', required=True)
    g.set_defaults(fn=cmd_gen)

    t = sub.add_parser('train')
    t.add_argument('--data', required=True); t.add_argument('--out', required=True)
    t.add_argument('--model', choices=['sfprm', 'esfprm'], required=True)
    t.add_argument('--d', type=int, default=64); t.add_argument('--H', type=int, default=4)
    t.add_argument('--dh', type=int, default=None)
    t.add_argument('--depth', type=int, default=2); t.add_argument('--n_mem', type=int, default=1024)
    t.add_argument('--lam', type=float, default=0.5); t.add_argument('--injection', default='geodesic')
    t.add_argument('--steps', type=int, default=4000); t.add_argument('--batch', type=int, default=128)
    t.add_argument('--windows', type=int, default=2); t.add_argument('--K', type=int, default=8)
    t.add_argument('--lr', type=float, default=2e-3); t.add_argument('--wd', type=float, default=0.01)
    t.add_argument('--seed', type=int, default=0); t.add_argument('--device', default='auto')
    t.add_argument('--monitor_energy', action='store_true'); t.add_argument('--armijo_train', action='store_true')
    t.add_argument('--unsigned_heads', action='store_true'); t.add_argument('--betaA', type=float, default=None)
    t.add_argument('--hidden', action='store_true'); t.add_argument('--kappaC', type=float, default=0.5)
    t.add_argument('--mlp', default='gelu', choices=['gelu', 'memory'])
    t.add_argument('--tied_readout', action='store_true'); t.add_argument('--tie_embed', action='store_true')
    t.add_argument('--kappaAns', type=float, default=0.0); t.add_argument('--betaAns', type=float, default=None)
    t.add_argument('--anneal_ratio', type=float, default=1.0); t.add_argument('--anneal_iters', type=int, default=0)
    t.add_argument('--rank_loss', type=float, default=0.0); t.add_argument('--rank_margin', type=float, default=1.0)
    t.add_argument('--rank_restarts', type=int, default=2); t.add_argument('--commit_eps', type=float, default=0.1)
    t.add_argument('--structured', action='store_true')
    t.add_argument('--br', type=int, default=2); t.add_argument('--bc', type=int, default=3)
    t.add_argument('--kappa0', type=float, default=0.5); t.add_argument('--kappaA', type=float, default=0.5)
    t.add_argument('--kappaM', type=float, default=0.1); t.add_argument('--betaM', type=float, default=None)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser('eval')
    e.add_argument('--data', required=True); e.add_argument('--ckpt', required=True); e.add_argument('--out', required=True)
    e.add_argument('--n_eval', type=int, default=500); e.add_argument('--chunk', type=int, default=4096)
    e.add_argument('--max_iter', type=int, default=128); e.add_argument('--tau', type=float, default=1e-4)
    e.add_argument('--patience', type=int, default=3); e.add_argument('--eta_min', type=float, default=1 / 64)
    e.add_argument('--n_jac', type=int, default=20); e.add_argument('--polish_steps', type=int, default=2000)
    e.add_argument('--polish_tol', type=float, default=1e-12); e.add_argument('--imag_tol', type=float, default=1e-8)
    e.add_argument('--B_max', type=int, default=64); e.add_argument('--sum_horizon', type=int, default=32)
    e.add_argument('--seed', type=int, default=0); e.add_argument('--device', default='auto')
    e.add_argument('--commit_eps', type=float, default=0.1)
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
