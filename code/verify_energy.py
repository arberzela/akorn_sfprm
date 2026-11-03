"""Numerical verification of the energy-based SFPRM statements (double precision, random instances).

Checks
  1. -grad E = injection + two-sided attention coupling + Hopfield self-drive (closed form vs autograd)
  2. the energy-attention coupling satisfies J_ts = J_st^T for every state
  3. at a stationary point, dF_E = I - Hess E on the tangent space: symmetric, real spectrum,
     matches central finite differences; a plain SFPRM map at a fixed point has complex pairs
  4. descent lemma: E(Exp_z(-eta grad E)) <= E(z) - eta(1 - eta L/2)|grad E|^2 for eta < 2/L
  5. implicit gradient: conjugate gradient on Hess E versus the truncated Neumann series
     (truncated BPTT) and versus finite differences of the loss at the fixed point
  6. equilibrium propagation: (1/beta)[d_theta E(z*_beta) - d_theta E(z*_0)] -> implicit gradient
"""
import math, sys, json
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from esfprm_pilot import EnergySFPRM, SFPRM, proj, exp_map, random_sphere, angle_factor, tangent_basis  # noqa

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
out = {}

T, d, H, K, n_sym = 6, 16, 2, 24, 4


def make_energy_model(seed=0, kappaA=0.3, kappaM=0.5, betaM=3.0, hidden=False):
    torch.manual_seed(seed)
    m = EnergySFPRM(n_sym, T, d, H, K, kappa0=1.0, kappaA=kappaA, kappaM=kappaM, betaM=betaM,
                    betaA=2.0, w_sigma=1.0 / math.sqrt(d), hidden=hidden)
    return m


# ---------------------------------------------------------------- 1. drive identity
m = make_energy_model()
clues = torch.randint(0, n_sym + 1, (1, T))
x = m.embed(clues).detach()
errs = []
for _ in range(200):
    z = random_sphere((1, T, d), 'cpu').requires_grad_(True)
    E = m.energy(z, x).sum()
    g, = torch.autograd.grad(E, z)
    errs.append((-proj(z, g) - m.drive(z, x)).abs().max().item())
out['drive_identity'] = max(errs)
mh = make_energy_model(hidden=True)
xh = mh.embed(clues).detach()
errs = []
for _ in range(100):
    w = random_sphere((1, 2 * T, d), 'cpu').requires_grad_(True)
    g, = torch.autograd.grad(mh.energy(w, xh).sum(), w)
    errs.append((-proj(w, g) - mh.drive(w, xh)).abs().max().item())
out['drive_identity_hidden'] = max(errs)

# ---------------------------------------------------------------- 2. coupling symmetry J_ts = J_st^T
with torch.no_grad():
    z = random_sphere((1, T, d), 'cpu')
    q, k = m._qk(z, z)
    logits = torch.einsum('bhte,bhse->bhts', q, k) * m.betaA.view(1, -1, 1, 1)
    eye = torch.eye(T, dtype=torch.bool)
    a = logits.masked_fill(eye, float('-inf')).softmax(-1)[0]            # (H,T,T)
    J = torch.zeros(T, T, d, d)
    for h in range(H):
        M1 = m.kappaA[h] * m.Wq[h].T @ m.Wk[h]                            # W_Q^T W_K
        for t in range(T):
            for s in range(T):
                J[t, s] += a[h, t, s] * M1 + a[h, s, t] * M1.T
    sym_err = max((J[t, s] - J[s, t].T).abs().max().item() for t in range(T) for s in range(T))
    # the attention part of the drive equals P_z sum_s J_ts z_s
    att_drive = torch.einsum('tsde,se->td', J, z[0])
    # recompute the drive without injection/memory by zeroing them
    m2 = make_energy_model()
    m2.load_state_dict(m.state_dict())
    m2.kappa0_.data.fill_(-40.0); m2.kappaM_.data.fill_(-40.0)           # softplus -> ~0 (kappaA is raw/signed)
    err_att = (m2.drive(z, x)[0] - proj(z[0], att_drive)).abs().max().item()
out['coupling_symmetry'] = sym_err
out['coupling_drive'] = err_att


# ---------------------------------------------------------------- helpers
def riem_hessian(model, x, z):
    """Riemannian Hessian of E at z on the tangent space, in the basis Q: Q^T (P D2E P - <grad_E,z> P) Q."""
    Tn, dn = z.shape
    z = z.detach()
    with torch.enable_grad():
        zf = z.reshape(-1).clone().requires_grad_(True)

        def Ef(zf_):
            return model.energy(zf_.view(1, Tn, dn), x.view(1, Tn, dn)).sum()

        Hs = torch.autograd.functional.hessian(Ef, zf).view(Tn, dn, Tn, dn).detach()
        g = torch.autograd.grad(Ef(zf), zf)[0].view(Tn, dn).detach()
    mu = (g * z).sum(-1)                                                 # <grad E, z_t>
    Q = tangent_basis(z)
    Hq = torch.einsum("tda,tdse,seb->tasb", Q, Hs, Q).reshape(Tn * (dn - 1), Tn * (dn - 1))
    Hq = Hq - torch.block_diag(*[mu[t] * torch.eye(dn - 1) for t in range(Tn)])
    return Hq.detach(), Q


def find_min(model, x, z0, eta, n_iter=200000, tol=1e-14):
    z = z0.clone()
    with torch.no_grad():
        for i in range(n_iter):
            v = model.drive(z, x)
            if v.norm(dim=-1).max() < tol:
                break
            z = exp_map(z, eta * v)
    return z, v.norm(dim=-1).max().item(), i


def tangent_jacobian(fmap, z, Q):
    Tn, dn = z.shape
    J = torch.func.jacrev(fmap)(z.reshape(-1)).detach().view(Tn, dn, Tn, dn)
    return torch.einsum('tda,tdse,seb->tasb', Q, J, Q).reshape(Tn * (dn - 1), Tn * (dn - 1))


# ---------------------------------------------------------------- 3. Jacobian at a stationary point
m = make_energy_model(seed=1)
x = m.embed(torch.randint(0, n_sym + 1, (1, T))).detach()
z0 = random_sphere((1, T, d), 'cpu')
zs, gnorm, nit = find_min(m, x, z0, eta=0.2)
out['stationary_gradnorm'] = gnorm
out['stationary_iters'] = nit
Hq, Q = riem_hessian(m, x[0], zs[0])
Jq = tangent_jacobian(lambda zf: m.proposal(zf.view(1, T, d), x).reshape(-1), zs[0], Q)
I = torch.eye(Hq.shape[0])
out['jacobian_equals_I_minus_Hess'] = (Jq - (I - Hq)).abs().max().item()
out['jacobian_asymmetry'] = (Jq - Jq.T).abs().max().item()
ev = np.linalg.eigvals(Jq.detach().numpy())
out['jacobian_max_imag'] = float(np.abs(ev.imag).max())
out['hessian_min_eig'] = float(np.linalg.eigvalsh(Hq.numpy()).min())
out['hessian_max_eig'] = float(np.linalg.eigvalsh(Hq.numpy()).max())
# central finite differences on the tangent space
eps = 1e-6
Jq = Jq.detach(); fd = torch.zeros_like(Jq)
with torch.no_grad():
    for j in range(Jq.shape[1]):
        t, a_ = divmod(j, d - 1)
        dz = torch.zeros(T, d); dz[t] = Q[t][:, a_]
        zp = exp_map(zs[0], eps * dz).view(1, T, d); zm = exp_map(zs[0], -eps * dz).view(1, T, d)
        diff = (m.proposal(zp, x) - m.proposal(zm, x))[0] / (2 * eps)
        fd[:, j] = torch.einsum('tda,td->ta', Q, diff).reshape(-1)
out['jacobian_vs_finite_diff'] = (Jq - fd).abs().max().item()

# a plain SFPRM map at a fixed point (small eigen learning rates so that the iteration contracts)
torch.manual_seed(3)
sf = SFPRM(n_sym, T, d, H, 1, lam=0.5)
for layer in sf.layers:
    layer.alpha_A.data.fill_(0.3); layer.alpha_M.data.fill_(0.3)
xs = sf.embed(torch.randint(0, n_sym + 1, (1, T))).detach()
z = random_sphere((1, T, d), 'cpu')
with torch.no_grad():
    for i in range(20000):
        zt = sf.proposal(z, xs)
        r = (1 - (z * zt).sum(-1)).max().item()
        z = zt
        if r < 1e-30:
            break
out['sfprm_fixed_point_resid'] = r
Qs = tangent_basis(z[0])
Js = tangent_jacobian(lambda zf: sf.proposal(zf.view(1, T, d), xs).reshape(-1), z[0], Qs)
evs = np.linalg.eigvals(Js.detach().numpy())
out['sfprm_jacobian_asymmetry'] = (Js - Js.T).abs().max().item()
out['sfprm_complex_fraction'] = float((np.abs(evs.imag) > 1e-10).mean())
out['sfprm_max_imag'] = float(np.abs(evs.imag).max())

# ---------------------------------------------------------------- 4. descent lemma
m = make_energy_model(seed=5, kappaA=0.5, kappaM=1.0, betaM=4.0)
x = m.embed(torch.randint(0, n_sym + 1, (1, T))).detach()
viol = {}
with torch.no_grad():
    # L along a geodesic step: the second derivative of E(Exp_z(t v)) is bounded by the max Hessian
    # eigenvalue along the curve; estimate L as the max tangent-Hessian eigenvalue over random states
    Ls = []
    for _ in range(50):
        z = random_sphere((1, T, d), 'cpu')
        Hq, _ = riem_hessian(m, x[0], z[0])
        Ls.append(np.linalg.eigvalsh(Hq.numpy()).max())
    L0 = float(max(Ls))
    # refine L along descent trajectories (states visited at eta = 0.5/L0) and along the segments
    for _ in range(20):
        z = random_sphere((1, T, d), 'cpu')
        for i in range(20):
            v = m.drive(z, x)
            for frac in (0.0, 0.5, 1.0):
                Hq, _ = riem_hessian(m, x[0], exp_map(z, 0.5 / L0 * frac * v)[0])
                Ls.append(np.linalg.eigvalsh(Hq.numpy()).max())
            z = exp_map(z, 0.5 / L0 * v)
    L = float(max(Ls))
    out['L_estimate_random'] = L0
    out['L_estimate'] = L
    for frac in [0.5, 1.0, 1.9, 3.0]:
        eta = frac / L
        n_viol = n_tot = 0
        worst = -1e9
        for _ in range(200):
            z = random_sphere((1, T, d), 'cpu')
            for i in range(20):
                v = m.drive(z, x)
                e0 = m.energy(z, x).item()
                zn = exp_map(z, eta * v)
                e1 = m.energy(zn, x).item()
                bound = e0 - eta * (1 - eta * L / 2) * (v * v).sum().item()
                worst = max(worst, e1 - bound)
                n_viol += int(e1 > e0 + 1e-12); n_tot += 1
                z = zn
        viol[f'eta={frac}/L'] = dict(increase_frac=n_viol / n_tot, max_violation_of_bound=worst)
out['descent'] = viol

# ---------------------------------------------------------------- 5. implicit gradient: CG vs Neumann vs finite differences
m = make_energy_model(seed=7)
x = m.embed(torch.randint(0, n_sym + 1, (1, T))).detach()
zs, gnorm, _ = find_min(m, x, random_sphere((1, T, d), 'cpu'), eta=0.2)
Hq, Q = riem_hessian(m, x[0], zs[0])
w = torch.randn(T, d)                                                    # loss l(z) = <w, z>  (linear readout)
gl = torch.einsum('tda,td->ta', Q, w).reshape(-1)                        # tangent gradient of l in basis Q
# exact adjoint
v_exact = torch.linalg.solve(Hq, gl)
# conjugate gradient
def cg(A, b, iters):
    xk = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = r @ r
    errs = []
    for k in range(iters):
        Ap = A @ p; alpha = rs / (p @ Ap); xk = xk + alpha * p; r = r - alpha * Ap
        rs_new = r @ r; p = r + (rs_new / rs) * p; rs = rs_new
        errs.append((xk - v_exact).norm().item() / v_exact.norm().item())
    return errs
eta = 1.0 / np.linalg.eigvalsh(Hq.numpy()).max()
# Neumann series for (eta H)^-1 = sum_k (I - eta H)^k  (truncated BPTT with window K)
def neumann(A, b, eta, iters):
    Mx = torch.eye(A.shape[0]) - eta * A
    term = b.clone(); acc = torch.zeros_like(b); errs = []
    for k in range(iters):
        acc = acc + term; term = Mx @ term
        errs.append((eta * acc - v_exact).norm().item() / v_exact.norm().item())
    return errs
kappa = float(np.linalg.eigvalsh(Hq.numpy()).max() / np.linalg.eigvalsh(Hq.numpy()).min())
e_cg = cg(Hq, gl, 400); e_ne = neumann(Hq, gl, eta, 4000)
def first_below(errs, tol):
    for i, e in enumerate(errs):
        if e < tol:
            return i + 1
    return None
out['implicit'] = {'condition_number': kappa,
                   'cg_iters_1e-6': first_below(e_cg, 1e-6), 'neumann_iters_1e-6': first_below(e_ne, 1e-6),
                   'cg_iters_1e-3': first_below(e_cg, 1e-3), 'neumann_iters_1e-3': first_below(e_ne, 1e-3)}
# parameter gradient of l(z*(theta)) via the adjoint vs finite differences, for the parameter kappa0_
theta = m.kappa0_
def grad_E_tangent(model, x, z):
    zf = z.reshape(-1).clone().requires_grad_(True)
    E = model.energy(zf.view(1, T, d), x).sum()
    g = torch.autograd.grad(E, zf, create_graph=True)[0].view(T, d)
    return torch.einsum('tda,td->ta', Q, proj(z, g)).reshape(-1)          # Riemannian gradient in basis Q
gE = grad_E_tangent(m, x, zs[0])
# d/dtheta of grad E (mixed derivative) applied with the adjoint:  dl/dtheta = - v^T d(grad E)/dtheta
dtheta = torch.autograd.grad(gE @ v_exact.detach(), theta)[0]
impl_grad = -dtheta.item()
with torch.no_grad():
    h = 1e-5
    th0 = theta.item()
    vals = []
    for sgn in [+1, -1]:
        theta.data.fill_(th0 + sgn * h)
        zz, _, _ = find_min(m, x, zs, eta=0.2, tol=1e-15)
        vals.append((w * zz[0]).sum().item())
    theta.data.fill_(th0)
    fd_grad = (vals[0] - vals[1]) / (2 * h)
out['implicit']['adjoint_vs_finite_diff'] = abs(impl_grad - fd_grad) / max(1e-12, abs(fd_grad))

# ---------------------------------------------------------------- 6. equilibrium propagation
# total energy F_beta = E + beta * l, l(z) = <w,z>; nudged minimum z*_beta; EP estimate of dl(z*_0)/dtheta
def find_min_nudged(model, x, z0, beta, eta=0.2, n_iter=200000, tol=1e-14):
    z = z0.clone()
    with torch.no_grad():
        for i in range(n_iter):
            v = model.drive(z, x) - beta * proj(z, w.view(1, T, d))
            if v.norm(dim=-1).max() < tol:
                break
            z = exp_map(z, eta * v)
    return z
def dE_dtheta(model, x, z):
    E = model.energy(z, x).sum()
    return torch.autograd.grad(E, theta)[0].item()
ep = {}
for beta in [1e-1, 1e-2, 1e-3]:
    zb = find_min_nudged(m, x, zs, beta)
    zb_minus = find_min_nudged(m, x, zs, -beta)
    one_sided = (dE_dtheta(m, x, zb) - dE_dtheta(m, x, zs)) / beta
    centered = (dE_dtheta(m, x, zb) - dE_dtheta(m, x, zb_minus)) / (2 * beta)
    ep[f'beta={beta}'] = dict(one_sided_rel_err=abs(one_sided - fd_grad) / abs(fd_grad),
                              centered_rel_err=abs(centered - fd_grad) / abs(fd_grad))
out['equilibrium_propagation'] = ep
out['reference_grad'] = fd_grad

print(json.dumps(out, indent=1))
with open(sys.argv[1] if len(sys.argv) > 1 else 'verify_energy.json', 'w') as f:
    json.dump(out, f, indent=1)
