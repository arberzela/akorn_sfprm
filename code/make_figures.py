"""Figures for the pilot.

usage: python make_figures.py OUT_DIR TAG sfprm_eval.json esfprm_eval.json [sfprm_ckpt.pt esfprm_ckpt.pt]

Writes OUT_DIR/spectrum_TAG.pdf, restarts_TAG.pdf, halting_TAG.pdf and, with checkpoints, training_TAG.pdf.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, tag = sys.argv[1], sys.argv[2]
sf = json.load(open(sys.argv[3]))
en = json.load(open(sys.argv[4]))
ckpts = sys.argv[5:7]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
C_SF, C_EN = '#1f4e79', '#b23a48'

# ---------------- spectrum: eigenvalues of dF at halted OOD states (first three puzzles pooled)
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharex=True, sharey=True)
for ax, res, name, col in zip(axes, [sf, en], ['SFPRM', 'Energy SFPRM'], [C_SF, C_EN]):
    eigs = res['test_ood'].get('jac_eigs') or res['test_id'].get('jac_eigs') or []
    ev = np.array([e for puzzle in eigs for e in puzzle]) if eigs else np.zeros((0, 2))
    if name != 'SFPRM' and len(ev):
        # the energy model's unit-step map has real eigenvalues 1 - h; show the damped map at
        # eta = 1/lambda_max (half the window), which maps them into [0, 1)
        h = 1 - ev[:, 0]
        ev = np.stack([1 - h / h.max(), ev[:, 1] / h.max()], 1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), lw=0.5, color='gray', zorder=0)
    ax.axhline(0, lw=0.4, color='gray', zorder=0)
    if len(ev):
        ax.scatter(ev[:, 0], ev[:, 1], s=2.5, alpha=0.5, linewidths=0, color=col)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel(r'$\mathrm{Re}\,\zeta$')
    if len(ev):
        frac = 100 * (np.abs(ev[:, 1]) > 1e-8).mean()
        ax.text(0.03, 0.04, f'{frac:.0f}% complex', transform=ax.transAxes, fontsize=6.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
axes[0].set_ylabel(r'$\mathrm{Im}\,\zeta$')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out, f'spectrum_{tag}.pdf'), bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- restarts: OOD board accuracy against B
fig, ax = plt.subplots(figsize=(3.3, 2.1))
styles = {'oracle': (':', 'oracle'), 'majority': ((0, (3, 1, 1, 1)), 'majority'),
          'anchor': ('--', 'anchor angle'), 'energy': ('-', 'energy'), 'energy_sum': ((0, (5, 1)), 'summed energy')}
for res, name, col, marker in [(sf, 'SFPRM', C_SF, 'o'), (en, 'Energy SFPRM', C_EN, 's')]:
    r = res['test_ood'].get('restarts', {})
    for key, (ls, lab) in styles.items():
        if key in r:
            Bs = sorted(int(b) for b in r[key])
            ys = [100 * r[key][str(b)] for b in Bs]
            ax.plot(Bs, ys, color=col, ls=ls, marker=marker, ms=2.5, lw=1, label=f'{name}: {lab}')
ax.set_xscale('log', base=2)
ax.set_xlabel('restarts $B$')
ax.set_ylabel('OOD board accuracy (%)')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=2.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out, f'restarts_{tag}.pdf'), bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- halting: distribution of iterations to halt (patience rule), ID and OOD
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.8), sharey=True)
for ax, split, title in zip(axes, ['test_id', 'test_ood'], ['ID', 'OOD']):
    for res, name, col in [(sf, 'SFPRM', C_SF), (en, 'Energy SFPRM', C_EN)]:
        mode = 'armijo' if (name != 'SFPRM' and f'armijo/iters_list' in res[split]) else 'patience'
        if mode == 'armijo':
            name = name + ' (Armijo)'
        it = np.array(res[split].get(f'{mode}/iters_list', []))
        ha = np.array(res[split].get(f'{mode}/halted_list', []), dtype=bool)
        if len(it) == 0:
            continue
        it = it[ha] if ha.any() else it
        xs = np.sort(it); ys = np.arange(1, len(xs) + 1) / len(it) * 100 * (ha.mean() if len(ha) else 1)
        ax.step(xs, ys, where='post', color=col, lw=1.2, label=name)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('iterations to halt')
    ax.set_xscale('log', base=2)
axes[0].set_ylabel('puzzles halted (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out, f'halting_{tag}.pdf'), bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ---------------- training curves
if len(ckpts) == 2:
    import torch
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    for ck, name, col in [(ckpts[0], 'SFPRM', C_SF), (ckpts[1], 'Energy SFPRM', C_EN)]:
        log = torch.load(ck, map_location='cpu', weights_only=False)['log']
        steps = [r['step'] for r in log]; acc = [100 * r['acc_board'] for r in log]
        ax.plot(steps, acc, color=col, lw=1, label=name)
    ax.set_xlabel('training step'); ax.set_ylabel('train board acc. (%)')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, f'training_{tag}.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
print('wrote figures to', out)
