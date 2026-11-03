"""Fill the pilot table placeholders in energy.tex from the evaluation JSONs.

usage: python fill_tables.py energy.tex s4_sfprm.json s4_esfprm.json s6_sfprm.json s6_esfprm.json
(a missing file leaves its placeholders as 'n/a')
"""
import json, sys, os, re
import numpy as np
import torch

tex_path = sys.argv[1]
paths = sys.argv[2:6]
res = [json.load(open(p)) if os.path.exists(p) else None for p in paths]
tags = ['4S', '4E', '6S', '6E']

def pct(v): return f'{100 * v:.1f}'
def fmt(v, nd=2): return f'{v:.{nd}f}'

def steps_to_solve(ckpt):
    """first training step at which the train board accuracy reaches 90% (logged every 50 steps)"""
    if not ckpt or not os.path.exists(ckpt):
        return 'n/a'
    log = torch.load(ckpt, map_location='cpu', weights_only=False)['log']
    for r in log:
        if r['acc_board'] >= 0.9:
            return str(r['step'])
    return f"$>{log[-1]['step']}$"

ckpts = {'4S': 'results/checkpoints/sw_S2.pt', '4E': 'results/checkpoints/sw_H7.pt',
         '6S': 'results/checkpoints/s6_sfprm.pt', '6E': 'results/checkpoints/s6_esfprm.pt'}
here = os.path.dirname(os.path.abspath(__file__))
vals = {}
for tag, r in zip(tags, res):
    if r is None:
        for key in 'PTAHICMLWENJO':
            vals[key + tag] = 'n/a'
        vals['A' + tag + 'I'] = vals['A' + tag + 'O'] = 'n/a'
        continue
    energy = r['model'] == 'esfprm'
    mode = 'armijo' if energy else 'patience'
    idr, ood = r['test_id'], r['test_ood']
    vals['P' + tag] = f"{r['n_params'] / 1000:.0f}"
    vals['T' + tag] = steps_to_solve(os.path.join(here, ckpts[tag]))
    vals['A' + tag + 'I'] = pct(idr[f'{mode}/board_acc'])
    vals['A' + tag + 'O'] = pct(ood[f'{mode}/board_acc'])
    vals['H' + tag] = pct(ood[f'{mode}/halt_frac'])
    vals['I' + tag] = f"{ood[f'{mode}/iters_median']:.0f} / {ood[f'{mode}/iters_p90']:.0f}"
    j = ood.get('jac') or idr.get('jac')
    if j and j['n']:
        vals['C' + tag] = pct(j['complex_frac']['mean'])
        vals['M' + tag] = fmt(j['max_imag']['mean'], 2) if j['max_imag']['mean'] > 1e-6 else '$0$'
        vals['W' + tag] = fmt(j['window']['mean'], 2)
        # lambda_max(Hess E) = 1 - min Re(zeta) for the energy model
        eigs = ood.get('jac_eigs') or idr.get('jac_eigs')
        if energy and eigs:
            lam = np.mean([1 - min(e[0] for e in puzzle) for puzzle in eigs])
            vals['L' + tag] = fmt(lam, 1)
        else:
            vals['L' + tag] = '--'
    else:
        for key in 'CMWL':
            vals[key + tag] = 'n/a'
    rs = ood.get('restarts', {})
    B = str(max(int(b) for b in rs['oracle'])) if rs else None
    vals['E' + tag] = pct(rs['energy'][B]) if (rs and 'energy' in rs) else '--'
    vals['N' + tag] = pct(rs['anchor'][B]) if rs else 'n/a'
    vals['J' + tag] = pct(rs['majority'][B]) if rs else 'n/a'
    vals['O' + tag] = pct(rs['oracle'][B]) if rs else 'n/a'

# full appendix table rows
rows = []
names = {'4S': ('$4\\times4$', 'SFPRM'), '4E': ('$4\\times4$', 'Energy'), '6S': ('$6\\times6$', 'SFPRM'), '6E': ('$6\\times6$', 'Energy')}
for tag, r in zip(tags, res):
    if r is None:
        continue
    task, model = names[tag]
    modes = ['patience', 'armijo'] if r['model'] == 'esfprm' else ['patience']
    for mode in modes:
        idr, ood = r['test_id'], r['test_ood']
        rows.append(f"    {task} & {model} ({'Pat.' if mode == 'patience' else 'Arm.'}) & {pct(idr[mode + '/board_acc'])} & "
                    f"{pct(ood[mode + '/board_acc'])} & {pct(idr[mode + '/halt_frac'])} & {pct(ood[mode + '/halt_frac'])} & "
                    f"{idr[mode + '/iters_median']:.0f} / {idr[mode + '/iters_p90']:.0f} & "
                    f"{ood[mode + '/iters_median']:.0f} / {ood[mode + '/iters_p90']:.0f} \\\\")
full_rows = '\n'.join(rows)
app_path = sys.argv[6] if len(sys.argv) > 6 else None
if app_path:
    a = open(app_path).read().replace('FULLROWS', full_rows)
    open(app_path, 'w').write(a)

s = open(tex_path).read()
# replace longer keys first (A4SI before A4S is not an issue since A4S alone is not a placeholder)
for key in sorted(vals, key=len, reverse=True):
    s = re.sub(r'(?<![A-Za-z])' + key + r'(?![A-Za-z])', vals[key], s)
open(tex_path, 'w').write(s)
print(json.dumps(vals, indent=1))
