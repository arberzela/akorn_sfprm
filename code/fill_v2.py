"""Fill the codebook-variant table (tab:v2) in energy.tex from evaluation JSONs and training checkpoints.

usage: python fill_v2.py energy.tex
Reads code/results/{s4_ev2,s4_ev2_fixedbeta,s4_ev2_norank,s4_sfprm_mem,s6_ev2,s6_sfprm_mem}_eval.json
and the matching checkpoints; a missing file leaves 'n/a'.
"""
import json, os, re, sys
import numpy as np
import torch

tex_path = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
cols = {'F4': 's4_ev2_norank', 'R4': 's4_ev2', 'B4': 's4_ev2_fixedbeta', 'M4': 's4_sfprm_mem',
        'F6': 's6_ev2', 'M6': 's6_sfprm_mem'}

def pct(v): return f'{100 * v:.1f}'

vals = {}
for col, name in cols.items():
    jp = os.path.join(here, 'results', f'{name}_eval.json')
    cp = os.path.join(here, 'results', 'checkpoints', f'{name}.pt')
    keys = ['P', 'T', 'H', 'I', 'C', 'L', 'W', 'E', 'N', 'K', 'G']
    if not os.path.exists(jp):
        for k in keys:
            vals[k + col] = 'n/a'
        vals['A' + col + 'I'] = vals['A' + col + 'O'] = 'n/a'
        continue
    r = json.load(open(jp))
    energy = r['model'] == 'esfprm'
    mode = 'armijo' if energy else 'patience'
    idr, ood = r['test_id'], r['test_ood']
    vals['P' + col] = f"{r['n_params'] / 1000:.0f}"
    log = torch.load(cp, map_location='cpu', weights_only=False)['log'] if os.path.exists(cp) else []
    t90 = next((str(x['step']) for x in log if x['acc_board'] >= 0.9), f"$>{log[-1]['step'] + 1}$" if log else 'n/a')
    vals['T' + col] = t90
    vals['A' + col + 'I'] = pct(idr[f'{mode}/board_acc'])
    vals['A' + col + 'O'] = pct(ood[f'{mode}/board_acc'])
    vals['H' + col] = pct(ood[f'{mode}/halt_frac'])
    med, p90 = ood[f'{mode}/iters_median'], ood[f'{mode}/iters_p90']
    vals['I' + col] = '--' if med != med else f"{med:.0f}/{p90:.0f}"
    j = ood.get('jac') or idr.get('jac')
    if j and j['n']:
        vals['C' + col] = pct(j['complex_frac']['mean'])
        vals['W' + col] = f"{j['window']['mean']:.2f}"
        eigs = ood.get('jac_eigs') or idr.get('jac_eigs')
        vals['L' + col] = f"{np.mean([1 - min(e[0] for e in pz) for pz in eigs]):.1f}" if (energy and eigs) else '--'
    else:
        vals['C' + col] = vals['W' + col] = vals['L' + col] = 'n/a'
    rs = ood.get('restarts', {})
    B = str(max(int(b) for b in rs['oracle'])) if rs else None
    vals['E' + col] = (pct(rs['energy'][B]) + '/' + pct(rs['energy_sum'][B])) if (rs and 'energy' in rs) else '--'
    vals['N' + col] = (pct(rs['anchor'][B]) + '/' + pct(rs['majority'][B]) + '/' + pct(rs['oracle'][B])) if rs else 'n/a'
    if f'{mode}/committed_frac' in ood:
        ac = ood[f'{mode}/acc_committed']; au = ood[f'{mode}/acc_uncommitted']
        f = lambda v: 'n/a' if (v != v) else pct(v)
        vals['K' + col] = f"{pct(ood[f'{mode}/committed_frac'])} ({f(ac)}/{f(au)})"
    else:
        vals['K' + col] = '--'
    rk = [x['rank_correct_lower'] for x in log if 'rank_correct_lower' in x]
    vals['G' + col] = pct(np.mean(rk[-5:])) if rk else '--'

s = open(tex_path).read()
for key in sorted(vals, key=len, reverse=True):
    s = re.sub(r'(?<![A-Za-z])' + key + r'(?![A-Za-z0-9])', vals[key], s)
open(tex_path, 'w').write(s)
print(json.dumps(vals, indent=1))
