"""Generate the main-text results table (tab_results.tex) from the evaluation JSONs.

usage: python make_main_table.py OUT.tex
Columns: 4x4 SFPRM, first energy model, codebook, codebook + ranking loss, SFPRM with memory MLP;
6x6 SFPRM, first energy model, codebook + ranking loss, SFPRM with memory MLP.
A missing evaluation file gives 'n/a' cells.
"""
import json, os, sys
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(here, 'results')
# (eval json, training checkpoint)
cols = [('s4_sfprm', 'sw_S2'), ('s4_esfprm', 'sw_H7'), ('s4_ev2_norank', 's4_ev2_norank'),
        ('s4_ev2', 's4_ev2'), ('s4_sfprm_mem', 's4_sfprm_mem'),
        ('s6_sfprm', 's6_sfprm'), ('s6_esfprm', 's6_esfprm'), ('s6_ev2', 's6_ev2'),
        ('s6_sfprm_mem', 's6_sfprm_mem')]
NA = 'n/a'


def pct(v):
    return NA if v is None or v != v else f'{100 * v:.1f}'


def column(ev, ck):
    jp = os.path.join(R, f'{ev}_eval.json')
    if not os.path.exists(jp):
        return None
    r = json.load(open(jp))
    energy = r['model'] == 'esfprm'
    mode = 'armijo' if energy else 'patience'
    idr, ood = r['test_id'], r['test_ood']
    c = {}
    c['params'] = f"{r['n_params'] / 1000:.0f}"
    cp = os.path.join(R, 'checkpoints', f'{ck}.pt')
    log = torch.load(cp, map_location='cpu', weights_only=False)['log'] if os.path.exists(cp) else []
    c['steps'] = next((str(x['step']) for x in log if x['acc_board'] >= 0.9),
                      f"$>{log[-1]['step'] + 1}$" if log else NA)
    c['id'] = pct(idr[f'{mode}/board_acc'])
    c['ood'] = pct(ood[f'{mode}/board_acc'])
    c['halted'] = pct(ood[f'{mode}/halt_frac'])
    med, p90 = ood[f'{mode}/iters_median'], ood[f'{mode}/iters_p90']
    c['iters'] = '--' if med != med else f'{med:.0f}/{p90:.0f}'
    j = ood.get('jac') if ood.get('jac', {}).get('n') else idr.get('jac')
    eigs = ood.get('jac_eigs') or idr.get('jac_eigs')
    if j and j.get('n'):
        c['complex'] = pct(j['complex_frac']['mean'])
        c['window'] = f"{j['window']['mean']:.2f}"
        c['lmax'] = f"{np.mean([1 - min(e[0] for e in pz) for pz in eigs]):.1f}" if (energy and eigs) else '--'
    else:
        c['complex'] = c['window'] = c['lmax'] = NA
    rs = ood.get('restarts', {})
    B = str(max(int(b) for b in rs['oracle'])) if rs else None
    c['energy'] = pct(rs['energy'][B]) if (rs and 'energy' in rs) else '--'
    c['anchor'] = pct(rs['anchor'][B]) if rs else NA
    c['majority'] = pct(rs['majority'][B]) if rs else NA
    c['oracle'] = pct(rs['oracle'][B]) if rs else NA
    if f'{mode}/committed_frac' in ood:
        c['committed'] = pct(ood[f'{mode}/committed_frac'])
        c['commacc'] = f"{pct(ood[f'{mode}/acc_committed'])}/{pct(ood[f'{mode}/acc_uncommitted'])}"
    else:
        c['committed'] = c['commacc'] = '--'
    return c


rows = [('params', r'Parameters ($\times10^3$)'), ('steps', r'Steps to $90\%$ train acc.'),
        ('id', r'Board acc.\ ID'), ('ood', r'Board acc.\ OOD'), ('halted', r'Halted ($\le128$ it.)'),
        ('iters', r'Iterations (med./p90)'), ('complex', r'Complex $\zeta$ (\%)'),
        ('lmax', r'$\lambda_{\max}(\operatorname{Hess}E)$'), ('window', r'Window (Cor.~\ref{cor:window})'),
        ('energy', r'$B{=}64$: lowest energy'), ('anchor', r'$B{=}64$: anchor angle'),
        ('majority', r'$B{=}64$: majority'), ('oracle', r'$B{=}64$: oracle'),
        ('committed', r'Committed boards'), ('commacc', r'Acc.\ committed/uncomm.')]
data = [column(ev, ck) for ev, ck in cols]
lines = [r'\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}', r'\toprule',
         r' & \multicolumn{5}{c}{$4\times4$} & \multicolumn{4}{c}{$6\times6$} \\',
         r'\cmidrule(lr){2-6}\cmidrule(lr){7-10}',
         r' & SFPRM & Energy & Codebook & $+$rank & Mem.\ MLP & SFPRM & Energy & $+$rank & Mem.\ MLP \\',
         r'\midrule']
for key, label in rows:
    cells = [(d[key] if d else NA) for d in data]
    lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    if key in ('ood', 'window', 'oracle'):
        lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}']
open(sys.argv[1], 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
