# Run index (all on 4x4 Sudoku unless noted; see `code/esfprm_pilot.py` for flags)

Energy-model development sweep, in chronological order. "plateau" = cell accuracy stalls at the
clue-only solution (~0.71 on 4x4), i.e. nothing learned for blank cells.

| log | model / flags | outcome |
|---|---|---|
| sw_sfprm | SFPRM, free positions, 400 steps | 94% train boards at step 400 |
| sw_S2 | SFPRM, structured positions, 400 steps | 100% at step 300 (used for 4x4 eval) |
| sw_e1..e4 | flat energy model, attractive heads, unit-logit W init; e2 smaller scales, e3 chordal injection, e4 lr 5e-3 | plateau |
| sw_A..D | flat, weaker injection / larger attention / sharper logits / shorter windows | collapse to chance |
| sw_H1..H3 | hidden tokens (H2 structured positions, H3 larger kappaA) | plateau / unstable |
| sw_H4 | hidden, structured, kappaA 0.2 (new small-W / large-beta init), eta=1 | plateau, energy rises in 44% of steps |
| sw_H5, sw_H6 | as H4 with Armijo training (H6: K=12) | monotone energy, plateau at 600 steps |
| sw_H7 | as H5, 2000 steps | escapes the plateau after ~500 steps, 94% boards by step 1300 (used for 4x4 eval) |
| sw_H8, sw_H9 | H5 with full BPTT (K=16) / lr 5e-3 | killed early (superseded by H7) |
| toy_sfprm/flat/hidden, toy_T*, toy_U* | routing toy (copy the right neighbour's digit), `code/diagnostics/routing_toy*.py` | U3/U4 (small W, large beta, kappaA 0.2) learn; large-W inits do not |
| s6_sfprm | SFPRM, 6x6, 4000 steps | 100% train boards from step ~900 |
| s6_esfprm | energy model, 6x6, hidden, structured, kappaA 0.2, Armijo, 4000 steps | see paper |

Codebook variant (second round; `--tied_readout --tie_embed --kappaAns 0.5`, hidden tokens, Armijo training):

| log | flags | purpose |
|---|---|---|
| s4_ev2 | + beta schedule (rho 8, K 8) + ranking loss 0.1 (R=2), 2000 steps | full codebook variant, 4x4 |
| s4_ev2_fixedbeta | ranking loss, no schedule | ablation: fixed beta |
| s4_ev2_norank | schedule, no ranking loss | ablation: no ranking loss |
| s4_sfprm_mem | SFPRM with `--mlp memory --n_mem 256`, 400 steps | proposal's ablation 1: only the MLP swapped |
| s6_ev2 | full variant, 6x6, 4000 steps | |
| s6_sfprm_mem | SFPRM memory-MLP, 6x6, 4000 steps | |
| *_eval | evaluation of the above (`*_eval.json` in `code/results/`) | |
