# SFPRM as an Operator-Split AKOrN — ICML 2026 submission (blind)

Converted from the report `sfprm_akorn_equivalence.pdf`.

## Files
| File | Purpose |
|---|---|
| `main.tex` | Main paper (blind review mode, `\usepackage{icml2026}`) |
| `appendix.tex` | Appendix A (preliminaries), B (proofs), C (verification setup); `\input` by `main.tex` |
| `references.bib` | Bibliography (entries marked `VERIFY` need completion) |
| `energy.tex` | Section 8: the energy-based SFPRM (definition, design choices, theory, training, pilot); `\input` by `main.tex` |
| `appendix_energy.tex` | Appendix D (proofs for the energy model), E (pilot experiment), F (numerical verification of the energy statements); `\input` by `main.tex` |
| `code/esfprm_pilot.py` | Pilot experiment: Sudoku generation, SFPRM and energy-SFPRM models, training, evaluation (`gen` / `train` / `eval`) |
| `code/verify_energy.py` | Double-precision verification of the energy-model identities (Table 3, lower block) |
| `code/make_figures.py` | Figures for the pilot (Jacobian spectra, accuracy against restarts) |
| `code/results/` | JSON outputs of the verification and pilot evaluation |
| `figures/` | Pilot figures (`spectrum.pdf`, `restarts.pdf`) |
| `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` | ICML 2026 style files (see provenance below) |

## Reproducing the pilot (CPU/MPS, minutes to an hour)
```
PY=python   # needs numpy, torch, scipy, matplotlib
$PY code/esfprm_pilot.py gen --n 6 --br 2 --bc 3 --out data6
$PY code/esfprm_pilot.py train --data data6 --out runs/sfprm.pt  --model sfprm  --structured --br 2 --bc 3
$PY code/esfprm_pilot.py train --data data6 --out runs/esfprm.pt --model esfprm --hidden --structured --br 2 --bc 3 --kappaA 0.2 --armijo_train
$PY code/esfprm_pilot.py eval  --data data6 --ckpt runs/sfprm.pt  --out code/results/sfprm_eval.json
$PY code/esfprm_pilot.py eval  --data data6 --ckpt runs/esfprm.pt --out code/results/esfprm_eval.json
$PY code/make_figures.py code/results/sfprm_eval.json code/results/esfprm_eval.json figures
$PY code/verify_energy.py code/results/verify_energy.json
```

## Compile
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or: latexmk -pdf main.tex
```
Tested with TeX Live 2023 (pdfTeX 1.40.25): compiles with no errors, no overfull boxes,
no undefined references; all fonts are embedded Type 1. Two benign warnings remain
(hyperref drops the `\\` from the title in PDF metadata; an empty anchor from the
template's title block).

For the camera-ready version use `\usepackage[accepted]{icml2026}`, fill in authors,
and add acknowledgements.

## Style-file provenance (please re-download)
The official zip (media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip) was not
reachable from the build machine. The included files come from public mirrors of the
official template (`icml2026.sty`, "version of 2025-10-29"); two independent mirrors
had byte-identical files (sha256 prefix of icml2026.sty: 7cdcf90f6a59c521). Replace
them with the files from the official ICML 2026 author kit before submitting.
