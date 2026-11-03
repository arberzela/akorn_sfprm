# Tools behind the energy-SFPRM section of `viz/sfprm_akorn.html`

| File | Purpose |
|---|---|
| `toy3.py` | Python prototype of the 4×4 toy Sudoku energy (cells on S³, digit prototypes e₁…e₄, repulsive attention, annealed codebook memory) |
| `sweep.py` | Parameter sweep used to pick the toy weights shown on the page |
| `export.py` | Exports the puzzles, the measured eigenvalues and the restart curves (from `code/results/*_eval.json`) into `vizdata.json`, which is embedded in the page |
| `section.html`, `section.js` | Source of the new section and its script as inserted into the page (the page may contain later edits) |
| `test_page_jsdom.js` | Loads the page in jsdom, exercises every control of the new section, reports script errors and NaN attributes (`npm install jsdom@24.1.3`, then `node test_page_jsdom.js ../../viz/sfprm_akorn.html`) |
| `dump_svgs_jsdom.js` | Dumps the rendered figures as standalone SVGs for a visual check |
