"use strict";
// ================= Energy-based SFPRM =================
// Uses the helpers of the main script ($, el, css, mulberry32, gauss, dot, nrm, add, scl, unit, proj, clamp, fmt, deg, reduceMotion, showTip, hideTip).
const VD = JSON.parse(document.getElementById("vizdata").textContent);
const TAU = 2 * Math.PI;
const wrapPi = (a) => { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; };
function softmax(ls) { const m = Math.max(...ls); const w = ls.map((v) => Math.exp(v - m)); const Z = w.reduce((a, b) => a + b, 0); return { p: w.map((v) => v / Z), lse: m + Math.log(Z) }; }
function arrowMarkers(svg, prefix, cols) {
  const defs = el("defs", {}, svg);
  cols.forEach((c) => { const mk = el("marker", { id: `${prefix}-${c}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs); el("path", { d: "M0,0 L10,5 L0,10 z", fill: `var(--${c})` }, mk); });
  return defs;
}
const supTxt = (k) => `10<tspan dy="-5" font-size="9">${String(k).replace("-", "−")}</tspan>`;

// ---------- E1 · rotation versus descent ----------
(function rotDemo() {
  const svg = $("rotPlane"), S = 100, HMIN = 0.25, E0 = [1.05, 0.85];
  const defs = el("defs", {}, svg);
  const cp = el("clipPath", { id: "rotClip" }, defs); el("rect", { x: -160, y: -130, width: 320, height: 260 }, cp);
  const g = el("g", { "clip-path": "url(#rotClip)" }, svg);
  el("text", { x: 156, y: -6, "text-anchor": "end", class: "svgtext" }, svg).textContent = "stiff mode";
  el("text", { x: 6, y: -118, class: "svgtext" }, svg).textContent = "slow mode";
  const par = () => ({ om: +$("rOm").value / 100, kap: +$("rKap").value, eta: +$("rEta").value / 100 });
  function lam(h1, h2, om) {                      // eigenvalues of M = [[h1, -om], [om, h2]]
    const tr = h1 + h2, disc = ((h1 - h2) / 2) ** 2 - om * om;
    if (disc >= 0) { const s = Math.sqrt(disc); return [[tr / 2 + s, 0], [tr / 2 - s, 0]]; }
    const s = Math.sqrt(-disc); return [[tr / 2, s], [tr / 2, -s]];
  }
  const rho = (L, eta) => Math.max(...L.map(([a, b]) => Math.hypot(1 - eta * a, eta * b)));
  const windowOf = (L) => Math.min(...L.map(([a, b]) => 2 * a / (a * a + b * b)));
  function run(h1, h2, om, eta) {
    let e = E0.slice(); const pts = [e], En = [0.5 * (h1 * e[0] ** 2 + h2 * e[1] ** 2)], incs = []; let it = -1; const n0 = Math.hypot(E0[0], E0[1]);
    for (let i = 1; i <= 400; i++) {
      e = [e[0] - eta * (h1 * e[0] - om * e[1]), e[1] - eta * (om * e[0] + h2 * e[1])]; pts.push(e);
      const E = 0.5 * (h1 * e[0] ** 2 + h2 * e[1] ** 2); if (E > En[En.length - 1] * (1 + 1e-9)) incs.push(i); En.push(E);
      const n = Math.hypot(e[0], e[1]); if (it < 0 && n < 1e-3 * n0) it = i; if (n > 1e3) break;
    }
    return { pts, En, incs, it };
  }
  const ev = $("rotEig"), U = 55;
  el("line", { x1: -112, x2: 112, y1: 0, y2: 0, stroke: "var(--rule)" }, ev); el("line", { x1: 0, x2: 0, y1: -76, y2: 76, stroke: "var(--rule)" }, ev);
  el("circle", { cx: 0, cy: 0, r: U, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.2 }, ev);
  el("circle", { cx: U, cy: 0, r: 3, fill: "var(--ink)" }, ev); el("text", { x: U + 4, y: 13, class: "svgtext" }, ev).textContent = "1";
  el("text", { x: -112, y: -66, class: "svgtext" }, ev).textContent = "eigenvalues of the step";
  const evg = el("g", {}, ev);
  const er = $("rotErr");
  function draw() {
    const p = par(), h1 = HMIN * p.kap, h2 = HMIN;
    $("rOmv").textContent = p.om.toFixed(2); $("rKapv").textContent = p.kap + "×"; $("rEtav").textContent = p.eta.toFixed(2);
    g.innerHTML = "";
    el("line", { x1: -160, x2: 160, y1: 0, y2: 0, stroke: "var(--rule)" }, g); el("line", { x1: 0, x2: 0, y1: -130, y2: 130, stroke: "var(--rule)" }, g);
    const Es = 0.5 * (h1 * E0[0] ** 2 + h2 * E0[1] ** 2);
    [1, 0.5, 0.25, 0.1, 0.03, 0.008].forEach((f) => el("ellipse", { cx: 0, cy: 0, rx: S * Math.sqrt(2 * f * Es / h1), ry: S * Math.sqrt(2 * f * Es / h2), fill: "none", stroke: "var(--s3)", "stroke-width": 1, opacity: 0.6 }, g));
    const R = run(h1, h2, p.om, p.eta);
    el("polyline", { points: R.pts.slice(0, 250).map((q) => `${(S * q[0]).toFixed(1)},${(-S * q[1]).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--s2)", "stroke-width": 1.8 }, g);
    R.pts.slice(0, 60).forEach((q, i) => el("circle", { cx: S * q[0], cy: -S * q[1], r: i === 0 ? 5 : 2.4, fill: R.incs.includes(i) ? "var(--bad)" : "var(--s2)" }, g));
    el("circle", { cx: 0, cy: 0, r: 4, fill: "var(--ink)" }, g);
    el("text", { x: S * E0[0] + 7, y: -S * E0[1] - 6, class: "svgmath" }, g).textContent = "e₀";
    el("text", { x: 6, y: 16, class: "svgtext" }, g).textContent = "fixed point";
    // eigenvalues
    const L = lam(h1, h2, p.om), cplx = L[0][1] !== 0, w = windowOf(L), r = rho(L, p.eta);
    evg.innerHTML = "";
    L.forEach(([a, b]) => {
      const zx = clamp(U * (1 - a), -110, 110), zy = clamp(-U * (-b), -74, 74);
      const dx = clamp(U * (1 - p.eta * a), -110, 110), dy = clamp(-U * (-p.eta * b), -74, 74);
      el("line", { x1: U, y1: 0, x2: zx, y2: zy, stroke: "var(--muted)", "stroke-dasharray": "3 3" }, evg);
      el("circle", { cx: zx, cy: zy, r: 6, fill: "none", stroke: "var(--s2)", "stroke-width": 2 }, evg);
      el("circle", { cx: dx, cy: dy, r: 4.5, fill: "var(--s1)", stroke: "var(--surface)", "stroke-width": 1.5 }, evg);
    });
    // best fixed step by scan
    let best = [Infinity, 0]; for (let k = 1; k < 600; k++) { const et = w * k / 600; const rr = rho(L, et); if (rr < best[0]) best = [rr, et]; }
    const itBest = Math.ceil(Math.log(1e-3) / Math.log(best[0]));
    const z1 = 1 - L[0][0], z2 = 1 - L[1][0];
    $("rotRead").innerHTML =
      `<dt>eigenvalues ζ of the unit step</dt><dd>${cplx ? `${z1.toFixed(2)} ± ${Math.abs(L[0][1]).toFixed(2)}i` : `${z1.toFixed(2)}, ${z2.toFixed(2)}`}</dd>` +
      `<dt>spectrum</dt><dd>${cplx ? '<span class="st-bad">complex pair</span>' : '<span class="st-ok">real</span>'}</dd>` +
      `<dt>stable for η &lt;</dt><dd>${w.toFixed(2)}</dd>` +
      `<dt>shrink per step at this η</dt><dd style="color:var(${r < 1 ? "--good" : "--bad"})">${r.toFixed(3)}</dd>` +
      `<dt>error ÷1000 after</dt><dd>${R.it > 0 ? R.it + " steps" : (r >= 1 ? "never (unstable)" : "> 400 steps")}</dd>` +
      `<dt>with the best η = ${best[1].toFixed(2)}</dt><dd>${itBest} steps</dd>` +
      `<dt>steps where ½eᵀHe rose</dt><dd>${R.incs.length}</dd>`;
    // energy trace
    er.innerHTML = ""; const K = 120, lo = -6, hi = 1;
    const X = (k) => 34 + 260 * k / K, Y = (v) => 108 - 92 * (clamp(Math.log10(v), lo, hi) - lo) / (hi - lo);
    [-6, -4, -2, 0].forEach((k) => { el("line", { x1: 34, x2: 294, y1: Y(Math.pow(10, k)), y2: Y(Math.pow(10, k)), stroke: "var(--rule)" }, er); el("text", { x: 30, y: Y(Math.pow(10, k)) + 4, "text-anchor": "end", class: "svgtext" }, er).innerHTML = supTxt(k); });
    const ratio = R.En.slice(0, K + 1).map((v) => Math.max(1e-9, v / R.En[0]));
    el("polyline", { points: ratio.map((v, k) => `${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--s2)", "stroke-width": 1.8 }, er);
    R.incs.filter((k) => k <= K).forEach((k) => el("circle", { cx: X(k), cy: Y(ratio[k]), r: 2.6, fill: "var(--bad)" }, er));
    el("text", { x: 34, y: 126, class: "svgtext" }, er).textContent = "½eᵀHe relative to the start, over 120 steps (log scale)";
  }
  ["rOm", "rKap", "rEta"].forEach((id) => $(id).addEventListener("input", draw));
  draw();
})();

// ---------- E2 · one cell, the memory term and its β knob ----------
(function cellDemo() {
  const svg = $("cellDial"), R = 80, MU = [0, 1, 2, 3].map((k) => k * Math.PI / 2), ETA = 0.15, RHO = 30, KA = 70, ITERS = 170;
  const P = (a, r = R) => [r * Math.sin(a), -r * Math.cos(a)];          // angle clockwise from the top
  const betaOf = () => 0.3 * Math.pow(100, +$("cB").value / 100);
  arrowMarkers(svg, "cd", ["s1"]);
  el("circle", { cx: 0, cy: 0, r: R, fill: "none", stroke: "var(--rule)", "stroke-width": 1.5 }, svg);
  const dyn = el("g", {}, svg);
  MU.forEach((m, k) => { const q = P(m), t = P(m, R + 44); el("circle", { cx: q[0], cy: q[1], r: 4, fill: "var(--ink)" }, svg); el("text", { x: t[0], y: t[1] + 5, "text-anchor": "middle", class: "svgtext", style: "font-weight:600;fill:var(--ink)" }, svg).textContent = "digit " + (k + 1); });
  let th = 0, trail = [], anim = 0, betaNow = null, msg = "";
  const E = (t, b, s, phi) => -s * Math.cos(t - phi) - softmax(MU.map((m) => b * Math.cos(t - m))).lse / b;
  function dE(t, b, s, phi) { const { p } = softmax(MU.map((m) => b * Math.cos(t - m))); let g = s * Math.sin(t - phi); MU.forEach((m, k) => { g += p[k] * Math.sin(t - m); }); return g; }
  const resp = (t, b) => softmax(MU.map((m) => b * Math.cos(t - m))).p;
  const nearest = (t) => { let k = 0, bd = 9; MU.forEach((m, i) => { const d = Math.abs(wrapPi(t - m)); if (d < bd) { bd = d; k = i; } }); return k; };
  const cv = $("cellCurve"), bars = $("cellBars");
  const ux = (t) => 40 + 400 * (((t + Math.PI / 4) % TAU + TAU) % TAU) / TAU;  // digits sit at interior x positions
  const inputs = () => ({ s: +$("cS").value / 100, phi: +$("cPhi").value * Math.PI / 180 });
  function draw() {
    const bF = betaOf(), b = betaNow ?? bF, { s, phi } = inputs();
    $("cBv").textContent = bF.toFixed(1); $("cPhiv").textContent = $("cPhi").value + "°"; $("cSv").textContent = s.toFixed(2); $("cStartv").textContent = $("cStart").value + "°";
    dyn.innerHTML = "";
    const N = 240, dens = []; for (let i = 0; i < N; i++) { const a = i / N * TAU; dens.push(MU.reduce((acc, m) => acc + Math.exp(b * (Math.cos(a - m) - 1)), 0)); }
    const dm = Math.max(...dens);
    el("polygon", { points: dens.map((d, i) => P(i / N * TAU, R + 4 + 26 * d / dm).map((v) => v.toFixed(1)).join(",")).join(" "), fill: "var(--s3)", "fill-opacity": 0.12, stroke: "var(--s3)", "stroke-width": 1.5 }, dyn);
    const tipA = P(phi, 16 + 60 * s); if (s > 0.01) el("line", { x1: 0, y1: 0, x2: tipA[0], y2: tipA[1], stroke: "var(--s1)", "stroke-width": 3, "marker-end": "url(#cd-s1)" }, dyn);
    if (trail.length > 1) el("polyline", { points: trail.map((t) => P(t, R).map((v) => v.toFixed(1)).join(",")).join(" "), fill: "none", stroke: "var(--s2)", "stroke-width": 2.5, opacity: 0.5 }, dyn);
    const q = P(th); el("circle", { cx: q[0], cy: q[1], r: 8, fill: "var(--s2)", stroke: "var(--surface)", "stroke-width": 2 }, dyn);
    // energy along the circle
    cv.innerHTML = ""; const M = 361, ts = Array.from({ length: M }, (_, i) => -Math.PI / 4 + i / (M - 1) * TAU);
    const Ev = ts.map((t) => E(t, b, s, phi)), lo = Math.min(...Ev), hi = Math.max(...Ev), Y = (v) => 156 - 132 * (v - lo) / Math.max(hi - lo, 1e-9);
    el("line", { x1: 40, x2: 440, y1: 156, y2: 156, stroke: "var(--rule)" }, cv);
    MU.forEach((m, k) => { const x = ux(m); el("line", { x1: x, x2: x, y1: 20, y2: 160, stroke: "var(--rule)" }, cv); el("text", { x, y: 176, "text-anchor": "middle", class: "svgtext" }, cv).textContent = "digit " + (k + 1); });
    const xp = ux(phi); el("line", { x1: xp, x2: xp, y1: 20, y2: 160, stroke: "var(--s1)", "stroke-dasharray": "4 3" }, cv);
    el("polyline", { points: ts.map((t, i) => `${(40 + 400 * i / (M - 1)).toFixed(1)},${Y(Ev[i]).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, cv);
    el("circle", { cx: ux(th), cy: Y(E(th, b, s, phi)), r: 6, fill: "var(--s2)", stroke: "var(--surface)", "stroke-width": 2 }, cv);
    el("text", { x: 40, y: 12, class: "svgtext" }, cv).textContent = `energy along the circle at β = ${b.toFixed(2)} (lower is better)`;
    el("text", { x: 440, y: 196, "text-anchor": "end", class: "svgtext" }, cv).textContent = "position on the circle";
    // soft assignment bars at the current beta
    bars.innerHTML = ""; const pr = resp(th, b), kmax = pr.indexOf(Math.max(...pr));
    pr.forEach((v, k) => { const x = 20 + k * 50; el("rect", { x, y: 52 - 44 * v, width: 32, height: Math.max(1, 44 * v), rx: 2, fill: k === kmax ? "var(--s2)" : "var(--muted)" }, bars); el("text", { x: x + 16, y: 68, "text-anchor": "middle", class: "svgtext" }, bars).textContent = k + 1; });
    el("text", { x: 0, y: 8, class: "svgtext" }, bars).textContent = "soft assignment";
    // readout
    let minima = 0; const G = Array.from({ length: 720 }, (_, i) => dE(i / 720 * TAU, b, s, phi)); for (let i = 0; i < 720; i++) if (G[i] < 0 && G[(i + 1) % 720] >= 0) minima++;
    const c = Math.max(...resp(th, bF));
    $("cRead").innerHTML = `<dt>β now</dt><dd>${b.toFixed(2)}</dd><dt>wells in the energy</dt><dd>${minima}</dd><dt>nearest digit</dt><dd>${nearest(th) + 1}</dd><dt>commitment (at final β)</dt><dd>${c.toFixed(2)} ${c > 0.9 ? '<span class="st-ok">committed</span>' : ""}</dd>`;
    $("cMsg").innerHTML = msg;
  }
  function descend(anneal) {
    cancelAnimationFrame(anim); th = +$("cStart").value * Math.PI / 180; trail = [th]; msg = ""; let it = 0;
    const one = () => { const { s, phi } = inputs(), bF = betaOf(); betaNow = anneal ? bF * Math.pow(RHO, -(1 - Math.min(it / KA, 1))) : bF; th = wrapPi(th - ETA * dE(th, betaNow, s, phi)); trail.push(th); it++; };
    const finish = () => { betaNow = null; const k = nearest(th), want = nearest(inputs().phi); msg = k === want ? `<span class="st-ok">✓ settled on digit ${k + 1}, the digit the constraints favour.</span>` : `<span class="st-bad">✗ committed to digit ${k + 1}; the constraints favour digit ${want + 1}.</span>`; draw(); };
    if (reduceMotion) { while (it < ITERS) one(); finish(); return; }
    const frame = () => { for (let k = 0; k < 2 && it < ITERS; k++) one(); draw(); if (it < ITERS) anim = requestAnimationFrame(frame); else finish(); };
    frame();
  }
  $("cFix").addEventListener("click", () => descend(false));
  $("cAnn").addEventListener("click", () => descend(true));
  $("cStart").addEventListener("input", () => { cancelAnimationFrame(anim); betaNow = null; th = +$("cStart").value * Math.PI / 180; trail = []; msg = ""; draw(); });
  ["cB", "cPhi", "cS"].forEach((id) => $(id).addEventListener("input", () => { msg = ""; draw(); }));
  th = +$("cStart").value * Math.PI / 180; draw();
})();

// ---------- E3/E4 · a 4×4 Sudoku solved by descending an energy ----------
const SUD = { k0: 1.0, kA: 1.5, bA: 4, kM: 1.0, bM: 12, rho: 20, K: 60, eta: 0.2, iters: 260 };
const PEERS = Array.from({ length: 16 }, (_, t) => Array.from({ length: 16 }, (_, s) => s).filter((s) => s !== t &&
  ((s >> 2) === (t >> 2) || (s & 3) === (t & 3) || ((s >> 3) === (t >> 3) && ((s & 3) >> 1) === ((t & 3) >> 1)))));
const sudBeta = (i, anneal) => anneal ? SUD.bM * Math.pow(SUD.rho, -(1 - Math.min(i / SUD.K, 1))) : SUD.bM;
const sudInit = (r) => { const z = new Float64Array(64); for (let t = 0; t < 16; t++) { let n = 0; for (let d = 0; d < 4; d++) { const v = gauss(r); z[4 * t + d] = v; n += v * v; } n = Math.sqrt(n); for (let d = 0; d < 4; d++) z[4 * t + d] /= n; } return z; };
const cellOf = (z, t) => [z[4 * t], z[4 * t + 1], z[4 * t + 2], z[4 * t + 3]];
// z holds 16 unit vectors in R^4, flattened; returns the energy and its Euclidean gradient
function sudEG(z, clues, bM) {
  let E = 0; const g = new Float64Array(64), l = new Float64Array(8);
  for (let t = 0; t < 16; t++) if (clues[t]) {                       // evidence: ½∠(z_t, e_clue)²
    const k = clues[t] - 1, c = clamp(z[4 * t + k], -1 + 1e-9, 1 - 1e-9), th = Math.acos(c);
    E += SUD.k0 * 0.5 * th * th; g[4 * t + k] -= SUD.k0 * th / Math.sqrt(1 - c * c);
  }
  for (let t = 0; t < 16; t++) {                                     // repulsive attention over row, column and box
    const P = PEERS[t], n = P.length; let m = -Infinity;
    for (let j = 0; j < n; j++) { const s = P[j]; l[j] = SUD.bA * (z[4 * t] * z[4 * s] + z[4 * t + 1] * z[4 * s + 1] + z[4 * t + 2] * z[4 * s + 2] + z[4 * t + 3] * z[4 * s + 3]); if (l[j] > m) m = l[j]; }
    let Z = 0; for (let j = 0; j < n; j++) { l[j] = Math.exp(l[j] - m); Z += l[j]; }
    E += SUD.kA / SUD.bA * (m + Math.log(Z));
    for (let j = 0; j < n; j++) { const s = P[j], a = SUD.kA * l[j] / Z; for (let d = 0; d < 4; d++) { g[4 * t + d] += a * z[4 * s + d]; g[4 * s + d] += a * z[4 * t + d]; } }
  }
  for (let t = 0; t < 16; t++) {                                     // codebook memory with prototypes e_1..e_4
    let m = -Infinity; for (let d = 0; d < 4; d++) { l[d] = bM * z[4 * t + d]; if (l[d] > m) m = l[d]; }
    let Z = 0; for (let d = 0; d < 4; d++) { l[d] = Math.exp(l[d] - m); Z += l[d]; }
    E -= SUD.kM / bM * (m + Math.log(Z)); for (let d = 0; d < 4; d++) g[4 * t + d] -= SUD.kM * l[d] / Z;
  }
  return { E, g };
}
// one Riemannian gradient step z <- Exp_z(-eta grad E), token-wise
function sudStep(z, clues, i, anneal) {
  const { g } = sudEG(z, clues, sudBeta(i, anneal)), out = new Float64Array(64);
  for (let t = 0; t < 16; t++) {
    const o = 4 * t; let gz = 0; for (let d = 0; d < 4; d++) gz += g[o + d] * z[o + d];
    const v = [0, 0, 0, 0]; let n = 0; for (let d = 0; d < 4; d++) { v[d] = -SUD.eta * (g[o + d] - gz * z[o + d]); n += v[d] * v[d]; } n = Math.sqrt(n);
    const c = Math.cos(n), sc = n < 1e-12 ? 1 : Math.sin(n) / n; let q = 0;
    for (let d = 0; d < 4; d++) { out[o + d] = c * z[o + d] + sc * v[d]; q += out[o + d] * out[o + d]; }
    q = Math.sqrt(q); for (let d = 0; d < 4; d++) out[o + d] /= q;
  }
  return out;
}
const sudDecode = (z) => Array.from({ length: 16 }, (_, t) => { let k = 0; for (let d = 1; d < 4; d++) if (z[4 * t + d] > z[4 * t + k]) k = d; return k + 1; });
const sudEnergy = (z, clues) => sudEG(z, clues, SUD.bM).E;
function sudSolve(clues, r, anneal) { let z = sudInit(r); for (let i = 0; i < SUD.iters; i++) z = sudStep(z, clues, i, anneal); return { z, pred: sudDecode(z), E: sudEnergy(z, clues) }; }
const sudParse = (s) => s.split("").map(Number);
const sameBoard = (a, b) => a.every((v, i) => v === b[i]);
function sudViolations(pred) { let v = 0; for (let t = 0; t < 16; t++) for (const s of PEERS[t]) if (s > t && pred[s] === pred[t]) v++; return v; }
function majorityPick(runs) { const cnt = new Map(); runs.forEach((r) => { const k = r.pred.join(""); cnt.set(k, (cnt.get(k) || 0) + 1); }); let bk = null, bc = -1; runs.forEach((r) => { const k = r.pred.join(""); if (cnt.get(k) > bc) { bc = cnt.get(k); bk = k; } }); return runs.find((r) => r.pred.join("") === bk); }
const lowestPick = (runs) => runs.reduce((a, b) => (b.E < a.E ? b : a));

let sudPuzzle = 0;
const sudAnneal = () => $("schA").checked;
(function sudDemo() {
  const svg = $("sud"), plot = $("sudPlot"), C = 78, O = 9;
  let st = null, anim = 0, seed = 7;
  function grid(z, clues, sol, b, done) {
    svg.innerHTML = "";
    for (let t = 0; t < 16; t++) {
      const r = t >> 2, c = t & 3, x = O + c * C, y = O + r * C;
      el("rect", { x, y, width: C, height: C, fill: clues[t] ? "var(--rule)" : "var(--surface)", stroke: "var(--rule)" }, svg);
      const zt = cellOf(z, t), pr = softmax(zt.map((v) => b * v)).p, k = pr.indexOf(Math.max(...pr));
      const commit = Math.max(...softmax(zt.map((v) => SUD.bM * v)).p), committed = commit > 0.9;
      const col = clues[t] ? "var(--ink)" : (done ? (k + 1 === sol[t] ? "var(--good)" : "var(--bad)") : "var(--ink)");
      el("text", { x: x + C / 2, y: y + 40, "text-anchor": "middle", style: `font:600 30px var(--f-ui);fill:${col}`, opacity: clues[t] || committed ? 1 : 0.28 }, svg).textContent = k + 1;
      pr.forEach((v, d) => el("rect", { x: x + 13 + d * 14, y: y + 70 - 18 * v, width: 10, height: Math.max(1, 18 * v), rx: 1.5, fill: d === k ? "var(--s2)" : "var(--muted)", opacity: clues[t] ? 0.5 : 1 }, svg));
    }
    el("rect", { x: O, y: O, width: 4 * C, height: 4 * C, fill: "none", stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
    el("line", { x1: O + 2 * C, x2: O + 2 * C, y1: O, y2: O + 4 * C, stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
    el("line", { y1: O + 2 * C, y2: O + 2 * C, x1: O, x2: O + 4 * C, stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
  }
  function drawPlot(trace, it) {
    plot.innerHTML = "";
    const X = (i) => 40 + 280 * i / SUD.iters;
    el("rect", { x: X(0), y: 8, width: X(SUD.K) - X(0), height: 86, fill: "var(--rule)", opacity: 0.5 }, plot);
    el("text", { x: X(2), y: 20, class: "svgtext" }, plot).textContent = "β rising";
    if (trace.length > 1) {
      const lo = Math.min(...trace), hi = Math.max(...trace), Y = (v) => 90 - 76 * (v - lo) / Math.max(hi - lo, 1e-9);
      el("polyline", { points: trace.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--s2)", "stroke-width": 2 }, plot);
      el("text", { x: 36, y: Y(hi) + 4, "text-anchor": "end", class: "svgtext" }, plot).textContent = hi.toFixed(1);
      el("text", { x: 36, y: Y(lo) + 4, "text-anchor": "end", class: "svgtext" }, plot).textContent = lo.toFixed(1);
    }
    el("line", { x1: 40, x2: 320, y1: 94, y2: 94, stroke: "var(--rule)" }, plot);
    el("text", { x: 40, y: 112, class: "svgtext" }, plot).textContent = `energy E over ${SUD.iters} iterations (iteration ${it})`;
  }
  function readout() {
    if (!st) return;
    const pred = sudDecode(st.z), done = st.i >= SUD.iters, b = sudBeta(st.i, st.anneal);
    let committed = 0; for (let t = 0; t < 16; t++) if (Math.max(...softmax(cellOf(st.z, t).map((v) => SUD.bM * v)).p) > 0.9) committed++;
    $("sudRead").innerHTML = `<dt>puzzle</dt><dd>${sudPuzzle + 1} of ${VD.puzzles.length}</dd><dt>iteration</dt><dd>${st.i}</dd><dt>β now</dt><dd>${b.toFixed(2)}</dd><dt>energy E</dt><dd>${st.trace.length ? st.trace[st.trace.length - 1].toFixed(2) : "–"}</dd><dt>committed cells</dt><dd>${committed} / 16</dd><dt>repeated digits in a unit</dt><dd>${sudViolations(pred)}</dd>`;
    $("sudMsg").innerHTML = done ? (sameBoard(pred, st.sol) ? '<span class="st-ok">✓ halted on the solution.</span>' : '<span class="st-bad">✗ halted on a wrong board: a local minimum. Try another start, or the restarts below.</span>') : "";
    grid(st.z, st.clues, st.sol, b, done); drawPlot(st.trace, st.i);
  }
  function load() {
    cancelAnimationFrame(anim);
    const [c, s] = VD.puzzles[sudPuzzle], clues = sudParse(c), sol = sudParse(s);
    const z0 = new Float64Array(64); clues.forEach((v, t) => { for (let d = 0; d < 4; d++) z0[4 * t + d] = v ? (d === v - 1 ? 1 : 0) : 0.5; });
    st = { clues, sol, z: z0, i: 0, trace: [], anneal: sudAnneal() };
    grid(st.z, clues, sol, SUD.bM, false); drawPlot([], 0);
    $("sudRead").innerHTML = `<dt>puzzle</dt><dd>${sudPuzzle + 1} of ${VD.puzzles.length}</dd><dt>clues</dt><dd>${clues.filter(Boolean).length}</dd>`;
    $("sudMsg").textContent = "Press “Solve from a random start”.";
  }
  function solve() {
    cancelAnimationFrame(anim);
    const [c, s] = VD.puzzles[sudPuzzle], r = mulberry32(seed++);
    st = { clues: sudParse(c), sol: sudParse(s), z: null, i: 0, trace: [], anneal: sudAnneal() };
    st.z = sudInit(r); st.trace.push(sudEnergy(st.z, st.clues));
    const one = () => { st.z = sudStep(st.z, st.clues, st.i, st.anneal); st.i++; st.trace.push(sudEnergy(st.z, st.clues)); };
    if (reduceMotion) { while (st.i < SUD.iters) one(); readout(); return; }
    const frame = () => { for (let k = 0; k < 3 && st.i < SUD.iters; k++) one(); readout(); if (st.i < SUD.iters) anim = requestAnimationFrame(frame); };
    frame();
  }
  $("sudRun").addEventListener("click", solve);
  $("sudNew").addEventListener("click", () => { sudPuzzle = (sudPuzzle + 1) % VD.puzzles.length; load(); document.dispatchEvent(new Event("sudpuzzle")); });
  ["schA", "schF"].forEach((id) => $(id).addEventListener("change", () => { if (st && st.i > 0) solve(); }));
  load();
})();

(function sudRestarts() {
  const svg = $("sudE"); let seed = 1000, runs = null;
  function draw() {
    svg.innerHTML = "";
    if (!runs) { el("text", { x: 230, y: 85, "text-anchor": "middle", class: "svgtext" }, svg).textContent = "Press “Run B restarts on this puzzle”."; return; }
    const Es = runs.map((r) => r.E), lo = Math.min(...Es), hi = Math.max(...Es), pad = 0.08 * Math.max(hi - lo, 0.5);
    const X = (e) => 30 + 400 * (e - (lo - pad)) / ((hi + pad) - (lo - pad));
    el("line", { x1: 30, x2: 430, y1: 120, y2: 120, stroke: "var(--rule)" }, svg);
    [lo, hi].forEach((v) => el("text", { x: X(v), y: 138, "text-anchor": "middle", class: "svgtext" }, svg).textContent = v.toFixed(1));
    el("text", { x: 230, y: 160, "text-anchor": "middle", class: "svgtext" }, svg).textContent = "final energy E of each restart (lower is better)";
    const sorted = runs.map((r, i) => [r, i]).sort((a, b) => a[0].E - b[0].E), occ = [];
    const best = lowestPick(runs);
    sorted.forEach(([r]) => {
      const x = X(r.E); let lvl = 0; while (occ.some(([ox, ol]) => ol === lvl && Math.abs(ox - x) < 13)) lvl++; occ.push([x, lvl]);
      const y = 106 - 15 * lvl;
      const c = el("circle", { cx: x, cy: y, r: 6, fill: r.ok ? "var(--good)" : "var(--bad)", stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
      c.addEventListener("mousemove", (e) => showTip(e, `E = ${r.E.toFixed(2)} · ${r.ok ? "solution" : `wrong board, ${sudViolations(r.pred)} repeated digits`}`)); c.addEventListener("mouseleave", hideTip);
      if (r === best) { el("circle", { cx: x, cy: y, r: 11, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, svg); el("text", { x: x + 14, y: y - 10, class: "svgtext" }, svg).textContent = "lowest energy"; }
    });
    const nOk = runs.filter((r) => r.ok).length, maj = majorityPick(runs);
    $("sudERead").innerHTML = `<dt>restarts that found the solution</dt><dd>${nOk} / ${runs.length}</dd><dt>lowest energy picks</dt><dd>${best.ok ? '<span class="st-ok">✓ solution</span>' : '<span class="st-bad">✗ wrong board</span>'}</dd><dt>majority vote picks</dt><dd>${maj.ok ? '<span class="st-ok">✓ solution</span>' : '<span class="st-bad">✗ wrong board</span>'}</dd>`;
  }
  function go() {
    const B = +$("sB").value, [c, s] = VD.puzzles[sudPuzzle], clues = sudParse(c), sol = sudParse(s), r = mulberry32(seed++), an = sudAnneal();
    runs = Array.from({ length: B }, () => { const o = sudSolve(clues, r, an); o.ok = sameBoard(o.pred, sol); return o; });
    draw();
  }
  $("sB").addEventListener("input", () => { $("sBv").textContent = $("sB").value; });
  $("sudRestart").addEventListener("click", go);
  document.addEventListener("sudpuzzle", () => { runs = null; $("sudERead").innerHTML = ""; draw(); });
  $("sBv").textContent = $("sB").value; draw();
})();

(function sudBatchDemo() {
  const svg = $("sudBatch"), Bs = [1, 2, 4, 8, 16], NP = 24, NR = 16; let seedBase = 5000;
  function draw(res) {
    svg.innerHTML = "";
    const X = (i) => 60 + 330 * i / (Bs.length - 1), Y = (v) => 222 - 196 * v;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el("line", { x1: 60, x2: 390, y1: Y(v), y2: Y(v), stroke: "var(--rule)" }, svg); el("text", { x: 54, y: Y(v) + 4, "text-anchor": "end", class: "svgtext" }, svg).textContent = Math.round(v * 100) + "%"; });
    Bs.forEach((B, i) => el("text", { x: X(i), y: 242, "text-anchor": "middle", class: "svgtext" }, svg).textContent = "B = " + B);
    el("text", { x: 60, y: 14, class: "svgtext" }, svg).textContent = "puzzles solved by the chosen restart";
    if (!res) return;
    const series = [[res.A.oracle, "ink-2", "2 3", "oracle"], [res.A.majority, "s2", "6 4", "majority"], [res.A.energy, "s2", null, "lowest energy"], [res.F.energy, "s1", null, "fixed β, lowest energy"]];
    series.forEach(([v, col, dash, name]) => {
      el("polyline", { points: v.map((y, i) => `${X(i)},${Y(y)}`).join(" "), fill: "none", stroke: `var(--${col})`, "stroke-width": 2, "stroke-dasharray": dash || "none" }, svg);
      v.forEach((y, i) => { el("circle", { cx: X(i), cy: Y(y), r: 3.5, fill: `var(--${col})` }, svg); const h = el("circle", { cx: X(i), cy: Y(y), r: 10, fill: "transparent" }, svg); h.addEventListener("mousemove", (e) => showTip(e, `${name}, B = ${Bs[i]}: ${(y * 100).toFixed(0)}%`)); h.addEventListener("mouseleave", hideTip); });
    });
    $("sudBatchRead").innerHTML = `<dt>one restart, annealed β</dt><dd>${(res.A.single * 100).toFixed(0)}%</dd><dt>one restart, fixed β</dt><dd>${(res.F.single * 100).toFixed(0)}%</dd><dt>B = 16, lowest energy, annealed</dt><dd>${(res.A.energy[4] * 100).toFixed(0)}%</dd><dt>B = 16, oracle, annealed</dt><dd>${(res.A.oracle[4] * 100).toFixed(0)}%</dd><dt>B = 16, lowest energy, fixed β</dt><dd>${(res.F.energy[4] * 100).toFixed(0)}%</dd>`;
  }
  function compute() {
    const btn = $("sudBatchBtn"); btn.disabled = true;
    const acc = { A: { energy: Bs.map(() => 0), majority: Bs.map(() => 0), oracle: Bs.map(() => 0), single: 0 }, F: { energy: Bs.map(() => 0), majority: Bs.map(() => 0), oracle: Bs.map(() => 0), single: 0 } };
    let p = 0; const base = seedBase; seedBase += 1000;
    const work = () => {
      const [c, s] = VD.puzzles[p], clues = sudParse(c), sol = sudParse(s);
      [["A", true], ["F", false]].forEach(([key, an]) => {
        const r = mulberry32(base + p); const runs = Array.from({ length: NR }, () => { const o = sudSolve(clues, r, an); o.ok = sameBoard(o.pred, sol); return o; });
        acc[key].single += runs.filter((o) => o.ok).length / NR / NP;
        Bs.forEach((B, i) => { const sub = runs.slice(0, B); acc[key].energy[i] += lowestPick(sub).ok / NP; acc[key].majority[i] += majorityPick(sub).ok / NP; acc[key].oracle[i] += sub.some((o) => o.ok) / NP; });
      });
      p++; btn.textContent = `Computing… ${p} / ${NP}`;
      if (p < NP) setTimeout(work, 0); else { draw(acc); btn.textContent = "Recompute with new random starts"; btn.disabled = false; }
    };
    draw(null); setTimeout(work, 30);
  }
  $("sudBatchBtn").addEventListener("click", compute);
  compute();
})();

// ---------- E5 · measured spectra and restart curves of the trained models ----------
(function realEigDemo() {
  const svg = $("realEig"), S = 80, XL = 2.05, YL = 1.8;
  el("line", { x1: -168, x2: 168, y1: 0, y2: 0, stroke: "var(--rule)" }, svg); el("line", { x1: 0, x2: 0, y1: -148, y2: 148, stroke: "var(--rule)" }, svg);
  el("circle", { cx: 0, cy: 0, r: S, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.5 }, svg);
  el("circle", { cx: S, cy: 0, r: 3, fill: "var(--ink)" }, svg); el("text", { x: S + 5, y: 14, class: "svgtext" }, svg).textContent = "1";
  el("text", { x: 164, y: -6, "text-anchor": "end", class: "svgtext" }, svg).textContent = "Re"; el("text", { x: 6, y: -138, class: "svgtext" }, svg).textContent = "Im";
  const G = el("g", {}, svg);
  const models = [["SFPRM", VD.eig_sfprm, "s1"], ["Energy model", VD.eig_energy, "s2"]];
  function stats(E, eta) { let rho = 0, cpx = 0, win = Infinity, off = 0; for (const [a, b] of E) { const da = 1 + eta * (a - 1), db = eta * b; rho = Math.max(rho, Math.hypot(da, db)); if (Math.abs(da) > XL || Math.abs(db) > YL) off++; if (Math.abs(b) > 1e-6) cpx++; const dz2 = (a - 1) ** 2 + b * b; if (a < 1 && dz2 > 1e-12) win = Math.min(win, 2 * (1 - a) / dz2); } return { rho, cpx: cpx / E.length, win, off }; }
  const bestEta = models.map(([, E]) => { const w = stats(E, 1).win; let best = [Infinity, 0]; for (let k = 1; k < 200; k++) { const et = w * k / 200, r = stats(E, et).rho; if (r < best[0]) best = [r, et]; } return best; });
  const iters = (r) => r < 1 ? Math.ceil(Math.log(1e-3) / Math.log(r)) : Infinity;
  function draw() {
    const eta = +$("rE").value / 100; $("rEv").textContent = eta.toFixed(2);
    G.innerHTML = "";
    models.forEach(([, E, col]) => { for (const [a, b] of E) { const da = 1 + eta * (a - 1), db = eta * b; if (Math.abs(da) <= XL && Math.abs(db) <= YL) el("circle", { cx: (S * da).toFixed(1), cy: (-S * db).toFixed(1), r: 1.8, fill: `var(--${col})`, opacity: 0.55 }, G); } });
    let html = "";
    models.forEach(([name, E], m) => {
      const st = stats(E, eta), it = iters(st.rho), [bR, bE] = bestEta[m];
      html += `<dt style="grid-column:1/-1;color:var(--ink);font-weight:600;margin-top:4px">${name}</dt>` +
        `<dt>complex eigenvalues</dt><dd>${(st.cpx * 100).toFixed(1)}%</dd>` +
        `<dt>stable for η &lt;</dt><dd>${st.win.toFixed(2)}</dd>` +
        `<dt>slowest mode ÷1000 at this η</dt><dd style="color:var(${st.rho < 1 ? "--good" : "--bad"})">${Number.isFinite(it) ? it + " steps" : "diverges"}</dd>` +
        `<dt>at its best η = ${bE.toFixed(2)}</dt><dd>${iters(bR)} steps</dd>` +
        (st.off ? `<dt>off the chart</dt><dd>${st.off}</dd>` : "");
    });
    $("realEigRead").innerHTML = html;
  }
  $("rE").addEventListener("input", draw); draw();
})();

(function realBDemo() {
  const svg = $("realB"), Bs = VD.Bs;
  const series = [
    ["sfprm", "anchor", "s1", "1.5", "3 3", "SFPRM, anchor angle"],
    ["sfprm", "majority", "s1", "2", "7 4", "SFPRM, majority vote"],
    ["sfprm", "oracle", "s1", "1.5", "1.5 3", "SFPRM, oracle"],
    ["first", "energy", "muted", "2", null, "first energy model, lowest energy"],
    ["code", "majority", "s2", "2", "7 4", "energy model, majority vote"],
    ["code", "oracle", "s2", "1.5", "1.5 3", "energy model, oracle"],
    ["code", "energy", "s2", "3", null, "energy model, lowest energy"],
  ];
  $("realBLegend").innerHTML = series.map(([, , col, , dash, name]) => `<span><span class="sw${dash ? " dash" : ""}" style="${dash ? `color:var(--${col})` : `background:var(--${col})`}"></span>${name}</span>`).join("");
  function draw() {
    const size = $("r4").checked ? "4" : "6", D = { sfprm: VD[`r${size}_sfprm`], code: VD[`r${size}_code`], first: VD[`r${size}_first`] };
    svg.innerHTML = "";
    const X = (i) => 56 + 380 * i / (Bs.length - 1), Y = (v) => 232 - 2.06 * v;
    [0, 25, 50, 75, 100].forEach((v) => { el("line", { x1: 56, x2: 436, y1: Y(v), y2: Y(v), stroke: "var(--rule)" }, svg); el("text", { x: 50, y: Y(v) + 4, "text-anchor": "end", class: "svgtext" }, svg).textContent = v + "%"; });
    Bs.forEach((B, i) => el("text", { x: X(i), y: 252, "text-anchor": "middle", class: "svgtext" }, svg).textContent = B);
    el("text", { x: 246, y: 268, "text-anchor": "middle", class: "svgtext" }, svg).textContent = "restarts B";
    el("text", { x: 56, y: 14, class: "svgtext" }, svg).textContent = `out-of-distribution board accuracy, ${size}×${size}`;
    series.forEach(([m, rule, col, w, dash, name]) => {
      const v = D[m] && D[m][rule]; if (!v) return;
      el("polyline", { points: v.map((y, i) => `${X(i)},${Y(y)}`).join(" "), fill: "none", stroke: `var(--${col})`, "stroke-width": w, "stroke-dasharray": dash || "none" }, svg);
      v.forEach((y, i) => { const h = el("circle", { cx: X(i), cy: Y(y), r: 8, fill: "transparent" }, svg); h.addEventListener("mousemove", (e) => showTip(e, `${name}, B = ${Bs[i]}: ${y.toFixed(1)}%`)); h.addEventListener("mouseleave", hideTip); });
    });
    const last = Bs.length - 1, sBest = Math.max(D.sfprm.anchor[last], D.sfprm.majority[last]);
    $("realBRead").innerHTML = `<dt>energy model, one restart</dt><dd>${D.code.energy[0].toFixed(1)}%</dd><dt>energy model, lowest energy of 64</dt><dd>${D.code.energy[last].toFixed(1)}%</dd><dt>energy model, oracle of 64</dt><dd>${D.code.oracle[last].toFixed(1)}%</dd><dt>SFPRM, one restart</dt><dd>${D.sfprm.majority[0].toFixed(1)}%</dd><dt>SFPRM, best rule at 64</dt><dd>${sBest.toFixed(1)}%</dd>`;
  }
  ["r4", "r6"].forEach((id) => $(id).addEventListener("change", draw)); draw();
})();
