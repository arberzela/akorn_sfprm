const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push("jsdomError: " + (e.detail ? e.detail.stack || e.detail : e.message)));
vc.on("error", (e) => errors.push("console.error: " + e));
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  beforeParse(window) {
    window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
    window.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: (t, k) => (k === "createImageData" ? () => ({ data: new Uint8ClampedArray(4 * 150 * 150) }) : k === "measureText" ? () => ({ width: 10 }) : () => {}), set: () => true });
    window.requestAnimationFrame = (f) => setTimeout(() => f(Date.now()), 0);
    window.cancelAnimationFrame = (id) => clearTimeout(id);
  } });
const w = dom.window, d = w.document;
const fire = (id, type) => { const e = d.getElementById(id); if (!e) { errors.push("missing #" + id); return; } e.dispatchEvent(new w.Event(type, { bubbles: true })); };
setTimeout(() => {
  // exercise every control of the new section
  for (const [id, vals] of [["rOm", [0, 80, 150]], ["rKap", [1, 20, 40]], ["rEta", [2, 100, 200]], ["cB", [0, 50, 100]], ["cPhi", [0, 359]], ["cS", [0, 100]], ["cStart", [10, 300]], ["rE", [1, 100]], ["sB", [1, 32]]]) for (const v of vals) { d.getElementById(id).value = v; fire(id, "input"); }
  d.getElementById("cB").value = 91; d.getElementById("cS").value = 35; d.getElementById("cPhi").value = 200; d.getElementById("cStart").value = 60; fire("cStart", "input");
  fire("cFix", "click"); fire("cAnn", "click");
  fire("sudRun", "click"); fire("sudNew", "click"); fire("sudRun", "click");
  d.getElementById("schF").checked = true; fire("schF", "change"); d.getElementById("schA").checked = true; fire("schA", "change");
  fire("sudRestart", "click");
  d.getElementById("r6").checked = true; fire("r6", "change"); fire("rE1", "click"); fire("rE01", "click");
}, 50);
setTimeout(() => {
  const txt = (id) => (d.getElementById(id) || {}).textContent;
  console.log("rotRead:", txt("rotRead").replace(/\s+/g, " ").slice(0, 200));
  console.log("cRead:", txt("cRead").replace(/\s+/g, " "), "| cMsg:", txt("cMsg"));
  console.log("sudRead:", txt("sudRead").replace(/\s+/g, " "), "| sudMsg:", txt("sudMsg"));
  console.log("sudERead:", txt("sudERead").replace(/\s+/g, " "));
  console.log("sudBatchRead:", txt("sudBatchRead").replace(/\s+/g, " "), "| btn:", txt("sudBatchBtn"));
  console.log("realEigRead:", txt("realEigRead").replace(/\s+/g, " "));
  console.log("realBRead:", txt("realBRead").replace(/\s+/g, " "));
  for (const id of ["rotPlane", "rotEig", "rotErr", "cellDial", "cellCurve", "cellBars", "sud", "sudPlot", "sudE", "sudBatch", "realEig", "realB"]) console.log(id, "children:", d.getElementById(id).querySelectorAll("*").length);
  const nan = [...d.querySelectorAll("#energy svg *")].filter((e) => [...e.attributes].some((a) => /NaN|undefined|Infinity/.test(a.value))).length;
  console.log("SVG attributes with NaN/undefined/Infinity:", nan);
  console.log(errors.length ? "ERRORS:\n" + errors.join("\n") : "no script errors");
  process.exit(0);
}, 25000);
