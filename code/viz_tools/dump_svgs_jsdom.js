const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8"), out = process.argv[3];
const vars = { "--bg": "#f5f6f8", "--surface": "#ffffff", "--ink": "#161a22", "--ink-2": "#4b5263", "--muted": "#7b8292", "--rule": "#dde1e8", "--accent": "#3346a8", "--s1": "#2a78d6", "--s2": "#eb6834", "--s3": "#1baf7a", "--good": "#0ca30c", "--bad": "#d03b3b", "--f-ui": "Helvetica", "--f-math": "Times" };
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: new VirtualConsole(),
  beforeParse(w) { w.matchMedia = () => ({ matches: false, addEventListener() {} }); w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: (t, k) => (k === "createImageData" ? () => ({ data: new Uint8ClampedArray(4 * 150 * 150) }) : k === "measureText" ? () => ({ width: 10 }) : () => {}), set: () => true }); w.requestAnimationFrame = (f) => setTimeout(() => f(0), 0); w.cancelAnimationFrame = (id) => clearTimeout(id); } });
const d = dom.window.document;
const click = (id) => d.getElementById(id).dispatchEvent(new dom.window.Event("click"));
setTimeout(() => { click("cAnn"); click("sudRun"); click("sudRestart"); }, 100);
setTimeout(() => {
  for (const id of ["rotPlane", "rotEig", "rotErr", "cellDial", "cellCurve", "cellBars", "sud", "sudPlot", "sudE", "sudBatch", "realEig", "realB"]) {
    const s = d.getElementById(id); let x = s.outerHTML;
    x = x.replace(/var\((--[a-z0-9-]+)\)/g, (m, v) => vars[v] || "#000");
    x = x.replace(/class="svgtext"/g, 'font-family="Helvetica" font-size="12" fill="#4b5263"').replace(/class="svgmath"/g, 'font-family="Times" font-style="italic" font-size="15" fill="#161a22"');
    x = x.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ');
    const vb = s.getAttribute("viewBox").split(" ").map(Number);
    x = x.replace(/<svg /, `<svg width="${vb[2] * 2}" height="${vb[3] * 2}" style="background:#fff" `);
    fs.writeFileSync(`${out}/${id}.svg`, x);
  }
  process.exit(0);
}, 20000);
