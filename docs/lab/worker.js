// The live demo's Python side, in a worker so compiling never blocks the page: Pyodide, numpy,
// cimo, scramjit and lab.py run here. The page sends code and controls; this sends back what each
// stage shows and, one frame at a time, the terrain's heights.
'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SJ_FILES = ['__init__.py', '_compiler.py', '_dispatch.py', '_errors.py', '_runtime.py', '_types.py'];
let py = null, lab = null, compiled = false, inFlight = false, ready = false, active = true, looping = false;

async function text(base, path) {
  const r = await fetch(new URL(path, base), { cache: 'no-cache' });
  if (!r.ok) throw new Error(path + ': ' + r.status);
  return r.text();
}
const status = msg => postMessage({ type: 'status', msg });

async function boot(base) {
  status('Loading Pyodide (CPython in WebAssembly, 7 MB)…');
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  status('Loading numpy…');
  await py.loadPackage('numpy');
  status('Loading cimo and scramjit (pure Python)…');
  py.FS.mkdirTree('/home/pyodide/scramjit');
  py.FS.writeFile('/home/pyodide/cimo.py', await text(base, 'py/cimo.py'));
  for (const f of SJ_FILES) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text(base, 'py/scramjit/' + f));
  py.FS.writeFile('/home/pyodide/lab.py', await text(base, 'lab/lab.py'));
  py.FS.writeFile('/home/pyodide/wat.py', await text(base, 'lab/wat.py'));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  status('Importing…');
  await py.runPythonAsync('import lab');
  lab = py.pyimport('lab');
  ready = true;
  postMessage({ type: 'ready' });
}

// work continuously in time-boxed steps; send the heights (a copy: the page gets its own buffer)
// whenever the page has drawn the last ones. Paused while the demo is out of sight.
function loop() {
  looping = false;
  if (!ready || !active) return;
  const r = JSON.parse(lab.step(compiled ? 22 : 14));
  if (!inFlight || r.error) {
    const hp = lab.heights(), hb = hp.getBuffer('f32');
    const h = new Float32Array(hb.data);
    hb.release(); hp.destroy();
    let vis = null;
    const vo = lab.overlay();
    if (vo) { const vb = vo.getBuffer('u8'); vis = new Uint8Array(vb.data); vb.release(); vo.destroy(); }
    inFlight = true;
    postMessage({ type: 'frame', rate: r.rate, error: r.error, compiled, h, vis }, vis ? [h.buffer, vis.buffer] : [h.buffer]);
  }
  kick();
}
function kick() { if (!looping) { looping = true; setTimeout(loop, 0); } }

onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === 'boot') return await boot(m.base);
    if (!ready) return;
    if (m.type === 'ack') inFlight = false;
    else if (m.type === 'active') { active = m.on; kick(); }
    else if (m.type === 'code') {
      compiled = m.compiled;
      const info = lab.set_code(m.preset, m.src, m.compiled);
      postMessage({ type: 'info', id: m.id, info: JSON.parse(info) });
      kick();
    } else if (m.type === 'reset') lab.reset(m.seed ?? 7);
    else if (m.type === 'wasm') {
      let t;
      try { t = lab.wasm_text(); } catch (err) { t = ';; could not disassemble: ' + err.message; }
      postMessage({ type: 'wat', text: t });
    }
  } catch (err) {
    postMessage({ type: 'fatal', msg: err.message });
  }
};
