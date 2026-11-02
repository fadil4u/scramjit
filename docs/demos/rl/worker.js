// Python, numpy, the compiler and the world all run here, so compiling never blocks the page.
'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SJ = ['__init__.py', '_compiler.py', '_dispatch.py', '_errors.py', '_interrupt.py', '_mathlib.py', '_preflight.py', '_runtime.py', '_types.py'];
let py, demo;
const text = async p => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.text(); };
const say = msg => postMessage({ type: 'status', msg });
const breathe = () => new Promise(r => setTimeout(r));

async function boot() {
  const t = performance.now();
  say('Loading Python');
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  say('Loading numpy');
  await py.loadPackage('numpy');
  say('Loading the compiler');
  py.FS.mkdirTree('/home/pyodide/scramjit');
  py.FS.writeFile('/home/pyodide/cimo.py', await text('../../py/cimo.py'));
  for (const f of SJ) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text('../../py/scramjit/' + f));
  py.FS.writeFile('/home/pyodide/envlab.py', await text('envlab.py'));
  py.FS.writeFile('/home/pyodide/demo_py.py', await text('demo_py.py'));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  await py.runPythonAsync('import demo_py');
  demo = py.pyimport('demo_py');
  postMessage({ type: 'ready', seconds: (performance.now() - t) / 1000 });
}

let ticket = 0;      // a newer load or training run makes an older training run stop

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'boot') await boot();
    if (m.type === 'load') {
      ticket++;
      say('Compiling');
      await breathe();
      postMessage({ type: 'loaded', id: m.id, ...JSON.parse(demo.load(m.code)) });
    }
    if (m.type === 'moves') postMessage({ type: 'moves', id: m.id, gen: m.gen, ...JSON.parse(demo.moves(m.n, m.budget, m.useQ)) });
    if (m.type === 'tally') postMessage({ type: 'tally', id: m.id, ...JSON.parse(demo.tally(m.budget, m.ms)) });
    if (m.type === 'train') {
      const mine = ++ticket;
      say('Compiling the training loop');
      await breathe();
      const s = JSON.parse(demo.train_start());
      postMessage({ type: 'trainstart', id: m.id, ...s });
      if (s.error) return;
      for (let i = 0; i < 10; i++) {
        const c = JSON.parse(demo.train_chunk(i));
        postMessage({ type: 'trainchunk', id: m.id, ...c });
        if (c.error) return;
        await breathe();
        if (mine !== ticket) return;
      }
      demo.train_done();
      postMessage({ type: 'traindone', id: m.id });
    }
  } catch (err) { postMessage({ type: 'failed', id: m.id, error: err.message, stage: 'worker' }); }
};
