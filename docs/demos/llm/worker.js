// Python, numpy, the compiler and the model all run in workers, so the page never blocks. One worker holds the
// weights and writes the story. A second, with no weights, compiles edits and hands over the compiled files.
'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SJ = ['__init__.py', '_compiler.py', '_dispatch.py', '_errors.py', '_interrupt.py', '_lift.py', '_mathlib.py', '_preflight.py', '_runtime.py', '_types.py', '_webs.py'];
const WEIGHTS = 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin';
const TOKENIZER = 'https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin';
const CACHE = 'scramjit-llm-1';
let py, demo;
const text = async p => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.text(); };
const say = msg => postMessage({ type: 'status', msg });
const breathe = () => new Promise(r => setTimeout(r));
const jsonOf = s => JSON.parse(s);

async function opened() { try { return await caches.open(CACHE); } catch { return null; } }

// A file from the browser's cache, or downloaded with progress and kept for the next visit.
async function download(url, label) {
  const cache = await opened();
  const kept = cache && await cache.match(url).catch(() => null);
  if (kept) return new Uint8Array(await kept.arrayBuffer());
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${label}: download failed (${r.status})`);
  const total = +r.headers.get('content-length') || 0, chunks = [], reader = r.body.getReader();
  let got = 0, shown = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (got - shown > 2e6) { shown = got; say(`Downloading ${label}: ${Math.round(got / 1e6)}${total ? ' of ' + Math.round(total / 1e6) : ''} MB`); postMessage({ type: 'alive' }); }
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  if (cache) await cache.put(url, new Response(out.slice())).catch(() => {});
  return out;
}

async function python() {
  say('Loading Python');
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  say('Loading numpy');
  await py.loadPackage('numpy');
  postMessage({ type: 'alive' });
  say('Loading the compiler');
  py.FS.mkdirTree('/home/pyodide/scramjit');
  py.FS.writeFile('/home/pyodide/cimo.py', await text('../../py/cimo.py'));
  for (const f of SJ) py.FS.writeFile('/home/pyodide/scramjit/' + f, await text('../../py/scramjit/' + f));
  py.FS.writeFile('/home/pyodide/lab.py', await text('../../lab/lab.py'));
  py.FS.writeFile('/home/pyodide/wat.py', await text('../../lab/wat.py'));
  py.FS.writeFile('/home/pyodide/demo_py.py', await text('demo_py.py'));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  await py.runPythonAsync('import demo_py');
  demo = py.pyimport('demo_py');
}

async function boot() {
  const t = performance.now();
  await python();
  const ready = (performance.now() - t) / 1000;
  py.FS.writeFile('/home/pyodide/tokenizer.bin', await download(TOKENIZER, 'the tokenizer'));
  py.FS.writeFile('/home/pyodide/weights.bin', await download(WEIGHTS, 'the model'));
  say('Loading the weights');
  await breathe();
  const info = jsonOf(demo.setup('/home/pyodide/weights.bin', '/home/pyodide/tokenizer.bin'));
  postMessage({ type: 'ready', python: ready, seconds: (performance.now() - t) / 1000, robot: demo.token_of('robot'), ...info });
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'boot') await boot();
    if (m.type === 'bootCompiler') { const t = performance.now(); await python(); postMessage({ type: 'compilerReady', seconds: (performance.now() - t) / 1000 }); }
    if (m.type === 'precompile') {
      await breathe();
      const r = jsonOf(demo.precompile(m.code));
      const files = (r.files || []).map(path => ({ path, data: py.FS.readFile(path) }));
      postMessage({ type: 'precompiled', id: m.id, seconds: r.seconds, error: r.error, files }, files.map(f => f.data.buffer));
    }
    if (m.type === 'adopt') for (const f of m.files) { py.FS.mkdirTree(f.path.slice(0, f.path.lastIndexOf('/'))); py.FS.writeFile(f.path, f.data); }
    if (m.type === 'compile') { await breathe(); postMessage({ type: 'compiled', id: m.id, ...jsonOf(demo.compile_forward(m.code, m.compiled)) }); }
    if (m.type === 'start') demo.start(m.prompt);
    if (m.type === 'wat') {
      let t;
      try { t = demo.wasm_text(m.code); } catch (err) { t = ';; could not disassemble: ' + err.message; }
      postMessage({ type: 'wat', code: m.code, text: t });
    }
    if (m.type === 'gen') postMessage({ type: 'text', ...jsonOf(demo.step(m.ms)) });
  } catch (err) { postMessage({ type: 'failed', compiler: m.type === 'bootCompiler' || m.type === 'precompile', error: err.message }); }
};
