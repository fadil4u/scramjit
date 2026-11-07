// pytest in Pyodide under Node, with this checkout's scramjit and tests and a one-file cimo:
//   npm i pyodide@0.27.7 && node tools/run_pyodide_tests.mjs tests/test_glists.py [-k expr]
// cimo comes from docs/py/cimo.py (tools/sync_site.py refreshes it), or the file in CIMO_PY. Tests that need what
// Pyodide lacks (resource, subprocess) skip themselves.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const repo = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const { loadPyodide } = createRequire(process.cwd() + '/')('pyodide');
const py = await loadPyodide();
await py.loadPackage(['numpy', 'pytest'], { messageCallback: () => {} });
py.FS.mkdirTree('/w/scramjit');
for (const f of fs.readdirSync(repo + '/scramjit')) if (f.endsWith('.py')) py.FS.writeFile('/w/scramjit/' + f, fs.readFileSync(repo + '/scramjit/' + f));
py.FS.writeFile('/w/cimo.py', fs.readFileSync(process.env.CIMO_PY || repo + '/docs/py/cimo.py'));
py.FS.mkdirTree('/w/tests');
for (const f of fs.readdirSync(repo + '/tests')) if (f.endsWith('.py')) py.FS.writeFile('/w/tests/' + f, fs.readFileSync(repo + '/tests/' + f));
const args = JSON.stringify(['-q', '-p', 'no:cacheprovider', ...process.argv.slice(2).map(a => a.startsWith('tests/') ? '/w/' + a : a)]);
py.runPython(`import sys, pytest; sys.path.insert(0, '/w'); sys.path.insert(0, '/w/tests'); r = pytest.main(${args}); print('EXIT', r)`);
