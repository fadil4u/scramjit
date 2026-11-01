// Run a Python file in Pyodide under Node, with this checkout's scramjit and a one-file cimo.
//   npm i pyodide@0.27.7 && node tools/run_pyodide.mjs tests/test_math_cases.py
// cimo comes from docs/py/cimo.py (tools/sync_site.py refreshes it), or the file in CIMO_PY.
// MOUNT=<dir> makes a host directory available at /mnt.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const repo = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const { loadPyodide } = createRequire(process.cwd() + '/')('pyodide');
const py = await loadPyodide();
await py.loadPackage('numpy', { messageCallback: () => {} });
py.FS.mkdirTree('/w/scramjit');
for (const f of fs.readdirSync(repo + '/scramjit')) if (f.endsWith('.py')) py.FS.writeFile('/w/scramjit/' + f, fs.readFileSync(repo + '/scramjit/' + f));
py.FS.writeFile('/w/cimo.py', fs.readFileSync(process.env.CIMO_PY || repo + '/docs/py/cimo.py'));
if (process.env.MOUNT) py.FS.mkdirTree('/mnt'), py.FS.mount(py.FS.filesystems.NODEFS, { root: process.env.MOUNT }, '/mnt');
const script = fs.readFileSync(process.argv[2], 'utf8');
py.FS.writeFile('/w/' + path.basename(process.argv[2]), script);
py.runPython(`import sys, runpy; sys.path.insert(0, '/w'); runpy.run_path('/w/${path.basename(process.argv[2])}', run_name='__main__')`);
