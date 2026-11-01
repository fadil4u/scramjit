"""Copy what the site's live demo loads into Pyodide: cimo as one file, scramjit's package, and
cimo's WebAssembly disassembler (docs/nn/wat.py in cimo).
Run after changing scramjit (or to pick up a newer cimo):  python3 tools/sync_site.py
cimo is taken from the environment, or from ../cimo/python next to this repository.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'docs', 'py')


def main():
    os.makedirs(os.path.join(OUT, 'scramjit'), exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, 'scramjit')):
        if f.endswith('.py'):
            shutil.copy(os.path.join(ROOT, 'scramjit', f), os.path.join(OUT, 'scramjit', f))
    env = dict(os.environ)
    cimo_src = os.path.join(ROOT, '..', 'cimo', 'python')
    if os.path.isdir(cimo_src):
        env['PYTHONPATH'] = cimo_src + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-m', 'cimo', 'onefile', '-o', os.path.join(OUT, 'cimo.py')], check=True, env=env)
    wat = os.path.join(ROOT, '..', 'cimo', 'docs', 'nn', 'wat.py')    # cimo's WebAssembly disassembler
    if os.path.exists(wat):
        shutil.copy(wat, os.path.join(ROOT, 'docs', 'lab', 'wat.py'))
    print('synced', sorted(os.listdir(os.path.join(OUT, 'scramjit'))), 'and cimo.py',
          os.path.getsize(os.path.join(OUT, 'cimo.py')) // 1024, 'KB')


if __name__ == '__main__':
    main()
