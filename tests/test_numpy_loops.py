"""numpy's own loops (_nploops), called where numpy's results aren't the C library's (SVML on AVX-512). Most machines
never need them, so SCRAMJIT_NUMPY_LOOPS=1 makes every compile use them where they exist, in a fresh process, and the
results must still be numpy's, bit for bit."""

import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ONE = ['exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
       'sinh', 'cosh', 'arcsinh', 'arccosh', 'arctanh', 'cbrt']
TWO = ['arctan2', 'hypot', 'fmod']

MAIN = r'''
import struct, sys, warnings
import numpy as np
from scramjit import scram
import loops_mod as m

xs = [0.0, -0.0, 0.5, -0.7, 1.0, 2.5, -3.25, 10.0, 1e-300, 1e300, float('inf'), float('nan')]
bits = lambda v: 'nan' if v != v else struct.pack('<d', float(v))
used = 0
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name in sys.argv[1].split(','):
        f = getattr(m, name)
        g = scram(f, fallback=False, cache=False)
        for x in xs:
            for y in xs:
                assert bits(g(x, y)) == bits(f(x, y)), (name, x, y)
        used += 'sj_np' in g.c()
    p = scram(m.power, fallback=False, cache=False)
    for x, y in ((2.0, 0.5), (7.3, -1.25), (2147483647.0, 2.5)):
        for dt in (np.float64, np.float32):
            a, b = np.array([x], dt), np.array([y], dt)
            assert bits(p(a, b)) == bits(m.power(a, b)), (x, y, dt)
        used += 'sj_np2' in p.c()
print('used', used)
'''


def test_numpys_loops_give_numpys_results(tmp_path):
    funcs = [f'def {n}(a, b):\n    return np.{n}(a{", b" if n in TWO else ""})\n' for n in ONE + TWO]
    funcs.append('def power(a, b):\n    return a[0] ** b[0]\n')
    (tmp_path / 'loops_mod.py').write_text('import numpy as np\n\n\n' + '\n\n'.join(funcs))
    (tmp_path / 'main.py').write_text(MAIN)
    env = dict(os.environ, SCRAMJIT_NUMPY_LOOPS='1', PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, '..', 'cimo', 'python'), str(tmp_path)]))
    r = subprocess.run([sys.executable, 'main.py', ','.join(ONE + TWO)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'used {len(ONE + TWO) + 3}' in r.stdout, r.stdout      # every function through numpy's loop, and each power
