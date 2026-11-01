"""The preflight report, the command line, the disk cache across processes, and the log."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from scramjit import scram

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
CIMO = os.path.join(ROOT, '..', 'cimo', 'python')

SIM = '''
import math
from scramjit import scram

@scram
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]

@scram
def grid(h):
    rows, cols = h.shape
    s = 0.0
    for i in range(rows):
        for j in range(cols):
            s += math.sqrt(h[i, j])
    return s

@scram
def sloppy(a, k: int):
    s = 0.0
    for i in range(3):
        s += a[k]
    return s

@scram(boundscheck=False)
def fast(a, k: int):
    return a[k]

@scram
def summarize(a):
    names = {}
    for i in range(len(a)):
        names[i] = str(a[i])
    return len(names)

def plain():
    return {}
'''
APP = '''
import sys
import numpy as np
import sim
x = np.ones(50)
y = np.zeros(50)
sim.ema(x, y, 0.5)
sim.ema(x, y, 1)
sim.grid(np.ones((4, 4)))
sim.summarize(x)
print('args', sys.argv[1:])
'''


def run(args, cwd, **env):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, CIMO]), PYTHONWARNINGS='ignore', PYTHONIOENCODING='utf-8', **env)
    return subprocess.run([sys.executable, *args], cwd=cwd, env=e, capture_output=True, text=True, timeout=120, encoding='utf-8')


@pytest.fixture
def proj(tmp_path):
    (tmp_path / 'sim.py').write_text(SIM)
    (tmp_path / 'app.py').write_text(APP)
    return tmp_path


def test_static_report(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py'], proj)
    assert r.returncode == 1, r.stderr
    assert '🟢 ema' in r.stdout and '🟢 grid' in r.stdout and '🔴 summarize' in r.stdout
    assert 'the Dict expression' in r.stdout and 'calls str()' in r.stdout
    assert 'plain' not in r.stdout
    assert '4 compiled · 1 Python' in r.stdout


def test_static_imports_nothing(tmp_path):
    (tmp_path / 'boom.py').write_text('raise SystemExit("imported!")\nfrom scramjit import scram\n@scram\ndef f(a: int):\n    return a + 1\n')
    r = run(['-m', 'scramjit', 'preflight', str(tmp_path)], tmp_path)
    assert r.returncode == 0 and '🟢 f' in r.stdout, r.stdout + r.stderr


def test_exit_zero_when_all_compiled(tmp_path):
    (tmp_path / 'ok.py').write_text('from scramjit import scram\n@scram\ndef f(a: int):\n    return a + 1\n')
    assert run(['-m', 'scramjit', 'preflight', 'ok.py'], tmp_path).returncode == 0
    assert run(['-m', 'scramjit', 'preflight', '--allow-partial', 'ok.py'], tmp_path).returncode == 0


def test_bad_usage(proj):
    assert run(['-m', 'scramjit'], proj).returncode == 2
    assert run(['-m', 'scramjit', 'preflight', 'missing.py'], proj).returncode == 2
    assert run(['-m', 'scramjit', 'preflight', 'sim.py:nothing'], proj).returncode == 2


def test_one_function_bounds(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:ema'], proj)
    assert r.returncode == 0
    lines = {ln.split()[0]: ln for ln in r.stdout.splitlines() if ln[:4].strip().isdigit()}
    # y[0] is checked; the loop's accesses are versioned: one test before it, no check inside
    assert '2 bounds checks kept' in next(v for v in lines.values() if 'y[0] = x[0]' in v)
    assert 'bounds hoisted out of the loop' in next(v for v in lines.values() if 'y[i - 1]' in v)
    assert 'hint' not in r.stdout


def test_kept_check_in_a_loop_gets_a_hint(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:sloppy'], proj)
    assert '1 bounds check kept' in r.stdout and 'runs on every iteration' in r.stdout


def test_checks_off(proj):
    assert 'bounds checks off' in run(['-m', 'scramjit', 'preflight', 'sim.py:fast'], proj).stdout


def test_one_function_python(proj):
    r = run(['-m', 'scramjit', 'preflight', 'sim.py:summarize'], proj)
    assert r.returncode == 1 and '🔴' in r.stdout and 'calls str()' in r.stdout


def test_run_report(proj):
    r = run(['-m', 'scramjit', 'preflight', '--run', 'app.py', 'one', 'two'], proj)
    assert "args ['one', 'two']" in r.stdout, r.stderr
    assert '% native' in r.stdout and '🟢 ema ×2' in r.stdout and '🔴 summarize' in r.stdout
    assert 'a arrives as float and int' in r.stdout
    assert 'failed: summarize is not compiled' in r.stdout
    assert r.returncode == 1


def test_run_module(proj):
    (proj / 'okmod.py').write_text('import numpy as np\nimport sim\nsim.grid(np.ones((2, 2)))\n')
    r = run(['-m', 'scramjit', 'preflight', '--run', '-m', 'okmod'], proj)
    assert r.returncode == 0 and '🟢 grid' in r.stdout, r.stdout + r.stderr


def test_run_passes_on_the_programs_failure(proj):
    (proj / 'bad.py').write_text('import sys\nsys.exit(7)\n')
    assert run(['-m', 'scramjit', 'preflight', '--run', 'bad.py'], proj).returncode == 7


def test_method_on_a_function(capsys):
    f = scram(_probe)
    f.preflight()
    assert 'assumed signature' in capsys.readouterr().out
    f(np.ones(3))
    f.preflight()
    out = capsys.readouterr().out
    assert 'compiled in' in out and 'types' in out
    f.preflight(f.signatures[0])
    assert 'compiled in' in capsys.readouterr().out
    assert 'int sj_kernel' in f.c()


def _probe(a):
    return a[0]


def test_log(proj):
    r = run(['app.py'], proj, SCRAMJIT_LOG='1')
    assert 'scramjit: compiled grid(float64[:, :]) in' in r.stderr
    assert 'scramjit: summarize(float64[:]) runs as Python: line' in r.stderr
    assert 'scramjit:' not in run(['app.py'], proj).stderr


def _images(d):
    return [f for f in os.listdir(d)] if os.path.isdir(d) else []


def test_a_second_process_reuses_the_disk_cache(tmp_path):
    (tmp_path / 'p.py').write_text(textwrap.dedent('''
        import sys, time
        import numpy as np
        from scramjit import scram
        @scram(fallback=False)
        def f(a, n: int):
            s = 0.0
            for k in range(n):
                for i in range(len(a)):
                    s += a[i] * %d
            return s
        t = time.perf_counter()
        print(f(np.ones(4), 2))
        print(f'{time.perf_counter() - t:.3f}', file=sys.stderr)
    ''' % (os.getpid() % 1000 + 3)))
    cc = tmp_path / 'cc'
    first = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert first.returncode == 0, first.stderr
    made = _images(cc)
    assert made
    before = {f: os.path.getmtime(cc / f) for f in made}
    second = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert second.stdout == first.stdout
    assert {f: os.path.getmtime(cc / f) for f in _images(cc)} == before     # read, not rebuilt


def test_cache_false_writes_nothing(tmp_path):
    (tmp_path / 'p.py').write_text(textwrap.dedent('''
        import numpy as np
        from scramjit import scram
        @scram(fallback=False, cache=False)
        def f(a):
            return a[0] * 91.5
        print(f(np.ones(2)))
    '''))
    cc = tmp_path / 'cc'
    r = run(['p.py'], tmp_path, CIMO_CACHE_DIR=str(cc))
    assert r.returncode == 0 and '91.5' in r.stdout, r.stderr
    assert not [f for f in _images(cc) if f.endswith('.img')]
