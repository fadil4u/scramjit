"""preflight --counts and --live: exact against a Python tracer, results unchanged, and readable mid-run."""

import json
import os
import pty
import select
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from scramjit import _dispatch

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'cimo', 'python')]),
           PYTHONIOENCODING='utf-8')


@pytest.fixture
def counting():
    saved = _dispatch._counts, _dispatch._profile
    _dispatch.enable_counts()
    yield
    _dispatch._counts, _dispatch._profile = saved


def traced(fn, *args):
    """fn's result and how many times Python ran each of its lines."""
    runs = {}
    code = fn.__code__

    def local(frame, event, arg):
        if event == 'line':
            runs[frame.f_lineno] = runs.get(frame.f_lineno, 0) + 1
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code is code else None

    sys.settrace(tracer)
    try:
        return fn(*args), runs
    finally:
        sys.settrace(None)


SOURCES = {
    'branches': '''
def f(x, n: int):
    """A docstring."""
    t = 0
    for i in range(n):
        if x[i % 7] > 0.5:
            t += 1
        elif x[i % 5] > 0.2:
            t += 2
            t += 3
        else:
            t -= 1
        t += 1
    return t
''',
    'jumps': '''
def f(x, n: int):
    t = 0
    for i in range(n):
        if i % 3 == 0:
            continue
        t += 1
        if i > n - 5:
            break
        t += 2
    t += 10
    return t
''',
    'while_else': '''
def f(x, n: int):
    i = 0
    t = 0
    while i < n:
        i += 1
        if i == 1000000:
            break
        t += i
    else:
        t = -t
    for j in range(3):
        t += j
    else:
        t += 1
    return t
''',
    'early_return': '''
def f(x, n: int):
    for i in range(n):
        for j in range(i):
            if j == 20:
                return i * j
            x[j % 7] = x[j % 7] * 0.5
    return -1
''',
    'versioned_loop': '''
def f(x, n: int):
    t = 0.0
    for i in range(1, len(x)):
        x[i] = 0.5 * x[i] + 0.5 * x[i - 1]
        if x[i] > 0.1:
            t += x[i]
    return t
''',
    'one_line': '''
def f(x, n: int):
    a = 1; b = 2
    for i in range(n):
        a += 1; b += a
    return a + b
''',
    'dict_argument': '''
def f(x, n: int):
    d = 0
    for i in range(n):
        if i % 2:
            d += i
    return d
''',
}


@pytest.mark.parametrize('name', sorted(SOURCES))
def test_counts_match_a_tracer(name, counting):
    f = _CountedFromSource(SOURCES[name])
    expected, truth = traced(f.__wrapped__, np.linspace(0, 1, 9), 40)
    got = f(np.linspace(0, 1, 9), 40)
    assert got == expected
    runs, heads = f.line_counts()
    assert runs
    for line, (kind, body) in heads.items():
        if kind == 'if':
            assert runs[line] == truth[line], (line, runs, truth)
    loops = {n for n, (k, _) in heads.items() if k == 'loop'}
    for line, v in truth.items():
        if line not in loops:
            assert runs.get(line) == v, (line, runs, truth)
    for line, v in runs.items():
        if line not in loops:
            assert truth.get(line, 0) == v, (line, runs, truth)


def _CountedFromSource(source):
    """A real @scram function, from a real file, since a compiled function needs its source."""
    import importlib.util
    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'counted_module.py')
    with open(path, 'w') as out:
        out.write('from scramjit import scram\n\n@scram' + textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location('counted_module', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.f


def test_branch_taken_and_loop_iterations(counting):
    f = _CountedFromSource('''
def f(n: int):
    t = 0
    for i in range(n):
        if i % 4 == 0:
            t += 1
    return t
''')
    assert f(100) == 25
    runs, heads = f.line_counts()
    first = min(runs)                            # t = 0, in the file written above
    assert runs[first] == 1
    assert heads[first + 1] == ('loop', 100)
    assert heads[first + 2] == ('if', 25)


def test_each_signature_adds_up(counting):
    f = _CountedFromSource('''
def f(a):
    t = a
    for i in range(10):
        t += i
    return t
''')
    assert f(1) == 46 and f(1.5) == 46.5
    runs, _ = f.line_counts()
    assert max(runs.values()) == 20 and len(f.signatures) == 2


def test_lifted_loops_are_counted(counting):
    f = _CountedFromSource('''
def f(x):
    d = {}
    for i in range(len(x)):
        x[i] = x[i] * 2.0
    d['a'] = 1
    return x[0]
''')
    with pytest.warns(Warning):
        assert f(np.ones(30)) == 2.0
    assert 30 in f.line_counts()[0].values()


def test_an_exception_keeps_the_counts_and_the_error(counting):
    f = _CountedFromSource('''
def f(x):
    for i in range(10):
        x[i] = 1.0
''')
    with pytest.raises(IndexError):
        f(np.zeros(4))
    assert 5 in f.line_counts()[0].values()      # the fifth iteration raised


def test_only_named_functions_are_counted():
    saved = _dispatch._counts, _dispatch._profile
    _dispatch.enable_counts({'wanted'})
    src = """
def f(n: int):
    t = 0
    for i in range(n):
        t += i
    return t
"""
    try:
        other = _CountedFromSource(src)
        other(10)
        assert other.line_counts() == ({}, {})
        wanted = _CountedFromSource(src)
        wanted._owner = 'wanted'
        wanted(10)
        assert 10 in wanted.line_counts()[0].values()
    finally:
        _dispatch._counts, _dispatch._profile = saved


def test_not_counting_leaves_the_code_alone():
    f = _CountedFromSource(SOURCES['jumps'])
    assert f(np.zeros(3), 10) is not None
    assert 'sj_c' not in f.c()
    assert f.line_counts() == ({}, {})


def test_counters_can_be_read_while_a_kernel_runs(counting):
    f = _CountedFromSource('''
def f(n: int):
    t = 0
    for i in range(n):
        t += i % 7
    return t
''')
    f(1)
    seen = []
    done = threading.Event()

    def watch():
        while not done.is_set():
            seen.append(max(f.line_counts()[0].values()))
            time.sleep(0.002)

    w = threading.Thread(target=watch)
    w.start()
    n = 400_000_000
    f(n)
    done.set()
    w.join()
    middle = [v for v in seen if 1 < v < n]
    assert middle, seen[:5]                      # updates arrive from inside one call
    assert middle == sorted(middle)


APP = '''
import numpy as np
from scramjit import scram

@scram
def settle(x, n: int):
    t = 0.0
    for k in range(n):
        for i in range(1, len(x)):
            x[i] = 0.5 * x[i] + 0.5 * x[i - 1]
            if x[i] > 0.25:
                t += 1.0
    return t

x = np.linspace(0, 1, 1000)
print('RESULT', settle(x, %d))
'''


def cli(tmp_path, *args, iters=20):
    (tmp_path / 'app.py').write_text(APP % iters)
    return subprocess.run([sys.executable, '-m', 'scramjit', 'preflight', *args, '--run', 'app.py'], cwd=tmp_path, env=ENV,
                          capture_output=True, text=True, encoding='utf-8', timeout=240)


def test_cli_counts_reports_hottest_lines(tmp_path):
    r = cli(tmp_path, '--counts')
    assert r.returncode == 0, r.stderr
    assert 'hottest lines' in r.stderr and '19,980  app.py:10' in r.stderr


def test_cli_results_do_not_change(tmp_path):
    plain = cli(tmp_path)
    for flags in (['--counts'], ['--live'], ['--counts', '--live']):
        r = cli(tmp_path, *flags)
        assert r.stdout == plain.stdout, flags
        assert r.returncode == plain.returncode


def test_cli_plain_run_is_not_instrumented(tmp_path):
    r = cli(tmp_path)
    assert 'hottest lines' not in r.stderr


def test_cli_one_function_shows_a_column(tmp_path):
    r = cli(tmp_path, 'app.py:settle', '--counts')
    assert r.returncode == 0, r.stderr
    assert 'per entry' in r.stderr and 'taken' in r.stderr and '19,980' in r.stderr
    assert 'hottest lines' not in r.stderr


@pytest.mark.parametrize('flag', ['--counts', '--live'])
def test_cli_needs_run(tmp_path, flag):
    r = subprocess.run([sys.executable, '-m', 'scramjit', 'preflight', flag], cwd=tmp_path, env=ENV, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and f'{flag} needs --run' in r.stderr


def test_cli_json_has_the_counts(tmp_path):
    r = cli(tmp_path, '--counts', '--json', '-o', 'out.json')
    assert r.returncode == 0, r.stderr
    row = json.loads((tmp_path / 'out.json').read_text())['functions'][0]
    assert row['counts']['lines']['10'] == 19980
    assert row['counts']['loop_iterations']['9'] == 19980


def test_cli_live_json_lines(tmp_path):
    r = cli(tmp_path, '--live', '--counts', '--json', '-o', 'run.jsonl', iters=1000000)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (tmp_path / 'run.jsonl').read_text().splitlines()]
    assert rows[-1]['type'] == 'report' and rows[-1]['functions'][0]['counts']['lines']['10'] == 999000000
    mid = [x for x in rows[:-1] if x['hottest']]
    assert mid and mid[0]['type'] == 'snapshot' and 0 < mid[0]['hottest'][0]['times'] < 999000000
    assert mid[0]['running'] == ['settle']       # one long call shows up while it's under way


def test_cli_live_piped_prints_lines(tmp_path):
    r = cli(tmp_path, '--live', iters=200)
    assert r.returncode == 0 and '\x1b[' not in r.stderr.split('scramjit', 1)[0]


@pytest.mark.skipif(sys.platform == 'win32', reason='needs a pty')
def test_cli_live_redraws_on_a_terminal(tmp_path):
    (tmp_path / 'app.py').write_text('print("BEFORE", flush=True)\n' + APP % 1000000 + 'print("AFTER", flush=True)\n')
    leader, follower = pty.openpty()
    p = subprocess.Popen([sys.executable, '-m', 'scramjit', 'preflight', '--live', '--counts', '--run', 'app.py'], cwd=tmp_path,
                         env=dict(ENV, TERM='xterm'), stdin=follower, stdout=follower, stderr=follower)
    os.close(follower)
    out = b''
    while True:
        ready, _, _ = select.select([leader], [], [], 240)
        if not ready:
            p.kill()
            raise AssertionError('timed out')
        try:
            chunk = os.read(leader, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    p.wait()
    text = out.decode('utf-8', 'replace')
    assert p.returncode == 0
    assert 'BEFORE' in text and 'AFTER' in text and 'RESULT' in text
    assert '\x1b[J' in text and '% native' in text and 'hottest lines' in text
