"""The C of a function doesn't depend on where it is in its file, or on the file's name, so cimo's cache keeps it
through edits elsewhere; tracebacks, preflight's notes and line counts still name the file's lines."""

import importlib.util
import traceback

import numpy as np

from scramjit._compiler import cimo_diagnostic

SOURCE = '''
import numpy as np
from scramjit import scram

@scram(cache=False)
def helper(x, i):
    return x[i] * 2.0

@scram(cache=False)
def outer(x, n):
    s = 0.0
    for i in range(n):
        s += x[i]
    return s + helper(x, n)

@scram(cache=False)
def biggest(x):
    return x.max()
'''


def load(path, padding):
    path.write_text(padding + SOURCE)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def where(mod, x, n):
    try:
        mod.outer(x, n)
    except IndexError as e:
        return [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(e.__traceback__) if f.name in ('outer', 'helper')]
    raise AssertionError('no IndexError')


def test_same_c_wherever_the_function_is(tmp_path):
    x = np.arange(4.0)
    a = load(tmp_path / 'first.py', '')
    b = load(tmp_path / 'second_name.py', '# moved\n' * 7)
    assert a.outer(x, 3) == b.outer(x, 3) == 3.0 + 6.0
    assert a.outer.c() == b.outer.c()
    a.biggest(x)
    b.biggest(x)
    assert a.biggest.c() == b.biggest.c()
    assert str(tmp_path) not in a.outer.c()


def test_tracebacks_name_the_files_lines(tmp_path):
    x = np.arange(4.0)
    a = load(tmp_path / 'first.py', '')
    b = load(tmp_path / 'second.py', '# moved\n' * 7)
    fa, fb = where(a, x, 4), where(b, x, 4)          # helper(x, 4) indexes past the end
    assert [n for _, _, n in fa] == ['outer', 'helper']
    assert [(name, line + 7) for _, line, name in fa] == [(name, line) for _, line, name in fb]
    lines = (tmp_path / 'second.py').read_text().split('\n')
    assert 'helper(x, n)' in lines[fb[0][1] - 1] and 'x[i] * 2.0' in lines[fb[1][1] - 1]
    assert fb[0][0].endswith('second.py')


def test_preflight_notes_name_the_files_lines(tmp_path):
    b = load(tmp_path / 'second.py', '# moved\n' * 7)
    x = np.arange(4.0)
    b.outer(x, 3)
    b.biggest(x)
    lines = (tmp_path / 'second.py').read_text().split('\n')
    (entry,) = [v for v in b.outer._cache.values() if hasattr(v, 'checks')]
    assert entry.checks or entry.hoisted, 'outer has a bounds check'
    for line in list(entry.checks) + list(entry.hoisted):
        assert '[' in lines[line - 1], line
    (entry,) = [v for v in b.biggest._cache.values() if hasattr(v, 'gotchas')]
    assert [lines[n - 1].strip() for n in entry.gotchas] == ['return x.max()']


def test_cimo_messages_point_at_python():
    frames = [('/w/lib.py', 'helper', 40)]
    assert cimo_diagnostic('sj:3: error: x; sj0:2: note', '/w/app.py', 10, frames) == '/w/app.py:13: error: x; /w/lib.py:42: note'
    assert cimo_diagnostic('nothing here, xsj:3', '/w/app.py', 10, frames) == 'nothing here, xsj:3'
