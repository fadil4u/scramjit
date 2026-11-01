"""Ctrl-C during a compiled call: real signals, to real processes. And that the chunked loops it needs
behave exactly like Python's around their chunk boundaries."""

import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from scramjit import scram
from util import same

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CIMO = os.path.join(ROOT, '..', 'cimo', 'python')
posix = pytest.mark.skipif(sys.platform == 'win32', reason='signals are sent differently')

COMMON = '''
import math, os, signal, sys, threading, time
import numpy as np
from scramjit import scram
print = __import__('functools').partial(print, flush=True)
'''


def start(tmp_path, body, kernels=''):
    (tmp_path / 'kernels.py').write_text(textwrap.dedent(kernels))
    (tmp_path / 'main.py').write_text(COMMON + 'import kernels\n' + textwrap.dedent(body))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, CIMO, str(tmp_path)]), PYTHONWARNINGS='ignore')
    return subprocess.Popen([sys.executable, 'main.py'], cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def interrupt(p, after=0.3):
    """Wait for the program to say go, send SIGINT, and return when it's finished."""
    assert p.stdout.readline().strip() == 'go', p.stderr.read()
    time.sleep(after)
    t = time.perf_counter()
    p.send_signal(signal.SIGINT)
    try:
        out, err = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:           # never leave a spinning program behind
        p.kill()
        out, err = p.communicate()
        raise AssertionError(f'still running 30 s after SIGINT: {out} {err}') from None
    return out, err, time.perf_counter() - t


LOOPS = {
    'for_range': ('def f(n: int):\n    s = 0\n    for i in range(n):\n        s = (s + i * i) % 1000003\n    return s\n', '(1 << 62)'),
    'for_step': ('def f(n: int):\n    s = 0\n    for i in range(0, n, 3):\n        s = (s + i) % 1000003\n    return s\n', '(1 << 62)'),
    'nested': ('def f(n: int):\n    s = 0\n    for i in range(n):\n        for j in range(n):\n            s = (s + i * j) % 1000003\n    return s\n', '(1 << 31)'),
    'while_true': ('def f(n: int):\n    s = 0\n    while True:\n        s = (s + n) % 1000003\n        if n == 1:\n            break\n    return s\n', '3'),
    'while_cond': ('def f(n: int):\n    s = 0\n    i = 0\n    while i < n:\n        s = (s + i) % 1000003\n        i += 1\n    return s\n', '(1 << 62)'),
    'generator': ('def f(n: int):\n    return sum(i % 7 for i in range(n))\n', '(1 << 62)'),
    'any_gen': ('def f(n: int):\n    return any(i < 0 for i in range(n))\n', '(1 << 62)'),
    'list_comp': ('def f(n: int):\n    a = [i % 7 for i in range(n)]\n    return a[0]\n', '(1 << 27)'),
}


@posix
@pytest.mark.parametrize('name', LOOPS)
def test_every_kind_of_loop_stops(tmp_path, name):
    src, arg = LOOPS[name]
    p = start(tmp_path, f'''
        f = scram(kernels.f, fallback=False)
        f(1)
        print('go')
        try:
            f({arg})
            print('finished')
        except KeyboardInterrupt:
            print('interrupted')
    ''', src)
    out, err, took = interrupt(p)
    assert 'interrupted' in out, out + err
    assert took < 5, took


@posix
def test_writes_so_far_stay_and_the_function_still_works(tmp_path):
    p = start(tmp_path, '''
        f = scram(kernels.fill, fallback=False)
        a = np.zeros(1 << 20)
        f(a, 1)
        print('go')
        try:
            f(a, 1 << 62)
        except KeyboardInterrupt:
            print('interrupted', a[0] > 0)
        b = np.zeros(3)
        f(b, 1)
        print('again', b[0])
        try:
            raise ValueError
        except ValueError:
            print('python is fine')
    ''', 'def fill(a, n: int):\n    for k in range(n):\n        for i in range(len(a)):\n            a[i] += 1.0\n')
    out, err, _ = interrupt(p)
    assert 'interrupted True' in out and 'again 1.0' in out and 'python is fine' in out, out + err


@posix
def test_an_uncaught_interrupt_ends_the_program_as_usual(tmp_path):
    p = start(tmp_path, '''
        f = scram(kernels.f, fallback=False)
        f(1)
        print('go')
        f(1 << 62)
    ''', LOOPS['for_range'][0])
    out, err, _ = interrupt(p)
    assert 'KeyboardInterrupt' in err and p.returncode != 0


@posix
def test_a_worker_thread_is_not_stopped(tmp_path):
    """Python raises KeyboardInterrupt on the main thread only."""
    p = start(tmp_path, '''
        f = scram(kernels.f, fallback=False)
        f(1)
        result = []
        t = threading.Thread(target=lambda: result.append(f(400_000_000)))
        t.start()
        print('go')
        try:
            while t.is_alive():
                time.sleep(0.01)
            print('no signal arrived')
        except KeyboardInterrupt:
            print('main interrupted')
        t.join()
        print('worker result', result == [f.__wrapped__(1000)] or len(result) == 1)
    ''', LOOPS['for_range'][0])
    out, err, _ = interrupt(p, after=0.1)
    assert 'main interrupted' in out and 'worker result True' in out, out + err


@posix
def test_a_custom_handler_doesnt_stop_the_kernel(tmp_path):
    """It might not raise, and a stopped kernel can't resume, so the call finishes, as it did before."""
    p = start(tmp_path, '''
        seen = []
        signal.signal(signal.SIGINT, lambda *a: seen.append(1))
        f = scram(kernels.f, fallback=False)
        f(1)
        print('go')
        r = f(600_000_000)
        print('finished', r > 0 or r == 0, seen)
    ''', LOOPS['for_range'][0])
    out, err, _ = interrupt(p, after=0.05)
    assert 'finished True [1]' in out, out + err


@posix
def test_a_handler_set_after_the_first_compile_is_respected(tmp_path):
    p = start(tmp_path, '''
        f = scram(kernels.f, fallback=False)
        f(1)
        seen = []
        signal.signal(signal.SIGINT, lambda *a: seen.append(1))
        print('go')
        r = f(600_000_000)
        print('finished', seen)
    ''', LOOPS['for_range'][0])
    out, err, _ = interrupt(p, after=0.05)
    assert 'finished [1]' in out, out + err


@posix
def test_another_signal_doesnt_stop_the_kernel(tmp_path):
    p = start(tmp_path, '''
        seen = []
        signal.signal(signal.SIGUSR1, lambda *a: seen.append(1))
        f = scram(kernels.f, fallback=False)
        f(1)
        def poke():
            time.sleep(0.1)
            os.kill(os.getpid(), signal.SIGUSR1)
        threading.Thread(target=poke).start()
        print('go')
        r = f(600_000_000)
        print('finished', seen)
    ''', LOOPS['for_range'][0])
    assert p.stdout.readline().strip() == 'go'
    out, err = p.communicate(timeout=60)
    assert 'finished [1]' in out, out + err


@posix
def test_a_signal_while_in_python_doesnt_stop_the_next_call(tmp_path):
    p = start(tmp_path, '''
        f = scram(kernels.f, fallback=False)
        f(1)
        print('go')
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print('in python')
        time.sleep(0.2)                     # the watcher has set its flag by now
        print('result', f(100_000) == f.__wrapped__(100_000))
    ''', LOOPS['for_range'][0])
    out, err, _ = interrupt(p)
    assert 'in python' in out and 'result True' in out, out + err


@posix
def test_a_forked_child_has_its_own_ctrl_c(tmp_path):
    p = start(tmp_path, '''
        f = scram(kernels.f, fallback=False)
        f(1)
        pid = os.fork()
        if pid == 0:
            print('go')
            try:
                f(1 << 62)
            except KeyboardInterrupt:
                os._exit(42)
            os._exit(1)
        time.sleep(0.5)
        os.kill(pid, signal.SIGINT)         # the child's, while the parent runs a kernel of its own
        r = f(200_000_000)
        _, status = os.waitpid(pid, 0)
        print('child', os.waitstatus_to_exitcode(status), 'parent finished')
    ''', LOOPS['for_range'][0])
    assert p.stdout.readline().strip() == 'go'
    out, err = p.communicate(timeout=60)
    assert 'child 42 parent finished' in out, out + err


# ---- Windows: a real console control event, in a console of the program's own

WINDOWS_COMMON = """
import ctypes, os, signal, sys, threading, time
import numpy as np
from scramjit import scram
import kernels
print = __import__('functools').partial(print, flush=True)
sent = []
def fire(after):
    time.sleep(after)
    sent.append(time.perf_counter())
    assert ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0)
f = scram(kernels.f, fallback=False)
f(1)
"""


def run_windows(tmp_path, kernels, body):
    (tmp_path / 'kernels.py').write_text(textwrap.dedent(kernels))
    (tmp_path / 'main.py').write_text(WINDOWS_COMMON + textwrap.dedent(body))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, CIMO, str(tmp_path)]), PYTHONWARNINGS='ignore')
    p = subprocess.Popen([sys.executable, 'main.py'], cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        out, err = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        raise AssertionError(f'still running after 60 s: {out} {err}') from None
    return out, err


windows = pytest.mark.skipif(sys.platform != 'win32', reason='console control events')


@windows
@pytest.mark.parametrize('name', LOOPS)
def test_windows_every_kind_of_loop_stops(tmp_path, name):
    src, arg = LOOPS[name]
    out, err = run_windows(tmp_path, src, f"""
        threading.Thread(target=fire, args=(0.3,), daemon=True).start()
        try:
            f({arg})
            print('finished')
        except KeyboardInterrupt:
            print('interrupted', time.perf_counter() - sent[0] < 5)
    """)
    assert 'interrupted True' in out, out + err


@windows
def test_windows_a_custom_handler_doesnt_stop_the_kernel(tmp_path):
    out, err = run_windows(tmp_path, LOOPS['for_range'][0], """
        seen = []
        signal.signal(signal.SIGINT, lambda *a: seen.append(1))
        threading.Thread(target=fire, args=(0.05,), daemon=True).start()
        r = f(400_000_000)
        time.sleep(0.05)
        print('finished', seen)
    """)
    assert 'finished [1]' in out, out + err


@windows
def test_windows_a_worker_thread_is_not_stopped(tmp_path):
    out, err = run_windows(tmp_path, LOOPS['for_range'][0], """
        done = []
        t = threading.Thread(target=lambda: done.append(f(300_000_000)))
        t.start()
        threading.Thread(target=fire, args=(0.1,), daemon=True).start()
        try:
            while t.is_alive():
                time.sleep(0.01)
            print('no interrupt')
        except KeyboardInterrupt:
            print('main interrupted')
        t.join()
        print('worker finished', len(done))
    """)
    assert 'main interrupted' in out and 'worker finished 1' in out, out + err


@windows
def test_windows_a_signal_while_in_python_doesnt_stop_the_next_call(tmp_path):
    out, err = run_windows(tmp_path, LOOPS['for_range'][0], """
        threading.Thread(target=fire, args=(0.1,), daemon=True).start()
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print('in python')
        time.sleep(0.2)
        print('whole', f(100000) == f.__wrapped__(100000))
    """)
    assert 'in python' in out and 'whole True' in out, out + err


# ---- chunk boundaries


def sums(n: int):
    s = 0
    i = -1
    for i in range(n):
        s += i
    return s, i                                 # the loop variable, after the loop


def from_to(a: int, b: int):
    s = 0
    c = 0
    for i in range(a, b):
        s ^= i
        c += 1
    return s, c


def stepped(a: int, b: int, k: int):
    s = 0
    c = 0
    for i in range(a, b, k):
        s ^= i
        c += 1
    return s, c


def breaks(n: int, at: int):
    c = 0
    i = -1
    for i in range(n):
        if i == at:
            break
        c += 1
    return c, i


def continues(n: int, every: int):
    c = 0
    for i in range(n):
        if i % every == 0:
            continue
        c += 1
    return c


def nested_break(n: int, at: int):
    c = 0
    for i in range(3):
        for j in range(n):
            if j == at:
                break
            c += 1
        c += 100
    return c


def in_array(a, at: int):
    c = 0
    for v in a:
        if c == at:
            break
        c += 1
    return c


def enumerated(a):
    s = 0
    for i, v in enumerate(a):
        s += i
    return s


def comp(n: int):
    a = [i for i in range(n)]
    if n > 0:
        return len(a), a[-1]
    return 0, -1


def gen(n: int):
    return sum(i for i in range(n))


def returns(n: int, at: int):
    for i in range(n):
        if i == at:
            return i
    return -1


def while_break(n: int):
    i = 0
    while True:
        i += 1
        if i >= n:
            break
    return i


C = 65536
EDGES = [0, 1, 2, C - 1, C, C + 1, 2 * C - 1, 2 * C, 2 * C + 1, 3 * C + 5]


@pytest.mark.parametrize('n', EDGES)
def test_chunk_boundaries(n):
    same(scram(sums, fallback=False), n)
    same(scram(comp, fallback=False), n)
    same(scram(gen, fallback=False), n)
    same(scram(enumerated, fallback=False), np.zeros(n))
    for at in (0, 1, C - 1, C, C + 1, 2 * C, n - 1, n, n + 1):
        same(scram(breaks, fallback=False), n, at)
        same(scram(returns, fallback=False), n, at)
        same(scram(in_array, fallback=False), np.zeros(n), at)
    same(scram(continues, fallback=False), n, 2)
    same(scram(continues, fallback=False), n, C)
    same(scram(while_break, fallback=False), n + 1)


@pytest.mark.parametrize('n, at', [(C + 5, C), (C + 5, 3), (2 * C + 1, 2 * C), (5, 100)])
def test_nested_breaks_leave_only_their_own_loop(n, at):
    same(scram(nested_break, fallback=False), n, at)


@pytest.mark.parametrize('a, b', [(0, C), (-5, C), (-C - 1, C + 1), (5, 5), (6, 5), (C, 0), ((1 << 63) - 10, (1 << 63) - 1),
                                  (-(1 << 63), -(1 << 63) + 7), ((1 << 63) - C - 3, (1 << 63) - 1)])
def test_ranges_near_the_ends_of_int64(a, b):
    same(scram(from_to, fallback=False), a, b)


@pytest.mark.parametrize('a, b, k', [(0, 3 * C, 3), (3 * C, 0, -3), (0, C, C), (0, C + 1, C), (0, 5, 100), ((1 << 63) - 40, (1 << 63) - 1, 7),
                                     (-(1 << 63), -(1 << 63) + 40, 7), ((1 << 63) - 1, (1 << 63) - 40, -7)])
def test_steps(a, b, k):
    same(scram(stepped, fallback=False), a, b, k)


def short_ranges(a):
    s = 0.0
    for i in range(1, 5):
        for j in range(-1, 2):
            if j == 1:
                break
            s += a[i + j]
        for k in range(0, 6, 2):
            if k == 4:
                continue
            s += k
    return s


def test_short_constant_ranges_need_no_chunks_and_still_behave():
    g = scram(short_ranges, fallback=False)
    same(g, np.arange(8.0))
    assert 'sj_stop' not in g.c()


def long_constant(a):
    s = 0.0
    for i in range(100000):
        s += a[0]
    return s


def test_a_long_constant_range_can_still_be_interrupted():
    g = scram(long_constant, fallback=False)
    same(g, np.ones(1))
    assert 'sj_stop' in g.c()


def no_loop(x: float):
    return x + 1


def test_a_function_with_no_loop_takes_no_flag():
    g = scram(no_loop, fallback=False)
    g(1.0)
    assert 'sj_stop' not in g.c()
