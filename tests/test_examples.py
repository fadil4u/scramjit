"""The examples run, compile, and agree with CPython (each script asserts that itself)."""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
EXAMPLES = sorted(f for f in os.listdir(os.path.join(ROOT, 'examples')) if f.endswith('.py'))


def run(args, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, '..', 'cimo', 'python')]),
               PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300, encoding='utf-8')


@pytest.mark.parametrize('name', EXAMPLES)
def test_example_runs_native(name, tmp_path):
    path = os.path.join(ROOT, 'examples', name)
    # any fallback is an error, except in the examples about falling back
    args = [path] if name in ('fallback.py', 'partly_compiled.py') else ['-W', 'error::scramjit.FallbackWarning', path]
    r = run(args, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_preflight_agrees(tmp_path):
    r = run(['-m', 'scramjit', 'preflight', os.path.join(ROOT, 'examples')], tmp_path)
    assert '25 compiled, 1 partly compiled, 2 run as Python' in r.stdout, r.stdout
    assert '✗ histogram_dict' in r.stdout and '✗ must_be_native' in r.stdout
