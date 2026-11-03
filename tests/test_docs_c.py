"""The C shown next to snippets in the docs is what scramjit generates for them now (tools/docs_c.py writes it)."""

import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def test_generated_c_in_the_docs_is_current():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'docs_c.py'), '--check'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
