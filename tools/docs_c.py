"""Put the C scramjit generates next to the Python snippets in the docs, as a second tab.

A snippet takes part when a comment with an example call follows it:

    ```python
    @scram
    def scale(x, k: float):
        ...
    ```
    <!-- c: scale(np.ones(4), 2.0) -->
    ```c-generated
    (written by this tool)
    ```

The snippet runs as a module of its own, after `import math`, `import numpy as np`, `from enum import Enum, IntEnum, auto`
and `from scramjit import scram`; the call compiles it, and the C of the function called (with any it calls) goes in
the `c-generated` block, without the prelude every kernel shares or the #line markers. The site shows the two as tabs.

    python tools/docs_c.py            rewrite the blocks
    python tools/docs_c.py --check    exit 1 if any block isn't what scramjit generates now (tests/test_docs_c.py)
"""

import ast
import glob
import importlib.util
import os
import re
import sys
import tempfile
import warnings

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

HEADER = 'import math\nimport numpy as np\nfrom enum import Enum, IntEnum, auto\nfrom scramjit import scram\n\n'
BLOCK = re.compile(r'(```python\n(?P<code>(?:(?!```).)*?)```\n)(?P<gap>\s*)<!-- c: (?P<call>.*?) -->\n(?:\s*```c-generated\n(?:(?!```).)*?```\n)?', re.S)


def generated_c(code, call, where):
    """The C the call compiles the snippet's function to."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'snippet.py')
        tree = ast.parse(code)                    # the snippet's example calls aren't part of it
        tree.body = [s for s in tree.body if not isinstance(s, ast.Expr)]
        with open(path, 'w') as f:
            f.write(HEADER + ast.unparse(tree) + '\n')
        spec = importlib.util.spec_from_file_location('snippet', path)
        mod = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter('error')            # a snippet shown with its C must compile
            spec.loader.exec_module(mod)
            name = ast.parse(call, mode='eval').body.func.id
            eval(call, vars(mod))
        source = vars(mod)[name].c()
    lines = [ln for ln in source.split('\n') if not ln.lstrip().startswith('#line')]
    start = next(i for i, ln in enumerate(lines) if re.match(r'\s*(static int sj_call_\d+\(|int sj_kernel\(|sj_kernel\()', ln)
                 or ln.startswith('int sj_kernel(') or re.match(r'^\w[\w\s\*]*\bsj_(kernel|call_\d+)\(', ln))
    out = '\n'.join(lines[start:]).strip('\n')
    return re.sub(r'\n{3,}', '\n\n', out)


def rewrite(text, where):
    def one(m):
        c = generated_c(m.group('code'), m.group('call'), where)
        return f"{m.group(1)}<!-- c: {m.group('call')} -->\n```c-generated\n{c}\n```\n"
    return BLOCK.sub(one, text)


def main():
    check = '--check' in sys.argv
    stale = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'docs', 'docs', 'content', '*.md'))):
        with open(path) as f:
            text = f.read()
        if '<!-- c: ' not in text:
            continue
        new = rewrite(text, path)
        if new != text:
            stale.append(os.path.basename(path))
            if not check:
                with open(path, 'w') as f:
                    f.write(new)
    if check and stale:
        print('generated C out of date in:', ', '.join(stale), '(run python tools/docs_c.py)')
        return 1
    print(('checked' if check else 'wrote'), 'C tabs;', 'stale: ' + ', '.join(stale) if stale else 'all current')
    return 0


if __name__ == '__main__':
    sys.exit(main())
