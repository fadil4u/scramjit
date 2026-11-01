# scramjit

**Python at native speed.** A JIT written in pure Python, for the loops the interpreter is
slowest at. Decorate a function; scramjit turns it into C and compiles it with
[cimo](https://www.cimo.dev) in your process, then runs it without the GIL. No LLVM, no C
compiler, no binary wheels.

```python
from scramjit import scram

@scram
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
```

Results are bit-identical to CPython's. What can't be compiled runs as Python, with a
`FallbackWarning`.

- Site and docs: https://farizrahman4u.github.io/scramjit/
- Spec and design decisions: [`requirements.md`](requirements.md)
- Development: `pip install -e .[dev]` with cimo installed (or checked out next to this
  repository), then `python -m pytest tests` and `python bench/m1.py`.

MIT licensed.
