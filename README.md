<div align="center">

<img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/logo.svg" alt="scramjit" width="96">

# scramjit

Python at native speed.

[Live demo](https://www.scramjit.dev/#lab) · [Docs](https://www.scramjit.dev/docs/)

<img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/race.gif" alt="The same Mandelbrot function rendered side by side on one clock, as plain Python and through scramjit." width="486">

<sub>The same function on one clock, native on arm64. scramjit is capped at one fractal frame per GIF frame; uncapped, it does 21x the iterations. <a href="https://github.com/farizrahman4u/scramjit/blob/main/tools/readme_gif.py">Script.</a></sub>

</div>

## Usage

```python
import numpy as np
from scramjit import scram

@scram
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]

x = np.random.rand(1_000_000)
y = np.zeros_like(x)
ema(x, y, 0.1)     # first call: compiles for these argument types, about 0.1 s
ema(x, y, 0.1)     # 3 ms; undecorated, 250 ms
```

scramjit is a JIT written in pure Python. On the first call for a combination of argument types, it
generates C specialized to those types and compiles it in-process with
[cimo](https://www.cimo.dev), a C compiler that is also pure Python. It needs no LLVM, C compiler or
binary wheels. Arrays are passed without copying, and compiled code releases the GIL.

## Install

```
pip install scramjit numpy
```

Requires CPython 3.9 or newer. numpy is only needed for your arrays.

## Live demo

<img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/demo.gif" alt="The live demo: a flock of boids runs as plain Python, updating every couple of seconds. scramjit is switched on, the Python is swept into generated C, and the flock moves smoothly." width="840">

[The demo](https://www.scramjit.dev/#lab) runs Pyodide, scramjit and cimo in the browser, with no
compile server, and targets WebAssembly. The code is editable and recompiles as you type.

| Demo | Loop | Speedup over Pyodide's CPython |
|:---|:---|---:|
| Terrain | Hydraulic erosion, each droplet following what the last carved | ~240x |
| Flock | 1,200 boids, all pairs | ~180x |
| Sand | Cellular automaton, updated in place | ~90x |
| Fractal | Mandelbrot, one escape loop per pixel | ~45x |

## Examples

[`examples/`](https://github.com/farizrahman4u/scramjit/tree/main/examples) has runnable scripts: n-body gravity, edit distance, union-find, Karplus-Strong
audio, a Mandelbrot renderer, Monte Carlo option pricing, threads, and what falling back looks like.

## Benchmarks

Arm64 Linux, from [`bench/kernels.py`](https://github.com/farizrahman4u/scramjit/blob/main/bench/kernels.py) and [`bench/m1.py`](https://github.com/farizrahman4u/scramjit/blob/main/bench/m1.py). The
hand-written C is compiled by cimo too.

| Kernel | CPython | scramjit | Hand-written C | Speedup |
|:---|---:|---:|---:|---:|
| Gray-Scott reaction-diffusion, 256×256 | 120 ms | 0.22 ms | 0.20 ms | 540x |
| Flood fill, 512×512 | 248 ms | 3.6 ms | 2.3 ms | 69x |
| Viewshed, 96×96 | 117 ms | 10.8 ms | 5.8 ms | 11x |
| Exponential moving average, per element | 272 ns | 3.11 ns | 3.09 ns | 88x |
| Edit distance, per cell | 468 ns | 4.33 ns | | 108x |
| Collatz, per step | 41 ns | 2.39 ns | | 17x |

Viewshed pays for exact `math` semantics; with `parity=False` it takes 7.3 ms. On Gray-Scott and
flood fill, scramjit's C compiled by `gcc -O3` matches hand-written C compiled the same way, so the
remaining gap there is cimo's backend ([`bench/gcc.py`](https://github.com/farizrahman4u/scramjit/blob/main/bench/gcc.py)). Four threads ran four calls
3.5x faster than running them one after another.

## How it works

```
function ──▶ syntax tree ──▶ type inference ──▶ C ──▶ cimo ──▶ machine code
                             from the first call's             arm64, x86-64,
                             argument types                    WebAssembly
```

The generated C implements Python's semantics, including floor division, negative indices and bounds
checks that raise `IndexError`. `range`, `zip`, `enumerate`, `map`, `filter` and comprehensions are
lowered to plain loops. Later calls with the same types go straight to the compiled code.

## What compiles

| Compiled | Runs as Python |
|:---|:---|
| `for`, `while`, `break`, `continue`, `if`, conditional expressions | Masks and index arrays: `x[x > 0]` |
| Views: `m[i]`, `x[a:b]`, `x[::-1]`, `m[:, j]`, `.T`, `for row in m`, as arguments and results | `x.reshape(...)`, `np.eye(n)` |
| Making arrays: `np.zeros(n)`, `x.copy()`, `np.arange`, `np.linspace`, as locals and results | `x[1:] += y` |
| Assigning to a slice: `x[1:3] = 0.0`, `m[i] = row`, overlap included | `np.sqrt(x)` on a whole array |
| Numbers, tuples, arrays and dicts at module level | Assigning to a global, other global objects |
| `range`, `enumerate`, `zip`, `map`, `filter`, comprehensions | numpy array expressions: `a * 2` |
| `sum(gen, 0.0)`, `min`, `max`, `abs`, `**`, tuple unpacking | Calls to undecorated Python functions, recursion |
| Calls to other `@scram` functions and to C from `cimo.compile`, usually inlined | Strings, objects |
| Numbers and n-D numpy arrays, including dtypes like `uint8` | Classes |
| All of `math`, natively and in Pyodide (float `sumprod` excepted) | Lists that grow with `append` |
| Dicts passed in, reading and writing keys it names | Dicts that gain keys, or built in the function |
| numpy functions on single numbers: `np.sqrt(x[i])`, `np.sin`, `np.minimum` | numpy calls that make arrays: `np.sqrt(x)`, `x.sum(axis=0)` |
| numpy reductions to one number: `x.sum()`, `np.max(m[i])`, `x.std()`, `x.argmax()` | `np.dot`, `np.linalg` |

`**` compiles. Where Python's result would change type, it raises `ValueError`: a negative `k` in
`n ** k` (Python gives a float) and a negative float `x` in `x ** 0.5` (Python gives a complex).

A function that can't be compiled runs as Python, with a `FallbackWarning` giving the reason:

```python
import warnings
import numpy as np
from scramjit import scram, CompileError

@scram
def count(a):
    seen = {}
    for i in range(len(a)):
        seen[a[i]] = i
    return len(seen)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    print(count(np.arange(3.0)))     # 3
print(caught[0].message)             # count(float64[:]) runs as Python: the Dict expression

@scram(fallback=False)               # raise CompileError instead
def strict(a):
    return {0: a[0]}[0]

try:
    strict(np.arange(3.0))
except CompileError as e:
    print(e)
```

## Preflight

`scramjit preflight` shows what compiles and which lines stop it, without importing your code:

```
$ scramjit preflight src/
src/sim.py
  ✓ ema        compiled
  ✓ grid       compiled
  ✗ summarize  runs as Python
      22 │     names = {}
         │             ^^ dicts aren't compiled
         │ hint: if the keys are small integers, use an array indexed by key; otherwise do this
         │       outside the compiled function

2 compiled, 1 runs as Python
```

It also lists undecorated functions with a loop that would compile if decorated, assuming their
arguments are numbers or arrays. `scramjit preflight --run app.py` (or `--run -m pytest`) reports what happened with the real types:
calls, compile time, and time in native code and in Python. Both exit non-zero if a function isn't
compiled, so they work as a CI check, and `--json` gives the report to tools. Add `--counts` for how often each line ran and
each branch was taken, and `--live` to watch it happen. `scramjit preflight
FILE:FUNCTION` or `f.preflight()` annotates one function line by line, including which bounds checks
were proven, hoisted out of the loop or kept. `scramjit c FILE:FUNCTION` prints the generated C, and
`SCRAMJIT_LOG=1` logs each compile and fallback. `scramjit --help` has the rest.

Compiled code is cached on disk: compiling took 0.53 s in the first process and 0.06 s in later
ones on a Gray-Scott kernel. `@scram(cache=False)` writes nothing.

## CPython parity

Compiled functions are tested against the same functions running undecorated, including on randomly
generated expressions. This covers floor division and modulo on negatives, `ZeroDivisionError`,
`IndexError`, `math` domain errors and numpy's scalar rules.

There are three exceptions: integers are 64-bit and wrap, and the two `**` cases above raise
`ValueError`. Compiled calls also skip numpy's warnings and `errstate`, are invisible to debuggers and
coverage, and cost about 1.4 µs each from Python, so a loop should be compiled too, not only a small helper. See
[Gotchas](https://www.scramjit.dev/docs/?p=gotchas).

| Option | Default | Effect |
|:---|:---|:---|
| `parity` | `True` | `False` gives IEEE results for float division by zero and `math`, and uses cimo's libm |
| `boundscheck` | `True` | `False` removes index checks |
| `fallback` | `True` | `False` raises `CompileError` instead of falling back |

Compiled code runs with the full privileges of the Python process, like an extension module. There
is no sandbox.

## Status

Pre-release.

- About 1,600 tests pass. Developed and benchmarked on arm64 Linux and in Pyodide, and run by hand on arm64 Windows and on x86-64 Windows
  (emulated), where a NaN's sign can differ. macOS is a cimo target that scramjit hasn't been run on, and none of these run in CI yet.
- A function that doesn't compile whole has its loops compiled on their own, where they fit.
- Dicts compile when passed in with keys that don't change. Next, the rest of the data structures: see `TODO.md`.
- Next: see [`TODO.md`](https://github.com/farizrahman4u/scramjit/blob/main/TODO.md).

## Development

```
pip install -e .[dev]       # with cimo installed, or checked out next to this repository
python -m pytest tests
python bench/kernels.py
```

MIT licensed.
