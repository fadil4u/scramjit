<div align="center">

<a href="https://www.scramjit.dev"><img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/banner.svg" alt="scramjit: Python at native speed" width="100%"></a>

**A JIT for Python's hot loops, written in pure Python.**<br>
Add one decorator. Your loop runs as machine code, without the GIL.<br>
No LLVM. No C compiler. No binary wheels. In the browser too.

[**Try it live**](https://www.scramjit.dev/#lab) &nbsp;·&nbsp; [Docs](https://www.scramjit.dev/docs/) &nbsp;·&nbsp; [Design spec](requirements.md)

<img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/race.gif" alt="A Mandelbrot dive rendered side by side on one clock: plain Python has finished 25 frames when scramjit has finished 118 and is far deeper into the set." width="100%">

<sub>The same function, the same clock, recorded live. scramjit is capped at one frame per GIF frame; uncapped, it's 21x here. <a href="tools/readme_gif.py">Reproduce it.</a></sub>

</div>

## One decorator

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
ema(x, y, 0.1)     # first call: compiles for these argument types, in about 0.1 s
ema(x, y, 0.1)     # 3 ms. The plain Python function takes 250 ms.
```

That's the whole API. This is the kind of loop numpy can't help with: each step depends on the one
before. `y` is written in place, with no copying, and the result is the same, bit for bit, as
running the function undecorated.

## Install

scramjit is pre-release, and the packages on PyPI are placeholders for now. Install both from GitHub:

```
pip install numpy "cimo @ git+https://github.com/farizrahman4u/cimo#subdirectory=python" \
                  "scramjit @ git+https://github.com/farizrahman4u/scramjit"
```

CPython 3.9 or newer. Both packages are plain Python files, with nothing to build, and numpy is only
for your arrays. Or skip installing and [run it in your browser](https://www.scramjit.dev/#lab).

## See it for yourself

<a href="https://www.scramjit.dev/#lab"><img src="https://raw.githubusercontent.com/farizrahman4u/scramjit/main/docs/readme/demos.png" alt="The four live demos: a 3D eroding terrain, a Mandelbrot dive, falling sand and water, and a flock of boids." width="100%"></a>

[The live demo](https://www.scramjit.dev/#lab) runs the entire toolchain in your browser, with no
compile server: CPython in WebAssembly, then scramjit, then cimo. Each simulation starts as plain
Python. Flip the switch and watch the same function compile to WebAssembly, then edit the code and
it recompiles as you type.

| Demo | The loop | Speedup in the browser |
|:---|:---|---:|
| **Terrain** | Raindrops carving a landscape, each following what the last one carved | ~240x |
| **Flock** | 1,200 boids, every one checking every other | ~180x |
| **Sand** | Sand and water you can draw with, cell by cell | ~90x |
| **Fractal** | An endless Mandelbrot dive, an escape loop per pixel | ~45x |

## How fast

Each kernel is plain Python, timed against the same algorithm hand-written in C and compiled by cimo.
Arm64 Linux, from [`bench/kernels.py`](bench/kernels.py) and [`bench/m1.py`](bench/m1.py):

| Kernel | CPython | scramjit | Hand-written C | Speedup |
|:---|---:|---:|---:|---:|
| Gray-Scott reaction-diffusion, 256×256 | 115 ms | **0.22 ms** | 0.20 ms | **520x** |
| Flood fill, 512×512 | 252 ms | **3.6 ms** | 2.4 ms | **70x** |
| Viewshed, 96×96 | 116 ms | **10.8 ms** | 5.9 ms | **11x** |
| Exponential moving average, per element | 272 ns | **3.11 ns** | 3.09 ns | **88x** |
| Edit distance, per cell | 468 ns | **4.33 ns** | | **108x** |
| Collatz, per step | 41 ns | **2.39 ns** | | **17x** |

The spread is real. Tight array loops get close to C, while viewshed pays for exact `math` semantics;
`parity=False` trades those for IEEE behaviour and brings it to 7.3 ms. Compiled by `gcc -O3`
rather than cimo, scramjit's C matches hand-written C on Gray-Scott and flood fill, so what remains
there is the backend, and [`bench/gcc.py`](bench/gcc.py) tracks it.

**Threads run in parallel**, because compiled code releases the GIL. Four threads ran the same work
3.5x faster than one thread running it four times in a row.

## How it works

```
 your function ──▶ syntax tree ──▶ type inference ──▶ C ──▶ cimo ──▶ machine code
                                   from the first                    arm64 · x86-64
                                   call's arguments                  WebAssembly
```

On the first call, scramjit reads the argument types and generates C specialized to them. C
semantics are made to match Python's: floor division, negative indices, bounds checks that raise
`IndexError`. Builtins like `range`, `zip` and `enumerate` don't exist at runtime, because they fuse
into one plain loop. [cimo](https://www.cimo.dev), a C compiler that is itself pure Python, turns
that into machine code in your process. Later calls with the same types go straight to it.

## Nothing breaks

scramjit can't compile everything, so what it can't compile keeps working. It runs as plain Python,
and a `FallbackWarning` says why:

```python
import warnings
import numpy as np
from scramjit import scram, CompileError

@scram
def count(a):
    seen = {}                 # dicts don't compile
    for i in range(len(a)):
        seen[a[i]] = i
    return len(seen)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    print(count(np.arange(3.0)))     # 3: correct, at Python speed
print(caught[0].message)             # count(float64[:]) runs as Python: ...

@scram(fallback=False)        # or make it an error, for code that must be native
def strict(a):
    return {0: a[0]}[0]

try:
    strict(np.arange(3.0))
except CompileError as e:
    print(e)
```

| Compiles to native code | Runs as Python, for now |
|:---|:---|
| `for`, `while`, `break`, `continue`, `if`, conditional expressions | Slices and array views: `a[1:]` |
| `range`, `enumerate`, `zip`, list comprehensions | numpy array expressions: `a * 2`, `a.sum()` |
| `sum(gen, 0.0)`, `min`, `max`, `abs`, tuple unpacking and swaps | Calls to your other Python functions |
| Numbers and n-D numpy arrays, including narrow dtypes like `uint8` | Dicts, strings, objects, `**`, `map`, `filter` |
| `math` functions (natively; in Pyodide, those with no C library code) | Lists that grow with `append` |

Where it fits: **loops that don't vectorize.** Recurrences, flood fills, dynamic programming,
simulations, data-dependent branches, scans over byte buffers. If your work is already one numpy
expression, numpy is already fast.

## Same results as CPython

Every compiled function is held to the result of running it undecorated, in differential tests
that include randomly generated expressions. That extends to the awkward parts: floor division and
modulo on negatives, `ZeroDivisionError`, `IndexError`, `math` domain errors, and numpy's scalar
rules. The one documented difference is that integers are 64-bit.

`@scram` takes `parity=False` to trade exactness for speed, and `boundscheck=False` to drop index
checks.

## Status

Early, and moving fast. Being straight about where it stands:

- **Works today:** everything above. 55 tests pass, developed and benchmarked on arm64 Linux and in
  Pyodide.
- **Not on PyPI yet:** both packages there are placeholders, so install from GitHub as shown above.
- **Not yet tested by CI on every platform:** x86-64, macOS and Windows are compiler targets through
  cimo, but scramjit's tests don't run there automatically yet.
- **Designed, not built:** the `scramjit preflight` report and the on-disk compile cache. Until
  then, the first call in each process compiles.
- **Next:** all of `math` in Pyodide, slices and views. See [`TODO.md`](TODO.md).

Compiled code runs with the full privileges of your Python process, like an extension module. There
is no sandbox.

## Contributing

Bug reports, benchmarks and pull requests are welcome, and so is the loop you want faster.
[`requirements.md`](requirements.md) has the design and the reasoning behind it.

```
pip install -e .[dev]       # with cimo installed, or checked out next to this repository
python -m pytest tests
python bench/kernels.py
```

---

<div align="center">

MIT licensed. Compiled by [cimo](https://www.cimo.dev), the C compiler written in pure Python.

</div>
