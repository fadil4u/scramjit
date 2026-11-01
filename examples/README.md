# Examples

Each is one file you can run: `python examples/nbody.py`. Each times the same function as plain
Python and through scramjit, and asserts the results are identical. They need numpy.

| Example | The loop | Why numpy can't do it | Python | scramjit | Speedup |
|:---|:---|:---|---:|---:|---:|
| [`nbody.py`](nbody.py) | Gravity between all pairs, leapfrog integration | Updates in place within a step; prints energy drift as a check | 1.32 s | 3.0 ms | 445x |
| [`edit_distance.py`](edit_distance.py) | Levenshtein distance | Each cell depends on three before it | 1.33 s | 10.3 ms | 129x |
| [`components.py`](components.py) | Union-find on a grid | Follows parent pointers for as long as the data says | 0.62 s | 12.4 ms | 50x |
| [`karplus_strong.py`](karplus_strong.py) | Plucked strings, written to a WAV | Each sample depends on one a period earlier | 0.80 s | 28.3 ms | 28x |
| [`mandelbrot.py`](mandelbrot.py) | Escape-time fractal, drawn in the terminal | Each pixel iterates a different number of times | 1.67 s | 76.7 ms | 22x |
| [`asian_option.py`](asian_option.py) | Monte Carlo option pricing | Path-dependent, with generator state across paths | 1.06 s | 86.3 ms | 12x |

Arm64 Linux, one run of each script. The first call for a set of argument types compiles, which takes
40 to 200 ms here; these times exclude it, and the scripts print it. `asian_option` gains least because
most of its time is in `log`, `cos` and `exp`, which are the same library functions either way.

Two more show behaviour rather than speed:

- [`threads.py`](threads.py): compiled code releases the GIL. Four threads ran 3.8x faster than the same
  calls one after another; plain Python threads gained 1.0x.
- [`fallback.py`](fallback.py): a function that doesn't compile runs as Python with a `FallbackWarning`.
  Shows the warning, `f.preflight()`, `fallback=False`, and a rewrite that compiles.
