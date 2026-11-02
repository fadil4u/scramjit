# Examples

Each is one file you can run: `python examples/nbody.py`. Each times the same function as plain
Python and through scramjit, and asserts the results are identical. They need numpy.

| Example | The loop | Why numpy can't do it | Python | scramjit | Speedup |
|:---|:---|:---|---:|---:|---:|
| [`nbody.py`](nbody.py) | Gravity between all pairs, leapfrog integration | Updates in place within a step; prints energy drift as a check | 1.57 s | 4.0 ms | 388x |
| [`edit_distance.py`](edit_distance.py) | Levenshtein distance | Each cell depends on three before it | 1.91 s | 15.4 ms | 124x |
| [`components.py`](components.py) | Union-find on a grid | Follows parent pointers for as long as the data says | 0.85 s | 20.3 ms | 42x |
| [`karplus_strong.py`](karplus_strong.py) | Plucked strings, written to a WAV | Each sample depends on one a period earlier | 0.99 s | 25.7 ms | 38x |
| [`mandelbrot.py`](mandelbrot.py) | Escape-time fractal, drawn in the terminal | Each pixel iterates a different number of times | 1.94 s | 103 ms | 19x |
| [`asian_option.py`](asian_option.py) | Monte Carlo option pricing | Path-dependent, with generator state across paths | 1.27 s | 103 ms | 12x |
| [`reductions.py`](reductions.py) | Spikes in a signal, with `x[a:b].mean()`, `.max()` and `.min()` on a sliding window | Each sample's window, and a stored list of what's found | 1.47 s | 62 ms | 24x |
| [`views.py`](views.py) | Gaussian elimination with rows and columns as views | Each step changes what the next sees; the pivot depends on the data | 1.65 s | 5.7 ms | 291x |
| [`numpy_functions.py`](numpy_functions.py) | A greedy route over a sphere, with `np.sin`, `np.arcsin`, `np.sqrt` | Each stop depends on the one before | 0.175 s | 2.4 ms | 74x |
| [`life.py`](life.py) | Game of Life, making a new board each generation and returning the last | A cell depends on its neighbours and on a test of them | 0.40 s | 3.1 ms | 131x |
| [`slices.py`](slices.py) | Heat flow, written with slice assignment and views, returning a view | Each step reads the last | 0.314 s | 0.41 ms | 776x |
| [`constants.py`](constants.py) | A filter with its settings at module level: a number, a tuple, an array, a dict | Clips each sample; each pass repeats over its own output | 0.52 s | 1.5 ms | 338x |
| [`config_dict.py`](config_dict.py) | An epidemic model with its settings and results in dicts | Each day depends on the one before | 0.081 s | 4.4 ms | 18x |

One run of each on arm64 Linux, on a busy machine, so expect faster and somewhat different numbers. The
first call for a set of argument types compiles, which took under a second here; these times exclude it,
and the scripts print it. `asian_option` gains least because most of its time is in `log`, `cos` and
`exp`, which are the same library functions either way.

`config_dict` gains less than most because its loop is only float arithmetic with no indexing, which CPython does comparatively well, and it's at the hardware's limit: hand-written C takes the same time per step through cimo and through `gcc -O3`.

Four more show behaviour rather than speed:

- [`threads.py`](threads.py): compiled code releases the GIL. Four threads ran 2.3x to 3.8x faster than the same
  calls one after another, depending on what else the machine was doing; plain Python threads gained
  nothing.
- [`fallback.py`](fallback.py): a function that doesn't compile runs as Python with a `FallbackWarning`.
  Shows the warning, `f.preflight()`, `fallback=False`, and a rewrite that compiles.
- [`hot_lines.py`](hot_lines.py): insertion sort, to run under `scramjit preflight --counts`: 100 million moves on one line, and exactly how often each line and branch ran.
- [`partly_compiled.py`](partly_compiled.py): a function that handles text can't compile whole, but its loop over numbers still does: 96x, with the rest as Python. Shows the warning and `f.preflight()` line by line.
