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
| [`calls.py`](calls.py) | A cart and pole as `reset`, `step` and `policy`, run by a function calling them | Each step depends on the last, and an episode ends when the pole falls | 1.95 s | 31.1 ms | 63x |
| [`inline_c.py`](inline_c.py) | Counting different values with a hash table, its hash in C from `cimo.compile` | How far each probe goes depends on the data | 0.222 s | 1.42 ms | 157x |
| [`slices.py`](slices.py) | Heat flow, written with slice assignment and views, returning a view | Each step reads the last | 0.314 s | 0.41 ms | 776x |
| [`narrow_types.py`](narrow_types.py) | `a * x + y` on `float32`, brightening `uint8` pixels, a `float32` dot product | numpy does this `axpy`; here the loop is yours, with numpy's wrapping and rounding | 0.18 s | 0.28 ms | 636x |
| [`xorshift.py`](xorshift.py) | A 32-bit xorshift random number generator, from `<<`, `>>`, `^` and masks | Each number comes from the one before | 0.22 s | 3.9 ms | 56x |
| [`hash64.py`](hash64.py) | FNV-1a 64 over 1 MB, splitmix64 into a `uint64` array, Fibonacci hashing of `uint64` keys | Each step depends on the one before | 0.10 s | 1.8 ms | 57x |
| [`enums.py`](enums.py) | Breadth-first search through a maze, with an `Enum` for each cell | Each step depends on what was found before | 1.74 s | 4.7 ms | 366x |
| [`boxes.py`](boxes.py) | The box overlapping a query most, boxes as nested tuples `((x0, y0), (x1, y1))` | A Python function call per box | 4.21 s | 1.1 ms | 3872x |
| [`lists.py`](lists.py) | A rolling maximum over a plain Python list, returning a list; all shortest paths in a list of lists, in place | Each window is scanned; every pair through every vertex | 0.31 s | 23 ms | 13x |
| [`constants.py`](constants.py) | A filter with its settings at module level: a number, a tuple, an array, a dict | Clips each sample; each pass repeats over its own output | 0.52 s | 1.5 ms | 338x |
| [`classes.py`](classes.py) | A temperature controller as a dataclass, with `@scram` on its method, which calls another, its state left on the object | Each step depends on the one before | 1.88 s | 39 ms | 48x |
| [`particles.py`](particles.py) | Gravity between 200 particles in a plain list of dataclass instances, with methods, a helper taking two particles and a constant at module level | Every particle pulls on every other, each step on the last | 0.27 s | 13 ms | 21x |
| [`vectors.py`](vectors.py) | Rays against a sphere with a small vector dataclass: a `Vec` made for each pixel, normalized by a method, handed to the sphere's method | Each pixel's hit depends on its ray, and its shade on the hit | 0.15 s | 5.4 ms | 27x |
| [`shared_target.py`](shared_target.py) | A bouncing ball that a camera and a spotlight, all dataclasses, both aim at: their `target` is the ball itself | Each step depends on the last; one object under three names | 0.24 s | 3.2 ms | 73x |
| [`vector_math.py`](vector_math.py) | A charged particle in a magnetic field, with a vector dataclass's `+`, `*` and `@` (cross product) making a new `Vec` at each step | Each step depends on the last; no `Vec` is allocated compiled | 0.47 s | 9.4 ms | 50x |
| [`search_tree.py`](search_tree.py) | 200,000 lookups in a binary search tree of dataclass nodes, `node = node.left if key < node.key else node.right`, counting visits on the nodes | Where each walk goes depends on the keys; a branch nothing can predict at every node keeps the gain small | 0.44 s | 73 ms | 6x |
| [`config_dict.py`](config_dict.py) | An epidemic model with its settings and results in dicts | Each day depends on the one before | 0.081 s | 4.4 ms | 18x |
| [`grow_lists.py`](grow_lists.py) (from 0.0.4) | Pairs of points closer than a radius, from a grid that is a list of lists built by `append`, returned as a list of `(i, j)` tuples | How many points share a cell, and which pairs are close, depends on the data | 0.42 s | 77 ms | 5x |
| [`sorting.py`](sorting.py) (from 0.0.4) | The busiest hour of a day of requests: each hour's intervals built as a list of tuples, sorted with `key=` and merged, the hours ranked with `sorted(..., reverse=True)` | Merging depends on the order and on how far the last interval reached | 0.51 s | 33 ms | 15x |
| [`sets.py`](sets.py) (from 0.0.4) | Islands on a sparse map, the land a set of `(x, y)` cells, flood-filled with a frontier set | Each island's extent depends on the cells found so far | 0.17 s | 64 ms | 3x |
| [`records.py`](records.py) (from 0.0.4) | Runs above a level in a signal: a list of dataclass instances passed in and appended to, a list of dicts returned | Each run depends on where the last one ended | 40 ms | 3.0 ms | 13x |
| [`whole_array.py`](whole_array.py) (from 0.0.4) | 1,000 damped oscillators stepped with `a = -k * x - c * v + drive` on whole arrays | numpy does this, a call and a temporary array per operator; compiled, one pass per line | 0.010 s | 2.3 ms | 4x |

One run of each on arm64 Linux, on a busy machine, so expect faster and somewhat different numbers. The
first call for a set of argument types compiles, which took under a second here; these times exclude it,
and the scripts print it. `asian_option` gains least because most of its time is in `log`, `cos` and
`exp`, which are the same library functions either way.

`config_dict` gains less than most because its loop is only float arithmetic with no indexing, which CPython does comparatively well, and it's at the hardware's limit: hand-written C takes the same time per step through cimo and through `gcc -O3`.

Four more show behaviour rather than speed:

- [`threads.py`](threads.py): compiled code releases the GIL. Four threads ran 2.3x to 3.8x faster than the same
  calls one after another, depending on what else the machine was doing; plain Python threads gained
  nothing. Then one loop split into four parts over a `ThreadPoolExecutor`, each a call on its own range of
  the same array: 3.9x, with the same bits.
- [`fallback.py`](fallback.py): a function that doesn't compile runs as Python with a `FallbackWarning`.
  Shows the warning, `f.preflight()`, `fallback=False`, and a rewrite that compiles.
- [`hot_lines.py`](hot_lines.py): insertion sort, to run under `scramjit preflight --counts`: 100 million moves on one line, and exactly how often each line and branch ran.
- [`partly_compiled.py`](partly_compiled.py): a function that handles text can't compile whole, but its loop over numbers still does: 96x, with the rest as Python. Shows the warning and `f.preflight()` line by line.
