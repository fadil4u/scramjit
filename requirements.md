# scramjit: requirements

**Python at native speed.** A JIT written in pure Python, for the loops the interpreter is slowest at. Built on
[cimo](https://github.com/farizrahman4u/cimo). PyPI name `scramjit` is reserved (placeholder
0.0.1, published 2026-11-01).

This is the living spec: what to build, why, what has been measured, and in what order.

## Engineering rules

- **Pure Python** at runtime: standard library plus `cimo` (itself pure Python, standard library
  only). No native extensions. Supports what cimo supports: CPython 3.9+ and Pyodide.
- **Use cimo's public API only.** Anything scramjit needs from cimo becomes a general cimo
  feature, not a private hook.
- **Never change results.** A compiled function returns what CPython returns for the same
  function undecorated, within the documented semantics below. Optimizations (narrow types,
  arrays for lists, fused loops) apply only when proven safe.
- **Differential testing from day one:** every compiled function is checked against CPython
  running it undecorated, including random programs within the subset.
- **Benchmarks:** follow cimo's rules (interleave configurations, best of several runs, mains
  power only). Compare against pure CPython and against other Python JITs where they run.
- **Record findings here, concisely.** This file is a spec, not a log.
- **Site and docs** (`docs/`, served by GitHub Pages) are written as for the released product:
  present tense, no roadmap or status notes. They follow this spec; where it is open, they show
  Python's behavior as the default.

## Pitch

"Native speed for your hot loop, with no compiler, no LLVM and no binaries." Works anywhere
Python runs, including the browser. The decorated code is plain Python: without scramjit it still
runs, slowly, in CPython, and stays debuggable and testable as Python.

**Where it shines:** code that doesn't vectorize and is held back by the interpreter and the GIL:
recurrences, data-dependent branches, flood fills, dynamic programming, simulations, scans over
byte buffers. Every iteration pays the interpreter in CPython, and threads queue on the GIL;
compiled, the loop runs as C with the GIL released. **Also a fit:** vectorizable code that must run
where only pure Python is allowed or available. The supported subset is numeric (scalars, arrays,
tuples, fixed lists), but the target is any loop-heavy code, not only "numeric Python".

## Users

In priority order:

1. **Python in the browser** (Pyodide, JupyterLite, marimo WASM, PyScript). Existing Python
   compilers don't run there; a slow loop has no fix today.
2. **Plugin authors in embedded Pythons** (QGIS, Blender, Houdini, Maya, Nuke, FreeCAD, KiCad),
   where compiled dependencies are banned or painful.
3. **Platforms existing JITs haven't reached yet:** new CPython releases, free-threaded builds, Windows
   arm64.
4. **Library authors** who want fast hot paths in a pure-Python package, with no wheel matrix.
   The largest group if first-call latency and caching are good enough.

Not for: ML (PyTorch/JAX); code that is already vectorized numpy or pandas in an environment
where numpy runs (it is already native loops); users happy with an existing JIT on a standard
desktop or server.

## Competition

- **LLVM-based JITs:** mature, vectorize via LLVM, often parallel and GPU support. Need LLVM
  wheels; don't run in Pyodide; lag new CPython releases.
- **Pythran, Cython (pure-Python mode), Shed Skin:** need a C/C++ toolchain on the target.
- **Mojo:** a separate language with Python-like syntax and its own toolchain; not Python. Don't
  position against it.
- **MicroPython's `@micropython.viper`:** the same idea for microcontrollers; not a competitor.

## API: a decorator

```python
from scramjit import scram

@scram
def saxpy(a: float, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]
```

The decorator is the whole user-facing API. Options (`boundscheck`, `cache`, `fallback`) are
keyword arguments to it. It is named `scram` (to scram: to go, fast) and imported as
`from scramjit import scram`; a callable module (`@scramjit`) was rejected because mypy and Pyright
report "module not callable" for every variant of the trick. The undecorated
function stays reachable as `f.__wrapped__` (the `functools` convention), for debugging and
comparing results.

## Compilation model: unboxed

Generated code never touches a `PyObject`. Arguments are unboxed at the call boundary (Python
scalars into C scalars, numpy arrays into pointer + shape + strides, without copying), the body
runs on C values only, and the result is boxed on return.

Compiled code never falls back to Python objects *inside* itself: that would be slow, and it can't
work without libpython (cimo links none, and Pyodide has none to link).

## Fallback: compiled, partly compiled, Python

What can't be compiled runs as plain Python instead, at two granularities:

- **Loop lifting.** Loops that fit the subset are lifted out of a function that otherwise doesn't
  and compiled on their own; the interpreter runs the rest and calls the compiled loop, passing in
  the variables it reads and taking back the ones it writes. Those must be unboxable.
- **Whole function.** If nothing compiles, or a call's argument types can't be unboxed (a `str`,
  say), that function and signature run as the original Python (`f.__wrapped__`).

So each decorated function, per signature, is in one of three states, shown everywhere with the
same colours: **compiled** (green, everything native), **partly compiled** (amber, some loops
native) or **Python** (red, nothing native). Fallback can't change results: the fallback is
CPython itself.

- Each fallback emits `scramjit.FallbackWarning` once per function and signature, naming the
  line and the reason. Standard `warnings` filters silence it or make it an error
  (`-W error::scramjit.FallbackWarning`).
- `@scram(fallback=False)` makes anything but compiled a `scramjit.CompileError` at the line.
  For code that must be native, and for CI.

## Builtins are lowering rules, not functions

No builtin exists at runtime as a callable. Each is a rule in the compiler that emits plain C:

- `for i in range(a, b, s)` → `for (i = a; i < b; i += s)`. A non-constant step gets the sign
  check Python implies; no range object exists.
- `enumerate`, `zip`, `reversed` over arrays and ranges → index loops.
- `map`, `filter`, generator expressions and comprehensions → one fused loop with no
  intermediate iterators. A result is materialized only when stored; `sum(x*x for x in a)`
  reduces in place.
- `len`, `sum`, `min`, `max`, `abs`, `any`, `all`, `math.*` → inline code or C math calls.

Iterator protocols (`__iter__`, `__next__`) never appear in generated code.

## Types

- **Inference** over the whole function, seeded by the argument types seen at call time and by
  type annotations. A variable that takes several types unifies by Python's rules (`int` and
  `float` → `float`); types that don't unify keep that code
  in Python (a fallback; a `CompileError` under `fallback=False`).
- **Annotations** are used, not ignored: as contracts checked at the call boundary, and as
  information for narrowing. Accepted: `int`, `float`, `bool`, numpy dtypes (`np.int8`,
  `np.float32`). Open: scramjit's own spellings for sized scalars and arrays (e.g. `i32`,
  `f32[:, :]`).
- **Narrowest type that is proven sufficient.** The compiler picks the smallest type that holds
  every value a variable can take, from annotations, argument dtypes and value-range analysis
  (a loop index bounded by an array length, `x % 256`, a `uint8` array element). Without proof,
  integers are `int64` and floats `float64`. The payoff is mainly in arrays and SIMD (smaller
  elements, more lanes per vector); narrowing scalars rarely helps on 64-bit CPUs.

## Containers

- **Tuples** → C structs.
- **Lists that never change size** become fixed-size C arrays: no `append`, `pop`, `extend`,
  `insert`, `del` or slice assignment on them, and no escape into code that might. Stack-allocated
  when small, constant-sized and not returned; heap otherwise.
- **Lists passed in from Python:** homogeneous numeric lists are copied into an array at the call
  boundary and copied back if mutated. That's O(n) per call, so large data should be numpy arrays.
- **Later:** growable lists, dicts, sets and strings as typed C containers. Not in the first version: each needs allocation and a runtime.

## Semantics

Each decision is explicit and tested against CPython:

- `int` is int64 unless narrowed; overflow wraps. No checked mode: numeric code rarely nears
  `2**63`, and checking every operation would cost every loop.
- `//` and `%` round toward negative infinity, as in Python, not truncate as in C.
- Negative indices count from the end, as in Python.
- Bounds checks are on by default and raise `IndexError` as Python does; opt out per function with
  `@scram(boundscheck=False)`.
- Division by zero, integer or float, raises `ZeroDivisionError` as Python does.
- Float evaluation order follows Python; no reassociation, no FMA contraction unless opted in.

## Execution model

On each call the decorator reads the argument types (Python type, dtype, number of dimensions,
layout), generates C specialized to them, compiles it with cimo, and caches the result by
signature, in memory and on disk.

Compiled code releases the GIL while it runs (as cimo kernels do), so Python threads calling a
compiled function run in parallel.

## Architecture

Python source → `ast` → type inference → typed IR → lowering (builtins, containers, narrowing)
→ C source with `#line` directives back to the Python file → cimo → machine code in the process
(native) or WASM (Pyodide).

Generating C, not cimo's IR, keeps the output readable and debuggable and reuses cimo's tested
frontend and loop optimizer. The C that scramjit emits should be the analyzable kind cimo
optimizes best (canonical counted loops, affine indexing, `restrict` where no aliasing is proven).

## Prior art

Study existing Python compilers and JITs for their strategies: type inference and typing of
builtins, lowering of loops, iterators and comprehensions, array analysis and loop fusion, the
compiler pipeline, dispatch and the on-disk cache, and error reporting. Record here what we
adopt, change or reject and why. "Better" means measurable: faster code, faster compiles,
clearer errors or fewer surprises.

## Visibility and profiling

Fallback makes speed silent to lose, so the state of every function, and of every line in it, must
be easy to see. Tools, all pure Python, behind one `scramjit` command (also `python -m scramjit`):

- **`scramjit preflight PATH`:** static, before running. Every `@scram` function under the path
  with its state (compiled, partly compiled, Python), and line by line what stays in Python and why.
  Annotations are used; unannotated arguments are assumed numeric or arrays.
- **`scramjit preflight --run SCRIPT`** (or `--run -m MODULE`, such as `-m pytest`): runs it and
  reports what actually happened, with the real argument types: per function and signature, state,
  calls, compile time or cache hit, time in native code and in Python, and the native share of the
  whole run (the coverage number). Ends with hints. Catches the type-driven fallbacks the static
  mode can't see; running the test suite under it is the CI check.
- Both modes exit non-zero on anything but compiled (`--allow-partial` accepts partly compiled).
- **`scramjit preflight PATH:FUNCTION`** (or `pkg.module:function`): the `f.preflight()` report for one
  function, from the command line. Static by default (one assumed signature); with `--run` it shows
  each signature compiled during the run. The target comes before `--run`, which takes the rest of
  the command line.
- **`f.preflight()`:** the same per-function report in a REPL or notebook: the Python source annotated per line: state (native or Python, and why),
  inferred and narrowed types, bounds checks proven, hoisted or kept, loops cimo optimized or not,
  copies at the call boundary. Ends with hints.
- **`f.signatures`, `f.c()`, `SCRAMJIT_LOG=1`:** the specializations and their states, the
  generated C, and one log line per compile, cache hit or fallback.

Hints must be actionable and name the line: what stops a loop compiling and how to rewrite it,
annotate an argument that arrives as several types, pass a numpy array instead of a list (copied per
call), make an array contiguous, and when a bounds check stays in a hot loop, what would let it go.

## What scramjit needs from cimo

Planned cimo features; scramjit is their first consumer:

- Compile a C string in memory and return callables, with no files involved.
- `#line` directives honored in diagnostics and debug info.
- A defined calling convention for numpy buffers and scalars.
- Low compile latency and a cache key scramjit can use.
- The per-loop optimization report (`cimo report`: optimized or not, and why) as data, mapped to
  `#line`, for the per-line preflight report.
- Symbols for profilers (perf map, jitdump) so `perf` names compiled functions.
- Possibly: a way to state no-alias guarantees scramjit can prove (`restrict`).
- Auto-vectorization (cimo's SIMD v2). C users can write vector types by hand; scramjit users
  write plain loops. Until it lands, claim "C speed for scalar loops", not parity with LLVM-based JITs.

## Milestones and acceptance criteria

**M0: subset check and prior-art study.** Write the planned cimo demo kernels (reaction-diffusion,
flood fill, viewshed) as plain Python, and hand-write the C scramjit should generate for each.
Study existing compilers' typing and lowering (see "Prior art"). Done when every kernel fits the subset or the
subset is amended here with the reason, and the findings are recorded here.

**M1: scalars, 1-D arrays, `range`.** `ast` → C for scalar arithmetic, `range` loops, `if`,
`while`, and 1-D numpy arrays; annotations as contracts; `int64`/`float64` only; in-memory cache.
Done when the M0 kernels that need only these features run, match CPython on random inputs, and
are benchmarked against pure CPython.

**M2: builtins, lists, narrowing.** Lowering rules for `enumerate`, `zip`, `map`, `filter`,
comprehensions and the reductions; fixed-size lists as arrays; narrow types from annotations,
dtypes and range analysis. Done when each rule has differential tests and the generated C for
each M0 kernel contains no iterator or boxing code.

**M3: n-D arrays, errors, disk cache, Pyodide.** Done when all M0 kernels run in native CPython
and in Pyodide, a second process reuses the disk cache, and every out-of-subset construct is
reported at its Python line (fallback warning, preflight, and `CompileError` under
`fallback=False`). Whole-function fallback lands in M1; it is the original function.

**M4: loop lifting.** Compile the loops that fit inside functions that don't. Done when a function
mixing Python-only setup (dicts, strings) with a numeric loop is partly compiled, matches CPython,
and preflight (static, `--run` and `f.preflight()`) shows the state per line.

## Risks

- **First-call latency** in pure Python decides whether library authors can use it. Measure cimo's
  compile time on generated C early.
- **Performance without vectorization** is well behind LLVM-based JITs on simple loops until cimo's
  auto-vectorizer exists.
- **Semantic drift** from CPython in edge cases (overflow, division, indexing, narrowing) breaks
  the "it's still Python" promise. Differential testing is the guard.
- **Silent slowness** from fallback: a decorated function that quietly runs as Python. The warning,
  preflight (static and `--run`) and `fallback=False` are the guard.
- **Scope creep** into growable lists, dicts and classes pulls in a runtime. Hold the line until
  the numeric subset is solid.
