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
- **Prompt for Agents** (`docs/agent-prompt.txt`) reads as for the released product, but
  agents act on it, so every instruction must work today: it gives the GitHub install until the
  PyPI release, and only what actually compiles. Update it when the subset, CLI or install changes.

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

The decorator is the whole user-facing API. Options (`boundscheck`, `cache`, `fallback`, `parity`)
are keyword arguments to it. It is named `scram` (to scram: to go, fast) and imported as
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
- A variable negative exponent in `n ** k` raises `ValueError`, where Python returns a float.
- A negative float to a fractional power raises `ValueError`, where Python returns a complex.
- **`math` functions are bit-exact with CPython on every platform**, Pyodide included. Use the
  same code CPython runs:
  - functions CPython implements itself (`hypot`, `dist`, `fsum`, `isqrt`, `gamma`, `lgamma`,
    ...) are ported into scramjit's runtime, with CPython's license notice;
  - functions CPython takes from the platform C library (`sin`, `atan2`, `exp`, `pow`, ...) call
    that same library: glibc, Apple's libm or the Windows CRT natively, and emscripten's in Pyodide,
    where `ctypes.CDLL(None)` gives the addresses (table indices), so nothing needs porting;
  - CPython's own handling around those calls (NaN, infinities, domain and range errors raising
    `ValueError` or `OverflowError`) is reproduced exactly.

  Differential tests compare every supported function against CPython on every platform, with
  zero differing results as the bar.
- **Parity can be traded for speed** with `@scram(parity=False)`, where matching CPython costs
  speed: float `/`, `//`, `%` by zero and `math` functions give IEEE inf and nan instead of
  raising (as numpy's functions do), `math` functions are cimo's C library and `hypot` is C's,
  and `int / int` and int-float comparisons use plain double arithmetic. Integer `//` and `%` by zero still
  raise (in C they trap). Later: FMA contraction, faster math functions, reassociation for
  vectorization. Parity stays the default; the code generator's job is to make it cost nothing.
- **numpy scalars are numpy's, not Python's.** A value read from an array is a numpy scalar:
  dividing it by zero gives inf, nan or 0 instead of raising, and int with np.int64 gives
  np.int64 (NEP 50). Types record numpy-ness; a variable holding both kinds (`total = 0.0`, then
  `total += x[i]`) carries a runtime flag. Narrow dtypes (float32, uint8) follow NEP 50 too
  (M2).

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

Study existing Python compilers and JITs for their strategies. Record here what we adopt, change
or reject and why. "Better" means measurable: faster code, faster compiles, clearer errors or
fewer surprises.

Findings from an LLVM-based JIT (studied for M0):

- **Dispatch:** adopt interned type codes, fast paths for scalars and arrays, and a cache from
  argument fingerprint to signature. Reject a "best compatible signature" search: exact-key
  lookup, compile on a miss. Memoize failed compiles; one global compile lock.
- **Inference:** adopt SSA plus a widen-only type lattice with an explicit Python-numeric join;
  report errors only after a fixed point. Use a worklist (they re-run every constraint per round).
- **Builtins:** they keep `range`, `enumerate` and `zip` as iterator structs and rely on LLVM to
  remove them. cimo can't, so rewrite them into counted loops in our IR. Write builtins as direct
  lowering rules, not source-level overloads compiled through the pipeline (a compile-latency
  cost there).
- **Comprehensions:** they fuse only `np.array([... for ...])` and don't support generator
  expressions. A producer node inlined into its consumer (sum, any, list build, for) is ours.
- **Arrays:** adopt pointer + shape + strides with the layout (C, F, any) in the signature.
  Their bounds checks are off by default, per access and never hoisted; ours stay on and are
  removed by range analysis and hoisting. Never emit arithmetic that is undefined on overflow.
- **Errors:** adopt their convention: an int status, results and exception info through out
  parameters, a static table of (exception, message, Python line) per raise site.
- **Loop lifting:** adopt their candidate rules (one entry; all exits to one block; no `return`
  or `yield` inside) and live-in/live-out computation. Change: the rest runs in the real
  interpreter (theirs compiles it to PyObject C-API calls, "little to no speedup"); lifted loops
  stay cacheable (theirs can't be cached). They removed fallback after users mistook slow paths
  for compiled code; our warning, preflight and `fallback=False` exist for exactly that.
- **Caching:** adopt atomic writes and the target in the key. Also key on transitive callees,
  captured globals and the scramjit and cimo builds; no pickle.
- **Diagnostics:** adopt per-line annotation and errors that list why each option failed. Name
  the user's variables, never IR temporaries; one primary reason per line.
- **Avoid their deviations from Python** where cheap: missing `UnboundLocalError`, frozen
  globals, copying nested containers in and out (keep copy-in/out to flat numeric lists, and
  check every element, not the first).

## Kernels and measurements (M0)

`kernels/`: Gray-Scott reaction-diffusion, flood fill and viewshed as plain Python, the C
scramjit should generate for each in `kernels/c/`, and `kernels/check.py`, which requires the C
(compiled by cimo) to give bit-identical results and the same exceptions at the same lines, after
the same writes, as CPython.

- **Subset fit:** Gray-Scott and viewshed fit as written. Flood fill fits once its stack is a
  preallocated array; the usual version (a list with `append`/`pop`) is partly compiled. Growable
  lists will be the commonest fallback; keep them out until the numeric subset is solid.
- **Speed** (arm64 Linux, best of 5 against one CPython run): Gray-Scott 96×96 about 480×, flood
  fill 256×256 about 75×, viewshed 64×64 about 20× (math calls and kept bounds checks in its
  inner loop). Cold compile 0.07 to 0.15 s per kernel.
- **FMA:** cimo contracts multiply-adds by default, and that changes Gray-Scott's results. With
  contraction off, all three match bit for bit. scramjit must compile with contraction off.
- **Semantics the C must reproduce:** Python's `round` is round-half-even (C `rint`, not C
  `round`); storing an int into a narrower array raises `OverflowError` in numpy, so it is range
  checked (once, if loop-invariant); a hoisted bounds check needs loop versioning, so the
  `IndexError` comes after the same writes as in Python; when branches merge, each statement
  keeps its own line; `//` and `%` on values read from arrays need the sign fix-up, while `%` on
  a loop index lowers to a compare.
- **Math functions:** cimo's C library differs from CPython's in the last bit for many inputs:
  `atan2` 18%, `hypot` 14% (CPython implements `hypot` itself), `sin` and `cos` 3%, `exp` and
  `pow` under 0.1%, `sqrt` and `log` never (100,000 random inputs). Decided: math functions are
  bit-exact too; see "Semantics".
- **Calls between compiled functions** aren't in the subset yet; kernels with a helper per ray or
  cell need them.

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
- Code as fast as gcc -O3 on scramjit's kernels. `bench/gcc.py` compiles scramjit's generated
  C and the hand-written C both ways; on the same C cimo is 1.3x (gray_scott), 1.9x (flood_fill)
  and 2.3x (viewshed) slower than gcc. Causes found: `rint` is a call (2.5x), the bundled `hypot`
  is 1.5x glibc's, no common subexpression elimination. Reported in cimo's `TODO.md`.
- Fused multiply-add contraction controlled per compile (today only the process-wide `CIMO_FMA`
  environment variable); scramjit needs it off, other users of cimo may want it on.
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
subset is amended here with the reason, and the findings are recorded here. **Done 2026-11-01**: see "Kernels and
measurements (M0)" and "Prior art"; one decision is open there (math functions).

**M1: scalars, 1-D arrays, `range`.** `ast` → C for scalar arithmetic, `range` loops, `if`,
`while`, and 1-D numpy arrays; annotations as contracts; `int64`/`float64` only; in-memory cache.
Done when the M0 kernels that need only these features run, match CPython on random inputs, and
are benchmarked against pure CPython. **Done 2026-11-01** (see "M1 findings").

**M1 findings.** Differential tests (`tests/`) plus 2000-program random expression runs (about
9,500 checks, numpy scalars included) found zero mismatches. CPython quirks reproduced: `float *
int` with the int equal to -1 is computed as a negation (a NaN's sign flips; other ints and
numpy scalars don't). cimo doesn't inline `static inline` functions (5.15 vs 3.07 ns on `ema`),
so hot helpers were macros, until cimo learned to inline (now `static inline`, no slowdown). Facts from unit-step `range` loops (index non-negative, below
`len(a)`) drop negative-index adjustment and bounds checks, and loop versioning (one test before
the loop, checked copy otherwise) removes the rest without changing where an `IndexError` is
raised. With those, parity and bounds checks cost nothing measurable on the M1 kernels:
`bench/m1.py`, arm64, ns per element, all modes within 3%:

| kernel | CPython | scramjit | speedup |
|---|---|---|---|
| ema (recurrence) | 260 | 3.10 (hand C 3.12) | 84x |
| ratio_sum (divisions) | 155 | 1.03 | 150x |
| count_below (int vs float compare) | 74 | 1.66 | 45x |
| prefix_max | 99 | 0.69 | 143x |
| edit_distance (DP) | 482 | 4.28 | 113x |
| collatz (int // and %) | 42 | 2.39 | 17x |

No M0 kernel uses only 1-D arrays, so M1 was measured on these.

**M2 progress (2026-11-01).** Built and differentially tested: loops over arrays, lists,
`enumerate`, `zip`, `reversed` (nested, tuple targets) as one counted loop; `sum`, `min`, `max`,
`any`, `all` over iterables and comprehensions, fused, each following CPython's `bltinmodule.c`
(sum's int, Neumaier-compensated float and generic phases; `any`/`all` short-circuit over
generators but not list comprehensions); tuples of numbers (swap assignment, returns, arguments);
lists that never change size (literals, `[x] * n`, single-loop comprehensions) as heap arrays
with Python's list errors; `math` functions bit-exact, calling the host libm CPython calls,
wrapped in `mathmodule.c`'s logic. Findings: `sum()` of floats with the default int start
returns the int 0 when empty, so it falls back with a hint to write `sum(..., 0.0)`; numpy ints
have no `__floor__`, so `math.floor(np.int64(x))` goes through float (precision lost above 2**53)
and `math.trunc` of one raises. Not yet: arithmetic on narrow dtypes (NEP 50), value-range
narrowing, `map`/`filter`.

**M0 kernels compiled (2026-11-01).** With C-contiguous n-D arrays (numpy's per-axis IndexError),
`a.shape`/`ndim`/`size`, narrow dtypes loaded, compared and stored (a Python int that doesn't
fit raises numpy's NEP 50 OverflowError, checked after the index, as numpy does; numpy ints
cast and wrap) and `round()`, the three M0 kernels compile unchanged and match CPython bit for
bit, errors included (`tests/test_kernels.py`); their generated C has no iterator or boxing
code. Range facts now name (array, axis): extents read from `a.shape` into variables assigned
once bound loops, `x % n` indexes in bounds, and `(i ± 1) % n` inside such a loop lowers to a
compare. `bench/kernels.py`, arm64:

| kernel | CPython | scramjit | hand C | vs CPython | vs hand C |
|---|---|---|---|---|---|
| gray_scott 256x256 | 119 ms | 0.51 ms | 0.20 ms | 233x | 2.5x |
| flood_fill 512x512 | 244 ms | 4.7 ms | 2.5 ms | 52x | 1.9x |
| viewshed 96x96 | 119 ms | 13.2 ms | 6.7 ms | 9x | 2.0x |

Against gcc (`bench/gcc.py`): scramjit's generated C compiled by gcc -O3 runs as fast as the
hand-written C compiled by gcc (gray_scott 0.17 vs 0.17 ms, flood_fill 1.9 vs 1.85 ms,
viewshed with `parity=False` 3.4 vs 2.5 ms), so what is left is cimo's backend, reported to cimo.
Getting there took: wraparound indices hoisted once per loop iteration, `x % n` indexes covered
by loop versioning (`n <= len` tested once), one-sided wraparound compares, facts from guards
(`if y > 0 and ...` makes `y - 1` nonnegative) and from assignments (every `x = p % w` puts x in
`[0, w)`), for numpy ints too, and one division per `//` or `%`.

viewshed's gap is parity: CPython-exact `math.hypot` (vector_norm through the host table) and
math error checks; with `parity=False` it runs in 7.9 ms (15x CPython, 1.2x hand C). gray_scott
and flood_fill are the same speed either way: their gaps are code generation (gray_scott's
wraparound compare repeated per access where the hand C computes it once per row).

**First run in Pyodide (2026-11-01).** The site's live demo (`docs/lab/`, three kernels:
hydraulic erosion, a viewshed beacon, thermal landslides) loads Pyodide, numpy, cimo
(`cimo onefile`) and scramjit in the page and compiles with full parity: erosion runs 170 to
200x faster than Pyodide's CPython (about 2,400 against 400,000 droplets per second), compiling
in about 2.5 s. `tools/sync_site.py` refreshes the copies the site loads. Exact `math.hypot`
is slow in WebAssembly (cimo's `fma` is software there), so the beacon uses `math.sqrt`.
All of `math` compiles in Pyodide too, bit-exact: `ctypes.CDLL(None)` gives libm's addresses in the
main module, which in WebAssembly are table indices, so kernels call the very functions `math` calls
(`tests/test_math_cases.py`, run natively and through `tools/run_pyodide.mjs`). A pure `sin`, `log`,
`exp` loop gains 12x there: most of its time is in libm.
What CPython implements itself (`gamma`, `lgamma`, `remainder`, `fsum`, `comb`, ...) is in
`scramjit/_mathlib.py`, included only in kernels that use it, so it adds nothing to other compiles.
`frexp` is implemented from the float's bits, because calling a libm function with a pointer traps in
WebAssembly (cimo's pointers don't match the signature; a table call with doubles and ints is fine).
Types that depend on the data still fall back: float `sumprod` (the int 0 when empty) and float
`prod` without `start=1.0`.
At compiled speed, erosion and landslides reach their steady state (a flat plain) in under a
second, so the demo adds clock-based uplift toward the starting terrain (2 s time constant) and
gentler droplets: the terrain shows the balance, and more work per second carves deeper.
Compiling takes 1.5 to 3 s and blocks the thread it runs on, so the demo runs Pyodide in a
Web Worker (`docs/lab/worker.js`): the page only draws and handles input, and gets each frame's
arrays as transferred buffers. A running compile can't be cancelled: interrupting Pyodide needs
SharedArrayBuffer, so cross-origin isolation headers, which GitHub Pages can't send. Queued edits
collapse into one compile of the newest code instead, and stale results are dropped. The demo now
has four tabs (terrain, Mandelbrot dive, falling sand, flocking); in this setup they run about
45x, 90x, 180x and 240x faster than Pyodide's CPython. A native run doesn't prove a kernel compiles
in Pyodide (see TODO on `math`): test there. Work and drawing stop while the demo is out of view.

**M2: builtins, lists, narrowing.** Lowering rules for `enumerate`, `zip`, `map`, `filter`,
comprehensions and the reductions; fixed-size lists as arrays; narrow types from annotations,
dtypes and range analysis. Done when each rule has differential tests, the generated C for
each M0 kernel contains no iterator or boxing code, and every supported `math` function gives
results identical to CPython's, natively and in Pyodide.

**M3: n-D arrays, errors, disk cache, Pyodide.** Done when all M0 kernels run in native CPython
and in Pyodide, a second process reuses the disk cache, and every out-of-subset construct is
reported at its Python line (fallback warning, preflight, and `CompileError` under
`fallback=False`). Whole-function fallback lands in M1; it is the original function.

**`**`.** Python's result type depends on the values in two cases: an int to a negative power is a
float, and a negative float to a fractional power is a complex. Both compile, to an int and a float,
and raise `ValueError` in the rare case: decided 2026-11-01, because these are too common to run as
Python and an error is never a wrong answer. A constant exponent, or a nonnegative constant base, rules
the case out. Rerunning or resuming in Python was considered and left for later. numpy's rules have one
type whatever the values, so `a[i] ** 0.5` gives `nan` as numpy does. This and every other difference
from CPython are in `docs/docs/content/gotchas.md`; keep that page complete. `parity=False` gives C's `pow`, and `x * x` for small powers. `x ** 2` calls libm's
`pow` as CPython does, which is why `v ** 2` over an array gains about 18x where `v * v` gains far more.

**Call boundary.** Calls were 3.2 µs with numbers and 6.6 µs with an array, half of it scramjit's wrapper.
Each signature now gets a small generated function: checks decided by the argument types run once when
it's made, and anything it can't vouch for at call time (an int past 64 bits, another dtype or layout, an
empty array) takes the general path, so it changes only speed. Status buffers come from a pool, not
thread-locals, because a signal handler can re-enter between the kernel returning and its result being
read. Calling `Kernel.address` directly would skip cimo's wrapper, but its trampoline is what lets `malloc`
and the stack guard find their per-thread context, so that isn't safe; the rest needs a cimo fast path.

**M3 progress.** The disk cache is cimo's: scramjit's C is deterministic, so a second process finds
it (Gray-Scott first call 0.53 s, then 0.06 s, mostly preprocessing the prelude); no second cache
needed. Preflight is built (`scramjit/_preflight.py`): static analysis sets each failing statement
aside and retries, so it reports every line, not only the first, though a line can fail only because
one it depends on did. Bounds notes come from the compiler, not from scanning the C, since a versioned
loop's slow copy always holds checks that don't run in the fast one.
The command line (`scramjit/cli/`) is built like cimo's: its parser, terminal and completion are copied
rather than imported, because they aren't cimo's public API. `--run` reports on stderr so the program
keeps stdout.

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
