# scramjit: requirements

**A Python JIT written in pure Python,** for the loops the interpreter is slowest at. Built on
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
import scramjit

@scramjit.jit
def saxpy(a: float, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]
```

The decorator is the whole user-facing API. Options (bounds checks, cache control) are keyword
arguments to it. The exact name (`@scramjit.jit` or bare `@scramjit`) is open. The undecorated
function stays reachable for debugging.

## Compilation model: unboxed

Generated code never touches a `PyObject`. Arguments are unboxed at the call boundary (Python
scalars into C scalars, numpy arrays into pointer + shape + strides, without copying), the body
runs on C values only, and the result is boxed on return.

Anything that can't be unboxed is a compile error that points at the Python line and says why.
There is no fallback to running on Python objects: it would be slow, and it can't work without
libpython (cimo links none, and Pyodide has none to link).

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
  `float` → `float`); types that don't unify are a compile error.
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

- `int` is int64 unless narrowed; overflow wraps. Open: a checked mode.
- `//` and `%` round toward negative infinity, as in Python, not truncate as in C.
- Negative indices count from the end, as in Python.
- Bounds checks: open (on by default with an opt-out, or off by default).
- Float division by zero: open (raise as Python does, or return inf as C does).
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

## What scramjit needs from cimo

Planned cimo features; scramjit is their first consumer:

- Compile a C string in memory and return callables, with no files involved.
- `#line` directives honored in diagnostics and debug info.
- A defined calling convention for numpy buffers and scalars.
- Low compile latency and a cache key scramjit can use.
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
and in Pyodide, a second process reuses the disk cache, and every out-of-subset construct gives an
error pointing at the Python line.

## Risks

- **First-call latency** in pure Python decides whether library authors can use it. Measure cimo's
  compile time on generated C early.
- **Performance without vectorization** is well behind LLVM-based JITs on simple loops until cimo's
  auto-vectorizer exists.
- **Semantic drift** from CPython in edge cases (overflow, division, indexing, narrowing) breaks
  the "it's still Python" promise. Differential testing is the guard.
- **Scope creep** into growable lists, dicts and classes pulls in a runtime. Hold the line until
  the numeric subset is solid.
