# scramjit: requirements

**Python at native speed.** A JIT written in pure Python, for the loops the interpreter is slowest at. Built on
[cimo](https://github.com/farizrahman4u/cimo). Published on PyPI as `scramjit`.

This is the living spec: what to build, why, what has been measured, and in what order.

## Engineering rules

- **Pure Python** at runtime: standard library plus `cimo` (itself pure Python, standard library
  only). No native extensions. Supports what cimo supports: CPython 3.9+ and Pyodide.
- **Compiled code is a cache** (decided 2026-11-02). It's built on the machine that runs it, from the Python source, and can always be
  built again. It is never distributed, in a package, a wheel or a single HTML file: an app carries its source and the compiler, whose
  340 KB compressed and first-launch compile are worth keeping "no binaries" (measured: about 3% of a 12.2 MB single-file app, 0.7 s of a 3.5 s launch). The LLM demo keeps the last 8 compiled sources in the browser's
  cache, about 360 KB, dropped when Python, cimo or scramjit changes: an edit made on an earlier visit took 0.33 s rather than 5.7. Turning scramjit on
  gains less, since the page must first wait for plain Python's current token, several seconds.
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

Not for: code that is already vectorized numpy or pandas in an environment
where numpy runs (it is already native loops); users happy with an existing JIT on a standard
desktop or server.

## Competition

- **LLVM-based JITs:** mature, vectorize via LLVM, often parallel and GPU support. Need LLVM
  wheels; don't run in Pyodide; lag new CPython releases.
- **Pythran, Cython (pure-Python mode), Shed Skin:** need a C/C++ toolchain on the target.
- **Mojo:** a separate language with Python-like syntax and its own toolchain; not Python. Don't
  position against it.
- **MicroPython's `@micropython.viper`:** the same idea for microcontrollers; not a competitor.

Measured (`bench/compare.py`, arm64, 2026-11-07; `TODO.md` #18): on par with Numba, typed Cython and gcc on stencils,
flood fill, dot, axpy and a histogram with exact floats (the histogram after one-test `int()` and range facts from an
`if`'s comparisons: 2x to 1.06x); behind on exact `hypot`/`round` (viewshed 5.6x), reordered float sums (Numba
fastmath 3x) and first-call compile time (2 to 8x). Ahead on Python
data: Numba doesn't compile a dict of settings or a list of dataclasses, and Cython as written doesn't speed them up.

## API: a decorator

```python
from scramjit import scram

@scram
def saxpy(a: float, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]
```

The decorator is the whole user-facing API. Options (`boundscheck`, `cache`, `fallback`, `parity`, `counts`)
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
- **Lists passed in from Python (built 2026-11-03, from 0.0.4):** a list of only ints, floats or bools is copied into an
  `array.array` at the call, and if the function stores into it, the elements whose value changed are copied back at
  every exit, so the others keep their identity. The same list twice runs as Python. A returned local list's memory is
  handed to the caller, copied into a list in C (`memoryview.cast().tolist()`) and released; returning the argument
  returns it. Copies cost about 10 ms per 200,000 floats, the type check 2 (`set(map(type, x))`).
  A rectangular list of lists is an array of Python numbers (`T.is_pylist`): one buffer, rows as views, so a row is a
  reference as in Python; `m[i, j]`, slices and replacing a row stay Python. A changed row is copied back in one go.
  `[[c] * w for _ in range(h)]` (and the comprehension form) is rewritten to one made block (`Tables`), its rows separate
  lists as in Python; only plain row values and sizes, since Python evaluates them once a row.
- **Python data structures, nested, as much as possible** (decided 2026-11-01): dicts, lists, tuples, classes
  and enums compile when they are **statically sized** (nothing resizes them at runtime) and **statically
  typed** (the element types they start with don't change). Enums and module-level constants are their
  values. Classes and dicts become structs; only fixed dicts for now, so a function that could add a key runs
  as Python. Loop optimizations and loop lifting come first, so that containers don't hide speedups still to
  be had from loops. Open, apart from dicts below: what a runtime value that breaks the rule does, aliasing, and the
  cost of checking and copying at the call. See `TODO.md`.
- **Nested tuples (built 2026-11-03).** One C variable per number, named by its path (`v_t__0__1`); values move as nested lists
  of (value, flag) following the type, and results take one pair of `sj_io` slots per number. A loop over a tuple holds every
  element first, as Python builds it, then selects element k by index: one counted loop, so `break` and `continue` need nothing
  new. Elements must share a type. Preflight takes a parameter's tuple shape from how it's unpacked, or what it's passed to.
- **SIMD (2026-11-03):** cimo vectorizes straight loops of 4- and 8-byte lanes, so the generated C keeps loop bodies free of
  stores that don't vary: a variable's numpy flag (`total = 0` then `total += x[i]`) is set once after a counted loop that ran,
  when every assignment gives a numpy value, one runs every iteration and nothing in the loop reads the flag.
- **Enums (built 2026-11-03).** A member of an `Enum` or `IntEnum` with int values is its value; the class, read from module
  level, is a hidden argument whose identity is part of the signature, so `Level.HIGH` is a literal. Two members of one class are
  the same member exactly when their values are equal, since an alias is the member it names; an `Enum` is never equal to anything
  else, and an `IntEnum` is also an int. A returned member is `cls(value)`. A class that defines its own `__eq__`, `__hash__`,
  `__bool__` or ordering stays Python. Preflight builds the enum from a class statement of int literals and `auto()`.
- **Fixed dicts (built 2026-11-02).** A dict argument is flattened in source: `p['a']` becomes a variable and `p` one parameter per key
  the function names, so all loop work applies and the C equals using locals. A dict's type is only those keys, so anything else in it
  can be anything, at no cost: a read took 2.6 µs per call against 2.6 with plain numbers, in a 1000-key dict too, 3.2 with a write (2.6
  only through a call plan that checks those keys; the general path takes 9). A write goes back at both of the kernel's exits, so
  stores before an exception survive. The kernel's written variables are kept out of web splitting, since every exit reads them.
  Exact `dict` only, since a subclass can run code on a read, and one dict passed twice runs as Python.
- **Instances (built 2026-11-05, from 0.0.4).** Flattened as dicts are, `p.a` becoming a variable. Python's lookup is what must be kept: at every call, walking the class's `__mro__`, an attribute compiles only if
  no code of the class would run: `__getattribute__` the default one, no `__getattr__`, nothing on the class under that name but a plain value, a slot or a namedtuple's field, and for a write the default `__setattr__` and the
  attribute on the instance already, since writing back one only on the class would add it even where no assignment ran. A call plan checks the same thing for 2.0 µs a call (10 without one; a dict takes 1.7) by comparing what each Python-made class holds
  under those names with what it held: built-in classes can't change. `self.__x` stays Python, as the source doesn't show the name Python gives it. An object and its own `__dict__` count as one object passed twice.
- **Methods (built 2026-11-05, from 0.0.4).** `p.m(...)` is a callee call: the function on the class (a plain one gets a `Function` of its own) takes
  the attributes it uses, through what it calls, as self, never the rest, which may have changed type in the caller. Its writes come back from its `sj_io`
  before its status is checked, so an exception keeps them. A writing call stays out of tests, loop headers and comprehensions, and `p__a = p__a` after its
  statement shows the write to every analysis that looks for assignments. Which function runs is the class's at each call, as for attributes; one on the
  instance runs Python. A call costs 5 to 6 ns more than inline code, as every call between compiled functions does: cimo keeps the result array in memory.
- **Lists of instances, passing instances (built 2026-11-05, from 0.0.4).** A list of one class's instances is a list of numbers per attribute used,
  copied in at the call and written back where a value changed (identity kept for the rest); `for p in ps`, `enumerate`, `p = ps[i]` make `p` an index,
  checked where Python would raise. An instance or element passed to a function or method goes as the attributes the callee uses (`_param_uses`, through
  what it passes it to), narrowed at compile time. Two elements of one list may be one instance, so a call given two compiles only if what it assigns
  through one it doesn't use through the other. Only real assignments count there: the `p__a = p__a` markers say a call may write, not that it does.
  A method's globals join the caller's when a call's types name the method (`_add_method_globals`), so the caller's globals can grow after its first call.
- **Making instances, inner objects, the rest (built 2026-11-05, from 0.0.4).** `v = C(...)` compiles when C's construction only stores: a
  dataclass's own `__init__` (recognized by its code having no file), an `__init__` of `self.a = expression of its parameters`, or none; and
  `type.__call__`, `object.__new__`, default lookups. The class is a hidden argument (`madeclass`), checked like an instance's class by a plan. `v`'s
  attributes are variables assigned in `__init__`'s order; a returned `v` is `cls.__new__(cls)` given them in that order with `object.__setattr__`,
  which is what the dict order and a frozen dataclass need. Inner objects are owners of their own (`'p.pos'`), flattened breadth first.
  `isinstance` folds from the class in the signature. Keyword-only parameters are compiled as parameters
  after the others, the Python fallback getting them back by keyword.
- **Object model (decided 2026-11-05, to build; `TODO.md`, "Object model").** The case-by-case flattening above can't compose: each
  shape is its own AST rewrite, so combinations fall back. Instead: every statically named access path is mapped, at the call, to the
  object it reaches; paths reaching one object share one variable, and that alias pattern is part of the signature, so kernels stay flat
  with nothing to alias. Only where the code chooses an object by data (indexing a collection that may share, rebinding a reference,
  walking a structure, keeping objects made in a loop) do objects live in per-class struct-of-arrays pools referenced by integer index.
  Rejected: struct pointers everywhere, since cimo doesn't keep fields in registers through them even with `restrict` (1.2x to 1.8x
  slower, measured), and CPython-style objects, whose cost is boxing, dispatch and hashing, not the indirection.
  **Built (2026-11-05)**, onto the existing case code rather than replacing it (`TODO.md`, "Object model: what's left"):
  the alias pattern (`T.same_as`; `_object_graph` walks the paths with one identity memo, an object handed to a callee is rebased or
  expanded, `handed`); two elements given to a callee go by reference (`T.elemref_of`: the lists and the index) only where what's
  assigned through one is used through the other, so code that compiled by value keeps its C; a list with repeats, or objects
  shared by elements, is pooled per path (`objlist_of(..., shared)`: lists of each object once, an index list per pooled level).
  A call returning a new instance (`returns_made`) hands back its attributes and the caller's name holds them; `a + b` is
  `type(a).__add__` unless it can return `NotImplemented` or `b` is another class (Python may try `b.__radd__` first); a new
  instance inside an expression is hoisted to where Python makes it, refused past a call that may assign or under a condition.
  A parameter only ever given new instances starts as a copy of the argument's numbers, valid only if nothing is written through
  it first and it isn't returned or compared with `is`. References chosen by data (`n = n.next`, `a.pos = b.pos`) index one
  pool per call of one class (`T.pool_of`, `_pool_plan`), None is -1, reading through it raises `AttributeError` (status 13,
  names in `ATTRIBUTE_NAMES`); the pool walk is O(objects reached) per call. Speed: tree lookups 21 ns per node against 170 in
  Python (6x, branch misses dominate); vector operators 50x; C unchanged on the earlier class examples.
  **2026-11-06:** pools of several classes; references given to compiled callees and methods as the caller's pool lists and an
  index (`T.poolref_of`), returned by them (`ref_return_class` follows the callee's code); lists of pooled objects as index lists
  (`T.reflist_of`); instances made and kept in a structure go into their class's lists, which have spare room, and a kernel that
  runs out stops (`POOL_FULL`) and is called again with twice the room, sound only because it wrote nothing but copies (so writing
  an array as well refuses), their types from existing objects or from what the constructor is given (a wrong guess is refused
  by the compiler's store checks, never silently converted). A pooled call plan (`_pooled_plan`, `_pool_build`) took a call
  from 200 µs + 5 µs per object to 10 µs + 0.45 µs; object plans walk into objects in attributes (3.3 µs). Static preflight
  follows operators, new instances from methods, and pools assumed from annotations. Every tree analysis reads one `_Facts`
  (one walk, cached on the tree with its parameter count, as a copy given hidden parameters must be walked again); calls are
  marshalled by `_marshal` with a packer per kind; `flatten_dicts` resolves objects through one `locate`. Generated C unchanged.
- **numpy on single numbers (built 2026-11-02).** 25 of 26 functions matched libm bit for bit on 160,024 inputs on arm64 and in Pyodide, so they
  call it; `np.tanh` doesn't, natively. Types are the trap: `np.abs(4)` is int64 natively and int32 in Pyodide, `np.floor(4)` changed in numpy 2.1,
  `np.sqrt(True)` is float16, so those stay Python. `np.minimum(0.0, -0.0)` differs by platform, so the compiler asks the numpy present.
- **Views (built 2026-11-02).** Pointer, extents and strides as plain variables, so no allocation and `row[j]` costs what `m[i, j]` does (1.04 against 1.01
  ns). Until these, arrays were parameters with fixed extents, and range facts, aliases, versioning and forwarding relied on it. Now an array assigned
  anywhere (`rebound`) gives no facts: its loops are versioned instead, tested at entry, unless it's assigned inside that loop. Removing any of these guards
  fails `tests/test_views.py`.
- **numpy reductions (built 2026-11-02).** `np.sum` runs one pairwise pass over everything, blocks of 128 in 8 accumulators, not seeded with the first element;
  a plain loop differs from about 100 elements. Pyodide's numpy sums 8192 at a time, so the compiler asks the numpy present with a fixed array. Signed zeros in
  `max` and `min` follow the scalar function's rule on both platforms; which of two NaNs of different signs comes back depends on numpy's loop layout, and isn't matched.
  Only forms giving one number compile; `axis` on n-D and `keepdims=True` return arrays. On 1,000,000 elements `sum` is within 15% of numpy (not SIMD there
  either), `max` 9x slower, `argmax` 4x, `all` 2.4x; preflight notes these only outside loops, since in a loop `m[i].sum()` wins up to 150x over a numpy call.
- **numpy compatibility (2026-11-02).** Slice assignment copies only if the two share memory, as numpy acts. A returned view is a real array over the argument's memory, wrapped through
  `__array_interface__` since a view's root may itself have gaps; a variable tracks which argument it's of, in functions that return arrays only, so others keep their C. A view can
  come in as an argument, unless its address or stride isn't a whole number of elements. Integer `mean` sums in 8192-element chunks on every build, since numpy converts through a buffer; `var` is numpy's steps exactly. `np.clip` of a
  value equal to a bound depends on the platform, as `max` does, so it's probed.
- **x86-64 (first run 2026-11-02, Windows, emulated, AVX2, Python 3.12).** The whole suite runs. `platform.machine()` says ARM64 there, and cimo handles it. All numpy functions on single
  numbers match; AVX-512 unchecked. `max` and `min` keep the last of two zeros, a third rule next to "by sign" (arm64) and "the first" (wasm). Python 3.12 words `1.0 % 0.0` differently from 3.13,
  so zero-division messages now come from the running Python. NaN signs differ as on Windows arm64.
- **numpy calls (decided 2026-11-02; see `TODO.md`):** whole-array arithmetic stays in Python, and MKL and BLAS stay out: not pure Python, and their results vary by library, CPU and thread count. A
  plain loop doesn't reproduce `np.sum` (pairwise), and was 6 to 12x slower than numpy on 1,000,000 elements. cimo has SIMD now, but not for loops with a condition (its `TODO.md`).
- **Arrays made in a function (built 2026-11-02).** Each is a `malloc`ed block behind a header, and a view remembers its block in a variable. No reference counts, so loops keep their C: at statements
  that make arrays and at the top of loops that do, if 1 MB or the live size has been allocated since the last time, blocks no variable names are freed. A stale variable only keeps memory longer. A result is copied
  into a numpy array, as numpy can't own a kernel's block and `resize` has to work; 3.1 µs returning 10 elements against 0.14 for `np.zeros`, 107 µs for 800 KB against 4.8, since numpy's pages stay untouched. Inside a loop 24 ns against 256 (8
  elements), 213 against 434 (8 KB), 10.9 µs against 6.8 (800 KB, every one mapped and faulted in). A block is `sj_release`d by the kernel's module: `module.free` exists only natively. The size limit is `sys.maxsize` bytes as in numpy,
  and also what keeps a request from reaching cimo's wasm heap, which truncates past 4 GiB (cimo `TODO.md`). `np.full(2, 300, dtype=np.uint8)` raises in newer numpy and wraps in 2.0.2 (Pyodide's), so the compiler asks the numpy present and wrapping stays Python.
  Generated C for functions that make no arrays is unchanged, byte for byte, on 10 examples.
- **Calling C (built 2026-11-02).** `k.f(x)`, with `k` from `cimo.compile`, puts `k.source` in the generated file, so cimo inlines it: 0.92 ns against 686 from Python. Calling by `kernel.address` isn't an alternative: it crashes once the callee uses `malloc`, and wasm has none. So
  a module with data that isn't `const` stays Python, as the copy wouldn't share it, and the file uses the module's `fma`, there being no `FP_CONTRACT` in cimo: helpers with `a * b + c` in one expression (`CONTRACTING`) refuse to mix with `fma` on. Conversions copy Python's: 32 bits and under cut as ctypes does, a float is rounded through a double, and
  what Python raises for, or treats oddly (`np.int64` to a 64-bit parameter is taken as an address, cimo `TODO.md`), stays Python. The module has a type of its own, like a function, so rebinding the name is a new signature.
- **Module-level constants (built 2026-11-02).** Hidden parameters, read at each call, not embedded literals: a new value needs no recompile, and arrays and dicts get them for free. Passing one costs about what an
  argument does, 0.2 µs, more than the 21-25 ns measured for only checking one. A literal gains only on `%`: `x[i % K]` 1.25 against 1.06 ns; `x[i] * R` and `range(K)` nothing measurable.
- **Later:** growable lists, dicts, sets and strings as typed C containers: each needs allocation and a runtime.

## Semantics

Each decision is explicit and tested against CPython:

- `int` is int64 unless narrowed; overflow wraps. No checked mode: numeric code rarely nears
  `2**63`, and checking every operation would cost every loop. `<<` wraps too, as multiplying by `2**k` does
  (0 from 64 bits on), so a hash masked to 32 bits (`(x << 13) & 0xFFFFFFFF`) is exact; `>>` is always exact, and a
  negative count raises `ValueError` as in Python. On numpy scalars, numpy's: a count outside the width gives 0, or -1.
  An int that can only be in `[0, 2**64)` (a literal past int64, `& 0xFFFFFFFFFFFFFFFF`, a sum, product, `|` or `^` of such
  values) is held as `uint64_t`, so 64-bit hashes are exact. Non-negativity is read from the source, following each name to
  every assignment (`nonneg`); a parameter it relies on is checked on entry, and a negative one reruns the call in Python.
  Literal-only int expressions are folded at Python's precision (`(1 << 64) - 1`). Rejected: an unsigned flag per value
  (what `h ^ k` means would depend on k's sign at run time), and 128-bit ints.
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
  `total += x[i]`) carries a runtime flag. Narrow dtypes (float32, uint8) follow NEP 50 too: a Python
  int must fit the dtype or raises `OverflowError`, except with `/`, and with `float32` a Python number is rounded
  to `float32` first, in comparisons as well (`np.float32(0.1) == 0.1` is true). A literal that becomes narrow (`total = 0.0`,
  then `total += x[i]`) is held in the narrow type with the flag, so it must fit exactly.

## Execution model

On each call the decorator reads the argument types (Python type, dtype, number of dimensions,
layout), generates C specialized to them, compiles it with cimo, and caches the result by
signature, in memory and on disk.

Compiled code releases the GIL while it runs (as cimo kernels do), so Python threads calling a
compiled function run in parallel. One loop split by hand into ranges over a `ThreadPoolExecutor` scaled
4.0x on 4 threads (2026-11-06), so a parallel loop of scramjit's own is worth building only for what
the hand split lacks: a proof of independence (`TODO.md`, Adoption, `# scram: parallel`). Arguments
copied in (lists, lists of instances, lists of lists) write back only what the call changed, against a
snapshot taken at the call, so concurrent calls on disjoint parts of one list keep each other's writes.

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

**Python data structures in an LLVM-based Python JIT** (studied 2026-11-01, by reading it and running it):

- Copying a Python list in and back at the call was tried and is being abandoned: nested lists are rejected, and
  its authors found that doing this consistently for nested structures is close to impossible. Reading one
  element of a million-item list cost about 395 ms per call, since the whole list converts. Its check is
  worth keeping: write back only if something wrote. And so is its answer to aliasing: a list met twice in
  one call maps to one native list, so a write through one name shows through the other.
- Its replacement is its own container types, which pass into compiled code in about 6 µs but cost users: they
  must change their code, and reading one from Python took about 4,000 ns against 28 for a list. Classes work
  the same way, through a decorator and a type spec. Plain instances, dataclasses and dicts passed in are
  rejected. scramjit's aim differs: ordinary Python objects, so that decorating changes nothing else.
- Its fixed-key string dicts are named tuples underneath, with constant keys only: the same plan as ours.
  Enums are their values, globals compile as constants, and tuples are structs, with variable indexing only
  when all elements share a type. All of this fits.
- Don't copy its silent differences. Storing 1.5 into an int list or class field truncates to 1 with no error,
  and a global changed after the first call keeps its old value. Our rule: an error, or a fallback, never a
  wrong answer.

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
- **Calls between compiled functions (built 2026-11-02).** One C file per function called from Python, with a static C function per callee and signature; cimo inlines small ones, so `sq(x[i])` costs 0.93 ns, as
  written out, and a `step` 9.1 ns against 3 µs from a Python loop. A branchy `clamp` costs more than written out, 4.66 against 2.44 ns. A callee has its own `sj_io` and its own heap: a block it returns moves to the caller's, since its sweeps
  only see its own variables. Each failing caller adds to a trace at the end of the top `sj_io`, for a traceback with every frame. A called function's globals are hidden arguments of the top function, by their own dict, so
  their guards and new values work as at any call; the name holding a function has a type of its own, so rebinding it is a new signature. In a statement with a call, every value read is put in a temporary first, since the
  call can write arrays; forwarding is off in such a loop. Default values: a call is bound as Python binds it, from `__defaults__` at each call (a default array is the same object; a changed `__defaults__` is seen), and a compiled caller leaving one out passes it as a constant if it's a number. Not built, each staying Python: recursion, which needs return types before inference, other defaults left out by a compiled caller, keyword-only parameters, dicts and lists as arguments, `module.f()`, and tagged parameters, since a callee can't rerun as Python.

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
- **`--counts` and `--live`** (with `--run`; built 2026-11-02): `--counts` compiles copies that count how often each line runs, for the
  hottest lines in the report, and with `FILE:FUNCTION` only that function, every line, loop iterations and branches taken. `--live` shows
  the report during the run: redrawn on a terminal, a line every 5 s when piped, JSON lines with `--json`. Separate because they cost
  different things, and plain `--run` must never slow down. Not `--watch`, which means "rerun on change" elsewhere.
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

**Line counts: what they cost.** Statements that always run together share one counter, and an `if` adds its test to its body's
counter without branching (`sj_c[n] += t`), its else being the difference. A store inside the body stopped if-conversion: a random branch
went from 0.69 to 4.6 ns (6.6x), `edit_distance` 3.8x. With this (interleaved, best of 7, arm64): `ema` 2.42 to 2.48 ns, matrix-vector
0.90 to 0.96, a random branch 0.70 to 0.90, `edit_distance` 1.26 to 1.82. So timing and counting are separate runs: counts are checked
against a Python tracer, and don't time anything. A kernel adds to its counters with the GIL released, so a Python thread reads them mid-call
(`tests/test_counts.py`); in Pyodide counts work but move only between calls. `--live` alone costs under 1% on calls of 12 µs. Only
native code is counted, so this can't see into libraries scramjit doesn't compile. `@scram(counts=True)` and `f.line_counts()` do the same where there's no command line; the LLM demo's heatmap uses them, compiling the counted copy in its second worker, and writes 51 tok/s against 60 while it's on. Spills per line would need them from cimo
(`cimo report` in its `TODO.md`).

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
numpy scalars don't). The wrapping `sj_add`, `sj_sub`, `sj_mul` and `sj_neg` are macros, not
`static inline`: cimo stops inlining into a function once it reaches 4000 IR instructions, in source
order, so in a large function the later loops call one-instruction helpers. A llama2.c forward pass
as one function (`bench/llm.py`) ran 28 tok/s natively with them as functions, 89 as macros; the M1
kernels don't change. Other small helpers can still hit that limit (proposed to cimo). Versioning
also takes `a[base + i]`, for `base` of Python ints unchanged in an innermost loop: `base >= 0 && stop
<= n - base` can't overflow and proves every index, wrapped or not. On `bench/llm.py`, default options
went from 77 to 92 tok/s natively and 68 to 88 in Pyodide. `boundscheck=False` gets it too, because it
also drops the negative-index test: 89 to 90 natively but 70 to 88 in Pyodide. The price is a second
copy of those loops: this function compiles in 4 s rather than 2.7 by default, 2.5 rather than 1.25
with `boundscheck=False`. A constant `base` stays with `a[i + c]`, or `a[i - 1]` would always take the
checked copy. `y[i - 1]` reads a temporary when `y[i] = ...` is the loop's only array store, on every iteration, in a
float64 or int64 array: with no other store, no view can change that element, so overlap needs no test. Up to `y[i - 4]`,
each in its own temporary, shifted after the store; past `y[i - 1]` it gains little (a biquad 5.80 to 5.68 ns per sample),
since those loads are issued iterations before they're needed. With stores to other arrays, a test before the loop
checks that none shares y's bytes (contiguous arrays, not assigned in the loop; lists have memory of their own), with
the plain loop otherwise: `y[i] = a * x[i] + (1 - a) * y[i - 1]; d[i] = x[i] - y[i]` 5.1 to 2.4 ns (arm64).
`ema` 3.09 to 2.39 ns (hand C 3.06, which cimo can't forward), `edit_distance` 2.22 to 1.71; in Pyodide
`ema` 4.20 to 3.55. Facts from `range` loops with a step of 1 or another positive literal (index
non-negative, below `len(a)`) drop negative-index adjustment and bounds checks, and loop versioning
(one test before the loop, checked copy otherwise) removes the rest without changing where an
`IndexError` is raised. In innermost loops the test also covers a positive literal step, using the
last value the variable takes, and a start not known to be nonnegative (`start >= 0` joins the
test): `y[i] = x[i] + x[i + 1]` over `range(0, n, 2)` 1.53 to 0.79 ms, over `range(lo, n)` 1.97 to
0.63 ms (2M elements, arm64), for 5% more compile time on `bench/llm.py`'s forward pass. Versioning's copies are capped per function, since cimo's compile
time grows with the square of a function's size (60 loops: 3,140 lines of C in 4.0 s, 6,020 versioned in 14.7 s): they
may bring it to 3,200 syntax nodes, at least 400, the most deeply nested loops first (14.7 to 5.1 s, `bench/llm.py` unchanged). With those, parity and bounds checks cost nothing measurable on the M1 kernels:
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
and `math.trunc` of one raises. Not yet: value-range narrowing, `map`/`filter`.

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

**M2 acceptance (2026-11-02).** Met as written: every rule has differential tests (`tests/test_m2.py`, `test_lazy.py`,
`test_for_else.py`), `tests/test_m2_acceptance.py` finds no iterator or boxing code in each M0 kernel's C, and `math` matches CPython
natively and in Pyodide (28,019 cases there). Arithmetic on narrow dtypes came later, with differential tests over every pair of dtypes (`tests/test_narrow.py`), and then `**`, `sum()`, numpy functions that are exact in double and rounded once, and reductions: each numpy reduction helper is copied per element type (`_mathlib.typed`), all-`float` for float32, since numpy totals float32 in float32. `axpy` on float32 is 0.15 ns per element against 0.26 on float64 (arm64). Range analysis still picks no narrow types: arrays already have the user's dtype, cimo has no byte or short lanes, and scalars gain nothing, so only temporary arrays could benefit. `for ... else`
needed definite assignment to follow breaks: `found` assigned before each break and in the else is assigned after the loop.

Found on the way, in code emitted before this: a size from the user could wrap (`[1.5] * (2**61 + 1)`: 8 times that is 8 bytes), so
`malloc` succeeded and the fill crashed the process. Now `MemoryError`, as in Python. cimo made signed overflow undefined the same
day; what scramjit emits still wraps through unsigned, checked by audit and by 36 cases against numpy at the extremes.

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

**Ctrl-C.** Python runs signal handlers between bytecodes, so a kernel used to run to its end. A thread now
waits on `signal.set_wakeup_fd` and sets a flag kernels check, and a kernel that sees it returns a status
that raises `KeyboardInterrupt` (`scramjit/_interrupt.py`). A check on every iteration is too dear: `ema`
3.08 to 5.16 ns, `edit_distance` 2.24 to 3.74, likely the volatile read stopping optimizations; a counter
was worse. Counted loops run in chunks of 65,536 instead, checking between them, which cost nothing
measurable on the benchmarks. Short constant ranges skip chunks (their setup cost 3x3 stencils 28%); a very
short variable range still pays about 20%. `while` loops check every iteration, which was cheap. It stops a
kernel only on the main thread and under Python's default handler, since a stopped kernel can't resume and
another handler might not raise. Tested with real signals on Linux and real console control events on Windows
arm64 (a console of the program's own, `CREATE_NO_WINDOW`, then `GenerateConsoleCtrlEvent`); macOS isn't yet.

**M3 progress.** The disk cache is cimo's: scramjit's C is deterministic, so a second process finds
it (Gray-Scott first call 0.53 s, then 0.06 s, mostly preprocessing the prelude); no second cache
needed. The C holds no position (2026-11-05): lines count from the function's `def` and `#line` names no file, so
moving a function or renaming its file hits the cache (a line above Gray-Scott was 86 ms warm, 562 ms before). Preflight is built (`scramjit/_preflight.py`): static analysis sets each failing statement
aside and retries, so it reports every line, not only the first, though a line can fail only because
one it depends on did. Bounds notes come from the compiler, not from scanning the C, since a versioned
loop's slow copy always holds checks that don't run in the fast one.
The command line (`scramjit/cli/`) is built like cimo's: its parser, terminal and completion are copied
rather than imported, because they aren't cimo's public API. `--run` reports on stderr so the program
keeps stdout.

**Variables that change type (2026-11-02).** Two steps, only after inference finds a conflict, so other functions keep
their C. `_webs.py` splits a name into webs by reaching definitions (`x = 1`, used, `x = 0.5`, used; `for v` over two
arrays). A web holding int and float, as in `s = 0; for v in x: s += v`, can't simply widen: after no iterations
Python has the int. So it's a double with a flag saying it's still an int, for ints a double holds exactly, in the only places
an int and that float behave the same: arithmetic and comparison with a float, and returning it. Every other read
fails the compile. An int parameter is checked on entry, where a kernel that has done nothing can return `RERUN`, so the
call runs as Python (needed for lifted loops, where `s` arrives as a parameter). No cost measured: `s = 0` and `s = 0.0`
both 1.05 ns per element. A random program test found what hand-written ones hadn't: a parameter's C type must
follow its signature, not its joined type (a float parameter later assigned `3`), and a numpy parameter whose variable
can also hold Python floats started with its flag off, an older bug.

**M4: loop lifting.** Compile the loops that fit inside functions that don't. Done when a function
mixing Python-only setup (dicts, strings) with a numeric loop is partly compiled, matches CPython,
and preflight (static, `--run` and `f.preflight()`) shows the state per line.

**M4 done (2026-11-02).** `scramjit/_lift.py`: a loop becomes `def f(live_in): <loop>; return live_out`, an ordinary
Function, so it specializes on run-time types and, if it doesn't compile, runs as Python with its inner loops as the
next candidates. So anything the compiler refuses (a variable maybe unbound, a type change) is simply not lifted, and
lifting only has to protect the frame: nothing in `try` or `with`, no function with closures, `locals()`, `del` or
`global`, and live-ins definitely assigned before the loop, or the call could raise where Python wouldn't. Live-outs
need real liveness: reusing `i` in the next loop is not reading it. Arrays stay correct because stores happen in order;
a frame that dies with an exception has no variables left to get wrong. A call costs more than a very short loop, so
lifting alone made a 3-iteration inner loop 10x slower than Python. `Tier` times tiny loops both ways and keeps the
faster, comparing minimums because single samples misled it (the second native call builds the call plan; a pause in a
long run is certain): now 0.7x at 3 iterations, 1.1x at 40, 71x at 100,000.

## Risks

- **First-call latency** in pure Python decides whether library authors can use it. Measure cimo's
  compile time on generated C early.
- **Performance without vectorization** is well behind LLVM-based JITs on simple loops until cimo's
  auto-vectorizer exists.
- **Semantic drift** from CPython in edge cases (overflow, division, indexing, narrowing) breaks
  the "it's still Python" promise. Differential testing is the guard.
- **Silent slowness** from fallback: a decorated function that quietly runs as Python. The warning,
  preflight (static and `--run`) and `fallback=False` are the guard.
- **Scope creep** into anything that resizes pulls in a runtime. Fixed-size, fixed-type structures don't, which
  is where the line now is (see "Containers").
