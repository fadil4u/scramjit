# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

- **Hot helpers are macros** (`scramjit/_runtime.py`: `sj_add`, `sj_floordiv`, `sj_truediv`, ...)
  because cimo compiles `static inline` functions as calls (5.15 vs 3.07 ns per element on
  `ema`). When cimo inlines small functions: turn them back into functions.
- **`CIMO_FMA=0` set before importing cimo** (`scramjit/_dispatch.py`, `tests/conftest.py`), plus
  a warning when it is on, because contraction is process-wide. When cimo takes `fma=` per
  compile: pass it, drop the environment variable and the warning; `exact=False` can then turn
  contraction on.
- **Math functions call the host libm through a table of addresses** (`_dispatch._libm_table`,
  `SJ_LIBM1` in the prelude), passed to each kernel that uses them, because cimo resolves `sin`
  etc. to its bundled libm. When cimo can link the host libm for math calls: call them directly.

## Next

0. **Math in Pyodide:** functions that involve no C library code (`sqrt`, `hypot`, `floor`,
   `fmod`, ...) compile there; the others fall back until ported from the libm Pyodide's CPython
   is built with (emscripten's), and checked in Pyodide.

1. **M2: builtins, lists, narrowing.** `enumerate`, `zip`, `reversed`, comprehensions and generator
   expressions fused into one loop, `sum`/`any`/`all`, `math.*` (bit-exact), fixed-size lists,
   tuples, narrowed types (and numpy's NEP 50 scalar arithmetic for the narrow dtypes).
2. **SSA for variables that change type** (`x = 1` then `x = 0.5`): today they fall back.
3. **Bounds-check hoisting beyond unit-step loops** over one variable (versioning covers
   `for i in range(a, len(x))` with `arr[i + c]` today).
4. **Store-to-load forwarding:** `y[i]` written then `y[i - 1]` read next iteration; keeping it in
   a register gives `ema` 2.39 ns against 3.07 (cimo doesn't do it).
5. **Call boundary cost:** a call is about a microsecond of Python (signature key, argument
   list, status buffer); measure and trim for small inputs.
