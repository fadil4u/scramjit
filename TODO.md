# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

- **Hot helpers are macros** (`scramjit/_runtime.py`: `sj_add`, `sj_floordiv`, `sj_truediv`, ...)
  because cimo compiles `static inline` functions as calls (5.15 vs 3.07 ns per element on
  `ema`). When cimo inlines small functions: turn them back into functions. **cimo inlines now**
  (commit `49f08c0`): functions of up to 40 IR instructions, including ones defined later, and
  `ema` with a helper matches the macro (3.33 ns). Remove the macro workaround.
- **`CIMO_FMA=0` set before importing cimo** (`scramjit/_dispatch.py`, `tests/conftest.py`), plus
  a warning when it is on, because contraction is process-wide. When cimo takes `fma=` per
  compile: pass it, drop the environment variable and the warning; `parity=False` can then turn
  contraction on.
- **Math functions call the host libm through a table of addresses** (`_dispatch._libm_table`,
  `SJ_LIBM1` in the prelude), passed to each kernel that uses them, because cimo resolves `sin`
  etc. to its bundled libm. When cimo can link the host libm for math calls: call them directly.

- **Code shaped around cimo's missing CSE:** wraparound indices hoisted to the top of the loop
  body (`hoist_wraps`), `//` and `%` computed as one division and one remainder into
  temporaries. Both help any compiler; revisit what else can go once cimo merges repeated
  expressions (cimo `TODO.md`: "Slower than gcc -O3 on scramjit's kernels").

- **The demo's WebAssembly tab calls `cimo.wasm.build`** (`docs/lab/lab.py`: `wasm_text`) for the
  module's bytes, which isn't cimo's documented API. When cimo exposes a compiled module's
  WebAssembly (or a disassembly), use that.

## Next

0. **All of `math` in Pyodide.** Only the functions with no C library code (`sqrt`, `fabs`, `fmod`,
   `copysign`, `hypot`, `floor`, `ceil`, `trunc`, `degrees`, `radians`, `isnan`, `isinf`,
   `isfinite`) compile there. Every other `math` function (`log` and `exp` families, the trig and
   hyperbolic ones, `atan2`, `pow`, ...) makes the whole function fall back to Python, which costs
   the entire speedup: the site's beacon and fractal demos had to be written around it. Compile
   all of them, bit-exact with Pyodide's CPython (its libm is emscripten's), and test each in
   Pyodide, not only natively. Then remove the workarounds in `docs/lab/` (beacon's `math.sqrt`,
   the fractal's colouring without `math.log`).

1. **Rest of M2:** narrow dtypes (float32, int32, uint8, ...) with numpy's NEP 50 scalar
   arithmetic, value-range narrowing; `map`/`filter`; nested tuples; lists of several for
   clauses. Then the M2 acceptance check on the M0 kernels (needs 2-D arrays from M3).
2. **M3:** n-D arrays (and strides), disk cache, Pyodide, out-of-subset reporting.
3. **SSA for variables that change type** (`x = 1` then `x = 0.5`): today they fall back.
4. **Bounds-check hoisting beyond unit-step loops** over one variable (versioning covers
   `for i in range(a, len(x))` with `arr[i + c]` today).
5. **Store-to-load forwarding:** `y[i]` written then `y[i - 1]` read next iteration; keeping it in
   a register gives `ema` 2.39 ns against 3.07 (cimo doesn't do it).
6. **Call boundary cost:** a call is about a microsecond of Python (signature key, argument
   list, status buffer); measure and trim for small inputs.
