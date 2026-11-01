# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

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
