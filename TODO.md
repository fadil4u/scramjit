# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

- **Float `math.sumprod` falls back:** Python returns the int 0 when it's empty, so its type depends on
  the data. Same for float `math.prod` without `start=1.0`, where the fix is in the user's hands.
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

1. **Rest of M2:** value-range narrowing; nested tuples; `for ... else`. Then the M2 acceptance check.
2. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`), and a `disk` label in the compile column,
   which needs cimo to say whether a compile hit its cache.
3. **Ctrl-C during a compiled call:** it waits until the call returns (10.7 s after the signal in one
   test), and a compiled infinite loop can only be killed. Check a flag at loop back edges, if that can
   be made cheap. Measure first.
4. **`n ** k` with a negative `k` raises today** (see Gotchas). Rerunning in Python, for functions that
   write nothing, or resuming in Python alongside loop lifting would remove the difference.
5. **SSA for variables that change type** (`x = 1` then `x = 0.5`): today they fall back.
6. **Bounds-check hoisting beyond unit-step loops** over one variable (versioning covers
   `for i in range(a, len(x))` with `arr[i + c]` today).
7. **Store-to-load forwarding:** `y[i]` written then `y[i - 1]` read next iteration; keeping it in
   a register gives `ema` 2.39 ns against 3.07 (cimo doesn't do it).
8. **Call boundary cost:** a compiled call measured about 3.3 µs (signature key, argument list, status
   buffer, ctypes-style call), against 83 ns for a small function in plain Python, so decorating a tiny
   function can make it about 40x slower. Trim it, and warn in preflight about functions with no loop.
