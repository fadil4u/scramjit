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

1. **Python data structures, nested, when statically sized and statically typed:** dicts, lists, tuples,
   classes and enums that don't resize at runtime and whose element types don't change. See "Containers" in
   `requirements.md`. Today only flat tuples of numbers and flat local lists compile. In this order, simplest
   and most useful first:
   - **Module-level constants:** `K = 3` and `TABLE = (1.0, 2.0)` read by a function. Both fall back today.
   - **Enums:** `Level.HIGH`, comparisons, and members as arguments, as their values.
   - **Nested tuples:** `((1, 2), 3)`, and tuples holding arrays, as locals, arguments and results.
   - **Lists as arguments, and nested:** `[[1.0, 2.0], [3.0, 4.0]]`; returning one. The spec's copy in and
     out at the call was never built.
   - **Classes:** attribute reads and writes on instances and dataclasses, then method calls.
   - **Dicts with fixed keys:** `d['a']` on a dict whose keys don't change, as a struct where the keys are
     constants, then a hash table where they aren't.

   To settle first, all of it needing differential tests: what happens when a value breaks the rule at runtime
   (a new key, an `int` slot given a float); two names for one object (`[[0] * 3] * 2`, one list passed twice),
   since writes must show through both; the cost of checking and copying a whole structure on every call; and
   results that must come back as real Python objects.
2. **Rest of M2:** value-range narrowing; `for ... else`. Then the M2 acceptance check.
3. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`), and a `disk` label in the compile column,
   which needs cimo to say whether a compile hit its cache.
4. **Test Ctrl-C on macOS.** `tests/test_interrupt.py` has run on Linux and Windows arm64, not macOS. It
   should work, since it uses `signal.set_wakeup_fd` and sockets, but nothing has checked it.
5. **A NaN's sign differs from CPython's on Windows** (arm64, the only one tried). `inf % 1.0`, `inf // 1.0`
   and `math.atan2(0.0, nan)` give `7ff8...` where CPython gives `fff8...`, and `nan * -1` gives `fff8...`
   where it gives `7ff8...`; `tests/test_m1.py` and `test_m2.py` fail there on these, 7 in all. The values are
   all NaN, and only `math.copysign` or the raw bytes can tell. Likely fix: call the CRT's `fmod` through the
   libm table, as for `sin`, and make the `* -1` negation depend on the platform. Nothing runs scramjit's tests
   on Windows automatically yet.
6. **`n ** k` with a negative `k` and `x ** 0.5` with a negative `x` raise today** (see Gotchas). Rerunning
   in Python, for functions that write nothing, or resuming in Python alongside loop lifting would remove
   the difference.
7. **SSA for variables that change type** (`x = 1` then `x = 0.5`): today they fall back.
8. **Bounds-check hoisting beyond unit-step loops** over one variable (versioning covers
   `for i in range(a, len(x))` with `arr[i + c]` today).
9. **Store-to-load forwarding:** `y[i]` written then `y[i - 1]` read next iteration; keeping it in
   a register gives `ema` 2.39 ns against 3.07 (cimo doesn't do it).
10. **Call boundary cost:** about 1.4 µs with numbers (was 3.2) and 2.4 µs with an array (was 6.6), through
   a function generated per signature (`_make_plan`). For numbers: ctypes about 450 ns (the floor), cimo's
   `Kernel.__call__` about 450 (see cimo's `TODO.md`), scramjit about 500. In Pyodide about 13 µs, not yet
   broken down. Break-even is 30 to 50 loop iterations per call. Rejected: caching an array's address (saves
   200 ns, but `resize(refcheck=False)` would corrupt memory silently); `__call__` by argument count (100 to
   150 ns, not worth its keyword edge cases).
