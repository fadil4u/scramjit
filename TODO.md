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

1. **Value-range narrowing:** the rest of M2's loop work, using the range facts the compiler already has.
2. **Bounds-check hoisting beyond unit-step loops** over one variable (versioning covers
   `for i in range(a, len(x))` with `arr[i + c]` today).
3. **Store-to-load forwarding:** `y[i]` written then `y[i - 1]` read next iteration; keeping it in
   a register gives `ema` 2.39 ns against 3.07 (cimo doesn't do it).
4. **Loop lifting (M4):** compile the loops that fit inside a function that doesn't, and let Python run the rest,
   which makes "partly compiled" real. The risk is matching Python when a loop raises partway: every variable
   must hold what it would in plain Python. Build it in small steps with differential tests from the start. It
   also gives "continue in Python", which could remove the `**` errors below.
5. **Python data structures, nested, when statically sized and statically typed, after the loop work above:** so
   that containers don't hide speedups we can still get from loops, each step must leave `bench/` unchanged.
   Dicts, lists, tuples, classes and enums that don't resize at runtime and whose element types don't change. See "Containers" in
   `requirements.md`. Today only flat tuples of numbers and flat local lists compile. In this order, simplest
   and most useful first:
   - **Module-level constants:** `K = 3` and `TABLE = (1.0, 2.0)` read by a function. Both fall back today. Compile
     them as constants, so loops can use them, and check each on every call, to recompile if it was rebound.
     Measured: 21 to 25 ns per constant on a 1.4 µs call, nothing for functions that read none. Compare with `==`
     after `is`, so an equal new object doesn't recompile, and stop recompiling a global that keeps changing.
   - **Enums:** `Level.HIGH`, comparisons, and members as arguments, as their values.
   - **Nested tuples:** `((1, 2), 3)`, and tuples holding arrays, as locals, arguments and results.
   - **Lists as arguments, and nested:** `[[1.0, 2.0], [3.0, 4.0]]`; returning one. The spec's copy in and
     out at the call was never built.
   - **Classes:** attribute reads and writes on instances and dataclasses, then method calls.
   - **Fixed dicts only, for now:** `d['a']` on a dict whose keys don't change, as a struct. A function that could
     add a key runs as Python.

   To settle first, all of it needing differential tests: what happens when a value breaks the rule at runtime
   (an `int` slot given a float); two names for one object (`[[0] * 3] * 2`, one list passed twice),
   since writes must show through both; the cost of checking and copying a whole structure on every call; and
   results that must come back as real Python objects.
6. **Call boundary cost:** about 1.4 µs with numbers (was 3.2) and 2.4 µs with an array (was 6.6), through
   a function generated per signature (`_make_plan`). For numbers: ctypes about 450 ns (the floor), cimo's
   `Kernel.__call__` about 450 (see cimo's `TODO.md`), scramjit about 500. In Pyodide about 13 µs, not yet
   broken down. Break-even is 30 to 50 loop iterations per call. Rejected: caching an array's address (saves
   200 ns, but `resize(refcheck=False)` would corrupt memory silently); `__call__` by argument count (100 to
   150 ns, not worth its keyword edge cases).
7. **SSA for variables that change type** (`x = 1` then `x = 0.5`): today they fall back.
8. **Rest of M2:** `for ... else`. Then the M2 acceptance check.
9. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`), and a `disk` label in the compile column,
   which needs cimo to say whether a compile hit its cache.
10. **`n ** k` with a negative `k` and `x ** 0.5` with a negative `x` raise today** (see Gotchas). Rerunning
   in Python, for functions that write nothing, or resuming in Python alongside loop lifting would remove
   the difference.
11. **A NaN's sign differs from CPython's on Windows** (arm64, the only one tried). `inf % 1.0`, `inf // 1.0`
   and `math.atan2(0.0, nan)` give `7ff8...` where CPython gives `fff8...`, and `nan * -1` gives `fff8...`
   where it gives `7ff8...`; `tests/test_m1.py` and `test_m2.py` fail there on these, 7 in all. The values are
   all NaN, and only `math.copysign` or the raw bytes can tell. Likely fix: call the CRT's `fmod` through the
   libm table, as for `sin`, and make the `* -1` negation depend on the platform. Nothing runs scramjit's tests
   on Windows automatically yet.
12. **Test Ctrl-C on macOS.** `tests/test_interrupt.py` has run on Linux and Windows arm64, not macOS. It
   should work, since it uses `signal.set_wakeup_fd` and sockets, but nothing has checked it.
13. **`parity=False` asks cimo for FMA, and it never happens.** cimo contracts `a * b + c` within one C
    expression, and scramjit gives each operation its own temporary. Nothing is lost on reductions:
    FMA makes them slower on arm64 (a `dot` through cimo: 2.15 ns per element with, 1.81 without;
    `bench/llm.py` through `gcc -O3`: 63 tok/s with, 82 without), probably the latency of the
    instruction the sum's next add waits on (not measured). Measure non-reductions (stencils) before making it fire or removing it.
