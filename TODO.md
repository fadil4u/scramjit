# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## From cimo

- **Signed overflow is now undefined in cimo, as in C** (`compile(wrapv=True)` for `-fwrapv`): `x + 1 > x` folds to true
  and `int` counters widen to 64 bits. Generated C must keep doing wrapping arithmetic through unsigned, as it
  does today (scramjit's 726 tests pass).

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
2. **Bounds-check hoisting, what's left:** versioning covers `arr[i + c]` and, in innermost loops, `arr[base + i]`
   with `base` built from Python ints that don't change in the loop. Not yet: steps other than 1 (RoPE's
   `range(0, dim, 2)` in `bench/llm.py`), a start that isn't a literal or `len()`, numpy int bases, and outer
   loops, where a second copy of everything inside costs more than it wins.
3. **Store-to-load forwarding, what's left:** `y[i - 1]` comes from a temporary when `y[i] = ...` is the
   loop's only array store and runs every iteration. Not yet: `y[i - 2]`, 2-D rows (`m[r, j - 1]`), and loops
   with other stores, which need a runtime test that the arrays don't overlap.
4. **Loop lifting, what's left:** a loop stays in Python if a variable read afterwards may be unbound
   (`for i ...` then reading `i`, with nothing assigned first), inside `try` or `with`, or containing `return`.
   "Partly compiled" is per function, not per signature. Tracebacks show one extra frame for a loop that raises.
5. **numpy calls in compiled functions:** `np.sqrt(x[i])`, `np.sum(x)`, `x.sum()`, `m[i].sum()` and `x * 2` all run as Python today.
   No cimo work is required for any step; its SIMD only adds speed. In this order:
   - **numpy functions on single numbers, first:** `np.sqrt`, `np.abs`, `np.exp`, `np.floor`, `np.minimum` and friends. No allocation, and
     probably the most common reason a numeric loop won't compile. Test bit for bit first: numpy may use its own `exp` and `sin`, not
     libm's. What differs gets ported, as in `_mathlib.py`, or stays in Python.
   - **Reductions on a whole array:** `sum`, `prod`, `min`, `max`, `mean`, `any`, `all`, `argmax`. A plain loop doesn't match `np.sum` (it differs from
     about 100 elements up), so port numpy's pairwise order. The gain is inside loops: outside one, lifting already gives numpy for the
     reduction and native code for the loop. numpy is faster on big arrays, with SIMD: at 1,000,000 elements `np.sum` 162 µs against our loop's
     1,019, `np.max` 114 against 1,358; at 8, 1.5 µs against 2.8, nearly all call overhead. So numpy should win above roughly 1,000 to 2,000
     elements, an estimate. **Undecided:** should a function that compiles whole because of this get slower on huge arrays until cimo
     has SIMD, with a gotcha and a preflight note, or do whole-array reductions stay in Python until then? Recommended: the former.
   - **Views:** `m[i]`, `x[a:b]`, only a pointer and a length. Reductions are much less useful without them.
   - **Not now, operations that create arrays:** `x * 2`, `a + b` need allocation, lifetimes, broadcasting and dtype rules, which is rewriting numpy.
     They stay in Python, with the loops around them lifted.
   - **Not `np.dot`, and no MKL or BLAS:** they would break "standard library plus cimo", and their results depend on the library, CPU and thread count.
     `A @ B` and `np.linalg` already run at full speed in Python. An opt-in through cimo's `link=` for BLAS inside a hot loop is possible
     later; untried.
6. **Python data structures, nested, when statically sized and statically typed, after the loop work above:** so
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
   - **Dicts, what's left:** a dict passed in compiles (see "Containers"). Not yet: nested dicts, a dict built in the function, returning
     one, `len(p)` and `'a' in p`, which are known from the keys. A function that could add a key still runs as Python.

   To settle first, all of it needing differential tests: what happens when a value breaks the rule at runtime
   (an `int` slot given a float); two names for one object (`[[0] * 3] * 2`, one list passed twice),
   since writes must show through both; the cost of checking and copying a whole structure on every call; and
   results that must come back as real Python objects.
7. **Call boundary cost:** about 1.4 µs with numbers (was 3.2) and 2.4 µs with an array (was 6.6), through
   a function generated per signature (`_make_plan`). For numbers: ctypes about 450 ns (the floor), cimo's
   `Kernel.__call__` about 450 (see cimo's `TODO.md`), scramjit about 500. In Pyodide about 13 µs, not yet
   broken down. Break-even is 30 to 50 loop iterations per call. Rejected: caching an array's address (saves
   200 ns, but `resize(refcheck=False)` would corrupt memory silently); `__call__` by argument count (100 to
   150 ns, not worth its keyword edge cases).
8. **Variables that change type, what's left:** a name used for separate things compiles (each web is its own
   variable), and so does an int that becomes a float, where the int is a literal up to 2^53 or a parameter, and it's only
   returned or combined with a float. Not yet: other uses of such a variable (`-s`, `s + 1`, `t = s`, `a[s]`, `if s:`), and
   `s = n` from an int variable, each of which keeps the function in Python.
9. **Narrow arithmetic, the part of M2 not built:** arithmetic on `int32`, `uint8` and the like, as numpy does it (NEP 50).
   Value-range narrowing (item 1) has little to win until cimo vectorizes.
10. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`), and a `disk` label in the compile column,
   which needs cimo to say whether a compile hit its cache.
11. **`n ** k` with a negative `k` and `x ** 0.5` with a negative `x` raise today** (see Gotchas). Rerunning
   in Python, for functions that write nothing, or resuming in Python alongside loop lifting would remove
   the difference.
12. **A NaN's sign differs from CPython's on Windows** (arm64, the only one tried). `inf % 1.0`, `inf // 1.0`
   and `math.atan2(0.0, nan)` give `7ff8...` where CPython gives `fff8...`, and `nan * -1` gives `fff8...`
   where it gives `7ff8...`; `tests/test_m1.py` and `test_m2.py` fail there on these, 7 in all. The values are
   all NaN, and only `math.copysign` or the raw bytes can tell. Likely fix: call the CRT's `fmod` through the
   libm table, as for `sin`, and make the `* -1` negation depend on the platform. Nothing runs scramjit's tests
   on Windows automatically yet.
13. **Test Ctrl-C on macOS.** `tests/test_interrupt.py` has run on Linux and Windows arm64, not macOS. It
   should work, since it uses `signal.set_wakeup_fd` and sockets, but nothing has checked it.
14. **`parity=False` asks cimo for FMA, and it never happens.** cimo contracts `a * b + c` within one C
    expression, and scramjit gives each operation its own temporary. Nothing is lost on reductions:
    FMA makes them slower on arm64 (a `dot` through cimo: 2.15 ns per element with, 1.81 without;
    `bench/llm.py` through `gcc -O3`: 63 tok/s with, 82 without), probably the latency of the
    instruction the sum's next add waits on (not measured). Measure non-reductions (stencils) before making it fire or removing it.
15. **Shift operators:** `>>` and `<<` fall back ("the RShift operator"). Hash, RNG and bit-packing loops use
    them constantly, and a model writing code reaches for them first. Python's ints don't wrap on `<<`, so decide
    how that fits the 64-bit rule in Gotchas.
16. **Calls between compiled functions** (already noted in the spec under "Calls between compiled functions"). A
    program built from `reset`, `step` and `observe` has to be pasted into one function by hand today: found
    building a training loop around generated environments, where it is the main obstacle.
17. **Compile time jumps on large functions.** In a generated 200-line training loop, adding a 4-line nested
    copy (`rec[t, y, x] = grid[y, x]` under one `if`) took the compile from about 3 s to 11 to 15 s on a loaded
    machine. The cause is bounds-check versioning: a planner over the same world is 4,410 lines of C and 205 loops
    with checks (16.4 s in cimo), 1,611 lines and 50 loops without (2.5 s). Limit how many copies versioning makes
    in one function. Compile time decides whether generated code is practical, since each candidate compiles once.
18. **float32 arithmetic falls back, with a misleading reason:** "arithmetic on numpy bools", because
    `arith_result` returns None for narrow types too. Say what is actually unsupported, then compile it:
    float64 doubles the memory of anything with float32 weights.
