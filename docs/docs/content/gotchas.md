# Gotchas

A compiled function returns what CPython returns for the same function undecorated, and the differential tests hold it to that. This page lists every way we know of that it doesn't, and the things that work differently around a compiled call. Preflight points out the ones it can see in your code.

## Results that differ from Python

### A negative exponent on `n ** k` raises, in a function that writes

If `n` and `k` are ints and `k` isn't a constant, `n ** k` compiles to an int, and Python's result is an int or a float depending on the sign of `k`. Compiled code has one type per expression, so a negative `k` can't give Python's float.

In a function that writes nothing (no stores into arrays, lists or dicts, no calls to other compiled functions or C), that call starts again as Python, which gives its own result:

```python
@scram
def power(n: int, k: int):
    return n ** k

power(2, 3)     # 8, compiled
power(2, -1)    # 0.5, as Python: the call ran as Python
power(0, -1)    # ZeroDivisionError, as Python
```
<!-- c: power(2, 3) -->
```c-generated
int sj_kernel(int64_t v_n, int64_t v_k, const uint64_t *sj_m, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = v_n;
    int64_t t2 = v_k;
    int64_t t3 = 0;
    if ((int64_t)t2 < 0) SJ_FAIL(9, 8, 0, 0);
    t3 = sj_ipow((int64_t)t1, (int64_t)t2);
    sj_io[0] = (int64_t)(t3);
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

A function that has written something can't start again, so there a negative `k` raises `ValueError`. We chose to compile it and raise in the rare case, because `n ** k` is too common to run as Python and `k` is almost never negative. numpy makes the same choice for its integers. It's an error, never a wrong answer.

If `k` can be negative, write `float(n) ** k`, which always gives a float and compiles. A constant exponent is exact: `n ** 2` is an int and `n ** -1` is a float, as in Python. `scramjit preflight` adds a note to each line this applies to:

```text
  ✓ power  compiled
      note: line 3 runs as Python if the exponent is negative
```

### A negative base to a fractional power raises, in a function that writes

On a Python float, `x ** 0.5` and `x ** y` compile to a float. Where Python's result would be a complex, a function that writes nothing runs that call as Python, as above, and one that writes raises `ValueError`:

```python
@scram
def root(x: float, out):
    out[0] = x
    return x ** 0.5

root(4.0, a)    # 2.0, as in Python
root(-8.0, a)   # ValueError; Python returns (1.7319121124709868e-16+2.8284271247461903j)
```
<!-- c: root(4.0, np.zeros(1)) -->
```c-generated
int sj_kernel(double v_x, double *v_out, int64_t v_out__n, const uint64_t *sj_m, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL;
    int64_t t2 = t1 < 0 ? t1 + v_out__n : t1;
    if ((uint64_t)t2 >= (uint64_t)v_out__n) { sj_io[4] = 0; SJ_FAIL(1, 8, t1, v_out__n); }
    double t3 = v_x;
    v_out[t2] = t3;
    double t4 = v_x;
    double t5 = 0x1.0000000000000p-1;
    double t6 = 0;
    int t7 = sj_float_pow(SJ_LIBM2(23), t4, t5, &t6);
    if (t7 == 1) SJ_FAIL(2, 9, 6, 0);
    if (t7 == 2) SJ_FAIL(4, 9, 4, 0);
    if (t7 == 3) SJ_FAIL(3, 9, 17, 0);
    { double t8 = t6; memcpy(&sj_io[0], &t8, 8); }
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

As with `n ** k`, this is an error, never a wrong answer. (Where Python's complex result would itself overflow, Python raises `OverflowError`; compiled code still raises `ValueError`.) An integer exponent is exact, as in Python: `x ** 2`, `x ** n` and `(-8.0) ** 3.0` work. So is a nonnegative constant base: `2.0 ** x` never raises this.

numpy values follow numpy's rules: `a[i] ** 0.5` on a negative element is `nan`, in Python and compiled alike. With `@scram(parity=False)`, `x ** 0.5` is `nan` too. If you need the complex result, leave the function undecorated. `scramjit preflight` adds a note to each line this applies to.

### Integers are 64-bit, and wrap

Python's integers are unbounded. Compiled integers are `int64`, and past that they wrap silently, **giving a wrong answer with no error**:

| Python | Python's result | Compiled |
| --- | --- | --- |
| `2 ** 62 + 2 ** 62` | `9223372036854775808` | `-9223372036854775808` |
| `2 ** 70` | `1180591620717411303424` | `0` |
| `math.factorial(25)` | `15511210043330985984000000` | `7034535277573963776` |
| `int(1e22)`, `math.floor(1e22)` | `10000000000000000000000` | `1864712049423024128` |
| `1 << 63`, `1 << 64` | `9223372036854775808`, `18446744073709551616` | `-9223372036854775808`, `0` |

`>>` is always exact, and so is a shift masked to fewer than 64 bits, such as `(x << 13) & 0xFFFFFFFF` in a 32-bit hash: masking keeps only the bits wrapping leaves alone. On numpy integers, shifts follow numpy, where a count past the width gives 0.

64-bit hashes are exact too. A value that can only be in `[0, 2 ** 64)` is held as an unsigned 64-bit int: a literal such as `0xCBF29CE484222325`, a result masked with `& 0xFFFFFFFFFFFFFFFF`, a mask written `(1 << 64) - 1`, and a variable that only ever holds such values. So `h = ((h ^ int(b)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF` gives Python's numbers, and so do xorshift64 and splitmix64 ([example](https://github.com/farizrahman4u/scramjit/blob/main/examples/hash64.py)). A parameter that becomes such a variable, like a seed, is checked when called: a negative one runs as Python. Such a variable mustn't also be assigned something that may be negative, like `h - k`; the function then stays Python and says so. Elements of `uint64` arrays are `np.uint64`, with numpy's rules: they wrap with Python ints and unsigned dtypes, and become `float64` with signed numpy ints and floats, so `x[i] + y[i]` on `uint64` and `int64` is a float. `int(x[i])` gives the Python int.

Checking every operation would cost speed on every loop, and numeric code rarely comes near `2 ** 63`. If yours might, leave that function undecorated, or use floats. An argument that doesn't fit in 64 bits is safe: that call runs as Python.

### On Windows, a NaN's sign can differ

A few operations give a NaN with the opposite sign bit from CPython's on Windows: `inf % 1.0`, `inf // 1.0`, `math.atan2(0.0, nan)` and `nan * -1`. Both are NaN and compare the same; only `math.copysign` or the raw bytes can tell. Seen on Windows arm64; Linux matches exactly.

## Things that work differently around a compiled call

The results are the same in all of these. What changes is what you can observe while the code runs.

### Small functions get slower

Crossing from Python into compiled code costs about 1.5 µs per call for numbers and 2.5 µs with an array, on arm64 Linux. A small Python function takes about 35 ns, so decorating `def add(a, b): return a + b` makes it about 40x slower. The crossing pays for itself at around **30 to 50 loop iterations per call**:

| Iterations per call | Plain Python | Compiled | |
| --- | ---: | ---: | ---: |
| 1 | 146 ns | 2,426 ns | 17x slower |
| 20 | 1,589 ns | 2,470 ns | 1.6x slower |
| 50 | 3,867 ns | 2,512 ns | 1.5x faster |
| 100 | 7,609 ns | 2,565 ns | 3x faster |
| 1,000 | 79,694 ns | 3,522 ns | 21x faster |

So decorate the function that holds the loop, not only a helper called from one: [a call from compiled code](loops-and-builtins.md#calls-between-functions) has no such cost, and a small one is inlined. In Pyodide the crossing is about 13 µs, so the break-even is higher. `scramjit preflight` adds a note to decorated functions with no loop, and only suggests decorating functions that have one.

### Ctrl-C can take a moment, and has limits

Ctrl-C stops a compiled call: loops check for it every 65,536 iterations, and `KeyboardInterrupt` is raised as usual. Arrays keep what was written so far, just as if Python had been interrupted partway. A `while` loop checks every iteration, and a `for` loop over a short constant range, like `range(-1, 2)`, doesn't check at all, so the enclosing loop does. A long stretch of work between checks delays it.

Sometimes the call still runs to the end first, and `KeyboardInterrupt` arrives when it returns:

- **With your own SIGINT handler** (`signal.signal(signal.SIGINT, ...)`). Your handler might not raise, and a stopped call can't resume.
- **When something else uses `signal.set_wakeup_fd`**, such as asyncio's `add_signal_handler`.
- **On a thread other than the main one**, where Python doesn't raise `KeyboardInterrupt` either, so that call keeps running.
- **If the first compile happens off the main thread**, until one happens on it.
- **In Pyodide**, where there's no Ctrl-C to catch.

### A short loop inside a Python loop gains nothing

In a function that doesn't compile whole, each lifted loop is a call, which costs more than a few iterations do in Python. scramjit times a loop whose calls are all tiny both ways and keeps the faster, so this is bounded: a 3-iteration loop in a Python loop ran at 0.7x plain Python, a 40-iteration one at about the same speed, and longer ones far faster. Making the outer loop compile too removes the call.

### numpy

numpy's warnings and `errstate`, whole-array calls that are slower outside a loop, and results that depend on the CPU are in [Working with numpy](numpy.md#gotchas).

### Debuggers, tracers and coverage don't see inside

`pdb`, `sys.settrace` and coverage tools see no lines of a compiled function. It will look uncovered even when tests run it. To debug or measure a function, call the original: `f.__wrapped__(...)`.

### Globals are read once per call

A compiled function reads the module-level values it uses when it's called. If another thread rebinds one while it runs, as a compiled function releases the GIL, it carries on with the value it started with. Writes to the contents of a global array are seen straight away, as in Python.

### Threads really do run at the same time

Compiled code releases the GIL, which is the point, but it also removes what was slowing down data races. Four threads each running `a[0] += 1` five million times on one array ended at about 6.5 million instead of 20 million. Give each thread its own array or its own part of one, and combine the results after they finish.

## Options that change this on purpose

- **`@scram(parity=False)`** trades exact results for speed: float `/`, `//` and `%` by zero and `math` functions give IEEE `inf` and `nan` instead of raising, `math` functions come from cimo's C library, and `x ** 0.5` gives `nan` for a negative base instead of raising.
- **`@scram(boundscheck=False)`** removes index checks. An index out of range then reads or writes other memory instead of raising `IndexError`, which can return garbage or crash the process. Use it only on code you've verified with checks on.

## What isn't a gotcha

Code that can't be compiled runs as Python, so its results can't differ, only its speed: anything preflight marks as running as Python. See [fallback and errors](errors.md).
