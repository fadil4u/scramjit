# Gotchas

A compiled function returns what CPython returns for the same function undecorated, and the differential tests hold it to that. This page lists every way we know of that it doesn't, and the things that work differently around a compiled call. Preflight points out the ones it can see in your code.

## Results that differ from Python

### A negative exponent on `n ** k` raises

If `n` and `k` are ints and `k` isn't a constant, `n ** k` compiles to an int. A negative `k` raises `ValueError`:

```python
@scram
def power(n: int, k: int):
    return n ** k

power(2, 3)     # 8, as in Python
power(2, -1)    # ValueError; Python returns 0.5
power(0, -1)    # ValueError; Python raises ZeroDivisionError
```

Compiled code has one type per expression, and Python's result here is an int or a float depending on the sign of `k`. We chose to compile it and raise in the rare case, because `n ** k` is too common to run as Python and `k` is almost never negative. numpy makes the same choice for its integers. It's an error, never a wrong answer.

If `k` can be negative, write `float(n) ** k`, which always gives a float and compiles. A constant exponent is exact: `n ** 2` is an int and `n ** -1` is a float, as in Python. `scramjit preflight` adds a note to each line this applies to:

```text
  ✓ power  compiled
      note: line 3 raises ValueError if the exponent is negative, where Python gives a float
```

### A negative base to a fractional power raises

On a Python float, `x ** 0.5` and `x ** y` compile to a float. Where Python's result would be a complex, they raise `ValueError`:

```python
@scram
def root(x: float):
    return x ** 0.5

root(4.0)       # 2.0, as in Python
root(-8.0)      # ValueError; Python returns (1.7319121124709868e-16+2.8284271247461903j)
```

As with `n ** k`, compiled code has one type per expression, and this is an error, never a wrong answer. (Where Python's complex result would itself overflow, Python raises `OverflowError`; compiled code still raises `ValueError`.) An integer exponent is exact, as in Python: `x ** 2`, `x ** n` and `(-8.0) ** 3.0` work. So is a nonnegative constant base: `2.0 ** x` never raises this.

numpy values follow numpy's rules: `a[i] ** 0.5` on a negative element is `nan`, in Python and compiled alike. With `@scram(parity=False)`, `x ** 0.5` is `nan` too. If you need the complex result, leave the function undecorated. `scramjit preflight` adds a note to each line this applies to.

### Integers are 64-bit, and wrap

Python's integers are unbounded. Compiled integers are `int64`, and past that they wrap silently, **giving a wrong answer with no error**:

| Python | Python's result | Compiled |
| --- | --- | --- |
| `2 ** 62 + 2 ** 62` | `9223372036854775808` | `-9223372036854775808` |
| `2 ** 70` | `1180591620717411303424` | `0` |
| `math.factorial(25)` | `15511210043330985984000000` | `7034535277573963776` |
| `int(1e22)`, `math.floor(1e22)` | `10000000000000000000000` | `1864712049423024128` |

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

### numpy's warnings and `errstate` don't apply

Compiled code does the arithmetic numpy would, and gets the same result, but numpy never sees it:

```python
def divide(a):
    return a[0] / a[1]

a = np.array([1.0, 0.0])
divide(a)                       # inf, with a RuntimeWarning
scram(divide)(a)                # inf, with no warning

with np.errstate(all='raise'):
    divide(a)                   # FloatingPointError
    scram(divide)(a)            # inf
```

Check for `inf` and `nan` yourself if you rely on these.

### A short loop inside a Python loop gains nothing

In a function that doesn't compile whole, each lifted loop is a call, which costs more than a few iterations do in Python. scramjit times a loop whose calls are all tiny both ways and keeps the faster, so this is bounded: a 3-iteration loop in a Python loop ran at 0.7x plain Python, a 40-iteration one at about the same speed, and longer ones far faster. Making the outer loop compile too removes the call.

### A few whole-array calls are slower outside a loop

numpy runs `max`, `min`, `argmax`, `argmin`, `any` and `all` with SIMD, which compiled code doesn't have yet. On 1,000,000 elements `x.max()` took 1.6 ms against numpy's 0.18, and `x.all()` 0.5 against 0.2. `x.sum()` and `x.mean()` are within about 15%, since numpy's sum isn't SIMD either. Inside a loop, where a numpy call costs about a microsecond each time, compiled code is far faster unless the arrays are huge. Preflight notes the calls this can affect that aren't in a loop.

### min and max of two different NaNs

`x.max()` with NaNs in the array gives NaN, as numpy does. If those NaNs have different signs, which one you get can differ, since numpy's depends on how its loops are laid out.

### numpy's exact results depend on the CPU it uses

The numpy functions that compile matched numpy bit for bit on arm64, in WebAssembly, and on x86-64 with AVX2. numpy can pick its own routines by CPU, notably with AVX-512, for functions such as `np.exp` and `np.sin`. Those may differ in the last bits from the C library's that compiled code calls. That case hasn't been checked. Results agree to rounding.

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
