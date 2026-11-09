# Working with numpy

scramjit works on numpy arrays in place and gives numpy's results in compiled loops. This page has what compiles, how it's held, and where it differs. Anything else runs as Python with numpy's own result, and the loops around it still compile if they fit.

## Arrays

Arrays are passed by pointer, shape and strides, never copied. Writes go to your array. The generated code is specialized to the dtype, the number of dimensions and the layout, so a C-contiguous array gets plain `a[i]` indexing.

Negative indices count from the end, as in Python. Indices are bounds-checked by default and raise `IndexError`; see [Semantics](semantics.md).

## Views

Indexing with fewer indices than dimensions, or with slices, gives a view, as in numpy. Nothing is copied and nothing is allocated, so a view costs a few variables, and reading through one runs as fast as `m[i, j]`:

```python
@scram
def normalize(m):
    for row in m:                   # each row is a view
        top = row[0]
        for j in range(len(row)):
            row[j] /= top

    first = m[:, 0]                 # a column
    tail = m[1:, 1:]                # a block
    backwards = first[::-1]         # any step, and views of views
    return first[0] + tail[0, 0] + backwards[0] + m[0][0]
```

Writes go to the array, and show through every other view of it. Slices follow Python's rules: bounds clamp, negatives count from the end, and a step of zero raises `ValueError`. `len()`, `.shape`, `.size` and `.ndim` work on a view.

A variable can be assigned a different view later, as `x = x[1:]` does. Passing one in works too, so `f(x[::2])` and `f(m.T)` compile, and so does returning one: you get a real numpy array sharing the memory, or the very same array if the function returns an argument untouched. `.T` works, and rows work in `for`, `enumerate()` and `zip()`, and in `a, b = m[0], m[1]`.

### Assigning to a slice

```python
x[1:3] = 0.0            # a number, into any view
m[i] = row              # an array, broadcast as numpy does
m[1:3] = row            # into each row
m[:, j] = m[:, k]       # columns
x[1:] = x[:-1]          # overlapping: as if the right side were copied first
```

A shape that doesn't broadcast raises numpy's `ValueError`. An array can go into a view of up to two dimensions; a number into any. Storing floats into an integer array runs as Python.

## What compiles

| | Compiles | Runs as Python |
| --- | --- | --- |
| Indexing | ints, slices of any step, `.T`, `m[i][j]` | `None`, `...`, masks `x[x > 0]`, index arrays `x[idx]` |
| Assigning | an element; a number or an array into a slice | `x[1:] += y`, a list into a slice |
| Functions on one number | [these](#functions-on-single-numbers) | `np.tanh`, `np.power` |
| Reductions to one number | [`sum`, `mean`, `std`, `var`, `max`, `argmax` and the rest](#reductions) | `axis=` on more than one dimension, `keepdims=True` |
| Making arrays | [`np.zeros`, `x.copy()`, `np.arange` and the rest](#making-arrays) | `x.reshape(...)`, `np.eye(n)` |
| Whole-array arithmetic | [`a * 2 + b`, `np.sqrt(a)`, broadcasting](#whole-array-arithmetic) | `//`, `%`, comparisons, more than two dimensions |
| Element types | `float64`, `int64`, and [`float32`, `int32`, `uint8` and friends](#narrow-dtypes), with arithmetic and most reductions, and `uint64` | on `float32` `np.sin`, `np.exp` and the like, `.std()`, `.var()` |
| Other | `len()`, `.shape`, `.size`, `.ndim`, returning arrays and views, alone or in a tuple | `np.dot`, `np.linalg`, `np.sort`, `np.cumsum` |

Anything in the right column runs as Python with numpy's own result, and the loops around it still compile if they fit.

## Making arrays

A function can make its own arrays, and what it returns is an ordinary numpy array, with the dtype and shape numpy would have given it:

```python
@scram
def smooth(x, passes):
    cur = x.copy()
    for _ in range(passes):
        nxt = np.zeros_like(cur)
        for i in range(1, len(cur) - 1):
            nxt[i] = (cur[i - 1] + cur[i] + cur[i + 1]) / 3
        cur = nxt
    return cur
```

These compile, up to four dimensions:

- `np.zeros`, `np.ones`, `np.empty`, `np.full`, and the same with `_like`
- `x.copy()`, `np.copy(x)`, `np.array(x)`, `x.flatten()`, `x.astype(...)` to the same kind of number or from integer to float, and `np.asarray(x)`, which is `x` itself
- `np.arange`, `np.linspace`
- `np.array([a, b])` and `np.array([[a, b], [c, d]])` of numbers, and of a local list with `dtype=` if it holds ints, since an empty list would give floats

They take `dtype=` with `float64`, `int64` and the narrower types. `order=` and `like=` stay in Python, as do a `_like` of an array in another memory layout, and `np.array` of mixed lists. Errors are numpy's own, on the line that asked.

Arrays no variable holds any more are freed while the function runs, so a loop making a new array each time doesn't build up memory. A small array made inside a loop costs about 24 ns, against 256 for numpy from Python. A very large one is a little slower than numpy's, at about 11 µs against 7 for 800 KB.

A returned array is copied once, into memory numpy owns, so that `resize` and everything else behave as they do on any array. Both copies exist for a moment, and a function returning a small array costs about 3 µs, against 0.15 for `np.zeros` itself: below a few hundred loop iterations, or for arrays near your memory limit, take the array as an argument instead.

## Whole-array arithmetic

Operators and numpy functions on whole arrays compile, giving numpy's result: the same dtype, shape and values, bit for bit.

```python
@scram
def step(y, x, k):
    for _ in range(10):
        y = y * 0.5 + np.sqrt(x) * k
    return y
```

- `+`, `-`, `*`, `/` and `**` between arrays and numbers, and unary `-`
- the numpy functions [on single numbers](#functions-on-single-numbers) applied to arrays: `np.sqrt`, `np.exp`, `np.sin`, `np.abs`, `np.maximum` and the rest
- broadcasting, as numpy does it: `m + row`, `col + row`, a number with an array; a shape that doesn't broadcast raises numpy's `ValueError`
- arrays of up to two dimensions, of `float64`, `int64` and the [narrow dtypes](#narrow-dtypes), and views of them, except a two-dimensional view with gaps such as `m.T`

A whole expression is computed in one pass, element by element, with no temporary arrays in between. The numbers in it must be plain values: names, constants, elements such as `w[k]`, and arithmetic on those. A call such as `a * f()` runs as Python, as do `//`, `%`, comparisons that give arrays of booleans, and an integer array to a power that may be negative. The loops around them still compile.

Up to tens of thousands of elements, this is faster than numpy from Python (`y = y * 0.5 + b` ten times: 7 µs against 12 for 1,000 elements, 42 against 59 for 10,000). On 1,000,000 elements it's still a little faster: ten passes took 6.0 ms against numpy's 6.9.

## Functions on single numbers

These compile on a single number, such as `np.sqrt(x[i])`, and give numpy's result and type, `np.float64`:

- `np.sqrt`, `np.exp`, `np.exp2`, `np.expm1`, `np.log`, `np.log2`, `np.log10`, `np.log1p`, `np.cbrt`
- `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`, `np.arctan`, `np.sinh`, `np.cosh`, `np.arcsinh`, `np.arccosh`, `np.arctanh`
- `np.abs`, `np.absolute`, `np.fabs`, `np.sign`, `np.square`, `np.floor`, `np.ceil`, `np.trunc`, `np.rint`
- `np.minimum`, `np.maximum`, `np.fmin`, `np.fmax`, `np.hypot`, `np.arctan2`, `np.fmod`, `np.copysign`
- `np.isnan`, `np.isinf`, `np.isfinite`, `np.signbit`, `np.float64(x)`, `np.int64(n)`
- on floats only: `np.round(x)`, `np.clip(x, lo, hi)`, `np.degrees`, `np.radians`, `np.rad2deg`, `np.deg2rad`
- the constants `np.pi`, `np.e`, `np.inf`, `np.nan`

Arguments can be Python ints and floats, `np.int64` and `np.float64`. A few combinations run as Python, because what numpy returns for them depends on the platform or the numpy version:

- `np.abs`, `np.sign`, `np.square`, `np.minimum` and `np.maximum` of Python ints: `np.abs(4)` is `int64` on most platforms and `int32` in Pyodide
- `np.floor`, `np.ceil` and `np.trunc` of any int: numpy 2.1 changed what these return

Bools and narrow types such as `np.float32` or `np.uint8` run as Python too. So do `np.tanh`, which is numpy's own and doesn't match the C library's in the last bits, and `np.power`, which has shortcuts for some exponents; use `**`. Other calls on whole arrays, such as `np.sqrt(x)`, run as Python for now.

## Reductions

`sum`, `prod`, `mean`, `std`, `var`, `max`, `min`, `any`, `all`, `argmax` and `argmin` compile as methods and as functions (`x.sum()`, `np.sum(x)`, and `np.amax` and `np.amin`), on arrays and on views. They give numpy's exact result, down to rounding, since `np.sum` adds in a particular order, and numpy's type:

```python
@scram
def spread(m):
    total = 0.0
    for i in range(len(m)):
        total += m[i].max() - m[i].min()
    return total / m.size
```
<!-- c: spread(np.ones((3, 4))) -->
```c-generated
int sj_kernel(double *v_m, int64_t v_m__n, int64_t v_m__n1, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    double v_total = 0;
    int v_total__np = 0;
    int64_t v_i = 0;
    double *v_sjv__0 = 0; int64_t v_sjv__0__n = 0;
    double *v_sjv__1 = 0; int64_t v_sjv__1__n = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    v_total = 0x0.0p+0;
    v_total__np = 0;
    int64_t t1 = 0LL, t2 = v_m__n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v_i = t3;
            double t6 = v_total;
            int t7 = v_total__np;
            int64_t t8 = v_i;
            int64_t t9 = 0;
            int64_t t10 = t8 < 0 ? t8 + v_m__n : t8;
            if ((uint64_t)t10 >= (uint64_t)v_m__n) { sj_io[4] = 0; SJ_FAIL(1, 5, t8, v_m__n); }
            t9 += t10 * (1LL * v_m__n1);
            v_sjv__0 = v_m + t9;
            v_sjv__0__n = v_m__n1;
            int64_t t11 = 1LL * v_sjv__0__n;
            int64_t t12 = 1LL;
            if (t11 == 0) SJ_FAIL(3, 5, 19, 0);
            double t13 = sj_np_max_f(v_sjv__0, t11, t12, 1);
            double t14 = t13;
            int64_t t15 = v_i;
            int64_t t16 = 0;
            int64_t t17 = t15 < 0 ? t15 + v_m__n : t15;
            if ((uint64_t)t17 >= (uint64_t)v_m__n) { sj_io[4] = 0; SJ_FAIL(1, 5, t15, v_m__n); }
            t16 += t17 * (1LL * v_m__n1);
            v_sjv__1 = v_m + t16;
            v_sjv__1__n = v_m__n1;
            int64_t t18 = 1LL * v_sjv__1__n;
            int64_t t19 = 1LL;
            if (t18 == 0) SJ_FAIL(3, 5, 20, 0);
            double t20 = sj_np_min_f(v_sjv__1, t18, t19, 1);
            double t21 = t20;
            double t22 = (t14 - t21);
            v_total = (t6 + t22);
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    if (t2 > t1) { v_total__np = 1; }
    double t23 = v_total;
    int t24 = v_total__np;
    int64_t t25 = (v_m__n * v_m__n1);
    if (!t24 && (double)(t25) == 0.0) SJ_FAIL(2, 6, 1, 0);
    { double t26 = (t23 / (double)(t25)); memcpy(&sj_io[0], &t26, 8); }
    sj_io[2] = t24;
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Inside a compiled loop there's no call to pay for, so short rows gain the most: `m[i].sum()` took 8 ns on rows of 8, against 1.2 µs in Python. They work on `float64` and `int64` arrays, of any shape if the elements are contiguous, and on one-dimensional views with gaps. `std` and `var` take `ddof=`.

Only forms that give one number compile. Everything else runs as Python, with numpy's result:

| Call | |
| --- | --- |
| `x.sum()`, `x.sum(axis=None)`, `x.sum(keepdims=False)` | compiles, any number of dimensions |
| `x.sum(axis=0)`, `x.sum(axis=-1)`, `x.sum(0)` on one dimension | compiles |
| `x.sum(axis=0)` on more than one dimension | Python: it returns an array |
| `x.sum(axis=(0, 1))`, an axis held in a variable | Python |
| `x.sum(keepdims=True)` | Python: it returns an array |
| `dtype=`, `out=`, `where=`, `initial=`, and `mean=`, or a `ddof` that isn't an integer | Python |

`x.sum(axis=1)` on a loop's hot path is worth writing as a loop over `m[i].sum()`, which compiles.

## Narrow dtypes

Arithmetic on `float32`, `int32`, `int16`, `int8`, `uint32`, `uint16` and `uint8` compiles, as array elements and as numpy scalars passed in, and gives what numpy gives:

```python
@scram
def brighten(image, amount):
    for i in range(len(image)):
        image[i] = image[i] + amount    # 250 + 10 wraps to 4
```
<!-- c: brighten(np.zeros(8, np.uint8), 10) -->
```c-generated
int sj_kernel(uint8_t *v_image, int64_t v_image__n, int64_t v_amount, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_image__n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v_i = t3;
            int64_t t6 = v_i;
            int64_t t7 = t6;
            uint8_t t8 = v_image[t7];
            uint8_t t9 = t8;
            int64_t t10 = v_amount;
            int64_t t11 = t10;
            if (t11 < 0LL || t11 > 255LL) SJ_FAIL(7, 4, t11, 5);
            int64_t t12 = v_i;
            int64_t t13 = t12;
            uint8_t t14 = ((uint8_t)((uint32_t)t9 + (uint32_t)t10));
            v_image[t13] = t14;
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

| With a `uint8` or `float32` | Result |
|:---|:---|
| a Python int | the same dtype; on an integer dtype an int that doesn't fit raises `OverflowError`, except with `/` |
| a Python float | `float32` stays `float32`; an integer dtype gives `float64` |
| another numpy scalar | numpy's promotion: `uint8` with `int8` is `int16`, `int32` with `float32` is `float64` |
| `/` of two integers | `float64` |

Comparisons follow the same rules, so `x[i] == 0.1` on a `float32` is true when `x[i]` is `np.float32(0.1)`.

Half the memory per element also means twice the numbers in each vector register. On arm64, `y[i] = a * x[i] + y[i]` takes 0.15 ns per element on `float32` against 0.26 on `float64`. Bytes and shorts compile but don't vectorize yet. A running float total doesn't vectorize on any dtype, since parity keeps the order of its additions.

A total may start as a literal, and is numpy's from the first `+=`:

```python
total = 0.0              # the Python float 0.0 if the loop doesn't run
for i in range(len(x)):
    total += x[i] * y[i]     # np.float32 from here on
```

The literal must fit the dtype exactly: `0`, `0.0` or `0.5` on `float32`, not `0.1` or `16777217`, and on an integer dtype an int, not a float. Starting from `0` on `float32`, the result is the int `0` when the loop doesn't run, as in Python. Such a variable can then only meet numpy scalars, or be compared with an int if it holds ints. Anything else keeps the function in Python.

Also compiled, with numpy's result:

- `**`: an integer dtype wraps, and a negative exponent raises `ValueError`. `float32` uses the C library's `powf`, as numpy does.
- `sum()`: the total wraps or rounds at each step, as in Python with numpy scalars. Starting from Python's `0`, it's `0` when there's nothing to add, on `float32` too.
- On `float32` scalars: `np.sqrt`, `np.abs`, `np.square`, `np.sign`, `np.floor`, `np.ceil`, `np.trunc`, `np.rint`, `np.copysign`, `np.minimum` and `np.maximum`, the tests such as `np.isnan`, and conversions such as `np.float32(x)` or `np.uint8(n)`. On integer dtypes, the same functions apart from those numpy gives a float for, since it picks a float as narrow as the int (`np.sqrt` of a `uint8` is a `float16`).
- Reductions: `.max()`, `.min()`, `.argmax()`, `.argmin()`, `.any()` and `.all()`; `.sum()`, `.prod()` and `.mean()`, with an unsigned int's total as `np.uint64`, as numpy gives it; `.std()` and `.var()` on integer dtypes. On `float32`, `.std()` and `.var()` stay Python.

Other numpy functions, such as `np.sin` or `np.exp` on `float32`, stay in Python: numpy has its own versions, whose last bits vary by CPU.

## Gotchas

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

### A few whole-array calls are slower outside a loop

On 1,000,000 `float64` elements on arm64, compiled `x.max()` took 0.18 ms against numpy's 0.11, and `x.argmax()` 0.26 against 0.31; `x.all()` and `x.sum()` are within 10% of numpy's. Narrower dtypes widen the gap, as numpy fits more in a register (`float32` `x.max()`: 0.10 ms against 0.06). Inside a loop, where a numpy call costs about a microsecond each time, compiled code is far faster unless the arrays are huge. Preflight notes the calls this can affect that aren't in a loop.

### min and max of two different NaNs

`x.max()` with NaNs in the array gives NaN, as numpy does. If those NaNs have different signs, which one you get can differ, since numpy's depends on how its loops are laid out.

### numpy's exact results depend on the CPU it uses

The numpy functions that compile matched numpy bit for bit on arm64, in WebAssembly, and on x86-64 with AVX2. numpy can pick its own routines by CPU, notably with AVX-512, for functions such as `np.exp` and `np.sin`. Those may differ in the last bits from the C library's that compiled code calls. That case hasn't been checked. Results agree to rounding.

## Python data structures

Tuples, lists, lists of lists, dicts, instances, enums and module-level constants have a page of their own: [Python data structures](data-structures.md).
