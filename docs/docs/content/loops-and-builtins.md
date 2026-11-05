# Loops and builtins

No builtin exists at runtime as a callable. Each one is a rule in the compiler that emits plain C, so `range`, `zip` and generator expressions cost nothing beyond the loop they describe. Iterator protocols (`__iter__`, `__next__`) never appear in generated code.

## `range`

```python
for i in range(a, b, s):
    ...
```

becomes

```c
for (int64_t i = a; i < b; i += s) {
    ...
}
```

No range object exists. A step that isn't a constant gets the sign check Python implies, so a negative step counts down as it does in Python.

`for` and `while` can have an `else`, which runs unless a `break` left the loop:

```python
for i in range(len(x)):
    if x[i] == target:
        found = i
        break
else:
    found = -1
```

## `enumerate`, `zip`, `reversed`

Over arrays and ranges these become index loops:

```python
for i, (u, v) in enumerate(zip(a, b)):
    out[i] = u * v
```

is one counted loop over the shorter of `a` and `b`, reading `a[i]` and `b[i]` directly.

A tuple works as well, when its elements share a type: `for w in (0.25, 0.5, 0.25)`, `for (x, y) in (p, q)`. It's built first, as in Python, then read by index.

## Comprehensions, `map`, `filter` and generator expressions

These fuse into one loop with no intermediate iterators. A result is materialized only when it is stored:

```python
total = sum(x * x for x in a if x > 0)
```

reduces in place:

```c
double total = 0;
for (int64_t i = 0; i < a_len; i++)
    if (a[i] > 0)
        total += a[i] * a[i];
```

## Reductions and math

`len`, `sum`, `min`, `max`, `abs`, `any`, `all` and `math.*` become inline code or C math calls.

numpy's functions and reductions, such as `np.sqrt(x[i])` and `x.sum()`, are in [Working with numpy](numpy.md#functions-on-single-numbers).

## Calls between functions

A compiled function can call another `@scram` function at module level, or imported from another module:

```python
@scram
def step(state, push):
    state[1] += push * 0.02
    state[0] += state[1] * 0.02
    return abs(state[0]) < 2.4

@scram
def run(state, n):
    for i in range(n):
        if not step(state, 1.0 if i % 2 else -1.0):
            return i
    return n
```
<!-- c: run(np.zeros(2), 10) -->
```c-generated
static int sj_call_0(double *v_state, int64_t v_state__n, double v_push, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 1LL;
    int64_t t2 = t1 < 0 ? t1 + v_state__n : t1;
    if ((uint64_t)t2 >= (uint64_t)v_state__n) { sj_io[4] = 0; SJ_FAIL(1, 3, t1, v_state__n); }
    double t3 = v_state[t2];
    double t4 = t3;
    double t5 = v_push;
    double t6 = 0x1.47ae147ae147bp-6;
    double t7 = (t5 * t6);
    v_state[t2] = (t4 + t7);
    int64_t t8 = 0LL;
    int64_t t9 = t8 < 0 ? t8 + v_state__n : t8;
    if ((uint64_t)t9 >= (uint64_t)v_state__n) { sj_io[4] = 0; SJ_FAIL(1, 4, t8, v_state__n); }
    double t10 = v_state[t9];
    double t11 = t10;
    int64_t t12 = 1LL;
    int64_t t13 = t12 < 0 ? t12 + v_state__n : t12;
    if ((uint64_t)t13 >= (uint64_t)v_state__n) { sj_io[4] = 0; SJ_FAIL(1, 4, t12, v_state__n); }
    double t14 = v_state[t13];
    double t15 = t14;
    double t16 = 0x1.47ae147ae147bp-6;
    double t17 = (t15 * t16);
    v_state[t9] = (t11 + t17);
    int64_t t18 = 0LL;
    int64_t t19 = t18 < 0 ? t18 + v_state__n : t18;
    if ((uint64_t)t19 >= (uint64_t)v_state__n) { sj_io[4] = 0; SJ_FAIL(1, 5, t18, v_state__n); }
    double t20 = v_state[t19];
    double t21 = fabs(t20);
    double t22 = 0x1.3333333333333p+1;
    sj_io[0] = (int64_t)((t21 < t22));
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

int sj_kernel(double *v_state, int64_t v_state__n, int64_t v_n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t *sj_tb = sj_io + 5; sj_tb[0] = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v_i = t3;
            double t6; int t7;
            int64_t t8 = v_i;
            int64_t t9 = 2LL;
            int64_t t10 = ((int64_t)t8 & 1LL);
            if ((t10) != 0) {
                t6 = 0x1.0000000000000p+0; t7 = 0;
            } else {
                double t11 = (-(0x1.0000000000000p+0));
                t6 = t11; t7 = 0;
            }
            double t12 = t6;
            int t13 = t7;
            double t14 = t12;
            int t15 = t13;
            int64_t t16[5];
            int t17 = sj_call_0(v_state, v_state__n, t14, sj_tb, t16);
            if (t17) { sj_tb[++sj_tb[0]] = ((int64_t)0 << 32) | t16[1]; sj_io[0] = t16[0]; sj_io[4] = t16[4]; SJ_FAIL(t17, 4, t16[2], t16[3]); }
            int t18 = (int)t16[0];
            int t19 = t18;
            int t20 = !(t19);
            if (t20) {
                sj_io[0] = (int64_t)(v_i);
                goto sj_done;
            }
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    sj_io[0] = (int64_t)(v_n);
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Each function compiles once per set of argument types it's called with, so `sq(2)` and `sq(2.0)` are two C functions. A small one is inlined and costs nothing: a loop calling `sq(x[i])` took 0.93 ns per element, the same as writing `x[i] * x[i]`. `step` above, called 1,000,000 times from `run`, costs a few ns each time, against 3 µs from a Python loop.

Arguments can be numbers, arrays, views and tuples of numbers, and a call can return any of these or a tuple holding arrays. An array passed in is the same memory, and one passed back is the same object: `y = f(x)` with `return x` gives you `x`. Everything follows Python: the order things are read in, exceptions, with a traceback showing every function, and any global a called function reads, in its own module.

A call runs as Python, with the same result, and says why, when:

- the function isn't decorated with `@scram`, or is in a closure or a method
- it calls itself, directly or through another (recursion)
- it doesn't compile for those argument types: the warning gives its reason
- it leaves out a parameter whose default isn't a number, or is passed a dict or a list, or a function

Inside a call there's no profile of its own: time in `step` counts toward `run` in `preflight --run`, and `counts=True` counts the lines of the outer function only.

## Calling C

`<<` and `>>` compile, so a 32-bit or 64-bit hash written in Python, with each step masked, is exact ([gotcha](gotchas.md#integers-are-64-bit-and-wrap)). For what's awkward in Python, such as SIMD intrinsics or an existing C library, write that part in C with [cimo](https://www.cimo.dev/) and call it:

```python
import cimo

hashing = cimo.compile(r"""
#include <stdint.h>
int64_t mix(int64_t value) {
    uint64_t h = (uint64_t)value;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    return (int64_t)(h >> 1);
}
""")

@scram
def bucket(values, counts):
    for i in range(len(values)):
        counts[hashing.mix(int(values[i])) % len(counts)] += 1
```
<!-- c: bucket(np.arange(8), np.zeros(4, np.int64)) -->
```c-generated
int sj_kernel(int64_t *v_values, int64_t v_values__n, int64_t *v_counts, int64_t v_counts__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = v_values__n;
    int64_t t2 = 0LL, t3 = t1;
    int64_t t4 = t2, t5 = t3;
    for (;;) {
        int64_t t6 = t5 > t4 && (uint64_t)t5 - (uint64_t)t4 > 65536 ? t4 + 65536 : t5;
        for (; t4 < t6; t4++) {
            v_i = t4;
            int64_t t7 = v_i;
            int64_t t8 = t7;
            int64_t t9 = v_values[t8];
            int64_t t10 = t9;
            int64_t t11 = t10;
            int64_t t12 = (int64_t)(t11);
            int64_t t13 = mix(t12);
            int64_t t14 = t13;
            int64_t t15 = t14;
            int64_t t16 = v_counts__n;
            int64_t t17 = t16;
            if (t17 == 0) SJ_FAIL(2, 4, 3, 0);
            int64_t t18 = 0, t19 = 0;
            if ((int64_t)t17 == -1) t18 = sj_neg((int64_t)t15); else if ((int64_t)t17 != 0) { t18 = (int64_t)t15 / (int64_t)t17; t19 = (int64_t)t15 % (int64_t)t17; }
            int t20 = t19 != 0 && ((t19 < 0) != ((int64_t)t17 < 0));
            int64_t t21 = (t19 + (t20 ? (int64_t)t17 : 0));
            int64_t t22 = t21;
            int64_t t23 = t22;
            int64_t t24 = v_counts[t23];
            int64_t t25 = t24;
            int64_t t26 = 1LL;
            v_counts[t23] = sj_add((int64_t)t25, (int64_t)t26);
        }
        if (t4 >= t5) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

The module's source goes into the same C file as the function calling it, so a small C function is inlined: `k.sq(x[i])` took 0.92 ns per element in a compiled loop, against 686 from a Python loop calling it. `f = k.f` at module level works as well. Arguments and results follow what the same call does from Python, which is what runs if any of this doesn't compile:

| C parameter | Compiles with | Otherwise |
| --- | --- | --- |
| `double`, `float` | any number | |
| `int32_t` and narrower | a Python int or bool, a numpy integer, cut to size | a float raises `ArgumentError` from Python |
| `int64_t`, `uint64_t` | a Python int or bool | `int(x[i])` for a numpy integer: cimo takes one as an address |
| `_Bool` | a bool | |
| `double *`, `const uint8_t *`, a VLA, `void *` | an array or view without gaps, of that element type | another type, a gap, a read-only array as non-`const` |

A result that is an integer other than `uint64_t` comes back as a Python int, and `double` or `float` as a Python float. A pointer, a struct, or a variadic function stays in Python.

Some modules can't compile this way, and stay in Python, with a warning saying why:

- **State:** a module with data that isn't `const`, such as a counter in a global or a static local. A second copy of its source wouldn't share that with the module you can call from Python.
- **Options:** `link=`, `host_headers=` or `include_dirs=`. `defines=` is fine, and so is `fma=`, as long as all the modules a function calls agree.
- **Math that must stay exact,** such as `math.gamma`, `np.std` or `math.remainder`, in a function calling a module compiled with the default `fma=`: the file has one setting. Pass `fma=False` to `cimo.compile`.
- **Two modules defining the same name,** in one function.

An `assert` or `abort()` in the C raises `cimo.KernelError`, naming `sj_kernel` rather than your function. A macro your C defines is also visible to the C scramjit generates, after it, so don't define one named like a C library function.

The C runs with the privileges of the process and isn't checked: a wrong length is a bad write, as anywhere in C.
