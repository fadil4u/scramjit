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

## numpy functions on single numbers

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

## numpy reductions

`sum`, `prod`, `mean`, `std`, `var`, `max`, `min`, `any`, `all`, `argmax` and `argmin` compile as methods and as functions (`x.sum()`, `np.sum(x)`, and `np.amax` and `np.amin`), on arrays and on views. They give numpy's exact result, down to rounding, since `np.sum` adds in a particular order, and numpy's type:

```python
@scram
def spread(m):
    total = 0.0
    for i in range(len(m)):
        total += m[i].max() - m[i].min()
    return total / m.size
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

Each function compiles once per set of argument types it's called with, so `sq(2)` and `sq(2.0)` are two C functions. A small one is inlined and costs nothing: a loop calling `sq(x[i])` took 0.93 ns per element, the same as writing `x[i] * x[i]`. `step` above, called 1,000,000 times from `run`, costs a few ns each time, against 3 µs from a Python loop.

Arguments can be numbers, arrays, views and tuples of numbers, and a call can return any of these or a tuple holding arrays. An array passed in is the same memory, and one passed back is the same object: `y = f(x)` with `return x` gives you `x`. Everything follows Python: the order things are read in, exceptions, with a traceback showing every function, and any global a called function reads, in its own module.

A call runs as Python, with the same result, and says why, when:

- the function isn't decorated with `@scram`, or is in a closure or a method
- it calls itself, directly or through another (recursion)
- it doesn't compile for those argument types: the warning gives its reason
- it has default values for parameters, or is passed a dict or a list, or a function

Inside a call there's no profile of its own: time in `step` counts toward `run` in `preflight --run`, and `counts=True` counts the lines of the outer function only.

## Calling C

`<<` and `>>` compile, so a 32-bit hash written in Python, with each step masked, is exact ([gotcha](gotchas.md#integers-are-64-bit-and-wrap)). For what's awkward in Python, such as a 64-bit hash that relies on unsigned overflow, write that part in C with [cimo](https://www.cimo.dev/) and call it:

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
