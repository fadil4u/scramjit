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
- the constants `np.pi`, `np.e`, `np.inf`, `np.nan`

Arguments can be Python ints and floats, `np.int64` and `np.float64`. A few combinations run as Python, because what numpy returns for them depends on the platform or the numpy version:

- `np.abs`, `np.sign`, `np.square`, `np.minimum` and `np.maximum` of Python ints: `np.abs(4)` is `int64` on most platforms and `int32` in Pyodide
- `np.floor`, `np.ceil` and `np.trunc` of any int: numpy 2.1 changed what these return

Bools and narrow types such as `np.float32` or `np.uint8` run as Python too. So do `np.tanh`, which is numpy's own and doesn't match the C library's in the last bits, and `np.power`, which has shortcuts for some exponents; use `**`. Calls on whole arrays, such as `np.sum(x)` or `np.sqrt(x)`, run as Python for now.
