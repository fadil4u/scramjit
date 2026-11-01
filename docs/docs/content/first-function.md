# Your first function

> [!PLANNED]
> This page describes the design. The examples will be checked by running them once [M1](roadmap.md#m1-scalars-1-d-arrays-range) lands.

## Decorate a loop

```python
import numpy as np
import scramjit

@scramjit.jit
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
```

This is a recurrence: each output depends on the one before it, so it can't be written as one vectorized numpy expression. In pure Python it runs one bytecode-interpreted iteration at a time.

## Call it

```python
x = np.random.rand(1_000_000)
y = np.empty_like(x)
ema(x, y, 0.1)
```

On this first call scramjit reads the argument types (`x` and `y` are 1-D `float64` arrays, `a` is a `float`), generates C specialized to them, compiles it with cimo and caches the result. Later calls with the same types skip straight to native code.

## What happened

scramjit turned the function into C close to what you would write by hand:

```c
void ema(const double *x, int64_t x_len, double *y, int64_t y_len, double a) {
    y[0] = x[0];
    for (int64_t i = 1; i < x_len; i++)
        y[i] = a * x[i] + (1 - a) * y[i - 1];
}
```

No Python objects appear in it: `range` became a counted loop, `len(x)` became the array length passed in at the call, and `a` arrived as a C `double`. (The exact C, including bounds checks, is being designed; see [Semantics](semantics.md).)

## The undecorated function

The original Python function stays reachable for debugging and for comparing results. How you get at it is **open** (for example an attribute on the compiled function).

## Options

The decorator is the whole API. Options such as bounds checks and cache control are keyword arguments to it. Whether the spelling is `@scramjit.jit` or a bare `@scramjit` is **open**.
