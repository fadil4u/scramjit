# Your first function

## Decorate a loop

```python
import numpy as np
from scramjit import scram

@scram
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
```

`scram` is scramjit's one decorator. This function is a recurrence: each output depends on the one before it, so it can't be written as one vectorized numpy expression. In pure Python it runs one bytecode-interpreted iteration at a time.

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

No Python objects appear in it: `range` became a counted loop, `len(x)` became the array length passed in at the call, and `a` arrived as a C `double`. (Bounds checks are left out here for clarity; see [Semantics](semantics.md).)

## The undecorated function

The original Python function stays reachable as `ema.__wrapped__`, the standard library's convention for decorated functions. Call it to debug in plain Python or to compare results:

```python
ref = np.empty_like(x)
ema.__wrapped__(x, ref, 0.1)
assert np.array_equal(y, ref)
```

## Options

The decorator is the whole API. Options are keyword arguments to it:

```python
@scram(boundscheck=False, cache=False)
def ema(x, y, a: float):
    ...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `boundscheck` | `True` | Check every index and raise `IndexError` as Python does. `False` drops the checks in loops you have verified. |
| `cache` | `True` | Keep compiled code on disk so other processes reuse it. Code is always cached in memory. |
