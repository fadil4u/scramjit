# Types and annotations

## Inferred from the call

scramjit infers a type for every variable over the whole function, seeded by the types of the arguments at call time. A new combination of argument types (a `float32` array instead of `float64`, say) compiles a new specialization; each one is cached by its signature.

A variable that takes several types unifies by Python's rules: `int` and `float` give `float`. Types that don't unify, such as a variable that is sometimes a number and sometimes an array, keep that code in Python; see [Fallback and errors](errors.md).

## Annotations are contracts

Annotations are used, not ignored. They are checked at the call boundary, and they give the compiler information it can narrow with.

```python
@scram
def scale(img, k: float):
    for i in range(len(img)):
        img[i] = img[i] * k
```

Calling `scale(img, "2")` fails at the boundary with a `TypeError`, as the annotation says it should, before any native code runs.

Accepted annotations:

| Annotation | Meaning |
| --- | --- |
| `int` | a Python `int`, held as `int64` unless narrowed |
| `float` | a Python `float`, held as `float64` |
| `bool` | a Python `bool` |
| `np.int8`, `np.float32`, … | a numpy scalar type of that size |

## A variable that changes type

One name can hold different types. If the uses never meet, as with `x = 1`, used, then `x = 0.5`, used, each is its own variable. An int that turns into a float also compiles:

```python
s = 0
for v in x:          # x holds floats
    s += v
```

As in Python, `s` is the int `0` if the loop doesn't run, and a float otherwise. That works when the int is a literal or an argument, and `s` is only combined with floats, compared with them, or returned. Any other use, such as `s + 1`, `-s`, `t = s` or `if s:`, could differ between the two, so it runs as Python.

## Narrow dtypes

Arithmetic on `float32`, `int32`, `int16`, `int8`, `uint32`, `uint16` and `uint8` compiles, as array elements and as numpy scalars passed in, and gives what numpy gives:

```python
@scram
def brighten(image, amount):
    for i in range(len(image)):
        image[i] = image[i] + amount    # 250 + 10 wraps to 4
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

The literal must have the same kind as the array and fit its dtype exactly: `0.0` or `0.5`, not `0.1`, on `float32`, and not `0` either. Such a variable can then only meet numpy scalars, or be compared with an int if it holds ints. Anything else keeps the function in Python.

Also in Python, with numpy's result: `**`, `sum()`, and [numpy functions and reductions](loops-and-builtins.md#numpy-functions-on-single-numbers) on these dtypes.

## The narrowest type that is proven sufficient

The compiler takes narrow types from annotations and array dtypes, such as an element read from a `uint8` array. It doesn't yet narrow from value ranges, such as a loop index bounded by an array length or `x % 256`.

Without a proof, integers are `int64` and floats `float64`. The payoff is mainly in arrays and SIMD (smaller elements, more lanes per vector); narrowing scalars rarely helps on 64-bit CPUs.

## Never on Python objects

Generated code never touches a Python object. Arguments are unboxed at the call: Python numbers into C scalars, numpy arrays into a pointer, shape and strides (without copying). The body runs on C values only, and the result is boxed on return.

Compiled code never falls back to Python objects inside itself: that would be slow, and it can't work without libpython, which cimo doesn't link and Pyodide has none of to link. Code that needs Python objects stays in Python instead, around the compiled parts; see [Fallback and errors](errors.md).
