# Types and annotations

> [!PLANNED]
> Designed, not built. M1 handles `int64` and `float64` only; narrowing arrives in M2.

## Inferred from the call

scramjit infers a type for every variable over the whole function, seeded by the types of the arguments at call time. A new combination of argument types (a `float32` array instead of `float64`, say) compiles a new specialization; each one is cached by its signature.

A variable that takes several types unifies by Python's rules: `int` and `float` give `float`. Types that don't unify, such as a variable that is sometimes a number and sometimes an array, are a compile error that points at the line.

## Annotations are contracts

Annotations are used, not ignored. They are checked at the call boundary, and they give the compiler information it can narrow with.

```python
@scramjit.jit
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

scramjit's own spellings for sized scalars and arrays (for example `i32` or `f32[:, :]`) are **open**.

## The narrowest type that is proven sufficient

The compiler picks the smallest type that holds every value a variable can take, from annotations, array dtypes and value-range analysis:

- a loop index bounded by an array length,
- `x % 256`,
- an element read from a `uint8` array.

Without a proof, integers are `int64` and floats `float64`. The payoff is mainly in arrays and SIMD (smaller elements, more lanes per vector); narrowing scalars rarely helps on 64-bit CPUs.

## Never on Python objects

Generated code never touches a Python object. Arguments are unboxed at the call: Python numbers into C scalars, numpy arrays into a pointer, shape and strides (without copying). The body runs on C values only, and the result is boxed on return.

Anything that can't be unboxed is a compile error that says why. There is no fallback to running on Python objects: it would be slow, and it can't work without libpython, which cimo doesn't link and Pyodide has none of to link.
