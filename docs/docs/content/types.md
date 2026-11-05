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
<!-- c: scale(np.ones(8), 2.0) -->
```c-generated
int sj_kernel(double *v_img, int64_t v_img__n, double v_k, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_img__n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v_i = t3;
            int64_t t6 = v_i;
            int64_t t7 = t6;
            double t8 = v_img[t7];
            double t9 = t8;
            double t10 = v_k;
            int64_t t11 = v_i;
            int64_t t12 = t11;
            double t13 = (t9 * t10);
            v_img[t12] = t13;
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

Calling `scale(img, "2")` fails at the boundary with a `TypeError`, as the annotation says it should, before any native code runs.

Accepted annotations:

| Annotation | Meaning |
| --- | --- |
| `int` | a Python `int`, held as `int64` unless narrowed; one that can only be in `[0, 2 ** 64)`, such as a value masked with `& 0xFFFFFFFFFFFFFFFF`, is held as `uint64` |
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

## numpy's dtypes

`float32`, `int32`, `uint8` and the other narrow dtypes compile and follow numpy's rules: see [Narrow dtypes](numpy.md#narrow-dtypes).

## The narrowest type that is proven sufficient

The compiler takes narrow types from annotations and array dtypes, such as an element read from a `uint8` array. It doesn't yet narrow from value ranges, such as a loop index bounded by an array length or `x % 256`.

Without a proof, integers are `int64` and floats `float64`. The payoff is mainly in arrays and SIMD (smaller elements, more lanes per vector); narrowing scalars rarely helps on 64-bit CPUs.

## Never on Python objects

Generated code never touches a Python object. Arguments are unboxed at the call: Python numbers into C scalars, numpy arrays into a pointer, shape and strides (without copying). The body runs on C values only, and the result is boxed on return.

Compiled code never falls back to Python objects inside itself: that would be slow, and it can't work without libpython, which cimo doesn't link and Pyodide has none of to link. Code that needs Python objects stays in Python instead, around the compiled parts; see [Fallback and errors](errors.md).
