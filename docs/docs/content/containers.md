# Arrays, lists and tuples

> [!PLANNED]
> 1-D numpy arrays arrive in M1, lists and tuples in M2, n-D arrays in M3.

## numpy arrays

Arrays are passed by pointer, shape and strides, never copied. Writes go to your array. The generated code is specialized to the dtype, the number of dimensions and the layout, so a C-contiguous array gets plain `a[i]` indexing.

Negative indices count from the end, as in Python. Whether bounds are checked by default is **open**; see [Semantics](semantics.md).

## Tuples

Tuples become C structs. Returning several values works as in Python:

```python
@scramjit.jit
def minmax(a):
    lo = hi = a[0]
    for v in a:
        lo = min(lo, v)
        hi = max(hi, v)
    return lo, hi
```

## Lists that never change size

A list that is never resized becomes a fixed-size C array. That means no `append`, `pop`, `extend`, `insert`, `del` or slice assignment on it, and no escape into code that might do those. It lives on the stack when it is small, constant-sized and not returned, and on the heap otherwise.

```python
@scramjit.jit
def blur3(src, dst):
    w = [0.25, 0.5, 0.25]          # a 3-element C array on the stack
    for i in range(1, len(src) - 1):
        dst[i] = w[0] * src[i - 1] + w[1] * src[i] + w[2] * src[i + 1]
```

## Lists passed in from Python

A homogeneous numeric list argument is copied into an array at the call and copied back if the function mutated it. That is O(n) per call, so pass large data as numpy arrays.

## Not yet

Growable lists, dicts, sets and strings, as typed C containers, come later. Each needs allocation and a runtime, and the numeric subset comes first.
