# numpy compatibility

## numpy arrays

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

### What numpy things compile

| | Compiles | Runs as Python |
| --- | --- | --- |
| Indexing | ints, slices of any step, `.T`, `m[i][j]` | `None`, `...`, masks `x[x > 0]`, index arrays `x[idx]` |
| Assigning | an element; a number or an array into a slice | `x[1:] += y`, a list into a slice |
| Functions on one number | [these](loops-and-builtins.md#numpy-functions-on-single-numbers) | `np.tanh`, `np.power` |
| Reductions to one number | [`sum`, `mean`, `std`, `var`, `max`, `argmax` and the rest](loops-and-builtins.md#numpy-reductions) | `axis=` on more than one dimension, `keepdims=True` |
| Making arrays | [`np.zeros`, `x.copy()`, `np.arange` and the rest](#making-arrays) | `x * 2`, `a + b`, `np.sqrt(x)`, `x.reshape(...)`, `np.eye(n)` |
| Element types | `float64`, `int64`, and [`float32`, `int32`, `uint8` and friends](types.md#narrow-dtypes), with arithmetic and most reductions, and `uint64` | on `float32` `np.sin`, `np.exp` and the like, `.std()`, `.var()` |
| Other | `len()`, `.shape`, `.size`, `.ndim`, returning arrays and views, alone or in a tuple | `np.dot`, `np.linalg`, `np.sort`, `np.cumsum` |

Anything in the right column runs as Python with numpy's own result, and the loops around it still compile if they fit.

### Making arrays

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
## Python data structures

Tuples, lists, lists of lists, dicts, enums and module-level constants have a page of their own: [Python data structures](data-structures.md).
