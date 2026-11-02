# Arrays, views, lists, tuples and dicts

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
| Making arrays | | `x * 2`, `a + b`, `np.zeros(n)`, `x.copy()`, `np.sqrt(x)` |
| Element types | `float64`, `int64`, and reading and storing smaller ones | arithmetic on `float32`, `int32`, `uint8` and friends |
| Other | `len()`, `.shape`, `.size`, `.ndim`, returning a view alone | `np.dot`, `np.linalg`, `np.sort`, `np.cumsum`, a view inside a returned tuple |

Anything in the right column runs as Python with numpy's own result, and the loops around it still compile if they fit. The biggest gap is making arrays: a function that starts with `y = np.zeros(n)` has to take `y` as an argument for now.

## Tuples

Tuples become C structs. Returning several values works as in Python:

```python
@scram
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
@scram
def blur3(src, dst):
    w = [0.25, 0.5, 0.25]          # a 3-element C array on the stack
    for i in range(1, len(src) - 1):
        dst[i] = w[0] * src[i - 1] + w[1] * src[i] + w[2] * src[i + 1]
```

## Lists passed in from Python

A homogeneous numeric list argument is copied into an array at the call and copied back if the function mutated it. That is O(n) per call, so pass large data as numpy arrays.

## Dicts with fixed keys

A dict passed in compiles when the function only reads and writes keys it names, so settings and results can stay in dicts. Each key becomes a plain variable, so a loop over `p['k']` generates the same code as one over a local:

```python
@scram
def advance(p, x, y):
    for i in range(p['n']):
        y[i] = x[i] * p['gain'] + p['offset']
    p['steps'] += 1
```

Only the keys the function names matter. Anything else in the dict can be anything, such as strings or other dicts, and isn't looked at, so a call costs about the same as passing those values one by one, however big the dict is.

A value can be a number, an array or a tuple of numbers. What you write goes back to your dict, as it would in Python: stores made before an exception are there afterwards, and key order doesn't change.

Each of these runs as Python, for that call, with the same results:

- a key the dict doesn't have, which raises `KeyError` as usual
- a value that can't be passed, such as a string you read
- one dict passed as two arguments, since a write must show through both names
- a dict subclass, such as `OrderedDict` or `defaultdict`

And each of these keeps the function, or the loop it's in, in Python: adding or deleting a key, a key in a variable (`p[k]`), `len(p)`, `'a' in p`, `p.get()`, iterating over `p`, returning `p`, and building a dict inside the function. Those loops can still be [lifted out](errors.md#loops-are-lifted-out) if they don't touch the dict that way.

## What stays in Python

Growable lists, dicts that gain keys, sets and strings. Each would need allocation and a runtime inside compiled code, so code that uses them runs as Python, and the loops around it that don't are still compiled. See [Fallback and errors](errors.md).
