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

Tuples can hold tuples, to any depth, as locals, arguments, results and module-level constants, with one C variable per number:

```python
@scram
def iou(a, b):
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    ...

iou(((0.0, 0.0), (2.0, 2.0)), ((1.0, 1.0), (3.0, 3.0)))
```

A tuple is indexed with constants (`box[0][1]`), unpacked, and looped over when its elements share a type: `for (a, b) in (p, q)`, with `break` and `continue`, and in comprehensions when the element is unpacked. A tuple holding arrays works as a result (`return a, b`), not yet as a local or an argument.

## Lists that never change size

A list that is never resized becomes a fixed-size C array. That means no `append`, `pop`, `extend`, `insert`, `del` or slice assignment on it, and no escape into code that might do those. It lives on the stack when it is small, constant-sized and not returned, and on the heap otherwise.

```python
@scram
def blur3(src, dst):
    w = [0.25, 0.5, 0.25]          # a 3-element C array on the stack
    for i in range(1, len(src) - 1):
        dst[i] = w[0] * src[i - 1] + w[1] * src[i] + w[2] * src[i + 1]
```

## Lists passed in, returned, and at module level (from 0.0.4)

A list argument whose elements are all ints, all floats or all bools is copied into C memory at the call. If the function writes to it, the elements that changed are copied back, also when it raises, so the caller sees what Python would have left; the others stay the very objects they were. Lists at module level work the same way.

```python
@scram
def rolling_max(prices, window: int):
    out = [0.0] * (len(prices) - window + 1)
    for i in range(len(out)):
        m = prices[i]
        for j in range(i + 1, i + window):
            m = prices[j] if prices[j] > m else m
        out[i] = m
    return out          # a new Python list
```

Returning a list built in the function gives a new Python list; returning the list passed in gives that list itself. The copies cost about 10 ms per 200,000 floats, so a large list is still better passed as a numpy array, and a call that only adds numbers up may not pay for them.

Still Python: an empty list (its element type isn't known), mixed ints and floats, numpy scalars as elements, the same list passed twice, storing a float into a list of ints, lists of lists, and `append` and the other methods that resize.

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

## Constants at module level

A function can read numbers, tuples of numbers, arrays and dicts from module level, and loops can use them:

```python
RATE = 0.25
TABLE = (1.0, 2.5)
WEIGHTS = np.hanning(16)

@scram
def apply(x):
    s = 0.0
    for i in range(len(WEIGHTS)):
        s += WEIGHTS[i] * x[i] * RATE
    return s + TABLE[0]
```

They work like arguments, read at each call. A new value needs no recompile, a new type compiles for that type, and writes to a global array or dict reach it. Each one costs about what an argument does, around 0.2 µs per call. One that isn't there raises `NameError` as usual. Any other global, such as a string, a list or `None`, keeps the function or loop in Python, as does assigning to a global with `global`.

## Enums

Members of an `Enum` or `IntEnum` whose values are ints compile, held as their values:

```python
class Cell(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2

@scram
def cell_of(v):
    return Cell.WALL if v > 0.5 else Cell.EMPTY

@scram
def walls(grid):
    n = 0
    for v in grid:
        if cell_of(v) is Cell.WALL:
            n += 1
    return n
```

`Cell.WALL`, comparisons with `==`, `!=`, `is` and `is not`, `.value`, and truth work as in Python: an `Enum` member equals only itself, an alias is the member it names, and an `IntEnum` member is also an int, so it orders, adds and compares with ints. Members are passed in, returned, and passed between compiled functions; what comes back is the member itself. The class is read from module level like the constants above, and a new class compiles for it.

Still Python: storing a member into an array (store `.value`), `.name`, `Cell(1)`, iterating over the class, `Flag` and string enums, and an enum that defines its own `__eq__`, `__hash__`, `__bool__` or ordering.

## What stays in Python

Growable lists, dicts that gain keys, sets and strings. Each would need allocation and a runtime inside compiled code, so code that uses them runs as Python, and the loops around it that don't are still compiled. See [Fallback and errors](errors.md).
