# Python data structures

scramjit compiles Python's own data structures where it can hold them in plain C memory: tuples, lists, lists of lists, dicts with fixed keys, attributes of instances (from 0.0.4), enums, and constants at module level. This page lists exactly what works for each, how it's held, what is copied and when, and what keeps a function in Python. Anything that doesn't compile runs as Python with the same result and a `FallbackWarning` that says why, and the loops around it that do compile still do (see [Fallback and errors](errors.md)).

The rule behind all of it: a structure compiles when it is **statically sized** (nothing resizes it at run time) and **statically typed** (the types of what it holds don't change). numpy arrays have their own page, [Working with numpy](numpy.md).

At a glance:

| | Compiles | Stays Python |
|:---|:---|:---|
| Tuples | numbers and tuples of them, to any depth; as locals, arguments, results and globals; constant indexes, unpacking, loops over them | an index in a variable, slices, `+`, comparing tuples, a tuple holding arrays except as a result |
| Lists | lists of numbers made in the function, passed in, returned, and at module level (from 0.0.4) | `append` and every method that resizes, `in`, slices, aliases (`ys = xs`), mixed ints and floats |
| Lists of lists | rectangular, passed in, made in the function and returned (from 0.0.4) | ragged, one row in two places, `m[i, j]`, slices, replacing a row, three levels made in the function |
| Dicts | passed in, reading and writing the keys the function names | building one, adding or deleting keys, `len`, `in`, `.get`, iterating, a key in a variable |
| Instances (from 0.0.4) | passed in or at module level, made in the function, `@scram` methods; attributes read and written, objects in attributes; methods; passing them on, returning them, `isinstance`; lists of them: plain classes, dataclasses, `__slots__`, namedtuples, `SimpleNamespace` | properties, static and class methods, adding an attribute, a list made in the function |
| Enums | `Enum` and `IntEnum` members with int values | `.name`, `Cell(1)`, iterating the class, `Flag` and string enums |
| Module-level constants | numbers, tuples, arrays, dicts, lists, enum classes, instances (from 0.0.4) | strings, `None`, assigning with `global` |
| Sets, strings | | all of them |

## Tuples

A tuple is held as one C variable per number, so it costs nothing to build or unpack. Returning several values works as in Python:

```python
@scram
def minmax(a):
    lo = hi = a[0]
    for v in a:
        lo = min(lo, v)
        hi = max(hi, v)
    return lo, hi           # a Python tuple
```
<!-- c: minmax(np.ones(4)) -->
```c-generated
int sj_kernel(double *v_a, int64_t v_a__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    double v_lo = 0;
    double v_hi = 0;
    double v_v = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL;
    int64_t t2 = t1 < 0 ? t1 + v_a__n : t1;
    if ((uint64_t)t2 >= (uint64_t)v_a__n) { sj_io[4] = 0; SJ_FAIL(1, 3, t1, v_a__n); }
    double t3 = v_a[t2];
    double t4 = t3;
    v_lo = t4;
    v_hi = t4;
    uint64_t t5 = (uint64_t)v_a__n;
    uint64_t t6 = 0, t7 = t5;
    for (;;) {
        uint64_t t8 = t7 > t6 && (uint64_t)t7 - (uint64_t)t6 > 65536 ? t6 + 65536 : t7;
        for (; t6 < t8; t6++) {
            v_v = v_a[t6];
            double t9 = v_lo;
            int t10 = 1;
            double t11 = v_v;
            if (t11 < t9) { t9 = t11; t10 = 1; }
            v_lo = t9;
            double t12 = v_hi;
            int t13 = 1;
            double t14 = v_v;
            if (t14 > t12) { t12 = t14; t13 = 1; }
            v_hi = t12;
        }
        if (t6 >= t7) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    double t15 = v_lo;
    double t16 = v_hi;
    { double t17 = t15; memcpy(&sj_io[4], &t17, 8); }
    sj_io[5] = 1;
    { double t18 = t16; memcpy(&sj_io[6], &t18, 8); }
    sj_io[7] = 1;
    goto sj_done;
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Tuples can hold tuples, to any depth, as locals, arguments, results and module-level constants:

```python
@scram
def iou(a, b):
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    w = min(ax1, bx1) - max(ax0, bx0)
    ...

iou(((0.0, 0.0), (2.0, 2.0)), ((1.0, 1.0), (3.0, 3.0)))
```

What works:

- **Indexing with a constant:** `box[0][1]`, `t[-1]`. The index must be a literal, since each element can have a type of its own.
- **Unpacking:** `a, b = t`, `(x0, y0), (x1, y1) = box`, and `for i, (a, b) in enumerate(zip(xs, ys))`. The number of names must match, as in Python, or the function stays Python.
- **`len(t)`**, which is a constant.
- **Looping over a tuple** whose elements share a type: `for w in (0.25, 0.5, 0.25)` and `for (a, b) in (p, q)`, with `break` and `continue`. The tuple is built first, as Python builds it, then read by index. In a comprehension, unpack the element: `[a * b for a, b in (p, q)]`.
- **A variable holding a tuple**, reassigned in a loop as long as its shape and types stay the same, such as a running best `(score, index)`.
- **Tuples passed to and returned from other compiled functions.**
- **Returning arrays in a tuple:** `return a, b`, where each is an array.

Still Python: an index in a variable (`t[i]`), slices (`t[1:]`), `t + u` and `t * 2`, comparing tuples (`t == (1, 2)`), `in`, a tuple holding arrays as a local or an argument (`p = (a, b)`), a tuple holding lists, a loop over a tuple whose elements differ in type (`for v in (1, x)` with a float `x`), and a comprehension variable that is a whole tuple.

## Lists

### Lists made in the function

A list of numbers made in the function and never resized becomes a C array on the heap, freed when the function returns or raises:

```python
@scram
def blur3(src, dst):
    w = [0.25, 0.5, 0.25]
    for i in range(1, len(src) - 1):
        dst[i] = w[0] * src[i - 1] + w[1] * src[i] + w[2] * src[i + 1]
```

They can be made as a literal (`[1.0, 2.0]`), repeated (`[0.0] * n`, with `n` known only at run time), or by a comprehension, with conditions and several `for`s (`[i * j for i in range(n) for j in range(n) if i != j]`). They can be indexed, with negative indexes and Python's `IndexError` when out of range, written, iterated over, passed to `len`, `sum`, `min`, `max`, `enumerate`, `zip` and `reversed`, reassigned to a new list, and returned (from 0.0.4).

Their element type is fixed by what they're made with. Storing a numpy scalar into a list of floats is fine: the list then holds both, as Python's would, and a flag per element says which. Storing a float into a list of ints keeps the function in Python, since the list's type would change.

Still Python, each for that function or the loop it's in:

- **Resizing:** `append`, `extend`, `insert`, `pop`, `remove`, `del xs[i]`, `xs += [..]`, slice assignment. A list that grows needs a runtime that compiled code doesn't have.
- **Methods and operators:** `sort`, `index`, `count`, `copy`, `x in xs`, `xs + ys`, `xs * 2` after it's made, slices (`xs[1:]`), `sorted(xs)`.
- **Aliases:** `ys = xs`, or putting a list in a tuple. Two names for one list must see each other's writes, which a C array assigned twice wouldn't give.
- **Mixed element types:** `[1, 2.5]`, or `[1.0, n]` with an int `n`.
- **Lists of tuples, lists of arrays,** and other lists that hold something other than numbers.

### Lists passed in (from 0.0.4)

A list argument whose elements are all ints, all floats or all bools is copied into C memory at the call. If the function stores into it, the elements whose value changed are copied back, at every exit, also when it raises, so the caller sees what Python would have left. The elements that didn't change stay the very objects they were.

```python
@scram
def tally(rolls, counts):
    for r in rolls:
        counts[r - 1] += 1          # counts, a Python list, is updated in place

@scram
def rolling_max(prices, window: int):
    out = [0.0] * (len(prices) - window + 1)
    for i in range(len(out)):
        m = prices[i]
        for j in range(i + 1, i + window):
            m = prices[j] if prices[j] > m else m
        out[i] = m
    return out                       # a new Python list
```
<!-- c: rolling_max([1.0, 3.0, 2.0, 4.0], 2) -->
```c-generated
int sj_kernel(double *v_prices, int64_t v_prices__n, int64_t v_window, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    double *v_out = NULL; int64_t v_out__n = 0;
    int64_t v_i = 0;
    double v_m = 0;
    int64_t v_j = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    double *t1 = NULL;
    double t4 = 0x0.0p+0;
    int64_t t5 = v_prices__n;
    int64_t t6 = v_window;
    int64_t t7 = sj_sub((int64_t)t5, (int64_t)t6);
    int64_t t8 = 1LL;
    int64_t t9 = sj_add((int64_t)t7, (int64_t)t8);
    int64_t t10 = (t9 > 0 ? t9 : 0);
    if ((uint64_t)(t10) > 9223372036854775807ULL / sizeof(double) / 1) SJ_FAIL(6, 3, 0, 0);
    int64_t t3 = 0;
    t1 = malloc(sizeof(double) * (size_t)((t10) * 1 > 0 ? (t10) * 1 : 1));
    if (!t1) SJ_FAIL(6, 3, 0, 0);
    int64_t t11 = 0, t12 = t10;
    for (;;) {
        int64_t t13 = t12 > t11 && (uint64_t)t12 - (uint64_t)t11 > 65536 ? t11 + 65536 : t12;
        for (; t11 < t13; t11++) {
            t1[t3] = t4; t3++;
        }
        if (t11 >= t12) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    free(v_out); v_out = t1; v_out__n = t3; t1 = NULL;
    int64_t t14 = 0LL, t15 = v_out__n;
    int64_t t16 = t14, t17 = t15;
    for (;;) {
        int64_t t18 = t17 > t16 && (uint64_t)t17 - (uint64_t)t16 > 65536 ? t16 + 65536 : t17;
        for (; t16 < t18; t16++) {
            v_i = t16;
            int64_t t19 = v_i;
            int64_t t20 = t19;
            if ((uint64_t)t20 >= (uint64_t)v_prices__n) SJ_FAIL(5, 5, 0, 0);
            double t21 = v_prices[t20];
            v_m = t21;
            int64_t t22 = v_i;
            int64_t t23 = 1LL;
            int64_t t24 = v_i;
            int64_t t25 = v_window;
            int64_t t26 = sj_add((int64_t)t22, (int64_t)t23), t27 = sj_add((int64_t)t24, (int64_t)t25);
            int64_t t28 = t26, t29 = t27;
            for (;;) {
                int64_t t30 = t29 > t28 && (uint64_t)t29 - (uint64_t)t28 > 65536 ? t28 + 65536 : t29;
                for (; t28 < t30; t28++) {
                    v_j = t28;
                    double t31; int t32;
                    int64_t t33 = v_j;
                    int64_t t34 = t33 < 0 ? t33 + v_prices__n : t33;
                    if ((uint64_t)t34 >= (uint64_t)v_prices__n) SJ_FAIL(5, 7, 0, 0);
                    double t35 = v_prices[t34];
                    double t36 = t35;
                    double t37 = v_m;
                    if ((t36 > t37)) {
                        int64_t t38 = v_j;
                        int64_t t39 = t38 < 0 ? t38 + v_prices__n : t38;
                        if ((uint64_t)t39 >= (uint64_t)v_prices__n) SJ_FAIL(5, 7, 0, 0);
                        double t40 = v_prices[t39];
                        t31 = t40; t32 = 0;
                    } else {
                        t31 = v_m; t32 = 0;
                    }
                    v_m = t31;
                }
                if (t28 >= t29) break;
                if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
            }
            int64_t t41 = v_i;
            int64_t t42 = t41;
            double t43 = v_m;
            v_out[t42] = t43;
        }
        if (t16 >= t17) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    sj_io[0] = (int64_t)(uintptr_t)v_out; sj_io[4] = v_out__n; sj_io[5] = 0; v_out = NULL;
    goto sj_done;
sj_done:
    free(v_out); free(t1);
    return 0;
sj_fail:
    free(v_out); free(t1);
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Returning a list made in the function gives a new Python list. Returning the list that was passed in gives that list itself, as in Python. Lists at module level are passed the same way: read at each call, and written back.

Each of these makes that call run as Python, with the same result:

- an empty list, whose element type can't be known
- mixed types (`[1, 2.5]`, `[1, True]`), numpy scalars as elements, or an int past 64 bits
- **the same list passed twice**, or a list and one of its rows: a write must show through both names, and two copies wouldn't do that

And these keep the function in Python: storing a float into a list of ints that was passed in (the caller's list would then hold both), and passing a list to another compiled function.

**Cost:** copying is O(n) per call: about 10 ms for 200,000 floats in and out, and 2 ms to check their types. A loop that only adds numbers up gains about 2x; a rolling maximum over 200,000 prices is 13x Python, copies included. Large data that's already in numpy arrays is passed without any copy, so prefer arrays where you have them.

### Lists of lists (from 0.0.4)

A **rectangular** list of lists of numbers passed in is copied into one block, row after row. A row is a view of that block, so it behaves as Python's row does, a reference: writing `row[j]` after `row = m[i]` changes `m`.

```python
@scram
def shortest_paths(d):
    """Floyd-Warshall, in place."""
    n = len(d)
    for k in range(n):
        dk = d[k]
        for i in range(n):
            di = d[i]
            for j in range(n):
                if di[k] + dk[j] < di[j]:
                    di[j] = di[k] + dk[j]
```

What works: `m[i][j]` to read and write, `row = m[i]`, `for row in m`, `len(m)` and `len(m[i])`, Python's `IndexError` for a row or an element out of range, and returning the list of lists that was passed in (which is that list itself). The rows that changed are copied back, each **in one go**, so in a row that changed, the numbers that didn't become equal new objects. Rows that didn't change keep everything.

A table **made in the function** is one block with nothing to copy:

```python
d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]     # each row its own list, as in Python
t = [[0.0 for _ in range(w)] for _ in range(h)]          # the same
```

Its rows are separate lists, as Python makes them, and a negative size gives no rows, as `range` does. Returning it gives new Python lists. An edit distance over two lists of 1,500 is 25x Python.

Still Python:

- **`[[0] * w] * h`**, whose rows are all one list in Python: writing `m[0][0]` changes every row. scramjit doesn't copy that sharing, so it doesn't compile it.
- **Ragged** lists of lists, or **one row in two places** (`[r, r]`) passed in.
- **`m[i, j]`**, which Python rejects with `TypeError`: write `m[i][j]`.
- **Slices** (`m[i][1:]`), which copy in Python, and **replacing a row** (`m[i] = new_row`), which rebinds it in Python.
- **Returning a row** (`return m[0]`), more than two levels made in the function, and a row value or size that is more than a plain expression (Python evaluates it again for every row).

**Cost:** about 100 ns per number each way. Floyd-Warshall on a 150x150 list of lists is 20x Python, but a single cheap pass over one, doing a few operations per number, gains nothing.

## Dicts

A dict **passed in** compiles when the function only reads and writes keys it names, with constant strings or ints:

```python
@scram
def advance(p, x, y):
    for i in range(p['n']):
        y[i] = x[i] * p['gain'] + p['offset']
    p['steps'] += 1
```
<!-- c: advance({'n': 4, 'gain': 2.0, 'offset': 1.0, 'steps': 0}, np.ones(4), np.zeros(4)) -->
```c-generated
int sj_kernel(double v_p__gain, int64_t v_p__n, double v_p__offset, int64_t v_p__steps, double *v_x, int64_t v_x__n, double *v_y, int64_t v_y__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_p__n;
    if (t2 > t1 && t2 <= v_x__n - (0) && t2 <= v_y__n - (0)) {
        int64_t t3 = t1, t4 = t2;
        for (;;) {
            int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
            for (; t3 < t5; t3++) {
                v_i = t3;
                int64_t t6 = v_i;
                int64_t t7 = t6;
                double t8 = v_x[t7];
                double t9 = t8;
                double t10 = v_p__gain;
                double t11 = (t9 * t10);
                double t12 = v_p__offset;
                int64_t t13 = v_i;
                int64_t t14 = t13;
                double t15 = (t11 + t12);
                v_y[t14] = t15;
            }
            if (t3 >= t4) break;
            if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
        }
    } else {
        int64_t t16 = t1, t17 = t2;
        for (;;) {
            int64_t t18 = t17 > t16 && (uint64_t)t17 - (uint64_t)t16 > 65536 ? t16 + 65536 : t17;
            for (; t16 < t18; t16++) {
                v_i = t16;
                int64_t t19 = v_i;
                int64_t t20 = t19;
                if ((uint64_t)t20 >= (uint64_t)v_x__n) { sj_io[4] = 0; SJ_FAIL(1, 4, t19, v_x__n); }
                double t21 = v_x[t20];
                double t22 = t21;
                double t23 = v_p__gain;
                double t24 = (t22 * t23);
                double t25 = v_p__offset;
                int64_t t26 = v_i;
                int64_t t27 = t26;
                if ((uint64_t)t27 >= (uint64_t)v_y__n) { sj_io[4] = 0; SJ_FAIL(1, 4, t26, v_y__n); }
                double t28 = (t24 + t25);
                v_y[t27] = t28;
            }
            if (t16 >= t17) break;
            if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
        }
    }
    int64_t t29 = v_p__steps;
    int64_t t30 = 1LL;
    v_p__steps = sj_add((int64_t)t29, (int64_t)t30);
sj_done:
    sj_io[5] = (int64_t)v_p__steps;
    sj_io[6] = 0;
    return 0;
sj_fail:
    sj_io[5] = (int64_t)v_p__steps;
    sj_io[6] = 0;
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Each key the function names becomes a plain variable, so a loop over `p['k']` compiles to the same code as one over a local. Only those keys matter: anything else in the dict, such as strings or other dicts, is never looked at, so a call costs about what passing those values one by one would, however big the dict is.

A value can be a number, an array, or a tuple of numbers. Writes go back to your dict, as they would in Python: stores made before an exception are there afterwards, and the order of keys doesn't change.

Each of these makes that call run as Python, with the same result:

- a key the dict doesn't have, which raises `KeyError` as usual
- a value that can't be held, such as a string the function reads, or a list
- the same dict passed as two arguments
- a dict subclass, such as `OrderedDict` or `defaultdict`

And each of these keeps the function, or the loop it's in, in Python: adding or deleting a key, a key in a variable (`p[k]`), `len(p)`, `'a' in p`, `p.get()`, `p.items()` and other methods, iterating over `p`, returning `p`, a dict inside a dict (`p['a']['b']`), and building a dict inside the function. Loops that don't touch the dict that way can still be [lifted out](errors.md#loops-are-lifted-out).

## Instances (from 0.0.4)

An object **passed in** compiles when the function only reads and writes attributes it names, and calls its methods. So does `self`, with `@scram` on a method. Nothing about the class has to change:

```python
import dataclasses

@dataclasses.dataclass
class Spring:
    k: float
    x: float
    v: float
    name: str = 'unused, so it can be anything'
    steps: int = 0

@scram
def advance(s, dt, n):
    for _ in range(n):
        s.v -= s.k * s.x * dt
        s.x += s.v * dt
        s.steps += 1
    return s.x
```
<!-- c: advance(Spring(1.0, 1.0, 0.0), 0.01, 10) -->
```c-generated
int sj_kernel(double v_s__k, int64_t v_s__steps, double v_s__v, double v_s__x, double v_dt, int64_t v_n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v__ = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v__ = t3;
            double t6 = v_s__v;
            double t7 = v_s__k;
            double t8 = v_s__x;
            double t9 = (t7 * t8);
            double t10 = v_dt;
            double t11 = (t9 * t10);
            v_s__v = (t6 - t11);
            double t12 = v_s__x;
            double t13 = v_s__v;
            double t14 = v_dt;
            double t15 = (t13 * t14);
            v_s__x = (t12 + t15);
            int64_t t16 = v_s__steps;
            int64_t t17 = 1LL;
            v_s__steps = sj_add((int64_t)t16, (int64_t)t17);
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    { double t18 = v_s__x; memcpy(&sj_io[0], &t18, 8); }
    goto sj_done;
sj_done:
    sj_io[5] = (int64_t)v_s__steps;
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_s__v, 8);
    sj_io[8] = 0;
    memcpy(&sj_io[9], &v_s__x, 8);
    sj_io[10] = 0;
    return 0;
sj_fail:
    sj_io[5] = (int64_t)v_s__steps;
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_s__v, 8);
    sj_io[8] = 0;
    memcpy(&sj_io[9], &v_s__x, 8);
    sj_io[10] = 0;
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

```python
@dataclasses.dataclass
class Controller:
    gain: float
    total: float = 0.0

    @scram
    def run(self, errors):
        for e in errors:
            self.total += e * self.gain
        return self.total
```

Each attribute the function uses becomes one C variable, read at the call. So `s.x` costs what a local does, all the loop work applies, and the C is what you'd get from locals. The object itself never reaches the C, so the rest of it can hold anything. A call takes about 2 µs, against 1.7 µs with a dict and 1.6 µs with the same numbers passed separately; with an array among the attributes, about 12 µs against 2.8 µs passed separately, since no call plan exists for that yet.

A written attribute goes back to the object when the function ends, however it ends, so what was stored before an exception is there afterwards, as in Python.

### What the class can do

scramjit looks the attributes up as Python would, **at every call**, and compiles only if doing so runs none of the class's own code. Anything else runs as Python, with a `FallbackWarning` naming the attribute, so these all behave as they always did:

| | Reading | Writing |
|:---|:---|:---|
| Plain class, dataclass, `SimpleNamespace` | compiles | compiles |
| `__slots__`, `@dataclass(slots=True)` | compiles | compiles |
| An attribute only on the class (`k = 3.0` in the class body) | compiles, and the instance's comes first if it has one | Python, since assigning adds it to the instance |
| `namedtuple`, `typing.NamedTuple` | compiles | Python, which raises |
| `@dataclass(frozen=True)` | compiles | Python, which raises |
| A class with `__setattr__` | compiles | Python, which calls it |
| `@property`, `@staticmethod`, `@classmethod` and other descriptors | Python | Python |
| A method, read without calling it (`f = p.norm`) | Python | |
| A class with `__getattribute__` or `__getattr__` | Python | Python |
| An attribute that isn't there | Python, which raises `AttributeError` | Python, which adds it |

Changing the class later is fine, since a new `@property` or `__setattr__` is noticed at the next call. So is giving an attribute a value of a new type, which compiles for that type, as with an argument.

Attributes can hold numbers, numpy arrays and scalars, tuples of numbers and enum members. An array is the array itself: stores into `p.x[i]` reach it, and `return p.x` returns it. Assigning a new array to `p.x` runs as Python.

An instance at module level works the same way, read at each call, like [other constants](#constants-at-module-level).

### Methods

A compiled function can call the object's methods, whether they're plain functions on the class or `@scram` ones, and they can call each other:

```python
import dataclasses

@dataclasses.dataclass
class Body:
    x: float
    v: float
    k: float

    def force(self):
        return -self.k * self.x

    def step(self, dt):
        self.v += self.force() * dt
        self.x += self.v * dt

@scram
def simulate(b, n, dt):
    for _ in range(n):
        b.step(dt)
    return b.x
```
<!-- c: simulate(Body(1.0, 0.0, 1.0), 10, 0.01) -->
```c-generated
static int sj_call_0(double v_self__k, double v_self__x, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    double t1 = (-(v_self__k));
    double t2 = v_self__x;
    { double t3 = (t1 * t2); memcpy(&sj_io[0], &t3, 8); }
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

static int sj_call_1(double v_self__k, double v_self__v, double v_self__x, double v_dt, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    double t1 = v_self__v;
    double t2 = v_self__k;
    double t3 = v_self__x;
    int64_t t4[5];
    int t5 = sj_call_0(t2, t3, sj_tb, t4);
    if (t5) { sj_tb[++sj_tb[0]] = ((int64_t)0 << 32) | t4[1]; sj_io[0] = t4[0]; sj_io[4] = t4[4]; SJ_FAIL(t5, 2, t4[2], t4[3]); }
    double t6_d; memcpy(&t6_d, &t4[0], 8); double t6 = (double)t6_d;
    double t7 = t6;
    double t8 = t7;
    double t9 = v_dt;
    double t10 = (t8 * t9);
    double t11 = t10;
    v_self__v = (t1 + t11);
    double t12 = v_self__x;
    double t13 = v_self__v;
    double t14 = v_dt;
    double t15 = (t13 * t14);
    v_self__x = (t12 + t15);
sj_done:
    memcpy(&sj_io[5], &v_self__v, 8);
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_self__x, 8);
    sj_io[8] = 0;
    return 0;
sj_fail:
    memcpy(&sj_io[5], &v_self__v, 8);
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_self__x, 8);
    sj_io[8] = 0;
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

int sj_kernel(double v_b__k, double v_b__v, double v_b__x, int64_t v_n, double v_dt, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t *sj_tb = sj_io + 9; sj_tb[0] = 0;
    int64_t v__ = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v__ = t3;
            double t6 = v_b__k;
            double t7 = v_b__v;
            double t8 = v_b__x;
            double t9 = v_dt;
            int64_t t10[9];
            int t11 = sj_call_1(t6, t7, t8, t9, sj_tb, t10);
            memcpy(&v_b__v, &t10[5], 8);
            memcpy(&v_b__x, &t10[7], 8);
            if (t11) { sj_tb[++sj_tb[0]] = ((int64_t)1 << 32) | t10[1]; sj_io[0] = t10[0]; sj_io[4] = t10[4]; SJ_FAIL(t11, 4, t10[2], t10[3]); }
            v_b__v = v_b__v;
            v_b__x = v_b__x;
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    { double t12 = v_b__x; memcpy(&sj_io[0], &t12, 8); }
    goto sj_done;
sj_done:
    memcpy(&sj_io[5], &v_b__v, 8);
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_b__x, 8);
    sj_io[8] = 0;
    return 0;
sj_fail:
    memcpy(&sj_io[5], &v_b__v, 8);
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_b__x, 8);
    sj_io[8] = 0;
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Each method compiles as a [call between functions](loops-and-builtins.md#calls-between-functions), with the attributes it uses, through the methods it calls too, as `self`. What it assigns comes back into the caller before anything else happens, so a method that raises leaves what it stored before, as in Python. Which method runs is decided as Python would decide it, at each call: a subclass's method, or one put on the class later, is the one that runs. One the instance holds itself (`p.step = ...`) runs as Python.

A call to a method costs about 5 ns more than the same code written in the loop, since cimo doesn't yet keep a call's results in registers once it inlines it. That's 15 ns a step against 8.5 for the loop above, or 128 ns as Python.

A method that assigns attributes must be called in a plain statement, such as `b.step(dt)`, `y = b.step(dt)`, `s += b.f()` or `return b.f()`, and not in an `if` or `while` test, a loop header, a comprehension or a lambda. One that only reads can be called anywhere. A method can read module-level values, which are read at each call as the function's own are. Keyword-only parameters, recursion, and a method that changes an attribute's type run as Python.

### Passing an instance on

An instance can be given to another `@scram` function, or to a method, as an argument: `dist(p, q)`, `a.dist2(b)`. It goes as the attributes that function uses, and what it assigns comes back the same way a method's does.

### Lists of instances

A plain list of instances of one class compiles too:

```python
import dataclasses

@dataclasses.dataclass
class Particle:
    x: float
    v: float = 0.0
    bounces: int = 0

    def drift(self, dt):
        self.x += self.v * dt
        if self.x < 0.0:
            self.v = -self.v
            self.bounces += 1

@scram
def step(particles, dt, n):
    for _ in range(n):
        for p in particles:
            p.v -= 9.8 * dt
            p.drift(dt)
    return sum((p.x for p in particles), 0.0)
```
<!-- c: step([Particle(1.0), Particle(2.0)], 0.01, 10) -->
```c-generated
static int sj_call_0(int64_t v_self__bounces, double v_self__v, double v_self__x, double v_dt, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    double t1 = v_self__x;
    double t2 = v_self__v;
    double t3 = v_dt;
    double t4 = (t2 * t3);
    v_self__x = (t1 + t4);
    double t5 = v_self__x;
    double t6 = 0x0.0p+0;
    if ((t5 < t6)) {
        v_self__v = (-(v_self__v));
        int64_t t7 = v_self__bounces;
        int64_t t8 = 1LL;
        v_self__bounces = sj_add((int64_t)t7, (int64_t)t8);
    }
sj_done:
    sj_io[5] = (int64_t)v_self__bounces;
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_self__v, 8);
    sj_io[8] = 0;
    memcpy(&sj_io[9], &v_self__x, 8);
    sj_io[10] = 0;
    return 0;
sj_fail:
    sj_io[5] = (int64_t)v_self__bounces;
    sj_io[6] = 0;
    memcpy(&sj_io[7], &v_self__v, 8);
    sj_io[8] = 0;
    memcpy(&sj_io[9], &v_self__x, 8);
    sj_io[10] = 0;
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

int sj_kernel(int64_t *v_particles__bounces, int64_t v_particles__bounces__n, double *v_particles__v, int64_t v_particles__v__n, double *v_particles__x, int64_t v_particles__x__n, double v_dt, int64_t v_n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t *sj_tb = sj_io + 5; sj_tb[0] = 0;
    int64_t v__ = 0;
    int64_t v_sji__p = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v__ = t3;
            int64_t t6 = v_particles__bounces__n;
            int64_t t7 = 0LL, t8 = t6;
            int64_t t9 = t7, t10 = t8;
            for (;;) {
                int64_t t11 = t10 > t9 && (uint64_t)t10 - (uint64_t)t9 > 65536 ? t9 + 65536 : t10;
                for (; t9 < t11; t9++) {
                    v_sji__p = t9;
                    int64_t t12 = v_sji__p;
                    int64_t t13 = t12;
                    if ((uint64_t)t13 >= (uint64_t)v_particles__v__n) SJ_FAIL(5, 5, 0, 0);
                    double t14 = v_particles__v[t13];
                    double t15 = t14;
                    double t16 = 0x1.399999999999ap+3;
                    double t17 = v_dt;
                    double t18 = (t16 * t17);
                    v_particles__v[t13] = (t15 - t18);
                    int64_t t19 = v_sji__p;
                    int64_t t20 = t19;
                    int64_t t21 = v_particles__bounces[t20];
                    int64_t t22 = t21;
                    int64_t t23 = t22;
                    int64_t t24 = v_sji__p;
                    int64_t t25 = t24;
                    if ((uint64_t)t25 >= (uint64_t)v_particles__v__n) SJ_FAIL(5, 6, 0, 0);
                    double t26 = v_particles__v[t25];
                    double t27 = t26;
                    double t28 = t27;
                    int64_t t29 = v_sji__p;
                    int64_t t30 = t29;
                    if ((uint64_t)t30 >= (uint64_t)v_particles__x__n) SJ_FAIL(5, 6, 0, 0);
                    double t31 = v_particles__x[t30];
                    double t32 = t31;
                    double t33 = t32;
                    double t34 = v_dt;
                    int64_t t35[11];
                    int t36 = sj_call_0(t23, t28, t33, t34, sj_tb, t35);
                    int64_t t37 = (int64_t)t35[5];
                    int64_t t38 = v_sji__p;
                    int64_t t39 = t38;
                    int64_t t40 = t37;
                    v_particles__bounces[t39] = t40;
                    double t41_d; memcpy(&t41_d, &t35[7], 8); double t41 = (double)t41_d;
                    int64_t t42 = v_sji__p;
                    int64_t t43 = t42;
                    if ((uint64_t)t43 >= (uint64_t)v_particles__v__n) SJ_FAIL(5, 6, 1, 0);
                    double t44 = t41;
                    v_particles__v[t43] = t44;
                    double t45_d; memcpy(&t45_d, &t35[9], 8); double t45 = (double)t45_d;
                    int64_t t46 = v_sji__p;
                    int64_t t47 = t46;
                    if ((uint64_t)t47 >= (uint64_t)v_particles__x__n) SJ_FAIL(5, 6, 1, 0);
                    double t48 = t45;
                    v_particles__x[t47] = t48;
                    if (t36) { sj_tb[++sj_tb[0]] = ((int64_t)0 << 32) | t35[1]; sj_io[0] = t35[0]; sj_io[4] = t35[4]; SJ_FAIL(t36, 6, t35[2], t35[3]); }
                    int64_t t49 = v_sji__p;
                    int64_t t50 = t49;
                    int64_t t51 = v_particles__bounces[t50];
                    int64_t t52 = v_sji__p;
                    int64_t t53 = t52;
                    int64_t t54 = t51;
                    v_particles__bounces[t53] = t54;
                    int64_t t55 = v_sji__p;
                    int64_t t56 = t55;
                    if ((uint64_t)t56 >= (uint64_t)v_particles__v__n) SJ_FAIL(5, 6, 0, 0);
                    double t57 = v_particles__v[t56];
                    int64_t t58 = v_sji__p;
                    int64_t t59 = t58;
                    if ((uint64_t)t59 >= (uint64_t)v_particles__v__n) SJ_FAIL(5, 6, 1, 0);
                    double t60 = t57;
                    v_particles__v[t59] = t60;
                    int64_t t61 = v_sji__p;
                    int64_t t62 = t61;
                    if ((uint64_t)t62 >= (uint64_t)v_particles__x__n) SJ_FAIL(5, 6, 0, 0);
                    double t63 = v_particles__x[t62];
                    int64_t t64 = v_sji__p;
                    int64_t t65 = t64;
                    if ((uint64_t)t65 >= (uint64_t)v_particles__x__n) SJ_FAIL(5, 6, 1, 0);
                    double t66 = t63;
                    v_particles__x[t65] = t66;
                }
                if (t9 >= t10) break;
                if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
            }
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    double t67 = 0.0, t68 = 0.0; int t69 = 0;
    int64_t t70 = 0LL, t71 = v_particles__bounces__n, t72 = 1LL;
    uint64_t t73 = t72 > 0 ? (t70 < t71 ? ((uint64_t)t71 - (uint64_t)t70 - 1) / (uint64_t)t72 + 1 : 0) : (t70 > t71 ? ((uint64_t)t70 - (uint64_t)t71 - 1) / (0 - (uint64_t)t72) + 1 : 0);
    t67 = (double)(0x0.0p+0);
    uint64_t t74 = 0, t75 = t73;
    for (;;) {
        uint64_t t76 = t75 > t74 && (uint64_t)t75 - (uint64_t)t74 > 65536 ? t74 + 65536 : t75;
        for (; t74 < t76; t74++) {
            int64_t t77 = (int64_t)((uint64_t)t70 + t74 * (uint64_t)t72);
            int64_t t78 = t77;
            int64_t t79 = t78 < 0 ? t78 + v_particles__x__n : t78;
            if ((uint64_t)t79 >= (uint64_t)v_particles__x__n) SJ_FAIL(5, 7, 0, 0);
            double t80 = v_particles__x[t79];
            double t81 = t80;
            if (!t69) { double tt = t67 + t81; if (fabs(t67) >= fabs(t81)) t68 += (t67 - tt) + t81; else t68 += (t81 - tt) + t67; t67 = tt; } else t67 = t67 + t81;
        }
        if (t74 >= t75) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    if (!t69 && t68 != 0.0 && isfinite(t68)) t67 += t68;
    { double t82 = t67; memcpy(&sj_io[0], &t82, 8); }
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

`for p in ps`, `for i, p in enumerate(ps)`, `p = ps[i]`, `ps[i].x`, `len(ps)`, `sum(p.x for p in ps)`, methods of an element, and passing an element on all compile. Each attribute the function uses is copied into a list of numbers at the call, so `p.x` costs what `xs[i]` does, and the values that changed are written back into the instances when it ends, however it ends. The copy takes about 0.15 µs per instance and attribute at each call. [`examples/particles.py`](https://github.com/farizrahman4u/scramjit/blob/main/examples/particles.py) is a full one: gravity between 200 particles, with methods, a helper function and a module-level constant.

Every element must be the same class and hold the same type in each attribute used (all floats, say, not some ints), and an instance can be in the list only once. Anything else runs the call as Python. Two elements of one list may be the same instance, so a call given two of them, as `pull(a, b)` in a loop over every pair, compiles only if what it assigns through one it doesn't use through the other. `pull` assigning `a.vx` while reading `b.x` is fine; assigning `a.x` while reading `b.x` runs as Python.

Using the list any other way (appending, `ps[i] = q`, slicing, passing it to a function, returning an element) runs as Python.

### Parameters the function doesn't use

A parameter the function never names can be anything: a label, a logger, an object it doesn't touch. It compiles, and nothing of it reaches the C.

### Making instances

`v = Vec(x, y)` in a compiled function compiles when making one runs none of the class's code but storing: a dataclass's own `__init__`, an `__init__` that only assigns `self.a = ...` from its parameters, or none at all. Defaults, keyword arguments, `kw_only`, `frozen`, `slots` and a `__post_init__` method work. The instance's attributes become variables, so making one in a loop costs nothing, and methods, passing it on and `isinstance` work as for one passed in:

```python
import dataclasses
import math

@dataclasses.dataclass
class Vec:
    x: float
    y: float = 0.0

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

@scram
def longest(n):
    best = 0.0
    for i in range(n):
        v = Vec(math.cos(i), math.sin(i) * 2.0)
        best = max(best, v.norm())
    return best
```
<!-- c: longest(10) -->
```c-generated
static int sj_call_0(double v_self__x, double v_self__y, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    double t1 = v_self__x;
    double t2 = v_self__x;
    double t3 = (t1 * t2);
    double t4 = v_self__y;
    double t5 = v_self__y;
    double t6 = (t4 * t5);
    double t7 = (t3 + t6);
    double t8 = t7;
    double t9 = sqrt(t8);
    if (t9 != t9 && t8 == t8) SJ_FAIL(3, 2, 4, 0);
    if (isinf(t9) && isfinite(t8)) SJ_FAIL(3, 2, 4, 0);
    { double t10 = t9; memcpy(&sj_io[0], &t10, 8); }
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

int sj_kernel(int64_t v_n, const uint64_t *sj_m, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t *sj_tb = sj_io + 5; sj_tb[0] = 0;
    double v_best = 0;
    int64_t v_i = 0;
    double v_sjc__v__x = 0;
    double v_sjc__v__y = 0;
    double v_v__x = 0;
    double v_v__y = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    v_best = 0x0.0p+0;
    int64_t t1 = 0LL, t2 = v_n;
    int64_t t3 = t1, t4 = t2;
    for (;;) {
        int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
        for (; t3 < t5; t3++) {
            v_i = t3;
            int64_t t6 = v_i;
            double t7 = (double)(t6);
            double t8 = SJ_LIBM1(1)(t7);
            if (t8 != t8 && t7 == t7) SJ_FAIL(3, 5, 4, 0);
            if (isinf(t8) && isfinite(t7)) SJ_FAIL(3, 5, 4, 0);
            v_sjc__v__x = t8;
            int64_t t9 = v_i;
            double t10 = (double)(t9);
            double t11 = SJ_LIBM1(0)(t10);
            if (t11 != t11 && t10 == t10) SJ_FAIL(3, 5, 4, 0);
            if (isinf(t11) && isfinite(t10)) SJ_FAIL(3, 5, 4, 0);
            double t12 = t11;
            double t13 = 0x1.0000000000000p+1;
            v_sjc__v__y = (t12 * t13);
            v_v__x = v_sjc__v__x;
            v_v__y = v_sjc__v__y;
            double t14 = v_best;
            int t15 = 0;
            double t16 = v_v__x;
            double t17 = v_v__y;
            int64_t t18[5];
            int t19 = sj_call_0(t16, t17, sj_tb, t18);
            if (t19) { sj_tb[++sj_tb[0]] = ((int64_t)0 << 32) | t18[1]; sj_io[0] = t18[0]; sj_io[4] = t18[4]; SJ_FAIL(t19, 6, t18[2], t18[3]); }
            double t20_d; memcpy(&t20_d, &t18[0], 8); double t20 = (double)t20_d;
            double t21 = t20;
            double t22 = t21;
            if (t22 > t14) { t14 = t22; t15 = 0; }
            double t23 = t14;
            int t24 = t15;
            v_best = t23;
        }
        if (t3 >= t4) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    { double t25 = v_best; memcpy(&sj_io[0], &t25, 8); }
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

Returning one gives a real instance of the class, its attributes in the order `__init__` assigns them, as Python would have made it. A `default_factory`, an `__init__` that does anything else (prints, calls, reads `self`), `__new__`, or a metaclass with its own `__call__` keep the function in Python. The class is part of the signature: give the name a new class and it compiles again, and a method changed on it later is noticed at the next call.

### Objects in attributes

`b.pos.x`, `self.pos.norm()` and `d.inner.pos.x += 1.0` compile: each number on the way is a variable, read at the call and written back if assigned. `return b.pos` returns that very object. Assigning an object (`a.pos = b.pos`) runs as Python.

### One object under several names

An object can be reached two ways: `f(p, p)`, `b.pos is b.vel`, `c.pos is a.pos` for two arguments, a node whose `next` is itself, or `p.absorb(p)` and `chase(p, p)` inside a compiled function. Which names share an object is part of the signature, like a type: each object is one set of variables however many names reach it, so a write through one name is seen through the others, as in Python, and nothing in the C can alias. Calling `f(p, p)` and then `f(p, q)` compiles twice, and each call checks which pattern it has. Two elements of one list (`ps[i]` and `ps[j]`) are chosen by data, so a call given both compiles only if what it assigns through one it doesn't use through the other.

### Other ways to use an instance

- `q = p` at the top of the function makes `q` another name for `p`; `best = p` in a loop over a list of instances, then `best.x` after it, works too.
- `return p`, `return ps[i]`, `return best` give back the very object.
- `isinstance(p, Shape)`, also with a tuple of classes, is decided when the function compiles, from the class in the signature. So is `isinstance(x, float)` for a number.
- A method can be called in an `if` or `while` test, or a loop header, even if it assigns attributes: `while body.step(dt):`.
- Methods and functions can take keyword-only parameters: `p.step(dt, damping=0.5)`.

Stays Python, for now:

- `hasattr(p, 'x')`, `getattr(p, name)`, `vars(p)`, and using the object any other way
- a method that assigns attributes, called inside a comprehension or a lambda
- an object in an attribute of a list's elements, or of an instance made in the function, and passing one on (`f(b.pos)`)
- a function that makes an instance, called from another compiled function
- an object and its own `__dict__` passed as two arguments
- `self.__x` in a class, as Python renames it

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
<!-- c: walls(np.ones(4)) -->
```c-generated
static int sj_call_0(double v_v, int64_t *sj_tb, int64_t *sj_io) {
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1; int t2;
    double t3 = v_v;
    double t4 = 0x1.0000000000000p-1;
    if ((t3 > t4)) {
        t1 = 1LL; t2 = 0;
    } else {
        t1 = 0LL; t2 = 0;
    }
    sj_io[0] = (int64_t)(t1);
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}

int sj_kernel(double *v_grid, int64_t v_grid__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t *sj_tb = sj_io + 5; sj_tb[0] = 0;
    int64_t v_n = 0;
    double v_v = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    v_n = 0LL;
    uint64_t t1 = (uint64_t)v_grid__n;
    uint64_t t2 = 0, t3 = t1;
    for (;;) {
        uint64_t t4 = t3 > t2 && (uint64_t)t3 - (uint64_t)t2 > 65536 ? t2 + 65536 : t3;
        for (; t2 < t4; t2++) {
            v_v = v_grid[t2];
            double t5 = v_v;
            int64_t t6[5];
            int t7 = sj_call_0(t5, sj_tb, t6);
            if (t7) { sj_tb[++sj_tb[0]] = ((int64_t)0 << 32) | t6[1]; sj_io[0] = t6[0]; sj_io[4] = t6[4]; SJ_FAIL(t7, 5, t6[2], t6[3]); }
            int64_t t8 = (int64_t)t6[0];
            int64_t t9 = t8;
            int64_t t10 = 1LL;
            int t11 = ((t9) == (t10));
            if (t11) {
                int64_t t12 = v_n;
                int64_t t13 = 1LL;
                v_n = sj_add((int64_t)t12, (int64_t)t13);
            }
        }
        if (t2 >= t3) break;
        if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
    }
    sj_io[0] = (int64_t)(v_n);
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

`Cell.WALL`, `==`, `!=`, `is`, `is not`, `.value` and truth all work as in Python. An `Enum` member equals only itself and is never equal to a number; an alias is the member it names; an `IntEnum` member is also an int, so it orders, adds and compares with ints. Members can be passed in, returned, put in conditional expressions, and passed between compiled functions, and what comes back is the member itself. The class is read from module level, and a different class compiles for itself.

Still Python: storing a member into an array or a list (store `.value`), `.name`, `Cell(1)`, iterating over the class, `Flag` and string enums, and an enum class that defines its own `__eq__`, `__hash__`, `__bool__` or ordering.

## Constants at module level

A function can read numbers, tuples of numbers, arrays, dicts, enum classes and (from 0.0.4) lists from module level:

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
<!-- c: apply(np.ones(16)) -->
```c-generated
int sj_kernel(double *v_x, int64_t v_x__n, double v_RATE, double v_TABLE__0, double v_TABLE__1, double *v_WEIGHTS, int64_t v_WEIGHTS__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    double v_s = 0;
    int v_s__np = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    v_s = 0x0.0p+0;
    v_s__np = 0;
    int64_t t1 = 0LL, t2 = v_WEIGHTS__n;
    if (t2 > t1 && t2 <= v_x__n - (0)) {
        int64_t t3 = t1, t4 = t2;
        for (;;) {
            int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
            for (; t3 < t5; t3++) {
                v_i = t3;
                double t6 = v_s;
                int t7 = v_s__np;
                int64_t t8 = v_i;
                int64_t t9 = t8;
                double t10 = v_WEIGHTS[t9];
                double t11 = t10;
                int64_t t12 = v_i;
                int64_t t13 = t12;
                double t14 = v_x[t13];
                double t15 = t14;
                double t16 = (t11 * t15);
                double t17 = v_RATE;
                double t18 = (t16 * t17);
                v_s = (t6 + t18);
            }
            if (t3 >= t4) break;
            if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
        }
    } else {
        int64_t t19 = t1, t20 = t2;
        for (;;) {
            int64_t t21 = t20 > t19 && (uint64_t)t20 - (uint64_t)t19 > 65536 ? t19 + 65536 : t20;
            for (; t19 < t21; t19++) {
                v_i = t19;
                double t22 = v_s;
                int t23 = v_s__np;
                int64_t t24 = v_i;
                int64_t t25 = t24;
                double t26 = v_WEIGHTS[t25];
                double t27 = t26;
                int64_t t28 = v_i;
                int64_t t29 = t28;
                if ((uint64_t)t29 >= (uint64_t)v_x__n) { sj_io[4] = 0; SJ_FAIL(1, 5, t28, v_x__n); }
                double t30 = v_x[t29];
                double t31 = t30;
                double t32 = (t27 * t31);
                double t33 = v_RATE;
                double t34 = (t32 * t33);
                v_s = (t22 + t34);
            }
            if (t19 >= t20) break;
            if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
        }
    }
    if (t2 > t1) { v_s__np = 1; }
    double t35 = v_s;
    int t36 = v_s__np;
    double t37 = v_TABLE__0;
    { double t38 = (t35 + t37); memcpy(&sj_io[0], &t38, 8); }
    sj_io[2] = t36;
    goto sj_done;
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

They behave like arguments, read at each call: a new value needs no recompile, a new type compiles for that type, and writes to a global array, list or dict reach it. Each costs about what an argument does, around 0.2 µs per call. A constant expression made only of literals, such as `MASK = (1 << 64) - 1`, keeps Python's value. A global that isn't there raises `NameError`, as usual.

Still Python: strings, `None`, functions that aren't `@scram`, objects other than [instances that compile](#instances-from-004), and assigning to a global with `global`.

## Sets and strings

Sets and strings keep a function in Python, as do lists and dicts that change size. Each would need allocation and a runtime inside compiled code.

## Gotchas

- **Attributes are read at the call, and written back at the end.** Inside one call that's invisible, but another thread changing the object during the call won't be seen, and won't see the writes until it ends. Nothing in compiled code can call back into Python, so only threads can see this.
- **Copies, not references, at the call (lists).** A list passed in is copied in and the changes copied back when the call ends. Inside one call that's invisible. But another thread changing the same list during the call, or a callback that reads it, would see the old values. Nothing in compiled code can call back into Python, so only threads can see this.
- **Identity in a changed row.** In a list of lists, a row whose numbers changed is replaced in one go, so `m[i][j] is old` can be false for a number that kept its value. Use `==`, as you would for numbers anyway.
- **`[[0] * w] * h` stays Python,** on purpose: its rows are one list. Write `[[0] * w for _ in range(h)]` for a table, which is also what you'd want in plain Python.
- **An empty list argument runs as Python** for that call, since its element type isn't known. A function that's usually given a non-empty list is compiled once, on the first call that is.
- **Mixed ints and floats** (`[1, 2.5]`) run as Python. Write `[1.0, 2.5]`.
- **Lists vs arrays:** a list's elements are Python numbers, with Python's rules (`1 / 0` raises, ints are 64-bit). An array's are numpy scalars, with numpy's (`np.float64(1) / 0` is `inf`). The same loop over the same values can differ between the two exactly where Python and numpy differ.
- **Preflight can't tell a list from an array** in source, since both are indexed the same way. It assumes an array; annotate `xs: list[float]` (or `list[int]`, `list[bool]`) to have it check a list.
