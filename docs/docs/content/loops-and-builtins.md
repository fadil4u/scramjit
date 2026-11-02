# Loops and builtins

No builtin exists at runtime as a callable. Each one is a rule in the compiler that emits plain C, so `range`, `zip` and generator expressions cost nothing beyond the loop they describe. Iterator protocols (`__iter__`, `__next__`) never appear in generated code.

## `range`

```python
for i in range(a, b, s):
    ...
```

becomes

```c
for (int64_t i = a; i < b; i += s) {
    ...
}
```

No range object exists. A step that isn't a constant gets the sign check Python implies, so a negative step counts down as it does in Python.

`for` and `while` can have an `else`, which runs unless a `break` left the loop:

```python
for i in range(len(x)):
    if x[i] == target:
        found = i
        break
else:
    found = -1
```

## `enumerate`, `zip`, `reversed`

Over arrays and ranges these become index loops:

```python
for i, (u, v) in enumerate(zip(a, b)):
    out[i] = u * v
```

is one counted loop over the shorter of `a` and `b`, reading `a[i]` and `b[i]` directly.

## Comprehensions, `map`, `filter` and generator expressions

These fuse into one loop with no intermediate iterators. A result is materialized only when it is stored:

```python
total = sum(x * x for x in a if x > 0)
```

reduces in place:

```c
double total = 0;
for (int64_t i = 0; i < a_len; i++)
    if (a[i] > 0)
        total += a[i] * a[i];
```

## Reductions and math

`len`, `sum`, `min`, `max`, `abs`, `any`, `all` and `math.*` become inline code or C math calls.
