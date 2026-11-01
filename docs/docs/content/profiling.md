# Profiling and visibility

There is no fallback to running on Python objects. A decorated function either compiles, or raises a compile error that points at the Python line. So if a call returns, it ran native code. The tools on this page show what was compiled, what it cost, what the compiler decided and what you can change.

## Profile a whole program

```console
$ python -m scramjit simulate.py --steps 2000
```

runs the script like `python -m cProfile` would, then prints a report:

```text
scramjit · 4.21 s run · 96% in native code
  function        calls   compile   native   share
  step            2,000   disk      3.81 s     91%
  flood             120   0.14 s    0.22 s      5%
  ema ×2              2   0.07 s    0.02 s     <1%

hints
  ema: a arrives as int and float (2 specializations); annotate a: float
  flood, line 12: list seeds is copied in and out on each call; pass a numpy array
```

- **native share** is the part of the whole run spent in compiled code. The rest is your Python. If a hot loop isn't in the table, it isn't decorated.
- **compile** is the time spent compiling, or `disk` when the code came from the [disk cache](caching.md).
- **×2** means the function was compiled for two signatures.

Arguments after the script name are passed to it. `python -m scramjit -m package.module` profiles a module, as with `cProfile`.

## Explain one function

`f.explain()` prints the function's source, annotated with what the compiler decided for each compiled signature:

```text
>>> mandel_rows.explain()
mandel_rows(int32[:, :], int, int, int)  38 ms · disk

 7  h, w = out.shape
 8  for j in range(y0, y1):        j: int64
 9      ci = 2.4 * j / h - 1.2     ci: float64
10      for i in range(w):         i: proven in bounds
14          while n < iters and …  not unrolled: exit depends on the data
17          out[j, i] = n          j: checked once per call, not per pixel
hints  none: no boxing, no copies
```

It reports:

- **types:** what each variable was inferred as, and where a type was narrowed;
- **bounds checks:** proven unnecessary, hoisted out of the loop, or kept, and what would let a kept one go;
- **loops:** which ones cimo's optimizer unrolled or otherwise improved, and why the others weren't;
- **the call boundary:** arrays passed in place, lists copied, strided arrays.

Pass a signature to explain only that one: `f.explain(f.signatures[0])`.

## Check before you run

```console
$ python -m scramjit check src/
src/regions.py:41  flood
  queue.append() grows a list; growable lists aren't supported. Use a numpy array as a stack.
src/stats.py:18  summarize
  {} creates a dict; dicts aren't supported.

2 of 9 @scram functions won't compile
```

`check` reads every `@scram` function under the path and reports constructs outside what scramjit compiles, without importing or running anything. It exits non-zero when it finds any, so it fits in CI. Type errors, such as a variable that is sometimes a number and sometimes an array, depend on the arguments, so they are reported on the first call.

## Specializations and the generated C

```python
>>> ema.signatures
[(float64[:], float64[:], float), (float64[:], float64[:], int)]
>>> print(ema.c(ema.signatures[0]))
```

`f.c()` returns the C scramjit generated, with `#line` directives pointing back at your Python. Without an argument it uses the most recent signature.

## Log compiles

```console
$ SCRAMJIT_LOG=1 python simulate.py
scramjit: compiled step(float64[:, :], float64[:, :], float) in 0.18 s
scramjit: flood(uint8[:, :], int64[:, :], int, int) from disk cache
```

One line per compile or cache hit. A function that keeps compiling new signatures is being called with arguments of changing types; annotate them, or convert once before the loop.

## Acting on hints

| Hint | Why it costs | Fix |
| --- | --- | --- |
| an argument arrives as several types | one compile and one specialization per type | annotate it, such as `a: float` |
| a list is copied in and out | O(n) work on every call | pass a numpy array |
| an array is strided | every index multiplies by a stride | `np.ascontiguousarray` once, outside the loop |
| a bounds check stays in a hot loop | a compare and branch per access | index with the loop variable over `range(len(a))`, or `@scram(boundscheck=False)` once verified |
