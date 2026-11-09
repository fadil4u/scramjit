# Compilation and caching

## When compilation happens

On each call the decorator reads the argument types: Python type, dtype, number of dimensions and layout. If it has no compiled version for that signature yet, it generates C specialized to it, compiles it with cimo and caches it.

## The GIL

Compiled code releases the GIL while it runs, so several Python threads calling compiled functions run in parallel. Compiling itself is Python code and holds the GIL like any other.

## Splitting one loop across threads

A loop whose iterations don't depend on each other can run on every core: give the function the range to work on, and call it once per part from a thread pool.

```python
from concurrent.futures import ThreadPoolExecutor

@scram
def settle(xs, ys, lo, hi):
    for i in range(lo, hi):
        s = xs[i]
        for _ in range(50):
            s = math.sin(s) + 0.5
        ys[i] = s

def split(fn, n, workers, pool, *data):
    bounds = [(k * n // workers, (k + 1) * n // workers) for k in range(workers)]
    list(pool.map(lambda b: fn(*data, *b), bounds))

with ThreadPoolExecutor(4) as pool:
    split(settle, len(xs), 4, pool, xs, ys)
```

On a million elements, four threads ran 3.9x faster than one call, eight 5x, with the same bits in `ys` ([`examples/threads.py`](https://github.com/farizrahman4u/scramjit/blob/main/examples/threads.py)). What to watch:

- **The parts must not depend on each other.** Nothing checks this. If one part reads what another writes (`ys[i - 1]`), the result depends on timing. Two parts writing the same element is a data race ([gotchas](gotchas.md#threads-really-do-run-at-the-same-time)).
- **Sums across parts:** give each part its own result (return it) and add them up after. For floats that changes the order of the additions, so the last bits can differ from one call over the whole range.
- **Make the parts big.** Four tiny parts through a pool took about 0.4 ms against 4 µs for one direct call (Linux, 8 cores), so a part should run for at least a millisecond. Keep one pool for the program rather than one per call.
- **What can be shared:** numpy arrays are passed without copying, so every part works on the same memory. Lists, lists of instances and lists of lists are copied in and written back at the end of each call, only the elements the call changed (rows, for a list of lists), so parts writing different elements of one list keep each other's writes. One instance or dict shared by threads is written back attribute by attribute, every attribute the function may write, so give each thread its own if more than one writes to it.
- **In the browser,** Pyodide has no threads, so this runs one part after another.

## Two caches

- **In memory,** for the life of the process.
- **On disk,** so a second process reuses the compiled code instead of compiling again.

The disk cache is keyed on the C generated for a function and its argument types, so it holds through edits elsewhere, through moving the function within its file and renaming the file, too. Changing the function, or one it calls, compiles again.

## First-call latency

Compiling happens in pure Python, so the first call to each specialization takes noticeably longer than the ones after it. The disk cache means that cost is paid once per machine, not once per process. Pass `cache=False` to keep compiled code in memory only.
