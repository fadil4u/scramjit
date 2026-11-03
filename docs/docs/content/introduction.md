# Introduction

scramjit is a just-in-time compiler for Python, written in pure Python, for the loops the interpreter is slowest at. Put `@scram` on a function with a hot loop; on the first call scramjit translates it to C, compiles that with [cimo](https://www.cimo.dev) inside your process, and from then on the call runs native code, without holding the GIL.

```python
import numpy as np
from scramjit import scram

@scram
def saxpy(a: float, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]

x = np.random.rand(1_000_000)
y = np.zeros_like(x)
saxpy(2.0, x, y)   # compiled on this call, native from then on
```
<!-- c: saxpy(2.0, np.ones(8), np.zeros(8)) -->
```c-generated
int sj_kernel(double v_a, double *v_x, int64_t v_x__n, double *v_y, int64_t v_y__n, volatile int *sj_stop, int64_t *sj_io) {
    *sj_stop = 0;
    int64_t v_i = 0;
    int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;
    int64_t t1 = 0LL, t2 = v_x__n;
    if (t2 > t1 && t2 <= v_y__n - (0)) {
        int64_t t3 = t1, t4 = t2;
        for (;;) {
            int64_t t5 = t4 > t3 && (uint64_t)t4 - (uint64_t)t3 > 65536 ? t3 + 65536 : t4;
            for (; t3 < t5; t3++) {
                v_i = t3;
                int64_t t6 = v_i;
                int64_t t7 = t6;
                double t8 = v_y[t7];
                double t9 = t8;
                double t10 = v_a;
                int64_t t11 = v_i;
                int64_t t12 = t11;
                double t13 = v_x[t12];
                double t14 = t13;
                double t15 = (t10 * t14);
                v_y[t7] = (t9 + t15);
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
                if ((uint64_t)t20 >= (uint64_t)v_y__n) { sj_io[4] = 0; SJ_FAIL(1, 12, t19, v_y__n); }
                double t21 = v_y[t20];
                double t22 = t21;
                double t23 = v_a;
                int64_t t24 = v_i;
                int64_t t25 = t24;
                double t26 = v_x[t25];
                double t27 = t26;
                double t28 = (t23 * t27);
                v_y[t20] = (t22 + t28);
            }
            if (t16 >= t17) break;
            if (*sj_stop) SJ_FAIL(8, 0, 0, 0);
        }
    }
sj_done:
    return 0;
sj_fail:
    sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;
    return sj_st;
}
```

There is no LLVM, no C compiler and no binary wheel involved: scramjit and cimo are both pure Python, so it works wherever Python runs, including Pyodide in the browser.

## Where it shines

Code that doesn't vectorize and is held back by the interpreter and the GIL:

- **Recurrences,** where each step depends on the one before (filters, integrators, running statistics).
- **Data-dependent branches and searches:** flood fills, path finding, dynamic programming, scans over byte buffers.
- **Simulations** that step state forward one cell, particle or agent at a time.

numpy is fast when the work fits one array expression; these don't. In CPython every iteration pays the interpreter, and threads take turns on the GIL. Compiled, the loop runs as C, and the GIL is released while it runs, so Python threads calling it run in parallel.

**Also a fit:** vectorizable array math written as plain loops, where only pure Python is allowed or numpy isn't available.

## Wherever Python runs

- **Python in the browser.** Pyodide, JupyterLite, marimo and PyScript have no Python compiler today, so a slow loop has no fix.
- **Plugins in embedded Pythons.** QGIS, Blender, Houdini, Maya, Nuke, FreeCAD and KiCad make compiled dependencies banned or painful.
- **Platforms other JITs haven't reached yet:** new CPython releases, free-threaded builds, Windows on arm64.
- **Fast hot paths in a pure-Python package,** with no wheel matrix to build and release.

## When not to

- Code that is already one vectorized numpy or pandas expression, where numpy runs: it is already running native loops.
- If an existing JIT already works for you on a standard desktop or server, it has a head start, notably SIMD vectorization.

## It is still Python

The decorated function is plain Python. Without scramjit it still runs, slowly, in CPython, and stays debuggable and testable as Python. scramjit's first rule is that it never changes results: a compiled function returns what CPython returns for the same function undecorated, within the documented [semantics](semantics.md).

## How these docs are organized

The first three parts are a walkthrough; read them in order the first time. **Next** at the bottom of each page takes you on. The **Reference** part has platforms and the FAQ. Press <kbd>/</kbd> anywhere to search.
