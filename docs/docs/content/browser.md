# In the browser

scramjit runs wherever cimo does. In Pyodide, JupyterLite, marimo and PyScript, cimo compiles to WebAssembly in the page and links it into Pyodide's memory, so a decorated function runs as WebAssembly instead of as interpreted Python.

```python
import micropip
await micropip.install("scramjit")

from scramjit import scram

@scram
def mandel(xs, ys, out, iters: int):
    for j in range(len(ys)):
        for i in range(len(xs)):
            zr = zi = 0.0
            n = 0
            while n < iters and zr * zr + zi * zi < 4.0:
                zr, zi = zr * zr - zi * zi + xs[i], 2 * zr * zi + ys[j]
                n += 1
            out[j, i] = n
```

This is the case scramjit exists for first: existing Python compilers don't run in the browser, so a slow loop there has no fix today.
