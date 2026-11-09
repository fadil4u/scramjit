# Install

```console
$ pip install scramjit
```

scramjit is pure Python. Its only dependency is [cimo](https://www.cimo.dev), which is itself pure Python with nothing outside the standard library. It runs on CPython 3.12 or newer, and in Pyodide.

## numpy

scramjit works on numpy arrays but does not require numpy: a function that only uses scalars and `range` needs no numpy at all. Arrays are read in place through the buffer they expose, without copying.

## In Pyodide, JupyterLite or marimo

```python
import micropip
await micropip.install("scramjit")
```

This installs cimo too. See [In the browser](browser.md).
