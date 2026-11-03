# Install

```console
$ pip install scramjit
```

scramjit is pure Python. Its only dependency is [cimo](https://www.cimo.dev), which is itself pure Python with nothing outside the standard library. It supports what cimo supports: CPython 3.9 or newer, and Pyodide.

## Before 0.0.3

The latest release on PyPI is 0.0.2. Calls between `@scram` functions, calling C, narrow dtypes, shifts and 64-bit hashes are on `main` only, until 0.0.3. To use them now, install both packages from GitHub:

```console
$ pip install "cimo @ git+https://github.com/farizrahman4u/cimo#subdirectory=python" "scramjit @ git+https://github.com/farizrahman4u/scramjit"
```

## numpy

scramjit works on numpy arrays but does not require numpy: a function that only uses scalars and `range` needs no numpy at all. Arrays are read in place through the buffer they expose, without copying.

## In Pyodide, JupyterLite or marimo

```python
import micropip
await micropip.install("scramjit")
```

This installs cimo too. See [In the browser](browser.md).
