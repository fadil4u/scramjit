# Platforms

scramjit is pure Python and runs wherever cimo does: CPython 3.9 or newer, and Pyodide.

| Platform | Code runs as | Loaded with |
| --- | --- | --- |
| Linux arm64 / x86-64 | native arm64, amd64 | `mmap` + `mprotect` (W^X) |
| macOS, Apple silicon | native arm64 | `MAP_JIT` |
| Windows x64 | native amd64 | `VirtualAlloc` + `VirtualProtect` |
| Pyodide (browser) | WebAssembly | a side module linked into Pyodide's memory |

The same decorated function runs unchanged on all of them; scramjit generates the same C, and cimo compiles it for the machine it is on.

## Tested

scramjit is developed and benchmarked on arm64 Linux and in Pyodide, where its whole test suite runs: about 2,800 differential tests against CPython and numpy. It has been run by hand on Windows arm64 and on Windows x86-64 under emulation, where a NaN's sign can differ from CPython's (`inf % 1.0` gives `nan` with the other sign bit). cimo runs on x86-64 Linux and macOS, but scramjit's tests haven't been run on them yet, and nothing runs in CI yet.

## Security

Compiled code runs with the full privileges of your Python process, like an extension module or ctypes. There is no sandbox.
