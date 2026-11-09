# Platforms

scramjit is pure Python and runs wherever cimo does on CPython 3.12 or newer, and in Pyodide.

| Platform | Code runs as | Loaded with |
| --- | --- | --- |
| Linux arm64 / x86-64 | native arm64, amd64 | `mmap` + `mprotect` (W^X) |
| macOS, Apple silicon | native arm64 | `MAP_JIT` |
| Windows arm64 / x64 | native arm64, amd64 | `VirtualAlloc` + `VirtualProtect` |
| Pyodide (browser) | WebAssembly | a side module linked into Pyodide's memory |

The same decorated function runs unchanged on all of them; scramjit generates the same C, and cimo compiles it for the machine it is on.

## Tested

About 3,000 differential tests compare each compiled function with CPython and numpy, bit for bit. CI runs them on Linux and Windows, x86-64 and arm64, macOS on Apple silicon, CPython 3.12, 3.13 (free-threaded too) and 3.14, and runs part of them in Pyodide. scramjit is developed and benchmarked on arm64 Linux. A NaN's sign bit may differ from CPython's (see [Gotchas](gotchas.md)). Intel Macs aren't supported: cimo doesn't generate code for them.

## Security

Compiled code runs with the full privileges of your Python process, like an extension module or ctypes. There is no sandbox.
