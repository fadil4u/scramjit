# Roadmap

The milestones and their acceptance criteria, from the [spec](https://github.com/farizrahman4u/scramjit/blob/main/requirements.md#milestones-and-acceptance-criteria).

## M0: subset check and prior-art study

Write the planned cimo demo kernels (reaction-diffusion, flood fill, viewshed) as plain Python, and hand-write the C scramjit should generate for each. Study existing compilers' typing and lowering.

**Done when** every kernel fits the subset, or the subset is amended with the reason, and the findings are recorded in the spec.

## M1: scalars, 1-D arrays, `range`

`ast` to C for scalar arithmetic, `range` loops, `if`, `while` and 1-D numpy arrays; annotations as contracts; `int64` and `float64` only; in-memory cache.

**Done when** the M0 kernels that need only these features run, match CPython on random inputs, and are benchmarked against pure CPython.

## M2: builtins, lists, narrowing

Lowering rules for `enumerate`, `zip`, `map`, `filter`, comprehensions and the reductions; fixed-size lists as arrays; narrow types from annotations, dtypes and range analysis.

**Done when** each rule has differential tests and the generated C for each M0 kernel contains no iterator or boxing code.

## M3: n-D arrays, errors, disk cache, Pyodide

**Done when** all M0 kernels run in native CPython and in Pyodide, a second process reuses the disk cache, and every construct outside the subset gives an error pointing at the Python line.
