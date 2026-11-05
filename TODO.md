# TODO

Biggest first. `requirements.md` is the spec; this is the work queue.

## Release

0. **0.0.3 released (2026-11-03),** with cimo 0.0.3, which it needs for calling C. The rule stays: the README, site and
   docs describe the released version, and a feature on `main` but not on PyPI is released the same day or marked
   "from 0.0.4" wherever it's mentioned (`AGENTS.md`). `tests/test_version.py` checks the version agrees everywhere it's written.

## Object model (next; replaces the case-by-case instance work)

Decided 2026-11-05 (see "Object model" in `requirements.md`). Instances, lists and dicts are flattened today by AST rewrites that each
know one shape (a parameter, an element, an inner object, a local made here, an argument), so every *combination* needs its own code
and the rest falls back: inner objects of list elements, a function making instances called from compiled code (`a.add(b)` returning
a new `Vec`), a list in an attribute, `[row, row]`, `b.pos is b.vel`, `callee(a, a)`, a writing method in a comprehension. The
workarounds show the design's limit: `p__a = p__a` write markers, "mark every inner number written", `check_shared`'s refusals,
parameter order kept equal in two places by hand (`leaf_types` vs `_push`; getting it wrong gave a wrong answer on 2026-11-05).
Replace it with one model in which composition is free.

**The model.**
1. **Static paths are flat variables.** At the call, the dispatcher walks every access path the function names by a fixed path
   (`a.x`, `a.y`, `b.pos.x`, `self.center.x` through a method, a dict's `p['k']`, through callees and methods transitively) and maps
   each to the object it lands on, by identity (one memo per call). Paths that land on one object share one variable or buffer:
   `a.y = a.x` gives one buffer, `b.pos is b.vel` one set of variables, `callee(a, a)` one set for both parameters. The **alias
   pattern** (which static paths share an object) is part of the signature, like a dtype: inside the kernel there is nothing to
   alias, every field is a register candidate, and today's flattening is the special case "every path a different object".
   Checking the pattern costs O(paths in the code), never O(data).
   **Built (2026-11-05)** for instances reached by static paths: `_object_graph` (dispatcher), `T.same_as`, `flatten_dicts` renaming a
   `same_as` parameter to its target's path, and `share_passed` for `f(a, a)`/`p.absorb(p)` in compiled code; an object handed to a callee
   gets its `same_as` rebased (inside it) or expanded into the target's fields (outside it), `handed` in `flatten_dicts`. `tests/test_alias_pattern.py`,
   `examples/shared_target.py`. Left of point 1: dicts and lists in the pattern (an instance's `__dict__` passed as a dict still refuses; one list
   under two names too), and plans for a pattern through attributes (they take the general path).
2. **Data-chosen objects use pools with integer indices.** Only where *which* object an access reaches depends on run-time data:
   - indexing a collection that may share (`ps[i].x` with `ps` holding duplicates or objects reachable elsewhere): a struct-of-arrays
     pool per class (one array per attribute, today's columns) and `ps` as an array of pool indices, so `ps[i].x` is
     `pool_x[ps_ix[i]]`. A list of distinct objects needs no index array: the pool *is* the list (today's layout, which vectorizes);
   - rebinding a reference (`a.pos = b.pos`, maybe under an `if`): `a.pos` becomes a stored index;
   - walking a structure (`node = node.next`, trees): the shape is not put in the signature (a recompile per length); `next` is an
     index field;
   - objects made in a loop and kept (stored into an attribute or a list): allocated in the pool, which grows.
   One indirection exactly where C would pay one; nothing anywhere else.
3. **One marshaller at the boundary**, replacing `_object_fields`/`_objlist_type`/`_dict_type`/list copies: walks the graph from the
   arguments and globals along the paths and pools the function needs, with the identity memo; checks each class as today (default
   `__getattribute__`, no descriptor on a used name, the `__mro__`/dict identity guards of `_object_guards` for plans); builds the
   variables, buffers and pools; after the kernel, writes back what changed (values that didn't change keep their identity, as
   `_copy_back` does), and turns pool objects that escape (returned, stored into an argument) into Python objects: new instances via
   `cls.__new__` plus `object.__setattr__` in `__init__`'s assignment order, as `instance_ret ('made', ...)` does now.
4. **Construction compiles `__init__` as an ordinary method** on a fresh pool slot (or flat variables when the instance never escapes
   and is never chosen by data), instead of `_init_recipe`'s "only assigns". Any compilable `__init__` works; `__new__`, a metaclass
   `__call__` and non-plain `__init__` still refuse.
5. **Calls between compiled functions pass references, not copies**: a static-path object as its variables by reference (C pointers to
   the caller's locals, or the callee inlined with the caller's variables), a pool object as its index. No result-slot copy-back
   (`take_writes`), no write markers: a callee's writes are where the caller reads. A callee returning a new instance returns a pool
   index (or its fields, when it can be flat in the caller), which makes `a.add(b)` and `__add__` work.

**What stays Python** (genuinely new surface, not combinations): attribute names known only at run time (`getattr(p, name)`,
`hasattr`, `vars`), adding attributes (the layout is fixed per signature), a variable holding different classes at different times
(later: a compile per class, or a tagged union), `__getattr__`. Properties are methods and can come after.

**Speed, measured 2026-11-05 through cimo** (spring step, ns per step, arm64): locals 5.2; a struct pointer 6.1; a `restrict` pointer
6.1; writes through one pointer and reads through another 11.6 against 6.4 for the same arithmetic in locals; `restrict` on both 11.6.
cimo honours `restrict` for dependences, not for keeping fields in registers (no LICM scalar promotion; its `TODO.md`, "Keep struct
fields in registers across a loop"). The model above avoids depending on that: static paths are locals, so only the pool indirection
for data-chosen objects costs anything, and that is a gather C would do too. Also worth asking cimo for: struct-path alias analysis
(`a->vx` and `b->x` can't overlap whether or not `a == b`), which `check_shared` does by hand today.

**How to build it.** The ~200 class tests from 2026-11-05 are the specification and must pass unchanged in observable behaviour
(`tests/test_classes.py`, `test_methods.py`, `test_instance_lists.py`, `test_passing_instances.py`, `test_instance_returns.py`,
`test_isinstance.py`, `test_made_instances.py`, `test_nested_instances.py`, `test_dicts.py`, `test_list_args.py`, `test_nested_lists.py`);
tests that assert a fallback for a combination the model now handles flip to compiled, each checked against Python. Then:
1. Path analysis on the tree: every static path per parameter/global/local, through methods (by class at key time, `_methods_called`)
   and callees (`_param_uses`), as one structure, replacing `_attrs_used`, `_element_attrs`, `_nested_uses`, `_passes`, `_returned`.
2. Signature = per-path types + alias pattern + pool shapes; marshaller as in 3 above, with call plans for the common shapes (numbers
   only, no sharing) so a call stays near 2 µs.
3. Lowering: one flatten for all owners (parameters, globals, made locals, inner objects, elements) driven by the path table, replacing
   `flatten_dicts`, `SelfArg`, `shown_writes`, `check_shared`, `take_writes`, `narrow_passed`; pools and indices as a new lowering for the
   four data-chosen cases.
4. Static preflight builds the same path table from class statements (`_static_self`, `_static_inner`, `_static_made` today).
5. Measure before and after: `bench/`, `examples/classes.py`, `particles.py`, `vectors.py`, `lists.py`, `config_dict.py`; nothing that
   compiles today may get slower.
Done when the "Stays Python, for now" lists in `data-structures.md` hold only the genuinely-new surface above, the class examples are no
slower, and `TODO.md`'s instance items are deleted.

## Demos

1. **Standalone demo pages, one per demo, with the LLM demo's shell.** `docs/demos/llm/` is the
   model: a full page at `scramjit.dev/demos/<name>/` with the canvas, the editable code, the
   scramjit switch, the stats strip (frames per second, iterations per second, speedup, compile
   time), the tabs for the syntax tree, C and WebAssembly, the share-an-edit link, and the
   "compiles stay between visits" cache. Make the shell one module (`docs/demos/shell/`: `page.js`,
   `worker.js`, the CSS) that each demo fills with a Python file, a draw function and presets, so a
   new demo is one `.py` and one small `.js`. Then:
   - **Existing lab demos as pages:** `demos/flock`, `demos/erosion`, `demos/sand`, `demos/fractal`
     from `docs/lab/*.py`, unchanged in substance, each with room to show the full code (the home
     page's lab truncates). The lab on the home page keeps linking to them.
   - **`demos/3d`: strange attractors.** Aizawa, Dadras, Halvorsen, Lorenz, Rössler, Thomas (cimo's
     docs have posters of each): a `@scram` function integrates a million points with RK4 per frame
     into a float32 array that three.js (already on the site) draws as a rotating point cloud;
     plain Python manages a few thousand. Presets switch the system; the parameters are editable
     and recompile live. Color by speed.
   - **`demos/lenia`: continuous cellular automaton.** Lenia (Chan 2019) with one kernel ring and
     growth function, 256x256 at first, the convolution as a plain loop (no FFT: that is the point),
     drawn to a 2D canvas with a colormap. Presets: Orbium and two other known creatures. Edit the
     growth function's `mu`/`sigma` and watch it change. SmoothLife as a second preset family if
     cheap.
   - **`demos/tracer`: progressive path tracer.** A Cornell-box scene of spheres and planes, one
     sample per pixel per frame accumulated into a float32 buffer, tone-mapped to the canvas: noise
     resolves to a clean image in seconds under scramjit, barely clears the first pass as Python.
     Editable material parameters and light. 256x256 to start; measure what Pyodide sustains.
   - **`demos/fluid`: the race.** Stable fluids (Stam 1999: diffuse, advect, project, about 80
     lines) with dye injected by the mouse, **two panels on one clock**: plain Python on the left,
     scramjit on the right, both given the same real time per frame as `tools/readme_gif.py` does
     for the Mandelbrot race, with the clock and iteration counts on screen and nothing staged.
     This is the social-media asset: it needs no explanation.
   - **A recording mode** on every page: `?record=1` hides everything but the canvas, the stats and
     the switch, sizes the canvas to 1920x1080 (2x device pixel ratio), and `?autorun=<seconds>` flips
     the switch after that many seconds, so a clean 1080p screen capture needs no clicks. Document it
     in `docs/demos/README.md` with the shot list (3 s plain Python, flip, 5 s smooth, one edit, hold).
   - Each page states its own numbers from a measurement in Pyodide on a named machine, as the lab
     table in the README does, and the README's demo table gains a row per new demo.
   - Order: the shell with the four existing demos first (they're known to work), then fluid (the
     asset), then 3d, lenia, tracer. Each lands with its page, a preset test in `tests/`, and a line
     in the README.

## Directions (priority order; each is weeks, not days, and has no oracle to diff against)

1. **Autodiff: gradients of any `@scram` function.** Source-level differentiation on scramjit's own
   IR (after type inference, before C): forward mode first (dual numbers per scalar, arrays of
   tangents), then reverse mode on loop bodies over arrays with a tape only where control flow
   needs one (Tapenade/Enzyme-style; the generated C stays plain loops cimo optimizes as usual).
   API: `grad(f)`, `jvp`, `vjp`, matching JAX's names for the subset. Correctness oracle: finite
   differences and, where it runs, JAX on the same function. Targets, in order: ODE/PDE parameter
   fitting (a `@scram` RK4 integrator differentiated through time), a physics-informed loss,
   inverse problems. It runs in Pyodide, so the demo is "fit a model to data in a tab."
   Hard parts, measured before designed: reverse mode through `while`, in-place array updates
   (checkpoint vs. recompute), and `math` functions (cimo's libm needs derivatives).
2. **Equation in, compiled solver out.** Tools that already generate C and shell out to a compiler
   run on cimo instead (cimo's `TODO.md`, "Directions"). scramjit's part: the Python-side ones,
   where the model is written as Python and the solver is a `@scram` loop: an ODE/SDE system from
   a right-hand side function, a stencil PDE from an update rule, with the time loop, boundary
   conditions and output stepping generated. First target: a minimal Devito-shaped API (grid,
   stencil expression, time loop) over scramjit, running in JupyterLite.
3. **GPU from a `@scram` function.** When cimo has a WGSL target (its `TODO.md`), `@scram(target='gpu')`
   maps a kernel over an index space: the function body becomes the per-element kernel, arrays
   become buffers, reductions are the first thing that won't map. Through WebGPU in the browser and
   wgpu natively; no CUDA, ROCm or Metal SDK. scramjit's work is the mapping rules (what a loop
   over `range(n)` means on a GPU, which loops stay on the CPU) and the buffer transfer, measured
   against the CPU path on one matmul and one stencil.
4. **Autotuning per machine.** Expose cimo's tuner (its `TODO.md`, "Directions") as
   `@scram(tune=True)`: the first call compiles the variants cimo proposes, times them on the
   user's CPU, keeps the best in the cache keyed by CPU model. Report what won in `preflight`.
5. **Model-written code, fast and sandboxed.** The product surface for code an LLM wrote: RL
   environments and reward functions, verifiers, data-pipeline steps, run as `@scram` on the WASM
   target in-process with a time limit and a memory limit, and natively when the code is trusted.
   Build the harness: `scramjit.sandbox(source, inputs, limits)` returning results or a typed
   failure; a Gymnasium-shaped environment API; a benchmark against a subprocess/container sandbox
   on throughput and startup. The measurement decides whether this is a product: do it before the API.

## Adoption (priority order, below Directions; each is days, and has an oracle: the existing behavior, numpy, or Numba)

Rules for every item here: no new dependency (numpy and the optional library under test stay dev
extras); every step lands with its tests in `tests/` and its page in `docs/docs/content/`, and the
README's feature table changes only when PyPI has the feature (Release #0); measure before and
after with `bench/`, and write the numbers into the item when done.

1. **`@scram(parallel=True)`: a loop over a thread pool, like Numba's `prange` without Numba.**
   What exists: compiled code releases the GIL, cimo gives each thread its own heap part
   (`caching.md`, cimo's `threads.md`). What's missing is the split of the work.
   - **API.** `@scram(parallel=True)` makes the function's *outermost* `for i in range(a, b)` loop
     (or `range(n)`) run in `scramjit.threads()` threads (default `os.cpu_count()`, settable with
     `scramjit.set_threads(n)` and `SCRAMJIT_THREADS`). Also `prange` as an explicit marker:
     `from scramjit import prange`, usable as `range` when undecorated, so code runs unchanged as Python.
   - **Lowering** (`_compiler`, then `_dispatch`): generate the C with two extra leading parameters
     `(long block, long nblocks)`; the chosen loop's bounds become `lo = a + block * chunk`,
     `hi = min(b, lo + chunk)` with `chunk = ceil((b - a) / nblocks)`. The dispatcher calls the
     compiled function `nblocks` times from a `ThreadPoolExecutor` kept per process (not per call:
     thread start costs more than a small loop). One call when `nblocks == 1` or the range is
     shorter than `nblocks * 1024` elements, measured: find the break-even on `bench/kernels.py`'s
     EMA-sized loops and put the number in the docs.
   - **What must refuse, with a clear `CompileError`** (and fall back to the serial compile, not to
     Python, with a `FallbackWarning` saying why): a loop-carried dependency in the chosen loop
     (`y[i]` read with `y[i-1]`, a scalar accumulated across iterations that isn't a reduction),
     writes to a scalar read after the loop, `break`/`return` inside the loop, calls to
     non-`@scram` functions. Detection: reuse the dependence analysis the bounds-check versioning
     uses (`checks`); anything it can't prove independent is refused. Start strict; widen with tests.
   - **Reductions.** `s += x[i]` and `s = max(s, ...)` across the parallel loop: each thread keeps
     its own `s` in a per-block output array; the dispatcher combines in Python in block order.
     For floats this changes the summation order against serial, so: allowed only with
     `parity=False`, documented, and the combine is left-to-right so the result is deterministic
     for a fixed thread count. With `parity=True` a float reduction refuses as above.
   - **Tests** (`tests/test_parallel.py`): results equal to the serial compile on every kernel in
     `tests/test_kernels.py` for 1, 2, 3, 8 threads and for ranges shorter than the thread count
     (including empty); each refused shape raises/falls back with the expected message; a
     stress loop of 1,000 calls on 8 threads with `malloc` inside (per-thread heaps); exceptions
     raised inside one block (an `IndexError` with `boundscheck=True`) propagate and the other
     blocks finish. Run the suite once under free-threaded Python 3.13t (`python3.13t`) in CI
     (`ubuntu-24.04-arm`, `deadsnakes` or `uv python install 3.13t`): no crash, same results.
   - **Browser.** Out of scope for this item; note in `browser.md` that `parallel=True` runs serial
     in Pyodide (no threads without COOP/COEP), with the Web Worker + `SharedArrayBuffer` version as
     a later item.
   - **Done when:** Gray-Scott and the flood-fill-by-rows kernel scale to at least 3x on 4 threads
     natively (`bench/parallel.py`, numbers in `threads.md`), Numba's `prange` on the same loop is
     within 1.3x, and `preflight` reports "parallel" or the reason it isn't.

2. **Shipping a package built on scramjit: pure-Python wheel, native speed, no compile for the user.**
   cimo's `distribution.md` covers the dependency and the vendored one-file cimo; what's missing is
   the scramjit side and shipping compiled code so a user never waits for a compile.
   - **Precompiled cache in the wheel.** When cimo's public cache export/import lands (cimo
     `TODO.md`: "public cache export and import"), add `scramjit build <package>`: imports the
     package in a subprocess with `SCRAMJIT_RECORD=1`, calls nothing, but compiles every `@scram`
     function for the signatures listed in a `scramjit.toml` (`[[precompile]] function = "pkg.mod.f"
     types = ["float64[:]", "float64[:]", "float"]`), one entry per target platform the build runs on
     (`linux-arm64`, `linux-x86_64`, `macos-arm64`, `windows-x86_64`, `windows-arm64`, `wasm32`), and
     writes the cache entries under `pkg/_scramjit_cache/<platform>/`. At import, `scram` looks
     there before compiling. Until cimo's export exists, the item stops at the signature list and
     the subprocess driver, tested with the private cache layout the LLM demo already uses.
   - **Signatures without a call.** `@scram` needs argument types to compile; add
     `f.compile("float64[:], float64[:], float")` (the type-string syntax `_types.py` already prints)
     so a package can compile at import time deliberately, and `scramjit.toml` uses the same strings.
   - **An example package** at `examples/package/`: `pyproject.toml` building a `py3-none-any` wheel,
     one `@scram` kernel, the `scramjit.toml`, a GitHub Actions workflow that builds the wheel once
     and runs its tests on the five native platforms and in Pyodide (`tools/run_pyodide.mjs`), and
     a README that is the guide. Copy its `pyproject.toml` into `docs/docs/content/packaging.md`.
   - **Tests:** `tests/test_packaging.py` builds the example wheel with `python -m build`, installs
     it in a temp venv, imports it with `SCRAMJIT_LOG=1`, and asserts no compile happened (log is
     empty) when the precompiled cache matches, and exactly one compile when it doesn't.
   - **Done when:** `pip install` of the example wheel followed by the first call takes under 50 ms
     on a machine the cache was built for, and the guide fits on one page.

3. **Notebook integrations: marimo, JupyterLite, Jupyter Book.** The Pyodide path works; the
   gap is that nobody finds it. Three small deliverables, in this order:
   - **marimo.** A `marimo` notebook in `examples/notebooks/` that runs in marimo's WASM mode
     (`marimo export html-wasm`) with one `@scram` kernel behind a slider; a PR or issue to marimo
     listing scramjit in their docs' "packages that work in WASM" page, with the notebook as the
     example. Check `micropip.install("scramjit")` works from a stock Pyodide index (both wheels are
     pure Python, so it should; if `cimo` isn't resolvable, that's a packaging bug to fix first).
   - **JupyterLite.** A `jupyterlite` site config in `docs/notebooks/` (`jupyter_lite_config.json`
     with `piplite` wheels for `cimo` and `scramjit`) deployed to `scramjit.dev/notebooks/`, with the
     Gray-Scott and the EMA notebooks. Pyodide's kernel runs in a worker, so the main-thread sync
     compile limit (cimo `requirements.md`, "Platform loading") doesn't apply; note it.
   - **Jupyter Book / Sphinx.** A `sphinx` directive `.. scramjit-demo::` is too much; instead a
     doc page `notebooks.md` showing the `thebe`/JupyterLite embedding so a book's figure runs a
     `@scram` kernel live, and one such page on the site as proof.
   - **Tests:** `tests/test_notebooks.py` executes each notebook with `nbclient` natively (dev extra)
     and `tools/run_pyodide.mjs` runs the kernels' code in Pyodide. **Done when:** both notebook
     environments show scramjit in their own docs or examples, or the PRs are open.

4. **Native UDFs over Arrow, DuckDB and Polars buffers.** Their columns expose the buffer protocol
   or `__array__` zero-copy, so a `@scram` function can be a vectorized UDF with no C++ extension.
   - **Arrow first** (`pyarrow` dev extra): accept a `pyarrow.Array`/`ChunkedArray` of a primitive
     type as an argument, mapped to numpy via `to_numpy(zero_copy_only=True)` in `_dispatch`
     (`_arg_type`, `_push`), with the validity bitmap passed as a second hidden `uint8[:]` argument
     when nulls exist and `scramjit.isnull(x, i)` to read it. Chunked arrays: one call per chunk,
     results concatenated, in order. Strings and nested types stay Python (fallback message says so).
   - **DuckDB:** `duckdb.create_function(name, f, type="arrow")` takes a function over Arrow
     batches; show `@scram` inside it on a 10-million-row aggregate, measured against DuckDB's own
     SQL and against a plain-Python UDF. **Polars:** `df.with_columns(pl.col("x").map_batches(f))`
     over `to_numpy()` the same way. One example each in `examples/udf_*.py`.
   - **Tests:** `tests/test_arrow.py` (skipped without `pyarrow`): zero-copy asserted by writing
     through the array and reading from Arrow; nulls; chunked; results against numpy. **Done when:**
     the DuckDB example runs the UDF within 2x of the equivalent SQL expression, numbers in a new
     `udfs.md`.

5. **`deterministic=True`: the same bits on every platform.** cimo bundles its own libm and controls
   instruction selection; `parity=False` already uses that libm. What's missing is the promise and
   the proof.
   - **Define it:** `@scram(deterministic=True)` implies `parity=False`, `fma=False` in the cimo
     compile (no contraction, which differs between arm64 and x86-64), no reassociated reductions,
     and `parallel=True` with a fixed thread count only. Document what it does *not* promise:
     numpy calls that stay Python (their results follow numpy's SIMD paths), and `uint64`/narrow
     dtypes until their rules are platform-independent (item 1 under "Next").
   - **Prove it:** `tests/test_deterministic.py` runs a fixed set of kernels (the lab demos,
     `bench/kernels.py`, the `math`-heavy viewshed) on 1,000 fixed inputs and writes SHA-256 of the
     outputs to `tests/data/deterministic.json`; CI on all five platforms and Pyodide compares
     against the committed hashes and fails on any difference. The first run finds the places cimo
     isn't deterministic (expect: libm `pow`/`exp` paths that differ between the scalar and vector
     versions, x86 `sqrt` vs arm64); each becomes a cimo issue, fixed or excluded with a reason.
   - **Done when:** the hash file is identical from every CI job and the doc page `determinism.md`
     lists exactly what is covered.

## From cimo

- **Signed overflow is now undefined in cimo, as in C** (`compile(wrapv=True)` for `-fwrapv`): `x + 1 > x` folds to true
  and `int` counters widen to 64 bits. Generated C must keep doing wrapping arithmetic through unsigned, as it
  does today (scramjit's 726 tests pass).

## Workarounds waiting on cimo

Each has a matching entry in cimo's `TODO.md`, which asks for a note here when it lands.

- **Float `math.sumprod` falls back:** Python returns the int 0 when it's empty, so its type depends on
  the data. Same for float `math.prod` without `start=1.0`, where the fix is in the user's hands.
- **Math functions call the host libm through a table of addresses** (`_dispatch._libm_table`,
  `SJ_LIBM1` in the prelude), passed to each kernel that uses them, because cimo resolves `sin`
  etc. to its bundled libm. When cimo can link the host libm for math calls: call them directly.

- **Code shaped around cimo's missing CSE:** wraparound indices hoisted to the top of the loop
  body (`hoist_wraps`), `//` and `%` computed as one division and one remainder into
  temporaries. Both help any compiler; revisit what else can go once cimo merges repeated
  expressions (cimo `TODO.md`: "Slower than gcc -O3 on scramjit's kernels").

- **The LLM demo moves compiled code by listing what changes in `CIMO_CACHE_DIR`** (`docs/demos/llm/demo_py.py`: `precompile`), which depends on
  cimo's cache layout. The same listing keeps compiles between visits. When cimo has public cache export and import (its `TODO.md`), use that for both.
  The demo's tabs also read the private `fn._cache`, for the C and the inferred types.

- **The demo's WebAssembly tab calls `cimo.wasm.build`** (`docs/lab/lab.py`: `wasm_text`) for the
  module's bytes, which isn't cimo's documented API. When cimo exposes a compiled module's
  WebAssembly (or a disassembly), use that.

## Line counts, what's left

Spills per line, once cimo can report them.

## Next

1. **SIMD through cimo, what's left:** what vectorizes was checked loop by loop (cimo's `vectorize.plan`, wrapped):
   `saxpy`, scaling, `a + b + c`, stencils, int `+`, `^`, `&`, 2-D rows, offset bases and float32 do, and int sums now that a
   variable's numpy flag is set after the loop, not in it (`sinkable_flags`; int32 sum about 2x, int64 1.2x, memory-bound).
   Not yet: anything with a condition (`relu`, `clip`, `max`), which needs cimo to vectorize them (its `TODO.md`); `math.sqrt`
   and other functions that check their domain on every element, which only `parity=False` lets vectorize (a check after the
   loop would need its stores undone); float sums and `dot`, kept scalar on purpose, since their order is Python's. Re-run the
   probe on `bench/` and the examples when cimo lands conditions.
2. **Narrow dtypes, what's left:** arithmetic, `**`, `sum()`, the exact numpy functions and most reductions compile (see "Narrow dtypes" in `numpy.md`).
   Not yet: `np.sin`, `np.exp` and the like on float32 (numpy's own SIMD versions: check them against numpy per CPU first), float functions of narrow ints (float16 results),
   `.std()` and `.var()` on float32 (numpy computes them in float32), `.sum()` of signed ints on wasm32 (numpy totals in int32 there), and `np.int8(x)` of a float.
   A literal becoming narrow must fit exactly: not `s = 0.1` on float32, nor a parameter, and such a variable can't meet a Python number afterwards, since the type depends on its flag.
   Speed, all in cimo's `TODO.md`: byte and short lanes (`uint8` brighten 0.24 ns per element against int32's 0.12) and the flag written in the loop.
   `bench/llm.py` still runs on float64 weights, from before float32 compiled: measure float32 (half the memory the matrix products read), with `bench/gcc.py` on `float` alongside.
3. **64-bit ints, what's left:** hashes masked with `& 0xFFFFFFFFFFFFFFFF` compile (`examples/hash64.py`). `np.uint64`
   elements compile with numpy's rules (`tests/test_uint64.py`). Not yet: unsigned 64-bit Python ints in tuples, dicts, as `/`, passed to
   compiled functions or `math`, compared with float32, as slice bounds, or choosing a row (`m[int(h)]` of a 2-D array).
   `examples/xorshift.py` takes 3.9 ms per million numbers against 2.05 for the same loop written in C on `uint32_t` through cimo:
   the masks sit on the chain each number waits on, and storing into `uint32` checks the range.
4. **Bounds-check hoisting, what's left:** versioning covers `arr[i + c]` and, in innermost loops, `arr[base + i]`
   with `base` built from Python ints that don't change in the loop, also with a positive literal step and a start
   checked at run time. Not yet: negative steps, numpy int bases, `arr[i // 2]` and `arr[2 * i]`, and outer loops,
   where a second copy of everything inside costs more than it wins.
5. **Store-to-load forwarding, what's left:** `y[i - 1]` to `y[i - 4]` come from temporaries when `y[i] = ...` is y's
   only store in the loop and runs every iteration, with other arrays stored to behind a test that they don't share
   y's memory. Not yet: 2-D rows (`m[r, j - 1]`), strided views among the other stores, and stores inside an `if`.
6. **Loop lifting, what's left:** a loop stays in Python if a variable read afterwards may be unbound
   (`for i ...` then reading `i`, with nothing assigned first), inside `try` or `with`, or containing `return`.
   "Partly compiled" is per function, not per signature. Tracebacks show one extra frame for a loop that raises.
   The first two are hard for one reason: compiled code has static types. A name the loop may leave unbound has no
   value of any type to come back as, and a `return` inside means returning either its value or the loop's outputs.
7. **numpy calls in compiled functions:** `x * 2` and other whole-array arithmetic still run as Python.
   No cimo work is required for any step; its SIMD only adds speed. In this order:
   - **numpy functions on single numbers: built.** See `numpy.md`. Left in Python: `np.tanh` and `np.power`, which aren't libm's, and
     combinations whose result depends on the platform or numpy version. Bit for bit checked on arm64, wasm and x86-64 with AVX2 (Windows, emulated). Check AVX-512, where numpy
     may use its own `exp` and `sin`. Not yet: `np.mod`, `np.floor_divide`, `np.degrees`, `np.radians`.
   - **Views: built,** as arguments and results too, with `.T`, slice assignment and tuples of views. Not yet: `a, b = m` (unpacking rows), `x[1:] += y`,
     a two-dimensional-or-more array assigned into three dimensions or more.
   - **Reductions: built,** with `std`, `var`, integer `mean` and `argmax` on wasm32, for one number from `float64` or `int64`. `axis=` that returns an array, `keepdims=True`, `dtype=`, `out=`, `where=`, `initial=` stay Python.
     Not yet: views with gaps in more than one dimension (numpy's order depends on strides), other dtypes.
   - **Arrays made inside a function: built.** See "Making arrays" in `numpy.md`. Not yet: `x.reshape()`, `np.eye`, `.size` or `.shape` directly on a call (`np.ones(n).size`), a conditional expression of arrays (`y = a if c else b`),
     five dimensions or more, `uint64` and `bool` dtypes, an array made in a comprehension. Returning one costs a copy (3.1 µs for 10 elements): worth removing if a Gymnasium-style `reset` or `step` shows it.
   - **Not now, whole-array arithmetic:** `x * 2`, `a + b` need broadcasting and dtype rules on top, which is rewriting numpy.
     They stay in Python, with the loops around them lifted.
   - **Not `np.dot`, and no MKL or BLAS:** they would break "standard library plus cimo", and their results depend on the library, CPU and thread count.
     `A @ B` and `np.linalg` already run at full speed in Python. An opt-in through cimo's `link=` for BLAS inside a hot loop is possible
     later; untried.
8. **Python data structures, nested, when statically sized and statically typed, after the loop work above:** so
   that containers don't hide speedups we can still get from loops, each step must leave `bench/` unchanged.
   Dicts, lists, tuples, classes and enums that don't resize at runtime and whose element types don't change. See "Containers" in
   `requirements.md`. Today only flat tuples of numbers and flat local lists compile. In this order, simplest
   and most useful first:
   - **Module-level constants: built.** Numbers, tuples, arrays and dicts are hidden arguments, read at each call, at about 0.2 µs each. Not as compile-time literals: only `x[i % K]` would gain, 1.25 to 1.06
     ns, against recompiling on every change. Not yet: lists at module level, and a global that is a list of numbers.
   - **Enums: built.** See "Enums" in `data-structures.md`. Not yet: `.name`, `Cell(1)`, iterating over the class, storing a member into an array, `Flag` and string enums.
   - **Nested tuples: built,** as locals, arguments, results and globals, and loops over tuples. Not yet: a tuple holding arrays as a local or an argument, a tuple indexed with a variable, a comprehension variable that is a tuple, and comparing tuples.
   - **Lists as arguments, returned, and at module level: built, from 0.0.4** (see `data-structures.md`). Lists of lists passed in
     too: one array, rows as views, so a row is a reference as in Python. Lists of lists made in the
     function too, from `[[c] * w for _ in range(h)]` and its comprehension form (`[[c] * w] * h`, one row repeated, stays Python).
     Returned too. Not yet: making more than two levels, returning a row, numpy scalars as elements, and a call plan for list
     arguments (each call takes the general path).
   - **Instances: attributes and method calls built, from 0.0.4** (see "Instances" in `data-structures.md`), passed in, as `self` and at module level. A method is a callee with the
     attributes it uses as self, its writes copied back before its status is checked; `p__a = p__a` after the statement shows the write to every analysis. Lists of instances (a list of
     numbers per attribute), passing an instance or an element on, methods reading globals, parameters never used, objects in attributes (`p.pos.x`), instances made in the
     function (`v = Vec(x, y)`, `_init_recipe`), `q = p`, returning an instance, an element or an inner object, `isinstance`, keyword-only parameters, a writing call in a test,
     `enumerate()` in a comprehension, and one object under several names (the alias pattern, "Object model" above), built too. Not yet: an object in an attribute of a list's elements or of an instance made here, passing an inner object on (`f(b.pos)`), a
     function making instances called from compiled code (its class's type is made for its own tree only), a writing method in a comprehension, `hasattr`/`getattr`/`vars`. A method
     writing into an inner object marks all of that object's numbers written (they go back unchanged). A list's copy is about 0.15 µs per instance and attribute, checked and
     built in Python twice (`_key`, `_run`): once would halve it. Static preflight finds a parameter's class by its methods, when one class in the module has them all, and
     follows made instances and inner objects from the class statements. A call plan for an array among the attributes: about 12 µs per call now, against 2 for numbers and 2.8 for the array passed separately.
   - **Dicts, what's left:** a dict passed in compiles (see "Containers"). Not yet: nested dicts, a dict built in the function, returning
     one, `len(p)` and `'a' in p`, which are known from the keys. A function that could add a key still runs as Python.

   To settle first, all of it needing differential tests: what happens when a value breaks the rule at runtime
   (an `int` slot given a float); two names for one object (`[[0] * 3] * 2`, one list passed twice),
   since writes must show through both; the cost of checking and copying a whole structure on every call; and
   results that must come back as real Python objects.
9. **Call boundary cost:** about 1.4 µs with numbers (was 3.2) and 2.4 µs with an array (was 6.6), through
   a function generated per signature (`_make_plan`). For numbers: ctypes about 450 ns (the floor), cimo's
   `Kernel.__call__` about 450 (see cimo's `TODO.md`), scramjit about 500. In Pyodide about 13 µs, not yet
   broken down. Break-even is 30 to 50 loop iterations per call. A Gymnasium-style `step` is below it, with no loop
   at all: CartPole from a Python loop does 396k steps/s compiled against 698k plain, and 53M with the loop compiled too, which calls between compiled functions now allow without pasting. Rejected: caching an array's address (saves
   200 ns, but `resize(refcheck=False)` would corrupt memory silently); `__call__` by argument count (100 to
   150 ns, not worth its keyword edge cases).
10. **Variables that change type, what's left:** a name used for separate things compiles (each web is its own
   variable), and so does an int that becomes a float, where the int is a literal up to 2^53 or a parameter, and it's only
   returned or combined with a float. Not yet: other uses of such a variable (`-s`, `s + 1`, `t = s`, `a[s]`, `if s:`), and
   `s = n` from an int variable, each of which keeps the function in Python.
11. **Rest of M3:** run the M0 kernels in Pyodide as a test, not only through the site; preflight's
   per-loop report once cimo exposes one (cimo's `TODO.md`), and a `disk` label in the compile column,
   which needs cimo to say whether a compile hit its cache.
12. **`n ** k` with a negative `k` and `x ** 0.5` with a negative `x`, what's left:** a function that writes nothing
   reruns the call as Python there. One that writes still raises: resuming in Python alongside loop lifting would remove
   that difference too.
13. **A NaN's sign differs from CPython's on Windows** (arm64, and x86-64 under emulation). `inf % 1.0`, `inf // 1.0`
   and `math.atan2(0.0, nan)` give `7ff8...` where CPython gives `fff8...`, and `nan * -1` gives `fff8...`
   where it gives `7ff8...`; `tests/test_m1.py` and `test_m2.py` fail there on these, 7 in all. The values are
   all NaN, and only `math.copysign` or the raw bytes can tell. Likely fix: call the CRT's `fmod` through the
   libm table, as for `sin`, and make the `* -1` negation depend on the platform. Nothing runs scramjit's tests
   on Windows automatically yet.
14. **Test Ctrl-C on macOS.** `tests/test_interrupt.py` has run on Linux and Windows arm64, not macOS. It
   should work, since it uses `signal.set_wakeup_fd` and sockets, but nothing has checked it.
15. **`parity=False` asks cimo for FMA, and it never happens.** cimo contracts `a * b + c` within one C
    expression, and scramjit gives each operation its own temporary. Nothing is lost on reductions:
    FMA makes them slower on arm64 (a `dot` through cimo: 2.15 ns per element with, 1.81 without;
    `bench/llm.py` through `gcc -O3`: 63 tok/s with, 82 without), probably the latency of the
    instruction the sum's next add waits on (not measured). Measure non-reductions (stencils) before making it fire or removing it.
16. **Calls between compiled functions, what's left:** recursion, defaults other than numbers left out by a compiled caller, keyword-only parameters, dicts and lists as arguments, a function passed as an argument, `module.f()`, and a line profile of what's called.
    **Calling C, what's left:** modules with state, which need calling across modules (cimo: a call by address crashes once the callee uses `malloc`), `include_dirs=` and `link=`, returning a pointer or `uint64_t`, a C function's module compiled with `fma=` on alongside exact math in one file (needs `FP_CONTRACT` in cimo).
17. **Compile time on large functions, what's left:** versioning's copies are capped (`VERSION_TOTAL`, `VERSION_MIN` in
    `_compiler.py`), most deeply nested loops first: a generated 60-loop function 14.7 to 5.1 s, `bench/llm.py` unchanged.
    The cost was mostly cimo's, now much less (cimo 2026-11-05): this shape of function, 60 loops all versioned (4,150 lines
    of C), 32.6 to 7.9 s, 120 small loops 21.4 to 2.7 s; what's left grows about 3x per doubling. Raise the cap to match.
    Depth is a guess at which loops run most: a profile (`counts=True`) would know.
18. **scramjit against Numba and Cython: measure, then close the gaps.** The README compares with
    CPython and hand-written C; the question everyone will ask is how it compares with what they use
    today. `bench/compare.py`, run on x86-64 Linux and arm64 Linux (an investor's or a reviewer's machine
    is x86), reported as one table in the README and the site:
    - Kernels: the existing `bench/kernels.py` set plus three where Numba is known to be strong (a
      float64 dot/axpy that it vectorizes, a stencil, a histogram) and one where it's weak (a loop with
      Python-level dict or tuple work).
    - Columns: CPython, scramjit, Numba (`@njit`, `fastmath=False` to match parity, and once with
      `fastmath=True`), Cython (typed, `boundscheck=False`), hand-written C via gcc -O3. Steady-state
      time, first-call time including compile, install size on disk (Numba + llvmlite about 100 MB;
      Cython needs a C compiler; scramjit about 1 MB), and whether it runs in Pyodide (only scramjit).
    - Report losses as they are; the "runs everywhere" claim doesn't need wins on every row.
    - Then improve, biggest gap first, each with a measurement before and after. Expected gaps and where
      they land: FP reductions that Numba vectorizes with `fastmath` (an opt-in `reassociate=True` that maps
      to cimo's `#pragma omp simd reduction`), 128-bit lanes against Numba's AVX2 on x86 (cimo's TODO: AVX2
      behind a check at load), bounds checks that Numba drops by proving ranges (#3 above), call overhead
      (1.4 µs against Numba's about 0.3 µs; a typed fast path in `_dispatch`).
    - Keep the script runnable by anyone: `pip install numba cython` are dev extras, never dependencies.
19. **Compile time.** A first call costs seconds, not the 0.1 s the README's `ema` example suggests: 0.53 s
    on Gray-Scott, 0.76 s native and 1.75 s in Pyodide for the LLM demo's forward pass, 3 s for a 200-line
    generated loop and up to 15 s with bounds-check versioning (#16). Every candidate in a generated-code
    workflow compiles once, so this decides what's practical. An optimization pass on the compiler
    itself, measured with `cProfile` on `bench/` and `bench/llm.py` in both runtimes, before and after:
    - Split the time between scramjit (parse, type inference, C generation) and cimo (preprocessor,
      front end, middle end, backend, encoder, link). Record the split here; cimo's share goes to its
      `TODO.md` ("Compile time").
    - In scramjit: generate less C (smaller versioned copies, no dead helpers, one prelude shared across
      compiles), and cache type inference per signature.
    - Across the boundary: cimo's compile of a kernel that reuses the prelude and libm table from a
      previous compile shouldn't redo their work; ask for a per-process cache of parsed headers and the
      prelude, or pass them pre-parsed.
    - Target: under 0.3 s warm for a 100-line function natively, under 1 s in Pyodide.
    - **An edit above a function recompiles it.** Line numbers and the file's path are in the C, so cimo's key changes: Gray-Scott first call 86 ms warm, 562 ms after
      adding one line above it, or renaming the file; a comment that keeps the lines stays warm. **Done here (2026-11-05):** the C's lines count from the function's
      `def` and `#line` names `sj` (or `sj<k>`, the k-th function called) instead of the file, put back for tracebacks (`_failed`), preflight's notes (`py_line`) and
      cimo's messages (`cimo_diagnostic`); moved 3 lines down and renamed, Gray-Scott's first call stayed warm. `tests/test_line_independence.py`.
      cimo's half is done too (2026-11-05: its key leaves out line markers, blank lines and indentation), so `#line` could name the real file again; `__LINE__`
      and `__FILE__` still go into the key wherever they're used.
    - **A warm first call (2026-11-05):** Gray-Scott about 47 ms: 13 generating C, 30 preprocessing it (14 of that the prelude), about 1 loading; `small` about 45, nearly all preprocessing.
      A cold one is 96% cimo (496 ms, 13 generating). A cache keyed on the Python would save only warm time, and needs cimo's public cache export and import to skip the preprocessing.
20. **Split `_compiler.py` into a package.** 4.8k lines holding every lowering in one file. Make
    `scramjit/_compiler/` with one module per concern, in the order the compile runs: `infer` (types),
    `index` (indexing, views, slices), `checks` (bounds checks and versioning), `calls` (`@scram` and C
    callees, inlining), `reduce` (builtins and numpy reductions), `make` (arrays and containers made
    inside a function), `emit` (C text). No behavior change: the suite and `bench/` are the gate, and
    the `docs/py/scramjit` copy is regenerated, not edited.
21. **Adversarial code reading before a public launch.** `_compiler.py` (4.8k lines) and `_dispatch.py`
    were written in days and no one has read them end to end. The parity tests check results; a read
    checks what they can't: a lowering that is right for every tested type combination and wrong for
    one the generator never produces, a bounds check hoisted when it shouldn't be, a fallback that
    silently changes semantics. Read `_compiler.py` by lowering (indexing and views first, then
    bounds-check versioning, then calls), and record what was checked and found here.
22. **Value-range narrowing:** the rest of M2's loop work, using the range facts the compiler already has. Little to win: arrays already have the user's dtype, cimo has no byte or short lanes, and scalars gain nothing, which leaves temporary arrays.
