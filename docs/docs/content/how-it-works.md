# How it works

```text
Python source → ast → type inference → typed IR → lowering → C (#line) → cimo → machine code or WASM
```

1. **Parse.** The function's source is read and parsed with the standard `ast` module.
2. **Infer types** over the whole function, seeded by the argument types at the call and by annotations.
3. **Typed IR.** Every value has a concrete C type.
4. **Lower.** Builtins become loops, containers become arrays and structs, and types are narrowed where that is proven safe.
5. **Emit C** with `#line` directives pointing back at the Python file.
6. **cimo** compiles the C to machine code in the process (native) or to WebAssembly (Pyodide).

## Why generate C

Generating C rather than cimo's internal IR keeps the output readable and debuggable, and reuses cimo's tested frontend and loop optimizer. scramjit aims to emit the kind of C cimo optimizes best: canonical counted loops, affine indexing, and `restrict` where no aliasing is proven.

## What scramjit needs from cimo

scramjit relies on cimo for:

- compiling a C string in memory into callables, with no files involved;
- `#line` directives honored in diagnostics;
- a calling convention for numpy buffers and scalars;
- a cache key for the compiled code.

## Performance

scramjit emits scalar C, and cimo compiles it with its loop optimizer. That makes loops that don't vectorize, the ones scramjit is for, run at C speed. On simple loops that vectorize well, LLVM-based JITs can be faster where they run.
