# Compilation and caching

## When compilation happens

On each call the decorator reads the argument types: Python type, dtype, number of dimensions and layout. If it has no compiled version for that signature yet, it generates C specialized to it, compiles it with cimo and caches it.

## The GIL

Compiled code releases the GIL while it runs, so several Python threads calling compiled functions run in parallel. Compiling itself is Python code and holds the GIL like any other.

## Two caches

- **In memory,** for the life of the process.
- **On disk,** so a second process reuses the compiled code instead of compiling again.

The disk cache is keyed on the C generated for a function and its argument types, so it holds through edits elsewhere: from 0.0.4, through moving the function within its file and renaming the file, too. Changing the function, or one it calls, compiles again.

## First-call latency

Compiling happens in pure Python, so the first call to each specialization takes noticeably longer than the ones after it. The disk cache means that cost is paid once per machine, not once per process. Pass `cache=False` to keep compiled code in memory only.
