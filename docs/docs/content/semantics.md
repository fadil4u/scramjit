# Semantics

A compiled function returns what CPython returns for the same function undecorated. Where Python and C disagree, scramjit picks a behavior explicitly and tests it against CPython: every compiled function is checked against the undecorated one, including on randomly generated programs within the subset.

| Topic | Python | C | scramjit |
| --- | --- | --- | --- |
| Integer size | unbounded | fixed | `int64` unless narrowed; overflow wraps |
| `//` and `%` | round toward negative infinity | truncate toward zero | Python's rule |
| Negative indices | count from the end | out of bounds | Python's rule |
| Bounds checks | always | never | on, raising `IndexError`; `boundscheck=False` drops them |
| Division by zero | raises `ZeroDivisionError` | `inf`, `nan` or a crash | raises `ZeroDivisionError` |
| Float evaluation order | left to right, as written | compiler may reassociate | as written: no reassociation, no FMA contraction unless opted in |
| `math` functions | the platform's C library, or CPython's own code | the C library at hand | the same code CPython runs, bit for bit, on every platform including Pyodide |

## Why integers wrap

Python integers never overflow; C integers do. Checking every operation would cost speed on every loop, so `int` is `int64` and wraps. Numeric code rarely comes near `2**63`, but if yours might, leave that function undecorated.

## Why floats stay in order

Reassociating `a + b + c` or contracting `a * b + c` into one fused multiply-add changes the last bits of the result. scramjit promises CPython's answer, so it does neither unless you opt in.
