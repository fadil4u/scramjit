# Semantics

A compiled function returns what CPython returns for the same function undecorated. Where Python and C disagree, scramjit picks a behavior explicitly and tests it against CPython: every compiled function is checked against the undecorated one, including on randomly generated programs within the subset.

> [!PLANNED]
> Designed, not built. Rows marked **open** are undecided.

| Topic | Python | C | scramjit |
| --- | --- | --- | --- |
| Integer size | unbounded | fixed | `int64` unless narrowed; overflow wraps. A checked mode is **open** |
| `//` and `%` | round toward negative infinity | truncate toward zero | Python's rule |
| Negative indices | count from the end | out of bounds | Python's rule |
| Bounds checks | always | never | **open**: on by default with an opt-out, or off by default |
| Float division by zero | raises `ZeroDivisionError` | `inf` or `nan` | **open** |
| Float evaluation order | left to right, as written | compiler may reassociate | as written: no reassociation, no FMA contraction unless opted in |

## Why integers wrap

Python integers never overflow; C integers do. Checking every operation would cost speed on every loop, so `int` is `int64` and wraps. A checked mode, which raises instead, is open. Numeric code rarely comes near `2**63`, but if yours might, keep that function in Python.

## Why floats stay in order

Reassociating `a + b + c` or contracting `a * b + c` into one fused multiply-add changes the last bits of the result. scramjit promises CPython's answer, so it does neither unless you opt in.
