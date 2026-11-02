"""C inside a compiled function, for what Python doesn't do well: a hash built from shifts and overflow. Compile the C with cimo as usual
and call it. Its source goes into the same C file as the function calling it, so a small function is inlined, and each call costs no more
than the C would. Python calling the same function works too, which is how this runs undecorated."""

import time

import cimo
import numpy as np

from scramjit import scram

hashing = cimo.compile(r"""
#include <stdint.h>

int64_t mix(int64_t value) {
    uint64_t h = (uint64_t)value;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int64_t)(h >> 1);
}
""")


@scram
def distinct(values, table):
    """How many different values there are, with an open-addressed hash table."""
    size = len(table)
    table[:] = -1
    found = 0
    for i in range(len(values)):
        value = int(values[i])
        slot = hashing.mix(value) % size
        while table[slot] != -1 and table[slot] != value:
            slot = (slot + 1) % size
        if table[slot] == -1:
            table[slot] = value
            found += 1
    return found


rng = np.random.default_rng(5)
values = rng.integers(0, 40_000, 200_000)
table = np.empty(1 << 17, np.int64)


def run(f):
    t = time.perf_counter()
    r = f(values, table)
    return time.perf_counter() - t, r


t0 = time.perf_counter()
run(distinct)
compiled_in = time.perf_counter() - t0

python_s, plain = run(distinct.__wrapped__)
native_s = min(run(distinct)[0] for _ in range(5))
_, native = run(distinct)
assert native == plain == len(np.unique(values))

print(f'{len(values)} values, {native} different')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
