"""A size from the user must fail like Python, never wrap to a small allocation that gets overrun."""

import pytest

from scramjit import scram


def repeated(n):
    a = [1.5] * n
    return len(a)


def repeated_pair(n):
    a = [1.5, 2.5] * n
    return len(a)


def comprehension(n):
    a = [i * 0.5 for i in range(n)]
    return len(a)


@pytest.mark.parametrize('n', [2**61, 2**61 + 1, 2**62, 2**63 - 1, 2**60 + 1])
def test_a_size_too_big_is_a_memory_error_not_a_crash(n):
    """The size in bytes used to wrap to something small: malloc succeeded and the fill ran off the end."""
    for f in (repeated, repeated_pair, comprehension):
        with pytest.raises(MemoryError):
            scram(fallback=False, cache=False)(f)(n)


@pytest.mark.parametrize('n', [-1, 0, 1, 3])
def test_small_sizes_still_work(n):
    for f in (repeated, repeated_pair, comprehension):
        assert scram(fallback=False, cache=False)(f)(n) == f(n)
