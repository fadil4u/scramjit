"""A module whose map isn't the builtin."""


def map(f, a):  # noqa: A001
    return [f(x) + 1 for x in a]


def rebound(a):
    return sum(map(abs, a), 0.0)
