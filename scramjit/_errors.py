class CompileError(Exception):
    """Raised under @scram(fallback=False) when a function isn't fully compiled."""


class FallbackWarning(UserWarning):
    """Some or all of a decorated function runs as Python instead of compiled code."""


class Unsupported(Exception):
    """Internal: a construct outside what scramjit compiles. Carries the Python line."""

    def __init__(self, node, reason):
        super().__init__(reason)
        self.lineno = getattr(node, 'lineno', None)
        self.node = node
        self.reason = reason
