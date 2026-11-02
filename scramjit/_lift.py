"""Loop lifting: when a function doesn't compile whole, its loops that fit are compiled on their own.

Each candidate loop becomes a function of its own, `def f(live_in): <loop>; return live_out`, and the
Python that remains calls it. That function is an ordinary scramjit Function, so it specializes on the
types it sees at run time; if it doesn't compile it runs as Python, where its inner loops are candidates
in turn. So a loop lifted wrongly can only be a loop that fails to compile.

What a compiled loop can't do is keep the enclosing frame's variables up to date while it runs. That is
only visible where the frame outlives an exception or can be inspected, so a loop inside try or with is
never a candidate, and neither is any loop of a function that shares or inspects its frame (closures,
locals(), eval, ...). del and global would need a dataflow this doesn't have, so those stay in Python too.
"""

import ast
import copy
import time
import types

_FRAME_READERS = {'locals', 'vars', 'dir', 'eval', 'exec', 'compile', 'globals', 'breakpoint', 'super'}
_FRAME_ATTRS = {'_getframe', 'currentframe', 'stack', 'f_locals', 'f_back'}
_NOT_IN_A_LOOP = (ast.Return, ast.Try, ast.With, ast.AsyncWith, ast.Import, ast.ImportFrom, ast.Match)
if hasattr(ast, 'TryStar'):
    _NOT_IN_A_LOOP += (ast.TryStar,)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_LOOPS = (ast.For, ast.While)


def can_lift(fdef, code=None):
    """Can any loop of this function run away from its frame? Without code (preflight reading source), what the
    code would say follows from the nested functions and yields looked for here."""
    if code is not None and (code.co_cellvars or code.co_freevars or code.co_flags & (0x20 | 0x80 | 0x100 | 0x200)):
        return False                     # 0x20...: generators and async
    for n in ast.walk(fdef):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.Match,
                          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and n is not fdef:
            return False
        if isinstance(n, ast.Name) and (n.id in _FRAME_READERS or _private(n.id)):
            return False
        if isinstance(n, ast.Attribute) and (n.attr in _FRAME_ATTRS or _private(n.attr)):
            return False
        if isinstance(n, _COMPREHENSIONS) and any(isinstance(x, ast.NamedExpr) for x in ast.walk(n)):
            return False
    return True


def local_names(fdef):
    """The function's local variables, from its source."""
    a = fdef.args
    names = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]}
    for s in fdef.body:
        names |= _split(s)[1]
    return names


def _private(name):
    """A name a class body would mangle: this code is compiled again, outside its class."""
    return name.startswith('__') and not name.endswith('__')


def _names(node, loads, stores):
    """The names an expression or simple statement reads and writes, without entering comprehensions'
    own variables."""
    if isinstance(node, _COMPREHENSIONS):
        inner_loads, inner_stores = set(), set()
        for c in ast.iter_child_nodes(node):
            _names(c, inner_loads, inner_stores)
        loads |= inner_loads - inner_stores
    elif isinstance(node, ast.Name):
        (loads if isinstance(node.ctx, ast.Load) else stores).add(node.id)
    else:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            loads.add(node.target.id)
        for c in ast.iter_child_nodes(node):
            _names(c, loads, stores)


def _split(node):
    loads, stores = set(), set()
    _names(node, loads, stores)
    return loads, stores


def _imported(s):
    return {(a.asname or a.name).split('.')[0] for a in s.names}


def _flow(stmts, defined, exposed=None, visit=None):
    """Walk statements in order, tracking the names definitely assigned. exposed collects names read before
    that. visit(loop, defined) is called on each loop, and says whether to walk into it. Returns the names
    definitely assigned afterwards. Anything unusual only makes this assign less, which is the safe side."""
    defined = set(defined)

    def reads(node):
        if exposed is not None:
            exposed.update(_split(node)[0] - defined)

    for s in stmts:
        if isinstance(s, _LOOPS):
            head = s.iter if isinstance(s, ast.For) else s.test
            reads(head)
            inside = defined | _split(head)[1]
            if isinstance(s, ast.For):
                loads = _split(s.target)
                if exposed is not None:
                    exposed.update(loads[0] - inside)
                inside |= loads[1]
            if visit is None or visit(s, defined):
                _flow(s.body, inside, exposed, visit)
            _flow(s.orelse, defined, exposed, visit)
        elif isinstance(s, ast.If):
            reads(s.test)
            base = defined | _split(s.test)[1]
            defined = _flow(s.body, base, exposed, visit) & _flow(s.orelse, base, exposed, visit)
        elif isinstance(s, (ast.Try, ast.With, ast.AsyncWith)) or (hasattr(ast, 'TryStar') and isinstance(s, ast.TryStar)):
            if exposed is not None:      # only reached for what a candidate excludes: read everything
                exposed.update(_split(s)[0])
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            defined |= _imported(s)
        else:
            reads(s)
            defined |= _split(s)[1]
    return defined


class Candidate:
    __slots__ = ('loop', 'params', 'out', 'slot')

    def __init__(self, loop, params, out):
        self.loop, self.params, self.out = loop, params, out


def candidates(fdef, local_names, root=None):
    """The outermost loops that can be lifted, walking into those that can't. root is the loop a lifted
    function exists for: never itself a candidate, or lifting it again would never end."""
    args = fdef.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    params |= {a.arg for a in (args.vararg, args.kwarg) if a}
    found = []

    def visit(loop, defined):
        if loop is root:
            return True
        c = _candidate(loop, defined, fdef, local_names)
        if c is None:
            return True
        found.append(c)
        return False

    _flow(fdef.body, params, None, visit)
    return found


def _candidate(loop, defined, fdef, local_names):
    if any(isinstance(n, _NOT_IN_A_LOOP) for n in ast.walk(loop)):
        return None
    exposed = set()
    _flow([loop], set(), exposed)
    written = _split(loop)[1] & local_names
    out = sorted(written & _live_after(fdef, loop))
    params = sorted((exposed & local_names) | set(out))
    if not set(params) <= defined:       # the call would raise where Python might not
        return None
    return Candidate(loop, params, out)


def _path(stmts, loop, owner=None):
    """From loop outward, each block on the way to it: (statements, loop's or its ancestor's index, the
    statement owning that block)."""
    for i, st in enumerate(stmts):
        if st is loop:
            return [(stmts, i, owner)]
        for field in ('body', 'orelse', 'finalbody'):
            inner = getattr(st, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                found = _path(inner, loop, st)
                if found:
                    return found + [(stmts, i, owner)]
    return None


def _live_after(fdef, loop):
    """The names that may be read after loop before being assigned: reusing i in the next loop is not
    reading it. A surrounding loop can run everything again, and its break and continue skip assignments,
    so nothing counts as assigned past one."""
    exposed, defined = set(), set()
    for stmts, i, owner in _path(fdef.body, loop):
        defined = _flow(stmts[i + 1:], defined, exposed)
        if isinstance(owner, _LOOPS):
            defined = set()
            _flow([owner], set(), exposed)
    return exposed


def _load(name):
    return ast.Name(name, ast.Load())


def _store(name):
    return ast.Name(name, ast.Store())


def loop_function(c, name):
    """def name(params): <loop>; return out"""
    body = [c.loop]
    if c.out:
        value = _load(c.out[0]) if len(c.out) == 1 else ast.Tuple([_load(o) for o in c.out], ast.Load())
        body.append(ast.Return(value))
    fdef = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[ast.arg(p) for p in c.params], kwonlyargs=[],
                                                         kw_defaults=[], defaults=[]),
                           body=body, decorator_list=[], returns=None, type_comment=None)
    if 'type_params' in ast.FunctionDef._fields:
        fdef.type_params = []
    ast.copy_location(fdef, c.loop)
    fdef.end_lineno = c.loop.end_lineno
    return ast.fix_missing_locations(fdef)


def _replace(node, loop, new):
    for field in ('body', 'orelse'):
        stmts = getattr(node, field, None)
        if isinstance(stmts, list):
            for i, s in enumerate(stmts):
                if s is loop:
                    stmts[i] = new
                    return True
                if isinstance(s, ast.stmt) and _replace(s, loop, new):
                    return True
    return False


def _call(c, slot):
    call = ast.Call(_load(slot), [_load(p) for p in c.params], [])
    if not c.out:
        s = ast.Expr(call)
    else:
        target = _store(c.out[0]) if len(c.out) == 1 else ast.Tuple([_store(o) for o in c.out], ast.Store())
        s = ast.Assign([target], call)
    ast.copy_location(s, c.loop)
    return ast.fix_missing_locations(s)


def build(fdef, free, like, filename, first):
    """fdef as a real function with like's globals and defaults, the names in free as its closure. A closure
    because lifted loops mustn't appear in the user's globals, and because a cell can be swapped cheaply."""
    inner = copy.deepcopy(fdef)
    inner.name = '__sj_inner'            # under its own name, a recursive call would find this copy
    inner.decorator_list = []
    inner.returns = None
    a = inner.args
    a.defaults, a.kw_defaults = [], [None] * len(a.kwonlyargs)
    for p in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
        p.annotation = None
    ast.increment_lineno(inner, first - 1)
    names = list(free)
    outer = ast.FunctionDef(name='__sj_outer', args=ast.arguments(posonlyargs=[], args=[ast.arg(n) for n in names], kwonlyargs=[],
                                                                  kw_defaults=[], defaults=[]),
                            body=[inner, ast.Return(_load(inner.name))], decorator_list=[], returns=None, type_comment=None)
    if 'type_params' in ast.FunctionDef._fields:
        outer.type_params = []
    ast.copy_location(outer, inner)
    ast.fix_missing_locations(outer)
    flags = like.__code__.co_flags & 0x1FF000 if like is not None else 0       # the __future__ flags
    code = compile(ast.Module([outer], []), filename, 'exec', flags, dont_inherit=True)
    space = {}
    exec(code, like.__globals__, space)  # noqa: S102
    fn = space['__sj_outer'](*[free[n] for n in names])
    fn.__name__ = fdef.name
    code = fn.__code__.replace(co_name=fdef.name)
    fn.__code__ = code.replace(co_qualname=like.__qualname__) if hasattr(code, 'co_qualname') else code
    fn.__defaults__, fn.__kwdefaults__ = like.__defaults__, like.__kwdefaults__
    fn.__module__, fn.__qualname__, fn.__doc__ = like.__module__, like.__qualname__, like.__doc__
    return fn


# Past this, a native call has done more work than Python could in the same time: no need to compare.
_CLEARLY_NATIVE = 20e-6
_SKIPPED = 2           # the first native call compiles, the second makes the call plan
_SAMPLES = 8
_RECHECK_EVERY = 1024
_RECHECKS = 3          # in a row: one slow call may be a pause, and a long run is sure to have one


class Tier:
    """Calls a lifted loop. A call into native code costs more than a very short loop does in Python, so a
    loop whose calls are all tiny is timed both ways and the faster kept, comparing minimums since pauses
    only add time. Once native wins, the closure's cell points straight at it and this object is out of the
    way. A loop kept in Python is looked at again from time to time, since loops grow."""

    __slots__ = ('native', 'python', 'cell', 'n', 'native_best', 'python_best', 'left', 'slow')

    def __init__(self, native, python):
        self.native, self.python = native, python
        self.cell = None
        self.n = 0
        self.native_best = self.python_best = float('inf')
        self.left = 0          # settled on Python: calls until the next look
        self.slow = 0

    def __call__(self, *args):
        if self.left:
            self.left -= 1
            if self.left:
                return self.python(*args)
            t = time.perf_counter()
            try:
                return self.python(*args)
            finally:
                if time.perf_counter() - t > _CLEARLY_NATIVE:
                    self.slow += 1
                    self.left = 1
                    if self.slow == _RECHECKS:
                        self.cell.cell_contents = self.native
                else:
                    self.slow = 0
                    self.left = _RECHECK_EVERY
        n = self.n
        self.n = n + 1
        native = n < _SKIPPED + _SAMPLES
        t = time.perf_counter()
        try:
            return (self.native if native else self.python)(*args)
        finally:
            spent = time.perf_counter() - t
            if native:
                if n >= _SKIPPED:
                    self.native_best = min(self.native_best, spent)
                if n == _SKIPPED + _SAMPLES - 1 and self.native_best > _CLEARLY_NATIVE:
                    self.cell.cell_contents = self.native
            else:
                self.python_best = min(self.python_best, spent)
                if n == _SKIPPED + 2 * _SAMPLES - 1:
                    if self.python_best < self.native_best:
                        self.left = _RECHECK_EVERY
                    else:
                        self.cell.cell_contents = self.native


def lift(fdef, fn, filename, first, make_native, root=None):
    """fn with its liftable loops replaced by calls. make_native(python_fn, tree, loop) gives the Function
    for a lifted loop. Returns (the new function, the natives), or (fn, []) if no loop can be lifted."""
    if not can_lift(fdef, fn.__code__):
        return fn, []
    local_names = set(fn.__code__.co_varnames)
    tree = copy.deepcopy(fdef)
    root_copy = None
    if root is not None:
        root_copy = next(n for n in tree.body if isinstance(n, _LOOPS))
    found = candidates(tree, local_names, root_copy)
    if not found:
        return fn, []
    free, natives, tiers = {}, [], {}
    for i, c in enumerate(found):
        slot = f'__sj_loop_{i}'
        loop_def = loop_function(c, fdef.name)
        python = build(loop_def, {}, fn, filename, first)
        native = make_native(python, loop_def, c.loop)
        natives.append(native)
        tier = Tier(native, python)      # plain, so a loop kept in Python costs little more than it did
        tiers[slot] = tier
        free[slot] = tier
        if not _replace(tree, c.loop, _call(c, slot)):
            raise AssertionError('a lifted loop was not found in its function')
    out = build(tree, free, fn, filename, first)
    for slot, cell in zip(out.__code__.co_freevars, out.__closure__):
        tiers[slot].cell = cell
    return out, natives
