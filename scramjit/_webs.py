"""One Python name can be several variables: x = 1, used, then x = 0.5, used. A web is a set of assignments and
reads linked by "this assignment can reach that read"; two webs of a name never meet, so each can have its own
type, and they get names of their own. Reaching definitions over the structured tree, loops to a fixed point.
"""

import ast
import copy


class _GiveUp(Exception):
    """A statement this doesn't follow: nothing is renamed, and the compiler judges the function as written."""


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    out = dict(a)
    for k, v in b.items():
        out[k] = out[k] | v if k in out else v
    return out


class _Flow:
    def __init__(self):
        self.parent = {}
        self.names = {}        # element -> its name
        self.breaks, self.continues = [], []

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] is not x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def read(self, node, st):
        for d in st.get(node.id, ()):
            self.names[node] = node.id
            self.union(node, d)

    def define(self, node, st):
        self.names[node] = node.id
        self.find(node)
        st[node.id] = frozenset((node,))

    def expr(self, e, st, shadow=frozenset()):
        if isinstance(e, ast.Name):
            if e.id in shadow:
                return
            if isinstance(e.ctx, ast.Load):
                self.read(e, st)
            else:
                self.define(e, st)
        elif isinstance(e, _COMPREHENSIONS):
            bound = {n.id for g in e.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
            if any(isinstance(n, ast.NamedExpr) for n in ast.walk(e)):
                raise _GiveUp
            for c in ast.iter_child_nodes(e):
                self.expr(c, st, shadow | bound)
        elif isinstance(e, ast.NamedExpr):
            self.expr(e.value, st, shadow)
            self.expr(e.target, st, shadow)
        else:
            for c in ast.iter_child_nodes(e):
                self.expr(c, st, shadow)

    def store(self, t, st):
        if isinstance(t, ast.Name):
            self.define(t, st)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for x in t.elts:
                self.store(x, st)
        elif isinstance(t, ast.Starred):
            self.store(t.value, st)
        else:
            self.expr(t, st)

    def block(self, stmts, st):
        for s in stmts:
            st = self.stmt(s, st)
            if st is None:
                return None
        return st

    def stmt(self, s, st):
        if isinstance(s, ast.Assign):
            self.expr(s.value, st)
            for t in s.targets:
                self.store(t, st)
        elif isinstance(s, ast.AugAssign):
            self.expr(s.value, st)
            if isinstance(s.target, ast.Name):
                self.read(s.target, st)
                self.define(s.target, st)
            else:
                self.expr(s.target, st)
        elif isinstance(s, ast.If):
            self.expr(s.test, st)
            return _merge(self.block(s.body, dict(st)), self.block(s.orelse, dict(st)))
        elif isinstance(s, (ast.For, ast.While)):
            return self.loop(s, st)
        elif isinstance(s, ast.Return):
            if s.value is not None:
                self.expr(s.value, st)
            return None
        elif isinstance(s, ast.Raise):
            for c in ast.iter_child_nodes(s):
                self.expr(c, st)
            return None
        elif isinstance(s, ast.Break):
            self.breaks[-1].append(st)
            return None
        elif isinstance(s, ast.Continue):
            self.continues[-1].append(st)
            return None
        elif isinstance(s, ast.Expr):
            self.expr(s.value, st)
        elif not isinstance(s, ast.Pass):
            raise _GiveUp
        return st

    def loop(self, s, st):
        if isinstance(s, ast.For):
            self.expr(s.iter, st)
        head = dict(st)
        while True:
            self.breaks.append([])
            self.continues.append([])
            inner = dict(head)
            if isinstance(s, ast.For):
                self.store(s.target, inner)
            else:
                self.expr(s.test, inner)
                head = _merge(head, inner)
            after = self.block(s.body, inner)
            breaks = self.breaks.pop()
            for c in self.continues.pop():
                after = _merge(after, c)
            grown = _merge(head, after)
            if grown == head:
                break
            head = grown
        out = self.block(s.orelse, dict(head))
        for b in breaks:
            out = _merge(out, b)
        return out


def split(fdef, keep=()):
    """A copy of fdef with each web under a name of its own, or None if every name is one variable. Names in
    keep stay whole: something reads them at every exit, whichever assignment got there."""
    tree = copy.deepcopy(fdef)
    flow = _Flow()
    st = {}
    a = tree.args
    for p in a.posonlyargs + a.args + a.kwonlyargs:
        flow.names[p] = p.arg
        flow.find(p)
        st[p.arg] = frozenset((p,))
    try:
        flow.block(tree.body, st)
    except _GiveUp:
        return None
    webs = {}
    for node, name in flow.names.items():
        webs.setdefault(name, {}).setdefault(flow.find(node), []).append(node)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    changed = False
    for name, groups in webs.items():
        if len(groups) < 2 or name in keep:
            continue
        def first(nodes):
            params = any(isinstance(n, ast.arg) for n in nodes)
            return (not params, min(n.lineno for n in nodes), min(n.col_offset for n in nodes))
        order = sorted(groups.values(), key=first)
        k = 0
        for nodes in order[1:]:
            k += 1
            while f'{name}__v{k}' in taken:
                k += 1
            for n in nodes:
                n.id = f'{name}__v{k}'
            changed = True
    return tree if changed else None
