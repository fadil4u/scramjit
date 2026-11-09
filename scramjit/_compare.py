"""Comparisons of composite values, as Python's: tuples, lists held by reference (_glist), dicts and instances, sets,
nested to any depth, with ==, !=, <, <=, > and >=, and the equality `in` and sorting rely on.

Python compares two sequences by finding the first place where the elements aren't equal (`==`, after an identity
check): if there is one, those two elements decide (by the operator); if not, the lengths do. `==` between sequences of
different lengths is false at once. A list, dict or set is one object, so two references to the same one are equal
(its elements are the very objects they are), as Python's identity check finds. Numbers are values here, not objects:
a NaN is never equal to itself, where Python finds the very same NaN object in two places equal (see gotchas.md).

Each pair of list types gets a C function for each operator (lists nest, so the code loops); tuples are inlined."""

import ast
import dataclasses

from . import _types as T

OPS = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>='}
COMPOSITE = ('tuple', 'glist', 'record', 'gset')
SAME = {'==': 1, '<=': 1, '>=': 1, '<': 0, '>': 0}       # an object against itself


def eq_kind(cls):
    """How instances of cls compare with ==: 'id' (object's own), 'fields' (a dataclass's generated __eq__: its
    compared fields, as a tuple, with an instance of the same class), or None (a method of the user's)."""
    if cls.__eq__ is object.__eq__:
        return 'id'
    if dataclasses.is_dataclass(cls) and cls.__dataclass_params__.eq and '__eq__' in cls.__dict__ \
            and getattr(cls.__eq__, '__code__', None) is not None and cls.__eq__.__code__.co_filename.startswith('<'):
        return 'fields'
    return None


def orders(cls):
    """A dataclass(order=True), its <, <=, > and >= generated: they compare its fields as tuples."""
    return dataclasses.is_dataclass(cls) and cls.__dataclass_params__.order and all(
        getattr(getattr(cls, m, None), '__code__', None) is not None and getattr(cls, m).__code__.co_filename.startswith('<')
        for m in ('__lt__', '__le__', '__gt__', '__ge__'))


def compared_fields(t):
    """A dataclass record's compared fields, in their order: (index in the record, type)."""
    keys = [k for k, _ in t.elem]
    out = []
    for f in dataclasses.fields(t.cls):
        if f.compare:
            if f.name not in keys:
                return None
            k = keys.index(f.name)
            out.append((k, t.elem[k][1]))
    return out


class Compare:
    # ---- types

    def cmp_is(self, e):
        """A comparison (==, !=, <, <=, >, >=, chained) with a tuple, list, dict, instance or set on either side."""
        if not isinstance(e, ast.Compare) or not all(type(o) in OPS for o in e.ops):
            return False
        for x in [e.left] + e.comparators:
            if isinstance(x, (ast.List, ast.ListComp, ast.Tuple, ast.Dict, ast.Set, ast.SetComp)):
                return True
        return any(self.cmp_type_of(x).kind in COMPOSITE for x in [e.left] + e.comparators)

    def cmp_type_of(self, x):
        return self.gl_value_type(x) if not isinstance(x, ast.Tuple) else self.typeof(x)

    def cmp_types(self, e):
        """The operands' types; an empty list literal takes its neighbour's (`xs == []`: it holds nothing, so any)."""
        xs = [e.left] + e.comparators
        ts = [None if isinstance(x, ast.List) and not x.elts else self.cmp_type_of(x) for x in xs]
        for i, t in enumerate(ts):
            if t is None:
                near = [u for u in (ts[i - 1] if i else None, ts[i + 1] if i + 1 < len(ts) else None) if u is not None and u.kind == 'glist']
                ts[i] = near[0] if near else T.glist_of(T.INT)
        return ts

    def cmp_type(self, e):
        if any(self.is_list_ctor(x) or isinstance(x, ast.Dict) for x in [e.left] + e.comparators):
            self.makes = True                            # a list or dict made there, on the heap
        ts = self.cmp_types(e)
        for op, a, b in zip(e.ops, ts, ts[1:]):
            self.cmp_check(OPS[type(op)], a, b, e)
        return T.BOOL

    def cmp_check(self, op, a, b, node):
        """Fails unless a op b compiles (and is what Python computes)."""
        if T.PENDING in (a, b) or (a.kind == 'glist' and a.elem is T.PENDING) or (b.kind == 'glist' and b.elem is T.PENDING):
            from ._glist import Pending
            raise Pending()
        if a.is_number and b.is_number:
            if T.is_narrow_dyn(a) or T.is_narrow_dyn(b) or 'dyn' in (a.np, b.np):
                self.fail(node, f'comparing {a.name} with {b.name} inside a tuple or a list')
            return
        if 'array' in (a.kind, b.kind) or 'list' in (a.kind, b.kind):
            self.fail(node, f'comparing {a.name} with {b.name}')
        if a.kind != b.kind:
            if op in ('==', '!='):
                return                                   # never equal
            self.fail(node, f'{op} between {a.name} and {b.name} (Python raises TypeError)')
        if a.kind == 'tuple':
            for x, y in zip(a.elem, b.elem):
                self.cmp_check('==', x, y, node)
                if op not in ('==', '!='):
                    self.cmp_check(op, x, y, node)
        elif a.kind == 'glist':
            self.cmp_check('==', a.elem, b.elem, node)
            if op not in ('==', '!='):
                self.cmp_check(op, a.elem, b.elem, node)
        elif a.kind == 'gset':
            self.cmp_check('==', a.elem, b.elem, node)
        elif a.kind == 'record':
            if a.cls is None and b.cls is None:          # dicts: equal keys, equal values
                if op not in ('==', '!='):
                    self.fail(node, f'{op} between dicts (Python raises TypeError)')
                bk = dict(b.elem)
                for k, x in a.elem:
                    if k in bk:
                        self.cmp_check('==', x, bk[k], node)
                return
            if (a.cls is None) != (b.cls is None) or a.cls is not b.cls:
                if op in ('==', '!=') and (a.cls is None or eq_kind(a.cls)) and (b.cls is None or eq_kind(b.cls)):
                    return                               # a dict and an instance, or two classes: not equal
                self.fail(node, f'{op} between {a.name} and {b.name}')
            if op in ('==', '!='):
                kind = eq_kind(a.cls)
                if kind is None:
                    self.fail(node, f'== on {a.cls.__name__}, whose __eq__ is its own')
                if kind == 'fields':
                    fs = compared_fields(a)
                    if fs is None:
                        self.fail(node, f'== on {a.cls.__name__}, with a field it does not hold')
                    for _, x in fs:
                        self.cmp_check('==', x, x, node)
                return
            if not orders(a.cls) or compared_fields(a) is None:
                self.fail(node, f'{op} on {a.cls.__name__} (only a dataclass with order=True compiles)')
            for _, x in compared_fields(a):
                self.cmp_check('==', x, x, node)
                self.cmp_check(op, x, x, node)

    # ---- values and code

    def cmp_value(self, x, t):
        """C for an operand: a number, a tree of numbers (a tuple), or a pointer (a list, a dict or instance, a set)."""
        if t.is_number:
            return self.hold(self.expr(x)[0], t)
        if t.kind == 'tuple':
            vals = self.texpr(x)
            leaves = iter(T.leaves(t))

            def held(v):
                return [held(y) if isinstance(y, list) else self.hold(y[0], next(leaves)) for y in v]
            return held(vals)
        if isinstance(x, ast.List) and not x.elts:
            return self.hold_ptr(self.gl_new(t, '0', self.line(x)))
        if t.kind == 'gset':
            return self.gs_ptr(x)
        return self.hold_ptr(self.gl_value(x, t))

    def cmp_tree(self, t, base, at=None):
        """A tuple held as a struct of its numbers (base): the C for each, nested as the tuple is."""
        at = [0] if at is None else at
        out = []
        for x in t.elem:
            if x.kind == 'tuple':
                out.append(self.cmp_tree(x, base, at))
            else:
                out.append(f'{base}.f{at[0]}')
                at[0] += 1
        return out

    def cmp_elem(self, t, c):
        """An element (or field) of type t at C lvalue c, as cmp_expr takes it."""
        return self.cmp_tree(t, c) if t.kind == 'tuple' else c

    def cmp_expr(self, op, a, av, b, bv):
        """C for av op bv (of types a, b), side-effect free."""
        if a.is_number and b.is_number:
            return self.cmp(av, a, bv, b, op)
        if op == '!=':
            return f'(!{self.cmp_expr("==", a, av, b, bv)})'
        if a.kind != b.kind:
            return '0'
        if a.kind == 'tuple':
            if op == '==':
                if len(a.elem) != len(b.elem):
                    return '0'
                return '(' + (' && '.join(self.cmp_expr('==', x, xv, y, yv) for x, xv, y, yv in zip(a.elem, av, b.elem, bv)) or '1') + ')'
            out = '1' if {'<': len(a.elem) < len(b.elem), '<=': len(a.elem) <= len(b.elem),
                          '>': len(a.elem) > len(b.elem), '>=': len(a.elem) >= len(b.elem)}[op] else '0'
            for x, xv, y, yv in reversed(list(zip(a.elem, av, b.elem, bv))):
                out = f'(!{self.cmp_expr("==", x, xv, y, yv)} ? {self.cmp_expr(op, x, xv, y, yv)} : {out})'
            return out
        return f'{self.cmp_fn(op, a, b)}({av}, {bv})'

    def cmp_fn(self, op, a, b):
        """The C function comparing two lists, dicts, instances or sets of these types."""
        reg = self.gl_registry('cmp_fns')
        key = (op, a.name, b.name, getattr(a, 'cls', None), getattr(b, 'cls', None))
        if key in reg:
            return reg[key]
        name = f'sjK{len(reg)}'
        reg[key] = name
        same = SAME[op]
        if a.kind == 'glist':
            ea, eb = a.elem, b.elem
            x = self.cmp_elem(ea, f'(({self.gl_ctype(ea)} *)a->p)[i]')
            y = self.cmp_elem(eb, f'(({self.gl_ctype(eb)} *)b->p)[i]')
            eq = self.cmp_expr('==', ea, x, eb, y)
            if op == '==':
                body = (f'if (a == b) return 1; if (a->n != b->n) return 0; '
                        f'for (int64_t i = 0; i < a->n; i++) if (!{eq}) return 0; return 1;')
            else:
                body = (f'if (a == b) return {same}; int64_t n = a->n < b->n ? a->n : b->n; '
                        f'for (int64_t i = 0; i < n; i++) if (!{eq}) return {self.cmp_expr(op, ea, x, eb, y)}; return a->n {op} b->n;')
            sig = 'sj_glist *a, sj_glist *b'
        elif a.kind == 'gset':
            body = self.gs_set_cmp_body(op, a, b)
            sig = 'sj_gset *a, sj_gset *b'
        else:                                            # dicts, instances
            sig = 'void *a, void *b'
            sa, sb = self.rc_struct(a), self.rc_struct(b)

            def field(t, s, base, k):
                return self.cmp_elem(t.elem[k][1], f'(({s} *){base})->f{k}')
            if a.cls is None and b.cls is None:
                bk = {k: i for i, (k, _) in enumerate(b.elem)}
                if {k for k, _ in a.elem} != set(bk):
                    body = 'return a == b;'
                else:
                    conds = [self.cmp_expr('==', t, field(a, sa, 'a', i), b.elem[bk[k]][1], field(b, sb, 'b', bk[k]))
                             for i, (k, t) in enumerate(a.elem)]
                    body = f'if (a == b) return 1; return {" && ".join(conds) or "1"};'
            elif a.cls is not b.cls or (op in ('==', '!=') and eq_kind(a.cls) == 'id'):
                body = 'return a == b;' if a.cls is b.cls else 'return 0;'
            else:                                        # a dataclass: its compared fields, as a tuple
                fs = compared_fields(a)
                tt = T.tuple_of([x for _, x in fs])
                body = f'if (a == b) return {same}; return {self.cmp_expr(op, tt, [field(a, sa, "a", k) for k, _ in fs], tt, [field(b, sb, "b", k) for k, _ in fs])};'
        self.unit.code.append(f'static int {name}({sig}) {{ {body} }}')
        return name

    def cmp_compare(self, e):
        """A comparison with composite operands, chained as Python chains: each operand once, the rest skipped once one
        is false."""
        operands = [e.left] + e.comparators
        ts = self.cmp_types(e)
        a = self.cmp_value(operands[0], ts[0])
        if len(e.ops) == 1:
            b = self.cmp_value(operands[1], ts[1])
            return self.hold(self.cmp_expr(OPS[type(e.ops[0])], ts[0], a, ts[1], b), T.BOOL), '0'
        r = self.temp()
        self.emit(f'int {r} = 1;')
        closes = 0
        for i, op in enumerate(e.ops):
            if i:
                self.open(f'if ({r})')
                closes += 1
            b = self.cmp_value(operands[i + 1], ts[i + 1])
            self.emit(f'{r} = {self.cmp_expr(OPS[type(op)], ts[i], a, ts[i + 1], b)};')
            a = b
        for _ in range(closes):
            self.close()
        return r, '0'

    def cmp_eq_elem(self, xt, x, et, elem):
        """Python's `in`: the element is the very object (a list, dict or set), or ==."""
        e = self.cmp_elem(et, elem) if isinstance(elem, str) else elem     # an lvalue, or a tuple's tree already
        eq = self.cmp_expr('==', xt, x, et, e)
        if xt.kind in ('glist', 'record', 'gset') and xt.kind == et.kind:
            return f'((void *)({x}) == (void *)({elem}) || {eq})'
        return eq
