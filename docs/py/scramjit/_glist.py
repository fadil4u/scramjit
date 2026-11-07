"""Lists held by reference, as Python holds them: they grow (append, pop, extend, insert, clear, del), nest (lists of
lists of tuples, to any depth) and can be shared (`b = a`, `rows.append(row)`, `[[0.0] * w] * h`).

In C a list is a pointer to a header on the heap of compiled arrays (np_make's sj_block heap):

    sj_glist { void *p; int64_t n, cap; sj_block *db; }

with its elements in a block of their own (db), replaced by a bigger one when it fills up. Elements are stored by value:
numbers as their C type, tuples of numbers as a struct of their leaves (one per tuple type, sjT<k>), lists as pointers
to their headers. The heap's mark-and-sweep follows a list into its elements with a mark function per list type
(sjM<k>), so what a list holds stays alive and what nothing holds is freed. A returned list is marked to stay, read
by the dispatcher into Python lists (shared lists staying shared), and released.

A list the function never grows and that holds only numbers keeps the faster representation of _compiler's 'list'
kind; these are the lists that grow, nest, or start empty. One element type per list: a list that would mix ints and
floats, or Python and numpy numbers, stays Python.
"""

import ast

from . import _types as T
from ._runtime import LIST_INDEX_ERROR, MEMORY_ERROR

GROWERS = ('append', 'pop', 'extend', 'insert', 'clear')

SNIPPET = """
typedef struct sj_glist { void *p; int64_t n, cap; sj_block *db; } sj_glist;
static int sj_glist_reserve(sj_heap *h, sj_glist *l, int64_t item, int64_t need) {
    if (need <= l->cap) return 1;
    int64_t cap = l->cap > 0 ? l->cap : 4;
    while (cap < need) { if (cap > (int64_t)1 << 56) return 0; cap *= 2; }
    sj_block *d = sj_make(h, 1, &cap, item, 0);
    if (!d) return 0;
    if (l->n) memcpy(d + 1, l->p, (size_t)(l->n * item));
    l->p = d + 1; l->db = d; l->cap = cap;
    return 1;
}
static sj_glist *sj_glist_new(sj_heap *h, int64_t item, int64_t cap) {
    int64_t one = 1;
    sj_block *b = sj_make(h, 1, &one, sizeof(sj_glist), 0);
    if (!b) return 0;
    sj_glist *l = (sj_glist *)(b + 1);
    l->p = 0; l->n = 0; l->cap = 0; l->db = 0;
    if (cap > 0 && !sj_glist_reserve(h, l, item, cap)) return 0;
    return l;
}
"""


class Pending(Exception):
    """A read of a list whose element type isn't known yet (`out = []`, nothing appended so far): inference comes back."""


def receiver(call):
    """For x.append(v) and the like: (the list expression, the method), else None."""
    f = call.func if isinstance(call, ast.Call) else None
    if isinstance(f, ast.Attribute) and f.attr in GROWERS and not call.keywords:
        return f.value, f.attr
    return None


def grown_names(tree):
    """Names the function grows or empties (x.append(...), del x[i], x += [...]) or makes empty (x = []): these lists
    are held by reference."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and receiver(n) and isinstance(receiver(n)[0], ast.Name):
            out.add(receiver(n)[0].id)
        elif isinstance(n, ast.Delete):
            out |= {t.value.id for t in n.targets if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)}
        elif isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and isinstance(n.target, ast.Name) \
                and isinstance(n.value, (ast.List, ast.ListComp)):
            out.add(n.target.id)
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.List) and not n.value.elts:
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


def unify(a, b):
    """The element type a list holding a and b has, or None if they mix."""
    if a is T.PENDING:
        return b
    if b is T.PENDING or a is b:
        return a
    if a.kind == 'glist' and b.kind == 'glist':
        e = unify(a.elem, b.elem)
        return None if e is None else T.glist_of(e)
    if a.kind == 'array' and b.kind == 'array' and a.elem is b.elem and a.ndim == b.ndim and not T.is_pylist(a) and not T.is_pylist(b):
        return T.array(a.elem, a.writable and b.writable, a.ndim, a.strided or b.strided)
    return None


def has_arrays(t):
    """Does a value of type t hold arrays (a list of them, at any depth, or in a returned tuple)?"""
    if t.kind == 'array':
        return True
    if t.kind in ('glist', 'tuple'):
        return any(has_arrays(e) for e in ([t.elem] if t.kind == 'glist' else t.elem))
    return False


class ArrayOf:
    """What a loop over a list of arrays yields: element k, an array record (see gl_value)."""

    def __init__(self, record, t):
        self.record, self.t = record, t


def leaves(t):
    """A tuple type's numbers, in order."""
    out = []
    for e in t.elem:
        out += leaves(e) if e.kind == 'tuple' else [e]
    return out


def unify_lists(a, b):
    """Two list types one variable has had: the richer one, or None if their elements mix."""
    return unify(a, b)


class GList:
    """The Compiler's part for these lists (mixed into Compiler)."""

    def gl_any(self):
        return any(t.kind == 'glist' for t in self.env.values())

    def gl_wants(self, v):
        """Is the list v made of something only these lists hold: lists, tuples, or one of these?"""
        if isinstance(v, ast.BinOp):
            v = v.left if isinstance(v.left, ast.List) else v.right
        if isinstance(v, ast.ListComp):
            if self.is_list_ctor(v.elt) or isinstance(v.elt, ast.Tuple):
                return True
            return self.comp_types(v, any_elem=True).kind in ('glist', 'tuple')
        for x in v.elts:
            if self.is_list_ctor(x) or isinstance(x, ast.Tuple):
                return True
            if isinstance(x, ast.Name) and self.env.get(x.id) is not None and self.env[x.id].kind in ('glist', 'tuple', 'array') \
                    and not T.is_pylist(self.env[x.id]):
                return True
            if self.elementwise_of(x) or (self.is_view(x) and self.typeof(x).kind == 'array'):   # an array: a view, made, or computed
                return True
        return False

    def gl_extend_call(self, s):
        """x += y, as x.extend(y)."""
        cache = self.__dict__.setdefault('gl_extends', {})
        if id(s) not in cache:
            f = ast.copy_location(ast.Attribute(s.target, 'extend', ast.Load()), s)
            cache[id(s)] = ast.copy_location(ast.Call(f, [s.value], []), s)
        return cache[id(s)]

    # ---- types
    def gl_elem_ok(self, node, t):
        if t is T.PENDING or t.kind == 'glist':
            return
        if t.kind == 'array':
            if T.is_pylist(t):
                self.fail(node, 'a list of lists of lists put into a list (build it with [] and append)')
            return
        if t.kind == 'tuple':
            for x in leaves(t):
                self.gl_elem_ok(node, x)
            return
        if not t.is_number or t.np == 'dyn' or t is T.U64:
            self.fail(node, f'a list of {t.name}')

    def gl_is(self, node):
        """Is node a list held by reference?"""
        if isinstance(node, ast.Name):
            t = self.env.get(node.id)
            return t is not None and t.kind == 'glist'
        if isinstance(node, (ast.Subscript, ast.Call)):
            if isinstance(node, ast.Subscript) and not self.gl_is(node.value):
                return False
            if isinstance(node, ast.Call) and not (receiver(node) and receiver(node)[1] == 'pop' and self.gl_is(receiver(node)[0])):
                return False
            return self.typeof(node).kind == 'glist'
        return False

    def gl_ctor_type(self, v):
        """The type of a list made here: [..], [], [..] * n, a comprehension."""
        if isinstance(v, ast.List):
            e = T.PENDING
            for x in v.elts:
                xt = self.gl_value_type(x)
                self.gl_elem_ok(x, xt)
                u = unify(e, xt)
                if u is None:
                    self.fail(v, f'a list holding {e.name} and {xt.name} (one element type per list)')
                e = u
            return T.glist_of(e)
        if isinstance(v, ast.ListComp):
            return T.glist_of(self.comp_types(v, any_elem=True))
        lit, k = (v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left)
        if self.typeof(k).kind not in ('int', 'bool'):
            self.fail(v, 'a list repeated by a non-integer')
        return self.gl_ctor_type(lit)

    def gl_value_type(self, x):
        """What putting x in a list puts there: a list made on the spot is one of these lists."""
        if self.is_list_ctor(x):
            return self.gl_ctor_type(x)
        t = self.typeof(x)
        if t.kind == 'list':
            self.fail(x, f'{x.id if isinstance(x, ast.Name) else "a list"} put into another list (make it with [] or append to it)')
        return t

    def gl_retype(self, node, t):
        """node, a list, now has type t (its element type became known, or richer)."""
        if isinstance(node, ast.Name):
            self.env[node.id] = t
            return
        if isinstance(node, ast.Subscript):
            self.gl_retype(node.value, T.glist_of(t))
            return
        self.fail(node, 'growing a list that is the result of a call')

    def gl_unify_into(self, node, lt, xt, what):
        u = unify(lt.elem, xt)
        if u is None:
            self.fail(node, f'{what} {xt.name} to a {lt.name} (one element type per list)')
        if u is not lt.elem:
            self.gl_retype(node, T.glist_of(u))

    def gl_infer_method(self, call):
        """x.append(v) and the other methods, as a statement or for pop's value: the types they imply."""
        recv, meth = receiver(call)
        lt = self.typeof(recv)
        if lt.kind != 'glist':
            self.fail(call, f'.{meth}() of {lt.name}')
        self.makes = True
        args = call.args
        want = {'append': 1, 'extend': 1, 'insert': 2, 'clear': 0}.get(meth)
        if any(isinstance(a, ast.Starred) for a in args) or (want is not None and len(args) != want) or (meth == 'pop' and len(args) > 1):
            self.fail(call, f'.{meth}() with these arguments')
        if meth in ('pop', 'insert') and args and self.typeof(args[0]).kind not in ('int', 'bool'):
            self.fail(call, f'.{meth}() with an index that is not an int')
        if meth in ('append', 'insert'):
            xt = self.gl_value_type(args[-1])
            self.gl_elem_ok(args[-1], xt)
            self.gl_unify_into(recv, lt, xt, 'appends')
        elif meth == 'extend':
            st = self.gl_value_type(args[0])
            if st.kind != 'glist':
                self.fail(call, f'.extend() with {st.name}')
            self.gl_unify_into(recv, lt, st.elem, 'extends with')
        if meth == 'pop':
            if lt.elem is T.PENDING:
                raise Pending()
            return lt.elem
        return T.NONE

    def gl_subscript_type(self, e):
        if isinstance(e.slice, ast.Slice):
            self.fail(e, 'a slice of a list (a copy in Python)')
        if self.typeof(e.slice).kind not in ('int', 'bool') or self.typeof(e.slice) in (T.U64, T.NP_UINT64):
            self.fail(e, f'a list indexed with {self.typeof(e.slice).name}')
        et = self.typeof(e.value).elem
        if et is T.PENDING:
            raise Pending()
        if et.kind == 'array':
            self.env[self.view_name(e)] = et
        return et

    def gl_store_type(self, target, t, s):
        lt = self.typeof(target.value)
        self.gl_subscript_type_index(target)
        self.gl_elem_ok(s, t)
        self.gl_unify_into(target.value, lt, t, 'stores')

    def gl_subscript_type_index(self, e):
        if isinstance(e.slice, ast.Slice):
            self.fail(e, 'assigning to a slice of a list')
        if self.typeof(e.slice).kind not in ('int', 'bool'):
            self.fail(e, f'a list indexed with {self.typeof(e.slice).name}')

    def gl_loop_check(self, s):
        """A loop over a list mustn't change a list of that type: Python's loop would see the change."""
        it = s.iter
        node = it.args[0] if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ('enumerate', 'reversed') and it.args else it
        if not self.gl_is(node):
            return
        lt = self.typeof(node)
        for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            r = receiver(n) if isinstance(n, ast.Call) else None
            hit = r is not None and self.gl_maybe(r[0], lt)
            if isinstance(n, ast.Delete):
                hit = any(isinstance(t, ast.Subscript) and self.gl_maybe(t.value, lt) for t in n.targets)
            if isinstance(n, ast.AugAssign) and self.gl_maybe(n.target, lt):
                hit = True
            if hit:
                self.fail(n, 'changes a list while a loop goes over a list of that type (Python\'s loop would see the change)')

    def gl_maybe(self, node, lt):
        try:
            return self.typeof(node) is lt
        except Exception:                       # noqa: BLE001 (not typed here: can't be that list)
            return False

    # ---- C
    def gl_ctype(self, t):
        if t.kind == 'glist':
            return 'sj_glist *'
        if t.kind == 'tuple':
            return self.gl_struct(t)
        if t.kind == 'array':
            return self.gl_record(t.ndim)
        return t.ctype

    def gl_record(self, ndim):
        """An array in a list: the record a returned array has (array_record_slots), so the dispatcher reads both alike."""
        reg = self.gl_registry('gl_records')
        if ndim not in reg:
            name = f'sjA{ndim}'
            self.unit.code.append(f'typedef struct {name} {{ int64_t r[{10 + 2 * ndim}]; }} {name};')
            reg[ndim] = name
        return reg[ndim]

    def gl_fill_record(self, rec, src):
        """The record of the array variable src into the struct rec."""
        t = self.env[src]
        self.emit(f'{rec}.r[0] = (int64_t)(uintptr_t)v_{src}; {rec}.r[1] = {f"v_{src}__b" if self.tracks else "-1"}; '
                  f'{rec}.r[2] = {f"v_{src}__o" if self.tracks else "0"}; {rec}.r[3] = 0;')
        self.open(f'if (v_{src}__h)')
        self.emit(f'sj_block *b_ = v_{src}__h; {rec}.r[3] = (int64_t)(uintptr_t)b_; {rec}.r[4] = (int64_t)(uintptr_t)(b_ + 1); {rec}.r[5] = b_->ndim;')
        for d in range(4):
            self.emit(f'{rec}.r[{6 + d}] = b_->shape[{d}];')
        self.close()
        for d in range(t.ndim):
            self.emit(f'{rec}.r[{10 + 2 * d}] = {self.extent(src, d)}; {rec}.r[{11 + 2 * d}] = {self.stride(src, d)};')

    def gl_bind_array(self, name, rec):
        """The array variable name from the record rec."""
        t = self.env[name]
        self.emit(f'v_{name} = ({t.ctype.strip()})(uintptr_t){rec}.r[0]; v_{name}__h = (sj_block *)(uintptr_t){rec}.r[3];')
        if self.tracks:
            self.emit(f'v_{name}__b = {rec}.r[1]; v_{name}__o = {rec}.r[2];')
        for d in range(t.ndim):
            self.emit(f'{self.extent(name, d)} = {rec}.r[{10 + 2 * d}];')
            if t.strided:
                self.emit(f'v_{name}__s{d} = {rec}.r[{11 + 2 * d}];')

    def gl_registry(self, key):
        return self.unit.__dict__.setdefault(key, {})

    def gl_struct(self, t):
        reg = self.gl_registry('gl_structs')
        if t.name not in reg:
            name = f'sjT{len(reg)}'
            fields = ' '.join(f'{x.ctype} f{k};' for k, x in enumerate(leaves(t)))
            self.unit.code.append(f'typedef struct {name} {{ {fields} }} {name};')
            reg[t.name] = name
        return reg[t.name]

    def gl_mark(self, t):
        """The name of the C function marking a list of type t and what it holds."""
        reg = self.gl_registry('gl_marks')
        if t.name not in reg:
            inner = self.gl_mark(t.elem) if t.elem.kind == 'glist' else None
            name = f'sjM{len(reg)}'
            body = (f' for (int64_t i_ = 0; i_ < l->n; i_++) {inner}(((sj_glist **)l->p)[i_], m);' if inner else '')
            if t.elem.kind == 'array':
                rec = self.gl_record(t.elem.ndim)
                body = f' for (int64_t i_ = 0; i_ < l->n; i_++) {{ sj_block *h_ = (sj_block *)(uintptr_t)(({rec} *)l->p)[i_].r[3]; if (h_) h_->mark = m; }}' 
            self.unit.code.append(f'static void {name}(sj_glist *l, int m) {{ if (!l) return; sj_block *b = (sj_block *)l - 1; '
                                  f'if (b->mark == m) return; b->mark = m; if (l->db) l->db->mark = m;{body} }}')
            reg[t.name] = name
        return reg[t.name]

    def gl_size(self, t):
        return f'(int64_t)sizeof({self.gl_ctype(t.elem)})'

    def gl_new(self, t, cap, line):
        self.helpers.add('glist')
        v = self.temp()
        self.emit(f'sj_glist *{v} = sj_glist_new(&sj_h, {self.gl_size(t)}, {cap});')
        self.emit(f'if (!{v}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
        return v

    def gl_value(self, x, et):
        """C for the value x puts in a list of et: a number, a struct of a tuple's leaves, a list's header."""
        if et.kind == 'tuple':
            vals = self.texpr(x)
            flat = []

            def walk(v):
                for y in v:
                    walk(y) if isinstance(y, list) else flat.append(y[0])
            walk(vals)
            st = self.gl_struct(et)
            tmp = self.temp()
            self.emit(f'{st} {tmp} = {{{", ".join(flat)}}};')
            return tmp
        if self.is_list_ctor(x):
            return self.gl_make_value(x, self.gl_ctor_type(x))
        if et.kind == 'array':
            src = self.materialize(x)
            rec = self.temp()
            self.emit(f'{self.gl_record(et.ndim)} {rec};')
            self.gl_fill_record(rec, src)
            return rec
        c, _ = self.expr(x)
        if et.kind == 'glist':
            return self.hold_ptr(c)
        return self.hold(self.coerce(c, self.typeof(x), et), et)

    def hold_ptr(self, c):
        v = self.temp()
        self.emit(f'sj_glist *{v} = {c};')
        return v

    def gl_push(self, l, et, val, line):
        ct = self.gl_ctype(et)
        self.emit(f'if ({l}->n == {l}->cap && !sj_glist_reserve(&sj_h, {l}, (int64_t)sizeof({ct}), {l}->n + 1)) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
        self.emit(f'(({ct} *){l}->p)[{l}->n++] = {val};')

    def gl_make_value(self, v, t):
        """Make the list v (a literal, a repetition or a comprehension) into a new header; its elements first, in order."""
        line = self.line(v)
        et = t.elem
        if isinstance(v, ast.List):
            vals = [self.gl_value(x, et) for x in v.elts]
            l = self.gl_new(t, str(len(vals)), line)
            for val in vals:
                self.gl_push(l, et, val, line)
            return l
        if isinstance(v, ast.ListComp):
            l = self.gl_new(t, '0', line)
            saved, saved_c, opened = {}, {}, 0
            try:
                for g in v.generators:
                    cnt, elem = self.iter_prep(self.iterable(g.iter))
                    k = self.temp()
                    self.open_counted('uint64_t', k, '0', cnt)
                    opened += 1
                    self.comp_bind_values(g.target, elem(k), saved, saved_c)
                    for c in g.ifs:
                        self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
                self.gl_push(l, et, self.gl_value(v.elt, et), line)
            finally:
                for _ in range(opened):
                    self.close()
                self.restore(saved, saved_c)
            return l
        lit, k = (v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left)
        kc = None
        if k is v.left:                                  # k * [..]: k first, as Python evaluates it
            kc = self.hold(self.coerce(self.expr(k)[0], self.typeof(k), T.INT), T.INT)
        vals = [self.gl_value(x, et) for x in lit.elts]
        if kc is None:
            kc = self.hold(self.coerce(self.expr(k)[0], self.typeof(k), T.INT), T.INT)
        reps = self.hold(f'({kc} > 0 ? {kc} : 0)', T.INT)
        if vals:
            self.emit(f'if ((uint64_t){reps} > (uint64_t)(((int64_t)1 << 56) / {len(vals)})) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
        l = self.gl_new(t, f'{reps} * {len(vals)}', line)
        r = self.temp()
        self.open_counted('int64_t', r, '0', reps)
        for val in vals:                                 # the same values each time: lists in it are shared, as in Python
            self.gl_push(l, et, val, line)
        self.close()
        return l

    def gl_index(self, l, idx_node, line, msg):
        """A checked index into list l, negative ones from the end. msg: which IndexError."""
        raw = self.hold(self.coerce(self.expr(idx_node)[0], self.typeof(idx_node), T.INT), T.INT)
        i = self.temp()
        self.emit(f'int64_t {i} = {raw} < 0 ? {raw} + {l}->n : {raw};')
        self.emit(f'if ((uint64_t){i} >= (uint64_t){l}->n) SJ_FAIL({LIST_INDEX_ERROR}, {line}, {msg}, 0);')
        return i

    def gl_read(self, e):
        """x[i]: (C value, flag)."""
        l = self.hold_ptr(self.expr(e.value)[0])
        i = self.gl_index(l, e.slice, self.line(e), 0)
        et = self.typeof(e)
        return f'(({self.gl_ctype(et)} *){l}->p)[{i}]', (flag_of(et) if et.kind != 'glist' else '0')

    def gl_store(self, target, v, vt):
        """x[i] = v: the list and index first, then the store (the value is already computed, as Python does)."""
        l = self.hold_ptr(self.expr(target.value)[0])
        i = self.gl_index(l, target.slice, self.line(target), 1)
        et = self.typeof(target.value).elem
        val = v[0] if et.kind in ('glist', 'array', 'tuple') else self.coerce(v[0], vt, et)
        self.emit(f'(({self.gl_ctype(et)} *){l}->p)[{i}] = {val};')

    def gl_method(self, call, want_value=False):
        """x.append(v) and the others. Returns pop's (C value, flag)."""
        recv, meth = receiver(call)
        line = self.line(call)
        lt = self.typeof(recv)
        et = lt.elem
        ct = self.gl_ctype(et)
        l = self.hold_ptr(self.expr(recv)[0])
        if meth == 'append':
            self.gl_push(l, et, self.gl_value(call.args[0], et), line)
        elif meth == 'clear':
            self.emit(f'{l}->n = 0;')
        elif meth == 'extend':
            src = self.gl_value(call.args[0], lt)          # a list made on the spot, or one held: evaluated first
            n = self.temp()
            self.emit(f'int64_t {n} = {src}->n;')          # x.extend(x) doubles x: its length before it grows
            self.emit(f'if ({n} && !sj_glist_reserve(&sj_h, {l}, (int64_t)sizeof({ct}), {l}->n + {n})) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
            self.emit(f'if ({n}) memmove(({ct} *){l}->p + {l}->n, {src}->p, sizeof({ct}) * (size_t){n}); {l}->n += {n};')
        elif meth == 'insert':
            raw = self.hold(self.coerce(self.expr(call.args[0])[0], self.typeof(call.args[0]), T.INT), T.INT)
            val = self.gl_value(call.args[1], et)
            i = self.temp()                                # Python clamps: no IndexError
            self.emit(f'int64_t {i} = {raw} < 0 ? {raw} + {l}->n : {raw}; if ({i} < 0) {i} = 0; if ({i} > {l}->n) {i} = {l}->n;')
            self.emit(f'if ({l}->n == {l}->cap && !sj_glist_reserve(&sj_h, {l}, (int64_t)sizeof({ct}), {l}->n + 1)) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
            self.emit(f'memmove(({ct} *){l}->p + {i} + 1, ({ct} *){l}->p + {i}, sizeof({ct}) * (size_t)({l}->n - {i})); (({ct} *){l}->p)[{i}] = {val}; {l}->n++;')
        elif meth == 'pop':
            out = self.temp()
            self.emit(f'{ct} {out};')
            if not call.args:
                self.emit(f'if (!{l}->n) SJ_FAIL({LIST_INDEX_ERROR}, {line}, 2, 0);')
                self.emit(f'{out} = (({ct} *){l}->p)[--{l}->n];')
            else:
                raw = self.hold(self.coerce(self.expr(call.args[0])[0], self.typeof(call.args[0]), T.INT), T.INT)
                i = self.temp()
                self.emit(f'if (!{l}->n) SJ_FAIL({LIST_INDEX_ERROR}, {line}, 2, 0);')
                self.emit(f'int64_t {i} = {raw} < 0 ? {raw} + {l}->n : {raw};')
                self.emit(f'if ((uint64_t){i} >= (uint64_t){l}->n) SJ_FAIL({LIST_INDEX_ERROR}, {line}, 3, 0);')
                self.emit(f'{out} = (({ct} *){l}->p)[{i}]; memmove(({ct} *){l}->p + {i}, ({ct} *){l}->p + {i} + 1, sizeof({ct}) * (size_t)({l}->n - {i} - 1)); {l}->n--;')
            if want_value:
                return out, (flag_of(et) if et.kind != 'glist' else '0')
        return None

    def gl_delete(self, target):
        l = self.hold_ptr(self.expr(target.value)[0])
        i = self.gl_index(l, target.slice, self.line(target), 1)
        ct = self.gl_ctype(self.typeof(target.value).elem)
        self.emit(f'memmove(({ct} *){l}->p + {i}, ({ct} *){l}->p + {i} + 1, sizeof({ct}) * (size_t)({l}->n - {i} - 1)); {l}->n--;')

    def gl_iter_prep(self, node):
        """For a loop over a list: (trip count, elem(k)). Nothing in the loop changes a list of its type (gl_loop_check)."""
        l = self.hold_ptr(self.expr(node)[0])
        et = self.typeof(node).elem
        ct = self.gl_ctype(et)
        if et.kind == 'tuple':
            def elem(k):
                at = [0]

                def walk(tt):
                    out = []
                    for x in tt.elem:
                        if x.kind == 'tuple':
                            out.append(walk(x))
                        else:
                            out.append((f'(({ct} *){l}->p)[{k}].f{at[0]}', x, flag_of(x)))
                            at[0] += 1
                    return out
                return walk(et)
            return f'(uint64_t){l}->n', elem
        if et.kind == 'array':
            return f'(uint64_t){l}->n', lambda k: ArrayOf(f'(({ct} *){l}->p)[{k}]', et)
        return f'(uint64_t){l}->n', lambda k: (f'(({ct} *){l}->p)[{k}]', et, flag_of(et) if et.kind != 'glist' else '0')

    def gl_materialize(self, e):
        """x[i] that is an array: its view variable, filled from the record."""
        c, _ = self.gl_read(e)
        name = self.view_name(e)
        rec = self.temp()
        self.emit(f'{self.gl_record(self.env[name].ndim)} {rec} = {c};')
        self.gl_bind_array(name, rec)
        return name

    def gl_texpr(self, e):
        """A tuple read from a list (x[i] or x.pop()): its leaves, held now."""
        if isinstance(e, ast.Call):
            c, _ = self.gl_method(e, want_value=True)
        else:
            c, _ = self.gl_read(e)
        t = self.typeof(e)
        s = self.temp()
        self.emit(f'{self.gl_struct(t)} {s} = {c};')
        at = [0]

        def walk(tt):
            out = []
            for x in tt.elem:
                if x.kind == 'tuple':
                    out.append(walk(x))
                else:
                    out.append((self.hold(f'{s}.f{at[0]}', x), flag_of(x)))
                    at[0] += 1
            return out
        return walk(t)

    def gl_return(self, value, slot=None):
        """Hand a list back: everything it reaches stays when the heap is freed, and the dispatcher reads and releases it.
        slot: its place among a returned tuple's values."""
        c, _ = self.expr(value)
        l = self.hold_ptr(c)
        self.emit(f'{self.gl_mark(self.typeof(value))}({l}, 2);')
        if slot is None:
            self.emit(f'sj_io[0] = (int64_t)(uintptr_t){l}; sj_io[5] = 0;')
        else:
            self.emit(f'sj_io[{slot}] = (int64_t)(uintptr_t){l}; sj_io[{slot + 1}] = 0;')

    def gl_marks(self):
        """The roots for a sweep: every list variable."""
        return ' '.join(f'{self.gl_mark(t)}(v_{n}, 1);' for n, t in self.env.items() if t.kind == 'glist')


def flag_of(t):
    return '1' if t.np is True else '0' if t.np is False else None
