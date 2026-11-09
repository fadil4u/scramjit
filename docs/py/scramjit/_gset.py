"""Sets, as CPython keeps them: an open-addressed hash table with CPython's hashes, probing (linear probes, then
perturbation), resizing, dummies for removed elements, and the same order of operations in each set operation, so a
set iterates in the order Python's would, and `pop` takes the element Python's would.

Elements: ints, floats, bools, and tuples of these (one type per set). A NaN's hash in CPython is its object's
address; here every NaN is a value of its own, with hash 0 (see gotchas.md).

In C a set is a pointer to a header {table, fill, used, mask, finger, table's block} on the heap of arrays, its table
mask + 1 entries {hash, state (0 empty, 1 active, 2 dummy), key}. Each element type has its own functions (SET_TEMPLATE).
"""

import ast
import sys

from . import _types as T
from ._runtime import KEY_ERROR, MEMORY_ERROR

HASH32 = sys.hash_info.width == 32    # a 32-bit CPython (Pyodide's): hashes modulo 2**31 - 1, xxHash's 32-bit primes

SNIPPET = ("""
typedef uint32_t sj_uh; typedef int32_t sj_sh;
#define SJ_XXP1 2654435761U
#define SJ_XXP2 2246822519U
#define SJ_XXP5 374761393U
#define SJ_XXROT(x) (((x) << 13) | ((x) >> 19))
""" if HASH32 else """
typedef uint64_t sj_uh; typedef int64_t sj_sh;
#define SJ_XXP1 11400714785074694791ULL
#define SJ_XXP2 14029467366897019727ULL
#define SJ_XXP5 2870177450012600261ULL
#define SJ_XXROT(x) (((x) << 31) | ((x) >> 33))
""") + """
#define SJ_HASH_BITS """ + ("31" if HASH32 else "61") + """
typedef struct sj_gset { void *t; int64_t fill, used, mask, finger; sj_block *db; } sj_gset;
static void *sj_gset_table(sj_heap *h, sj_gset *s, int64_t esize, int64_t size) {
    sj_block *b = sj_make(h, 1, &size, esize, 1);
    if (!b) return 0;
    s->t = b + 1; s->db = b; s->mask = size - 1;
    return s->t;
}
static sj_gset *sj_gset_new(sj_heap *h, int64_t esize) {
    int64_t one = 1;
    sj_block *b = sj_make(h, 1, &one, sizeof(sj_gset), 1);
    if (!b) return 0;
    sj_gset *s = (sj_gset *)(b + 1);
    s->fill = s->used = s->finger = 0;
    return sj_gset_table(h, s, esize, 8) ? s : 0;
}
/* CPython's hash of an int: modulo 2**61 - 1 (2**31 - 1 on 32 bits), with its sign; -1 is -2 */
static inline int64_t sj_hash_int(int64_t x) {
    const uint64_t P = ((uint64_t)1 << SJ_HASH_BITS) - 1;
    uint64_t a = x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x;
    a = (a & P) + (a >> SJ_HASH_BITS);                 /* modulo a Mersenne prime: a fold, and once more at most */
    a = (a & P) + (a >> SJ_HASH_BITS);
    if (a >= P) a -= P;
    int64_t h = x < 0 ? -(int64_t)a : (int64_t)a;
    return h == -1 ? -2 : h;
}
/* CPython's hash of a float (_Py_HashDouble): equal to its int's for an integral float. A NaN: 0 here */
static int64_t sj_hash_double(double v) {
    if (isinf(v)) return v > 0 ? 314159 : -314159;
    if (v != v) return 0;
    const uint64_t P = ((uint64_t)1 << SJ_HASH_BITS) - 1;
    int e;
    double m = frexp(v, &e);
    uint64_t sign = 1, x = 0;
    if (m < 0) { sign = (uint64_t)-1; m = -m; }
    while (m) {
        x = ((x << 28) & P) | x >> (SJ_HASH_BITS - 28);
        m *= 268435456.0;
        e -= 28;
        uint64_t y = (uint64_t)m;
        m -= (double)y;
        x += y;
        if (x >= P) x -= P;
    }
    e = e >= 0 ? e % SJ_HASH_BITS : SJ_HASH_BITS - 1 - ((-1 - e) % SJ_HASH_BITS);
    x = ((x << e) & P) | x >> (SJ_HASH_BITS - e);
    sj_uh r = (sj_uh)(x * sign);
    if (r == (sj_uh)-1) r = (sj_uh)-2;
    return (int64_t)(sj_sh)r;
}
static inline int64_t sj_bits_of(double d) { int64_t b; memcpy(&b, &d, 8); return b; }
"""

# Per element type: @N@ the prefix, @E@ the entry type, @K@ the key type; @N@_hash and @N@_eq come before.
SET_TEMPLATE = """
static int64_t @N@_look(const sj_gset *s, @K@ key, int64_t hh) {          /* the slot holding key, or -1 (set_lookkey) */
    const @E@ *t = (const @E@ *)s->t;
    uint64_t mask = (uint64_t)s->mask, i = (uint64_t)hh & mask, perturb = (uint64_t)(sj_uh)hh;
    for (;;) {
        uint64_t j = i;
        int probes = i + 9 <= mask ? 9 : 0;
        for (;;) {
            if (t[j].st == 0) return -1;
            if (t[j].st == 1 && t[j].h == hh && @N@_eq(t[j].k, key)) return (int64_t)j;
            if (!probes--) break;
            j++;
        }
        perturb >>= 5;
        i = (i * 5 + 1 + perturb) & mask;
    }
}
static void @N@_clean(@E@ *t, uint64_t mask, @K@ key, int64_t hh) {       /* set_insert_clean */
    uint64_t i = (uint64_t)hh & mask, perturb = (uint64_t)(sj_uh)hh;
    for (;;) {
        if (t[i].st == 0) break;
        if (i + 9 <= mask) {
            uint64_t j;
            for (j = 1; j <= 9; j++) if (t[i + j].st == 0) break;
            if (j <= 9) { i += j; break; }
        }
        perturb >>= 5;
        i = (i * 5 + 1 + perturb) & mask;
    }
    t[i].h = hh; t[i].st = 1; t[i].k = key;
}
static int @N@_resize(sj_heap *h, sj_gset *s, int64_t minused) {          /* set_table_resize */
    uint64_t size = 8;
    while (size <= (uint64_t)minused) size <<= 1;
    if (size == 8 && s->mask == 7 && s->fill == s->used) return 1;
    @E@ *old = (@E@ *)s->t;
    int64_t oldmask = s->mask;
    @E@ *t = (@E@ *)sj_gset_table(h, s, (int64_t)sizeof(@E@), (int64_t)size);
    if (!t) return 0;
    for (int64_t i = 0; i <= oldmask; i++) if (old[i].st == 1) @N@_clean(t, size - 1, old[i].k, old[i].h);
    s->fill = s->used;
    return 1;
}
static int @N@_add(sj_heap *h, sj_gset *s, @K@ key, int64_t hh) {         /* set_add_entry */
    @E@ *t = (@E@ *)s->t;
    uint64_t mask = (uint64_t)s->mask, i = (uint64_t)hh & mask, perturb = (uint64_t)(sj_uh)hh, j;
    int64_t freeslot = -1;
    for (;;) {
        j = i;
        int probes = i + 9 <= mask ? 9 : 0;
        for (;;) {
            if (t[j].st == 0) goto unused;
            if (t[j].st == 1) { if (t[j].h == hh && @N@_eq(t[j].k, key)) return 1; }
            else freeslot = (int64_t)j;
            if (!probes--) break;
            j++;
        }
        perturb >>= 5;
        i = (i * 5 + 1 + perturb) & mask;
    }
  unused:
    if (freeslot >= 0) { t[freeslot].h = hh; t[freeslot].st = 1; t[freeslot].k = key; s->used++; return 1; }
    t[j].h = hh; t[j].st = 1; t[j].k = key;
    s->fill++; s->used++;
    if ((uint64_t)s->fill * 5 < mask * 3) return 1;
    return @N@_resize(h, s, s->used > 50000 ? s->used * 2 : s->used * 4);
}
static int @N@_discard(sj_gset *s, @K@ key, int64_t hh) {                 /* set_discard_entry: 1 if it was there */
    int64_t j = @N@_look(s, key, hh);
    if (j < 0) return 0;
    @E@ *t = (@E@ *)s->t;
    t[j].st = 2; t[j].h = -1;
    s->used--;
    return 1;
}
static int @N@_clear(sj_heap *h, sj_gset *s) {
    if (!sj_gset_table(h, s, (int64_t)sizeof(@E@), 8)) return 0;
    s->fill = s->used = 0;
    return 1;
}
static int @N@_pop(sj_gset *s, @K@ *out) {                                 /* set_pop: from the finger on */
    @E@ *t = (@E@ *)s->t;
    if (!s->used) return 0;
    int64_t j = s->finger & s->mask;
    while (t[j].st != 1) { j++; if (j > s->mask) j = 0; }
    *out = t[j].k;
    t[j].st = 2; t[j].h = -1;
    s->used--;
    s->finger = j + 1;
    return 1;
}
static int @N@_merge(sj_heap *h, sj_gset *so, const sj_gset *other) {     /* set_merge */
    if (so == other || other->used == 0) return 1;
    if ((so->fill + other->used) * 5 >= so->mask * 3)
        if (!@N@_resize(h, so, (so->used + other->used) * 2)) return 0;
    const @E@ *o = (const @E@ *)other->t;
    @E@ *t = (@E@ *)so->t;
    if (so->fill == 0 && so->mask == other->mask && other->fill == other->used) {
        for (int64_t i = 0; i <= other->mask; i++) t[i] = o[i];
        so->fill = other->fill; so->used = other->used;
        return 1;
    }
    if (so->fill == 0) {
        so->fill = so->used = other->used;
        for (int64_t i = 0; i <= other->mask; i++) if (o[i].st == 1) @N@_clean(t, (uint64_t)so->mask, o[i].k, o[i].h);
        return 1;
    }
    for (int64_t i = 0; i <= other->mask; i++) if (o[i].st == 1 && !@N@_add(h, so, o[i].k, o[i].h)) return 0;
    return 1;
}
static sj_gset *@N@_copy(sj_heap *h, const sj_gset *s) {
    sj_gset *r = sj_gset_new(h, (int64_t)sizeof(@E@));
    return r && @N@_merge(h, r, s) ? r : 0;
}
static sj_gset *@N@_union(sj_heap *h, const sj_gset *a, const sj_gset *b) {        /* set_or */
    sj_gset *r = @N@_copy(h, a);
    return r && (a == b || @N@_merge(h, r, b)) ? r : 0;
}
static sj_gset *@N@_inter(sj_heap *h, const sj_gset *so, const sj_gset *other) {   /* set_intersection */
    if (so == other) return @N@_copy(h, so);
    sj_gset *r = sj_gset_new(h, (int64_t)sizeof(@E@));
    if (!r) return 0;
    if (other->used > so->used) { const sj_gset *x = so; so = other; other = x; }
    const @E@ *o = (const @E@ *)other->t;
    for (int64_t i = 0; i <= other->mask; i++)
        if (o[i].st == 1 && @N@_look(so, o[i].k, o[i].h) >= 0 && !@N@_add(h, r, o[i].k, o[i].h)) return 0;
    return r;
}
static int @N@_diff_update(sj_heap *h, sj_gset *so, const sj_gset *other) {       /* set_difference_update_internal */
    if (so == other) return @N@_clear(h, so);
    if ((other->used >> 3) > so->used) { other = @N@_inter(h, so, other); if (!other) return 0; }
    const @E@ *o = (const @E@ *)other->t;
    for (int64_t i = 0; i <= other->mask; i++) if (o[i].st == 1) @N@_discard(so, o[i].k, o[i].h);
    if ((uint64_t)(so->fill - so->used) <= (uint64_t)so->mask / 4) return 1;
    return @N@_resize(h, so, so->used > 50000 ? so->used * 2 : so->used * 4);
}
static sj_gset *@N@_diff(sj_heap *h, const sj_gset *so, const sj_gset *other) {   /* set_difference */
    if ((so->used >> 2) > other->used) {
        sj_gset *r = @N@_copy(h, so);
        return r && @N@_diff_update(h, r, other) ? r : 0;
    }
    sj_gset *r = sj_gset_new(h, (int64_t)sizeof(@E@));
    if (!r) return 0;
    const @E@ *t = (const @E@ *)so->t;
    for (int64_t i = 0; i <= so->mask; i++)
        if (t[i].st == 1 && @N@_look(other, t[i].k, t[i].h) < 0 && !@N@_add(h, r, t[i].k, t[i].h)) return 0;
    return r;
}
static int @N@_xor_update(sj_heap *h, sj_gset *so, const sj_gset *other) {        /* set_symmetric_difference_update */
    if (so == other) return @N@_clear(h, so);
    const @E@ *o = (const @E@ *)other->t;
    for (int64_t i = 0; i <= other->mask; i++)
        if (o[i].st == 1 && !@N@_discard(so, o[i].k, o[i].h) && !@N@_add(h, so, o[i].k, o[i].h)) return 0;
    return 1;
}
static sj_gset *@N@_xor(sj_heap *h, const sj_gset *so, const sj_gset *other) {    /* set_symmetric_difference */
    sj_gset *r = sj_gset_new(h, (int64_t)sizeof(@E@));
    return r && @N@_merge(h, r, other) && @N@_xor_update(h, r, so) ? r : 0;
}
static int @N@_and_update(sj_heap *h, sj_gset *so, const sj_gset *other) {        /* set_intersection_update: its table */
    sj_gset *r = @N@_inter(h, so, other);
    if (!r) return 0;
    so->t = r->t; so->db = r->db; so->fill = r->fill; so->used = r->used; so->mask = r->mask;
    return 1;
}
static int @N@_subset(const sj_gset *a, const sj_gset *b) {                       /* set_issubset */
    if (a->used > b->used) return 0;
    const @E@ *t = (const @E@ *)a->t;
    for (int64_t i = 0; i <= a->mask; i++) if (t[i].st == 1 && @N@_look(b, t[i].k, t[i].h) < 0) return 0;
    return 1;
}
static int @N@_disjoint(const sj_gset *a, const sj_gset *b) {
    if (a->used > b->used) { const sj_gset *x = a; a = b; b = x; }
    const @E@ *t = (const @E@ *)a->t;
    for (int64_t i = 0; i <= a->mask; i++) if (t[i].st == 1 && @N@_look(b, t[i].k, t[i].h) >= 0) return 0;
    return 1;
}
"""

METHODS = ('add', 'discard', 'remove', 'pop', 'clear', 'update', 'copy', 'union', 'intersection', 'difference',
           'symmetric_difference', 'issubset', 'issuperset', 'isdisjoint', 'intersection_update', 'difference_update',
           'symmetric_difference_update')
CHANGES = ('add', 'discard', 'remove', 'pop', 'clear', 'update', 'intersection_update', 'difference_update',
           'symmetric_difference_update')
BINOPS = {ast.BitOr: 'union', ast.BitAnd: 'inter', ast.Sub: 'diff', ast.BitXor: 'xor'}
AUGOPS = {ast.BitOr: 'merge', ast.BitAnd: 'and_update', ast.Sub: 'diff_update', ast.BitXor: 'xor_update'}
NAMED = {'union': 'union', 'intersection': 'inter', 'difference': 'diff', 'symmetric_difference': 'xor'}
NAMED_UPDATE = {'intersection_update': 'and_update', 'difference_update': 'diff_update', 'symmetric_difference_update': 'xor_update'}
KEY_KINDS = {'int': 0, 'float': 1, 'bool': 2}


NOT_CONSTANT = object()
FOLDS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
         ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}


def folded(e):
    """What CPython's compiler folds e to (a number or a tuple of them), or NOT_CONSTANT."""
    if isinstance(e, ast.Constant) and type(e.value) in (int, float, bool):
        return e.value
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd, ast.Invert)):
        v = folded(e.operand)
        if v is NOT_CONSTANT or type(v) is tuple or (isinstance(e.op, ast.Invert) and type(v) is float):
            return NOT_CONSTANT
        return -v if isinstance(e.op, ast.USub) else +v if isinstance(e.op, ast.UAdd) else ~v
    if isinstance(e, ast.BinOp) and type(e.op) in FOLDS:
        a, b = folded(e.left), folded(e.right)
        if NOT_CONSTANT in (a, b) or tuple in (type(a), type(b)):
            return NOT_CONSTANT
        try:
            v = FOLDS[type(e.op)](a, b)
        except (ArithmeticError, ValueError):
            return NOT_CONSTANT
        return v if type(v) in (int, float) and (type(v) is float or abs(v) < 1 << 64) else NOT_CONSTANT
    if isinstance(e, ast.Tuple) and isinstance(e.ctx, ast.Load):
        vs = [folded(x) for x in e.elts]
        return NOT_CONSTANT if any(v is NOT_CONSTANT for v in vs) else tuple(vs)
    return NOT_CONSTANT


def set_receiver(call):
    """For s.add(x) and the other set methods: (the set expression, the method), else None."""
    f = call.func if isinstance(call, ast.Call) else None
    if isinstance(f, ast.Attribute) and f.attr in METHODS and not call.keywords:
        return f.value, f.attr
    return None


def element_ok(t):
    """Can a set hold values of type t (hashable here, one representation per value)?"""
    if t.kind == 'tuple':
        return all(element_ok(x) for x in t.elem)
    return t in (T.INT, T.FLOAT, T.BOOL)


class GSet:
    """The Compiler's part for sets (mixed into Compiler)."""

    # ---- types
    def gs_ctor(self, x):
        """A set made here: {a, b}, a comprehension, set() or set(iterable)."""
        return isinstance(x, (ast.Set, ast.SetComp)) or (isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'set'
                                                        and self.is_builtin('set', 'set') and len(x.args) <= 1 and not x.keywords)

    def gs_is(self, node):
        if self.gs_ctor(node):
            return True
        if isinstance(node, ast.Name):
            t = self.env.get(node.id)
            return t is not None and t.kind == 'gset'
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            return (self.gl_is(node.value) or self.rc_is(node.value)) and self.typeof(node).kind == 'gset'
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            return self.gs_is(node.left) and self.gs_is(node.right)
        if isinstance(node, ast.Call) and set_receiver(node) and set_receiver(node)[1] in NAMED and self.gs_is(set_receiver(node)[0]):
            return True
        if isinstance(node, ast.Call) and set_receiver(node) and set_receiver(node)[1] == 'copy' and self.gs_is(set_receiver(node)[0]):
            return True
        if isinstance(node, ast.Call) and self.callee_of(node) is not None:
            return self.typeof(node).kind == 'gset'
        return False

    def gs_elem_check(self, node, t):
        if t is T.PENDING:
            return
        if not element_ok(t):
            self.fail(node, f'a set of {t.name} (sets of ints, floats, bools and tuples of these compile)')

    def gs_ctor_type(self, x):
        self.makes = True
        if isinstance(x, ast.Set):
            e = T.PENDING
            for y in x.elts:
                if isinstance(y, ast.Starred):
                    self.fail(x, 'a set literal with * in it')
                yt = self.typeof(y)
                self.gs_elem_check(y, yt)
                u = self.gs_unify(e, yt)
                if u is None:
                    self.fail(x, f'a set holding {e.name} and {yt.name} (one element type per set)')
                e = u
            return T.gset_of(e)
        if isinstance(x, ast.SetComp):
            et = self.comp_types(ast.ListComp(x.elt, x.generators), any_elem=True)   # its element, as a list's would be
            self.gs_elem_check(x, et)
            return T.gset_of(et)
        if not x.args:
            return T.gset_of(T.PENDING)
        src = x.args[0]
        if self.gs_is(src):
            return self.typeof(src)
        et = self.gs_src_elem(src)
        if isinstance(et, list):
            et = T.tuple_of(et)
        self.gs_elem_check(x, et)
        return T.gset_of(et)

    def gs_src_elem(self, src):
        """The element type of an iterable a set is made from or updated with: a loop's, a generator's, a literal's."""
        if isinstance(src, ast.GeneratorExp):
            return self.comp_types(ast.ListComp(src.elt, src.generators), any_elem=True)
        if isinstance(src, (ast.List, ast.Tuple)):
            e = T.PENDING
            for y in src.elts:
                if isinstance(y, ast.Starred):
                    self.fail(src, 'a set made from a literal with * in it')
                u = self.gs_unify(e, self.typeof(y))
                if u is None:
                    self.fail(src, f'a set made from {e.name} and {self.typeof(y).name} (one element type per set)')
                e = u
            return e
        et = self.elem_struct(self.no_rows(self.iterable(src), src))
        return T.tuple_of(et) if isinstance(et, list) else et

    def gs_unify(self, a, b):
        if a is T.PENDING:
            return b
        if b is T.PENDING or a is b:
            return a
        return None

    def gs_type(self, node):
        """The type of a set expression."""
        if self.gs_ctor(node):
            return self.gs_ctor_type(node)
        if isinstance(node, ast.BinOp):
            return self.gs_binop_type(node)
        if isinstance(node, ast.Call) and set_receiver(node):
            return self.gs_infer_method(node)
        return self.typeof(node)

    def gs_binop_type(self, e):
        a, b = self.gs_type(e.left), self.gs_type(e.right)
        self.makes = True
        u = self.gs_unify(a.elem, b.elem)
        if u is None:
            self.fail(e, f'{a.name} and {b.name} combined (one element type per set)')
        if u is T.PENDING:
            from ._glist import Pending
            raise Pending()
        return T.gset_of(u)

    def gs_unify_into(self, node, st, xt, what):
        u = self.gs_unify(st.elem, xt)
        if u is None:
            self.fail(node, f'{what} {xt.name} to a {st.name} (one element type per set)')
        if u is not st.elem:
            self.gl_retype(node, T.gset_of(u))

    def gs_infer_method(self, call):
        """s.add(x) and the others: the types they imply, and what they return."""
        recv, meth = set_receiver(call)
        st = self.gs_type(recv)
        if st.kind != 'gset':
            self.fail(call, f'.{meth}() of {st.name}')
        self.makes = True
        args = call.args
        want = {'add': 1, 'discard': 1, 'remove': 1, 'pop': 0, 'clear': 0, 'copy': 0}.get(meth, 1)
        if len(args) != want or any(isinstance(a, ast.Starred) for a in args):
            self.fail(call, f'.{meth}() with these arguments')
        if meth in ('add', 'discard', 'remove'):
            xt = self.typeof(args[0])
            if meth == 'add':
                self.gs_elem_check(args[0], xt)
                self.gs_unify_into(recv, st, xt, 'adds')
                return T.NONE
            if meth == 'remove' and xt.kind == 'tuple':
                self.fail(call, '.remove() of a tuple (its KeyError would show the tuple)')
            if st.elem is not T.PENDING and xt is not st.elem and not (xt.is_number and st.elem.is_number):
                return T.NONE                            # can't be there (nothing to discard; remove raises)
            return T.NONE
        if meth in ('pop', 'copy') or meth in NAMED or meth in ('issubset', 'issuperset', 'isdisjoint') or meth in NAMED_UPDATE:
            if st.elem is T.PENDING and meth != 'copy':
                from ._glist import Pending
                raise Pending()
        if meth == 'pop':
            return st.elem
        if meth == 'clear':
            return T.NONE
        if meth == 'copy':
            return st
        if meth == 'update':
            src = args[0]
            if self.gs_is(src):
                xt = self.gs_type(src).elem
            else:
                xt = self.gs_src_elem(src)
            self.gs_elem_check(src, xt)
            self.gs_unify_into(recv, st, xt, 'adds')
            return T.NONE
        other = self.gs_type(args[0]) if self.gs_is(args[0]) else None
        if other is None:
            self.fail(call, f'.{meth}() of something other than a set')
        if self.gs_unify(st.elem, other.elem) is None:
            self.fail(call, f'.{meth}() of {st.name} with {other.name}')
        if meth in ('issubset', 'issuperset', 'isdisjoint'):
            return T.BOOL
        if meth in NAMED_UPDATE:
            return T.NONE
        return T.gset_of(self.gs_unify(st.elem, other.elem))

    def gs_aug_type(self, s):
        st, ot = self.gs_type(s.target), self.gs_type(s.value)
        self.makes = True
        if self.gs_unify(st.elem, ot.elem) is None or type(s.op) not in AUGOPS:
            self.fail(s, f'{st.name} {type(s.op).__name__}= {ot.name}')
        if type(s.op) is ast.BitOr:
            self.gs_unify_into(s.target, st, ot.elem, 'adds')

    def gs_in_type(self, e):
        st = self.gs_type(e.comparators[0])
        if st.elem is T.PENDING:
            from ._glist import Pending
            raise Pending()
        xt = self.typeof(e.left)
        if not (xt.is_number or xt.kind == 'tuple'):
            self.fail(e, f'{xt.name} in a set')
        if xt.kind == 'tuple' and st.elem.kind == 'tuple' and xt is not st.elem:
            self.fail(e, f'{xt.name} in {st.name} (a tuple of other types than its elements)')
        if xt.np or xt is T.U64:
            self.fail(e, f'{xt.name} in a set')
        return T.BOOL

    def gs_loop_check(self, s, node):
        """A loop over a set mustn't change a set of that type: Python's would see it, or raise."""
        st = self.typeof(node)
        for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            r = set_receiver(n) if isinstance(n, ast.Call) else None
            hit = r is not None and r[1] in CHANGES and self.gl_maybe(r[0], st)
            if isinstance(n, ast.AugAssign) and self.gl_maybe(n.target, st):
                hit = True
            if hit:
                self.fail(n, "changes a set while a loop goes over a set of that type (Python's loop would see it, or raise)")

    # ---- C
    def gs_fn(self, et):
        """The prefix of the C functions for sets of et (made once per element type)."""
        reg = self.gl_registry('gs_fns')
        if et.name in reg:
            return reg[et.name]
        self.helpers.add('gset')
        n = f'sjZ{len(reg)}'
        reg[et.name] = n
        k = self.gl_ctype(et)
        self.unit.code.append(f'typedef struct {n}e {{ int64_t h; int64_t st; {k} k; }} {n}e;')
        if et.kind == 'tuple':
            fields = []

            def walk(t):
                for x in t.elem:
                    walk(x) if x.kind == 'tuple' else fields.append(x)
            walk(et)
            lines, at, count = [], [0], [0]

            def thash(t):                                 # CPython's tuple hash (xxHash's lanes), nested as the tuple is
                count[0] += 1
                acc = f'a{count[0]}'
                lines.append(f'sj_uh {acc} = SJ_XXP5;')
                for x in t.elem:
                    if x.kind == 'tuple':
                        lane = thash(x)
                    else:
                        lane = f'(sj_uh){self.gs_leaf_hash(x, f"x.f{at[0]}")}'
                        at[0] += 1
                    lines.append(f'{acc} += {lane} * SJ_XXP2; {acc} = SJ_XXROT({acc}); {acc} *= SJ_XXP1;')
                lines.append(f'{acc} += (sj_uh){len(t.elem)} ^ (SJ_XXP5 ^ (sj_uh)3527539UL); if ({acc} == (sj_uh)-1) {acc} = 1546275796;')
                return acc
            top = thash(et)
            self.unit.code.append(f'static int64_t {n}_hash({k} x) {{ {" ".join(lines)} return (int64_t)(sj_sh){top}; }}')
            eq = ' && '.join(f'a.f{i} == b.f{i}' for i in range(len(fields))) or '1'
            self.unit.code.append(f'static inline int {n}_eq({k} a, {k} b) {{ return {eq}; }}')
        else:
            self.unit.code.append(f'static inline int64_t {n}_hash({k} x) {{ return {self.gs_leaf_hash(et, "x")}; }}')
            self.unit.code.append(f'static inline int {n}_eq({k} a, {k} b) {{ return a == b; }}')
        self.unit.code.append(SET_TEMPLATE.replace('@N@', n).replace('@E@', f'{n}e').replace('@K@', k))
        return n

    def gs_leaf_hash(self, t, c):
        if t.kind == 'float':
            return f'sj_hash_double({c})'
        if t.kind == 'bool':
            return f'(int64_t)(({c}) != 0)'
        return f'sj_hash_int((int64_t)({c}))'

    def gs_new(self, t, line):
        self.helpers.add('glist')
        n = self.gs_fn(t.elem if t.elem is not T.PENDING else T.INT)
        v = self.temp()
        self.emit(f'sj_gset *{v} = sj_gset_new(&sj_h, (int64_t)sizeof({n}e));')
        self.emit(f'if (!{v}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
        return v

    def gs_key(self, x, et):
        """The C value of x as a key of a set of et, held in a variable."""
        if et.kind == 'tuple':
            return self.gl_value(x, et)
        c, _ = self.expr(x)
        return self.hold(self.coerce(c, self.typeof(x), et), et)

    def gs_hold(self, c):
        v = self.temp()
        self.emit(f'sj_gset *{v} = {c};')
        return v

    def gs_ok(self, call, line):
        self.emit(f'if (!({call})) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')

    def gs_ptr(self, x):
        """A set expression, evaluated: its header."""
        if self.gs_ctor(x):
            return self.gs_hold(self.gs_make(x, self.gs_type(x)))
        if isinstance(x, ast.BinOp):
            return self.gs_hold(self.gs_binop(x)[0])
        if isinstance(x, ast.Call) and set_receiver(x):
            return self.gs_hold(self.gs_method(x, want_value=True)[0])
        return self.gs_hold(self.expr(x)[0])

    def gs_make(self, x, t):
        """A set made here, as CPython makes it."""
        line = self.line(x)
        et = t.elem
        if isinstance(x, ast.Set):
            if len(x.elts) > 2 and all(folded(y) is not NOT_CONSTANT for y in x.elts):
                return self.gs_constant(x, t)            # {1, 2, 3}: CPython merges a constant frozenset into a new set
            keys = [self.gs_key(y, et) for y in x.elts]  # all made first, then added in order (BUILD_SET)
            s = self.gs_new(t, line)
            n = self.gs_fn(et)
            for k in keys:
                self.gs_ok(f'{n}_add(&sj_h, {s}, {k}, {n}_hash({k}))', line)
            return s
        if isinstance(x, ast.SetComp):
            s = self.gs_new(t, line)
            n = self.gs_fn(et)
            saved, saved_c, opened = {}, {}, 0
            try:
                for g in x.generators:
                    cnt, elem = self.iter_prep(self.iterable(g.iter))
                    k = self.temp()
                    self.open_counted('uint64_t', k, '0', cnt)
                    opened += 1
                    self.comp_bind_values(g.target, elem(k), saved, saved_c)
                    for c in g.ifs:
                        self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
                key = self.gs_key(x.elt, et)
                self.gs_ok(f'{n}_add(&sj_h, {s}, {key}, {n}_hash({key}))', line)
            finally:
                for _ in range(opened):
                    self.close()
                self.restore(saved, saved_c)
            return s
        if not x.args:
            return self.gs_new(t, line)
        src = x.args[0]
        if self.gs_is(src):                              # set(s): a copy, as a merge into a new set
            o = self.gs_ptr(src)
            n = self.gs_fn(et)
            s = self.temp()
            self.emit(f'sj_gset *{s} = {n}_copy(&sj_h, {o});')
            self.gs_ok(s, line)
            return s
        s = self.gs_new(t, line)
        self.gs_add_all(s, src, et, line)
        return s

    def gs_add_all(self, s, src, et, line):
        """Each element of src, added in order (set_update_iterable)."""
        n = self.gs_fn(et)
        if isinstance(src, (ast.List, ast.Tuple)):        # the list or tuple made first, then each added
            keys = [self.gs_key(y, et) for y in src.elts]
            for k in keys:
                self.gs_ok(f'{n}_add(&sj_h, {s}, {k}, {n}_hash({k}))', line)
            return
        if isinstance(src, ast.GeneratorExp):
            saved, saved_c, opened = {}, {}, 0
            try:
                for g in src.generators:
                    cnt, elem = self.iter_prep(self.iterable(g.iter))
                    k = self.temp()
                    self.open_counted('uint64_t', k, '0', cnt)
                    opened += 1
                    self.comp_bind_values(g.target, elem(k), saved, saved_c)
                    for c in g.ifs:
                        self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
                key = self.gs_key(src.elt, et)
                self.gs_ok(f'{n}_add(&sj_h, {s}, {key}, {n}_hash({key}))', line)
            finally:
                for _ in range(opened):
                    self.close()
                self.restore(saved, saved_c)
            return
        cnt, elem = self.iter_prep(self.iterable(src))
        k = self.temp()
        self.open_counted('uint64_t', k, '0', cnt)
        v = elem(k)
        if isinstance(v, list):                          # a tuple: its struct
            flat = []

            def walk(y):
                for z in y:
                    walk(z) if isinstance(z, list) else flat.append(z[0])
            walk(v)
            key = self.temp()
            self.emit(f'{self.gl_struct(et)} {key} = {{{", ".join(flat)}}};')
        else:
            from ._compiler import TupleOf
            if isinstance(v, TupleOf):
                flat = []

                def walk2(y):
                    for z in y:
                        walk2(z) if isinstance(z, list) else flat.append(z[0])
                walk2(v.vals)
                key = self.temp()
                self.emit(f'{self.gl_struct(et)} {key} = {{{", ".join(flat)}}};')
            else:
                c, vt, _ = v
                key = self.hold(self.coerce(c, vt, et), et)
        self.gs_ok(f'{n}_add(&sj_h, {s}, {key}, {n}_hash({key}))', line)
        self.close()

    def gs_constant(self, x, t):
        """{1, 2, 3}: CPython builds a frozenset of the constants when it compiles the function, and merges it into a
        new set. That frozenset's table is read from the function's code (_pysets), so the merge copies what it copies."""
        from . import _pysets
        line = self.line(x)
        et = t.elem
        vals = [folded(y) for y in x.elts]
        fz = self.gs_code_constant(frozenset(vals))
        n = self.gs_fn(et)
        table = _pysets.read(fz) if fz is not None else None
        if table is None:                                # (no layout: the frozenset as adding them in order makes it)
            keys = [self.gs_key(y, et) for y in x.elts]
            s = self.gs_new(t, line)
            for k in keys:
                self.gs_ok(f'{n}_add(&sj_h, {s}, {k}, {n}_hash({k}))', line)
            return s
        fill, used, mask, _, slots = table
        entries = []
        for o, h in slots:
            if o is None:
                entries.append('{0, 0}')
            elif o is _pysets.DUMMY:
                entries.append('{-1, 2}')
            else:
                entries.append(f'{{{h}LL, 1, {self.gs_const_key(o, et)}}}')
        reg = self.gl_registry('gs_consts')
        name = f'sjF{len(reg)}'
        reg[name] = True
        self.unit.code.append(f'static const {n}e {name}t[{mask + 1}] = {{{", ".join(entries)}}};')
        self.unit.code.append(f'static const sj_gset {name} = {{(void *){name}t, {fill}, {used}, {mask}, 0, 0}};')
        s = self.gs_new(t, line)
        self.gs_ok(f'{n}_merge(&sj_h, {s}, &{name})', line)
        return s

    def gs_const_key(self, o, et):
        if et.kind == 'tuple':
            flat = []

            def walk(y):
                for z in y:
                    walk(z) if type(z) is tuple else flat.append(z)
            walk(o)
            return '{' + ', '.join(self.gs_const_num(v) for v in flat) + '}'
        return self.gs_const_num(o)

    def gs_const_num(self, v):
        if type(v) is float:
            return repr(v) if v == v and v not in (float('inf'), float('-inf')) else ('NAN' if v != v else ('INFINITY' if v > 0 else '-INFINITY'))
        return f'{int(v)}LL'

    def gs_code_constant(self, value):
        """The frozenset constant equal to value in the function's code, or None."""
        fn = getattr(self.unit.stack[-1], '_fn', None)  # (equal frozensets in one module's code are one object)
        code = getattr(fn, '__code__', None)
        if code is None:
            return None
        todo = [code]
        while todo:
            c = todo.pop()
            for k in c.co_consts:
                if type(k) is frozenset and k == value and all(type(a) is type(b) for a, b in zip(sorted(k, key=repr), sorted(value, key=repr))):
                    return k
                if hasattr(k, 'co_consts'):
                    todo.append(k)
        return None

    def gs_binop(self, e):
        t = self.typeof(e)
        n = self.gs_fn(t.elem)
        a = self.gs_ptr(e.left)
        b = self.gs_ptr(e.right)
        r = self.temp()
        self.emit(f'sj_gset *{r} = {n}_{BINOPS[type(e.op)]}(&sj_h, {a}, {b});')
        self.gs_ok(r, self.line(e))
        return r, '0'

    def gs_augassign(self, s):
        st = self.typeof(s.target)
        n = self.gs_fn(st.elem)
        a = self.gs_ptr(s.target)
        b = self.gs_ptr(s.value)
        self.gs_ok(f'{n}_{AUGOPS[type(s.op)]}(&sj_h, {a}, {b})', self.line(s))

    def gs_method(self, call, want_value=False):
        recv, meth = set_receiver(call)
        line = self.line(call)
        st = self.gs_type(recv)
        et = st.elem if st.elem is not T.PENDING else T.INT
        n = self.gs_fn(et)
        s = self.gs_ptr(recv)
        if meth in ('add', 'discard', 'remove'):
            xt = self.typeof(call.args[0])
            if xt is not et and not (xt.is_number and et.is_number) and meth != 'add':
                self.expr(call.args[0])                  # another kind of value: never there
                if meth == 'remove':
                    self.fail(call, '.remove() of a value of another type')
                return ('0', '0')
            if meth != 'add' and xt.is_number and et.is_number and xt is not et:
                return self.gs_mixed(call, s, n, xt, et, meth, line)
            k = self.gs_key(call.args[0], et)
            if meth == 'add':
                self.gs_ok(f'{n}_add(&sj_h, {s}, {k}, {n}_hash({k}))', line)
            elif meth == 'discard':
                self.emit(f'{n}_discard({s}, {k}, {n}_hash({k}));')
            else:
                self.emit(f'if (!{n}_discard({s}, {k}, {n}_hash({k}))) SJ_FAIL({KEY_ERROR}, {line}, {self.gs_key_bits(k, et)}, {KEY_KINDS[et.kind]});')
            return ('0', '0')
        if meth == 'pop':
            out = self.temp()
            self.emit(f'{self.gl_ctype(et)} {out};')
            self.emit(f'if (!{n}_pop({s}, &{out})) SJ_FAIL({KEY_ERROR}, {line}, 0, 3);')
            return out, '0'
        if meth == 'clear':
            self.gs_ok(f'{n}_clear(&sj_h, {s})', line)
            return ('0', '0')
        if meth == 'copy':
            r = self.temp()
            self.emit(f'sj_gset *{r} = {n}_copy(&sj_h, {s});')
            self.gs_ok(r, line)
            return r, '0'
        src = call.args[0]
        if meth == 'update':
            if self.gs_is(src):
                o = self.gs_ptr(src)
                self.gs_ok(f'{n}_merge(&sj_h, {s}, {o})', line)
            else:
                self.gs_add_all(s, src, et, line)
            return ('0', '0')
        o = self.gs_ptr(src)
        if meth in NAMED:
            r = self.temp()
            self.emit(f'sj_gset *{r} = {n}_{NAMED[meth]}(&sj_h, {s}, {o});')
            self.gs_ok(r, line)
            return r, '0'
        if meth in NAMED_UPDATE:
            self.gs_ok(f'{n}_{NAMED_UPDATE[meth]}(&sj_h, {s}, {o})', line)
            return ('0', '0')
        if meth == 'issubset':
            return self.hold(f'{n}_subset({s}, {o})', T.BOOL), '0'
        if meth == 'issuperset':
            return self.hold(f'{n}_subset({o}, {s})', T.BOOL), '0'
        return self.hold(f'{n}_disjoint({s}, {o})', T.BOOL), '0'

    def gs_mixed(self, call, s, n, xt, et, meth, line):
        """discard/remove of a number of another kind (1.0 from a set of ints): it is there if a value equal to it is,
        and it has that value's hash."""
        c, _ = self.expr(call.args[0])
        x = self.hold(c, xt)
        k = self.gs_as(x, xt, et)
        same = self.cmp(k, et, x, xt, '==')
        found = self.temp()
        self.emit(f'int {found} = ({same}) && {n}_discard({s}, {k}, {n}_hash({k}));')
        if meth == 'remove':
            self.emit(f'if (!{found}) SJ_FAIL({KEY_ERROR}, {line}, {self.gs_key_bits(x, xt)}, {KEY_KINDS[xt.kind]});')
        return ('0', '0')

    def gs_as(self, x, xt, et):
        """x (of number type xt) as a key of type et, where it can be one; whether it is that value is checked after."""
        k = self.temp()
        if xt.kind == 'float' and et.kind != 'float':     # (a float that is no int64: never equal to one)
            self.emit(f'{self.gl_ctype(et)} {k} = ({x} == {x} && fabs({x}) < 9.2e18) ? ({self.gl_ctype(et)})({x}) : 0;')
        else:
            self.emit(f'{self.gl_ctype(et)} {k} = ({self.gl_ctype(et)})({x});')
        return k

    def gs_key_bits(self, k, t):
        if t.kind == 'float':
            return f'sj_bits_of({k})'
        return f'(int64_t)({k})'

    def gs_in(self, e):
        right = e.comparators[0]
        st = self.gs_type(right)
        et = st.elem
        xt = self.typeof(e.left)
        negate = isinstance(e.ops[0], ast.NotIn)
        if xt.is_number != et.is_number:                 # a number in a set of tuples, or the reverse: never there
            self.texpr(e.left) if xt.kind == 'tuple' else self.expr(e.left)
            self.gs_ptr(right)
            return ('1' if negate else '0'), '0'
        n = self.gs_fn(et)
        if xt.is_number and xt is not et:                # 1.0 in a set of ints: there if an equal value is, with its hash
            c, _ = self.expr(e.left)
            x = self.hold(c, xt)
            s = self.gs_ptr(right)
            k = self.gs_as(x, xt, et)
            r = self.hold(f'(({self.cmp(k, et, x, xt, "==")}) && {n}_look({s}, {k}, {n}_hash({k})) >= 0)', T.BOOL)
        else:
            k = self.gs_key(e.left, et)
            s = self.gs_ptr(right)
            r = self.hold(f'({n}_look({s}, {k}, {n}_hash({k})) >= 0)', T.BOOL)
        return (f'(!{r})' if negate else r), '0'

    def gs_iter_prep(self, node):
        """For a loop over a set: (trip count, elem(k)), going through its table in order, as Python's iterator does."""
        st = self.typeof(node)
        et = st.elem
        n = self.gs_fn(et)
        s = self.gs_ptr(node)
        cur = self.temp()
        self.emit(f'int64_t {cur} = 0;')

        def elem(k):
            v = self.temp()
            self.emit(f'while ((({n}e *){s}->t)[{cur}].st != 1) {cur}++;')
            self.emit(f'{self.gl_ctype(et)} {v} = (({n}e *){s}->t)[{cur}++].k;')
            if et.kind == 'tuple':
                at = [0]

                def walk(tt):
                    out = []
                    for x in tt.elem:
                        if x.kind == 'tuple':
                            out.append(walk(x))
                        else:
                            out.append((f'{v}.f{at[0]}', x, '0'))
                            at[0] += 1
                    return out
                return walk(et)
            return (v, et, '0')
        return f'(uint64_t){s}->used', elem

    def gs_tree_vals(self, tree):
        """A tuple's tree of C values (cmp_tree) as texpr gives one: (value, flag) leaves."""
        return [self.gs_tree_vals(x) if isinstance(x, list) else (x, '0') for x in tree]

    def gs_set_cmp_body(self, op, a, b):
        n = self.gs_fn(a.elem)
        if op == '==':
            return f'return a == b || (a->used == b->used && {n}_subset(a, b));'
        if op == '<=':
            return f'return {n}_subset(a, b);'
        if op == '>=':
            return f'return {n}_subset(b, a);'
        if op == '<':
            return f'return a->used < b->used && {n}_subset(a, b);'
        return f'return b->used < a->used && {n}_subset(b, a);'
