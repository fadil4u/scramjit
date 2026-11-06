"""The types scramjit compiles, and how Python and numpy values map onto them.

A value read from a numpy array is a numpy scalar, which behaves differently from a Python number
of the same value: np.float64(1) / 0 is inf where 1.0 / 0 raises, and int + np.int64 is np.int64.
So every scalar type records whether it is a Python number or a numpy scalar.
"""

import struct


class Type:
    __slots__ = ('name', 'ctype', 'kind', 'np', 'elem', 'writable', 'ndim', 'strided', 'py', 'methods', 'cls', 'writes', 'recipe')

    def __init__(self, name, ctype, kind, np=False, elem=None, writable=True):
        self.name = name          # how signatures print it
        self.ctype = ctype        # the C type of a value (arrays: the element pointer)
        self.kind = kind          # 'int', 'float', 'bool', 'array', 'none'
        self.np = np              # a numpy scalar rather than a Python number
        self.elem = elem          # arrays: the element type (a numpy scalar type)
        self.writable = writable  # arrays
        self.ndim = 0             # arrays: dimensions
        self.strided = False      # arrays: a view whose elements aren't next to each other
        self.py = None            # a narrow-or-literal variable: the kind of its Python number, when it isn't the narrow one's
        self.methods = None       # instances: method name -> (the attributes a call of it uses, those it assigns), through what it calls too
        self.cls = None           # instances: the class
        self.writes = frozenset() # instances: every attribute anything given it may assign
        self.recipe = None        # a class instances are made of in the function: how its __init__ assigns (see madeclass_of)

    def __repr__(self):
        return self.name

    @property
    def is_number(self):
        return self.kind in ('int', 'float', 'bool')


INT = Type('int', 'int64_t', 'int')
FLOAT = Type('float', 'double', 'float')
BOOL = Type('bool', 'int', 'bool')
NP_INT64 = Type('np.int64', 'int64_t', 'int', np=True)
NP_FLOAT64 = Type('np.float64', 'double', 'float', np=True)
NP_BOOL = Type('np.bool', 'int', 'bool', np=True)
# Elements of uint64 arrays. numpy wraps them in uint64 with Python ints and unsigned dtypes, and makes float64 of them with
# signed numpy ints and floats (uint64_result).
NP_UINT64 = Type('np.uint64', 'uint64_t', 'int', np=True)
# A variable that holds a Python number at some times and a numpy scalar of the same kind at
# others (total = 0.0, then total += x[i]): its C value plus a runtime flag.
INT_DYN = Type('int | np.int64', 'int64_t', 'int', np='dyn')
FLOAT_DYN = Type('float | np.float64', 'double', 'float', np='dyn')
BOOL_DYN = Type('bool | np.bool', 'int', 'bool', np='dyn')
NONE = Type('None', 'void', 'none')
# A Python int known to be in [0, 2**64): a literal past int64, a value masked with & 0xFFFFFFFFFFFFFFFF, or a variable that only
# ever holds such values. Held unsigned, so 64-bit hashes come out as Python's numbers.
U64 = Type('int (unsigned 64-bit)', 'uint64_t', 'int')

# numpy scalars of the narrow dtypes. Arithmetic on them follows numpy's NEP 50 rules: see narrow_result.
NARROW = {
    'float32': Type('np.float32', 'float', 'float', np=True),
    'int32': Type('np.int32', 'int32_t', 'int', np=True),
    'int16': Type('np.int16', 'int16_t', 'int', np=True),
    'int8': Type('np.int8', 'int8_t', 'int', np=True),
    'uint32': Type('np.uint32', 'uint32_t', 'int', np=True),
    'uint16': Type('np.uint16', 'uint16_t', 'int', np=True),
    'uint8': Type('np.uint8', 'uint8_t', 'int', np=True),
}
INT_RANGE = {'int32': (-(1 << 31), (1 << 31) - 1), 'int16': (-(1 << 15), (1 << 15) - 1), 'int8': (-128, 127),
             'uint32': (0, (1 << 32) - 1), 'uint16': (0, 65535), 'uint8': (0, 255)}


# A variable that starts as a literal and becomes a narrow scalar (total = 0.0, then total += x[i], which is
# np.float32 from then on): held in the narrow type, since the literal fits it exactly, plus the runtime flag.
NARROW_DYN = {n: Type(f'{"float" if t.kind == "float" else "int"} | {t.name}', t.ctype, t.kind, np='dyn', elem=t) for n, t in NARROW.items()}
# s = 0, then s += x[i] on float32: the int 0 until the first addition. A float holds a small int exactly; returned while
# the flag says it's still Python's, it comes back as an int.
INT_OR_F32 = Type('int | np.float32', 'float', 'float', np='dyn', elem=NARROW['float32'])
INT_OR_F32.py = 'int'


def is_narrow(t):
    return t.np is True and t.name not in ('np.int64', 'np.float64', 'np.bool', 'np.uint64')


def is_narrow_dyn(t):
    return t.np == 'dyn' and t.elem is not None and t.is_number


def as_narrow(t):
    """The narrow type a value is when it's numpy's: t, or what a narrow-or-literal variable becomes."""
    return t.elem if is_narrow_dyn(t) else t


def holds_exactly(t, v):
    """Does narrow type t hold the number v exactly? v is None if it isn't a literal."""
    if v is None:
        return False
    if t.kind == 'int':
        return type(v) is int and INT_RANGE[t.name[3:]][0] <= v <= INT_RANGE[t.name[3:]][1]
    if v != v or v in (float('inf'), float('-inf')):
        return True
    try:
        return struct.unpack('<f', struct.pack('<f', v))[0] == v
    except OverflowError:
        return False


def narrow_or_literal(a, b):
    """The type of a variable assigned a Python number and a narrow scalar of the same kind, or None."""
    if is_narrow_dyn(a):
        return a if b in (a.elem, INT if a.kind == 'int' or a.py == 'int' else FLOAT) else None
    if is_narrow_dyn(b):
        return narrow_or_literal(b, a)
    if is_narrow(b):
        a, b = b, a
    if is_narrow(a) and b is (INT if a.kind == 'int' else FLOAT):
        return NARROW_DYN[a.name[3:]]
    if a is NARROW['float32'] and b is INT:
        return INT_OR_F32
    return None

_BY_KIND = {('int', False): INT, ('int', True): NP_INT64, ('int', 'dyn'): INT_DYN,
            ('float', False): FLOAT, ('float', True): NP_FLOAT64, ('float', 'dyn'): FLOAT_DYN,
            ('bool', False): BOOL, ('bool', True): NP_BOOL, ('bool', 'dyn'): BOOL_DYN}


def scalar(kind, np):
    return _BY_KIND[(kind, np)]


def join_np(*nps):
    """numpy-ness of a result from its operands': numpy if any is, dynamic if any may be."""
    if True in nps:
        return True
    if 'dyn' in nps:
        return 'dyn'
    return False


def join(a, b):
    """The type of a variable that is assigned values of types a and b, or None."""
    if a is b:
        return a
    if a is U64 or b is U64 or a is NP_UINT64 or b is NP_UINT64:
        return None
    if a.is_number and b.is_number and has_narrow(a, b):
        return None
    if a.is_number and b.is_number and a.kind == b.kind:
        return scalar(a.kind, 'dyn')
    if a.kind == 'list' and b.kind == 'list':
        e = join(a.elem, b.elem)
        return None if e is None else list_of(e)
    if a.kind == 'tuple' and b.kind == 'tuple' and len(a.elem) == len(b.elem):
        elems = [join(x, y) for x, y in zip(a.elem, b.elem)]
        return None if None in elems else tuple_of(elems)
    return None


def join_returns(a, b):
    """The type a function returning a and b in different places returns. An array needn't keep one type, as a
    variable must: each comes back as what it is."""
    if a.kind == 'array' and b.kind == 'array' and a.elem is b.elem and a.ndim == b.ndim:
        return array(a.elem, a.writable and b.writable, a.ndim, a.strided or b.strided)
    if a.kind == 'tuple' and b.kind == 'tuple' and len(a.elem) == len(b.elem):
        elems = [join_returns(x, y) for x, y in zip(a.elem, b.elem)]
        return None if None in elems else tuple_of(elems)
    return join(a, b)


def is_pylist(t):
    """An array that is a list of lists of Python numbers (or a row of one)."""
    return t.kind == 'array' and t.elem.np is False


def is_c_module(x):
    return type(x).__name__ in ('Module', 'WasmModule') and type(x).__module__.startswith('cimo.')


def is_c_function(x):
    return type(x).__name__ in ('Kernel', 'WasmKernel') and type(x).__module__.startswith('cimo.')


def function_of(fn):
    """A @scram function, as what a global name holds. Each has a type of its own, so a new one is a new signature. No C."""
    t = getattr(fn, '_type', None)
    if t is None:
        t = fn._type = Type(f'function {getattr(fn, "__name__", "C module")}', None, 'function', elem=fn)
    return t


_ENUMS = {}


def enum_ok(cls):
    """An Enum or IntEnum class compiled code can hold members of as their values: int values in int64, and equality,
    hashing and truth as Enum and IntEnum define them."""
    import enum
    if not (isinstance(cls, type) and issubclass(cls, enum.Enum)) or issubclass(cls, (enum.Flag, str)):
        return False
    base = enum.IntEnum if issubclass(cls, int) else enum.Enum
    for c in cls.__mro__:                     # a class between it and Enum or IntEnum defining these changes them
        if c is base:
            break
        if any(m in vars(c) for m in ('__eq__', '__ne__', '__hash__', '__bool__', '__lt__', '__le__', '__gt__', '__ge__')):
            return False
    vals = [m.value for m in cls]
    return bool(vals) and all(type(v) is int and -(1 << 63) <= v < 1 << 63 for v in vals)


def enum_class_of(cls):
    """An enum class, as a global names it: no C value; a new class is a new signature."""
    key = ('class', cls)
    if key not in _ENUMS:
        _ENUMS[key] = Type(f'enum {cls.__name__}', None, 'enumclass', elem=cls)
    return _ENUMS[key]


def classref_of(cls):
    """A class a global names, other than an enum: for isinstance(). No C value; a new class is a new signature."""
    key = ('ref', cls)
    if key not in _ENUMS:
        _ENUMS[key] = Type(f'class {cls.__qualname__}', None, 'classref', elem=cls)
    return _ENUMS[key]


_MADE = {}


def madeclass_of(cls, recipe, methods, writes, method_fields):
    """A class the function makes instances of, as a global names it. recipe: {'params': [(name, default or None,
    keyword-only)], 'assigns': [(attribute, ('param', name) | ('const', value) | ('expr', ast))], 'post_init': bool}.
    methods, writes, method_fields: as for an instance's type. No C value; a new class is a new signature."""
    key = (cls, tuple(sorted((m, tuple(sorted(u)), tuple(sorted(w))) for m, (u, w) in methods.items())), tuple(sorted(writes)),
           tuple((m, id(ft)) for m, ft in method_fields))
    if key not in _MADE:
        t = _MADE[key] = Type(f'class {cls.__qualname__}', None, 'madeclass', elem=cls)
        t.recipe, t.methods, t.writes, t.cls, t.py = recipe, dict(methods), frozenset(writes), cls, cls.__qualname__
        t.recipe['methods'] = list(method_fields)
    return _MADE[key]


def enum_member_of(cls):
    """A member of an enum class, held as its value. An IntEnum member is also an int."""
    key = ('member', cls)
    if key not in _ENUMS:
        _ENUMS[key] = Type(cls.__name__, 'int64_t', 'enum', elem=cls)
    return _ENUMS[key]


def is_int_enum(t):
    return t.kind == 'enum' and issubclass(t.elem, int)


_DICTS = {}
_OBJECTS = {}


def dict_of(fields):
    """A dict, by the keys a function uses [(key, type)]: in C, one variable per key."""
    key = tuple((repr(k), t.name) for k, t in fields)
    if key not in _DICTS:
        _DICTS[key] = Type('dict[' + ', '.join(f'{k}: {n}' for k, n in key) + ']', None, 'dict', elem=list(fields))
    return _DICTS[key]


def object_of(name, fields, methods=None, cls=None, writes=()):
    """An instance, by the attributes a function uses [(attribute, type)]: in C, one variable per attribute. A method it
    calls is an attribute of a function's type, which is that function's own, and methods says what each call uses and assigns."""
    key = (name, id(cls), tuple((k, t.name if t.kind != 'function' else id(t)) for k, t in fields),
           tuple(sorted((m, tuple(sorted(u)), tuple(sorted(w))) for m, (u, w) in (methods or {}).items())), tuple(sorted(writes)))
    if key not in _OBJECTS:
        shown = ', '.join(f'{k}()' if t.kind == 'function' else f'{k}: {t.name}' for k, t in fields)
        t = _OBJECTS[key] = Type(f'{name}({shown})', None, 'object', elem=list(fields))
        t.methods = {m: (frozenset(u), frozenset(w)) for m, (u, w) in (methods or {}).items()}
        t.py = name                         # the class's name
        t.cls, t.writes = cls, frozenset(writes)
    return _OBJECTS[key]


_SAME = {}


def same_as(i, path=()):
    """A parameter, or an object in an attribute, that is the very object parameter i's path reaches (i None: the root of
    the parameter it's in). Part of the signature, like a type: both names are one set of variables in the C."""
    key = (i, tuple(path))
    if key not in _SAME:
        where = ('' if i is None else f'#{i}') + ''.join('.' + a for a in path)
        _SAME[key] = Type(f'same as {where or "itself"}', None, 'same', elem=key)
    return _SAME[key]


_POOLREFS = {}


def ref_of(cls):
    """A reference to an object of a pooled class, chosen by data (n = n.next, a.pos = b.pos): its index in the pool, or
    -1 for None. cls None: only ever None seen."""
    if cls not in _POOLREFS:
        _POOLREFS[cls] = Type(f'ref {cls.__qualname__ if cls else "None"}', 'int64_t', 'ref', elem=cls)
        _POOLREFS[cls].cls = cls
    return _POOLREFS[cls]


_POOLS = {}


def pool_of(pools, roots, made=()):
    """The objects references reach, each once, a pool per class: [(class, [(number attribute, type)], [(reference attribute,
    the class it reaches, or None)])], a list per number and an index list per reference. roots: [(argument, path)] they're
    walked from, path '' for the argument itself."""
    key = (tuple((c, tuple((a, t.name) for a, t in ns), tuple(rs)) for c, ns, rs in pools), tuple(roots), tuple(made))
    if key not in _POOLS:
        shown = '; '.join(f'{c.__qualname__ if c else "None"}(' + ', '.join([f'{a}: {t.name}' for a, t in ns] + [f'{a}: ref' for a, _ in rs]) + ')'
                          for c, ns, rs in pools)
        shown += ''.join(f'; makes {pools[k][0].__qualname__}' for k, _ in made)
        _POOLS[key] = Type(f'pool {shown}', None, 'pool', elem=([(c, list(ns), list(rs)) for c, ns, rs in pools], list(roots), list(made)))
    return _POOLS[key]


NOPOOL = Type('no pool', None, 'nopool')

_REFLIST = {}


def reflist_of(cls):
    """A list of objects kept in a pool (or None): a list of their indexes, written back as the objects."""
    if cls not in _REFLIST:
        t = _REFLIST[cls] = Type(f'list[ref {cls.__qualname__ if cls else "None"}]', None, 'reflist', elem=cls)
        t.cls = cls
    return _REFLIST[cls]


_POOLREF = {}


def poolref_of(pool, cls, lists=True):
    """A reference given to a compiled callee: its index, and, for the first such argument of a call (lists), the caller's
    pool lists, so the callee reads and writes the caller's objects."""
    key = (pool, cls, lists)
    if key not in _POOLREF:
        t = _POOLREF[key] = Type(f'ref {cls.__qualname__ if cls else "None"} into {pool.name}' + ('' if lists else ' (lists given before)'),
                                 None, 'poolref', elem=(pool, cls, lists))
        t.cls = cls
    return _POOLREF[key]

_INLIST = {}


def inlist_of(i):
    """An argument that is an element of argument i, a list of instances: its place there, so its attributes are that
    list's, and a write through either name is seen through the other."""
    if i not in _INLIST:
        _INLIST[i] = Type(f'element of #{i}', 'int64_t', 'inlist', elem=i)
    return _INLIST[i]


_REFS = {}


def elemref_of(elem):
    """An element of a list of instances, given to a compiled callee by reference: the attributes' lists and its index, so
    two of them that are one instance are one memory, and a write through either is seen through both."""
    if elem not in _REFS:
        _REFS[elem] = Type(f'ref {elem.name}', None, 'elemref', elem=elem)
    return _REFS[elem]


_OBJLISTS = {}


def objlist_of(elem, shared=frozenset()):
    """A list of instances, all one class, by the attributes a function uses (elem, an object type): in C, a list of
    numbers per attribute, by path. shared: the paths ('' for the list itself, 'vel' for its elements' vel) where one
    object is at several places, so the lists there are of each object once (a pool), and an index list says which is
    at each place (strided holds this set)."""
    shared = frozenset([''] if shared is True else shared or ())
    if (elem, shared) not in _OBJLISTS:
        shown = (' with repeats' if '' in shared else '') + ''.join(f' sharing .{p}' for p in sorted(shared) if p)
        t = _OBJLISTS[elem, shared] = Type(f'list[{elem.name}]{shown}', None, 'objlist', elem=elem)
        t.strided = shared
    return _OBJLISTS[elem, shared]


def leaves(t):
    """A tuple type's numbers, in order, through nested tuples."""
    out = []
    for et in t.elem:
        out.extend(leaves(et) if et.kind == 'tuple' else [et])
    return out


def is_number_tree(t):
    """A number, or a tuple of numbers and such tuples."""
    return t.is_number or (t.kind == 'tuple' and all(is_number_tree(et) for et in t.elem))


_TUPLES = {}


def tuple_of(elems):
    """A tuple of numbers: in C, one variable per element (a struct, in effect)."""
    key = tuple(e.name for e in elems)
    if key not in _TUPLES:
        _TUPLES[key] = Type(f'tuple[{", ".join(key)}]', None, 'tuple', elem=list(elems))
    return _TUPLES[key]


_ARRAYS = {}


def array(elem, writable=True, ndim=1, strided=False):
    key = (elem.name, writable, ndim, strided)
    if key not in _ARRAYS and elem.np is False:      # a list of lists: Python numbers, rows that are references
        t = Type('list[' * ndim + elem.name + ']' * ndim, elem.ctype + ' *', 'array', elem=elem, writable=True)
        t.ndim = ndim
        t.strided = strided
        _ARRAYS[key] = t
    if key not in _ARRAYS:
        name = elem.name[3:] + '[' + ', '.join([':'] * ndim) + ']'
        if strided:
            name += ' (strided)'
        if not writable:
            name += ' (read-only)'
        t = Type(name, elem.ctype + ' *', 'array', elem=elem, writable=writable)
        t.ndim = ndim
        t.strided = strided
        _ARRAYS[key] = t
    return _ARRAYS[key]


def bits(t):
    return {'int8_t': 8, 'uint8_t': 8, 'int16_t': 16, 'uint16_t': 16, 'int32_t': 32, 'uint32_t': 32}.get(t.ctype, 64)


def promote(a, b):
    """numpy's type for two numpy scalars, at least one of them narrow."""
    if a.kind == 'float' or b.kind == 'float':
        x, y = (a, b) if a.kind == 'float' else (b, a)
        if x is NP_FLOAT64 or y is NP_FLOAT64:
            return NP_FLOAT64
        return x if y.kind == 'float' or bits(y) <= 16 else NP_FLOAT64
    if a is NP_INT64 or b is NP_INT64:
        return NP_INT64
    ua, ub = a.ctype[0] == 'u', b.ctype[0] == 'u'
    width = max(bits(a), bits(b))
    if ua != ub:                            # the signed one must hold the unsigned one's values
        unsigned_bits, signed_bits = (bits(a), bits(b)) if ua else (bits(b), bits(a))
        width = signed_bits if signed_bits > unsigned_bits else unsigned_bits * 2
        return NP_INT64 if width > 32 else NARROW[f'int{width}']
    return NARROW[f'{"u" if ua else ""}int{width}']


def narrow_result(a, b, op):
    """NEP 50, with a narrow scalar on at least one side. A Python number takes the numpy scalar's type if it's the
    same kind or lower: a Python float with a narrow int gives float64, and a Python int must fit (checked when run)."""
    if a.np == 'dyn' or b.np == 'dyn':
        # Whichever the variable holds, the other side being numpy's decides. Otherwise the answer depends on the flag.
        if not (is_narrow_dyn(a) and b.np is True or is_narrow_dyn(b) and a.np is True):
            return None
        a, b = as_narrow(a), as_narrow(b)
    if not a.np or not b.np:
        n, py = (a, b) if a.np else (b, a)
        r = NP_FLOAT64 if n.kind == 'int' and py.kind == 'float' else n
    else:
        r = promote(a, b)
    return NP_FLOAT64 if op == 'Div' and r.kind == 'int' else r


def has_narrow(a, b):
    return is_narrow(a) or is_narrow(b) or is_narrow_dyn(a) or is_narrow_dyn(b)


def arith_problem(a, b, op):
    """Why a <op> b doesn't compile, when arith_result says it doesn't."""
    if (a.np and a.kind == 'bool') or (b.np and b.kind == 'bool'):
        return 'arithmetic on numpy bools'
    if NP_UINT64 in (a, b):
        return f'{a.name} and {b.name} with this operator (numpy raises TypeError)'
    if op == 'Pow':
        return f'** on {a.name} and {b.name}: numpy uses the narrow type\'s own pow'
    return f'{a.name} with {b.name}: the type of the result would depend on what the variable holds at run time'


def compare_in(a, b):
    """The C type numpy compares these in, when an int meets a float or either is np.float32; None when the two are
    compared as ints. A Python float or int becomes np.float32 first (np.float32(0.1) == 0.1 is true). Nothing
    else is rounded: numpy's integer comparisons are exact."""
    if a.kind != 'float' and b.kind != 'float':
        return None
    if not has_narrow(a, b):
        return 'double' if a.kind != b.kind and (a.np is True or b.np is True) else None
    r = arith_result(as_narrow(a), as_narrow(b), 'Add')
    return r.ctype if r is not None else 'double'


def uint64_result(a, b, op):
    """numpy's type for np.uint64 with another number, or None where numpy raises TypeError (bitwise with a signed int or a
    float) or the type would depend on a flag."""
    other = b if a is NP_UINT64 else a
    if other.np == 'dyn' or (other.np and other.kind == 'bool' and op not in ('BitAnd', 'BitOr', 'BitXor', 'Add', 'Sub', 'Mult')):
        return None
    signed_numpy = other.np is True and other.kind == 'int' and other is not NP_UINT64 and not other.ctype.startswith('uint')
    if other.kind == 'float' or signed_numpy:
        return None if op in ('BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift') else NP_FLOAT64
    return NP_FLOAT64 if op == 'Div' else NP_UINT64


def arith_result(a, b, op):
    """The type of a <op> b for numbers, following Python and numpy's promotion (NEP 50).

    Returns None when the combination isn't compiled: see arith_problem.
    """
    if NP_UINT64 in (a, b):
        return uint64_result(a, b, op)
    if (a.np and a.kind == 'bool') or (b.np and b.kind == 'bool'):
        return None
    if has_narrow(a, b):
        return narrow_result(a, b, op)
    np = join_np(a.np, b.np)
    if a.kind == 'float' or b.kind == 'float' or op == 'Div':
        return scalar('float', np)
    return scalar('int', np)


def compare_result(a, b):
    return scalar('bool', join_np(a.np, b.np))


_LISTS = {}


def list_of(elem):
    """A list of numbers that never changes size: a heap array (and a numpy flag per element if
    some elements may be numpy scalars)."""
    if elem.name not in _LISTS:
        _LISTS[elem.name] = Type(f'list[{elem.name}]', elem.ctype + ' *', 'list', elem=elem)
    return _LISTS[elem.name]
