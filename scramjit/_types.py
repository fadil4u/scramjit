"""The types scramjit compiles, and how Python and numpy values map onto them.

A value read from a numpy array is a numpy scalar, which behaves differently from a Python number
of the same value: np.float64(1) / 0 is inf where 1.0 / 0 raises, and int + np.int64 is np.int64.
So every scalar type records whether it is a Python number or a numpy scalar.
"""


class Type:
    __slots__ = ('name', 'ctype', 'kind', 'np', 'elem', 'writable')

    def __init__(self, name, ctype, kind, np=False, elem=None, writable=True):
        self.name = name          # how signatures print it
        self.ctype = ctype        # the C type of a value (arrays: the element pointer)
        self.kind = kind          # 'int', 'float', 'bool', 'array', 'none'
        self.np = np              # a numpy scalar rather than a Python number
        self.elem = elem          # arrays: the element type (a numpy scalar type)
        self.writable = writable  # arrays

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
# A variable that holds a Python number at some times and a numpy scalar of the same kind at
# others (total = 0.0, then total += x[i]): its C value plus a runtime flag.
INT_DYN = Type('int | np.int64', 'int64_t', 'int', np='dyn')
FLOAT_DYN = Type('float | np.float64', 'double', 'float', np='dyn')
BOOL_DYN = Type('bool | np.bool', 'int', 'bool', np='dyn')
NONE = Type('None', 'void', 'none')

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
    if a.is_number and b.is_number and a.kind == b.kind:
        return scalar(a.kind, 'dyn')
    if a.kind == 'list' and b.kind == 'list':
        e = join(a.elem, b.elem)
        return None if e is None else list_of(e)
    if a.kind == 'tuple' and b.kind == 'tuple' and len(a.elem) == len(b.elem):
        elems = [join(x, y) for x, y in zip(a.elem, b.elem)]
        return None if None in elems else tuple_of(elems)
    return None


_TUPLES = {}


def tuple_of(elems):
    """A tuple of numbers: in C, one variable per element (a struct, in effect)."""
    key = tuple(e.name for e in elems)
    if key not in _TUPLES:
        _TUPLES[key] = Type(f'tuple[{", ".join(key)}]', None, 'tuple', elem=list(elems))
    return _TUPLES[key]


_ARRAYS = {}


def array(elem, writable=True):
    key = (elem.name, writable)
    if key not in _ARRAYS:
        name = {'np.int64': 'int64', 'np.float64': 'float64'}[elem.name] + '[:]'
        if not writable:
            name += ' (read-only)'
        _ARRAYS[key] = Type(name, elem.ctype + ' *', 'array', elem=elem, writable=writable)
    return _ARRAYS[key]


def arith_result(a, b, op):
    """The type of a <op> b for numbers, following Python and numpy's promotion (NEP 50).

    Returns None when the combination isn't compiled (numpy bools in arithmetic, where numpy's
    rules differ from Python's).
    """
    if (a.np and a.kind == 'bool') or (b.np and b.kind == 'bool'):
        return None
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
