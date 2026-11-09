"""CPython's sets, table and all: read a set's hash table as CPython laid it out, and make a set with a given table.

A compiled function keeps sets as CPython does (_gset): the same hashes, probing, resizing and dummies, so the same
iteration order. A set passed in is read as it is, slot for slot, and a set returned or written back is given the
kernel's table slot for slot, so that what Python then does with it (adds, discards, iteration) goes as it would
have. That needs CPython's PySetObject layout, which is checked against a set made here before it is relied on; if
it doesn't match (another implementation, a free-threaded build, no ctypes), `exact` is False and sets are rebuilt
from their elements, in table order: the same elements, perhaps in another order (see gotchas.md).
"""

import struct
import sys

try:
    import ctypes
except ImportError:                     # pragma: no cover
    ctypes = None

DUMMY = object()                        # a slot that held an element and was emptied (CPython's dummy)


class _Layout:
    def __init__(self):
        P = ctypes.sizeof(ctypes.c_void_p)
        self.P = P
        self.head = object.__basicsize__                 # refcount and type
        self.fields = struct.Struct('<' + ('q' if P == 8 else 'i') * 6)   # fill, used, mask, table, hash, finger
        self.entry = struct.Struct('<qq' if P == 8 else '<ii')            # key, hash
        self.small = self.head + 6 * P                   # the smalltable, inside the object
        if set.__basicsize__ != self.small + 8 * 2 * P + P:
            raise ValueError('another layout')
        api = ctypes.pythonapi
        self.incref = api.Py_IncRef
        self.incref.argtypes = [ctypes.py_object]
        self.calloc = api.PyMem_Calloc
        self.calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.calloc.restype = ctypes.c_void_p
        s = {1, 9}                                       # 1 and 9 share a slot of 8: 9 goes on; emptied, it's a dummy
        s.discard(9)
        _, _, mask, table, _, _ = self.fields.unpack(ctypes.string_at(id(s) + self.head, self.fields.size))
        self.dummy = next(k for k, h in self.entries(table, mask) if h == -1)

    def entries(self, table, mask):
        return list(self.entry.iter_unpack(ctypes.string_at(table, (mask + 1) * self.entry.size)))

    def read(self, s):
        fill, used, mask, table, _, finger = self.fields.unpack(ctypes.string_at(id(s) + self.head, self.fields.size))
        objs = {id(x): x for x in s}
        out = []
        for k, h in self.entries(table, mask):
            out.append((None, 0) if k == 0 else (DUMMY, -1) if k == self.dummy else (objs[k], h))
        if used != len(s) or sum(1 for o, _ in out if o is not None and o is not DUMMY) != used:
            raise ValueError('not a table as expected')
        return fill, used, mask, finger, out

    def write(self, s, fill, used, mask, finger, slots):
        """Make the empty set s hold slots: (object, hash), (DUMMY, -1) or (None, 0) each, mask + 1 of them."""
        assert not s and len(slots) == mask + 1
        base = id(s)
        _, _, smask, stable, shash, _ = self.fields.unpack(ctypes.string_at(base + self.head, self.fields.size))
        if mask == 7:
            table = base + self.small
        else:
            table = self.calloc(mask + 1, self.entry.size)
            if not table:
                raise MemoryError()
        raw = bytearray((mask + 1) * self.entry.size)
        for i, (o, h) in enumerate(slots):
            if o is None:
                continue
            if o is DUMMY:
                self.entry.pack_into(raw, i * self.entry.size, self.dummy, -1)
            else:
                self.incref(o)                           # the set holds it
                self.entry.pack_into(raw, i * self.entry.size, id(o), h)
        ctypes.memmove(table, bytes(raw), len(raw))
        ctypes.memmove(base + self.head, self.fields.pack(fill, used, mask, table, shash, finger), self.fields.size)


def _probe():
    if ctypes is None or sys.implementation.name != 'cpython' or not getattr(sys, '_is_gil_enabled', lambda: True)():
        return None
    try:
        lay = _Layout()
        src = set()                                      # check it: a table read, written into a new set, behaves the same
        for x in (3, 11, 19, 27, 5, 1 << 40, -7, 2.5, (1, 2)):
            src.add(x)
        src.discard(19)
        got = lay.read(src)
        dst = set()
        lay.write(dst, *got[:4], got[4])
        if list(dst) != list(src) or len(dst) != len(src) or 27 not in dst or 19 in dst:
            return None
        for x in (19, 35, 43, 1 << 41):
            src.add(x)
            dst.add(x)
        src.discard(3)
        dst.discard(3)
        if list(dst) != list(src) or src.pop() != dst.pop() or list(dst) != list(src):
            return None
        big = set(range(100))
        dst = set()
        lay.write(dst, *lay.read(big)[:4], lay.read(big)[4])
        if list(dst) != list(big):
            return None
        return lay
    except Exception:                                    # noqa: BLE001 (any surprise: rebuild instead)
        return None


LAYOUT = _probe()
exact = LAYOUT is not None


def read(s):
    """(fill, used, mask, finger, slots) of a set, slots as LAYOUT.write takes them; None if that can't be done."""
    if LAYOUT is None:
        return None
    try:
        return LAYOUT.read(s)
    except Exception:                                    # noqa: BLE001
        return None


def make(fill, used, mask, finger, slots):
    """A new set with this table, slot for slot; without the layout, the same elements added in table order."""
    s = set()
    if LAYOUT is not None and used:
        LAYOUT.write(s, fill, used, mask, finger, slots)
        return s
    for o, _ in slots:
        if o is not None and o is not DUMMY:
            s.add(o)
    return s


def replace(s, fill, used, mask, finger, slots):
    """Make s, a set passed in, hold this table: its old elements released first (as clear() releases them)."""
    s.clear()
    if LAYOUT is not None and used:
        LAYOUT.write(s, fill, used, mask, finger, slots)
    else:
        for o, _ in slots:
            if o is not None and o is not DUMMY:
                s.add(o)


def layout(s):
    """Without CPython's layout: the table of a set made by adding s's elements in its order (CPython's algorithm)."""
    mask, fill, used = 7, 0, 0
    slots = [(None, 0)] * 8

    def clean(table, mask, o, h):
        i, perturb = h & mask, h & 0xFFFFFFFFFFFFFFFF
        while True:
            if table[i][0] is None:
                table[i] = (o, h)
                return
            if i + 9 <= mask:
                for j in range(i + 1, i + 10):
                    if table[j][0] is None:
                        table[j] = (o, h)
                        return
            perturb >>= 5
            i = (i * 5 + 1 + perturb) & mask
    for o in s:
        h = hash(o)
        clean(slots, mask, o, h)
        fill += 1
        used += 1
        if fill * 5 >= mask * 3:
            size = 8
            while size <= (used * 2 if used > 50000 else used * 4):
                size <<= 1
            old, mask = slots, size - 1
            slots = [(None, 0)] * size
            for oo, hh in old:
                if oo is not None:
                    clean(slots, mask, oo, hh)
    return fill, used, mask, 0, slots
