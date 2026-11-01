"""A small WebAssembly disassembler: module bytes -> WebAssembly text (WAT) for chosen functions.
Covers the instructions cimo emits. Used by the website's training demo (docs/nn.js)."""

import struct

_NUM = ('i32.eqz i32.eq i32.ne i32.lt_s i32.lt_u i32.gt_s i32.gt_u i32.le_s i32.le_u i32.ge_s '
        'i32.ge_u i64.eqz i64.eq i64.ne i64.lt_s i64.lt_u i64.gt_s i64.gt_u i64.le_s i64.le_u '
        'i64.ge_s i64.ge_u f32.eq f32.ne f32.lt f32.gt f32.le f32.ge f64.eq f64.ne f64.lt f64.gt '
        'f64.le f64.ge i32.clz i32.ctz i32.popcnt i32.add i32.sub i32.mul i32.div_s i32.div_u '
        'i32.rem_s i32.rem_u i32.and i32.or i32.xor i32.shl i32.shr_s i32.shr_u i32.rotl i32.rotr '
        'i64.clz i64.ctz i64.popcnt i64.add i64.sub i64.mul i64.div_s i64.div_u i64.rem_s '
        'i64.rem_u i64.and i64.or i64.xor i64.shl i64.shr_s i64.shr_u i64.rotl i64.rotr f32.abs '
        'f32.neg f32.ceil f32.floor f32.trunc f32.nearest f32.sqrt f32.add f32.sub f32.mul f32.div '
        'f32.min f32.max f32.copysign f64.abs f64.neg f64.ceil f64.floor f64.trunc f64.nearest '
        'f64.sqrt f64.add f64.sub f64.mul f64.div f64.min f64.max f64.copysign i32.wrap_i64 '
        'i32.trunc_f32_s i32.trunc_f32_u i32.trunc_f64_s i32.trunc_f64_u i64.extend_i32_s '
        'i64.extend_i32_u i64.trunc_f32_s i64.trunc_f32_u i64.trunc_f64_s i64.trunc_f64_u '
        'f32.convert_i32_s f32.convert_i32_u f32.convert_i64_s f32.convert_i64_u f32.demote_f64 '
        'f64.convert_i32_s f64.convert_i32_u f64.convert_i64_s f64.convert_i64_u f64.promote_f32 '
        'i32.reinterpret_f32 i64.reinterpret_f64 f32.reinterpret_i32 f64.reinterpret_i64 '
        'i32.extend8_s i32.extend16_s i64.extend8_s i64.extend16_s i64.extend32_s').split()
NUMERIC = {0x45 + i: n for i, n in enumerate(_NUM)}
MEM = {0x28: 'i32.load', 0x29: 'i64.load', 0x2A: 'f32.load', 0x2B: 'f64.load', 0x2C: 'i32.load8_s',
       0x2D: 'i32.load8_u', 0x2E: 'i32.load16_s', 0x2F: 'i32.load16_u', 0x30: 'i64.load8_s',
       0x31: 'i64.load8_u', 0x32: 'i64.load16_s', 0x33: 'i64.load16_u', 0x34: 'i64.load32_s',
       0x35: 'i64.load32_u', 0x36: 'i32.store', 0x37: 'i64.store', 0x38: 'f32.store',
       0x39: 'f64.store', 0x3A: 'i32.store8', 0x3B: 'i32.store16', 0x3C: 'i64.store8',
       0x3D: 'i64.store16', 0x3E: 'i64.store32'}
SAT = ['i32.trunc_sat_f32_s', 'i32.trunc_sat_f32_u', 'i32.trunc_sat_f64_s', 'i32.trunc_sat_f64_u',
       'i64.trunc_sat_f32_s', 'i64.trunc_sat_f32_u', 'i64.trunc_sat_f64_s', 'i64.trunc_sat_f64_u']
VT = {0x7F: 'i32', 0x7E: 'i64', 0x7D: 'f32', 0x7C: 'f64', 0x70: 'funcref'}


class _R:
    def __init__(self, b, i=0):
        self.b, self.i = b, i

    def byte(self):
        self.i += 1
        return self.b[self.i - 1]

    def u(self):
        r = s = 0
        while True:
            x = self.byte()
            r |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return r

    def s(self):
        r = s = 0
        while True:
            x = self.byte()
            r |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return r - (1 << s) if x & 0x40 else r

    def name(self):
        n = self.u()
        self.i += n
        return bytes(self.b[self.i - n:self.i]).decode('utf-8', 'replace')


def _fmt_f(x):
    r = repr(x)
    return r if 'e' in r or 'n' in r else r.rstrip('0').rstrip('.') if '.' in r else r


def disassemble(mod, only=None):
    """WAT text for the module's functions named in `only` (all if None), and a summary."""
    mod = bytes(mod)
    r = _R(mod, 8)
    types, nimp, imports, ftypes, exports, bodies, names, sections = [], 0, [], [], {}, [], {}, []
    while r.i < len(mod):
        sid = r.byte()
        size = r.u()
        end = r.i + size
        if sid == 0:
            nm = r.name()
            sections.append(('custom "%s"' % nm, size))
            if nm == 'name':
                while r.i < end:
                    sub, sz = r.byte(), r.u()
                    if sub == 1:
                        for _ in range(r.u()):
                            k = r.u()
                            names[k] = r.name()
                    else:
                        r.i += sz
        else:
            sections.append(({1: 'type', 2: 'import', 3: 'function', 7: 'export', 9: 'elem',
                              10: 'code', 11: 'data'}.get(sid, str(sid)), size))
        if sid == 1:
            for _ in range(r.u()):
                r.byte()
                p = [VT[r.byte()] for _ in range(r.u())]
                q = [VT[r.byte()] for _ in range(r.u())]
                types.append((p, q))
        elif sid == 2:
            for _ in range(r.u()):
                m, f = r.name(), r.name()
                kind = r.byte()
                if kind == 0:
                    imports.append((m, f, r.u()))
                elif kind == 1:
                    r.byte(); fl = r.byte(); r.u()
                    if fl & 1: r.u()
                elif kind == 2:
                    fl = r.byte(); r.u()
                    if fl & 1: r.u()
                elif kind == 3:
                    r.byte(); r.byte()
            nimp = len(imports)
        elif sid == 3:
            ftypes = [r.u() for _ in range(r.u())]
        elif sid == 7:
            for _ in range(r.u()):
                nm = r.name()
                kind = r.byte()
                idx = r.u()
                if kind == 0:
                    exports[idx] = nm
        elif sid == 10:
            for _ in range(r.u()):
                sz = r.u()
                bodies.append((r.i, r.i + sz))
                r.i += sz
        r.i = end
    for k, (m, f, _) in enumerate(imports):
        names.setdefault(k, f)

    def fname(k):
        return '$' + names.get(k, 'f%d' % k)

    out = []
    for n, (a, b) in enumerate(bodies):
        k = nimp + n
        nm = names.get(k, 'f%d' % k)
        if only is not None and nm not in only:
            continue
        p, q = types[ftypes[n]]
        head = '(func %s' % fname(k)
        if k in exports:
            head += ' (export "%s")' % exports[k]
        if p:
            head += ' (param %s)' % ' '.join(p)
        if q:
            head += ' (result %s)' % ' '.join(q)
        out.append(head)
        c = _R(mod, a)
        loc = []
        for _ in range(c.u()):
            cnt, t = c.u(), VT[c.byte()]
            loc.append('%d %s' % (cnt, t) if cnt > 1 else t)
        if loc:
            out.append('  (local %s)' % ' '.join(loc))
        depth = 1
        while c.i < b:
            op = c.byte()
            ind = '  ' * depth
            if op == 0x0B:
                depth -= 1
                if depth == 0:
                    break
                out.append('  ' * depth + 'end')
                continue
            if op in (0x02, 0x03, 0x04):
                bt = c.byte()
                out.append(ind + {2: 'block', 3: 'loop', 4: 'if'}[op] +
                           ('' if bt == 0x40 else ' (result %s)' % VT[bt]))
                depth += 1
            elif op == 0x05:
                out.append('  ' * (depth - 1) + 'else')
            elif op in (0x0C, 0x0D):
                out.append(ind + ('br %d' if op == 0x0C else 'br_if %d') % c.u())
            elif op == 0x0E:
                ls = [c.u() for _ in range(c.u())]
                out.append(ind + 'br_table %s %d' % (' '.join(map(str, ls)), c.u()))
            elif op in (0x00, 0x01, 0x0F, 0x1A, 0x1B):
                out.append(ind + {0: 'unreachable', 1: 'nop', 0x0F: 'return', 0x1A: 'drop',
                                  0x1B: 'select'}[op])
            elif op == 0x10:
                out.append(ind + 'call ' + fname(c.u()))
            elif op == 0x11:
                t = c.u(); c.u()
                out.append(ind + 'call_indirect (type %d)' % t)
            elif 0x20 <= op <= 0x24:
                out.append(ind + ['local.get', 'local.set', 'local.tee', 'global.get',
                                  'global.set'][op - 0x20] + ' %d' % c.u())
            elif op in MEM:
                al, off = c.u(), c.u()
                out.append(ind + MEM[op] + (' offset=%d' % off if off else ''))
            elif op == 0x41 or op == 0x42:
                out.append(ind + ('i32.const %d' if op == 0x41 else 'i64.const %d') % c.s())
            elif op == 0x43:
                out.append(ind + 'f32.const ' + _fmt_f(struct.unpack('<f', mod[c.i:c.i + 4])[0]))
                c.i += 4
            elif op == 0x44:
                out.append(ind + 'f64.const ' + _fmt_f(struct.unpack('<d', mod[c.i:c.i + 8])[0]))
                c.i += 8
            elif op in NUMERIC:
                out.append(ind + NUMERIC[op])
            elif op == 0xFC:
                sub = c.u()
                if sub < 8:
                    out.append(ind + SAT[sub])
                elif sub == 10:
                    c.i += 2
                    out.append(ind + 'memory.copy')
                elif sub == 11:
                    c.i += 1
                    out.append(ind + 'memory.fill')
                else:
                    out.append(ind + ';; 0xfc %d' % sub)
            else:
                out.append(ind + ';; unknown opcode 0x%02x' % op)
                break
        out.append(')')
        out.append('')
    others = sorted(names.get(nimp + n, '') for n in range(len(bodies))
                    if only is not None and names.get(nimp + n) not in only)
    return '\n'.join(out), sections, [o for o in others if o]
