"""Preflight: which functions compile, and which lines keep them from it.

Static (`scramjit preflight PATH`) reads source without importing or running it. With `--run` a
program runs and the report covers what happened with the real argument types. `f.preflight()`
reports one function. A function compiles whole or falls back whole until loop lifting exists, so
"partly compiled" doesn't occur yet.
"""

import ast
import importlib.util
import math
import os
import runpy
import sys
import textwrap
import time

from . import _dispatch, _types as T
from ._compiler import Compiler
from ._errors import Unsupported

GREEN, RED = '\U0001f7e2', '\U0001f534'
RULE = '─' * 45
_OTHER = object()     # a module-level name whose value static analysis can't know


# ---- static analysis
def module_globals(tree):
    """What compiling needs from a module, without running it: math, and which names are taken."""
    out = {}
    for s in tree.body:
        if isinstance(s, ast.Import):
            for a in s.names:
                out[(a.asname or a.name).split('.')[0]] = math if a.name == 'math' else _OTHER
        elif isinstance(s, ast.ImportFrom):
            for a in s.names:
                out[a.asname or a.name] = getattr(math, a.name, _OTHER) if s.module == 'math' else _OTHER
        elif isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            out[s.name] = _OTHER
        elif isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(s.value)
                    except ValueError:
                        out[t.id] = _OTHER
    return out


def _is_scram(d):
    d = d.func if isinstance(d, ast.Call) else d
    return (isinstance(d, ast.Name) and d.id == 'scram') or (isinstance(d, ast.Attribute) and d.attr == 'scram')


def _options(fdef):
    opts = {'boundscheck': True, 'parity': True}
    for d in fdef.decorator_list:
        if isinstance(d, ast.Call) and _is_scram(d):
            for k in d.keywords:
                if k.arg in opts and isinstance(k.value, ast.Constant):
                    opts[k.arg] = bool(k.value.value)
    return opts


def assumed_signature(fdef):
    """Argument types from annotations; the rest are assumed to be numbers or float64 arrays, by how
    the function uses them."""
    ann = {'float': T.FLOAT, 'int': T.INT, 'bool': T.BOOL}
    ndim = {}
    for n in ast.walk(fdef):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            k = len(n.slice.elts) if isinstance(n.slice, ast.Tuple) else 1
            ndim[n.value.id] = max(ndim.get(n.value.id, 0), k)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in ('shape', 'size', 'ndim'):
            ndim.setdefault(n.value.id, 1)
        elif isinstance(n, ast.For) and isinstance(n.iter, ast.Name):
            ndim.setdefault(n.iter.id, 1)
        elif isinstance(n, ast.comprehension) and isinstance(n.iter, ast.Name):
            ndim.setdefault(n.iter.id, 1)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('len', 'enumerate', 'zip', 'reversed'):
            for a in n.args:
                if isinstance(a, ast.Name):
                    ndim.setdefault(a.id, 1)
    # rows, cols = a.shape says how many dimensions a has
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Attribute) \
                and n.value.attr == 'shape' and isinstance(n.value.value, ast.Name):
            ndim[n.value.value.id] = max(ndim.get(n.value.value.id, 1), len(n.targets[0].elts))
    sig = []
    a = fdef.args
    for p in a.posonlyargs + a.args:
        name = ast.unparse(p.annotation) if p.annotation is not None else None
        if name in ann:
            sig.append(ann[name])
        elif p.arg in ndim:
            sig.append(T.array(T.NP_FLOAT64, True, ndim[p.arg]))
        else:
            sig.append(T.FLOAT)
    return sig


def best_analysis(fdef, globals_, filename, offset, opts):
    """Analyze under the assumed signature. An unannotated array whose values are used as indices
    isn't float64, so before calling a function uncompilable, try its arrays as int64 too. Returns
    (signature, compiler or None, problems)."""
    sig = assumed_signature(fdef)
    first = None
    arrays = [i for i, t in enumerate(sig) if t.kind == 'array']
    for mask in range(1 << min(len(arrays), 4)):
        trial = list(sig)
        for bit, i in enumerate(arrays[:4]):
            if mask >> bit & 1:
                trial[i] = T.array(T.NP_INT64, True, sig[i].ndim)
        c, problems = analyze(fdef, trial, globals_, filename, offset, opts)
        if c is not None:
            return trial, c, problems
        first = first or (trial, c, problems)
    return first


def _innermost_stmt(fdef, line):
    """The list holding the innermost statement on this line, and its index."""
    found = None
    for node in ast.walk(fdef):
        for field in ('body', 'orelse', 'finalbody'):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                for i, s in enumerate(stmts):
                    if isinstance(s, ast.stmt) and s.lineno <= line <= (s.end_lineno or s.lineno):
                        if found is None or s.lineno >= found[0][found[1]].lineno:
                            found = (stmts, i)
    return found


def analyze(fdef, sig, globals_, filename, offset, opts):
    """Compile as far as possible. Returns (compiler or None, [(line, reason)...]). A failing
    statement is set aside and compiling retried, so every line that keeps the function in
    Python is found, not only the first."""
    import copy
    tree = copy.deepcopy(fdef)
    problems = []
    for _ in range(25):
        c = None
        try:
            c = Compiler(tree, list(sig), globals_, filename, offset, opts['boundscheck'], opts['parity'])
            c.generate()
            return (c if not problems else None), problems
        except Unsupported as u:
            line = u.lineno
            problems.append(((line or 1) + offset, u.reason))
            where = _innermost_stmt(tree, line) if line and line > tree.lineno else None
            if where is None:
                return None, problems
            stmts, i = where
            stmts[i] = ast.copy_location(ast.Pass(), stmts[i])
    return None, problems


def scan_file(path):
    """(FunctionDef, globals) for each @scram function in a file, without importing it."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    try:
        tree = ast.parse(src, path)
    except SyntaxError:
        return []
    g = module_globals(tree)
    return [(n, g) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(_is_scram(d) for d in n.decorator_list)]


def python_files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith('.py')]
    return out


def static_report(paths, out):
    """The report for every @scram function under these paths. Returns how many aren't compiled."""
    good = bad = 0
    for p in paths:
        for path in python_files(p):
            funcs = scan_file(path)
            if not funcs:
                continue
            print(path, file=out)
            width = max(len(f.name) for f, _ in funcs)
            for fdef, g in funcs:
                _, c, problems = best_analysis(fdef, g, path, 0, _options(fdef))
                if c is not None:
                    good += 1
                    print(f'  {GREEN} {fdef.name:<{width}}  compiled', file=out)
                else:
                    bad += 1
                    print(f'  {RED} {fdef.name:<{width}}  Python', file=out)
                    for line, why in problems:
                        print(f'      {line:>4}  {why}', file=out)
    print(f'\n{good} compiled · {bad} Python', file=out)
    return bad


# ---- one function, line by line
def _subscript_lines(fdef, offset):
    return {n.lineno + offset for n in ast.walk(fdef) if isinstance(n, ast.Subscript)}


def _loop_lines(fdef, offset):
    return {n + offset for s in ast.walk(fdef) if isinstance(s, (ast.For, ast.While))
            for b in s.body for n in range(b.lineno, (b.end_lineno or b.lineno) + 1)}


def line_report(lines, first, fdef, offset, title, status, bounds, problems, types, out, hints=()):
    print(title, file=out)
    print(status, file=out)
    print(RULE, file=out)
    kept, hoisted, checked = bounds or ({}, (), True)
    indexed = _subscript_lines(fdef, offset)
    in_loop = _loop_lines(fdef, offset)
    bad = {}
    for line, why in problems:
        bad.setdefault(line, why)
    body = fdef.body
    if len(body) > 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]                                 # a docstring isn't code
    body_start = body[0].lineno + offset
    width = max(len(s) for s in lines) if lines else 0
    hints = list(hints)
    for i, text in enumerate(lines):
        n = first + i
        if n < body_start or not text.strip():
            mark, note = ' ', ''
        elif problems:
            mark, note = RED, bad.get(n, '')
        else:
            mark, note = GREEN, ''
            if n in indexed and not checked:
                note = 'bounds checks off'
            elif n in indexed:
                k = kept.get(n, 0)
                note = (f'{k} bounds check{"s" if k != 1 else ""} kept' if k
                        else 'bounds hoisted out of the loop' if n in hoisted else 'bounds proven')
                if k and n in in_loop:
                    hints.append(f'line {n}: a bounds check runs on every iteration; index with the loop variable '
                                 'over range(len(a)), or use @scram(boundscheck=False) once verified')
        print(f'{n:>4} {mark} {text:<{width}}  {note}'.rstrip(), file=out)
    print(RULE, file=out)
    if types:
        print('types  ' + ', '.join(f'{k}: {v}' for k, v in types.items()), file=out)
    for h in dict.fromkeys(hints):
        print(f'hint   {h}', file=out)


def function_report(f, signature, out):
    """f.preflight(): every signature it has seen (or the one given), else one assumed signature."""
    fdef = f._tree
    lines = []
    if fdef is not None:
        try:
            import inspect
            lines = textwrap.dedent(inspect.getsource(f._fn)).splitlines()
        except (OSError, TypeError):
            pass
    offset = f._first - 1
    where = os.path.basename(f._filename)
    keys = [tuple(signature)] if signature is not None else list(f._cache)
    if fdef is None:
        print(f'{f.__name__}\nPython · {where}\n{f._why_not}', file=out)
        return
    if not keys:
        trial, c, problems = best_analysis(fdef, f._fn.__globals__, f._filename, offset,
                                           {'boundscheck': f._boundscheck, 'parity': f._parity})
        sig = ', '.join(t.name for t in trial)
        state = 'compiled' if c is not None else 'Python'
        line_report(lines, f._first, fdef, offset, f'{f.__name__}({sig})', f'{state} · {where} · assumed signature, not called yet',
                    (c.checks, c.hoisted_lines, f._boundscheck) if c else None, problems,
                    {k: t.name for k, t in c.env.items()} if c else {}, out)
        return
    for key in keys:
        entry = f._cache.get(key)
        names = ', '.join(t.name if t is not None else '?' for t in key)
        if isinstance(entry, _dispatch.Compiled):
            ms = f._compile_s.get(key)
            when = f' · compiled in {ms * 1000:.0f} ms' if ms is not None else ''
            line_report(lines, f._first, fdef, offset, f'{f.__name__}({names})', f'compiled · {where}{when}',
                        (entry.checks, entry.hoisted, f._boundscheck), [], entry.locals, out)
        else:
            line, why = f._why.get(key, (f._first, 'not compiled'))
            line_report(lines, f._first, fdef, offset, f'{f.__name__}({names})', f'Python · {where}', None, [(line, why)], {}, out)


def target_report(target, out):
    """`scramjit preflight FILE:FUNCTION` (or pkg.module:function), statically."""
    path, _, name = target.rpartition(':')
    if not os.path.exists(path):
        spec = importlib.util.find_spec(path)
        if spec is None or not spec.origin:
            print(f'scramjit: no file or module {path}', file=sys.stderr)
            return 2
        path = spec.origin
    with open(path, encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src, path)
    g = module_globals(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            sig, c, problems = best_analysis(n, g, path, 0, _options(n))
            start = n.decorator_list[0].lineno if n.decorator_list else n.lineno
            text = src.splitlines()[start - 1:n.end_lineno]
            state = 'compiled' if c is not None else 'Python'
            line_report(text, start, n, 0, f'{name}({", ".join(t.name for t in sig)})',
                        f'{state} · {os.path.basename(path)} · assumed signature',
                        (c.checks, c.hoisted_lines, _options(n)['boundscheck']) if c else None, problems,
                        {k: t.name for k, t in c.env.items()} if c else {}, out)
            return 0 if c is not None else 1
    print(f'scramjit: no function {name} in {path}', file=sys.stderr)
    return 2


# ---- a program, run
def run_program(args):
    """Run a script (or -m module) with its own argv. Returns its exit status and how long it took."""
    _dispatch.enable_profile()
    t0 = time.perf_counter()
    status = 0
    saved = sys.argv
    try:
        if args[0] == '-m':
            sys.argv = args[1:]
            runpy.run_module(args[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = args
            sys.path.insert(0, os.path.dirname(os.path.abspath(args[0])))
            runpy.run_path(args[0], run_name='__main__')
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved
    return status, time.perf_counter() - t0


def run_report(total, out):
    """The table for what ran. Returns how many functions weren't compiled."""
    rows = []
    for f in sorted(_dispatch.functions(), key=lambda f: f.__name__):
        st = f._stats
        if not st:
            continue
        calls = sum(s[0] for s in st.values())
        native = sum(s[1] for s in st.values())
        python = sum(s[2] for s in st.values())
        comp = sum(f._compile_s.get(k, 0.0) for k in st)
        sigs = len(st)
        compiled = [k for k in st if isinstance(f._cache.get(k), _dispatch.Compiled)]
        rows.append((f, calls, comp, native, python, sigs, len(compiled)))
    native = sum(r[3] for r in rows)
    print(f'\nscramjit · {total:.2f} s run · {native / total * 100 if total else 0:.0f}% native', file=out)
    secs = lambda v: f'{v:.2f} s' if v else '–'  # noqa: E731
    width = max([len(r[0].__name__) + (3 if r[5] > 1 else 0) for r in rows] + [8])
    print(f'   {"function":<{width}}  {"calls":>6}  {"compile":>7}  {"native":>7}  {"python":>7}', file=out)
    bad = 0
    hints = []
    for f, calls, comp, nat, py, sigs, ok in rows:
        name = f.__name__ + (f' ×{sigs}' if sigs > 1 else '')
        mark = GREEN if ok == sigs else RED
        bad += ok != sigs
        print(f'{mark} {name:<{width}}  {calls:>6,}  {secs(comp):>7}  {secs(nat):>7}  {secs(py):>7}', file=out)
        for k in f._stats:
            if k in f._why:
                line, why = f._why[k]
                hints.append(f'{f.__name__}, line {line}: {why}')
        if sigs > 1:
            per = [{i: t.name for i, t in enumerate(k) if t is not None} for k in f._stats]
            for i in sorted(set().union(*per)):
                seen = {p.get(i) for p in per}
                if len(seen) > 1:
                    names = f._fn.__code__.co_varnames
                    hints.append(f'{f.__name__}: {names[i]} arrives as {" and ".join(sorted(s for s in seen if s))}; one compile per type')
    if not rows:
        print('no @scram function was called', file=out)
    if hints:
        print('\nhints', file=out)
        for h in hints:
            print(f'  {h}', file=out)
    failed = [r[0].__name__ for r in rows if r[6] != r[5]]
    if failed:
        print(f'\nfailed: {", ".join(failed)} {"is" if len(failed) == 1 else "are"} not compiled', file=out)
    return bad


def main(argv=None, out=None):
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = 'usage: scramjit preflight [--allow-partial] [PATH|FILE:FUNCTION ...] [--run SCRIPT|-m MODULE [ARGS ...]]'
    if not argv or argv[0] != 'preflight':
        print(usage, file=sys.stderr)
        return 2
    argv = argv[1:]
    run = None
    if '--run' in argv:
        i = argv.index('--run')
        argv, run = argv[:i], argv[i + 1:]
        if not run or (run[0] == '-m' and len(run) < 2):
            print(usage, file=sys.stderr)
            return 2
    allow_partial = '--allow-partial' in argv     # accepted: nothing is partly compiled until loop lifting
    if allow_partial:
        argv.remove('--allow-partial')
    if any(a.startswith('-') for a in argv):
        print(usage, file=sys.stderr)
        return 2
    targets = [a for a in argv if ':' in a and not os.path.exists(a)]
    paths = [a for a in argv if a not in targets]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f'scramjit: no such path: {missing[0]}', file=sys.stderr)
        return 2
    if run is not None:
        status, total = run_program(run)
        if targets:
            for t in targets:
                name = t.rpartition(':')[2]
                for f in _dispatch.functions():
                    if f.__name__ == name:
                        function_report(f, None, out)
            bad = sum(not all(isinstance(f._cache.get(k), _dispatch.Compiled) for k in f._stats)
                      for f in _dispatch.functions() if f.__name__ in {t.rpartition(':')[2] for t in targets})
        else:
            bad = run_report(total, out)
        return status or (1 if bad else 0)
    if not targets and not paths:
        print(usage, file=sys.stderr)
        return 2
    rc = 0
    for t in targets:
        rc = max(rc, target_report(t, out))
    if paths:
        rc = max(rc, 1 if static_report(paths, out) else 0)
    return rc
