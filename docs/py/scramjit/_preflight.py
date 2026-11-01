"""Preflight: which functions compile, and which lines keep them from it.

Static analysis reads source without importing or running it. A run reports what happened with the
real argument types. A function compiles whole or falls back whole until loop lifting exists, so
"partly compiled" doesn't occur yet. Rendering is in cli/report.py.
"""

import ast
import copy
import math
import os
import re

from . import _types as T
from ._compiler import Compiler
from ._errors import Unsupported

_OTHER = object()     # a module-level name whose value static analysis can't know
SKIP_DIRS = {'__pycache__', 'node_modules', 'venv', 'env', 'site-packages', 'build', 'dist'}

# What to do about a reason: the first match wins. Hints must be something to act on.
HINTS = [
    (r'^dict', 'if the keys are small integers, use an array indexed by key; otherwise do this outside the compiled function'),
    (r'^sets|^set comp', 'if the values are small integers, use a boolean array indexed by value'),
    (r'^f-strings|calls (str|print|repr|format)\(\)', 'keep text and printing out of the function: return numbers and format them after the call'),
    (r'slices|views', 'loop over the indices you want: for i in range(lo, hi): a[i]'),
    (r'^calls [\w.]*\.\w+\(\)', 'write it as a loop; methods and numpy calls aren\'t compiled'),
    (r'^calls \w+\(\)', 'put its body here, or decorate and call it from Python; a compiled function can\'t call Python functions'),
    (r'empty list|a list assigned', 'allocate a numpy array of the final size and pass it in'),
    (r'must keep one type', 'use a new name for the new type'),
    (r'global or undefined name', 'pass it in as an argument'),
    (r'keeps? one type|would be None', 'make it the same type on every path'),
    (r'multi-dimensional array', 'index the rows: for i in range(rows): a[i, j]'),
    (r'closures', 'pass what it reads as arguments, and define it at module level'),
    (r'expression statement', 'assign the result, or move the call out of the function'),
    (r'is not available', 'define it in a file, not in exec() or a one-line REPL session'),
]


def hint_for(reason):
    for pattern, hint in HINTS:
        if re.search(pattern, reason):
            return hint
    return None


class Problem:
    """Why one line keeps a function in Python."""

    __slots__ = ('line', 'col', 'width', 'reason', 'hint')

    def __init__(self, line, col, width, reason):
        self.line, self.col, self.width, self.reason = line, col, width, reason
        self.hint = hint_for(reason)

    def as_dict(self):
        return {'line': self.line, 'column': self.col, 'reason': self.reason, 'hint': self.hint}


class Analysis:
    """One function under one signature. compiler is None when it doesn't compile."""

    __slots__ = ('signature', 'compiler', 'problems')

    def __init__(self, signature, compiler, problems):
        self.signature, self.compiler, self.problems = signature, compiler, problems

    @property
    def state(self):
        return 'compiled' if self.compiler is not None else 'python'


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


def is_scram(d):
    d = d.func if isinstance(d, ast.Call) else d
    return (isinstance(d, ast.Name) and d.id == 'scram') or (isinstance(d, ast.Attribute) and d.attr == 'scram')


def options(fdef):
    opts = {'boundscheck': True, 'parity': True}
    for d in fdef.decorator_list:
        if isinstance(d, ast.Call) and is_scram(d):
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


def _bound_names(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def analyze(fdef, sig, globals_, filename, offset, opts, first_only=False):
    """Compile as far as possible. A failing statement is set aside and compiling retried, so every
    line that keeps the function in Python is found, not only the first. A failure that only reads a
    name a set-aside statement would have bound follows from that one, so it isn't reported."""
    tree = copy.deepcopy(fdef)
    problems = []
    gone = set()
    for _ in range(25):
        try:
            c = Compiler(tree, list(sig), globals_, filename, offset, opts['boundscheck'], opts['parity'])
            c.generate()
            return Analysis(sig, c if not problems else None, problems)
        except Unsupported as u:
            node, line = u.node, u.lineno
            reads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if isinstance(node, ast.AST) else set()
            if not (reads & gone) and not any(f"'{g}'" in u.reason for g in gone):
                col = getattr(node, 'col_offset', None)
                same = getattr(node, 'end_lineno', None) == line
                width = (node.end_col_offset - col) if col is not None and same else 1
                problems.append(Problem((line or 1) + offset, col, width, u.reason))
            where = _innermost_stmt(tree, line) if line and line > tree.lineno and not first_only else None
            if where is None:
                if not problems:
                    problems.append(Problem((line or 1) + offset, None, 1, u.reason))
                return Analysis(sig, None, problems)
            stmts, i = where
            gone |= _bound_names(stmts[i])
            stmts[i] = ast.copy_location(ast.Pass(), stmts[i])
    return Analysis(sig, None, problems)


def best_analysis(fdef, globals_, filename, offset, opts, first_only=False):
    """Analyze under the assumed signature. An unannotated array whose values are used as indices
    isn't float64, so before calling a function uncompilable, try its arrays as int64 too."""
    sig = assumed_signature(fdef)
    first = None
    arrays = [i for i, t in enumerate(sig) if t.kind == 'array']
    for mask in range(1 << min(len(arrays), 4)):
        trial = list(sig)
        for bit, i in enumerate(arrays[:4]):
            if mask >> bit & 1:
                trial[i] = T.array(T.NP_INT64, True, sig[i].ndim)
        a = analyze(fdef, trial, globals_, filename, offset, opts, first_only)
        if a.compiler is not None:
            return a
        first = first or a
    return first


class FileReport:
    __slots__ = ('path', 'source', 'functions', 'error', 'candidates')

    def __init__(self, path, source, functions, error=None, candidates=()):
        self.path, self.source, self.functions, self.error = path, source, functions, error
        self.candidates = candidates


def _is_candidate(fdef, path):
    """Worth suggesting @scram for: an undecorated module-level function with a loop to speed up, and
    not a test."""
    base = os.path.basename(path)
    if fdef.decorator_list or fdef.name.startswith('test_') or base.startswith('test_') or base == 'conftest.py':
        return False
    a = fdef.args
    if not (a.posonlyargs or a.args):
        return False                                    # nothing comes in, so nothing to specialize on
    return any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(fdef))


def read_file(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def scan_file(path, suggest=True):
    """A FileReport for every @scram function in a file, and the undecorated functions that would
    compile, without importing it; None if it has neither."""
    source = read_file(path)
    try:
        tree = ast.parse(source, path)
    except SyntaxError as e:
        return FileReport(path, source, [], e)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(is_scram(d) for d in n.decorator_list)]
    g = module_globals(tree)
    candidates = []
    if suggest:
        for f in tree.body:
            if isinstance(f, ast.FunctionDef) and _is_candidate(f, path):
                a = best_analysis(f, g, path, 0, options(f), first_only=True)
                if a.compiler is not None:
                    candidates.append((f, a))
    if not funcs and not candidates:
        return None
    return FileReport(path, source, [(f, best_analysis(f, g, path, 0, options(f))) for f in funcs], candidates=candidates)


def python_files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS)
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith('.py')]
    return out


def has_loop(fdef):
    return any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(fdef))


def subscript_lines(fdef, offset):
    return {n.lineno + offset for n in ast.walk(fdef) if isinstance(n, ast.Subscript)}


def loop_lines(fdef, offset):
    return {n + offset for s in ast.walk(fdef) if isinstance(s, (ast.For, ast.While))
            for b in s.body for n in range(b.lineno, (b.end_lineno or b.lineno) + 1)}
