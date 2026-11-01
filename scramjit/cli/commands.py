"""The `scramjit` command: its command tree and what each command does."""

import ast
import difflib
import importlib.util
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
import warnings

from .parse import Command, UsageError
from .term import Spinner


def build_tree():
    from .. import __version__
    root = Command('scramjit', f'scramjit {__version__}, Python at native speed',
                   description='A JIT written in pure Python. Decorate a function with `@scram` and its loops run '
                               'as machine code. This command shows what compiles, and why not when it doesn\'t.')
    root.opt('-h', '--help', flag=True, help='Show help for a command')
    root.opt('--version', flag=True, help='Show the version and exit')
    root.opt('--color', metavar='when', choices=('auto', 'always', 'never'), default='auto', help='Colored output')
    root.example('scramjit preflight src/', 'what compiles, and which lines stop the rest')
    root.example('scramjit preflight --run -m pytest', 'what happened with the real argument types')
    root.example('scramjit c src/sim.py:ema', 'the C generated for a function')

    p = root.command(
        'preflight', 'Show what compiles, and which lines stop the rest', func=cmd_preflight, aliases=('check',),
        description='Reads every `@scram` function under the paths without importing or running anything. '
                    'Annotations are used; unannotated arguments are assumed to be numbers or arrays. Also lists '
                    'undecorated functions with a loop that would compile if decorated; these don\'t affect the '
                    'exit status.\n\n'
                    'With `--run`, runs a program and reports what happened with the real argument types: '
                    'the fallbacks static analysis can\'t see, and where the time went.\n\n'
                    'Exits with 1 if any function runs as Python, so it works as a CI check.')
    p.arg('paths', 'Files or directories (default: the current directory), or `FILE:FUNCTION` for one function '
                   'line by line', nargs='*')
    p.opt('--run', metavar='program', help='Run this script, or `-m module`, with the arguments after it. '
                                           'The report goes to stderr; the program keeps stdout')
    p.opt('-q', '--quiet', flag=True, help='Show only what doesn\'t compile')
    p.opt('--no-suggest', flag=True, help='Don\'t list undecorated functions that would compile')
    p.opt('--json', flag=True, help='Print the report as JSON')
    p.opt('--allow-partial', flag=True, help='Accept partly compiled functions (none exist until loop lifting)')
    p.example('scramjit preflight src/')
    p.example('scramjit preflight src/sim.py:ema', 'one function, line by line')
    p.example('scramjit preflight --run simulate.py --steps 2000')
    p.example('scramjit preflight --run -m pytest tests/', 'in CI: every function, with the types it really gets')

    c = root.command(
        'c', 'Print the C generated for a function', func=cmd_c,
        description='Generates C for one function without running anything, from its annotations and, for '
                    'the rest, assumed types. `#line` directives point back at your Python.')
    c.arg('function', '`FILE:FUNCTION` or `package.module:FUNCTION`')
    c.opt('--runtime', flag=True, help='Include the runtime the C is compiled with')
    c.example('scramjit c src/sim.py:ema')
    c.example('scramjit c src/sim.py:ema --runtime > ema.c')

    cache = root.command(
        'cache', 'Inspect or clear the compile cache',
        description='Compiled code is kept in cimo\'s cache, keyed by content, so a second process doesn\'t '
                    'compile again.')
    cache.command('info', 'Show where the cache is and how big it is', func=_cache('info'))
    clear = cache.command('clear', 'Delete cached files', func=_cache('clear'))
    clear.opt('--modules', flag=True, help='Only compiled modules')
    clear.opt('--runtime', flag=True, help='Only runtime library builds')
    cache.command('path', 'Print the cache directory', func=_cache('path'))

    comp = root.command(
        'completion', 'Print a shell completion script', func=cmd_completion,
        description='Prints a script that makes your shell complete scramjit commands, options and files.')
    comp.arg('shell', 'bash, zsh or fish', complete=None)
    comp.example('scramjit completion bash >> ~/.bashrc')
    comp.example('scramjit completion zsh > "${fpath[1]}/_scramjit"')
    comp.example('scramjit completion fish > ~/.config/fish/completions/scramjit.fish')

    h = root.command('help', 'Show help for a command', func=cmd_help)
    h.arg('command', 'Command name, e.g. `preflight` or `cache clear`', nargs='*', complete=None)

    root.command('__complete', 'Shell completion backend', func=cmd_complete, hidden=True) \
        .arg('words', nargs='rest', complete=None)
    return root


class Failure(Exception):
    """A command failed; the message was printed. status: the exit status."""

    def __init__(self, status=1):
        Exception.__init__(self, status)
        self.status = status


def _error(ctx, message, hint=None, suggestion=None):
    e = ctx.err
    e.write(f'{e.style("error:", "error")} {e.style(message, "bold")}')
    if suggestion:
        e.write('')
        e.write(f'  Did you mean {e.style(suggestion, "bold")}?')
    if hint:
        e.write('')
        e.write(f'  {e.style("hint:", "hint")} {e.code(hint)}')


def _close_path(path):
    """A sibling of a path that doesn't exist, close to it."""
    d = os.path.dirname(path) or '.'
    try:
        near = difflib.get_close_matches(os.path.basename(path), os.listdir(d), n=1, cutoff=0.6)
    except OSError:
        return None
    return os.path.join(os.path.dirname(path), near[0]) if near else None


def _is_target(a):
    return ':' in a and not os.path.exists(a) and not a.startswith(':')


def _resolve(ctx, target):
    """FILE:FUNCTION -> (path, source, FunctionDef, globals), or a helpful failure."""
    from .. import _preflight as P
    where, _, name = target.rpartition(':')
    path = where
    if not os.path.exists(path):
        spec = None
        if not where.endswith('.py') and '/' not in where:
            try:
                spec = importlib.util.find_spec(where)
            except (ImportError, ValueError):
                spec = None
        if spec is None or not spec.origin:
            near = _close_path(where)
            _error(ctx, f'no such file or module: {where}', suggestion=f'{near}:{name}' if near else None)
            raise Failure(2)
        path = spec.origin
    source = P.read_file(path)
    try:
        tree = ast.parse(source, path)
    except SyntaxError as e:
        _error(ctx, f'{e.msg} in {path}:{e.lineno}')
        raise Failure(2) from None
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if name not in funcs:
        near = difflib.get_close_matches(name, list(funcs), n=1, cutoff=0.5)
        hint = None
        if not near and funcs:
            hint = 'functions in this file: ' + ', '.join(f'`{f}`' for f in sorted(funcs))
        _error(ctx, f'no function {name} in {path}', suggestion=f'{where}:{near[0]}' if near else None, hint=hint)
        raise Failure(2)
    return path, source, funcs[name], P.module_globals(tree)


def cmd_preflight(ctx):
    from .. import _preflight as P
    from . import report
    v = ctx.values
    if v.get('run') is not None:
        return _preflight_run(ctx, v['run'])
    args = v.get('paths') or []
    targets = [a for a in args if _is_target(a)]
    paths = [a for a in args if a not in targets] or ([] if targets else ['.'])
    for p in paths:
        if not os.path.exists(p):
            near = _close_path(p)
            _error(ctx, f'no such file or directory: {p}', suggestion=near)
            raise Failure(2)
    status = 0
    out = ctx.out
    for k, t in enumerate(targets):
        path, source, fdef, g = _resolve(ctx, t)
        a = P.best_analysis(fdef, g, path, 0, P.options(fdef))
        if k:
            out.write('')
        report.one_static(out, fdef, a, source, path, P.options(fdef)['boundscheck'])
        status = max(status, a.compiler is None)
    if not paths:
        return status
    files = [f for p in paths for f in P.python_files(p)]
    reports = []
    with Spinner(ctx.err, f'Reading {len(files)} files') if len(files) > 20 else _Nothing() as sp:
        for k, f in enumerate(files):
            if sp and sp.live:
                sp.update(f'Reading {f} ({k + 1}/{len(files)})')
            r = P.scan_file(f, suggest=not v.get('no_suggest'))
            if r is not None:
                reports.append(r)
    if v.get('json'):
        report.static_json(sys.stdout, reports)
        return max(status, any(r.error is not None or any(a.compiler is None for _, a in r.functions) for r in reports))
    if not reports:
        e = ctx.err
        e.write(f'{e.style("note:", "note")} no {e.style("@scram", "code")} functions under {", ".join(paths)}')
        e.write(f'  {e.style("hint:", "hint")} {e.code("decorate a function: `from scramjit import scram`, then `@scram`")}')
        return status
    good, bad, broken = report.static(out, reports, v.get('quiet'))
    if not (v.get('quiet') and not bad and not broken):
        out.write('')
    report.summary(out, good, bad, broken, 0 if v.get('quiet') else sum(len(r.candidates) for r in reports))
    return max(status, 1 if bad or broken else 0)


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _preflight_run(ctx, program):
    from .. import _dispatch
    from .. import FallbackWarning
    from . import report
    if program[0] == '-m' and len(program) < 2:
        raise UsageError('--run -m needs a module', ctx.command)
    if program[0] != '-m' and not os.path.exists(program[0]):
        near = _close_path(program[0])
        _error(ctx, f'no such script: {program[0]}', suggestion=near)
        raise Failure(2)
    _dispatch.enable_profile()
    warnings.filterwarnings('ignore', category=FallbackWarning)     # the report says it, with hints
    status = 0
    saved = sys.argv, list(sys.path)
    t0 = time.perf_counter()
    try:
        if program[0] == '-m':
            sys.argv = program[1:]
            sys.path.insert(0, os.getcwd())
            runpy.run_module(program[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = program
            sys.path.insert(0, os.path.dirname(os.path.abspath(program[0])))
            runpy.run_path(program[0], run_name='__main__')
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        status = 130
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        status = 1
    finally:
        sys.argv, sys.path[:] = saved[0], saved[1]
    total = time.perf_counter() - t0
    sys.stdout.flush()
    if ctx.values.get('json'):
        bad = report.run_json(sys.stderr, total, status)
    else:
        bad = report.run(ctx.err, total, status)
    if ctx.values.get('paths'):
        names = {a.rpartition(':')[2] for a in ctx.values['paths'] if ':' in a}
        for f in _dispatch.functions():
            if f.__name__ in names:
                ctx.err.write('')
                report.function_report(f, None, ctx.err.stream)
    return status or (1 if bad else 0)


def cmd_c(ctx):
    from .. import _preflight as P
    from .._runtime import PRELUDE
    from . import report
    target = ctx.values['function']
    if ':' not in target:
        raise UsageError(f"'{target}' doesn't name a function", ctx.command, hint='write it as `FILE:FUNCTION`, such as `src/sim.py:ema`')
    path, source, fdef, g = _resolve(ctx, target)
    a = P.best_analysis(fdef, g, path, 0, P.options(fdef))
    if a.compiler is None:
        e = ctx.err
        e.write(f'{e.style("error:", "error")} {e.style(fdef.name + " runs as Python, so it has no C", "bold")}')
        lines = source.splitlines()
        for p in a.problems:
            report._excerpt(e, lines, p, 2)
        return 1
    c = P.Compiler(fdef, list(a.signature), g, path, 0, P.options(fdef)['boundscheck'], P.options(fdef)['parity']).generate()
    ctx.err.write(ctx.err.style(f'// {fdef.name}({", ".join(t.name for t in a.signature)}), assumed from annotations and use', 'muted'))
    sys.stdout.write((PRELUDE if ctx.values.get('runtime') else '') + c)
    return 0


def _cache(sub):
    def run(ctx):
        args = [sys.executable, '-m', 'cimo', '--color', 'always' if ctx.out.color else 'never', 'cache', sub]
        if ctx.values.get('modules'):
            args.append('--modules')
        if ctx.values.get('runtime'):
            args.append('--runtime')
        return subprocess.call(args)
    return run


def cmd_help(ctx):
    from .parse import render_help
    cmd = ctx.command.root
    for w in ctx.values.get('command') or ():
        sub = cmd.find_command(w)
        if sub is None:
            raise UsageError(f"unknown command '{w}'", cmd,
                             difflib.get_close_matches(w, [c.name for c in cmd.visible_commands()], n=3, cutoff=0.6))
        cmd = sub
    ctx.out.write(render_help(cmd, ctx.out))
    return 0


def cmd_completion(ctx):
    from . import completion
    shell = ctx.values['shell']
    scripts = {'bash': completion.BASH, 'zsh': completion.ZSH, 'fish': completion.FISH}
    if shell not in scripts:
        raise UsageError(f"unsupported shell '{shell}'", ctx.command,
                         difflib.get_close_matches(shell, list(scripts), n=1, cutoff=0.5), hint='supported: bash, zsh, fish')
    sys.stdout.write(scripts[shell])
    return 0


def cmd_complete(ctx):
    from .parse import complete
    cands, kind = complete(ctx.command.root, ctx.values['words'] or [''])
    for c in cands:
        sys.stdout.write(c + '\n')
    if kind:
        sys.stdout.write(f'__{kind}__\n')
    return 0


def version_text():
    from .. import __version__
    try:
        import cimo
        with_cimo = f'cimo {cimo.__version__}, '
    except ImportError:
        with_cimo = ''
    return f'scramjit {__version__} ({with_cimo}Python {platform.python_version()}, {sys.platform} {platform.machine()})'
