"""The live demo's driver, run in Pyodide: terrain, presets, and each pipeline stage's output."""

import ast
import json
import linecache
import time
import warnings

import numpy as np

import scramjit
from scramjit._runtime import PRELUDE

N = 128
PRESETS = {
    'erosion': {'entry': 'erode', 'unit': 'droplets'},
    'beacon': {'entry': 'viewshed', 'unit': 'sightlines'},     # one per cell: a line of sight
    'landslides': {'entry': 'slide', 'unit': 'cells'},
}
S = {'k': 1, 'row': 0, 'angle': 0.0, 'rates': {}}


def terrain(seed):
    """Fractal value noise (numpy, vectorized: this part isn't the demo), squared into valleys."""
    rng = np.random.default_rng(seed)
    h = np.zeros((N, N))
    amp = 1.0
    for octave in range(6):
        k = 2 ** (octave + 2)
        g = rng.random((k + 1, k + 1))
        t = np.linspace(0, k, N)
        i = np.minimum(t.astype(int), k - 1)
        f = t - i
        rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
        h += (rows[:, i] * (1 - f) + rows[:, i + 1] * f) * amp
        amp *= 0.5
    h = (h - h.min()) / (h.max() - h.min())
    ridges = 1 - np.abs(2 * h - 1)                    # ridged: sharp crests, long valleys
    h = 0.55 * ridges ** 2 + 0.45 * h ** 1.6
    return (h - h.min()) / (h.max() - h.min())


def reset(seed=7):
    S['h'] = terrain(seed)
    S['vis'] = np.zeros((N, N), np.uint8)
    S['rng'] = np.random.default_rng(seed)
    S['row'] = 0


def outline(node, depth=0, out=None, limit=60):
    """A compact syntax tree: node types with names, one per line."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    label = type(node).__name__
    for attr in ('name', 'id', 'arg', 'attr'):
        if isinstance(getattr(node, attr, None), str):
            label += f' {getattr(node, attr)}'
            break
    if isinstance(node, ast.Constant):
        label += f' {node.value!r}'
    if hasattr(node, 'op'):
        label += f' {type(node.op).__name__}'
    if not isinstance(node, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
        out.append('  ' * depth + label)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
                outline(child, depth + 1, out, limit)
    return out


def set_code(preset, src, compiled):
    """Load the editor's code, as Python or through scramjit; return what each stage shows."""
    S['preset'] = preset
    S['k'] = 1
    name = f'<{preset}.py>'
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    info = {'state': 'python'}
    try:
        tree = ast.parse(src, name)
        ns = {}
        exec(compile(tree, name, 'exec'), ns)
        fn = ns[PRESETS[preset]['entry']]
    except Exception as e:  # noqa: BLE001
        S['fn'] = None
        return json.dumps({'state': 'error', 'error': f'{type(e).__name__}: {e}'})
    lines = outline(tree.body[-1])
    info['ast'] = '\n'.join(lines) + ('\n  …' if len(lines) >= 60 else '')
    S['fn'] = fn
    if not compiled:
        return json.dumps(info)
    f = scramjit.scram(fn)
    t = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            run(f, 1, warm=True)          # compiles on the first call
    except Exception as e:  # noqa: BLE001
        S['fn'] = None
        return json.dumps({'state': 'error', 'error': f'{type(e).__name__}: {e}', 'ast': info['ast']})
    info['compile_ms'] = round((time.perf_counter() - t) * 1000)
    fell = [str(x.message) for x in w if issubclass(x.category, scramjit.FallbackWarning)]
    S['fn'] = f
    if fell:
        info.update(state='fallback', reason=fell[0])
        return json.dumps(info)
    entry = next(v for v in f._cache.values() if hasattr(v, 'source'))
    info.update(state='compiled', c=entry.source,
                types='\n'.join(f'{k}: {v}' for k, v in entry.locals.items()))
    return json.dumps(info)


def wasm_text():
    """The compiled function as WebAssembly text: cimo builds the module from the same C (its cache
    makes that quick) and cimo's disassembler prints scramjit's kernel."""
    f = S.get('fn')
    entry = next((v for v in getattr(f, '_cache', {}).values() if hasattr(v, 'source')), None)
    if entry is None:
        return 'Turn scramjit on to compile.'
    import cimo.wasm
    import wat
    # not cimo's documented API yet: see TODO.md ("Workarounds waiting on cimo")
    mod, _ = cimo.wasm.build(PRELUDE + entry.source)
    text, sections, others = wat.disassemble(mod, {'sj_kernel'})
    head = [f';; scramjit\'s kernel as WebAssembly text, disassembled from the {len(mod) / 1024:.1f} KB module',
            ';; cimo compiled in this page (sections: ' + ', '.join(f'{n.replace("custom ", "")} {z} B' for n, z in sections) + ')',
            f';; the module also holds {len(others)} runtime functions', '']
    return '\n'.join(head) + text


def run(f, k, warm=False):
    """Do k units of the current preset's work with f."""
    p = S['preset']
    if p == 'erosion':
        seeds = S['rng'].random(2 * k)
        f(S['h'] if not warm else S['h'].copy(), seeds, 0.05, 4.0, 0.3, 0.3, 0.02, 40)
        return k
    if p == 'beacon':
        a = S['angle']
        oy, ox = int(N / 2 + np.sin(a) * N * 0.3), int(N / 2 + np.cos(a) * N * 0.3)
        r0 = S['row'] if not warm else 0
        r1 = min(N, r0 + k)
        f(S['h'], S['vis'] if not warm else S['vis'].copy(), oy, ox, 0.08, r0, r1 if not warm else 1)
        if not warm:
            S['row'] = 0 if r1 >= N else r1
            if r1 >= N:
                S['angle'] = a + 0.12
        return (r1 - r0) * N                 # cells: one sightline each
    r0 = S['row'] if not warm else 0     # landslides
    r1 = min(N, r0 + k)
    f(S['h'] if not warm else S['h'].copy(), 0.012, 0.25, r0, r1 if not warm else 2)
    if not warm:
        S['row'] = 0 if r1 >= N else r1
    return (r1 - r0) * N                     # cells


def step(budget_ms):
    """Work for about budget_ms; return units done per second. The batch size adapts."""
    if S.get('fn') is None:
        return json.dumps({'rate': 0})
    t = time.perf_counter()
    done = 0
    while (time.perf_counter() - t) * 1000 < budget_ms:
        k = S['k']
        t1 = time.perf_counter()
        try:
            done += run(S['fn'], k)
        except Exception as e:  # noqa: BLE001
            S['fn'] = None
            return json.dumps({'rate': 0, 'error': f'{type(e).__name__}: {e}'})
        dt = max(time.perf_counter() - t1, 1e-5) * 1000
        S['k'] = max(1, min(int(k * min(2.0, (budget_ms / 3) / dt)) or 1, 20000))
    rate = done / max(time.perf_counter() - t, 1e-6)
    return json.dumps({'rate': rate, 'unit': PRESETS[S['preset']]['unit']})


def heights():
    return S['h'].astype(np.float32)


def overlay():
    return S['vis'] if S.get('preset') == 'beacon' else None


reset()
