"""The worker's Python side. One function per thing the page asks for; every result is JSON."""
import json
import time

import envlab

# Bounds checks multiply the planner's code through loop versioning (16 s to compile, against 2.5 s). Off here, a bad
# index can only hurt this tab's WebAssembly heap. Worlds that ship were run with checks on: see dev/worker.py.
PLANNER_OPTIONS = 'fallback=False, cache=False, boundscheck=False'
PLANS, TRAINERS = {}, {}
S = {}


def _fail(stage, e):
    return json.dumps({'error': f'{type(e).__name__}: {str(e)[-400:]}', 'stage': stage})


def load(code):
    """Parse and compile the planner, and start an episode."""
    try:
        c, f = envlab.parse_env(code)
    except envlab.EnvError as e:
        return json.dumps({'error': str(e), 'stage': 'check'})
    t = time.perf_counter()
    try:
        if code not in PLANS:
            PLANS[code] = envlab.load_plan(envlab.build_planner(f, options=PLANNER_OPTIONS))
        lab = envlab.Lab(c, None)
        lab.planned(PLANS[code], 1, rollouts=0, depth=1, eps_m=1000)
    except Exception as e:
        return _fail('compile', e)
    S.clear()
    S.update(code=code, c=c, f=f, lab=lab, plan=PLANS[code], trained=False, evals={}, tallies={})
    return json.dumps({'name': c['NAME'], 'description': c['DESCRIPTION'], 'glyphs': c['GLYPHS'], 'w': c['W'], 'h': c['H'],
                       'actions': c['ACTIONS'], 'seconds': time.perf_counter() - t})


def _rollouts(budget):
    """Simulated steps per move -> rollouts per action and their depth."""
    if budget <= 1:
        return 0, 1
    n = S['c']['ACTIONS']
    depth = max(2, min(60, S['c']['MAX_STEPS'], int(budget ** 0.5)))
    return max(1, budget // (n * depth)), depth


def moves(n, budget, use_q):
    """n real moves, thinking `budget` simulated steps for each."""
    lab = S['lab']
    rollouts, depth = _rollouts(budget)
    eps = 100 if use_q and S['trained'] else 1000
    t = time.perf_counter()
    try:
        _, done = lab.planned(S['plan'], n, rollouts=rollouts, depth=depth, eps_m=eps, n_rec=n, fresh=False)
    except Exception as e:
        return _fail('run', e)
    return json.dumps({'frames': lab.rec.tolist(), 'f': lab.recf.tolist(), 'episodes': done, 'wins': int(lab.stats[2]),
                       'simulated': n * rollouts * depth * S['c']['ACTIONS'], 'seconds': time.perf_counter() - t})


def tally(budget, ms):
    """Play unseen episodes at one thinking level for about ms milliseconds. Returns that level's running count."""
    t = S['tallies'].setdefault(budget, {'wins': 0, 'episodes': 0, 'seconds': 0.0, 'simulated': 0})
    lab = S['evals'].get(budget)
    if lab is None:
        lab = S['evals'][budget] = envlab.Lab(S['c'], None)
        lab.stats[5] = 1.0       # start with a reset
    lab.q = S['lab'].q
    rollouts, depth = _rollouts(budget)
    eps = 100 if S['trained'] else 1000
    per_move = max(1, rollouts * depth * S['c']['ACTIONS'])
    n = max(1, min(400, int(S.get('rate', 5e6) * ms / 1000 / per_move / 2)))
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < ms / 1000:
            t0 = time.perf_counter()
            _, done = lab.planned(S['plan'], n, rollouts=rollouts, depth=depth, eps_m=eps, fresh=False)
            t['episodes'] += done
            t['wins'] += int(lab.stats[2])
            t['simulated'] += n * rollouts * depth * S['c']['ACTIONS']
            S['rate'] = max(1e5, n * per_move / max(1e-6, time.perf_counter() - t0))
    except Exception as e:
        return _fail('run', e)
    t['seconds'] += time.perf_counter() - start
    return json.dumps({'budget': budget, **t})


def train_start():
    """Compile the training loop, with bounds checks, since it runs the world tens of millions of times."""
    t = time.perf_counter()
    try:
        code = S['code']
        if code not in TRAINERS:
            TRAINERS[code] = envlab.load_run(envlab.build_source(S['f']))
        lab = envlab.Lab(S['c'], TRAINERS[code])
        lab.go(1, 1000, False)
        random_r, _ = lab.go(200_000, 1000, False)
    except Exception as e:
        return _fail('compile', e)
    S.update(train_lab=lab, steps=0, spent=0.0)
    return json.dumps({'seconds': time.perf_counter() - t, 'random': random_r, 'random_win': lab.win_rate})


def train_chunk(i):
    lab = S['train_lab']
    try:
        t = time.perf_counter()
        lab.go(2_000_000, max(30, 200 - 20 * i), True, alpha=max(0.02, 0.3 * 0.7 ** i))
        S['spent'] += time.perf_counter() - t
        S['steps'] += 2_000_000
        ret, _ = lab.go(100_000, 30, False)
    except Exception as e:
        return _fail('run', e)
    return json.dumps({'steps': S['steps'], 'ret': float(ret), 'win': lab.win_rate, 'rate': S['steps'] / S['spent']})


def train_done():
    S['lab'].q = S['train_lab'].q
    S['trained'] = True
    S['tallies'] = {}
