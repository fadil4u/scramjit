"""Evaluate one environment: runs in its own process under a memory limit, killed by the parent on timeout."""
import json, os, resource, sys, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path[:0] = [RL, os.path.join(RL, '..', '..', 'py')]    # the same scramjit and cimo the page loads
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
import numpy as np
import envlab
TRAIN_CHUNKS, CHUNK = 10, 1_000_000


def evaluate(text):
    out = {}
    try:
        c, f = envlab.parse_env(text)
    except envlab.EnvError as e:
        return {'stage': 'check', 'error': str(e)}
    out.update(name=c['NAME'], description=c['DESCRIPTION'], w=c['W'], h=c['H'], glyphs=c['GLYPHS'], max_steps=c['MAX_STEPS'],
               actions=c['ACTIONS'])
    src = envlab.build_source(f)
    try:
        t = time.perf_counter()
        run = envlab.load_run(src)
        lab = envlab.Lab(c, run)
        lab.go(1, 1000, False)
        out['compile_s'] = round(time.perf_counter() - t, 2)
    except Exception as e:
        return {**out, 'stage': 'compile', 'error': f'{type(e).__name__}: {str(e)[-400:]}'}
    try:
        lab = envlab.Lab(c, run)
        random_r, random_eps = lab.go(300_000, 1000, False)
        out.update(random_return=round(random_r, 4), random_episodes=random_eps, random_win=round(lab.win_rate, 3))
        curve = []
        t0 = time.perf_counter()
        train_t = 0.0
        for i in range(TRAIN_CHUNKS):
            eps = max(30, 200 - 20 * i)
            alpha = max(0.02, 0.3 * 0.7 ** i)
            t = time.perf_counter()
            lab.go(CHUNK, eps, True, alpha=alpha)
            train_t += time.perf_counter() - t
            g, ge = lab.go(100_000, 30, False)
            curve.append({'steps': (i + 1) * CHUNK, 'seconds': round(time.perf_counter() - t0, 3), 'greedy': round(g, 4), 'episodes': ge})
        out['curve'] = curve
        out['steps_per_s'] = round(TRAIN_CHUNKS * CHUNK / train_t)
        g, _ = lab.go(300_000, 30, False)
        out.update(greedy_return=round(g, 4), trained_win=round(lab.win_rate, 3))
        rec, recf = lab.record(envlab.load_run(envlab.build_source(f, compiled=False, record=True), 'replay'), min(400, c['MAX_STEPS'] * 3))
        out['frames'] = rec.tolist()
        out['framef'] = recf.tolist()
    except Exception as e:
        return {**out, 'stage': 'run', 'error': f'{type(e).__name__}: {str(e)[-400:]}'}
    # the same source undecorated: speed, and whether the results match bit for bit
    try:
        plain = envlab.load_run(envlab.build_source(f, compiled=False), 'plain')
        a, b = envlab.Lab(c, run), envlab.Lab(c, plain)
        n = 20_000
        a.go(n, 100, True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = time.perf_counter()
            b.go(n, 100, True)
            py = time.perf_counter() - t
        out['python_steps_per_s'] = round(n / py)
        out['parity'] = bool(np.array_equal(a.q, b.q) and a.seed == b.seed)
    except Exception as e:
        out['parity_error'] = f'{type(e).__name__}: {str(e)[-200:]}'
    # planning: bounds checks stay on here, so every world that ships was exercised with them
    try:
        t = time.perf_counter()
        plan = envlab.load_plan(envlab.build_planner(f))
        lab2 = envlab.Lab(c, run)
        lab2.planned(plan, 1, rollouts=1, depth=1)
        out['plan_compile_s'] = round(time.perf_counter() - t, 2)
        lab2.q[:] = 0
        t = time.perf_counter()
        pr, pe = lab2.planned(plan, 4000, rollouts=16, depth=40, eps_m=1000)
        out.update(plan_return=round(pr, 4), plan_win=round(lab2.win_rate, 3), plan_episodes=pe,
                   plan_moves_per_s=round(4000 / (time.perf_counter() - t)))
    except Exception as e:
        return {**out, 'stage': 'run', 'error': f'planner: {type(e).__name__}: {str(e)[-400:]}'}
    best = max(out['trained_win'], out['plan_win'])
    if out['random_episodes'] > 100_000:
        verdict = 'degenerate'     # episodes end almost at once
    elif out['random_win'] > 0.5:
        verdict = 'too-easy'       # acting at random already wins
    elif best >= 0.8:
        verdict = 'solved'
    elif best >= 0.3:
        verdict = 'partial'
    else:
        verdict = 'unsolved'
    return {**out, 'stage': 'done', 'verdict': verdict}


if __name__ == '__main__':
    res = evaluate(open(sys.argv[1]).read())
    json.dump(res, open(sys.argv[2], 'w'))
