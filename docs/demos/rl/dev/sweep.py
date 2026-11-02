"""Win rate at each thinking level, untrained and guided by a trained table, for one world in a run file."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path[:0] = [RL, os.path.join(RL, '..', '..', 'py')]
import envlab
LEVELS = [1, 100, 1000, 10000, 100000, 1000000]
def rd(c, b):
    if b <= 1: return 0, 1
    d = max(2, min(60, c['MAX_STEPS'], int(b ** .5)))
    return max(1, b // (c['ACTIONS'] * d)), d
def sweep(c, plan, q, seconds):
    out = []
    for b in LEVELS:
        lab = envlab.Lab(c, None); lab.stats[5] = 1.0
        if q is not None: lab.q = q
        wins = eps = 0; R, D = rd(c, b); t = time.perf_counter()
        n = 1 if b >= 100000 else 20
        while time.perf_counter() - t < seconds * (3 if b >= 100000 else 1):
            _, e = lab.planned(plan, n, rollouts=R, depth=D, eps_m=100 if q is not None else 1000, fresh=False)
            wins += int(lab.stats[2]); eps += e
        out.append(f'{wins/eps:4.0%}/{eps:<4}' if eps else '  -/0   ')
    return out
e = json.load(open(sys.argv[1]))
e = [x for x in e if x['name'] == sys.argv[2]][0]
c, f = envlab.parse_env(e['code'])
plan = envlab.load_plan(envlab.build_planner(f, options='fallback=False, cache=False, boundscheck=False'))
run = envlab.load_run(envlab.build_source(f))
a = sweep(c, plan, None, 1.5)
l = envlab.Lab(c, run)
for i in range(10): l.go(2_000_000, max(30, 200 - 20 * i), True, alpha=max(0.02, 0.3 * 0.7 ** i))
l.go(100_000, 30, False); tw = l.win_rate
b = sweep(c, plan, l.q, 1.5)
print(f"{c['NAME']:<20} max {c['MAX_STEPS']:<4} plain: " + ' '.join(a) + f"\n{'':<20} table alone {tw:4.0%} guided: " + ' '.join(b), flush=True)
