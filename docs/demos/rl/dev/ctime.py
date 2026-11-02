"""Where a generated function's compile time goes: size of the C, and time in scramjit vs cimo."""
import os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path[:0] = [RL, os.path.join(RL, '..', '..', 'py')]
import cimo, envlab
c, f = envlab.parse_env(open(HERE + '/keydoor.py').read())
orig = cimo.compile
LOG = {}
def spy(src, *a, **k):
    LOG['c_lines'] = src.count('\n'); LOG['c_loops'] = len(re.findall(r'\bfor\s*\(|\bwhile\s*\(', src)); LOG['src'] = src
    t = time.perf_counter(); r = orig(src, *a, **k); LOG['cimo'] = time.perf_counter() - t
    return r
cimo.compile = spy
import scramjit._dispatch as D
for name in dir(D):
    if getattr(getattr(D, name), '__name__', '') == 'compile' and getattr(D, name) is orig: setattr(D, name, spy)
def measure(label, build, load, call):
    LOG.clear(); fn = load(build)
    t = time.perf_counter(); call(fn); tot = time.perf_counter() - t
    print(f"{label:<34} total {tot:5.1f} s  cimo {LOG.get('cimo', 0):5.1f} s  C lines {LOG.get('c_lines')}  loops {LOG.get('c_loops')}", flush=True)
    return LOG.get('src', '')
def tr(fn): envlab.Lab(c, fn).go(1, 1000, False)
def pl(fn): envlab.Lab(c, run).planned(fn, 1, rollouts=1, depth=1)
run = envlab.load_run(envlab.build_source(f))
a = measure('train', envlab.build_source(f), envlab.load_run, tr)
open('/tmp/claude-1000/train.c', 'w').write(a)
p = measure('planner', envlab.build_planner(f), envlab.load_plan, pl)
open('/tmp/claude-1000/plan.c', 'w').write(p)
measure('planner, no bounds checks', envlab.build_planner(f, options='fallback=False, cache=False, boundscheck=False'), envlab.load_plan, pl)
