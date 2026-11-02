"""Environment source -> checked -> spliced into a training loop and a planner -> compiled.

An environment is untrusted text. It is parsed, never imported: constants are read with literal_eval,
the three function bodies must pass the whitelist, and they are pasted into TEMPLATE so that the whole
training loop compiles as one function (scramjit has no calls between compiled functions).
"""
import ast
import importlib.util
import os
import sys
import tempfile

import numpy as np

ALLOWED = (ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass,
           ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Subscript, ast.Tuple,
           ast.Name, ast.Constant, ast.Attribute, ast.Load, ast.Store, ast.operator, ast.unaryop, ast.boolop,
           ast.cmpop, ast.keyword)
CALLS = {'range', 'min', 'max', 'abs', 'int', 'float', 'bool', 'len'}
MATH = {'sqrt', 'floor', 'ceil', 'sin', 'cos', 'exp', 'log', 'pi', 'hypot', 'atan2', 'fabs'}
PARAMS = {'reset': ['s', 'grid', 'obs', 'rnd', 'w', 'h'],
          'step': ['s', 'grid', 'obs', 'rnd', 'w', 'h', 'action'],
          'observe': ['s', 'grid', 'obs', 'w', 'h']}
CONSTS = {'NAME': str, 'DESCRIPTION': str, 'W': int, 'H': int, 'ACTIONS': int, 'MAX_STEPS': int, 'OBS': int,
          'GLYPHS': dict, 'AGENT_X': int, 'AGENT_Y': int}


class EnvError(Exception):
    pass


def _check_body(fn):
    for st in fn.body:
        for n in ast.walk(st):
            if not isinstance(n, ALLOWED):
                raise EnvError(f'{fn.name}, line {getattr(n, "lineno", "?")}: {type(n).__name__} is not allowed')
            if isinstance(n, ast.Name) and (n.id.startswith('t_') or n.id.startswith('_')):
                raise EnvError(f'{fn.name}: names starting with t_ or _ are reserved: {n.id}')
            if isinstance(n, ast.Attribute):
                if not (isinstance(n.value, ast.Name) and n.value.id == 'math' and n.attr in MATH):
                    raise EnvError(f'{fn.name}, line {n.lineno}: only math.{{{",".join(sorted(MATH))}}} are allowed')
            if isinstance(n, ast.Call):
                ok = (isinstance(n.func, ast.Name) and n.func.id in CALLS) or isinstance(n.func, ast.Attribute)
                if not ok:
                    raise EnvError(f'{fn.name}, line {n.lineno}: only calls to {sorted(CALLS)} and math.* are allowed')
            if isinstance(n, ast.Subscript) and any(isinstance(x, ast.Slice) for x in ast.walk(n.slice)):
                raise EnvError(f'{fn.name}, line {n.lineno}: slices are not allowed')
            if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float, bool)):
                raise EnvError(f'{fn.name}, line {n.lineno}: only numeric constants are allowed')


def parse_env(text):
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise EnvError(f'syntax error: {e}') from None
    consts, fns = {}, {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            try:
                consts[st.targets[0].id] = ast.literal_eval(st.value)
            except ValueError:
                raise EnvError(f'{st.targets[0].id} must be a literal') from None
        elif isinstance(st, ast.FunctionDef):
            fns[st.name] = st
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            pass
        else:
            raise EnvError(f'line {st.lineno}: only constants and the three functions are allowed at top level')
    for k, t in CONSTS.items():
        if not isinstance(consts.get(k), t):
            raise EnvError(f'missing constant {k} ({t.__name__})')
    if not (2 <= consts['ACTIONS'] <= 8 and 1 <= consts['OBS'] <= 64 and consts['W'] * consts['H'] <= 4096
            and 1 <= consts['MAX_STEPS'] <= 2000):
        raise EnvError('ACTIONS 2-8, OBS 1-64, W*H <= 4096, MAX_STEPS <= 2000')
    for name, params in PARAMS.items():
        fn = fns.get(name)
        if fn is None:
            raise EnvError(f'missing function {name}')
        got = [a.arg for a in fn.args.args]
        if got != params:
            raise EnvError(f'{name} must take ({", ".join(params)}), got ({", ".join(got)})')
        _check_body(fn)
    return consts, fns


def _body(fn, indent):
    out = []
    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue    # docstring
        out.extend(ast.unparse(st).splitlines())
    return '\n'.join(' ' * indent + line for line in out) or ' ' * indent + 'pass'


RECORD = '''        if t_i < n_rec:
            for t_j in range(h):
                for t_k in range(w):
                    rec[t_i, t_j, t_k] = grid[t_j, t_k]
            recf[t_i, 0] = s[ax]
            recf[t_i, 1] = s[ay]
            recf[t_i, 2] = reward
            recf[t_i, 3] = 1.0 if done or t_len >= max_steps else 0.0
            recf[t_i, 4] = action
'''

TEMPLATE = '''import math
{IMPORT}

{DECORATOR}
def run(q, s, grid, obs, rnd, stats, rec, recf, n_rec, ax, ay, w, h, n_actions, n_rnd, qmask, max_steps, n_steps,
        seed, eps_m, alpha, gamma, learn):
    t_x = seed
    t_need = True
    t_h = 0
    t_ret = 0.0
    t_len = 0
    t_eps = 0
    t_sum = 0.0
    t_wins = 0
    for t_i in range(n_steps):
        if t_need:
            for t_j in range(n_rnd):
                t_x = (t_x * 1103515245 + 12345) & 2147483647
                rnd[t_j] = t_x // 256
{RESET}
{OBSERVE0}
            t_h = 0
            for t_j in range(len(obs)):
                t_h = (t_h * 1000003 + int(obs[t_j])) & qmask
            t_need = False
            t_ret = 0.0
            t_len = 0
        t_x = (t_x * 1103515245 + 12345) & 2147483647
        t_o = (t_x // 256) % n_actions
        t_x = (t_x * 1103515245 + 12345) & 2147483647
        action = t_o
        if (t_x // 256) % 1000 >= eps_m:
            t_b = q[t_h, t_o]
            for t_j in range(1, n_actions):
                t_a = (t_o + t_j) % n_actions
                if q[t_h, t_a] > t_b:
                    t_b = q[t_h, t_a]
                    action = t_a
        for t_j in range(8):
            t_x = (t_x * 1103515245 + 12345) & 2147483647
            rnd[t_j] = t_x // 256
        reward = 0.0
        done = False
        win = False
{STEP}
        t_len += 1
{RECORD}{OBSERVE1}
        t_n = 0
        for t_j in range(len(obs)):
            t_n = (t_n * 1000003 + int(obs[t_j])) & qmask
        if learn:
            t_b = q[t_n, 0]
            for t_j in range(1, n_actions):
                if q[t_n, t_j] > t_b:
                    t_b = q[t_n, t_j]
            if done:
                t_b = 0.0
            q[t_h, action] += alpha * (reward + gamma * t_b - q[t_h, action])
        t_h = t_n
        t_ret += reward
        if win:
            t_wins += 1
        if done or t_len >= max_steps:
            t_eps += 1
            t_sum += t_ret
            t_need = True
    stats[0] = t_eps
    stats[1] = t_sum
    stats[2] = t_wins
    return t_x
'''


PLANNER = '''import math
{IMPORT}

{DECORATOR}
def plan(q, s, grid, obs, rnd, stats, s0, g0, score, rec, recf, n_rec, ax, ay, w, h, n_actions, n_rnd, qmask, max_steps,
         n_steps, seed, rollouts, depth, eps_m, gamma, fresh):
    t_x = seed
    t_need = fresh
    if stats[5] > 0.5:
        t_need = True
    t_len = int(stats[3])
    t_ret = stats[1] if not fresh else 0.0
    t_sum = 0.0
    t_eps = 0
    t_wins = 0
    t_h = 0
    t_real = 0
    t_best = 0
    for t_i in range(n_steps):
        if t_need:
            for t_j in range(n_rnd):
                t_x = (t_x * 1103515245 + 12345) & 2147483647
                rnd[t_j] = t_x // 256
{RESET}
            t_need = False
            t_len = 0
            t_ret = 0.0
        for t_j in range(len(s)):
            s0[t_j] = s[t_j]
        for t_j in range(h):
            for t_k in range(w):
                g0[t_j, t_k] = grid[t_j, t_k]
        for t_j in range(n_actions):
            score[t_j] = 0.0
        t_a = n_actions if rollouts == 0 else 0     # no rollouts: every score ties, so a random move
        t_r = 0
        t_real = 0
        while t_real == 0:
            # n_actions * rollouts simulations, then one real step with the best action
            if t_a >= n_actions:
                t_real = 1
                t_best = (t_x // 256) % n_actions
                t_b = score[t_best]
                for t_j in range(n_actions):
                    if score[t_j] > t_b:
                        t_b = score[t_j]
                        t_best = t_j
                for t_j in range(len(s)):
                    s[t_j] = s0[t_j]
                for t_j in range(h):
                    for t_k in range(w):
                        grid[t_j, t_k] = g0[t_j, t_k]
            t_g = 0.0
            t_disc = 1.0
            action = t_best if t_real == 1 else t_a
            t_steps = 1 if t_real == 1 else depth
            for t_d in range(t_steps):
                for t_j in range(8):
                    t_x = (t_x * 1103515245 + 12345) & 2147483647
                    rnd[t_j] = t_x // 256
                reward = 0.0
                done = False
                win = False
{STEP}
                t_g += t_disc * reward
                t_disc *= gamma
                if t_real == 1:
                    t_ret += reward
                    t_len += 1
                    if t_i < n_rec:
                        for t_j in range(h):
                            for t_k in range(w):
                                rec[t_i, t_j, t_k] = grid[t_j, t_k]
                        recf[t_i, 0] = s[ax]
                        recf[t_i, 1] = s[ay]
                        recf[t_i, 2] = reward
                        recf[t_i, 3] = 1.0 if done or t_len >= max_steps else 0.0
                        recf[t_i, 4] = action
                        recf[t_i, 5] = 1.0 if win else 0.0
                    if win:
                        t_wins += 1
                    if done or t_len >= max_steps:
                        t_eps += 1
                        t_sum += t_ret
                        t_need = True
                    break
                if done:
                    break
{OBSERVE}
                t_h = 0
                for t_j in range(len(obs)):
                    t_h = (t_h * 1000003 + int(obs[t_j])) & qmask
                t_x = (t_x * 1103515245 + 12345) & 2147483647
                t_o = (t_x // 256) % n_actions
                t_x = (t_x * 1103515245 + 12345) & 2147483647
                action = t_o
                if (t_x // 256) % 1000 >= eps_m:
                    t_b = q[t_h, t_o]
                    for t_j in range(1, n_actions):
                        t_c = (t_o + t_j) % n_actions
                        if q[t_h, t_c] > t_b:
                            t_b = q[t_h, t_c]
                            action = t_c
            if t_real == 0:
                score[t_a] += t_g
                for t_j in range(len(s)):
                    s[t_j] = s0[t_j]
                for t_j in range(h):
                    for t_k in range(w):
                        grid[t_j, t_k] = g0[t_j, t_k]
                t_r += 1
                if t_r >= rollouts:
                    t_r = 0
                    t_a += 1
    stats[0] = t_eps
    stats[1] = t_ret if not t_need else 0.0
    stats[2] = t_wins
    stats[3] = 0 if t_need else t_len
    stats[5] = 1.0 if t_need else 0.0
    stats[4] = t_sum
    return t_x
'''


def build_planner(fns, compiled=True, options='fallback=False, cache=False'):
    return PLANNER.format(
        IMPORT='from scramjit import scram' if compiled else '',
        DECORATOR=f'@scram({options})' if compiled else '',
        RESET=_body(fns['reset'], 12),
        STEP=_body(fns['step'], 16),
        OBSERVE=_body(fns['observe'], 16))


def load_plan(source, tag='plan'):
    d = tempfile.mkdtemp(prefix='rlproto')
    path = os.path.join(d, f'{tag}.py')
    with open(path, 'w') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(f'{tag}_{id(path)}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.plan


def build_source(fns, compiled=True, options='fallback=False, cache=False', record=False):
    return TEMPLATE.format(
        RECORD=RECORD if record else '',
        IMPORT='from scramjit import scram' if compiled else '',
        DECORATOR=f'@scram({options})' if compiled else '',
        RESET=_body(fns['reset'], 12),
        OBSERVE0=_body(fns['observe'], 12),
        STEP=_body(fns['step'], 8),
        OBSERVE1=_body(fns['observe'], 8))


def load_run(source, tag='env'):
    d = tempfile.mkdtemp(prefix='rlproto')
    path = os.path.join(d, f'{tag}.py')
    with open(path, 'w') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(f'{tag}_{id(path)}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.run


class Lab:
    QBITS = 18

    def __init__(self, consts, run):
        self.c = consts
        self.run = run
        self.q = np.zeros((1 << self.QBITS, consts['ACTIONS']))
        self.s = np.zeros(32, np.int64)
        self.grid = np.zeros((consts['H'], consts['W']), np.int64)
        self.obs = np.zeros(consts['OBS'], np.int64)
        self.rnd = np.zeros(1024, np.int64)
        self.stats = np.zeros(8)
        self.seed = 12345
        self.rec = np.zeros((1, consts['H'], consts['W']), np.int64)
        self.recf = np.zeros((1, 6))
        self.s0 = np.zeros(32, np.int64)
        self.g0 = np.zeros((consts['H'], consts['W']), np.int64)
        self.score = np.zeros(consts['ACTIONS'])

    def planned(self, plan, n, rollouts=8, depth=30, eps_m=100, gamma=0.99, n_rec=0, fresh=True):
        """n real steps, each chosen by simulating every action with the world's own step. fresh=False continues
        the episode in progress, so a page can stream moves. Returns mean return of finished episodes and how many."""
        c = self.c
        if n_rec:
            self.rec = np.zeros((n_rec, c['H'], c['W']), np.int64)
            self.recf = np.zeros((n_rec, 6))
        if fresh:
            self.stats[:] = 0
        self.seed = int(plan(self.q, self.s, self.grid, self.obs, self.rnd, self.stats, self.s0, self.g0, self.score,
                             self.rec, self.recf, n_rec, c['AGENT_X'], c['AGENT_Y'], c['W'], c['H'], c['ACTIONS'], 1024,
                             (1 << self.QBITS) - 1, c['MAX_STEPS'], n, self.seed, rollouts, depth, eps_m, gamma, fresh))
        eps = self.stats[0]
        self.win_rate = self.stats[2] / eps if eps else 0.0
        return self.stats[4] / eps if eps else 0.0, int(eps)

    def record(self, plain_run, frames, eps_m=50):
        """An episode replay, through the undecorated source: 400 frames take milliseconds."""
        c = self.c
        fast, self.run = self.run, plain_run
        self.rec = np.zeros((frames, c['H'], c['W']), np.int64)
        self.recf = np.zeros((frames, 5))
        self.go(frames, eps_m, False, n_rec=frames)
        self.run = fast
        return self.rec, self.recf

    def go(self, n, eps_m, learn, alpha=0.2, gamma=0.99, n_rec=0):
        c = self.c
        self.seed = int(self.run(self.q, self.s, self.grid, self.obs, self.rnd, self.stats, self.rec, self.recf,
                                 n_rec, c['AGENT_X'], c['AGENT_Y'], c['W'], c['H'], c['ACTIONS'], 1024,
                                 (1 << self.QBITS) - 1, c['MAX_STEPS'], n, self.seed, eps_m, alpha, gamma, learn))
        eps = self.stats[0]
        self.win_rate = self.stats[2] / eps if eps else 0.0
        return self.stats[1] / eps if eps else 0.0, int(eps)
