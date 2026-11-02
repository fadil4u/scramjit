"""The curriculum: a model writes environments, each is compiled and trained in seconds, results go back to the model."""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
EMPTY = tempfile.mkdtemp(prefix='rlgen')    # no project files for the model to pick up
SYSTEM = open(HERE + '/prompt.md').read().replace('{EXAMPLE}', open(HERE + '/keydoor.py').read())
CLEAN = {k: v for k, v in os.environ.items() if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDE_CODE_SESSION_ID')}


def ask(model, prompt):
    t = time.time()
    r = subprocess.run(['claude', '-p', '--model', model, '--tools', '', '--no-session-persistence', '--system-prompt', SYSTEM, prompt],
                       cwd=EMPTY, env=CLEAN, capture_output=True, text=True, timeout=400)
    if r.returncode:
        raise RuntimeError(r.stderr[-300:] or r.stdout[-300:])
    m = re.search(r'```(?:python)?\n(.*?)```', r.stdout, re.S)
    return (m.group(1) if m else r.stdout), time.time() - t


def evaluate(code):
    with tempfile.TemporaryDirectory() as d:
        src, dst = d + '/env.py', d + '/out.json'
        open(src, 'w').write(code)
        try:
            p = subprocess.run([sys.executable, HERE + '/worker.py', src, dst], capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return {'stage': 'run', 'error': 'timed out after 240 s: an infinite loop in reset or step?'}
        if not os.path.exists(dst):
            return {'stage': 'run', 'error': f'worker died, exit {p.returncode}: {p.stderr[-300:]}'}
        return json.load(open(dst))


def summary(arch):
    if not arch:
        return 'None yet: only the example exists, and agents solve it.'
    lines = []
    for e in arch:
        if e.get('stage') == 'done':
            lines.append(f"- {e['name']}: {e['description']} [{e['verdict']}: wins, random {e['random_win']:.0%}, trained {e['trained_win']:.0%}, planning {e['plan_win']:.0%}]")
    return '\n'.join(lines) or 'None succeeded yet.'


def one(model, brief, arch, retries):
    prompt = f"Environments so far:\n{summary(arch)}\n\n{brief}"
    log = []
    fixes = 0
    code, secs = ask(model, prompt)
    t_llm = secs
    while True:
        r = evaluate(code)
        log.append({'stage': r.get('stage'), 'error': r.get('error')})
        if r.get('stage') == 'done' or fixes >= retries:
            break
        fixes += 1
        fix = f"{prompt}\n\nYour previous attempt:\n```python\n{code}\n```\nfailed at the {r['stage']} stage with:\n{r['error']}\n\nFix it. Output the whole corrected code block."
        code, secs = ask(model, fix)
        t_llm += secs
    r.update(code=code, attempts=log, fixes=fixes, llm_seconds=round(t_llm, 1))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--batch', type=int, default=3)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--out', default=HERE + '/runs/run2.json')
    a = ap.parse_args()
    arch = []
    briefs = ['Propose one new environment, different in its core mechanic from those so far and from the example, and a little harder than the hardest that was learned. Vary the mechanic: pushing blocks, moving hazards, switches and doors, resource gathering, stealth, timing, chasing, herding, building. Make it fun to watch.',
              'Propose one new environment whose core challenge needs a sequence of subgoals or planning ahead, unlike those so far. Make it fun to watch.',
              'Propose one new environment with something that moves on its own (enemies, currents, falling objects, other creatures) that the agent must react to, unlike those so far. Make it fun to watch.']
    t0 = time.time()
    for rnd in range(a.rounds):
        with cf.ThreadPoolExecutor(a.batch) as ex:
            fs = [ex.submit(one, a.model, briefs[i % len(briefs)], list(arch), a.retries) for i in range(a.batch)]
            for f in fs:
                try:
                    r = f.result()
                except Exception as e:
                    r = {'stage': 'llm', 'error': str(e)}
                r['round'] = rnd
                arch.append(r)
                if r.get('stage') == 'done':
                    print(f"[{time.time()-t0:5.0f}s] round {rnd} {r['name']:<26} {r['verdict']:<10} win: random {r['random_win']:>4.0%} trained {r['trained_win']:>4.0%} plan {r['plan_win']:>4.0%}  "
                          f"compile {r['compile_s']:.1f}+{r['plan_compile_s']:.1f}s  {r['steps_per_s']/1e6:.1f}M steps/s {r.get('plan_moves_per_s',0)} moves/s parity {r.get('parity')} fixes {r['fixes']}", flush=True)
                else:
                    print(f"[{time.time()-t0:5.0f}s] round {rnd} FAILED at {r.get('stage')} after {r.get('fixes')} fixes: {str(r.get('error'))[:200]}", flush=True)
        json.dump(arch, open(a.out, 'w'))


if __name__ == '__main__':
    main()
