"""envs.json for the page, from a curriculum run: the worlds that are solved and don't win at random.
Run after dev/loop.py:  python3 dev/build.py dev/runs/run2.json"""
import json, os, sys
RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = []
for e in json.load(open(sys.argv[1])):
    if e.get('verdict') == 'solved':
        out.append({k: e[k] for k in ('name', 'description', 'code', 'random_win', 'trained_win', 'plan_win')})
# worlds where thinking visibly pays first; those where it needs the table (dev/sweep.py) last
ORDER = ['guard_crossing', 'firefly_meadow', 'patrol_heist', 'crate_gate_bat', 'conveyor_crossing', 'reef_dodger', 'pen_herder',
         'boulder_miner', 'vault_chain', 'alchemist_lair']
out.sort(key=lambda e: ORDER.index(e['name']) if e['name'] in ORDER else len(ORDER))
json.dump(out, open(os.path.join(RL, 'envs.json'), 'w'))
print(len(out), 'worlds')
