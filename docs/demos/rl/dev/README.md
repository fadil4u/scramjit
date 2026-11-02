# How the demo's worlds are made and tested

Worlds are parsed, never imported, checked against a whitelist (`../envlab.py`), and spliced into a training loop and a
planner, each compiled as one function. These scripts use `docs/py/`, the same scramjit and cimo the page loads.

- `loop.py`: a model writes worlds, each is evaluated, results go back to the model. Uses the `claude` CLI and `prompt.md`:
  `python3 dev/loop.py --rounds 5 --batch 4 --out dev/runs/run2.json`
- `worker.py`: evaluates one world in its own process under memory and time limits, bounds checks on.
- `build.py`: `envs.json` from a run: `python3 dev/build.py dev/runs/run2.json`
- `sweep.py`: win rate at each thinking level for one world, untrained and guided by a table.
- `bench_env.py keydoor.py`: plain Python, one process per episode, compiled. `bench_pyodide.py`: the same in Pyodide.
- `ctime.py`: where compile time goes.
- `drive.mjs`: the whole page in headless Chromium: thinking levels, edits, training, hostile code, phone layout.
