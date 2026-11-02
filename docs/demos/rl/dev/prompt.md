You design small reinforcement-learning environments as source code. Output one Python code block and nothing else.

The code is parsed, never imported, and the bodies of three functions are pasted into a compiled training loop.
Required top-level constants (literals only):

NAME (short snake_case str), DESCRIPTION (one sentence), W, H (grid size, W*H <= 4096), ACTIONS (2 to 8), MAX_STEPS (<= 2000),
OBS (length of the observation, 1 to 64), GLYPHS (dict from grid value to one character, for rendering),
AGENT_X, AGENT_Y (which entries of s hold the agent's position, for rendering).

Required functions, with exactly these parameters:

def reset(s, grid, obs, rnd, w, h):        start an episode: fill grid and s
def step(s, grid, obs, rnd, w, h, action): apply action; assign the variables `reward` (always a float), `done` and `win` (bools)
def observe(s, grid, obs, w, h):           fill obs with small integers

- s: int64 array of length 32, your persistent state. grid: int64 array, shape (h, w), indexed grid[y, x]. obs: int64 array of length OBS.
- rnd: int64 array of non-negative random ints below 2**23. In reset, rnd[0..1023] are fresh. In step, only rnd[0..7] are fresh. Use rnd[k] % n.
- `reward` starts each step at 0.0, `done` and `win` at False. Assign them directly. Never use `return`.
- Set `win = True` on the one step where the agent fully achieves the world's goal (along with `done = True`). Dying or running out of time is never a win.

The language is a strict subset of Python:
- Statements: assignment, augmented assignment, for over range(), while, if/elif/else, break, continue, pass. Local variables are fine.
- Allowed calls: range, min, max, abs, int, float, bool, len, math.sqrt, floor, ceil, sin, cos, exp, log, hypot, atan2, fabs. No helper functions, imports, lists, tuples, strings, classes, slices or comprehensions.
- Operators: + - * / // % ** and comparisons, and/or/not, & | ^. The shift operators << and >> do not compile: use * and //.
- A variable must keep one type: `reward = 1` is an error, write `reward = 1.0`. Do not assign both ints and floats to the same name.
- Reading outside an array raises an error: check bounds before indexing grid and s. Python's negative indexing still applies, so check for < 0.
- Local variables defined in step are gone next step in spirit: keep anything persistent in s or grid.

The agent is tabular Q-learning. It hashes obs into a table of 2**18 states and has about 10 million steps of experience, with epsilon-greedy exploration.
So obs must be small and sufficient: if two situations need different actions, their obs must differ. Prefer relative directions, nearby cells and flags to absolute coordinates, unless the state space is small. The total number of distinct observations should stay well below 100,000. Rewards should be roughly in -1 to 1 per event, and an episode's return should be clearly better under good play than random play. Random layouts each episode are encouraged, since they prove real skill.

Here is a working example:

```python
{EXAMPLE}
```
