# RL demo: scramjit.dev/demos/rl

A model writes worlds as Python; each compiles and trains in the visitor's tab, with no server.

## Done
- [x] Worlds set `win`; a world counts as solved at 80% wins, and only solved worlds ship.
- [x] A planner that runs the world's own compiled `step`, with thinking per move on a slider.
- [x] Win rate at each thinking level, live, from unseen episodes.
- [x] Train a Q-table, which then guides the simulations.
- [x] Editor, watchdog for hangs, hostile code rejected, site's look, phone layout.

## Next
- [ ] Before and after: the same world, one agent thinking and one not, side by side.
- [ ] Live generation: paste an API key and a prompt, watch a new world get written, compiled, solved.
- [ ] Curriculum view: worlds appear one by one, each building on the last.
- [ ] One-click edits that show the agent adapting (add a wall, make a guard faster).
- [ ] Shareable links: a world's source in the URL.
- [ ] Better art than letters on squares.
- [ ] Try on real phones and in Firefox and Safari; only headless Chromium so far.
- [ ] Stop work when the page is out of view, not only when the tab is hidden.
- [ ] Link from the home page and the docs; add to `llms.txt`.

## Open
- Planning gets worse with long lookahead in worlds of many subgoals (`vault_chain`, `alchemist_lair`), and needs the table.
- Episode counts at 1M steps per move are small: the bar stays faint below 5 episodes.
- The planner runs without bounds checks in the page (16 s to compile with them, 2.5 s without), so an edited world can
  corrupt the tab's wasm heap, though nothing outside it. Shipped worlds were run with checks on in `dev/worker.py`.

## Gaps in scramjit it depends on
`TODO.md` items 14-17: shifts, calls between compiled functions, compile time on large functions, float32.
