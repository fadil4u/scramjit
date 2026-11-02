# scramjit docs

Served from `docs/` by GitHub Pages. No build step: `docs.js` fetches the Markdown and renders it
in the browser, so a pushed edit is live once Pages redeploys (about a minute). Pages are fetched
with `cache: 'no-cache'`, so readers get the new version on their next load. After changing
`docs.js` or `docs.css`, update the `?v=` on their tags in `index.html`.

The Prompt for Agents corner here peels back over the page to show the prompt (`../agent-prompt.txt`), as the
landing page's banner does. Both use `../agent.js` and `../agent.css`; after changing those, update
their `?v=` in both `index.html` files.

- `content/<page>.md`: one page each, plain Markdown (GitHub flavored: tables, fenced code).
- `../llms.txt`: the index of these raw pages for agents; add new pages there too (`tests/test_site.py`
  checks it).
- `nav.json`: the order and grouping. Parts with `"walk": true` are the walkthrough (numbered,
  with progress and Next links); the others are reference.

Writing pages:

- Link to another page as `[text](types.md)` or `[text](types.md#a-heading)`; headings get ids
  from their text (`## Annotations are contracts` -> `#annotations-are-contracts`).
- Code fences: `python`, `c`, `console` (lines starting with `$ ` are commands), `text`.
- Callouts: a blockquote starting with `[!NOTE]`, `[!TIP]`, or `[!WARNING]`.
- Write as for the released product: present tense, no roadmap or status notes. `requirements.md`
  is the source of truth for behavior; keep the docs consistent with it.

Preview locally: `python3 -m http.server -d docs` and open http://localhost:8000/docs/.
