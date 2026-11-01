# scramjit docs

Served from `docs/` by GitHub Pages. No build step: `docs.js` fetches the Markdown and renders it
in the browser, so a pushed edit is live once Pages redeploys (about a minute). Pages are fetched
with `cache: 'no-cache'`, so readers get the new version on their next load. After changing
`docs.js` or `docs.css`, update the `?v=` on their tags in `index.html`.

- `content/<page>.md`: one page each, plain Markdown (GitHub flavored: tables, fenced code).
- `nav.json`: the order and grouping. Parts with `"walk": true` are the walkthrough (numbered,
  with progress and Next links); the others are reference.

Writing pages:

- Link to another page as `[text](types.md)` or `[text](types.md#a-heading)`; headings get ids
  from their text (`## Annotations are contracts` -> `#annotations-are-contracts`).
- Code fences: `python`, `c`, `console` (lines starting with `$ ` are commands), `text`.
- Callouts: a blockquote starting with `[!NOTE]`, `[!TIP]`, `[!WARNING]` or `[!PLANNED]`
  (for designed but unbuilt features).
- `requirements.md` is the source of truth. The docs describe the code as it is and mark what
  isn't built yet as Planned; decisions the spec lists as open are called open here too.

Preview locally: `python3 -m http.server -d docs` and open http://localhost:8000/docs/.
