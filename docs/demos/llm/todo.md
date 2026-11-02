# Live LLM demo

Live at scramjit.dev/demos/llm once pushed. `tools/drive_llm.mjs` drives it in headless Chromium.

## Next

- Keep the first compile for return visits: store the compiled files in the browser's cache, as the weights are.
- A bigger model: stories110M works at 9 tok/s today (440 MB download).
- Share an edit as a link.
- Real phones, Firefox, Safari.
- Link from the home page and docs, and `llms.txt`.

## Needs from scramjit

- float32 arithmetic: weights are widened to float64, doubling memory.
