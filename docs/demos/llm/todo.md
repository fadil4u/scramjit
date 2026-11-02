# Live LLM demo

Live at scramjit.dev/demos/llm once pushed. `tools/drive_llm.mjs` drives it in headless Chromium.

## Next

- Host the weights ourselves, or add a mirror: the Hugging Face download took 20 to 156 s across test runs.
- A bigger model: stories110M works at 9 tok/s today (440 MB download).
- Real phones, Firefox, Safari.
- Link from the home page and docs, and `llms.txt`.

## Needs from scramjit

- float32 arithmetic: weights are widened to float64, doubling memory.
