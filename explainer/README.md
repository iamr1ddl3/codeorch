# CodeOrch Explained — Lesson 1

An interactive, step-by-step walkthrough of how one spec travels through the CodeOrch pipeline
(Orchestrator → Planner → Coder ‖ Tester → Quality gate → Reviewer → Documenter), and what each
agent writes to the context store.

## Open it

Double-click `index.html` (or drag it into Chrome, Safari, Edge or Firefox). No server, build step
or internet connection is needed.

Keep this folder together — `index.html` loads its font from `fonts/`. Without that folder the page
still works, just in the system monospace font.

## Use it

- **Next / Back**, the `←` `→` keys, or click any step number in the bottom bar.
- Append `#5` to the URL to open step 5 directly (e.g. to jump straight to the retry or crash step).
- If the computer has *Reduce motion* turned on, the page shows each step without animation.

Timings and gate scores in the walkthrough are illustrative (rounded RouterBench-corpus averages),
not a live run.

## Files

| File | What it is |
|---|---|
| `index.html` | The whole lesson: markup, styles, diagram and script |
| `fonts/` | JetBrains Mono (400/500/700/800), self-hosted |
| `fonts/OFL.txt` | The font's SIL Open Font License, required when redistributing it |
