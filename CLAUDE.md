# argmin — Claude instructions

Ing's archive of optimization algorithms. He writes each algorithm by hand, runs it, and debugs it, to understand the method and get hands-on experience. `README.md` indexes the algorithms; `Plan.md` (in ai-mentor-core's plan convention) lists what's in the archive and what's next.

**This repo is managed from the math vault** (Ing's call, 2026-11-02). Sessions run from `~/devpro/ing/math-notes`, under that vault's mentor conventions, `Profile.psych.md` and safety protocol; this repo holds the code. If a session was opened directly in this folder, tell Ing at the start and suggest running it from math-notes instead. As a fallback, his calibration is imported here:

@~/devpro/ing/math-notes/Profile.psych.md

- **Ing writes the algorithms.** Never write an algorithm's implementation, its update steps, or its derivation for him. Hints are one sentence; the answer comes only after two hints or an explicit ask.
- **What Claude may do:** scaffold a new folder (`<algorithm>/<problem>/` with an empty `main.py` and a `README.md` skeleton), add a problem generator or plotting helper when asked, run the code, read tracebacks with him, and check a result against an independent invariant (a residual, a duality gap, a known closed-form solution) before trusting it.
- **Layout:** one folder per algorithm, one subfolder per worked problem, each with its own `README.md` and `main.py`. Add a row to the table in `README.md` and to `Plan.md` when a new one lands.
- **This repo is public.** Nothing personal goes in: no notes about Ing, no references to his profile beyond the import line above.
- **Commit** after each landed algorithm or problem, with a conventional message (`feat: <algorithm> on <problem>`).
