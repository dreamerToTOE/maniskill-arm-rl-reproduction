# Reproduction Execution Plan / 分阶段复现计划

**Project:** ManiSkill 3 manipulation with PickCube/PushCube/PegInsertionSide
**Status:** PLANNED; no results have been executed or validated.

## Prior-learning checkpoint (required before coding)
Produce an explanation of (1) scientific/engineering problem; (2) main solution and why; (3) innovations vs baselines; (4) full published experiments; (5) which parts are reproducible from open source; (6) proposed numerical simulation scope. Do not silently omit experiments and do not claim original quantitative reproduction based on simulation visuals alone.

## Gates
- **M0** — environment audit: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M1** — task introspection: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M2** — official baseline: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M3** — PPO/SAC: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M4** — ablations: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M5** — robust evaluation: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.
- **M6** — optional FR3 extension: record setup and acceptance evidence in docs/PROGRESS.md; if incomplete mark BLOCKED with cause.

## Project-specific actions
1. Verify task IDs / official PPO and SAC example entrypoints from pinned ManiSkill 3 commit, rather than assuming both algorithms exist for each environment.
2. Enumerate observation state vectors, action type, robot model, reset randomization, reward decomposition, collision behavior and time limit.
3. Design matched PPO/SAC comparison only for tasks with equivalent observations, actions and fair environment-step accounting.
4. Track action direction cosine, near-target saturation, contact regime, distance-to-target bins and failed trajectories.
5. Compare PickCube / PushCube / PegInsertionSide; export JSON/CSV metrics, curve statistics and reproducible videos.
6. Optional FR3 transfer: separate source policy ideas from solver/controller changes and log simulation mismatch.

## Minimum acceptance
- Reproducible pinned source SHA, environment manifest and deterministic smoke test.
- Paper/baseline code map and honest difference log.
- A run manifest for **each** metric or figure; no fabricated results.
- At least 3 random seeds *if computationally feasible*; otherwise explicit budget and uncertainty.
- Held-out evaluation conditions separated from training, raw episodes and summary plots.
- Metrics: success rate, steps-to-success, sample efficiency, force/collision proxy, policy saturation.
- Completion report must distinguish validated claims, partial reproductions, extra extensions, and blocked claims.

## Stop/go rules
Stop after three distinct failed attempts on the same blocker; escalate cause and proposed alternatives. Do not silently lower model/environment complexity or invent synthetic 'paper-equivalent' evidence. Prefer numerical simulation verification where physical testing unavailable, clearly labeled. No unapproved long/high-cost runs.
