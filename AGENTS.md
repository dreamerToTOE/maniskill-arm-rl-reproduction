# Codex / Agent Working Agreement

## Purpose
This repository holds a **not-yet-executed** reproduction. The agent must report actual work, not convert plans into accomplishments.

## Mandatory startup
1. Read README, this file, the full upstream methods/paper/docs, and existing logs. Identify **problem → method → innovation → experiments → dependencies → expected metrics** before coding.
2. Extract upstream commit/tag, upstream license, data provenance, supported simulator/task/action versions; record in `docs/UPSTREAM_LOCK.md` before installing.
3. Propose what counts as exact reproduction vs adaptation vs new extension. Ask the user to choose scope when important experiments or resources differ.
4. Maintain a visible progress log with dates, commit hashes and command outcomes. No unsupported percentage estimates.

## Implementation policy
- Never replace existing main research projects or their environments. Create an isolated Python environment and do not silently upgrade system CUDA / NVIDIA driver / ROS2 / Isaac Sim.
- Do not copy the entire upstream implementation into this repo without reviewing its license, attribution, and asset/data constraints. Prefer separately cloning an immutable upstream tag/SHA.
- Make the smallest testable change. Keep baseline untouched and clearly differentiate official baseline from own modifications.
- Seeds, randomization, observation/action spaces, units, control step, train/validation/test splits, simulator params and target metrics must be saved.
- Never claim success, real-robot deployment, original-paper numerical agreement, or statistical improvement without measured artifacts.
- Never launch overnight training, large GPU batches, download restricted data, or perform irreversible operations without explicit user-approved budget.

## Three-attempt failure protocol
On 3 meaningfully different unsuccessful attempts at the **same blocker**, stop blind retries; classify as (A) local environment/implementation issue, (B) upstream open-source/data deficiency, (C) paper ambiguity, or (D) resource constraint. Provide evidence and bounded next steps. A smaller, numerical-simulation-only equivalent requires explicit labeling.

## Required artifacts for every completed gate
`docs/PROGRESS.md`: objective / changed files / commit / command / result / next action; `docs/EXPERIMENT_MATRIX.md`: experiment ID, configs, pinned SHA and output path; `docs/FAILURE_LOG.md`: failures; outputs in versioned `results/` summaries; plots or videos with reproducible generating scripts. Checkpoints and datasets must be gitignored unless deliberately approved.

## Reporting template
- Target gate & reproduction claim:
- Source paper/upstream exact location:
- Code changes & local commit:
- Tests and commands executed (pass/fail):
- Metrics and artifact locations:
- Deviations / uncertainty:
- Next proposed bounded run (not automatically launched):
