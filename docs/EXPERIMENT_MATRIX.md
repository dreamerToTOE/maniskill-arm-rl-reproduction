# Experimental Matrix / 实验矩阵

No experiment has been run. Fill rows **before** launch, then attach exact evidence paths and status. Never infer quality from a demo video.

| ID | Gate | Algorithm/controller | Task, env & upstream SHA | Seed | Train steps & wall-clock | Eval episodes / held-out setup | Metrics (success rate, steps-to-success, sample efficiency, force/collision proxy, policy saturation) | Artifact path | State |
|---|---|---|---|---|---|---|---|---|---|
| EXP-000 | preparation | — | TBD | — | 0 | — | — | — | NOT_STARTED |

## Protocol invariants
- Preserve joint limits/physics settings unless an experiment explicitly changes them.
- Keep RL comparison observation/action/termination scheme and evaluation budget equivalent where possible.
- Report deterministic/stochastic policy evaluation separately, with mean/std or confidence interval and per-seed data.
- Save training curves, configs, checkpoint file hashes, outputs and failure videos.
- For negative outcomes, document what failed without changing evaluation criteria retroactively.
