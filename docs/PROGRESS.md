# Work Log / 复现进度日志

**Status 2026-10-19: M0 SOURCE_AUDIT_COMPLETE / LEARNER_AND_HOST_CHECK_PENDING. No training, experimental reproduction, or hardware validation performed.**

## Running log
| Date | Gate | Repo base commit | Work | Evidence | Outcome | Next |
|---|---|---|---|---|---|---|
| 2026-10-19 | INIT | historical | repository governance and plans initialized | README / AGENTS / docs | SETUP_ONLY | source audit |
| 2026-10-19 | M0 | 9cde33674d3da1122b6085513909fbd9356968bb | Read eight local governance/planning documents; verified upstream v3.0.1 commit a4a4f9272ad64b1564035874b605ceb687b63ed8, task source, PPO/SAC scripts, baseline commands, licensing metadata, paper and current issues; documented algorithm data-flow, task matrix, learners' questions and host risks. | docs/M0_UPSTREAM_AUDIT.md; docs/M0_TASK_ALGORITHM_MATRIX.md; docs/M0_ALGORITHM_WALKTHROUGH.md; docs/M0_LEARNING_CHECK.md; docs/UPSTREAM_LOCK.md | SOURCE_REVIEW_ONLY / TRAINING_NOT_RUN | researcher's M0 answers + approved host audit/smoke |

| 2026-10-19 | M0-LEARNING-01 | 4f37dc02732df6b0ccb597059cf79350f715bac4 | Reviewed researcher answers to three M0 theory questions; identified Actor/Critic vs advantage confusion and missing SAC direct-Q policy objective; provided corrections and focused recheck. | docs/M0_LEARNING_CHECK.md | PARTIAL / NOT_PASSED; Q1 pass, Q2 fail, Q3 partial | researcher reanswers 3 focused questions; NO TRAINING |

## Explicit unknowns/blockers
1. Exact local Python/Torch/SAPIEN/CUDA runtime and available VRAM are **not observed** (no SSH/local-run tool); requested read-only audit needed.
2. SAC tuned example for PegInsertionSide not located in pinned README; do not label official benchmark without evidence.
3. Controller and reward/action dimensions not runtime checked; default scripts compare different control modes.
4. Researcher first M0 answer reviewed; Q1 pass, Q2 core misconceptions, Q3 partial. Focused three-question recheck pending; exact M1 smoke configuration and time/compute approval not given.
5. Upstream issue #1469 warns about EE rotational-action sign for v3.0.1; do not modify upstream or claim this locally reproduced.

## Hard boundaries
- Dual-FR3 main project frozen pending 5 coordinated-arm paper reproductions and explicit resume approval.
- No local clone, download, pip install, RTX benchmarking, ROS2/Isaac Sim/CUDA/driver changes, RL training, 3-seed runs or model-weight assertions in M0 audit.
- Experiment matrix EXP-000 remains NOT_STARTED; no fabricated metrics or progress percentages.
