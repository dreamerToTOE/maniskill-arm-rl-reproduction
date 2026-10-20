# Work Log / 复现进度日志

**Status 2026-10-20: M0 source/theory and learner-provided hardware/VRAM/disk/pip audit COMPLETE; limited isolated dependency installation and CPU smoke APPROVED / NOT_RUN; M1 CPU smoke NOT_RUN; NO training or local simulation.**

## Running log
| Date | Gate | Repo base commit | Work | Evidence | Outcome | Next |
|---|---|---|---|---|---|---|
| 2026-10-19 | INIT | historical | repository governance and plans initialized | README / AGENTS / docs | SETUP_ONLY | source audit |
| 2026-10-19 | M0 | 9cde33674d3da1122b6085513909fbd9356968bb | Read eight local governance/planning documents; verified upstream v3.0.1 commit a4a4f9272ad64b1564035874b605ceb687b63ed8, task source, PPO/SAC scripts, baseline commands, licensing metadata, paper and current issues; documented algorithm data-flow, task matrix, learners' questions and host risks. | docs/M0_UPSTREAM_AUDIT.md; docs/M0_TASK_ALGORITHM_MATRIX.md; docs/M0_ALGORITHM_WALKTHROUGH.md; docs/M0_LEARNING_CHECK.md; docs/UPSTREAM_LOCK.md | SOURCE_REVIEW_ONLY / TRAINING_NOT_RUN | researcher's M0 answers + approved host audit/smoke |

| 2026-10-19 | M0-LEARNING-01 | 4f37dc02732df6b0ccb597059cf79350f715bac4 | Reviewed researcher answers to three M0 theory questions; identified Actor/Critic vs advantage confusion and missing SAC direct-Q policy objective; provided corrections and focused recheck. | docs/M0_LEARNING_CHECK.md | PARTIAL / NOT_PASSED; Q1 pass, Q2 fail, Q3 partial | researcher reanswers 3 focused questions; NO TRAINING |

| 2026-10-19 | M0 follow-up / M1-PREP | fa58dc1624f8d55b607cc435dea4aa879d650c49 | Reviewed further PPO/SAC answers and corrected remaining conceptual details; source-mapped PickCube state/action/reward/reset/done and prepared bounded CPU introspection script plus approval checklist. | docs/M0_LEARNING_CHECK.md; docs/M1_START_HERE.md; scripts/m1_inspect_pickcube.py | THEORY_BROADLY_COVERED / M1_PREPARED / CPU_SMOKE_NOT_RUN | read-only Ubuntu audit; propose isolated dependency plan and seek explicit smoke approval |

| 2026-10-20 | M0-HOST-AUDIT / M1-PREFLIGHT | 046a11bd14bd47240c5f4826ec6951580850a132 | Learner executed read-only audit_environment.py and provided output: Python 3.10.12, kernel 6.8/glibc 2.35, RTX3070 total 8192 MiB, driver 580.178.04, NumPy 1.24.4, no torch/gymnasium/sapien/mani_skill in active interpreter. Source-verified candidate distro wheels; requested free VRAM/disk and installation approval before isolated setup. | Learner-provided stdout; docs/UPSTREAM_LOCK.md | HOST_VERSION_AUDITED / INTEGRATION_NOT_RUN / INSTALL_NOT_APPROVED | audit free VRAM/disk; obtain bounded venv + CPU smoke approval; NO TRAINING |

| 2026-10-20 | M0-HOST-PREFLIGHT / M1-INSTALL-PROPOSAL | c6149438a428271a21f49539728e8e001c761ddc | Learner ran nvidia-smi (RTX 3070 total 8192 MiB, free 5957 MiB, driver 580.178.04), df -h (418G available), system pip (22.0.2). Verified PyPI published cp310 SAPIEN 3.0.0 wheel and Torch 2.5.1 cu121 option; authored bounded installation approval plan. | docs/UPSTREAM_LOCK.md; docs/M1_INSTALL_APPROVAL.md; learner terminal output | RESOURCE_PREFLIGHT_COMPLETE / USER_APPROVAL_PENDING / INSTALL_NOT_RUN | obtain explicit user permission for isolated install + CPU 3-step smoke; do not train |

| 2026-10-20 | M1-INSTALL-GATE / TRAINING-COST-REVIEW | 41775caebf100cd1ce5199b882f034c3e37acb8b | User explicitly approved previously proposed isolated Python installation and one 3-step CPU PickCube smoke. Audited pinned v3.0.1 PPO/SAC official workload: PPO state PickCube and PushCube 4096 envs/50M steps/seed each; PegInsertionSide 2048 envs/75M steps/seed; SAC readme PickCube/PushCube 32 envs/500k steps. RTX 3070 8GiB (5957 MiB free) supports sensible first local test, but local training throughput/unrestricted 4096 env VRAM remain unmeasured. Documented no-rent-now, benchmark-before-rent criteria. | docs/M1_INSTALL_APPROVAL.md; docs/M1_TRAINING_EFFICIENCY_PLAN.md | INSTALL_AND_CPU_SMOKE_APPROVED_NOT_RUN / TRAINING_NOT_APPROVED | local Codex install+bounded smoke; separately approve GPU throughput or RL pilot later |

| 2026-10-20 | M1/M2 PORTABILITY DECISION | d8cb6f75bbf82662525fae64ad6365b174e396bf | Researcher chose local-first development and small pilots, cloud GPU migration only if measured training throughput/VRAM later warrant it. Added portable experiment checklist, config/manifest/checkpoint/replay requirements and launch gates; no cloud account provisioned or training executed. | docs/CLOUD_MIGRATION_PLAYBOOK.md; docs/M1_TRAINING_EFFICIENCY_PLAN.md | CLOUD_RENTAL_DEFERRED / MIGRATION_DESIGN_READY | finish approved isolated M1 setup and CPU smoke |

## Explicit unknowns/blockers
1. Read-only user-supplied audit confirms Python 3.10.12, RTX 3070 total 8192 MiB/**5957 MiB free**, driver 580.178.04, **418G free disk**, system pip 22.0.2; other venv interpreters and PyTorch CUDA wheel remain unobserved. No local shell access.
2. SAC tuned example for PegInsertionSide not located in pinned README; do not label official benchmark without evidence.
3. Controller and reward/action dimensions not runtime checked; default scripts compare different control modes.
4. Researcher follow-up understanding improved: PPO V/GAE vs SAC twin-Q/entropy largely correct; still needs source-backed checks for rsample/controller semantics. Host audit completed and user approved *isolated installation plus three-zero-action CPU smoke* only. No GPU training or microbenchmark permission and no actual shell execution evidence. M0 not marked PASSED.
5. Upstream issue #1469 warns about EE rotational-action sign for v3.0.1; do not modify upstream or claim this locally reproduced.

## Hard boundaries
- Dual-FR3 main project frozen pending 5 coordinated-arm paper reproductions and explicit resume approval.
- No local clone, download, pip install, RTX benchmarking, ROS2/Isaac Sim/CUDA/driver changes, RL training, 3-seed runs or model-weight assertions in M0 audit.
- Experiment matrix EXP-000 remains NOT_STARTED; no fabricated metrics or progress percentages.
