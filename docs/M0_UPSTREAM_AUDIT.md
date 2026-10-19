# M0 upstream and paper audit — 2026-10-19

Status: SOURCE_READ / HOST_NOT_TESTED / TRAINING_NOT_RUN. No claims of locally reproduced curves.

## Problem and contribution
Paper: https://arxiv.org/html/2410.00425v2 (2025 revision; initial arXiv 2024). ManiSkill3 addresses insufficient GPU-parallel state+visual data throughput, heterogeneous scenes, and task-building overhead. Based on SAPIEN / PhysX, it adds fast batched physics and raster rendering, heterogeneous task/robot scenes, batched object-oriented APIs, robot controllers, datasets, teleoperation and RL/IL example baselines. It is **not an original PPO or SAC invention**.

## Original experiments (do not conflate them)
- Platform benchmark: state and RGB(+depth, segmentation) simulation/rendering FPS versus alternatives across resolution/cameras, GPU memory, heterogeneous tasks. Hardware/settings matter; quoted max 30k+ FPS is not a universal speed guarantee. See paper III and XI.
- Learning speed: PPO PickCube state/RGB ManiSkill3 GPU versus ManiSkill2 CPU on RTX 4090, five seeds and 95% CI; ~1 minute state and ~10 minutes RGB in the paper's particular setup. See IV-C/Fig.11.
- Other RL baselines: PPO / TD-MPC2 emphasized in appendix IX; SAC available but paper explicitly says **not heavily benchmarked**.
- Imitation learning: BC/ACT/Diffusion Policy task tables; PerAct with vision and augmentations. These results are NOT PPO/SAC success rates.
- Sim2Real: Koch arm RGB pick with domain randomization; paper reports 22/24 real successes over 3 training runs after ~1h training on RTX 4090. Our project has **no real-world experiment**.
- Limits: not all scenes batch efficiently; domain gaps remain. See paper V.

## Verified source and reproducibility
Proposed reference tag v3.0.1 = a4a4f9272ad64b1564035874b605ceb687b63ed8; this is a *source pin proposal* and has not been locally cloned or installed. Official GitHub release 2026-04-21.
Official repo: https://github.com/mani-skill/ManiSkill
Official source license: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE (Apache-2.0); other datasets/assets/third-party rights require their own review: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE-3RD-PARTY
Setup dependency constraints: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/setup.py : Python >=3.9, sapien >=3.0.0 on Linux, gymnasium >=0.29.1; no exact torch/CUDA wheel pinned by upstream setup.
- Task environment sources: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/pick_cube.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/push_cube.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/peg_insertion_side.py
- Core environment: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/sapien_env.py ; sim/control defaults: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/utils/structs/types.py
- Robot/controller: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/agents/robots/panda/panda.py
- PPO: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/ppo.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/ppo_fast.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/baselines.sh
- SAC: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/sac/sac.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/sac/README.md

## Open upstream issues worth tracking (2026-10-19 inspection; NOT self-reproduced)
- #1469: pd_ee_delta_pose normalized rotation sign issue in v3.0.1 (affects rotation semantics; confirm before any peg action experiments): https://github.com/mani-skill/ManiSkill/issues/1469
- #1499 and #1501: newer main trajectory loader label/filter problems; do not automatically attribute to pinned v3.0.1: https://github.com/mani-skill/ManiSkill/issues/1499 , https://github.com/mani-skill/ManiSkill/issues/1501
- #1071: controller interpolation GPU behavior: https://github.com/mani-skill/ManiSkill/issues/1071
- #1023: asks about absent comparable SAC baselines.sh/control-mode differences: https://github.com/mani-skill/ManiSkill/issues/1023
- #899: older SAC alpha-gradient issue is *closed*; pinned sac.py draws log_pi under no_grad for alpha update: https://github.com/mani-skill/ManiSkill/issues/899

## Claim taxonomy
A. Framework smoke: environment boots, reset/step shapes, rewards, no training.
B. Official baseline repeat: unchanged official code/config at pinned SHA, retrain/evaluate and log own raw metrics.
C. Paper claim reproduction: reproduce a specified paper figure/protocol (hardware, settings, statistics), with explicit deviations.
D. Independent comparison: matched PPO/SAC obs/action/evaluation settings, clearly marked *our experiment*.
E. Novel extension: new observation, reward or controller; never call this an official result.

Scope decision still pending researcher review. Five-paper FR3 freeze remains mandatory.
