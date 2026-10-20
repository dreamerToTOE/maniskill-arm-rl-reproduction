# M1/M2/M3 GPU training efficiency and rent-vs-local decision — PRE-TRAINING ANALYSIS

**Date:** 2026-10-20
**Status:** ANALYSIS_ONLY. CPU isolated install and three-step smoke approved by researcher; **no installation, simulator benchmark, policy optimization, or rented GPU launched by ChatGPT**. Future GPU benchmark and PPO/SAC training each require a separate approval.

## Measured host facts (user-run, not assistant-run)

- Active Python 3.10.12, Linux kernel 6.8/glibc 2.35, system pip 22.0.2; user-side independent project.
- NVIDIA RTX 3070, **8192 MiB total, 5957 MiB free** snapshot; driver 580.178.04. About **5.82 GiB** currently free; it could vary with other processes.
- Filesystem free **418 GB**. Torch/ManiSkill/SAPIEN missing from the inspected interpreter; planned isolated install pending local execution. No direct evidence of training throughput or GPU PhysX peak memory.
- **Do not describe system as RTX 4070 or as having full 8GiB available to a single process.**

## Verified pinned ManiSkill v3.0.1 official workloads

Direct pinned source: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/baselines.sh

| Workload | Parallel envs | Env steps per seed | Seeds in reference file | Total for reference seeds |
|---|---:|---:|---:|---:|
| PushCube-v1 state PPO (`ppo_fast.py`) | 4096 | 50,000,000 | 3 | 150,000,000 |
| PickCube-v1 state PPO (`ppo_fast.py`) | 4096 | 50,000,000 | 3 | 150,000,000 |
| PegInsertionSide-v1 state PPO (`ppo_fast.py`) | 2048 | 75,000,000 | 3 | 225,000,000 |
| **Three PPO tasks total** | – | – | – | **525,000,000** |

- PPO fast baseline uses 8 update epochs, 32 minibatches, CUDA Graphs; `ppo_fast.py` needs optional `torchrl` and `tensordict`. Number of envs influences both memory and actual throughput; blindly applying 4096 environments to a 3070 with ~5.82 GiB presently free can OOM and is **not approved**.
- Direct pinned SAC source: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/sac/README.md . PickCube and PushCube examples use **32 envs, 500,000 env steps per task, UTD=0.5, buffer 500,000, controller `pd_ee_delta_pos`**. PPO baseline controller is `pd_joint_delta_pos` by default; an uncontrolled PPO vs SAC speed/success comparison is **not a valid algorithm-only comparison**.
- Baseline documentation mentions minutes-scale state PPO on certain GPUs/configs but **does not establish RTX 3070 throughput**. ManiSkill quickstart's **200k+ pure state simulation fps on 4090** excludes PPO gradient updates, evaluation, logging and transfer overhead: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html .
- Image-based RGB/RGBD workloads are substantially more VRAM-heavy. Reference docs describe RGB PPO examples <15 GB and SAC RGBD examples <16 GB for their showcased settings, which **are not proof of 8GB compatibility**. Prefer cloud 24GiB tier or scientifically justified lower camera resolution/number envs with changed config recorded, not silently treat modified config as official reproduction.

## Hypothetical full-training walltime scenarios (NOT observed GPU results)

Use **end-to-end aggregate env steps per wall-second** `SPS` during actual policy optimization, including rollout + backward passes (and preferably evaluation/checkpoint), *not pure simulator FPS*.

`hours = total_environment_steps / measured_SPS / 3600`.

| Hypothetical end-to-end SPS | 50M steps one PPO seed | 75M steps one Peg seed | Three-task PPO 3-seed volume (525M) |
|---:|---:|---:|---:|
| 5,000 | 2.78 h | 4.17 h | 29.17 h |
| 20,000 | 0.69 h | 1.04 h | 7.29 h |
| 50,000 | 0.28 h | 0.42 h | 2.92 h |

These are ONLY arithmetic scenarios. Walltime may differ significantly across tasks, PPO and SAC, batch sizes, update settings, performance changing over run, eval and checkpoints. SAC 500k steps is not automatically cheap: update-to-data ratio and replay sampling costs matter.

## Proposed next performance investigation — separate approval required

Do M1 approved isolated wheel installation + **one 3-zero-action CPU smoke** first. Only after it passes, request separate user approval for:

1. **GPU state-only simulator/forward benchmark** for PickCube using candidate parallel counts [64, 128, 256, 512], then optionally 1024 if current VRAM headroom permits. Enforce <=60 seconds measurement per setting, walltime <=8 minutes, no optimizer or policy training. Print startup/steady-state memory with `nvidia-smi` and torch peak VRAM, engine `physx_cuda`, setup time vs warm steady-state sim SPS. No RGB rendering or checkpoints. Stop on OOM/driver errors. **Measurement of sim SPS alone cannot estimate training SPS without additional profiling.**
2. **Short complete training pilot** (separate approval, after GPU smoke): 1 small state-based PickCube PPO seed, bounded to e.g. <=100,000 env steps and <=10 minutes walltime, with full rollout + gradients + eval as feasible. Keep PPO configuration, env count, rollout length, evaluations, memory and success logging. Its aim is throughput/VRAM profiling and overfit diagnostics, **not final converged training or official 50M reproduction**.
3. For SAC, separately profile 32 parallel envs, UTD settings and GPU/CPU replay storage before making cross-algorithm cost projections. Do not add this to a PPO pilot without permission.
4. After measurement, calculate `projected hours(seed/task) = reference steps / measured steady-state end-to-end SPS / 3600` with uncertainty and quote cloud provider hourly price. Compare local electricity/opportunity cost and data transfer to 24GB cloud GPU. Cloud rental is not a prerequisite for M1, source review, CPU smoke, or initial modest state-based pilots.

## Rent-vs-local decision thresholds (practical heuristics, not proven constraints)

- **Start local**: M1, M2 smoke/short state-based rollout, 1 controlled baseline pilot. RTX 3070 8GiB can be viable for small-scale state-based tasks if memory confirmed; no need to rent yet.
- **Stay local** if sample performance is adequate and total planned seed hours are acceptable, with adequate VRAM headroom and stable CUDA/SAPIEN.
- **Consider 24 GiB cloud GPU (e.g. RTX 4090-class)** if 2048/4096 environment faithful reference configs OOM or require unacceptably small parallel counts, repeated runs take >4–6h per seed, full multi-seed volume creates days of walltime, or RGB/RGBD experiments exceed 8GB.
- Reserve cost decision until measured throughput and actual provider price are known. GPU rental, cloud storage, transfers, and all large runs need approval.

## Scope guards

No full PPO/SAC training approved. No PPO vs SAC claims without matching controllers, observation, reward, episode horizon, evaluation, and environment-step accounting. No FR3 main project change. Never fabricate RTX 3070 FPS or success rates. `training_efficiency_status = ESTIMATE_ONLY`, `local_training_fps = UNKNOWN`, `rental_decision = DEFER_PENDING_BENCH`.
