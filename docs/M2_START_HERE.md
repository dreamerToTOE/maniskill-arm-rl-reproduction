# M2 START HERE — PickCube official PPO baseline (study first; no training)

> **2026-10-21, state: SOURCE-VERIFIED / LEARNING-PACK-CORRECTED / **STAGE A0 PASSED (12/12, 3.496 s)** / **STAGE A1 PASSED (4/4, 27.706 s)** / PHYSX-CUDA-VERIFIED / PARALLEL-SCALING-MEASURED-TO-256 / SIM-THROUGHPUT-MEASURED / **STAGE-B-PASSED / 256/512/1024-PILOTS-PASSED (loop 4435/8236/15117, VRAM flat to 1024).**
>
> Prerequisite: M1 PASSED at commit `d08e5870368c12644ff385e17f78068ad4bbb774`. Researcher approved **M1 isolated installation + 3-zero-action CPU smoke only**. This is not approval for M2 GPU simulation benchmarks or PPO training.
>
> Upstream: ManiSkill **v3.0.1** source SHA `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
>
> 2026-10-21 additions: [`M2_SOURCE_REVIEW.md`](M2_SOURCE_REVIEW.md) (line-by-line verification of the learning pack, three corrections, official-vs-teaching arithmetic, dependency gaps, RTX 3070 risk) and [`M2_GPU_PILOT_PROPOSAL.md`](M2_GPU_PILOT_PROPOSAL.md) (bounded Stage A probe + Stage B pilot, awaiting approval). GPU visibility inside the agent shell was **re-tested and is now positive** — see §9 of the review; this supersedes the earlier "no device nodes" reading without implying any driver change.
>
> 2026-10-21 (later): the original 64→4096 Stage A sweep was **rejected** by [`M2_INDEPENDENT_REVIEW.md`](M2_INDEPENDENT_REVIEW.md) and has been repaired — see [`M2_GPU_PROBE_REPAIR.md`](M2_GPU_PROBE_REPAIR.md). Stage A is now split into **A0** (`scripts/m2_gpu_smoke_a0.py`, 1 env / 3 steps / ≤120 s, needs no new dependency) and **A1** (`scripts/m2_gpu_sim_probe.py` + `scripts/m2_a1_run_all.py`, 16/64/128/256, one isolated process per level). **A0 has been executed three times, and the third one PASSED.** 12:58: the agent session exited **8** at the VRAM-telemetry gate without building anything (no `/dev/nvidia*` there). 15:41: the researcher's terminal built the scene and stepped 3 times but the gate returned **11/12** because our horizon walker missed `ManiSkillVectorEnv._env` - a script bug. After fixing it, 15:44: **12/12, exit 0, 3.496 s**, `horizon_source=TimeLimitWrapper`, `max_episode_steps=50`, ~1.33 GiB device VRAM, torch allocator peak 8.2 MiB, and results reproducible against the 15:41 run. Details in `M2_GPU_PROBE_REPAIR.md` §7.2b-7.5.
>
> 2026-10-21 (A1): authorized, hardened and statically verified (51 audit + 82 self-test checks, 0 failed). The 16:09 agent-session attempt was `BLOCKED` (exit 8, zero levels), but the 16:18 run in the researcher's terminal **PASSED: 4/4 levels, exit 0, 27.706 s**, giving aggregate simulator SPS **905.8 / 3342.0 / 6437.4 / 11746.1** at 16/64/128/256 with scaling efficiency 1.000/0.922/0.888/0.811, and device VRAM **essentially flat** (peak 3912/3867/3931/3931 MiB, >= 4261 MiB free) — the ~1.3 GiB PhysX scene is a fixed cost. Details: [`M2_A1_EXECUTION_REPORT.md`](M2_A1_EXECUTION_REPORT.md).
>
> 2026-10-21 (Stage B): pilot authorized, code-audited, dependency installed (`tensorboard 2.21.0`), supervised (`scripts/m2_stage_b_pilot.py`). **Attempt 1 (20:51) was cut at 29/40 by a supervisor stdout-buffering bug; attempt 2 (21:08) PASSED cleanly — 40/40 iterations, 256,000 training + 3,200 eval env steps, 9 checkpoints, end-to-end 2400.4 SPS / loop 2600, peak VRAM 5137 MiB, explained variance −1.03 → +0.85, eval success 0.0 (expected at this budget).** Details: [`M2_STAGE_B_EXECUTION_REPORT.md`](M2_STAGE_B_EXECUTION_REPORT.md), GPU root cause: [`M2_GPU_VISIBILITY.md`](M2_GPU_VISIBILITY.md).

## M2 scope and scientific claim

Reproduce the **official state-based PickCube PPO baseline** in an auditable way, separating three stages:
1. **M2-READ**: source-map the official algorithm and task, deliver *theory one-pager + code guide* **before** asking comprehension questions; prepare a training comparison table and a concrete run budget. This phase is approved as document/source analysis; no training.
2. **M2-HOST-READONLY — DONE (2026-10-20)**: researcher ran `.venv-maniskill/bin/python` in own normal Ubuntu terminal and reported `CUDA: True`, `GPU: NVIDIA GeForce RTX 3070`. This establishes PyTorch GPU visibility in that terminal, **not** SAPIEN/PhysX `physx_cuda` operation, VRAM headroom, training speed or visibility inside Codex's agent shell. Record stdout as user-supplied evidence; no GPU sim/stress has run.
3. **M2-EXECUTION — separate approval required**: (a) bounded GPU environment smoke/profiling with memory and walltime limits, (b) bounded end-to-end PPO pilot, (c) only then official multi-seed reproduction and comparison. Each substantive run must be separately budgeted and explicitly approved.

The project follows local-first debugging with RTX 3070 8GB, cloud migration only after actual throughput/peak VRAM indicates it is useful. Main FR3 repo remains frozen. Nothing in this M2 document authorizes launching training.

## Baseline identity: do not silently conflate two official scripts

| Layer | Baseline path in pinned source | Role |
|---|---|---|
| Transparent teaching reference | `examples/baselines/ppo/ppo.py` | Easy-to-read PPO loop (Gaussian Actor, V-Critic, rollout, GAE, clipped update) |
| Official listed performance baseline | `examples/baselines/ppo/ppo_fast.py` | Same broad PPO method with TensorDict/torch.compile/CUDA graphs optimizations |
| Official PickCube command | `examples/baselines/ppo/baselines.sh` | PickCube **state**, 4096 environments, `num_steps=4`, `update_epochs=8`, `num_minibatches=32`, `total_timesteps=50_000_000` per seed, `--cudagraphs`, seeds **9351,4796,1788** |
| Independent task/controller source | `mani_skill/envs/tasks/tabletop/pick_cube.py`; `mani_skill/agents/robots/panda/panda.py` | Task success/rewards and `pd_joint_delta_pos` mapping already audited in M1 |

Reference: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/baselines.sh

**WARNING:** `ppo.py` defaults (512 env, 50 rollout steps, 4 update epochs, 10M steps) **differ** from the official 50M/4096/4/8 run in `baselines.sh`. Understanding `ppo.py` is not proof of reproducing the `ppo_fast.py` baseline. Optimized implementation needs optional `torchrl`/`tensordict`, not verified in the installed venv. No optional installs authorized as part of M2-READ.

For the 4096×4 official training batch, total samples per update = **16384 env steps**, with **32 minibatches of 512 transitions** (ignoring exact script rounding/final iteration effects). This is official published *configuration*, **not an observed local capacity claim**. 8GB GPU VRAM is NOT assumed to fit it. 50M requested env steps per seed times 3 seeds represents roughly 150M cumulative reference steps, but exact execution count is floor-divided by the batch size.

## M2 researcher learning deliverable (read FIRST)

Read [M2_LEARNING_PACK.md](M2_LEARNING_PACK.md), which includes:
- Why we use Actor `π(a|s)` and Critic `V(s)`, and why `Advantage` is needed.
- Data path: action sample → bounded controller input → step/reward/done → rollout buffer → timeout-aware GAE → PPO clip/value/entropy → gradients → evaluation/logging.
- Four source anchor clusters (environment+policy, rollout, GAE, update+metrics) with pinned exact line ranges and the one question for each.
- Honest differences: official high-throughput config vs educational reference, `clip_action` mismatch (Gaussian logprob computed preclip while environment receives clipped action), termination/truncation, `ppo_fast.py` optional packages, and evaluation protocol.
- **Three learner questions appear only after explanations**; learner answers are not pre-filled or graded from templates.

## Currently permitted M2 actions

- Read/update source mapping and docs, propose pinned manifest/config schema and future cost/VRAM envelopes.
- Obtain read-only GPU visibility from researcher's normal local terminal (do not infer from `nvidia-smi` seen in another shell).
- Static code inspection/read-only version reporting only; **no** CPU/GPU simulation beyond previously authorized M1 three-step smoke; no PPO/SAC policy rollout with gradient updates.
- No pip installs, video capture, checkpoints, model downloads, modifications to system drivers/CUDA/ROS2/Isaac Sim, or changes to other repositories.

## Researcher-side CUDA read-only check (NOT an optimization run)

In the learner's normal Ubuntu desktop terminal **not the agent sandbox**, within the independent repo:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
./.venv-maniskill/bin/python -c 'import torch; print("torch",torch.__version__,"cuda_runtime",torch.version.cuda,"available",torch.cuda.is_available()); print("gpu",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT_VISIBLE")'
```

Collect commands, timestamps, and outputs. `torch.cuda.is_available()` should be tested in this interpreter before proposing a `physx_cuda` pilot. If False, determine environment/shell visibility; do not propose driver reinstall without diagnostic evidence. No remote or cloud GPU is needed for M2 source reading.

## Approval gates and evidence requirements

- `M2_PREPARED`: learning pack, source map, exact baseline-vs-pilot configuration matrix, budget/risk plan, read-only CUDA result and three researcher answers (do not fabricate).
- `M2_PILOT`: only after explicit approved task/env_count/steps/seed/VRAM/walltime/storage budget, record actual stdout, GPU max usage, real end-to-end SPS, intermediate success, checkpoint evidence and failure logs. Do **not** claim convergence from a tiny pilot.
- `M2_PASSED`: independent official-scope evidence and learner acceptance, with any reduced budget or changed controllers clearly labeled `ADAPTATION/PARTIAL` instead of `EXACT BASELINE`. No post-hoc definition changes. Full historical M0 rubric is tracked separately.

**Status: M2-READ PREPARED + SOURCE-CORRECTED; TORCH CUDA=TRUE IN BOTH the researcher terminal and (as of 2026-10-21) the agent shell; GPU PHYSICS UNVERIFIED; PPO SCRIPTS CANNOT IMPORT (missing deps); PPO TRAINING NOT_RUN.**


## Learner terminal CUDA check — 2026-10-20

**Command the researcher ran** from `~/lmy/maniskill-arm-rl-reproduction`:

```bash
./.venv-maniskill/bin/python -c \
'import torch; print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT_VISIBLE")'
```

**Verbatim stdout:**
```text
CUDA: True
GPU: NVIDIA GeForce RTX 3070
```

Interpretation: torch has access to RTX 3070 in **researcher's regular desktop shell**. This contradicts neither `/dev/nvidia*` absence nor CUDA false previously observed *inside the Codex agent shell*; execution contexts differ. It is a positive prerequisite, **NOT a pass of ManiSkill GPU simulation**, not a training result, and not proof that the agent process can see or operate the device. Avoid installing/reinstalling GPU drivers or CUDA as a reaction to the old sandbox result. Next proposed test is a bounded PickCube `physx_cuda` 1-env no-optimizer smoke, separate approval required, preferably executed in GPU-visible terminal. Do not execute yet.
