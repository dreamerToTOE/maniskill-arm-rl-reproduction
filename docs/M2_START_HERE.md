# M2 START HERE — PickCube official PPO baseline (study first; no training)

> **2026-10-21, state: SOURCE-VERIFIED / LEARNING-PACK-CORRECTED / GPU-VISIBLE-IN-AGENT-SHELL / PHYSX-CUDA-UNVERIFIED / PILOT-PROPOSAL-PENDING / TRAINING-NOT-RUN.**
>
> Prerequisite: M1 PASSED at commit `d08e5870368c12644ff385e17f78068ad4bbb774`. Researcher approved **M1 isolated installation + 3-zero-action CPU smoke only**. This is not approval for M2 GPU simulation benchmarks or PPO training.
>
> Upstream: ManiSkill **v3.0.1** source SHA `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
>
> 2026-10-21 additions: [`M2_SOURCE_REVIEW.md`](M2_SOURCE_REVIEW.md) (line-by-line verification of the learning pack, three corrections, official-vs-teaching arithmetic, dependency gaps, RTX 3070 risk) and [`M2_GPU_PILOT_PROPOSAL.md`](M2_GPU_PILOT_PROPOSAL.md) (bounded Stage A probe + Stage B pilot, awaiting approval). GPU visibility inside the agent shell was **re-tested and is now positive** — see §9 of the review; this supersedes the earlier "no device nodes" reading without implying any driver change.

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
