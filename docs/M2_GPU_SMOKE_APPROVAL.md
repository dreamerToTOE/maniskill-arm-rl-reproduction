# M2 preliminary GPU PhysX smoke — PROPOSED, AWAITING APPROVAL

Researcher's normal Ubuntu terminal reports `torch.cuda.is_available()==True`, `torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3070'`, using the installed `.venv-maniskill`. The Codex agent execution shell earlier lacked NVIDIA device nodes; GPU tests must be launched from a GPU-visible environment or stopped, not routed through an invisible sandbox and treated as a CUDA failure.

**Purpose:** answer the narrow question “does the pinned ManiSkill 3.0.1 / SAPIEN 3.0.0 `physx_cuda` backend create a single PickCube Panda env and perform bounded steps on this host?”. It does NOT measure PPO training throughput, success or convergence.

**Not authorized yet.** Proposed only for the researcher's explicit approval after reviewing this file.

## Proposed fixed budget

- Task `PickCube-v1`, Panda, state observation, `pd_joint_delta_pos`, normalized dense reward, `sim_backend=physx_cuda`.
- 1 environment, seed 0, **3 zero-action environment steps** only; no Actor/Critic, no optimizer, no RL training, no renders or videos, no optional pip installs or source rebuilds.
- Hard wall-clock **<=120 seconds** for actual smoke (tool should use external `timeout 120s`), one attempt initially. Log initialization latency and step result. If it fails, stop and diagnose before any repeat.
- Resource safeguards: record fresh `nvidia-smi` and current available VRAM; **do not start if free VRAM <2048 MiB**, if another job is active or device nodes unavailable. Stop on OOM/driver/Vulkan errors. No unbounded loop. Stop if process or other GPU activity interferes with user's other jobs. No claims about VRAM until measured.
- Permit a **minimal separate GPU smoke script** rather than changing the M1 CPU evidence script in-place; no changes to CUDA/driver/ROS2/IsaacSim/FR3 repo.
- Write `results/m2_gpu_physx_smoke.json` with command, source commit, package versions, GPU model, driver/CUDA info, peak memory if available, elapsed wall time, obs/action/reward/done and exit status. Keep logs under `results/raw`; no checkpoint, no training.
- If agent sandbox has GPU invisible, output `BLOCKED_IN_AGENT_SHELL` and provide the exact command to execute from user terminal; do not attempt sudo or driver/package remediation.

## Next step after this (separate approval again)

If and only if GPU simulation smoke is successful and learning review completed: request a short environment-parallel throughput microbenchmark using 64/128/256/512 env candidates (each <=60s, whole experiment <=8 minutes), followed by an independently approved <=100k-step PPO pilot; treat pure sim FPS and end-to-end PPO SPS as distinct. Use actual RTX 3070 measurements before considering cloud rental.
