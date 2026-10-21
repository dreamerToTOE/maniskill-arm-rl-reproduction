# Independent review of M2 Codex deliverables (2026-10-21)

Scope: source and script review ONLY; **no GPU simulation, pip installation, model training, cloud provisioning or experimentation executed** by this review. ManiSkill pinned v3.0.1, commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`. Codex source review commit `42ecf43b0623fb069773e99a39ea9182892bbd53`.

## Summary

The Codex-source-mapping work is largely sound and M1 was passed. PyTorch now reportedly detects RTX 3070 also in the local agent shell; these observations **do not** prove `physx_cuda` works. **Do not approve/run the proposed original Stage A 64→4096 sweep until the following defects are corrected.**

### Blocking defects in `scripts/m2_gpu_sim_probe.py`

1. **Wrong parallel action shape**: `zero=np.zeros(env.single_action_space.shape)` produces shape `(8,)`. Pinned `BaseEnv._step_action` at `sapien_env.py:1081-1110` only adds a batch dimension when `num_envs == 1`. At `num_envs>1` the controller requires `(num_envs, 8)`; use a correctly sized batched tensor on the correct device. Validate before stepping.
2. **Walltime claims unenforced**: `--max-seconds` applies only inside the measured loop, not GPU scene creation, reset, or warmup; a "12 min total" deadline is not implemented at all. Add an external process-level `timeout` and ensure each independent configuration is separately bounded, so one hung initialization does not exhaust the whole budget.
3. **Episode semantics omitted**: the code calls bare `gym.make` / raw `env.step` for 20 warmup + 100 measured actions and never resets at the default 50-step truncation. Use the same `ManiSkillVectorEnv` auto-reset semantics as the PPO baseline, or explicitly define resets and document why a different simulator-only steady-state protocol is valid. No extrapolation to end-to-end PPO SPS.
4. **VRAM guard ineffective**: `free<500 MiB` is far too late to protect a desktop 8GB GPU, and free memory is only sampled before/after each level. PyTorch peak allocation does **not** capture all SAPIEN PhysX allocations. Conservatively reject starting a level if VRAM is already constrained, monitor `nvidia-smi` separately where feasible, and stop **before** memory is near exhaustion. Do not assume a precheck can predict GPU-scene allocation; OOM at initialization remains possible.
5. **Too many envs for first contact**: 4096 is reference-scale workload, not an essential smoke test. First test should be one env + three zero actions (not seven sizes and 12 minutes), and verify CUDA physics backend, device, obs/action/reward/done. Then, only after another bounded approval and script fix, try progressive scale such as 16/64/128/256, and 512 only if memory headroom justifies.
6. **Dependencies separated by stage**: the standalone GPU probe uses ManiSkill/SAPIEN/Torch, not PPO or TensorBoard, so **installing `tensorboard` is NOT a prerequisite for the first GPU smoke**. Installing `tensorboard` is a potential separate step before `ppo.py` PPO pilot. `tensordict` and `wandb` are necessary for `ppo_fast.py` at pinned commit, not for the first smoke.

### Additional scientific accuracy notes

- `ppo.py` uses `final_observation` for bootstrap at both success termination and time-limit truncation. For an actual terminal state, nonzero `V(s_T)` produces an extra `γV(s_T)` contribution; its **sign is not fixed** (depends on learned `V(s_T)`), and magnitude is not measured. Thus the original report's unconditional "raises the target" claim is unsupported. This review corrected the learning pack and source review. At truncation, bootstrap from the pre-reset end observation is sensible under a continuing-time-limit formulation. Evaluate according to the actual environment's fixed-50-step success definition: a timeout without success is not automatically a "systematically underestimated" success rate.
- The code's `next_not_done` breaks the GAE recurrence at both termination and truncation. That fact alone is not proof the behavior is always biased in one direction; preserve evidence and compare with a justified reference before modifying the frozen baseline.
- For state size 42 and action size 8, the readable PPO two 3×256-hidden-layer networks have roughly **0.29M parameters combined**, not 0.6M. This does not affect the conclusion that PhysX parallel environments can dominate 8GB usage.
- `ppo.py` computes old action logprob on sampled unbounded Gaussian actions, then clips actions sent to `env.step`. This discrepancy should be monitored if a later pilot is authorized; it is not a reason to edit upstream PPO unilaterally.
- Before future PPO pilot, verify the actual Tyro `--help` boolean CLI syntax (e.g. `--no-capture-video` instead of assuming `--capture_video=False`) and exact accessible path to upstream `ppo.py`. Do not claim a successful run from an unexecuted command skeleton.

## Proposed next gate — require explicit user authorization

**GPU Smoke A0** (preferred): single `PickCube-v1` Panda `physx_cuda` env, state observation, `pd_joint_delta_pos`, seed 0, **exactly 3 zero-action steps**, **external process wall-clock ≤120s**, no video, model, optimizer, dependencies, extra assets, or cloud GPU. Fresh `nvidia-smi` snapshot and stop if GPU busy or VRAM headroom insufficient (suggest >=2048 MiB free as a minimum). Record exit code, step output, GPU info, elapsed time and memory observations. If `physx_cuda` is not available, STOP and report. This test **does not** measure throughput.

**Future parallel simulator profile A1** (not yet authorized): only after A0 PASS and repaired probe/code review. Consider 16/64/128/256 envs initially, with >=1.5–2GiB free VRAM reserve, true external timeouts including setup, correctly batched GPU actions, and appropriate episode resets. Stage B end-to-end PPO training needs another approval and only then install the required `tensorboard` into the isolated venv with pinned version and `pip check`.

**Current status:** M2_SOURCE_VERIFIED / ORIGINAL_GPU_SWEEP_UNSAFE_TO_RUN / PPO_TRAINING_NOT_RUN / GPU_PHYSX_NOT_RUN.
