# M1 START HERE — PickCube task introspection (executed; awaiting review)

> State: **ENV_INSTALLED / CPU_SMOKE_EXECUTED_PASSED / INTERFACE_RUNTIME_VERIFIED / M1_NOT_MARKED_PASSED (researcher review pending)**. Date: 2026-10-20.
> Full evidence, measured numbers and deviations: **[M1_EXECUTION_REPORT.md](M1_EXECUTION_REPORT.md)**.
> Independent repo: dreamerToTOE/maniskill-arm-rl-reproduction
> Executed base commit: `4a4e5ff92fd4a1de13e9b0ed57a09d15d363e902`
> Upstream proposal: ManiSkill v3.0.1, immutable commit `a4a4f9272ad64b1564035874b605ceb687b63ed8` (PyPI wheel 3.0.1 actually installed)

## Scope and gating

M0 source audit and the learner's central PPO/SAC conceptual review are sufficiently advanced to **prepare** M1. Learner provided a real read-only host audit on 2026-10-20: CPython 3.10.12, RTX 3070 with 8192 MiB total, NVIDIA driver 580.178.04, NumPy 1.24.4; no torch/sapien/gymnasium/mani_skill distributions in that interpreter. Formal M0 gate still open because isolated dependency integration and CPU smoke evidence are missing (free VRAM 5957 MiB and disk free 418G observed). On 2026-10-20 researcher approved the *bounded isolated installation and one CPU smoke*, not training or GPU microbench. Do not falsely mark M0 PASSED. Prepare M1 without training or changing the main dual-FR3 project.

Researcher owns acceptance/experimental design; Codex owns environment inspection, isolated setup on user-approved host, implementation and evidence. Do NOT run training, install system packages, upgrade CUDA/drivers/ROS2/Isaac Sim, or modify other repos.

## Task-specific source map — pinned v3.0.1

| Investigation | Exact upstream source | Expected behavior from code reading |
|---|---|---|
| Gym task registration / horizon | `mani_skill/envs/tasks/tabletop/pick_cube.py` `@register_env("PickCube-v1", max_episode_steps=50)` | 50 environment control steps max, unless wrapper/horizon overridden |
| Panda robot/controller | `mani_skill/agents/robots/panda/panda.py` `_controller_configs` | `pd_joint_delta_pos` normalized joint-position increments, arm ±0.1 rad, gripper included; `pd_ee_delta_pos` arm translation ±0.1 m/axis, gripper included |
| State observation | `mani_skill/envs/sapien_env.py` `_get_obs_state_dict`, `_flatten_raw_obs`; `pick_cube.py` `_get_obs_extra` | `state_dict = {"agent": proprioception(qpos,qvel,...), "extra": task signals}`; `state` is a flattened tensor, task extra includes grasp indicator, tcp and goal pose, object pose + relative vectors |
| Reset/randomization | `pick_cube.py` `_initialize_episode`, `sapien_env.py` `reset` | Random cube XY/yaw and goal XY/Z; try same seed, record what actually matches; do not assume GPU bitwise determinism |
| Reward | `pick_cube.py` `compute_dense_reward`, `compute_normalized_dense_reward` | reach + grasp + grasp-gated place + placed-gated static, success override 5; normalized reward /5 |
| Success | `pick_cube.py` `evaluate` | distance(cube,goal) <= goal_thresh + Panda arm static threshold, no force-based safety criterion |
| Dynamics/done | `sapien_env.py` `step` / `_step_action`, `mani_skill/utils/structs/types.py` `SimConfig` | default 100Hz sim, 20Hz control = 5 physics steps per env.step. Terminated on success/fail; horizon truncation comes from wrapper |
| CPU vs GPU wrappers | `mani_skill/utils/wrappers/gymnasium.py` `CPUGymWrapper`; `mani_skill/vector/wrappers/gymnasium.py` `ManiSkillVectorEnv` | vectorized/CPU shapes, reset/auto-reset and ignore_terminations change semantics; must log wrappers |

Source tree: https://github.com/mani-skill/ManiSkill/tree/a4a4f9272ad64b1564035874b605ceb687b63ed8/

## Work package M1-A — read-only host audit and review (no install)

1. **DONE user-reported**: `python3 scripts/audit_environment.py` returned Python 3.10.12, RTX 3070 8GB, driver 580.178.04; output captured in docs/UPSTREAM_LOCK.md. **ALSO DONE user-reported**: `nvidia-smi` → 5957 MiB free VRAM; `df -h .` → 418G disk free; `python3 -m pip --version` → global pip 22.0.2. **Next:** obtain approval for proposed isolated pip install and bounded CPU smoke in `docs/M1_INSTALL_APPROVAL.md`.
2. Confirm Python, torch CUDA runtime, SAPIEN/gymnasium, GPU free VRAM, target isolated venv path; confirm source SHA. No system changes. Record hardware evidence in `docs/UPSTREAM_LOCK.md` and `docs/PROGRESS.md`.
3. Propose exact Python/Torch/SAPIEN wheel pins after inspecting local output and dependency compatibility. Installation requires explicit user permission. Disk/network costs TBD until solved; *do not imply zero installs*.

## Work package M1-B — bounded CPU state smoke (researcher approved, not yet run)

After dependencies are installed in a dedicated isolated environment, under the user's **2026-10-20 explicit approval of this concrete smoke test**, run from orchestration repo with ManiSkill v3.0.1 importable:

```bash
# Inspect structured fields and execute at most THREE zero actions
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 python scripts/m1_inspect_pickcube.py \
  --obs-mode state_dict --action-mode zero --steps 3 --seed 0 \
  --output results/m1_pickcube_state_dict_zero_seed0.json

# Inspect the flattened state representation, separately (optional)
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 python scripts/m1_inspect_pickcube.py \
  --obs-mode state --action-mode zero --steps 3 --seed 0 \
  --output results/m1_pickcube_state_zero_seed0.json
```

Expected command upper limit: **three environment steps per command, one CPU environment, no renders and no gradient update**, normally <5 minutes after dependencies/assets are installed; runtime unverified on this machine. CPU smoke avoids planned GPU parallel training allocation; runtime may still initialize graphics libraries depending on local installation. Approximate *training* VRAM requirement: none; runtime GPU allocation to be verified rather than invented. Disk for environment/wheels/assets **not measured yet**; filesystem has 418G free, with a separately proposed installation cap in `docs/M1_INSTALL_APPROVAL.md`. Stop if download prompts, driver/Vulkan failure, missing assets, unbounded run, or unexpected resources.

If the smoke is successful, **separately** request approval for an opt-in tiny arm action (normalized 0.1 on arm joint 1) to inspect controller scaling. It is NOT a grasping demo and cannot establish baseline success.

## Required evidence and acceptance

- Current installed version manifest, source SHA, exact commands, timestamp/host, smoke stdout and JSON artifacts.
- Print full structured observation key names, tensor shapes/dtypes and flattened dimension; action low/high and dimensions, controller mode, sim/control rates, reward mode.
- At every step record scalar reward, info success / grasp / placed / static if present, and terminated/truncated; capture the chosen action and number of steps.
- Verify (not assume) gripper dimension, action unit scaling, reset reproducibility, trial and time-limit semantics; note when only code is inspected rather than execution verified.
- No claims of training or learning. `M1 PASSED` only after researcher's 3-question review AND evidence of actual local reset/step smoke. M0 completion must be separately recorded once host audit and smoke approval criteria are met.

## Three researcher understanding questions

> **先读学习包：[docs/M1_LEARNING_PACK.md](M1_LEARNING_PACK.md)**（原理一页纸 + 代码导读 + 三道题的完整上下文）。
> 下面的问题来自执行前的准备文档，学习包里已用本次实测数据重写并补充了提示来源。

1. Why do `obs_mode=state_dict` and `obs_mode=state` expose the same underlying information in different formats? Which features are privileged simulation ground truth?
2. Under Panda `pd_joint_delta_pos`, is a normalized action `a[0]=0.1` a 0.1-radian realized joint displacement? Trace policy output -> controller scaling -> PD -> PhysX; what else may prevent actual motion?
3. In PickCube, why is `is_grasped=True` insufficient for success? Distinguish `success`, `terminated`, `truncated` and how `ignore_terminations` alters rollouts.

### M1 status
ENV_INSTALLED (`.venv-maniskill`, pip check clean) / CPU_SMOKE_PASSED (3 zero actions, exit 0) / INTERFACE_RUNTIME_VERIFIED (42-dim state, 8-dim action, 100/20 Hz, horizon 50) / GPU_UNVERIFIED_IN_AGENT_SHELL / M1_NOT_MARKED_PASSED / TRAINING_NOT_RUN

Executed command (actual):

```bash
TMPDIR=/home/ubuntu2004/.cache/ms-m1-tmp MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
./.venv-maniskill/bin/python scripts/m1_inspect_pickcube.py \
  --obs-mode state_dict --action-mode zero --steps 3 --seed 0 \
  --output results/m1_pickcube_state_dict_zero_seed0.json   # exit 0
```

Measured: reward `0.06530120 / 0.06530116 / 0.06530116`; `terminated=False`, `truncated=False`,
`success=False`, `is_grasped=False`, `is_obj_placed=False`, `is_robot_static=True`.
The reward was independently reproduced from `‖tcp−cube‖` via `(1−tanh(5d))/5`, confirming the source formula.

Two deviations were required and are documented in the report §6: `TMPDIR` redirection
(`/tmp` is a 10 MB tmpfs; pip errno 28) and a **real bug fix in this script** — `horizon` read
`env.spec.max_episode_steps`, which is `None` under gymnasium 1.1.1, instead of the
`TimeLimitWrapper` budget. The pre-fix artifact is preserved under `results/raw/PREFIX_horizon_null_*`.

### Host utilization actually consumed
- venv on disk: 6.21 GB (budget 12 GB); downloads well under the 5 GB cap (torch 780 MB + triton 210 MB + CUDA libs + ~100 MB ManiSkill + 51 MB SAPIEN).
- Extra GPU/render assets: **0 bytes** — PickCube used packaged assets, no download prompt appeared.
