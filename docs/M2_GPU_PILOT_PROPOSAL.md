# M2 GPU 运行报批 — 分阶段资源上限方案

> 状态：**PROPOSAL_ONLY / NOT_APPROVED / NOT_RUN — ORIGINAL STAGE A CURRENTLY HELD FOR TECHNICAL REPAIR**
>
> **2026-10-21 independent review:** Do **not** run the Stage A 64→4096 sweep as written. The script passes an unbatched `(8,)` action to multi-env simulation (the upstream code only auto-batches when `num_envs==1`); startup and warmup are not included in the advertised per-level timer; there is no enforced total wall-clock limit; the raw env is stepped beyond the 50-step episode horizon without a documented reset strategy; VRAM protection is too weak. See `docs/M2_INDEPENDENT_REVIEW.md`. **First approved candidate should instead be a 1-env, 3-step `physx_cuda` smoke ≤120s; no `tensorboard` is required for this smoke.** Fix and statically review the multi-env probe before seeking new approval for any larger sweep.
> 日期 2026-10-21 · 上游 ManiSkill v3.0.1 `a4a4f9272ad64b1564035874b605ceb687b63ed8`
> 依据：`docs/M2_SOURCE_REVIEW.md`（源码核对）与只读 GPU 诊断（`results/raw/m2_gpu_readonly_diag.log`）
> 本文是**申请**，不是执行记录。以下每一阶段都需要研究者单独批准后才可运行。

---

## 0. 先决条件：三个依赖包（需批准，且与 GPU 无关）

实测两个官方脚本在当前 venv 里**都无法 import**：

| 包 | 谁需要 | 现状 |
|---|---|---|
| `tensorboard` | `ppo.py` 与 `ppo_fast.py` 都硬性 import | MISSING |
| `tensordict` | 仅 `ppo_fast.py` | MISSING |
| `wandb` | `ppo_fast.py` 顶层无条件 import；`ppo.py` 仅 `--track` 时导入 | MISSING |

**建议**：Stage A 与 Stage B 只用 `ppo.py`，因此**只需装 `tensorboard`**。
`tensordict` + `wandb` 留到确实要跑官方 `ppo_fast.py` 时再申请，避免无谓扩容。

```bash
# 需批准后才执行
./.venv-maniskill/bin/python -m pip install --no-cache-dir 'tensorboard>=2.16,<3'
```
预估增量 200–400 MB（含 protobuf/grpcio 等），实际以安装后 `du` 为准。

---

## Stage A — GPU 仿真冒烟与吞吐探针（无策略、无优化器、无梯度）

**目的**：回答"`physx_cuda` 在本机能不能建起来、4096 并行要多少显存、纯仿真 SPS 是多少"。
**不回答**：训练速度、收敛、成功率。

| 项目 | 设定 |
|---|---|
| 脚本 | `scripts/m2_gpu_sim_probe.py`（已写好，**未运行**） |
| 环境 | `PickCube-v1` / `panda` / `state` / `pd_joint_delta_pos` / `normalized_dense` / `physx_cuda` |
| 并行数 | **64 → 128 → 256 → 512 → 1024 → 2048 → 4096**（逐级，遇错即停） |
| 每级预算 | warmup 20 步 + 计时 100 步，**单级实测墙钟 ≤ 60 s** |
| 总墙钟上限 | **≤ 12 分钟**（含各级 setup；7 级 × ~90 s 的保守估计） |
| 动作 | 零动作（与 M1 smoke 同策略，避免引入策略语义） |
| 显存监控 | 每级记录 `torch.cuda.max_memory_allocated/reserved` + `nvidia-smi memory.free`（前后各一次） |
| 显存下限 | **free < 500 MiB 即中止后续等级** |
| 停止条件 | CUDA OOM / RuntimeError / `/dev/nvidia*` 消失 / 总时超 12 min / free < 500 MiB |
| 产出 | `results/m2_gpu_sim_probe.json`（约 10 KB）+ stdout 日志到 `results/raw/` |
| 不产出 | 无 checkpoint、无视频、无 tensorboard、无 replay buffer |

**验收判据（用于决定 Stage B 的并行数）**：
取"峰值显存 ≤ 6500 MiB 且 free ≥ 1000 MiB"的最大等级作为 Stage B 的 `num_envs` 候选，上限 1024。

> 说明：`render_mode=None`（探针脚本显式设置），而官方 `ppo.py` 用 `rgb_array`。两者的显存/SPS **不可直接等价**，报告里会分开写。

---

## Stage B — 短 PPO 端到端 pilot（打管道，不追收敛）

**目的**：验证"采样 → GAE → PPO 更新 → 评估 → 存 ckpt → tensorboard"整条链能跑通，并测**端到端 SPS**（含梯度更新）。

| 项目 | 设定 |
|---|---|
| 脚本 | 上游 `ppo.py`（**不修改**），只通过命令行覆盖 |
| 命令骨架 | `python ppo.py --env_id=PickCube-v1 --num_envs=<A 的结论> --num_steps=50 --update_epochs=4 --num_minibatches=32 --total_timesteps=256000 --seed=9351 --capture_video=False --track=False` |
| 并行数 | 由 Stage A 决定，**上限 1024**（若 4096 也放得下，仍建议先用 1024 做管道测试） |
| 总步数上限 | **256,000**（`256000 // (num_envs×50)` 个 iteration，512 env 时为 10 个） |
| 墙钟上限 | **≤ 20 分钟** |
| 磁盘上限 | **≤ 300 MB**（tensorboard events + 2 个 `.pt`，每个约 2–3 MB） |
| 评估 | `eval_freq=5`，`num_eval_envs=8`，仅看管道是否通 |
| 显存下限 | free < 400 MiB 即中止 |
| 停止条件 | OOM / NaN-Inf loss / 墙钟超限 / 连续 2 分钟无新 iteration 输出 / `/dev/nvidia*` 消失 |
| 产出 | `runs/PickCube-v1__ppo__9351__<ts>/`（events、ckpt_1.pt、final_ckpt.pt）+ 日志 `results/raw/` + 一行 SPS 记录进 `results/m2_pilot_summary.json` |

### 明确不能从这个 pilot 得出的结论

- ❌ 不能声称"复现了官方基线"：官方是 `ppo_fast.py` + 4096 env + 4 step + 8 epoch + 50M step + cudagraphs × 3 seeds；这里是 `ppo.py` + 256k step + 可能不同 env 数，**必须标 `ADAPTATION / PLUMBING_ONLY`**。
- ❌ 不能声称"PPO 收敛/失败"：256k 步在 PickCube 上通常 `success_rate ≈ 0`，这与算法好坏无关。
- ❌ 不能把纯仿真 SPS 当训练 SPS：Stage A 不含反向传播。
- ✅ 可以得出的：能否跑通、峰值显存、端到端 SPS 量级、以及**是否值得继续本地跑**。

### 算力外推（纯算术，非实测）

`hours(one seed) = 50,000,000 / SPS / 3600`，SPS 取 Stage B 的**端到端**值：

| 假设端到端 SPS | 单 seed 50M | 三 seed PickCube |
|---:|---:|---:|
| 3,000 | 4.63 h | 13.9 h |
| 6,000 | 2.31 h | 6.94 h |
| 10,000 | 1.39 h | 4.17 h |

这些只是把公式摆出来；**真实值未测**，不得在报告里当作结论。若实测落点使三 seed 超过 ~2 天，才进入"考虑 24 GB 云卡"的讨论（仍另需批准）。

---

## 审批清单（请逐项勾选后再执行）

- [ ] **A-1** 批准安装 `tensorboard`（约 200–400 MB）
- [ ] **A-2** 批准运行 Stage A 探针：7 个并行等级、单级 ≤60 s、总 ≤12 min、free<500 MiB 即停
- [ ] **B-1** 批准 Stage B pilot：`ppo.py`、≤256k env steps、≤20 min、≤300 MB、无 wandb、无视频
- [ ] **B-2** 指定 Stage B 的 `num_envs`（或授权我按 Stage A 结论自动取"峰值 ≤6500 MiB 的最大值，上限 1024"）
- [ ] 确认不在本轮申请：官方 4096×50M 三 seed、`ppo_fast.py`（需 tensordict+wandb）、云 GPU、checkpoint 下载

---

## 执行时必记的审计字段（每次 GPU 运行）

`timestamp` / `git commit` / `/dev/nvidia*` 是否存在 / `nvidia-smi` 机型-总显存-空闲 / `torch.__version__` + `torch.version.cuda` / `torch.cuda.is_available()` / `num_envs` / `num_steps` / `total_timesteps` / `seed` / `sim_backend` / `render_mode` / 峰值显存 / 端到端 SPS / 退出码 / 产物路径。

理由见 `docs/M2_SOURCE_REVIEW.md` §9.3：**agent 壳的 GPU 可见性依赖会话**，不记录现场证据就无法审计。
