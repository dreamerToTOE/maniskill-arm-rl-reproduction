# M2 GPU 运行报批 — 分阶段资源上限方案

> 状态：**A0 PASSED (12/12) / A1 PASSED (4/4, 27.706 s) / STAGE B PASSED (40/40, e2e 2400.4 SPS, 5137 MiB peak)**
>
> **2026-10-21 A1 结果（实测）**：16/64/128/256 的聚合仿真 SPS = **905.8 / 3342.0 / 6437.4 / 11746.1**，
> 扩展效率 1.000/0.922/0.888/0.811；设备级显存峰值 **3912 / 3867 / 3931 / 3931 MiB**，全程 free ≥ 4261 MiB。
> **显存在 256 以内几乎不随并发增长**（边际 1.77 / 1.00 / 0.00 MiB per env，~1.3 GiB 场景是固定成本），
> 所以本阶段的约束不是显存而是每步固定开销。torch 峰值仅 8.2–8.6 MiB（≈1/160，不能用来估 PhysX）。
> 详见 `docs/M2_A1_EXECUTION_REPORT.md` §6-§7。**Stage B 建议 `num_envs=128`**（预登记规则取 256 再减半）。
>
> **2026-10-21 修复完成：** 原 Stage A 64→4096 扫描已被独立复核否决并冻结。修复后的方案拆成两个脚本，
> 逐项修复见 `docs/M2_GPU_PROBE_REPAIR.md`，静态检查证据见 `results/raw/m2_static_checks.log`：
> **A0** = `scripts/m2_gpu_smoke_a0.py`（1 env / 3 步零动作 / 内部 100 s + 外部 `timeout 120` / 零依赖），
> **A1** = `scripts/m2_gpu_sim_probe.py`（单等级）+ `scripts/m2_a1_run_all.py`（每级独立进程的驱动器），
> 默认 16/64/128/256，512 需 `--allow-512`，>256 在驱动器层拒绝。
> **执行情况：A0 已于 15:44 由研究者在终端 PASS（12/12）；A1 于 16:09 在本会话 `BLOCKED`（0 等级）。**
>
> **2026-10-21 independent review:** Do **not** run the Stage A 64→4096 sweep as written. The script passes an unbatched `(8,)` action to multi-env simulation (the upstream code only auto-batches when `num_envs==1`); startup and warmup are not included in the advertised per-level timer; there is no enforced total wall-clock limit; the raw env is stepped beyond the 50-step episode horizon without a documented reset strategy; VRAM protection is too weak. See `docs/M2_INDEPENDENT_REVIEW.md`. **First approved candidate should instead be a 1-env, 3-step `physx_cuda` smoke ≤120s; no `tensorboard` is required for this smoke.** Fix and statically review the multi-env probe before seeking new approval for any larger sweep.
> 日期 2026-10-21 · 上游 ManiSkill v3.0.1 `a4a4f9272ad64b1564035874b605ceb687b63ed8`
> 依据：`docs/M2_SOURCE_REVIEW.md`（源码核对）与只读 GPU 诊断（`results/raw/m2_gpu_readonly_diag.log`）
> 本文是**申请**，不是执行记录。以下每一阶段都需要研究者单独批准后才可运行。

---

## 0. 先决条件：依赖与阶段解耦

实测两个官方脚本在当前 venv 里**都无法 import**：

| 包 | 谁需要 | 现状 | A0 需要吗 |
|---|---|---|---|
| `tensorboard` | `ppo.py` 与 `ppo_fast.py` 都硬性 import | **已安装 2.21.0（2026-10-21，Stage B 授权）** | 不需要 |
| `tensordict` | 仅 `ppo_fast.py` | MISSING（**未安装，也不在本阶段范围**） | 不需要 |
| `wandb` | `ppo_fast.py` 顶层无条件 import；`ppo.py` 仅 `--track` 时导入 | MISSING（**未安装；本轮用 `--no-track`**） | 不需要 |

**修正（独立复核第 6 条）**：A0 只用到 `torch` / `gymnasium` / `mani_skill` / `sapien`，**一个包都不用装**。
`tensorboard` 是 Stage B（真的跑 `ppo.py`）的前置，已于 2026-10-21 单独批准并安装；
`tensordict` + `wandb` 只属于 `ppo_fast.py`，**本轮明确不装**。

```bash
# 实际执行过（2026-10-21），结果：tensorboard 2.21.0 / protobuf 7.36.2 / grpcio 1.84.0
# / markdown 3.10.3 / werkzeug 3.1.9 / tensorboard-data-server 0.7.2；pip check 干净；
# 磁盘增量约 64 MB（未保留 wheel 缓存）。venv 由 5.9 GB → 约 5.97 GiB。
./.venv-maniskill/bin/python -m pip install --no-cache-dir 'tensorboard>=2.16,<3'
```
预估增量 200–400 MB（含 protobuf/grpcio 等），实际以安装后 `du` 为准。

## 0.1 Stage A0 实际执行命令（**已 PASS，12/12**）

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv
timeout --signal=TERM --kill-after=10s 120 \
  ./.venv-maniskill/bin/python scripts/m2_gpu_smoke_a0.py \
    --env-id PickCube-v1 --robot panda --seed 0 --steps 3 \
    --budget-seconds 100 --min-free-mib 3072 --abort-free-mib 1536 \
    --output results/m2_gpu_smoke_a0.json 2>&1 | tee results/raw/m2_gpu_smoke_a0.log
echo "exit=${PIPESTATUS[0]}"
```

通过判据、退出码含义与 A0 不能证明的事项：`docs/M2_GPU_PROBE_REPAIR.md` §3。


---

## Stage A — 已拆分为 A0（正确性）与 A1（吞吐）

**已被否决的形态**：单脚本 64→4096 扫描 + `--max-seconds` 只覆盖计时循环。见 `docs/M2_INDEPENDENT_REVIEW.md`。

### Stage A0 — 单环境 `physx_cuda` 冒烟（**这是下一步唯一应执行的 GPU 动作**）

**目的**：回答"`physx_cuda` 在本机能不能建 1 个场景、reset、走 3 步"。
**不回答**：吞吐、多环境显存、训练可跑性、收敛。
完整规格与通过判据见 `docs/M2_GPU_PROBE_REPAIR.md` §3；命令见本文 §0.1。

| 项目 | 设定 |
|---|---|
| 脚本 | `scripts/m2_gpu_smoke_a0.py`（**已修复并通过静态检查，未运行**） |
| 环境数 / 步数 | **1** / **恰好 3** 步零动作 |
| 动作 | `torch.zeros((1, 8), device=cuda)`，显式断言形状 |
| 后端 | `sim_backend="physx_cuda"`（显式；回读 `backend.sim_backend` 与 `gpu_sim_enabled` 校验） |
| 时间 | 内部 100 s（monotonic + watchdog）/ 外部 `timeout 120` |
| 显存 | 起跑需 free ≥ 3072 MiB；跌破 1536 MiB 即中止 |
| 依赖 | **零新增**（不需要 `tensorboard`） |
| 产出 | `results/m2_gpu_smoke_a0.json` + `results/raw/m2_gpu_smoke_a0.log` |

### Stage A1 — 并行吞吐探针（**2026-10-21 已获授权；已加固并通过静态检查；执行 `BLOCKED`**）

| 项目 | 设定 |
|---|---|
| 脚本 | `scripts/m2_gpu_sim_probe.py`（单等级）+ `scripts/m2_a1_run_all.py`（驱动器） |
| 并行数 | **16 → 64 → 128 → 256**（512 需 `--allow-512`；>256 在驱动器层直接拒绝） |
| 隔离方式 | **每个等级一个独立进程、独立 JSON、独立日志**；任一等级失败即停止，剩余等级记为 `NOT_RUN_AFTER_FAILURE`；`attempts: 1`，无自动重试 |
| 每级预算 | warmup 10 步 + 计时 100 步，**单级 ≤ 90 s**（含 build/reset/warmup/measure/cleanup） |
| 进程预算 | **≤ 120 s**（import + build + steps + cleanup，monotonic + watchdog） |
| 外部上限 | 每级 `timeout --signal=TERM --kill-after=10s 150`；整轮外部上限 4×150 = 600 s |
| 动作 | 零动作，形状 `(num_envs, 8)`，device=cuda |
| 回合语义 | `ManiSkillVectorEnv(auto_reset=True)`，记录 step 级与 env 级 terminated/truncated/auto_reset；**零截断即判失败** |
| 显存 | 起始闸 3072 MiB、运行闸 1536 MiB、每 10 步设备级采样；torch 峰值只作下界（记录但不用于估算） |
| 遥测 | **强制**：`nvidia-smi` 不可用则在 `import torch` 之前退出 8，不建任何场景 |
| 产出 | `results/m2_a1_benchmark_summary.json` + `results/raw/m2_a1_n{16,64,128,256}.{json,log}` |

**执行命令（一行，研究者终端；详见 `docs/M2_A1_EXECUTION_REPORT.md` §6）**：

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
./.venv-maniskill/bin/python scripts/m2_a1_run_all.py 2>&1 | tee results/raw/m2_a1_launcher.log
```

退出码：`0` = 四级全过；`1` = 部分完成；`2` = 首级失败；`8` = 遥测不可用（本轮即此）。

**验收判据（决定 Stage B 的并行数）**：取满足
`设备级 used 峰值 ≤ 6500 MiB`、`free_min ≥ 1000 MiB`、`scaling_efficiency ≥ 0.5`、且无 error/warning
的**最大**等级，然后**再减半**作为 Stage B 首个 pilot 的 `num_envs`（训练路径还要分配
策略/优化器状态，官方 fast 脚本还要 `rgb_array` 渲染）。若 16 级都过不了 headroom 判据，
则本地 Stage B 不可行。

**判据结果（2026-10-21，四级全部满足）**：max = **256** → 减半 → **Stage B `num_envs = 128`**。

> 当前状态：**PASS** — 2026-10-21 16:18 研究者终端执行，`verdict=PASS`、4/4 级、exit 0、27.706 s。
> 五问结论见 `docs/M2_A1_EXECUTION_REPORT.md` §7；报告 §8 给出 `num_envs=128` 的**临时**预算。

> `render_mode=None` 与官方 `ppo.py` 的 `rgb_array` 不可等价；A1 的 SPS 也不含前向/反向，不可外推为训练速度。

---

## Stage B — 短 PPO 端到端 pilot（打管道，不追收敛）

**目的**：验证"采样 → GAE → PPO 更新 → 评估 → 存 ckpt → tensorboard"整条链能跑通，并测**端到端 SPS**（含梯度更新）。

| 项目 | 设定 |
|---|---|
| 脚本 | 上游 `ppo.py`（**不修改**），只通过命令行覆盖 |
| 命令骨架 | **Tyro 是 kebab-case，不能用 `--capture_video=False`**（那样会在解析阶段直接报错）。已用 `--help` 核实，实际命令：`python ppo.py --env-id PickCube-v1 --exp-name PickCube-v1__ppo__9351__stageB --num-envs 128 --num-steps 50 --total-timesteps 256000 --update-epochs 4 --num-minibatches 32 --num-eval-envs 8 --eval-freq 5 --seed 9351 --no-capture-video --no-track --save-model` |
| 并行数 | **128**（A1 结论：256 达标后减半；显存余量充足，余量留给策略/优化器/渲染） |
| 总步数上限 | **256,000** → `256000 // (128×50)` = **40 个 iteration** |
| 墙钟上限 | **≤ 1200 s**（外部 `timeout 1260` 兜底） |
| 磁盘上限 | **≤ 300 MB**。**checkpoint 数量是 9 个不是 2 个**：`ppo.py` 在 `iteration % eval_freq == 1` 时存盘 ⇒ 第 1/6/11/16/21/26/31/36 次 = **8 个周期 ckpt**，再加收尾的 `final_ckpt.pt` = **9 个 `.pt`**（每份约 1.2–3 MB，合计 < 30 MB），主要占用是 tensorboard events |
| 评估 | `eval_freq=5`（共 8 次评估，每次 50 步 × 8 env = 400 env-steps），`num_eval_envs=8`，仅看管道是否通 |
| 显存下限 | free < 1536 MiB 即中止（起始闸 3072 MiB） |
| 停止条件 | CUDA OOM / NaN-Inf loss / 墙钟超限 / 连续 120 s 无新 iteration / 显存跌破下限 / 输出 > 300 MB |
| 产出 | `runs/PickCube-v1__ppo__9351__stageB/`（events + 9 个 ckpt）+ 逐行日志 `results/raw/m2_stage_b_pilot.log` + `results/m2_pilot_summary.json` |

> **审计发现（2026-10-21）**：`ppo.py:195` 把 `render_mode="rgb_array"` **硬编码**在 `env_kwargs` 里，
> 且没有对应 CLI 开关。所以 Stage B 一定带渲染，A1（`render_mode=None`）的 3931 MiB 峰值**不能**直接套用，
> 显存会更高。这是本阶段最大的未知风险，也是 1536 MiB 运行闸存在的理由。
>
> **实测（attempt 2，2026-10-21 21:08，PASS）**：**40/40 iteration，e2e 2400.4 SPS / loop 2600**，
> 峰值显存 **5137 MiB**（比 A1 无渲染的 3931 高 **1206 MiB / +31%**，验证了 `render_mode` 硬编码预判），
> free 最低 2682 MiB；256,000 训练步 + 3,200 评估步，9 个 ckpt。attempt 1（20:51）被监督器 stdout
> 缓冲 bug 误杀在 29/40，已修（`-u` + 两段式看门狗）。详见 `docs/M2_STAGE_B_EXECUTION_REPORT.md`。
>
> **256-env pilot（2026-10-22 11:22，PASS）**：agent 会话恢复 GPU 可见性后直接跑（`verdict: OK`）。
> **20/20 iteration，e2e 2885.8 / loop 4435 SPS（128 的 loop 是 2600，+71%）**，峰值显存
> **4601 MiB / 最低 free 3217 MiB**——加一倍环境**没有**触发显存墙（128→256 的场景+渲染成本基本不随
> env 数涨，与 A1 一致）。稳态外推：256 env ≈ **3.13 h/seed、三 seed ~9.4 h**（128 是 ~16 h）。
> 详见 `docs/M2_STAGE_B_256_PILOT.md`。
>
> **512-env pilot（2026-10-22 11:47，PASS）**：研究者明确授权测 512（A1 硬上限），"不够就杀停"。
> 结果**没触发杀停**：10/10 iteration，loop **8236** SPS（128→256→512 = 2600→4435→8236，3.17× 于 128、
> 效率 0.79），峰值显存 **4777 MiB / 最低 free 3041 MiB**——到 512 显存仍是平的。
> 稳态外推：512 env ≈ **1.7 h/seed、三 seed ~5 h**。详见 `docs/M2_STAGE_B_512_PILOT.md`。
>
> **1024-env pilot（2026-10-22 11:57，PASS）**：研究者授权测 1024（超出项目 512 硬上限，属显式授权）。
> 5/5 iteration，loop **15117** SPS（2600→4435→8236→15117，5.81× 于 128），峰值显存 **4862 MiB /
> 最低 free 2956 MiB**——**到 1024 显存仍是平的**，推翻了"8 GiB 瓶颈是并行环境"的旧判断。
> 稳态外推：1024 env ≈ **0.92 h/seed、三 seed ~2.8 h**。详见 `docs/M2_STAGE_B_1024_PILOT.md`。

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
- [x] **A-2** 批准运行 Stage A1 探针（2026-10-21）：16/64/128/256、单级 ≤90 s、每级独立进程 + 外部 `timeout 150`、起始闸 3072 MiB、运行闸 1536 MiB、失败即停 —— **已执行 `PASS`（4/4，exit 0，27.706 s），结论 `num_envs=128`**
- [ ] **B-0** 安装 `tensorboard`（约 200–400 MB）并 `pip check` —— 尚未批准
- [ ] **B-1** 批准 Stage B pilot：`ppo.py`、`num_envs=128`、≤256k env steps、≤20 min、≤300 MB、无 wandb、无视频
- [ ] **B-2** 确认 `num_envs=128`（或改为 256；A1 显示显存足够，但 256 没有减半余量）
- [ ] 确认不在本轮申请：官方 4096×50M 三 seed、`ppo_fast.py`（需 tensordict+wandb）、云 GPU、checkpoint 下载

---

## 执行时必记的审计字段（每次 GPU 运行）

`timestamp` / `git commit` / `/dev/nvidia*` 是否存在 / `nvidia-smi` 机型-总显存-空闲 / `torch.__version__` + `torch.version.cuda` / `torch.cuda.is_available()` / `num_envs` / `num_steps` / `total_timesteps` / `seed` / `sim_backend` / `render_mode` / 峰值显存 / 端到端 SPS / 退出码 / 产物路径。

理由见 `docs/M2_SOURCE_REVIEW.md` §9.3：**agent 壳的 GPU 可见性依赖会话**，不记录现场证据就无法审计。
