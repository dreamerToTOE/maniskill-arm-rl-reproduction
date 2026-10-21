# M2 GPU Probe — Repair Report (Stage A0 ready, nothing executed)

> 日期 2026-10-21 · 状态 **PROBE_REPAIRED / STATIC_CHECKS_PASS / A0_ATTEMPTED_ONCE_BLOCKED_ON_GPU_VISIBILITY / A1_NOT_AUTHORIZED**
> 执行记录见 §7：A0 已按批准命令跑过**一次**，在建场景之前被强制遥测闸拦下（退出码 8），本轮 session 内无 `/dev/nvidia*`。
> 依据：`docs/M2_INDEPENDENT_REVIEW.md`（6 项阻塞缺陷）、本文件（逐项修复对照）
> 上游锁定不变：ManiSkill `3.0.1` @ `a4a4f9272ad64b1564035874b605ceb687b63ed8`，SAPIEN `3.0.0`，torch `2.5.1`
> **本轮没有执行任何 GPU 仿真、没有安装任何依赖、没有创建过 CUDA context。**

---

## 1. 两个脚本，两种问题

原方案把「1 个环境能不能跑起来」和「N 个环境有多快」混在一个 64→4096 的扫描里。这是错的方向：
前者是**正确性**问题（后端是否真的在 GPU 上、obs/action/reward/done 是否合理），后者才是**性能**问题。
一旦第一级就卡住，12 分钟的预算全被一个尚未验证的前置假设吃掉。

| 文件 | 阶段 | 环境数 | 步数 | 默认时间 | 目的 | 状态 |
|---|---|---|---|---|---|---|
| `scripts/m2_gpu_smoke_a0.py` | **A0** | 1 | 3（零动作） | 内部 100 s / 外部 120 s | 证明 `physx_cuda` 能建、能 reset、能走 3 步 | **已修复，未运行** |
| `scripts/m2_gpu_sim_probe.py` | **A1** | 16/64/128/256（512 需显式授权） | 10 预热 + 100 计时 | 每级 90 s / 全程 600 s | 模拟器 SPS 随并行数变化 | **已重写，未获授权** |

---

## 2. 逐项修复对照（7 条要求 + 独立复核 6 项）

### 2.1 动作必须分批为 `(num_envs, action_dim)`

**缺陷**：原脚本 `np.zeros(env.single_action_space.shape)` → 形状 `(8,)`。
**源码依据**：`sapien_env.py:1105-1110`

```python
if set_action:
    if self.num_envs == 1 and action_is_unbatched:
        action = common.batch(action)      # ← 只有 n==1 才自动补 batch 维
    self.agent.set_action(action)
```

即 `num_envs>1` 时上游**不会**替你补维度，一个 `(8,)` 会被直接交给控制器。
**修复**：两个脚本都只通过 `batched_action_shape(num_envs, action_shape)` 构造动作，
并断言 `tuple(action.shape) == (num_envs, *action_shape)`；张量建在 `env.unwrapped.device` 上。

```python
shape = batched_action_shape(1, tuple(env.single_action_space.shape))   # (1, 8)
action = torch.zeros(shape, dtype=torch.float32, device=unwrapped.device)
assert tuple(action.shape) == shape
```

静态审计会拒绝任何 `*.zeros(... single_action_space ...)` 的裸用法（`scripts/m2_static_audit.py`）。

### 2.2 墙钟必须覆盖初始化、reset、预热、计时与清理

**缺陷**：`--max-seconds` 只包住计时循环；场景创建（GPU 上建 articulation/contact buffer）与 warmup 不受控；
"12 分钟总预算"根本没有实现。

**修复**：三层时间约束，任何一层单独都足以终止进程。

| 层 | 机制 | 覆盖范围 |
|---|---|---|
| L1 外部 | 命令里的 `timeout --signal=TERM --kill-after=10s 120` | 整个进程，兜底（含 SIGTERM 不响应时强杀） |
| L2 进程 | `Deadline` 以 `time.monotonic()` 锚定在**模块导入时刻**（`_T0`），每个 phase 边界与**每一步**前 `check()` | 导入、建环境、reset、每步、清理 |
| L3 线程 | `arm_watchdog()` 的 daemon `threading.Timer`，到点写盘后 `os._exit(124)` | 卡在不可中断的 CUDA 调用里也能退出 |

A1 还多一层**每级独立预算**（`--level-seconds 90`）：某一级建场景卡死时，该级被放弃、进程停止，
不会把后面几级的预算吃光。所有计时都扣除 `nvidia-smi` 采样自身耗时（`vram_sampling_overhead_s`）。

证据是**增量写盘**：`Recorder.phase()` 在每个 phase 结束后就 `os.replace` 落盘一次，
所以即使被外部 `kill -9`，JSON 里也留着"死在哪一 phase、已用多少秒"。

### 2.3 截断与自动重置

**缺陷**：裸 `env.step` 走 20+100 步，而 horizon 是 50，脚本从不处理到点截断。
**修复**：两个脚本都用 `ManiSkillVectorEnv(base, auto_reset=True, ignore_terminations=False)`，
即 PPO 基线同款语义（`ppo.py:212-213`）。源码行为（`vector/wrappers/gymnasium.py:164-175`）：

```python
dones = torch.logical_or(terminations, truncations)
if dones.any() and self.auto_reset:
    final_obs = torch_clone_dict(obs)          # 重置前的 o_{t+1}
    obs, infos = self.reset(options=dict(env_idx=env_idx))
    infos["final_observation"] = final_obs
```

A1 每级记录 `counters = {terminated, truncated, auto_reset}`，并要求 `truncated > 0`
（10+100 = 110 步 > 50 步 horizon，截断**必然**发生，不发生说明观测/时间限制读错了）。
A0 只有 3 步，且零动作下不可能成功，因此脚本把「3 步内出现 done」判为 **`UNEXPECTED_DONE` 失败**
（退出码 6），而不是静默继续——这比"忽略 done"更严格。

**边界说明**：A1 测的是"模拟器在稳态下的步进速度"，auto-reset 会周期性地混入 reset 开销，
所以这个数字**偏保守**，且**不能**当作 PPO 的端到端 SPS 外推（后者还含前向、反向、GAE、eval）。

### 2.4 8 GiB 显存的保守保护

**缺陷**：`free<500 MiB` 才停——那时已经太晚；且只在每级前后采样一次。

**修复**：三道闸，全部读**设备级**数字。

| 闸 | 默认值 | 语义 |
|---|---|---|
| 起始闸 `--min-free-mib` | **3072 MiB** | 建场景**之前**就判定；不足则 `SKIPPED_LOW_VRAM`（退出码 3），不尝试 |
| 运行闸 `--abort-free-mib` | **1536 MiB** | 每 `--vram-every 10` 步采一次；跌破则抛出 `MemoryError` → `VRAM_GUARD_TRIPPED`，**主动停**而不是等 OOM |
| 级别上限 | **≤512 env**（512 需 `--allow-512`） | 1024/2048/4096 直接拒绝（`validate_env_counts` 抛错，退出码 5） |

选 3072 MiB 起始闸的理由：本机 8192 MiB 里桌面已占约 1043 MiB，A0 只需一个场景；
留 3 GiB 意味着"不把桌面挤到 swap/被杀"。1536 MiB 停闸是为了在 OOM 之前留下可回收余量，
因为 PhysX 的显存失败往往表现为 `RuntimeError` 而非干净的 `OutOfMemoryError`。

**没有**使用 `torch.cuda.set_per_process_memory_fraction` 作为保护：它只约束 PyTorch 的 caching allocator，
对 SAPIEN/PhysX 直接发起的 CUDA 分配无效，用它会造成"以为有保护"的错觉。脚本里对此有注释说明。

### 2.5 单环境冒烟与并行吞吐彻底分离

见 §1 表。两者不共享 CLI、不共享输出文件、不共享通过判据：
A0 输出 `results/m2_gpu_smoke_a0.json`（正确性证据），A1 输出 `results/m2_gpu_sim_probe.json`（吞吐证据）。

### 2.6 不得用 PyTorch 分配量估算 SAPIEN/PhysX 总显存

**修复**：输出里两类数字**分列且明确标注**。

| 字段 | 来源 | 含义 |
|---|---|---|
| `torch_peak_alloc_mib` / `torch_peak_reserved_mib` | `torch.cuda.max_memory_*` | **只是 torch 分配器的下界**，看不见 PhysX |
| `device_free_mib`（前/后/逐步最小值 `vram_free_min_mib`） | `nvidia-smi memory.free` | 设备级真实余量 |
| `device_used_delta_mib` | `nvidia-smi memory.used` 前后差 | 该场景的设备级显存成本（含桌面噪声） |
| `device_used_peak_mib_est` | `total − min(free samples)` | 运行期间设备级峰值（估算，含其他进程） |
| `other_compute_apps` | `nvidia-smi --query-compute-apps` | 起跑前谁在占卡，避免把别人的占用算成自己的 |

`attribution_note` 字段把这段话写进每一份 JSON，防止以后有人只贴 torch 数字。

### 2.7 保持 pinned 基线不变

- 不 `pip install`、不 `git submodule`、不改上游任何文件（上游本来也没 vendor 进仓库）。
- 脚本运行时断言 `mani_skill == 3.0.1`、`sapien == 3.0.0`、`torch` 前缀 `2.5.1`，不匹配则 `VERSION_MISMATCH`（退出码 7）；
  只有显式 `--allow-version-mismatch` 才继续，且会把 mismatch 写进 JSON。
- 记录 `torch.version.cuda` 与 `gymnasium` 版本，便于事后复现。
- **新发现的坑（原方案会踩）**：`sim_backend` 默认是 `"auto"`，而 `sapien_env.py:233-237` 规定
  `num_envs == 1` 时 `auto` → **`physx_cpu`**。若 A0 不显式传 `physx_cuda`，会在 CPU 上跑完 3 步并"通过"，
  得到一个**假阳性**。所以两个脚本都硬编码 `sim_backend="physx_cuda"`，并在建完环境后回读校验：

```python
backend_name = str(unwrapped.backend.sim_backend)     # 必须 == "physx_cuda"
gpu_sim      = bool(unwrapped.gpu_sim_enabled)        # 必须 is True
```

任一不满足 → `FAIL_SIM_BACKEND`（退出码 2），不写 PASS。

---

### 2.8 显存遥测缺失 = 建场景前硬中止（第二轮补强）

**问题**：第一版在 `nvidia-smi` 读不到时只把 `verdict` 记成 `UNKNOWN` 并继续，等于"看不见显存也要跑"。
这既违反 8 GiB 的安全前提，也让事后无法审计——PhysX 的分配量只有设备级计数器看得见。

**修复**：`telemetry_verdict()` 成为强制闸，位置在 `import torch` **之前**：

| 情况 | 判定 | 行为 |
|---|---|---|
| PATH 上找不到 `nvidia-smi` | `UNAVAILABLE` | 退出码 **8**，`status=VRAM_TELEMETRY_UNAVAILABLE`，`error` 写明"not found on PATH" |
| `nvidia-smi` 非零退出 | `UNAVAILABLE` | 退出码 8，`error` 写明 `exited <rc>: <stderr 前两行>`（例如 rc=9 "couldn't communicate with the NVIDIA driver"） |
| 输出无法解析 / `memory.free` 非非负整数 / 空 stdout | `UNAVAILABLE` | 退出码 8，`error` 写明具体字段 |
| 运行中逐步采样丢失 | `UNAVAILABLE` | 立刻中止，`status=VRAM_TELEMETRY_LOST`，退出码 8 |

诊断宽度是故意的：『不在 PATH』『退出码 9 连不上驱动』『输出不可解析』是三种不同的运维问题，
不能塌缩成同一句"遥测不可用"。`state["vram"]["telemetry"]["diagnostics"]` 里保存
`binary / returncode / stderr_head / exception / command`，供事后定位。

**没有**任何"遥测未知但继续"的分支——`grep -n "UNKNOWN" ` 在 A0 里只剩 `vram_verdict` 的返回值定义，
调用方一律走中止路径。

### 2.9 程序化验收门：12 条，任一失败即非零退出（第二轮补强）

**问题**：第一版跑完就 `status="PASS"`，等于"跑完即通过"。这违反了 AGENTS.md 的
"never claim success without measured artifacts"。

**修复**：`acceptance_checks()` 只从已记录的证据里判 12 条，`finalize_with_gate()` 把结果写进
`state["acceptance"]` 并决定退出码。**证据缺失一律算失败**，所以被中止的运行无法被事后拼成 PASS。

| # | 判据 | 证据字段 |
|---|---|---|
| 1 | 全程无异常到达终点 | `run_completed is True` |
| 2 | 后端解析为 `physx_cuda` | `backend.resolved` |
| 3 | `gpu_sim_enabled is True` | `backend.gpu_sim_enabled` |
| 4 | horizon 为 50 | `backend.max_episode_steps` |
| 5 | 动作形状为 `[1, 8]` | `action.actual_shape` |
| 6 | 恰好记录 3 步 | `len(steps)` |
| 7 | 每步 `terminated == 0` | `steps[*].terminated` |
| 8 | 每步 `truncated == 0` | `steps[*].truncated` |
| 9 | `elapsed_steps == [1,2,3]` | `steps[*].elapsed_steps` |
| 10 | 设备级显存遥测齐全（前/后/每步） | `vram.device_before/after`、`steps[*].vram_free_mib` |
| 11 | 跑完剩余显存 ≥ 中止阈值 | `vram.device_after.memory.free` |
| 12 | 墙钟 < min(120 s 外部, 内部预算) | `elapsed_s` |

退出码规则：12 条全过 → 0；否则若非提前失败则 `status=GATE_FAILED` 且退出码 **9**；
若是提前失败（如后端不符）则保留其专属退出码（2/3/6/8），但仍写满 12 条判据。

## 3. Stage A0 规格（待批准执行）

| 项 | 值 |
|---|---|
| 任务 | `PickCube-v1`，`robot_uids="panda"` |
| 后端 | `sim_backend="physx_cuda"`（显式，回读校验） |
| 环境数 | **1** |
| 观测 | `obs_mode="state"`（42 维） |
| 控制 | `control_mode="pd_joint_delta_pos"`（动作 8 维） |
| 奖励 | `reward_mode="normalized_dense"` |
| 渲染 | `render_mode=None`（无 rgb_array、无视频） |
| 种子 | `reset(seed=0)` |
| 步数 | **恰好 3** 步，动作为全零、形状 `(1, 8)`、dtype float32、device=cuda |
| 时间 | 内部 100 s（monotonic + watchdog），外部 `timeout 120` |
| 禁止 | 无策略、无优化器、无梯度、无 checkpoint、无 wandb/tensorboard、无额外依赖、无云端 |
| 输出 | `results/m2_gpu_smoke_a0.json`（增量写盘）+ stdout |

**通过判据**：不再由人工看数字决定，改由 §2.9 的 12 条程序化验收门判定。
`results/m2_gpu_smoke_a0.json` 的 `acceptance.{ok,passed,total,checks,failed_ids}` 就是结论本身；
`status == "PASS"` 与退出码 0 必须同时成立。

**A0 明确不能证明的事**：不能证明吞吐、不能证明多环境显存、不能证明训练可跑、不能证明收敛。
它只回答"CUDA PhysX 在这台机器上能不能起来"。

### 3.1 拟执行命令（**尚未执行，等你点头**）

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction

# 前 30 秒只读体检（不建环境）：
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv

# 正式 A0：外部 120 s 硬上限 + 内部 100 s 预算
timeout --signal=TERM --kill-after=10s 120 \
  ./.venv-maniskill/bin/python scripts/m2_gpu_smoke_a0.py \
    --env-id PickCube-v1 --robot panda --seed 0 --steps 3 \
    --budget-seconds 100 --min-free-mib 3072 --abort-free-mib 1536 \
    --output results/m2_gpu_smoke_a0.json 2>&1 | tee results/raw/m2_gpu_smoke_a0.log
echo "exit=${PIPESTATUS[0]}"
```

`timeout` 的退出码 124 表示外部强杀；脚本自身退出码含义见脚本 docstring（0/2/3/4/5/6/7）。

---

## 4. 本轮实际跑过的静态检查

机器可读汇总：**`results/m2_static_checks.json`**（已入库）；原始 stdout：`results/raw/m2_static_checks.log`
（`results/raw/` 按仓库约定 gitignore，属本地原始日志，可随时用下面一行重生成）。

```bash
./.venv-maniskill/bin/python scripts/m2_static_audit.py --json results/m2_static_checks.json
```

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法编译 | `python -m py_compile scripts/m2_gpu_smoke_a0.py scripts/m2_gpu_sim_probe.py scripts/m2_static_audit.py` | 3/3 通过 |
| 策略审计 | `python scripts/m2_static_audit.py` | **27 passed, 0 failed** |
| 逻辑自测（A0） | `python scripts/m2_gpu_smoke_a0.py --self-test` | **16 passed, 0 failed** |
| 逻辑自测（A1） | `python scripts/m2_gpu_sim_probe.py --self-test` | **18 passed, 0 failed** |
| 守卫拒绝 | `--steps 4` / `--env-counts 16,4096` / `--env-counts 16,512` | 分别退出 5，均按预期拒绝 |

`m2_static_audit.py` 是 AST 级审计（不是 grep），逐文件断言：
不 import `torch.optim`/`tensorboard`/`wandb`/`tensordict`/`torchrl`；
不出现 `backward`/`zero_grad`/`Adam`/`SGD`/`SummaryWriter` 等符号；
字符串里没有 `pip install`/`conda install`/`curl`/`wget`/`cudagraphs`/`--track`；
每个 `gym.make` 都显式带 `sim_backend="physx_cuda"`；
动作必须经 `batched_action_shape` 构造；必须有 `Timer/Thread` 看门狗与 `monotonic` 预算；
A0 的 `--steps` 默认必须是 3、内部预算必须 < 120。

自测完全不创建 CUDA context、不实例化任何环境，只验证纯函数：动作形状、VRAM 判定、预算算术、
wrapper 链 horizon 解析、JSON 原子写、看门狗线程确实会触发。

---

## 5. 仍未解决 / 需要你注意的残余风险

1. **OOM 仍可能在建场景时发生**：`--min-free-mib` 只是起跑前的必要条件，不能预测 PhysX 场景的分配量。
   因此 A0 的通过/失败本身就是"这台机器能不能承受 1 个 CUDA 场景"的答案，不存在事前保证。
2. **agent 壳的 GPU 可见性会随会话变化**（M1→M2 已发生过一次）。每次运行必须把
   `/dev/nvidia*` 与 `nvidia-smi` 快照留在证据里，否则无法审计；A0 已经自动记录后者。
3. **`timeout` 对 CUDA 的强杀可能留下脏的 GPU 上下文**：`--kill-after=10s` 是必要的，但强杀后
   建议再看一次 `nvidia-smi` 确认显存已释放（A0 的 `vram.device_after` 只能记录正常退出路径）。
4. **A1 的 SPS 不可外推**：`render_mode=None` 与官方 `rgb_array` 不同，且无策略、无梯度。
   `scope_warning` 字段已写进 JSON，报告里也必须照样写。
5. **PPO 依赖仍未安装**：`tensorboard`/`tensordict`/`wandb` 全部 MISSING，且**与 A0 无关**。
   独立复核第 6 条已明确：装 `tensorboard` 是 Stage B 的前置，不是 A0 的前置。
6. **Tyro 布尔参数语法未验证**（`--no-capture-video` 之类）：等真要跑 `ppo.py` 时先看 `--help`，
   不能用猜的命令行宣布"跑通了"。

---

## 6. 本轮明确没有做的事（第二轮之后的实际情况见 §7）

未执行 A1；未创建 CUDA context；未实例化任何 ManiSkill 环境；未安装任何包；
未下载 checkpoint；未训练；未改上游源码；未改驱动/CUDA/ROS2/Isaac Sim；未租云 GPU。
A0 已按批准命令执行**一次**，结果见 §7。

---

## 7. A0 执行记录（2026-10-21，唯一一次）

**命令**（与批准版逐字一致）：

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv
timeout --signal=TERM --kill-after=10s 120 \
  ./.venv-maniskill/bin/python scripts/m2_gpu_smoke_a0.py \
    --env-id PickCube-v1 --robot panda --seed 0 --steps 3 \
    --budget-seconds 100 --min-free-mib 3072 --abort-free-mib 1536 \
    --output results/m2_gpu_smoke_a0.json 2>&1 | tee results/raw/m2_gpu_smoke_a0.log
```

**结果：未进入仿真。退出码 8，`status=VRAM_TELEMETRY_UNAVAILABLE`，1.2 秒结束，`steps` 为空。**

```
[vram] telemetry verdict=UNAVAILABLE :: nvidia-smi missing, non-zero exit, unparsable output, or timeout
FATAL: device-level VRAM telemetry is mandatory for Stage A0 and could not be read before env creation:
       ...; aborting without building any scene
acceptance: 1/12 passed, failed_ids=[1,2,3,4,5,6,7,8,9,10,11]
status=VRAM_TELEMETRY_UNAVAILABLE elapsed=1.204s -> results/m2_gpu_smoke_a0.json
exit=8
```

（第一版的 reason 文案还不够精确，已在同轮补成"`exited 9: couldn't communicate with the NVIDIA driver`"，
但**没有**为了刷新文案而重跑 A0——批准的是"恰好一次"，证据文件保持原始状态；
精确诊断另存 `results/raw/m2_a0_gpu_unavailable_diag.log`。）

### 7.1 根因：本轮 session 又看不到 GPU 设备节点

| 检查（只读） | 输出 |
|---|---|
| `ls -l /dev/nvidia*` | **No such file or directory** |
| `which nvidia-smi` | `/usr/bin/nvidia-smi`（二进制在） |
| `nvidia-smi --query-gpu=memory.free ...` | `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.` **rc=9** |
| `/proc/driver/nvidia/version` | `NVRM 580.178.04`（**驱动模块已加载**） |
| `lspci` | `GA104 [GeForce RTX 3070 Lite Hash Rate]`（**卡在**） |
| venv 内 `torch.cuda.is_available()` | **False**，`device_count=0`（`Can't initialize NVML`） |

结论：不是驱动故障、不是卡不在、不是 torch 装错，而是**本 session 没有暴露 `/dev/nvidia*` 设备节点**——
与 2026-10-20 M1 那次同一类问题（`docs/M2_INDEPENDENT_REVIEW.md` §5 与 `PROGRESS.md` 未知项 #1 已预警"会话级、会回退"）。
10-21 上午的阳性读数（`nvidia-smi` 正常、`torch.cuda=True`）说明可见性在两个方向都会变。

**没做的事**：没有改驱动、没有改设备权限、没有加 `--privileged`、没有换容器、没有重试 A0。

### 7.2 这次失败证明了什么、没证明什么

- **证明了**：强制遥测闸有效——在**没有任何 CUDA 上下文、没有建任何场景**的情况下干净退出，退出码 8，
  `steps` 为空，12 条验收门中 11 条判失败，JSON 里留着 reason 与诊断字段。这本身是 §2.8 的正向证据。
- **没有证明**：`physx_cuda` 能否建场景、能否 reset、能否走 3 步、显存占用多少。
  **A0 依然是 NOT PASSED**，M2 的 `PHYSX_CUDA_UNVERIFIED` 状态不变。

### 7.3 恢复路径（需要研究者操作，不需要改代码）

1. 在你自己的桌面终端确认 `nvidia-smi` 正常（那里一直是可用的）；
2. 重开一次宿主应用/会话，让新进程重新拿到 `/dev/nvidia*`（10-21 09:57 那次就是这么恢复的）；
3. 恢复后用 `ls -l /dev/nvidia*` + `nvidia-smi` 复核，再**重新批准一次** A0（因为"恰好一次"已经用掉）。

脚本无需改动：同一份代码在 GPU 可见时按 12 条判据给出 PASS/FAIL，不可见时以退出码 8 拒绝。
