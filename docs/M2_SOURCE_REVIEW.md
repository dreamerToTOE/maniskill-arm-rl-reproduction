# M2 Source Review — 官方 PPO 基线源码核对报告

> 日期 2026-10-21 · 状态 **SOURCE_VERIFIED / LEARNING_PACK_CORRECTED / GPU_VISIBLE_IN_AGENT_SHELL / TRAINING_NOT_RUN**
> 上游：ManiSkill `v3.0.1` = `a4a4f9272ad64b1564035874b605ceb687b63ed8`
> 本文只做静态源码核对与只读环境诊断；**没有**运行训练、GPU 仿真或基准。
> 配套：学习包 `docs/M2_LEARNING_PACK.md`（已按本文修正）；训练报批 `docs/M2_GPU_PILOT_PROPOSAL.md`。

---

## 1. 结论摘要

| # | 检查项 | 结论 |
|---|---|---|
| 1 | 学习包给出的四段行号映射 | **全部落在正确位置**（逐条核对见 §3） |
| 2 | `ppo_fast.py` 依赖 `torchrl` | **❌ 不准确，已修正**：该文件 0 处引用 `torchrl`，实际需要 `tensordict` |
| 3 | 依赖是否齐全 | **❌ 两个脚本当前都无法 import**：都缺 `tensorboard`；`ppo_fast.py` 还缺 `tensordict`、`wandb` |
| 4 | GAE 的终止/截断边界语义 | **包内原为"待查证"，现已查清**并写明机制与作者注释 |
| 5 | 官方 PickCube 配置 | 与 `baselines.sh` 逐参数一致（4096/4/8/32/50M/16/`--cudagraphs`/3 seeds） |
| 6 | 8GB 的风险点 | 不在 PPO 网络（很小），而在 **4096 个并行 PhysX 环境 + `render_mode="rgb_array"`** |
| 7 | GPU 可见性 | **已解释并已改变**：agent 壳内现在 `nvidia-smi` 正常、`torch.cuda.is_available()=True` |

---

## 2. 源码溯源

三个文件通过 **GitHub Contents API** 只读取回（`api.github.com` 可达；`raw.githubusercontent.com` 在本网络被重置）。文件**未**提交进本仓库（上游 Apache-2.0，不 vendor），只记录 blob SHA 与获取方式：

| 文件 | 行数 | 字节 | blob SHA |
|---|---:|---:|---|
| `examples/baselines/ppo/ppo.py` | 471 | 22,982 | `66802de0653061da550d6600d6346be374dbf974` |
| `examples/baselines/ppo/ppo_fast.py` | 523 | 22,681 | `969e56401768e3c2fe7a25a19efb5cfae4ae4aa6` |
| `examples/baselines/ppo/baselines.sh` | 226 | 7,727 | `1af67fe34378374f8a842b1296d568455a178962` |

机器可读记录：`results/m2_source_provenance.json`（含 API URL 与复现命令）。

---

## 3. 学习包核对：逐条判定

判定用词：**CONFIRMED**=行号/表述正确；**CORRECTED**=原文有误已改；**UPGRADED**=原为存疑，现已查清。

| 学习包原文 | 实际源码 | 判定 |
|---|---|---|
| `ppo.py:25–112` Args | `@dataclass Args` 在 24，字段至 113（`num_iterations`） | CONFIRMED（边界差 1 行，不影响阅读） |
| `ppo.py:121–161` Agent | `class Agent` 121–161（`actor_mean` 133–141，`actor_logstd` 142，`get_action_and_value` 154–161） | CONFIRMED |
| `ppo.py:192–213` env setup | 192 device；195 `env_kwargs`（`obs_mode="state"`, `render_mode="rgb_array"`, `sim_backend="physx_cuda"`）；198–199 `gym.make`；212–213 `ManiSkillVectorEnv` | CONFIRMED |
| `ppo.py:247–265` storage | 247–253 六个 buffer；265 `action_space_low/high`；266–267 `clip_action` | CONFIRMED |
| `ppo.py:308–332` rollout | 308 `for step`；315 `get_action_and_value`；321 `envs.step(clip_action(action))`；322 `next_done=term|trunc`；323 reward；325–331 `final_observation→final_values` | CONFIRMED |
| `ppo.py:333–374` GAE | 334–335 `next_value`；338 反向循环；345 `real_next_values`；372–373 delta/GAE；374 `returns=adv+values` | CONFIRMED |
| `ppo.py:384–438` update | 396–397 `logratio/ratio`；401–403 KL/clipfrac；413–415 `pg_loss`；419–430 `v_loss`；432–438 loss/backward/`clip_grad_norm_`/`step` | CONFIRMED |
| `ppo.py:449–462` 指标 | 449–456 loss/KL/clipfrac/explained_var；457–462 SPS 与耗时 | CONFIRMED |
| "`ppo_fast.py` 使用额外 `torchrl`、`tensordict`" | `torchrl` 出现 **0 次**；实际 import：`tensordict`(20)、`wandb`(27)、`tqdm`(24)、`tensorboard`(26) | **CORRECTED** |
| "默认 `ent_coef=0.0` 的前提下不会产生该项梯度" | 90 行 `ent_coef: float = 0.0`；433 行 `loss = pg_loss - ent_coef*entropy + vf_coef*v_loss` | CONFIRMED |
| "`ppo.py` 采样无界高斯，logprob 用采样动作，env 收 clip 后动作" | 315 采样→318 存 logprob→321 `clip_action` 后 step | CONFIRMED |
| "`next_done = terminations \| truncations`，经 `final_observation` 取 `final_values`" | 322 / 331 | CONFIRMED |
| "本代码 GAE 的 `next_not_done` 与 Bootstrap 边界须对照推导，不得先断言对错" | 机制已完全查清，见 §5；且作者在 373 行留有注释自认应区分 termination | **UPGRADED** |

---

## 4. `ppo.py` 精读地图（按数据流，非按行号顺序）

```
Args(24–113)  ← 所有超参都在 dataclass 默认值里，命令行可覆盖
  ↓
Agent(121–161)
   critic:   101→256→256→256→1, Tanh, layer_init(√2)
   actor_mean: 101→256→256→256→8, 末层 std=0.01√2
   actor_logstd: 可学习参数，初值 -0.5（即 std≈0.607），expand 到 8 维
  ↓
env(192–214)  physx_cuda + num_envs 并行 + ManiSkillVectorEnv(auto_reset, ignore_terminations)
  ↓
buffer(247–253)  obs/actions/logprobs/rewards/dones/values，形状 [num_steps, num_envs, ...]
  ↓
rollout(272–332)  存 s_t,a_t,logp_old,V(s_t)；env.step(clip(a_t))；记录 final_values
  ↓
GAE(333–374)  δ_t, Â_t, R_t=Â_t+V_t（见 §5）
  ↓
update(384–443)  K epochs × 32 minibatch：ratio → clip → pg/v/entropy → Adam
  ↓
log/eval(276–300, 445–462)  eval 用 deterministic 均值动作；每 25 iter 存 ckpt
```

**Actor/Critic 的确切身份**（回答"更新了哪些网络"）：
- `actor_mean` 与 `actor_logstd` 同属 `agent.parameters()`，由**同一个 Adam**（245 行）更新；熵项经 `actor_logstd` 影响梯度。
- `critic` 只由 `v_loss` 更新，`vf_coef=0.5` 加权。
- 观测是 42 维 state（M1 已实测），动作 8 维，**没有任何视觉编码器**。

---

## 5. GAE 与 Bootstrap：把"存疑"变成确定结论

### 5.1 向量环境的自动重置机制（已读源码确认）

`mani_skill/vector/wrappers/gymnasium.py::ManiSkillVectorEnv.step`：

```python
dones = terminations | truncations          # 注意：两者并列
if dones.any() and self.auto_reset:
    final_obs = torch_clone_dict(obs)       # ← 先克隆"重置前"的 o_{t+1}
    obs, infos = self.reset(options=dict(env_idx=dones))
    infos["final_observation"] = final_obs  # ← 真正的下一观测保存在这里
    infos["final_info"] = final_info
    infos["_final_info"] = dones
```

所以：**`next_obs` 是重置后的新 episode 起点，而 `final_observation` 才是刚结束那个 episode 的 o_{t+1}。**
这正是 `ppo.py:331` 必须单独存 `final_values` 的原因——否则在用 `next_obs` 算 bootstrap 时会把"新 episode 的初值"当成"旧 episode 的末值"。

`partial_reset` 的实际作用（`ignore_terminations = not partial_reset`，212 行）：

| 参数 | `ignore_terminations` | 行为 |
|---|---|---|
| `partial_reset=True`（默认，官方训练用） | False | **成功即结束**并重置；到 50 步截断也重置 |
| `partial_reset=False`（官方评估用 `eval_partial_reset=False`） | True | 成功**不**结束，只在 50 步截断时重置（无限时域风格） |

### 5.2 于是 GAE 分支可以精确描述

```python
next_not_done  = 1 - done                      # done = term|trunc
real_next_values = next_not_done * V(next_obs) + final_values[t]
delta_t = r_t + γ·real_next_values − V(s_t)
Â_t = delta_t + γλ·next_not_done·Â_{t+1}       # 373 行
```

- **截断时**：`next_not_done=0` → 用 `final_values[t] = V(o_{t+1})`，即**用真实的后续观测做 bootstrap**。这对截断是**正确**的（截断不是真终止，未来回报不为 0）。
- **成功终止时**：同样走到 `final_values[t]`，于是目标里多了一项 `γ·V(成功状态)`。严格理论下终止状态应为 0（只保留 `r_t`）。**这是一处真实的小偏差**，方向是把价值目标抬高。
- **λ 递归在两种 done 处都被切断**（`next_not_done` 同时乘进 `Â_{t+1}`）：截断处切断后向传播会略微短视，但方向保守。
- 作者自己在 **373 行**留了注释：`# Here actually we should use next_not_terminated, but we don't have lastgaelam if terminated`。

→ 结论：**不能说这段实现"错误"，但它确实把 termination 和 truncation 当同一种边界处理**。学习者必须能指出这个偏差及其方向，而不是背"GAE 公式"。这也是 M2 理解题 Q3 的落点。

---

## 6. `ppo.py`（教学版）vs `ppo_fast.py`（官方性能版）

| 维度 | `ppo.py` | `ppo_fast.py` |
|---|---|---|
| 定位 | 可读参考实现 | 官方 baseline（`baselines.sh` 调用它） |
| 网络 | 同构 MLP（256×3） | 同构，但用 `TensorDictModule` 包装（300 行） |
| 经验容器 | 六个独立 `torch.zeros` buffer（247–253） | `TensorDict`（235–252 `rollout()` 内堆叠） |
| GAE | 主循环内联（333–374） | 独立函数 `gae()`（189–215），可被 `CudaGraphModule` 捕获 |
| 更新 | 双层循环内联（389–441） | 独立函数 `update()`（255–299）+ `TensorDictModule`，按 minibatch 外部调用（484–497） |
| 加速 | 无 | `--compile`（torch.compile）、`--cudagraphs`（`CudaGraphModule`，403–420） |
| 推理参数 | 与训练同一模块 | 额外 `agent_inference` + 分离参数（`from_module(...).data`，396–400），避免 CUDA graph 与 autograd 冲突 |
| 评估动作 | `get_action(deterministic=True)`（283） | `agent.actor_mean(eval_obs)`（440）——**等价**，都是确定性均值 |
| 依赖 | gymnasium, torch, tyro, tqdm, **tensorboard** | 以上 + **tensordict, wandb** |
| 默认规模 | 512 env × 50 step，10M steps，4 epochs | 512 env × 50 step，10M steps，4 epochs（**与官方命令不同**） |
| 官方 PickCube 配置 | — | 4096 env × **4** step，**8** epochs，32 minibatch，**50M** steps，16 eval env，`--cudagraphs`，`--track` |
| `torchrl` | 不用 | **不用**（学习包原写"需要 torchrl"，已修正） |

### 官方配置的算术（可核对）

```
batch_size      = 4096 × 4   = 16384
minibatch_size  = 16384 / 32 = 512
num_iterations  = 50,000,000 // 16384 = 3051
实际采集步数    = 3051 × 16384 = 49,987,584   （略少于 50M，因为整除截断）
seeds           = 9351, 4796, 1788 → 三 seed 合计 149,962,752 env steps
```

注意 `--track` 会初始化 wandb，且 `ppo_fast.py` 在**模块顶层** `import wandb`（27 行）——即使不加 `--track` 也必须安装，否则直接 ImportError。

---

## 7. 依赖缺口（实测，非推测）

在当前 `.venv-maniskill` 内逐个 import 的结果：

| 包 | ppo.py 是否需要 | ppo_fast.py 是否需要 | 当前状态 |
|---|---|---|---|
| `tensorboard` | **是**（15 行 `from torch.utils.tensorboard import SummaryWriter`） | **是**（26 行） | **MISSING**（`ModuleNotFoundError: No module named 'tensorboard'`） |
| `tensordict` | 否 | **是**（20/28/29 行） | **MISSING** |
| `wandb` | 仅 `--track` 时延后导入（221） | **顶层无条件**（27 行） | **MISSING** |
| `tqdm` | 否 | 是（24 行） | 已装 4.70.1 |
| `tyro` | 是 | 是 | 已装 1.0.16 |

**因此：两个脚本现在都无法启动**，报错都在 import 阶段，与 GPU、PhysX、显存无关。
安装这三个包属于"新增可选依赖"，**需要单独批准**；本文不安装。

预估磁盘增量（未安装，仅列名，不当作实测）：`tensorboard` 会拉入 protobuf/grpcio/markdown 等，`wandb` 拉入 requests/psutil 等，量级约 200–400 MB，需在批准时确认。

---

## 8. RTX 3070 8GB 的风险分析

关键判断：**PPO 的网络和 batch 不是瓶颈，并行 PhysX 环境才是。**

| 项目 | 量级 | 说明 |
|---|---|---|
| Actor+Critic 参数 | ~0.6 M（4×256×256 MLP ×2） | 可忽略 |
| batch 张量 | 16384 × (42 obs + 8 act + 3 scalar) ≈ 数 MB | 可忽略 |
| 优化器状态 | Adam 两份矩 ≈ 数 MB | 可忽略 |
| **4096 并行环境** | **未知，主要风险** | 每个 env 的 articulation/contact buffer 在 GPU 上；ManiSkill 公开说法是 state 任务可跑数千 env，但**不是 3070 + 8GB 的实测** |
| `render_mode="rgb_array"`（195 行） | 额外渲染目标 | 官方即使在 state 观测下也设了 rgb_array；`RecordEpisode` 会写 eval 视频。8GB 上这是可省的开销 |

可能的降级路径（**每条都必须标注为 ADAPTATION，不能称作官方复现**）：
1. 降 `num_envs`（如 4096→1024/512）并**按比例调 `total_timesteps`**？——不行，步数是采样总量，不应随 env 数缩放；应保持 50M，只是更慢。
2. 关 `--cudagraphs` 或 `--compile`（省显存/避免 graph 捕获失败），速度下降。
3. `--capture_video=False` 省渲染开销。
4. 关 `--track`（但仍需安装 wandb）。

**不能做的事**：把 4096 改小后仍称"官方基线复现"。必须记录 `ADAPTATION`。

---

## 9. GPU 可见性：问题已定位且现状已改变

### 9.1 只读诊断证据（2026-10-21 10:27 UTC+9）

```
/dev/nvidia0, nvidiactl, nvidia-modeset, nvidia-uvm, nvidia-uvm-tools   全部存在
nvidia-smi --query-gpu=... → RTX 3070, 8192 MiB total, 6775 MiB free, driver 580.178.04
nvidia-smi 详情            → 1043 MiB / 8192 MiB 已用, GPU-Util 10%（桌面占用）
torch.cuda.is_available()  → True, device_count=1, capability (8,6) = sm_86
```

原始输出：`results/raw/m2_gpu_readonly_diag.log`。

### 9.2 与 M1 那次失败的对比

| 时间 | agent 壳内 `/dev/nvidia*` | `nvidia-smi` | `torch.cuda` |
|---|---|---|---|
| 2026-10-20 10:32 | **不存在** | rc=9 | False |
| 2026-10-21 10:27 | **存在**（均为 `crw-rw-rw-`，时间戳 10-21 09:57） | 正常 | **True** |

设备节点时间戳 09:57 与 cgroup（`app-gnome-workbuddy-…scope`）说明：**宿主应用在 09:57 重启后，新的会话进程拿到了 GPU 设备节点**。此前是会话级设备暴露缺失，不是驱动故障、不是卡不在、也不是 torch 装错。

### 9.3 结论

- 之前 M1 报告里 "CUDA_UNVERIFIED" 的**原因已解释**：agent 壳当时处于无设备节点的受限命名空间；**不需要**动驱动或设备权限（我们也没动）。
- 研究者在**自己的终端**早已是 `CUDA: True`，与本结论一致，不矛盾。
- ⚠️ 但 **PhysX/CUDA 仿真本身仍未验证**：`torch.cuda.is_available()=True` 只证明 PyTorch 能用 GPU，不证明 `sim_backend="physx_cuda"` 能建 4096 个 env、能分配显存、能跑出速度。这正是下一步要报批的 smoke。
- 另外提醒：agent 壳的 GPU 可见性是**会话相关**的，可能再次变化。任何 GPU 运行的证据必须记录当时的 `/dev/nvidia*` 与 `nvidia-smi` 输出，否则无法审计。

---

## 10. 与学习包的关系（已同步修改）

`docs/M2_LEARNING_PACK.md` 本轮改动：
1. §3 风险 #4：删除"需要 `torchrl`"，改为实测依赖 `tensordict`/`wandb`/`tensorboard`，并注明当前 **MISSING**。
2. §3 风险 #3：由"须对照推导"升级为**已查清的机制描述**（含 `final_observation` 原理、termination 与 truncation 同处理的偏差方向、作者注释位置）。
3. §2.3 补一句：`loss = pg_loss − ent_coef·entropy + vf_coef·v_loss`（433 行），并指出 actor_mean 与 actor_logstd 共用一个 Adam。
4. 新增 §3.1 官方 vs 教学配置的算术表（16384 / 512 / 3051 / 49,987,584）。

**未填任何"研究者答案"**；§4 的三道题保持原样，等研究者本人作答。

---

## 11. 本轮明确未做的事

未安装 `tensorboard`/`tensordict`/`wandb`；未运行 `ppo.py`/`ppo_fast.py`；未启动任何 `physx_cuda` 仿真或吞吐基准；未下载 checkpoint；未改驱动/CUDA/ROS2/Isaac Sim；未触碰双 FR3 主工程；未租云 GPU。
下一步的资源上限方案见 `docs/M2_GPU_PILOT_PROPOSAL.md`（待研究者批准）。
