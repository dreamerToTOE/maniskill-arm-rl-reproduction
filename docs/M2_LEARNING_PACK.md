# M2 Learning Pack — PickCube 官方 PPO：原理一页纸 + 源码导读

> **先读完第 1–3 节，再回答第 4 节。** 与 M1 一样，Codex 不应先出十几个小问才提供教学材料。
> 来源：ManiSkill v3.0.1, source SHA `a4a4f9272ad64b1564035874b605ceb687b63ed8`; 主源码 `examples/baselines/ppo/ppo.py`；官方基准配置参照 `examples/baselines/ppo/ppo_fast.py` + `baselines.sh`。**当前未运行 PPO 训练**。

## 1. 先建立整体视角（1分钟）

M1 解决环境交互：`s_t → a_t → controller → PhysX → (s_{t+1},r_t,terminated,truncated)`。

M2 研究如何更新策略：
```
Actor πθ(a|s) + Critic Vφ(s)
    ↓ collect fresh on-policy rollouts
Rollout {obs, sampled_action, old_logprob, reward, done, value}
    ↓ compute temporal-difference residual, then GAE
Advantage Â_t and value target R_t
    ↓ repeated PPO minibatch updates
Actor: clipped surrogate objective; Critic: return regression
    ↓ log/evaluate, then collect new rollouts
```

**Actor：给出动作分布**。本源代码采用高斯 `Normal(mean(s),std)`，动作向量与 M1 的 `pd_joint_delta_pos` 一样是 8 维（7 臂+1 夹爪）。Actor 不直接预测方块能否成功，也不是直接输出力矩。

**Critic：输出一个标量 `V(s)`**，预测从该状态继续执行策略所能获得的折扣回报；这不是 Actor 的动作分布，也不是 SAC 的 `Q(s,a)`。

## 2. 原理一页纸：四个最重要的公式

### 2.1 为什么要算 Advantage？

`V(s_t)` 估计“处在状态 s_t 的平均未来收益”；动作具体好坏要比较**执行这次动作之后实际获得的信息与此前的预期**。一步 TD 残差：

```
δ_t = r_t + γ * V(s_{t+1}) − V(s_t)
```

这是**非终止过渡**的简化形式；episode 结束/超时需按脚本的 done、final_observation 和 Bootstrap 分支处理，不能对全部边界都套用同一个 next V。

`GAE(λ)` 用多个后续 TD 残差降低纯一步估计的噪声：

```
Â_t = δ_t + γ λ (1−done_boundary) Â_(t+1)
R_t(target) = Â_t + V(s_t)
```

`γ`（discount）重视未来多久；`λ` 平衡偏差与方差。Critic 拟合 `R_t`，Actor 依赖 `Â_t`。`done_boundary` 与 value Bootstrap 要结合具体 wrapper 的 terminated/truncated 处理，不能用一句“所有 done 都不 Bootstrap”替代源代码。

### 2.2 为什么 PPO 引入新旧策略概率比？

```
ρ_t(θ) = exp(log π_new(a_t|s_t) − log π_old(a_t|s_t))
```

同一条**旧策略采集**的数据，重算新策略对当时动作的概率；当 `ρ` 太大/太小，Clipping 防止仅靠这一条样本驱动策略大幅改变：

```
L_clip(θ) = E[min(ρ_t*Â_t, clip(ρ_t,1−ε,1+ε)*Â_t)]
```

数学上 Actor 想**最大化**这项；代码最小化其负号等价形式（`max(-ρA, -clip(ρ)A)`）。对于正负优势，裁剪方向不同，不能把 `ρ` 简单截断后当作实际策略更新或权重限制。

### 2.3 Actor/Critic 最后各更新什么？

```
L_total = L_policy + c_v * L_value − c_e * Entropy
L_value = 0.5 * mean((V_new(s_t) − R_t)^2)
```

这是 `ppo.py` 的 Loss 形状（**433 行原文**：`loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef`）；
`actor_mean` 与可学习的 `actor_logstd` 同属 `agent.parameters()`，由**同一个 Adam**（245 行）更新；
`critic` 只由 `v_loss` 更新；梯度经 `clip_grad_norm_(max_grad_norm=0.5)`（437 行）裁剪。
`entropy` 项鼓励探索（默认 `ent_coef=0.0` 时该项对梯度贡献为 0）。

### 2.4 一个最小数值例子（纯教学，不是运行结果）

若非终止 `r=0.1`, `γ=0.8`, `V(s)=0.2`, `V(s')=0.3`，
`δ = 0.1 + 0.8×0.3 − 0.2 = 0.14`。
若 `Â=2`, `ρ=1.5`, `ε=0.2`，两个 surrogate 分别为 3 与 2.4，PPO 取 2.4，避免对该正优势样本过度加权。这只是公式说明，不是实测 RL 指标。

## 3. 代码导读：按四个模块阅读，不要逐行背 500 行

**以下行号来自 v3.0.1 锁定的 `ppo.py`，不是未来自动更新的 main。**

| 阅读顺序 | 文件及行号 | 做什么 | 读完要会讲什么 |
|---|---|---|---|
| 1 | `ppo.py:25–112` Args, `121–161` Agent, `192–213` env setup | 参数、Actor Gaussian、Critic V、`physx_cuda`、VectorEnv | 状态如何变动作，为什么用 `V(s)` 而不是 `Q(s,a)` |
| 2 | `ppo.py:247–265,308–332` | 创建 rollout buffer，采样动作并存 `logprob_old`、reward、value、done | 为什么每轮 PPO 都要先采集**当前策略的新数据** |
| 3 | `ppo.py:333–374` | backward GAE、Bootstrap，`returns=advantages+values` | 终止和第50步截断为什么不能混淆 |
| 4 | `ppo.py:384–438,449–462` | `ratio`、clip、value/entropy loss、Adam update、SPS/KL/clipfrac | 更新了哪些网络，什么时候 PPO 更新可能过大/不稳定 |

源码浏览：
- https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/ppo.py
- https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/ppo_fast.py

### 3.1 官方配置 vs 教学配置的算术（2026-10-21 核对 `baselines.sh`）

| 量 | `ppo.py` 默认 | 官方 PickCube（`baselines.sh` → `ppo_fast.py`） |
|---|---|---|
| `num_envs` | 512 | **4096** |
| `num_steps` | 50 | **4** |
| batch = envs×steps | 25,600 | **16,384** |
| `num_minibatches` → minibatch | 32 → 800 | 32 → **512** |
| `update_epochs` | 4 | **8** |
| `total_timesteps` | 10,000,000 | **50,000,000** |
| iterations = total // batch | 390 | **3051** |
| 实际采集步数 | 9,984,000 | **49,987,584**（整除截断，略少于 50M） |
| 加速 | — | `--cudagraphs`（+ 可选 `--compile`） |
| eval envs / seeds | 8 / 1 | **16 / 9351, 4796, 1788** |

→ 「理解 `ppo.py`」不等于「复现官方基线」。另注意 **`num_steps=4`** 意味着每轮只采 4 步、
一个 episode（50 步）要跨 13 个左右 iteration 才走完，**不是**"每 episode 只能走 4 步"。

### 3.2 跑起来还缺什么（实测，2026-10-21）

| 包 | `ppo.py` | `ppo_fast.py` | 当前 |
|---|---|---|---|
| `tensorboard` | 必须（15 行硬 import） | 必须（26 行） | **缺失** |
| `tensordict` | 不需要 | 必须（20/28/29 行） | **缺失** |
| `wandb` | 仅 `--track` 时才导入 | **顶层无条件**（27 行） | **缺失** |
| `tqdm` / `tyro` | 否 / 是 | 是 / 是 | 已装 |

安装这些属于新增依赖，**需单独批准**；见 `docs/M2_GPU_PILOT_PROPOSAL.md`。

### 来源上的关键区别与风险

1. **教学 `ppo.py` ≠ 官方大规模 PickCube 性能基准本身**。官方 `baselines.sh` 选 `ppo_fast.py`、4096 env、4 rollout steps、8 update epochs、32 minibatches、50M steps/seed、cudagraphs、三 seeds。取教育用途的小配置来运行时要标记适配实验。
2. `ppo.py` 采样无界高斯动作，用采样动作算 `logprob`，但运行 `envs.step(clip_action(action))`。**Policy log probability 和实际发给控制器的动作可能不同**；未来审查饱和比率（动作越界比例）和偏差，而不是假装完全同一分布。
3. `next_done = terminations | truncations`（322 行），并通过 `final_observation` 获取额外 `final_values`（331 行）。
   **机制已查清（2026-10-21 源码核对，详见 `docs/M2_SOURCE_REVIEW.md` §5）**：
   向量环境在 `dones` 处先克隆"重置前"的观测存进 `infos["final_observation"]`，再重置；
   所以 `next_obs` 是新 episode 的起点，`final_observation` 才是旧 episode 真正的 `o_{t+1}`。
   → 截断时用它做 Bootstrap 是**正确**的；成功终止时也用它，等于在终止状态上多加了一项
   `γ·V(s_T)`，而严格理论下真终止状态应当不再 Bootstrap（取 0）。**这是潜在的价值目标偏差，但偏差方向取决于当前 Critic 对该终止状态预测的正负，不能不经实测断言一定抬高；偏差大小也未量化**。
   λ 递归在两种 done 处都被切断。作者在 373 行自注：
   `# Here actually we should use next_not_terminated, but we don't have lastgaelam if terminated`。
   因此**不能说这段实现"错"，但它确实把 termination 与 truncation 当同一种边界处理**。
4. `ppo_fast.py` 的额外依赖**实测为 `tensordict`、`wandb`、`tqdm`、`tensorboard`**；
   **`torchrl` 在该文件中出现 0 次**（早期文档写"需要 torchrl"是错的，已修正）。
   注意 `wandb` 在 27 行**顶层无条件 import**，即使不加 `--track` 也必须安装。
   当前 venv 三者全缺，**两个脚本都会在 import 阶段失败**（与 GPU 无关）；安装需单独批准。
5. 训练 `SPS` 是包含 rollout 和梯度更新的端到端速度；`rollout_fps` 只是采样速度。CPU 三步零动作 Smoke 不能用于估算 RTX 3070 训练成本。
6. 官方 PPO 默认 `control_mode=pd_joint_delta_pos`，官方 SAC PickCube 示例 `pd_ee_delta_pos`，未匹配控制器的成效差异不能归因于 PPO vs SAC。

## 4. 研究者理解验收（读完上面内容才答）

1. 为什么 `ppo.py` 中 Actor 需要 `logprob_old`，Critic 输出 `V(s)`，而不是把 reward 直接当作 Advantage？
2. `num_envs=4096, num_steps=4` 时一次采样批是多少？为何“rollout 4 步”**不等于**“每个 Episode 只能走 4 步”？
3. 为什么第50步的 `truncated=True` 不能不加区别当成真实失败/终止？在 GAE 和训练评估里分别会引发什么问题？

答题记录应包含原始答案与评阅结论。不要让 Codex 给出标准答案冒充研究者理解。此阶段是 M2 概念/source 学习，未被标记 M2 PASSED。

## 5. 什么才是下一步允许的验证？

优先在研究者本地的**正常 Ubuntu 终端**运行 `torch.cuda.is_available()` 和 `nvidia-smi` 的只读检查，结果按真实设备节点、解释器路径记录。Codex 代理壳曾报告 CUDA False，这不能证明 RTX 3070 在用户桌面会话无法用。

之后应独立提交**具体 GPU smoke 与短 PPO pilot 的资源上限申请**：环境并行数尝试范围、种子、最大 steps、wall-clock、VRAM 停止条件、是否产生任何 checkpoint、磁盘上限。未经审批，不运行 `ppo.py` 或 `ppo_fast.py` 的训练入口，也不把官方训练步数直接缩小后称为“完整复现”。
