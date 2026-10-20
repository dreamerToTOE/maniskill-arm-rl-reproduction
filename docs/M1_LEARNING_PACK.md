# M1 学习包 / Learning Pack — 观测、动作、奖励、生命周期

> 用途：M1 的**学习交付物**。按 `docs/HUMAN_CODEX_WORKFLOW.md` 第三节，每个关卡必须先给
> 「原理一页纸 + 代码导读 + 需要研究者回答的问题」，研究者理解后才进入下一关。
> 所有数值来自 2026-10-20 实测（`results/m1_pickcube_state_dict_zero_seed0.json`、`results/raw/m1_interface_probe.json`）；
> 所有公式标注了对应源码位置（已安装 wheel `mani_skill==3.0.1`）。
> 读完本文件再回答 §5 的三道题；看完答案前不建议去看 §5 末尾的"常见误解"。

---

> **2026-10-20 源码复核勘误**：原版部分解释误写“归一化动作 0.1 = 0.1 rad”及“所有未成功状态奖励≤0.6”；已修正。保留历史对话/日志供追溯。建议先读 [M1 重新整理版](M1_REORGANIZED_LESSON.md) 建立整体框架，再回本文件查源码和实测。

## 1. 本关只需掌握四件事

1. **动作不是结果，是"目标位置"**：策略输出 → 缩放 → 关节目标 → PD 伺服 → 物理。中间有 4 层。
2. **观测是仿真给的"上帝视角"**：42 维里有一部分真机上拿不到，这决定了迁移难度。
3. **奖励是分段的 + 有门控**：四项按条件打开，成功时直接覆盖成常数。
4. **episode 有两条独立出口**：`terminated`（成功）与 `truncated`（超时），来源完全不同。

---

## 2. 原理一页纸

### 2.1 动作链路：策略输出 → 关节目标 → 力矩 → 位移

策略输出 `a ∈ [-1, 1]^8`。**第 1 步：拆包**。`CombinedController.action_mapping = {"arm": [0,7], "gripper": [7,8]}`
→ 臂用 `a[0:7]`，夹爪用 `a[7]`。

**第 2 步：归一化反缩放**（`agents/controllers/base_controller.py:125 _preprocess_action` →
`_clip_and_scale_action`）：

```
先 clip 到 [-1, 1]，再线性映射：  â = lower + (a + 1) / 2 · (upper − lower)
```

Panda 的参数（实测读出的 `config`）：

| 子控制器 | lower | upper | use_delta | 含义 |
|---|---|---|---|---|
| `arm`（7 维） | −0.1 | +0.1 | **True** | 关节角**增量**，单位 rad |
| `gripper`（1 维） | −0.01 | +0.04 | **False** | 手指关节**绝对目标**，单位 m |

所以臂：`a_i = 1 → â_i = +0.1 rad`；夹爪：`a = −1 → −0.01 m`（闭合）、`a = +1 → +0.04 m`（张开）。

**第 3 步：算目标位置**（`agents/controllers/pd_joint_pos.py:76 set_action`）：

```python
_start_qpos  = 当前 qpos
if use_delta and not use_target:
    _target_qpos = _start_qpos + â      # ← 每步都从"当前"重新加
elif use_delta and use_target:
    _target_qpos = _target_qpos + â     # ← 累积（本配置未启用）
else:
    _target_qpos = â                    # 绝对值（夹爪走这里）
```

**这是全关最重要的一行**：因为 `use_target=False`，目标**不累积**。
连续两步给 0.1，第二步的目标是"那时的新 qpos + 0.1"，不是"初始 qpos + 0.2"。

**第 4 步：PD 伺服 + 物理**（`envs/sapien_env.py:1123`）：

```
每个 control step 内重复 sim_freq/control_freq = 100/20 = 5 次 scene.step()
PhysX 驱动按 PD 律出力：  τ = k_p (q* − q) − k_d q̇ ,  |τ| ≤ force_limit
Panda: k_p = 1000 N·m/rad, k_d = 100 N·m·s/rad, force_limit = 100
```

因此 **归一化 `a[0]=0.1` 对应关节目标偏移 `0.01 rad`；只有 `a[0]=1.0` 才对应 `0.1 rad`，且目标偏移不等于实际位移**。实际位移会被这些因素削减：
PD 有限刚度下的稳态误差（重力矩要求 `k_p·e = τ_gravity`，必然存在 `e`）、力矩限幅、
5 个物理步内来不及收敛、碰撞/关节限位。

**零动作时到底发生了什么（实测，逐关节）**：

| 关节 | reset | 第 3 步 | 变化 |
|---|---|---|---|
| `panda_joint1..7` | 0.0353 / 0.4007 / 0.0196 / −1.9187 / 0.0374 / 2.3366 / 0.8044 | **完全相同** | **0.0（一位都没动）** |
| `panda_finger_joint1/2` | 0.0400 | 0.0206 | **−0.0194（在闭合）** |

原因有两半：手臂是 `use_delta=True` 且 `use_target=False`，目标每步重设为**当时的 qpos**，
所以零动作的**关节目标是维持当时的位置**（3 步实测臂关节位移为 0；这并不能单凭数据证明 PD 驱动力恰好抵消重力，也不保证所有状态均零位移）；而夹爪是绝对目标，
零动作经 `0.5(high+low)+0.5(high−low)·a` 映射到 **区间中点 0.015 m**（实测读出的
`_target_qpos` = 0.014999999664723873），于是手指从 0.04 一路收到 0.0206。

→ **"零动作"对手臂是"保持"，对夹爪是"半开"命令。这是本关最容易踩的坑。**

### 2.2 观测组装：42 维从哪来

调用链：`sapien_env.py:546 _get_obs_state_dict` → `_get_obs_agent` + `_get_obs_extra`。

```
_get_obs_agent  → agent.get_proprioception()（agents/base_agent.py:339）
                → {"qpos": 9, "qvel": 9}     # 7 臂关节 + 2 手指关节
                # 若控制器有内部状态（use_target=True），这里会多一个 "controller": {"target_qpos": ...}
                # 本配置 use_target=False → get_state() 返回 {} → 观测里没有 controller 字段

_get_obs_extra  → (pick_cube.py:132)
                → is_grasped(1), tcp_pose(7), goal_pos(3)
                → if "state" in self.obs_mode:  obj_pose(7), tcp_to_obj_pos(3), obj_to_goal_pos(3)
```

合计 9+9+1+7+3+7+3+3 = **42**。`obs_mode="state"` 与 `"state_dict"` 信息量相同，只是前者被
`utils/common.py:195 flatten_state_dict` 按字典插入顺序拼成一个 `(1,42)` 张量（实测已验证两者一致）。

**哪些是"特权信息"**（真机上拿不到、必须靠感知或额外传感器）：

| 特征 | 真机可得性 |
|---|---|
| `qpos` / `qvel` | ✅ 编码器直接给 |
| `tcp_pose` | ✅ 正运动学算得出 |
| `goal_pos` | ⚠️ 视任务设定，通常人为给定 |
| `obj_pose`(7) | ❌ 需要物体位姿估计（视觉/标记） |
| `is_grasped`(1) | ❌ 由**接触力**判定（`panda.py:237`，力≥0.5 N 且夹角≤85°），真机需力/触觉传感 |
| `tcp_to_obj_pos` / `obj_to_goal_pos` | ❌ 依赖 `obj_pose` |

### 2.3 奖励：四项 + 门控 + 成功覆盖

`pick_cube.py:161 compute_dense_reward`，记 `d₁ = ‖cube − tcp‖`，`d₂ = ‖goal − cube‖`：

```
r = 1 − tanh(5·d₁)                     # A. reaching ∈ [0,1)，贴上去才趋近 1
  + 1·[is_grasped]                     # B. 抓住 +1
  + (1 − tanh(5·d₂))·[is_grasped]      # C. 只有抓住后才给"搬运/放置"进度
  + (1 − tanh(5·‖q̇_arm‖))·[is_obj_placed]   # D. 只有放到位后才奖励"别抖"
r[success] = 5                          # 成功直接覆盖
normalized = r / 5                      # 输出范围 [0, 1]
```

三个要点：

- **门控（gating）**：C 乘 `is_grasped`，D 乘 `is_obj_placed`。没抓住时 C、D 恒为 0，
  所以"把方块推近目标"这种不抓的投机行为拿不到 C 的分。
- **`tanh` 的作用**：把距离映射到有界区间且近距离梯度大 —— 靠近时每厘米的收益远大于远处。
- **`r[success] = 5` 是硬覆盖**，不是加和。**当方块尚未到位**（`is_obj_placed=False`），静止项关闭，前三项至多原始 3（归一化≤0.6）；**当方块到位但臂仍未静止**，静止奖励项可能为正，所以不能把 0.6 当成所有未成功状态的理论上限。成功时原始奖励直接覆盖为 5（归一化 1.0），并非固定从 0.6 跳变 0.4。这是"稠密塑形 + 稀疏成功"的混合奖励，
  也是 PPO 早期容易先学会 reach、很久才学会 grasp 的原因。

### 2.4 生命周期：两条独立出口

```
env.step(a)
  ├─ _step_action(a): 5 × scene.step()            # 物理推进 0.05 s
  ├─ _elapsed_steps += 1
  ├─ info  = get_info() = {elapsed_steps} ∪ evaluate()
  │            evaluate() = success, is_obj_placed, is_robot_static, is_grasped
  ├─ reward = get_reward(info)
  └─ terminated = info["success"]（没有 fail 键时）
     truncated  = 全 False ← base env 永远返回 False！
```

`truncated` 由外层补：`utils/registration.py:127 TimeLimitWrapper`
→ `truncated = (elapsed_steps >= 50)`。

- **`success = is_obj_placed & is_robot_static`**（`pick_cube.py:147`）：
  `is_obj_placed = ‖goal − cube‖ ≤ 0.025 m`；`is_robot_static = max|q̇_arm| ≤ 0.2 rad/s`
  （`panda.py:267`，注意 `qvel[..., :-2]` **排除了两根手指**）。
  → 方块到位但手臂还在动 = `is_obj_placed=True`、`success=False`。
- **没有失败终止**：`PickCube-v1` 不返回 `fail`，所以掉地上也不会提前结束，只能等 `truncated`。
- **50 步 = 50 个 control step = 2.5 s 仿真时间**（20 Hz），不是 50 个物理步。
- **不自动重置**：`gym.make` 出来是 `TimeLimitWrapper → OrderEnforcing → PickCubeEnv`，
  没有 auto-reset 包装器，`terminated/truncated=True` 之后必须自己 `reset()`。
  向量化封装（`ManiSkillVectorEnv`）才有自动重置，且 `ignore_terminations` 会改变语义。
- **种子的作用点**：`pick_cube.py:106 _initialize_episode` 决定方块 xy∈±0.1、z=0.02、yaw 随机；
  目标 xy∈±0.1、z = 方块z + U(0, 0.3)；臂初始 qpos 噪声 0.02 rad。实测 seed 0 两次运行**逐位相同**。

---

## 3. 代码导读：按这个顺序读 5 处

先打开 `.venv-maniskill/lib/python3.10/site-packages/mani_skill/`。

| # | 文件:行 | 符号 | 读的时候只回答一个问题 |
|---|---|---|---|
| 1 | `envs/sapien_env.py:1042` | `step` | 一个 `env.step` 里，`terminated` 和 `truncated` 各自被赋了什么？谁赋的？ |
| 2 | `envs/sapien_env.py:1073` | `_step_action` | 循环体跑几次？由哪个除法决定？（看 1123 行和 283 行） |
| 3 | `agents/controllers/pd_joint_pos.py:76` | `set_action` | `use_delta=True, use_target=False` 时，`_target_qpos` 用到的是**当前** qpos 还是**上个**目标？ |
| 4 | `agents/controllers/base_controller.py:125` | `_preprocess_action` | 策略的 ±1 是在哪一行变成 ±0.1 的？ |
| 5 | `envs/tasks/tabletop/pick_cube.py:147,161` | `evaluate` / `compute_dense_reward` | 哪几项乘了 `is_grasped` / `is_obj_placed`？`success` 由哪两个条件与运算？ |

辅助但值得扫一眼：`pick_cube.py:132 _get_obs_extra`（那行 `if "state" in self.obs_mode`）、
`utils/registration.py:127 TimeLimitWrapper`（10 行就能看完）、`agents/robots/panda/panda.py:237 is_grasping`。

---

## 4. 实测数据对照（把公式和数字对上）

| 现象 | 实测值 | 对应公式 |
|---|---|---|
| reset 后 `‖tcp−cube‖` | 0.16342215 | `d₁` |
| reset 后 `‖goal−cube‖` | 0.27600583 | `d₂` |
| 零动作第 1 步奖励 | **0.06530120** | `(1−tanh(5·0.1634))/5 = 0.06530115` ✅ |
| 第 2、3 步奖励 | 0.06530116 | 同上（qpos 略沉，`d₁` 微增） |
| 每步 `is_robot_static` | True | `max|q̇_arm| ≤ 0.2`，静止时成立 |
| `is_grasped` / `is_obj_placed` | False / False | 零动作没碰到方块 |
| `qpos` 变化（3 步零动作） | 臂 7 关节 **0.0**；手指 −0.0194 | §2.1 表格（夹爪收到中点 0.015） |
| horizon | 50 | `TimeLimitWrapper._max_episode_steps` |

---

## 5. 需要你回答的三个问题

> 都可以只用本文件 + 上面 5 处代码回答，不需要额外知识。答完我会逐条对源码核对，
> 对不上我会指出具体行，不会替你圆过去。

**Q1｜动作语义**
在 `pd_joint_delta_pos` 下，某一步策略输出 `a = [0.1, 0, 0, 0, 0, 0, 0, 1]`。
(1) 关节 1 的**目标位置**相对当前 qpos 偏了多少 rad？是怎么算出来的？
(2) 这等于"关节 1 立刻转过 0.1 rad"吗？如果不是，列出至少两个让它转不到 0.1 rad 的原因。
(3) 如果第 2 步输出 `[0.1, 0, ...]`，关节 1 的目标是"当前 +0.1"还是"初始 +0.2"？为什么？
(4) 这组动作里夹爪会张开还是闭合？依据是哪两个数？

**Q2｜观测与特权信息**
(1) `obs_mode="state_dict"` 和 `"state"` 的信息量为什么相同？差别只在哪一步？
(2) 为什么 `state_dict` 里也会出现 `obj_pose`、`tcp_to_obj_pos`？是哪一行代码导致的，
它是"设计意图"还是"副作用"？
(3) 如果把这套策略迁到真机，42 维里哪些**拿不到**？其中 `is_grasped` 为什么特别难，它是怎么算出来的？

**Q3｜奖励与终止**
(1) 为什么 `is_grasped=True` 不足以完成任务？`success` 由哪两个布尔量与运算得到？
(2) 说清 `success`、`terminated`、`truncated` 三者的**判定来源**分别在哪（哪个文件/哪个对象）。
(3) 一个 episode 跑到第 50 步时，方块刚好进入 2.5 cm 但手臂仍在动。此时
`is_obj_placed` / `is_robot_static` / `success` / `terminated` / `truncated` 各是什么？为什么？
(4) 区分“方块未到位但未成功”和“方块已到位、手臂仍在运动而未成功”：为什么前者有 0.6 的简单上界，而后者不能笼统断言奖励≤0.6？成功时为什么归一化奖励直接变为 1.0？

---

## 6. 常见误解（先别看，答完再对照）

<details>
<summary>点开查看</summary>

- ❌「动作 0.1 = 转 0.1 rad」→ 归一化动作 0.1 的目标偏移是 **0.01 rad**；归一化动作 1 才偏移 0.1 rad；PD 有动态误差、力矩限幅、5 步内来不及收敛。
- ❌「零动作应该完全不动」→ 手臂确实不动（实测 delta = 0.0），但**夹爪会收到区间中点 0.015 m**
  而闭合一段。"零动作"不等于"什么都不做"。
- ❌「delta 会自动累积」→ 本配置 `use_target=False`，**不累积**；累积是 `pd_joint_target_delta_pos`（`use_target=True`）。
- ❌「`truncated` 是环境给的」→ base env 的 `step` 恒返回 False，真正的截断来自外层 `TimeLimitWrapper`。
- ❌「失败会提前结束」→ PickCube 不返回 `fail`，掉地上也只是继续跑到第 50 步。
- ❌「`state_dict` 是设计上排除了 `obj_pose`」→ 是 `"state" in "state_dict"` 子串判断顺带命中的。
- ❌「夹爪也是 delta」→ 夹爪 `use_delta=False`，是绝对目标位置，语义和手臂**不一样**。

</details>

---

## 7. 答完之后

- 我按 Q1–Q3 逐条对源码核对，记录在 `docs/M1_LEARNING_CHECK.md`（新增 M1 段落）。
- 三条都过 → M1 可标 `PASSED`（技术证据已在 `docs/M1_EXECUTION_REPORT.md`）。
- 有偏差 → 指出薄弱点、安排针对性复看，原始回答保留，不润色成"已通过"。
- 之后才轮到 GPU 可见性确认 / GPU 吞吐微基准 / M2 官方基线复现（均需单独批准）。
