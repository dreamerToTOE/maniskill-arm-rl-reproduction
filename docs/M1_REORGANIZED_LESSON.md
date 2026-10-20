# ManiSkill M1 — 观测、动作、奖励、终止：从机器人视角重新学习

> 2026-10-20 概念优先版本；源于已提交的 `docs/M1_EXECUTION_REPORT.md` 和上游 ManiSkill v3.0.1 源码。M1 独立 Python 环境和 PickCube CPU 三步零动作 Smoke 已完成；**PPO/SAC 训练 NOT_RUN，GPU agent shell 未见设备节点，M1 学习验收待答。**

## 0. 为什么学 M1？

PPO/SAC 已经回答如何利用样本更新 Actor/Critic。M1 回答“样本从哪里来，策略动作如何驱动真实物理仿真”：`obs_t → policy/action_t → controller → PhysX → obs_{t+1}, reward_t, terminated, truncated`。现阶段尚未训练网络，使用人工零动作观察接口。

## 1. Observation：Actor 看到了什么？

`PickCube-v1 / Panda / state_dict` 实测：`agent.qpos=9`（7个臂+2个手指）、`agent.qvel=9`、`extra.is_grasped=1`、`tcp_pose=7`、`goal_pos=3`、`obj_pose=7`、`tcp_to_obj_pos=3`、`obj_to_goal_pos=3`；总计 **42 维**。`state` 只是将同一数据展平为 `(1,42)`，不是另一种感知模式。 `obs_mode='state_dict'` 仍包含 `obj_pose`，因为源码判断 `if 'state' in self.obs_mode` 同时匹配字符串 `state` 和 `state_dict`。

`qpos/qvel` 可由真机编码器获取，TCP 可由 FK 计算，但 `obj_pose` 等是仿真中直接提供的特权真值，真机要另做视觉/位置估计；`is_grasped` 在此仿真里根据双指接触力与接触方向判定，不能假设真机天然可读。

源码：`mani_skill/envs/sapien_env.py::_get_obs_state_dict`；`mani_skill/envs/tasks/tabletop/pick_cube.py::_get_obs_extra`。

## 2. Action：策略给命令，控制器尝试实现，物理决定结果

本次动作是 8 维连续归一化量 `a∈[-1,1]^8`。前 7 维是臂关节；最后一维是夹爪。配置 `control_mode=pd_joint_delta_pos`：

- **手臂**：`Δq_i=0.1*a_i` rad，`q_i,target=q_i,current+Δq_i`。`a_i=0.1 → +0.01 rad`（不是 +0.1 rad）；`a_i=1 → +0.1 rad`。目标≠实际关节位移。
- **夹爪**：`q_gripper,target=-0.01 + (a_g+1)*0.025` m，为**绝对目标**。`a_g=-1` 闭合方向，`a_g=+1` 张开方向，`a_g=0` 中间目标 **0.015 m**。单个夹爪动作维度映射至两个 mimic 指关节。
- **不按旧目标累积**：臂的 `use_delta=True, use_target=False`，每步 `q_target=q_current+Δq`，不是上次目标+Δq。但如果实际 qpos 逐步增加，那么实际角度仍可以持续积累，不能理解为“关节永远不会累积运动”。
- **PD + 物理仿真**：`sim_freq=100 Hz`、`control_freq=20 Hz`，一个 `env.step` 运行 5 个 PhysX 物理子步。驱动有刚度、阻尼、力矩和动力学限制，`Δq_target` 不是立即的 `Δq_actual`。
- **三步零动作实测**：7 个臂关节 qpos 均未变；手指由 0.0400 m 向目标 0.015 m 运动，第三步约 0.0206 m。因此全零动作不等于“完全不动”。但这种观察**不能证明**驱动恰好抵消重力，也不保证所有初始状态下都完全静止。

源码：`mani_skill/agents/robots/panda/panda.py::_controller_configs`；`mani_skill/agents/controllers/base_controller.py::_preprocess_action`；`mani_skill/agents/controllers/pd_joint_pos.py::set_action`；`mani_skill/envs/sapien_env.py::_step_action`。

## 3. Reward：分阶段引导，但不等于 Success

定义 `d_1=||cube-tcp||`、`d_2=||goal-cube||`。原始密集奖励：

```
reaching = 1 - tanh(5*d1)
grasping = 1 if is_grasped else 0
placing  = (1 - tanh(5*d2)) if is_grasped else 0
static   = (1 - tanh(5*||qvel_arm||)) if is_obj_placed else 0
raw_reward = reaching + grasping + placing + static
if success: raw_reward = 5
normalized_dense = raw_reward / 5
```

- **门控 gating**：未抓到方块不领取 `placing`，没放到位不领取 `static`。
- 三步零动作测试 `is_grasped=False`、`is_obj_placed=False`，只有 reaching 生效，数值约 **0.0653012**，由 `d_1≈0.16342215 m` 直接反算吻合。
- **成功不要求 grasped**：`success=is_obj_placed AND is_robot_static`。只要把方块推到目标附近并静止，也可能满足源码成功条件；训练评估时应额外核查抓取行为。
- **更正“未成功时归一化奖励上限固定 0.6”**：只有 `is_obj_placed=False` 时 static 项为零，前三项至多原始 3/归一化 0.6。若 `is_obj_placed=True` 且 `is_robot_static=False`，static 奖励仍可为正；尽管没有成功，归一化值理论上可能高于 0.6。不要使用固定 0.4 的必然跳差解释所有终止奖励。
- 成功时硬覆盖原始奖励 5 → 归一化 **1.0**，与平滑进度奖励不同。

源码：`mani_skill/envs/tasks/tabletop/pick_cube.py::evaluate`、`compute_dense_reward`、`compute_normalized_dense_reward`。

## 4. Episode：是谁决定任务完成和超时？

`success = (||cube-goal|| <= 0.025 m) AND is_static(0.2)`。Panda 静止判断只检查 7 个臂关节而非手指，速度绝对值阈值为 0.2 rad/s。

- `terminated`：本任务的 base environment 根据 `success` 返回是否按任务逻辑结束；不要求 `is_grasped`。
- `truncated`：base env 自身返回 False；外层 ManiSkill `TimeLimitWrapper` 监视 `elapsed_steps >= 50` 并设为 True。50 控制步 = 2.5 秒**模拟**时间（20 Hz），不是墙钟耗时。
- 第 50 步方块到位但臂还在动：`is_obj_placed=True`、`is_robot_static=False`、`success=False`、`terminated=False`、`truncated=True`。
- 第 50 步同时成功：按此包装器逻辑 `terminated` 和 `truncated` **都可能为 True**。
- 真实最大步数取 wrapper `_max_episode_steps=50`；此依赖组合下 `env.spec.max_episode_steps=None`，不能把后者误读为无时间上限。

源码：`mani_skill/envs/sapien_env.py::step`、`mani_skill/utils/registration.py::TimeLimitWrapper`、`pick_cube.py::evaluate`。

## 5. 最小源码阅读顺序（每次只带一个问题）

1. `pick_cube.py` 看 `_get_obs_extra`：机器人拿到了哪些仿真真值？
2. `panda.py` 看 `_controller_configs` + `pd_joint_pos.py::set_action`：动作 0.1 对应多少关节目标偏移？夹爪零动作目标是多少？
3. `sapien_env.py` 看 `_step_action`：为什么一个 RL 动作对应 5 个物理步？
4. `pick_cube.py` 看 `compute_dense_reward` + `evaluate`：高奖励与成功条件有何区别？
5. `registration.py::TimeLimitWrapper` 看 `step`：`terminated` 与 `truncated` 谁负责设置？

**不要**在理论关要求研究者记住所有行号；先说清接口和因果链，再定位具体实现即可。

## 6. 学习验收（研究者答题前不要公布标准答案）

1. **动作**：`a=[0.1,0,0,0,0,0,0,0]` 时，第一个关节目标偏移多少？夹爪收到什么目标？连续两步会不会基于旧 target 累加？
2. **观测**：为何 `state_dict` 里有 `obj_pose`？哪些变量在真机需要感知而非直接读取？
3. **奖励与结束**：方块推到目标且臂静止、但从未抓住时能否成功？第 50 步方块到位、臂未静止时 `terminated/truncated` 分别为何值？

学习验收尚未发生；M1 状态仍是 **SMOKE_PASSED / UNDERSTANDING_PENDING**。GPU 在本地 agent shell 不可见（用户终端可见性需要独立检查），并未开始训练。
