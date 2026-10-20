# M1 Execution Report — isolated env + PickCube CPU 3-step smoke

> **Status: M1_EVIDENCE_COMPLETE / SMOKE_PASSED_RUNTIME_VERIFIED / M1_NOT_MARKED_PASSED（等待研究者理解关）/ TRAINING_NOT_RUN / GPU_NOT_ACCESSIBLE_IN_AGENT_SHELL**
> Date: 2026-10-20
> Repo: `dreamerToTOE/maniskill-arm-rl-reproduction` @ `4a4e5ff92fd4a1de13e9b0ed57a09d15d363e902` (执行前基线)
> Upstream source pin: ManiSkill `v3.0.1` = `a4a4f9272ad64b1564035874b605ceb687b63ed8`
> 所有数值均来自本机实际运行输出，非引用论文或文献预期值。

## 1. 执行摘要 / Summary

| 项目 | 结果 | 证据 |
|---|---|---|
| 隔离环境安装 | **成功** | `.venv-maniskill/`，Python 3.10.12，`pip check` 无冲突 |
| 依赖解析 | **一次通过**，未出现 resolver conflict，未强装 | 安装日志 `results/raw/`（pip 输出） |
| 磁盘增量 | **6.21 GB**（venv 6,210,349,110 B），预算 12 GB 内 | `results/m1_environment_manifest.json` |
| PickCube CPU 3 步 smoke | **通过**，exit code 0 | `results/m1_pickcube_state_dict_zero_seed0.json` |
| PyTorch CUDA | runtime = 12.1（cu121 wheel）；`torch.cuda.is_available()` = **False（本 shell 无 GPU 设备节点）** | manifest |
| 训练 / GPU benchmark | **未运行（NOT_RUN）**，按约束不做 | — |
| M1 判定 | **不标记 PASSED**：证据已齐，但研究者理解关（3 题）尚未回答 | 本文档 §7 |

## 2. 环境安装 / Environment installation

### 2.1 实际执行命令（本机 shell）

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
git status --short                     # 干净
git pull --ff-only origin main         # 046a11b..4a4e5ff Fast-forward，无冲突

/usr/bin/python3 -m venv .venv-maniskill          # 必须用 /usr/bin/python3 = 3.10.12
./.venv-maniskill/bin/python -m pip install --upgrade pip     # 22.0.2 -> 26.2.1

# 与批准文档唯一不同处：TMPDIR 重定向（原因见 §6.1）
export TMPDIR=/home/ubuntu2004/.cache/ms-m1-tmp

./.venv-maniskill/bin/python -m pip install --no-cache-dir 'torch==2.5.1' \
    --index-url https://download.pytorch.org/whl/cu121
./.venv-maniskill/bin/python -m pip install --no-cache-dir \
    'mani_skill==3.0.1' 'sapien==3.0.0' 'gymnasium==1.1.1' 'numpy==1.24.4'
./.venv-maniskill/bin/python -m pip check
```

### 2.2 实测版本（`importlib.metadata`，非推测）
| 包 | 版本 | 说明 |
|---|---|---|
| Python | **3.10.12** | `/usr/bin/python3`，venv 内 `sys.version` 一致 |
| venv 路径 | `.venv-maniskill/bin/python` | `is_venv = True`，未触碰系统解释器 |
| torch | **2.5.1+cu121** | 官方 PyTorch CDN wheel |
| torch CUDA runtime | **12.1** | 编译期 CUDA；cudnn 9.1.0（90100） |
| sapien | **3.0.0** | cp310 manylinux_2_28 wheel |
| mani-skill | **3.0.1** | PyPI wheel（上游 git pin 仅作源码参照，非二进制同一性） |
| gymnasium | **1.1.1** | 与批准候选一致，未冲突 |
| numpy | **1.24.4** | 与批准候选一致，未冲突 |
| scipy / opencv / trimesh / mplib / toppra / pytorch-kinematics | 1.15.3 / 4.11.0.86 / 5.1.1 / 0.1.1 / 0.6.10 / 0.7.6 | mani_skill 传递依赖，自动解析 |
| `pip check` | **No broken requirements found.** | returncode 0 |

完整 96 条依赖锁定：`results/m1_requirements.lock.txt`；机器可读清单：`results/m1_environment_manifest.json`。

### 2.3 wheel 溯源（SHA-256，取自 PyPI 元数据）

| 包 | wheel | SHA-256 |
|---|---|---|
| mani-skill 3.0.1 | `mani_skill-3.0.1-py3-none-any.whl` | `685de2f03c300b1ede49881a1bf6306ad062082d39c8d3be8b8e85603f32e33a` |
| sapien 3.0.0 | `sapien-3.0.0-cp310-cp310-manylinux_2_28_x86_64.whl` | `c57837cc6998f42dad1355c2332b1d0e5b815acb438f4f38f250de8e2576cc90` |
| gymnasium 1.1.1 | `gymnasium-1.1.1-py3-none-any.whl` | `9c167ec0a2b388666e37f63b2849cd2552f7f5b71938574c637bb36487eb928a` |
| numpy 1.24.4 | `numpy-1.24.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `7ffe43c74893dbf38c2b0a1f5428760a1a9c98285553c89e12d70a96a7f3a4d6` |
| torch 2.5.1+cu121 | 未保留 wheel（`--no-cache-dir`） | **未记录**（如实标注，不编造） |

### 2.4 环境重建 / Recreate on another machine

整个安装已收敛为一条命令，不需要再记 §2.1 的细节：

```bash
bash scripts/setup_maniskill_env.sh              # 建 venv + 装依赖 + 写 manifest/lock
bash scripts/setup_maniskill_env.sh --check-only # 只体检：解释器版本 + /tmp 容量
```

脚本行为：强制 `/usr/bin/python3`（3.10.x，匹配 SAPIEN cp310 wheel）；检测 `/tmp` < 3 GB 时改用
`$HOME/.cache/ms-m1-tmp` 并写入 venv 的 `activate`；`pip check` 非零即失败退出；最后生成
`results/m1_environment_manifest.json` 与 `results/m1_requirements.lock.txt`。
全程不 sudo、不改系统解释器/驱动/CUDA/ROS2/Isaac Sim。迁移云机时直接跑这条命令即可，
这也是 `docs/CLOUD_MIGRATION_PLAYBOOK.md` 要求的"远程干净安装"路径。

## 3. Smoke test 结果 / Bounded CPU smoke

### 3.1 命令与退出码

```bash
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
./.venv-maniskill/bin/python scripts/m1_inspect_pickcube.py \
  --obs-mode state_dict --action-mode zero --steps 3 --seed 0 \
  --output results/m1_pickcube_state_dict_zero_seed0.json
# exit code = 0
```

配置实测确认：`PickCube-v1` / `panda` / `physx_cpu` / `state_dict` / `pd_joint_delta_pos` / `normalized_dense` / `num_envs=1` / `seed=0` / 零动作 / 3 步。

### 3.2 逐步实测值

| t | action | reward | terminated | truncated | success | is_grasped | is_obj_placed | is_robot_static | elapsed_steps |
|---|---|---|---|---:|---|---|---|---|---|
| 0 | 8×0.0 | **0.06530120** | False | False | False | False | False | **True** | 1 |
| 1 | 8×0.0 | **0.06530116** | False | False | False | False | False | **True** | 2 |
| 2 | 8×0.0 | **0.06530116** | False | False | False | False | False | **True** | 3 |

**奖励公式经数值反算验证**（不是只看源码）：reset 后 `‖tcp−cube‖ = 0.16342215`，
`(1 − tanh(5×0.16342215))/5 = 0.06530115`，与实测 `0.06530120` 在 1e-7 量级一致；
`is_grasped=False` 时 `place` 项贡献为 0，因此奖励只剩 reach 项 —— 与源码分支一致。

### 3.3 产物

| 路径 | 内容 |
|---|---|
| `results/m1_pickcube_state_dict_zero_seed0.json` | 修正后 smoke 结果（`horizon=50`） |
| `results/raw/PREFIX_horizon_null_m1_pickcube_state_dict_zero_seed0.json` | **修正前**原始产物（`horizon=null`），失败证据保留 |
| `results/raw/m1_smoke_stdout.log` / `m1_smoke_stderr.log` | 完整 stdout/stderr |
| `results/raw/m1_interface_probe.json` | 接口探针（观测/动作/控制器/reset/3 步） |

## 4. 接口实测分析 / Interfaces: runtime facts vs source expectations

标注约定：**[实测]** 来自本机运行；**[源码]** 来自已安装 wheel 源码阅读；两者同时给出时才作为可依赖结论。

### 4.1 Observation

**[实测]** `obs_mode="state_dict"` 顶层只有 `agent`、`extra` 两组：

| 组 | key | shape | dtype | 来源 |
|---|---|---|---|---|
| agent | `qpos` | (1, 9) | float32 | 本体感受：7 臂关节 + 2 指关节 |
| agent | `qvel` | (1, 9) | float32 | 本体感受：关节速度 |
| extra | `is_grasped` | (1,) | **bool** | 任务判定（接触力 + 角度） |
| extra | `tcp_pose` | (1, 7) | float32 | 末端位姿（位置 3 + 四元数 4） |
| extra | `goal_pos` | (1, 3) | float32 | 目标点位置 |
| extra | `obj_pose` | (1, 7) | float32 | 方块位姿 |
| extra | `tcp_to_obj_pos` | (1, 3) | float32 | 相对向量 |
| extra | `obj_to_goal_pos` | (1, 3) | float32 | 相对向量 |

- **总维度 [实测]：42** = 9 + 9 + 1 + 7 + 3 + 7 + 3 + 3。
  另用 `obs_mode="state"` 单独 reset 验证：返回 `(1, 42)` float32 Tensor，`matches_state_dict_sum = True`。
- **全部为仿真真值（simulator ground truth）**：没有任何传感器噪声、遮挡或渲染管线介入；
  `obj_pose`、`goal_pos`、`is_grasped` 在真机上无法直接获得（`is_grasped` 由接触力计算，源码注释也说明现实中只能靠夹爪闭合量"hack"）。
- **[源码] 为何 state_dict 也带 `obj_pose` 等相对量**：`pick_cube.py::_get_obs_extra` 用
  `if "state" in self.obs_mode:` 判断，而 `"state_dict"` 字符串**包含** `"state"` 子串，故同样注入这 3 个 key。
  这是字符串子串判断的副作用，不是显式设计。
- `agent.qpos` 实测 reset 值（seed 0）：`[0.0353, 0.4007, 0.0196, -1.9187, 0.0374, 2.3366, 0.8044, 0.04, 0.04]`，
  末端两维 0.04 = 夹爪完全张开，与 `panda.py` 夹爪 `upper=0.04` 一致。

### 4.2 Action

- **[实测]** `single_action_space` = `Box(-1.0, 1.0, (8,), float32)`。
- **[实测]** `CombinedController` 由两个子控制器拼接，`action_mapping = {"arm": [0,7], "gripper": [7,8]}`：
  - `arm`：`PDJointPosController`，7 维（`panda_joint1..7`）
  - `gripper`：`PDJointPosMimicController`，1 维（`panda_finger_joint1`，mimic 到 joint2）
- **[实测/源码]** 归一化语义（两者一致）：
  - 臂：`normalize_action=True`，`lower=-0.1, upper=0.1`，`use_delta=True`
    → 动作 `a_i ∈ [-1,1]` 先 clip 再线性映射到 `[-0.1, +0.1]` rad，作为**相对当前 qpos 的增量**：
    `target_qpos = qpos_now + 0.1·a_i`。
  - 夹爪：`normalize_action=True`，`lower=-0.01, upper=0.04`，**`use_delta=False`**
    → 动作是**绝对目标关节位置**：`a=-1 → -0.01`（闭合），`a=+1 → 0.04`（张开）。
    **手臂是增量、夹爪是绝对值**，两者语义不对称，容易踩坑。
- **[源码] 控制链**：`env.step` → `_step_action` → `controller.set_action` → `_preprocess_action`
  （`clip_and_scale_action`）→ `set_drive_targets` → 每个 control step 内跑
  `_sim_steps_per_control = sim_freq/control_freq = 100/20 = 5` 次 `scene.step()` → PhysX PD 驱动（stiffness 1000 / damping 100 / force_limit 100）。
- **[实测] 零动作不等于完全静止**：3 步内 qpos 相对 reset 的最大偏移 0.0098 → 0.0158 → 0.0194 rad，
  原因是重力下的 PD 稳态沉降；但 `is_robot_static` 仍为 True（阈值 0.2 rad/s）。

### 4.3 Reward

- **[实测]** 三步奖励 ≈ **0.0653012 / 0.0653012 / 0.0653012**（`normalized_dense`）。
- **[源码]** `pick_cube.py::compute_dense_reward`：

```python
tcp_to_obj_dist = ‖cube.p − tcp.p‖
reward  = 1 − tanh(5·tcp_to_obj_dist)          # reaching
reward += is_grasped                            # +1 若抓住
reward += (1 − tanh(5·obj_to_goal_dist)) · is_grasped   # grasp 门控的 place
reward += (1 − tanh(5·‖qvel_arm‖)) · is_obj_placed      # placed 门控的 static
reward[success] = 5                             # 成功硬覆盖
```

`compute_normalized_dense_reward` = 上式 `/5`。
- **成功与稠密奖励的区别**：`success = is_obj_placed & is_robot_static`，
  即"方块进 2.5 cm 球内 **且** 机械臂关节速度 ≤ 0.2 rad/s"。
  `is_obj_placed` 单独为真不足以成功（臂还在动就不算）。奖励是**连续塑形**，
  在未抓取时 place 项被门控为 0，所以零动作下奖励恒等于 reach 项，不会随"接近目标"而涨。
- 成功时奖励跳到 **5**（归一化后 1.0），与之前的平滑值不连续 —— 这是"稀疏成功 + 稠密塑形"的混合设计。

### 4.4 Episode lifecycle

- **[实测] wrapper 链**：`TimeLimitWrapper(ManiSkill) → OrderEnforcing → PickCubeEnv`。
- **[实测] 时间上限 = 50 个 control step**（取自 wrapper 的 `_max_episode_steps`）。
  ⚠️ **注意**：`env.spec.max_episode_steps` 在 gymnasium 1.1.1 下返回 `None`（见 §6.2），
  只看 spec 会误判为"无上限"。
- **[实测] terminated**：`PickCubeEnv.evaluate()` 只返回 `success/is_obj_placed/is_robot_static/is_grasped`，
  没有 `fail` 键 → base `step` 中 `terminated = info["success"]`，**只有成功才 terminated**；无失败终止条件。
- **[实测] truncated**：base env 的 `step` **永远返回全 False**；截断只由外层 `TimeLimitWrapper` 在
  `elapsed_steps >= 50` 时置真。因此"到时间"和"成功"是两条独立出口。
- **[实测] reset**：`reset(seed=0)` 两次（含两个独立进程实例）得到**完全逐位相同**的
  `obj_pose / goal_pos / qpos`；`seed=1` 则不同 → CPU 后端下固定种子可复现。
  随机化范围 [源码]：方块 xy ∈ ±0.1、z = cube_half_size = 0.02、yaw 随机；
  目标 xy ∈ ±0.1、z = cube_z + U(0, 0.3)；臂初始 qpos 噪声 0.02 rad。
- **[实测] 频率**：`sim_freq = 100 Hz`，`control_freq = 20 Hz`，`sim_dt = 0.01 s`，
  `control_dt = 0.05 s`，**每个 env.step = 5 个物理步**。50 步上限 = 2.5 s 仿真时间。
- **[实测] info 键**：`elapsed_steps, success, is_grasped, is_obj_placed, is_robot_static`（reset 时另有 `reconfigure`）。
- **[实测] 几何参数**：`goal_thresh = 0.025 m`，`cube_half_size = 0.02`，`max_goal_height = 0.3`。

## 5. 重要源码位置 / Source map（已安装 wheel，版本 3.0.1）

| 关注点 | 文件与符号 |
|---|---|
| 任务注册与时间上限 | `mani_skill/envs/tasks/tabletop/pick_cube.py` `@register_env("PickCube-v1", max_episode_steps=50)` |
| 随机化 | 同文件 `_initialize_episode` |
| 观测扩展 | 同文件 `_get_obs_extra` |
| 成功判定 | 同文件 `evaluate` |
| 稠密/归一化奖励 | 同文件 `compute_dense_reward` / `compute_normalized_dense_reward` |
| 几何超参 | `mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py` `PICK_CUBE_CONFIGS["panda"]` |
| 状态观测组装 | `mani_skill/envs/sapien_env.py` `_get_obs_state_dict` / `_get_obs_agent` |
| step 与 terminated | 同文件 `step`、`get_info`、`_step_action` |
| 截断语义 | `mani_skill/utils/registration.py` `TimeLimitWrapper`、`register_env` |
| 控制器动作缩放 | `mani_skill/agents/controllers/pd_joint_pos.py` `set_action`；`base_controller.py` `_preprocess_action` / `_clip_and_scale_action` |
| Panda 控制器配置 | `mani_skill/agents/robots/panda/panda.py` `_controller_configs` |
| 抓取判定 | 同文件 `is_grasping`（接触力 ≥0.5 N 且 夹角 ≤85°）、`is_static(0.2)` |

## 6. 遇到的问题与处理 / Problems and resolutions

### 6.1 pip 安装首轮失败：`OSError: [Errno 28] 设备上没有空间`（10.5 MB / 780 MB）

- **现象**：`torch` 下载到 10.5 MB 即报 errno 28，但 `df -h /` 显示 418 GB 可用。
- **根因定位**：`df -h /tmp` → **tmpfs 仅 10 MB**。pip 把临时解包/下载文件放 `/tmp`，
  与根分区容量无关。属于**执行环境限制**，不是磁盘不足、也不是依赖冲突。
- **处理**：`export TMPDIR=/home/ubuntu2004/.cache/ms-m1-tmp`（位于 418 GB 根分区）后重试即成功。
- **已固化（2026-10-20 补）**：新增 `scripts/setup_maniskill_env.sh`，自动检测 `/tmp` 可用空间
  （阈值 3 GB），不足则把 `TMPDIR` 指向 `$HOME/.cache/ms-m1-tmp`（可用 `MANISKILL_TMPDIR` 覆盖），
  再把同一 `export` 幂等地写进 `.venv-maniskill/bin/activate`。
  验证：`bash scripts/setup_maniskill_env.sh --check-only` 正确报出 `/tmp` 仅 10 MB；
  `source .venv-maniskill/bin/activate` 后 `tempfile.gettempdir()` 指向大分区，
  且一次 17.3 MB 的 `pip download`（大于 10 MB tmpfs）成功。以后不需要再手敲 `TMPDIR`。
- **是否属于"三次失败"**：不属于。同一 blocker 共 1 次失败 + 1 次根因修正即通过。
- **对可复现性的影响**：普通机器 `/tmp` 通常足够大，脚本会**跳过**该分支；已记入 `docs/FAILURE_LOG.md`。
  迁移云机时若 `/tmp` 为 tmpfs，脚本同样自动兜底。

### 6.2 `scripts/m1_inspect_pickcube.py` 的 `horizon` 字段误报为 `null`

- **现象**：脚本输出 `"horizon": null`，但任务在源码里注册为 `max_episode_steps=50`。
- **根因**：脚本读的是 `env.spec.max_episode_steps`。ManiSkill 通过
  `additional_wrappers=[WrapperSpec(..., kwargs=dict(max_episode_steps=50))]` 注册自己的
  `TimeLimitWrapper`，真实的 50 保存在 wrapper 实例的 `_max_episode_steps` 上，
  而 gymnasium 1.1.1 的 `EnvSpec.max_episode_steps` **保持为 None**。
  于是"没有上限"是假象；实际运行中到第 50 步仍会 truncate。
- **处理（最小必要修正）**：新增 `episode_horizon()`，沿 wrapper 链读取 `_max_episode_steps`，
  读不到才回退到 spec；并新增 `wrappers` 字段如实记录 wrapper 链。未改动任何仿真/奖励/动作逻辑。
- **证据保全**：修正前的原始产物保存在
  `results/raw/PREFIX_horizon_null_m1_pickcube_state_dict_zero_seed0.json`，未覆盖、未删除。
- **影响面**：仅影响"报告读数"，不影响 smoke 本身是否通过。

### 6.3 本机 GPU 在 agent shell 内不可见（环境限制，未绕过）

- **现象**：`nvidia-smi` 失败（returncode 9），`torch.cuda.is_available() = False`，`device_count = 0`。
- **根因定位**（三层证据）：
  1. `/proc/driver/nvidia/version` 存在 → `NVRM version: 580.178.04`，驱动**已加载**；
  2. `lspci` → `GA104 [GeForce RTX 3070 Lite Hash Rate]`，硬件在；
  3. `ls /dev/nvidia*` → **不存在**。
  → 即 agent 的执行命名空间**没有暴露 GPU 设备节点**，不是驱动损坏、不是显卡不存在、也不是安装错误。
- **处理**：不尝试修改驱动或权限（明确超范围）。CPU smoke 本来就用 `physx_cpu`，SAPIEN 回落到
  CPU 渲染设备（日志：`Requested render device "sapien_cuda" ... Falling back to "cpu"`），因此测试有效。
- **后果**：M1 的"PyTorch CUDA 可用性"只能报告为 **在 agent shell 内不可用**，
  研究者本人终端下是否可用**未被本次执行验证**（其 2026-10-20 审计显示 5957 MiB free，说明用户侧可见 GPU）。
  **GPU 吞吐与显存占用依旧 UNKNOWN**，不得由本次结果推断。

## 7. 与计划配置的偏差 / Deviations

| # | 计划 | 实际 | 理由 / 影响 |
|---|---|---|---|
| 1 | `python -m venv` 使用默认 python3 | 显式 `/usr/bin/python3` | agent shell 默认 `python3` 是 WorkBuddy 托管的 **3.13.12**；必须用 3.10.12，否则 SAPIEN cp310 wheel 不匹配 |
| 2 | 直接 pip install | 额外设置 `TMPDIR` | §6.1，`/tmp` 仅 10 MB |
| 3 | 脚本原样运行 | 修正 `horizon` 读数 | §6.2，属真实逻辑缺陷，最小修正 |
| 4 | 预期 GPU 可用 | GPU 在 agent shell 不可见 | §6.3，环境限制，未绕过、未降级依赖 |
| 5 | 预期 `pip` 升级到最新 | 22.0.2 → 26.2.1（venv 内） | 未触碰系统 pip 22.0.2 |
| 6 | — | 额外新建 `scripts/m1_env_manifest.py` | 为满足"可复现 manifest"要求，只读采集，不参与仿真 |

未发生：sudo / apt / 系统包变更、驱动或 CUDA Toolkit 变更、ROS2/Isaac Sim 变更、
其他仓库（含双 FR3 主工程）改动、GL 资产下载、checkpoint 下载、训练、GPU benchmark、云租用。

## 8. 本地是否可用于后续 GPU 性能测试 / Readiness

**结论：CPU 侧就绪；GPU 侧"未验证"，且在当前 agent 通道下不能验证。**

- 可确认：依赖图完整、`pip check` 干净、CPU 物理仿真与任务逻辑可运行、种子可复现。
- **不可确认**：RTX 3070 上的 CUDA 可用性、显存峰值、并行环境数上限、训练吞吐。
  本次 `torch.cuda.is_available()=False` 是**设备节点不可见**导致，不能读成"3070 不能跑"，
  也不能读成"3070 能跑"。
- 因此**不建议**据此直接批准 GPU benchmark；应先由研究者在自己终端确认
  `nvidia-smi` 与 `torch.cuda.is_available()`（一行命令即可），再走单独审批。

## 9. 下一步建议（均需单独批准，未执行）

1. **研究者本机 GPU 可见性确认**（0 风险，1 条命令）：在 `.venv-maniskill` 下
   `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。
2. 若可见 → 申请 **GPU state-only 仿真吞吐微基准**（并行数 64/128/256/512，每次 ≤60 s，总 ≤8 min，无优化器）。
3. 之后再申请**短训练 pilot**（PickCube state PPO，≤100k env steps，≤10 min）。
4. 在 M2 之前补：`pd_ee_delta_pos` 对照说明（SAC 官方示例用该控制器，PPO 用 joint），避免控制器混淆的比较。

## 10. 状态与证据索引

- `M1_SMOKE = PASSED`（exit 0，3 步，证据齐）
- `M1_INTERFACE_AUDIT = RUNTIME_VERIFIED`（观测/动作/奖励/生命周期均有实测值）
- `M0 = NOT_MARKED_PASSED`（隔离依赖与 CPU smoke 证据已补，但研究者理解关未完成）
- `M1 = NOT_MARKED_PASSED`（同上；理解关是独立验收项）
- `TRAINING = NOT_RUN`，`GPU_BENCHMARK = NOT_RUN`，`CLOUD = NOT_PROVISIONED`

证据文件：
`results/m1_pickcube_state_dict_zero_seed0.json`、`results/m1_environment_manifest.json`、
`results/m1_requirements.lock.txt`、`results/raw/m1_interface_probe.json`、
`results/raw/m1_smoke_stdout.log`、`results/raw/m1_smoke_stderr.log`、
`results/raw/PREFIX_horizon_null_m1_pickcube_state_dict_zero_seed0.json`。
