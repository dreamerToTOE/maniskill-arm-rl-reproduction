# ManiSkill Arm RL Reproduction | 机械臂强化学习复现

> Status: **PLANNED / 未开始复现** · Job focus: robotic manipulation + RL · Owner: dreamerToTOE

## Objective / 目标
以 **ManiSkill 3** 为基座复现机械臂操作任务中的基线强化学习算法，在可控的实验设置下，对比 PPO / SAC 训练性能、样本效率、接触操作稳定性和泛化效果。先完成公开 Baseline，再开展明确标识的个人扩展；严禁将上游结果写成个人实验结果。

- Upstream: https://github.com/mani-skill/ManiSkill
- Official docs: https://maniskill.readthedocs.io/
- Paper: https://arxiv.org/abs/2410.00425
- Related existing project: https://github.com/dreamerToTOE/fr3-robot-rl-control-lab

## Tasks / 任务
1. **PickCube**: pick-and-place, sparse vs dense reward and joint vs end-effector action.
2. **PushCube**: contact-rich manipulation, reward shaping and action safety.
3. **PegInsertionSide**: contact-rich insertion, train/evaluate with controls and criteria supported by pinned upstream; record any SAC baseline implementation work separately.
4. **FR3 transfer (optional)**: map ideas to the existing Isaac Sim FR3 push-into-slot environment without claiming identical physics.

## Reproduction milestones / 复现关卡
| Gate | Deliverable | Acceptance |
|---|---|---|
| M0 | Repo, upstream pin, environment audit | Versions, GPU memory, task list and reproducible setup recorded |
| M1 | Environment and observation audit | Explain state/action/reward/reset; zero-action + scripted smoke tests |
| M2 | Official baseline replay | Obtain learning curves, checkpoints and >=3 seeds; note if impossible |
| M3 | PPO vs SAC comparison | Same task/observation/evaluation budget; explicitly mark algorithm support and custom baselines |
| M4 | Reward/observation ablations | At least 2 meaningful ablations with failure diagnostics |
| M5 | Robustness and report | Independent evaluation, task success CI, plots, videos, method-to-code mapping |
| M6 | Optional FR3 application | Cross-simulator differences, contact limitations and new evaluation |

**Do not assume every task provides both official PPO and SAC.** Verify official training scripts for the pinned commit before scheduling an experiment; if absent, record a new implementation rather than calling it an official reproduction.

## Repo guide
- [AGENTS.md](AGENTS.md) — Codex collaboration and approval constraints
- [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md) — detailed step-by-step execution
- [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md) — immutable experiment schema
- [docs/PAPER_CODE_ALIGNMENT.md](docs/PAPER_CODE_ALIGNMENT.md) — algorithm/paper/code comparison
- [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md) — failures, root causes and retry cap
- [docs/RESUME_EVIDENCE.md](docs/RESUME_EVIDENCE.md) — evidence required before CV statements

## Principle
**Zero experimental results at initialization.** Every claim must point to a real local artifact and pinned upstream SHA; final report distinguishes exact reproduction / approximation / extension / not completed. No real-world deployment claim without physical tests.

## Local setup
Clone this orchestration repo and the upstream source **separately**. Use upstream's pinned installation instructions (do not blindly upgrade the user's existing Isaac Sim/ROS2 environments).

## 人与 Codex 学习分工（必读）

- [职责分工与逐阶段学习验收](docs/HUMAN_CODEX_WORKFLOW.md) — 研究者主导原理、实验原则和故障判断；Codex 主导实现、实验、日志；每阶段提供核心公式/源码导读、3 道理解检查题和实验证据。
- **研究边界**：五篇小论文复现完成并获得明确恢复指令前，双 FR3 主工程不推进、不被本仓库修改。独立开源复现可以正常进行。
- **[新会话从 M0 开始：完整交接清单](docs/M0_START_HERE.md)** — 先读论文/上游任务、核对 PPO/SAC 支持与环境兼容性；不直接启动大规模训练。

## Final mock interview / 项目面试验收（独立关卡）

- [正式验收标准与本项目专属问题](docs/FINAL_INTERVIEW_GATE.md)：3 分钟项目陈述、公式/控制原理、源码追踪、实验与统计、失败诊断和个人贡献。建议 100 分制，达到 80 分且各模块达标才通过，重大来源/实验真实性错误直接不通过。
- [面试原始记录与评分模板](docs/INTERVIEW_RECORD_TEMPLATE.md)：记录面试官提问、研究者本人原始回答、分项得分、弱项与补强方案；未实际面试保持 NOT_STARTED。
- **复现技术验收与求职面试验收分开记录**：Codex 可以组织材料和出题，但不能替研究者应答，也不能自动标记 PASS。项目未完成时可以进行阶段练习，不算最终验收。
