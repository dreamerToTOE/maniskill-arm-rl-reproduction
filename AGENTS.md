# Codex / Agent Working Agreement

## Purpose
This repository holds a **not-yet-executed** reproduction. The agent must report actual work, not convert plans into accomplishments.

## Mandatory startup
1. Read README, this file, the full upstream methods/paper/docs, and existing logs. Identify **problem → method → innovation → experiments → dependencies → expected metrics** before coding.
2. Extract upstream commit/tag, upstream license, data provenance, supported simulator/task/action versions; record in `docs/UPSTREAM_LOCK.md` before installing.
3. Propose what counts as exact reproduction vs adaptation vs new extension. Ask the user to choose scope when important experiments or resources differ.
4. Maintain a visible progress log with dates, commit hashes and command outcomes. No unsupported percentage estimates.

## Implementation policy
- Never replace existing main research projects or their environments. Create an isolated Python environment and do not silently upgrade system CUDA / NVIDIA driver / ROS2 / Isaac Sim.
- Do not copy the entire upstream implementation into this repo without reviewing its license, attribution, and asset/data constraints. Prefer separately cloning an immutable upstream tag/SHA.
- Make the smallest testable change. Keep baseline untouched and clearly differentiate official baseline from own modifications.
- Seeds, randomization, observation/action spaces, units, control step, train/validation/test splits, simulator params and target metrics must be saved.
- Never claim success, real-robot deployment, original-paper numerical agreement, or statistical improvement without measured artifacts.
- Never launch overnight training, large GPU batches, download restricted data, or perform irreversible operations without explicit user-approved budget.

## Three-attempt failure protocol
On 3 meaningfully different unsuccessful attempts at the **same blocker**, stop blind retries; classify as (A) local environment/implementation issue, (B) upstream open-source/data deficiency, (C) paper ambiguity, or (D) resource constraint. Provide evidence and bounded next steps. A smaller, numerical-simulation-only equivalent requires explicit labeling.

## Required artifacts for every completed gate
`docs/PROGRESS.md`: objective / changed files / commit / command / result / next action; `docs/EXPERIMENT_MATRIX.md`: experiment ID, configs, pinned SHA and output path; `docs/FAILURE_LOG.md`: failures; outputs in versioned `results/` summaries; plots or videos with reproducible generating scripts. Checkpoints and datasets must be gitignored unless deliberately approved.

## Reporting template
- Target gate & reproduction claim:
- Source paper/upstream exact location:
- Code changes & local commit:
- Tests and commands executed (pass/fail):
- Metrics and artifact locations:
- Deviations / uncertainty:
- Next proposed bounded run (not automatically launched):

## 研究者与 Codex 分工（强制读取）

**Before any execution, read [docs/HUMAN_CODEX_WORKFLOW.md](docs/HUMAN_CODEX_WORKFLOW.md).**

- 研究者主导研究问题、创新、核心算法公式、实验原则和故障判断，并最终能够独立面试答辩；Codex 讲解、实现、运行、记录和提供证据。
- 每关必须交付原理与源码映射、3 个需要研究者回答的问题、验证数据、待审批决策；没有证据不标记完成。
- 主 FR3 工程在五篇小论文复现完成并获用户明确许可前**保持冻结**；本仓库可独立复现但禁止擅自迁移/修改主工程。
- MPC 等非当前任务所需知识按需讲解，不得以“先学完全部理论”为复现前置条件。
- 新窗口/新 Codex 会话仍应服从上述约束，不能仅阅读本文件而跳过链接中的完整分工表。

## 最终面试验收 Gate（必须执行，不等于复现完成）

- **复现完成与面试通过是两个独立状态**。严格阅读 [docs/FINAL_INTERVIEW_GATE.md](docs/FINAL_INTERVIEW_GATE.md)，按项目对应的理论、源码、实验及故障诊断能力出题。
- Codex 整理真实实验数据、源码映射和题库；**研究者自己**在正式模拟面试时进行 3 分钟陈述、公式推导、代码定位与问题诊断，禁止 Codex 代答或实时提示。
- 每个独立复现阶段仍需至少 3 道学习检查；最终面试只在确有可核实验数据后标记 READY，若项目尚未完成可做 PRACTICE_ONLY。
- 按 [docs/INTERVIEW_RECORD_TEMPLATE.md](docs/INTERVIEW_RECORD_TEMPLATE.md) 保存真实问答和打分。未开展的面试必须标记 NOT_STARTED，不得因代码、训练完成而自动判定 PASS。
- 不合格须记录薄弱点、安排针对性复习和二次提问；原始回答保留，不能人为润色成“已通过”。
