# M0 START HERE — ManiSkill Arm RL Reproduction

> Status: **NOT_STARTED at handoff**. This is a concrete onboarding and audit plan, not proof that baseline algorithms or environments have been reproduced.
> GitHub: https://github.com/dreamerToTOE/maniskill-arm-rl-reproduction
> Upstream: https://github.com/mani-skill/ManiSkill
> Paper: https://arxiv.org/abs/2410.00425

## Important research boundary
- The existing dual FR3 main project is frozen until all five small-paper reproductions are finished and the researcher explicitly approves restart.
- This independent ManiSkill repository can proceed with learning, upstream reading, its own isolated environment and subsequently approved baseline experiments.
- Do **not** edit `dual-arm-embodied-palletizing`, `fr3-robot-rl-control-lab`, `dual-arm-force-control-mujoco-isaacsim`, or other research repos while performing this task.

## M0 objectives (do these first, in this order)

### 0. Read existing governance and scope
Read `AGENTS.md`, `docs/HUMAN_CODEX_WORKFLOW.md`, `README.md`, `docs/REPRODUCTION_PLAN.md`, `docs/UPSTREAM_LOCK.md`, `docs/PROGRESS.md`, `docs/EXPERIMENT_MATRIX.md`, and current GitHub issues. Do not change the already agreed learning/approval model.

### 1. Explain research framework vs algorithm reproduction to the learner
Produce a clear Chinese briefing with:
- What ManiSkill 3 is (a SAPIEN-based manipulation simulation/benchmark framework, **not** a new standalone PPO/SAC algorithm).
- How `environment reset/step -> observation -> action -> reward/terminated/truncated -> rollout or replay buffer -> policy update -> evaluation` maps to actual ManiSkill interfaces and supported PPO/SAC reference training code.
- Official goals and source of PickCube, PushCube and PegInsertionSide; clarify what the controller/action and reward components mean.
- Paper vs benchmark distinction: which scientific claims correspond to an original paper and which will be our custom baseline comparison. **No direct claim of full scientific reproduction simply by running official demos.**

### 2. Upstream audit and reference alignment
- Inspect exact upstream repository, docs, paper and license; propose a **version/tag/SHA to pin** and explain why.
- In pinned reference, verify exact task IDs, default robot model, observation/actions, reset distribution, dense/sparse reward, termination/success, control frequency, available built-in PPO and SAC training scripts. Keep `supported/unsupported/unknown` separately for each task/algorithm pair.
- Locate training entry points, env wrappers, PPO Actor/Critic and losses, SAC Twin-Q/Actor/Target update and replay logic where actually available.
- Produce concise formula-to-code anchors: PPO clipped objective/GAE; SAC critic loss, double Q, reparameterized actor and entropy alpha objective. Label any missing code as unavailable or an independent future implementation; do not invent locations.

### 3. Environment and hardware audit (no invasive setup)
- Confirm actual OS, Python, NVIDIA GPU VRAM, driver, CUDA, PyTorch and dependency compatibility with your proposed pinned ManiSkill/SAPIEN.
- Use **read-only** `python scripts/audit_environment.py` if working on the learner's Ubuntu host, and quote the output. Tool-less cloud browsing cannot inspect their local GPU: mark it unverified.
- Do not upgrade or replace existing ROS2/Isaac Sim/CUDA toolchains; use separate venv/Conda if user approves a minimal install.
- Before running even a smoke test, propose commands, source SHA, disk/GPU footprint and a bounded run (<~5 min for first smoke); only proceed if compatible and user permits local execution.

### 4. Pick a smallest feasible experimental goal; hold for learner decision
Propose starting with a simpler supported PickCube **official baseline** as an instrumentation/learning milestone, followed by PushCube contact analysis and PegInsertionSide. If the project's cited examples and verified official baselines suggest another order, explain evidence and request learner's choice. Do **not** start multi-hour PPO/SAC jobs, 3-seed training, modified rewards or FR3 migration in M0.

## M0 deliverables
1. `docs/M0_UPSTREAM_AUDIT.md`: problem/method/innovations/experiment claims, official framework features, upstream paper vs task claims and links.
2. `docs/M0_TASK_ALGORITHM_MATRIX.md`: task x PPO/SAC official status, obs/action controller, reward/success, code entrypoints; unsupported explicitly.
3. `docs/UPSTREAM_LOCK.md`: verified upstream commits/release and library/license/data/system constraints (unverified fields remain TBD).
4. `docs/M0_ALGORITHM_WALKTHROUGH.md`: readable state-action-learning loop, PPO and SAC formula-to-source mapping with location evidence, no black-box prose.
5. `docs/M0_LEARNING_CHECK.md`: at least three learner questions with answers hidden/separated so the learner can reason first. Include:
   - What actually updates the Actor/Critic and which env data feed these losses?
   - Why do action scaling / reward terms affect contact insertion and near-goal behavior?
   - What distinguishes an official checkpoint demo from true retraining and robust evaluation?
6. `docs/PROGRESS.md`: exact audited paths, current SHA, executed checks, failures, next bounded run.

## Initial acceptance gate
**M0 PASSED** only when:
- The researcher can explain task observations/actions/reward/done and PPO/SAC train/eval data flow in their own words;
- The task/algorithm feasibility matrix is backed by source evidence;
- Environment risk and cost are clear;
- Researcher approves the next specific smoke-test / baseline scope.

No algorithm reproduction / training runs are assumed complete at M0. If source version differs, update documentation with evidence before changing design.

## Exact expected report to the learner
- Key takeaways, in **simple Chinese**, no excessive theoretical prerequisites;
- Source SHA and 5–10 exact file/function references (or explicitly say not verified);
- PPO/SAC theory applied to this *actual* benchmark, not standalone textbook paragraphs;
- Suggested first small test with time, VRAM, install impact estimates;
- Three questions for learner's independent reasoning;
- Approved/not-approved boundary and status;
- What's blocked/unknown and how to resolve it.
