# M1 researcher learning review — 2026-10-20

**Status: PARTIAL_PASS / CORRECTION_REQUIRED; M1 NOT PASSED.** CPU smoke and interface measurements passed separately; PPO/SAC training NOT_RUN; GPU benchmark NOT_RUN.

## Verbatim first researcher answers

1. `关节目标增量是0.05，等于实际关节位移。`
2. `在真机上不一定能完整的获得obj——pose，有观测误差等。`
3. `reward是由多项叠加的，而成功的要求是cube到位（一定阈值内）+机械臂停下来。二者不具有等价关系`

## Critical grading, evidence and corrections

| Item | Grade | Explanation |
|---|---|---|
| Q1 normalized arm action =0.5 | **PARTIAL / NOT PASSED** | Correct: `0.1 * 0.5 = +0.05 rad` **target delta**. Incorrect: target delta does **not** necessarily equal actual joint displacement over five simulation substeps. `q_target=q_current + delta` then PD/PhysX determines `q_actual`. Recheck needed. |
| Q2 `obj_pose` sim vs real | **BASIC PASS** | In simulation `obj_pose` is privileged ground-truth state from physics engine, not camera observation. In real robot must estimate from sensors/vision + calibration/coordinate transformations; occlusion/noise/delay are possible. Researcher mentioned observation inaccuracy correctly but should identify source and perception. |
| Q3 reward vs success | **PASS** | Dense reward has reaching/grasp/placing/static terms with gates; success requires cube-goal <=0.025 m and Panda arm per-joint speed absolute max <=0.2 rad/s. `is_grasped` not part of success criterion, so high reward does not imply success. Note static is a threshold, not exact zero velocity. |

### Targeted correction/recheck for Q1 (only one required)

**Given** initial joint angle `q0=0.40 rad`, normalized first-joint action `a1=0.5`, and **hypothetical** measured joint angle after five physics steps `q1_actual=0.43 rad`:
- What target did the controller command during this control step?
- What was the **actual displacement** and what is the target-tracking error?
- If on the next control step the policy again outputs `a1=0.5`, what is the new joint target given `use_delta=True,use_target=False`?

Researcher must independently compute; don't mark passed solely for reading the tutor solution. Expected internally for reviewer: 0.45 rad target; +0.03 rad actual displacement; +0.02 rad residual error; second-step target 0.48 rad (assuming q1 actual 0.43 rad). Do not copy these numbers into the researcher's answers.

## Gate state

`M1_TEACHING_MATERIAL=DELIVERED`; `M1_CPU_SMOKE=PASSED`; `M1_THEORY=RECHECK_PENDING`; `M1_OVERALL=NOT_PASSED`; `GPU_LOCAL_USABILITY=NOT_VERIFIED`; `TRAINING=NOT_RUN`.

Source: `mani_skill/agents/robots/panda/panda.py::_controller_configs` and `mani_skill/agents/controllers/pd_joint_pos.py::set_action`, official v3.0.1 source pin `a4a4f9272ad64b1564035874b605ceb687b63ed8`; `docs/M1_EXECUTION_REPORT.md` for local smoke data.
