# M0 task × RL algorithm / interface matrix

Verified **in upstream tag v3.0.1** (a4a4f9272ad64b1564035874b605ceb687b63ed8); no host-side run. Source presence != successful retraining.

| Task | PPO official code/example | SAC official code/example | Robot default | Max steps | Distinctive observation / reward / success |
|---|---|---|---|---|---|
| PickCube-v1 | YES: ppo/examples.sh + ppo/baselines.sh (50M / 3 seeds in baselines.sh) | YES: sac/README.md (500k state example and RGBD PickCube sample) | panda | 50 | Extra state: tcp_pose, goal_pos, obj_pose and relative vectors; dense reach + grasp + transfer + static; success object at goal **and arm static** |
| PushCube-v1 | YES: ppo/README.md + baselines.sh (50M / 3 seeds) | YES: sac/README.md (500k state example) | panda | 50 | Extra state: tcp_pose, goal_pos, obj_pose; dense reach behind cube + gated placement + table-height; success cube near target and on table |
| PegInsertionSide-v1 | YES: ppo/examples.sh (250M demo command) AND ppo/baselines.sh (75M / 3 seeds accelerated config) | NO **documented tuned PegInsertionSide SAC example in pinned sac/README.md**; generic sac.py has env_id, but compat/performance UNVERIFIED — not an official Peg SAC baseline | panda_wristcam | 100 | Extra state: tcp_pose, peg_pose/size, hole_pose/radius; dense grasp + lateral prealignment + insertion; geometric insertion success |

State observations include agent proprioception composed by BaseEnv, not only the task-specific _get_obs_extra. Do not state an observation tensor dimension before runtime introspection. State input usually exposes privileged ground-truth object poses; RGB policy input differs and must be labeled.
Reward modes: sparse/dense/normalized_dense/none in BaseEnv; default training script logging config records normalized_dense; verify actual instantiated mode at runtime. Normalized reward: Pick /5, Push /4, Peg /10.
Task source: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/pick_cube.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/push_cube.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/tasks/tabletop/peg_insertion_side.py
Baseline source: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/README.md ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/examples.sh ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/ppo/baselines.sh ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/sac/README.md

## Controller / timing constraints
PPO ppo.py and SAC sac.py **default** control_mode=pd_joint_delta_pos but official SAC README examples explicitly set pd_ee_delta_pos, while PPO examples typically retain joint delta. These are different action semantics and not a fair algorithm-only comparison. Panda configuration: joint delta commanded range [-0.1,+0.1] rad per selected arm joint; EE delta position target [-0.1,+0.1] m/axis; controller PD/IK underneath. Actual normalized action/scaling and gripper dimension to be introspected in M1.
Core defaults SimConfig sim_freq=100 Hz, control_freq=20 Hz, hence each env.step=5 PhysX integration steps unless overridden.
Source: https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/agents/robots/panda/panda.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/utils/structs/types.py ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/mani_skill/envs/sapien_env.py

## Reset / done rules to verify
Pick cube XY and goal XYZ randomized in _initialize_episode; Push cube XY randomized ±0.1m and target shifted along x; Peg geometry sampled on scene reconfiguration and pose randomized on episode init.
BaseEnv.step: terminated from success or fail; raw truncated returned false; ManiSkillVectorEnv / Gym TimeLimit can supply truncation at horizon; do not equate success with timeout.
The RL wrapper's ignore_terminations/partial_reset, bootstrapping handling and eval reconfiguration frequency change sample validity. Need exact config log before a fair study.

Next bounded step (requires approval): isolated single-env CPU reset/step shape audit and optional small GPU 1/16-env smoke; zero training. Do not promise 3 task × 2 algorithms on 4070 without memory test.
