# Interview Skills from This Reproduction / 项目关联面试技术栈

**Focus:** 机械臂 RL / Manipulation
**Target job families:** 机械臂操作、仿真 RL 训练、装配/接触控制算法实习

## Core required skills
- PPO clipped loss and GAE
- SAC twin-Q / replay / entropy
- action scaling, reward shaping and contact-heavy success definition
- baseline fairness
- PyTorch
- observations / reward bugs
- sparse vs dense rewards
- generalization

## Portfolio / interview drill
Inspect one task's reward function and reset distribution; reproduce a 3-seed PPO curve; explain why a reach policy can saturate near the goal; compare one equal-budget PPO/SAC experiment with held-out evaluation.

## Evidence policy
- Knowledge != implemented code != verified experiment. Demonstrate the exact level using code links and logs.
- Prepare one 3-minute overview and one 15-minute whiteboard walkthrough.
- Be ready to answer: Why this algorithm? Which baselines? What are training and evaluation metrics? How do you debug failures? What is your independent contribution?
- If this remains a numerical simulation, **do not** advertise physical robot experience, Sim2Real transfer or field deployment.

## Links
See [plan](REPRODUCTION_PLAN.md), [experiment matrix](EXPERIMENT_MATRIX.md), [resume evidence](RESUME_EVIDENCE.md).
