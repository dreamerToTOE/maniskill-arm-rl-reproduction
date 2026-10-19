# M0 learning and interview checkpoint

Status: PENDING_RESEARCHER_ANSWERS, **not passed**. Researcher should answer *before* reading solution hints in a separate review session.

## Questions (answer in your own words)
1. ManiSkill / SAPIEN / PPO (or SAC) each performs what role? Walk through obs -> action -> controller -> physics -> reward.
2. Why is a policy producing pd_ee_delta_pos not equivalent to directly outputting force/torque? What will happen when an action is normalized then limited by the controller?
3. In PPO, what does the Actor output and what does the Critic output? What trajectories are used to compute GAE and the clipped loss?
4. In SAC, what five items are sampled from ReplayBuffer? Why do we need two Qs, why rsample(), and how is alpha updated?
5. Official PPO uses joint delta while listed SAC examples use EE delta. Why is a direct win/loss comparison scientifically unfair?
6. In PegInsertionSide, why can the distance/reward improve while safe contact manipulation remains unsolved? What extra metrics are needed?
7. How do you tell apart a downloaded checkpoint demo, an unchanged official baseline retraining, paper figure reproduction, and a new ablation?

## Review-only answer rubric (not auto-marked)
Pass when researcher independently identifies: policy/controller/physics separation; grounded source functions; PPO on-policy GAE and likelihood ratio; SAC off-policy twin Q and entropy and target network; controller/action confounds; success versus shaped reward; reproducible provenance and statistical evidence. Mistakes must be described specifically and discussed before opening M1.

## Current status
No responses, no approval, no environment run. Next: reviewer asks Q1–Q3 interactively, checks responses, then Q4–Q7. Do not prematurely mark M0 PASSED.
