# M0 learning and interview checkpoint

Status: FIRST_ATTEMPT_REVIEWED / CORRECTION_REQUIRED, **not passed**. Researcher should answer *before* reading solution hints in a separate review session.

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
First independent response reviewed on 2026-10-19; corrections below. No environment audit or training run approved. Do not mark M0 PASSED.


## First learner attempt and mentor review — 2026-10-19

**Q1 — ManiSkill/agent/control:** Researcher explained ManiSkill as providing RL platform/feedback; policies can output joint or EE increments and in principle lower-level actuation. **PASS with qualifications**: Observation is not all environment parameters; SAPIEN does physics; torque commands are not equivalent to raw motor voltage/PWM.

**Q2 — PPO Actor/Critic:** Researcher said Actor *learns A (state value change)* and Critic *learns difference between target and realized value*. **CORE MISCONCEPTION / NOT PASSED**. Actor learns a policy distribution pi_theta(a|s); Critic predicts V_phi(s). Advantage A^pi=Q^pi(s,a)-V^pi(s) is an estimated relative action quality, not raw delta V. GAE uses rewards plus bootstrapped state values; critic loss regresses V against return targets, its prediction error is a training signal, not output.

**Q3 — SAC:** Researcher correctly described replay of historical data and two Q estimates selecting a minimum, but did not understand lack of PPO GAE. **PARTIAL / NOT PASSED**. Replay stores (s,a,r,s_next,done/bootstrapping mask). Both Q critics are trained; min primarily reduces overestimation. SAC learns action-conditioned Q(s,a) and optimizes alpha*log pi(a|s)-min(Q1,Q2), gradients pass through rsample/reparameterized actions, so PPO's sampled-trajectory GAE is not needed. Still to explain entropy autotuning and slowly updated target critics.

**Safety/provenance:** These are evaluations of learner-written conceptual answers, not measured RL results. Training **NOT_RUN**, host **NOT_AUDITED**, source tag v3.0.1 commit a4a4f9272ad64b1564035874b605ceb687b63ed8.

### Focused recheck (researcher must answer independently)
1. In PPO, which function does Actor approximate, which does Critic approximate, and how does positive/negative advantage change action probability? Can a Critic directly provide the next action?
2. Given V(s)=5, r=2, V(s')=4, gamma=0.9 and a non-terminal step, compute one-step TD delta and value target. Why is delta not merely V(s')-V(s)?
3. In SAC, list the five replay entries, explain both critics' target and actor objective, and why this objective uses Q directly without PPO's GAE; what does alpha do?

**Gate:** Await answers and review. Do not proceed to training. M0 source review complete, learner gate pending.
