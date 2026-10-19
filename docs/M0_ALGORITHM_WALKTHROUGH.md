# M0: Algorithm-and-source walkthrough (source read, not executed)

Pin proposal: v3.0.1 / a4a4f9272ad64b1564035874b605ceb687b63ed8. Paths are relative to https://github.com/mani-skill/ManiSkill/tree/v3.0.1/

## End-to-end data flow
gym.make(env_id, obs_mode="state", control_mode=...) -> reset(seed) -> obs
-> policy network -> action (joint-delta or EE-delta; **not commanded force**) -> BaseEnv.step -> agent.set_action -> controller -> multiple PhysX steps -> info/evaluate -> observation/reward/terminated -> PPO rollout OR SAC replay storage -> gradient update -> evaluation.
Main anchors: mani_skill/envs/sapien_env.py [step ~1042, _step_action ~1073]; task files [evaluate, _get_obs_extra, compute_dense_reward]; mani_skill/agents/robots/panda/panda.py [_controller_configs ~76]; examples/baselines/{ppo/ppo.py,sac/sac.py}; mani_skill/utils/wrappers/ for vectorization.

## PPO (on-policy)
Source: examples/baselines/ppo/ppo.py
- class Agent lines 121–161: actor_mean MLP (3x256/Tanh) and trainable actor_logstd build Gaussian policy; critic MLP predicts V(s); get_action_and_value returns action, logprob, entropy, V(s).
- lines 247–253 store current rollout (obs, action, old logpi, reward, dones, V), reset ~258, env.step ~321, record post-timeout final_obs when available.
- lines 334–374 GAE backwards; delta_t = r_t + gamma * bootstrap_next_V - V(s_t), A_t = delta_t + gamma*lambda*(1-done)*A_(t+1), with special truncation bootstrap handling and an alternate finite-horizon branch. Returns=advantages+values.
- lines 389–415 update_epochs/minibatches, rho=exp(logpi_new-logpi_old); L_clip = -mean(min(rho*A, clip(rho,1-eps,1+eps)*A)). Signs in implementation are max of the two negatives.
- lines 417–438 value MSE (sometimes clipped), entropy regularization, backward/Adam and grad clipping. Actor+Critic updated from a fresh finite rollout; no replay of old-policy trajectories across collection rounds.
- Important: rollout clips actions to env Box before actuation; logprob is obtained on sampled pre-clip action. Record action saturation for diagnosis.

## SAC (off-policy)
Source: examples/baselines/sac/sac.py
- class ReplayBuffer 136–184: (obs, next_obs, action, reward, done) stored by environment/time and randomly sampled.
- SoftQNetwork 187–202 accepts concat(s,a), outputs Q; two online and two target critics instantiated ~343–354.
- Actor 209–255: mean/logstd MLP, Normal.rsample for reparameterization, tanh + rescale to action bounds; adjusted logprob accounts for tanh Jacobian.
- lines 432–467: random initial action until learning starts, then sample policy action, step and store real next_obs and bootstrap-stop flags.
- lines 478–499: sample replay; target y=r+gamma*(1-stop_bootstrap)*(min(Q1_target(s',a'),Q2_target(s',a'))-alpha*logpi(a'|s')); regress both online Q values to target.
- lines 501–511: actor loss E[alpha*logpi(a|s)-min(Q1(s,a),Q2(s,a))]; reparameterization transmits gradient through the action into actor.
- lines 513–525: alpha entropy autotune on **detached log_pi** (sample under torch.no_grad at 514); optimize log_alpha. lines 527–532: Polyak target update Qtarget=tau Qonline+(1-tau)Qtarget.
- No PPO-style advantage A(s,a) is required because sampled Q(s,a) directly evaluates chosen actions; Q is learned, NOT a fixed environment score.

## Worked robot action semantics
Panda controller configs in mani_skill/agents/robots/panda/panda.py:
- pd_joint_delta_pos -> normalized arm action maps to joint-position target increments with ±0.1 rad bounds; arm is position-controlled by lower-level PD.
- pd_ee_delta_pos -> normalized translation action maps to EE target increment with ±0.1m/axis bounds; IK resolves target to joints, PD drives joints.
- pd_ee_delta_pose includes rotational increments. Existing upstream issue #1469 reports rotation-direction inversion for this controller in v3.0.1; first validate controller realization before insertion work.
These are not raw robot torque/force policy outputs. gripper control coexists with arm controller. Exact full action vector and observation dimensions must be read after environment creation.

## Failure hypotheses and student responsibility
1. Reward hacking: near-goal distance reward improves while actual success remains low; inspect reward components and held-out success.
2. Action saturation/scaling: too large control increments produce poor near-contact precision; log saturation and net displacement instead of blaming actor immediately.
3. PPO horizon/truncation: incorrect bootstrapping causes bad value estimates; inspect timeout, final_observation and done handling.
4. SAC actor/critic/entropy: diverging Q, badly tuned alpha, replay biases; inspect Q losses, alpha, action histogram, policy grad and replay sampling.
5. Fair comparison: official PPO joint-delta vs official SAC EE-delta changes controller, cannot isolate PPO-vs-SAC.
No training performed; all effects are hypotheses to be checked with original logs later.
