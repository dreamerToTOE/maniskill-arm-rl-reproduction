# Resume Evidence / 简历证据清单

Project: ManiSkill manipulation RL

**All results remain NOT_STARTED at project creation.** Do not report upstream demo performance as personal results.

## Intended experience / 必须实证后才能写入简历
- Designed and validated [specific task and simulator pinned versions] with [verified observations/actions/rewards].
- Reproduced [actual algorithm] for [seed count] seeds, [env steps] training interactions, [eval episodes] held-out evaluations.
- Diagnosed [measured failure case] and implemented [clearly authored solution], yielding [measured improvement + statistical uncertainty].
- Validated robustness across [unseen inputs, friction, model or contact parameters]; differentiate simulated vs real.

## Evidence gates
- [ ] Source tag + SHA + license and dependencies captured in UPSTREAM_LOCK.md.
- [ ] Independently executable train and evaluate commands.
- [ ] Code diff showing personal contribution rather than copying upstream.
- [ ] Evaluation results (per-seed CSV/JSON, curves, videos) tracked and reproducible.
- [ ] Negative tests, ablation and fair control baseline.
- [ ] Honest failure and missing-scope notes.
- [ ] 3-minute and 10-minute project interview narratives.

Proposed benchmark: PickCube / PushCube / PegInsertionSide; PPO/SAC where actually available.
Primary metrics: task success, sample efficiency, action saturation, force/contact proxies.

## Interview checks
What problem is solved? Which calculations are classical control vs learned? What are the exact simulator and policy action rates? How was robustness measured? Which part is yours? What cannot yet be claimed?
