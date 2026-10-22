# M2 — Full 50M/3-seed PickCube PPO reproduction (ADAPTATION)

> **Status: `PASS` — 3/3 seeds converged, final eval success 1.0, mean eval success ~0.78.**
> This is the first "official-budget" result of the reproduction: 3 seeds × 50,000,000
> env steps of PPO on the pinned upstream `ppo.py`, at 2048 parallel envs, on the local
> RTX 3070 (8 GiB), in the researcher's terminal (the agent session has no GPU).
>
> Date: 2026-10-22 16:37–18:24 KST · Base commit: `ceef64a`
> Classification: **ADAPTATION / PLUMBING + CONVERGED** — `ppo.py` (readable) @ 2048 envs
> (not the official `ppo_fast.py` @ 4096 envs / 4 steps / cudagraphs), but same task,
> same 50M budget, same seed set.

## 1. Headline

| Seed | Wall | e2e SPS | loop SPS | peak VRAM | min free | eval success (mean) | final eval | steps → 0.5 success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9351 | 35.2 min | 23,669 | 23,808 | 6022 MiB | 1797 MiB | 0.7625 | **1.0** | 10.24 M |
| 4796 | 35.5 min | 23,461 | 23,622 | 5819 MiB | 2000 MiB | 0.8000 | **1.0** | 10.24 M |
| 1788 | 36.8 min | 22,611 | 22,755 | 5873 MiB | 1945 MiB | 0.7875 | **1.0** | 10.24 M |
| **mean** | **35.8 min** | **23,247** | **23,395** | **5905 MiB** | **1914 MiB** | **0.783** | **1.0** | **10.24 M** |

All three seeds: 488 iterations, 49,971,200 training env steps + 8,000 evaluation env
steps (20 evals × 400), 21 checkpoint files each, exit 0, no OOM, no NaN, no kill.
Total wall clock **≈ 1.79 h** for the three-seed run.

## 2. Did it converge? — Yes, cleanly

| Seed | success at 2.56M | 5.12M | 7.68M | 10.24M | 12.8M | 15.36M+ |
|---:|---:|---:|---:|---:|---:|---:|
| 9351 | 0.0 | 0.125 | 0.25 | 0.5 | 0.875 | 1.0 (then ~1.0, dips to 0.75-0.875 near end) |
| 4796 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 (flat) |
| 1788 | 0.0 | 0.125 | 0.125 | 0.5 | 1.0 | 1.0 (flat) |

- **Steps to 0.5 success ≈ 10.24 M** on all three seeds (≈ 20 % of the budget) — a
  consistent, reproducible sample-efficiency signature.
- **Steps to 1.0 success**: 10.24 M (seed 4796) to 15.36 M (seed 9351).
- The critic converged hard: explained variance → **0.968–0.982**, value loss ~0.01.
- Final evaluation = **100 % success** on all three seeds; mean over the 20 evaluations
  is **0.78** (the final evals were on the highest-quality checkpoints).

This matches the official ManiSkill PickCube state-PPO result (~0.8 success rate).
It is **not** a claim of beating or exactly matching `ppo_fast.py` — it is the readable
script at a different width — but the task is solved from scratch in all three seeds.

## 3. Throughput / resource profile (measured, not estimated)

- **End-to-end SPS 22.6k–23.7k** — higher than the 2048-env pilot (15.1k loop) because
  that pilot ran only 2 cold iterations; the steady-state loop SPS is ~23.4k.
- **Peak VRAM 5.8–6.0 GiB** of 8 GiB, min free 1.8–2.0 GiB — the 1536 MiB abort floor
  was never approached. The `rgb_array` render + 2048 envs is comfortably within budget.
- 50M steps in **~35 min/seed**, ~1.79 h for the 3-seed volume, on an 8 GiB card.

## 4. Deviations from the official baseline (all deliberate, pre-registered)

| Deviation | Official | Here | Why |
|---|---|---|---|
| Script | `ppo_fast.py` | `ppo.py` | no `tensordict`/`wandb`; readable baseline |
| Env count | 4096 | **2048** | 8 GiB; A1/256/512/1024/2048 pilots showed 2048 fits with 2 GiB headroom |
| Steps/rollout | 4 | 50 | `ppo.py` default |
| Update epochs | 8 | 4 | `ppo.py` default |
| CUDA graphs | yes | no | `ppo_fast.py` feature |
| Seeds | 9351/4796/1788 | same | unchanged |

Nothing in the algorithm, optimizer, GAE, reward, or network was modified.

## 5. What is licensed / not licensed

**Licensed:** the full PPO loop runs end-to-end at official budget and converges to
~0.8 mean / 1.0 final success on all three seeds, at measured throughput and VRAM.

**Not licensed:**
- ❌ "official baseline reproduction" — that is `ppo_fast.py` @ 4096 envs / 4 steps /
  8 epochs / cudagraphs, still not run;
- ❌ comparison with published numbers beyond "consistent with ~0.8";
- ❌ any sample-efficiency claim beyond the measured steps-to-success.

## 6. Artifacts

| Path | Content |
|---|---|
| `results/m2_full_n2048_s{9351,4796,1788}.json` | per-seed full metrics (committed) |
| `results/m2_full_eval_curves.json` | 20-point success/return curves per seed (committed) |
| `runs/PickCube-v1__ppo__{seed}__n2048_50M/` | 21 checkpoints + events per seed (gitignored) |
| `results/raw/m2_full_n2048_s{seed}.log` | child stdout (gitignored) |

## 7. Next step

Optionally: (a) run the official `ppo_fast.py` @ 4096 envs for a literal baseline
reproduction (needs `tensordict`+`wandb`, 4096 fit is still unproven but likely), or
(b) move to PushCube / PegInsertionSide, or (c) the PPO-vs-SAC comparison. None started;
each needs its own approval.
