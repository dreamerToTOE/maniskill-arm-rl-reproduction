# M2 Stage B — 1024-env PPO pilot (reference-scale width on an 8 GiB card)

> **Status: `PASS` — 5/5 iterations, exit 0, 26.24 s, loop SPS 15117, VRAM still flat.**
> Explicitly authorized by the researcher (2026-10-22). **This exceeds the project's
> documented ceiling of 512 envs** (the A1 hard cap; 1024 was "refused outright" in the
> A1 probe's `validate_env_counts`). The user is the authority here, and the safety net
> is the same: the supervisor's 1536 MiB VRAM floor and OOM handling "kill if the GPU
> can't take it" — it took it.
>
> Date: 2026-10-22 11:57 KST · Base commit: `edcdd0e` · Classification: **ADAPTATION / PLUMBING_ONLY**
> Ran in the agent session (`verdict: OK`, 6745 MiB free at start).
> Upstream blob `66802de0…`, `render_mode="rgb_array"` hardcoded and untouched.

## 1. It fits — and so does the memory question

| | 128 | 256 | 512 | **1024** |
|---|---:|---:|---:|---:|
| **Peak device VRAM** | 5137 | 4601 | 4777 | **4862 MiB** |
| **Min free VRAM** | 2682 | 3217 | 3041 | **2956 MiB** |
| OOM / floor trip | no | no | no | **no** |

**The GPU memory does not scale with env count in this range.** Peak stays ~4.6–5.1 GiB
from 128 all the way to 1024 parallel PickCube envs (state obs, 8-dim action), rendering
included. The scene and the `rgb_array` pipeline are fixed costs; SAPIEN evidently reuses
render/buffer allocations rather than multiplying them per env. 1024 envs left 2956 MiB
free — still ~2× the 1536 MiB abort floor.

This **reverses the standing assumption** from the M2 source review ("the 8 GiB bottleneck
is parallel PhysX environments"). For state-observation PickCube, env count is **not** the
VRAM driver; it is the throughput lever.

## 2. Throughput — 5.8× over 128, still ~0.92 marginal efficiency

| | 128 | 256 | 512 | **1024** |
|---|---:|---:|---:|---:|
| Loop SPS (last) | 2600 | 4435 | 8236 | **15117** |
| Rollout fps (mean) | 3978 | 6449 | 11456 | **20791** |
| Rollout time / iter | 1.617 s | 2.004 s | 2.249 s | **2.474 s** |
| Update time / iter | 0.444 s | 0.477 s | 0.463 s | 0.468 s |
| End-to-end SPS | 2400.4 | 2885.8 | 6687.6 | **9756.1** |
| Wall (256k steps) | 106.65 s | 88.71 s | 38.28 s | **26.24 s** |

- Speedup vs 128: 15117/2600 = **5.81×** for 8× envs → overall efficiency **0.73**.
- The 512→1024 doubling is **1.84×** (efficiency **0.92**) — scaling barely decays even at
  this width.
- Rollout time 2.25 → 2.47 s (+10 %) for double the env-steps, the same signature as every
  width below it.
- **Caveat, sharpened**: only **5 iterations** (256k / (1024×50) = 5) and SPS was still
  climbing (10411 → 13229 → 14318 → 14841 → **15117**) when it stopped. The last value is
  the conservative figure; steady state could be a bit higher, but 5 iterations is too thin
  to claim it.

## 3. Learning metrics (sanity only)

5 iterations, 1 evaluation: policy −0.0316, value 0.0027, KL 0.0301, explained variance
→ +0.585, `eval/success_once` 0.0. Finite and sane; nothing about PPO is drawable from
0.51 % of the official budget — as with every pilot.

## 4. Projection for the full 50M run

Using measured loop SPS (last):

| Env count | loop SPS | 50M / seed | 3 seeds |
|---:|---:|---:|---:|
| 128 | 2600 | 5.34 h | 16.0 h |
| 256 | 4435 | 3.13 h | 9.4 h |
| 512 | 8236 | 1.69 h | ~5 h |
| **1024** | **15117** | **~0.92 h** | **~2.8 h** |

**At 1024 envs a 3-seed PickCube run is ~2.8 hours.** The remaining uncertainty is the
5-iteration sample (thermal/steady-state over hours), not memory (2.9 GiB headroom).

## 5. What this opens, and what it does not

- **Opened (unproven, needs a test):** 2048 and 4096 envs *might also fit*, given the flat
  VRAM trend. 4096 is the official baseline width — if it fits, the official `ppo_fast.py`
  configuration becomes locally testable. **This is an extrapolation, not a measurement.**
- **Not licensed:** any claim that 4096 fits, any baseline reproduction, any convergence,
  or any comparison to published numbers. All four runs are plumbing probes at 0.51 % of
  the official budget.
- **Boundary note:** 1024 (and any larger width) is outside the project's written 512 cap;
  it is recorded here because the researcher explicitly authorized it, and the cap should
  be updated if larger widths are to become routine.

## 6. Caveats

- 5 iterations, 4 VRAM samples, one seed, SPS still rising — the thinnest window of the
  four pilots; treat 15117 as an order-of-magnitude, not a point estimate.
- Peak `memory.used` absolutes are confounded by session-to-session desktop baseline;
  compare deltas and free-memory floors, not absolutes.
- `render_mode="rgb_array"` (upstream hardcode) is on in all runs, so they are comparable.

## 7. Artifacts

| Path | Content |
|---|---|
| `results/m2_pilot_n1024_summary.json` | full metrics (committed) |
| `results/raw/m2_stage_b_n1024_pilot.log` | child stdout (gitignored) |
| `runs/PickCube-v1__ppo__9351__stageB_n1024/` | 2 checkpoints + events (gitignored) |
