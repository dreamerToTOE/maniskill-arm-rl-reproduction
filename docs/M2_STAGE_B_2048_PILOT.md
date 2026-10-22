# M2 Stage B — 2048-env PPO pilot (half of the official width, still fits)

> **Status: `PASS` — 2/2 iterations, exit 0, 20.18 s, loop SPS 22475, VRAM still flat.**
> Explicitly authorized by the researcher (2026-10-22). This is now **half the official
> baseline width** (4096 envs) and far beyond the project's 512 hard cap. Same safety net:
> the supervisor's 1536 MiB floor and OOM handling — "kill if the GPU can't take it".
> It took it.
>
> Date: 2026-10-22 12:14 KST · Base commit: `d83fe95` · Classification: **ADAPTATION / PLUMBING_ONLY**
> Ran in the agent session (`verdict: OK`, 6761 MiB free at start, 40 GB RAM available,
> `ulimit -v` unlimited). Upstream blob `66802de0…`, `render_mode="rgb_array"` untouched.

## 1. It fits — the flat-VRAM trend holds through 2048

| | 128 | 256 | 512 | 1024 | **2048** |
|---|---:|---:|---:|---:|---:|
| **Peak device VRAM** | 5137 | 4601 | 4777 | 4862 | **5148 MiB** |
| **Min free VRAM** | 2682 | 3217 | 3041 | 2956 | **2670 MiB** |
| OOM / floor trip | no | no | no | no | **no** |

Peak VRAM is essentially flat (4.6–5.1 GiB) across **a 16× range** of env counts. The GPU
PhysX scene and the `rgb_array` pipeline are fixed costs; env count is a throughput lever,
not a memory driver — now established over 128→2048, not just extrapolated.

## 2. Throughput — scaling is now decaying, but still paying

| | 128 | 256 | 512 | 1024 | **2048** |
|---|---:|---:|---:|---:|---:|
| Loop SPS (last) | 2600 | 4435 | 8236 | 15117 | **22475** |
| Rollout fps (mean) | 3978 | 6449 | 11456 | 20791 | **35788** |
| Rollout time / iter | 1.617 | 2.004 | 2.249 | 2.474 | **2.863 s** |
| Update time / iter | 0.444 | 0.477 | 0.463 | 0.468 | 0.489 s |
| End-to-end SPS | 2400.4 | 2885.8 | 6687.6 | 9756.1 | **10148.7** |
| Wall (256k steps) | 106.65 s | 88.71 s | 38.28 s | 26.24 s | **20.18 s** |

- Speedup vs 128: 22475/2600 = **8.64×** for 16× envs → overall efficiency **0.54**.
- The marginal doubling 1024→2048 is **1.49×** (efficiency **0.74**) — the first clear
  decline from 0.92 (512→1024). Diminishing returns have set in, but a 1.49× doubling is
  still a good trade.
- **Caveat, sharpened again**: only **2 iterations** and **2 VRAM samples**. SPS was still
  climbing (17930 → 22475). This is a fit-check plus an order-of-magnitude throughput
  read, nothing more.

## 3. Projection for the full 50M run

| Env count | loop SPS | 50M / seed | 3 seeds |
|---:|---:|---:|---:|
| 128 | 2600 | 5.34 h | 16.0 h |
| 256 | 4435 | 3.13 h | 9.4 h |
| 512 | 8236 | 1.69 h | ~5 h |
| 1024 | 15117 | 0.92 h | ~2.8 h |
| **2048** | **22475** | **~0.62 h** | **~1.85 h** |

At 2048 envs a 3-seed PickCube run is **~1.9 hours** — by a 2-iteration estimate.

## 4. What this means for the reproduction

- **2048 envs = half the official 4096 width, and it fits in 8 GiB with 2.7 GiB headroom.**
  The earlier "4096 won't fit in 8 GiB" belief is now *empirically dubious* for state-obs
  PickCube: the trend is flat, and the next doubling would land ~5.3–5.5 GiB **if** it
  holds. Whether 4096 specifically fits is **still an extrapolation**, not a measurement.
- If 4096 fits, the official `ppo_fast.py` configuration becomes locally testable — a
  materially different situation for the whole reproduction plan. That test (4096) is the
  obvious next fit-check, **not started, not authorized**.
- Throughput gains are now diminishing (0.74 marginal), so the practical return on 4096
  over 2048 is roughly 1.5×, not 2×.

## 5. Caveats

- 2 iterations / 2 VRAM samples / one seed / SPS still climbing — the thinnest window of
  the series; treat 22475 as an order-of-magnitude.
- Peak `memory.used` absolutes are confounded by session-to-session desktop baseline;
  compare free-memory floors, not absolutes.
- 2048 (and 1024 before it) are outside the project's written 512 cap; recorded because
  the researcher explicitly authorized them.

## 6. Artifacts

| Path | Content |
|---|---|
| `results/m2_pilot_n2048_summary.json` | full metrics (committed) |
| `results/raw/m2_stage_b_n2048_pilot.log` | child stdout (gitignored) |
| `runs/PickCube-v1__ppo__9351__stageB_n2048/` | 2 checkpoints + events (gitignored) |
