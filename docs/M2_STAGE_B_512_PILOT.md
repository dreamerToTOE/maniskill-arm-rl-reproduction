# M2 Stage B — 512-env PPO pilot (the ceiling: does it fit, what does it buy?)

> **Status: `PASS` — 10/10 iterations, exit 0, 38.28 s, loop SPS 8236, VRAM flat.**
> Explicitly authorized by the researcher (2026-10-22); this exceeds the 256-env ceiling
> used through Stage B, and 512 is the hard cap (`HARD_MAX_ENVS`) the A1 probe enforces.
> Same pinned upstream `ppo.py`, only `num_envs` 256 → 512, same 256k-step budget.
>
> Date: 2026-10-22 11:47 KST · Base commit: `9702b75` · Classification: **ADAPTATION / PLUMBING_ONLY**
> Ran in the agent session (`verdict: OK`, 6704 MiB free at start, only
> `gnome-remote-desktop-daemon` at 157 MiB resident).

## 1. It fits — no kill was needed

The request was "test 512 envs; kill it if the GPU can't take it". The GPU took it:

| | 128 | 256 | 512 |
|---|---:|---:|---:|
| **Peak device VRAM** | 5137 MiB | 4601 MiB | **4777 MiB** |
| **Min free VRAM** | 2682 MiB | 3217 MiB | **3041 MiB** |
| OOM / floor trip | no | no | **no** |

Peak used stayed ~4.6–5.1 GiB across 128→256→512 (the absolute differences are
dominated by the desktop baseline drifting between sessions, not by env count). The
scene + `rgb_array` render cost is **env-count-flat even at 512** — the memory wall the
audit worried about does not exist in this range. 512 envs left 3041 MiB free, 2× the
1536 MiB abort floor.

## 2. Throughput — still scaling, ~79 % efficient over 128

| | 128 | 256 | 512 |
|---|---:|---:|---:|
| Loop SPS (last) | 2600 | 4435 | **8236** |
| Loop SPS (mean) | 2575 | — | 7848 |
| Rollout fps (mean) | 3978 | 6449 | **11456** |
| Rollout time / iter | 1.617 s | 2.004 s | **2.249 s** |
| Update time / iter | 0.444 s | 0.477 s | 0.463 s |
| End-to-end SPS | 2400.4 | 2885.8 | **6687.6** |
| Wall (256k steps) | 106.65 s | 88.71 s | **38.28 s** |

- Speedup vs 128: 8236/2600 = **3.17×** for 4× the envs → **efficiency 0.79**.
- The marginal doubling 256→512 is **1.86×** (efficiency 0.93), actually *better* than
  128→256 (0.85) — the fixed per-step overhead amortises more as width grows.
- Rollout time only 2.00 → 2.25 s (+12 %) for twice the env-steps: same "2× work for
  1.12× wall" signature seen at every width.
- **Caveat**: only 10 iterations (256k / (512×50) = 10), and the SPS was still rising
  when it ended (5703 → 7374 → … → 8236). The last value is the conservative figure;
  a longer run could settle a little higher.

## 3. Learning metrics (sanity only)

10 iterations, 2 evaluations: policy −0.0114, value 0.0057, KL 0.0254, clipfrac 0.309,
explained variance → +0.606, `eval/success_once` 0.0. Finite and sane; at 0.51 % of the
official budget, no conclusion about PPO is drawable — same as every other pilot.

## 4. Projection for the full 50M run

Using measured loop SPS, three ways:

| Env count | loop SPS | 50M / seed | 3 seeds |
|---:|---:|---:|---:|
| 128 | 2600 | 5.34 h | 16.0 h |
| 256 | 4435 | 3.13 h | 9.4 h |
| **512** | **8236 (last) / 7848 (mean)** | **1.69–1.77 h** | **~5.1–5.3 h** |

**512 envs brings a 3-seed PickCube run to ~5 hours locally.** The remaining uncertainty
is not VRAM (3.0 GiB headroom) but whether the 10-iteration throughput holds over hours
(thermal, allocator) — the only way to know is the run itself.

## 5. Caveats and what "512" is not

- 512 is the A1 **hard cap**; the official baseline is **4096 envs**, which is a
  different (cloud-scale) regime and does not fit in 8 GiB. 512 is still an
  ADAPTATION, not the official configuration.
- One run, one seed, 10 iterations, 5 s-sampled VRAM — short window, ±3 % timing noise,
  no confidence interval, spikes between VRAM samples invisible.
- `render_mode="rgb_array"` (upstream hardcode) is on in all three runs, so they are
  comparable to each other.
- Peak `memory.used` absolute values are confounded by session-to-session desktop
  baseline; compare deltas and free-memory floors, not absolutes.

## 6. Artifacts

| Path | Content |
|---|---|
| `results/m2_pilot_n512_summary.json` | full metrics (committed) |
| `results/raw/m2_stage_b_n512_pilot.log` | child stdout (gitignored) |
| `runs/PickCube-v1__ppo__9351__stageB_n512/` | 3 checkpoints + events (gitignored) |
