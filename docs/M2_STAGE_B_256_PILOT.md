# M2 Stage B — 256-env PPO pilot (throughput + VRAM at double width)

> **Status: `PASS` — 20/20 iterations, exit 0, 88.71 s.**
> Same pinned upstream `ppo.py`, only `num_envs` 128 → 256. Answers the one open
> question Stage B left: *does doubling the env count pay for itself, and does it fit
> in 8 GiB now that the training path renders?*
>
> Date: 2026-10-22 11:22 KST · Base commit: `4cf7ab0` · Classification: **ADAPTATION / PLUMBING_ONLY**
> Executed **in the agent session** — GPU visibility returned with the new session
> (`verdict: OK`, free 6534 MiB, `torch.cuda.is_available()=True`, device_count 1).
> Upstream blob `66802de0…`, `render_mode="rgb_array"` hardcoded and untouched.

## 1. The one-number answer

| | 128 envs (Stage B) | 256 envs (this pilot) | Δ |
|---|---:|---:|---:|
| End-to-end SPS | 2400.4 | **2885.8** | +20 % |
| Loop SPS (`charts/SPS`) | 2600 | **4435** | +71 % |
| Rollout fps | 3978 | **6449** | +62 % |
| Rollout time / iter | 1.617 s | 2.004 s | +24 % (for 2× the work) |
| Update time / iter | 0.444 s | 0.477 s | +7 % |
| Peak device VRAM (sampled) | 5137 MiB | **4601 MiB** | −536 MiB* |
| Min free VRAM | 2682 MiB | **3217 MiB** | +535 MiB |
| Wall clock (256k steps) | 106.65 s | **88.71 s** | −17 % |
| Iterations / checkpoints | 40 / 9 | 20 / **5** (4 periodic + final) | halved |
| Eval success rate | 0.0 | 0.0 | — |

\* The absolute `peak used` is **not** directly comparable across the two runs: the
desktop baseline differed (idle used ≈1732 MiB at 21:08 vs ≈1284 MiB at 11:22). The
incremental scene+render cost is ~3.3–3.4 GiB in both — i.e. it **does not scale with
env count** over this range, consistent with A1 (which saw flat VRAM up to 256 without
rendering). What is unambiguous: **256 envs fits comfortably, min free never left
3217 MiB, and the 1536 MiB floor was never approached.**

## 2. Throughput — the doubling pays ~71 % in the loop

- Scaling efficiency from 128 → 256 is **4435 / (2 × 2600) = 0.85** — almost identical
  to A1's simulator-only efficiency of 0.81 at 256. The training loop's fixed per-step
  cost behaves like the simulator's; doubling width keeps ~85 % of the linear gain.
- Rollout fps 3978 → 6449 (+62 %) and rollout_time only 1.62 → 2.00 s (+24 %) for
  double the env-steps — same "2× work for 1.24× wall" signature seen in A1.
- The **end-to-end** gain is only +20 % because the 256 run carried ~31 s of overhead
  (≈25 s first-time build of a 256-env rendered scene + 4 evals + import) against
  88.7 s total, whereas the 128 run had already amortised its build (8.2 s). The steady
  state is the loop figure, **4435 SPS**.

## 3. VRAM — double width, no memory wall

Peak sampled **4601 MiB used / 3217 MiB free** on the actual training path (256 train
envs + 8 eval envs, `rgb_array`). This closes the only real risk left open after Stage B:
256 envs with rendering is **not** a VRAM problem on this card. (The audit warning about
`render_mode` hardening still stands conceptually, but the increment turned out
env-count-flat, so it does not bite at 256.)

## 4. Learning metrics — sanity only, not a convergence result

| Metric | 128 (40 iters) | 256 (20 iters) |
|---|---:|---:|
| policy loss (last) | +0.0276 | +0.0356 |
| value loss (last) | 0.0202 | 0.0198 |
| approx KL (last) | 0.0313 | 0.0469 |
| clipfrac (last) | 0.398 | 0.343 |
| explained variance (last) | +0.848 | +0.697 |
| eval/return (last) | 13.12 | 8.51 |
| eval/success_once | 0.0 | 0.0 |

Both runs are tiny fractions of the official budget (0.51 %); the numbers are finite and
sane in both, and neither is evidence for or against PPO on PickCube. Note the two runs
are **not** the same experiment rescaled — 256 envs halves the iteration count (20 vs 40)
for the same 256k steps, so the learning dynamics differ; do not read the two columns as
"128 vs 256 with equal training".

## 5. Decision input for the next step

Using the measured **steady-state loop SPS** (startup amortises to nothing at 50M steps):

| Env count | 50M / seed | 3 seeds |
|---:|---:|---:|
| 128 | 5.34 h | ~16.0 h |
| **256** | **3.13 h** | **~9.4 h** |

**256 envs is viable and roughly halves the wall time** (≈9.4 h for a 3-seed PickCube
run) with plenty of VRAM headroom. This is the configuration to take into a full
reproduction *if and when* that run is approved — and a 256-env `ppo_fast.py`/4096-env
discussion is separate and still requires tensordict/wandb and likely a larger GPU.

## 6. Caveats

- One run, one seed, warm-ish start (the 256-env scene was a first-time build here; a
  repeat would shave the ~25 s build). ±3 % timing noise, no confidence interval.
- Peak VRAM is a 5 s-sampled device figure; short spikes between samples are invisible.
- `render_mode=None` vs `rgb_array`: this is the rendered path (matches upstream).
- Baseline VRAM drifted between the two runs (different sessions), so compare deltas and
  free-memory floors, not absolute `memory.used`.

## 7. Artifacts

| Path | Content |
|---|---|
| `results/m2_pilot_n256_summary.json` | full metrics (committed) |
| `results/raw/m2_stage_b_n256_pilot.log` | child stdout (gitignored) |
| `runs/PickCube-v1__ppo__9351__stageB_n256/` | 5 checkpoints + events (gitignored) |
