# M2 Stage A1 — GPU Parallel Simulation Benchmark

> **Status: `PASS` — 4/4 levels OK, launcher exit 0, total wall clock 27.706 s.**
> Executed 2026-10-21 16:18 in the researcher's GPU-visible terminal, after the agent
> session refused with exit 8 at 16:09 (only an empty `/dev/nvidia-caps` there).
>
> Base commit at execution: `cf1e698` · upstream pin ManiSkill `v3.0.1` =
> `a4a4f9272ad64b1564035874b605ceb687b63ed8` · binary pin `mani-skill==3.0.1`,
> `sapien==3.0.0`, `torch==2.5.1+cu121` · GPU RTX 3070 8192 MiB.

**Headline numbers** (all measured, see §6):

| envs | aggregate sim SPS | device VRAM peak | free at peak | scaling efficiency |
|---:|---:|---:|---:|---:|
| 16 | 905.8 | 3912 MiB | 4280 MiB | 1.00 |
| 64 | 3342.0 | 3867 MiB | 4325 MiB | 0.92 |
| 128 | 6437.4 | 3931 MiB | 4261 MiB | 0.89 |
| 256 | 11746.1 | 3931 MiB | 4261 MiB | 0.81 |

**The decisive finding is not the throughput: device VRAM is flat at ~3.9 GiB from 16
to 256 envs.** The ~1.3 GiB GPU PhysX scene is a fixed cost, and 240 extra envs added
**0 MiB** of *sampled* peak between 128 and 256. Memory is therefore not what limits
this stage on an 8 GiB card — the per-step serialisation cost is.

---

## 1. What Stage A1 measures — and what it does not

| Measures | Does not measure |
|---|---|
| Simulator steps/s for zero actions at 16/64/128/256 envs | Training throughput (no forward pass, no backward, no optimizer) |
| Device-level VRAM (SAPIEN/PhysX included) per env count | Official-baseline reproduction (that is 4096 envs × 50M steps × 3 seeds) |
| Initialisation, warmup and measured-step wall clock separately | Rendering cost (this run uses `render_mode=None`) |
| Termination/truncation/auto-reset counts across the 50-step horizon | Convergence, success rate, or any learning claim |

`render_mode=None` is deliberate: the official `ppo_fast.py` PickCube path uses
`render_mode="rgb_array"`, so its VRAM and SPS are **not** interchangeable with these
numbers (unknown #9 in `docs/PROGRESS.md`). **Simulation SPS is not training SPS** — any
training-speed number derived from this table is an unverified extrapolation (§8).

---

## 2. Frozen protocol (as executed)

| Item | Value |
|---|---|
| Env / robot | `PickCube-v1` / `panda` |
| Observation | `state` (42-dim) |
| Control | `pd_joint_delta_pos` (8-dim: arm 7 + gripper 1) |
| Reward | `normalized_dense` (recorded, unused by zero actions) |
| Backend | `physx_cuda`, pinned explicitly, re-read after construction |
| Render mode | `None` |
| Seed | 0 on every level's `reset` |
| Actions | all zeros, `(num_envs, 8)` float32 on `cuda:0` |
| Levels | 16 → 64 → 128 → 256, strictly ascending, 512/4096 never requested |
| Warmup | 10 steps, discarded |
| Measured | 100 steps (all four levels reached exactly 100) |
| Per-level internal budget | 90 s (used: 3.4–4.5 s) |
| Per-process internal budget | 120 s (used: 5.7–6.8 s) |
| External per-level timeout | 150 s `timeout --signal=TERM --kill-after=10s` (never hit) |
| VRAM start gate | 3072 MiB free (baseline free was 4325–4420 MiB) |
| VRAM abort gate | 1536 MiB (min free observed 4261 MiB — never approached) |
| VRAM sampling | every 10 steps, device level, sampling time subtracted from `measure_s` |
| Isolation | one OS process per level, one JSON + one log per level |
| Retries | none |

Episode semantics: `ManiSkillVectorEnv(auto_reset=True, ignore_terminations=False)`,
horizon 50 read from `TimeLimitWrapper` (`horizon_source=TimeLimitWrapper` on all four
levels). 110 steps per level therefore crosses the boundary twice; truncations were
observed on every level (32 / 128 / 256 / 512 env-truncations), which is the evidence
that the auto-reset path is real rather than assumed.

---

## 3. Guards that were exercised

| # | Guard | Evidence from this run |
|---|---|---|
| 1 | Batched action `(n, 8)` via `batched_action_shape()` | `action.shape` recorded as `[16,8] / [64,8] / [128,8] / [256,8]`, `cuda:0`, float32 |
| 2 | `sim_backend="physx_cuda"` + post-build re-read | all four levels resolved `physx_cuda`, `gpu_sim_enabled=True` |
| 3 | Requested vs reported `num_envs` | equal on all four |
| 4 | Auto-reset counters + `FAIL_NO_TRUNCATION` | truncations present on all four; zero would have failed the level |
| 5 | Telemetry before `import torch` | passed at startup; the 16:09 blocked run proves the refusal path |
| 6 | 3072 / 1536 MiB gates | never triggered; closest approach 4261 MiB free |
| 7 | Device counters authoritative | torch peak 8.2–8.6 MiB vs device ~1.29–1.43 GiB — a ~160× gap |
| 8 | Nested time limits | longest level 4.5 s of a 90 s budget; no watchdog fire |
| 9 | One process per level | 4 independent JSON + 4 logs, all present |
| 10 | Stop on first failure | not triggered (all OK) |
| 11 | Pinned versions asserted | 3.0.1 / 3.0.0 / 2.5.1+cu121, no mismatch |
| 12 | No optimizer / tensorboard / tensordict / wandb | AST-audited; 51 audit checks green |

Static gates re-run immediately before execution: 51 audit + 82 self-test checks, 0 failed
(`results/m2_static_checks.json`). Fail-closed behaviour was proven twice before the real
run (PATH stripped → exit 8 in 0.049 s; launcher preflight at 16:09 → `BLOCKED`, exit 8,
zero levels).

---

## 4. Artifacts

| Path | Content | Committed |
|---|---|---|
| `results/m2_a1_benchmark_summary.json` | all four rows + derived scaling + verdict | yes |
| `results/raw/m2_a1_n{16,64,128,256}.json` | per-level full evidence (~6.1 KB each) | no (`results/raw/` gitignored) |
| `results/raw/m2_a1_n{16,64,128,256}.log` | per-level stdout/stderr + argv + exit code | no |
| `results/raw/m2_a1_launcher.log` | launcher transcript | no |
| `docs/M2_A1_EXECUTION_REPORT.md` | this document | yes |

The committed summary embeds the complete per-level measurement set, so it is
self-sufficient for audit even though the raw directory is ignored.

---

## 5. Execution history (both attempts)

| Time | Where | Result |
|---|---|---|
| 16:09 | agent session | `BLOCKED`, exit 8, 0 levels — only an empty `/dev/nvidia-caps`, `nvidia-smi` rc=9, `torch.cuda.device_count()=0`. Nothing built. |
| 16:18 | researcher's terminal | **`PASS`, exit 0, 4/4 levels, 27.706 s** |

Command (verbatim):

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
./.venv-maniskill/bin/python scripts/m2_a1_run_all.py 2>&1 | tee results/raw/m2_a1_launcher.log
echo "exit=${PIPESTATUS[0]}"
```

The launcher expanded it into four `timeout --signal=TERM --kill-after=10s 150 …` children,
one per env count, in ascending order.

---

## 6. Measurements — the 12 required fields

| # | Field | 16 | 64 | 128 | 256 |
|---|---|---:|---:|---:|---:|
| 1 | Environments (`num_envs`) | 16 | 64 | 128 | 256 |
| 2 | Initialisation (`build_s`) | 1.169 s | 1.292 s | 1.439 s | 1.712 s |
| 3 | Warmup, 10 steps (`warmup_s`) | 0.209 s | 0.207 s | 0.212 s | 0.235 s |
| 4 | Measured simulation (`measure_s`) | 1.766 s | 1.915 s | 1.988 s | 2.179 s |
| 5 | **Aggregate simulation SPS** | **905.8** | **3342.0** | **6437.4** | **11746.1** |
| 6 | Device used before init | 2224 MiB | 2133 MiB | 2133 MiB | 2133 MiB |
| 7 | **Device used peak** (sampled) | **3912 MiB** | **3867 MiB** | **3931 MiB** | **3931 MiB** |
| 8 | Free VRAM at peak | 4280 MiB | 4325 MiB | 4261 MiB | 4261 MiB |
| 9 | Increase vs baseline | +1276 MiB | +1361 MiB | +1425 MiB | +1425 MiB |
| 10 | Terminated / truncated / auto-reset (env-level) | 0 / 32 / 32 | 1 / 128 / 129 | 1 / 256 / 257 | 3 / 512 / 515 |
| 11 | Errors / warnings | none / none | none / none | none / none | none / none |
| 12 | Exit code / status | 0 / OK | 0 / OK | 0 / OK | 0 / OK |
| — | Total experiment wall clock | | | | **27.706 s** |

Supporting per-level detail:

| n | import_s | reset_s | steps done | env-steps | ms per `env.step` | SPS/env | process wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2.213 | 0.014 | 100 | 1600 | 17.665 | 56.61 | 5.736 s |
| 64 | 2.218 | 0.024 | 100 | 6400 | 19.150 | 52.22 | 6.180 s |
| 128 | 2.190 | 0.036 | 100 | 12800 | 19.884 | 50.29 | 6.228 s |
| 256 | 2.210 | 0.060 | 100 | 25600 | 21.794 | 45.88 | 6.756 s |

`torch_peak_alloc`: 8.2 / 8.3 / 8.4 / 8.6 MiB; `torch_peak_reserved`: 22.0 MiB on all
four. Recorded as a **torch-only lower bound** — it is ~160× below the device view and
must never be used to size PhysX.

---

## 7. Analysis

### 7.1 Throughput scaling — near-linear, with mild and predictable decay

| n | SPS | speedup vs 16 | ideal speedup | efficiency | marginal SPS per added env |
|---:|---:|---:|---:|---:|---:|
| 16 | 905.8 | 1.000 | 1.0 | 1.000 | — |
| 64 | 3342.0 | 3.689 | 4.0 | 0.922 | 50.8 |
| 128 | 6437.4 | 7.107 | 8.0 | 0.888 | 48.4 |
| 256 | 11746.1 | 12.968 | 16.0 | 0.811 | 41.5 |

- 16× the environments buys **12.97×** the throughput, i.e. 81 % efficiency.
- Efficiency decays monotonically **0.922 → 0.888 → 0.811**, and marginal SPS per added
  env drops **50.8 → 48.4 → 41.5** (−18 %). This is the signature of a fixed per-step
  cost (kernel launches, buffer copies, Python/torch dispatch) paid once per `env.step`
  regardless of width.
- The cleanest statement of it: one `env.step` covering **all** envs takes 17.7 ms at
  n=16 but only 21.8 ms at n=256 — 16× the work for 1.23× the wall time.
- **No cliff** in the tested range: the last doubling still delivered 1.82×.

### 7.2 VRAM scaling — the fixed scene dominates, not the env count

| n | used after build | used peak (sampled) | Δ vs idle | marginal Δ per added env |
|---:|---:|---:|---:|---:|
| 16 | 3548 MiB | 3912 MiB | +1276 MiB | — |
| 64 | 3492 MiB | 3867 MiB | +1361 MiB | 1.77 MiB |
| 128 | 3556 MiB | 3931 MiB | +1425 MiB | 1.00 MiB |
| 256 | 3556 MiB | 3931 MiB | +1425 MiB | **0.00 MiB** |

- The GPU PhysX scene costs **~1.28–1.43 GiB regardless of env count** — consistent with
  the A0 single-env measurement (~1.33 GiB). Up to 256 envs, width is nearly free in VRAM.
- Idle baseline moved 2224 → 2133 MiB between levels (desktop noise), which is why deltas
  and free floors are reported instead of absolute used values.
- Worst-point headroom is **4261 MiB free** — 2.8× the 1536 MiB abort gate. VRAM is
  **not** the binding constraint for this stage.

**Caveat, stated plainly:** the peak is a *sampled* value (every 10 steps), so a transient
spike between samples would be missed, and "128 and 256 identical" means "no sustained
growth at 100-step / sample-10 resolution", not "provably identical allocation". A 0 MiB
marginal cost is plausible for PhysX pre-allocated buffer pools but is **not** proven here.

### 7.3 Is memory stable?

Yes, over the measured window: 10 samples per level, minimum free memory never left the
4261–4325 MiB band, no downward drift across samples, no allocator warnings. A 100-step
window (≈2.2 s) cannot detect a multi-minute leak; that is outside this stage.

### 7.4 Diminishing returns?

Present but mild **in throughput**, absent **in memory**:

- Throughput: 81 % efficiency at 256, with a monotone decay that suggests a further drop
  at 512 (extrapolation — **not measured**, and 512 was never authorized).
- Memory: no measurable return to gain beyond 128 envs.

Practical reading: throughput per VRAM byte peaks at 256; throughput per unit of per-step
latency is already 89 % saturated at 128.

### 7.5 Which configuration suits a short training pilot?

Applying the rule **pre-registered before the run** (largest n with peak ≤ 6500 MiB,
free_min ≥ 1000 MiB, efficiency ≥ 0.5, no warnings — then halve it for the first pilot):

| Criterion | 16 | 64 | 128 | 256 |
|---|---|---|---|---|
| peak ≤ 6500 MiB | pass | pass | pass | pass |
| free_min ≥ 1000 MiB | pass | pass | pass | pass |
| efficiency ≥ 0.5 | pass | pass | pass | pass |
| no warnings/errors | pass | pass | pass | pass |

→ largest qualifying = **256**, halved = **128** for the first training pilot.

### 7.6 One observation that needs care

Zero actions terminated **1 / 1 / 3** env-episodes (at 64 / 128 / 256; none at 16), always
as `terminated`, never with a warning. With a static robot and no action, the only
plausible route to `terminated` in PickCube is a randomised initial object pose that
already satisfies `is_obj_placed & is_robot_static`. That reading is **unverified** — the
run recorded counts, not per-env causes. It affects nothing in the SPS/VRAM numbers
(3 events in 28,160 env-steps ≈ 0.01 %), but it is a reason not to read a success rate out
of this stage.

---

## 8. Stage B recommendation and provisional budget

**Recommended `num_envs` for the Stage B pilot: 128.**

Why 128 and not 256: the halving is the pre-registered safety rule, and it buys margin for
three costs A1 does not pay — policy + optimizer state, GAE/update buffers, and rendering
if the pilot path enables `rgb_array`. It costs 1.8× throughput (6437 vs 11746 SPS)
against a 20-minute budget that is nowhere near binding. If the pilot later shows both
VRAM and per-iteration time are far from limits, 256 is the natural second step — as a new
authorization, not a silent change.

Provisional resource budget (**PROVISIONAL / unverified** where marked):

| Item | Estimate | Basis |
|---|---|---|
| Env steps | 256,000 | approved pilot ceiling, unchanged |
| Envs × steps | 128 × 50 = 6,400 / iteration | — |
| Iterations | 40 | 256,000 / 6,400 |
| Pure-sim floor | 256,000 / 6437 SPS ≈ **40 s** | measured simulator SPS at 128; a **lower bound on wall time**, since training adds work |
| Realistic end-to-end | **1.5–7 min** | PROVISIONAL: assumes end-to-end SPS is 10–50 % of simulator SPS (640–3200 SPS); **not measured** |
| VRAM | scene ~1.43 GiB + policy/optimizer (≈0.29 M params) + activations for a 6,400-sample batch | scene measured; PyTorch side **unmeasured** |
| Risk | VRAM, not time | peak 3931 MiB of 8192 MiB leaves 4261 MiB, but rendering would change that |

`sps_end_to_end` is **unknown**; dividing 256,000 by it is the only honest arithmetic
available, and it must be replaced by a measured number the moment the pilot runs.

---

## 9. Limitations

1. **100-step window** (~2.2 s per level): no thermal throttling, no multi-minute leak, no
   allocator fragmentation over time. These are steady-state *samples*, not a soak test.
2. **Zero actions**: near-static scenes. Contact-rich trajectories may cost differently;
   treat these SPS as a **best case** for this task.
3. **`render_mode=None`**: the official fast baseline renders, so its memory and speed are
   not comparable with this table.
4. **Sampled peak**: 10-step granularity, so short spikes are invisible (§7.2 caveat).
5. **Four points, one seed, one repetition**: `reset(seed=0)` is deterministic per level,
   but GPU timing still varies by a few percent run to run; the efficiency numbers carry
   roughly ±3 % noise, not a confidence interval.
6. **The desktop was not idle**: baseline used memory drifted 2224 → 2133 MiB and the card
   is shared with a desktop session. Absolute memory numbers are not portable to another
   machine; the deltas and the free floor are.
7. **No claim about 512 or 4096** — never run; the launcher refuses > 256.

## 10. Next actions

1. Stage B needs a **separate approval**: install `tensorboard` in the isolated venv
   (~200–400 MB, `pip check` after), then run upstream `ppo.py` unmodified with
   `num_envs=128`, ≤256k env steps, ≤20 min, ≤300 MB, no wandb, no video.
2. Nothing in this stage may be reported as a baseline reproduction or as training
   throughput.
3. If the pilot's measured end-to-end SPS makes three-seed PickCube exceed ~2 days, only
   then does a 24 GB cloud card become a discussion (still requiring approval).
