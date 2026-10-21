# M2 Stage A1 — GPU Parallel Simulation Benchmark

> **Status: PREPARED / NOT EXECUTED — `BLOCKED` on agent-session GPU visibility.**
> The harness, guards and evidence format are finished and statically verified, but
> **no level was run and no simulation number in this document was measured.**
> Stage A1 is `BLOCKED`, not `PASS`/`PARTIAL`/`FAIL`: `PASS`/`PARTIAL`/`FAIL` describe
> an experiment that ran, and none did.
>
> Date: 2026-10-21 · Base commit: `148a0ea` (A0 PASSED 12/12)
> Upstream pin: ManiSkill `v3.0.1` = `a4a4f9272ad64b1564035874b605ceb687b63ed8`
> Binary pin: PyPI `mani-skill==3.0.1`, `sapien==3.0.0`, `torch==2.5.1+cu121`
> Authorization: researcher approved A1 on 2026-10-21 (16→64→128→256, ≤256 only).

---

## 0. Why this document exists before the numbers

Stage A1 was authorized and prepared, but the agent shell became GPU-blind again
(§5). Rather than run four levels that would deterministically exit 8, the run was
handed to the researcher's terminal with the exact commands in §6. This document
therefore records: the frozen protocol, the guards that make it fail closed, the
static evidence, the blocked-attempt evidence, and the exact analysis that will be
applied to the numbers once they exist. **Every measurement field is marked
`NOT_MEASURED` until §7 is filled in from a real run.**

---

## 1. What Stage A1 measures — and what it does not

| Measures | Does not measure |
|---|---|
| Simulator steps/s for zero actions at 16/64/128/256 envs | Training throughput (no forward pass, no backward, no optimizer) |
| Device-level VRAM (SAPIEN/PhysX included) per env count | Official-baseline reproduction (that is 4096 envs × 50M steps × 3 seeds) |
| Initialisation, warmup and measured-step wall clock separately | Rendering cost (this run uses `render_mode=None`) |
| Termination/truncation/auto-reset counts across the 50-step horizon | Convergence, success rate, or any learning claim |

`render_mode=None` is deliberate: the official `ppo_fast.py` PickCube path uses
`render_mode="rgb_array"`, so its VRAM and SPS are **not** interchangeable with
these numbers (recorded as unknown #9 in `docs/PROGRESS.md`).

**Simulation SPS is not training SPS.** Any training-speed number derived from
this table is an unverified extrapolation and must be labelled as such (§8).

---

## 2. Frozen protocol

| Item | Value |
|---|---|
| Env / robot | `PickCube-v1` / `panda` |
| Observation | `state` (42-dim) |
| Control | `pd_joint_delta_pos` (8-dim: arm 7 + gripper 1) |
| Reward | `normalized_dense` (recorded, unused by zero actions) |
| Backend | `physx_cuda` — pinned explicitly, re-read after construction |
| Render mode | `None` |
| Seed | 0 on every level's `reset` |
| Actions | all zeros, `(num_envs, 8)` float32 on `cuda:0` |
| Levels | 16 → 64 → 128 → 256, strictly ascending, **512 and 4096 forbidden** |
| Warmup | 10 steps, discarded |
| Measured | ≤100 steps (a level that does fewer is a failure) |
| Per-level internal budget | 90 s (build + reset + warmup + measure + cleanup) |
| Per-process internal budget | 120 s (import + build + steps + cleanup), monotonic + watchdog |
| External per-level timeout | 150 s `timeout --signal=TERM --kill-after=10s` |
| VRAM start gate | free ≥ 3072 MiB, else the level is not started |
| VRAM abort gate | free < 1536 MiB, else the level aborts at the next sample |
| VRAM sampling | every 10 steps, device level (`nvidia-smi`), sampling time subtracted |
| Isolation | one OS process per level, one JSON + one log per level |
| Retries | none — every record carries `attempts: 1` |

Episode semantics: `ManiSkillVectorEnv(auto_reset=True, ignore_terminations=False)`.
With horizon 50 and 110 steps per level, truncation is guaranteed; a level that
records **zero truncations** is a hard failure (`FAIL_NO_TRUNCATION`), because it
means the episode boundary was never crossed and the rollout is not evidence of
anything.

---

## 3. Guards, and the failure they prevent

| # | Guard | Where | Prevents |
|---|---|---|---|
| 1 | Batched action `(n, 8)` built only via `batched_action_shape()` | `m2_gpu_sim_probe.py` | `(8,)` reaching the controller for n>1 (`sapien_env.py:1108` auto-batches only at n=1) |
| 2 | `sim_backend="physx_cuda"` pinned + post-build `backend.sim_backend` / `gpu_sim_enabled` re-read | probe | `auto` silently resolving to `physx_cpu` at n=1 (`sapien_env.py:233-237`) |
| 3 | Requested vs reported `num_envs` equality check | probe | a launcher/tyro mismatch silently measuring the wrong width |
| 4 | Auto-reset counters + `FAIL_NO_TRUNCATION` | probe | stepping past the horizon with no reset policy, i.e. meaningless timing |
| 5 | Mandatory telemetry, checked **before** `import torch` | probe | sizing a level blind; exit 8 with no scene built |
| 6 | 3072 MiB start gate / 1536 MiB abort gate | probe | OOM on an 8 GiB card with a desktop attached |
| 7 | Device counters authoritative, torch allocator recorded as a lower bound | probe | under-estimating PhysX by ~200× (A0: 8.2 MiB torch vs ~1.33 GiB device) |
| 8 | Three nested time limits + launcher backstop | probe + launcher | a hung scene build eating the next level's budget |
| 9 | One process per level | launcher | one crash/OOM/hard-kill destroying the other levels' records |
| 10 | Stop on first non-OK level; remaining marked `NOT_RUN_AFTER_FAILURE` | launcher | a half-run sweep being reported as a complete one |
| 11 | Pinned versions asserted at runtime (3.0.1/3.0.0/2.5.1) | probe | accidentally measuring a different stack |
| 12 | No optimizer / tensorboard / tensordict / wandb import (AST-audited) | all three scripts | this stage drifting into training |

Static evidence (2026-10-21, this commit):

- `scripts/m2_static_audit.py --json results/m2_static_checks.json` → **51 checks passed, 0 failed** (A0 17 + A1 24 + launcher 10, including the gate-ordering check and the budget pins).
- Self-tests, pure logic, no CUDA context: **A0 36 + A1 24 + launcher 22 = 82 checks, 0 failed**.
- Fail-closed proof: with `PATH` stripped, the probe exits **8** in **0.049 s** with `levels: []` and `reason: "nvidia-smi not found on PATH"` — no scene, no CUDA context.
- Launcher fail-closed proof: preflight blocked → verdict `BLOCKED`, four rows `NOT_RUN_PREFLIGHT_BLOCKED`, exit **8** (`results/raw/m2_a1_preflight_blocked.json`).

---

## 4. Artifacts

| Path | Content | Committed |
|---|---|---|
| `results/m2_a1_benchmark_summary.json` | aggregate: 4 rows + derived scaling + verdict | yes (currently `BLOCKED`) |
| `results/raw/m2_a1_n{16,64,128,256}.json` | per-level full evidence | no (`results/raw/` is gitignored) |
| `results/raw/m2_a1_n{16,64,128,256}.log` | per-level stdout/stderr + argv + exit code | no |
| `docs/M2_A1_EXECUTION_REPORT.md` | this document | yes |
| `results/m2_static_checks.json` | static audit summary | yes |

Because `results/raw/` is gitignored, the aggregate summary **embeds the full
per-level measurement set** for every level that ran, so the committed artifact is
self-sufficient for audit.

---

## 5. Why it did not run here (evidence)

Read-only snapshot, full log in `results/raw/m2_a1_gpu_unavailable_diag.log`:

| Probe | Result |
|---|---|
| `ls -l /dev/nvidia*` | only `/dev/nvidia-caps` (an **empty directory**); no `/dev/nvidia0`, `nvidiactl`, `nvidia-uvm`, `nvidia-uvm-tools` |
| `nvidia-smi --query-gpu=...` | rc=9, "couldn't communicate with the NVIDIA driver" |
| `/proc/driver/nvidia/version` | `NVRM 580.178.04` — module loaded |
| `lspci` | `01:00.0 GA104 [GeForce RTX 3070 LHR]` — card present |
| venv torch | `2.5.1+cu121`, `is_available False`, `device_count 0` |
| launcher preflight | `telemetry_verdict=UNAVAILABLE`, `blocked=true`, driver exit **8** |

This is the same session-scoped namespace issue documented in
`docs/M2_GPU_PROBE_REPAIR.md` §7.3 and `docs/PROGRESS.md` unknown #12: the machine
and driver are healthy (A0 passed in the researcher's terminal at 15:44 on the same
host), the *agent session* has no NVIDIA character devices. No driver, permission,
CUDA or container change was attempted.

At 15:44 the researcher's terminal reported free VRAM 5592–5654 MiB, so the
3072 MiB start gate is satisfiable there. That is context, **not** a measurement
of this stage.

---

## 6. Exact commands (researcher's GPU-visible terminal)

Everything below runs in one shell; the launcher spawns one isolated process per
level and stops at the first failure.

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction

# 0) read-only preflight: is the GPU visible *in this terminal*?
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
ls -l /dev/nvidia* | head
echo "---"

# 1) static gates (no CUDA context, ~2 s)
./.venv-maniskill/bin/python scripts/m2_gpu_sim_probe.py --self-test
./.venv-maniskill/bin/python scripts/m2_a1_run_all.py --self-test
./.venv-maniskill/bin/python scripts/m2_static_audit.py --json results/m2_static_checks.json

# 2) show the four per-level commands without touching the GPU
./.venv-maniskill/bin/python scripts/m2_a1_run_all.py --dry-run

# 3) the authorized benchmark: 16 -> 64 -> 128 -> 256, one process each,
#    90 s internal per level, 150 s external per level, stop on first failure
mkdir -p results/raw
./.venv-maniskill/bin/python scripts/m2_a1_run_all.py 2>&1 | tee results/raw/m2_a1_launcher.log
echo "launcher_exit=${PIPESTATUS[0]}"
```

Expected exit codes from the launcher: `0` PASS (all four), `1` PARTIAL (some
levels OK, then a stop), `2` FAIL (first level failed), `8` BLOCKED (no telemetry).
Per-level exit codes from the probe: `0` OK, `2` backend mismatch, `3` VRAM gate,
`4` budget, `5` OOM/error, `7` version, `8` no telemetry, `9` level-level assertion
(incomplete measurement, or zero truncations at n≥2 steps beyond the horizon).

A single level can be re-run on request (this is a *new* authorized action, not an
automatic retry):

```bash
timeout --signal=TERM --kill-after=10s 150 \
  ./.venv-maniskill/bin/python scripts/m2_gpu_sim_probe.py \
    --env-counts 64 --warmup-steps 10 --steps 100 \
    --level-seconds 90 --total-seconds 120 --external-limit-s 150 \
    --min-free-mib 3072 --abort-free-mib 1536 --vram-every 10 \
    --output results/raw/m2_a1_n64.json 2>&1 | tee results/raw/m2_a1_n64.log
```

Total external ceiling for the whole sweep: 4 × 150 s = **600 s**, plus overhead.

---

## 7. Measurements

### 7.1 Required table — **NOT_MEASURED**

No row below exists yet. The field list is fixed now so the numbers cannot be
cherry-picked afterwards.

| # | Field | JSON key | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 1 | Environments | `num_envs` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 2 | Initialisation time | `init_s` (`build_s`) | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 3 | Warmup time | `warmup_s` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 4 | Measured simulation time | `measure_s` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 5 | Aggregate simulation SPS | `sim_sps` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 6 | VRAM before init | `device_used_baseline_mib` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 7 | Peak VRAM during execution | `device_used_peak_mib` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 8 | Remaining free VRAM | `vram_free_min_mib` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 9 | Increase vs baseline | `device_used_increase_mib` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 10 | Terminated / truncated / auto-reset | `counters_total` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 11 | Errors / warnings | `error`, `warnings` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| 12 | Exit code / status | `exit_code`, `status` | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| — | Total experiment wall clock | `total_wall_clock_s` | NOT_MEASURED | | | |

The single hard anchor available today is A0 (1 env, 3 steps): device used
2227→3591 MiB, i.e. **~1.33 GiB for one GPU scene**, with a torch allocator peak of
8.2 MiB. It is a 1-env reference point, **not** a row of this table.

### 7.2 Analysis that will be applied (fixed in advance)

- **Throughput scaling**: `speedup(n) = SPS(n)/SPS(16)`.
- **Scaling efficiency**: `eff(n) = speedup(n)/(n/16)`. `eff < 1` ⇒ per-step
  overhead grows with width (diminishing returns); `eff ≈ 1` ⇒ near-linear.
- **Marginal gain**: `ΔSPS/Δn` between adjacent levels; the level where it
  collapses bounds the useful width on this card.
- **VRAM scaling**: report both `Δused(16)` (fixed scene cost + 16 envs) and the
  incremental `(Δused(n) − Δused(16))/(n − 16)`, because the first level carries
  the fixed PhysX context and must not be used as a per-env slope.
- **Stability**: whether `vram_free_min_mib` stays above the 1536 MiB abort floor
  with margin, and whether free memory drifts across samples (leak signature).
- **Stage B width rule** (pre-registered): choose the largest n with
  `device_used_peak_mib ≤ 6500`, `vram_free_min_mib ≥ 1000`, `eff ≥ 0.5`, and no
  error/warning; then **halve it** for the first PPO pilot because the training
  path also allocates policy/optimizer state and (for the fast script) rendering.
  If 16 itself fails the headroom rule, Stage B is not viable locally.

---

## 8. Provisional Stage B budget (unverified, formula only)

`sps` below is a placeholder, **not** a measurement. With the pre-registered
256k-step pilot (`docs/M2_GPU_PILOT_PROPOSAL.md` §Stage B):

`wall = 256,000 / SPS_end_to_end`, and `SPS_end_to_end < SPS_sim` by an unknown
factor because gradients, GAE and evaluation are absent from A1.

| Assumed end-to-end SPS | 256k steps |
|---:|---:|
| 1,000 | 4.3 min |
| 3,000 | 1.4 min |
| 6,000 | 0.7 min |

Even the slowest case fits the 20-minute pilot budget *if* the simulation rate
holds; the real risk is VRAM, not time. **No Stage B number may be quoted until
that pilot runs.** All of this is `PROVISIONAL`.

---

## 9. Limitations and honest status

1. **Nothing in §7 was measured.** A1 is `BLOCKED`, not passed.
2. Four levels × 100 steps is a **short-window** measurement: it does not capture
   thermal throttling, allocator fragmentation over minutes, or multi-minute leak
   behaviour.
3. `render_mode=None` under-reports the memory of the official fast baseline.
4. Zero actions produce a near-static scene; contact-rich states may allocate and
   compute differently. SPS here is an upper bound for this task, not a guarantee.
5. The 8 GiB card is shared with a desktop session; the baseline `memory.used`
   varies run to run, which is why deltas and free-memory floors are reported
   instead of absolute used values.
6. Only 4 of the 6 authorized protection points are exercised end-to-end here
   (telemetry gate, VRAM gates, budgets, and the launcher's stop-on-failure are
   all proven; the OOM path and the backend-mismatch path are code-level only).

## 10. Next actions

1. Researcher runs §6 in a GPU-visible terminal (A0 already proved `physx_cuda`
   works there at 15:44 on this host).
2. This document's §7.1 is filled from `results/m2_a1_benchmark_summary.json`,
   the Stage B width is selected by the §7.2 rule, and the verdict is set to
   `PASS` / `PARTIAL` / `FAIL` **only then**.
3. Stage B stays unapproved until that happens; `tensorboard` stays uninstalled.
