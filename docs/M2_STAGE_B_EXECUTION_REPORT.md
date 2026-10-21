# M2 Stage B — PPO Pilot (plumbing validation)

> **Status: PREPARED / NOT EXECUTED — `BLOCKED` on agent-session GPU access.**
> Phase 1 (code audit), Phase 2 (dependency install) and the supervisor are done and
> verified. **Phase 4/5 did not run: zero iterations, zero measured metrics.**
> A verdict of PASS/PARTIAL/FAIL would describe an experiment; none happened.
>
> Date: 2026-10-21 · Base commit: `416fd22` · Classification: **ADAPTATION / PLUMBING_ONLY**
> Upstream: `examples/baselines/ppo/ppo.py` @ ManiSkill `v3.0.1`
> = `a4a4f9272ad64b1564035874b605ceb687b63ed8`, blob `66802de0653061da550d6600d6346be374dbf974`
> Env: `PickCube-v1`, 128 envs, 256,000 env steps, seed 9351, RTX 3070 8 GiB.

---

## 0. What this pilot is and is not

**Is:** an engineering check that one loop of *sampling → GAE → PPO update → eval →
TensorBoard → checkpoint* completes on this host within budget.

**Is not:** a baseline reproduction (official fast path is `ppo_fast.py`, 4096 envs,
4 steps, 8 epochs, 50M steps, cudagraphs, 3 seeds), a convergence test, or evidence
about whether PPO "works" on PickCube. 256k steps is ~0.5 % of the official budget;
a near-zero success rate here is expected and uninformative.

---

## 1. Phase 1 — pre-execution audit (done)

### 1.1 Source provenance — is this really the pinned file?

| Check | Result |
|---|---|
| File used | `/home/ubuntu2004/.cache/ms-upstream-ppo/ppo.py` (outside the repo; upstream is **not vendored**) |
| Size / lines | 22,982 bytes / 470 lines |
| Computed git blob SHA-1 | `66802de0653061da550d6600d6346be374dbf974` |
| Recorded in `results/m2_source_provenance.json` | **identical** |
| Other `ppo.py` on disk | none (`find` over `/home/ubuntu2004`), and the PyPI wheel does **not** ship `examples/` |
| Interpreter | `.venv-maniskill/bin/python` = Python 3.10.12; `mani_skill 3.0.1` resolves to the venv's `site-packages` |
| Supervisor re-checks the blob before launching | yes — `BLOCKED_PPO_MISMATCH` with exit 8 otherwise |

So the command cannot silently run a stale copy or a second installation: there is
exactly one candidate file and its bytes are verified against the pin at launch time.

### 1.2 CLI syntax — the documented skeleton was wrong

`ppo.py` uses `tyro.cli(Args)`, and docstring fields become **kebab-case** flags.
`--help` (run inside the venv after installing tensorboard) confirms:

```
--num-envs INT          --num-steps INT         --total-timesteps INT
--update-epochs INT     --num-minibatches INT   --num-eval-envs INT
--eval-freq INT         --seed INT
--capture-video, --no-capture-video
--track, --no-track
--save-model, --no-save-model
```

The earlier proposal wrote `--capture_video=False --track=False`, which tyro rejects
before any training starts. The audited argv uses `--no-capture-video --no-track`.

### 1.3 Requested configuration, as verified in the argv

| Parameter | Requested | In argv | Upstream default | Note |
|---|---|---|---|---|
| `num_envs` | 128 | `--num-envs 128` | 512 | ADAPTATION (8 GiB card) |
| `num_steps` | 50 | `--num-steps 50` | 50 | same |
| `total_timesteps` | 256000 | `--total-timesteps 256000` | 10,000,000 | ADAPTATION |
| `update_epochs` | 4 | `--update-epochs 4` | 4 | same |
| `num_minibatches` | 32 | `--num-minibatches 32` | 32 | same |
| `num_eval_envs` | 8 | `--num-eval-envs 8` | 8 | same |
| `eval_freq` | 5 | `--eval-freq 5` | 25 | ADAPTATION (denser checkpoints) |
| `seed` | 9351 | `--seed 9351` | 1 | one official seed, not the 3-seed set |
| `capture_video` | False | `--no-capture-video` | True | avoids `RecordEpisode` + video VRAM |
| `track` | False | `--no-track` | False | no wandb (not installed) |
| `save_model` | True | `--save-model` | True | same |

Implicit parameters that matter and were **not** touched (upstream defaults):
`gamma=0.8`, `gae_lambda=0.9`, `clip_coef=0.2`, `target_kl=0.1`, `vf_coef=0.5`,
`max_grad_norm=0.5`, `norm_adv=True`, `ent_coef=0.0`, `anneal_lr=False`,
`learning_rate=3e-4`, `partial_reset=True` (→ `ignore_terminations=False`),
`eval_partial_reset=False`, `reconfiguration_freq=None`, `eval_reconfiguration_freq=1`,
`control_mode="pd_joint_delta_pos"`, `obs_mode="state"`, `sim_backend="physx_cuda"`.

Derived arithmetic (from `ppo.py:176-178`): batch = 128×50 = **6,400**;
minibatch = 6400//32 = **200**; iterations = 256000//6400 = **40**.

### 1.4 Audit findings that change expectations

1. **`render_mode` is hardcoded to `"rgb_array"`** at `ppo.py:195`, with no CLI
   switch. A1 measured 3931 MiB peak with `render_mode=None`, so **the pilot's VRAM
   will be higher than A1's** and the 1536 MiB floor is doing real work. This is the
   single largest unknown of the phase. It is *not* modified (that would change the
   upstream implementation); it is documented as a deviation from A1's measurement
   conditions instead.
2. **Checkpoint count: 9, not 2.** `ppo.py:297` saves at `iteration % eval_freq == 1`,
   so with `eval_freq=5` and 40 iterations it fires at iterations 1, 6, 11, 16, 21,
   26, 31, 36 → **8 periodic** `ckpt_*.pt`; `ppo.py:465` adds `final_ckpt.pt` at the
   end → **9 files** (~1.2–3 MB each). Earlier docs said "2 个 .pt"; corrected in
   `M2_GPU_PILOT_PROPOSAL.md`.
3. **Evaluation runs 8 times** (same `iteration % eval_freq == 1` condition), each
   50 steps × 8 envs = **400 eval env-steps**, total **3,200**. These are counted
   separately from the 256,000 training steps.
4. **Optimizer updates cannot be read exactly.** Upstream prints no per-update count,
   and `target_kl=0.1` can break an epoch early (`ppo.py:405,440`). 40×4×32 = 5,120
   is therefore an **upper bound**, stated as such wherever it appears.
5. **`ManiSkillVectorEnv(..., record_metrics=True)`** is used for both train and eval,
   so the event file carries episode metrics (`success_once`, `return`, `episode_len`)
   under both `train/` and `eval/` — that is where the eval success rate comes from,
   not from a separate evaluator.
6. Upstream `ppo.py` computes `charts/SPS` from `start_time = time.time()` taken
   **after** env creation (`ppo.py:257`), so its SPS excludes import, scene build and
   the first reset. The supervisor therefore reports `end_to_end_sps` separately.

### 1.5 Documentation corrected in this pass

| File | Was | Now |
|---|---|---|
| `docs/M2_START_HERE.md` | A1 "AUTHORIZED-BUT-BLOCKED (0 levels)" | A1 **PASSED 4/4, 27.706 s**, with SPS/VRAM numbers; Stage B state added |
| `docs/M2_GPU_PILOT_PROPOSAL.md` | "A0 拟执行命令（待批准）", `--capture_video=False` skeleton, "2 个 .pt", "free < 400 MiB 即中止" | A0 已 PASS；kebab-case 命令；**9 个 ckpt 的推导**；1536 MiB 运行闸；`render_mode` 硬编码告警；tensorboard 已装版本 |
| `docs/M2_GPU_PILOT_PROPOSAL.md` §0 | tensorboard "MISSING" | 2.21.0 installed, with the exact version set and ~64 MB footprint |

---

## 2. Phase 2 — dependencies (done)

Authorized scope: `tensorboard` only, inside the isolated venv.

```bash
export TMPDIR=/home/ubuntu2004/.cache/ms-m1-tmp
./.venv-maniskill/bin/python -m pip install --no-cache-dir 'tensorboard>=2.16,<3'
./.venv-maniskill/bin/python -m pip check
```

| Package | Version |
|---|---|
| `tensorboard` | **2.21.0** |
| `tensorboard-data-server` | 0.7.2 |
| `protobuf` | 7.36.2 |
| `grpcio` | 1.84.0 |
| `markdown` | 3.10.3 |
| `werkzeug` | 3.1.9 |

- `pip check` → **No broken requirements found.**
- `from torch.utils.tensorboard import SummaryWriter` works; a probe event file was
  written and read back successfully.
- Disk: 55,032 KB of package trees + 9,652 KB of `.dist-info` ≈ **64 MB** added
  (no wheel cache kept). venv is now 6,264,584,585 bytes ≈ **5.83 GiB**.
- **Not installed** (deliberately): `wandb`, `tensordict`, `torchrl`.
- No existing package was upgraded or removed; `torch 2.5.1+cu121`, `mani_skill 3.0.1`,
  `sapien 3.0.0`, `gymnasium 1.1.1`, `numpy 1.24.4` unchanged.
- `ppo.py --help` parses and prints the full flag list (exit 0) — Phase 1.2 above.

Resource budget check: 64 MB ≪ the 300 MB output cap and ≪ the 200–400 MB estimate.
No major environment change, so the "stop if installation requires major changes"
condition did not trigger.

---

## 3. Phase 3 — GPU environment (blocked here, ready there)

This agent session still cannot reach the GPU. `scripts/gpu_visibility_probe.py`:

```
verdict: NO_NODES   (exit 3)
/dev: full, null, random, shm, tty, urandom, zero   (no NVIDIA nodes)
nvidia-smi: rc=9 "couldn't communicate with the NVIDIA driver"
```

Root cause is unchanged from the 16:26 investigation (`docs/M2_GPU_VISIBILITY.md`):
a per-invocation `/dev` tmpfs plus a seccomp `ioctl` allow-list. No further bypass
was attempted — the instruction forbids it and the mechanism is already proven.

Preflight conditions the supervisor enforces on the executing side:

| Condition | Gate | Current agent session | Researcher terminal (expected) |
|---|---|---|---|
| GPU probe | `nvidia-smi` rc 0 | rc 9 → `BLOCKED_NO_GPU`, exit 8 | rc 0 (A0/A1 passed there) |
| Free VRAM | ≥ 3072 MiB before start | n/a | A1 saw 4325–4420 MiB free |
| Conflicting process | `--query-compute-apps` recorded | n/a | to be read from `preflight.compute_apps` |
| Pinned source | blob SHA-1 == `66802de0…` | verified | re-verified at launch |
| Wall / VRAM / storage / progress | 1200 s / 1536 MiB / 300 MB / 120 s | armed | armed |

---

## 4. Phase 4 — the exact command (ready to run, not run)

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction

# 0) GPU sanity (read-only)
./.venv-maniskill/bin/python scripts/gpu_visibility_probe.py

# 1) one bounded pilot; the supervisor owns the kill switches
mkdir -p results/raw
./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py 2>&1 | tee results/raw/m2_stage_b_console.log
echo "pilot_exit=${PIPESTATUS[0]}"
```

The supervisor expands to (verified by `--dry-run`):

```
.venv-maniskill/bin/python /home/ubuntu2004/.cache/ms-upstream-ppo/ppo.py
  --env-id PickCube-v1 --exp-name PickCube-v1__ppo__9351__stageB
  --num-envs 128 --num-steps 50 --total-timesteps 256000
  --update-epochs 4 --num-minibatches 32 --num-eval-envs 8 --eval-freq 5
  --seed 9351 --no-capture-video --no-track --save-model
```

If a hard external backstop is wanted on top of the supervisor's own 1200 s clock:

```bash
timeout --signal=TERM --kill-after=20s 1260 \
  ./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py
```

Exit codes: `0` PASS (40/40 iterations), `5` FAIL (child error / partial), `8`
BLOCKED (no GPU / low VRAM / ppo blob mismatch), `9` KILLED (a resource violation).

### Kill switches (all active in the supervisor)

| Trigger | Threshold | Checked |
|---|---|---|
| Wall clock | 1200 s | every 2 s (monotonic) |
| Device free VRAM | < 1536 MiB | every ~5 s (`nvidia-smi`) |
| No iteration progress | 120 s without an `Epoch:` line | every 2 s |
| Output storage | > 300 MB | every ~5 s (`du` on the run dir) |
| Non-finite loss | NaN/Inf in last `policy_loss`/`value_loss`/`approx_kl` | every ~5 s (event file) |
| CUDA OOM | child exits non-zero | captured, log preserved, **no retry** |

On any trigger the child is `SIGTERM`ed, the reason is written to
`results/m2_pilot_summary.json` before exit, and the checkpoint/event files already
on disk are left untouched.

---

## 5. Phase 5 — metrics

**No row below was measured: the pilot never started (0 iterations).**
The field names are frozen now so they cannot be chosen after the fact.

| # | Metric | JSON key | Value |
|---|---|---|---|
| 1 | Total training environment steps | `steps.train_env_steps` | NOT_MEASURED (expected 256,000) |
| 2 | Evaluation environment steps (separate) | `steps.eval_env_steps` | NOT_MEASURED (expected 3,200) |
| 3 | Completed iterations | `iterations_completed` | NOT_MEASURED (expected 40) |
| 4 | Optimizer updates | `steps.optimizer_updates_upper_bound` | NOT_MEASURED (bound 5,120) |
| 5 | Training-loop SPS | `sps.loop_from_stdout` / `charts/SPS` | NOT_MEASURED |
| 6 | **End-to-end SPS** | `sps.end_to_end` | NOT_MEASURED |
| 7 | Rollout time per iteration | `tensorboard["time/rollout_time"]` | NOT_MEASURED |
| 8 | PPO update time per iteration | `tensorboard["time/update_time"]` | NOT_MEASURED |
| 9 | Peak device VRAM | `vram.peak_used_mib` | NOT_MEASURED |
| 10 | Minimum free VRAM | `vram.min_free_mib` | NOT_MEASURED |
| 11 | Policy loss | `tensorboard["losses/policy_loss"]` | NOT_MEASURED |
| 12 | Value loss | `tensorboard["losses/value_loss"]` | NOT_MEASURED |
| 13 | Approx KL | `tensorboard["losses/approx_kl"]` | NOT_MEASURED |
| 14 | Clip fraction | `tensorboard["losses/clipfrac"]` | NOT_MEASURED |
| 15 | Explained variance | `tensorboard["losses/explained_variance"]` | NOT_MEASURED |
| 16 | Evaluation success rate | `tensorboard["eval/success_once"]` | NOT_MEASURED |
| 17 | Checkpoints | `checkpoints` | NOT_MEASURED (expected 8 + 1) |
| 18 | Exit code / verdict | `returncode`, `status` | NOT_MEASURED |

Expected artifact layout on a successful run:

```
runs/PickCube-v1__ppo__9351__stageB/
  events.out.tfevents.*        # all scalars, a few hundred KB
  ckpt_1.pt  ckpt_6.pt  ckpt_11.pt  ckpt_16.pt
  ckpt_21.pt ckpt_26.pt ckpt_31.pt  ckpt_36.pt
  final_ckpt.pt                # 8 periodic + 1 final = 9
results/raw/m2_stage_b_pilot.log          # full child stdout
results/raw/m2_stage_b_console.log        # supervisor console
results/m2_pilot_summary.json             # all of the above, parsed
```

### 5.1 Expected throughput (arithmetic only, NOT a measurement)

Stage A1 measured 6,437.4 simulator SPS at 128 envs. Stage B adds a forward pass per
step, a backward pass per minibatch, and 8 evaluations. So `SPS_end_to_end < 6,437`
by an unknown factor, and the wall clock is bounded above by the 1200 s ceiling:

| If end-to-end SPS is | 256,000 steps take | Fits 1200 s? |
|---|---:|---|
| 6,437 (A1, unattainable — no gradients) | 40 s | yes |
| 3,000 | 85 s | yes |
| 1,500 | 171 s | yes |
| 500 | 512 s | yes |
| 200 | 1,280 s | **no — would be killed** |

The 1200 s limit therefore only bites below ~213 SPS, i.e. it is a safety net, not a
binding constraint. **All five rows are assumptions, not data.**

---

## 6. Phase 6 — scientific interpretation

Classification: **ADAPTATION / PLUMBING_ONLY**.

Not licensable from this pilot, even after it runs:

- ❌ official baseline reproduction — different script (`ppo.py` vs `ppo_fast.py`),
  different env count (128 vs 4096), different horizon (50M vs 256k), no cudagraphs,
  one seed instead of three;
- ❌ PPO convergence, or that PPO "failed" — 256k steps ≈ 0.5 % of the official
  budget, and a success rate near zero on PickCube at this length is expected;
- ❌ any comparison with published numbers;
- ❌ simulator SPS reported as training SPS, or the reverse.

Licensable after it runs: the pipeline executes end to end within a fixed budget, the
observed peak VRAM under `rgb_array`, the measured end-to-end SPS order of magnitude,
and whether local continuation is worth it at all.

---

## 7. Deviations from the upstream baseline (all deliberate)

| Deviation | Upstream | Here | Why |
|---|---|---|---|
| Env count | 512 (script default) / 4096 (official fast) | **128** | 8 GiB card; A1 → halve |
| Total steps | 10M / 50M | **256,000** | plumbing only |
| `eval_freq` | 25 | **5** | more checkpoints inside a short run |
| Seeds | 9351/4796/1788 | **9351 only** | one bounded run |
| Video | `capture_video=True` | **off** | avoids `RecordEpisode` overhead/VRAM |
| Tracking | off by default | off | wandb not installed |
| `render_mode` | `rgb_array` (hardcoded) | **same, untouched** | changing it would alter upstream; documented instead |

Nothing in the algorithm, optimizer, GAE, reward, or network architecture was modified.

---

## 8. Failure analysis

No failure to analyse yet — the pilot did not start. The only failures in this phase
were on the *preparation* path and are recorded in `docs/FAILURE_LOG.md`:
a broken docstring/CLI mismatch in the old proposal (fixed), and the agent-session GPU
block (root-caused at 16:26, not fixable from inside).

## 9. Next actions

1. Researcher runs §4 in a GPU-visible terminal. Expected runtime: well under 1200 s.
2. Paste `results/m2_pilot_summary.json`; §5 is then filled from it and the verdict
   becomes PASS / PARTIAL / FAIL / KILLED.
3. Only then: decide whether local 50M-step runs are viable, using the measured
   end-to-end SPS (§5.1 formula, not the simulated one).
4. No second attempt and no 256-env variant without a new authorization.
