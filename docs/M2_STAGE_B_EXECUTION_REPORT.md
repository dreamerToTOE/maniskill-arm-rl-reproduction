# M2 Stage B — PPO Pilot (plumbing validation)

> **Status: `PARTIAL` — the pipeline ran end to end, but the authorized run was cut at
> 29/40 iterations by a supervisor bug, not by a training failure.**
> 185,600 training env steps, 6/8 periodic checkpoints, 6 evaluations, **end-to-end
> SPS 1516.8**, loop SPS 2563, device peak 4937 MiB.
> **The kill was my instrumentation defect**: the child's stdout was block-buffered
> (piped), so the liveness watchdog saw zero lines and fired at 120 s while PPO was
> happily training. Fixed (`-u` + `PYTHONUNBUFFERED=1` + a startup grace).
> A clean 40/40 run is **attempt 2** and needs the researcher's go-ahead.
>
> Date: 2026-10-21 · Attempt 1 executed by the researcher at 20:51–20:53 KST
> Base commit: `6f86e8b` · Classification: **ADAPTATION / PLUMBING_ONLY**
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

## 3. Phase 3 — GPU environment and how the run was launched

The agent session still cannot reach the GPU (`NO_NODES`, exit 3), so attempt 1 was
launched by the researcher in their terminal at 20:51 KST with:

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py 2>&1 | tee results/raw/m2_stage_b_console.log
echo "exit=${PIPESTATUS[0]}"
```

Preflight that the supervisor recorded before launching:

| Condition | Required | Observed | Verdict |
|---|---|---|---|
| `nvidia-smi` | rc 0 | rc 0 | pass |
| Free VRAM at start | ≥ 3072 MiB | **6460 MiB** | pass |
| Conflicting compute apps | none heavy | `gnome-remote-desktop-daemon`, 157 MiB | pass (desktop, not training) |
| `ppo.py` blob SHA | `66802de0…` | `66802de0…` | pass |

So the environment was clean; the failure was entirely on the supervision side (§8).

---

## 4. Phase 4 — what actually happened

The supervisor killed the child at **122.36 s** with the message
`no iteration progress for 120s`, and reported `iterations=0`, `e2e_sps=0.0`.
**That report was wrong.** The artifacts prove the opposite:

| Artifact | Content | Meaning |
|---|---|---|
| `runs/PickCube-v1__ppo__9351__stageB/ckpt_{1,6,11,16,21,26}.pt` | 6 files, 1,155,150–1,155,683 B each | checkpointing worked, at exactly the expected iterations |
| `events.out.tfevents.*` | 30,250 B, 22 scalar tags, 29 logged iterations | TensorBoard logging + full metric record |
| `results/raw/m2_stage_b_pilot.log` | **0 bytes** | the child's stdout never reached the supervisor — block-buffered pipe |
| last logged `time/step` | 185,600 = 29 × 6,400 | **29 of 40 iterations completed** |

Root cause: `subprocess.Popen(..., stdout=PIPE, text=True)` makes the child's stdout a
pipe, and CPython then block-buffers it (4–8 KB). `ppo.py`'s prints ("####", "Running
training", "Epoch: n", "SPS: n") sat in that buffer, so the `Epoch:`-based watchdog
never saw a line, and `SIGTERM` discarded the buffer — hence a 0-byte log.

Two defects, both mine, both fixed:

1. **Buffering.** The child is now launched with `-u` (interpreter flag, not a change to
   `ppo.py`) and `PYTHONUNBUFFERED=1`.
2. **The watchdog clocked startup with the loop rule.** Import + two PhysX scene builds
   + render-target setup took **≈50 s** here (122.36 s wall − 72.4 s of loop), so even
   with output arriving, a 120 s window measured from `t0` is the wrong instrument.
   The watchdog is now two-phase: a **300 s startup grace** until the first output line,
   then the **120 s** loop rule. The 1200 s wall clock still bounds everything.

---

## 5. Phase 5 — measurements (attempt 1, harvested from the artifacts)

Rebuilt with `--harvest`, which reads the event file and the surviving checkpoints
instead of trusting the lost stdout. Machine-readable: `results/m2_pilot_summary.json`.

| # | Metric | Value |
|---|---|---|
| 1 | Training environment steps | **185,600** (29 × 6,400) of the 256,000 authorized |
| 2 | Evaluation environment steps (separate) | **2,400** (6 evaluations × 50 steps × 8 envs) |
| 3 | Completed iterations | **29 / 40** |
| 4 | Optimizer updates | ≤ **3,712** (29×4×32 upper bound; `target_kl` can break early, upstream prints no count) |
| 5 | Training-loop SPS (`charts/SPS`) | last **2563**, mean **2514.3**, first 1451 (cold) |
| 6 | **End-to-end SPS** | **1516.8** = 185,600 / 122.36 s |
| 7 | Rollout time per iteration | mean **1.614 s** (min 1.328, max 1.817) |
| 8 | PPO update time per iteration | mean **0.461 s** (min 0.158, max 0.850) |
| 9 | Peak device VRAM | **4937 MiB** |
| 10 | Minimum free VRAM | **2882 MiB** (floor was 1536 — never close) |
| 11 | Policy loss | last **+0.0151**, mean −0.0009, range −0.0467…+0.0468 |
| 12 | Value loss | last **0.0134**, mean 0.0101, min 0.0024 |
| 13 | Approx KL | last **0.0358**, max **0.1033** (crossed `target_kl=0.1` once) |
| 14 | Clip fraction | last **0.3656**, mean 0.3406 |
| 15 | Explained variance | **−1.03 → +0.832** (mean 0.553) — the critic went from worse than the mean to useful |
| 16 | Evaluation success rate | **0.0** on all 6 evaluations (train `success_once` also 0) |
| 17 | Checkpoints | **6 periodic, 0 final**, 1.156 MB each (6.9 MB total) |
| 18 | Exit code / verdict | child `-15` (SIGTERM), supervisor `9`, **KILLED** |

Supporting detail: `train/return` **2.48 → 9.96** monotone-ish over 29 iterations;
`losses/entropy` 7.35 → 6.42; `time/rollout_fps` mean **3989** (pure sampling);
output storage 6.9 MB of the 300 MB cap.

**Where the three SPS numbers come from, and why they differ:**

| Figure | Value | What it includes |
|---|---:|---|
| rollout fps | 3,989 | sampling only (6,400 env-steps ÷ rollout_time) |
| `charts/SPS` | 2,563 | sampling + GAE + 128 optimizer steps + logging, excludes import/build |
| end-to-end | 1,517 | the whole child: import, 2 scene builds, 6 evals, everything |

The gap between 1,517 and 2,563 is the **≈50 s startup** — which is exactly why the
instruction's warning ("upstream SPS does not include the whole process lifetime") mattered.

---

## 6. Phase 6 — scientific interpretation

Classification: **ADAPTATION / PLUMBING_ONLY**. What attempt 1 *does* establish:

- ✅ environment sampling, GAE, PPO updates, evaluation, TensorBoard logging and
  checkpointing all execute on this host, with the pinned upstream script unmodified;
- ✅ VRAM under the real training path (`render_mode="rgb_array"`, 128 envs + 8 eval envs)
  peaks at **4937 MiB** — **+1006 MiB (+26 %)** over A1's render-free 3931 MiB. That is
  the audit finding from §1.4 measured, and it leaves 2882 MiB free, far above the floor;
- ✅ the loss/return curves move in a sane direction over 29 iterations (explained
  variance −1.03 → 0.83, return 2.48 → 9.96).

What it does **not** establish, and must not be claimed:

- ❌ 40/40 completion or a final checkpoint — the run was truncated by tooling;
- ❌ any success-rate conclusion: **0.0 at 185k steps is expected** (0.37 % of the
  official 50M budget) and says nothing about whether PPO works on PickCube;
- ❌ baseline reproduction (different script, env count, horizon, one seed);
- ❌ PPO convergence, or any comparison with published numbers.

**Is the pipeline validated?** For the objective in the instruction — "sampling → rollout
→ GAE → PPO optimization → evaluation → TensorBoard → checkpoint" — **yes, every stage
produced evidence**. The stage is `PARTIAL` only because the authorized run did not reach
40 iterations and no `final_ckpt.pt` exists.

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
| `render_mode` | `rgb_array` (hardcoded) | **same, untouched** | changing it would alter upstream; measured instead |
| Child buffering | n/a | `python -u` | supervisor visibility; does not touch `ppo.py` |

Nothing in the algorithm, optimizer, GAE, reward, or network architecture was modified.

---

## 8. Failure analysis

| Question | Answer |
|---|---|
| Did PPO fail? | **No.** 29 iterations completed, losses finite, explained variance rising, 6 checkpoints written. |
| Did the GPU fail? | No. Peak 4937 MiB, min free 2882 MiB, no OOM, no NaN. |
| Did the harness fail? | **Yes, twice, both in the supervisor:** piped stdout was block-buffered (0-byte log), and the 120 s liveness rule was applied to the ≈50 s startup window. |
| Class | (A) local implementation — instrumentation, not the experiment |
| Evidence | 0-byte child log vs 30 KB event file + 6 checkpoints; `kill_reason="no iteration progress for 120s"` with `iterations=0` while `time/step` reached 185,600 |
| Fix | `-u` + `PYTHONUNBUFFERED=1`; two-phase watchdog (300 s startup grace, then 120 s); iteration count now reconciled against the event file (`max(stdout, events)`); `--harvest` rebuilds metrics from artifacts if stdout is ever lost again |
| Retry policy | **no automatic retry** — attempt 1 consumed the authorization; attempt 2 requires an explicit go-ahead |

---

## 9. Next actions

1. **Attempt 2 (needs go-ahead).** Same command, fixed supervisor; expect 40/40 in
   ≈170 s and 9 checkpoint files:
   ```bash
   cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
   rm -rf runs/PickCube-v1__ppo__9351__stageB        # start from a clean run dir
   ./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py 2>&1 | tee results/raw/m2_stage_b_console.log
   ```
2. Then §5 is filled with 40/40 numbers, including `final_ckpt.pt` and the 8th evaluation.
3. **Scaling decision, using the measured 1,517 e2e / 2,563 loop SPS** (not the A1
   simulator number): at 128 envs, 50M steps ≈ **5.4 h per seed** by loop SPS (≈9.2 h by
   this run's end-to-end, which is startup-inflated at 122 s), i.e. ≈**16 h for three
   seeds**. The official 4096-env configuration would be far faster but is unmeasured
   here and does not fit in 8 GiB.
