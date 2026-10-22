# M2 Stage B — PPO Pilot (plumbing validation)

> **Status: `PASS` — 40/40 iterations, exit 0, 106.65 s, 9 checkpoint files.**
> The pipeline is validated end to end on the pinned upstream script: sampling → rollout
> → GAE → PPO update → evaluation → TensorBoard → checkpoint.
> **End-to-end SPS 2400.4**, loop SPS 2600, peak device VRAM **5137 MiB**, min free 2682 MiB.
> 256,000 training env steps + 3,200 evaluation env steps, both counted separately.
>
> Date: 2026-10-21 · Attempt 1 (20:51) was killed at 29/40 by a supervisor buffering bug;
> attempt 2 (21:08) is the clean run recorded here. Both in the researcher's terminal —
> the agent session still has no GPU (`docs/M2_GPU_VISIBILITY.md`).
> Base commit: `63520da` · Classification: **ADAPTATION / PLUMBING_ONLY**
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

## 3. Phase 3 — GPU environment and how the runs were launched

The agent session cannot reach the GPU (`NO_NODES`, exit 3), so both attempts ran in the
researcher's terminal. Command (attempt 2, verbatim):

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
rm -rf runs/PickCube-v1__ppo__9351__stageB
./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py 2>&1 | tee results/raw/m2_stage_b_console.log
echo "exit=${PIPESTATUS[0]}"
```

Preflight recorded by the supervisor (attempt 2):

| Condition | Required | Observed | Verdict |
|---|---|---|---|
| `nvidia-smi` | rc 0 | rc 0 | pass |
| Free VRAM at start | ≥ 3072 MiB | **6460 MiB** (attempt 1: 6460) | pass |
| Conflicting compute apps | none heavy | `gnome-remote-desktop-daemon`, 157 MiB | pass (desktop, not training) |
| `ppo.py` blob SHA | `66802de0…` | `66802de0…` | pass |

---

## 4. Phase 4 — what happened

### 4.1 Attempt 2 (21:08) — PASS

```
Epoch: 1 … Epoch: 40, global_step=249600
model saved to runs/…/final_ckpt.pt
status=PASS iterations=40 wall=106.65s e2e_sps=2400.4   exit=0
```

40/40 iterations, 8 evaluations at iterations 1/6/11/16/21/26/31/36, **9 checkpoint
files** (8 periodic + `final_ckpt.pt`), 9.96 MB of output. The stdout now arrives line by
line, so the two-phase watchdog saw every `Epoch:` line and never fired.

### 4.2 Attempt 1 (20:51) — KILLED at 29/40, kept as the record of the bug

Supervisor killed it at 122.36 s reporting `no iteration progress for 120s`,
`iterations=0`, `e2e_sps=0.0`. **That report was wrong**: the artifacts showed 29/40
iterations (`time/step` 185,600), 6 checkpoints, a 30 KB event file — and a **0-byte
child log**.

Root cause: `subprocess.Popen(..., stdout=PIPE)` makes the child's stdout a pipe, so
CPython block-buffers it (4–8 KB); the `Epoch:` lines never reached the watchdog and
`SIGTERM` discarded the buffer. Compounding it, the 120 s rule was clocked from `t0`,
and that run's **cold start** (first-ever `rgb_array` render-target compile plus two
PhysX scene builds) took ≈50 s.

Fixed before attempt 2: `python -u` + `PYTHONUNBUFFERED=1`; two-phase watchdog (300 s
grace until the first output line, then 120 s/iteration, 1200 s wall on top); iteration
count reconciled as `max(stdout, event file)`; `--harvest` can rebuild metrics from
artifacts if stdout is ever lost again. **Warm start is only 8.2 s** (106.65 s total −
98.5 s of loop), which is why the cold-start cost is now documented separately rather
than folded into "startup".

---

## 5. Phase 5 — measurements (attempt 2, 40/40)

Machine-readable: `results/m2_pilot_summary.json` (`status=PASS`, `harvested` absent —
these are the live values).

| # | Metric | Value |
|---|---|---|
| 1 | Training environment steps | **256,000** (40 × 6,400) — exactly the authorized budget |
| 2 | Evaluation environment steps (separate) | **3,200** (8 evaluations × 50 steps × 8 envs) |
| 3 | Completed iterations | **40 / 40** |
| 4 | Optimizer updates | ≤ **5,120** (40×4×32 upper bound; `target_kl` can break early, upstream prints no count) |
| 5 | Training-loop SPS (`charts/SPS`) | last **2600**, mean **2575.5**, min 1701 (first iteration), max 2818 |
| 6 | **End-to-end SPS** | **2400.4** = 256,000 / 106.65 s |
| 7 | Rollout time per iteration | mean **1.617 s** (min 1.238, max 1.798) |
| 8 | PPO update time per iteration | mean **0.444 s** (min 0.155, max 0.510) |
| 9 | Peak device VRAM | **5137 MiB** (73 % of 8192) |
| 10 | Minimum free VRAM | **2682 MiB** (floor 1536 — never close) |
| 11 | Policy loss | last **+0.0276**, mean +0.0046, range −0.0467…+0.0525 |
| 12 | Value loss | last **0.0202**, mean 0.0129, min 0.0024 |
| 13 | Approx KL | last **0.0313**, mean 0.0397, max **0.1218** (twice above `target_kl=0.1`) |
| 14 | Clip fraction | last **0.398**, mean 0.358 |
| 15 | Explained variance | last **+0.848**, mean 0.635, first **−1.030** |
| 16 | Evaluation success rate | **0.0** on all 8 evaluations |
| 17 | Checkpoints | **9** = 8 periodic + 1 final, 1.156 MB each, **9.92 MB** total |
| 18 | Exit code / verdict | **0 / PASS** |

Evaluation curve (8 points, 3,200 env steps total):

| step | 0 | 32k | 64k | 96k | 128k | 160k | 192k | 224k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eval/return` | 2.760 | 5.877 | 8.044 | 8.030 | 10.194 | **15.134** | 14.812 | 13.116 |
| `eval/reward` | 0.0552 | 0.1175 | 0.1609 | 0.1606 | 0.2039 | **0.3027** | 0.2962 | 0.2623 |
| `eval/success_once` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Every evaluation episode ran the full 50 steps (`eval_episode_len` = 50.0 throughout), so
the return is a genuine 50-step return, not a shortened one.

**Where the three SPS numbers come from:**

| Figure | Value | What it includes |
|---|---:|---|
| rollout fps (`time/rollout_fps`) | 3,978 mean | sampling only (6,400 env-steps ÷ rollout_time) |
| `charts/SPS` | 2,600 | sampling + GAE + 128 optimizer steps + logging; excludes import/build |
| end-to-end | 2,400 | the whole child: import, 2 scene builds, 8 evaluations, everything |

Warm start = **8.2 s**, i.e. 7.7 % of wall clock. At 50M steps that amortises to nothing.

---

## 6. Phase 6 — scientific interpretation

Classification: **ADAPTATION / PLUMBING_ONLY**. Established:

- ✅ the full loop runs on the pinned upstream `ppo.py`, unmodified: 256,000 env steps,
  40 PPO updates, 8 evaluations, TensorBoard logging, 9 checkpoints, exit 0, inside a
  1200 s budget with 106.65 s used;
- ✅ **VRAM under the real training path: 5137 MiB peak** (`render_mode="rgb_array"`,
  128 train + 8 eval envs) — **+1206 MiB (+31 %) over A1's render-free 3931 MiB**,
  confirming the pre-flight audit; 2682 MiB still free, so 128 envs is safe on 8 GiB;
- ✅ numerically sane optimisation: explained variance **−1.03 → +0.85**, value loss
  bounded in [0.002, 0.030], entropy 7.35 → 6.13, KL mostly below the 0.1 target.

Not established, and must not be claimed:

- ❌ **no success**: `eval/success_once` = 0 on all 8 evaluations. At 0.51 % of the
  official 50M budget this is the expected outcome, not a verdict on PPO. (One training
  step at global_step 200,448 logged `train/success_once` = 1.0 — a single completed
  episode did succeed — but 8 eval envs × 1 episode each can easily miss a rare success,
  and one event is not a rate.)
- ❌ baseline reproduction: `ppo.py` not `ppo_fast.py`, 128 envs not 4096, 256k not 50M,
  one seed not three, no cudagraphs;
- ❌ convergence, final policy quality, or any comparison with published numbers;
- ❌ that this SPS generalises: it is one run on one desktop card with a desktop session
  resident.

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
| Did PPO fail? | No. 40/40 iterations, finite losses, explained variance rising, 9 checkpoints. |
| Did the GPU fail? | No. Peak 5137 MiB, min free 2682 MiB, no OOM, no NaN. |
| Did the harness fail? | **Yes, on attempt 1 only** — piped stdout was block-buffered (0-byte log) and the 120 s rule was clocked from `t0` over a ≈50 s cold start. Fixed; attempt 2 passed with no incidents. |
| Class | (A) local implementation — instrumentation, never the experiment |
| Evidence | attempt 1: 0-byte child log vs 30 KB event file + 6 ckpts; attempt 2: 40/40 with line-by-line stdout |
| Retry policy | attempt 1 consumed the authorization; attempt 2 was explicitly approved. No further run is planned |

---

## 9. Next actions

1. **Local viability, from measured numbers** (128 envs, `ppo.py`, seed 9351):

   | Basis | 50M steps / seed | 3 seeds |
   |---|---:|---:|
   | loop SPS 2600 | **5.34 h** | **16.0 h** |
   | end-to-end 2400 (this run) | 5.79 h | 17.4 h |

   At 128 envs a 3-seed PickCube run is **~16–17 h** on this card — feasible locally
   within the "under ~2 days" bar, with no cloud rental needed. The official 4096-env
   configuration would be far faster but does not fit in 8 GiB.

2. **If throughput matters more than locality**, the lever is env count, not the GPU.
   **This is now measured**: the 256-env pilot (2026-10-22) ran 20/20 at loop SPS 4435
   (+71 % over 128) with **flat VRAM** (peak 4601 MiB, min free 3217 MiB) — see
   `docs/M2_STAGE_B_256_PILOT.md`. 256 envs ≈ **3.13 h/seed, ~9.4 h / 3 seeds**.

3. **Do not** raise `total_timesteps` and call it a baseline reproduction: that still
   requires `ppo_fast.py`, 4096 envs, the official 4-step/8-epoch schedule, cudagraphs,
   and three seeds — none of which this stage measures.

4. Success rate stays untested at this budget. Any learning claim requires the full
   50M-step run, which needs its own approval.
