# Local-first ManiSkill reproduction → optional cloud GPU migration

**Decision recorded 2026-10-20.** Keep developing and debugging locally on the user's RTX 3070 8GB. After approved local smoke and any **separately authorized** bounded benchmarks, migrate the reproducible experiment package to a 24GB-class or suitable cloud GPU **only when measured time/VRAM justifies cost**. This is a workflow decision, not cloud rental or training authorization.

## A. What 'package the project' actually means

**DO include** (version-controlled unless too large):
1. This repository's source scripts and exact git commit; separately record pinned original ManiSkill v3.0.1 commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`, local source/wheel origin and hashes.
2. Configuration files with **named task/env/controller/reward/observation**, max steps, random seeds, GPU/CPU simulator backend, control frequency, batch size, horizon and PPO/SAC hyperparameters, evaluation settings, and logging paths. Do not bake hardware assumptions into algorithms.
3. An **actual post-install** reproducible Python lock/manifest (`pip freeze` + Python/Torch/CUDA/SAPIEN/ManiSkill versions, `pip check`, distribution wheel hashes where feasible). Do not fabricate a lock based solely on proposed requirements. Optionally build/test a Dockerfile later if it reduces deployment risk; not a prerequisite for M1.
4. A one-command or documented CLI entry point for *approved* smoke, training, evaluation and export, separating modes and prohibiting accidental long training.
5. `manifest.json` per run: git SHAs, machine GPU/driver, relevant package versions, task/controller IDs, experiment config, seed, env-steps, walltime/SPS, peak VRAM, reward/success definitions, exit status and result/artifact paths.
6. Unit checks and run logs; plots, CSV metrics and small reports. Pin assets/dataset versions and licenses independently; use explicit remote data paths if needed.

**DO NOT bundle or push as normal git files**:
- the whole local Ubuntu installation, NVIDIA drivers/CUDA Toolkit, ROS2/IsaacSim, or the `.venv-maniskill` directory (platform/environment-specific, large, not portable).
- Hugely sized replay buffers, checkpoints, videos, generated USD/assets, raw trajectories or secrets/API credentials. Transfer needed files using `rsync/scp` or an approved artifact store, with checksums and version/source metadata.
- system-specific absolute filesystem paths, CUDA device indexes or hardcoded local hostnames; use configurable `DATA_ROOT`, `OUTPUT_ROOT`, `DEVICE` or equivalent.

## B. Preserve progress correctly

There are **two distinct migration cases**:

1. **Before serious training** (preferred): move code/config/asset manifest, reconstruct cloud env from hashes, verify one CPU/GPU smoke and bounded simulator behavior; launch fresh cloud seeds after explicit approval. This is simple and easy to audit.
2. **Resuming an actual local training run** (more complex): an Actor-only `.pt` is *not* a true training-resumption checkpoint.
   - PPO: persist policy + critic, optimizer state, global env steps, scheduler settings, seeds/RNG states, and the code's intended rollout/update boundaries.
   - SAC: persist actor, both online critics + target critics, all optimizer states, temperature `alpha` and optimizer state, replay buffer with cursor, steps/update counters and RNG state. Replay contents may be large; estimate transfer cost first.
   - GPU/PhysX or dependency variation can prevent bitwise continuation: record differences, verify evaluation separately and do not assert identical training behavior.

If full checkpoint state is not implemented by the reference baseline, document `WEIGHTS_ONLY` and **restart an independent seed** rather than falsely claiming exact continuation.

## C. Decision to migrate (measured, not guessed)

Complete approved isolated local install and 3-step CPU PickCube smoke first; actual outcome still `NOT_RUN` as of decision. Then request separate bounded approval for GPU simulator/profile and short full-training pilot.

Collect **complete training throughput** (`environment steps / wall second`) including optimizer updates, and peak memory, task success diagnostics and eval overhead. Compare official 50M/75M PPO reference steps and SAC 500k reference steps against projected local walltime. Consider migrating if local baseline needs >4-6h per seed, reference env count OOMs, multi-seed aggregate takes days, or vision inputs exceed 8GB. This is a heuristic, not a strict fact.

Choose cloud GPU based on actual workload peak memory (often 24GB-class for reference high-parallel/vision work, if justified), GPU-hour quote, data-egress, availability, expected walltime and reproducible CUDA/SAPIEN compatibility. Cloud provider/tool choice later, after approval. Do not silently change PPO vs SAC controller mode for 'performance comparisons'; record controller confounds.

## D. Future migration acceptance checklist

- [ ] Source repos pinned by exact commit; experiment config and seed logged
- [ ] Exact installed dependency manifest captured; remote env installed in isolation and passes `pip check`
- [ ] Assets/datasets explicitly versioned, licensed, checksum verified; no missing downloads
- [ ] Local approved minimal CPU smoke passed and cloud minimal smoke re-validated
- [ ] Training and evaluation CLI accept explicit output/data directories with no hardcoded local paths
- [ ] Before cloud run, estimate cost (hourly rate x expected run time), VRAM headroom and set budget
- [ ] Experiment manifests and checkpoints produced according to actual reference support; `weights-only` distinguished from resumable
- [ ] Each cloud run labeled GPU/driver/CUDA/simulator/seed/steps; compare metrics under matched protocol
- [ ] Researcher **explicitly approves** provision/rental, GPU run configuration, estimated cost and total training budget
- [ ] On completion, archive small reports and transfer large outputs with checksums; no secrets committed

## M1 immediate Codex scope (English)

**For currently approved M1 installation and three-step CPU PickCube smoke only:** keep project environment isolated, avoid host-specific paths, save the actual installed package manifest and source provenance, write evidence to docs/PROGRESS.md. Do NOT spend time building a Docker image or staging cloud infrastructure. Do NOT launch a GPU benchmark or training without new approval. Read this cloud portability plan before structuring any later experiment runner.

Status: **LOCAL_FIRST_DECIDED / CLOUD_MIGRATION_PLAYBOOK_DOCUMENTED / TRAINING_NOT_RUN / CLOUD_NOT_PROVISIONED**.
