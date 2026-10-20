# M1 isolated installation & bounded smoke — APPROVAL REQUEST (NOT EXECUTED)

Date 2026-10-20. User-provided host audit **RTX 3070 8192 MiB total / 5957 MiB free**, driver **580.178.04**, 418G disk free on root filesystem, Python **3.10.12**, system **pip 22.0.2**; `torch`/`sapien`/`gymnasium`/`mani_skill` not found in this Python interpreter.

**Current approval: PENDING**. Documentation and preparing a task script do not authorize installing or simulating on the student's local host.

## Scope and candidate versions (review prior to execution)

- New isolated local Python venv: `.venv-maniskill/` within **only** this repository (gitignored).
- Python 3.10.12 from host. Keep the system pip/driver/CUDA/IsaacSim/ROS2 and all other repositories unchanged. No sudo or apt/system changes.
- PyTorch candidate: `torch==2.5.1` CUDA 12.1 wheel from official PyTorch wheel index; this is a **proposed** release, not tested with ManiSkill here. Driver 580.178.04 should be new enough in principle; run local test to confirm.
- `mani_skill==3.0.1` from published PyPI wheel; upstream source reference is separately pinned to `a4a4f9272ad64b1564035874b605ceb687b63ed8`. These pins **do not establish binary source SHA identity**; verify actual import version and record wheel/hashes in manifest.
- `sapien==3.0.0`, whose CPython 3.10 Linux manylinux_2_28 wheel is published; `gymnasium==1.1.1` and `numpy==1.24.4` are **tentative** compatible versions, not verified together with transitive dependencies. If solver conflicts, STOP and propose a minimally changed pin with evidence; do not force packages.
- Baseline data/checkpoints/datasets and GPU training are **out of scope**.

Official reference: https://pytorch.org/get-started/previous-versions/ ; https://pypi.org/project/mani-skill/ ; https://pypi.org/project/sapien/3.0.0/ ; https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/setup.py .

Published upstream individual wheel sizes: ManiSkill **101.7 MB**, SAPIEN **51.3 MB**; total download and installed footprint will be larger due to Torch/CUDA and transitive deps. **Proposed hard budget:** stop if downloads projected > 5 GB or disk growth > 12 GB; request separate approval to exceed. GPU training allocation 0; CPU smoke does not imply no GPU/render initialization allocation (measure if unexpectedly happens).

## Proposed commands — ONLY AFTER EXPLICIT APPROVAL

```bash
# In maniskill-arm-rl-reproduction (after confirming git status and path)
python3 -m venv .venv-maniskill
source .venv-maniskill/bin/activate
python -m pip install --upgrade pip
# Torch wheel source: official PyTorch CDN. Torch only; no unnecessary vision/audio.
python -m pip install --no-cache-dir 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
# Pin published simulation framework and basic env deps. Stop if resolver conflicts.
python -m pip install --no-cache-dir 'mani_skill==3.0.1' 'sapien==3.0.0' 'gymnasium==1.1.1' 'numpy==1.24.4'
python -m pip check
python - <<'PY'
import importlib.metadata as md
import torch, sapien, gymnasium
print({x: md.version(x) for x in ['torch','sapien','gymnasium','mani-skill','numpy']})
print('torch CUDA runtime:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
PY

# M1 CPU-only, 1 env, exactly <=3 zero actions, no optimizer or training.
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 python scripts/m1_inspect_pickcube.py \
  --obs-mode state_dict --action-mode zero --steps 3 --seed 0 \
  --output results/m1_pickcube_state_dict_zero_seed0.json
```

If venv creation cannot bootstrap pip, STOP; do not run `sudo apt install python3-venv` without authorization. If any dependency install or import unexpectedly changes system paths or asks to download assets, STOP and report. If the CUDA torch candidate is incompatible, STOP and report; no generic `pip install -U torch` substitution. No benchmark, checkpoint fetch, random rollout, extra smoke variants, large simulation or PPO/SAC updates.

## What Codex must return to the learner

1. Actual local path, chosen Python/pip and installed version/torch CUDA report, pip freeze or environment lock file, resolver warnings, `pip check`, disk delta and NVIDIA free VRAM snapshot.
2. Exact CPU smoke command/exit code, stdout and `results/m1_pickcube_state_dict_zero_seed0.json`, observation keys/shapes, action bounds and dimensions, reward/success/done flags; any missing result explicitly `NOT_RUN`.
3. Diff of `docs/UPSTREAM_LOCK.md` and `docs/PROGRESS.md` reflecting observed evidence. No success claim if import/runtime fails.
4. Stop and report after at most three distinct attempts at a single issue. No training until separately approved.

## GitHub/CodeX English handoff

> Before executing, check that the researcher has explicitly approved **isolated Python wheel installation and one bounded CPU PickCube smoke test**. If approval is absent, **do not execute**. Read AGENTS.md, docs/HUMAN_CODEX_WORKFLOW.md, docs/M1_START_HERE.md, and this document. Review current host facts, repo state, and PyPI/Torch release metadata. Under approval, work exclusively in a new gitignored local venv, never system environments or FR3 repos. Use the pinned ManiSkill source as a reference and report the installed wheel versions separately. Limit smoke to one CPU `PickCube-v1` environment and three zero actions; never train an agent. Record actual stdout, dependency manifest, disk usage, and outcomes, or precise reasons for blocking.
