# Upstream & Dependency Audit / 依赖与原始版本锁定

Audit date 2026-10-19; user-supplied read-only Ubuntu audit 2026-10-20. **PROPOSED SOURCE PIN / NOT LOCALLY INSTALLED**.

## Suggested immutable source
- Repository: https://github.com/mani-skill/ManiSkill (historically haosulab/ManiSkill)
- Git release/tag v3.0.1 (2026-04-21) → commit a4a4f9272ad64b1564035874b605ceb687b63ed8; tree dc931fb9ea2f7c039623d1d419767fae56f217ae
- Source package setup.py version 3.0.1, Python >=3.9, SAPIEN >=3.0.0 on Linux, Gymnasium >=0.29.1, NumPy >=1.22; this is **minimum package metadata**, not a tested installed exact lock.
- Official setup/installation docs propose installing torch separately compatible with local GPU/driver; do not indiscriminately run pip --upgrade against system environment.
- Source license Apache-2.0 at https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE; check https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE-3RD-PARTY plus separately downloaded model/assets/demonstration licensing before redistribution.
- Paper https://arxiv.org/html/2410.00425v2 ; PPO/SAC baseline locations in docs/M0_ALGORITHM_WALKTHROUGH.md.

## Environment — read-only first
User previously reported Ubuntu 22.04.5, RTX 4070, driver 570.211.01, CUDA 12.8, ROS2 Humble / Isaac Sim in a separate setup. **Historical user report, not measured by this M0 session**; GPU VRAM model-specific/available memory, nvidia-smi state, current driver and Python/PyTorch/SAPIEN/Torch CUDA wheel are TBD. No local shell access to user host was used. Do not assume CUDA Toolkit and PyTorch CUDA runtime version must match exactly.
Preferred separately created venv/Conda, candidate Python 3.10/3.11; exact torch wheel + sapien pins are PENDING tested compatibility. CUDA driver, system toolkit, ROS2, Isaac Sim must stay untouched.

## Actual learner-host audit — 2026-10-20 UTC 01:38:29 (user-provided stdout)
From `python3 scripts/audit_environment.py`, supplied by the learner. This is **real user-provided shell output**, not a command run by the assistant or Codex:
- CPython **3.10.12**; `Linux-6.8.0-138-generic-x86_64-with-glibc2.35`. OS distribution release not independently verified.
- `NVIDIA GeForce RTX 3070`, total **8192 MiB**, driver **580.178.04**. This supersedes any earlier assumption of RTX 4070 for **this host/environment**. Do not infer that prior hardware did not exist.
- In this Python interpreter: `numpy==1.24.4`; package metadata lookup reports **torch, gymnasium, mani_skill, sapien = null**. This does not prove these packages are absent in other Conda/venv interpreters.
- GPU **free VRAM unknown** (audit reports total only). Disk free, `python3 -m pip --version`, `vulkaninfo` and current torch CUDA wheel unknown; host install has not started.
- Relative compatibility: package metadata of ManiSkill v3.0.1 specifies Python >=3.9 and NumPy >=1.22, Gymnasium >=0.29.1, SAPIEN >=3.0.0 on Linux. CPython 3.10, glibc 2.35 satisfy relevant wheel tags for SAPIEN v3.0.0's Linux cp310 manylinux_2_28 wheel. **Wheel/host compatibility is preliminary, not an executed integration test.**
- PyTorch official previous versions documents CUDA 12.1 wheels for `torch==2.5.1`; **one tentative candidate only** for an isolated environment, not validated with ManiSkill 3.0.1 or SAPIEN 3.0.0. Do not silently choose or mix torch packages in a global Python install.
- Source: https://pypi.org/project/sapien/3.0.0/ ; https://pypi.org/project/mani-skill/ ; https://pytorch.org/get-started/previous-versions/

**Next read-only checks before any installation:** `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv`, `df -h .`, `python3 -m pip --version`, `git status --short` and verify existing virtual environment paths. Explain whether another Python env already includes Torch; never assume `null` means machine-wide absence.

**Install authorization gate:** propose creating `.venv-maniskill` **inside this independent repo**, no CUDA/driver/ROS2/IsaacSim upgrade. Request user permission to download/install user-space Python wheels and run CPU 3-step test. Do not run until approved. Planned smoke is 1 env, 3 zero-actions, `physx_cpu`, `state_dict`, no renderer or RL optimizer. Network/disk footprint: PyPI ManiSkill 3.0.1 wheel ~101.7 MB and SAPIEN 3.0.0 CPython3.10 wheel ~51.3 MB; CUDA Torch and nvidia libraries add substantially more and depend on solver. Check free disk and show package plan first; do not promise total disk/VRAM without measurement.

## Noninvasive audit requests for local agent (do NOT auto-run install)
- nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
- nvidia-smi
- python3 --version ; which python3 ; python3 -m pip --version
- python3 -c 'import importlib.util; print({k: importlib.util.find_spec(k) is not None for k in ["torch","sapien","mani_skill","gymnasium"]})'
- If dependencies already in chosen environment, print installed torch.version.cuda, torch.__version__, sapien.__version__, torch.cuda.is_available().
- If present in our repo: python scripts/audit_environment.py (read-only); otherwise do not pretend the script exists.
Only after review: estimate disk/assets and GPU memory and obtain explicit smoke permission. Suggested first 1-env CPU state-only reset/step (<5 min once dependencies exist); no weights, no visual datasets, no long GPU training.

## Flags and evidence
| Component | Version proposed / known | Verified scope | Status |
|---|---|---|---|
| ManiSkill upstream | tag v3.0.1 / a4a4f9272ad64b1564035874b605ceb687b63ed8 | GitHub commit + task/baseline files | SOURCE_VERIFIED / NOT_CLONED |
| SAPIEN | 3.0.0 candidate, missing in active interpreter | setup.py + published cp310 Linux wheel | INSTALL_PENDING / INTEGRATION_UNTESTED |
| Gymnasium | >=0.29.1, missing in audited interpreter | upstream setup.py + audit | INSTALL_PENDING |
| Python | 3.10.12 confirmed for active interpreter | Learner audit 2026-10-20 and setup.py minimum | HOST_VERSION_CONFIRMED |
| Torch + CUDA runtime | absent in audited interpreter; 2.5.1/cu121 possible candidate | Learner audit + official torch archived wheels; full compatibility untested | PIN_AND_INSTALL_PENDING |
| GPU | RTX 3070, 8192 MiB total, driver 580.178.04 | Learner audit 2026-10-20; free VRAM unknown | HOST_DEVICE_CONFIRMED / FREE_PENDING |
| Assets/demos | no download authorized | license review pending | PENDING |
| Trained baseline/checkpoint | none | no training executed | NOT_RUN |

Additional source and issue risk details: docs/M0_UPSTREAM_AUDIT.md.
