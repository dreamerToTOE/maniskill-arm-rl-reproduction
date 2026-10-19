# Upstream & Dependency Audit / 依赖与原始版本锁定

Audit date 2026-10-19. **PROPOSED SOURCE PIN / NOT LOCALLY INSTALLED**.

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
| SAPIEN | Linux >=3.0.0 | setup.py metadata only | EXACT_PIN_PENDING |
| Gymnasium | >=0.29.1 | setup.py metadata only | EXACT_PIN_PENDING |
| Python | >=3.9; candidate 3.10/3.11 | setup.py metadata; host unobserved | HOST_PENDING |
| Torch + CUDA runtime | choose compatible wheel after host audit | install docs | HOST_PENDING |
| GPU | RTX4070 historical, current VRAM unknown | prior user report only | HOST_PENDING |
| Assets/demos | no download authorized | license review pending | PENDING |
| Trained baseline/checkpoint | none | no training executed | NOT_RUN |

Additional source and issue risk details: docs/M0_UPSTREAM_AUDIT.md.
