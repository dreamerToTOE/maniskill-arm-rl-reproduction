# Upstream & Dependency Audit / 依赖与原始版本锁定

Audit date 2026-10-19; user-supplied read-only Ubuntu audit 2026-10-20. **UPDATED 2026-10-20: isolated environment INSTALLED and CPU smoke EXECUTED — see `docs/M1_EXECUTION_REPORT.md`.**

## Suggested immutable source
- Repository: https://github.com/mani-skill/ManiSkill (historically haosulab/ManiSkill)
- Git release/tag v3.0.1 (2026-04-21) → commit a4a4f9272ad64b1564035874b605ceb687b63ed8; tree dc931fb9ea2f7c039623d1d419767fae56f217ae
- Source package setup.py version 3.0.1, Python >=3.9, SAPIEN >=3.0.0 on Linux, Gymnasium >=0.29.1, NumPy >=1.22; this is **minimum package metadata**, not a tested installed exact lock.
- Official setup/installation docs propose installing torch separately compatible with local GPU/driver; do not indiscriminately run pip --upgrade against system environment.
- Source license Apache-2.0 at https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE; check https://github.com/mani-skill/ManiSkill/blob/a4a4f9272ad64b1564035874b605ceb687b63ed8/LICENSE-3RD-PARTY plus separately downloaded model/assets/demonstration licensing before redistribution.
- Paper https://arxiv.org/html/2410.00425v2 ; PPO/SAC baseline locations in docs/M0_ALGORITHM_WALKTHROUGH.md.

## Environment — read-only first
Prior conversation referenced an RTX 4070 in another context. **Current user-supplied measurements for the ManiSkill host are RTX 3070 with 8 GiB total, 5957 MiB free, NVIDIA driver 580.178.04**; never merge the two presumed machine configurations. Linux reports kernel 6.8/glibc 2.35, Python 3.10.12, NumPy 1.24.4. Current Torch/SAPIEN/Gymnasium are missing in the inspected interpreter; no shell access to the user's host is provided to ChatGPT. Do not assume CUDA Toolkit and PyTorch CUDA runtime version must match exactly.
Preferred isolated `.venv-maniskill` running Python 3.10.12; proposed package pins PyTorch 2.5.1 cu121, SAPIEN 3.0.0, ManiSkill 3.0.1, Gymnasium 1.1.1 are **candidates pending dependency resolution and local integration**. Source Git SHA pin and PyPI distribution version are related but not equivalent evidence of binary identity. CUDA driver, system toolkit, ROS2, Isaac Sim must stay untouched.

## Actual learner-host audit — 2026-10-20 UTC 01:38:29 (user-provided stdout)From `python3 scripts/audit_environment.py`, supplied by the learner. This is **real user-provided shell output**, not a command run by the assistant or Codex:
- CPython **3.10.12**; `Linux-6.8.0-138-generic-x86_64-with-glibc2.35`. OS distribution release not independently verified.
- `NVIDIA GeForce RTX 3070`, total **8192 MiB**, driver **580.178.04**. This supersedes any earlier assumption of RTX 4070 for **this host/environment**. Do not infer that prior hardware did not exist.
- In this Python interpreter: `numpy==1.24.4`; package metadata lookup reports **torch, gymnasium, mani_skill, sapien = null**. This does not prove these packages are absent in other Conda/venv interpreters.
- GPU **free VRAM now observed: 5957 MiB**. **Disk free: 418G**; global **pip 22.0.2**. Vulkan and chosen torch CUDA wheel remain untested; host install has not started.
- Relative compatibility: package metadata of ManiSkill v3.0.1 specifies Python >=3.9 and NumPy >=1.22, Gymnasium >=0.29.1, SAPIEN >=3.0.0 on Linux. CPython 3.10, glibc 2.35 satisfy relevant wheel tags for SAPIEN v3.0.0's Linux cp310 manylinux_2_28 wheel. **Wheel/host compatibility is preliminary, not an executed integration test.**
- PyTorch official previous versions documents CUDA 12.1 wheels for `torch==2.5.1`; **one tentative candidate only** for an isolated environment, not validated with ManiSkill 3.0.1 or SAPIEN 3.0.0. Do not silently choose or mix torch packages in a global Python install.
- Source: https://pypi.org/project/sapien/3.0.0/ ; https://pypi.org/project/mani-skill/ ; https://pytorch.org/get-started/previous-versions/

**Preflight completed — user-supplied stdout 2026-10-20:**
- `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv` → RTX 3070, **8192 MiB total, 5957 MiB free**, driver **580.178.04**. Free VRAM is a snapshot, not a training allocation guarantee.
- `df -h .` → `/dev/sdb3` 724G size, 270G used, **418G available**, 40% usage, mount `/`.
- `python3 -m pip --version` → **pip 22.0.2**, **system** `/usr/lib/python3/dist-packages/pip`, Python 3.10. Do NOT upgrade this global interpreter; create an isolated environment and bootstrap/upgrade pip there only.
- CPU-only state-smoke can proceed after isolated dependency installation and the learner's approval. Available disk/VRAM do not prove dependencies are installed or compatible. No host commands were executed by ChatGPT itself; all outputs are user-provided.
- `git status --short`, `python3 -m venv --help`, and exact virtual environment availability remain unchecked; these are noninvasive checks for the approved local Codex task.

**Install authorization gate:** **approved by researcher on 2026-10-20** for the precise isolated installation and CPU 3-step smoke documented at `docs/M1_INSTALL_APPROVAL.md`; user must hand instructions to their local Codex (no ChatGPT shell access). This does NOT approve GPU throughput benchmark or actual model training; no CUDA/driver/ROS2/IsaacSim upgrade. PyPI ManiSkill 3.0.1 wheel 101.7 MB and SAPIEN 3.0.0 CPython3.10 wheel 51.3 MB; CUDA Torch and NVIDIA libraries add substantially more. Current disk free 418GB; staged installation cap proposed and subject to approval.

## Noninvasive audit requests for local agent (do NOT auto-run install)
- nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
- nvidia-smi
- python3 --version ; which python3 ; python3 -m pip --version
- python3 -c 'import importlib.util; print({k: importlib.util.find_spec(k) is not None for k in ["torch","sapien","mani_skill","gymnasium"]})'
- If dependencies already in chosen environment, print installed torch.version.cuda, torch.__version__, sapien.__version__, torch.cuda.is_available().
- If present in our repo: python scripts/audit_environment.py (read-only); otherwise do not pretend the script exists.
Only after review: estimate disk/assets and GPU memory and obtain explicit smoke permission. Suggested first 1-env CPU state-only reset/step (<5 min once dependencies exist); no weights, no visual datasets, no long GPU training.

## Actual installed environment — 2026-10-20 (agent-executed, evidence-backed)

Path: `.venv-maniskill/` **inside this repo only** (gitignored, 6,210,349,110 B). Interpreter
`/usr/bin/python3` 3.10.12 (`sys.prefix` = the venv, `is_venv = True`). System pip 22.0.2,
system CUDA/driver, ROS2 and Isaac Sim untouched; no sudo/site-packages change.

| Package | Installed (verified via `importlib.metadata`) | Wheel provenance |
|---|---|---|
| torch | **2.5.1+cu121** (CUDA runtime 12.1, cuDNN 9.1.0) | `download.pytorch.org/whl/cu121`; SHA-256 NOT recorded (wheel not retained) |
| sapien | **3.0.0** | `sapien-3.0.0-cp310-cp310-manylinux_2_28_x86_64.whl`, sha256 `c57837cc6998f42dad1355c2332b1d0e5b815acb438f4f38f250de8e2576cc90` |
| mani-skill | **3.0.1** (PyPI wheel; **not** the git tree) | `mani_skill-3.0.1-py3-none-any.whl`, sha256 `685de2f03c300b1ede49881a1bf6306ad062082d39c8d3be8b8e85603f32e33a` |
| gymnasium | **1.1.1** | sha256 `9c167ec0a2b388666e37f63b2849cd2552f7f5b71938574c637bb36487eb928a` |
| numpy | **1.24.4** | cp310 manylinux x86_64, sha256 `7ffe43c74893dbf38c2b0a1f5428760a1a9c98285553c89e12d70a96a7f3a4d6` |
| pip (in venv) | 26.2.1 | bootstrapped 22.0.2 → upgraded inside venv |

- `pip check` → **No broken requirements found.** The candidate pins resolved on the **first**
  attempt; no resolver conflict, no forced/`--no-deps` install. Full transitive freeze (96 lines):
  `results/m1_requirements.lock.txt`; machine-readable record: `results/m1_environment_manifest.json`.
- **Source pin ≠ binary identity**: `a4a4f927…` remains the *source reference*; the executed code is the
  published 3.0.1 wheel. Version strings agree, hashes of the git tree were not compared to the wheel.
- **GPU caveat**: inside the agent shell `nvidia-smi` returns rc=9 and `/dev/nvidia*` does not exist,
  so `torch.cuda.is_available()` is False there. `/proc/driver/nvidia/version` reports 580.178.04 and
  `lspci` shows `GA104 [GeForce RTX 3070 Lite Hash Rate]`, i.e. the driver is loaded and the card is
  present — the device nodes are simply not exposed to the agent namespace. **CUDA usability in the
  researcher's own terminal is UNVERIFIED by this run.**
- Runtime note (environment, not upstream): `/tmp` on this host is a 10 MB tmpfs, so `TMPDIR` had to be
  redirected for pip. SAPIEN also warned `Failed to create /root/.cache for shader cache` (non-fatal).

## Flags and evidence
| Component | Version proposed / known | Verified scope | Status |
|---|---|---|---|
| ManiSkill upstream | tag v3.0.1 / a4a4f9272ad64b1564035874b605ceb687b63ed8 | GitHub commit + task/baseline files | SOURCE_VERIFIED / PYPI_WHEEL_INSTALLED |
| SAPIEN | **3.0.0 installed** | `import sapien` OK; `Engine()` + `RenderSystem()` construct on CPU | INSTALLED / IMPORT_VERIFIED / CPU_RENDER_FALLBACK |
| Gymnasium | **1.1.1 installed** | import OK; `gym.make` + wrappers verified | INSTALLED |
| Python | **3.10.12 in venv** | `sys.version`, `sys.executable` | VENV_VERIFIED |
| Torch + CUDA runtime | **2.5.1+cu121 installed**, runtime 12.1 | import + version OK; `cuda.is_available()=False` in agent shell | INSTALLED / CUDA_UNVERIFIED |
| GPU | RTX 3070, 8192 MiB total, driver 580.178.04 | `lspci` + `/proc/driver/nvidia/version`; no `/dev/nvidia*` in agent namespace | PRESENT / NOT_ACCESSIBLE_IN_AGENT_SHELL |
| Assets/demos | PickCube needed no extra download (`MS_SKIP_ASSET_DOWNLOAD_PROMPT=1`, no prompt hit) | smoke run | OK_FOR_PICKCUBE |
| Trained baseline/checkpoint | none | no training executed | NOT_RUN |
| CPU smoke evidence | 3 zero actions, seed 0, exit 0 | `results/m1_pickcube_state_dict_zero_seed0.json` | PASSED |

Additional source and issue risk details: docs/M0_UPSTREAM_AUDIT.md.
Execution detail: docs/M1_EXECUTION_REPORT.md.

Additional source and issue risk details: docs/M0_UPSTREAM_AUDIT.md.
