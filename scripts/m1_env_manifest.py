#!/usr/bin/env python3
"""Collect the M1 environment manifest (read-only, no simulation).

Usage:
    python scripts/m1_env_manifest.py --output results/m1_environment_manifest.json

Records interpreter, package versions, `pip check`, torch CUDA runtime and GPU
availability, disk/VRAM state and git SHAs. Must be run with the isolated
`.venv-maniskill` interpreter. Never trains or steps an environment.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = [
    "torch",
    "sapien",
    "mani-skill",
    "gymnasium",
    "numpy",
    "scipy",
    "opencv-python",
    "trimesh",
    "mplib",
    "toppra",
    "pytorch-kinematics",
    "h5py",
    "tyro",
    "huggingface-hub",
]


def run(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {"cmd": " ".join(cmd), "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:2000]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import importlib.metadata as md

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None

    import torch

    torch_info = {
        "version": torch.__version__,
        "cuda_runtime_compiled": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "cpu_count": __import__("os").cpu_count(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "is_venv": sys.prefix != sys.base_prefix,
            "venv_prefix": sys.prefix,
        },
        "packages": versions,
        "torch": torch_info,
        "pip_check": run([sys.executable, "-m", "pip", "check"]),
        "disk": run(["df", "-B1", str(Path.cwd())]),
        "venv_size": run(["du", "-sb", str(Path(sys.prefix))]),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                           "--format=csv,noheader"]),
        "git": {
            "project_head": run(["git", "rev-parse", "HEAD"])["stdout"],
            "project_status": run(["git", "status", "--short"])["stdout"],
            "upstream_pin": "a4a4f9272ad64b1564035874b605ceb687b63ed8",
            "upstream_tag": "v3.0.1",
        },
        "env_vars_used": ["MS_SKIP_ASSET_DOWNLOAD_PROMPT=1", "TMPDIR=<large filesystem>"],
        "note": (
            "Installed ManiSkill comes from the PyPI wheel mani-skill==3.0.1; the upstream git pin "
            "is recorded as source reference, not as binary identity."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
