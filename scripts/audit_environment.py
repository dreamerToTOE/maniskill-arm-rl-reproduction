#!/usr/bin/env python3
"""Read-only local toolchain audit. Does not install dependencies or run training."""
from __future__ import annotations
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def gpu_status() -> str:
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not present"
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return str(exc)


def main() -> None:
    packages = ["torch", "numpy", "gymnasium", "mani_skill", "mujoco",
                "robosuite", "rsl-rl", "isaaclab", "stable-baselines3",
                "sapien", "genesis-world"]
    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_status(),
        "packages": {name: package_version(name) for name in packages},
        "note": "Read-only audit; versions alone do not establish compatibility."
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
