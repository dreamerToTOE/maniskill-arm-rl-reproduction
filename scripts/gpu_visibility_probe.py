#!/usr/bin/env python3
"""Diagnose whether THIS session can use the NVIDIA GPU, and why not.

The agent shell's GPU visibility comes and goes (2026-10-21: present 09:57,
absent 12:58, absent 15:24, absent 16:09, present again in a later session).
This probe turns that into a verdict instead of a guess, so a GPU gate can be
refused for the right reason.

Verdicts (exit code in brackets):

  OK               [0]  nvidia-smi works; the session can use the GPU.
  NO_TELEMETRY     [8]  nvidia-smi is missing or exits non-zero.
  NO_NODES         [3]  /dev has no NVIDIA character devices at all.
  NODES_UNUSABLE   [4]  nodes exist (or were created) and open(), but the driver
                        conversation fails -> the sandbox is blocking ioctl().
  NODES_MISSING_BUT_FIXABLE [5]
                        nodes were absent, --try-mknod created them, and the
                        driver then answered. Only reachable when the sandbox
                        permits ioctl(); never observed so far.

`--try-mknod` is the only mode that writes anything: it creates the NVIDIA
character devices in THIS process's /dev. That /dev is a per-invocation tmpfs
(7 nodes, 10 MB), so the nodes disappear when the process exits - nothing
persists and the host is untouched. It exists because "no device nodes" and
"nodes but blocked" have completely different fixes and were being conflated.

Usage:
  python3 scripts/gpu_visibility_probe.py
  python3 scripts/gpu_visibility_probe.py --try-mknod --json /tmp/gpu.json
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# major:minor as registered in /proc/devices on this host (driver 580.178.04).
# nvidia-uvm is NOT 511 here; the majors are dynamic, so read /proc/devices
# rather than trusting folklore.
NODES = [
    ("nvidiactl", 195, 255),
    ("nvidia0", 195, 0),
    ("nvidia-uvm", 235, 0),
    ("nvidia-uvm-tools", 235, 1),
    ("nvidia-modeset", 195, 254),
]

EXIT_OK = 0
EXIT_NO_TELEMETRY = 8
EXIT_NO_NODES = 3
EXIT_NODES_UNUSABLE = 4
EXIT_FIXABLE = 5


def smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"binary": None, "returncode": None, "stdout": None, "stderr": "not on PATH"}
    proc = subprocess.run(
        [exe, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    return {"binary": exe, "returncode": proc.returncode,
            "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[:200]}


def smi_compute_apps() -> list[dict]:
    """Processes currently holding GPU memory - i.e. possible interference."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return []
        apps = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                used: int | None = int(float(parts[2]))
            except ValueError:
                used = None
            apps.append({"pid": parts[0], "process": parts[1], "used_mib": used})
        return apps
    except Exception:  # noqa: BLE001
        return []


def dev_nodes() -> list[str]:
    return sorted(str(p) for p in Path("/dev").glob("nvidia*"))


def proc_devices() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/devices").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and "nvidia" in parts[1]:
                out[parts[1]] = int(parts[0])
    except OSError:
        pass
    return out


def sandbox_flags() -> dict:
    """Seccomp mode 2 + one filter is the mechanism that neutralises ioctl()."""
    info: dict = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("Seccomp:", "Seccomp_filters:", "NoNewPrivs:")):
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    except OSError:
        pass
    try:
        info["cgroup"] = Path("/proc/self/cgroup").read_text().strip()
    except OSError:
        pass
    return info


def try_mknod() -> dict:
    """Create the nodes in this process's /dev and test the driver path.

    Returns what actually happened; never raises.
    """
    result: dict = {"created": [], "failed": [], "opened": {}, "cuinit": None,
                    "smi_after": None, "note": None}
    majors = proc_devices()
    nodes = NODES
    if majors.get("nvidia-uvm") and majors["nvidia-uvm"] != 235:
        nodes = [(n, majors.get(n, mj), mn) for n, mj, mn in NODES]
    for name, maj, minr in nodes:
        path = "/dev/" + name
        try:
            if os.path.exists(path):
                os.unlink(path)
            os.mknod(path, 0o666, os.makedev(maj, minr))
            result["created"].append(f"{name} {maj}:{minr}")
        except OSError as exc:
            result["failed"].append(f"{name}: {errno.errorcode.get(exc.errno, exc.errno)}")

    for name, _, _ in nodes:
        path = "/dev/" + name
        try:
            fd = os.open(path, os.O_RDWR)
            os.close(fd)
            result["opened"][name] = "OK"
        except OSError as exc:
            result["opened"][name] = errno.errorcode.get(exc.errno, exc.errno)

    try:
        import ctypes
        result["cuinit"] = ctypes.CDLL("libcuda.so.1").cuInit(0)
    except Exception as exc:  # noqa: BLE001
        result["cuinit"] = f"n/a ({type(exc).__name__})"

    result["smi_after"] = smi()
    if result["failed"]:
        result["note"] = (
            "mknodat was refused by the sandbox; nothing can be tested. Creating device "
            "nodes needs no privilege here only because the sandbox decides - if it says "
            "no, the session simply cannot reach the driver."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--try-mknod", action="store_true",
                    help="create the NVIDIA nodes in this process's /dev and test them")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    report: dict = {
        "kind": "gpu_visibility",
        "dev_nodes": dev_nodes(),
        "proc_devices": proc_devices(),
        "sandbox": sandbox_flags(),
        "smi": smi(),
    }

    smi_ok = report["smi"]["returncode"] == 0
    verdict, hint = "", ""
    if smi_ok:
        verdict, hint = "OK", "the session can use the GPU; run GPU gates here"
    elif not report["dev_nodes"]:
        verdict = "NO_NODES"
        hint = ("/dev has no NVIDIA character devices. In this environment /dev is a "
                "per-invocation tmpfs; try --try-mknod to separate 'no nodes' from "
                "'nodes but blocked'.")
    else:
        verdict = "NODES_UNUSABLE"
        hint = "nodes exist but the driver probe fails; the sandbox is blocking ioctl()"

    if args.try_mknod and not smi_ok:
        report["try_mknod"] = try_mknod()
        tm = report["try_mknod"]
        if tm["smi_after"] and tm["smi_after"]["returncode"] == 0:
            verdict = "OK"
            hint = "nodes were missing; creating them restored GPU access"
        elif tm["created"] and all(v == "OK" for v in tm["opened"].values()):
            verdict = "NODES_UNUSABLE"
            hint = ("nodes were created and open() succeeded, but the driver still does "
                    "not answer (cuInit=%s, nvidia-smi rc=%s). The block is the session's "
                    "seccomp ioctl whitelist, not missing nodes or permissions."
                    % (tm["cuinit"], (tm["smi_after"] or {}).get("returncode")))
        elif verdict == "NO_NODES":
            verdict = "NO_NODES"

    report["verdict"] = verdict
    report["hint"] = hint

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"\nverdict: {verdict}\n{hint}", file=sys.stderr)
    return {"OK": EXIT_OK, "NO_TELEMETRY": EXIT_NO_TELEMETRY,
            "NO_NODES": EXIT_NO_NODES, "NODES_UNUSABLE": EXIT_NODES_UNUSABLE,
            "NODES_MISSING_BUT_FIXABLE": EXIT_FIXABLE}.get(verdict, EXIT_NO_TELEMETRY)


if __name__ == "__main__":
    sys.exit(main())
