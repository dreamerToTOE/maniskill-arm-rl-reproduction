#!/usr/bin/env python3
"""Stage A1 - bounded parallel `physx_cuda` simulator probe (simulator SPS only).

One invocation measures ONE env count. `scripts/m2_a1_run_all.py` runs
16 -> 64 -> 128 -> 256 as four independent processes so that a failure in one
level cannot corrupt or truncate the record of another, and stops the sweep on
the first failure. A single invocation is still accepted for a repeat of one
level, and `--env-counts 16,64` works but is NOT the recommended path.

Fixes carried over from the independent review
(`docs/M2_INDEPENDENT_REVIEW.md`):

  1. actions are always `(num_envs, action_dim)` and live on the env device;
     `BaseEnv._step_action` only auto-batches when `num_envs == 1`
     (sapien_env.py:1108), so an unbatched `(8,)` action is invalid for n > 1.
  2. wall clock is enforced for the WHOLE process: a monotonic global budget
     anchored at module import, a per-level budget covering build + reset +
     warmup + measure + cleanup, and a daemon watchdog that hard-exits. The
     launcher adds an external `timeout` as a fourth, process-level backstop.
  3. episodes are rolled out through `ManiSkillVectorEnv(auto_reset=True)`, the
     same wrapper the pinned baseline uses. With a 50-step horizon and
     10 + 100 steps per level, truncation happens by construction, and a level
     that records zero truncations is treated as a FAILURE (the boundary was
     not actually crossed, so the rollout tells us nothing).
  4. VRAM guarding is conservative for an 8 GiB desktop card: refuse to start a
     level with less than `--min-free-mib` free, sample device-level free memory
     every `--vram-every` steps, and abort a level before exhaustion rather than
     waiting for an OOM. Running out of free memory is a hard stop, not a retry.
  5. single-env smoke (A0) and parallel throughput (A1) are separate scripts.
  6. memory is reported from `nvidia-smi` device counters (which see
     SAPIEN/PhysX allocations); the PyTorch allocator peak is recorded only as a
     torch-only lower bound and never used as the estimate of total scene
     memory.
  7. device-level VRAM telemetry is MANDATORY and is checked BEFORE `import
     torch`: no telemetry means exit 8 with no scene built, because a level
     whose memory cannot be bounded must not be started at all.
  8. the pinned stack (ManiSkill 3.0.1 / SAPIEN 3.0.0 / torch 2.5.1) is asserted
     and nothing upstream is modified; no install, no download, no training.

Default sweep is 16/64/128/256. 512 requires `--allow-512` in addition to the
per-level VRAM gate. The old 1024/2048/4096 levels are refused outright.

This measures SIMULATOR steps/second only. It is not training throughput, not a
baseline reproduction, and must never be compared with published numbers.

Usage (static self-check, no CUDA touched):
  ./.venv-maniskill/bin/python scripts/m2_gpu_sim_probe.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

PINNED_MANISKILL = "3.0.1"
PINNED_SAPIEN = "3.0.0"
PINNED_TORCH = "2.5.1"
HARD_MAX_ENVS = 512

EXIT_PASS = 0
EXIT_BACKEND = 2
EXIT_LOW_VRAM = 3
EXIT_BUDGET = 4
EXIT_ERROR = 5
EXIT_VERSION = 7
EXIT_NO_VRAM_TELEMETRY = 8
EXIT_LEVEL_FAILED = 9

_T0 = time.monotonic()


def elapsed() -> float:
    return round(time.monotonic() - _T0, 3)


class BudgetExhausted(RuntimeError):
    pass


class Deadline:
    def __init__(self, total_s: float, anchor: float | None = None) -> None:
        self.total_s = float(total_s)
        self.anchor = _T0 if anchor is None else float(anchor)

    def used(self) -> float:
        return time.monotonic() - self.anchor

    def remaining(self) -> float:
        return self.total_s - self.used()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def check(self, phase: str) -> None:
        if self.expired():
            raise BudgetExhausted(
                f"budget {self.total_s:.0f}s exhausted at '{phase}' (used {self.used():.1f}s)"
            )


def smi_query_ex(fields: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Device-level numbers plus WHY the probe failed.

    Device counters are the only view that includes SAPIEN/PhysX allocations, so
    the diagnostics matter as much as the numbers: "nvidia-smi not on PATH",
    "nvidia-smi exit 9 - cannot talk to the driver" and "unparsable output" are
    three different operator problems and must not collapse into one message.
    """
    diag: dict[str, Any] = {
        "binary": None,
        "returncode": None,
        "stderr_head": None,
        "exception": None,
        "command": None,
    }
    exe = shutil.which("nvidia-smi")
    diag["binary"] = exe
    if exe is None:
        diag["exception"] = "nvidia-smi not found on PATH"
        return None, diag
    cmd = [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    diag["command"] = " ".join(cmd)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:  # noqa: BLE001
        diag["exception"] = f"{type(exc).__name__}: {exc}"
        return None, diag
    diag["returncode"] = out.returncode
    if out.returncode != 0:
        diag["stderr_head"] = (out.stderr or out.stdout or "").strip().splitlines()[:2]
        return None, diag
    lines = out.stdout.strip().splitlines()
    if not lines:
        diag["exception"] = "empty stdout"
        return None, diag
    keys = [f.strip() for f in fields.split(",")]
    vals = [v.strip() for v in lines[0].split(",")]
    parsed: dict[str, Any] = {}
    for k, v in zip(keys, vals):
        if k.startswith("memory") or k.startswith("utilization"):
            try:
                parsed[k] = int(float(v))
            except ValueError:
                parsed[k] = None
        else:
            parsed[k] = v
    return parsed, diag


def smi_query(fields: str) -> dict[str, Any] | None:
    """Device-level numbers. This is the only view that includes PhysX memory."""
    return smi_query_ex(fields)[0]


def smi_compute_apps() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return []
        apps = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                used: int | None = int(float(parts[2]))
            except ValueError:
                used = None
            apps.append({"pid": parts[0], "process": parts[1], "used_mib": used})
        return apps
    except Exception:
        return []


def device_nodes() -> dict[str, Any]:
    """Snapshot `/dev/nvidia*`: GPU visibility here is session-scoped."""
    try:
        nodes = sorted(str(p) for p in Path("/dev").glob("nvidia*"))
    except Exception as exc:  # noqa: BLE001
        return {"present": None, "nodes": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"present": bool(nodes), "nodes": nodes}


def vram_gate(free_mib: int | None, min_free_mib: int) -> str:
    """Conservative pre-flight decision: REFUSE / OK, UNKNOWN only if blind."""
    if free_mib is None:
        return "UNKNOWN"
    return "OK" if free_mib >= min_free_mib else "REFUSE"


def telemetry_verdict(probe: dict[str, Any] | None,
                      diag: dict[str, Any] | None = None) -> tuple[str, str]:
    """Decide whether device-level VRAM telemetry is usable.

    A level whose memory cannot be bounded must not be started: without a
    `nvidia-smi` reading there is no way to prove the 8 GiB card was not pushed
    to the edge, and no way to audit the run afterwards. So a missing or broken
    probe is a hard stop *before* any scene is built.
    """
    if probe is None:
        if diag:
            rc = diag.get("returncode")
            err = diag.get("stderr_head")
            if diag.get("binary") is None:
                return "UNAVAILABLE", "nvidia-smi not found on PATH"
            if rc not in (None, 0):
                detail = " | ".join(err) if isinstance(err, list) else str(err or "")
                return "UNAVAILABLE", f"nvidia-smi exited {rc}: {detail}"[:300]
            if diag.get("exception"):
                return "UNAVAILABLE", f"nvidia-smi probe failed: {diag['exception']}"[:300]
        return "UNAVAILABLE", "nvidia-smi missing, non-zero exit, unparsable output, or timeout"
    free = probe.get("memory.free")
    if free is None:
        return "UNAVAILABLE", f"nvidia-smi returned no parsable memory.free (got {probe!r})"
    if not isinstance(free, int) or free < 0:
        return "UNAVAILABLE", f"nvidia-smi memory.free is not a non-negative int: {free!r}"
    return "AVAILABLE", f"memory.free={free} MiB"


def batched_action_shape(num_envs: int, action_shape: tuple[int, ...]) -> tuple[int, ...]:
    """`(num_envs, *action_shape)` - never a bare `(action_dim,)`.

    `BaseEnv._step_action` (sapien_env.py:1108) only inserts a batch dimension
    when `num_envs == 1`; for `num_envs > 1` an unbatched action is handed
    straight to the controller. We always batch explicitly.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    return (int(num_envs), *tuple(int(x) for x in action_shape))


def validate_env_counts(counts: list[int], allow_512: bool) -> list[int]:
    """Refuse reference-scale workloads on an 8 GiB desktop card."""
    for n in counts:
        if n < 1:
            raise ValueError(f"invalid env count {n}")
        if n > HARD_MAX_ENVS:
            raise ValueError(
                f"env count {n} exceeds the hard cap {HARD_MAX_ENVS}; the original "
                "64-4096 sweep is not authorized - see docs/M2_INDEPENDENT_REVIEW.md"
            )
        if n == 512 and not allow_512:
            raise ValueError("512 envs requires --allow-512 in addition to the VRAM gate")
    return counts


def _inner_wrapper(cur: Any) -> Any:
    """Next env down the chain: gymnasium uses `.env`, ManiSkillVectorEnv uses `._env`."""
    for attr in ("env", "_env"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur:
            return nxt
    return None


def episode_horizon(env: Any) -> tuple[int | None, list[str], str | None]:
    horizon: int | None = None
    source: str | None = None
    names: list[str] = []
    cur = env
    for _ in range(16):
        names.append(type(cur).__name__)
        candidate = getattr(cur, "_max_episode_steps", None)
        if horizon is None and isinstance(candidate, int):
            horizon = candidate
            source = type(cur).__name__
        nxt = _inner_wrapper(cur)
        if nxt is None:
            break
        cur = nxt
    return horizon, names, source


def arm_watchdog(budget_s: float, on_fire) -> threading.Timer:
    timer = threading.Timer(max(0.05, float(budget_s)), on_fire)
    timer.daemon = True
    timer.start()
    return timer


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    check("batch shape n=16", batched_action_shape(16, (8,)) == (16, 8))
    check("batch shape n=256", batched_action_shape(256, (8,)) == (256, 8))
    check("batch never bare", batched_action_shape(2, (8,)) != (8,))
    try:
        batched_action_shape(-1, (8,))
        check("reject negative", False)
    except ValueError:
        check("reject negative", True)

    check("gate ok", vram_gate(4096, 3072) == "OK")
    check("gate refuse", vram_gate(1024, 3072) == "REFUSE")
    check("gate unknown", vram_gate(None, 3072) == "UNKNOWN")

    verdict, reason = telemetry_verdict({"memory.free": 4096})
    check("telemetry available", verdict == "AVAILABLE", reason)
    verdict, reason = telemetry_verdict(None, {"binary": None})
    check("telemetry no binary", verdict == "UNAVAILABLE" and "PATH" in reason, reason)
    verdict, reason = telemetry_verdict(None, {"binary": "/usr/bin/nvidia-smi",
                                              "returncode": 9,
                                              "stderr_head": ["couldn't communicate"]})
    check("telemetry rc9 distinct", verdict == "UNAVAILABLE" and "exited 9" in reason, reason)
    verdict, reason = telemetry_verdict({"memory.free": None})
    check("telemetry unparsable", verdict == "UNAVAILABLE", reason)

    try:
        validate_env_counts([16, 64, 128, 256], allow_512=False)
        check("accept default sweep", True)
    except ValueError as exc:
        check("accept default sweep", False, str(exc))
    for bad, allow, label in [
        ([4096], False, "reject 4096"),
        ([512], False, "512 needs opt-in"),
        ([0], False, "reject zero"),
    ]:
        try:
            validate_env_counts(bad, allow_512=allow)
            check(label, False)
        except ValueError:
            check(label, True)
    try:
        validate_env_counts([512], allow_512=True)
        check("512 allowed with opt-in", True)
    except ValueError as exc:
        check("512 allowed with opt-in", False, str(exc))

    d = Deadline(5.0, anchor=time.monotonic())
    check("deadline fresh", not d.expired())
    d2 = Deadline(5.0, anchor=time.monotonic() - 6.0)
    check("deadline expired", d2.expired())
    try:
        d2.check("selftest")
        check("deadline raises", False)
    except BudgetExhausted:
        check("deadline raises", True)

    class _W:
        def __init__(self, inner=None, steps=None):
            self.env = inner
            if steps is not None:
                self._max_episode_steps = steps

    horizon, names, source = episode_horizon(_W(_W(_W(steps=50))))
    check("horizon walk", horizon == 50, f"{horizon} {names} {source}")

    class _V:
        def __init__(self, inner):
            self._env = inner

    horizon_v, names_v, _ = episode_horizon(_V(_W(_W(steps=50))))
    check("horizon reaches through ManiSkillVectorEnv", horizon_v == 50, f"{horizon_v} {names_v}")

    nodes = device_nodes()
    check("device node snapshot is shaped correctly",
          isinstance(nodes.get("nodes"), list) and isinstance(nodes.get("present"), bool),
          str(nodes)[:120])

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a" / "b.json"
        atomic_write_json(p, {"x": [1, 2]})
        check("atomic json", json.loads(p.read_text())["x"] == [1, 2])

    fired: list[str] = []
    t = arm_watchdog(0.05, lambda: fired.append("x"))
    t.join(timeout=1.0)
    check("watchdog fires", bool(fired))

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return EXIT_PASS if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A1: bounded parallel physx_cuda simulator probe (simulator SPS only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--robot", default="panda")
    p.add_argument("--env-counts", default="16",
                   help="one level per invocation is recommended; lists are allowed but "
                        "not isolated from each other")
    p.add_argument("--allow-512", action="store_true",
                   help="permit 512 envs (still subject to the VRAM gate)")
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="steps discarded before timing (crosses the 50-step truncation)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--level-seconds", type=float, default=90.0,
                   help="budget for build+reset+warmup+measure+cleanup of ONE level")
    p.add_argument("--total-seconds", type=float, default=120.0,
                   help="budget for the whole process (import + build + steps + cleanup)")
    p.add_argument("--external-limit-s", type=float, default=150.0,
                   help="the external `timeout` value used by the launcher; recorded and "
                        "asserted, not enforced here")
    p.add_argument("--min-free-mib", type=int, default=3072,
                   help="refuse to start a level below this much free device memory")
    p.add_argument("--abort-free-mib", type=int, default=1536,
                   help="abort a level if free device memory drops below this")
    p.add_argument("--vram-every", type=int, default=10,
                   help="sample device VRAM every N steps (sampling time is subtracted)")
    p.add_argument("--output", type=Path, default=Path("results/m2_gpu_sim_probe.json"))
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--allow-version-mismatch", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    try:
        counts = validate_env_counts([int(x) for x in args.env_counts.split(",")], args.allow_512)
    except ValueError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.warmup_steps + args.steps < 1:
        print("refusing: no steps requested", file=sys.stderr)
        return EXIT_ERROR

    state: dict[str, Any] = {
        "script": "m2_gpu_sim_probe.py",
        "stage": "A1",
        "status": "RUNNING",
        "scope_warning": (
            "SIMULATOR steps/second only: zero actions, no policy, no optimizer, no "
            "evaluation, no render_mode=rgb_array. Not training throughput and not a "
            "baseline reproduction. Do not compare against published numbers."
        ),
        "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "pinned": {"mani_skill": PINNED_MANISKILL, "sapien": PINNED_SAPIEN, "torch": PINNED_TORCH},
        "config": {
            "env_id": args.env_id, "robot": args.robot, "num_envs_levels": counts,
            "obs_mode": "state", "control_mode": "pd_joint_delta_pos",
            "reward_mode": "normalized_dense", "sim_backend": "physx_cuda",
            "render_mode": None, "warmup_steps": args.warmup_steps, "steps": args.steps,
            "level_seconds": args.level_seconds, "total_seconds": args.total_seconds,
            "external_limit_s": args.external_limit_s,
            "min_free_mib": args.min_free_mib, "abort_free_mib": args.abort_free_mib,
            "vram_every": args.vram_every,
        },
        "device_nodes_before": device_nodes(),
        "other_compute_apps": smi_compute_apps(),
        "levels": [],
    }
    output = args.output

    def flush() -> None:
        state["elapsed_s"] = elapsed()
        atomic_write_json(output, state)

    # --- mandatory telemetry gate, deliberately BEFORE `import torch` ----------
    state["phase"] = "telemetry"
    probe, diag = smi_query_ex(
        "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu")
    verdict, reason = telemetry_verdict(probe, diag)
    state["gpu"] = probe
    state["vram_telemetry"] = {"verdict": verdict, "reason": reason, "diagnostics": diag}
    state["telemetry_gate_s"] = elapsed()
    flush()
    if verdict != "AVAILABLE":
        state["status"] = "VRAM_TELEMETRY_UNAVAILABLE"
        state["error"] = (
            f"device-level VRAM telemetry unavailable ({reason}); refusing to size a "
            "parallel level blind. No scene was built."
        )
        state["run_completed"] = False
        flush()
        print(f"TELEMETRY UNAVAILABLE: {reason}", file=sys.stderr, flush=True)
        print(f"status={state['status']} elapsed={elapsed()}s -> {output}", flush=True)
        return EXIT_NO_VRAM_TELEMETRY

    global_deadline = Deadline(args.total_seconds)

    def _on_timeout() -> None:
        state["status"] = "BUDGET_EXCEEDED"
        state["timed_out_at"] = state.get("phase", "unknown")
        try:
            flush()
        finally:
            print(f"TIMEOUT after {elapsed()}s at {state.get('timed_out_at')}", flush=True)
            os._exit(EXIT_BUDGET)

    timer = arm_watchdog(args.total_seconds, _on_timeout)
    exit_code = EXIT_PASS
    try:
        state["phase"] = "import"
        t_imp = time.monotonic()
        import torch
        import gymnasium as gym
        import mani_skill
        import sapien
        import mani_skill.envs  # noqa: F401
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        state["versions"] = {
            "mani_skill": getattr(mani_skill, "__version__", "unknown"),
            "sapien": getattr(sapien, "__version__", "unknown"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gymnasium": getattr(gym, "__version__", "unknown"),
        }
        state["import_s"] = round(time.monotonic() - t_imp, 3)
        mism = [k for k, bad in {
            "mani_skill": state["versions"]["mani_skill"] != PINNED_MANISKILL,
            "sapien": state["versions"]["sapien"] != PINNED_SAPIEN,
            "torch": not state["versions"]["torch"].startswith(PINNED_TORCH),
        }.items() if bad]
        state["version_mismatches"] = mism
        if mism and not args.allow_version_mismatch:
            state["status"] = "VERSION_MISMATCH"
            flush()
            print(f"refusing: version mismatch {mism}", file=sys.stderr)
            return EXIT_VERSION
        if not torch.cuda.is_available():
            state["status"] = "FAIL_SIM_BACKEND"
            state["error"] = "torch.cuda.is_available() is False"
            flush()
            return EXIT_BACKEND
        state["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        }
        global_deadline.check("import")
        flush()

        for n in counts:
            global_deadline.check(f"level{n}")
            level_deadline = Deadline(args.level_seconds)
            entry: dict[str, Any] = {"num_envs": n, "status": "RUNNING"}
            state["levels"].append(entry)
            state["phase"] = f"level{n}"
            env = None
            level_t0 = time.monotonic()

            def lcheck(phase: str) -> None:
                level_deadline.check(f"n={n}:{phase}")
                global_deadline.check(f"n={n}:{phase}")

            warnlog: list[str] = []

            def _capture_warning(message, category, filename, lineno, file=None, line=None) -> None:
                if len(warnlog) < 50:
                    warnlog.append(f"{category.__name__}: {message}")

            orig_showwarning = warnings.showwarning
            try:
                lcheck("preflight")
                warnings.showwarning = _capture_warning
                warnings.simplefilter("always")
                info0 = smi_query("memory.total,memory.used,memory.free,utilization.gpu") or {}
                free_before = info0.get("memory.free")
                gverdict = vram_gate(free_before, args.min_free_mib)
                entry["vram_before"] = info0
                entry["vram_verdict"] = gverdict
                flush()
                if gverdict == "REFUSE":
                    entry["status"] = "SKIPPED_LOW_VRAM"
                    entry["error"] = (
                        f"free device memory {free_before} MiB < min-free {args.min_free_mib} MiB; "
                        "level not started"
                    )
                    exit_code = EXIT_LOW_VRAM
                    flush()
                    print(f"[n={n}] skip: free={free_before} MiB < {args.min_free_mib}", flush=True)
                    break
                if gverdict == "UNKNOWN":
                    entry["status"] = "VRAM_TELEMETRY_UNAVAILABLE"
                    entry["error"] = (
                        "nvidia-smi telemetry disappeared before the level; refusing to run blind"
                    )
                    exit_code = EXIT_NO_VRAM_TELEMETRY
                    flush()
                    print(f"[n={n}] {entry['error']}", flush=True)
                    break

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                lcheck("build")
                t = time.monotonic()
                base = gym.make(
                    args.env_id,
                    num_envs=n,
                    robot_uids=args.robot,
                    obs_mode="state",
                    control_mode="pd_joint_delta_pos",
                    reward_mode="normalized_dense",
                    sim_backend="physx_cuda",
                    render_mode=None,
                )
                env = ManiSkillVectorEnv(base, auto_reset=True, ignore_terminations=False)
                entry["build_s"] = round(time.monotonic() - t, 3)
                unwrapped = env.unwrapped
                backend_name = str(getattr(getattr(unwrapped, "backend", None), "sim_backend", "?"))
                gpu_sim = bool(getattr(unwrapped, "gpu_sim_enabled", False))
                actual_envs = int(getattr(unwrapped, "num_envs", -1))
                entry["backend"] = {
                    "requested": "physx_cuda", "resolved": backend_name,
                    "gpu_sim_enabled": gpu_sim, "num_envs": actual_envs,
                }
                horizon, chain, horizon_source = episode_horizon(env)
                entry["max_episode_steps"] = horizon
                entry["horizon_source"] = horizon_source
                entry["wrapper_chain"] = chain
                info_b = smi_query("memory.total,memory.used,memory.free") or {}
                entry["vram_after_build"] = info_b
                flush()
                if actual_envs != n:
                    entry["status"] = "FAIL_ENV_COUNT"
                    entry["error"] = f"requested {n} envs, env reports {actual_envs}"
                    exit_code = EXIT_LEVEL_FAILED
                    flush()
                    break
                if backend_name != "physx_cuda" or not gpu_sim:
                    entry["status"] = "FAIL_SIM_BACKEND"
                    entry["error"] = f"resolved={backend_name} gpu_sim_enabled={gpu_sim}"
                    exit_code = EXIT_BACKEND
                    flush()
                    break

                lcheck("reset")
                t = time.monotonic()
                env.reset(seed=0)
                torch.cuda.synchronize()
                entry["reset_s"] = round(time.monotonic() - t, 3)

                shape = batched_action_shape(n, tuple(env.single_action_space.shape))
                action = torch.zeros(shape, dtype=torch.float32, device=unwrapped.device)
                assert tuple(action.shape) == shape, f"{tuple(action.shape)} != {shape}"
                entry["action"] = {"shape": list(action.shape), "device": str(action.device),
                                   "dtype": str(action.dtype), "values": "all zeros"}

                counters_total = {"terminated_steps": 0, "terminated_envs": 0,
                                  "truncated_steps": 0, "truncated_envs": 0,
                                  "auto_reset_steps": 0, "auto_reset_envs": 0}
                counters_measure = dict(counters_total)
                phase = {"name": "warmup"}
                vram_samples: list[tuple[int, int]] = []
                smi_time = 0.0

                def run_steps(n_steps: int) -> int:
                    nonlocal smi_time
                    done = 0
                    for i in range(n_steps):
                        lcheck("step")
                        obs, rew, term, trunc, info = env.step(action)
                        done += 1
                        t_envs = int(term.sum().item()) if hasattr(term, "sum") else int(bool(term))
                        r_envs = int(trunc.sum().item()) if hasattr(trunc, "sum") else int(bool(trunc))
                        for c in (counters_total, counters_measure if phase["name"] == "measure"
                                  else None):
                            if c is None:
                                continue
                            c["terminated_steps"] += int(t_envs > 0)
                            c["terminated_envs"] += t_envs
                            c["truncated_steps"] += int(r_envs > 0)
                            c["truncated_envs"] += r_envs
                        if "final_observation" in info:
                            n_done = int(term.sum().item() + trunc.sum().item())
                            for c in (counters_total, counters_measure if phase["name"] == "measure"
                                      else None):
                                if c is None:
                                    continue
                                c["auto_reset_steps"] += 1
                                c["auto_reset_envs"] += n_done
                        if args.vram_every > 0 and (i + 1) % args.vram_every == 0:
                            ts = time.monotonic()
                            info_v = smi_query("memory.free,memory.used")
                            smi_time += time.monotonic() - ts
                            free_now = (info_v or {}).get("memory.free")
                            used_now = (info_v or {}).get("memory.used")
                            if free_now is not None:
                                vram_samples.append((i + 1, free_now))
                                if free_now < args.abort_free_mib:
                                    raise MemoryError(
                                        f"free device memory {free_now} MiB < "
                                        f"{args.abort_free_mib} MiB at step {i + 1}"
                                    )
                            if used_now is not None:
                                entry["device_used_peak_mib"] = max(
                                    entry.get("device_used_peak_mib") or 0, used_now)
                    return done

                lcheck("warmup")
                t_w = time.monotonic()
                warm_done = run_steps(args.warmup_steps)
                torch.cuda.synchronize()
                entry["warmup_s"] = round(time.monotonic() - t_w - smi_time, 3)
                entry["warmup_steps_done"] = warm_done
                entry["counters_warmup"] = dict(counters_total)

                lcheck("measure")
                phase["name"] = "measure"
                smi_time_measure = 0.0
                smi_time = 0.0
                t_m = time.monotonic()
                done = run_steps(args.steps)
                torch.cuda.synchronize()
                smi_time_measure = smi_time
                measure_s = max(1e-9, time.monotonic() - t_m - smi_time_measure)
                entry["measure_s"] = round(measure_s, 3)
                entry["measure_steps_done"] = done
                entry["total_env_steps"] = done * n
                entry["sim_steps_per_s"] = round(done / measure_s, 2)
                entry["sim_sps"] = round(done * n / measure_s, 1)
                entry["per_env_sps"] = round(done / measure_s, 2)
                entry["ms_per_sim_step"] = round(1000.0 * measure_s / max(1, done), 3)
                entry["vram_sampling_overhead_s"] = round(smi_time_measure, 3)
                entry["counters_measure"] = dict(counters_measure)
                entry["counters_total"] = dict(counters_total)
                entry["truncation_observed"] = counters_total["truncated_envs"] > 0
                entry["warnings"] = sorted(set(warnlog))[:20]

                entry["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                entry["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
                info_after = smi_query("memory.total,memory.used,memory.free") or {}
                entry["vram_after"] = info_after
                if vram_samples:
                    entry["vram_free_min_mib"] = min(v for _, v in vram_samples)
                    entry["vram_free_samples"] = [{"step": s, "free_mib": v} for s, v in vram_samples]
                used_total = info_after.get("memory.total") or 0
                if used_total and entry.get("vram_free_min_mib") is not None:
                    entry["device_used_peak_mib_est"] = used_total - entry["vram_free_min_mib"]
                base_used = info0.get("memory.used")
                used_now = info_after.get("memory.used")
                entry["device_used_baseline_mib"] = base_used
                entry["device_used_delta_mib"] = (
                    used_now - base_used if used_now is not None and base_used is not None else None
                )
                entry["device_used_increase_vs_idle_mib"] = entry["device_used_delta_mib"]
                entry["attribution_note"] = (
                    "device_used_* come from nvidia-smi and include SAPIEN/PhysX allocations "
                    "plus desktop noise; torch_peak_* is a torch-only lower bound and must NOT "
                    "be read as total scene memory."
                )

                if done != args.steps:
                    entry["status"] = "FAIL_INCOMPLETE_MEASUREMENT"
                    entry["error"] = f"measured {done}/{args.steps} steps"
                    exit_code = EXIT_LEVEL_FAILED
                    flush()
                    break
                if not entry["truncation_observed"]:
                    entry["status"] = "FAIL_NO_TRUNCATION"
                    entry["error"] = (
                        f"{args.warmup_steps + args.steps} steps with horizon {horizon} "
                        "recorded zero truncations; the episode boundary was not crossed, "
                        "so the rollout is not evidence of auto-reset behaviour"
                    )
                    exit_code = EXIT_LEVEL_FAILED
                    flush()
                    break

                entry["status"] = "OK"
                flush()
                print(json.dumps({k: entry[k] for k in
                                  ("num_envs", "status", "build_s", "sim_sps", "counters_total",
                                   "device_used_delta_mib", "vram_free_min_mib") if k in entry},
                                 ensure_ascii=False), flush=True)

            except BudgetExhausted as exc:
                entry["status"] = "BUDGET_EXCEEDED"
                entry["error"] = str(exc)
                exit_code = EXIT_BUDGET
                flush()
                print(f"[n={n}] budget exceeded: {exc}", flush=True)
                break
            except MemoryError as exc:
                entry["status"] = "VRAM_GUARD_TRIPPED"
                entry["error"] = str(exc)
                exit_code = EXIT_LOW_VRAM
                flush()
                print(f"[n={n}] VRAM guard: {exc}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                low = str(exc).lower()
                entry["status"] = "OOM" if "out of memory" in low else "ERROR"
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                exit_code = EXIT_ERROR
                flush()
                print(f"[n={n}] {entry['status']}: {entry['error']}", flush=True)
                break
            finally:
                t_c = time.monotonic()
                warnings.showwarning = orig_showwarning
                entry.setdefault("warnings", sorted(set(warnlog))[:20])
                if env is not None:
                    try:
                        env.close()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                entry["cleanup_s"] = round(time.monotonic() - t_c, 3)
                entry["level_elapsed_s"] = round(time.monotonic() - level_t0, 3)
                flush()

        state["run_completed"] = True
        levels = state["levels"]
        if levels and all(lv.get("status") == "OK" for lv in levels):
            state["status"] = "PASS"
        elif any(lv.get("status") == "OK" for lv in levels):
            state["status"] = "PARTIAL"
            if exit_code == EXIT_PASS:
                exit_code = EXIT_LEVEL_FAILED
        elif levels:
            state["status"] = "NO_LEVEL_COMPLETED"
            if exit_code == EXIT_PASS:
                exit_code = EXIT_LEVEL_FAILED
        else:
            state["status"] = "NO_LEVEL_RUN"
            if exit_code == EXIT_PASS:
                exit_code = EXIT_LEVEL_FAILED
        state["run_completed"] = state["status"] in {
            "PASS", "PARTIAL", "NO_LEVEL_COMPLETED", "NO_LEVEL_RUN"
        }
    except BudgetExhausted as exc:
        state["status"] = "BUDGET_EXCEEDED"
        state["error"] = str(exc)
        exit_code = EXIT_BUDGET
    except Exception as exc:  # noqa: BLE001
        state["status"] = "ERROR"
        state["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        exit_code = EXIT_ERROR
    finally:
        timer.cancel()
        state["finished_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["device_nodes_after"] = device_nodes()
        state["gpu_after"] = smi_query("memory.total,memory.used,memory.free")
        flush()
        print(f"status={state['status']} elapsed={elapsed()}s -> {output}", flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
