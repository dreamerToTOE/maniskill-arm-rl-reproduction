#!/usr/bin/env python3
"""Stage A0 - single-environment `physx_cuda` smoke test (ManiSkill v3.0.1).

Answers exactly one question: **does the CUDA PhysX backend build one PickCube
environment, reset it with seed 0, and survive 3 zero-action steps?**

It does NOT measure throughput. For the parallel scaling probe see
`scripts/m2_gpu_sim_probe.py` (Stage A1, not authorized).

Hard constraints baked in:
  * 1 env, `PickCube-v1`, `panda`, `state` obs, `pd_joint_delta_pos`,
    `normalized_dense` reward, `render_mode=None`, seed 0.
  * exactly 3 zero actions, batched to shape `(num_envs, action_dim)`.
  * `sim_backend="physx_cuda"` is passed EXPLICITLY and then verified against
    `env.unwrapped.backend.sim_backend` and `env.unwrapped.gpu_sim_enabled`.
    This matters: with the default `sim_backend="auto"` a single-env run
    silently falls back to `physx_cpu` (sapien_env.py:233-237) and would
    produce a false PASS.
  * wall clock is bounded by a monotonic deadline checked at every phase
    boundary and before every step, plus a watchdog thread that hard-exits.
    The proposed command also wraps the process in `timeout 120`.
  * VRAM telemetry from `nvidia-smi` is MANDATORY. If the tool is missing, fails,
    or returns an unparsable value, the script aborts BEFORE creating the
    environment, records which probe failed, and exits non-zero. There is no
    "telemetry unknown, continue anyway" path. PyTorch's allocator peak is
    recorded only as a torch-only lower bound; SAPIEN/PhysX allocations are not
    visible to it.
  * no policy, no optimizer, no gradient, no training, no video, no logging
    backend, no downloads, no dependency installation.
  * a programmatic acceptance gate re-checks 12 conditions after the run. A
    single failed condition forces `status=GATE_FAILED` and a non-zero exit
    code, so a "finished" run can never be reported as PASS on vibes.

Exit codes:
  0 PASS | 2 FAIL_SIM_BACKEND | 3 SKIPPED_LOW_VRAM | 4 BUDGET_EXCEEDED
  5 ERROR | 6 UNEXPECTED_DONE | 7 VERSION_MISMATCH
  8 VRAM_TELEMETRY_UNAVAILABLE | 9 ACCEPTANCE_GATE_FAILED

Usage (static self-check, no CUDA touched):
  ./.venv-maniskill/bin/python scripts/m2_gpu_smoke_a0.py --self-test
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
from pathlib import Path
from typing import Any

PINNED_MANISKILL = "3.0.1"
PINNED_SAPIEN = "3.0.0"
PINNED_TORCH = "2.5.1"

EXIT_PASS = 0
EXIT_BACKEND = 2
EXIT_LOW_VRAM = 3
EXIT_BUDGET = 4
EXIT_ERROR = 5
EXIT_UNEXPECTED_DONE = 6
EXIT_VERSION = 7
EXIT_NO_VRAM_TELEMETRY = 8
EXIT_GATE_FAILED = 9

EXPECTED_HORIZON = 50
EXPECTED_ACTION_SHAPE = (1, 8)
EXPECTED_ELAPSED_STEPS = [1, 2, 3]

_T0 = time.monotonic()


def elapsed() -> float:
    return round(time.monotonic() - _T0, 3)


class BudgetExhausted(RuntimeError):
    """Raised when the monotonic deadline is passed."""


class Deadline:
    """Monotonic budget anchored at process start, not at first use.

    Anchoring at `_T0` (module import) is deliberate: interpreter start-up and
    the torch/CUDA import cost are part of the wall clock the researcher pays.
    """

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
                f"budget of {self.total_s:.0f}s exhausted at phase '{phase}' "
                f"(used {self.used():.1f}s)"
            )


class Recorder:
    """Incremental JSON evidence.

    Written after *every* phase so that an external `timeout` kill still leaves
    a readable artefact instead of an empty file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "script": "m2_gpu_smoke_a0.py",
            "stage": "A0",
            "status": "RUNNING",
            "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pinned": {
                "mani_skill": PINNED_MANISKILL,
                "sapien": PINNED_SAPIEN,
                "torch": PINNED_TORCH,
            },
            "timings_s": {},
            "steps": [],
            "vram": {},
        }

    def phase(self, name: str, started: float) -> float:
        dt = round(time.monotonic() - started, 3)
        self.data["timings_s"][name] = dt
        self.data["elapsed_s"] = elapsed()
        self.flush()
        return dt

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return []
        apps = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                used = int(float(parts[2]))
            except ValueError:
                used = None
            apps.append({"pid": parts[0], "process": parts[1], "used_mib": used})
        return apps
    except Exception:
        return []


def vram_verdict(free_mib: int | None, min_free_mib: int) -> str:
    """Conservative pre-flight decision.

    Returns REFUSE when the device has less free memory than the reserve we
    insist on before even creating the GPU PhysX scene.
    Returns UNKNOWN only when telemetry is missing - and the caller treats
    UNKNOWN as a hard abort (`EXIT_NO_VRAM_TELEMETRY`), never as "carry on".
    """
    if free_mib is None:
        return "UNKNOWN"
    if free_mib < min_free_mib:
        return "REFUSE"
    return "OK"


def telemetry_verdict(probe: dict[str, Any] | None,
                     diag: dict[str, Any] | None = None) -> tuple[str, str]:
    """Decide whether device-level VRAM telemetry is usable.

    VRAM telemetry is mandatory for Stage A0: without a `nvidia-smi` reading we
    cannot bound the PhysX allocation, cannot prove the 8 GiB card was not
    pushed to the edge, and cannot audit the run afterwards. So a missing or
    broken probe is a hard stop *before* any scene is built.

    Returns (verdict, reason).
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


def acceptance_checks(state: dict[str, Any], abort_free_mib: int,
                      budget_seconds: float, external_limit_s: float = 120.0) -> list[dict[str, Any]]:
    """The 12 programmatic PASS conditions for Stage A0.

    Every condition is evaluated from recorded evidence only. A condition whose
    evidence is absent counts as failed - that is the point: an aborted run must
    not be reconstructable as a pass.
    """
    checks: list[dict[str, Any]] = []

    def add(cid: int, name: str, ok: bool, detail: str = "") -> None:
        checks.append({"id": cid, "name": name, "ok": bool(ok), "detail": str(detail)[:300]})

    steps = state.get("steps") or []
    backend = state.get("backend") or {}
    action = state.get("action") or {}
    vram = state.get("vram") or {}

    add(1, "run reached the end without exception", state.get("run_completed") is True,
        f"status={state.get('status')} error={state.get('error')}")

    add(2, "backend resolved to physx_cuda", backend.get("resolved") == "physx_cuda",
        f"resolved={backend.get('resolved')}")

    add(3, "gpu_sim_enabled is True", backend.get("gpu_sim_enabled") is True,
        f"gpu_sim_enabled={backend.get('gpu_sim_enabled')}")

    add(4, f"max_episode_steps == {EXPECTED_HORIZON}", backend.get("max_episode_steps") == EXPECTED_HORIZON,
        f"max_episode_steps={backend.get('max_episode_steps')}")

    actual_shape = tuple(action.get("actual_shape") or ())
    add(5, f"action shape == {list(EXPECTED_ACTION_SHAPE)}", actual_shape == EXPECTED_ACTION_SHAPE,
        f"actual_shape={list(actual_shape)}")

    add(6, "exactly 3 steps recorded", len(steps) == 3, f"len(steps)={len(steps)}")

    add(7, "no termination in any step", bool(steps) and all(s.get("terminated") == 0 for s in steps),
        f"terminated={[s.get('terminated') for s in steps]}")

    add(8, "no truncation in any step", bool(steps) and all(s.get("truncated") == 0 for s in steps),
        f"truncated={[s.get('truncated') for s in steps]}")

    elapsed_steps = [s.get("elapsed_steps") for s in steps]
    add(9, f"elapsed_steps == {EXPECTED_ELAPSED_STEPS}", elapsed_steps == EXPECTED_ELAPSED_STEPS,
        f"elapsed_steps={elapsed_steps}")

    have_before = (vram.get("device_before") or {}).get("memory.free") is not None
    have_after = (vram.get("device_after") or {}).get("memory.free") is not None
    step_samples = [s.get("vram_free_mib") for s in steps]
    add(10, "device-level VRAM telemetry present (before/after/every step)",
        have_before and have_after and bool(steps) and all(v is not None for v in step_samples),
        f"before={have_before} after={have_after} step_samples={step_samples}")

    free_after = (vram.get("device_after") or {}).get("memory.free")
    add(11, f"device free after run >= abort threshold {abort_free_mib} MiB",
        free_after is not None and free_after >= abort_free_mib,
        f"free_after={free_after}")

    elapsed_s = state.get("elapsed_s")
    within = (
        isinstance(elapsed_s, (int, float))
        and elapsed_s < external_limit_s
        and elapsed_s < float(budget_seconds)
    )
    add(12, f"wall clock < min({external_limit_s:.0f}s external, {budget_seconds:.0f}s internal)",
        within, f"elapsed_s={elapsed_s}")

    return checks


def finalize_with_gate(state: dict[str, Any], abort_free_mib: int, budget_seconds: float,
                       fallback_exit: int) -> int:
    """Attach acceptance results and return the exit code that matches them.

    Any failed condition overrides the caller's exit code, so an early abort or a
    run that merely "finished" cannot exit 0.
    """
    checks = acceptance_checks(state, abort_free_mib, budget_seconds)
    ok = all(c["ok"] for c in checks)
    state["acceptance"] = {
        "ok": ok,
        "passed": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
        "checks": checks,
        "failed_ids": [c["id"] for c in checks if not c["ok"]],
    }
    if ok:
        state["status"] = "PASS"
        return EXIT_PASS
    if state.get("status") in {None, "RUNNING", "PASS"}:
        state["status"] = "GATE_FAILED"
    reasons = "; ".join(f"#{c['id']} {c['name']} ({c['detail']})" for c in checks if not c["ok"])
    state["gate_failure_reason"] = reasons
    state.setdefault("error", reasons)
    return EXIT_GATE_FAILED if fallback_exit in {EXIT_PASS, EXIT_GATE_FAILED} else fallback_exit



def batched_action_shape(num_envs: int, action_shape: tuple[int, ...]) -> tuple[int, ...]:
    """`(num_envs, *action_shape)` - never a bare `(action_dim,)`.

    `BaseEnv._step_action` (sapien_env.py:1108) only inserts a batch dimension
    when `num_envs == 1`; for `num_envs > 1` an unbatched action is passed
    straight to the controller. We always batch explicitly.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    return (int(num_envs), *tuple(int(x) for x in action_shape))


def _inner_wrapper(cur: Any) -> Any:
    """Next env down the chain.

    gymnasium wrappers expose `.env`; `ManiSkillVectorEnv` stores the wrapped env
    in `._env` (vector/wrappers/gymnasium.py:46-48). Walking only `.env` stops at
    the vector wrapper and silently loses the horizon - that is exactly the bug
    that made the first host-side A0 report `wrapper_chain=['ManiSkillVectorEnv']`
    and `max_episode_steps=None`.
    """
    for attr in ("env", "_env"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur:
            return nxt
    return None


def episode_horizon(env: Any) -> tuple[int | None, list[str], str | None]:
    """Walk the wrapper chain for the real step budget.

    gymnasium 1.1.1 leaves `env.spec.max_episode_steps` as None for ManiSkill;
    the budget lives on ManiSkill's own `TimeLimitWrapper._max_episode_steps`.
    Returns (horizon, chain_names, source_wrapper) so the evidence says *which*
    layer supplied the number instead of just asserting one.
    """
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


def _tensor_stats(x: Any) -> dict[str, Any]:
    """Small, JSON-safe fingerprint of an observation tensor."""
    try:
        flat = x.float().flatten()
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "sum": round(float(flat.sum()), 6),
            "absmax": round(float(flat.abs().max()), 6),
            "first8": [round(float(v), 6) for v in flat[:8].tolist()],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_self_test() -> int:
    """Pure logic checks. Imports numpy only; never touches CUDA or ManiSkill."""
    import numpy as np

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    check("batched shape n=1", batched_action_shape(1, (8,)) == (1, 8),
          str(batched_action_shape(1, (8,))))
    check("batched shape n=256", batched_action_shape(256, (8,)) == (256, 8),
          str(batched_action_shape(256, (8,))))
    check("batched shape never bare", batched_action_shape(4, (8,)) != (8,))
    try:
        batched_action_shape(0, (8,))
        check("reject n=0", False)
    except ValueError:
        check("reject n=0", True)

    arr = np.zeros(batched_action_shape(7, (8,)), dtype=np.float32)
    check("numpy zeros shape + dtype", arr.shape == (7, 8) and arr.dtype == np.float32)

    check("vram ok", vram_verdict(4096, 3072) == "OK")
    check("vram refuse", vram_verdict(2048, 3072) == "REFUSE")
    check("vram unknown", vram_verdict(None, 3072) == "UNKNOWN")

    d = Deadline(10.0, anchor=time.monotonic())
    check("deadline fresh", not d.expired() and d.remaining() <= 10.0)
    d2 = Deadline(10.0, anchor=time.monotonic() - 11.0)
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

    chain = _W(_W(_W(steps=50)))
    horizon, names, source = episode_horizon(chain)
    check("horizon walk", horizon == 50, f"horizon={horizon} chain={names}")
    check("horizon names", len(names) == 3, str(names))
    check("horizon source recorded", source == "_W", str(source))

    class _V:
        """Mimics ManiSkillVectorEnv: inner env lives in `_env`, not `env`."""

        def __init__(self, inner):
            self._env = inner

    vec_chain = _V(_W(_W(steps=50)))
    horizon_v, names_v, source_v = episode_horizon(vec_chain)
    check("horizon reaches through ManiSkillVectorEnv", horizon_v == 50,
          f"horizon={horizon_v} chain={names_v}")
    check("vector wrapper chain recorded", names_v == ["_V", "_W", "_W"] and source_v == "_W",
          str(names_v))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nested" / "out.json"
        atomic_write_json(p, {"a": 1})
        check("atomic json write", p.is_file() and json.loads(p.read_text())["a"] == 1)
        rec = Recorder(p)
        rec.phase("build", time.monotonic())
        check("recorder flush", json.loads(p.read_text())["timings_s"].get("build") is not None)

    r = on_fire_factory_probe()
    check("watchdog factory callable", callable(r))

    check("telemetry none", telemetry_verdict(None)[0] == "UNAVAILABLE")
    check("telemetry empty dict", telemetry_verdict({})[0] == "UNAVAILABLE")
    check("telemetry missing free", telemetry_verdict({"memory.used": 100})[0] == "UNAVAILABLE")
    check("telemetry negative free", telemetry_verdict({"memory.free": -1})[0] == "UNAVAILABLE")
    check("telemetry string free", telemetry_verdict({"memory.free": "4096"})[0] == "UNAVAILABLE")
    check("telemetry ok", telemetry_verdict({"memory.free": 6775})[0] == "AVAILABLE")

    rc9 = {"binary": "/usr/bin/nvidia-smi", "returncode": 9,
           "stderr_head": ["NVIDIA-SMI has failed because it couldn't communicate with the "
                           "NVIDIA driver."], "exception": None, "command": "nvidia-smi ..."}
    verdict9, reason9 = telemetry_verdict(None, rc9)
    check("telemetry rc=9 names the exit code", verdict9 == "UNAVAILABLE" and "exited 9" in reason9,
          reason9)
    check("telemetry rc=9 keeps the driver message", "communicate with the NVIDIA driver" in reason9)
    no_binary = {"binary": None, "returncode": None, "stderr_head": None,
                 "exception": "nvidia-smi not found on PATH", "command": None}
    check("telemetry missing binary is distinguished",
          "not found on PATH" in telemetry_verdict(None, no_binary)[1])

    good = {
        "run_completed": True,
        "status": "PASS",
        "backend": {"resolved": "physx_cuda", "gpu_sim_enabled": True, "max_episode_steps": 50},
        "action": {"actual_shape": [1, 8]},
        "steps": [
            {"terminated": 0, "truncated": 0, "elapsed_steps": 1, "vram_free_mib": 6700},
            {"terminated": 0, "truncated": 0, "elapsed_steps": 2, "vram_free_mib": 6600},
            {"terminated": 0, "truncated": 0, "elapsed_steps": 3, "vram_free_mib": 6600},
        ],
        "vram": {"device_before": {"memory.free": 6775}, "device_after": {"memory.free": 6600}},
        "elapsed_s": 12.5,
    }
    ok_checks = acceptance_checks(good, 1536, 100.0)
    check("gate: good run 12/12", len(ok_checks) == 12 and all(c["ok"] for c in ok_checks),
          str([c["id"] for c in ok_checks if not c["ok"]]))

    bad_step = {**good, "steps": [dict(s) for s in good["steps"]]}
    bad_step["steps"][1]["truncated"] = 1
    check("gate: truncation caught",
          not all(c["ok"] for c in acceptance_checks(bad_step, 1536, 100.0)))

    no_telemetry = {**good, "vram": {"device_before": {"memory.free": None},
                                     "device_after": {"memory.free": 6600}}}
    failed_ids = [c["id"] for c in acceptance_checks(no_telemetry, 1536, 100.0) if not c["ok"]]
    check("gate: missing telemetry caught", 10 in failed_ids, str(failed_ids))

    slow = {**good, "elapsed_s": 130.0}
    check("gate: wall clock caught",
          not all(c["ok"] for c in acceptance_checks(slow, 1536, 100.0)))

    stuck = {**good, "run_completed": False, "status": "ERROR"}
    check("gate: incomplete run caught",
          not all(c["ok"] for c in acceptance_checks(stuck, 1536, 100.0)))

    probe_state = json.loads(json.dumps(good))
    code = finalize_with_gate(probe_state, 1536, 100.0, EXIT_PASS)
    check("gate: finalize pass -> exit 0", code == EXIT_PASS and probe_state["acceptance"]["ok"])

    failing = json.loads(json.dumps(stuck))
    code = finalize_with_gate(failing, 1536, 100.0, EXIT_PASS)
    check("gate: finalize fail -> non-zero",
          code != 0 and failing["acceptance"]["ok"] is False,
          f"exit={code} status={failing.get('status')}")

    early = {**json.loads(json.dumps(stuck)), "status": "FAIL_SIM_BACKEND"}
    code = finalize_with_gate(early, 1536, 100.0, EXIT_BACKEND)
    check("gate: early abort keeps its own exit code",
          code == EXIT_BACKEND and early["status"] == "FAIL_SIM_BACKEND", f"exit={code}")

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return EXIT_PASS if not failures else 1


def on_fire_factory_probe():
    """Tiny helper used by the self-test to assert the watchdog callback shape."""
    fired: list[str] = []

    def _cb() -> None:
        fired.append("x")

    t = arm_watchdog(0.05, _cb)
    t.join(timeout=1.0)
    assert fired, "watchdog thread did not fire"
    return _cb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A0: 1-env physx_cuda PickCube smoke, 3 zero actions, bounded.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--robot", default="panda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3, help="zero-action steps; A0 is fixed at 3")
    p.add_argument("--budget-seconds", type=float, default=100.0,
                   help="internal monotonic budget (< external `timeout 120`)")
    p.add_argument("--min-free-mib", type=int, default=3072,
                   help="refuse to start below this much free device memory")
    p.add_argument("--abort-free-mib", type=int, default=1536,
                   help="abort if free device memory drops below this during the run")
    p.add_argument("--output", type=Path, default=Path("results/m2_gpu_smoke_a0.json"))
    p.add_argument("--self-test", action="store_true", help="run pure logic checks and exit")
    p.add_argument("--allow-version-mismatch", action="store_true",
                   help="continue even if the installed stack differs from the pinned versions")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.steps != 3:
        print(f"refusing: Stage A0 is fixed at exactly 3 steps, got {args.steps}", file=sys.stderr)
        return EXIT_ERROR

    rec = Recorder(args.output)
    state = rec.data
    state["config"] = {
        "env_id": args.env_id,
        "robot": args.robot,
        "num_envs": 1,
        "seed": args.seed,
        "steps": args.steps,
        "obs_mode": "state",
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "sim_backend": "physx_cuda",
        "render_mode": None,
        "budget_seconds": args.budget_seconds,
        "min_free_mib": args.min_free_mib,
        "abort_free_mib": args.abort_free_mib,
    }

    deadline = Deadline(args.budget_seconds)

    def _on_timeout() -> None:
        state["status"] = "BUDGET_EXCEEDED"
        state["timed_out_at_phase"] = state.get("phase", "unknown")
        state["elapsed_s"] = elapsed()
        try:
            rec.flush()
        finally:
            print(f"TIMEOUT after {elapsed()}s at phase {state.get('phase')}", flush=True)
            os._exit(EXIT_BUDGET)

    timer = arm_watchdog(args.budget_seconds, _on_timeout)
    env = None
    exit_code = EXIT_ERROR

    def finish(code: int) -> int:
        """Attach the 12-condition acceptance gate, then return the exit code."""
        state["elapsed_s"] = elapsed()
        return finalize_with_gate(state, args.abort_free_mib, args.budget_seconds, code)

    try:
        state["phase"] = "vram_telemetry"
        probe, probe_diag = smi_query_ex(
            "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
        )
        tel_verdict, tel_reason = telemetry_verdict(probe, probe_diag)
        state["vram"]["telemetry"] = {
            "source": "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "required": True,
            "verdict": tel_verdict,
            "reason": tel_reason,
            "probe": probe,
            "nvidia_smi_path": shutil.which("nvidia-smi"),
            "diagnostics": probe_diag,
        }
        state["vram"]["device_before"] = probe
        state["vram"]["compute_apps_before"] = smi_compute_apps()
        rec.flush()
        print(f"[vram] telemetry verdict={tel_verdict} :: {tel_reason}", flush=True)
        if tel_verdict != "AVAILABLE":
            state["status"] = "VRAM_TELEMETRY_UNAVAILABLE"
            state["error"] = (
                "device-level VRAM telemetry is mandatory for Stage A0 and could not be "
                f"read before env creation: {tel_reason}; aborting without building any scene"
            )
            rec.flush()
            print(f"FATAL: {state['error']}", file=sys.stderr)
            return finish(EXIT_NO_VRAM_TELEMETRY)

        free_before = probe.get("memory.free") if probe else None
        verdict = vram_verdict(free_before, args.min_free_mib)
        state["vram"]["preflight_verdict"] = verdict
        rec.flush()
        print(f"[vram] preflight verdict={verdict} free={free_before} MiB"
              f" (need >= {args.min_free_mib})", flush=True)
        if verdict == "REFUSE":
            state["status"] = "SKIPPED_LOW_VRAM"
            state["error"] = (
                f"free device memory {free_before} MiB < required {args.min_free_mib} MiB; "
                "no scene was built"
            )
            rec.flush()
            return finish(EXIT_LOW_VRAM)

        t = time.monotonic()
        import torch
        import gymnasium as gym
        import mani_skill
        import sapien
        import mani_skill.envs  # noqa: F401  (registers the tasks)
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        rec.phase("import", t)
        state["versions"] = {
            "mani_skill": getattr(mani_skill, "__version__", "unknown"),
            "sapien": getattr(sapien, "__version__", "unknown"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gymnasium": getattr(gym, "__version__", "unknown"),
        }
        mismatches = {
            "mani_skill": state["versions"]["mani_skill"] != PINNED_MANISKILL,
            "sapien": state["versions"]["sapien"] != PINNED_SAPIEN,
            "torch": not state["versions"]["torch"].startswith(PINNED_TORCH),
        }
        state["version_mismatches"] = [k for k, v in mismatches.items() if v]
        if state["version_mismatches"] and not args.allow_version_mismatch:
            state["status"] = "VERSION_MISMATCH"
            rec.flush()
            print(f"refusing: version mismatch {state['version_mismatches']}", file=sys.stderr)
            return finish(EXIT_VERSION)
        deadline.check("import")

        state["phase"] = "cuda_preflight"
        state["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if state["cuda"]["available"]:
            state["cuda"]["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            state["cuda"]["capability"] = f"{props.major}.{props.minor}"
            state["cuda"]["total_mib"] = round(props.total_memory / 2**20)
        else:
            state["status"] = "FAIL_SIM_BACKEND"
            state["error"] = "torch.cuda.is_available() is False; physx_cuda cannot be used"
            rec.flush()
            print("FATAL: CUDA not available in this interpreter", file=sys.stderr)
            return finish(EXIT_BACKEND)
        deadline.check("cuda_preflight")

        state["phase"] = "build_env"
        t = time.monotonic()
        base_env = gym.make(
            args.env_id,
            num_envs=1,
            robot_uids=args.robot,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            reward_mode="normalized_dense",
            sim_backend="physx_cuda",
            render_mode=None,
        )
        env = ManiSkillVectorEnv(base_env, auto_reset=True, ignore_terminations=False)
        rec.phase("build_env", t)

        state["phase"] = "verify_backend"
        unwrapped = env.unwrapped
        backend_name = str(getattr(getattr(unwrapped, "backend", None), "sim_backend", "unknown"))
        gpu_sim = bool(getattr(unwrapped, "gpu_sim_enabled", False))
        state["backend"] = {
            "requested": "physx_cuda",
            "resolved": backend_name,
            "gpu_sim_enabled": gpu_sim,
            "sim_device": str(getattr(unwrapped, "_sim_device", "unknown")),
            "env_device": str(getattr(unwrapped, "device", "unknown")),
            "num_envs": int(getattr(unwrapped, "num_envs", -1)),
        }
        horizon, chain, horizon_source = episode_horizon(env)
        state["backend"]["wrapper_chain"] = chain
        state["backend"]["max_episode_steps"] = horizon
        state["backend"]["horizon_source"] = horizon_source
        state["spaces"] = {
            "single_action_space": list(env.single_action_space.shape),
            "action_space": list(env.action_space.shape),
            "single_obs_space_len": int(getattr(env.single_observation_space, "shape", (0,))[0]),
        }
        rec.flush()
        if backend_name != "physx_cuda" or not gpu_sim:
            state["status"] = "FAIL_SIM_BACKEND"
            state["error"] = (
                f"requested physx_cuda but resolved backend={backend_name}, gpu_sim_enabled={gpu_sim}"
            )
            rec.flush()
            print(f"FATAL: {state['error']}", file=sys.stderr)
            return finish(EXIT_BACKEND)
        deadline.check("verify_backend")

        state["phase"] = "reset"
        t = time.monotonic()
        obs, info = env.reset(seed=args.seed)
        torch.cuda.synchronize()
        rec.phase("reset", t)
        state["reset"] = {
            "seed": args.seed,
            "obs": _tensor_stats(obs),
            "info_keys": sorted(str(k) for k in (info or {}).keys()),
            "elapsed_steps_after_reset": int(unwrapped.elapsed_steps.max().item()),
            "vram_free_mib": (smi_query("memory.free") or {}).get("memory.free"),
        }
        rec.flush()
        deadline.check("reset")

        state["phase"] = "steps"
        shape = batched_action_shape(1, tuple(env.single_action_space.shape))
        action = torch.zeros(shape, dtype=torch.float32, device=unwrapped.device)
        assert tuple(action.shape) == shape, f"action shape {tuple(action.shape)} != {shape}"
        state["action"] = {
            "requested_shape": list(shape),
            "actual_shape": list(action.shape),
            "dtype": str(action.dtype),
            "device": str(action.device),
            "values": "all zeros",
        }
        rec.flush()

        torch.cuda.reset_peak_memory_stats()
        for i in range(1, args.steps + 1):
            deadline.check(f"step{i}")
            t = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(action)
            torch.cuda.synchronize()
            dt = rec.phase(f"step{i}", t)

            term_i = int(bool(terminated.any().item())) if hasattr(terminated, "any") else int(bool(terminated))
            trunc_i = int(bool(truncated.any().item())) if hasattr(truncated, "any") else int(bool(truncated))
            vram_probe, vram_probe_diag = smi_query_ex("memory.free")
            tel_ok, tel_why = telemetry_verdict(vram_probe, vram_probe_diag)
            step_rec = {
                "step": i,
                "wall_s": dt,
                "reward": round(float(reward.float().mean().item()), 6),
                "terminated": term_i,
                "truncated": trunc_i,
                "success": bool(info["success"].any().item()) if "success" in info else None,
                "elapsed_steps": int(unwrapped.elapsed_steps.max().item()),
                "auto_reset_used": "final_observation" in info,
                "obs": _tensor_stats(obs),
                "vram_free_mib": (vram_probe or {}).get("memory.free"),
                "vram_telemetry": {"verdict": tel_ok, "reason": tel_why},
                "torch_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }
            state["steps"].append(step_rec)
            rec.flush()
            print(json.dumps(step_rec, ensure_ascii=False), flush=True)

            if tel_ok != "AVAILABLE":
                state["status"] = "VRAM_TELEMETRY_LOST"
                state["error"] = (
                    f"device-level VRAM telemetry lost at step {i}: {tel_why}; "
                    "A0 requires device-level memory evidence for every step, aborting"
                )
                rec.flush()
                print(f"FATAL: {state['error']}", file=sys.stderr)
                return finish(EXIT_NO_VRAM_TELEMETRY)

            free_now = step_rec["vram_free_mib"]
            if free_now is not None and free_now < args.abort_free_mib:
                state["status"] = "VRAM_GUARD_TRIPPED"
                state["error"] = (
                    f"free device memory {free_now} MiB < abort threshold {args.abort_free_mib} MiB"
                )
                rec.flush()
                return finish(EXIT_ERROR)

            if trunc_i or term_i:
                state["status"] = "UNEXPECTED_DONE"
                state["error"] = (
                    f"done at step {i} of {args.steps} with zero action "
                    f"(terminated={term_i}, truncated={trunc_i}); "
                    "A0 expects no episode boundary within 3 steps"
                )
                rec.flush()
                print(f"FAIL: {state['error']}", file=sys.stderr)
                return finish(EXIT_UNEXPECTED_DONE)

        state["phase"] = "measure_memory"
        torch.cuda.synchronize()
        state["vram"]["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        state["vram"]["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        after, after_diag = smi_query_ex("memory.total,memory.used,memory.free,utilization.gpu")
        state["vram"]["device_after"] = after
        state["vram"]["device_after_diagnostics"] = after_diag
        state["vram"]["device_after_verdict"] = telemetry_verdict(after, after_diag)[0]
        state["vram"]["compute_apps_after"] = smi_compute_apps()
        state["vram"]["attribution_note"] = (
            "torch_peak_* counts only the PyTorch caching allocator. SAPIEN/PhysX "
            "allocate directly through CUDA, so total usage must be read from "
            "nvidia-smi memory.used before/after; the delta is the device-level "
            "cost of this scene (includes desktop noise)."
        )
        rec.flush()

        state["run_completed"] = True
        exit_code = finish(EXIT_PASS)
    except BudgetExhausted as exc:
        state["status"] = "BUDGET_EXCEEDED"
        state["error"] = str(exc)
        exit_code = EXIT_BUDGET
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "out of memory" in low or isinstance(exc, MemoryError):
            state["status"] = "OOM"
        elif "cuda" in low and "not available" in low:
            state["status"] = "FAIL_SIM_BACKEND"
        else:
            state["status"] = "ERROR"
        state["error"] = f"{type(exc).__name__}: {msg[:500]}"
        exit_code = EXIT_ERROR
    finally:
        timer.cancel()
        state["phase"] = "cleanup"
        t = time.monotonic()
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        rec.phase("cleanup", t)
        state["elapsed_s"] = elapsed()
        state["finished_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec.flush()
        acc = state.get("acceptance") or {}
        if acc:
            print(f"acceptance: {acc.get('passed')}/{acc.get('total')} passed, "
                  f"failed_ids={acc.get('failed_ids')}", flush=True)
        print(f"status={state['status']} elapsed={state['elapsed_s']}s -> {args.output}",
              flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
