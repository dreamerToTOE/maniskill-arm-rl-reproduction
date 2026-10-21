#!/usr/bin/env python3
"""Stage A1 launcher - one isolated process per env count, stop on first failure.

Why a launcher instead of a single process:
  * "each configuration separately so a failure does not affect the remaining
    experiment records" - every level writes its own JSON and its own log, so a
    crash, OOM or hard kill at 256 envs cannot truncate the 16/64/128 evidence;
  * "a genuine per-configuration timeout" - the external `timeout` bounds the
    *whole* child process, including scene construction, so a hung GPU build
    cannot silently consume the next level's budget;
  * "stop the entire experiment immediately if any level fails" - the next level
    is simply never spawned; the remaining ones are recorded as
    NOT_RUN_AFTER_FAILURE rather than being left ambiguous;
  * "do not automatically retry" - every record carries `attempts: 1`, and there
    is no retry loop anywhere in this file.

Four nested guards per level: external `timeout` (150 s) > child process budget
(`--total-seconds 120`, monotonic + watchdog) > child level budget
(`--level-seconds 90`, covering build/reset/warmup/measure/cleanup) > the
launcher's own `subprocess` timeout (180 s) as the last backstop.

This measures SIMULATOR steps/second only - no policy, no optimizer, no
rendering. It is not training throughput and not a baseline reproduction.

Usage:
  ./.venv-maniskill/bin/python scripts/m2_a1_run_all.py --self-test
  ./.venv-maniskill/bin/python scripts/m2_a1_run_all.py --dry-run
  ./.venv-maniskill/bin/python scripts/m2_a1_run_all.py          # real run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m2_gpu_sim_probe import (  # noqa: E402  (no torch import at module level)
    device_nodes,
    smi_compute_apps,
    smi_query_ex,
    telemetry_verdict,
)

DEFAULT_LEVELS = [16, 64, 128, 256]
LEVEL_BUDGET_S = 90.0
PROCESS_BUDGET_S = 120.0
EXTERNAL_LIMIT_S = 150.0
LAUNCHER_BACKSTOP_S = 180.0
MIN_FREE_MIB = 3072
ABORT_FREE_MIB = 1536

EXIT_PASS = 0
EXIT_PARTIAL = 1
EXIT_FAIL = 2
EXIT_BLOCKED = 8


def build_command(python: str, probe: Path, n: int, out: Path,
                  level_budget: float, process_budget: float,
                  external_limit: float, min_free: int, abort_free: int,
                  vram_every: int) -> list[str]:
    """The exact argv of one level. Kept pure so it can be unit-tested."""
    inner = [
        python, str(probe),
        "--env-id", "PickCube-v1",
        "--robot", "panda",
        "--env-counts", str(n),
        "--warmup-steps", "10",
        "--steps", "100",
        "--level-seconds", f"{level_budget:g}",
        "--total-seconds", f"{process_budget:g}",
        "--external-limit-s", f"{external_limit:g}",
        "--min-free-mib", str(min_free),
        "--abort-free-mib", str(abort_free),
        "--vram-every", str(vram_every),
        "--output", str(out),
    ]
    exe = shutil.which("timeout")
    if exe is None:
        return inner
    return [exe, "--signal=TERM", "--kill-after=10s", f"{external_limit:g}", *inner]


def extract_measurements(payload: dict[str, Any], exit_code: int) -> dict[str, Any]:
    """Flatten one level's evidence into the 12 required measurements."""
    levels = payload.get("levels") or []
    lv = levels[0] if levels else {}
    counters = lv.get("counters_total") or {}
    total_mib = (lv.get("vram_after") or {}).get("memory.total") \
        or (payload.get("gpu") or {}).get("memory.total")
    free_min = lv.get("vram_free_min_mib")
    return {
        "num_envs": lv.get("num_envs"),
        "status": lv.get("status", payload.get("status")),
        "exit_code": exit_code,
        "attempts": 1,
        # 1. initialisation
        "init_s": lv.get("build_s"),
        "import_s": payload.get("import_s"),
        "reset_s": lv.get("reset_s"),
        "level_elapsed_s": lv.get("level_elapsed_s"),
        "process_elapsed_s": payload.get("elapsed_s"),
        # 2. warmup
        "warmup_s": lv.get("warmup_s"),
        "warmup_steps_done": lv.get("warmup_steps_done"),
        # 3. measured simulation
        "measure_s": lv.get("measure_s"),
        "measure_steps_done": lv.get("measure_steps_done"),
        "total_env_steps": lv.get("total_env_steps"),
        # 4. aggregate throughput
        "sim_sps": lv.get("sim_sps"),
        "sim_steps_per_s": lv.get("sim_steps_per_s"),
        "ms_per_sim_step": lv.get("ms_per_sim_step"),
        # 5. memory, device level
        "device_used_baseline_mib": lv.get("device_used_baseline_mib"),
        "device_used_peak_mib": lv.get("device_used_peak_mib_est"),
        "device_used_after_build_mib": (lv.get("vram_after_build") or {}).get("memory.used"),
        "vram_free_min_mib": free_min,
        "vram_free_after_mib": (lv.get("vram_after") or {}).get("memory.free"),
        "device_used_increase_mib": lv.get("device_used_delta_mib"),
        "device_total_mib": total_mib,
        "headroom_min_mib": (total_mib - lv["device_used_peak_mib_est"])
        if (isinstance(total_mib, int) and isinstance(lv.get("device_used_peak_mib_est"), int))
        else None,
        # 6. torch allocator, a lower bound only
        "torch_peak_alloc_mib": lv.get("torch_peak_alloc_mib"),
        "torch_peak_reserved_mib": lv.get("torch_peak_reserved_mib"),
        # 7. episode boundary
        "counters_total": counters,
        "counters_warmup": lv.get("counters_warmup"),
        "counters_measure": lv.get("counters_measure"),
        "truncation_observed": lv.get("truncation_observed"),
        # 8. warnings / errors
        "warnings": lv.get("warnings") or [],
        "error": lv.get("error"),
        # 9. shape/backend evidence
        "backend": lv.get("backend"),
        "max_episode_steps": lv.get("max_episode_steps"),
        "horizon_source": lv.get("horizon_source"),
        "action": lv.get("action"),
        "vram_verdict": lv.get("vram_verdict"),
        "vram_free_samples": lv.get("vram_free_samples"),
    }


def compute_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Speedup vs the first *measured* level, and scaling efficiency.

    Efficiency = speedup / env-ratio. Below 1.0 means per-step overhead grows
    with parallelism (diminishing returns); it is a simulator-only number and
    says nothing about training throughput.
    """
    valid = [r for r in rows if r.get("status") == "OK" and r.get("sim_sps")]
    if not valid:
        return rows
    base = valid[0]
    base_sps = float(base["sim_sps"])
    base_n = int(base["num_envs"])
    for r in rows:
        if r.get("status") != "OK" or not r.get("sim_sps"):
            r["speedup_vs_base"] = None
            r["scaling_efficiency"] = None
            r["sps_per_env"] = None
            r["vram_mib_per_env"] = None
            continue
        sps = float(r["sim_sps"])
        n = int(r["num_envs"])
        speedup = sps / base_sps
        r["speedup_vs_base"] = round(speedup, 3)
        r["scaling_efficiency"] = round(speedup / (n / base_n), 4)
        r["sps_per_env"] = round(sps / n, 2)
        inc = r.get("device_used_increase_mib")
        r["vram_mib_per_env"] = round(inc / n, 1) if isinstance(inc, (int, float)) else None
    return rows


def summarize(rows: list[dict[str, Any]], wall_clock_s: float,
              preflight: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Verdict + aggregate payload. Pure function: unit-tested with fakes."""
    ok = [r for r in rows if r.get("status") == "OK" and r.get("exit_code") == 0]
    failed = [r for r in rows if r.get("status") not in {"OK", None} and
              r.get("status") != "NOT_RUN_AFTER_FAILURE"]
    if preflight.get("blocked"):
        verdict = "BLOCKED"
    elif len(ok) == len(rows) and rows:
        verdict = "PASS"
    elif ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    executed = [r for r in rows if r.get("status") not in
                {"NOT_RUN_PREFLIGHT_BLOCKED", "NOT_RUN_AFTER_FAILURE"}]
    return {
        "kind": "m2_a1_benchmark_summary",
        "stage": "A1",
        "verdict": verdict,
        "levels_executed": len(executed),
        "notes": ([
            "verdict BLOCKED: device-level VRAM telemetry was unavailable, so no level "
            "was spawned. This file documents a refused attempt, not a benchmark. "
            "Rerunning the launcher overwrites it."
        ] if verdict == "BLOCKED" else []),
        "scope_warning": (
            "SIMULATOR steps/second only (zero actions, render_mode=None, no policy, "
            "no gradients). This is NOT training throughput, NOT a reproduction of the "
            "official baseline, and must not be compared with published numbers."
        ),
        "config": cfg,
        "preflight": preflight,
        "rows": rows,
        "levels_ok": [r["num_envs"] for r in ok],
        "levels_failed": [{"num_envs": r.get("num_envs"), "status": r.get("status"),
                           "error": r.get("error")} for r in failed],
        "levels_not_run": [r["num_envs"] for r in rows
                           if r.get("status") == "NOT_RUN_AFTER_FAILURE"],
        "total_wall_clock_s": round(wall_clock_s, 3),
        "attempts_per_level": 1,
        "generated_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def preflight_gpu() -> dict[str, Any]:
    probe, diag = smi_query_ex(
        "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu")
    verdict, reason = telemetry_verdict(probe, diag)
    return {
        "telemetry_verdict": verdict,
        "telemetry_reason": reason,
        "device_nodes": device_nodes(),
        "gpu": probe,
        "diagnostics": diag,
        "other_compute_apps": smi_compute_apps(),
        "blocked": verdict != "AVAILABLE",
    }


def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    cmd = build_command("/py", Path("p.py"), 16, Path("o.json"), 90.0, 120.0, 150.0,
                        3072, 1536, 10)
    check("command carries one level", "--env-counts" in cmd and "16" in cmd, str(cmd[:4]))
    check("command carries step protocol",
          cmd.count("10") >= 1 and cmd[cmd.index("--steps") + 1] == "100")
    check("command carries both budgets",
          cmd[cmd.index("--level-seconds") + 1] == "90"
          and cmd[cmd.index("--total-seconds") + 1] == "120")
    check("command carries VRAM gates",
          cmd[cmd.index("--min-free-mib") + 1] == "3072"
          and cmd[cmd.index("--abort-free-mib") + 1] == "1536")
    if shutil.which("timeout"):
        check("external timeout wraps the child",
              cmd[0] == shutil.which("timeout") and "150" in cmd[:4], str(cmd[:4]))

    ok_payload = {
        "import_s": 2.0, "elapsed_s": 12.0, "status": "PASS",
        "gpu": {"memory.total": 8192},
        "levels": [{
            "num_envs": 16, "status": "OK", "build_s": 3.0, "reset_s": 0.1,
            "warmup_s": 0.4, "measure_s": 2.0, "measure_steps_done": 100,
            "warmup_steps_done": 10, "total_env_steps": 1600, "sim_sps": 800.0,
            "sim_steps_per_s": 50.0, "ms_per_sim_step": 20.0,
            "device_used_baseline_mib": 2500, "device_used_peak_mib_est": 3400,
            "vram_free_min_mib": 4792, "device_used_delta_mib": 900,
            "vram_after": {"memory.total": 8192, "memory.used": 3400, "memory.free": 4792},
            "vram_after_build": {"memory.used": 3200},
            "counters_total": {"terminated_envs": 0, "truncated_envs": 32,
                               "auto_reset_envs": 32, "terminated_steps": 0,
                               "truncated_steps": 2, "auto_reset_steps": 2},
            "truncation_observed": True, "torch_peak_alloc_mib": 8.2,
            "level_elapsed_s": 9.0,
        }],
    }
    m = extract_measurements(ok_payload, 0)
    check("extract num_envs", m["num_envs"] == 16)
    check("extract sps", m["sim_sps"] == 800.0)
    check("extract baseline memory", m["device_used_baseline_mib"] == 2500)
    check("extract peak estimate", m["device_used_peak_mib"] == 3400)
    check("extract headroom", m["headroom_min_mib"] == 4792)
    check("extract truncation counters", m["counters_total"]["truncated_envs"] == 32)
    check("extract warnings default empty", m["warnings"] == [])

    rows = [
        {"num_envs": 16, "status": "OK", "exit_code": 0, "sim_sps": 800.0,
         "device_used_increase_mib": 900},
        {"num_envs": 64, "status": "OK", "exit_code": 0, "sim_sps": 2600.0,
         "device_used_increase_mib": 1800},
        {"num_envs": 128, "status": "ERROR", "exit_code": 5, "sim_sps": None,
         "device_used_increase_mib": None},
        {"num_envs": 256, "status": "NOT_RUN_AFTER_FAILURE", "exit_code": None,
         "sim_sps": None, "device_used_increase_mib": None},
    ]
    rows = compute_scaling(rows)
    check("speedup uses the first OK level", rows[1]["speedup_vs_base"] == 3.25)
    check("efficiency = speedup / env-ratio",
          rows[1]["scaling_efficiency"] == round(3.25 / 4, 4), str(rows[1]["scaling_efficiency"]))
    check("failed level has no scaling numbers", rows[2]["speedup_vs_base"] is None)

    s = summarize(rows, 41.0, {"blocked": False}, {"levels": [16, 64, 128, 256]})
    check("partial verdict when one level fails", s["verdict"] == "PARTIAL", s["verdict"])
    check("not-run level listed", s["levels_not_run"] == [256], str(s["levels_not_run"]))
    check("failed level captured with error field",
          s["levels_failed"] and s["levels_failed"][0]["num_envs"] == 128)
    check("no retry budget", s["attempts_per_level"] == 1)

    all_ok = compute_scaling([
        {"num_envs": 16, "status": "OK", "exit_code": 0, "sim_sps": 800.0,
         "device_used_increase_mib": 900},
        {"num_envs": 64, "status": "OK", "exit_code": 0, "sim_sps": 3200.0,
         "device_used_increase_mib": 2000},
    ])
    check("full pass verdict",
          summarize(all_ok, 20.0, {"blocked": False}, {})["verdict"] == "PASS")
    check("all-failed verdict is FAIL",
          summarize([{"num_envs": 16, "status": "OOM", "exit_code": 5}],
                    5.0, {"blocked": False}, {})["verdict"] == "FAIL")
    check("blocked preflight wins over rows",
          summarize([], 0.1, {"blocked": True}, {})["verdict"] == "BLOCKED")

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return EXIT_PASS if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A1 launcher: 16/64/128/256 as isolated processes, stop on failure.")
    p.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    p.add_argument("--python", default=str(Path(".venv-maniskill/bin/python").absolute()),
                   help="interpreter for the child processes; must be the isolated venv one "
                        "(the path is deliberately NOT symlink-resolved, or the venv would be "
                        "silently dropped and mani_skill would be missing)")
    p.add_argument("--probe", type=Path, default=Path("scripts/m2_gpu_sim_probe.py"))
    p.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    p.add_argument("--summary", type=Path, default=Path("results/m2_a1_benchmark_summary.json"))
    p.add_argument("--level-seconds", type=float, default=LEVEL_BUDGET_S)
    p.add_argument("--total-seconds", type=float, default=PROCESS_BUDGET_S,
                   help="budget of ONE child process")
    p.add_argument("--external-limit-s", type=float, default=EXTERNAL_LIMIT_S)
    p.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    p.add_argument("--abort-free-mib", type=int, default=ABORT_FREE_MIB)
    p.add_argument("--vram-every", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-level commands without touching the GPU")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    levels = [int(x) for x in args.levels.split(",")]
    if levels != sorted(levels) or len(set(levels)) != len(levels):
        print("refusing: levels must be strictly ascending", file=sys.stderr)
        return EXIT_FAIL
    if max(levels) > 256 and not args.dry_run:
        print("refusing: Stage A1 is authorized up to 256 envs only", file=sys.stderr)
        return EXIT_FAIL

    cfg = {
        "env_id": "PickCube-v1", "robot": "panda", "levels": levels,
        "warmup_steps": 10, "measure_steps": 100,
        "level_budget_s": args.level_seconds, "process_budget_s": args.total_seconds,
        "external_timeout_s": args.external_limit_s,
        "launcher_backstop_s": LAUNCHER_BACKSTOP_S,
        "min_free_mib": args.min_free_mib, "abort_free_mib": args.abort_free_mib,
        "vram_every": args.vram_every, "obs_mode": "state",
        "control_mode": "pd_joint_delta_pos", "reward_mode": "normalized_dense",
        "sim_backend": "physx_cuda", "render_mode": None, "actions": "zero",
        "ppo_installed": False, "dependencies_installed": False,
    }

    if args.dry_run:
        print(json.dumps({"config": cfg, "commands": [
            {"level": n, "argv": build_command(
                args.python, args.probe, n, args.raw_dir / f"m2_a1_n{n}.json",
                args.level_seconds, args.total_seconds, args.external_limit_s,
                args.min_free_mib, args.abort_free_mib, args.vram_every)}
            for n in levels]}, indent=2))
        return EXIT_PASS

    t_all = time.monotonic()
    if not Path(args.python).is_file():
        print(f"refusing: interpreter not found: {args.python}", file=sys.stderr)
        return EXIT_FAIL
    pre = preflight_gpu()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if pre["blocked"]:
        for n in levels:
            rows.append({"num_envs": n, "status": "NOT_RUN_PREFLIGHT_BLOCKED",
                         "exit_code": None, "attempts": 1, "error": pre["telemetry_reason"]})
        payload = summarize(rows, time.monotonic() - t_all, pre, cfg)
        payload["verdict"] = "BLOCKED"
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        print(f"BLOCKED: {pre['telemetry_reason']}", file=sys.stderr)
        print(f"-> {args.summary}")
        return EXIT_BLOCKED

    stopped = False
    for n in levels:
        if stopped:
            rows.append({"num_envs": n, "status": "NOT_RUN_AFTER_FAILURE",
                         "exit_code": None, "attempts": 1,
                         "error": "not launched: an earlier level failed (no retry, no continuation)"})
            continue
        out_json = args.raw_dir / f"m2_a1_n{n}.json"
        log_path = args.raw_dir / f"m2_a1_n{n}.log"
        cmd = build_command(args.python, args.probe, n, out_json,
                            args.level_seconds, args.total_seconds, args.external_limit_s,
                            args.min_free_mib, args.abort_free_mib, args.vram_every)
        print(f"[n={n}] launching: {' '.join(cmd)}", flush=True)
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=LAUNCHER_BACKSTOP_S, check=False)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        wall = round(time.monotonic() - t0, 3)
        log_path.write_text(
            f"# cmd: {' '.join(cmd)}\n# wall_s: {wall}\n# exit: {rc}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n", encoding="utf-8")
        payload: dict[str, Any] = {}
        if out_json.is_file():
            try:
                payload = json.loads(out_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = {"status": "UNREADABLE_JSON", "error": str(exc), "levels": []}
        row = extract_measurements(payload, rc)
        row["attempts"] = 1
        row["child_wall_s"] = wall
        row["external_timeout_hit"] = timed_out or rc == 124
        row["log"] = str(log_path)
        row["raw_json"] = str(out_json)
        if row["status"] is None:
            row["status"] = "NO_OUTPUT"
        rows.append(row)
        print(f"[n={n}] status={row['status']} rc={rc} wall={wall}s sim_sps={row.get('sim_sps')}",
              flush=True)
        if row["status"] != "OK" or rc != 0:
            stopped = True
            print(f"[n={n}] STOPPING the experiment: this level failed "
                  f"(status={row['status']}, rc={rc})", flush=True)

    rows = compute_scaling(rows)
    payload = summarize(rows, time.monotonic() - t_all, pre, cfg)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"\nverdict={payload['verdict']} ok={payload['levels_ok']} "
          f"failed={[f['num_envs'] for f in payload['levels_failed']]} "
          f"wall={payload['total_wall_clock_s']}s -> {args.summary}", flush=True)
    return {"PASS": EXIT_PASS, "PARTIAL": EXIT_PARTIAL}.get(payload["verdict"], EXIT_FAIL)


if __name__ == "__main__":
    sys.exit(main())
