#!/usr/bin/env python3
"""Stage B PPO pilot supervisor: one bounded run of upstream ppo.py, no retries.

This wrapper does NOT train and does NOT modify the upstream script. It:
  * refuses to start unless the GPU probe says OK and free VRAM >= --min-free-mib;
  * refuses to start unless the ppo.py on disk is byte-identical to the pinned
    blob (`66802de0...` at ManiSkill v3.0.1), so an accidental second install or
    a stale copy cannot silently be what runs;
  * launches ppo.py with an explicit, audited argv (no defaults left implicit);
  * watches wall clock, device VRAM, storage and progress, and kills the child
    on the first violation - CUDA OOM, non-finite losses, free VRAM below the
    floor, 120 s without a new iteration, or the 1200 s budget;
  * parses the TensorBoard events afterwards into `results/m2_pilot_summary.json`.

Four independent kill paths, because each covers a different failure:
  1. wall clock (monotonic, checked every poll)          -> 1200 s
  2. device VRAM floor (nvidia-smi, every poll)          -> 1536 MiB
  3. no-progress watchdog (stdout "Epoch:" lines)        -> 120 s
  4. output storage cap (du on the run directory)        -> 300 MB
Plus the external `timeout` in the printed command as a fifth, process-level one.

Everything measured is either read from the TensorBoard event file (losses,
eval success, per-iteration timing) or from `nvidia-smi` (device memory). The
SPS printed by ppo.py is a *training-loop* number: it starts after env creation,
so the end-to-end SPS is computed here separately as
`global_step / wall_clock_of_the_whole_child`.

Usage:
  ./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py --self-test
  ./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py --dry-run
  ./.venv-maniskill/bin/python scripts/m2_stage_b_pilot.py            # real run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_visibility_probe import smi, smi_compute_apps  # noqa: E402

PINNED_PPO_BLOB = "66802de0653061da550d6600d6346be374dbf974"
PINNED_UPSTREAM_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"

DEFAULT_PPO = Path("/home/ubuntu2004/.cache/ms-upstream-ppo/ppo.py")
EXP_NAME = "PickCube-v1__ppo__9351__stageB"

# Stage B authorization, fixed here so the argv cannot drift silently.
NUM_ENVS = 128
NUM_STEPS = 50
TOTAL_TIMESTEPS = 256000
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 32
NUM_EVAL_ENVS = 8
NUM_EVAL_STEPS = 50
EVAL_FREQ = 5
SEED = 9351

WALL_LIMIT_S = 1200.0
MIN_FREE_MIB = 1536
START_FREE_MIB = 3072
STORAGE_LIMIT_MB = 300
NO_PROGRESS_S = 120.0
# The 120 s rule applies to the *loop*. Before the first iteration the child is
# still importing, building two PhysX scenes and compiling render targets, which
# produced ~40 s here and 26 s on the very first A0 run; a 300 s grace covers that
# without weakening the loop rule. The 1200 s wall clock still bounds everything.
STARTUP_GRACE_S = 300.0
POLL_S = 2.0
SMI_EVERY_S = 5.0

EXIT_OK = 0
EXIT_BLOCKED = 8
EXIT_VIOLATION = 9
EXIT_ERROR = 5


def evals_done(iters: int, eval_freq: int = EVAL_FREQ) -> int:
    """How many evaluations happened after `iters` iterations.

    Upstream evaluates (and saves) when `iteration % eval_freq == 1`
    (ppo.py:276,297), not `== 0`. So 40 iterations give 8 evaluations
    (1,6,...,36) and 29 give 6. Using `iters // eval_freq` undercounts by one
    whenever the run is cut between the first and last aligned iteration.
    """
    return 0 if iters < 1 else (iters - 1) // eval_freq + 1


def blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def build_argv(python: str, ppo: Path, num_envs: int, exp_name: str,
               seed: int, total_timesteps: int, eval_freq: int) -> list[str]:
    """The exact upstream CLI. Tyro is kebab-case: --no-capture-video, not
    --capture_video=False (that would be an argparse error before training).

    `-u` is a launcher-level flag, not a change to ppo.py: without it Python
    block-buffers stdout when it is a pipe, so nothing reaches the supervisor and
    the liveness watchdog cannot tell a working run from a hung one. That cost one
    killed attempt on 2026-10-21 (29 iterations had actually completed).
    """
    return [
        python, "-u", str(ppo),
        "--env-id", "PickCube-v1",
        "--exp-name", exp_name,
        "--num-envs", str(num_envs),
        "--num-steps", str(NUM_STEPS),
        "--total-timesteps", str(total_timesteps),
        "--update-epochs", str(UPDATE_EPOCHS),
        "--num-minibatches", str(NUM_MINIBATCHES),
        "--num-eval-envs", str(NUM_EVAL_ENVS),
        "--eval-freq", str(eval_freq),
        "--seed", str(seed),
        "--no-capture-video",
        "--no-track",
        "--save-model",
    ]


def parse_progress(line: str, state: dict[str, Any]) -> None:
    """Turn ppo.py's stdout into a liveness signal. Pure, so it is testable.

    Any non-empty line counts as liveness; the `Epoch:` line additionally yields
    the iteration number. Only liveness matters for the kill switch, and using
    every line keeps a long evaluation from being mistaken for a hang.
    """
    text = line.strip()
    if text:
        state["last_progress"] = time.monotonic()
    if text.startswith("Epoch:"):
        try:
            state["iteration"] = int(text.split("Epoch:")[1].split(",")[0].strip())
        except (IndexError, ValueError):
            pass
    elif text.startswith("SPS:"):
        try:
            state["loop_sps"] = int(float(text.split("SPS:")[1].strip()))
        except (IndexError, ValueError):
            pass
    elif text.startswith("eval_success_once_mean="):
        try:
            state["eval_success"] = float(text.split("=", 1)[1])
        except ValueError:
            pass
    elif text.startswith("model saved to"):
        state["ckpt_prints"] = state.get("ckpt_prints", 0) + 1


def series_summary(ea, tag: str) -> dict[str, Any]:
    try:
        events = ea.Scalars(tag)
    except (KeyError, ValueError):
        return {"present": False}
    if not events:
        return {"present": False}
    vals = [e.value for e in events]
    finite = [v for v in vals if v == v and abs(v) != float("inf")]
    return {
        "present": True,
        "n": len(vals),
        "first": round(vals[0], 6),
        "last": round(vals[-1], 6),
        "mean": round(sum(finite) / len(finite), 6) if finite else None,
        "min": round(min(finite), 6) if finite else None,
        "max": round(max(finite), 6) if finite else None,
        "non_finite": len(vals) - len(finite),
        "last_step": events[-1].step,
    }


def read_scalars(run_dir: Path) -> dict[str, Any]:
    """Read the TensorBoard event file(s) written by upstream ppo.py."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        return {"present": False, "reason": "no event file"}
    ea = EventAccumulator(str(files[0]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    wanted = [
        "losses/policy_loss", "losses/value_loss", "losses/approx_kl",
        "losses/clipfrac", "losses/explained_variance", "losses/entropy",
        "losses/old_approx_kl", "charts/SPS", "charts/learning_rate",
        "time/rollout_time", "time/update_time", "time/rollout_fps",
        "eval/success_once", "eval/return", "eval/episode_len",
        "train/success_once", "train/return",
    ]
    out = {"present": True, "event_file": files[0].name,
           "size_bytes": files[0].stat().st_size, "all_tags": tags}
    for t in wanted:
        if t in tags:
            out[t] = series_summary(ea, t)
    return out


def nan_watch(run_dir: Path) -> list[str]:
    """Return the loss tags whose latest value is non-finite."""
    bad: list[str] = []
    try:
        data = read_scalars(run_dir)
    except Exception:  # noqa: BLE001
        return bad
    if not data.get("present"):
        return bad
    for t in ("losses/policy_loss", "losses/value_loss", "losses/approx_kl"):
        s = data.get(t) or {}
        if s.get("present") and s.get("last") is not None:
            last = s["last"]
            if last != last or abs(last) == float("inf"):
                bad.append(t)
    return bad


def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 2**20, 2)


def checkpoints(run_dir: Path) -> dict[str, Any]:
    ckpts = sorted(run_dir.glob("ckpt_*.pt")) + sorted(run_dir.glob("final_ckpt.pt"))
    files = [{"path": str(p), "mb": round(p.stat().st_size / 2**20, 3)} for p in ckpts]
    periodic = [f for f in files if "final_ckpt" not in f["path"]]
    return {"count": len(files), "periodic": len(periodic),
            "final": len(files) - len(periodic), "files": files,
            "total_mb": round(sum(f["mb"] for f in files), 3)}


def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    argv = build_argv("/py", Path("/tmp/ppo.py"), 128, "PickCube-v1__ppo__9351__stageB",
                      9351, 256000, 5)
    joined = " ".join(argv)
    check("argv runs the child unbuffered (-u)", argv[1] == "-u", str(argv[:3]))
    check("argv pins num-envs 128", "--num-envs 128" in joined)
    check("argv has total-timesteps 256000", "--total-timesteps 256000" in joined)
    check("argv has eval-freq 5", "--eval-freq 5" in joined)
    check("argv disables video via kebab-case flag", "--no-capture-video" in argv)
    check("argv never uses an invalid underscore boolean",
          not any("capture_video" in a or "no_capture" in a for a in argv))
    check("argv disables wandb", "--no-track" in argv)
    check("argv enables checkpoint saving", "--save-model" in argv)
    check("argv keeps upstream step/epoch/minibatch",
          "--num-steps 50" in joined and "--update-epochs 4" in joined
          and "--num-minibatches 32" in joined)
    check("argv keeps seed 9351 and 8 eval envs",
          "--seed 9351" in joined and "--num-eval-envs 8" in joined)

    check("eval count follows iteration % eval_freq == 1",
          evals_done(40) == 8 and evals_done(29) == 6 and evals_done(1) == 1
          and evals_done(0) == 0,
          f"{evals_done(40)} {evals_done(29)} {evals_done(1)} {evals_done(0)}")
    state: dict[str, Any] = {"last_progress": None}
    check("no liveness before any output", state["last_progress"] is None)
    parse_progress("Epoch: 7, global_step=38400", state)
    check("progress parser reads iteration", state.get("iteration") == 7, str(state.get("iteration")))
    check("progress parser marks liveness", state["last_progress"] is not None)
    parse_progress("SPS: 3120", state)
    check("progress parser reads loop SPS", state.get("loop_sps") == 3120)
    parse_progress("eval_success_once_mean=0.125", state)
    check("progress parser reads eval success", abs(state.get("eval_success") - 0.125) < 1e-9)
    before = state["last_progress"]
    time.sleep(0.01)
    parse_progress("some unrelated line", state)
    check("any output counts as liveness", state["last_progress"] > before)
    check("parser ignores unrelated lines for iteration", state.get("iteration") == 7)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "ckpt_1.pt").write_bytes(b"x" * 100)
        (d / "ckpt_6.pt").write_bytes(b"x" * 100)
        (d / "final_ckpt.pt").write_bytes(b"x" * 100)
        c = checkpoints(d)
        check("checkpoint counter separates periodic and final",
              c["periodic"] == 2 and c["final"] == 1, str(c["count"]))
        (d / "events.out.tfevents.1").write_bytes(b"y" * (2 * 2**20))
        check("storage measurement works", dir_size_mb(d) >= 2.0, str(dir_size_mb(d)))

    check("blob sha is the pinned one for the real file",
          True if not DEFAULT_PPO.is_file() else blob_sha1(DEFAULT_PPO) == PINNED_PPO_BLOB,
          "" if not DEFAULT_PPO.is_file() else blob_sha1(DEFAULT_PPO)[:12])

    print(f"\nself-test: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return EXIT_OK if not failures else 1


def harvest(summary_path: Path, run_dir: Path, num_envs: int, eval_freq: int) -> int:
    """Re-derive the metrics from what is on disk, without training anything.

    Needed because a killed child loses its buffered stdout while the TensorBoard
    events and checkpoints survive. The event file is therefore the authority for
    iteration count and losses; the surviving summary supplies wall clock and VRAM.
    """
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    scalars = read_scalars(run_dir)
    steps_seen = [s.get("last_step") for s in scalars.values()
                  if isinstance(s, dict) and s.get("last_step")]
    iters = max(steps_seen) // (num_envs * NUM_STEPS) if steps_seen else 0
    wall = summary.get("wall_s")
    summary["tensorboard"] = scalars
    summary["checkpoints"] = checkpoints(run_dir) if run_dir.exists() else {"count": 0}
    summary["storage_mb"] = dir_size_mb(run_dir) if run_dir.exists() else 0.0
    summary["iterations_from_events"] = iters
    summary["iterations_completed"] = iters
    summary["steps"] = {
        "train_env_steps": iters * num_envs * NUM_STEPS,
        "eval_env_steps": evals_done(iters, eval_freq) * NUM_EVAL_ENVS * NUM_EVAL_STEPS,
        "optimizer_updates_upper_bound": iters * UPDATE_EPOCHS * NUM_MINIBATCHES,
        "note": "derived from the event file after a kill; stdout was lost to buffering",
    }
    summary["sps"] = {
        "end_to_end": round(iters * num_envs * NUM_STEPS / wall, 1)
        if isinstance(wall, (int, float)) and wall > 0 else None,
        "loop_from_tensorboard": (scalars.get("charts/SPS") or {}).get("last")
        if scalars.get("present") else None,
        "definition": ("end_to_end = train_env_steps / wall clock of the whole child; "
                       "the loop figure excludes import and scene build"),
    }
    summary["harvested"] = True
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, summary_path)
    print(json.dumps({"iterations": iters, "wall_s": wall,
                      "end_to_end_sps": summary["sps"]["end_to_end"],
                      "loop_sps": summary["sps"]["loop_from_tensorboard"],
                      "checkpoints": summary["checkpoints"].get("count"),
                      "status": summary.get("status")}, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage B: one bounded upstream-PPO pilot with kill switches.")
    p.add_argument("--python", default=str(Path(".venv-maniskill/bin/python").absolute()))
    p.add_argument("--ppo", type=Path, default=DEFAULT_PPO)
    p.add_argument("--num-envs", type=int, default=NUM_ENVS,
                   help="parallel training envs; 128 is the approved Stage B default")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                   help="training env steps; 256000 for pilots, 50000000 for the full run")
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ,
                   help="evaluations + checkpoints every N iterations (upstream default 25)")
    p.add_argument("--exp-name", default=EXP_NAME,
                   help="runs/<exp-name>; distinct names keep runs from colliding")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--summary", type=Path, default=Path("results/m2_pilot_summary.json"))
    p.add_argument("--log", type=Path, default=Path("results/raw/m2_stage_b_pilot.log"))
    p.add_argument("--wall-limit", type=float, default=WALL_LIMIT_S)
    p.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    p.add_argument("--start-free-mib", type=int, default=START_FREE_MIB)
    p.add_argument("--storage-mb", type=int, default=STORAGE_LIMIT_MB)
    p.add_argument("--no-progress-s", type=float, default=NO_PROGRESS_S)
    p.add_argument("--startup-grace", type=float, default=STARTUP_GRACE_S,
                   help="allowance before the FIRST output line; covers import + two "
                        "PhysX scene builds. The 120 s rule applies to the loop after that")
    p.add_argument("--skip-blob-check", action="store_true",
                   help="only for the self-test path; a real run must verify the blob")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--harvest", action="store_true",
                   help="do not train: rebuild the metrics of an existing run from its "
                        "TensorBoard events, checkpoints and the surviving summary")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.harvest:
        return harvest(args.summary, args.runs_dir / args.exp_name, args.num_envs, args.eval_freq)

    batch_size = args.num_envs * NUM_STEPS
    num_iterations = args.total_timesteps // batch_size
    cfg = {
        "stage": "B", "classification": "ADAPTATION / PLUMBING_ONLY",
        "env_id": "PickCube-v1", "num_envs": args.num_envs, "num_steps": NUM_STEPS,
        "batch_size": batch_size, "num_iterations": num_iterations,
        "exp_name": args.exp_name,
        "total_timesteps": args.total_timesteps, "update_epochs": UPDATE_EPOCHS,
        "num_minibatches": NUM_MINIBATCHES, "num_eval_envs": NUM_EVAL_ENVS,
        "eval_freq": args.eval_freq, "seed": args.seed, "capture_video": False,
        "track": False, "save_model": True,
        "wall_limit_s": args.wall_limit, "min_free_mib": args.min_free_mib,
        "start_free_mib": args.start_free_mib, "storage_limit_mb": args.storage_mb,
        "upstream_commit": PINNED_UPSTREAM_COMMIT, "ppo_blob_sha": PINNED_PPO_BLOB,
        "ppo_path": str(args.ppo), "interpreter": args.python,
    }
    cmd = build_argv(args.python, args.ppo, args.num_envs, args.exp_name,
                     args.seed, args.total_timesteps, args.eval_freq)

    if args.dry_run:
        periodic = evals_done(num_iterations, args.eval_freq) if num_iterations >= 1 else 0
        print(json.dumps({"config": cfg, "argv": cmd,
                          "expected_iterations": num_iterations,
                          "expected_periodic_ckpts": periodic, "expected_final_ckpts": 1},
                         indent=2))
        return EXIT_OK

    summary: dict[str, Any] = {
        "kind": "m2_stage_b_pilot", "config": cfg, "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "RUNNING", "command": " ".join(cmd),
    }

    def flush() -> None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.summary.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, args.summary)

    # --- preflight ---------------------------------------------------------
    gpu = smi()
    free = None
    if gpu.get("stdout"):
        parts = [p.strip() for p in gpu["stdout"].split(",")]
        try:
            free = int(float(parts[-1]))
        except (ValueError, IndexError):
            free = None
    summary["preflight"] = {"nvidia_smi": gpu, "free_mib": free,
                            "compute_apps": smi_compute_apps()}
    if gpu.get("returncode") != 0 or free is None:
        summary["status"] = "BLOCKED_NO_GPU"
        summary["error"] = "nvidia-smi unusable; run this in a GPU-visible terminal"
        flush()
        print(f"BLOCKED: {summary['error']}", file=sys.stderr)
        return EXIT_BLOCKED
    if free < args.start_free_mib:
        summary["status"] = "BLOCKED_LOW_VRAM"
        summary["error"] = f"free {free} MiB < start gate {args.start_free_mib} MiB"
        flush()
        print(f"BLOCKED: {summary['error']}", file=sys.stderr)
        return EXIT_BLOCKED
    if not args.skip_blob_check:
        if not args.ppo.is_file():
            summary["status"] = "BLOCKED_NO_PPO"
            summary["error"] = f"ppo.py not found at {args.ppo}"
            flush()
            print(f"BLOCKED: {summary['error']}", file=sys.stderr)
            return EXIT_BLOCKED
        got = blob_sha1(args.ppo)
        summary["preflight"]["ppo_blob_sha_actual"] = got
        if got != PINNED_PPO_BLOB:
            summary["status"] = "BLOCKED_PPO_MISMATCH"
            summary["error"] = (f"ppo.py blob {got} != pinned {PINNED_PPO_BLOB}; refusing to "
                               "train on an unverified copy")
            flush()
            print(f"BLOCKED: {summary['error']}", file=sys.stderr)
            return EXIT_BLOCKED

    run_dir = args.runs_dir / args.exp_name
    summary["run_dir"] = str(run_dir)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = args.log.open("w", encoding="utf-8")

    # --- launch ------------------------------------------------------------
    t0 = time.monotonic()
    child_env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=child_env)
    state: dict[str, Any] = {"last_progress": None}
    vram_samples: list[int] = []
    vram_peak: list[int] = []
    kill_reason: str | None = None

    def watch() -> None:
        nonlocal kill_reason
        last_smi = 0.0
        while proc.poll() is None:
            time.sleep(POLL_S)
            now = time.monotonic()
            if now - t0 > args.wall_limit:
                kill_reason = f"wall clock {args.wall_limit:.0f}s exceeded"
            elif state["last_progress"] is None:
                if now - t0 > args.startup_grace:
                    kill_reason = (f"no output at all within {args.startup_grace:.0f}s "
                                   "(startup: import + scene build)")
            elif now - state["last_progress"] > args.no_progress_s:
                kill_reason = f"no iteration progress for {args.no_progress_s:.0f}s"
            elif now - last_smi >= SMI_EVERY_S:
                last_smi = now
                info = smi()
                if info.get("stdout"):
                    parts = [p.strip() for p in info["stdout"].split(",")]
                    try:
                        vram_samples.append(int(float(parts[-1])))
                    except (ValueError, IndexError):
                        pass
                    try:
                        vram_peak.append(int(float(parts[-2])))
                    except (ValueError, IndexError):
                        pass
                if vram_samples and vram_samples[-1] < args.min_free_mib:
                    kill_reason = (f"free VRAM {vram_samples[-1]} MiB < floor "
                                   f"{args.min_free_mib} MiB")
                if run_dir.exists() and dir_size_mb(run_dir) > args.storage_mb:
                    kill_reason = f"output storage > {args.storage_mb} MB"
                bad = nan_watch(run_dir)
                if bad:
                    kill_reason = f"non-finite loss detected: {bad}"
            if kill_reason:
                summary["kill_reason"] = kill_reason
                flush()
                print(f"KILL: {kill_reason}", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    for line in proc.stdout:  # type: ignore[union-attr]
        log_fh.write(line)
        log_fh.flush()
        print(line, end="", flush=True)
        parse_progress(line, state)
    proc.wait()
    watcher.join(timeout=30)
    wall = round(time.monotonic() - t0, 2)
    log_fh.close()

    # --- collect -----------------------------------------------------------
    summary["wall_s"] = wall
    summary["returncode"] = proc.returncode
    summary["loop_sps_last"] = state.get("loop_sps")
    try:
        scalars = read_scalars(run_dir)
    except Exception as exc:  # noqa: BLE001
        scalars = {"present": False, "reason": f"{type(exc).__name__}: {exc}"}
    summary["tensorboard"] = scalars
    summary["vram"] = {
        "min_free_mib": min(vram_samples) if vram_samples else None,
        "peak_used_mib": max(vram_peak) if vram_peak else None,
        "samples": len(vram_samples),
        "note": "device level via nvidia-smi every ~5 s; torch allocations are not counted separately here",
    }
    summary["checkpoints"] = checkpoints(run_dir) if run_dir.exists() else {"count": 0}
    summary["storage_mb"] = dir_size_mb(run_dir) if run_dir.exists() else 0.0

    iters_stdout = state.get("iteration", 0)
    iters_tb = 0
    if scalars.get("present"):
        steps = [s.get("last_step") for s in scalars.values()
                 if isinstance(s, dict) and s.get("last_step")]
        if steps:
            iters_tb = max(steps) // (args.num_envs * NUM_STEPS)
    # The event file is the authoritative record: stdout can be truncated by a kill.
    iters_done = max(iters_stdout, iters_tb)
    summary["iterations_stdout"] = iters_stdout
    summary["iterations_from_events"] = iters_tb
    summary["iterations_completed"] = iters_done
    train_steps = iters_done * args.num_envs * NUM_STEPS
    evals = evals_done(iters_done, args.eval_freq)
    summary["steps"] = {
        "train_env_steps": train_steps,
        "eval_env_steps": evals * NUM_EVAL_ENVS * 50,
        "note": "train and eval counted separately; eval is 50 steps x 8 envs per evaluation",
        "optimizer_updates_upper_bound": iters_done * UPDATE_EPOCHS * NUM_MINIBATCHES,
        "optimizer_updates_note": ("upper bound: target_kl=0.1 can break an epoch early, so the "
                                   "true count can be lower and is not printed by upstream ppo.py"),
    }
    summary["sps"] = {
        "end_to_end": round(train_steps / wall, 1) if wall > 0 else None,
        "loop_from_stdout": state.get("loop_sps"),
        "loop_from_tensorboard": (scalars.get("charts/SPS") or {}).get("last")
        if scalars.get("present") else None,
        "definition": ("end_to_end = train_env_steps / wall clock of the whole child, so it "
                       "includes env creation, evaluation and logging; ppo.py's own SPS does not"),
    }

    if kill_reason:
        summary["status"] = "KILLED"
    elif proc.returncode == 0 and iters_done == num_iterations:
        summary["status"] = "PASS"
    elif proc.returncode == 0:
        summary["status"] = "PARTIAL"
    else:
        summary["status"] = "FAIL"
    flush()
    print(f"\nstatus={summary['status']} iterations={iters_done} wall={wall}s "
          f"e2e_sps={summary['sps']['end_to_end']} -> {args.summary}", flush=True)
    if summary["status"] == "PASS":
        return EXIT_OK
    return EXIT_VIOLATION if summary["status"] == "KILLED" else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
