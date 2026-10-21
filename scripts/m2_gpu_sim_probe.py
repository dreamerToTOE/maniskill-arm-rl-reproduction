#!/usr/bin/env python3
"""M2 bounded GPU simulation probe. NO policy, NO optimizer, NO training.

Measures, for a range of parallel env counts on `physx_cuda`:
  - setup wall time, warmup time
  - steady-state simulator steps/second (pure sim, zero actions)
  - peak PyTorch VRAM and nvidia-smi free VRAM after the run
and stops early on OOM or a VRAM floor breach.

This is NOT a training-throughput measurement. Pure simulator SPS excludes
gradient updates, evaluation and logging; see docs/M2_GPU_PILOT_PROPOSAL.md.

Requires explicit researcher approval. Writes a JSON summary, nothing else.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def smi_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--reward-mode", default="normalized_dense")
    ap.add_argument("--env-counts", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="wall budget per env count")
    ap.add_argument("--min-free-mib", type=int, default=500, help="abort below this free VRAM")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import gymnasium as gym
    import mani_skill.envs

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available in this interpreter; aborting before any simulation.")

    counts = [int(x) for x in args.env_counts.split(",")]
    result = {
        "env_id": args.env_id, "robot": args.robot, "sim_backend": "physx_cuda",
        "obs_mode": "state", "control_mode": args.control_mode, "reward_mode": args.reward_mode,
        "render_mode": None, "warmup_steps": args.warmup_steps, "measured_steps": args.steps,
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        "free_mib_before": smi_free_mib(),
        "runs": [],
    }

    for n in counts:
        entry: dict = {"num_envs": n}
        env = None
        try:
            free_before = smi_free_mib()
            entry["free_mib_before"] = free_before
            if free_before is not None and free_before < args.min_free_mib:
                entry["status"] = "SKIPPED_LOW_VRAM"
                result["runs"].append(entry)
                break

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            env = gym.make(
                args.env_id, num_envs=n, robot_uids=args.robot,
                obs_mode="state", control_mode=args.control_mode,
                reward_mode=args.reward_mode, sim_backend="physx_cuda", render_mode=None,
            )
            env.reset(seed=0)
            torch.cuda.synchronize()
            entry["setup_s"] = round(time.perf_counter() - t0, 3)

            zero = np.zeros(env.single_action_space.shape, dtype=np.float32)
            t1 = time.perf_counter()
            for _ in range(args.warmup_steps):
                env.step(zero)
            torch.cuda.synchronize()
            entry["warmup_s"] = round(time.perf_counter() - t1, 3)

            done, sps_vals, t2 = 0, [], time.perf_counter()
            while done < args.steps and (time.perf_counter() - t2) < args.max_seconds:
                env.step(zero)
                done += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t2
            entry["measured_steps_done"] = done
            entry["measure_s"] = round(elapsed, 3)
            entry["sim_sps"] = round(done * n / elapsed, 1) if elapsed > 0 else None
            entry["env_sps"] = round(done / elapsed, 2) if elapsed > 0 else None
            entry["capped_by_walltime"] = bool(done < args.steps)
            entry["peak_torch_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            entry["peak_torch_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
            entry["free_mib_after"] = smi_free_mib()
            entry["status"] = "OK"
        except torch.cuda.OutOfMemoryError as exc:
            entry["status"] = "OOM"
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        except RuntimeError as exc:
            msg = str(exc)
            entry["status"] = "CUDA_OOM" if "out of memory" in msg.lower() else "RUNTIME_ERROR"
            entry["error"] = f"{type(exc).__name__}: {msg[:300]}"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            import torch as _t
            _t.cuda.empty_cache()
        result["runs"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if entry["status"] in {"OOM", "CUDA_OOM", "ERROR", "RUNTIME_ERROR"}:
            break

    result["free_mib_after_all"] = smi_free_mib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
