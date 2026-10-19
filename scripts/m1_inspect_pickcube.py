#!/usr/bin/env python3
"""M1 bounded PickCube introspection. Zero training, CPU simulation only.

Requires a separately approved and installed ManiSkill v3.0.1 environment.
Default: one Panda PickCube instance, three zero-actions, no rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def shape_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): shape_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [shape_tree(v) for v in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return type(value).__name__


def to_small_json(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.size <= 12:
        return arr.tolist()
    return {"shape": list(arr.shape), "min": float(arr.min()), "max": float(arr.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-mode", choices=["state", "state_dict"], default="state_dict")
    ap.add_argument("--action-mode", choices=["zero", "small-joint-test"], default="zero")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if not 1 <= args.steps <= 10:
        ap.error("--steps must be 1..10; training is not supported.")
    if args.action_mode == "small-joint-test" and args.steps > 5:
        ap.error("small-joint-test allows at most five steps.")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers the task environments

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        robot_uids="panda",
        obs_mode=args.obs_mode,
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        render_mode=None,
    )
    try:
        base = env.unwrapped
        action_space = base.single_action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise RuntimeError("Expected a continuous Box action space.")
        if len(action_space.shape) != 1:
            raise RuntimeError("Expected a flat single-agent action vector.")
        obs, reset_info = env.reset(seed=args.seed)
        result = {
            "task": "PickCube-v1",
            "robot": "panda",
            "obs_mode": args.obs_mode,
            "control_mode": base.control_mode,
            "reward_mode": base.reward_mode,
            "backend": "physx_cpu",
            "seed": args.seed,
            "action_mode": args.action_mode,
            "steps_budget": args.steps,
            "horizon": env.spec.max_episode_steps if env.spec else None,
            "sim_freq_hz": int(base.sim_config.sim_freq),
            "control_freq_hz": int(base.sim_config.control_freq),
            "action_space": {
                "shape": list(action_space.shape),
                "low": np.asarray(action_space.low).tolist(),
                "high": np.asarray(action_space.high).tolist(),
            },
            "initial_observation": shape_tree(obs),
            "reset_info_shape": shape_tree(reset_info),
            "steps": [],
        }
        for t in range(args.steps):
            action = np.zeros(action_space.shape, dtype=np.float32)
            if args.action_mode == "small-joint-test":
                action[0] = np.float32(0.1)
            obs, reward, terminated, truncated, info = env.step(action)
            fields = {
                k: to_small_json(info[k])
                for k in ("success", "is_grasped", "is_obj_placed", "is_robot_static", "elapsed_steps")
                if k in info
            }
            result["steps"].append({
                "t": t,
                "action": action.tolist(),
                "reward": to_small_json(reward),
                "terminated": to_small_json(terminated),
                "truncated": to_small_json(truncated),
                "info": fields,
                "obs_shape": shape_tree(obs),
            })
            if bool(np.asarray(to_small_json(terminated)).any()) or bool(np.asarray(to_small_json(truncated)).any()):
                break
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
