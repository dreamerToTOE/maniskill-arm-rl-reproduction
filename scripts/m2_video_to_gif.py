#!/usr/bin/env python3
"""Convert ManiSkill eval mp4s to GIF + keyframe PNGs.

Why: the eval videos are H.264 mp4s, and a stock Linux player (Totem) usually
lacks the H.264 decoder, so they won't play. GIF and PNG need no codecs and can
be viewed anywhere - including inline in the agent chat.

Usage:
  ./.venv-maniskill/bin/python scripts/m2_video_to_gif.py runs/*/test_videos/0.mp4
  ./.venv-maniskill/bin/python scripts/m2_video_to_gif.py --width 512 --stride 2 <mp4...>

Outputs next to each mp4 (or in --out-dir):
  <name>.gif        downscaled animation, viewable without codecs
  <name>_f{i}.png   keyframes (first / 25% / 50% / 75% / last)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def convert(mp4: Path, width: int, stride: int, out_dir: Path | None) -> list[Path]:
    reader = imageio.get_reader(str(mp4))
    meta = reader.get_meta_data()
    frames = [f for i, f in enumerate(reader) if i % stride == 0]
    reader.close()
    if not frames:
        print(f"{mp4}: no frames readable")
        return []

    h, w = frames[0].shape[:2]
    scale = min(1.0, width / w)
    if scale < 1.0:
        from PIL import Image
        new_size = (int(w * scale), int(h * scale))
        frames = [np.asarray(Image.fromarray(f).resize(new_size)) for f in frames]

    fps = float(meta.get("fps", 30) or 30) / max(1, stride)
    out_dir = out_dir or mp4.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mp4.stem

    gif_path = out_dir / f"{stem}.gif"
    imageio.mimsave(str(gif_path), frames, fps=round(fps), loop=0)

    png_paths: list[Path] = []
    n = len(frames)
    for tag, idx in [("first", 0), ("q25", n // 4), ("mid", n // 2),
                     ("q75", 3 * n // 4), ("last", n - 1)]:
        p = out_dir / f"{stem}_f{idx:03d}_{tag}.png"
        imageio.imwrite(str(p), frames[idx])
        png_paths.append(p)

    return [gif_path] + png_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mp4s", nargs="+", type=Path)
    ap.add_argument("--width", type=int, default=512, help="output GIF width in px")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    total = 0
    for mp4 in args.mp4s:
        if not mp4.is_file():
            print(f"skip (missing): {mp4}")
            continue
        outs = convert(mp4, args.width, args.stride, args.out_dir)
        for p in outs:
            print(f"{p}  ({p.stat().st_size / 1024:.0f} KB)")
        total += len(outs)
    print(f"\n{total} files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
