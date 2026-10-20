#!/usr/bin/env bash
# Create (or repair) the isolated ManiSkill environment for this repo.
#
# Why this script exists beyond "just pip install": on some hosts /tmp is a tiny
# tmpfs (10 MB on the machine this was authored on). pip unpacks large wheels
# there and dies with `OSError: [Errno 28] No space left on device` after a few
# megabytes, even when the root filesystem has hundreds of GB free. This script
# detects that and points TMPDIR at a large filesystem before installing.
#
# Usage:
#   bash scripts/setup_maniskill_env.sh              # create/repair env
#   bash scripts/setup_maniskill_env.sh --check-only  # only print diagnostics
#
# Never touches the system interpreter, system pip, CUDA toolkit, driver,
# ROS2, Isaac Sim, or any other repository. No sudo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-maniskill"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
MIN_TMP_BYTES=$((3 * 1024 * 1024 * 1024))  # 3 GB headroom for big wheels

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

say() { printf '[setup] %s\n' "$*"; }

# ---------------------------------------------------------------- interpreter
# The default `python3` on PATH may be a newer interpreter than the SAPIEN
# cp310 wheel supports, so require an explicit 3.10.
PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
say "interpreter: $PYTHON_BIN ($PY_VER)"
case "$PY_VER" in
  3.10.*) ;;
  *) say "ERROR: need Python 3.10.x to match the published SAPIEN cp310 wheel."; exit 1 ;;
esac

# --------------------------------------------------------------------- TMPDIR
tmp_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '; }
CUR_TMP="${TMPDIR:-/tmp}"
AVAIL="$(tmp_avail "$CUR_TMP" || echo 0)"
say "temp dir $CUR_TMP has $((AVAIL / 1024 / 1024)) MB available"

if (( AVAIL < MIN_TMP_BYTES )); then
  BIG_TMP="${MANISKILL_TMPDIR:-$HOME/.cache/ms-m1-tmp}"
  mkdir -p "$BIG_TMP"
  BIG_AVAIL="$(tmp_avail "$BIG_TMP" || echo 0)"
  if (( BIG_AVAIL < MIN_TMP_BYTES )); then
    say "ERROR: $CUR_TMP is small and $BIG_TMP also lacks space ($((BIG_AVAIL / 1024 / 1024)) MB)."
    say "       Point MANISKILL_TMPDIR at a large filesystem and retry."
    exit 1
  fi
  export TMPDIR="$BIG_TMP"
  say "small /tmp detected -> TMPDIR=$TMPDIR ($((BIG_AVAIL / 1024 / 1024)) MB free)"
else
  say "temp dir is adequate, leaving TMPDIR=${TMPDIR:-<unset>}"
fi

if (( CHECK_ONLY )); then
  say "check-only: no changes made"
  exit 0
fi

# ----------------------------------------------------------------------- venv
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  say "creating $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  say "reusing existing $VENV_DIR"
fi
PIP=("$VENV_DIR/bin/python" -m pip)

"${PIP[@]}" install --upgrade pip >/dev/null
say "pip: $("$VENV_DIR/bin/python" -m pip --version)"

# Make the TMPDIR workaround stick for interactive use:
#   source .venv-maniskill/bin/activate
# re-running this block is idempotent (guarded by a marker comment).
if ! grep -q 'ms-m1-tmp' "$VENV_DIR/bin/activate" 2>/dev/null; then
  {
    printf '\n# ManiSkill M1: this host has a tiny /tmp tmpfs, pip needs room for large wheels.\n'
    printf 'if [ -d "%s" ]; then export TMPDIR="%s"; fi\n' "$TMPDIR" "$TMPDIR"
  } >> "$VENV_DIR/bin/activate"
  say "added TMPDIR export to $VENV_DIR/bin/activate"
fi

# ------------------------------------------------------------------ packages
# torch first (largest download, own index), then the simulation stack.
say "installing torch 2.5.1+cu121 ..."
"${PIP[@]}" install --no-cache-dir 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121

say "installing mani_skill / sapien / gymnasium / numpy ..."
"${PIP[@]}" install --no-cache-dir \
  'mani_skill==3.0.1' 'sapien==3.0.0' 'gymnasium==1.1.1' 'numpy==1.24.4'

say "pip check:"
"${PIP[@]}" check

# -------------------------------------------------------------------- manifest
"$VENV_DIR/bin/python" "$REPO_ROOT/scripts/m1_env_manifest.py" \
  --output "$REPO_ROOT/results/m1_environment_manifest.json" >/dev/null
"${PIP[@]}" freeze > "$REPO_ROOT/results/m1_requirements.lock.txt"
say "wrote results/m1_environment_manifest.json and results/m1_requirements.lock.txt"

cat <<EOF

Done. Next:
  source .venv-maniskill/bin/activate
  echo \$TMPDIR                       # should be a large filesystem
  python -c "import torch;print(torch.cuda.is_available())"

Then the approved smoke (1 env, zero actions, 3 steps, no training):
  MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 python scripts/m1_inspect_pickcube.py \\
    --obs-mode state_dict --action-mode zero --steps 3 --seed 0 \\
    --output results/m1_pickcube_state_dict_zero_seed0.json
EOF
