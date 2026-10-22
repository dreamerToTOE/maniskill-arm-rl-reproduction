#!/usr/bin/env bash
# Restart WorkBuddy when the GPU is not visible.
#
# *** MUST BE RUN IN YOUR NORMAL UBUNTU TERMINAL, NOT INSIDE THE AGENT. ***
# The WorkBuddy agent sandbox has no DBus ("systemctl --user" -> "Failed to
# connect to bus") and runs in its own PID namespace, so it cannot restart its
# own host. This script needs the host terminal's DBus + process access.
#
# Honest caveat: restarting WorkBuddy does NOT reliably restore GPU visibility
# in the agent sandbox. It worked on 2026-10-22 11:22 but did NOT work on the
# 15:41 restart (still NO_NODES). This script automates the *attempt*; if the
# GPU stays absent after --max-restarts, the fix is not "restart more".
#
# Mechanism:
#   1. probe the GPU with nvidia-smi
#   2. if absent, stop the app (systemd --user scope, falling back to pkill),
#      then relaunch the desktop binary detached
#   3. wait --wait-after seconds and re-probe, up to --max-restarts times
set -u

APP_BIN="/opt/WorkBuddy/workbuddy"
SCOPE_GLOB="app-gnome-workbuddy-*.scope"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
WAIT_AFTER="${WAIT_AFTER:-20}"

gpu_ok() {
  nvidia-smi --query-gpu=name --format=csv,noheader,nounits >/dev/null 2>&1
}

stop_app() {
  # Prefer the systemd --user scope (GNOME-launched apps), else pkill the binary.
  if systemctl --user stop "$SCOPE_GLOB" 2>/dev/null; then
    echo "stopped app via systemd scope"
  else
    pkill -f "$APP_BIN" 2>/dev/null && echo "stopped app via pkill"
  fi
}

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  if gpu_ok; then
    echo "GPU OK (attempt $attempt)"
    exit 0
  fi
  echo "[$attempt/$MAX_RESTARTS] GPU not visible -> restarting WorkBuddy"
  stop_app
  sleep 3
  nohup "$APP_BIN" >/dev/null 2>&1 &
  disown
  echo "relaunched $APP_BIN; waiting ${WAIT_AFTER}s"
  sleep "$WAIT_AFTER"
done

if gpu_ok; then
  echo "GPU OK after $MAX_RESTARTS restart(s)"
  exit 0
fi
echo "STILL_NO_GPU after $MAX_RESTARTS restart(s): restarting WorkBuddy does not reliably restore GPU visibility"
exit 1
