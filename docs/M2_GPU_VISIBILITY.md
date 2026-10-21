# GPU visibility in the agent session — investigation and remedies

> Date: 2026-10-21 · Host: RTX 3070 8 GiB, driver 580.178.04 · Repo commit: `c02208e`
> Question: the researcher's terminal has the GPU, the agent session does not.
> Can it be repaired **from inside** the session?
> **Answer: no.** Evidence below, plus the two remedies that do work.

## 1. Verdict in one table

| Step | Result | Meaning |
|---|---|---|
| `nvidia-smi --query-gpu=...` | rc=9, "couldn't communicate with the NVIDIA driver" | the driver interface is not reachable |
| `ls /dev/nvidia*` | only `/dev` = 7 nodes (`null`, `zero`, `full`, `random`, `urandom`, `tty`, `shm`); no NVIDIA nodes | `/dev` is a **per-invocation tmpfs**, not the host devtmpfs |
| `/proc/driver/nvidia/version` | `NVRM 580.178.04` | the module is loaded and visible |
| `/proc/driver/nvidia/gpus/0000:01:00.0/information` | `RTX 3070`, `Device Minor: 0`, `GPU Excluded: No` | the driver itself sees the card |
| `/sys/bus/pci/devices/0000:01:00.0/driver` | `-> …/drivers/nvidia` | the card is bound to the driver |
| `CapEff` | `0x1ffffffffff` — includes `MKNOD`, `SYS_ADMIN`, `SYS_PTRACE` | capabilities are **not** the limit |
| `mknod` (coreutils) | EPERM | the CLI is wrapped by the sandbox |
| `os.mknod()` (mknodat syscall) | **succeeds** | so the kernel path is open, and nodes can be created |
| `open("/dev/nvidiactl", O_RDWR)` | **OK** | the device node is accepted by the driver |
| `cuInit(0)` | **100** `CUDA_ERROR_NO_DEVICE` | the driver answers "no devices" |
| `ioctl(TCGETS)` / `TIOCGWINSZ` on a tty | **OK** | `ioctl` is not globally blocked |
| `ioctl(0x12345678)` / NVIDIA request on `/dev/nvidiactl` | **ENOTTY** | only a **whitelist** of ioctl requests is allowed |
| `Seccomp` | mode `2`, `Seccomp_filters: 1`, `NoNewPrivs: 1` | a seccomp filter is active in this session |

**Diagnosis:** the session runs a seccomp filter that (a) presents a minimal
per-invocation `/dev` tmpfs and (b) passes only an allow-list of `ioctl()`
requests, returning `ENOTTY` for everything else. The NVIDIA driver's entire
control surface is `ioctl`. So even with perfect device nodes, `cuInit` and
NVML cannot talk to the driver. This is **not** a driver fault, a permissions
problem, a missing card, or a broken torch install.

## 2. What was actually attempted (and is reproducible)

`scripts/gpu_visibility_probe.py --try-mknod` does the one experiment that
distinguishes "no nodes" from "nodes but blocked":

1. reads the real majors from `/proc/devices` (`195 nvidia/nvidiactl/nvidia-modeset`,
   `235 nvidia-uvm`, `238 nvidia-caps` — the uvm major is **dynamic** here, not the
   classic 511, which is why folklore majors must never be hardcoded);
2. creates `nvidiactl 195:255`, `nvidia0 195:0`, `nvidia-uvm 235:0`,
   `nvidia-uvm-tools 235:1`, `nvidia-modeset 195:254` via `os.mknod`;
3. opens each node and calls `cuInit`, then re-runs `nvidia-smi`.

Result from the live session (`results/raw/m2_gpu_visibility_probe.json`):

```
created : nvidiactl 195:255, nvidia0 195:0, nvidia-uvm 235:0,
          nvidia-uvm-tools 235:1, nvidia-modeset 195:254   (all OK)
opened  : all five OK
cuinit  : 100
smi     : rc=9
verdict : NODES_UNUSABLE  (exit 4)
```

So the node creation is not the blocker, and it is worth being precise about
what it does *not* imply: the nodes live in a 10 MB tmpfs that is recreated for
every command, so they vanish immediately — the host `/dev` is never touched,
and no state leaks between runs.

## 3. Why it cannot be fixed from inside

Any in-session workaround has to deliver NVIDIA ioctls to the driver, and the
filter sits between the process and the driver. Escape routes were checked and
are closed:

| Route | Result |
|---|---|
| Re-run with the tool's sandbox disabled | identical (`Seccomp: 2` survives) |
| `systemd-run --user …` | `Failed to connect to bus` — no user DBus in the session |
| Borrow the desktop's `/dev` via `/proc/<pid>/root` | `echo $$` = 1 (own PID namespace); desktop PIDs are not visible |
| `unshare -m` + remount `/dev` | `unshare` succeeds, but the seccomp filter is inherited by children, so `ioctl` stays neutered |
| `strace` the failing call | `PTRACE_SETOPTIONS: EPERM` — no syscall tracing available |

The one thing that *did* change the symptom was creating nodes: it moved the
verdict from `NO_NODES` to `NODES_UNUSABLE`, which is exactly the distinction
the reports needed. It does not restore GPU access.

## 4. Remedies that work

**R1 — run the GPU gate in the researcher's terminal (proven twice).**
A0 passed there on 2026-10-21 15:44, and A1 passed on 16:18 (`4/4`, exit 0,
27.7 s). This is the current operating mode and needs nothing installed.

**R2 — restart the WorkBuddy app to obtain a fresh session namespace.**
On 2026-10-21 at 09:57 the NVIDIA nodes appeared in the agent session right
after an app restart (device timestamps match), and the same session later lost
them at 12:58. So a restart *can* produce a GPU-visible session — but it is not
guaranteed and not durable. If you try it, check first, every time:

```bash
cd /home/ubuntu2004/lmy/maniskill-arm-rl-reproduction
./.venv-maniskill/bin/python scripts/gpu_visibility_probe.py --try-mknod
```

`verdict: OK` (exit 0) → the agent session can run GPU gates directly.
Anything else → use R1; do not start a GPU gate that the launcher will refuse.

**Do not** pursue, for this project: driver reinstall, `chmod`/`chown` on device
nodes, privileged containers, or editing the app's sandbox policy mid-experiment.
None of those change a seccomp allow-list, and each risks the host that currently
works.

## 5. Consequence for the experiment pipeline

Every GPU script in this repo already fails closed, which is why the split-brain
never produced a false number:

| Situation | Behaviour |
|---|---|
| Agent session, no nodes | A0 / A1 exit **8** before `import torch`; nothing is built |
| Agent session, nodes but blocked ioctl | identical: the probe reports `NODES_UNUSABLE` (exit 4), the gates still exit 8 |
| Researcher terminal | A0 12/12, A1 4/4 PASS |

Recommended habit before any GPU gate: run the probe and paste its verdict into
the report. It converts "the agent couldn't see the GPU" into a checkable claim.
