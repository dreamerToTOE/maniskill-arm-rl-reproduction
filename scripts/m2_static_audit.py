#!/usr/bin/env python3
"""Static audit for the M2 GPU probe scripts. No CUDA, no simulation, no network.

Checks that the two Stage-A scripts cannot silently do something they are not
allowed to do, and that the specific defects raised in
`docs/M2_INDEPENDENT_REVIEW.md` are actually fixed in the code:

  * compile cleanly under the pinned interpreter;
  * never import a training stack (torch.optim, tensorboard, wandb, tensordict,
    torchrl) and never call an optimiser step / backward;
  * never shell out to an installer or downloader;
  * pass `sim_backend="physx_cuda"` explicitly to every `gym.make` call
    (a single-env run with the default `auto` silently uses `physx_cpu`);
  * build actions through `batched_action_shape(...)` and never a bare
    `np.zeros(env.single_action_space.shape)`;
  * bound wall clock twice: monotonic deadline + watchdog thread;
  * read device-level VRAM from `nvidia-smi`;
  * keep A0 at exactly 3 steps and under a 120 s external ceiling.

Exit code 0 = all checks pass.
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import sys
import tempfile
from pathlib import Path

FORBIDDEN_MODULES = {
    "torch.optim",
    "torchrl",
    "tensordict",
    "wandb",
    "tensorboard",
    "torch.utils.tensorboard",
    "stable_baselines3",
}
FORBIDDEN_IDENTS = {
    "backward",
    "zero_grad",
    "Adam",
    "SGD",
    "AdamW",
    "optimizer",
    "optim",
    "SummaryWriter",
    "PPO",
    "SAC",
}
FORBIDDEN_STRINGS = (
    "pip install",
    "conda install",
    "uv pip",
    "wget ",
    "curl ",
    "torch.hub",
    "huggingface",
    "cudagraphs",
    "--track",
)
REQUIRED_IDENTS = {
    "batched_action_shape",
    "Deadline",
    "arm_watchdog",
    "smi_query",
}
PER_FILE_REQUIRED = {
    "m2_gpu_smoke_a0.py": {
        "vram_verdict",
        "episode_horizon",
        "telemetry_verdict",
        "acceptance_checks",
        "finalize_with_gate",
    },
    "m2_gpu_sim_probe.py": {
        "vram_gate",
        "validate_env_counts",
        "ManiSkillVectorEnv",
        "telemetry_verdict",
        "device_nodes",
    },
    "m2_a1_run_all.py": {
        "build_command",
        "extract_measurements",
        "compute_scaling",
        "summarize",
        "preflight_gpu",
    },
    "m2_stage_b_pilot.py": {
        "build_argv",
        "parse_progress",
        "read_scalars",
        "checkpoints",
        "dir_size_mb",
        "blob_sha1",
    },
}
# Scripts that never build an env: skip the gym.make / action / watchdog checks.
NO_ENV_SCRIPTS = {"m2_a1_run_all.py", "m2_stage_b_pilot.py"}
# Reading tensorboard *events* is not the same as importing the training stack:
# the Stage B supervisor parses what PPO already wrote.
ALLOWED_MODULES = {"m2_stage_b_pilot.py": {"tensorboard"}}
REQUIRED_STRINGS = ("nvidia-smi", "physx_cuda")


class Audit:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.src = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.src, filename=str(path))
        self.failures: list[str] = []
        self.passes = 0

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passes += 1
            print(f"[PASS] {self.path.name}: {name}")
        else:
            self.failures.append(f"{self.path.name}: {name}{(' :: ' + detail) if detail else ''}")
            print(f"[FAIL] {self.path.name}: {name}{(' :: ' + detail) if detail else ''}")

    def imported_modules(self) -> set[str]:
        mods: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mods.add(node.module)
        return mods

    def identifiers(self) -> set[str]:
        return {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)
        }

    def strings(self) -> list[str]:
        out = []
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
        return out

    def run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            try:
                py_compile.compile(str(self.path), cfile=str(Path(td) / "x.pyc"), doraise=True)
                self.check("compiles under pinned interpreter", True)
            except py_compile.PyCompileError as exc:
                self.check("compiles under pinned interpreter", False, str(exc)[:200])

        mods = self.imported_modules()
        allowed = ALLOWED_MODULES.get(self.path.name, set())
        bad_mods = sorted(
            m for m in mods
            if any(m == b or m.startswith(b + ".") for b in FORBIDDEN_MODULES)
            and not any(m == a or m.startswith(a + ".") for a in allowed)
        )
        self.check("no training/logging stack imported", not bad_mods, str(bad_mods))

        idents = self.identifiers()
        bad_idents = sorted(i for i in idents if i in FORBIDDEN_IDENTS)
        self.check("no optimiser/backward/algorithm symbols", not bad_idents, str(bad_idents))

        strs = self.strings()
        bad_strings = sorted({s for s in strs for f in FORBIDDEN_STRINGS if f in s})
        self.check("no installer/downloader strings", not bad_strings, str(bad_strings)[:200])

        missing = sorted(
            r for r in (REQUIRED_IDENTS | PER_FILE_REQUIRED.get(self.path.name, set()))
            if r not in idents
        )
        if self.path.name in NO_ENV_SCRIPTS:
            # the launcher exists to spawn isolated child processes
            required_idents = PER_FILE_REQUIRED.get(self.path.name, set())
            missing = sorted(r for r in required_idents if r not in idents)
        self.check("required guards present", not missing, str(missing))

        if self.path.name not in NO_ENV_SCRIPTS:
            missing_s = [r for r in REQUIRED_STRINGS if not any(r in s for s in strs)]
            self.check("required primitives present", not missing_s, str(missing_s))

        if self.path.name in NO_ENV_SCRIPTS:
            if self.path.name == "m2_a1_run_all.py":
                self.check("launcher pins physx_cuda in its config",
                           any("physx_cuda" in s for s in strs))
                self.check("launcher uses an external timeout",
                           any("timeout" in s for s in strs))
                self.check("launcher records not-run levels after a failure",
                           "NOT_RUN_AFTER_FAILURE" in self.src)
                self.check("launcher has no retry loop",
                           '"attempts": 1' in self.src and "while True" not in self.src)
                self.check("launcher stops after the first failure", "stopped = True" in self.src)
            else:
                self.check("stage B pins the upstream ppo.py blob",
                           "66802de0653061da550d6600d6346be374dbf974" in self.src)
                self.check("stage B disables video with a kebab-case flag",
                           "--no-capture-video" in self.src)
                self.check("stage B has a dedicated blocked exit code", "EXIT_BLOCKED" in self.src)
                self.check("stage B kills on a VRAM floor and a wall clock",
                           "min_free_mib" in self.src and "wall_limit" in self.src)
                self.check("stage B has no retry path",
                           "while True" not in self.src
                           and "for attempt" not in self.src
                           and "Popen(" in self.src and self.src.count("Popen(") == 1)
                self.check("stage B never imports torch (monitor only)", "torch" not in mods)
            return

        make_calls = [
            n for n in ast.walk(self.tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "make"
        ]
        self.check("gym.make call found", bool(make_calls), str(len(make_calls)))
        for call in make_calls:
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            node = kw.get("sim_backend")
            ok = isinstance(node, ast.Constant) and node.value == "physx_cuda"
            self.check("gym.make pins sim_backend=physx_cuda", ok,
                       "missing explicit kwarg" if node is None else ast.dump(node)[:60])

        bare_zeros = [
            n for n in ast.walk(self.tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "zeros"
            and any(
                isinstance(a, ast.Attribute) and a.attr == "single_action_space"
                for a in n.args
            )
        ]
        self.check("no bare single-env action tensor", not bare_zeros, str(len(bare_zeros)))

        uses_batched = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "batched_action_shape"
            for n in ast.walk(self.tree)
        )
        self.check("actions built via batched_action_shape", uses_batched)

        has_timer = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"Timer", "Thread"}
            for n in ast.walk(self.tree)
        )
        self.check("watchdog thread armed", has_timer)

        has_monotonic = "monotonic" in {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)}
        self.check("monotonic deadline used", has_monotonic)


def _arg_defaults(a: Audit) -> dict[str, object]:
    """argparse defaults, resolving module-level constants (`default=WALL_LIMIT_S`)."""
    consts: dict[str, object] = {}
    for node in a.tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant):
                consts[tgt.id] = node.value.value
    defaults: dict[str, object] = {}
    for node in ast.walk(a.tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        args = [x.value for x in node.args if isinstance(x, ast.Constant)]
        if not args:
            continue
        flag = str(args[0]).lstrip("-")
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            if isinstance(kw.value, ast.Constant):
                defaults[flag] = kw.value.value
            elif isinstance(kw.value, ast.Name) and kw.value.id in consts:
                defaults[flag] = consts[kw.value.id]
    return defaults


def audit_stage_b_specifics(a: Audit) -> None:
    """Stage B is 128 envs / 256k steps / 1200 s / 1536 MiB; no video, no wandb."""
    d = _arg_defaults(a)
    a.check("Stage B wall limit default is 1200 s",
            d.get("wall-limit") == 1200.0, str(d.get("wall-limit")))
    a.check("Stage B VRAM floor default is 1536 MiB",
            d.get("min-free-mib") == 1536, str(d.get("min-free-mib")))
    a.check("Stage B start gate default is 3072 MiB",
            d.get("start-free-mib") == 3072, str(d.get("start-free-mib")))
    a.check("Stage B storage cap default is 300 MB",
            d.get("storage-mb") == 300, str(d.get("storage-mb")))
    a.check("Stage B no-progress watchdog default is 120 s",
            d.get("no-progress-s") == 120.0, str(d.get("no-progress-s")))
    a.check("Stage B pins 128 envs / 50 steps / 256k timesteps",
            "NUM_ENVS = 128" in a.src and "NUM_STEPS = 50" in a.src
            and "TOTAL_TIMESTEPS = 256000" in a.src)
    a.check("Stage B pins eval_freq 5 and seed 9351",
            "EVAL_FREQ = 5" in a.src and "SEED = 9351" in a.src)
    a.check("Stage B ends with a final checkpoint expectation",
            "final_ckpt" in a.src and "periodic" in a.src)


def audit_a1_specifics(a: Audit) -> None:
    """Stage A1 is fixed at 10 warmup + 100 measured steps, 90 s/level, 8 GiB gates."""
    d = _arg_defaults(a)
    a.check("A1 warmup default is 10", d.get("warmup-steps") == 10, str(d.get("warmup-steps")))
    a.check("A1 measured steps default is 100", d.get("steps") == 100, str(d.get("steps")))
    a.check("A1 per-level budget is 90 s", d.get("level-seconds") == 90.0, str(d.get("level-seconds")))
    a.check("A1 process budget is 120 s", d.get("total-seconds") == 120.0, str(d.get("total-seconds")))
    a.check("A1 declares the external 150 s limit",
            d.get("external-limit-s") == 150.0, str(d.get("external-limit-s")))
    a.check("A1 start gate is 3072 MiB", d.get("min-free-mib") == 3072, str(d.get("min-free-mib")))
    a.check("A1 abort gate is 1536 MiB", d.get("abort-free-mib") == 1536, str(d.get("abort-free-mib")))
    a.check("A1 has a dedicated no-telemetry exit code", "EXIT_NO_VRAM_TELEMETRY" in a.src)
    a.check("A1 refuses a level without truncation evidence", "FAIL_NO_TRUNCATION" in a.src)
    a.check("A1 records auto-reset counters", "auto_reset_envs" in a.src)

    src = a.src
    gate_line = None
    import_line = None
    for node in ast.walk(a.tree):
        # the *call* to the gate, not its definition
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "telemetry_verdict" and gate_line is None):
            gate_line = node.lineno
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names]
            if "torch" in names and import_line is None:
                import_line = node.lineno
    a.check("A1 telemetry gate precedes `import torch`",
            gate_line is not None and import_line is not None and gate_line < import_line,
            f"gate_line={gate_line} import_line={import_line}")
    a.check("A1 refuses low VRAM with its own exit code", "EXIT_LOW_VRAM" in src)


def audit_a0_specifics(a: Audit) -> None:
    """Stage A0 is fixed at 1 env / 3 steps / <= 120 s."""
    defaults = {"steps": None, "budget-seconds": None}
    for node in ast.walk(a.tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        args = [x.value for x in node.args if isinstance(x, ast.Constant)]
        if not args:
            continue
        flag = str(args[0]).lstrip("-")
        if flag not in defaults:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                defaults[flag] = kw.value.value
    a.check("A0 steps default is 3", defaults["steps"] == 3, str(defaults["steps"]))
    budget = defaults["budget-seconds"]
    a.check("A0 internal budget < 120 s external", isinstance(budget, (int, float)) and budget < 120,
            str(budget))
    a.check("A0 pins num_envs=1", "num_envs=1" in a.src.replace(" ", ""))
    a.check("A0 has a dedicated no-telemetry exit code",
            "EXIT_NO_VRAM_TELEMETRY" in a.src and "EXIT_GATE_FAILED" in a.src)
    a.check("A0 telemetry gate is consulted", "telemetry_verdict(" in a.src)


def run_self_tests(paths: list[str]) -> list[dict]:
    """Execute each script's `--self-test` (pure logic, no CUDA) and count results."""
    import subprocess

    out = []
    for p in paths:
        proc = subprocess.run(
            [sys.executable, p, "--self-test"],
            capture_output=True, text=True, check=False,
        )
        passes = proc.stdout.count("[PASS]")
        fails = proc.stdout.count("[FAIL]")
        out.append({
            "script": p,
            "returncode": proc.returncode,
            "passed": passes,
            "failed": fails,
            "ok": proc.returncode == 0 and fails == 0 and passes > 0,
        })
        print(f"[{'PASS' if out[-1]['ok'] else 'FAIL'}] self-test {p}: "
              f"{passes} passed, {fails} failed (rc={proc.returncode})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=["scripts/m2_gpu_smoke_a0.py",
                                                 "scripts/m2_gpu_sim_probe.py",
                                                 "scripts/m2_a1_run_all.py",
                                                 "scripts/m2_stage_b_pilot.py"])
    ap.add_argument("--json", type=Path, default=None,
                    help="write a machine-readable summary here")
    ap.add_argument("--no-self-test", action="store_true",
                    help="audit only; skip the pure-logic self-tests")
    args = ap.parse_args(argv)

    failures = 0
    total_pass = 0
    per_file: list[dict] = []
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[FAIL] missing file {f}")
            failures += 1
            continue
        a = Audit(p)
        a.run()
        if p.name == "m2_gpu_smoke_a0.py":
            audit_a0_specifics(a)
        if p.name == "m2_gpu_sim_probe.py":
            audit_a1_specifics(a)
        if p.name == "m2_stage_b_pilot.py":
            audit_stage_b_specifics(a)
        total_pass += a.passes
        failures += len(a.failures)
        per_file.append({"script": f, "passed": a.passes,
                         "failed": len(a.failures), "failures": a.failures})

    self_tests = []
    if not args.no_self_test:
        print()
        self_tests = run_self_tests([f for f in args.files if Path(f).is_file()])
        failures += sum(0 if s["ok"] else 1 for s in self_tests)

    summary = {
        "kind": "static_checks",
        "generated_by": "scripts/m2_static_audit.py",
        "note": "No CUDA context, no environment instantiation, no installation.",
        "audit": {"total_passed": total_pass, "total_failed": failures, "files": per_file},
        "self_tests": self_tests,
        "ok": failures == 0,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(__import__("json").dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    print(f"\nstatic audit: {total_pass} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
