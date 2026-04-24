#!/usr/bin/env python3
"""
MiniGo AlphaZero — Continuous Training Supervisor

Launches the four continuous workers (train, selfplay, gatekeeper,
optional rate) as subprocesses, each with its own CUDA_VISIBLE_DEVICES.
Tees stdout/stderr into logs/<ts>/<proc>.log, restarts crashed workers
with exponential backoff, and archives old run artifacts on `init`.

Subcommands
  init     Archive previous run (if any), run bootstrap, seed
           models/accepted/v000000000.onnx + latest symlink.
  run      Launch the four workers and supervise until Ctrl-C.
  status   Show current run state (accepted models, logs path).
"""

import argparse
import datetime
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BUILD_DIR = PROJECT_ROOT / "build"

TRAINING_DIR = PROJECT_ROOT / "training"
MODELS_DIR = PROJECT_ROOT / "models"
RATINGS_DIR = PROJECT_ROOT / "ratings"
LOGS_ROOT = PROJECT_ROOT / "logs"


# ═══════════════════════════════════════════════════════════
#  Init: archive + bootstrap
# ═══════════════════════════════════════════════════════════

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def archive_previous():
    """Archive training/, models/, ratings/, logs/* from a prior run.
    Move (don't delete) into archive-<ts>/ sibling dirs under each parent
    so nothing is lost."""
    ts = timestamp()
    moved = []
    for src in [TRAINING_DIR, MODELS_DIR, RATINGS_DIR]:
        if src.is_dir() and any(src.iterdir()):
            dst = src.parent / f"{src.name}.archive-{ts}"
            shutil.move(str(src), str(dst))
            moved.append(f"{src.name} -> {dst.name}")
    # Archive logs/ contents (not logs itself)
    if LOGS_ROOT.is_dir() and any(LOGS_ROOT.iterdir()):
        dst = LOGS_ROOT.parent / f"logs.archive-{ts}"
        shutil.move(str(LOGS_ROOT), str(dst))
        moved.append(f"logs -> {dst.name}")
    return moved


def build_if_needed():
    selfplay = BUILD_DIR / "selfplay"
    evaluate = BUILD_DIR / "evaluate"
    if selfplay.is_file() and evaluate.is_file():
        return
    print(f"[build] building C++ binaries into {BUILD_DIR}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                   cwd=str(BUILD_DIR), check=True)
    subprocess.run(["make", f"-j{os.cpu_count() or 4}"],
                   cwd=str(BUILD_DIR), check=True)


def bootstrap_model(args):
    """Run scripts/export_onnx.py --init to produce the seed model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    accepted = MODELS_DIR / "accepted"
    candidates = MODELS_DIR / "candidates"
    rejected = MODELS_DIR / "rejected"
    for d in [accepted, candidates, rejected]:
        d.mkdir(parents=True, exist_ok=True)

    seed_path = accepted / "v000000000.onnx"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "export_onnx.py"),
        "--init", "--output", str(seed_path),
        "--board", str(args.board), "--arch", args.arch,
        "--filters", str(args.filters), "--blocks", str(args.blocks),
    ]
    if args.arch == "vit":
        cmd += ["--d-model", str(args.d_model), "--depth", str(args.depth),
                "--heads", str(args.heads),
                "--kv-groups", str(args.kv_groups),
                "--mlp-ratio", str(args.mlp_ratio)]
    subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=True)

    # Atomic symlink swap for latest
    latest = accepted / "latest"
    tmp = accepted / ".latest.tmp"
    if tmp.exists():
        tmp.unlink()
    tmp.symlink_to("v000000000.onnx")
    os.replace(str(tmp), str(latest))
    print(f"[init] seeded {seed_path} (accepted/latest -> v000000000.onnx)")


# ═══════════════════════════════════════════════════════════
#  Subprocess supervision
# ═══════════════════════════════════════════════════════════

class Worker:
    """Tracks one worker subprocess and restart backoff state."""

    MAX_BACKOFF_S = 300.0
    MAX_RESTARTS_IN_WINDOW = 3
    WINDOW_S = 600.0

    def __init__(self, name, build_cmd_fn, env_overrides, log_path):
        self.name = name
        self.build_cmd_fn = build_cmd_fn   # () -> list[str]
        self.env_overrides = env_overrides
        self.log_path = log_path
        self.proc = None
        self.log_fh = None
        self.start_times = []
        self.backoff = 1.0
        self.disabled = False

    def spawn(self):
        env = os.environ.copy()
        env.update(self.env_overrides)
        cmd = self.build_cmd_fn()
        self.log_fh = open(self.log_path, "a", buffering=1)
        self.log_fh.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] SPAWN "
            f"{' '.join(cmd)} | env: {self.env_overrides}\n")
        self.log_fh.flush()
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=self.log_fh, stderr=subprocess.STDOUT)
        self.start_times.append(time.time())
        # Prune window
        cutoff = time.time() - self.WINDOW_S
        self.start_times = [t for t in self.start_times if t >= cutoff]

    def poll(self):
        if self.proc is None:
            return None
        return self.proc.poll()

    def terminate(self, grace_s=10.0):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except OSError:
                    pass
        if self.log_fh:
            try:
                self.log_fh.close()
            except OSError:
                pass

    def should_disable(self):
        return len(self.start_times) >= self.MAX_RESTARTS_IN_WINDOW

    def handle_exit(self, rc, sup_log):
        sup_log("CRASH", proc=self.name, rc=rc)
        if self.log_fh:
            try:
                self.log_fh.close()
            except OSError:
                pass
        self.log_fh = None
        if self.should_disable():
            self.disabled = True
            sup_log("DISABLED", proc=self.name,
                    reason=f">{self.MAX_RESTARTS_IN_WINDOW} restarts in {self.WINDOW_S}s")
            return
        backoff = min(self.backoff, self.MAX_BACKOFF_S)
        self.backoff = min(self.backoff * 2.0, self.MAX_BACKOFF_S)
        sup_log("RESTART", proc=self.name, after_s=f"{backoff:.1f}")
        time.sleep(backoff)
        self.spawn()


# ═══════════════════════════════════════════════════════════
#  Command builders
# ═══════════════════════════════════════════════════════════

def train_cmd(args, log_dir):
    gpus = args.train_gpus.split(",")
    n = len(gpus)
    port = 29500 + (os.getpid() % 1000)
    cmd = ["torchrun", f"--nproc_per_node={n}",
           f"--master_port={port}",
           str(SCRIPTS_DIR / "train_continuous.py"),
           "--pool-dir", str(TRAINING_DIR / "selfplay"),
           "--candidates-dir", str(MODELS_DIR / "candidates"),
           "--checkpoint", str(TRAINING_DIR / "checkpoints" / "training.pt"),
           "--status-file", str(TRAINING_DIR / "status.json"),
           "--log-dir", log_dir,
           "--board", str(args.board),
           "--arch", args.arch,
           "--filters", str(args.filters),
           "--blocks", str(args.blocks),
           "--d-model", str(args.d_model),
           "--depth", str(args.depth),
           "--heads", str(args.heads),
           "--kv-groups", str(args.kv_groups),
           "--mlp-ratio", str(args.mlp_ratio),
           "--batch-size", str(args.batch_size),
           "--base-lr", str(args.base_lr),
           "--warmup-steps", str(args.warmup_steps),
           "--lr-milestones", args.lr_milestones,
           "--lr-gamma", str(args.lr_gamma),
           "--replay-target", str(args.replay_target),
           "--n-augmentations", str(args.n_augmentations),
           "--ring-games", str(args.ring_games),
           "--bucket-cap-mult", str(args.bucket_cap_mult),
           "--min-window-games", str(args.min_window_games),
           "--min-ring-rows", str(args.min_ring_rows),
           "--export-every", str(args.export_every),
           "--status-publish-every", str(args.status_publish_every),
           "--log-every", str(args.log_every),
           "--value-ramp-steps", str(args.value_ramp_steps),
           "--score-ramp-steps", str(args.score_ramp_steps),
           ]
    if args.fp8:
        cmd.append("--fp8")
    return cmd


def selfplay_cmd(args, log_dir):
    return [sys.executable, str(SCRIPTS_DIR / "selfplay_driver.py"),
            "--pool-dir", str(TRAINING_DIR / "selfplay"),
            "--accepted-dir", str(MODELS_DIR / "accepted"),
            "--status-file", str(TRAINING_DIR / "status.json"),
            "--log-dir", log_dir,
            "--build-dir", str(BUILD_DIR),
            "--games-per-batch", str(args.selfplay_batch_games),
            "--window-games", str(args.window_games),
            "--sims", str(args.selfplay_sims),
            "--nn-server-threads", str(len(args.nn_device_ids_selfplay.split(","))),
            "--nn-device-ids", args.nn_device_ids_selfplay,
            "--max-batch", str(args.max_batch),
            "--c-puct", str(args.c_puct),
            "--dirichlet-alpha", str(args.dirichlet_alpha),
            "--dirichlet-epsilon", str(args.dirichlet_epsilon),
            "--temp-threshold", str(args.temp_threshold),
            "--komi", str(args.komi),
            "--score-scale", str(args.score_scale),
            ]


def gate_cmd(args, log_dir):
    return [sys.executable, str(SCRIPTS_DIR / "gatekeeper.py"),
            "--candidates-dir", str(MODELS_DIR / "candidates"),
            "--accepted-dir", str(MODELS_DIR / "accepted"),
            "--rejected-dir", str(MODELS_DIR / "rejected"),
            "--log-dir", log_dir,
            "--build-dir", str(BUILD_DIR),
            "--games", str(args.gate_games),
            "--sims", str(args.gate_sims),
            "--threshold", str(args.gate_threshold),
            "--poll-interval", str(args.gate_poll_interval),
            "--nn-server-threads", str(len(args.nn_device_ids_gate.split(","))),
            "--nn-device-ids", args.nn_device_ids_gate,
            "--max-batch", str(args.max_batch),
            "--c-puct", str(args.c_puct),
            "--komi", str(args.komi),
            "--score-scale", str(args.score_scale),
            ]


def rate_cmd(args, log_dir):
    return [sys.executable, str(SCRIPTS_DIR / "rate.py"),
            "--accepted-dir", str(MODELS_DIR / "accepted"),
            "--ratings-dir", str(RATINGS_DIR),
            "--log-dir", log_dir,
            "--build-dir", str(BUILD_DIR),
            "--pool-size", str(args.rating_pool_size),
            "--games-per-pair", str(args.rating_games),
            "--sims", str(args.rating_sims),
            "--interval", str(args.rating_interval),
            "--nn-server-threads", str(len(args.nn_device_ids_rate.split(","))),
            "--nn-device-ids", args.nn_device_ids_rate,
            "--max-batch", str(args.max_batch),
            "--c-puct", str(args.c_puct),
            "--komi", str(args.komi),
            "--score-scale", str(args.score_scale),
            ]


# ═══════════════════════════════════════════════════════════
#  run
# ═══════════════════════════════════════════════════════════

def make_sup_log(log_dir):
    path = os.path.join(log_dir, "supervisor.log")
    fh = open(path, "a", buffering=1)
    def log(tag, **kv):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        line = f"[{ts}] {tag}" + (f" {kvs}" if kvs else "")
        fh.write(line + "\n")
        print(line, flush=True)
    return log


def cmd_run(args):
    # Sanity: accepted/latest must exist.
    if not (MODELS_DIR / "accepted" / "latest").exists() and \
       not (MODELS_DIR / "accepted" / "latest").is_symlink():
        print("ERROR: no accepted/latest.  Run 'python scripts/run_continuous.py init' first.")
        sys.exit(2)

    build_if_needed()

    # Timestamped logs dir.  LOGS_ROOT/current is a symlink to the active one
    # for convenience.
    ts = timestamp()
    log_dir = LOGS_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    current = LOGS_ROOT / "current"
    if current.is_symlink() or current.exists():
        try:
            current.unlink()
        except OSError:
            pass
    os.symlink(ts, current)
    log_dir_str = str(current)  # workers log via the symlink for stable path

    sup_log = make_sup_log(str(log_dir))
    sup_log("SUPERVISOR_START",
            train_gpus=args.train_gpus,
            selfplay_gpus=args.selfplay_gpus,
            gate_gpus=args.gate_gpus,
            rate_gpus=args.rate_gpus,
            rating=args.rating)

    workers = []
    workers.append(Worker(
        "train", lambda: train_cmd(args, log_dir_str),
        env_overrides={"CUDA_VISIBLE_DEVICES": args.train_gpus},
        log_path=str(log_dir / "train.stdio.log"),
    ))
    workers.append(Worker(
        "selfplay", lambda: selfplay_cmd(args, log_dir_str),
        env_overrides={"CUDA_VISIBLE_DEVICES": args.selfplay_gpus},
        log_path=str(log_dir / "selfplay.stdio.log"),
    ))
    workers.append(Worker(
        "gatekeeper", lambda: gate_cmd(args, log_dir_str),
        env_overrides={"CUDA_VISIBLE_DEVICES": args.gate_gpus},
        log_path=str(log_dir / "gatekeeper.stdio.log"),
    ))
    if args.rating:
        workers.append(Worker(
            "rate", lambda: rate_cmd(args, log_dir_str),
            env_overrides={"CUDA_VISIBLE_DEVICES": args.rate_gpus},
            log_path=str(log_dir / "rate.stdio.log"),
        ))

    for w in workers:
        w.spawn()
        sup_log("SPAWN", proc=w.name, pid=w.proc.pid,
                gpus=w.env_overrides.get("CUDA_VISIBLE_DEVICES", "-"))

    # Graceful shutdown
    shutting_down = {"flag": False}
    def _handle(sig, _f):
        if shutting_down["flag"]:
            # Second Ctrl-C — hard kill
            for w in workers:
                try:
                    if w.proc and w.proc.poll() is None:
                        w.proc.kill()
                except OSError:
                    pass
            sys.exit(130)
        shutting_down["flag"] = True
        sup_log("SIGNAL_FORWARD", sig=sig)
        for w in workers:
            try:
                if w.proc and w.proc.poll() is None:
                    w.proc.send_signal(sig)
            except OSError:
                pass
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # Supervise loop
    try:
        while not shutting_down["flag"]:
            any_alive = False
            for w in workers:
                if w.disabled:
                    continue
                rc = w.poll()
                if rc is None:
                    any_alive = True
                    continue
                w.handle_exit(rc, sup_log)
                if not w.disabled:
                    any_alive = True
            if not any_alive:
                sup_log("ALL_DEAD", workers=[w.name for w in workers])
                break
            time.sleep(2.0)
    finally:
        for w in workers:
            w.terminate()
        sup_log("SUPERVISOR_STOP")


# ═══════════════════════════════════════════════════════════
#  init / status
# ═══════════════════════════════════════════════════════════

def cmd_init(args):
    if not args.yes:
        print("About to archive previous run artifacts:")
        for d in [TRAINING_DIR, MODELS_DIR, RATINGS_DIR, LOGS_ROOT]:
            if d.exists() and any(d.iterdir()):
                print(f"  {d} -> archive-<ts>")
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return

    moved = archive_previous()
    for m in moved:
        print(f"[init] archived {m}")

    (TRAINING_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (TRAINING_DIR / "selfplay").mkdir(parents=True, exist_ok=True)
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    build_if_needed()
    bootstrap_model(args)
    print(f"[init] done.  Start training with:  "
          f"python {Path(__file__).relative_to(PROJECT_ROOT)} run ...")


def cmd_status(args):
    accepted = sorted(glob.glob(str(MODELS_DIR / "accepted" / "v*.onnx")))
    candidates = sorted(glob.glob(str(MODELS_DIR / "candidates" / "v*.onnx")))
    rejected = sorted(glob.glob(str(MODELS_DIR / "rejected" / "v*.onnx")))

    latest_link = MODELS_DIR / "accepted" / "latest"
    latest_target = None
    if latest_link.is_symlink():
        latest_target = os.readlink(str(latest_link))

    print("================ continuous run status ================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Accepted:     {len(accepted)} "
          f"(latest -> {latest_target or '-'})")
    print(f"Candidates:   {len(candidates)} queued")
    print(f"Rejected:     {len(rejected)}")
    print(f"Pool (games): {len(glob.glob(str(TRAINING_DIR / 'selfplay' / 'g_*.bin.zst')))}")

    status_path = TRAINING_DIR / "status.json"
    if status_path.is_file():
        try:
            import json
            with open(status_path) as f:
                s = json.load(f)
            print(f"Status:       step={s.get('step')} "
                  f"lr={s.get('lr')} score_ramp={s.get('score_ramp'):.3f} "
                  f"value_weight={s.get('value_weight')}")
        except (OSError, ValueError):
            pass

    logs_current = LOGS_ROOT / "current"
    if logs_current.is_symlink():
        print(f"Logs:         {logs_current} -> {os.readlink(str(logs_current))}")
    print("=======================================================")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def add_run_args(p):
    # GPU assignment — defaults single-GPU (everything on 0)
    p.add_argument("--selfplay-gpus", default="0")
    p.add_argument("--train-gpus", default="0")
    p.add_argument("--gate-gpus", default="0")
    p.add_argument("--rate-gpus", default="0")
    p.add_argument("--nn-device-ids-selfplay", default="0,0")
    p.add_argument("--nn-device-ids-gate", default="0")
    p.add_argument("--nn-device-ids-rate", default="0")

    # Model
    p.add_argument("--arch", default="resnet", choices=["resnet", "vit"])
    p.add_argument("--board", type=int, default=9)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-groups", type=int, default=2)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--fp8", action="store_true")

    # Training
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--base-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--lr-milestones", default="100000,400000,1500000")
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--replay-target", type=float, default=4.0)
    p.add_argument("--n-augmentations", type=int, default=8)
    p.add_argument("--ring-games", type=int, default=2000)
    p.add_argument("--bucket-cap-mult", type=int, default=64)
    p.add_argument("--min-window-games", type=int, default=2000)
    p.add_argument("--min-ring-rows", type=int, default=10240)
    p.add_argument("--export-every", type=int, default=5000)
    p.add_argument("--status-publish-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--value-ramp-steps", type=int, default=30000)
    p.add_argument("--score-ramp-steps", type=int, default=50000)

    # Selfplay
    p.add_argument("--selfplay-batch-games", type=int, default=300)
    p.add_argument("--selfplay-sims", type=int, default=500)
    p.add_argument("--window-games", type=int, default=80000)

    # Gatekeeper
    p.add_argument("--gate-games", type=int, default=200)
    p.add_argument("--gate-sims", type=int, default=150)
    p.add_argument("--gate-threshold", type=float, default=0.5)
    p.add_argument("--gate-poll-interval", type=float, default=30.0)

    # Rating
    p.add_argument("--rating", action="store_true",
                   help="Enable optional rating loop (costs extra GPU time)")
    p.add_argument("--rating-games", type=int, default=80)
    p.add_argument("--rating-sims", type=int, default=200)
    p.add_argument("--rating-pool-size", type=int, default=5)
    p.add_argument("--rating-interval", type=float, default=7200.0)

    # MCTS / game (shared across workers)
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.15)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.22)
    p.add_argument("--temp-threshold", type=int, default=12)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--score-scale", type=float, default=18.0)
    p.add_argument("--max-batch", type=int, default=256)


def main():
    parser = argparse.ArgumentParser(
        description="MiniGo continuous training supervisor")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init",
                            help="Archive prior run and bootstrap seed model")
    p_init.add_argument("--arch", default="resnet", choices=["resnet", "vit"])
    p_init.add_argument("--board", type=int, default=9)
    p_init.add_argument("--filters", type=int, default=64)
    p_init.add_argument("--blocks", type=int, default=5)
    p_init.add_argument("--d-model", type=int, default=192)
    p_init.add_argument("--depth", type=int, default=8)
    p_init.add_argument("--heads", type=int, default=6)
    p_init.add_argument("--kv-groups", type=int, default=2)
    p_init.add_argument("--mlp-ratio", type=int, default=4)
    p_init.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")

    p_run = sub.add_parser("run", help="Launch continuous workers")
    add_run_args(p_run)

    sub.add_parser("status", help="Show current run state")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
