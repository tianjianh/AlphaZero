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
    """Tracks one worker subprocess and restart backoff state.

    Each worker is launched as a session leader (start_new_session=True)
    so the supervisor can reach the whole worker tree (Python wrapper +
    any C++ child it spawns) via os.killpg.  Shutdown is two-stage:

    1. Graceful: SIGTERM to the wrapper PID ONLY (self.proc.terminate()).
       The wrapper sets a stop flag and lets the in-flight C++ child
       finish the current batch/match.  Supervisor waits up to
       graceful_timeout seconds.
    2. Hard: SIGKILL to the whole process group (os.killpg).  Used on
       grace-timeout or on second Ctrl-C.  Guarantees no orphans.
    """

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
            stdout=self.log_fh, stderr=subprocess.STDOUT,
            start_new_session=True)   # own session/pgroup = reachable via killpg
        self.start_times.append(time.time())
        cutoff = time.time() - self.WINDOW_S
        self.start_times = [t for t in self.start_times if t >= cutoff]

    def poll(self):
        if self.proc is None:
            return None
        return self.proc.poll()

    def request_graceful_stop(self):
        """Send SIGTERM to the wrapper PID only.  The wrapper's handler
        sets a stop flag and lets its in-flight C++ child finish the
        current batch/match.  Does NOT signal the whole group — that
        would kill the C++ child immediately."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()   # SIGTERM to the wrapper PID
            except OSError:
                pass

    def kill_group(self):
        """SIGKILL the whole process group — wrapper + any C++ child.
        Use only after a grace timeout or on hard-shutdown escalation."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    def wait(self, timeout):
        """Block until the wrapper exits or timeout elapses.  Returns
        True if it exited, False on timeout."""
        if self.proc is None:
            return True
        try:
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def close_log(self):
        if self.log_fh:
            try:
                self.log_fh.close()
            except OSError:
                pass
            self.log_fh = None

    def should_disable(self):
        return len(self.start_times) >= self.MAX_RESTARTS_IN_WINDOW

    def handle_exit(self, rc, sup_log):
        sup_log("CRASH", proc=self.name, rc=rc)
        self.close_log()
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
           "--weight-decay", str(args.weight_decay),
           "--replay-target", str(args.replay_target),
           "--n-augmentations", str(args.n_augmentations),
           "--ring-games", str(args.ring_games),
           "--bucket-cap-mult", str(args.bucket_cap_mult),
           "--min-window-games", str(args.min_window_games),
           "--min-ring-rows", str(args.min_ring_rows),
           "--sample-batch-timeout-s", str(args.sample_batch_timeout_s),
           "--export-every", str(args.export_every),
           "--status-publish-every", str(args.status_publish_every),
           "--log-every", str(args.log_every),
           "--value-ramp-steps", str(args.value_ramp_steps),
           "--score-ramp-steps", str(args.score_ramp_steps),
           "--value-weight-start", str(args.value_weight_start),
           "--value-weight-end", str(args.value_weight_end),
           "--score-mean-weight-start", str(args.score_mean_weight_start),
           "--score-mean-weight-end", str(args.score_mean_weight_end),
           "--policy-weight", str(args.policy_weight),
           "--score-stdev-weight", str(args.score_stdev_weight),
           "--score-belief-weight", str(args.score_belief_weight),
           "--ownership-weight", str(args.ownership_weight),
           "--opp-policy-weight", str(args.opp_policy_weight),
           ]
    if args.fp8:
        cmd.append("--fp8")
    return cmd


def _resolve_nn_server_threads(proc_name, device_ids_str, explicit):
    """Derive nn-server-threads: if --<proc>-nn-server-threads is unset,
    default to len(--<proc>-nn-device-ids).  If set, validate it matches.
    The underlying C++ binary also enforces this; we fail early with a
    clearer message."""
    n_ids = len([x for x in device_ids_str.split(",") if x.strip()])
    if explicit is None:
        return n_ids
    if explicit != n_ids:
        raise SystemExit(
            f"ERROR: --{proc_name}-nn-server-threads={explicit} must equal "
            f"len(--{proc_name}-nn-device-ids)={n_ids} "
            f"(device_ids='{device_ids_str}')")
    return explicit


def selfplay_cmd(args, log_dir):
    nn_st = _resolve_nn_server_threads(
        "selfplay", args.selfplay_nn_device_ids, args.selfplay_nn_server_threads)
    return [sys.executable, str(SCRIPTS_DIR / "selfplay_driver.py"),
            "--pool-dir", str(TRAINING_DIR / "selfplay"),
            "--accepted-dir", str(MODELS_DIR / "accepted"),
            "--status-file", str(TRAINING_DIR / "status.json"),
            "--log-dir", log_dir,
            "--build-dir", str(BUILD_DIR),
            "--games-per-batch", str(args.selfplay_batch_games),
            "--window-games", str(args.window_games),
            "--sims", str(args.selfplay_sims),
            "--threads", str(args.selfplay_threads),
            "--search-threads", str(args.selfplay_search_threads),
            "--nn-server-threads", str(nn_st),
            "--nn-device-ids", args.selfplay_nn_device_ids,
            "--max-batch", str(args.max_batch),
            "--c-puct", str(args.c_puct),
            "--dirichlet-alpha", str(args.dirichlet_alpha),
            "--dirichlet-epsilon", str(args.dirichlet_epsilon),
            "--temp-threshold", str(args.temp_threshold),
            "--komi", str(args.komi),
            "--score-scale", str(args.score_scale),
            "--score-weight-max", str(args.score_weight_max),
            ]


def gate_cmd(args, log_dir):
    nn_st = _resolve_nn_server_threads(
        "gate", args.gate_nn_device_ids, args.gate_nn_server_threads)
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
            "--threads", str(args.gate_threads),
            "--search-threads", str(args.gate_search_threads),
            "--nn-server-threads", str(nn_st),
            "--nn-device-ids", args.gate_nn_device_ids,
            "--max-batch", str(args.max_batch),
            "--c-puct", str(args.c_puct),
            "--komi", str(args.komi),
            "--score-scale", str(args.score_scale),
            ]


def rate_cmd(args, log_dir):
    nn_st = _resolve_nn_server_threads(
        "rate", args.rate_nn_device_ids, args.rate_nn_server_threads)
    return [sys.executable, str(SCRIPTS_DIR / "rate.py"),
            "--accepted-dir", str(MODELS_DIR / "accepted"),
            "--ratings-dir", str(RATINGS_DIR),
            "--log-dir", log_dir,
            "--build-dir", str(BUILD_DIR),
            "--pool-size", str(args.rating_pool_size),
            "--games-per-pair", str(args.rating_games),
            "--sims", str(args.rating_sims),
            "--interval", str(args.rating_interval),
            "--threads", str(args.rate_threads),
            "--search-threads", str(args.rate_search_threads),
            "--nn-server-threads", str(nn_st),
            "--nn-device-ids", args.rate_nn_device_ids,
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


def _heartbeat_snapshot():
    """Read filesystem + status.json state for a single-line roll-up.
    Never throws — returns '-' placeholders on any failure so the
    heartbeat never interrupts the supervise loop."""
    def _count(patterns):
        import glob as _glob
        total = 0
        for p in patterns:
            total += len(_glob.glob(p))
        return total

    pool = _count([str(TRAINING_DIR / "selfplay" / "g_*.bin.zst")])
    candidates = _count([str(MODELS_DIR / "candidates" / "v*.onnx")])
    accepted = _count([str(MODELS_DIR / "accepted" / "v*.onnx")])
    rejected = _count([str(MODELS_DIR / "rejected" / "v*.onnx")])

    latest = "-"
    latest_link = MODELS_DIR / "accepted" / "latest"
    if latest_link.is_symlink():
        latest = os.readlink(str(latest_link))

    step = lr = score_ramp = "-"
    status_path = TRAINING_DIR / "status.json"
    if status_path.is_file():
        try:
            import json as _json
            with open(status_path) as f:
                s = _json.load(f)
            step = s.get("step", "-")
            lr = s.get("lr", "-")
            if isinstance(lr, float):
                lr = f"{lr:.2e}"
            sr = s.get("score_ramp", "-")
            score_ramp = f"{sr:.2f}" if isinstance(sr, (int, float)) else sr
        except (OSError, ValueError):
            pass

    return {
        "step": step,
        "lr": lr,
        "score_ramp": score_ramp,
        "pool": pool,
        "candidates": candidates,
        "accepted": accepted,
        "rejected": rejected,
        "latest": latest,
    }


def cmd_run(args):
    # Sanity: accepted/latest must exist.
    if not (MODELS_DIR / "accepted" / "latest").exists() and \
       not (MODELS_DIR / "accepted" / "latest").is_symlink():
        print("ERROR: no accepted/latest.  Run 'python scripts/run_continuous.py init' first.")
        sys.exit(2)

    build_if_needed()

    # Shared TRT engine cache — set once here and propagated to every
    # worker's env.  The C++ binaries read MINIGO_TRT_CACHE to locate
    # the cache dir; without this, cache keys depend on each worker's
    # cwd, defeating gatekeeper→selfplay plan reuse.
    trt_cache_dir = (MODELS_DIR / "trt_cache").resolve()
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

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

    base_env = {"MINIGO_TRT_CACHE": str(trt_cache_dir)}

    def _env(gpus):
        e = dict(base_env)
        e["CUDA_VISIBLE_DEVICES"] = gpus
        return e

    workers = []
    workers.append(Worker(
        "train", lambda: train_cmd(args, log_dir_str),
        env_overrides=_env(args.train_gpus),
        log_path=str(log_dir / "train.stdio.log"),
    ))
    workers.append(Worker(
        "selfplay", lambda: selfplay_cmd(args, log_dir_str),
        env_overrides=_env(args.selfplay_gpus),
        log_path=str(log_dir / "selfplay.stdio.log"),
    ))
    workers.append(Worker(
        "gatekeeper", lambda: gate_cmd(args, log_dir_str),
        env_overrides=_env(args.gate_gpus),
        log_path=str(log_dir / "gatekeeper.stdio.log"),
    ))
    if args.rating:
        workers.append(Worker(
            "rate", lambda: rate_cmd(args, log_dir_str),
            env_overrides=_env(args.rate_gpus),
            log_path=str(log_dir / "rate.stdio.log"),
        ))

    for w in workers:
        w.spawn()
        sup_log("SPAWN", proc=w.name, pid=w.proc.pid,
                gpus=w.env_overrides.get("CUDA_VISIBLE_DEVICES", "-"))

    # Two-stage shutdown:
    #   1st signal:  SIGTERM to each wrapper PID only (not the group).
    #                Wrapper's handler flips a stop flag and lets its
    #                in-flight C++ child (selfplay batch / gate match)
    #                run to completion.  Supervise loop falls out of
    #                its poll; finally-block waits up to
    #                --graceful-timeout seconds for each worker.
    #   Timeout or 2nd signal:
    #                SIGKILL to each worker's entire process group.
    #                Wrapper + C++ child die together; no orphans.
    shutting_down = {"flag": False, "escalated": False}
    def _handle(sig, _f):
        if shutting_down["flag"]:
            # Second Ctrl-C — escalate to group-kill for the supervise
            # loop's finally-block to see.
            shutting_down["escalated"] = True
            sup_log("SIGNAL_ESCALATE", sig=sig)
            for w in workers:
                w.kill_group()
            return
        shutting_down["flag"] = True
        sup_log("SIGNAL_GRACEFUL", sig=sig, grace_s=args.graceful_timeout)
        for w in workers:
            w.request_graceful_stop()
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # Heartbeat: roll up run state into one line per interval so the
    # operator can see forward progress without grep'ing per-worker
    # logs.  HEARTBEAT_INTERVAL_S is cheap (one scandir + one small
    # JSON read) and happens on the supervise thread, so it never
    # blocks worker restart handling.
    HEARTBEAT_INTERVAL_S = args.heartbeat_interval
    last_heartbeat = 0.0

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

            now = time.time()
            if HEARTBEAT_INTERVAL_S > 0 and (now - last_heartbeat) >= HEARTBEAT_INTERVAL_S:
                alive = [w.name for w in workers if w.proc and w.poll() is None]
                dead = [w.name for w in workers if w.disabled]
                snap = _heartbeat_snapshot()
                sup_log("HEARTBEAT",
                        step=snap["step"], lr=snap["lr"],
                        score_ramp=snap["score_ramp"],
                        pool=snap["pool"],
                        cand=snap["candidates"],
                        accepted=snap["accepted"],
                        rejected=snap["rejected"],
                        latest=snap["latest"],
                        alive=",".join(alive) or "-",
                        dead=",".join(dead) or "-")
                last_heartbeat = now

            time.sleep(2.0)
    finally:
        # Every worker gets its OWN graceful_timeout window — not a
        # shared budget that the first slow worker eats up.  We wait
        # for all of them in parallel on a thread pool so a long
        # gatekeeper match doesn't starve rate's grace period.
        def _wait_then_kill(w):
            if w.wait(timeout=args.graceful_timeout):
                return ("CLEAN_EXIT",
                        {"rc": w.proc.returncode if w.proc else None})
            w.kill_group()
            w.wait(timeout=5.0)
            return ("KILL_GROUP",
                    {"reason": f"graceful timeout ({args.graceful_timeout}s) exceeded"})

        import concurrent.futures
        alive_workers = [w for w in workers if w.proc is not None]
        if alive_workers:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(alive_workers),
                    thread_name_prefix="shutdown") as ex:
                futures = {ex.submit(_wait_then_kill, w): w for w in alive_workers}
                for fut in concurrent.futures.as_completed(futures):
                    w = futures[fut]
                    tag, fields = fut.result()
                    sup_log(tag, proc=w.name, **fields)
                    w.close_log()

        # Belt-and-braces: anything still alive (e.g., SIGKILL race)
        # gets one more group kill before we exit.
        for w in workers:
            if w.proc and w.proc.poll() is None:
                w.kill_group()
        sup_log("SUPERVISOR_STOP",
                escalated=shutting_down.get("escalated", False))


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
    # Per-worker GPU + NN assignment.  Naming convention: every
    # per-worker flag is prefixed with the worker name (--selfplay-*,
    # --train-*, --gate-*, --rate-*) so `--help | grep ^--selfplay-`
    # enumerates everything tied to one worker.
    #
    # For a given worker:
    #   --<proc>-gpus                set as CUDA_VISIBLE_DEVICES
    #   --<proc>-nn-device-ids       passed to the C++ binary; 0-indexed
    #                                against the VISIBLE set (not physical)
    #   --<proc>-nn-server-threads   count of NN batching threads; must
    #                                equal len(--<proc>-nn-device-ids)
    p.add_argument("--selfplay-gpus", default="0")
    p.add_argument("--train-gpus", default="0")
    p.add_argument("--gate-gpus", default="0")
    p.add_argument("--rate-gpus", default="0")
    p.add_argument("--selfplay-nn-device-ids", default="0,0")
    p.add_argument("--gate-nn-device-ids", default="0")
    p.add_argument("--rate-nn-device-ids", default="0")
    p.add_argument("--selfplay-nn-server-threads", type=int, default=None,
                   help="NN server threads for selfplay; defaults to "
                        "len(--selfplay-nn-device-ids) if unset")
    p.add_argument("--gate-nn-server-threads", type=int, default=None,
                   help="NN server threads for gatekeeper; defaults to "
                        "len(--gate-nn-device-ids) if unset")
    p.add_argument("--rate-nn-server-threads", type=int, default=None,
                   help="NN server threads for rate; defaults to "
                        "len(--rate-nn-device-ids) if unset")

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
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--replay-target", type=float, default=4.0)
    p.add_argument("--n-augmentations", type=int, default=8)
    p.add_argument("--ring-games", type=int, default=2000)
    p.add_argument("--bucket-cap-mult", type=int, default=64)
    p.add_argument("--min-window-games", type=int, default=2000)
    p.add_argument("--min-ring-rows", type=int, default=10240)
    p.add_argument("--sample-batch-timeout-s", type=float, default=30.0)
    p.add_argument("--export-every", type=int, default=5000)
    p.add_argument("--status-publish-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--value-ramp-steps", type=int, default=30000)
    p.add_argument("--score-ramp-steps", type=int, default=50000)
    # Head-weight endpoints (the ramps interpolate from *_start to *_end).
    p.add_argument("--value-weight-start", type=float, default=1.0)
    p.add_argument("--value-weight-end", type=float, default=2.0)
    p.add_argument("--score-mean-weight-start", type=float, default=0.004)
    p.add_argument("--score-mean-weight-end", type=float, default=0.010)
    # Fixed head weights (no ramp, but overridable per-run via CLI).
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--score-stdev-weight", type=float, default=0.006)
    p.add_argument("--score-belief-weight", type=float, default=0.035)
    p.add_argument("--ownership-weight", type=float, default=0.85)
    p.add_argument("--opp-policy-weight", type=float, default=0.1)

    # Selfplay
    p.add_argument("--selfplay-batch-games", type=int, default=300)
    p.add_argument("--selfplay-sims", type=int, default=500)
    p.add_argument("--window-games", type=int, default=80000)
    p.add_argument("--score-weight-max", type=float, default=0.06,
                   help="MCTS score weight at full ramp (selfplay reads "
                        "score_ramp from status.json and multiplies)")
    p.add_argument("--selfplay-threads", type=int, default=0,
                   help="Parallel game workers (0 = os.cpu_count())")
    p.add_argument("--selfplay-search-threads", type=int, default=16,
                   help="MCTS search threads per move")

    # Gatekeeper
    p.add_argument("--gate-games", type=int, default=200)
    p.add_argument("--gate-sims", type=int, default=150)
    p.add_argument("--gate-threshold", type=float, default=0.5)
    p.add_argument("--gate-poll-interval", type=float, default=30.0)
    p.add_argument("--gate-threads", type=int, default=0,
                   help="Parallel match workers (0 = os.cpu_count())")
    p.add_argument("--gate-search-threads", type=int, default=16)

    # Rating
    p.add_argument("--rating", action="store_true",
                   help="Enable optional rating loop (costs extra GPU time)")
    p.add_argument("--rating-games", type=int, default=80)
    p.add_argument("--rating-sims", type=int, default=200)
    p.add_argument("--rating-pool-size", type=int, default=5)
    p.add_argument("--rating-interval", type=float, default=7200.0)
    p.add_argument("--rate-threads", type=int, default=0,
                   help="Parallel pair workers (0 = os.cpu_count())")
    p.add_argument("--rate-search-threads", type=int, default=16)

    # MCTS / game (shared across workers)
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.15)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.22)
    p.add_argument("--temp-threshold", type=int, default=12)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--score-scale", type=float, default=18.0)
    p.add_argument("--max-batch", type=int, default=256)

    # Shutdown
    p.add_argument("--graceful-timeout", type=float, default=180.0,
                   help="Seconds to wait for workers to finish their "
                        "current batch/match after a shutdown request "
                        "before SIGKILL'ing the worker process group")

    # Heartbeat
    p.add_argument("--heartbeat-interval", type=float, default=60.0,
                   help="Seconds between supervisor HEARTBEAT lines "
                        "(pool size, step, candidates, accepted, latest). "
                        "0 disables.")


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
