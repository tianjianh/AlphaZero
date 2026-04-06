"""
MiniGo AlphaZero — Training Pipeline

Usage:
  python run_loop.py init small                     # 9x9, 64f/5b, ~100 iters
  python run_loop.py init large                     # 9x9, 128f/10b, ~200 iters
  python run_loop.py init quick                     # 5x5, 32f/3b, 5 iters (test)
  python run_loop.py init --board 9 --filters 96 --blocks 8   # custom arch

  python run_loop.py train                          # auto-detect GPUs, all cores
  python run_loop.py train --threads 64 --nn-device-ids 0,0,1,1
  python run_loop.py train --iterations 20          # run at most 20 more iters

  python run_loop.py status                         # show progress
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_DIR / "build"
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "training" / "selfplay"
LOGS_DIR = PROJECT_DIR / "training" / "logs"
CHECKPOINT_DIR = PROJECT_DIR / "training" / "checkpoints"
STATE_FILE = PROJECT_DIR / "training" / "state.json"
PLAN_FILE = PROJECT_DIR / "training" / "plan.json"
TRAIN_LOG = LOGS_DIR / "train.log"
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def version_onnx(v):
    return MODELS_DIR / f"v{v:04d}.onnx"


def version_checkpoint(v):
    return CHECKPOINT_DIR / f"v{v:04d}.pt"


def best_onnx():
    return MODELS_DIR / "best.onnx"


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def num_cores():
    return os.cpu_count() or 4


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{timestamp()}] {msg}"
    print(line, flush=True)
    if LOGS_DIR.is_dir():
        with open(LOGS_DIR / "pipeline.log", "a") as f:
            f.write(line + "\n")


def tlog(msg):
    if TRAIN_LOG.is_file():
        with open(TRAIN_LOG, "a") as f:
            f.write(msg + "\n")


def tlog_section(title):
    tlog("")
    tlog("════════════════════════════════════════════════════════════════")
    tlog(f"  {title}")
    tlog(f"  {timestamp()}")
    tlog("════════════════════════════════════════════════════════════════")


def vstr(v):
    return f"v{v:04d}"


# ══════════════════════════════════════════════════════════
#  State
# ══════════════════════════════════════════════════════════

def read_state():
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {"pipeline_iter": 0, "best_version": 0, "total_games": 0, "total_promotions": 0}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


# ══════════════════════════════════════════════════════════
#  Plan
# ══════════════════════════════════════════════════════════

def read_plan():
    if not PLAN_FILE.is_file():
        print("ERROR: No training plan found (training/plan.json).")
        print("Run 'python run_loop.py init small' (or large/quick) first.")
        sys.exit(1)
    return json.loads(PLAN_FILE.read_text())


def get_stage_for_iter(plan, it):
    for i, s in enumerate(plan["stages"]):
        if s["start"] <= it <= s["end"]:
            return {**s, "idx": i + 1}
    print(f"ERROR: iteration {it} is outside all stages in training/plan.json")
    sys.exit(1)


def get_total_iterations(plan):
    return plan["stages"][-1]["end"]


def build_data_window(plan, end_iter):
    start = max(1, end_iter - plan["window_size"] + 1)
    dirs = []
    for w in range(start, end_iter + 1):
        d = DATA_DIR / f"iter_{w:04d}"
        if d.is_dir():
            dirs.append(str(d))
    return ",".join(dirs)


# ══════════════════════════════════════════════════════════
#  Stage presets
# ══════════════════════════════════════════════════════════

def generate_stages(preset, board, filters, blocks):
    if preset == "quick":
        return [{"name": "Quick test", "start": 1, "end": 5,
                 "games": 20, "sims": 100, "epochs": 5, "lr": "2e-3", "eval_games": 0}]
    if preset == "small":
        return [
            {"name": "Warm up",     "start": 1,  "end": 5,   "games": 500,  "sims": 400, "epochs": 10, "lr": "2e-3", "eval_games": 0},
            {"name": "Explore",     "start": 6,  "end": 25,  "games": 1500, "sims": 600, "epochs": 15, "lr": "1e-3", "eval_games": 100},
            {"name": "Strengthen",  "start": 26, "end": 60,  "games": 2500, "sims": 600, "epochs": 15, "lr": "5e-4", "eval_games": 100},
            {"name": "Polish",      "start": 61, "end": 100, "games": 3000, "sims": 800, "epochs": 20, "lr": "1e-4", "eval_games": 100},
        ]
    if preset == "large":
        return [
            {"name": "Warm up",     "start": 1,   "end": 10,  "games": 1000, "sims": 600,  "epochs": 10, "lr": "2e-3", "eval_games": 0},
            {"name": "Explore",     "start": 11,  "end": 50,  "games": 3000, "sims": 800,  "epochs": 15, "lr": "1e-3", "eval_games": 200},
            {"name": "Strengthen",  "start": 51,  "end": 130, "games": 4000, "sims": 1000, "epochs": 20, "lr": "5e-4", "eval_games": 200},
            {"name": "Master",      "start": 131, "end": 200, "games": 5000, "sims": 1200, "epochs": 20, "lr": "1e-4", "eval_games": 200},
        ]
    # custom
    total = max(30, min(300, 60 * filters * blocks // 320))
    s1 = max(3, total * 8 // 100)
    s2 = total * 25 // 100
    s3 = total * 58 // 100
    s4 = total
    bg = max(50, 500 * board * board // 81)
    return [
        {"name": "Warm up",     "start": 1,      "end": s1, "games": bg,     "sims": 400, "epochs": 10, "lr": "2e-3", "eval_games": 0},
        {"name": "Explore",     "start": s1 + 1,  "end": s2, "games": bg * 3, "sims": 600, "epochs": 15, "lr": "1e-3", "eval_games": 100},
        {"name": "Strengthen",  "start": s2 + 1,  "end": s3, "games": bg * 5, "sims": 600, "epochs": 15, "lr": "5e-4", "eval_games": 100},
        {"name": "Polish",      "start": s3 + 1,  "end": s4, "games": bg * 6, "sims": 800, "epochs": 20, "lr": "1e-4", "eval_games": 100},
    ]


def generate_plan(board, filters, blocks, preset, arch="resnet",
                   d_model=192, depth=8, heads=6, kv_groups=2, mlp_ratio=4):
    komi = 7.5 if board >= 13 else 6.5
    batch_size = 1024
    eval_threshold = 0.55
    window = 20
    temp_threshold = 15
    dirichlet_alpha = 0.15
    score_weight = 0.02

    if preset == "quick":
        batch_size = 64; window = 5; eval_threshold = 0.5; temp_threshold = 8
    elif preset == "large":
        window = 30

    if board >= 13:
        dirichlet_alpha = 0.03
        temp_threshold = 30

    plan = {
        "arch": arch, "board": board, "komi": komi,
        "filters": filters, "blocks": blocks,
        "batch_size": batch_size, "eval_threshold": eval_threshold,
        "window_size": window, "c_puct": 1.5,
        "dirichlet_alpha": dirichlet_alpha, "dirichlet_epsilon": 0.25,
        "temp_threshold": temp_threshold, "score_weight": score_weight,
        "stages": generate_stages(preset, board, filters, blocks),
    }
    if arch == "vit":
        plan.update({"d_model": d_model, "depth": depth, "heads": heads,
                      "kv_groups": kv_groups, "mlp_ratio": mlp_ratio})
    return plan


# ══════════════════════════════════════════════════════════
#  Hardware
# ══════════════════════════════════════════════════════════

def detect_gpu_count():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return len([l for l in out.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def detect_hardware():
    hw = {
        "threads": num_cores(),
        "search_threads": 16,
        "selfplay_instances": 1,
        "max_batch": 256,
    }
    gpu_count = detect_gpu_count()
    if gpu_count >= 2:
        hw["nn_server_threads"] = gpu_count * 2
        hw["nn_device_ids"] = ",".join(f"{g},{g}" for g in range(gpu_count))
    elif gpu_count == 1:
        hw["nn_server_threads"] = 2
        hw["nn_device_ids"] = "0,0"
    else:
        hw["nn_server_threads"] = 1
        hw["nn_device_ids"] = "0"
    return hw


# ══════════════════════════════════════════════════════════
#  Build
# ══════════════════════════════════════════════════════════

def build_if_needed():
    selfplay = BUILD_DIR / "selfplay"
    evaluate = BUILD_DIR / "evaluate"
    if selfplay.is_file() and evaluate.is_file():
        return
    log("Building C++ components...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                   cwd=BUILD_DIR, check=True)
    subprocess.run(["make", f"-j{num_cores()}"], cwd=BUILD_DIR, check=True)


# ══════════════════════════════════════════════════════════
#  Self-play
# ══════════════════════════════════════════════════════════

def run_selfplay(iter_data, model, games, sims, hw, plan):
    mcts_flags = [
        "--c-puct", str(plan["c_puct"]),
        "--dirichlet-alpha", str(plan["dirichlet_alpha"]),
        "--dirichlet-epsilon", str(plan["dirichlet_epsilon"]),
        "--temp-threshold", str(plan["temp_threshold"]),
        "--komi", str(plan["komi"]),
        "--score-weight", str(plan["score_weight"]),
    ]
    base_cmd = [
        str(BUILD_DIR / "selfplay"),
        "--model", str(model),
        "--sims", str(sims),
        "--search-threads", str(hw["search_threads"]),
        "--max-batch", str(hw["max_batch"]),
        "--nn-server-threads", str(hw["nn_server_threads"]),
        "--nn-device-ids", hw["nn_device_ids"],
        "--output", str(iter_data),
    ] + mcts_flags

    instances = hw["selfplay_instances"]
    if instances <= 1:
        subprocess.run(base_cmd + ["--games", str(games),
                                   "--threads", str(hw["threads"])], check=True)
    else:
        gpii = math.ceil(games / instances)
        tpii = math.ceil(hw["threads"] / instances)
        log(f"  ({gpii} games x {tpii} threads x {instances} instances)")
        procs = []
        for _ in range(instances):
            p = subprocess.Popen(base_cmd + ["--games", str(gpii),
                                             "--threads", str(tpii)])
            procs.append(p)
        failed = 0
        for p in procs:
            if p.wait() != 0:
                log(f"WARNING: selfplay pid {p.pid} failed")
                failed += 1
        if failed:
            log("WARNING: some selfplay instances failed")


# ══════════════════════════════════════════════════════════
#  Compress selfplay data
# ══════════════════════════════════════════════════════════

def compress_selfplay(iter_data):
    """Compress .bin → .bin.zst to save disk."""
    bins = list(Path(iter_data).glob("game_*.bin"))
    if not bins:
        return
    try:
        subprocess.run(["zstd", "--rm", "-q"] + [str(b) for b in bins],
                       check=False, capture_output=True)
    except FileNotFoundError:
        pass  # zstd not installed, leave uncompressed


# ══════════════════════════════════════════════════════════
#  Count existing games
# ══════════════════════════════════════════════════════════

def count_existing_games(iter_data):
    d = Path(iter_data)
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("*.bin")) + \
           sum(1 for _ in d.glob("*.bin.zst")) + \
           sum(1 for _ in d.glob("*.bin.gz"))


# ══════════════════════════════════════════════════════════
#  INIT
# ══════════════════════════════════════════════════════════

def cmd_init(args):
    board = args.board
    filters = args.filters
    blocks = args.blocks
    arch = args.arch
    preset = args.preset or "custom"

    print("============================================")
    print("  MiniGo Training — Init")
    print("============================================")
    print(f"  Preset:     {preset}")
    print(f"  Arch:       {arch}")
    print(f"  Board:      {board}x{board}")
    if arch == "resnet":
        print(f"  Network:    {filters}f x {blocks}b")
    else:
        print(f"  Network:    d={args.d_model} depth={args.depth} heads={args.heads} kv={args.kv_groups}")
    print()

    print("This will DELETE all existing training data:")
    print("  models/              (ONNX model files)")
    print("  training/            (selfplay data, eval games, checkpoints, logs)")
    print("  trt_cache/           (TensorRT engine cache)")
    print("  training/plan.json   (training schedule)")
    print()

    if not args.yes:
        confirm = input("Continue? [y/N] ").strip()
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    print("Clearing...")
    for d in [MODELS_DIR, PROJECT_DIR / "training", PROJECT_DIR / "trt_cache"]:
        if d.is_dir():
            shutil.rmtree(d)
    for f in [PROJECT_DIR / "training_plan", PROJECT_DIR / "training_plan.json"]:
        if f.is_file():
            f.unlink()

    # Create fresh directories
    for d in [MODELS_DIR, DATA_DIR, LOGS_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate training plan
    print("Generating training plan...")
    plan = generate_plan(board, filters, blocks, preset, arch=arch,
                         d_model=args.d_model, depth=args.depth,
                         heads=args.heads, kv_groups=args.kv_groups,
                         mlp_ratio=args.mlp_ratio)
    PLAN_FILE.write_text(json.dumps(plan, indent=2) + "\n")

    # Create initial random model (v0000) and set as best
    print("Creating initial model (v0000)...")
    export_cmd = [
        sys.executable, str(SCRIPTS_DIR / "export_onnx.py"),
        "--output", str(version_onnx(0)),
        "--board", str(board), "--arch", arch,
        "--filters", str(filters), "--blocks", str(blocks),
        "--init",
    ]
    if arch == "vit":
        export_cmd += ["--d-model", str(args.d_model), "--depth", str(args.depth),
                       "--heads", str(args.heads), "--kv-groups", str(args.kv_groups),
                       "--mlp-ratio", str(args.mlp_ratio)]
    subprocess.run(export_cmd, cwd=SCRIPTS_DIR, check=True)
    shutil.copy2(version_onnx(0), best_onnx())

    # Initialize state
    save_state({"pipeline_iter": 0, "best_version": 0,
                "total_games": 0, "total_promotions": 0})

    # Initialize train.log
    total_iters = get_total_iterations(plan)
    total_games = sum(s["games"] * (s["end"] - s["start"] + 1) for s in plan["stages"])

    with open(TRAIN_LOG, "w") as f:
        f.write(f"""MiniGo AlphaZero — Training Log
════════════════════════════════════════════════════════════════
Initialized:  {timestamp()}
Preset:       {preset}
Arch:         {arch}
Architecture: {board}x{board} board, {f'{filters} filters, {blocks} blocks' if arch == 'resnet' else f'd_model={plan.get("d_model")}, depth={plan.get("depth")}, heads={plan.get("heads")}, kv_groups={plan.get("kv_groups")}'}
Batch size:   {plan['batch_size']}
Window size:  {plan['window_size']} iterations (data streamed via mmap, no memory limit)
Komi:         {plan['komi']}
MCTS:         c_puct={plan['c_puct']}  dirichlet_alpha={plan['dirichlet_alpha']}  dirichlet_eps={plan['dirichlet_epsilon']}  temp_threshold={plan['temp_threshold']}
Score weight: {plan['score_weight']}
Eval gate:    {plan['eval_threshold']} win rate threshold

Training Plan:
""")
        for i, s in enumerate(plan["stages"]):
            gate = f"{s['eval_games']} games" if s["eval_games"] > 0 else "off"
            f.write(f"  Stage {i+1} {s['name']:14s}  iter {s['start']:3d}-{s['end']:<3d}  "
                    f"{s['games']:4d} games  {s['sims']:4d} sims  {s['epochs']:2d} epochs  "
                    f"lr={s['lr']:<5s}  gate={gate}\n")
        f.write(f"\nTotal: {total_iters} iterations, ~{total_games} games\n")
        f.write("Hardware will be logged when training starts.\n")
        f.write("════════════════════════════════════════════════════════════════\n")

    # Print the plan to console
    print()
    print("============================================")
    print("  Training Plan")
    print("============================================")
    print(f"  {'Stage':14s} {'Iters':>8s} {'Games':>7s} {'Sims':>6s} {'Epoch':>6s} {'LR':>7s} {'Gate':>6s}")
    print("  " + "─" * 60)
    for i, s in enumerate(plan["stages"]):
        gate = f"{s['eval_games']}g" if s["eval_games"] > 0 else "off"
        print(f"  {s['name']:14s} {s['start']:4d}-{s['end']:<3d} {s['games']:5d} {s['sims']:6d} "
              f"{s['epochs']:6d} {s['lr']:>7s} {gate:>6s}")
    print("  " + "─" * 60)
    print(f"  Total:         {total_iters} iters, ~{total_games} games")
    print()
    print(f"  Model:         {best_onnx()}")
    print(f"  Plan:          {PLAN_FILE}")
    print(f"  Log:           {TRAIN_LOG}")
    print()
    print("  Ready! Run:  python run_loop.py train")
    print("============================================")


# ══════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════

def cmd_status(args):
    if not PLAN_FILE.is_file():
        print("No training initialized. Run 'python run_loop.py init small' first.")
        return

    plan = read_plan()
    state = read_state()
    total_iters = get_total_iterations(plan)
    pct = state["pipeline_iter"] * 100 // total_iters if total_iters > 0 else 0

    print()
    print("============================================")
    print("  MiniGo Training Status")
    print("============================================")
    arch = plan.get("arch", "resnet")
    if arch == "resnet":
        print(f"  Architecture:   {plan['board']}x{plan['board']}, {plan['filters']}f x {plan['blocks']}b (resnet)")
    else:
        print(f"  Architecture:   {plan['board']}x{plan['board']}, d={plan['d_model']} depth={plan['depth']} "
              f"heads={plan['heads']} kv={plan['kv_groups']} (vit)")
    print(f"  Komi:           {plan['komi']}")
    print(f"  Score weight:   {plan['score_weight']}")
    print(f"  Progress:       {state['pipeline_iter']} / {total_iters} iterations ({pct}%)")
    print(f"  Best model:     {vstr(state['best_version'])} ({state['total_promotions']} promotions)")
    print(f"  Total games:    {state['total_games']}")
    print()

    print(f"  {'Stage':14s} {'Iters':>8s} {'Games':>7s} {'Sims':>6s} {'Epoch':>6s} {'LR':>7s} {'Gate':>6s}   ")
    print("  " + "─" * 66)
    for i, s in enumerate(plan["stages"]):
        gate = f"{s['eval_games']}g" if s["eval_games"] > 0 else "off"
        status = ""
        if state["pipeline_iter"] >= s["end"]:
            status = "DONE"
        elif state["pipeline_iter"] >= s["start"]:
            status = f"<- iter {state['pipeline_iter']}"
        print(f"  {s['name']:14s} {s['start']:4d}-{s['end']:<3d} {s['games']:5d} {s['sims']:6d} "
              f"{s['epochs']:6d} {s['lr']:>7s} {gate:>6s}   {status}")
    print("  " + "─" * 66)

    if state["pipeline_iter"] >= total_iters:
        print()
        print(f"  Training COMPLETE. Final model: {best_onnx()}")
    else:
        print()
        print("  Resume:  python run_loop.py train")
    print("============================================")
    print()


# ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════

def cmd_train(args):
    hw = detect_hardware()

    # Apply CLI overrides
    if args.threads is not None:
        hw["threads"] = args.threads
    if args.search_threads is not None:
        hw["search_threads"] = args.search_threads
    if args.selfplay_instances is not None:
        hw["selfplay_instances"] = args.selfplay_instances
    if args.nn_server_threads is not None:
        hw["nn_server_threads"] = args.nn_server_threads
    if args.nn_device_ids is not None:
        hw["nn_device_ids"] = args.nn_device_ids
    if args.max_batch is not None:
        hw["max_batch"] = args.max_batch

    plan = read_plan()
    state = read_state()
    build_if_needed()

    # Apply komi override
    if args.komi is not None:
        plan["komi"] = args.komi

    total_iters = get_total_iterations(plan)
    if state["pipeline_iter"] >= total_iters:
        print(f"Training already complete ({state['pipeline_iter']}/{total_iters} iterations).")
        print("To restart: python run_loop.py init small")
        return

    start_iter = state["pipeline_iter"] + 1
    end_iter = total_iters
    if args.iterations and args.iterations > 0:
        end_iter = min(end_iter, start_iter + args.iterations - 1)

    # Validate nn_device_ids count matches nn_server_threads
    device_ids = hw["nn_device_ids"].split(",")
    if len(device_ids) != hw["nn_server_threads"]:
        print(f"ERROR: nn-device-ids has {len(device_ids)} entries but "
              f"nn-server-threads is {hw['nn_server_threads']}. Must match.")
        sys.exit(1)

    print("============================================")
    print("  MiniGo Training")
    print("============================================")
    arch = plan.get("arch", "resnet")
    if arch == "vit":
        print(f"  Architecture:     {plan['board']}x{plan['board']}, d={plan['d_model']} depth={plan['depth']} heads={plan['heads']} kv={plan['kv_groups']} (vit)")
    else:
        print(f"  Architecture:     {plan['board']}x{plan['board']}, {plan['filters']}f x {plan['blocks']}b (resnet)")
    print(f"  Iterations:       {start_iter} .. {end_iter}  (of {total_iters})")
    print(f"  Best model:       {vstr(state['best_version'])}")
    print(f"  Threads:          {hw['threads']}")
    print(f"  Search threads:   {hw['search_threads']}")
    print(f"  Selfplay inst:    {hw['selfplay_instances']}")
    print(f"  NN servers:       {hw['nn_server_threads']}")
    print(f"  NN devices:       {hw['nn_device_ids']}")
    print(f"  Max batch (NN):   {hw['max_batch']}")
    print(f"  Batch size (SGD): {plan['batch_size']}")
    print(f"  Komi:             {plan['komi']}")
    print(f"  MCTS:             c_puct={plan['c_puct']} alpha={plan['dirichlet_alpha']} "
          f"eps={plan['dirichlet_epsilon']} temp={plan['temp_threshold']}")
    print(f"  Score weight:     {plan['score_weight']}")
    print("============================================")
    print()

    # Log training session
    tlog_section(f"TRAINING SESSION  iter {start_iter}..{end_iter}")
    if arch == "vit":
        tlog(f"  Architecture:     {plan['board']}x{plan['board']}, d={plan['d_model']} depth={plan['depth']} heads={plan['heads']} kv={plan['kv_groups']} (vit)")
    else:
        tlog(f"  Architecture:     {plan['board']}x{plan['board']}, {plan['filters']}f x {plan['blocks']}b (resnet)")
    tlog(f"  Komi:             {plan['komi']}")
    tlog(f"  Batch size:       {plan['batch_size']}")
    tlog(f"  Data window:      last {plan['window_size']} iterations")
    tlog(f"  Eval threshold:   {plan['eval_threshold']}")
    tlog(f"  MCTS:             c_puct={plan['c_puct']}  alpha={plan['dirichlet_alpha']}  "
         f"eps={plan['dirichlet_epsilon']}  temp={plan['temp_threshold']}")
    tlog(f"  Score weight:     {plan['score_weight']}")
    tlog("  Hardware:")
    tlog(f"    Threads:          {hw['threads']}")
    tlog(f"    Search threads:   {hw['search_threads']}")
    tlog(f"    Selfplay inst:    {hw['selfplay_instances']}")
    tlog(f"    NN servers:       {hw['nn_server_threads']}")
    tlog(f"    NN devices:       {hw['nn_device_ids']}")
    tlog(f"    Max batch (NN):   {hw['max_batch']}")
    tlog("  State:")
    tlog(f"    Best model:       {vstr(state['best_version'])}")
    tlog(f"    Total games:      {state['total_games']}")
    tlog(f"    Promotions:       {state['total_promotions']}")

    log(f"Training starting: iter={state['pipeline_iter']} best={vstr(state['best_version'])} "
        f"games={state['total_games']}")

    for it in range(start_iter, end_iter + 1):
        state["pipeline_iter"] = it
        stage = get_stage_for_iter(plan, it)

        print()
        log("============================================")
        log(f"  ITERATION {it}/{total_iters}  |  Stage: {stage['name']}  |  "
            f"Best: {vstr(state['best_version'])}  |  LR: {stage['lr']}")
        log("============================================")

        tlog("")
        tlog(f"── ITERATION {it}/{total_iters}  Stage: {stage['name']} ──────────────────────")
        tlog(f"  games={stage['games']}  sims={stage['sims']}  epochs={stage['epochs']}  "
             f"lr={stage['lr']}  eval={stage['eval_games']}  best={vstr(state['best_version'])}")

        # ── Phase 1: Self-play ─────────────────────────────
        iter_data = DATA_DIR / f"iter_{it:04d}"
        iter_data.mkdir(parents=True, exist_ok=True)

        existing = count_existing_games(iter_data)
        if existing >= stage["games"]:
            log(f"Phase 1 — Selfplay: SKIP ({existing} games exist)")
            tlog(f"  Phase 1 selfplay: SKIP ({existing} games exist)")
        else:
            games_needed = stage["games"] - existing
            selfplay_model = version_onnx(state["best_version"])
            log(f"Phase 1 — Selfplay: {games_needed} games, {stage['sims']} sims, "
                f"model {vstr(state['best_version'])}...")
            tlog(f"  Phase 1 selfplay: {games_needed} games, {stage['sims']} sims/move")
            tlog(f"    model={selfplay_model}  threads={hw['threads']}  search_threads={hw['search_threads']}")
            tlog(f"    nn_servers={hw['nn_server_threads']}  devices={hw['nn_device_ids']}  "
                 f"instances={hw['selfplay_instances']}")

            t0 = time.time()
            run_selfplay(iter_data, selfplay_model, games_needed, stage["sims"], hw, plan)
            sp_time = int(time.time() - t0)
            state["total_games"] += games_needed

            compress_selfplay(iter_data)

            per_game = f"{sp_time / games_needed:.2f}" if games_needed > 0 else "?"
            tlog(f"    done: {sp_time}s ({per_game}s/game)  total_games={state['total_games']}")
            log(f"Phase 1 — Selfplay done ({sp_time}s). Total games: {state['total_games']}")

        # ── Phase 2: Training ──────────────────────────────
        candidate_onnx = version_onnx(it)
        candidate_ckpt = version_checkpoint(it)

        if candidate_onnx.is_file() and candidate_ckpt.is_file():
            log(f"Phase 2 — Training: SKIP ({vstr(it)} exists)")
            tlog(f"  Phase 2 training: SKIP ({vstr(it)} exists)")
        else:
            train_ckpt = CHECKPOINT_DIR / "training.pt"
            window_dirs = build_data_window(plan, it)
            if not window_dirs:
                log("ERROR: no data in window")
                sys.exit(1)

            n_window = len(window_dirs.split(","))
            log(f"Phase 2 — Training: {stage['epochs']} epochs, lr={stage['lr']}, "
                f"batch={plan['batch_size']}...")
            tlog(f"  Phase 2 training: {stage['epochs']} epochs, lr={stage['lr']}, "
                 f"batch={plan['batch_size']}")
            tlog(f"    window={n_window} dirs")

            t0 = time.time()

            # Detect GPUs for DDP training
            train_gpus = detect_gpu_count()

            train_args = [
                str(SCRIPTS_DIR / "train.py"),
                "--data", window_dirs,
                "--checkpoint", str(train_ckpt),
                "--epochs", str(stage["epochs"]),
                "--batch-size", str(plan["batch_size"]),
                "--lr", stage["lr"],
                "--board", str(plan["board"]),
                "--arch", arch,
                "--filters", str(plan["filters"]),
                "--blocks", str(plan["blocks"]),
                "--output-onnx", str(candidate_onnx),
                "--log-file", str(TRAIN_LOG),
            ]
            if arch == "vit":
                train_args += [
                    "--d-model", str(plan["d_model"]),
                    "--depth", str(plan["depth"]),
                    "--heads", str(plan["heads"]),
                    "--kv-groups", str(plan["kv_groups"]),
                    "--mlp-ratio", str(plan["mlp_ratio"]),
                ]

            if train_gpus > 1:
                import random
                port = 29500 + random.randint(0, 999)
                cmd = ["torchrun", f"--nproc_per_node={train_gpus}",
                       f"--master_port={port}"] + train_args
            else:
                cmd = [sys.executable] + train_args

            subprocess.run(cmd, cwd=SCRIPTS_DIR, check=True)

            tr_time = int(time.time() - t0)

            if not candidate_onnx.is_file():
                log(f"ERROR: training failed — {candidate_onnx} not produced")
                tlog("    ERROR: training failed — no ONNX produced")
                sys.exit(1)
            if train_ckpt.is_file():
                shutil.copy2(train_ckpt, candidate_ckpt)

            log(f"Phase 2 — Training done ({tr_time}s). Candidate: {vstr(it)}")

        # ── Phase 3: Evaluation & Gating ───────────────────
        if stage["eval_games"] > 0 and state["best_version"] > 0:
            log(f"Phase 3 — Eval: {vstr(it)} vs {vstr(state['best_version'])} "
                f"({stage['eval_games']} games, {stage['sims']} sims)...")
            tlog(f"  Phase 3 eval: {vstr(it)} vs {vstr(state['best_version'])}  "
                 f"{stage['eval_games']} games  {stage['sims']} sims")

            t0 = time.time()
            eval_dir = PROJECT_DIR / "training" / "eval" / f"iter_{it:04d}"
            eval_dir.mkdir(parents=True, exist_ok=True)

            eval_cmd = [
                str(BUILD_DIR / "evaluate"),
                "--model1", str(candidate_onnx),
                "--model2", str(version_onnx(state["best_version"])),
                "--games", str(stage["eval_games"]),
                "--threads", str(hw["threads"]),
                "--search-threads", str(hw["search_threads"]),
                "--max-batch", str(hw["max_batch"]),
                "--nn-server-threads", str(hw["nn_server_threads"]),
                "--nn-device-ids", hw["nn_device_ids"],
                "--sims", str(stage["sims"]),
                "--c-puct", str(plan["c_puct"]),
                "--komi", str(plan["komi"]),
                "--score-weight", str(plan["score_weight"]),
                "--threshold", str(plan["eval_threshold"]),
                "--output", str(eval_dir),
            ]

            result = subprocess.run(eval_cmd, capture_output=True, text=True)
            eval_output = result.stdout + result.stderr
            print(eval_output, end="")
            eval_result = result.returncode
            ev_time = int(time.time() - t0)

            # Parse results
            wr_match = re.search(r"win rate:\s*([\d.]+%)", eval_output)
            m1w_match = re.search(r"Model 1 wins:\s*(\d+)", eval_output)
            m2w_match = re.search(r"Model 2 wins:\s*(\d+)", eval_output)
            drw_match = re.search(r"Draws:\s*(\d+)", eval_output)
            verdict_match = re.search(r"RESULT:\s*(PASS|FAIL)", eval_output)

            wr = wr_match.group(1) if wr_match else "?"
            m1w = m1w_match.group(1) if m1w_match else "?"
            m2w = m2w_match.group(1) if m2w_match else "?"
            drw = drw_match.group(1) if drw_match else "?"
            verdict = verdict_match.group(1) if verdict_match else "?"

            tlog(f"    Win rate: {wr} ({m1w}-{m2w}-{drw})  {verdict}  {ev_time}s")

            if eval_result == 0:
                log(f"Phase 3 — PROMOTED {vstr(it)} {wr} (beats {vstr(state['best_version'])}) [{ev_time}s]")
                shutil.copy2(candidate_onnx, best_onnx())
                state["best_version"] = it
                state["total_promotions"] += 1
            else:
                log(f"Phase 3 — REJECTED {vstr(it)} {wr} [{ev_time}s]")
        else:
            if state["best_version"] == 0:
                log(f"Phase 3 — Auto-promote {vstr(it)} (first model)")
            else:
                log(f"Phase 3 — Auto-promote {vstr(it)} (no gate this stage)")
            tlog(f"  Phase 3: auto-promote {vstr(it)}")
            shutil.copy2(candidate_onnx, best_onnx())
            state["best_version"] = it
            state["total_promotions"] += 1

        save_state(state)
        log(f"Iteration {it} done. Best: {vstr(state['best_version'])}  "
            f"Promotions: {state['total_promotions']}/{it}")
        tlog(f"  Result: best={vstr(state['best_version'])}  promotions={state['total_promotions']}/{it}  "
             f"total_games={state['total_games']}")

    print()
    print("============================================")
    if end_iter >= total_iters:
        print("  Training Complete!")
    else:
        print(f"  Session Complete (paused at iter {end_iter})")
        print("  Resume: python run_loop.py train")
    print(f"  Best model:     {vstr(state['best_version'])}")
    print(f"  Total games:    {state['total_games']}")
    print(f"  Promotions:     {state['total_promotions']}")
    print(f"  Model file:     {best_onnx()}")
    print("============================================")

    tlog_section("SESSION END")
    tlog(f"  Best: {vstr(state['best_version'])}  Games: {state['total_games']}  "
         f"Promotions: {state['total_promotions']}")


# ══════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MiniGo AlphaZero — Training Pipeline",
        epilog="""Examples:
  python run_loop.py init small
  python run_loop.py train                          # auto-detect GPUs
  python run_loop.py train --nn-device-ids 0,0,1,1  # manual GPU assignment
  python run_loop.py status
  python run_loop.py train --iterations 10          # run 10 more iters then stop
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ──
    p_init = sub.add_parser("init", help="Initialize training (clears previous state)")
    p_init.add_argument("preset", nargs="?", choices=["quick", "small", "large"],
                        help="Preset: quick (5x5 test), small (9x9), large (9x9 deep)")
    p_init.add_argument("--arch", default="resnet", choices=["resnet", "vit"],
                        help="Model architecture (default: resnet)")
    p_init.add_argument("--board", type=int, default=None)
    p_init.add_argument("--filters", type=int, default=None)
    p_init.add_argument("--blocks", type=int, default=None)
    p_init.add_argument("--d-model", type=int, default=192)
    p_init.add_argument("--depth", type=int, default=8)
    p_init.add_argument("--heads", type=int, default=6)
    p_init.add_argument("--kv-groups", type=int, default=2)
    p_init.add_argument("--mlp-ratio", type=int, default=4)
    p_init.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")

    # ── train ──
    p_train = sub.add_parser("train", help="Start or resume training")
    p_train.add_argument("--threads", type=int, default=None,
                         help="Worker threads (default: all cores)")
    p_train.add_argument("--search-threads", type=int, default=None)
    p_train.add_argument("--selfplay-instances", type=int, default=None)
    p_train.add_argument("--nn-server-threads", type=int, default=None)
    p_train.add_argument("--nn-device-ids", type=str, default=None)
    p_train.add_argument("--max-batch", type=int, default=None)
    p_train.add_argument("--iterations", type=int, default=None,
                         help="Max iterations to run this session")
    p_train.add_argument("--komi", type=float, default=None,
                         help="Override komi from training plan")

    # ── status ──
    sub.add_parser("status", help="Show training progress")

    args = parser.parse_args()

    if args.command == "init":
        # Resolve preset defaults
        if args.preset == "quick":
            args.board = args.board or 5; args.filters = args.filters or 32; args.blocks = args.blocks or 3
        elif args.preset == "large":
            args.board = args.board or 9; args.filters = args.filters or 128; args.blocks = args.blocks or 10
        elif args.preset == "small":
            args.board = args.board or 9; args.filters = args.filters or 64; args.blocks = args.blocks or 5
        else:
            args.board = args.board or 9; args.filters = args.filters or 64; args.blocks = args.blocks or 5
            if args.preset is None:
                args.preset = "custom"
        cmd_init(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
