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


def get_stage_config(stage, plan):
    """Merge stage overrides with plan defaults for training and mcts params.

    Any key from training/mcts can be overridden per-stage by adding it
    directly to the stage dict.  E.g. a stage with "score_weight": 0.1
    overrides the global mcts.score_weight for that stage only.
    """
    t = dict(plan["training"])
    mc = dict(plan["mcts"])
    for k in t:
        if k in stage:
            t[k] = stage[k]
    for k in mc:
        if k in stage:
            mc[k] = stage[k]
    return t, mc


def build_data_window(window_size, end_iter):
    start = max(1, end_iter - window_size + 1)
    dirs = []
    for w in range(start, end_iter + 1):
        d = DATA_DIR / f"iter_{w:04d}"
        if d.is_dir():
            dirs.append(str(d))
    return ",".join(dirs)


# ══════════════════════════════════════════════════════════
#  Stage presets
# ══════════════════════════════════════════════════════════

def _stage(name, start, end, games, sims, epochs, lr, eval_games,
           score_weight, score_weight_loss, window_size,
           c_puct, temp_threshold, dirichlet_epsilon):
    """Build a stage dict with all per-stage overrides."""
    return {"name": name, "start": start, "end": end,
            "games": games, "sims": sims, "epochs": epochs, "lr": lr,
            "eval_games": eval_games,
            "score_weight": score_weight, "score_weight_loss": score_weight_loss,
            "window_size": window_size,
            "c_puct": c_puct, "temp_threshold": temp_threshold,
            "dirichlet_epsilon": dirichlet_epsilon}

# Per-stage exploration schedule (explore→exploit as model strengthens)
#                    score_wt  sw_loss  window  c_puct  temp  dir_eps
_EXPLORE = [
    # Bootstrap:     aggressive explore — network is random
    (0.0,   0.05,    3,      2.0,    20,   0.30),
    # Warm up:       still exploring, score head too noisy
    (0.0,   0.05,    3,      1.75,   18,   0.28),
    # Early gated:   tiny score, keep exploring
    (0.02,  0.1,     4,      1.5,    15,   0.25),
    # Consolidate:   still exploring — model needs diverse data
    (0.02,  0.1,     6,      1.5,    15,   0.25),
    # Steady:        start exploiting, score head becoming useful
    (0.05,  0.15,    8,      1.25,   12,   0.22),
    # Overnight:     moderate exploitation
    (0.1,   0.2,     8,      1.1,    12,   0.20),
]
# Large preset uses wider windows for later stages
_EXPLORE_LARGE = list(_EXPLORE)
_EXPLORE_LARGE[3] = (0.1, 0.5, 8, 1.25, 15, 0.22)
_EXPLORE_LARGE[4] = (0.15, 0.5, 10, 1.1, 12, 0.20)
_EXPLORE_LARGE[5] = (0.15, 0.5, 10, 1.1, 12, 0.20)


def generate_stages(preset, board, filters, blocks, arch="resnet"):
    vit = (arch == "vit")
    if preset == "quick":
        lr = "8e-4" if vit else "2e-3"
        return [{"name": "Quick test", "start": 1, "end": 5,
                 "games": 20, "sims": 100, "epochs": 5, "lr": lr, "eval_games": 0}]

    if vit:
        lrs = ["8e-4", "6e-4", "5e-4", "4e-4", "3e-4", "2e-4"]
    else:
        lrs = ["1.2e-3", "9e-4", "6e-4", "4.5e-4", "3e-4", "2e-4"]

    if preset == "small":
        if vit:
            #        score_wt swl    win  cpuct temp  eps
            return [
                _stage("Bootstrap",       1,  4,  500, 256, 3, lrs[0], 0,
                       0.0,  0.05, 3,  2.0,  20, 0.30),
                _stage("Warm up",         5,  8,  700, 384, 3, lrs[1], 0,
                       0.0,  0.05, 4,  1.75, 18, 0.28),
                _stage("Early gated",     9, 14, 1000, 512, 4, lrs[2], 120,
                       0.02, 0.08, 4,  1.5,  15, 0.25),
                _stage("Consolidate",    15, 22, 1200, 512, 5, lrs[3], 200,
                       0.03, 0.10, 6,  1.4,  14, 0.23),
                _stage("Steady improve", 23, 32, 1400, 640, 5, lrs[4], 240,
                       0.05, 0.12, 8,  1.25, 12, 0.20),
                _stage("Overnight extend",33,48, 1600, 768, 5, lrs[5], 240,
                       0.08, 0.15, 8,  1.1,  10, 0.18),
            ]
        ex = _EXPLORE
        ep = [3, 3, 3, 4, 5, 5]
        return [
            _stage("Bootstrap",       1,  4,   400, 200, ep[0], lrs[0], 0,   *ex[0]),
            _stage("Warm up",         5,  8,   600, 300, ep[1], lrs[1], 0,   *ex[1]),
            _stage("Early gated",     9, 14,   900, 400, ep[2], lrs[2], 100, *ex[2]),
            _stage("Consolidate",    15, 22,  1000, 400, ep[3], lrs[3], 200, *ex[3]),
            _stage("Steady improve", 23, 32,  1200, 500, ep[4], lrs[4], 200, *ex[4]),
            _stage("Overnight extend",33,48,  1400, 500, ep[5], lrs[5], 200, *ex[5]),
        ]
    if preset == "large":
        ex = _EXPLORE_LARGE
        vit_e = [3, 3, 4, 5, 6, 6] if vit else [3, 3, 3, 4, 5, 5]
        return [
            _stage("Bootstrap",        1,   6,  800, 300, vit_e[0], lrs[0], 0,   *ex[0]),
            _stage("Warm up",          7,  15, 1200, 400, vit_e[1], lrs[1], 0,   *ex[1]),
            _stage("Early gated",     16,  30, 2000, 600, vit_e[2], lrs[2], 200, *ex[2]),
            _stage("Consolidate",     31,  60, 3000, 600, vit_e[3], lrs[3], 200, *ex[3]),
            _stage("Steady improve",  61, 120, 4000, 800, vit_e[4], lrs[4], 200, *ex[4]),
            _stage("Overnight extend",121, 200,5000,1000, vit_e[5], lrs[5], 200, *ex[5]),
        ]

    # custom
    total = max(30, min(300, 60 * filters * blocks // 320))
    s1 = max(3, total * 8 // 100)
    s2 = total * 17 // 100
    s3 = total * 29 // 100
    s4 = total * 46 // 100
    s5 = total * 67 // 100
    s6 = total
    bg = max(50, 500 * board * board // 81)
    ex = _EXPLORE
    return [
        _stage("Bootstrap",        1,    s1, bg,     200, 8, lrs[0], 0,   *ex[0]),
        _stage("Warm up",         s1+1,  s2, bg*2,   300, 6, lrs[1], 0,   *ex[1]),
        _stage("Early gated",     s2+1,  s3, bg*3,   400, 4, lrs[2], 100, *ex[2]),
        _stage("Consolidate",     s3+1,  s4, bg*4,   400, 3, lrs[3], 200, *ex[3]),
        _stage("Steady improve",  s4+1,  s5, bg*5,   500, 3, lrs[4], 200, *ex[4]),
        _stage("Overnight extend",s5+1,  s6, bg*6,   500, 3, lrs[5], 200, *ex[5]),
    ]


def generate_plan(board, filters, blocks, preset, arch="resnet",
                   d_model=192, depth=8, heads=6, kv_groups=2, mlp_ratio=4):
    komi = 7.5 if board >= 13 else 6.5
    batch_size = 256 if arch == "vit" else 512
    eval_threshold = 0.53 if arch == "vit" else 0.52
    window = 8 if arch == "vit" else 6
    temp_threshold = 14 if arch == "vit" else 15
    dirichlet_alpha = 0.15
    score_weight = 0.03 if arch == "vit" else 0.02

    if preset == "quick":
        batch_size = 64; window = 3; eval_threshold = 0.5; temp_threshold = 8
    elif preset == "large":
        window = 10

    if board >= 13:
        dirichlet_alpha = 0.03
        temp_threshold = 30

    # Model config (read-only — architecture identity for logging)
    if arch == "vit":
        model_cfg = {"arch": arch, "board": board,
                     "d_model": d_model, "depth": depth, "heads": heads,
                     "kv_groups": kv_groups, "mlp_ratio": mlp_ratio}
    else:
        model_cfg = {"arch": arch, "board": board,
                     "filters": filters, "blocks": blocks}

    return {
        "model": model_cfg,
        "training": {
            "batch_size": batch_size,
            "window_size": window,
            "eval_threshold": eval_threshold,
            "policy_weight": 1.0,
            "value_weight": 1.0,
            "score_weight_loss": 0.15 if arch == "vit" else 0.5,
            "fp8": False,
        },
        "mcts": {
            "komi": komi,
            "c_puct": 1.45 if arch == "vit" else 1.5,
            "dirichlet_alpha": dirichlet_alpha,
            "dirichlet_epsilon": 0.25,
            "temp_threshold": temp_threshold,
            "score_weight": score_weight,
            "score_scale": 10.0,
        },
        "stages": generate_stages(preset, board, filters, blocks, arch),
    }


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

def run_selfplay(iter_data, model, games, sims, hw, mc):
    mcts_flags = [
        "--c-puct", str(mc["c_puct"]),
        "--dirichlet-alpha", str(mc["dirichlet_alpha"]),
        "--dirichlet-epsilon", str(mc["dirichlet_epsilon"]),
        "--temp-threshold", str(mc["temp_threshold"]),
        "--komi", str(mc["komi"]),
        "--score-weight", str(mc["score_weight"]),
        "--score-scale", str(mc["score_scale"]),
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

    m = plan["model"]
    t = plan["training"]
    mc = plan["mcts"]

    with open(TRAIN_LOG, "w") as f:
        arch_desc = (f'{m["filters"]} filters, {m["blocks"]} blocks' if arch == 'resnet'
                     else f'd_model={m["d_model"]}, depth={m["depth"]}, heads={m["heads"]}, kv_groups={m["kv_groups"]}')
        f.write(f"""MiniGo AlphaZero — Training Log
════════════════════════════════════════════════════════════════
Initialized:  {timestamp()}
Preset:       {preset}
Arch:         {arch}
Architecture: {board}x{board} board, {arch_desc}
Batch size:   {t['batch_size']}
Window size:  {t['window_size']} iterations (data streamed via mmap, no memory limit)
Komi:         {mc['komi']}
MCTS:         c_puct={mc['c_puct']}  dirichlet_alpha={mc['dirichlet_alpha']}  dirichlet_eps={mc['dirichlet_epsilon']}  temp_threshold={mc['temp_threshold']}
Score weight: {mc['score_weight']}
Score scale:  {mc['score_scale']}
Loss weights: policy={t['policy_weight']} value={t['value_weight']} score={t['score_weight_loss']}
Eval gate:    {t['eval_threshold']} win rate threshold

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
    m = plan["model"]
    mc = plan["mcts"]
    total_iters = get_total_iterations(plan)
    pct = state["pipeline_iter"] * 100 // total_iters if total_iters > 0 else 0

    print()
    print("============================================")
    print("  MiniGo Training Status")
    print("============================================")
    arch = m["arch"]
    if arch == "resnet":
        print(f"  Architecture:   {m['board']}x{m['board']}, {m['filters']}f x {m['blocks']}b (resnet)")
    else:
        print(f"  Architecture:   {m['board']}x{m['board']}, d={m['d_model']} depth={m['depth']} "
              f"heads={m['heads']} kv={m['kv_groups']} (vit)")
    print(f"  Komi:           {mc['komi']}")
    print(f"  Score weight:   {mc['score_weight']}")
    print(f"  Score scale:    {mc['score_scale']}")
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

    m = plan["model"]
    t = plan["training"]
    mc = plan["mcts"]
    arch = m["arch"]

    # Apply komi override
    if args.komi is not None:
        mc["komi"] = args.komi

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
    if arch == "vit":
        print(f"  Architecture:     {m['board']}x{m['board']}, d={m['d_model']} depth={m['depth']} heads={m['heads']} kv={m['kv_groups']} (vit)")
    else:
        print(f"  Architecture:     {m['board']}x{m['board']}, {m['filters']}f x {m['blocks']}b (resnet)")
    print(f"  Iterations:       {start_iter} .. {end_iter}  (of {total_iters})")
    print(f"  Best model:       {vstr(state['best_version'])}")
    print(f"  Threads:          {hw['threads']}")
    print(f"  Search threads:   {hw['search_threads']}")
    print(f"  Selfplay inst:    {hw['selfplay_instances']}")
    print(f"  NN servers:       {hw['nn_server_threads']}")
    print(f"  NN devices:       {hw['nn_device_ids']}")
    print(f"  Max batch (NN):   {hw['max_batch']}")
    print(f"  Batch size (SGD): {t['batch_size']}")
    print(f"  Komi:             {mc['komi']}")
    print(f"  MCTS:             c_puct={mc['c_puct']} alpha={mc['dirichlet_alpha']} "
          f"eps={mc['dirichlet_epsilon']} temp={mc['temp_threshold']}")
    print(f"  Score weight:     {mc['score_weight']}")
    print(f"  Score scale:      {mc['score_scale']}")
    print(f"  Loss weights:     policy={t['policy_weight']} "
          f"value={t['value_weight']} score={t['score_weight_loss']}")
    print("============================================")
    print()

    # Log training session
    tlog_section(f"TRAINING SESSION  iter {start_iter}..{end_iter}")
    if arch == "vit":
        tlog(f"  Architecture:     {m['board']}x{m['board']}, d={m['d_model']} depth={m['depth']} heads={m['heads']} kv={m['kv_groups']} (vit)")
    else:
        tlog(f"  Architecture:     {m['board']}x{m['board']}, {m['filters']}f x {m['blocks']}b (resnet)")
    tlog(f"  Komi:             {mc['komi']}")
    tlog(f"  Batch size:       {t['batch_size']}")
    tlog(f"  Data window:      last {t['window_size']} iterations")
    tlog(f"  Eval threshold:   {t['eval_threshold']}")
    tlog(f"  MCTS:             c_puct={mc['c_puct']}  alpha={mc['dirichlet_alpha']}  "
         f"eps={mc['dirichlet_epsilon']}  temp={mc['temp_threshold']}")
    tlog(f"  Score weight:     {mc['score_weight']}")
    tlog(f"  Score scale:      {mc['score_scale']}")
    tlog(f"  Loss weights:     policy={t['policy_weight']} "
         f"value={t['value_weight']} score={t['score_weight_loss']}")
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
        st, smc = get_stage_config(stage, plan)

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
            run_selfplay(iter_data, selfplay_model, games_needed, stage["sims"], hw, smc)
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
            window_dirs = build_data_window(st["window_size"], it)
            if not window_dirs:
                log("ERROR: no data in window")
                sys.exit(1)

            n_window = len(window_dirs.split(","))
            log(f"Phase 2 — Training: {stage['epochs']} epochs, lr={stage['lr']}, "
                f"batch={st['batch_size']}...")
            tlog(f"  Phase 2 training: {stage['epochs']} epochs, lr={stage['lr']}, "
                 f"batch={st['batch_size']}")
            tlog(f"    window={n_window} dirs")

            t0 = time.time()

            # Detect GPUs for DDP training
            train_gpus = detect_gpu_count()

            train_args = [
                str(SCRIPTS_DIR / "train.py"),
                "--data", window_dirs,
                "--checkpoint", str(train_ckpt),
                "--epochs", str(stage["epochs"]),
                "--batch-size", str(st["batch_size"]),
                "--lr", stage["lr"],
                "--board", str(m["board"]),
                "--arch", arch,
                "--filters", str(m.get("filters", 64)),
                "--blocks", str(m.get("blocks", 5)),
                "--output-onnx", str(candidate_onnx),
                "--log-file", str(TRAIN_LOG),
                "--policy-weight", str(st["policy_weight"]),
                "--value-weight", str(st["value_weight"]),
                "--score-weight-loss", str(st["score_weight_loss"]),
            ] + (["--fp8"] if st.get("fp8", False) else [])
            if arch == "vit":
                train_args += [
                    "--d-model", str(m["d_model"]),
                    "--depth", str(m["depth"]),
                    "--heads", str(m["heads"]),
                    "--kv-groups", str(m["kv_groups"]),
                    "--mlp-ratio", str(m["mlp_ratio"]),
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
                "--c-puct", str(smc["c_puct"]),
                "--komi", str(smc["komi"]),
                "--score-weight", str(smc["score_weight"]),
                "--threshold", str(st["eval_threshold"]),
                "--score-scale", str(smc["score_scale"]),
                "--output", str(eval_dir),
            ]

            result = subprocess.run(eval_cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            eval_output = result.stdout
            print(eval_output, end="")
            eval_result = result.returncode
            ev_time = int(time.time() - t0)

            # Detect crash (return code < 0 = signal, or 2 = error exit)
            if eval_result < 0 or eval_result == 2:
                log(f"Phase 3 — EVAL CRASHED (exit {eval_result}), auto-promoting {vstr(it)} [{ev_time}s]")
                tlog(f"    EVAL CRASHED exit={eval_result}  {ev_time}s — auto-promote")
                shutil.copy2(candidate_onnx, best_onnx())
                state["best_version"] = it
                state["total_promotions"] += 1
            else:
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
