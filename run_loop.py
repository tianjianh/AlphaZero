#!/usr/bin/env python3
"""
MiniXiangqi training pipeline.

This script orchestrates:
1. self-play with the current best model
2. policy/value training on recent self-play windows
3. head-to-head evaluation for promotion gating

The search/runtime architecture stays in C++ (threaded MCTS + batched NN
evaluation). This script only coordinates the loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_DIR / "build"
MODELS_DIR = PROJECT_DIR / "models"
TRAINING_DIR = PROJECT_DIR / "training"
DATA_DIR = TRAINING_DIR / "selfplay"
EVAL_DIR = TRAINING_DIR / "eval"
LOGS_DIR = TRAINING_DIR / "logs"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
STATE_FILE = TRAINING_DIR / "state.json"
PLAN_FILE = TRAINING_DIR / "plan.json"
TRAIN_LOG = LOGS_DIR / "train.log"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

BOARD_ROWS = 10
BOARD_COLS = 9
HISTORY_LENGTH = 4


def version_onnx(version: int) -> Path:
    return MODELS_DIR / f"v{version:04d}.onnx"


def version_checkpoint(version: int) -> Path:
    return CHECKPOINT_DIR / f"v{version:04d}.pt"


def best_onnx() -> Path:
    return MODELS_DIR / "best.onnx"


def num_cores() -> int:
    return os.cpu_count() or 4


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    if LOGS_DIR.is_dir():
        with open(LOGS_DIR / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def tlog(message: str = "") -> None:
    if TRAIN_LOG.parent.is_dir():
        with open(TRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def tlog_section(title: str) -> None:
    tlog()
    tlog("============================================================")
    tlog(f"  {title}")
    tlog(f"  {timestamp()}")
    tlog("============================================================")


def vstr(version: int) -> str:
    return f"v{version:04d}"


def read_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {
        "pipeline_iter": 0,
        "best_version": 0,
        "total_games": 0,
        "total_promotions": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def read_plan() -> dict:
    if not PLAN_FILE.is_file():
        print("ERROR: No training plan found.")
        print("Run 'python run_loop.py init small' first.")
        sys.exit(1)
    return json.loads(PLAN_FILE.read_text())


def get_stage_for_iter(plan: dict, iteration: int) -> dict:
    for idx, stage in enumerate(plan["stages"], start=1):
        if stage["start"] <= iteration <= stage["end"]:
            return {**stage, "idx": idx}
    print(f"ERROR: iteration {iteration} is outside all configured stages")
    sys.exit(1)


def get_total_iterations(plan: dict) -> int:
    return plan["stages"][-1]["end"]


def get_stage_config(stage: dict, plan: dict) -> tuple[dict, dict]:
    training = dict(plan["training"])
    mcts = dict(plan["mcts"])
    for key in tuple(training.keys()):
        if key in stage:
            training[key] = stage[key]
    for key in tuple(mcts.keys()):
        if key in stage:
            mcts[key] = stage[key]
    return training, mcts


def build_data_window(window_size: int, end_iter: int) -> str:
    start_iter = max(1, end_iter - window_size + 1)
    dirs = []
    for version in range(start_iter, end_iter + 1):
        path = DATA_DIR / f"iter_{version:04d}"
        if path.is_dir():
            dirs.append(str(path.resolve()))
    return ",".join(dirs)


def _stage(
    name: str,
    start: int,
    end: int,
    games: int,
    sims: int,
    epochs: int,
    lr: str,
    eval_games: int,
    window_size: int,
    c_puct: float,
    temp_threshold: int,
    dirichlet_epsilon: float,
    eval_threshold: float | None = None,
) -> dict:
    stage = {
        "name": name,
        "start": start,
        "end": end,
        "games": games,
        "sims": sims,
        "epochs": epochs,
        "lr": lr,
        "eval_games": eval_games,
        "window_size": window_size,
        "c_puct": c_puct,
        "temp_threshold": temp_threshold,
        "dirichlet_epsilon": dirichlet_epsilon,
    }
    if eval_threshold is not None:
        stage["eval_threshold"] = eval_threshold
    return stage


_SMALL_STAGE_CFG = [
    (3, 2.0, 20, 0.30),
    (4, 1.75, 18, 0.28),
    (4, 1.5, 15, 0.25),
    (6, 1.5, 15, 0.25),
    (8, 1.25, 12, 0.22),
    (8, 1.1, 10, 0.20),
]

_LARGE_STAGE_CFG = [
    (4, 2.0, 20, 0.30, None),
    (4, 1.75, 18, 0.28, None),
    (6, 1.5, 15, 0.25, 0.53),
    (6, 1.5, 15, 0.25, 0.54),
    (10, 1.3, 12, 0.22, 0.55),
    (12, 1.25, 12, 0.20, 0.55),
]


def generate_stages(preset: str, filters: int, blocks: int) -> list[dict]:
    if preset == "quick":
        return [
            _stage(
                "Quick test",
                1,
                5,
                games=20,
                sims=100,
                epochs=5,
                lr="2e-3",
                eval_games=0,
                window_size=3,
                c_puct=1.5,
                temp_threshold=8,
                dirichlet_epsilon=0.25,
            )
        ]

    if preset == "small":
        cfg = _SMALL_STAGE_CFG
        return [
            _stage("Bootstrap",        1,  4,  400, 200, 3, "1.2e-3",   0, *cfg[0]),
            _stage("Warm up",          5,  8,  600, 300, 3, "9e-4",     0, *cfg[1]),
            _stage("Early gated",      9, 14,  900, 400, 3, "6e-4",   100, *cfg[2]),
            _stage("Consolidate",     15, 22, 1000, 400, 4, "4.5e-4", 200, *cfg[3]),
            _stage("Steady improve",  23, 32, 1200, 500, 5, "3e-4",   200, *cfg[4]),
            _stage("Overnight extend",33, 48, 1400, 500, 5, "2e-4",   200, *cfg[5]),
        ]

    if preset == "large":
        cfg = _LARGE_STAGE_CFG
        return [
            _stage("Bootstrap",        1,  4,  400, 200, 3, "1.2e-3",   0, cfg[0][0], cfg[0][1], cfg[0][2], cfg[0][3], cfg[0][4]),
            _stage("Warm up",          5,  8,  600, 300, 3, "9e-4",     0, cfg[1][0], cfg[1][1], cfg[1][2], cfg[1][3], cfg[1][4]),
            _stage("Early gated",      9, 14,  900, 400, 3, "6e-4",   100, cfg[2][0], cfg[2][1], cfg[2][2], cfg[2][3], cfg[2][4]),
            _stage("Consolidate",     15, 24, 1100, 450, 4, "4.5e-4", 200, cfg[3][0], cfg[3][1], cfg[3][2], cfg[3][3], cfg[3][4]),
            _stage("Steady improve",  25, 40, 1300, 550, 5, "2e-4",   200, cfg[4][0], cfg[4][1], cfg[4][2], cfg[4][3], cfg[4][4]),
            _stage("Overnight extend",41, 72, 1400, 600, 3, "1e-4",   200, cfg[5][0], cfg[5][1], cfg[5][2], cfg[5][3], cfg[5][4]),
        ]

    if preset == "xlarge":
        cfg = [
            (3, 2.0, 20, 0.30),
            (4, 1.75, 18, 0.28),
            (6, 1.5, 15, 0.25),
            (8, 1.25, 15, 0.22),
            (10, 1.1, 12, 0.20),
            (10, 1.1, 12, 0.20),
        ]
        return [
            _stage("Bootstrap",        1,   6,  800,  300, 3, "1.2e-3", 0,   *cfg[0]),
            _stage("Warm up",          7,  15, 1200,  400, 3, "9e-4",   0,   *cfg[1]),
            _stage("Early gated",     16,  30, 2000,  600, 3, "6e-4",   200, *cfg[2]),
            _stage("Consolidate",     31,  60, 3000,  600, 4, "4.5e-4", 200, *cfg[3]),
            _stage("Steady improve",  61, 120, 4000,  800, 5, "3e-4",   200, *cfg[4]),
            _stage("Overnight extend",121,200, 5000, 1000, 5, "2e-4",   200, *cfg[5]),
        ]

    total = max(30, min(240, 60 * filters * blocks // 320))
    cut1 = max(3, total * 8 // 100)
    cut2 = total * 17 // 100
    cut3 = total * 29 // 100
    cut4 = total * 46 // 100
    cut5 = total * 67 // 100
    bg = max(100, 400 * filters * blocks // 320)
    cfg = _SMALL_STAGE_CFG
    return [
        _stage("Bootstrap",        1,        cut1, bg,     200, 4, "1.2e-3",   0, *cfg[0]),
        _stage("Warm up",          cut1 + 1, cut2, bg * 2, 300, 4, "9e-4",     0, *cfg[1]),
        _stage("Early gated",      cut2 + 1, cut3, bg * 3, 400, 4, "6e-4",   100, *cfg[2]),
        _stage("Consolidate",      cut3 + 1, cut4, bg * 4, 450, 4, "4.5e-4", 200, *cfg[3]),
        _stage("Steady improve",   cut4 + 1, cut5, bg * 5, 500, 5, "3e-4",   200, *cfg[4]),
        _stage("Overnight extend", cut5 + 1, total, bg * 6, 600, 5, "2e-4",  200, *cfg[5]),
    ]


def generate_plan(filters: int, blocks: int, preset: str) -> dict:
    num_workers = max(1, min(4, num_cores() // 2))
    batch_size = 64 if preset == "quick" else 256
    window_size = 3 if preset == "quick" else 6
    eval_threshold = 0.50 if preset == "quick" else 0.55

    return {
        "model": {
            "rows": BOARD_ROWS,
            "cols": BOARD_COLS,
            "history_length": HISTORY_LENGTH,
            "filters": filters,
            "blocks": blocks,
        },
        "training": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "weight_decay": 1e-4,
            "window_size": window_size,
            "eval_threshold": eval_threshold,
            "policy_weight": 1.0,
            "value_weight": 1.0,
        },
        "mcts": {
            "c_puct": 1.5,
            "dirichlet_alpha": 0.30,
            "dirichlet_epsilon": 0.25,
            "temp_threshold": 18,
            "win_loss_weight": 1.0,
            "score_weight": 0.0,
            "score_scale": 1000.0,
        },
        "stages": generate_stages(preset, filters, blocks),
    }


def detect_gpu_count() -> int:
    """Return number of CUDA GPUs available (0 if CUDA is absent)."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c",
             "import torch, sys; sys.stdout.write(str(torch.cuda.device_count()) if torch.cuda.is_available() else '0')"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        return max(0, int(out.strip() or "0"))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return 0


def detect_hardware() -> dict:
    """Auto-configure hardware knobs from the detected GPU topology.

    Single-process layout (matches the Go multi-gpu branch):
      * 0 GPUs  -> 1 selfplay instance, 1 NN server thread (CPU fallback)
      * 1 GPU   -> 1 instance, 2 NN servers on GPU 0 (0,0)
      * N GPUs  -> 1 instance, 2N NN servers, device_ids = "0,0,1,1,..."

    One selfplay process owns all GPUs.  This keeps the in-process TRT
    engine-build mutex effective — launching multiple selfplay processes
    would race on the trt_cache/ file on a cold start.
    """
    gpus = detect_gpu_count()
    hw = {
        "threads": num_cores(),
        "search_threads": 16,
        "selfplay_instances": 1,
        "max_batch": 256,
        "gpu_count": max(0, gpus),
    }
    if gpus <= 0:
        hw["nn_server_threads"] = 1
        hw["nn_device_ids"] = "0"
    elif gpus == 1:
        hw["nn_server_threads"] = 2
        hw["nn_device_ids"] = "0,0"
    else:
        hw["nn_server_threads"] = gpus * 2
        hw["nn_device_ids"] = ",".join(f"{g},{g}" for g in range(gpus))
    return hw


def build_if_needed() -> None:
    """Configure + build the C++ binaries.  Skips CMake when up-to-date."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    needed_binaries = ("selfplay", "evaluate", "play", "benchmark")
    all_present = all((BUILD_DIR / name).is_file() for name in needed_binaries)
    if not all_present or not (BUILD_DIR / "CMakeCache.txt").is_file():
        subprocess.run(
            [
                "cmake",
                "-S", str(PROJECT_DIR),
                "-B", str(BUILD_DIR),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "-j", str(num_cores())],
        check=True,
    )


def run_and_tee(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        tlog(line.rstrip("\n"))
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def run_selfplay(iter_data: Path, model: Path, games: int, sims: int, hw: dict, mc: dict) -> None:
    mcts_flags = [
        "--c-puct", str(mc["c_puct"]),
        "--dirichlet-alpha", str(mc["dirichlet_alpha"]),
        "--dirichlet-epsilon", str(mc["dirichlet_epsilon"]),
        "--temp-threshold", str(mc["temp_threshold"]),
        "--win-loss-weight", str(mc["win_loss_weight"]),
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
    ] + mcts_flags

    # Single selfplay process owns all GPUs; the in-process TRT build
    # mutex then serializes engine construction across NN server threads.
    subprocess.run(
        base_cmd + [
            "--output", str(iter_data),
            "--games", str(games),
            "--threads", str(hw["threads"]),
        ],
        check=True,
    )


def compress_selfplay(iter_data: Path) -> None:
    bins = sorted(iter_data.rglob("game_*.bin"))
    if not bins:
        return
    try:
        subprocess.run(
            ["zstd", "--rm", "-q", *[str(path) for path in bins]],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass


def count_existing_games(iter_data: Path) -> int:
    if not iter_data.is_dir():
        return 0
    return (
        sum(1 for _ in iter_data.rglob("*.bin")) +
        sum(1 for _ in iter_data.rglob("*.bin.zst")) +
        sum(1 for _ in iter_data.rglob("*.bin.gz"))
    )


def cmd_init(args: argparse.Namespace) -> None:
    preset = args.preset or "custom"
    filters = args.filters
    blocks = args.blocks

    print("============================================")
    print("  MiniXiangqi Training Init")
    print("============================================")
    print(f"  Preset:          {preset}")
    print(f"  Board:           {BOARD_ROWS}x{BOARD_COLS}")
    print(f"  History length:  {HISTORY_LENGTH}")
    print(f"  Network:         {filters} filters x {blocks} blocks")
    print()
    print("This will delete:")
    print("  models/")
    print("  training/")
    print("  trt_cache/   (if present)")
    print()

    if not args.yes:
        confirm = input("Continue? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return

    for path in [MODELS_DIR, TRAINING_DIR, PROJECT_DIR / "trt_cache"]:
        if path.is_dir():
            shutil.rmtree(path)

    for path in [MODELS_DIR, DATA_DIR, EVAL_DIR, LOGS_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    plan = generate_plan(filters, blocks, preset)
    PLAN_FILE.write_text(json.dumps(plan, indent=2) + "\n")

    export_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "export_onnx.py"),
        "--output", str(version_onnx(0)),
        "--rows", str(plan["model"]["rows"]),
        "--cols", str(plan["model"]["cols"]),
        "--history-length", str(plan["model"]["history_length"]),
        "--filters", str(plan["model"]["filters"]),
        "--blocks", str(plan["model"]["blocks"]),
    ]
    subprocess.run(export_cmd, cwd=str(PROJECT_DIR), check=True)
    shutil.copy2(version_onnx(0), best_onnx())

    save_state({
        "pipeline_iter": 0,
        "best_version": 0,
        "total_games": 0,
        "total_promotions": 0,
    })

    total_iters = get_total_iterations(plan)
    total_games = sum(
        stage["games"] * (stage["end"] - stage["start"] + 1)
        for stage in plan["stages"]
    )

    with open(TRAIN_LOG, "w", encoding="utf-8") as f:
        f.write(
            "MiniXiangqi Training Log\n"
            "============================================================\n"
            f"Initialized:     {timestamp()}\n"
            f"Preset:          {preset}\n"
            f"Board:           {BOARD_ROWS}x{BOARD_COLS}\n"
            f"History length:  {HISTORY_LENGTH}\n"
            f"Network:         {filters} filters x {blocks} blocks\n"
            f"Batch size:      {plan['training']['batch_size']}\n"
            f"Window size:     {plan['training']['window_size']}\n"
            f"Eval threshold:  {plan['training']['eval_threshold']}\n"
            f"MCTS:            c_puct={plan['mcts']['c_puct']} "
            f"alpha={plan['mcts']['dirichlet_alpha']} "
            f"eps={plan['mcts']['dirichlet_epsilon']} "
            f"temp={plan['mcts']['temp_threshold']}\n"
            f"Utility blend:   win_loss={plan['mcts']['win_loss_weight']} "
            f"score={plan['mcts']['score_weight']} "
            f"scale={plan['mcts']['score_scale']}\n"
            f"Loss weights:    policy={plan['training']['policy_weight']} "
            f"value={plan['training']['value_weight']}\n"
            "Stages:\n"
        )
        for idx, stage in enumerate(plan["stages"], start=1):
            gate = f"{stage['eval_games']} games" if stage["eval_games"] > 0 else "off"
            f.write(
                f"  {idx}. {stage['name']:16s} "
                f"iter {stage['start']:3d}-{stage['end']:<3d} "
                f"games={stage['games']:4d} sims={stage['sims']:4d} "
                f"epochs={stage['epochs']:2d} lr={stage['lr']} gate={gate}\n"
            )
        f.write(f"Total: {total_iters} iterations, ~{total_games} selfplay games\n")
        f.write("============================================================\n")

    print()
    print("============================================")
    print("  Training Plan")
    print("============================================")
    print(f"  {'Stage':16s} {'Iters':>8s} {'Games':>7s} {'Sims':>6s} {'Epoch':>6s} {'LR':>8s} {'Gate':>7s}")
    print("  " + "-" * 68)
    for stage in plan["stages"]:
        gate = f"{stage['eval_games']}g" if stage["eval_games"] > 0 else "off"
        print(
            f"  {stage['name']:16s} {stage['start']:4d}-{stage['end']:<3d} "
            f"{stage['games']:5d} {stage['sims']:6d} {stage['epochs']:6d} "
            f"{stage['lr']:>8s} {gate:>7s}"
        )
    print("  " + "-" * 68)
    print(f"  Total:            {total_iters} iterations, ~{total_games} games")
    print()
    print(f"  Model:            {best_onnx()}")
    print(f"  Plan:             {PLAN_FILE}")
    print(f"  Log:              {TRAIN_LOG}")
    print("============================================")


def cmd_status(_args: argparse.Namespace) -> None:
    if not PLAN_FILE.is_file():
        print("No training initialized. Run 'python run_loop.py init small' first.")
        return

    plan = read_plan()
    state = read_state()
    total_iters = get_total_iterations(plan)
    pct = state["pipeline_iter"] * 100 // total_iters if total_iters else 0

    model = plan["model"]
    training = plan["training"]
    mcts = plan["mcts"]

    print()
    print("============================================")
    print("  MiniXiangqi Training Status")
    print("============================================")
    print(f"  Board:           {model['rows']}x{model['cols']}")
    print(f"  History length:  {model['history_length']}")
    print(f"  Network:         {model['filters']} filters x {model['blocks']} blocks")
    print(f"  Batch size:      {training['batch_size']}")
    print(f"  Window size:     {training['window_size']}")
    print(f"  Eval threshold:  {training['eval_threshold']}")
    print(f"  MCTS:            c_puct={mcts['c_puct']} alpha={mcts['dirichlet_alpha']} "
          f"eps={mcts['dirichlet_epsilon']} temp={mcts['temp_threshold']}")
    print(f"  Utility blend:   win_loss={mcts['win_loss_weight']} "
          f"score={mcts['score_weight']} scale={mcts['score_scale']}")
    print(f"  Progress:        {state['pipeline_iter']} / {total_iters} iterations ({pct}%)")
    print(f"  Best model:      {vstr(state['best_version'])} ({state['total_promotions']} promotions)")
    print(f"  Total games:     {state['total_games']}")
    print()
    print(f"  {'Stage':16s} {'Iters':>8s} {'Games':>7s} {'Sims':>6s} {'Epoch':>6s} {'LR':>8s} {'Gate':>7s}  Status")
    print("  " + "-" * 80)
    for stage in plan["stages"]:
        gate = f"{stage['eval_games']}g" if stage["eval_games"] > 0 else "off"
        if state["pipeline_iter"] >= stage["end"]:
            stage_status = "DONE"
        elif state["pipeline_iter"] >= stage["start"]:
            stage_status = f"iter {state['pipeline_iter']}"
        else:
            stage_status = ""
        print(
            f"  {stage['name']:16s} {stage['start']:4d}-{stage['end']:<3d} "
            f"{stage['games']:5d} {stage['sims']:6d} {stage['epochs']:6d} "
            f"{stage['lr']:>8s} {gate:>7s}  {stage_status}"
        )
    print("  " + "-" * 80)
    if state["pipeline_iter"] >= total_iters:
        print(f"  Training complete. Final model: {best_onnx()}")
    else:
        print("  Resume with: python run_loop.py train")
    print("============================================")


def cmd_train(args: argparse.Namespace) -> None:
    hw = detect_hardware()
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

    if state["pipeline_iter"] >= get_total_iterations(plan):
        print("Training is already complete.")
        return

    build_if_needed()

    model = plan["model"]
    training = dict(plan["training"])
    mcts = dict(plan["mcts"])

    for key in ("batch_size", "num_workers", "weight_decay", "policy_weight", "value_weight"):
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value
            log(f"CLI override: training.{key} = {value}")

    for key in ("win_loss_weight", "score_weight", "score_scale"):
        value = getattr(args, key, None)
        if value is not None:
            mcts[key] = value
            log(f"CLI override: mcts.{key} = {value}")

    device_ids = [token.strip() for token in hw["nn_device_ids"].split(",") if token.strip()]
    if len(device_ids) != hw["nn_server_threads"]:
        print(
            f"ERROR: --nn-device-ids has {len(device_ids)} entries but "
            f"--nn-server-threads is {hw['nn_server_threads']}."
        )
        sys.exit(1)

    total_iters = get_total_iterations(plan)
    start_iter = state["pipeline_iter"] + 1
    end_iter = total_iters
    if args.iterations and args.iterations > 0:
        end_iter = min(total_iters, start_iter + args.iterations - 1)

    print("============================================")
    print("  MiniXiangqi Training")
    print("============================================")
    print(f"  Board:           {model['rows']}x{model['cols']}")
    print(f"  History length:  {model['history_length']}")
    print(f"  Network:         {model['filters']} filters x {model['blocks']} blocks")
    print(f"  Iterations:      {start_iter} .. {end_iter} (of {total_iters})")
    print(f"  Best model:      {vstr(state['best_version'])}")
    print(f"  Threads:         {hw['threads']}")
    print(f"  Search threads:  {hw['search_threads']}")
    print(f"  Selfplay inst:   {hw['selfplay_instances']}")
    print(f"  NN servers:      {hw['nn_server_threads']}")
    print(f"  NN devices:      {hw['nn_device_ids']}")
    print(f"  Max batch:       {hw['max_batch']}")
    print(f"  Batch size:      {training['batch_size']}")
    print(f"  Num workers:     {training['num_workers']}")
    print(f"  Weight decay:    {training['weight_decay']}")
    print(f"  MCTS:            c_puct={mcts['c_puct']} alpha={mcts['dirichlet_alpha']} "
          f"eps={mcts['dirichlet_epsilon']} temp={mcts['temp_threshold']}")
    print(f"  Utility blend:   win_loss={mcts['win_loss_weight']} "
          f"score={mcts['score_weight']} scale={mcts['score_scale']}")
    print(f"  Loss weights:    policy={training['policy_weight']} value={training['value_weight']}")
    print("============================================")
    print()

    tlog_section(f"TRAINING SESSION iter {start_iter}..{end_iter}")
    tlog(f"Board: {model['rows']}x{model['cols']} history={model['history_length']}")
    tlog(f"Network: {model['filters']} filters x {model['blocks']} blocks")
    tlog(f"Best model: {vstr(state['best_version'])}")
    tlog(f"Threads={hw['threads']} search_threads={hw['search_threads']} selfplay_instances={hw['selfplay_instances']}")
    tlog(f"NN servers={hw['nn_server_threads']} devices={hw['nn_device_ids']} max_batch={hw['max_batch']}")
    tlog(f"Training: batch={training['batch_size']} workers={training['num_workers']} weight_decay={training['weight_decay']}")
    tlog(f"MCTS: c_puct={mcts['c_puct']} alpha={mcts['dirichlet_alpha']} eps={mcts['dirichlet_epsilon']} temp={mcts['temp_threshold']}")
    tlog(f"Utility: win_loss={mcts['win_loss_weight']} score={mcts['score_weight']} scale={mcts['score_scale']}")
    tlog(f"Loss weights: policy={training['policy_weight']} value={training['value_weight']}")

    for iteration in range(start_iter, end_iter + 1):
        state["pipeline_iter"] = iteration
        stage = get_stage_for_iter(plan, iteration)
        st, smc = get_stage_config(stage, {"training": training, "mcts": mcts})

        log("============================================================")
        log(
            f"ITERATION {iteration}/{total_iters} | Stage: {stage['name']} | "
            f"Best: {vstr(state['best_version'])} | LR: {stage['lr']}"
        )
        log("============================================================")
        tlog()
        tlog(f"ITERATION {iteration}/{total_iters} stage={stage['name']}")
        tlog(
            f"  games={stage['games']} sims={stage['sims']} epochs={stage['epochs']} "
            f"lr={stage['lr']} eval_games={stage['eval_games']}"
        )

        iter_data = DATA_DIR / f"iter_{iteration:04d}"
        iter_data.mkdir(parents=True, exist_ok=True)
        existing_games = count_existing_games(iter_data)

        if existing_games >= stage["games"]:
            log(f"Phase 1 - Selfplay: skip ({existing_games} games already exist)")
            tlog(f"Phase 1 selfplay: skip ({existing_games} existing)")
        else:
            games_needed = stage["games"] - existing_games
            selfplay_model = version_onnx(state["best_version"])
            if not selfplay_model.is_file():
                print(f"ERROR: missing selfplay model {selfplay_model}")
                sys.exit(1)

            log(
                f"Phase 1 - Selfplay: {games_needed} games, "
                f"{stage['sims']} sims/move using {vstr(state['best_version'])}"
            )
            t0 = time.time()
            run_selfplay(iter_data, selfplay_model, games_needed, stage["sims"], hw, smc)
            compress_selfplay(iter_data)
            elapsed = int(time.time() - t0)
            state["total_games"] += games_needed
            per_game = elapsed / games_needed if games_needed else 0.0
            log(f"Phase 1 - Selfplay done ({elapsed}s, {per_game:.2f}s/game)")
            tlog(f"Phase 1 selfplay done: {elapsed}s total_games={state['total_games']}")

        candidate_onnx = version_onnx(iteration)
        candidate_ckpt = version_checkpoint(iteration)
        if candidate_onnx.is_file() and candidate_ckpt.is_file():
            log(f"Phase 2 - Training: skip ({vstr(iteration)} already exists)")
            tlog(f"Phase 2 training: skip ({vstr(iteration)} exists)")
        else:
            window_dirs = build_data_window(st["window_size"], iteration)
            if not window_dirs:
                print("ERROR: no selfplay data available in the training window")
                sys.exit(1)

            train_ckpt = CHECKPOINT_DIR / "training.pt"
            train_py_args = [
                str(SCRIPTS_DIR / "train.py"),
                "--data", window_dirs,
                "--checkpoint", str(train_ckpt),
                "--epochs", str(stage["epochs"]),
                "--batch-size", str(st["batch_size"]),
                "--lr", str(stage["lr"]),
                "--weight-decay", str(st["weight_decay"]),
                "--rows", str(model["rows"]),
                "--cols", str(model["cols"]),
                "--history-length", str(model["history_length"]),
                "--filters", str(model["filters"]),
                "--blocks", str(model["blocks"]),
                "--num-workers", str(st["num_workers"]),
                "--policy-weight", str(st["policy_weight"]),
                "--value-weight", str(st["value_weight"]),
                "--output-onnx", str(candidate_onnx),
            ]

            gpu_count = hw.get("gpu_count", 0)
            if gpu_count > 1:
                torchrun = shutil.which("torchrun")
                if torchrun:
                    train_cmd = [
                        torchrun,
                        f"--nproc_per_node={gpu_count}",
                        *train_py_args,
                    ]
                    log(f"  training via torchrun with {gpu_count} GPUs")
                else:
                    log("  torchrun not found — falling back to single-process training")
                    train_cmd = [sys.executable, *train_py_args]
            else:
                train_cmd = [sys.executable, *train_py_args]

            log(
                f"Phase 2 - Training: epochs={stage['epochs']} "
                f"lr={stage['lr']} batch={st['batch_size']} "
                f"window={st['window_size']}"
            )
            t0 = time.time()
            run_and_tee(train_cmd, cwd=PROJECT_DIR)
            elapsed = int(time.time() - t0)

            if not candidate_onnx.is_file():
                print(f"ERROR: training did not produce {candidate_onnx}")
                sys.exit(1)
            if train_ckpt.is_file():
                shutil.copy2(train_ckpt, candidate_ckpt)

            log(f"Phase 2 - Training done ({elapsed}s). Candidate: {vstr(iteration)}")
            tlog(f"Phase 2 training done: {elapsed}s")

        if stage["eval_games"] > 0 and state["best_version"] > 0:
            baseline_onnx = version_onnx(state["best_version"])
            eval_dir = EVAL_DIR / f"iter_{iteration:04d}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_threshold = st.get("eval_threshold", training["eval_threshold"])
            eval_cmd = [
                str(BUILD_DIR / "evaluate"),
                "--model1", str(candidate_onnx),
                "--model2", str(baseline_onnx),
                "--games", str(stage["eval_games"]),
                "--threads", str(hw["threads"]),
                "--search-threads", str(hw["search_threads"]),
                "--max-batch", str(hw["max_batch"]),
                "--nn-server-threads", str(hw["nn_server_threads"]),
                "--nn-device-ids", hw["nn_device_ids"],
                "--sims", str(stage["sims"]),
                "--c-puct", str(smc["c_puct"]),
                "--win-loss-weight", str(smc["win_loss_weight"]),
                "--score-weight", str(smc["score_weight"]),
                "--score-scale", str(smc["score_scale"]),
                "--threshold", str(eval_threshold),
                "--output", str(eval_dir),
            ]

            log(
                f"Phase 3 - Eval: {vstr(iteration)} vs {vstr(state['best_version'])} "
                f"({stage['eval_games']} games)"
            )
            result = subprocess.run(
                eval_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            eval_output = result.stdout
            print(eval_output, end="")
            for line in eval_output.splitlines():
                tlog(line)

            if result.returncode not in (0, 1):
                # Eval binary crashed.  Rather than deadlocking the pipeline we
                # auto-promote the candidate and continue - same policy as the
                # Go multi-gpu branch.  The next eval will catch a bad model.
                log(
                    f"Phase 3 - WARNING: evaluate exited with code "
                    f"{result.returncode}; auto-promoting candidate to keep the loop alive"
                )
                tlog(f"Phase 3 eval: crashed (rc={result.returncode}); auto-promoted")
                shutil.copy2(candidate_onnx, best_onnx())
                state["best_version"] = iteration
                state["total_promotions"] += 1
                save_state(state)
                log(
                    f"Iteration {iteration} done (eval crashed). "
                    f"Best={vstr(state['best_version'])} "
                    f"Promotions={state['total_promotions']} "
                    f"Total games={state['total_games']}"
                )
                continue

            win_rate_match = re.search(r"Model 1 win rate:\s*([\d.]+)%", eval_output)
            wins1_match = re.search(r"Model 1 wins:\s*(\d+)", eval_output)
            wins2_match = re.search(r"Model 2 wins:\s*(\d+)", eval_output)
            draws_match = re.search(r"Draws:\s*(\d+)", eval_output)
            win_rate = win_rate_match.group(1) + "%" if win_rate_match else "?"
            record = (
                wins1_match.group(1) if wins1_match else "?",
                wins2_match.group(1) if wins2_match else "?",
                draws_match.group(1) if draws_match else "?",
            )

            if result.returncode == 0:
                shutil.copy2(candidate_onnx, best_onnx())
                state["best_version"] = iteration
                state["total_promotions"] += 1
                log(f"Phase 3 - PROMOTED {vstr(iteration)} ({record[0]}-{record[1]}-{record[2]}, {win_rate})")
                tlog(f"Phase 3 eval: promoted ({record[0]}-{record[1]}-{record[2]}, {win_rate})")
            else:
                log(f"Phase 3 - REJECTED {vstr(iteration)} ({record[0]}-{record[1]}-{record[2]}, {win_rate})")
                tlog(f"Phase 3 eval: rejected ({record[0]}-{record[1]}-{record[2]}, {win_rate})")
        else:
            shutil.copy2(candidate_onnx, best_onnx())
            state["best_version"] = iteration
            state["total_promotions"] += 1
            log(f"Phase 3 - Auto-promote {vstr(iteration)}")
            tlog("Phase 3 eval: auto-promote")

        save_state(state)
        log(
            f"Iteration {iteration} done. Best={vstr(state['best_version'])} "
            f"Promotions={state['total_promotions']} Total games={state['total_games']}"
        )
        tlog(
            f"Iteration {iteration} result: best={vstr(state['best_version'])} "
            f"promotions={state['total_promotions']} total_games={state['total_games']}"
        )

    print()
    print("============================================")
    if end_iter >= total_iters:
        print("  Training Complete")
    else:
        print(f"  Session Complete (paused at iter {end_iter})")
        print("  Resume with: python run_loop.py train")
    print(f"  Best model:      {vstr(state['best_version'])}")
    print(f"  Total games:     {state['total_games']}")
    print(f"  Promotions:      {state['total_promotions']}")
    print(f"  Model file:      {best_onnx()}")
    print("============================================")
    tlog_section("SESSION END")
    tlog(
        f"Best={vstr(state['best_version'])} "
        f"games={state['total_games']} promotions={state['total_promotions']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniXiangqi training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_loop.py init small -y
  python run_loop.py train
  python run_loop.py train --iterations 5
  python run_loop.py train --threads 8 --search-threads 8
  python run_loop.py status
""",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Initialize training state and create v0000")
    p_init.add_argument(
        "preset",
        nargs="?",
        choices=["quick", "small", "large", "xlarge"],
        help="Training preset. Omit for a custom Xiangqi run using explicit filters/blocks.",
    )
    p_init.add_argument("--filters", type=int, default=None, help="Residual tower channels")
    p_init.add_argument("--blocks", type=int, default=None, help="Residual block count")
    p_init.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    p_train = sub.add_parser("train", help="Start or resume training")
    p_train.add_argument("--threads", type=int, default=None, help="Parallel selfplay workers")
    p_train.add_argument("--search-threads", type=int, default=None, help="MCTS search threads per move")
    p_train.add_argument("--selfplay-instances", type=int, default=None, help="Parallel selfplay processes")
    p_train.add_argument("--nn-server-threads", type=int, default=None, help="NN server threads")
    p_train.add_argument("--nn-device-ids", type=str, default=None, help="Comma-separated device ids")
    p_train.add_argument("--max-batch", type=int, default=None, help="Max NN batch size")
    p_train.add_argument("--iterations", type=int, default=None, help="Max iterations to run this session")
    p_train.add_argument("--batch-size", type=int, default=None, help="Override SGD batch size")
    p_train.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count")
    p_train.add_argument("--weight-decay", type=float, default=None, help="Override Adam weight decay")
    p_train.add_argument("--policy-weight", type=float, default=None, help="Override policy loss weight")
    p_train.add_argument("--value-weight", type=float, default=None, help="Override value loss weight")
    p_train.add_argument("--win-loss-weight", type=float, default=None, help="Override MCTS win/loss blend")
    p_train.add_argument("--score-weight", type=float, default=None, help="Override MCTS score blend")
    p_train.add_argument("--score-scale", type=float, default=None, help="Override MCTS score scale")

    sub.add_parser("status", help="Show training progress")

    args = parser.parse_args()

    if args.command == "init":
        if args.preset == "quick":
            args.filters = args.filters or 32
            args.blocks = args.blocks or 3
        elif args.preset == "small":
            args.filters = args.filters or 64
            args.blocks = args.blocks or 5
        elif args.preset in ("large", "xlarge"):
            args.filters = args.filters or 128
            args.blocks = args.blocks or 10
        else:
            args.filters = args.filters or 128
            args.blocks = args.blocks or 10
        cmd_init(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
