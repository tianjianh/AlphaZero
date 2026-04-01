#!/bin/bash
#
# MiniGo AlphaZero — Full Training Loop
#
# This script automates the hybrid C++/Python training cycle:
#   1. Export current model to ONNX
#   2. Run C++ self-play to generate training data
#   3. Train model in Python/PyTorch
#   4. ONNX model is auto-exported by train.py
#   5. Repeat
#
# Training is resumable: stop at any time and restart to continue.
# Self-play data accumulates across iterations (not deleted).
#
# Multi-instance selfplay:
#   --selfplay-instances 2  runs two selfplay processes in parallel,
#   each generating half the games.  Both write to the same output dir.
#   This maximizes GPU utilization when the GPU can handle more concurrent
#   search threads than a single process provides.
#
# Usage:
#   ./run_loop.sh                          # default settings (OpenCL GPU, all cores)
#   ./run_loop.sh --iterations 50          # custom iteration count
#   ./run_loop.sh --selfplay-instances 2   # 2 parallel selfplay processes
#   ./run_loop.sh --quick                  # fast test mode
#   ./run_loop.sh --backend eigen          # CPU-only fallback

set -e

# ── Cross-platform helpers ──────────────────────────────────
num_cores() {
    if command -v nproc &>/dev/null; then
        nproc
    elif command -v sysctl &>/dev/null; then
        sysctl -n hw.ncpu
    else
        echo 4
    fi
}

# Prefer python3, fall back to python
PYTHON=python3
if ! command -v python3 &>/dev/null; then
    PYTHON=python
fi

# ── Defaults ────────────────────────────────────────────────
ITERATIONS=30
GAMES_PER_ITER=100
SIMS=400
BOARD=9
EPOCHS=15
BATCH_SIZE=256
THREADS=$(num_cores)
FILTERS=64
BLOCKS=5
SEARCH_THREADS=16
SELFPLAY_INSTANCES=1

# ── Parse args ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) ITERATIONS=$2; shift 2;;
        --games) GAMES_PER_ITER=$2; shift 2;;
        --sims) SIMS=$2; shift 2;;
        --board) BOARD=$2; shift 2;;
        --epochs) EPOCHS=$2; shift 2;;
        --batch) BATCH_SIZE=$2; shift 2;;
        --threads) THREADS=$2; shift 2;;
        --search-threads) SEARCH_THREADS=$2; shift 2;;
        --selfplay-instances) SELFPLAY_INSTANCES=$2; shift 2;;
        --quick)
            ITERATIONS=3; GAMES_PER_ITER=10; SIMS=100; BOARD=5
            EPOCHS=5; BATCH_SIZE=64; THREADS=$(num_cores); FILTERS=32; BLOCKS=3
            echo "=== QUICK TEST MODE ==="
            shift;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--iterations N] [--games N] [--sims N] [--board N]"
            echo "          [--epochs N] [--batch N] [--threads N]"
            echo "          [--search-threads N] [--selfplay-instances N] [--quick]"
            exit 1;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
CHECKPOINT="checkpoints/best_model.pt"
MODEL_ONNX="model.onnx"

echo "============================================"
echo "  MiniGo AlphaZero Training Loop"
echo "============================================"
echo "  Board:              ${BOARD}x${BOARD}"
echo "  Filters:            ${FILTERS}"
echo "  ResBlocks:          ${BLOCKS}"
echo "  Iterations:         ${ITERATIONS}"
echo "  Games/iter:         ${GAMES_PER_ITER}"
echo "  MCTS sims:          ${SIMS}"
echo "  Search threads:     ${SEARCH_THREADS}"
echo "  Worker threads:     ${THREADS}"
echo "  Selfplay instances: ${SELFPLAY_INSTANCES}"
echo "  Epochs:             ${EPOCHS}"
echo "============================================"
echo

# Build C++ if needed
if [ ! -f "${BUILD_DIR}/selfplay" ]; then
    echo "Building C++ components..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(num_cores)
    cd "${PROJECT_DIR}"
    echo
fi

mkdir -p checkpoints

# Initial ONNX export (random if no checkpoint exists)
if [ ! -f "${PROJECT_DIR}/${MODEL_ONNX}" ]; then
    echo "Step 0: Exporting initial ONNX model..."
    cd "${PROJECT_DIR}/scripts"
    $PYTHON export_onnx.py \
        --checkpoint "${PROJECT_DIR}/${CHECKPOINT}" \
        --output "${PROJECT_DIR}/${MODEL_ONNX}" \
        --board ${BOARD} --filters ${FILTERS} --blocks ${BLOCKS} \
        --init
    cd "${PROJECT_DIR}"
    echo
fi

# ── Training loop ───────────────────────────────────────────
for iter in $(seq 1 ${ITERATIONS}); do
    echo "============================================"
    echo "  ITERATION ${iter}/${ITERATIONS}"
    echo "============================================"

    # Create per-iteration data directory (data accumulates)
    ITER_DATA="selfplay_data/iter_$(printf '%04d' ${iter})"
    mkdir -p "${ITER_DATA}"

    # ── 1. Self-play in C++ ─────────────────────────────────
    echo
    echo ">> Self-play: ${GAMES_PER_ITER} games (${SELFPLAY_INSTANCES} instance(s))..."

    if [ "${SELFPLAY_INSTANCES}" -le 1 ]; then
        # Single instance — simple case
        time "${BUILD_DIR}/selfplay" \
            --model "${MODEL_ONNX}" \
            --games ${GAMES_PER_ITER} \
            --threads ${THREADS} \
            --search-threads ${SEARCH_THREADS} \
            --output "${ITER_DATA}" \
            --sims ${SIMS}
    else
        # Multi-instance: split games and threads across N processes
        GAMES_PER_INSTANCE=$(( (GAMES_PER_ITER + SELFPLAY_INSTANCES - 1) / SELFPLAY_INSTANCES ))
        THREADS_PER_INSTANCE=$(( (THREADS + SELFPLAY_INSTANCES - 1) / SELFPLAY_INSTANCES ))

        echo "   (${GAMES_PER_INSTANCE} games × ${THREADS_PER_INSTANCE} threads per instance)"

        pids=()
        for i in $(seq 1 ${SELFPLAY_INSTANCES}); do
            "${BUILD_DIR}/selfplay" \
                --model "${MODEL_ONNX}" \
                --games ${GAMES_PER_INSTANCE} \
                --threads ${THREADS_PER_INSTANCE} \
                --search-threads ${SEARCH_THREADS} \
                --output "${ITER_DATA}" \
                --sims ${SIMS} &
            pids+=($!)
        done

        # Wait for all instances to finish
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                echo "WARNING: selfplay instance (pid $pid) failed"
                failed=1
            fi
        done
        if [ $failed -ne 0 ]; then
            echo "Some selfplay instances failed — continuing with available data"
        fi
    fi

    # ── 2. Train in Python ──────────────────────────────────
    echo
    echo ">> Training: ${EPOCHS} epochs..."
    cd "${PROJECT_DIR}/scripts"
    time $PYTHON train.py \
        --data "${PROJECT_DIR}/selfplay_data" \
        --checkpoint "${PROJECT_DIR}/${CHECKPOINT}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --board ${BOARD} \
        --filters ${FILTERS} \
        --blocks ${BLOCKS} \
        --output-onnx "${PROJECT_DIR}/${MODEL_ONNX}"
    cd "${PROJECT_DIR}"

    # ── 3. Backup checkpoint ────────────────────────────────
    cp "${CHECKPOINT}" "checkpoints/model_iter_$(printf '%04d' ${iter}).pt"

    echo
    echo "Iteration ${iter} complete!"
    echo
done

echo "============================================"
echo "  Training complete!"
echo "  Final model: ${CHECKPOINT}"
echo "  ONNX model:  ${MODEL_ONNX}"
echo "============================================"
