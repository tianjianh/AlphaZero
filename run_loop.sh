#!/bin/bash
#
# MiniGo AlphaZero — Training Pipeline
#
# Two commands:
#   init   — choose network size, generate training plan, start fresh
#   train  — run (or resume) training, pass hardware params
#
# Usage:
#   ./run_loop.sh init small                     # 9x9, 64f/5b, ~60 iters
#   ./run_loop.sh init large                     # 9x9, 128f/10b, ~200 iters
#   ./run_loop.sh init quick                     # 5x5, 32f/3b, 5 iters (test)
#   ./run_loop.sh init --board 9 --filters 96 --blocks 8   # custom arch
#
#   ./run_loop.sh train                          # auto-detect GPUs, all cores
#   ./run_loop.sh train --threads 64 --nn-device-ids 0,0,1,1
#   ./run_loop.sh train --iterations 20          # run at most 20 more iters
#
#   ./run_loop.sh status                         # show progress

set -eo pipefail

# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

num_cores() {
    if command -v nproc &>/dev/null; then nproc
    elif command -v sysctl &>/dev/null; then sysctl -n hw.ncpu
    else echo 4; fi
}

# ── Python / conda ─────────────────────────────────────────
activate_conda() {
    local env="${MINIGO_CONDA_ENV:-alphazero}"
    local conda_bin=""
    for p in /opt/miniforge3/bin/conda /opt/conda/bin/conda \
             ~/miniconda3/bin/conda ~/anaconda3/bin/conda; do
        [ -x "$p" ] && { conda_bin="$p"; break; }
    done
    if [ -n "$conda_bin" ]; then
        eval "$($conda_bin shell.bash hook)" 2>/dev/null
        conda activate "$env" 2>/dev/null && return 0
    fi
    return 1
}

activate_conda 2>/dev/null || true

PYTHON=python3
command -v python3 &>/dev/null || PYTHON=python

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
MODELS_DIR="${PROJECT_DIR}/models"
DATA_DIR="${PROJECT_DIR}/training/selfplay"
LOGS_DIR="${PROJECT_DIR}/training/logs"
CHECKPOINT_DIR="${PROJECT_DIR}/training/checkpoints"
STATE_FILE="${PROJECT_DIR}/training/state"
PLAN_FILE="${PROJECT_DIR}/training_plan"
TRAIN_LOG="${PROJECT_DIR}/training/logs/train.log"

# ── Logging ────────────────────────────────────────────────
log() {
    local msg="[$(timestamp)] $*"
    echo "$msg"
    [ -d "${LOGS_DIR}" ] && echo "$msg" >> "${LOGS_DIR}/pipeline.log"
}

# Detailed log — written to train.log for post-hoc analysis
tlog() {
    [ -f "${TRAIN_LOG}" ] && echo "$*" >> "${TRAIN_LOG}"
}

tlog_section() {
    tlog ""
    tlog "════════════════════════════════════════════════════════════════"
    tlog "  $*"
    tlog "  $(timestamp)"
    tlog "════════════════════════════════════════════════════════════════"
}

# ── Pipeline state ─────────────────────────────────────────
read_state() {
    PIPELINE_ITER=0; BEST_VERSION=0; TOTAL_GAMES=0; TOTAL_PROMOTIONS=0
    [ -f "$STATE_FILE" ] && source "$STATE_FILE"
}

save_state() {
    cat > "$STATE_FILE" << EOF
PIPELINE_ITER=$PIPELINE_ITER
BEST_VERSION=$BEST_VERSION
TOTAL_GAMES=$TOTAL_GAMES
TOTAL_PROMOTIONS=$TOTAL_PROMOTIONS
EOF
}

# ── Plan helpers ───────────────────────────────────────────
read_plan() {
    if [ ! -f "$PLAN_FILE" ]; then
        echo "ERROR: No training plan found."
        echo "Run './run_loop.sh init small' (or large/quick) first."
        exit 1
    fi
    source "$PLAN_FILE"
}

# Get the stage parameters for a given iteration.
# Sets: STAGE_GAMES STAGE_SIMS STAGE_EPOCHS STAGE_LR STAGE_EVAL_GAMES STAGE_NAME STAGE_IDX
get_stage_for_iter() {
    local iter=$1
    for s in $(seq 1 $PLAN_NUM_STAGES); do
        local stage_var="PLAN_STAGE_${s}"
        local name_var="PLAN_STAGE_NAME_${s}"
        read -r s_start s_end s_games s_sims s_epochs s_lr s_eval <<< "${!stage_var}"
        if [ "$iter" -ge "$s_start" ] && [ "$iter" -le "$s_end" ]; then
            STAGE_GAMES=$s_games; STAGE_SIMS=$s_sims; STAGE_EPOCHS=$s_epochs
            STAGE_LR=$s_lr; STAGE_EVAL_GAMES=$s_eval
            STAGE_NAME="${!name_var}"; STAGE_IDX=$s
            return 0
        fi
    done
    echo "ERROR: iteration $iter is outside all stages in training_plan"
    exit 1
}

get_total_iterations() {
    local stage_var="PLAN_STAGE_${PLAN_NUM_STAGES}"
    read -r _ s_end _ <<< "${!stage_var}"
    echo "$s_end"
}

# ── Version helpers ────────────────────────────────────────
version_onnx()       { printf "${MODELS_DIR}/v%04d.onnx" "$1"; }
version_checkpoint() { printf "${CHECKPOINT_DIR}/v%04d.pt" "$1"; }
best_onnx()          { echo "${MODELS_DIR}/best.onnx"; }

# ── Data window ────────────────────────────────────────────
build_data_window() {
    local end=$1
    local start=$(( end - PLAN_WINDOW_SIZE + 1 ))
    [ $start -lt 1 ] && start=1
    local dirs=""
    for w in $(seq $start $end); do
        local d="${DATA_DIR}/iter_$(printf '%04d' $w)"
        if [ -d "$d" ]; then
            [ -n "$dirs" ] && dirs="${dirs},"
            dirs="${dirs}${d}"
        fi
    done
    echo "$dirs"
}


# ── Hardware auto-detection ────────────────────────────────
detect_hardware() {
    THREADS=$(num_cores)
    SEARCH_THREADS=16
    SELFPLAY_INSTANCES=1

    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$gpu_count" -ge 2 ]; then
            # Multi-GPU: 2 server threads per GPU (pipelining)
            NN_SERVER_THREADS=$(( gpu_count * 2 ))
            NN_DEVICE_IDS=$(
                for g in $(seq 0 $((gpu_count - 1))); do printf "%d,%d," "$g" "$g"; done \
                | sed 's/,$//'
            )
        elif [ "$gpu_count" -eq 1 ]; then
            NN_SERVER_THREADS=2
            NN_DEVICE_IDS="0,0"
        else
            NN_SERVER_THREADS=1
            NN_DEVICE_IDS="0"
        fi
    else
        NN_SERVER_THREADS=1
        NN_DEVICE_IDS="0"
    fi
}

# ── Self-play ──────────────────────────────────────────────
run_selfplay() {
    local iter_data="$1" model="$2" games_needed="$3" sims="$4"

    if [ "${SELFPLAY_INSTANCES}" -le 1 ]; then
        "${BUILD_DIR}/selfplay" \
            --model "${model}" \
            --games ${games_needed} \
            --threads ${THREADS} \
            --search-threads ${SEARCH_THREADS} \
            --max-batch ${MAX_BATCH} \
            --nn-server-threads ${NN_SERVER_THREADS} \
            --nn-device-ids ${NN_DEVICE_IDS} \
            --output "${iter_data}" \
            --sims ${sims}
    else
        local gpii=$(( (games_needed + SELFPLAY_INSTANCES - 1) / SELFPLAY_INSTANCES ))
        local tpii=$(( (THREADS + SELFPLAY_INSTANCES - 1) / SELFPLAY_INSTANCES ))
        log "  (${gpii} games x ${tpii} threads x ${SELFPLAY_INSTANCES} instances)"

        local pids=()
        for i in $(seq 1 ${SELFPLAY_INSTANCES}); do
            "${BUILD_DIR}/selfplay" \
                --model "${model}" \
                --games ${gpii} \
                --threads ${tpii} \
                --search-threads ${SEARCH_THREADS} \
                --max-batch ${MAX_BATCH} \
                --nn-server-threads ${NN_SERVER_THREADS} \
                --nn-device-ids ${NN_DEVICE_IDS} \
                --output "${iter_data}" \
                --sims ${sims} &
            pids+=($!)
        done
        local failed=0
        for pid in "${pids[@]}"; do
            wait $pid || { log "WARNING: selfplay pid $pid failed"; failed=1; }
        done
        [ $failed -ne 0 ] && log "WARNING: some selfplay instances failed"
    fi
}

# ── Build ──────────────────────────────────────────────────
build_if_needed() {
    if [ ! -f "${BUILD_DIR}/selfplay" ] || [ ! -f "${BUILD_DIR}/evaluate" ]; then
        log "Building C++ components..."
        mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(num_cores)
        cd "${PROJECT_DIR}"
    fi
}

# ══════════════════════════════════════════════════════════
#  INIT
# ══════════════════════════════════════════════════════════

generate_plan_stages() {
    local preset=$1 board=$2 filters=$3 blocks=$4
    case "$preset" in
        quick)
            cat << 'EOF'
PLAN_NUM_STAGES=1
PLAN_STAGE_NAME_1="Quick test"
PLAN_STAGE_1="1 5 20 100 5 2e-3 0"
EOF
            ;;
        small)
            cat << 'EOF'
PLAN_NUM_STAGES=4
PLAN_STAGE_NAME_1="Warm up"
PLAN_STAGE_1="1 5 500 400 10 2e-3 0"
PLAN_STAGE_NAME_2="Explore"
PLAN_STAGE_2="6 25 1500 600 15 1e-3 100"
PLAN_STAGE_NAME_3="Strengthen"
PLAN_STAGE_3="26 60 2500 600 15 5e-4 100"
PLAN_STAGE_NAME_4="Polish"
PLAN_STAGE_4="61 100 3000 800 20 1e-4 100"
EOF
            ;;
        large)
            cat << 'EOF'
PLAN_NUM_STAGES=4
PLAN_STAGE_NAME_1="Warm up"
PLAN_STAGE_1="1 10 1000 600 10 2e-3 0"
PLAN_STAGE_NAME_2="Explore"
PLAN_STAGE_2="11 50 3000 800 15 1e-3 200"
PLAN_STAGE_NAME_3="Strengthen"
PLAN_STAGE_3="51 130 4000 1000 20 5e-4 200"
PLAN_STAGE_NAME_4="Master"
PLAN_STAGE_4="131 200 5000 1200 20 1e-4 200"
EOF
            ;;
        custom)
            local total=$(( 60 * filters * blocks / 320 ))
            [ $total -lt 30 ] && total=30
            [ $total -gt 300 ] && total=300

            local s1_end=$(( total * 8 / 100 ))
            local s2_end=$(( total * 25 / 100 ))
            local s3_end=$(( total * 58 / 100 ))
            local s4_end=$total
            [ $s1_end -lt 3 ] && s1_end=3

            local s2_start=$(( s1_end + 1 ))
            local s3_start=$(( s2_end + 1 ))
            local s4_start=$(( s3_end + 1 ))

            # Base games scale with board complexity
            local bg=$(( 500 * board * board / 81 ))
            [ $bg -lt 50 ] && bg=50

            cat << EOF
PLAN_NUM_STAGES=4
PLAN_STAGE_NAME_1="Warm up"
PLAN_STAGE_1="1 ${s1_end} ${bg} 400 10 2e-3 0"
PLAN_STAGE_NAME_2="Explore"
PLAN_STAGE_2="${s2_start} ${s2_end} $((bg * 3)) 600 15 1e-3 100"
PLAN_STAGE_NAME_3="Strengthen"
PLAN_STAGE_3="${s3_start} ${s3_end} $((bg * 5)) 600 15 5e-4 100"
PLAN_STAGE_NAME_4="Polish"
PLAN_STAGE_4="${s4_start} ${s4_end} $((bg * 6)) 800 20 1e-4 100"
EOF
            ;;
    esac
}

generate_plan() {
    local board=$1 filters=$2 blocks=$3 preset=$4

    local batch_size=1024 eval_threshold="0.55" window=20

    case "$preset" in
        quick) batch_size=64; window=5; eval_threshold="0.5";;
        large) window=30;;
    esac

    cat > "$PLAN_FILE" << EOF
# MiniGo Training Plan
# Generated: $(timestamp)
# Preset: ${preset}
#
# ── Architecture ──────────────────────────────────────────
# Board size for the Go game (9 = 9x9 board)
PLAN_BOARD=${board}
# Number of convolutional filters — controls network width
PLAN_FILTERS=${filters}
# Number of residual blocks — controls network depth
PLAN_BLOCKS=${blocks}
# Mini-batch size for gradient descent. Larger = faster epochs,
# more stable gradients. Scale with dataset size (aim for 100+ batches/epoch).
PLAN_BATCH_SIZE=${batch_size}

# ── Training Parameters ──────────────────────────────────
# Candidate model must win >= this fraction of evaluation games
# to be promoted over the current best model.
PLAN_EVAL_THRESHOLD=${eval_threshold}
# Sliding window: number of recent selfplay iterations whose data is
# loaded for training. Keeps training focused on games from models of
# similar strength while retaining enough history for diversity.
# This is the standard AlphaZero approach (AlphaGo Zero used the last
# 500K games; KataGo uses a similar sliding window).
# Data is streamed from disk via memory-mapped I/O — no memory limit.
PLAN_WINDOW_SIZE=${window}

# ── Training Stages ──────────────────────────────────────
# Each stage defines parameters for a range of iterations.
# You can edit these to tune the training schedule.
#
# Format:  "start_iter  end_iter  games  sims  epochs  lr  eval_games"
#
#   start_iter, end_iter  — iteration range for this stage (inclusive)
#   games                 — selfplay games generated per iteration
#   sims                  — MCTS simulations per move during selfplay
#   epochs                — training epochs per iteration
#   lr                    — Adam optimizer learning rate
#   eval_games            — games played to gate candidate vs best model
#                           0 = skip gating, auto-promote every iteration
#
EOF

    generate_plan_stages "$preset" "$board" "$filters" "$blocks" >> "$PLAN_FILE"
}

cmd_init() {
    local board=9 filters=64 blocks=5 preset="" yes_flag=0

    case "${1:-}" in
        quick) preset=quick; board=5; filters=32; blocks=3; shift;;
        small) preset=small; board=9; filters=64; blocks=5; shift;;
        large) preset=large; board=9; filters=128; blocks=10; shift;;
    esac

    while [[ $# -gt 0 ]]; do
        case $1 in
            --board)   board=$2; shift 2;;
            --filters) filters=$2; shift 2;;
            --blocks)  blocks=$2; shift 2;;
            -y|--yes)  yes_flag=1; shift;;
            *) echo "Unknown init option: $1"; exit 1;;
        esac
    done

    [ -z "$preset" ] && preset=custom

    echo "============================================"
    echo "  MiniGo Training — Init"
    echo "============================================"
    echo "  Preset:     ${preset}"
    echo "  Board:      ${board}x${board}"
    echo "  Network:    ${filters}f x ${blocks}b"
    echo

    # Confirm before clearing
    echo "This will DELETE all existing training data:"
    echo "  models/              (ONNX model files)"
    echo "  training/            (selfplay data, checkpoints, logs)"
    echo "  trt_cache/           (TensorRT engine cache)"
    echo "  training_plan        (training schedule)"
    echo

    if [ $yes_flag -eq 0 ]; then
        read -p "Continue? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
    fi

    echo "Clearing..."
    rm -rf "${MODELS_DIR}" "${PROJECT_DIR}/training" "${PROJECT_DIR}/trt_cache"
    rm -f "${PLAN_FILE}" "${PROJECT_DIR}/pipeline_config"

    # Create fresh directories
    mkdir -p "${MODELS_DIR}" "${DATA_DIR}" "${LOGS_DIR}" "${CHECKPOINT_DIR}"

    # Generate training plan
    echo "Generating training plan..."
    generate_plan "$board" "$filters" "$blocks" "$preset"

    # Create initial random model (v0000) and set as best
    echo "Creating initial model (v0000)..."
    cd "${PROJECT_DIR}/scripts"
    $PYTHON export_onnx.py \
        --output "$(version_onnx 0)" \
        --board ${board} --filters ${filters} --blocks ${blocks} \
        --init
    cd "${PROJECT_DIR}"
    cp "$(version_onnx 0)" "$(best_onnx)"

    # Initialize state
    PIPELINE_ITER=0; BEST_VERSION=0; TOTAL_GAMES=0; TOTAL_PROMOTIONS=0
    save_state

    # Initialize train.log with full header
    source "$PLAN_FILE"
    local total_iters
    total_iters=$(get_total_iterations)
    local total_games=0

    cat > "${TRAIN_LOG}" << EOF
MiniGo AlphaZero — Training Log
════════════════════════════════════════════════════════════════
Initialized:  $(timestamp)
Preset:       ${preset}
Architecture: ${board}x${board} board, ${filters} filters, ${blocks} blocks
Batch size:   ${PLAN_BATCH_SIZE}
Window size:  ${PLAN_WINDOW_SIZE} iterations (data streamed via mmap, no memory limit)
Eval gate:    ${PLAN_EVAL_THRESHOLD} win rate threshold

Training Plan:
EOF

    for s in $(seq 1 $PLAN_NUM_STAGES); do
        local sv="PLAN_STAGE_${s}" nv="PLAN_STAGE_NAME_${s}"
        read -r ss se sg ssm sep slr sev <<< "${!sv}"
        local gate_str="${sev} games"; [ "$sev" -eq 0 ] && gate_str="off"
        printf "  Stage %d %-14s  iter %3d-%-3d  %4d games  %4d sims  %2d epochs  lr=%-5s  gate=%s\n" \
            "$s" "${!nv}" "$ss" "$se" "$sg" "$ssm" "$sep" "$slr" "$gate_str" >> "${TRAIN_LOG}"
        total_games=$(( total_games + sg * (se - ss + 1) ))
    done

    cat >> "${TRAIN_LOG}" << EOF

Total: ${total_iters} iterations, ~${total_games} games
Hardware will be logged when training starts.
════════════════════════════════════════════════════════════════
EOF

    # Print the plan to console
    echo
    echo "============================================"
    echo "  Training Plan"
    echo "============================================"
    printf "  %-14s %8s %7s %6s %6s %7s %6s\n" \
        "Stage" "Iters" "Games" "Sims" "Epoch" "LR" "Gate"
    echo "  ────────────────────────────────────────────────────────────"
    for s in $(seq 1 $PLAN_NUM_STAGES); do
        local sv="PLAN_STAGE_${s}" nv="PLAN_STAGE_NAME_${s}"
        read -r ss se sg ssm sep slr sev <<< "${!sv}"
        local gate_str="${sev}g"; [ "$sev" -eq 0 ] && gate_str="off"
        printf "  %-14s %4d-%-3d %5d %6d %6d %7s %6s\n" \
            "${!nv}" "$ss" "$se" "$sg" "$ssm" "$sep" "$slr" "$gate_str"
    done
    echo "  ────────────────────────────────────────────────────────────"
    echo "  Total:         ${total_iters} iters, ~${total_games} games"
    echo
    echo "  Model:         $(best_onnx)"
    echo "  Plan:          ${PLAN_FILE}"
    echo "  Log:           ${TRAIN_LOG}"
    echo
    echo "  Ready! Run:  ./run_loop.sh train"
    echo "============================================"
}

# ══════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════

cmd_status() {
    if [ ! -f "$PLAN_FILE" ]; then
        echo "No training initialized. Run './run_loop.sh init small' first."
        exit 0
    fi

    read_plan
    read_state

    local total_iters
    total_iters=$(get_total_iterations)
    local pct=0
    [ "$total_iters" -gt 0 ] && pct=$(( PIPELINE_ITER * 100 / total_iters ))

    echo
    echo "============================================"
    echo "  MiniGo Training Status"
    echo "============================================"
    echo "  Architecture:   ${PLAN_BOARD}x${PLAN_BOARD}, ${PLAN_FILTERS}f x ${PLAN_BLOCKS}b"
    echo "  Progress:       ${PIPELINE_ITER} / ${total_iters} iterations (${pct}%)"
    echo "  Best model:     v$(printf '%04d' $BEST_VERSION) (${TOTAL_PROMOTIONS} promotions)"
    echo "  Total games:    ${TOTAL_GAMES}"
    echo

    printf "  %-14s %8s %7s %6s %6s %7s %6s   %s\n" \
        "Stage" "Iters" "Games" "Sims" "Epoch" "LR" "Gate" ""
    echo "  ────────────────────────────────────────────────────────────────"
    for s in $(seq 1 $PLAN_NUM_STAGES); do
        local sv="PLAN_STAGE_${s}" nv="PLAN_STAGE_NAME_${s}"
        read -r ss se sg ssm sep slr sev <<< "${!sv}"
        local gate_str="${sev}g"; [ "$sev" -eq 0 ] && gate_str="off"

        local status=""
        if [ "$PIPELINE_ITER" -ge "$se" ]; then
            status="DONE"
        elif [ "$PIPELINE_ITER" -ge "$ss" ]; then
            status="<- iter ${PIPELINE_ITER}"
        fi

        printf "  %-14s %4d-%-3d %5d %6d %6d %7s %6s   %s\n" \
            "${!nv}" "$ss" "$se" "$sg" "$ssm" "$sep" "$slr" "$gate_str" "$status"
    done
    echo "  ────────────────────────────────────────────────────────────────"

    if [ "$PIPELINE_ITER" -ge "$total_iters" ]; then
        echo
        echo "  Training COMPLETE. Final model: $(best_onnx)"
    else
        echo
        echo "  Resume:  ./run_loop.sh train"
    fi
    echo "============================================"
    echo
}

# ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════

cmd_train() {
    # Auto-detect hardware defaults
    detect_hardware
    MAX_BATCH=256
    local max_iters=0  # 0 = run until plan ends

    while [[ $# -gt 0 ]]; do
        case $1 in
            --threads)             THREADS=$2; shift 2;;
            --search-threads)      SEARCH_THREADS=$2; shift 2;;
            --selfplay-instances)  SELFPLAY_INSTANCES=$2; shift 2;;
            --nn-server-threads)   NN_SERVER_THREADS=$2; shift 2;;
            --nn-device-ids)       NN_DEVICE_IDS=$2; shift 2;;
            --max-batch)           MAX_BATCH=$2; shift 2;;
            --iterations)          max_iters=$2; shift 2;;
            --help|-h)
                cat << 'EOF'
./run_loop.sh train [hardware options]

Hardware options (only affects speed, not training quality):
  --threads N             Worker threads (default: all cores)
  --search-threads N      MCTS search threads per move (default: 16)
  --selfplay-instances N  Parallel selfplay processes (default: 1)
  --nn-server-threads N   NN server threads (default: auto-detect)
  --nn-device-ids IDS     GPU indices, comma-sep (default: auto-detect)
  --max-batch N           Max GPU batch size for NN server (default: 256)
  --iterations N          Max iterations to run this session (default: all)

GPU auto-detection: 2 server threads per GPU with pipelining.
  1 GPU  → --nn-server-threads 2 --nn-device-ids 0,0
  2 GPUs → --nn-server-threads 4 --nn-device-ids 0,0,1,1
EOF
                exit 0;;
            *) echo "Unknown train option: $1 (try --help)"; exit 1;;
        esac
    done

    read_plan
    read_state
    build_if_needed

    local total_iters
    total_iters=$(get_total_iterations)

    if [ "$PIPELINE_ITER" -ge "$total_iters" ]; then
        echo "Training already complete (${PIPELINE_ITER}/${total_iters} iterations)."
        echo "To restart: ./run_loop.sh init small"
        exit 0
    fi

    local start_iter=$(( PIPELINE_ITER + 1 ))
    local end_iter=$total_iters
    if [ "$max_iters" -gt 0 ]; then
        local cap=$(( start_iter + max_iters - 1 ))
        [ $cap -lt $end_iter ] && end_iter=$cap
    fi

    echo "============================================"
    echo "  MiniGo Training"
    echo "============================================"
    echo "  Architecture:     ${PLAN_BOARD}x${PLAN_BOARD}, ${PLAN_FILTERS}f x ${PLAN_BLOCKS}b"
    echo "  Iterations:       ${start_iter} .. ${end_iter}  (of ${total_iters})"
    echo "  Best model:       v$(printf '%04d' $BEST_VERSION)"
    echo "  Threads:          ${THREADS}"
    echo "  Search threads:   ${SEARCH_THREADS}"
    echo "  Selfplay inst:    ${SELFPLAY_INSTANCES}"
    echo "  NN servers:       ${NN_SERVER_THREADS}"
    echo "  NN devices:       ${NN_DEVICE_IDS}"
    echo "  Max batch (NN):   ${MAX_BATCH}"
    echo "  Batch size (SGD): ${PLAN_BATCH_SIZE}"
    echo "============================================"
    echo

    # Log training session to train.log
    tlog_section "TRAINING SESSION  iter ${start_iter}..${end_iter}"
    tlog "  Architecture:     ${PLAN_BOARD}x${PLAN_BOARD}, ${PLAN_FILTERS}f x ${PLAN_BLOCKS}b"
    tlog "  Batch size:       ${PLAN_BATCH_SIZE}"
    tlog "  Data window:      last ${PLAN_WINDOW_SIZE} iterations"
    tlog "  Eval threshold:   ${PLAN_EVAL_THRESHOLD}"
    tlog "  Hardware:"
    tlog "    Threads:          ${THREADS}"
    tlog "    Search threads:   ${SEARCH_THREADS}"
    tlog "    Selfplay inst:    ${SELFPLAY_INSTANCES}"
    tlog "    NN servers:       ${NN_SERVER_THREADS}"
    tlog "    NN devices:       ${NN_DEVICE_IDS}"
    tlog "    Max batch (NN):   ${MAX_BATCH}"
    tlog "  State:"
    tlog "    Best model:       v$(printf '%04d' $BEST_VERSION)"
    tlog "    Total games:      ${TOTAL_GAMES}"
    tlog "    Promotions:       ${TOTAL_PROMOTIONS}"

    log "Training starting: iter=${PIPELINE_ITER} best=v$(printf '%04d' $BEST_VERSION) games=${TOTAL_GAMES}"

    for iter in $(seq $start_iter $end_iter); do
        PIPELINE_ITER=$iter
        get_stage_for_iter $iter

        echo
        log "============================================"
        log "  ITERATION ${iter}/${total_iters}  |  Stage: ${STAGE_NAME}  |  Best: v$(printf '%04d' $BEST_VERSION)  |  LR: ${STAGE_LR}"
        log "============================================"

        tlog ""
        tlog "── ITERATION ${iter}/${total_iters}  Stage: ${STAGE_NAME} ──────────────────────"
        tlog "  games=${STAGE_GAMES}  sims=${STAGE_SIMS}  epochs=${STAGE_EPOCHS}  lr=${STAGE_LR}  eval=${STAGE_EVAL_GAMES}  best=v$(printf '%04d' $BEST_VERSION)"

        # ── Phase 1: Self-play ─────────────────────────────
        ITER_DATA="${DATA_DIR}/iter_$(printf '%04d' ${iter})"
        mkdir -p "${ITER_DATA}"

        existing_games=$(find "${ITER_DATA}" -maxdepth 1 \( -name "*.bin" -o -name "*.bin.gz" \) 2>/dev/null | wc -l)
        if [ "$existing_games" -ge "$STAGE_GAMES" ]; then
            log "Phase 1 — Selfplay: SKIP (${existing_games} games exist)"
            tlog "  Phase 1 selfplay: SKIP (${existing_games} games exist)"
        else
            local games_needed=$(( STAGE_GAMES - existing_games ))
            # Use versioned model file directly (TRT cache persists per version)
            local SELFPLAY_MODEL
            SELFPLAY_MODEL="$(version_onnx $BEST_VERSION)"
            log "Phase 1 — Selfplay: ${games_needed} games, ${STAGE_SIMS} sims, model v$(printf '%04d' $BEST_VERSION)..."
            tlog "  Phase 1 selfplay: ${games_needed} games, ${STAGE_SIMS} sims/move"
            tlog "    model=${SELFPLAY_MODEL}  threads=${THREADS}  search_threads=${SEARCH_THREADS}"
            tlog "    nn_servers=${NN_SERVER_THREADS}  devices=${NN_DEVICE_IDS}  instances=${SELFPLAY_INSTANCES}"

            local t_start=$SECONDS
            run_selfplay "${ITER_DATA}" "${SELFPLAY_MODEL}" "${games_needed}" "${STAGE_SIMS}"
            local sp_time=$((SECONDS - t_start))
            TOTAL_GAMES=$((TOTAL_GAMES + games_needed))

            # Compress .bin → .bin.gz to save disk (~100x smaller)
            gzip "${ITER_DATA}"/game_*.bin 2>/dev/null || true

            tlog "    done: ${sp_time}s ($(echo "scale=2; $sp_time / $games_needed" | bc 2>/dev/null || echo "?")s/game)  total_games=${TOTAL_GAMES}"
            log "Phase 1 — Selfplay done (${sp_time}s). Total games: ${TOTAL_GAMES}"
        fi

        # ── Phase 2: Training ──────────────────────────────
        local CANDIDATE_ONNX; CANDIDATE_ONNX="$(version_onnx $iter)"
        local CANDIDATE_CKPT; CANDIDATE_CKPT="$(version_checkpoint $iter)"

        if [ -f "${CANDIDATE_ONNX}" ] && [ -f "${CANDIDATE_CKPT}" ]; then
            log "Phase 2 — Training: SKIP (v$(printf '%04d' $iter) exists)"
            tlog "  Phase 2 training: SKIP (v$(printf '%04d' $iter) exists)"
        else
            local TRAIN_CKPT="${CHECKPOINT_DIR}/training.pt"
            local WINDOW_DIRS
            WINDOW_DIRS=$(build_data_window $iter)
            if [ -z "$WINDOW_DIRS" ]; then
                log "ERROR: no data in window"; exit 1
            fi

            # Count window dirs for logging
            local n_window_dirs
            n_window_dirs=$(echo "$WINDOW_DIRS" | tr ',' '\n' | wc -l)

            log "Phase 2 — Training: ${STAGE_EPOCHS} epochs, lr=${STAGE_LR}, batch=${PLAN_BATCH_SIZE}..."
            tlog "  Phase 2 training: ${STAGE_EPOCHS} epochs, lr=${STAGE_LR}, batch=${PLAN_BATCH_SIZE}"
            tlog "    window=${n_window_dirs} dirs"

            local t_start=$SECONDS

            # Detect GPUs for DDP training
            local train_gpus=1
            if command -v nvidia-smi &>/dev/null; then
                train_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
            fi

            local train_cmd="$PYTHON"
            local train_args=""
            if [ "$train_gpus" -gt 1 ]; then
                local port=$((29500 + RANDOM % 1000))
                train_cmd="torchrun --nproc_per_node=${train_gpus} --master_port=${port}"
                train_args=""  # torchrun replaces $PYTHON
            fi

            cd "${PROJECT_DIR}/scripts"
            $train_cmd train.py \
                --data "${WINDOW_DIRS}" \
                --checkpoint "${TRAIN_CKPT}" \
                --epochs ${STAGE_EPOCHS} \
                --batch-size ${PLAN_BATCH_SIZE} \
                --lr ${STAGE_LR} \
                --board ${PLAN_BOARD} \
                --filters ${PLAN_FILTERS} \
                --blocks ${PLAN_BLOCKS} \
                --output-onnx "${CANDIDATE_ONNX}" \
                --log-file "${TRAIN_LOG}"
            cd "${PROJECT_DIR}"

            local tr_time=$((SECONDS - t_start))

            if [ ! -f "${CANDIDATE_ONNX}" ]; then
                log "ERROR: training failed — ${CANDIDATE_ONNX} not produced"
                tlog "    ERROR: training failed — no ONNX produced"
                exit 1
            fi
            [ -f "${TRAIN_CKPT}" ] && cp "${TRAIN_CKPT}" "${CANDIDATE_CKPT}"

            log "Phase 2 — Training done (${tr_time}s). Candidate: v$(printf '%04d' $iter)"
        fi

        # ── Phase 3: Evaluation & Gating ───────────────────
        if [ "$STAGE_EVAL_GAMES" -gt 0 ] && [ "$BEST_VERSION" -gt 0 ]; then
            log "Phase 3 — Eval: v$(printf '%04d' $iter) vs v$(printf '%04d' $BEST_VERSION) (${STAGE_EVAL_GAMES} games, ${STAGE_SIMS} sims)..."
            tlog "  Phase 3 eval: v$(printf '%04d' $iter) vs v$(printf '%04d' $BEST_VERSION)  ${STAGE_EVAL_GAMES} games  ${STAGE_SIMS} sims"

            local t_start=$SECONDS
            local eval_tmp
            eval_tmp=$(mktemp)

            set +e
            "${BUILD_DIR}/evaluate" \
                --model1 "${CANDIDATE_ONNX}" \
                --model2 "$(version_onnx $BEST_VERSION)" \
                --games ${STAGE_EVAL_GAMES} \
                --threads ${THREADS} \
                --search-threads ${SEARCH_THREADS} \
                --max-batch ${MAX_BATCH} \
                --nn-server-threads ${NN_SERVER_THREADS} \
                --nn-device-ids ${NN_DEVICE_IDS} \
                --sims ${STAGE_SIMS} \
                --threshold ${PLAN_EVAL_THRESHOLD} \
                2>&1 | tee "$eval_tmp"
            local eval_result=$?
            set -e

            local ev_time=$((SECONDS - t_start))

            # Parse evaluation results for structured log
            local wr m1w m2w drw verdict
            wr=$(grep "win rate:" "$eval_tmp" | grep -oP '[\d.]+%' | head -1)
            m1w=$(grep "Model 1 wins:" "$eval_tmp" | grep -oP '\d+' | head -1)
            m2w=$(grep "Model 2 wins:" "$eval_tmp" | grep -oP '\d+' | head -1)
            drw=$(grep "Draws:" "$eval_tmp" | grep -oP '\d+' | head -1)
            verdict=$(grep "RESULT:" "$eval_tmp" | grep -oP 'PASS|FAIL')
            rm -f "$eval_tmp"

            tlog "    Win rate: ${wr:-?} (${m1w:-?}-${m2w:-?}-${drw:-?})  ${verdict:-?}  ${ev_time}s"

            if [ $eval_result -eq 0 ]; then
                log "Phase 3 — PROMOTED v$(printf '%04d' $iter) ${wr} (beats v$(printf '%04d' $BEST_VERSION)) [${ev_time}s]"
                cp "${CANDIDATE_ONNX}" "$(best_onnx)"
                BEST_VERSION=$iter
                TOTAL_PROMOTIONS=$((TOTAL_PROMOTIONS + 1))
            else
                log "Phase 3 — REJECTED v$(printf '%04d' $iter) ${wr} [${ev_time}s]"
            fi
        else
            [ "$BEST_VERSION" -eq 0 ] \
                && log "Phase 3 — Auto-promote v$(printf '%04d' $iter) (first model)" \
                || log "Phase 3 — Auto-promote v$(printf '%04d' $iter) (no gate this stage)"
            tlog "  Phase 3: auto-promote v$(printf '%04d' $iter)"
            cp "${CANDIDATE_ONNX}" "$(best_onnx)"
            BEST_VERSION=$iter
            TOTAL_PROMOTIONS=$((TOTAL_PROMOTIONS + 1))
        fi

        save_state
        log "Iteration ${iter} done. Best: v$(printf '%04d' $BEST_VERSION)  Promotions: ${TOTAL_PROMOTIONS}/${iter}"
        tlog "  Result: best=v$(printf '%04d' $BEST_VERSION)  promotions=${TOTAL_PROMOTIONS}/${iter}  total_games=${TOTAL_GAMES}"
    done

    echo
    echo "============================================"
    if [ "$end_iter" -ge "$total_iters" ]; then
        echo "  Training Complete!"
    else
        echo "  Session Complete (paused at iter ${end_iter})"
        echo "  Resume: ./run_loop.sh train"
    fi
    echo "  Best model:     v$(printf '%04d' $BEST_VERSION)"
    echo "  Total games:    ${TOTAL_GAMES}"
    echo "  Promotions:     ${TOTAL_PROMOTIONS}"
    echo "  Model file:     $(best_onnx)"
    echo "============================================"

    tlog_section "SESSION END"
    tlog "  Best: v$(printf '%04d' $BEST_VERSION)  Games: ${TOTAL_GAMES}  Promotions: ${TOTAL_PROMOTIONS}"
}

# ══════════════════════════════════════════════════════════
#  USAGE
# ══════════════════════════════════════════════════════════

cmd_usage() {
    cat << 'EOF'
MiniGo AlphaZero — Training Pipeline

Commands:
  init <preset>   Initialize training (clears previous state)
  train           Start or resume training
  status          Show training progress

Presets for init:
  quick           5x5, 32f/3b, 5 iterations (pipeline test)
  small           9x9, 64f/5b, 100 iterations, ~240K games
  large           9x9, 128f/10b, 200 iterations, ~800K games

Custom init:
  init --board 9 --filters 96 --blocks 8

Examples:
  ./run_loop.sh init small
  ./run_loop.sh train                          # auto-detect GPUs
  ./run_loop.sh train --nn-device-ids 0,0,1,1  # manual GPU assignment
  ./run_loop.sh status
  ./run_loop.sh train --iterations 10          # run 10 more iters then stop
EOF
}

# ══════════════════════════════════════════════════════════
#  DISPATCH
# ══════════════════════════════════════════════════════════

case "${1:-}" in
    init)   shift; cmd_init "$@";;
    train)  shift; cmd_train "$@";;
    status) cmd_status;;
    -h|--help|help) cmd_usage;;
    "") cmd_usage;;
    *) echo "Unknown command: $1"; echo; cmd_usage; exit 1;;
esac
