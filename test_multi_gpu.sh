#!/bin/bash
#
# MiniGo Multi-GPU Test Suite
#
# Tests the multi-GPU NNEvaluator on a machine with 2+ GPUs.
# Run from the project root after building with OpenCL backend.
#
# Usage:
#   ./test_multi_gpu.sh                    # full test suite
#   ./test_multi_gpu.sh --model model.onnx # custom model
#   ./test_multi_gpu.sh --quick            # fast smoke test only
#
# Prerequisites:
#   mkdir -p build && cd build && cmake .. -DMINIGO_BACKEND=opencl && make -j$(nproc) && cd ..
#

set -e

MODEL="model.onnx"
BUILD_DIR="build"
QUICK=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL=$2; shift 2;;
        --build-dir) BUILD_DIR=$2; shift 2;;
        --quick) QUICK=1; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

SELFPLAY="${BUILD_DIR}/selfplay"
BENCHMARK="${BUILD_DIR}/benchmark"

if [ ! -f "$SELFPLAY" ] || [ ! -f "$BENCHMARK" ]; then
    echo "ERROR: Binaries not found in ${BUILD_DIR}/"
    echo "Build first: mkdir -p build && cd build && cmake .. -DMINIGO_BACKEND=opencl && make -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

PASS=0
FAIL=0
RESULTS=""

run_test() {
    local name="$1"
    local cmd="$2"
    local expect="$3"  # "pass" or "time" (capture timing)

    echo "──────────────────────────────────────────"
    echo "TEST: $name"
    echo "CMD:  $cmd"
    echo ""

    local t0=$(date +%s%N)
    if eval "$cmd"; then
        local t1=$(date +%s%N)
        local ms=$(( (t1 - t0) / 1000000 ))
        echo ""
        echo "RESULT: PASS (${ms}ms)"
        PASS=$((PASS + 1))
        RESULTS="${RESULTS}\n  PASS  ${name} (${ms}ms)"
    else
        echo ""
        echo "RESULT: FAIL"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}\n  FAIL  ${name}"
    fi
    echo ""
}

echo "============================================"
echo "  MiniGo Multi-GPU Test Suite"
echo "============================================"
echo "  Model:     $MODEL"
echo "  Build:     $BUILD_DIR"
echo "  Backend:   opencl"
echo "============================================"
echo ""

# ── Test 1: Baseline — 1 server, GPU 0 ────────────────────
run_test "1-server, GPU 0 (baseline selfplay)" \
    "$SELFPLAY --model $MODEL --games 4 --threads 2 --search-threads 4 --sims 32 --nn-server-threads 1 --nn-device-ids 0 --output /tmp/minigo_test1"

# ── Test 2: 2 servers, same GPU 0,0 ───────────────────────
run_test "2-server, same GPU 0,0 (selfplay)" \
    "$SELFPLAY --model $MODEL --games 4 --threads 2 --search-threads 4 --sims 32 --nn-server-threads 2 --nn-device-ids 0,0 --output /tmp/minigo_test2"

# ── Test 3: 2 servers, 2 GPUs 0,1 ─────────────────────────
run_test "2-server, 2 GPUs 0,1 (selfplay)" \
    "$SELFPLAY --model $MODEL --games 4 --threads 2 --search-threads 4 --sims 32 --nn-server-threads 2 --nn-device-ids 0,1 --output /tmp/minigo_test3"

# ── Test 4: 4 servers, 2 GPUs 0,0,1,1 ─────────────────────
run_test "4-server, 2 GPUs 0,0,1,1 (selfplay)" \
    "$SELFPLAY --model $MODEL --games 8 --threads 4 --search-threads 8 --sims 32 --nn-server-threads 4 --nn-device-ids 0,0,1,1 --output /tmp/minigo_test4"

if [ "$QUICK" -eq 1 ]; then
    echo "============================================"
    echo "  Quick mode — skipping benchmark tests"
    echo "============================================"
else

# ── Test 5: Benchmark — 1 server, GPU 0 ───────────────────
run_test "Benchmark, 1 server, GPU 0" \
    "stdbuf -oL $BENCHMARK --model $MODEL --nn-iters 50 --games 2 --threads 1 --sims 32 --nn-server-threads 1 --nn-device-ids 0"

# ── Test 6: Benchmark — 2 servers, 2 GPUs ─────────────────
run_test "Benchmark, 2 servers, GPUs 0,1" \
    "stdbuf -oL $BENCHMARK --model $MODEL --nn-iters 50 --games 4 --threads 2 --sims 32 --nn-server-threads 2 --nn-device-ids 0,1"

# ── Test 7: Benchmark — 4 servers, 2 GPUs ─────────────────
run_test "Benchmark, 4 servers, GPUs 0,0,1,1" \
    "stdbuf -oL $BENCHMARK --model $MODEL --nn-iters 50 --games 4 --threads 4 --sims 32 --nn-server-threads 4 --nn-device-ids 0,0,1,1"

# ── Test 8: Large model (if available) ────────────────────
if [ -f "model_large.onnx" ]; then
    run_test "Large model, 2 servers, GPUs 0,1" \
        "$SELFPLAY --model model_large.onnx --games 4 --threads 2 --search-threads 4 --sims 32 --nn-server-threads 2 --nn-device-ids 0,1 --output /tmp/minigo_test8"
else
    echo "── Skipping large model test (model_large.onnx not found)"
    echo ""
fi

# ── Test 9: Stability — 10-run repeat ─────────────────────
echo "──────────────────────────────────────────"
echo "TEST: Stability — 10 sequential selfplay runs"
echo ""
STABILITY_PASS=0
for i in $(seq 1 10); do
    if $SELFPLAY --model $MODEL --games 2 --threads 1 --search-threads 2 --sims 16 --nn-server-threads 2 --nn-device-ids 0,1 --output /tmp/minigo_stability_$i 2>&1 | tail -1; then
        STABILITY_PASS=$((STABILITY_PASS + 1))
    fi
done
echo ""
if [ "$STABILITY_PASS" -eq 10 ]; then
    echo "RESULT: PASS (10/10)"
    PASS=$((PASS + 1))
    RESULTS="${RESULTS}\n  PASS  Stability 10/10"
else
    echo "RESULT: FAIL ($STABILITY_PASS/10)"
    FAIL=$((FAIL + 1))
    RESULTS="${RESULTS}\n  FAIL  Stability ${STABILITY_PASS}/10"
fi
echo ""

fi  # end of non-quick tests

# ── Summary ────────────────────────────────────────────────
echo "============================================"
echo "  TEST SUMMARY"
echo "============================================"
echo -e "$RESULTS"
echo ""
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "============================================"

# Cleanup
rm -rf /tmp/minigo_test* /tmp/minigo_stability_*

exit $FAIL
