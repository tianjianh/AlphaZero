#!/usr/bin/env bash
# kata1 ONNX → A733 NBG (asymmetric int8 + hybrid fp16 fallback).
#
# The fp16 path produced by tools/onnx_to_a733_docker.sh runs almost
# entirely on the VIP9000's PPU/Shader engines (89 SP + 71 SH ops, only
# 2 NN-core ops out of 195 — see A733_CONVERSION.md §12).  This is
# because VIP9000 NN cores are int8 dataflow; fp16 convs fall back to
# the shader path which is ~10× slower than the NN cores.
#
# This script runs the int8 quantization path:
#   1. Re-import the ONNX (we can't just reuse the fp16 IR since
#      quantize wants the import to share its --output-data file).
#   2. Run pegasus.py quantize with the calibration data from
#      tools/a733_gen_calib.py.  --quantizer asymmetric_affine
#      --qtype int8 --hybrid lets layers that quantize poorly fall
#      back to fp16 automatically (kata1's score/value heads are
#      typical fall-back candidates because their dynamic ranges blow
#      up after softmax).
#   3. Re-export with --dtype quantized + --model-quantize <name>.quantize.
#      This packs the NBG with int8 NN-core ops where possible.
#
# Prereqs:
#   * tools/onnx_to_a733_docker.sh has been run at least once for this
#     batch (the .unshared.onnx is what we feed in — same as the fp16
#     path).
#   * tools/a733_gen_calib.py has been run, producing
#     build/a733_calib/{state_spatial,state_global}/*.npy and the
#     dataset txt files.
#
# Usage:
#   bash tools/onnx_to_a733_quantize_docker.sh           # bs=1, default
#   bash tools/onnx_to_a733_quantize_docker.sh 1
#   bash tools/onnx_to_a733_quantize_docker.sh 4
#   ITER=200 bash tools/onnx_to_a733_quantize_docker.sh 1
#
# Outputs:
#   models/<base>.a733.bs<N>.int8/network_binary.nb
#   models/<base>.a733.bs<N>.int8/nbg_meta.json
#   models/<base>.a733.bs<N>.int8/<base>.a733.bs<N>.unshared_int8.quantize  (calibration table)

set -euo pipefail

BS="${1:-1}"
BASE="${BASE:-kata1-b10c128}"
IMAGE="ubuntu-npu:v2.0.10.1"
ITER="${ITER:-150}"          # quantize iterations (default = num samples roughly)
ALGO="${ALGO:-kl_divergence}"
QUANTIZER="${QUANTIZER:-asymmetric_affine}"
QTYPE="${QTYPE:-int8}"
CALIB_DIR="${CALIB_DIR:-build/a733_calib}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_NAME="${BASE}.a733.bs${BS}.unshared"
ONNX_PATH="${REPO_ROOT}/models/${ONNX_NAME}.onnx"
OUT_DIR="${REPO_ROOT}/models/${BASE}.a733.bs${BS}.int8"

CHIP="VIP9000NANODI_PID0X1000003B"

# ── Prereq checks ──────────────────────────────────────────────────────

for cmd in docker; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found" >&2; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon unreachable; sudo or add to docker group" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: $IMAGE not loaded; see A733_CONVERSION.md §3.2" >&2; exit 1; }

[ -f "$ONNX_PATH" ] || { echo "ERROR: $ONNX_PATH missing; run tools/onnx_to_a733_docker.sh $BS first to ensure ONNX exists" >&2; exit 1; }

CALIB_SP_LIST="${REPO_ROOT}/${CALIB_DIR}/dataset0_spatial.txt"
CALIB_GL_LIST="${REPO_ROOT}/${CALIB_DIR}/dataset1_global.txt"
[ -f "$CALIB_SP_LIST" ] || { echo "ERROR: calibration data missing — run tools/a733_gen_calib.py first" >&2; exit 1; }
[ -f "$CALIB_GL_LIST" ] || { echo "ERROR: calibration data missing — run tools/a733_gen_calib.py first" >&2; exit 1; }

mkdir -p "$OUT_DIR"

echo "[a733-int8] image:        $IMAGE"
echo "[a733-int8] source:       $ONNX_PATH"
echo "[a733-int8] calib:        ${CALIB_DIR}/dataset{0,1}_*.txt"
echo "[a733-int8] quantizer:    $QUANTIZER ($QTYPE, $ALGO, $ITER iters, hybrid)"
echo "[a733-int8] target chip:  $CHIP"
echo "[a733-int8] output:       $OUT_DIR"
echo "[a733-int8] running in container..."

# ── Container body ─────────────────────────────────────────────────────
#
# We sed-edit the inputmeta YAML to point at our calibration TEXT files
# (and to pass NPY rather than the default raw-binary .txt format the
# autogen yaml assumes).  Then quantize, then export.

docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -e BS="$BS" \
    -e ONNX_NAME="$ONNX_NAME" \
    -e CHIP="$CHIP" \
    -e ITER="$ITER" \
    -e ALGO="$ALGO" \
    -e QUANTIZER="$QUANTIZER" \
    -e QTYPE="$QTYPE" \
    -e CALIB_DIR="$CALIB_DIR" \
    -w /workspace \
    "$IMAGE" \
    /bin/bash -c '
        set -e

        # Same ldconfig + env setup as the fp16 path
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator/lib"   >  /etc/ld.so.conf.d/vivante.conf
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/lib"       >> /etc/ld.so.conf.d/vivante.conf
        ldconfig

        export VSIMULATOR_CONFIG="$CHIP"
        export VSIMULATOR_SHADER_CORE_COUNT=1
        export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin

        WORK=/workspace/build/a733/${ONNX_NAME}_int8
        rm -rf "$WORK"; mkdir -p "$WORK"
        cp /workspace/models/${ONNX_NAME}.onnx "$WORK/${ONNX_NAME}.onnx"
        cd "$WORK"

        echo "[a733-int8] 1) import ONNX (bs=${BS})"
        python3 ${ACUITY_PATH}/pegasus.py import onnx \
            --model           ${ONNX_NAME}.onnx \
            --output-model    ${ONNX_NAME}.json \
            --output-data     ${ONNX_NAME}.data \
            --inputs          "state_spatial state_global" \
            --input-size-list "${BS},22,9,9#${BS},19" \
            --outputs         "policy_logits value score_mean score_stdev ownership" \
            --size-with-batch "true#true" 2>&1 | grep -E "Error|Save|End importing|----" | tail -5

        echo "[a733-int8] 2) generate inputmeta yaml + repoint at TEXT/TENSOR calibration"
        python3 ${ACUITY_PATH}/pegasus.py generate inputmeta \
            --model              ${ONNX_NAME}.json \
            --separated-database \
            --input-meta-output  ${ONNX_NAME}_inputmeta.yml 2>&1 | grep -E "Error|Generate|----" | tail -3

        # The auto-generated inputmeta has TWO traps for tensor inputs:
        #   * category: image       -> Acuity applies image-style
        #     preprocessing (most importantly reverse_channel which
        #     reorders the 22 spatial planes like RGB->BGR).
        #   * reverse_channel: true -> the actual flag that does it.
        # Both produce calibration on scrambled inputs, so the
        # observed activation ranges are nonsense and the resulting
        # int8 scales are calibrated for the wrong distribution. The
        # on-board runtime feeds fp32 inputs straight to the NBG and
        # does NOT apply inputmeta preprocessing, so the scales no
        # longer match anything and accuracy collapses.
        #
        # Write a clean inputmeta from scratch with category=undefined
        # and reverse_channel=false; point at our calibration text
        # lists.  Acuity reads each line as a path to a numpy-loadtxt
        # ASCII file (one float per line, flat row-major).
        sed "s|/root/proj/AlphaZero/|/workspace/|g" \
            /workspace/${CALIB_DIR}/dataset0_spatial.txt \
            > /tmp/dataset0_spatial.txt
        sed "s|/root/proj/AlphaZero/|/workspace/|g" \
            /workspace/${CALIB_DIR}/dataset1_global.txt \
            > /tmp/dataset1_global.txt

        # Extract the lids Acuity assigned (e.g. state_spatial_146 /
        # state_global_147 for kata1; can shift between models).
        SP_LID=$(grep -oE "lid: state_spatial_[0-9]+" ${ONNX_NAME}_inputmeta.yml | head -1 | awk "{print \$2}")
        GL_LID=$(grep -oE "lid: state_global_[0-9]+"  ${ONNX_NAME}_inputmeta.yml | head -1 | awk "{print \$2}")
        cat > ${ONNX_NAME}_inputmeta.yml <<EOF
input_meta:
  databases:
  - path: /tmp/dataset0_spatial.txt
    type: TEXT
    ports:
    - lid: ${SP_LID}
      category: undefined
      dtype: float32
      sparse: false
      tensor_name:
      layout: nchw
      shape: [${BS}, 22, 9, 9]
      fitting: scale
      preprocess:
        reverse_channel: false
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm: [0, 1, 2, 3]
      redirect_to_output: false
  - path: /tmp/dataset1_global.txt
    type: TEXT
    ports:
    - lid: ${GL_LID}
      category: undefined
      dtype: float32
      sparse: false
      tensor_name:
      layout: nchw
      shape: [${BS}, 19]
      fitting: scale
      preprocess:
        reverse_channel: false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm: [0, 1]
      redirect_to_output: false
EOF
        echo "----- inputmeta head -----"
        head -25 ${ONNX_NAME}_inputmeta.yml
        echo "--------------------------"

        echo "[a733-int8] 3) quantize (--quantizer ${QUANTIZER} --qtype ${QTYPE} --hybrid --algorithm ${ALGO} --iterations ${ITER})"
        # Note: --output-dir places the .quantize file relative to cwd,
        # not absolute.  --batch-size is the calibration batch (use 1 to
        # match the NPY shape).
        # The --rebuild flow expects a pre-existing .quantize file that
        # it then refreshes.  For a cold run you need to seed it with a
        # minimal valid JSON; pegasus.py errors out at "quantize file
        # ... does not exist" before --rebuild logic kicks in
        # otherwise.  --rebuild-all then walks every tensor and
        # populates the table from the calibration set.
        #
        # --batch-size must match the model batch dim. Calibration
        # samples are per-sample [1,22,9,9]/[1,19]; Acuity stacks
        # BS of them to fill the static-batch input.
        QUANT_OUT=${ONNX_NAME}_${QTYPE}.quantize
        printf '"'"'{"version":1,"tensors":[]}\n'"'"' > "$QUANT_OUT"
        # Cap iterations so we never demand more samples than we have
        # (TOTAL_SAMPLES = ITER * BS).  ~150 stacks is plenty for
        # kl_divergence on a small fp16-targeted network.
        QITER=$(( ITER < 75 ? ITER : 75 ))
        python3 ${ACUITY_PATH}/pegasus.py quantize \
            --model              ${ONNX_NAME}.json \
            --model-data         ${ONNX_NAME}.data \
            --device             CPU \
            --with-input-meta    ${ONNX_NAME}_inputmeta.yml \
            --model-quantize     "$QUANT_OUT" \
            --quantizer          "$QUANTIZER" \
            --qtype              "$QTYPE" \
            --hybrid \
            --compute-entropy \
            --algorithm          "$ALGO" \
            --batch-size         "$BS" \
            --iterations         "$QITER" \
            --rebuild-all        2>&1 | tee quantize.log | grep -E "Error|quantiz|Hybrid|fall|Save|----|^I |^W |^E " | tail -25

        QUANT_FILE=$(ls -1 *.quantize 2>/dev/null | head -1)
        if [ -z "$QUANT_FILE" ]; then
            echo "[a733-int8] ERROR: quantize step produced no .quantize file" >&2
            tail -50 quantize.log >&2
            exit 1
        fi
        # Sanity: a populated quantize table is hundreds of KB; a stub
        # is ~30 B.  If quantize silently failed (allow_pickle, missing
        # calibration files, etc.) the file would still be the seed
        # JSON we wrote and the export would silently produce an
        # un-quantized NBG.
        QUANT_BYTES=$(stat -c %s "$QUANT_FILE")
        if [ "$QUANT_BYTES" -lt 1000 ]; then
            echo "[a733-int8] ERROR: quantize step left an empty calibration table ($QUANT_BYTES B)." >&2
            echo "[a733-int8]   This usually means the calibration data path or format was wrong." >&2
            tail -50 quantize.log >&2
            exit 1
        fi
        echo "[a733-int8] produced calibration table: $QUANT_FILE ($QUANT_BYTES bytes)"

        echo "[a733-int8] 4) export ovxlib + pack NBG (quantized)"
        python3 ${ACUITY_PATH}/pegasus.py export ovxlib \
            --model              ${ONNX_NAME}.json \
            --model-data         ${ONNX_NAME}.data \
            --model-quantize     "$QUANT_FILE" \
            --dtype              quantized \
            --target-ide-project linux64 \
            --with-input-meta    ${ONNX_NAME}_inputmeta.yml \
            --pack-nbg-unify \
            --optimize           "$CHIP" \
            --viv-sdk            /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator \
            --output-path        wksp/${ONNX_NAME}_int8 2>&1 | grep -E "Error|Saving|Pack|nbg|----" | tail -10

        # The .nb lands in either wksp/ or wksp_nbg_unify/, same trap as
        # the fp16 path.
        OUT=/workspace/models/'"$BASE"'.a733.bs${BS}.int8
        mkdir -p "$OUT"
        NB=$(find wksp wksp_nbg_unify -name "network_binary.nb" -type f 2>/dev/null | head -1)
        META=$(find wksp wksp_nbg_unify -name "nbg_meta.json"   -type f 2>/dev/null | head -1)
        if [ -z "$NB" ]; then
            echo "[a733-int8] ERROR: NBG not produced" >&2
            find . -type f -name "*.nb" -o -name "*.json" | head -20 >&2
            exit 1
        fi
        cp "$NB"    "$OUT/network_binary.nb"
        [ -n "$META" ] && cp "$META" "$OUT/nbg_meta.json"
        cp "$QUANT_FILE" "$OUT/$QUANT_FILE"
        echo "[a733-int8] wrote $OUT/network_binary.nb ($(stat -c %s "$OUT/network_binary.nb") bytes)"
    '

# ── Header check + engine breakdown ───────────────────────────────────

echo
python3 - <<PY
import struct
p = "$OUT_DIR/network_binary.nb"
with open(p, "rb") as f: head = f.read(12)
magic, ver, target = struct.unpack("<III", head)
ok = (magic == 0x4e4d5056 and target == 0x1000003B and ((ver == 0x1001E) or (ver >> 16) == 0x2))
print(f"[a733-int8] {p}")
print(f"[a733-int8]   magic   = {magic:#x}    (expect 0x4e4d5056 = 'VPMN')")
print(f"[a733-int8]   version = {ver:#x}      (NBG v2.0.0 from v6.30.22)")
print(f"[a733-int8]   target  = {target:#x}  (expect 0x1000003b)")
print(f"[a733-int8]   {'PASS' if ok else 'FAIL — header bytes wrong'}")
PY

echo
echo "[a733-int8] op-engine breakdown via nbinfo:"
docker run --rm -v "${REPO_ROOT}:/workspace" -w /workspace "$IMAGE" \
    /root/nbinfo -o "models/${BASE}.a733.bs${BS}.int8/network_binary.nb" 2>&1 \
    | grep -E "^Operation Type:" | sort | uniq -c | sort -rn \
    | sed 's/^/[a733-int8]   /'
echo
echo "[a733-int8] (NN/NN_LUT ops use the NPU NN cores; SP/SH are PPU/shader fallback;"
echo "[a733-int8]  TP/NT are tensor-processor ops.  More NN = better.)"
