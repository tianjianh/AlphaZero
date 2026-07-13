#!/usr/bin/env bash
# Turnkey: kata1 ONNX → Allwinner A733 NBG, via Allwinner's official
# Acuity Toolkit Docker image (ubuntu-npu:v2.0.10.1, internally Acuity
# v6.30.22).  This is the conversion path documented in
# docs/A733_CONVERSION.md; the pip-acuitylite path (removed, see §6.1)
# is broken (produces NBGs with target=0x15 that the on-board viplite
# v2.0.3 rejects with nbglk_valid_nbg_check[920]).
#
# Prereqs (the script checks these):
#   * Real x86_64 Linux host with Docker (NOT a nested container — the
#     VIP9000 simulator inside gen_nbg needs /proc mountable; see
#     docs/A733_CONVERSION.md §6.3).
#   * `ubuntu-npu:v2.0.10.1` already loaded.  This image is not on a
#     public registry — see docs/A733_CONVERSION.md §3.2 for how to fetch
#     it from Allwinner's Synology netdisk and `docker load` it.
#   * The matching `models/kata1-b10c128.a733.bs<N>.unshared.onnx`
#     already produced from the .txt.gz weights.  See §3.1.
#
# Usage:
#   bash tools/onnx_to_a733_docker.sh           # builds bs=1
#   bash tools/onnx_to_a733_docker.sh 1
#   bash tools/onnx_to_a733_docker.sh 4         # builds bs=4
#   BASE=somethingelse bash tools/onnx_to_a733_docker.sh 1
#
# Outputs:
#   models/<base>.a733.bs<N>.fp16/network_binary.nb
#   models/<base>.a733.bs<N>.fp16/nbg_meta.json

set -euo pipefail

BS="${1:-1}"
BASE_DEFAULT="kata1-b10c128"
BASE="${BASE:-$BASE_DEFAULT}"
IMAGE="ubuntu-npu:v2.0.10.1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_NAME="${BASE}.a733.bs${BS}.unshared"
ONNX_PATH="${REPO_ROOT}/models/${ONNX_NAME}.onnx"
OUT_DIR="${REPO_ROOT}/models/${BASE}.a733.bs${BS}.fp16"

# The chip target.  NOTE: NOT _PLUS_ despite Allwinner's pegasus_setup.sh
# v3 saying so — the actual .config file shipped in ubuntu-npu:v2.0.10.1
# is named without _PLUS_.  See docs/A733_CONVERSION.md §0 / §6.1.
CHIP="VIP9000NANODI_PID0X1000003B"

# ── Sanity checks ──────────────────────────────────────────────────────

if ! command -v docker >/dev/null; then
    echo "ERROR: docker not found.  Install Docker on this host first." >&2
    echo "  See https://docs.docker.com/engine/install/" >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: cannot talk to the Docker daemon." >&2
    echo "  Try: sudo bash $0 $*" >&2
    echo "  Or:  add yourself to the docker group: sudo usermod -aG docker \$USER" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: Docker image $IMAGE is not loaded on this host.

This image is not on Docker Hub.  Get it from Allwinner's Synology netdisk:

  SID=\$(curl -sL -G "https://netstorage.allwinnertech.com:5001/webapi/entry.cgi" \\
      --data-urlencode "api=SYNO.Core.Sharing.Login" \\
      --data-urlencode "version=1" \\
      --data-urlencode "method=login" \\
      --data-urlencode "sharing_id=Mh23BhPHq" \\
      | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['sharing_sid'])")
  curl -L "https://netstorage.allwinnertech.com:5001/fsdownload/Mh23BhPHq/share.zip" \\
      -b "sharing_sid=\$SID" -o share.zip
  unzip -q share.zip
  cd docker_images_v2.0.x
  unzip -q ubuntu-npu_v2.0.10.1.tar.zip
  sudo docker load -i ubuntu-npu_v2.0.10.1.tar

Total ~2.9 GB compressed, 7.4 GB uncompressed tar, then loaded into
Docker.  See docs/A733_CONVERSION.md §3.2 for full context.
EOF
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    cat >&2 <<EOF
ERROR: source ONNX not found: $ONNX_PATH

Produce it first (in the alphazero conda env):

    conda activate alphazero
    python tools/kata_export_for_rknn.py \\
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \\
        --board 9 --batch ${BS} --opset 13 \\
        --output models/${BASE}.a733.bs${BS}.onnx

    python -c "import onnx; from onnxsim import simplify; \\
        m,_=simplify(onnx.load('models/${BASE}.a733.bs${BS}.onnx')); \\
        onnx.save(m,'models/${BASE}.a733.bs${BS}.onnx')"

    python tools/onnx_rknn_mitigations.py \\
        --input  models/${BASE}.a733.bs${BS}.onnx \\
        --output models/${BASE}.a733.bs${BS}.unshared.onnx \\
        --unshare-initializers
EOF
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "[a733] image:      $IMAGE"
echo "[a733] source:     $ONNX_PATH"
echo "[a733] target chip: $CHIP"
echo "[a733] output dir: $OUT_DIR"
echo "[a733] running in container..."

# ── The actual conversion (see docs/A733_CONVERSION.md §3.3 for annotation) ─

docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -e BS="$BS" \
    -e ONNX_NAME="$ONNX_NAME" \
    -e CHIP="$CHIP" \
    -w /workspace \
    "$IMAGE" \
    /bin/bash -c '
        set -e

        # Make gen_nbg find libGAL/libOpenVX/libovxlib (vsimulator/lib) +
        # libvdtproxy (common/lib).  Both directories are required.
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator/lib"   >  /etc/ld.so.conf.d/vivante.conf
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/lib"       >> /etc/ld.so.conf.d/vivante.conf
        ldconfig

        export VSIMULATOR_CONFIG="$CHIP"
        export VSIMULATOR_SHADER_CORE_COUNT=1
        export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin

        WORK=/workspace/build/a733/${ONNX_NAME}
        rm -rf "$WORK"; mkdir -p "$WORK"
        cp /workspace/models/${ONNX_NAME}.onnx "$WORK/${ONNX_NAME}.onnx"
        cd "$WORK"

        echo "[a733] 1) import ONNX → Acuity IR (bs=${BS})"
        python3 ${ACUITY_PATH}/pegasus.py import onnx \
            --model           ${ONNX_NAME}.onnx \
            --output-model    ${ONNX_NAME}.json \
            --output-data     ${ONNX_NAME}.data \
            --inputs          "state_spatial state_global" \
            --input-size-list "${BS},22,9,9#${BS},19" \
            --outputs         "policy_logits value score_mean score_stdev ownership" \
            --size-with-batch "true#true" 2>&1 | grep -E "Error|Warning|Save|End importing|----" | tail -8

        echo "[a733] 2) generate inputmeta yaml"
        python3 ${ACUITY_PATH}/pegasus.py generate inputmeta \
            --model              ${ONNX_NAME}.json \
            --separated-database \
            --input-meta-output  ${ONNX_NAME}_inputmeta.yml 2>&1 | grep -E "Error|Generate|----" | tail -3

        echo "[a733] 3) export ovxlib + pack NBG"
        python3 ${ACUITY_PATH}/pegasus.py export ovxlib \
            --model              ${ONNX_NAME}.json \
            --model-data         ${ONNX_NAME}.data \
            --dtype              float \
            --target-ide-project linux64 \
            --with-input-meta    ${ONNX_NAME}_inputmeta.yml \
            --pack-nbg-unify \
            --optimize           "$CHIP" \
            --viv-sdk            /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator \
            --output-path        wksp/${ONNX_NAME}_fp16 2>&1 | grep -E "Error|Saving|Pack|nbg|----" | tail -10

        # ----- collect -----
        # Acuity v6.30.22 with --pack-nbg-unify writes the .nb to a
        # sibling directory `<output-path>_nbg_unify/`, not under the
        # `--output-path` itself.  Search both.
        OUT=/workspace/models/'"$BASE"'.a733.bs${BS}.fp16
        mkdir -p "$OUT"
        NB=$(find wksp wksp_nbg_unify -name "network_binary.nb" -type f 2>/dev/null | head -1)
        META=$(find wksp wksp_nbg_unify -name "nbg_meta.json"   -type f 2>/dev/null | head -1)
        if [ -z "$NB" ]; then
            echo "[a733] ERROR: NBG not produced — listing build dir" >&2
            find . -type f | head -30 >&2
            exit 1
        fi
        cp "$NB"   "$OUT/network_binary.nb"
        [ -n "$META" ] && cp "$META" "$OUT/nbg_meta.json"
        echo "[a733] wrote $OUT/network_binary.nb ($(stat -c %s "$OUT/network_binary.nb") bytes)"
    '

# ── Header check (the smoke test) ──────────────────────────────────────

echo
python3 - <<PY
import struct, sys
p = "$OUT_DIR/network_binary.nb"
with open(p, "rb") as f: head = f.read(12)
magic, ver, target = struct.unpack("<III", head)
# Acuity v6.30.22 (in ubuntu-npu:v2.0.10.1) emits NBG format v2.0.0
# (0x20000); the toolkit's host-side libOpenVX advertises both
# nbglk_create_v1_video_memory and nbglk_create_v2_video_memory, so v2
# NBGs are expected to load on viplite v2.0.3+.  Accept either v1.0.30
# (0x1001E, older toolkits) or v2.0.x (0x2xxxx, this image).
magic_ok  = (magic == 0x4e4d5056)
target_ok = (target == 0x1000003B)
ver_ok    = (ver == 0x1001E) or (ver >> 16) == 0x2
ok = magic_ok and target_ok and ver_ok
print(f"[a733] {p}")
print(f"[a733]   magic   = {magic:#x}    (expect 0x4e4d5056 = 'VPMN')")
print(f"[a733]   version = {ver:#x}      (acuity v6.30.22 in ubuntu-npu:v2.0.10.1 emits 0x20000 = NBG v2.0.0)")
print(f"[a733]   target  = {target:#x} (expect 0x1000003b)")
print(f"[a733]   {'PASS — ship to the Cubie A7A and run vpm_run smoke test (docs/A733_CONVERSION.md §7.1).' if ok else 'FAIL — these bytes will be rejected by on-board viplite.'}")
sys.exit(0 if ok else 1)
PY
