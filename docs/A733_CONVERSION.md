# A733 Conversion — kata1 ONNX → Allwinner A733 NBG (Vivante VIP9000)

> **STATUS (2026-04-29): WORKING.** `tools/onnx_to_a733_docker.sh`
> produces NBGs the on-board viplite v2.0.3.2 accepts at
> `vip_create_network`. Both bs=1 and bs=4 verified on the Cubie A7A
> via `vpm_run` smoke test — `create network 0` and `vpm run ret=0`
> both return clean. Header signature: `target=0x1000003B,
> version=0x20000` (NBG v2.0.0, same as Allwinner's own `yolact/v3`
> sample).
>
> The pip-`acuitylite` path at `tools/onnx_to_a733.py` does NOT work —
> it produces NBGs with `target=0x15` because the bundled chip table
> doesn't contain A733's PID. That script is kept for reference with a
> deprecation banner; do not use it.
>
> The empirical hard requirement (validated by the on-board
> `nbglk_valid_nbg_check`) is `target=0x1000003B`. The NBG format
> version (bytes 4..7) is **not** strictly required to be `0x1001E` as
> an earlier draft claimed: Allwinner's own ai-sdk v3 examples ship
> NBGs in both v1 (`0x1001E` / `0x10020`) and v2 (`0x20000`) formats
> against the same chip; viplite v2.0.3+ loads both.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Source ONNX | `tools/kata_export_for_rknn.py` + `onnxsim` + `tools/onnx_rknn_mitigations.py --unshare-initializers` (same chain as the RKNN target). |
| Toolkit | **Allwinner's `ubuntu-npu:v2.0.10.1` Docker image** (2.9 GB). NOT pip `acuitylite==6.51.0` — its bundled chip table is missing A733's PID. |
| Host requirement | **Real x86_64 Linux** with Docker. A nested container (Docker-in-Docker without `--privileged`) can compile `gen_nbg` but its VIP9000 simulator `vsi_nn_CreateGraph()` fails silently — see §6.3. |
| `VSIMULATOR_CONFIG` | `VIP9000NANODI_PID0X1000003B` (NOT `_PLUS_` — Allwinner's `pegasus_setup.sh v3` says `_PLUS_` but the actual `.config` file shipped in the image has the no-`_PLUS_` name). |
| fp16 path | `bash tools/onnx_to_a733_docker.sh <BS>` → `models/kata1-b10c128.a733.bs<BS>.fp16/network_binary.nb`. Lossless vs ORT but only 2 NN-core ops at bs=1 (zero at bs=4) → ~80 ms / call on board. Use only as a sanity baseline. |
| int8 path (production) | `bash tools/a733_gen_calib.py … && bash tools/onnx_to_a733_quantize_docker.sh <BS>` → `models/kata1-b10c128.a733.bs<BS>.int8/network_binary.nb`. 46–144 NN-core ops, **~110× faster than fp16** on board. **bs=4 is the production default** (top-1 79% / top-5 100% vs ORT). See §3.5 for the calibration recipe and the inputmeta trap that takes top-1 from 87% → 3% if you miss it. |
| Output check | `xxd network_binary.nb \| head -1` → bytes 0..3 must be `VPMN`, bytes 8..11 must be `3b 00 00 10` (target `0x1000003B`). Bytes 4..7 are the NBG format version (`0x20000` = NBG v2.0.0) — what v6.30.22 emits. |
| On-board check | `vpm_run -s sample.txt -l 5 -b 1` should print `create network 0: NNN us` and `vpm run ret=0`, NOT `[…]nbglk_valid_nbg_check[920], binary target=…` (see §7). |

**Bottom line for someone re-doing this on a real Linux host:**

```bash
# fp16 NBGs (cheap reference, slow on board)
bash tools/onnx_to_a733_docker.sh 1
bash tools/onnx_to_a733_docker.sh 4

# int8 NBGs (production, ~110× faster on board)
.venv/bin/python tools/a733_gen_calib.py \
    --output-dir build/a733_calib --num-games 80 \
    --onnx-model models/kata1-b10c128.a733.bs1.unshared.onnx \
    --strong-frac 0.7 --random-game-frac 0.2 --seed 1234
bash tools/onnx_to_a733_quantize_docker.sh 1
bash tools/onnx_to_a733_quantize_docker.sh 4

xxd models/kata1-b10c128.a733.bs1.fp16/network_binary.nb | head -1
xxd models/kata1-b10c128.a733.bs1.int8/network_binary.nb | head -1
# Both expect: 5650 4d4e 0000 0200 3b00 0010 ...   (VPMN, ver 0x20000, target 0x1000003B)
```

If those bytes match, ship the `.nb` to the Cubie A7A. The runtime
resolver in `src/vip9000_compute.cpp` picks `.int8/` first and falls
back to `.fp16/`; set `VIP9000_FORCE_PRECISION=fp16` to pin the
slower path for an A/B comparison. Run §7 to verify on-board.

---

## 1. What this is and isn't

| In scope | Out of scope |
|---|---|
| Converting kata1 ONNX → `.nb` on an x86_64 Linux host with Docker | The `awnn` / VIPLite runtime on the Cubie A7A board (covered by README.md "VIP9000 NPU Backend" section) |
| Producing an NBG that the on-board viplite v2.0.3.2-AW-2024-08-30 accepts at `vip_create_network` | The C++ `Vip9000ComputeHandle` backend in `src/vip9000_compute.cpp` (works once the NBG header validates and the `.quantize` table is correct) |
| Both the **fp16** path (sanity baseline) and the **int8 hybrid** path (production — ~110× faster) | Int16 / w8a16 / pcq variants — int8 hybrid is the right point on this NPU's accuracy/speed curve |
| Why pip `acuitylite==6.51.0` cannot do either path | The `build/vip9000_accuracy` harness and `VIP9000_FORCE_PRECISION` env var — those live on the board side |

The artifacts this doc is responsible for producing:

```
models/kata1-b10c128.a733.bs<N>.fp16/network_binary.nb        (slow but lossless)
models/kata1-b10c128.a733.bs<N>.fp16/nbg_meta.json
models/kata1-b10c128.a733.bs<N>.int8/network_binary.nb        (production)
models/kata1-b10c128.a733.bs<N>.int8/nbg_meta.json
models/kata1-b10c128.a733.bs<N>.int8/<base>_int8.quantize     (calibration table)
```

---

## 2. Why fp16, why Vivante, what's the NPU

The Cubie A7A's Allwinner A733 SoC carries one **VeriSilicon
VIP9000 NanoDI** NPU core, single shader core, ~3 INT8 TOPS rated.
The on-board `cat /sys/kernel/debug/viplite/vip_info` reports:

```
dev0 hw0 info: pid=0x1000003b, date=0x20230518, ver1=0x9000, ver2=0x9202.
```

The corresponding host-side simulator config file (in the Acuity
image at `/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/cfg/`)
is `VIP9000NANODI_PID0X1000003B.config`:

```
chipModel     = 0x00009000;
chipRevision  = 0x00009202;
productID     = 0x05090009;
ecoID         = 0x08000000;
customerID    = 0x1000003b;
```

`chipRevision=0x9202` matches the on-board `ver2=0x9202`; `customerID`
matches the `pid` reported by viplite. Confirmation that v6.30.22's
chip table knows this part: `strings libEmulator.so | grep
VIP9000NANODI_PID0X1000003B` returns a match (it's in both the
image's `Vivante_IDE/.../vsimulator/lib/libEmulator.so` and the
acuitylib wheel's `vsi_sdk/prebuilt-sdk/x86_64_linux/lib/libEmulator.so`).

VIP9000's spec sheet says fp16 runs at the same rate as int8 — but on
this particular A733 silicon and toolkit pairing, kata1's fp16 NBG
schedules ~93% of its ops onto the slow shader/PPU path (89 SP + 71
SH + 29 NN_LUT vs only 2 NN-core ops at bs=1; bs=4 fp16 puts **zero**
ops on the NN cores). The int8 NBG produced by the §3.5 quantize
path schedules 46–144 ops onto the NN cores (per `nbinfo -o`), and
on-board it runs **~110× faster than fp16** (see README §VIP9000
backend: 0.72 ms vs 80.8 ms per bs=1 call). The README has the
detailed perf table; this doc is the converter-side recipe.

Contrast with the RK3576/3588 path documented in `RKNN_CONVERSION.md`
§11.2 where int16 runs at ⅓ rate and w8a16 has codegen bugs.

---

## 3. The working conversion pipeline

Performed on a **real x86_64 Linux host with Docker** (NOT a nested
container — see §6.3 for why). The recipe below assumes Ubuntu 22.04
with an empty home directory; adjust apt for other distros.

### 3.0 Bring-up on a fresh box (one-time)

If you're regenerating on a host that doesn't already have Docker /
the right Python deps:

```bash
# Docker engine — official Ubuntu repo
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl gnupg lsb-release python3-venv python3-pip unzip p7zip-full zip
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Python venv with just enough to drive the ONNX export.
# The Acuity toolkit itself runs inside the Docker image — we don't
# pip-install it here.
cd /root/proj/AlphaZero
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-cache-dir onnx onnxsim onnxruntime numpy
```

Disk: the pipeline needs ~25 GB free during the Allwinner image fetch
and load (2.9 GB compressed share, 7.4 GB tar, ~16 GB once Docker
expands it). The ONNX export needs ~12 GB RAM peak (PyTorch CPU forward
pass at bs=4); a 6 GB / 4 GB-swap box is enough. Total wall-clock from
scratch on a residential connection: ~25 min, dominated by the image
download.

### 3.1 Source ONNX (same as the RKNN path)

This step is shared with the RKNN target — no A733-specific changes.

```bash
# Get kata1 weights (one-time; ~14 MB)
curl -L -o kata1-b10c128-s1141046784-d204142634.txt.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s1141046784-d204142634.txt.gz

# Use the venv created in §3.0 (or your existing alphazero conda env —
# either works as long as torch+onnx+onnxsim are importable)
PY=.venv/bin/python   # or: PY=python   inside `conda activate alphazero`

for bs in 1 4; do
    $PY tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --batch $bs --opset 13 \
        --output models/kata1-b10c128.a733.bs${bs}.onnx

    $PY -c "import onnx; from onnxsim import simplify; \
        m,_=simplify(onnx.load('models/kata1-b10c128.a733.bs${bs}.onnx')); \
        onnx.save(m,'models/kata1-b10c128.a733.bs${bs}.onnx')"

    $PY tools/onnx_rknn_mitigations.py \
        --input  models/kata1-b10c128.a733.bs${bs}.onnx \
        --output models/kata1-b10c128.a733.bs${bs}.unshared.onnx \
        --unshare-initializers
done
```

Output: `models/kata1-b10c128.a733.bs{1,4}.unshared.onnx`. These are
~12 MB each, opset 13, static batch (1 or 4), with KataGo's
BN.weight≡running_var alias split into per-consumer copies. ORT
forward output is bit-identical pre/post the unshare step (verified
by `tools/onnx_rknn_mitigations.py`'s parity check).

**Why opset 13 + 4D-gpool:** the older Acuity v6.30.22 importer
chokes on opset 17's tensor-name patterns and on the 2D ReduceMean
gpool topology. The `tools/kata_export_for_rknn.py` script monkey-
patches `katago_arch._gpool_stats` to use `GlobalAveragePool +
GlobalMaxPool` (4D) and exports at opset 13 — this is the same fix
that worked for RKNN (RKNN_CONVERSION.md §10.4) and it carries over
to A733.

**Why `--unshare-initializers`:** kata1 trains BN with
`track_running_stats=False`, so `BN.weight=1.0` and `running_var=1.0`
are both all-ones tensors. PyTorch's ONNX exporter dedupes them — 22
BN nodes end up sharing the same initializer in two slots
(`weight` and `running_var`) of every `BatchNormalization` node.
Acuity v6.30.22's ruler-matcher walks BN inputs by slot expecting
distinct tensors and dies with `TypeError: 'NoneType' object is not
iterable` on the **first** BN it sees (specifically
`policy_head/p1_bn` for kata1 because BFS hits it first). Splitting
the alias into per-consumer copies keeps forward output bit-identical
and bypasses the matcher bug. The same script is reused unchanged
from the RKNN bs=1 fp16-saturation fix (RKNN_CONVERSION.md §16).

### 3.2 Pull the Allwinner Docker image

The image isn't on Docker Hub or any public registry. It's distributed
as a zip on Allwinner's Synology netdisk:
[`https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq`](https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq)
(the page is JS-rendered Synology File Station — the browser path
opens a directory listing).

The unhelpful path is "open in browser, click download." The useful
path is the share's direct-download URL once you've grabbed a session
cookie:

```bash
# 1) Get a sharing_sid via the public Sharing.Login API
SID=$(curl -sL -G "https://netstorage.allwinnertech.com:5001/webapi/entry.cgi" \
    --data-urlencode "api=SYNO.Core.Sharing.Login" \
    --data-urlencode "version=1" \
    --data-urlencode "method=login" \
    --data-urlencode "sharing_id=Mh23BhPHq" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['sharing_sid'])")

# 2) Download (~2.9 GB, the literal filename suffix is ignored — the share
#    only has one downloadable archive at the root, served regardless of
#    what name you put after /fsdownload/<sharekey>/)
mkdir -p /tmp/aw && cd /tmp/aw
curl -L "https://netstorage.allwinnertech.com:5001/fsdownload/Mh23BhPHq/share.zip" \
    -b "sharing_sid=$SID" -o share.zip --progress-bar

# 3) Unwrap the nested archives. The outer share.zip contains
#    docker_images_v2.0.x/ which has the inner tar.zip plus an
#    ubuntu-npu_v2.0.10.1.tar.zip_md5sum.txt — verify the MD5 before
#    loading (~13 minutes lost if the tar is corrupt).
unzip -q share.zip                              # → docker_images_v2.0.x/
cd docker_images_v2.0.x
md5sum -c <(awk 'NF==2{print $1"  "$2}' ubuntu-npu_v2.0.10.1.tar.zip_md5sum.txt)
unzip -q ubuntu-npu_v2.0.10.1.tar.zip          # → ubuntu-npu_v2.0.10.1.tar (7.4 GB)

# 4) Load into Docker
docker load -i ubuntu-npu_v2.0.10.1.tar         # tags as ubuntu-npu:v2.0.10.1
docker images | grep ubuntu-npu                 # verify (~16 GB on disk)

# 5) Free the staging area — the tar/zip aren't needed after `docker load`
rm -rf /tmp/aw
```

The image bundles:

* `acuity-toolkit-whl-6.30.22/bin/pegasus.py` — the offline conversion
  driver (Python 3.8). Acuity v6.30.22 is the version we need;
  v6.51.0 (PyPI) does NOT have A733 in its chip table.
* `Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator/lib/` — host-side
  x86_64 OpenVX implementation (`libOpenVX.so`, `libGAL.so`,
  `libEmulator.so`, etc.) that lets the toolkit codegen + run the
  `gen_nbg` binary against a software simulator of the chip.
* `Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/cfg/<chip>.config` —
  per-chip simulator settings keyed by the `VSIMULATOR_CONFIG` env
  var. `VIP9000NANODI_PID0X1000003B.config` is the one we want.

`docker_images_v2.0.x.zip` also contains `ubuntu-npu_v2.0.10.1.tar.zip_md5sum.txt`
— check it before loading.

### 3.3 Run the conversion in the container

This is what `tools/onnx_to_a733_docker.sh` does. The annotated
sequence:

```bash
sudo docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    ubuntu-npu:v2.0.10.1 \
    /bin/bash -c '
        # ----- env setup -----
        # ldconfig: gen_nbg links against libGAL/libOpenVX/libovxlib (in
        # vsimulator/lib) AND libvdtproxy (in common/lib).  Both must be
        # findable.
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator/lib"   >  /etc/ld.so.conf.d/vivante.conf
        echo "/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/lib"       >> /etc/ld.so.conf.d/vivante.conf
        ldconfig

        # NPU target — the Allwinner pegasus_setup.sh v3 incorrectly
        # exports VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B with
        # an underscore-PLUS-underscore that does NOT match any .config
        # file shipped in the image.  The actual file is named without
        # _PLUS_; setting that name here makes the simulator find it.
        export VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B
        export VSIMULATOR_SHADER_CORE_COUNT=1
        export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin

        BS=${BATCH_SIZE:-1}
        BASE=kata1-b10c128.a733.bs${BS}.unshared
        WORK=/workspace/build/a733/${BASE}
        rm -rf "${WORK}"; mkdir -p "${WORK}"
        cp /workspace/models/${BASE}.onnx "${WORK}/${BASE}.onnx"
        cd "${WORK}"

        # ----- 1) import ONNX → Acuity IR (.json + .data) -----
        # Gotcha: the multi-input separator for --inputs is a SPACE,
        # but for --input-size-list it is "#".  Within one shape
        # tensor (e.g. 1,22,9,9) the dimensions are comma-separated.
        # Gotcha: --size-with-batch must be "true" per input; a
        # static-batch ONNX input like state_spatial[1,22,9,9] is
        # interpreted as N=1,C=22,H=9,W=9 only with this flag.
        # Without it, --input-size-list gets parsed as if the batch
        # dim were absent and the importer dies with IndexError in
        # smart_toolkit.ShapeComputation._conv_shape on the first Conv.
        python3 ${ACUITY_PATH}/pegasus.py import onnx \
            --model           ${BASE}.onnx \
            --output-model    ${BASE}.json \
            --output-data     ${BASE}.data \
            --inputs          "state_spatial state_global" \
            --input-size-list "1,22,9,9#1,19" \
            --outputs         "policy_logits value score_mean score_stdev ownership" \
            --size-with-batch "true#true"

        # ----- 2) generate inputmeta yaml -----
        python3 ${ACUITY_PATH}/pegasus.py generate inputmeta \
            --model              ${BASE}.json \
            --separated-database \
            --input-meta-output  ${BASE}_inputmeta.yml

        # ----- 3) export ovxlib + pack NBG -----
        # Gotchas:
        #   --dtype float       => fp16 NPU dataflow (Acuity vocabulary:
        #                          "float" means fp16-on-VIP9000)
        #   --pack-nbg-unify    => bake the .nb directly (otherwise
        #                          the toolkit only emits ovxlib C
        #                          scaffolding; you would have to
        #                          cross-compile + run on aarch64 to
        #                          get a .nb)
        #   --optimize <CFG>    => the SAME string as VSIMULATOR_CONFIG,
        #                          NOT a chip name and NOT an opt level
        #   --viv-sdk <DIR>     => points to <root>/cmdtools/vsimulator
        #                          (the toolkit appends /lib + /lib/x64_linux
        #                          internally)
        python3 ${ACUITY_PATH}/pegasus.py export ovxlib \
            --model              ${BASE}.json \
            --model-data         ${BASE}.data \
            --dtype              float \
            --target-ide-project linux64 \
            --with-input-meta    ${BASE}_inputmeta.yml \
            --pack-nbg-unify \
            --optimize           VIP9000NANODI_PID0X1000003B \
            --viv-sdk            /root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator \
            --output-path        wksp/${BASE}_fp16

        # ----- 4) collect NBG + meta -----
        # Acuity v6.30.22 with --pack-nbg-unify writes the .nb to a
        # *sibling* directory `<output-path>_nbg_unify/`, not under the
        # `--output-path` itself.  Search both — early drafts of the
        # script searched only `wksp/` and reported "ERROR: NBG not
        # produced" while the .nb was actually sitting in
        # `wksp_nbg_unify/`.
        OUT=/workspace/models/kata1-b10c128.a733.bs${BS}.fp16
        mkdir -p "${OUT}"
        NB=$(find wksp wksp_nbg_unify -name network_binary.nb -type f | head -1)
        META=$(find wksp wksp_nbg_unify -name nbg_meta.json    -type f | head -1)
        cp "$NB"   "${OUT}/network_binary.nb"
        cp "$META" "${OUT}/nbg_meta.json"
    '
```

End-to-end runs in ~30 s for bs=1, ~45 s for bs=4 on a typical x86
host (most of which is the Acuity import + tensorflow startup, not
the actual NBG codegen).

### 3.4 Verify the header

```bash
python3 -c "
import struct
for p in ['models/kata1-b10c128.a733.bs1.fp16/network_binary.nb',
          'models/kata1-b10c128.a733.bs4.fp16/network_binary.nb']:
    with open(p,'rb') as f: h = f.read(12)
    magic, ver, target = struct.unpack('<III', h)
    ok = (magic == 0x4e4d5056 and target == 0x1000003B)
    print(f'{p}: magic={magic:#x} version={ver:#x} target={target:#x}  {\"OK\" if ok else \"BAD\"}')
"
```

Expect (with v6.30.22 in `ubuntu-npu:v2.0.10.1`):

```
models/kata1-b10c128.a733.bs1.fp16/network_binary.nb: magic=0x4e4d5056 version=0x20000 target=0x1000003b  OK
models/kata1-b10c128.a733.bs4.fp16/network_binary.nb: magic=0x4e4d5056 version=0x20000 target=0x1000003b  OK
```

If `target=0x15` you've used the pip-`acuitylite` path — read §6.1
for what's wrong. The `version=0x20000` is the NBG v2.0.0 format that
v6.30.22 produces; the toolkit's host-side `libOpenVX.so` advertises
both `nbglk_create_v1_video_memory` and `nbglk_create_v2_video_memory`,
and Allwinner's own ai-sdk ships v3 sample NBGs in both formats
(`examples/vpm_run/operator/v3/network_binary.nb` is `0x1001E`,
`examples/yolact/model/v3/yolact.nb` is `0x20000`). Both load fine on
viplite v2.0.3+.

### 3.5 Quantise to int8 hybrid (the production path)

The fp16 NBGs from §3.3 are bit-exact vs ORT but ~93% of their ops
land on the slow shader/PPU path because the VIP9000 NN cores want
int8 dataflow. Quantising the trunk convs to int8 keeps the heads
near-fp16 quality and lights up the NN cores — empirically ~110×
faster on board (README §VIP9000 backend) with top-1 87% / top-5
100% on a 100-position held-out eval at bs=1; bs=4 closes the gap
further.

**Step 3.5.1 — generate calibration data** with `tools/a733_gen_calib.py`.
Plays 9×9 self-play games using kata1's own ONNX policy at
`temperature=1.0` (so the resulting positions are *in-distribution*
for kata1's activations), with 20% of whole games dropped to a
pure-random eye-aware policy for tactical-position diversity.
Snapshots are taken at moves 2/5/8/12/16/21/28/36/45/55/65/75 to
cover opening, midgame, and endgame evenly. Encoder is a faithful
port of `src/katago_inputs.cpp` (planes 14-17 / 20-21 zeroed,
matching the C++ version).

```bash
.venv/bin/python tools/a733_gen_calib.py \
    --output-dir   build/a733_calib \
    --num-games    80 \
    --onnx-model   models/kata1-b10c128.a733.bs1.unshared.onnx \
    --strong-frac  0.7 \
    --random-game-frac 0.2 \
    --temperature  1.0 \
    --seed         1234
# → 960 samples (≈ 320 opening / 400 midgame / 240 endgame), each as
#   build/a733_calib/state_spatial/NNNN.tensor   (np.savetxt-readable
#   ASCII fp32, 1782 floats for [22,9,9])
#   build/a733_calib/state_global/NNNN.tensor    (19 floats for [19])
# plus dataset0_spatial.txt / dataset1_global.txt index files.
```

**Step 3.5.2 — run quantize + export** with
`tools/onnx_to_a733_quantize_docker.sh`. The wrapper handles three
non-obvious traps inside the container — see §6.6 for the full
post-mortem; the script gets each one right:

* It overwrites Acuity's auto-generated `<name>_inputmeta.yml` with a
  hand-written one that sets `category: undefined` and
  `reverse_channel: false`. The default `category: image` reorders
  the 22 spatial planes like an RGB→BGR swap, calibrating against
  scrambled input and producing a `.quantize` whose scales don't
  match anything the runtime feeds (the on-board doesn't apply
  inputmeta preprocessing). This single bit takes top-1 from
  87% → 3%.
* It seeds an empty `{"version":1,"tensors":[]}` `.quantize` file
  before calling `pegasus.py quantize --rebuild-all`. Without this,
  pegasus errors out at *"quantize file '...' does not exist"* before
  any rebuild logic runs.
* Calibration files use TEXT mode + `np.savetxt`-style ASCII
  (`np.loadtxt` is what Acuity's TextDataset loader calls under the
  hood). `type: NPY` mode silently fails with *"Cannot load file
  containing pickled data when allow_pickle=False"* even on plain
  float arrays.

```bash
# bs=4 is the production default (top-1 79% / top-5 100% on board)
QUANTIZER=perchannel_symmetric_affine ALGO=kl_divergence ITER=400 \
  bash tools/onnx_to_a733_quantize_docker.sh 4

# bs=1 — keep around for the latency-sensitive path
QUANTIZER=perchannel_symmetric_affine ALGO=kl_divergence ITER=400 \
  bash tools/onnx_to_a733_quantize_docker.sh 1
```

`--quantizer perchannel_symmetric_affine` gives each conv weight a
per-output-channel scale (per-tensor `asymmetric_affine` on weights
loses ~1% top-1 because the single scale must cover all 128 output
channels of every conv). Activations stay per-tensor int8 asymmetric
either way — that's correct for activations.

After both batch-size runs you'll have:

```
models/kata1-b10c128.a733.bs1.int8/network_binary.nb     ~2.5 MB (vs 6.3 MB fp16)
models/kata1-b10c128.a733.bs1.int8/nbg_meta.json
models/kata1-b10c128.a733.bs1.int8/<base>_int8.quantize  the calibration table
models/kata1-b10c128.a733.bs4.int8/network_binary.nb     ~6.6 MB (vs 6.9 MB fp16)
models/kata1-b10c128.a733.bs4.int8/nbg_meta.json
models/kata1-b10c128.a733.bs4.int8/<base>_int8.quantize
```

The `.quantize` companion file is the per-tensor scale/zp table that
the on-board input-quantize / output-dequantize code in
`src/vip9000_compute.cpp` consumes (input encode = `q = round(x/scale) + zp`,
output decode = `x = (q - zp) * scale`). Always ship it alongside the
`.nb`.

The script's final stage runs `nbinfo -o` on the produced NBG and
prints the op-engine breakdown. A correctly quantised int8 NBG shows
**46–48 NN ops at bs=1, 140-148 at bs=4**, vs **0–2** for the fp16
path. If the breakdown still looks fp16-shaped, the calibration
silently failed — check the `.quantize` file size (should be ~230 KB
for kata1-b10c128, not the 30-byte stub).

### 3.6 Validate accuracy host-side before shipping

`pegasus.py inference --dtype quantized` runs the network through the
same VIP9000 simulator the on-board NBG executes on. Generate a
held-out eval set with a *different* `--seed` than the calibration:

```bash
.venv/bin/python tools/a733_gen_calib.py \
    --output-dir build/a733_eval --num-games 30 \
    --onnx-model models/kata1-b10c128.a733.bs1.unshared.onnx \
    --strong-frac 0.7 --random-game-frac 0.2 --seed 99999
```

…then drive `pegasus.py inference` with the eval set and compare its
outputs to `onnxruntime` on the same inputs (top-1/3/5 policy
agreement, value/score MAEs). The exact recipe lives in commit
history under `/tmp/eval_quant.sh` from the calibration debug
session; folding it into a checked-in tool is on the TODO list. A
healthy bs=1 int8 should hit ≥85% top-1 / ≥99% top-3 on this eval
set, with score_mean MAE under 1.5 pts.

Wraps §3.3 plus a guard for the image-load step. Default behaviour:
finds `models/kata1-b10c128.a733.bs${BS}.unshared.onnx` (where
`BS=$1`, default 1), produces `models/kata1-b10c128.a733.bs${BS}.fp16/`.

```bash
bash tools/onnx_to_a733_docker.sh 1   # build bs=1 NBG
bash tools/onnx_to_a733_docker.sh 4   # build bs=4 NBG
```

Prereqs (the script checks them):

* `docker` available and the calling user is in the `docker` group
  (or the script is run with `sudo`).
* `ubuntu-npu:v2.0.10.1` already loaded (`docker images | grep
  ubuntu-npu`). The script does not pull/load — that's a one-time
  step described in §3.2 since the image isn't on a public registry.
* The expected `.unshared.onnx` exists. If it doesn't, the script
  prints the §3.1 commands to produce it.

The script also xxd's the resulting `.nb` header and prints PASS/FAIL
based on the byte check from §3.4.

---

## 5. Pre-built artifacts

Once the §3 pipeline produces a header-validated NBG, pack everything
the on-board side needs in one archive (~32 MB combined):

```bash
zip -r kata1-b10c128.a733.fp16.zip \
    models/kata1-b10c128.a733.bs1.unshared.onnx \
    models/kata1-b10c128.a733.bs1.fp16/ \
    models/kata1-b10c128.a733.bs4.unshared.onnx \
    models/kata1-b10c128.a733.bs4.fp16/

# Or the smaller-but-slower 7z version (~24 MB):
for bs in 1 4; do
    7za a -t7z -mx=9 -m0=lzma2 -ms=on \
        kata1-b10c128.a733.bs${bs}.fp16.7z \
        models/kata1-b10c128.a733.bs${bs}.unshared.onnx \
        models/kata1-b10c128.a733.bs${bs}.fp16/
done
```

The archives are gitignored (`.gitignore: *.zip`, `*.7z`). On the Cubie:

```bash
# Cubie A7A side
scp kata1-b10c128.a733.fp16.zip cubie:~/
ssh cubie 'cd /root/proj/AlphaZero && unzip -o ~/kata1-b10c128.a733.fp16.zip'
# The .nb lands at models/kata1-b10c128.a733.bs1.fp16/network_binary.nb;
# the C++ runtime (vip9000_compute.cpp) finds it via extension swap on
# LoadedModel::model_path.  See README.md "VIP9000 NPU Backend"
# (path-resolution table) for the exact rules.
```

---

## 6. What we tried that doesn't work — and why

### 6.1 Pip `acuitylite==6.51.0` (the path currently in `tools/onnx_to_a733.py`)

This is the first thing we tried because it's a simple `pip install`
in the existing alphazero conda env. It produces an NBG that **the
on-board viplite v2.0.3.2-AW-2024-08-30 rejects at load time** with:

```
[0xaab16750]nbglk_valid_nbg_check[920], binary target=0x15, actually target=0x1000003B
[0xaab16750]vip_create_network[193], fail to create network status=-4
```

Why: the wheel's bundled simulator at
`<wheel>/acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux/lib/libEmulator.so`
has a **chip table that does not contain A733's PID `0x1000003B`**.

```bash
$ strings .../acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux/lib/libEmulator.so \
    | grep -E "PID0X|VIP9000"
CC8000L_PID0X55
CC8000_PID0X51
...
# Nothing matching VIP9000NANODI_PID0X1000003B.
```

When `VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B` is unmatched
the codegen **silently falls back to a generic VIP9000 default** and
writes the fallback's PID (`0x15`, low-byte truncated) into the NBG
header. There's no env var or kwarg to override this — the chip
table is compiled into the closed-source `.so`, and PyPI doesn't have
a newer wheel that includes A733 (6.51.0 is the latest as of
2026-04-29; available versions are 6.51.0, 6.48.0, 6.45.0, 6.42.0 —
none have A733 in their chip tables; verified by the same `strings`
check on each).

The pip path also writes NBG **format version `0x00020005`**, but
even after rewriting bytes 8..11 of the emitted NBG to `3b 00 00 10`
(target patched), `vip_create_network` still fails with `status=-4`:
the chip ID is just one of several encoded fields, and the underlying
compiled command stream was generated for the wrong ISA variant. The
header rewrite isn't a useful workaround.

(NB: an earlier draft of this section claimed viplite v2.0.3 strictly
needs format `0x0001001E`. That was wrong — Allwinner's own
`yolact/v3` sample ships at `0x00020000` and loads fine. The Docker
path's NBG is `0x20000` and was verified on-board with `vpm_run` ret=0
on 2026-04-29. So both v1 and v2 NBGs work; the only true requirement
is the right chip ID, which the pip path bungles.)

`tools/onnx_to_a733.py` and `tools/a733_verify.py` are kept in the
repo for reference but they should not be used for production
artifacts. The top of `onnx_to_a733.py` has been updated with a
deprecation banner.

### 6.2 PyPI alternates and acuitylite version sweep

Searched PyPI for `ovxlib`, `vivante`, `pegasus-acuity`, `awnn`. The
only thing that exists is `awnn` 0.0.2 which is unrelated (a Python
package by a third party, not Allwinner's awnn runtime). Nothing on
PyPI ships A733's chip table.

GitHub mirrors checked: `airockchip/rknn-toolkit2` (RKNN, wrong
target), `VeriSilicon/acuity-models` (model zoo, no toolkit),
`ZIFENG278/ai-sdk` (the user-side ai-sdk, wraps pegasus shell scripts
but doesn't bundle the toolkit binaries). None contain a
publicly-redistributable Acuity build.

Conclusion: **the only obtainable Acuity with A733 support is
Allwinner's `ubuntu-npu:v2.0.10.1` Docker image**, distributed via
the Synology netdisk URL in §3.2.

### 6.3 Trying to run Allwinner's Acuity in a nested container

If your build host is itself a Docker container (e.g., a CI runner,
a vast.ai instance, a cloud dev container), you'll hit a wall at
`gen_nbg` even with the right toolkit. We got:

| Stage | Status in nested container |
|---|---|
| pull and `docker load` (with `dockerd --storage-driver=vfs`) | ✅ |
| extract image layers manually with `tar` and chroot in | ✅ |
| `pegasus.py import onnx` (compiled Python only) | ✅ |
| `pegasus.py generate inputmeta` | ✅ |
| `pegasus.py export ovxlib` (without `--pack-nbg-unify`) | ✅ produces ovxlib C scaffolding |
| `gcc` of the scaffolding into `gen_nbg` binary | ✅ |
| **`gen_nbg` actually running and creating an OpenVX graph** | ❌ `vsi_nn_CreateGraph()` returns NULL silently |

The simulator inside `libGAL.so` / `libOpenVX.so` needs `/proc`
mounted (it parses `/proc/cpuinfo` and tries to mmap `/proc/self/...`
files). Inside a nested container without `CAP_SYS_ADMIN`, you can't
`mount -t proc` even into a chroot, and bind-mounting from outer
`/proc` doesn't propagate the kernel proc semantics the simulator
expects. We do have `CAP_SYS_CHROOT` and `CAP_MKNOD` (so chroot +
manual `/dev/{null,zero,urandom,...}` via `mknod` works for the
Python parts), but not the others.

`proot -q qemu-x86_64-static` (userspace ptrace-based fakechroot
that's supposed to bypass `noexec` and friends) ran but `gen_nbg`
**segfaulted** during the data-packing phase — likely qemu's signal
handling colliding with libGAL's ptrace-using simulator threads.

`unshare --user --mount --propagation=private chroot` failed at
`unshare: Operation not permitted` — the outer container doesn't
expose user namespaces.

Conclusion: this conversion needs **a real Linux host with Docker
running on a real kernel**, not a nested container. (Or, if you're
on a nested container with `--privileged=true`, that should work
too — but vast.ai / typical CI runners don't grant that.)

### 6.4 Patching the broken NBG bytes

Tempting after reading §6.1. Doesn't work:

```bash
# Patch the chip ID bytes in place
python3 -c "import struct; f=open('p.nb','r+b'); f.seek(8); f.write(struct.pack('<I',0x1000003B))"
# On the Cubie:
$ vpm_run -s sample.txt -l 1
[0x88cfd750]vip_create_network[193], fail to create network status=-4
```

Same `status=-4`. The NBG carries an entire compiled command stream
encoded for a different ISA variant; a single 4-byte header rewrite
doesn't change that, and the kernel ioctl validates the command
stream too. Don't bother.

### 6.5 TIM-VX as a workaround

TIM-VX is VeriSilicon's open-source aarch64-native runtime that JIT-
compiles graphs at load time — no offline NBG step. It would skip
the entire chip-table problem because the JIT runs against the actual
hardware's chip ID. But our project's `nn_evaluator.cpp` /
`Vip9000ComputeHandle` architecture expects an NBG-shaped backend
(per-K precompiled, fixed batch, single `predict_batch` call), and
TIM-VX is a graph-JIT runtime — adapting would be a significant
refactor, out of scope for the conversion side.

If you're starting from scratch and don't have the C++ runtime
constraints, TIM-VX is a viable alternative. We documented this for
completeness, not because we recommend it for this project.

### 6.6 The inputmeta `category: image` trap (took int8 top-1 from 87% → 3%)

The first int8 NBG we shipped to the board got **3% top-1, 0.73 value
MAE, 27 pt score MAE** on `build/vip9000_accuracy` — essentially
chance-level outputs (commit `093a1ff`). On-board input-quantize was
correct (verified end-to-end with the fp16 NBG running on the same
data path); the on-board NBG header was correct. The bug was on the
host, in calibration:

```yaml
# Auto-generated by `pegasus.py generate inputmeta` for kata1:
ports:
- lid: state_spatial_146
  category: image          # ← BAD: the auto-default for any input
  layout: nchw
  shape: [1, 22, 9, 9]
  preprocess:
    reverse_channel: true  # ← BAD: the auto-default with category=image
    preproc_node_params:
      add_preproc_node: false
      preproc_type: TENSOR
```

`add_preproc_node: false` means *"don't fuse a preprocessing op into
the graph,"* so we expected `reverse_channel: true` to be a no-op
during export. **But it isn't a no-op during quantize.** Acuity's
quantize pass applies the inputmeta preprocessing to every
calibration sample before observing activations, then writes the
observed scales/zps to the `.quantize` file. With
`reverse_channel: true`, all 22 spatial planes were reordered (think
RGB→BGR but on a 22-channel tensor): plane 0 (on-board mask, all-1)
became plane 21 (encore, all-0), every stone color got mirrored,
liberty planes ended up where ladder planes should be, etc. The
network still ran — it just saw an entirely different input
distribution than it ever saw at training time, so the activation
ranges the calibrator measured were nonsense.

On the runtime side, `src/vip9000_compute.cpp` does NOT apply
inputmeta preprocessing — it feeds the user's fp32 directly into the
NBG, then quantizes per the `.quantize` table. So the calibration
distribution and the inference distribution disagreed on every
plane. fp16 was unaffected (no `.quantize` file → no host-side scale
applied to a model trained without one).

The fp16 NBG passed `vpm_run` and produced sensible outputs because
*its export path* honoured `add_preproc_node: false` and didn't bake
the channel-reverse into the network. Only the `.quantize` table on
the int8 path carried the corrupted ranges.

**Fix**: write the inputmeta from scratch with `category: undefined`
and `reverse_channel: false`, which is what
`tools/onnx_to_a733_quantize_docker.sh` now does. Held-out eval went
from 3.0% → 87.0% top-1 on bs=1 int8 (with the same calibration
data; it was the inputmeta alone). bs=4 went 22.5% → 85% on the
same calibration, then 79% → 99% top-5 once we also broadened the
calibration to use kata1-policy-driven self-play instead of pure
random play.

This trap doesn't show up anywhere in Allwinner's ai-sdk samples
because every one of them is an actual image network (lenet,
resnet50, yolov5, yolact, MobileNetV2, ShuffleNetV2). For tensor
inputs, the auto-generated inputmeta is wrong by default and every
example you can crib from gets it "right" only because they're
3-channel networks where reverse_channel:true is just RGB↔BGR.

---

## 7. On-board verification

Once you have a header-validated NBG, smoke-test on the Cubie before
the full benchmark. The on-board side (build/vip9000_smoke + the C++
backend) is documented in README.md's "VIP9000 NPU Backend" section.
Both bs=1 and bs=4 NBGs from the Docker pipeline have been confirmed
to pass §7.1 on the Cubie A7A as of 2026-04-29.

### 7.1 `vpm_run` smoke test — VERIFIED PASSING

`vpm_run` is from the `ai-sdk` ([github.com/ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk)),
prebuilt at `examples/vpm_run/install/etc/npu/vpm_run/vpm_run` on the
Cubie. It's a standalone NBG loader — exercises the same
`vip_create_network` / `vip_run_network` path the C++ backend uses,
no minigo dependency.

```bash
ssh cubie

cat > /tmp/smoke.txt <<EOF
[network]
/root/AlphaZero/models/kata1-b10c128.a733.bs1.fp16/network_binary.nb
EOF

# bs=1 inputs: state_spatial[1,22,9,9] fp16 + state_global[1,19] fp16
#            = (22*9*9 + 19) * 2 bytes = 3602 bytes total across both inputs
# vpm_run wants per-input zeros if you skip its input-comparison step.

export LD_LIBRARY_PATH=/root/proj/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
/root/proj/ai-sdk/examples/vpm_run/install/etc/npu/vpm_run/vpm_run \
    -s /tmp/smoke.txt -l 5 -b 1 2>&1 \
    | grep -E "create network|prepare network|profile inference|target=|status=|ret="
```

PASS pattern:

```
create network 0: 1944 us.
prepare network 0: 1064 us.
profile inference time=2855us, cycle=2853769
profile inference time=...
profile inference time=...
profile inference time=...
profile inference time=...
vpm run ret=0
```

FAIL pattern (the broken-pip NBG):

```
[0xaab16750]nbglk_valid_nbg_check[920], binary target=0x15, actually target=0x1000003B
[0xaab16750]vip_create_network[193], fail to create network status=-4
Network creating failed.
vpm run ret=-1
```

Anything starting with `[0x...]nbglk_valid_nbg_check` or
`fail to create network` means the NBG didn't pass validation.
Re-check §3.4's xxd byte check on the host.

### 7.2 Full minigo benchmark

After §7.1 passes:

```bash
cd /root/proj/AlphaZero
./build/benchmark --model models/kata1-b10c128.a733.bs1.unshared.onnx \
    --max-batch 1 --nn-iters 100 --games 0 --threads 0
```

The C++ backend (`src/vip9000_compute.cpp` on `multi-gpu` — not yet
checked in as of 2026-04-29) prints its sanity check at startup:

```
VIP9000 NBG header: target=0x1000003b version=0x20000 bytes=6336808
NNEvaluator thread 0 (gpu 0): VIP9000 OK
```

If you see `target=0x15 ... does not match this hardware's CID
0x1000003b`, the NBG you shipped is the broken pip one — back to
§3.

Expected throughput: ResNet-50 on this NPU is documented at ~1 TOPS
achieved (Anton Maltsev's blog, [zlodeibaal/radxa-cubie-a7a](https://medium.com/@zlodeibaal/radxa-cubie-a7a-f7401a185694)).
kata1-b10c128 has fewer MACs than ResNet-50, so single-thread
inference should land in the 3–5 ms range. With 1 NPU core and
per-thread `vip_dup_network`, throughput scales by thread count
roughly linearly until DRAM bandwidth saturates (typically 2–3
threads on this part).

---

## 8. The conversion gotchas, condensed

A reference of the non-obvious bits, for someone hitting one of them
later.

| Symptom | Root cause | Fix |
|---|---|---|
| pip `acuitylite==6.51.0` produces NBG with `target=0x15`, on-board rejects | Wheel's chip table missing `0x1000003B` | Use Allwinner Docker image (§3.2) |
| `pegasus import onnx` dies with `TypeError: 'NoneType' object is not iterable` on `policy_head/p1_bn` | kata1's BN.weight≡running_var aliasing | `tools/onnx_rknn_mitigations.py --unshare-initializers` |
| `pegasus import onnx` dies with `ValueError: invalid literal for int() with base 10: '9;1'` | Wrong separator in `--input-size-list` | Use `#` between inputs, `,` within shape (e.g. `1,22,9,9#1,19`) |
| `pegasus import onnx` dies with `IndexError: list index out of range` in `smart_toolkit._conv_shape` | Importer treats `--input-size-list` as no-batch by default | Add `--size-with-batch true#true` |
| `pegasus export ovxlib --pack-nbg-unify` dies with `please set correct target name in config file` | Toolkit looks for `<viv-sdk>/../common/cfg/<VSIMULATOR_CONFIG>.config`; the file with `_PLUS_` doesn't exist | Use `VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B` (no `_PLUS_`); pass the matching string to `--optimize` |
| `pegasus export ovxlib --pack-nbg-unify` dies with `Fatal model generation error: 32512` (or `127 << 8`) | gen_nbg can't find shared libs | `ldconfig` with `vsimulator/lib` AND `common/lib` paths added |
| `pegasus export ovxlib --pack-nbg-unify` dies with `Fatal model generation error: 65280` | gen_nbg ran but `vsi_nn_CreateGraph()` returned NULL | Real Linux host with `/proc` accessible; nested-container chroots don't work (§6.3) |
| `pegasus export ovxlib` finishes with `Error(0),Warning(0)` but the script reports "ERROR: NBG not produced" | Acuity v6.30.22 with `--pack-nbg-unify` writes the `.nb` to `wksp_nbg_unify/` (sibling), not under `--output-path`'s directory | Search both `wksp/` AND `wksp_nbg_unify/`; the current `tools/onnx_to_a733_docker.sh` does this |
| `pegasus.py quantize` errors out at *"quantize file '...' does not exist"* before doing anything | `--rebuild` / `--rebuild-all` both expect a pre-existing `.quantize` file | Seed an empty `{"version":1,"tensors":[]}` JSON before invoking; `tools/onnx_to_a733_quantize_docker.sh` does this |
| Quantize step finishes "successfully" but the resulting NBG has the same op breakdown as fp16 | The `.quantize` file is the empty 30-byte stub — quantize silently failed during calibration loading | Check `stat -c %s <name>_int8.quantize`; healthy is ~230 KB. Most common cause: `type: NPY` in inputmeta (use `type: TEXT` + `np.savetxt`-format ASCII files), or calibration paths not visible inside the container |
| Calibration runs but every per-channel logit / score / value max is way smaller than ORT's actual max | Acuity's auto-generated inputmeta has `category: image` + `reverse_channel: true`, which scrambles the 22 spatial planes during quantize even though `add_preproc_node: false` — see §6.6 | Replace the auto-generated yaml with `category: undefined` + `reverse_channel: false`; `tools/onnx_to_a733_quantize_docker.sh` writes a clean inputmeta from scratch |
| On-board int8 NBG returns near-chance top-1 (~3%), score MAE >20 pts, but fp16 NBG works fine | Almost always the `category: image` trap above | Re-quantize after fixing the inputmeta; expect ≥85% top-1 / ≤1.5 pt score MAE on a held-out eval set |
| `vpm_run` on the board prints `nbglk_valid_nbg_check[920], binary target=...` | NBG header chip ID doesn't match the actual NPU | Re-check §3.4 host-side header bytes; if those are right then the NBG was built against a different chip target — verify `--optimize` and `VSIMULATOR_CONFIG` both say `VIP9000NANODI_PID0X1000003B` |

---

## 9. File reference

| File | Role | Status |
|---|---|---|
| `tools/kata_export_for_rknn.py` | KataGo `.txt.gz` → ONNX (4D-gpool, opset 13). Shared with the RKNN target. | ✅ |
| `tools/onnx_rknn_mitigations.py` | `--unshare-initializers` (math-equivalent BN initializer split). Same script as the RKNN bs=1 fp16 fix. | ✅ |
| `tools/onnx_to_a733_docker.sh` | fp16 path entry point. Wraps the §3.3 pipeline. Run on a real Linux host with Docker + `ubuntu-npu:v2.0.10.1` loaded. | ✅ (sanity baseline) |
| `tools/a733_gen_calib.py` | Self-play calibration-data generator for the int8 path. Drives 70% of moves with kata1's own ONNX policy via `onnxruntime`, 30% random eye-aware (with 20% of whole games dropped to pure-random for tactical diversity). Snapshots cover opening / midgame / endgame evenly. Encodes via Python port of `src/katago_inputs.cpp`. | ✅ (production) |
| `tools/onnx_to_a733_quantize_docker.sh` | int8 path entry point. Generates a clean inputmeta (workaround for the §6.6 trap), seeds the `.quantize` file (workaround for the `--rebuild-all` requires-file trap), runs `pegasus.py quantize` then `export ovxlib --dtype quantized --pack-nbg-unify`, and reports the op-engine breakdown. Defaults to `perchannel_symmetric_affine` int8 weights + per-tensor int8 activations, `kl_divergence` algorithm, 400 iterations. | ✅ (production) |
| `tools/onnx_to_a733.py` | Pip-`acuitylite` path. Produces NBGs with `target=0x15` that the on-board runtime rejects. | ❌ DO NOT USE |
| `tools/a733_verify.py` | Host-side ORT vs Acuity simulator parity. Worked for the old pip path (just for fp16 noise floor); doesn't run inside the Docker conversion flow. | ⚠️ optional, host-only |
| `models/<base>.a733.bs<N>.unshared.onnx` | Source ONNX (canonical reference for downstream parity). | — |
| `models/<base>.a733.bs<N>.fp16/network_binary.nb` | fp16 NBG (slow on board — ~93% of ops fall through to PPU). Header bytes 8..11 must be `3b 00 00 10`. | — |
| `models/<base>.a733.bs<N>.int8/network_binary.nb` | int8 NBG (production — ~110× faster). Same header check. | — |
| `models/<base>.a733.bs<N>.int8/<base>_int8.quantize` | Per-tensor scale/zp table the on-board input/output quant code consumes. Always ship alongside the `.nb`. | — |
| `models/<base>.a733.bs<N>.{fp16,int8}/nbg_meta.json` | Per-NBG input/output names, shapes, dtypes. | — |
| README.md "VIP9000 NPU Backend" section | On-board runtime notes (the C++ backend side, including `VIP9000_FORCE_PRECISION` env var, perf table, and the `build/vip9000_accuracy` harness). Read in tandem with this doc. | — |

---

## 10. What's deliberately not here

* **Performance numbers on the actual hardware.** Live in README.md's
  VIP9000 backend section — fp16 vs int8 timing table, multi-thread
  scaling, batch-size sweep.
* **A checked-in `pegasus.py inference` accuracy harness.** The
  calibration debug session in 2026-04-30 used a one-off
  `/tmp/eval_quant.sh` + Python comparator to drive `pegasus.py
  inference --dtype quantized` against ORT and report top-K /
  per-head MAE. The on-board equivalent is `build/vip9000_accuracy`
  (commit `093a1ff`); a host-side checked-in version is on the TODO
  list.
* **Int16 / w8a16 / per-channel-float8 paths.** Briefly explored:
  int16 with `--quantizer dynamic_fixed_point` works (5 MB NBG, 42
  NN ops at bs=1) but doesn't materially beat int8-hybrid on
  accuracy and runs slower because the NN cores prefer int8
  dataflow. Skipped as a production option.
* **Deeper debugging of why nested containers can't run gen_nbg.**
  Documented as a wall in §6.3; not worth chasing further when a real
  host exists.

---

## 11. Sources

* On-board diagnostics that motivated this rewrite: README.md
  "VIP9000 NPU Backend" section (this repo, `multi-gpu` branch).
* Allwinner ai-sdk: <https://github.com/ZIFENG278/ai-sdk> (pegasus
  shell scripts in `scripts/`, viplite runtime libs in `viplite-tina/`).
* Acuity Toolkit Docker image archive: Allwinner Synology netdisk
  <https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq>.
* Radxa Cubie A7A NPU dev guide: <https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie_acuity_sdk>
  (the pages currently 404 in some browsers — content is loaded via
  Docusaurus JS).
* Anton Maltsev — practical Cubie A7A NPU notes:
  <https://medium.com/@zlodeibaal/radxa-cubie-a7a-f7401a185694>.
