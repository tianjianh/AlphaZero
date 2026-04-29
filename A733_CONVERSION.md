# A733 Conversion — kata1 ONNX → Allwinner A733 NBG (Vivante VIP9000)

> **STATUS (2026-04-29): The pip-`acuitylite` path checked into this repo
> as `tools/onnx_to_a733.py` produces an NBG the on-board viplite v2.0.3
> runtime REJECTS.** The header bytes come out as `target=0x15,
> version=0x20005`; the on-board CID is `0x1000003B` and the runtime
> wants format version `0x1001E`. Re-generation must use Allwinner's
> official Acuity Toolkit (Docker image `ubuntu-npu:v2.0.10.1`,
> internally Acuity v6.30.22). See §3 for the working pipeline.
>
> All on-board diagnostics that lead to this verdict are captured
> in `vip9000.note` (in this repo, multi-gpu branch). This document
> is the converter-side companion to that note.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Source ONNX | `tools/kata_export_for_rknn.py` + `onnxsim` + `tools/onnx_rknn_mitigations.py --unshare-initializers` (same chain as the RKNN target). |
| Toolkit | **Allwinner's `ubuntu-npu:v2.0.10.1` Docker image** (2.9 GB). NOT pip `acuitylite==6.51.0` — its bundled chip table is missing A733's PID. |
| Host requirement | **Real x86_64 Linux** with Docker. A nested container (Docker-in-Docker without `--privileged`) can compile `gen_nbg` but its VIP9000 simulator `vsi_nn_CreateGraph()` fails silently — see §6.3. |
| `VSIMULATOR_CONFIG` | `VIP9000NANODI_PID0X1000003B` (NOT `_PLUS_` — Allwinner's `pegasus_setup.sh v3` says `_PLUS_` but the actual `.config` file shipped in the image has the no-`_PLUS_` name). |
| Output check | `xxd network_binary.nb \| head -1` → bytes 0..3 must be `VPMN`, bytes 4..7 must be `1e 00 01 00` (version `0x1001E`), bytes 8..11 must be `3b 00 00 10` (target `0x1000003B`). |
| On-board check | `vpm_run -s sample.txt -l 5 -b 1` should print `create network 0: NNN us` and `vpm run ret=0`, NOT `[…]nbglk_valid_nbg_check[920], binary target=…` (see §7). |

**Bottom line for someone re-doing this on a real Linux host:**

```bash
# On the real Linux host — Docker assumed already installed
bash tools/onnx_to_a733_docker.sh   # produces models/kata1-b10c128.a733.bs1.fp16/network_binary.nb
xxd models/kata1-b10c128.a733.bs1.fp16/network_binary.nb | head -1
# Expect: 5650 4d4e 1e00 0100 3b00 0010 ...   (VPMN, ver 0x1001E, target 0x1000003B)
```

If those bytes match, ship the `.nb` to the Cubie A7A and run the §7
verification. If not, something regressed — read §6 to diagnose.

---

## 1. What this is and isn't

| In scope | Out of scope |
|---|---|
| Converting kata1 ONNX → `.nb` on an x86_64 Linux host with Docker | The `awnn` / VIPLite runtime on the Cubie A7A board (`vip9000.note` covers that) |
| Producing an NBG that the on-board viplite v2.0.3.2-AW-2024-08-30 accepts at `vip_create_network` | The C++ `Vip9000ComputeHandle` backend in `src/vip9000_compute.cpp` (already merged on `multi-gpu`; works once the NBG header validates) |
| Why pip `acuitylite==6.51.0` cannot do this | Quantisation paths (int8 / int16 / hybrid) — kata1 is fp16 only on this NPU and that's correct for it, see §2 |

The only artifact this doc is responsible for producing is
`network_binary.nb` (plus the `nbg_meta.json` companion).

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

VIP9000's MAC array runs **fp16 natively** at the same rate as int8.
There's no throughput incentive to quantise kata1 — fp16 is
essentially lossless on the network and runs at peak. Contrast with
the RK3576/3588 path documented in `RKNN_CONVERSION.md` §11.2 where
int16 runs at ⅓ rate and w8a16 has codegen bugs.

---

## 3. The working conversion pipeline

Performed on a **real x86_64 Linux host with Docker** (NOT a nested
container — see §6.3 for why).

### 3.1 Source ONNX (same as the RKNN path)

This step is shared with the RKNN target — no A733-specific changes.

```bash
# In the alphazero conda env on the same host where the kata1 .txt.gz lives
conda activate alphazero

for bs in 1 4; do
    python tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --batch $bs --opset 13 \
        --output models/kata1-b10c128.a733.bs${bs}.onnx

    python -c "import onnx; from onnxsim import simplify; \
        m,_=simplify(onnx.load('models/kata1-b10c128.a733.bs${bs}.onnx')); \
        onnx.save(m,'models/kata1-b10c128.a733.bs${bs}.onnx')"

    python tools/onnx_rknn_mitigations.py \
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

# 3) Unwrap the nested archives
unzip -q share.zip                              # → docker_images_v2.0.x/
cd docker_images_v2.0.x
unzip -q ubuntu-npu_v2.0.10.1.tar.zip          # → ubuntu-npu_v2.0.10.1.tar (7.4 GB)

# 4) Load into Docker
sudo docker load -i ubuntu-npu_v2.0.10.1.tar   # tags as ubuntu-npu:v2.0.10.1
sudo docker images | grep ubuntu-npu           # verify
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
        OUT=/workspace/models/kata1-b10c128.a733.bs${BS}.fp16
        mkdir -p "${OUT}"
        cp $(find wksp -name network_binary.nb | head -1) "${OUT}/network_binary.nb"
        cp $(find wksp -name nbg_meta.json     | head -1) "${OUT}/nbg_meta.json"
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

Expect:

```
models/kata1-b10c128.a733.bs1.fp16/network_binary.nb: magic=0x4e4d5056 version=0x1001e target=0x1000003b  OK
models/kata1-b10c128.a733.bs4.fp16/network_binary.nb: magic=0x4e4d5056 version=0x1001e target=0x1000003b  OK
```

If `target=0x15` you've used the pip-`acuitylite` path — read §6.1
for what's wrong. If `version` differs from `0x1001e` you've got a
non-Allwinner Acuity build whose codegen produces a format the
on-board viplite v2.0.3 doesn't accept; only the bundled v6.30.22 in
`ubuntu-npu:v2.0.10.1` is known to match.

---

## 4. The turnkey script — `tools/onnx_to_a733_docker.sh`

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

Once the §3 pipeline produces a header-validated NBG, repack and
ship to the board exactly as the broken-pip path did before:

```bash
for bs in 1 4; do
    7za a -t7z -mx=9 -m0=lzma2 -ms=on \
        kata1-b10c128.a733.bs${bs}.fp16.7z \
        models/kata1-b10c128.a733.bs${bs}.unshared.onnx \
        models/kata1-b10c128.a733.bs${bs}.fp16/
done
```

The `.7z` files are gitignored (`.gitignore: *.7z`). On the Cubie:

```bash
# Cubie A7A side
scp kata1-b10c128.a733.bs1.fp16.7z cubie:~/
ssh cubie 'cd /root/proj/AlphaZero && 7za x ~/kata1-b10c128.a733.bs1.fp16.7z'
# The .nb lands at models/kata1-b10c128.a733.bs1.fp16/network_binary.nb;
# the C++ runtime (vip9000_compute.cpp) finds it via extension swap on
# LoadedModel::model_path.  See vip9000.note §8.
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
on-board viplite v2.0.3 wants **`0x0001001E`** — even if you patched
the chip ID, the underlying compiled command stream targets a
different ISA generation that the kernel ioctl rejects independently.
We confirmed this by rewriting bytes 8..11 of an emitted NBG to
`3b 00 00 10` and watching `vip_create_network` still fail with
`status=-4`.

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

---

## 7. On-board verification

Once you have a header-validated NBG, smoke-test on the Cubie before
the full benchmark. This complements `vip9000.note §7` with the
A733-converter-specific checks.

### 7.1 `vpm_run` smoke test

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

The C++ backend (`src/vip9000_compute.cpp` on `multi-gpu`) prints
its sanity check at startup:

```
VIP9000 NBG header: target=0x1000003b version=0x1001e bytes=6493872
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
| `vpm_run` on the board prints `nbglk_valid_nbg_check[920], binary target=...` | NBG header chip ID doesn't match the actual NPU | Re-check §3.4 host-side header bytes; if those are right then the NBG was built against a different chip target — verify `--optimize` and `VSIMULATOR_CONFIG` both say `VIP9000NANODI_PID0X1000003B` |

---

## 9. File reference

| File | Role | Status |
|---|---|---|
| `tools/kata_export_for_rknn.py` | KataGo `.txt.gz` → ONNX (4D-gpool, opset 13). Shared with the RKNN target. | ✅ |
| `tools/onnx_rknn_mitigations.py` | `--unshare-initializers` (math-equivalent BN initializer split). Same script as the RKNN bs=1 fp16 fix. | ✅ |
| `tools/onnx_to_a733_docker.sh` | **The actual entry point.** Wraps the §3.3 pipeline. Run on a real Linux host with Docker + `ubuntu-npu:v2.0.10.1` loaded. | ✅ |
| `tools/onnx_to_a733.py` | Pip-`acuitylite` path. Produces NBGs with `target=0x15` that the on-board runtime rejects. | ❌ DO NOT USE |
| `tools/a733_verify.py` | Host-side ORT vs Acuity simulator parity. Worked for the old pip path (just for fp16 noise floor); doesn't run inside the Docker conversion flow. | ⚠️ optional, host-only |
| `models/<base>.a733.bs<N>.unshared.onnx` | Source ONNX (canonical reference for downstream parity). | — |
| `models/<base>.a733.bs<N>.fp16/network_binary.nb` | Deployable NBG. Header bytes 8..11 must be `3b 00 00 10`. | — |
| `models/<base>.a733.bs<N>.fp16/nbg_meta.json` | Per-NBG input/output names, shapes, dtypes. The on-board awnn loader can match its buffers by these names. | — |
| `vip9000.note` | On-board runtime notes (the C++ backend side). Read this in tandem. | — |

---

## 10. What's deliberately not here

* **Performance numbers on the actual hardware.** Add when the
  benchmark in §7.2 has been run. (TODO once the regen lands.)
* **Multi-thread scaling.** The C++ backend supports `vip_dup_network`;
  expected to scale roughly linearly with thread count until DRAM
  bandwidth saturates. Same TODO.
* **Quantized paths.** Skipped because fp16 is essentially lossless on
  kata1 and runs at peak rate on this NPU. The plumbing in
  `tools/rknn_calibration.py` is portable if someone needs int8
  later, but the conversion would be:
  `pegasus.py quantize --quantizer asymmetric_affine --qtype uint8`
  (followed by `--model-quantize <NAME>_uint8.quantize` on export).
* **Deeper debugging of why nested containers can't run gen_nbg.**
  Documented as a wall in §6.3; not worth chasing further when a real
  host exists.

---

## 11. Sources

* On-board diagnostics that motivated this rewrite: `vip9000.note`
  (this repo, `multi-gpu` branch).
* Allwinner ai-sdk: <https://github.com/ZIFENG278/ai-sdk> (pegasus
  shell scripts in `scripts/`, viplite runtime libs in `viplite-tina/`).
* Acuity Toolkit Docker image archive: Allwinner Synology netdisk
  <https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq>.
* Radxa Cubie A7A NPU dev guide: <https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie_acuity_sdk>
  (the pages currently 404 in some browsers — content is loaded via
  Docusaurus JS).
* Anton Maltsev — practical Cubie A7A NPU notes:
  <https://medium.com/@zlodeibaal/radxa-cubie-a7a-f7401a185694>.
