#!/usr/bin/env python3
"""ONNX → A733 NBG via the pip `acuitylite` wheel — **DO NOT USE**.

⚠️ DEPRECATED.  This script produced an NBG with header `target=0x15,
   version=0x00020005` that the on-board viplite v2.0.3.2-AW-2024-08-30
   on the Cubie A7A REJECTS at `vip_create_network` with:

      [...]nbglk_valid_nbg_check[920], binary target=0x15, actually target=0x1000003B
      [...]vip_create_network[193], fail to create network status=-4

   Root cause: pip `acuitylite==6.51.0`'s bundled simulator at
   `<wheel>/acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux/lib/libEmulator.so`
   has a chip table that does NOT contain A733's PID `0x1000003B`.  The
   codegen silently falls back to a generic VIP9000 default and bakes
   the wrong PID into the NBG header.  No env var or kwarg overrides
   this — the chip table is compiled into the closed-source .so, and
   PyPI doesn't ship a newer wheel with A733 support (verified across
   acuitylite 6.42.0, 6.45.0, 6.48.0, 6.51.0).

   THE WORKING PATH IS:  bash tools/onnx_to_a733_docker.sh

   Which runs Allwinner's official Acuity v6.30.22 inside the
   `ubuntu-npu:v2.0.10.1` Docker image (its libEmulator.so chip
   table DOES contain `VIP9000NANODI_PID0X1000003B`, the right
   chip for the Cubie A7A's A733).  See docs/A733_CONVERSION.md for the
   full pipeline and on-board diagnostics that demonstrated the
   rejection.

   The original implementation that this banner replaced is preserved
   in the git history of this file (commits 827de13 and earlier on
   the multi-gpu branch).
"""

import sys

print(
    "\n"
    "ERROR: tools/onnx_to_a733.py is deprecated.\n"
    "\n"
    "       It uses pip `acuitylite` whose bundled chip table does not\n"
    "       contain A733's PID 0x1000003B; the NBG it produces is\n"
    "       rejected by the on-board viplite runtime with status=-4 at\n"
    "       vip_create_network.\n"
    "\n"
    "       Use the Docker-based path instead:\n"
    "\n"
    "           bash tools/onnx_to_a733_docker.sh 1   # bs=1\n"
    "           bash tools/onnx_to_a733_docker.sh 4   # bs=4\n"
    "\n"
    "       That script needs a real x86_64 Linux host with Docker and\n"
    "       Allwinner's `ubuntu-npu:v2.0.10.1` image already loaded.\n"
    "       Read docs/A733_CONVERSION.md for the full procedure and context.\n",
    file=sys.stderr,
)
sys.exit(2)
