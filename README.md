# MiniXiangqi C++ — AlphaZero-Style Chinese Chess

This repository is a **Chinese chess (Xiangqi) AI** built from the original
MiniGo/KataGo-style C++ search architecture.  The preferred runtimes are
**TensorRT on Linux + NVIDIA** and **Metal on macOS Apple Silicon**; both are
auto-selected from `cmake`.

The project goal of this port was not to throw away the old engine shape and
start over. The goal was to keep the parts that matter architecturally:

- multi-threaded MCTS
- batched neural-network evaluation
- one `NNResultBuf` per search thread
- a persistent async bot worker thread
- tree reuse across moves
- a clean split between game rules, search, evaluator, and UI

What changed is the domain:

- **Go** became **Xiangqi**
- the board is now fixed at **10 rows × 9 columns**
- pass/komi/territory logic is gone
- moves are now **source-square → destination-square**
- the default backends are **TensorRT** (Linux + NVIDIA, SM 7.5+) and **Metal**
  (macOS Apple Silicon); `cmake` picks whichever is available
- the network is now a simpler **policy + value** residual net designed for Xiangqi

The `xqwlight_win32/` folder is treated as a **reference implementation for
rules and move semantics**. This port does **not** embed or link that Win32
engine directly. Instead, its rules were studied and reimplemented in the
project’s own game layer so the existing search/runtime architecture could be
preserved.

## Status

The current port is:

- built and verified on **macOS / Apple Silicon** (Metal)
- built and verified on **Linux + NVIDIA** (TensorRT 10 + CUDA 12)
- capable of:
  - interactive play in ncurses
  - self-play data generation
  - offline evaluation matches
  - PyTorch training
  - ONNX export

Verified locally during the port (on both platforms):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
python3 scripts/export_onnx.py --output /tmp/xiangqi_test.onnx
./build/selfplay --model /tmp/xiangqi_test.onnx --games 1 --threads 1 \
  --search-threads 1 --sims 4 --max-batch 4 \
  --nn-server-threads 1 --nn-device-ids 0
```

`cmake` auto-selects the backend:

- Apple Silicon → `metal`
- Linux + NVIDIA GPU (SM ≥ 7.5, TensorRT installed) → `tensorrt`
- Linux + NVIDIA GPU without TensorRT → `cuda`
- anything else with OpenCL → `opencl`
- fallback → `eigen` (CPU)

The smoke test above loads the exported ONNX model and runs one self-play game
through whichever backend was selected at build time.

## What Was Preserved

The port intentionally kept the original high-level architecture:

### 1. Threaded MCTS

`src/mcts.cpp` still uses:

- multiple search threads per root
- virtual loss during descent
- explicit node state transitions:
  - `NODE_UNEVALUATED`
  - `NODE_EXPANDING`
  - `NODE_EXPANDED`
- tree reuse with `make_move()`
- root Dirichlet noise

### 2. Batched Evaluator Server

`src/nn_evaluator.cpp` still follows the same design:

- search threads push leaf requests into a shared queue
- evaluator server threads pop batches
- one compute handle is owned by each server thread
- the queue self-balances naturally across evaluator threads

### 3. Async Bot Controller

`src/async_bot.cpp` still provides:

- one long-lived worker thread
- synchronous `gen_move`
- external `play_move`
- optional background ponder
- optional periodic analysis callbacks

### 4. C++/Python Split

The runtime remains:

- **C++** for rules, search, play, self-play, evaluation, inference integration
- **Python/PyTorch** for training and ONNX export

## What Changed

### 1. The Game

The original `GoGame` interface is now aliased to `XiangqiGame`:

- rules live in [include/game.h](include/game.h)
- implementation lives in [src/game.cpp](src/game.cpp)

The game layer now handles:

- 10×9 Xiangqi board state
- piece placement and startup position
- legal move generation
- legality filtering for self-check
- check detection
- checkmate/stalemate detection via “no legal move”
- repetition tracking
- history-plane feature encoding
- ICCS-like move string conversion (`a0a1` style)

### 2. Fixed Board Dimensions

The Go engine used a variable square board. Xiangqi does not.

The port now treats the board as fixed:

- rows: `10`
- cols: `9`
- area: `90`

This matters because a lot of the original code assumed:

- `board_size`
- square boards
- action count = `board_size * board_size + 1`

Those assumptions were replaced with:

- `board_rows`
- `board_cols`
- `board_area()`
- action count = `board_area * board_area`

Default config lives in [include/config.h](include/config.h).

### 3. Move Encoding

There is no pass move in Xiangqi.

Every move is encoded as:

```text
action = src_square * board_area + dst_square
```

with:

- `board_area = 90`
- action range = `[0, 8099]`

So the policy head size is:

```text
90 * 90 = 8100
```

### 4. Neural Network Heads

The original Go project had a larger multi-head network.

This Xiangqi port simplifies the model to:

- **policy logits**: `[B, 8100]`
- **value** (internally a 3-class **WDL** head → `[B, 3]` logits)

The ONNX export wrapper ([scripts/export_onnx.py](scripts/export_onnx.py))
collapses the WDL head into a single scalar `P(win) - P(loss)` via softmax,
so the graph has exactly two outputs:

- `policy_logits` : `[B, 8100]`
- `value`         : `[B, 1]` ∈ `[-1, 1]`

Backends read the scalar value directly.  The TensorRT backend also handles
the raw 3-logit form in case the wrapper is bypassed.

### 5. Training Data Format

The self-play file format changed from the old Go-oriented layout to a
rectangular-board Xiangqi format.

Current format:

```text
magic   : u16   = 0x4D47
version : u16   = 3
count   : i32
rows    : i32
cols    : i32

Per record:
  [state_size: i32]
  [state: f32 × state_size]
  [policy_size: i32]
  [policy: f32 × policy_size]
  [value: f32]
  [score: f32]
  [ownership: f32 × (rows * cols)]
  [opponent_action: i32]
```

Notes:

- `ownership` and `opponent_action` are still written for compatibility with the
  existing record structure, but the current training loop only consumes:
  - state
  - policy
  - value
  - score
- header `version = 3` is the key format discriminator for the Xiangqi port

## Xiangqi Rules Engine

The rules engine is the most important domain change in the repository.

### Implemented Piece Rules

The game layer enforces:

- **King / General**
  - one orthogonal step inside the palace
  - flying general capture if the file is clear
- **Advisor / Guard**
  - one diagonal step inside the palace
- **Bishop / Elephant**
  - two-point diagonal move
  - cannot cross the river
  - blocked by an “elephant eye”
- **Knight / Horse**
  - L-shaped move
  - blocked by a “horse leg”
- **Rook / Chariot**
  - sliding orthogonal move
- **Cannon**
  - rook-like movement without capture
  - must jump exactly one screen piece to capture
- **Pawn / Soldier**
  - one step forward before crossing river
  - may move sideways after crossing river

### Check Detection

`is_square_attacked()` checks attacks from:

- pawns
- knights
- rooks
- cannons
- opposing king on a clear file

### Terminal Conditions

The current engine ends the game when:

- the side to move has no legal move
  - winner is the opponent
- a repeated position inside the current non-capturing sequence reaches the
  XQWLight-style adjudication threshold
  - the perpetual checker loses
  - if both sides are perpetually checking, the result is a draw
  - if neither side is perpetually checking, the repeated position is a draw
- the self-play move limit is reached
  - forced draw

### Repetition And Long-Check

Repetition handling now follows the same core approach used by
`xqwlight_win32/XQWL06.CPP`:

- scan backward only through the current reversible sequence
  - the scan stops at the root or the most recent capture
- compare the current side-to-move hash against prior same-side-to-move hashes
- track whether the current side's repeating moves are all checks
- track whether the opponent's repeating moves are all checks

When the recurrence threshold is reached, adjudication is:

- current side perpetually checking: current side loses
- opponent perpetually checking: current side wins
- both sides perpetually checking: draw
- repeated position without perpetual checking on either side: draw

This keeps the game result aligned with XQWLight's long-check behavior instead
of collapsing every repeated position into a simple threefold draw.

## Feature Encoding

Default history length:

- `4`

Piece planes per snapshot:

- `14`
  - 7 piece types for side to move
  - 7 piece types for the opponent

Extra plane:

- `1` side-to-move plane

So default input channels are:

```text
4 * 14 + 1 = 57
```

Default input tensor shape:

```text
[batch, 57, 10, 9]
```

This encoding is produced by `XiangqiGame::encode()`.

## Neural Network Architecture

The Xiangqi network lives in [scripts/model.py](scripts/model.py).

Default model:

- residual CNN
- 10 residual blocks (mixed **SE** and **GPool** — Plain blocks are supported
  but not used by default)
- 128 trunk channels
- policy head:
  - 1×1 conv to 4 channels
  - batch norm + ReLU
  - fully connected layer to 8100 logits
- value head:
  - 1×1 conv to 2 channels
  - batch norm + ReLU
  - FC 256 (+ ReLU)
  - FC 3 (WDL logits)
  - the export wrapper adds a final softmax + `P(win) - P(loss)` to emit a
    scalar value in `[-1, 1]`

The ONNX loader in [src/loaded_model.cpp](src/loaded_model.cpp)
expects these parameter names:

- `input_conv.*`
- `input_bn.*`
- `res_blocks.N.conv1.*`
- `res_blocks.N.bn1.*`
- `res_blocks.N.conv2.*`
- `res_blocks.N.bn2.*`
- `policy_conv.*`
- `policy_bn.*`
- `policy_fc.*`
- `value_conv.*`
- `value_bn.*`
- `value_fc1.*`
- `value_fc2.*`

The exporter deliberately keeps initializer names stable so the lightweight C++
ONNX loader can recover weights without a heavyweight ONNX runtime dependency.

## Backends

### Metal Backend (macOS Apple Silicon)

Relevant files:

- [include/metal_compute.h](include/metal_compute.h)
- [src/metal_compute.mm](src/metal_compute.mm)

The Metal path uses:

- `MTLCreateSystemDefaultDevice()`
- `MPSGraph`
- per-server-thread compute handles
- FP32 input tensors
- graph-side conv/batchnorm/relu/fc execution

### TensorRT Backend (Linux + NVIDIA)

Relevant files:

- [include/tensorrt_compute.h](include/tensorrt_compute.h)
- [src/tensorrt_compute.cpp](src/tensorrt_compute.cpp)

The TensorRT path:

- parses the exported ONNX model directly via `nvonnxparser`
- compiles a per-GPU TRT engine with a dynamic batch-size optimization profile
  (`[1, max_batch_size]`) the first time the model is loaded
- caches the compiled engine to `trt_cache/` so subsequent runs start in under
  a second (cache key includes the TRT version, GPU name, max batch, and
  precision, so mixed setups don't collide)
- auto-picks precision from the GPU's compute capability:
  - SM ≥ 8.0 (Ampere, Ada, Hopper, Blackwell) → **BF16** (same range as FP32,
    avoids score-head NaNs that FP16 can hit)
  - SM 7.x (Volta/Turing) → **FP16**
  - older → **FP32**
- uses `cudaStreamPerThread` so each NN-server thread has its own CUDA stream;
  two server threads on the same GPU run concurrently without host-side
  contention
- serializes engine lifecycle and `createExecutionContext` / `~IExecutionContext`
  under a per-engine mutex (required by TRT 10 even when inference itself is
  parallel)

The current code identifies the engine's I/O tensors by name (`input`,
`policy_logits`, `value`) with a shape-based fallback; optional heads
(`score_mean`, `score_stdev`, `ownership`) are bound only if the ONNX exports
them.

### Supported / Intended Runtime Path

Documented and tested paths:

- **Linux + NVIDIA** (TensorRT 10 + CUDA 12, SM ≥ 7.5)
- **macOS Apple Silicon** (Metal / MPSGraph)

Other backends (`cuda`, `opencl`, `eigen`) still exist in the tree as
fall-backs.  They compile and run but are not the primary validation target
of this port.

## ncurses Play UI

Interactive play lives in [src/main_play.cpp](src/main_play.cpp).

The UI was redesigned from a Go board to a Xiangqi board:

- 10 visible ranks
- 9 files (`a` through `i`)
- river gap labeled `Chu He` / `Han Jie`
- colored red/black pieces
- source-square selection with a cursor
- ICCS-like typed move entry (`b2e2` style)

### UI Controls

- arrows / `WASD`: move cursor
- `Enter` / `Space`: select piece, then select destination
- alphanumeric typing: enter a move string directly
- `Backspace`: clear typed move or deselect piece
- `a`: toggle live analysis
- `p`: toggle background ponder
- `q`: quit

## Build

### Linux + NVIDIA (TensorRT)

System requirements:

- NVIDIA GPU with compute capability ≥ **7.5** (Turing or newer)
- CUDA Toolkit 12.x
- TensorRT 10.x (`libnvinfer-dev`, `libnvonnxparsers-dev`, `libnvinfer-headers-dev`)
- Eigen3 (`libeigen3-dev`), CMake ≥ 3.16, ncursesw (`libncursesw5-dev`)

Ubuntu/Debian:

```bash
# CUDA and TensorRT come from the NVIDIA repo; see
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/
sudo apt install cmake libeigen3-dev libncursesw5-dev \
                 libnvinfer-dev libnvinfer-headers-dev libnvonnxparsers-dev
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

On a Linux host with a TRT-compatible NVIDIA GPU the build auto-selects:

```text
Backend: tensorrt
```

If TensorRT is missing (or the GPU is SM < 7.5, e.g. V100) the build falls
back to:

```text
Backend: cuda
```

To force a particular backend regardless of auto-detection:

```bash
cmake -S . -B build -DMINIGO_BACKEND=tensorrt   # or cuda | opencl | eigen
```

### macOS / Apple Silicon

```bash
xcode-select --install
brew install cmake eigen
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

On Apple Silicon the build will auto-select:

```text
Backend: metal
```

## Python Dependencies

Training/export require Python packages such as:

```bash
pip install torch onnx zstandard
```

Notes:

- `train.py --help` now works even if `zstandard` is not installed yet
- reading `.zst` self-play files still requires `zstandard`

## Commands

### Export a Model

```bash
python3 scripts/export_onnx.py \
  --checkpoint training/checkpoints/training.pt \
  --output models/model.onnx
```

### Train

```bash
python3 scripts/train.py \
  --data training/selfplay \
  --epochs 10 \
  --batch-size 256 \
  --checkpoint training/checkpoints/training.pt \
  --output-onnx models/model.onnx
```

### Interactive Play

```bash
./build/play --model models/model.onnx
```

### Self-Play

```bash
./build/selfplay --model models/model.onnx --games 100 --threads 4
```

### Evaluate Two Models

```bash
./build/evaluate \
  --model1 models/candidate.onnx \
  --model2 models/baseline.onnx \
  --games 100
```

### Benchmark

```bash
./build/benchmark --model models/model.onnx
```

### Full self-play + training loop

`run_loop.py` orchestrates the AlphaZero-style loop (self-play → train →
head-to-head eval gate) using the C++ binaries above.  It generates a plan
file (`training/plan.json`) describing the stages of the run — number of
iterations, games per iteration, simulations per move, learning-rate
schedule, optional promotion gate — and then iterates over that plan.

```bash
# Choose one preset:
#   quick   ~  5 iter,   20 games / iter,  100 sims    (smoke test)
#   small   ~ 48 iter,  400-1400 games,    200-500 sims
#   large   ~ 72 iter,  400-1400 games,    200-600 sims
#   xlarge  ~200 iter, 800-5000 games,    300-1000 sims
python3 run_loop.py init small -y

# Run until the final planned iteration:
python3 run_loop.py train

# At any time:
python3 run_loop.py status
```

The generated plan:

- scales model size with the preset (quick=32×3, small=64×5, large/xlarge=128×10)
- raises `games` and `sims` gradually across stages so the early iterations
  churn through many quick games and the later iterations spend more compute
  per move
- enables a head-to-head promotion gate from "Early gated" onwards
- auto-configures hardware knobs (NN server threads, device ids, batch size)
  from the detected GPU count via `detect_hardware()`

You can override `--filters` / `--blocks` at `init` time to pin the network
size independently of the preset.

## CLI Reference

This section documents the **current** Xiangqi CLI, not the old Go CLI.

### Common C++ Runtime Knobs

These options are now intentionally shared wherever they make sense:

| Argument | Meaning | Default |
|---|---|---:|
| `--sims` | MCTS simulations per move/search | `800` |
| `--search-threads` | search threads per root | `16` in tools that expose it |
| `--max-batch` | maximum evaluator batch size | `256` |
| `--c-puct` | exploration constant | `1.5` |
| `--win-loss-weight` | value contribution in utility | `1.0` |
| `--score-weight` | extra score term in utility | `0.0` |
| `--score-scale` | score scaling constant | `1000.0` |
| `--nn-server-threads` | evaluator server thread count | `1` |
| `--nn-device-ids` | GPU ids for evaluator servers | `"0"` |

### Intentional Differences Between Tools

The overlapping knobs are now consistent. The remaining differences are
intentional because the tools do different jobs:

- `play` adds:
  - `--pvs`
  - `--random`
- `selfplay` adds:
  - `--games`
  - `--threads`
  - `--output`
  - `--dirichlet-alpha`
  - `--dirichlet-epsilon`
  - `--temp-threshold`
- `benchmark` adds:
  - `--nn-iters`
  - `--games`
  - `--threads`
- `evaluate` adds:
  - `--model1`
  - `--model2`
  - `--threshold`
  - `--output`

### `play`

```text
./build/play [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--model PATH` | ONNX model to load | `models/best.onnx` |
| `--sims N` | MCTS simulations | `800` |
| `--search-threads N` | search threads | `16` |
| `--max-batch N` | evaluator batch cap | `256` |
| `--win-loss-weight F` | utility weight for value | `1.0` |
| `--score-weight F` | utility weight for score term | `0.0` |
| `--score-scale F` | score utility scale | `1000.0` |
| `--c-puct F` | exploration constant | `1.5` |
| `--nn-server-threads N` | evaluator server threads | `1` |
| `--nn-device-ids IDS` | comma-separated device ids | `"0"` |
| `--pvs N` | top moves shown in live analysis | `5` |
| `--random` | use random legal-move opponent instead of NN | off |

### `selfplay`

```text
./build/selfplay [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--model PATH` | ONNX model | `models/best.onnx` |
| `--games N` | number of games | `100` |
| `--threads N` | parallel self-play workers | `1` |
| `--search-threads N` | search threads per move | `16` |
| `--max-batch N` | evaluator batch cap | `256` |
| `--output DIR` | self-play output directory | `training/selfplay` |
| `--sims N` | simulations per move | `800` |
| `--c-puct F` | exploration constant | `1.5` |
| `--dirichlet-alpha F` | root noise alpha | `0.30` |
| `--dirichlet-epsilon F` | root noise blend | `0.25` |
| `--temp-threshold N` | stochastic opening move count | `18` |
| `--win-loss-weight F` | utility weight for value | `1.0` |
| `--score-weight F` | utility weight for score term | `0.0` |
| `--score-scale F` | score utility scale | `1000.0` |
| `--nn-server-threads N` | evaluator server threads | `1` |
| `--nn-device-ids IDS` | comma-separated device ids | `"0"` |

### `benchmark`

```text
./build/benchmark [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--model PATH` | ONNX model | `models/best.onnx` |
| `--sims N` | MCTS simulations | `800` |
| `--nn-iters N` | NN inference iterations | `1000` |
| `--games N` | self-play benchmark games | `5` |
| `--threads N` | self-play workers | `1` |
| `--search-threads N` | search threads per move | `16` |
| `--max-batch N` | evaluator batch cap | `256` |
| `--c-puct F` | exploration constant | `1.5` |
| `--win-loss-weight F` | utility weight for value | `1.0` |
| `--score-weight F` | utility weight for score term | `0.0` |
| `--score-scale F` | score utility scale | `1000.0` |
| `--nn-server-threads N` | evaluator server threads | `1` |
| `--nn-device-ids IDS` | comma-separated device ids | `"0"` |

### `evaluate`

```text
./build/evaluate [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--model1 PATH` | candidate model | required |
| `--model2 PATH` | baseline model | required |
| `--games N` | number of games | `100` |
| `--threads N` | parallel workers | `1` |
| `--search-threads N` | search threads per move | `16` |
| `--sims N` | simulations per move | `800` |
| `--max-batch N` | evaluator batch cap | `256` |
| `--threshold FLOAT` | pass/fail win-rate threshold | `0.55` |
| `--c-puct F` | exploration constant | `1.5` |
| `--win-loss-weight F` | utility weight for value | `1.0` |
| `--score-weight F` | utility weight for score term | `0.0` |
| `--score-scale F` | score utility scale | `1000.0` |
| `--output DIR` | write match records as text files | unset |
| `--nn-server-threads N` | evaluator server threads per model | `1` |
| `--nn-device-ids IDS` | comma-separated device ids | `"0"` |

### `scripts/train.py`

```text
python3 scripts/train.py [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--data` | comma-separated self-play directories | `training/selfplay` |
| `--checkpoint` | checkpoint path | `training/checkpoints/training.pt` |
| `--epochs` | epoch count | `10` |
| `--batch-size` | batch size | `256` |
| `--lr` | Adam learning rate | `1e-3` |
| `--weight-decay` | Adam weight decay | `1e-4` |
| `--rows` | board rows | `10` |
| `--cols` | board cols | `9` |
| `--history-length` | snapshots in input encoding | `4` |
| `--filters` | trunk channels | `128` |
| `--blocks` | residual block count | `10` |
| `--num-workers` | DataLoader workers | `4` |
| `--policy-weight` | policy loss weight | `1.0` |
| `--value-weight` | value loss weight | `1.0` |
| `--output-onnx` | ONNX export path after training | `models/model.onnx` |

### `scripts/export_onnx.py`

```text
python3 scripts/export_onnx.py [options]
```

| Argument | Meaning | Default |
|---|---|---:|
| `--checkpoint` | checkpoint to export | `training/checkpoints/training.pt` |
| `--output` | output ONNX file | `models/model.onnx` |
| `--rows` | board rows | `10` |
| `--cols` | board cols | `9` |
| `--history-length` | snapshots in input encoding | `4` |
| `--filters` | trunk channels | `128` |
| `--blocks` | residual block count | `10` |

## What CLI Arguments Changed From the Go Version

This is the most important command-line migration summary.

### Removed From C++ Runtime Tools

These Go-specific arguments are gone because they no longer make sense for
fixed-board Xiangqi:

- `--board`
- `--komi`

Why:

- Xiangqi is always `10x9`
- there is no pass/komi-based Go scoring model in this port

### Replaced In Python Scripts

The old square-board-style Python options were replaced by explicit Xiangqi
board geometry and simpler model knobs.

Replaced:

- old `--board`
- old multi-architecture knobs tied to the Go codebase

Current Xiangqi shape options:

- `--rows`
- `--cols`
- `--history-length`
- `--filters`
- `--blocks`

### Removed From Training/Export

These old Go/KataGo-oriented options were intentionally dropped from the new
Python stack:

- `--arch`
- `--d-model`
- `--depth`
- `--heads`
- `--kv-groups`
- `--mlp-ratio`
- auxiliary-loss knobs from the old multi-head network

Why:

- the port currently supports one Xiangqi residual network
- the active runtime path is Metal-first and policy/value-only
- simplifying the model/export path made the C++ ONNX loader and backend much
  easier to verify

### Output Behavior Changes

- `evaluate --output` now writes **plain text match records**, not SGF
- `play` now expects **Xiangqi move entry**, not Go coordinates or pass
- self-play data is now **V3** with explicit `rows` and `cols`

## Consistency Summary

Are the command-line arguments consistent now?

**Yes, where they overlap.**

The shared MCTS/runtime knobs are now aligned across the C++ tools:

- `--sims`
- `--search-threads`
- `--max-batch`
- `--c-puct`
- `--win-loss-weight`
- `--score-weight`
- `--score-scale`
- `--nn-server-threads`
- `--nn-device-ids`

The remaining differences are intentional and tied to tool purpose.

## Known Limitations

- the port is documented and validated around **TensorRT on Linux** and
  **Metal on macOS**; the other backends compile but are not the primary
  validation target
- the current training loop uses only policy/value supervision even though the
  self-play record still stores extra trailing fields
- `evaluate --output` writes text records, not a standard Xiangqi notation file
- the Win32 engine in `xqwlight_win32/` is a rules reference, not a linked runtime dependency

## Key Files

| Area | File |
|---|---|
| Xiangqi rules | [include/game.h](include/game.h) |
| Xiangqi rules impl | [src/game.cpp](src/game.cpp) |
| shared config | [include/config.h](include/config.h) |
| MCTS | [include/mcts.h](include/mcts.h) / [src/mcts.cpp](src/mcts.cpp) |
| async bot | [include/async_bot.h](include/async_bot.h) / [src/async_bot.cpp](src/async_bot.cpp) |
| evaluator | [include/nn_evaluator.h](include/nn_evaluator.h) / [src/nn_evaluator.cpp](src/nn_evaluator.cpp) |
| ONNX loader | [include/loaded_model.h](include/loaded_model.h) / [src/loaded_model.cpp](src/loaded_model.cpp) |
| Metal backend | [include/metal_compute.h](include/metal_compute.h) / [src/metal_compute.mm](src/metal_compute.mm) |
| TensorRT backend | [include/tensorrt_compute.h](include/tensorrt_compute.h) / [src/tensorrt_compute.cpp](src/tensorrt_compute.cpp) |
| training pipeline | [run_loop.py](run_loop.py) |
| play UI | [src/main_play.cpp](src/main_play.cpp) |
| self-play tool | [src/main_selfplay.cpp](src/main_selfplay.cpp) |
| evaluate tool | [src/main_evaluate.cpp](src/main_evaluate.cpp) |
| benchmark tool | [src/main_benchmark.cpp](src/main_benchmark.cpp) |
| model definition | [scripts/model.py](scripts/model.py) |
| training | [scripts/train.py](scripts/train.py) |
| ONNX export | [scripts/export_onnx.py](scripts/export_onnx.py) |

## Short Version

This repository is no longer “MiniGo with a few edits.”

It is now a **Xiangqi AlphaZero-style engine** with first-class **TensorRT on
Linux + NVIDIA** and **Metal on macOS Apple Silicon** backends, keeping the
original project's best architectural property: the search/evaluator threading
model.
