# KataGo Integration — Technical Notes

This is a maintainer-facing companion to `KATAGO_INFERENCE.md` (which is
user-facing). Read this when you need to understand or extend the
plumbing, not when you just want to run kata1.

It's organized by mental model first, then by file/function so you can
trace any change back to its motivation.

- `KATAGO_INFERENCE.md` → "how do I use it"
- **this doc** → "how does it work, why was it changed this way, what
  invariants are load-bearing"

The integration was implemented as commit `f8c3411` ("feat(katago):
runtime-dispatched KataGo ONNX inference on TensorRT") on the
`multi-gpu` branch. Cross-reference that diff for verbatim context.

---

## Table of contents

- [1. Mental model](#1-mental-model)
- [2. Format detection — `LoadedModel`](#2-format-detection--loadedmodel)
- [3. Wire-format parser — `onnx_loader`](#3-wire-format-parser--onnx_loader)
- [4. Encoder dispatch — `BatchEvaluator` + `NNEvaluator`](#4-encoder-dispatch--batchevaluator--nnevaluator)
- [5. NN result buffer — additional output fields](#5-nn-result-buffer--additional-output-fields)
- [6. MCTS — call-site renames + ownership wiring](#6-mcts--call-site-renames--ownership-wiring)
- [7. Game state — `recent_actions_` + `is_ko_ban`](#7-game-state--recent_actions_--is_ko_ban)
- [8. V7 input encoder — `katago_inputs.{h,cpp}`](#8-v7-input-encoder--katago_inputshcpp)
- [9. TensorRT backend — dual-input branch](#9-tensorrt-backend--dual-input-branch)
- [10. Boundary guards — selfplay, benchmark sec5, non-TRT backends](#10-boundary-guards--selfplay-benchmark-sec5-non-trt-backends)
- [11. Python tooling — converter / parser / parity test](#11-python-tooling--converter--parser--parity-test)
- [12. Coexistence map — how a single binary handles both formats](#12-coexistence-map--how-a-single-binary-handles-both-formats)
- [13. Test surface](#13-test-surface)
- [14. Audit](#14-audit)

---

## 1. Mental model

The integration adds a **single boundary**: `model->format`. Everything
on the inference path flows through the same code; everything on the
training path is forbidden from ever seeing a KataGo model.

```
                     [LoadedModel::load]
                              │
                  ┌───────────┴───────────┐
                  │                       │
           format == MiniGo         format == KataGo
                  │                       │
        game.encode() (17 ch)    encode_for_katago() (22+19)
                  │                       │
                  └───────────┬───────────┘
                              │
                       [TensorRT backend]
                              │
                  ┌───────────┴───────────┐
                  │                       │
        single-input branch      dual-input branch
        (state)                  (state_spatial + state_global)
                  │                       │
                  └───────────┬───────────┘
                              │
                       same NNOutput
                  (policy / value / score / score_sd / ownership)
                              │
                          [MCTS]
                  (format-agnostic, never sees ModelFormat)

Excluded from KataGo path:
  - main_selfplay         → return 2 at startup
  - main_benchmark sec 5  → SKIPPED message
  - Eigen / CUDA / OpenCL / Metal / RKNN backends
                          → throw at create_handle
  - mcts.cpp:699          → stays game.encode() (defense in depth)
  - V2 selfplay record format / 8-fold augmentation
                          → unreachable; not modified
```

Two non-negotiable invariants:

1. **MiniGo-format runs are byte-identical before vs. after.** No
   existing MiniGo flag, behavior, training output, engine cache key,
   or selfplay record format changes. Every C++ edit is either
   strictly additive (new field / new branch / new override) or a
   five-line guard that only fires on `format == KataGo`.
2. **KataGo weights cannot reach the training pipeline.** The
   selfplay binary refuses them; benchmark section 5 skips them; the
   selfplay `encode` call site at `mcts.cpp:699` is left as the
   hardcoded `game.encode()` so even if a future caller bypassed
   the entry guard, augmentation would still see MiniGo-shaped
   states.

If you're considering a change that violates either invariant, the
right answer is almost certainly to add another guard, not to relax
the invariant.

---

## 2. Format detection — `LoadedModel`

### Files
- `include/loaded_model.h` (additive)
- `src/loaded_model.cpp` (additive arm before existing detection)

### What and why

KataGo's exported ONNX has two graph inputs (`state_spatial`,
`state_global`); MiniGo's has one. We detect this via a new
graph-input enumerator (see §3) rather than by sniffing initializer
names — a KataGo network has no initializer name a MiniGo net wouldn't
also have, but the input *count* and *names* are a robust signal.

### Specific edits

`include/loaded_model.h` — add an enum and two fields:
```cpp
enum class ModelFormat {
    MiniGo = 0,
    KataGo = 1,
};

class LoadedModel {
    ...
    ModelFormat format = ModelFormat::MiniGo;
    int input_global_channels = 0;          // 0 for MiniGo, 19 for KataGo
};
```

`src/loaded_model.cpp::LoadedModel::load` — first arm of the function
body, executed BEFORE the existing ResNet/ViT detection:
```cpp
auto inputs = onnx_parser::parse_onnx_graph_inputs(model_path);
const onnx_parser::OnnxGraphIO* sp = nullptr;
const onnx_parser::OnnxGraphIO* gl = nullptr;
for (auto& i : inputs) {
    if (i.name == "state_spatial") sp = &i;
    else if (i.name == "state_global") gl = &i;
}
if (sp && gl) {
    model->format = ModelFormat::KataGo;
    model->model_type = "katago";
    // sp->dims = [batch, C, H, W]; gl->dims = [batch, num_global]
    model->input_channels        = (int)sp->dims[1];
    model->board_size            = (int)sp->dims[2];   // == sp->dims[3]
    model->input_global_channels = (int)gl->dims[1];
    std::cout << "Model loaded: type=katago board=" << model->board_size
              << " channels=" << model->input_channels
              << "+" << model->input_global_channels << "\n";
    return model;        // skip MiniGo's weight load (TRT parses ONNX itself)
}
// ... existing ResNet/ViT detection unchanged ...
```

The early `return model;` skips the MiniGo weight-extraction helpers
(`load_conv`, `load_bn`, `load_fc`). This is correct because all
non-KataGo C++ backends rebuild weights from `LoadedModel::*` fields,
but the TensorRT backend (the only KataGo-supported backend)
doesn't — it parses the ONNX directly via `nvonnxparser`. So
KataGo's `LoadedModel` only needs metadata.

### Invariant

Existing MiniGo ONNX files don't contain `state_spatial` /
`state_global` inputs, so the KataGo arm never fires for them. The
"Model loaded: …" log line is unchanged for MiniGo runs.

---

## 3. Wire-format parser — `onnx_loader`

### Files
- `include/onnx_loader.h` (new struct + new function declaration)
- `src/onnx_loader.cpp` (new helpers, ~110 LoC)

### What and why

The pre-existing parser only walked initializers (`GraphProto.initializer`,
field 5), enough to recover weight tensors. To detect KataGo we need
the **graph inputs** (`GraphProto.input`, field 11), which list each
named input tensor's shape. We extended the in-tree minimal protobuf
parser rather than depending on libprotobuf — same dependency posture
as the rest of the project.

### Specific edits

`include/onnx_loader.h` — additive:
```cpp
struct OnnxGraphIO {
    std::string name;
    std::vector<int64_t> dims;       // -1 for symbolic dims
};
std::vector<OnnxGraphIO> parse_onnx_graph_inputs(const std::string& path);
```

`src/onnx_loader.cpp` — three new static helpers + one public function:
- `parse_value_info(Reader r) -> OnnxGraphIO` — walks
  `ValueInfoProto.name` (field 1) and `ValueInfoProto.type` (field 2,
  TypeProto → tensor_type → shape → dim → dim_value).
- `parse_graph_inputs(Reader r) -> vector<OnnxGraphIO>` — walks
  `GraphProto.input` (field 11), invokes `parse_value_info` per entry.
- `parse_model_inputs(const uint8_t*, size_t) -> vector<OnnxGraphIO>` —
  walks `ModelProto.graph` (field 7) and delegates.
- `parse_onnx_graph_inputs(const std::string& path)` — public entry,
  reads file into a buffer and calls `parse_model_inputs`.

ONNX wire-format field numbers are documented inline as a comment
block above `parse_value_info` so future maintainers don't need to
go look them up:
```
GraphProto.input  is field 11 (repeated ValueInfoProto)
ValueInfoProto.name  is field 1 (string)
ValueInfoProto.type  is field 2 (TypeProto)
  TypeProto.tensor_type  is field 1
    Tensor.shape  is field 2
      Shape.dim   is field 1
        Dimension.dim_value  is field 1 (int64)
        Dimension.dim_param  is field 2 (string)  → -1
```

### Invariant

The existing `parse_onnx_file()` (initializer enumeration) is left
alone. New helpers are reachable only via the new public function.

---

## 4. Encoder dispatch — `BatchEvaluator` + `NNEvaluator`

### Files
- `include/batch_evaluator.h` (additive virtual on the abstract base)
- `include/nn_evaluator.h` (override declaration)
- `src/nn_evaluator.cpp` (override body — single function)

### What and why

An earlier draft put the format dispatch at every `game.encode(state)`
call site. That was wrong: MCTS doesn't hold a `LoadedModel*`, and
`main_evaluate` builds **one** `Config` shared by both MCTS instances,
so a single `Config::format` field would be ambiguous in cross-format
matches. Putting the dispatch on the evaluator (which already owns
`std::shared_ptr<LoadedModel>`) is the right OO answer and fits the
minimal-change requirement.

### Specific edits

`include/batch_evaluator.h`:

1. Add `#include "game.h"` so the inline default body has the full
   `GoGame` definition (otherwise only a forward declaration is
   visible and the inline body fails to compile in dependent TUs).
2. Add a virtual member with a default body that preserves MiniGo
   behavior:
```cpp
virtual void encode_state(const GoGame& game, std::vector<float>& out) const {
    game.encode(out);
}
```

`include/nn_evaluator.h`:
```cpp
void encode_state(const GoGame& game, std::vector<float>& out) const override;
```

`src/nn_evaluator.cpp` (added at the top of the namespace block,
before `NNEvaluator` constructor):
```cpp
void NNEvaluator::encode_state(const GoGame& game, std::vector<float>& out) const {
    if (model_->format == ModelFormat::KataGo) {
        encode_for_katago(game, model_.get(), out);
    } else {
        game.encode(out);
    }
}
```

This is the **only** place in the C++ where format is consulted on
the encode path. Adding a third format later (e.g. KataGo V8) means
extending this one function.

### Invariant

Any subclass of `BatchEvaluator` that doesn't override `encode_state`
gets MiniGo's encoder for free (default body). This is what
`DirectEvaluator` (Eigen backend) relies on.

---

## 5. NN result buffer — additional output fields

### Files
- `include/batch_evaluator.h` — `NNResultBuf` struct
- `src/nn_evaluator.cpp` — server-thread writeback + catch-path reset
  + buf-to-`NNOutput` reconstructions

### What and why

`NNOutput` already had `score`, `score_sd`, `ownership` (added for
MiniGo's score head). But `NNResultBuf` — the per-search-thread
queue entry that the server thread fills — was only carrying
`policy`, `value`, `score`. So `score_sd` and `ownership` were
computed by the GPU and then dropped before MCTS could see them.
That was a latent MiniGo bug; KataGo needs both fields, so we fixed
it as part of this work.

### Specific edits

`include/batch_evaluator.h::NNResultBuf` — added two fields at the end:
```cpp
struct NNResultBuf {
    /* existing fields */
    float              score_sd = 0.0f;     // NEW
    std::vector<float> ownership;            // NEW
};
```

`src/nn_evaluator.cpp` — four call sites updated:

1. `server_loop` success path (was at `:166-173`, now `:181-189`):
```cpp
buf->policy    = std::move(all_results[i].policy);
buf->value     = all_results[i].value;
buf->score     = all_results[i].score;
buf->score_sd  = all_results[i].score_sd;     // NEW
buf->ownership = std::move(all_results[i].ownership);  // NEW
buf->done      = true;
```

2. `server_loop` catch path (when `predict_batch` throws):
```cpp
int board_area = model_->board_size * model_->board_size;
for (auto* buf : batch) {
    ...
    buf->score    = 0.0f;
    buf->score_sd = 0.0f;                       // NEW
    buf->ownership.assign(board_area, 0.0f);    // NEW
    buf->done = true;
}
```
Without the reset, a buf reused across (success → throw) calls would
leak stale ownership/uncertainty.

3. `evaluate_with_buf()` return — now constructs
   `{policy, value, score, score_sd, ownership}` (5 fields).

4. `evaluate()` batch return — same 5-field construction.

### Invariant

`NNOutput`'s field layout is unchanged — only the buf-to-`NNOutput`
copy is now complete. Any caller that already worked keeps working;
callers that read the new fields now get real data instead of zeros.

---

## 6. MCTS — call-site renames + ownership wiring

### Files
- `include/mcts.h` (one new private member)
- `src/mcts.cpp` (3 call-site renames + 2 small additions)

### What and why

MCTS itself is format-agnostic; we want it to stay that way. So the
encode dispatch goes through the evaluator, not into MCTS. There are
4 call sites of `game.encode(state)` in `mcts.cpp` (lines 248, 340,
413, 699) — three are inference paths and get renamed; one is the
selfplay record-builder and stays as-is.

Additionally, `AnalysisInfo::root_ownership` was declared on the
`MCTS::AnalysisInfo` struct but never populated — a latent MiniGo
bug (the `play` binary's `o` overlay would never light up). We fix it
here so KataGo's ownership head reaches the analysis HUD.

### Specific edits

**Renames (3 of 4 `game.encode()` call sites)** — all on inference paths:

| Line | Function | Was | Now |
|---|---|---|---|
| 248 | `search_thread_loop` | `game_copy_ptr->encode(state);` | `evaluator_->encode_state(*game_copy_ptr, state);` |
| 340 | `search_single_threaded` | `game_copy_ptr->encode(leaf.state);` | `evaluator_->encode_state(*game_copy_ptr, leaf.state);` |
| 413 | `search` (root eval, fresh root) | `game.encode(state_enc);` | `evaluator_->encode_state(game, state_enc);` |

**Deliberately left alone:**

| Line | Function | Why |
|---|---|---|
| 699 | `self_play_game_impl` (per-step `state` for V2 records) | Selfplay rejects KataGo at entry. Leaving the hardcoded `game.encode()` here makes the MiniGo-only invariant explicit at the call site — defense in depth in case a future caller bypasses `main_selfplay`'s guard. |

**Ownership wiring (2 small additions):**

`include/mcts.h` — new private member:
```cpp
std::vector<float> root_nn_ownership_;   // [board²] from root NN eval
```
Stored at the MCTS class, not on `MCTSNode`, because only the root
ever needs it for analysis.

`src/mcts.cpp::search` — at the root expansion site (was `:421`):
```cpp
new_root->nn_score    = root_nn_output.score;
new_root->nn_score_sd = root_nn_output.score_sd;
root_nn_ownership_    = root_nn_output.ownership;     // NEW
new_root->state.store(NODE_EXPANDED, ...);
```

`src/mcts.cpp::get_analysis` — copy into `AnalysisInfo`:
```cpp
info.root_ownership = root_nn_ownership_;     // NEW
```

### Invariant

MCTS doesn't `#include "loaded_model.h"`, doesn't see
`ModelFormat`, doesn't hold a `LoadedModel*`. The only thing that
changed about MCTS's behavior is that:
- encode goes through the evaluator (which forwards to the same
  `game.encode()` for MiniGo, so MiniGo runs are byte-identical), and
- `root_ownership` is populated (was always supposed to be).

---

## 7. Game state — `recent_actions_` + `is_ko_ban`

### Files
- `include/game.h` (additive)
- `src/game.cpp` (additive helpers + maintenance in `play`/`reset`/`copy`)

### What and why

KataGo V7 history planes encode WHERE moves were played (one-hot per
ply), not what the board looked like. MiniGo's `ring_buf_` stores
board snapshots, which can't reliably recover move locations
(captures change multiple cells; passes change zero cells). So we
add a small auxiliary array `recent_actions_[5]` that the V7 encoder
reads.

The V7 ko plane (`6`) needs the simple-ko location. That used to be
derivable only from `prev_board` + `has_prev_board`, which are
private. Rather than make them public, we expose a single semantic
accessor: `is_ko_ban(action)`.

### Specific edits

`include/game.h` — add two public accessors and one private array:
```cpp
class GoGame {
public:
    int recent_action(int steps_back) const;   // 0=most recent
    bool is_ko_ban(int action) const;
private:
    static constexpr int RECENT_ACTIONS_CAP = 5;
    std::array<int, RECENT_ACTIONS_CAP> recent_actions_;
    int recent_actions_count_ = 0;
};
```

`src/game.cpp::reset` — added two lines:
```cpp
recent_actions_.fill(-2);     // -2 = sentinel "no move"
recent_actions_count_ = 0;
```

`src/game.cpp::copy` — added two lines:
```cpp
g.recent_actions_       = recent_actions_;
g.recent_actions_count_ = recent_actions_count_;
```

`src/game.cpp::play` — inserted **after** pass normalization,
**before** `prev_board` snapshot (this ordering matters — we record
the *normalized* action so passes show up as `PASS_MOVE`, not the
pre-normalized `n*n` slot):
```cpp
if (action == n * n) action = PASS_MOVE;     // existing line
// NEW: record action history
for (int i = RECENT_ACTIONS_CAP - 1; i > 0; --i)
    recent_actions_[i] = recent_actions_[i - 1];
recent_actions_[0] = action;
if (recent_actions_count_ < RECENT_ACTIONS_CAP) ++recent_actions_count_;
```

`src/game.cpp::is_ko_ban` (new function):
- Returns false fast if no `prev_board`, action is pass/out-of-range,
  or the cell is non-empty.
- Simulates the move on a stack-allocated test board (no heap
  allocation in the hot path):
  - Place the stone
  - Capture any neighboring opponent groups with 0 liberties
  - Suicide check on the played group → if suicide, it's not ko
  - Compare resulting test board to `prev_board` via `memcmp`. If
    equal, it's the simple-ko ban location.

### Invariant

MiniGo's encoder (`GoGame::encode`) doesn't read `recent_actions_`
at all — it uses `ring_buf_` (board snapshots). MiniGo runs are
byte-identical. The new array is maintained even on MiniGo runs
(small fixed cost, no heap allocation), to keep the `play`/`copy`
code paths unconditional.

---

## 8. V7 input encoder — `katago_inputs.{h,cpp}`

### Files
- `include/katago_inputs.h` (new, declaration only)
- `src/katago_inputs.cpp` (new, ~270 LoC — port of upstream's `fillRowV7`)

### What and why

This is the C++ port of `cpp/neuralnet/nninputs.cpp::NNInputs::fillRowV7`
from upstream KataGo. It produces the 22-spatial + 19-global float
vector that the kata1 V7 networks were trained against, packed into a
single `std::vector<float>`:
```
out[0 .. 22*H*W)        : spatial planes, row-major [C, H, W]
out[22*H*W .. + 19)     : global feature vector
```

The TRT backend splits this flat layout back into the two TRT inputs
at `predict_batch` time (see §9). Routing both pieces through the
same `state` byte array preserves the existing `NNResultBuf`'s
`state_data` / `state_size` semantics.

### Public API

`include/katago_inputs.h`:
```cpp
void encode_for_katago(const GoGame& game,
                       const LoadedModel* model,
                       std::vector<float>& out);
```

The `model` parameter is currently used only for assertion (22 + 19
match) — leaving it in the signature so future format variants
(different channel counts) can branch on it without an API change.

### Internal helpers

`src/katago_inputs.cpp`, all in an anonymous namespace:
- `group_liberty_count(game, r, c) -> int` — exact liberty count of
  the group containing `(r, c)`, stack-allocated visited buffers.
- `compute_area(game, area)` — Tromp-Taylor flood-fill: each cell
  → BLACK / WHITE / EMPTY based on stone presence and surround-color
  rule. Used for planes 18-19. **Approximation:** upstream uses
  `Board::calculateArea` with pass-alive logic; this is plain TT.
- `sp_idx(plane, r, c, H, W)` — flat-index helper.

### Plane-by-plane (spatial)

| Plane | Content | Source | Note |
|---|---|---|---|
| 0 | on-board mask | constant 1 | square boards only |
| 1 | own stones (current player) | `game.board[r][c] == pla` | |
| 2 | opp stones | `game.board[r][c] == opp` | |
| 3 | own/opp stones with **exactly 1** liberty | `group_liberty_count == 1` | |
| 4 | … with **exactly 2** liberties | `== 2` | |
| 5 | … with **exactly 3** liberties | `== 3` | upstream uses `==`, NOT `>=`; 4+ libs → no flag |
| 6 | ko-banned point | `game.is_ko_ban(action)` | simple ko |
| 7 | ko-recap-blocked (encore) | **0** — no encore support | |
| 8 | unused in V7 | **0** — exact, upstream zeroes too | |
| 9 | location of opp's previous move (1 ply ago) | `recent_action(0)` | also sets `gl[0]=1` if pass |
| 10 | location of own move 2 plies ago | `recent_action(1)` | `gl[1]=1` for pass |
| 11 | … 3 plies ago | `recent_action(2)` | `gl[2]=1` |
| 12 | … 4 plies ago | `recent_action(3)` | `gl[3]=1` |
| 13 | … 5 plies ago | `recent_action(4)` | `gl[4]=1` |
| 14-17 | ladder features | **0** — TODO, port upstream's `searchIsLadderCapturedAttackerFirst` | |
| 18 | own's Tromp-Taylor area | `compute_area` | |
| 19 | opp's TT area | `compute_area` | |
| 20-21 | second-encore start colors | **0** — no encore | |

### Global features (19)

| Idx | Content | Formula |
|---|---|---|
| 0-4 | pass flags for plies 1..5 ago | set by the history loop above |
| 5 | komi normalized | `selfKomi / 20.0`, `selfKomi = +komi (white) / -komi (black)`, clamped to ±(boardArea+1) |
| 6-7 | ko rule | `(0,0)` for simple ko (only mode supported) |
| 8 | multi-stone-suicide | `0` (TT disallows) |
| 9 | scoring rule | `0` = area (only mode supported) |
| 10-11 | tax rule | `(0,0)` = TAX_NONE |
| 12-13 | encore phase | `(0,0)` = none |
| 14 | pass-would-end-phase | `1.0 if game.consecutive_passes >= 1 else 0.0` (approximation) |
| 15 | komi parity wave | piecewise triangle wave, peaks at half-komi values; matches upstream |
| 16-18 | unused in V7 | `0` |

### Invariant

`encode_for_katago` is reached only via
`NNEvaluator::encode_state` when `model->format == KataGo`. MiniGo
runs never hit this code path. No global state. No heap allocation
beyond the output `std::vector<float>` and a few small intermediate
arrays in `compute_area`.

---

## 9. TensorRT backend — dual-input branch

### File
- `src/tensorrt_compute.cpp` (~205 LoC of net change, all guarded)

### What and why

KataGo ONNX has two inputs (`state_spatial`, `state_global`); MiniGo
has one. The output side is identical for both (same names, same
shapes — by design of our converter; see §11). So the only divergence
inside the TRT backend is on the **input** half: name discovery,
optimization profile, device buffer allocation, and predict-time
upload + binding. ~30 LoC of net-new logic, the rest is plumbing.

### Specific edits

#### Engine name discovery — `TensorRTComputeHandle` ctor I/O loop

`TRTDeviceState` (per-GPU shared) gains two name fields:
```cpp
std::string input_name;            // MiniGo
std::string input_spatial_name;    // KataGo
std::string input_global_name;     // KataGo
```

The name-match loop (the existing string-based discovery already used
for outputs) gets an input-side branch:
```cpp
if (mode == nvinfer1::TensorIOMode::kINPUT) {
    if      (sname == "state_spatial") dev.input_spatial_name = name;
    else if (sname == "state_global")  dev.input_global_name  = name;
    else                                dev.input_name         = name;
}
```

After the loop, cross-check that the engine's I/O shape matches
`model->format` (catches the case where someone hand-edits an ONNX
to add inputs, or the name-match table got out of sync):
```cpp
bool is_katago_engine = !dev.input_spatial_name.empty()
                     && !dev.input_global_name.empty();
bool is_minigo_engine = !dev.input_name.empty();
if (model->format == ModelFormat::KataGo && !is_katago_engine) throw …;
if (model->format == ModelFormat::MiniGo && !is_minigo_engine) throw …;
```

#### Optimization profile — generalized loop

Was hard-coded for input 0. Now iterates over `network->getNbInputs()`
and sets min/opt/max dims for each, treating `dims.d[0]` as the
dynamic batch dimension and leaving the rest fixed:
```cpp
auto* profile = builder->createOptimizationProfile();
int nb_inputs = network->getNbInputs();
int opt_batch = std::max(1, max_batch_size / 2);
for (int i = 0; i < nb_inputs; ++i) {
    auto* inp = network->getInput(i);
    auto dims = inp->getDimensions();
    nvinfer1::Dims min_dims = dims, opt_dims = dims, max_dims = dims;
    min_dims.d[0] = 1;
    opt_dims.d[0] = opt_batch;
    max_dims.d[0] = max_batch_size;
    profile->setDimensions(inp->getName(),
                           nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(inp->getName(),
                           nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(inp->getName(),
                           nvinfer1::OptProfileSelector::kMAX, max_dims);
}
config->addOptimizationProfile(profile);
```

This is a strict generalization — for `nb_inputs == 1`, behavior is
identical to before. The `nvinfer1::Dims4` literal was replaced with
`nvinfer1::Dims` so the same code handles the 4-D `state_spatial`
and the 2-D `state_global`.

#### Per-handle Impl — extra fields

`TensorRTComputeHandle::Impl` gets:
```cpp
ModelFormat format = ModelFormat::MiniGo;          // copied from model in ctor
int input_global_channels = 0;                      // KataGo: 19
float* d_input_spatial  = nullptr;                  // KataGo only
float* d_input_global   = nullptr;                  // KataGo only
std::string input_spatial_name;
std::string input_global_name;
```

The format flag is copied from `model->format` in the ctor alongside
the existing `board_size` / `input_channels` / `max_batch_size`
assignments.

#### Device buffer allocation — branched on format

```cpp
if (I.format == ModelFormat::KataGo) {
    cudaMalloc(&I.d_input_spatial, max_batch * 22 * HW * 4);
    cudaMalloc(&I.d_input_global,  max_batch * 19 * 4);
} else {
    cudaMalloc(&I.d_input,         max_batch * input_channels * HW * 4);
}
```

Output buffers (`d_policy`, `d_value`, `d_score`, `d_score_sd`,
`d_ownership`) are allocated unconditionally — same shapes for both
formats by design of our converter.

`Impl::~Impl` adds two `cudaFree` calls for the new buffers, no-op
when they're nullptr (MiniGo path).

#### `predict_batch` — input upload split

The KataGo branch demuxes the encoder's flat `[22*H*W spatial][19
global]` layout into the two TRT inputs:

```cpp
if (I.format == ModelFormat::KataGo) {
    const int sp_per = I.input_channels * HW;
    const int gl_per = I.input_global_channels;
    std::vector<float> flat_spatial((size_t)N * sp_per, 0.0f);
    std::vector<float> flat_global ((size_t)N * gl_per, 0.0f);
    for (int n = 0; n < N; ++n) {
        const auto& s = states[n];
        if ((int)s.size() != sp_per + gl_per)
            throw std::runtime_error("TensorRT KataGo: state size mismatch ...");
        std::memcpy(flat_spatial.data() + n * sp_per,        s.data(),          sp_per * 4);
        std::memcpy(flat_global.data()  + n * gl_per, s.data() + sp_per, gl_per * 4);
    }
    cudaMemcpyAsync(I.d_input_spatial, flat_spatial.data(), …, cudaStreamPerThread);
    cudaMemcpyAsync(I.d_input_global,  flat_global.data(),  …, cudaStreamPerThread);

    nvinfer1::Dims4 sp_dims(N, I.input_channels, H, W);
    nvinfer1::Dims2 gl_dims(N, I.input_global_channels);
    exec_ctx->setInputShape(I.input_spatial_name.c_str(), sp_dims);
    exec_ctx->setInputShape(I.input_global_name.c_str(),  gl_dims);
    exec_ctx->setTensorAddress(I.input_spatial_name.c_str(), I.d_input_spatial);
    exec_ctx->setTensorAddress(I.input_global_name.c_str(),  I.d_input_global);
} else {
    /* existing MiniGo single-input path: flat upload + setInputShape + setTensorAddress */
}
```

The output read-back (cudaMemcpy from `d_policy` / `d_value` / …) and
`NNOutput` packing are unchanged from the MiniGo path. Same names,
same shapes, same post-processing (none, because the converter baked
all post-processing into the ONNX graph).

### Invariant

When `model->format == MiniGo`:
- `nb_inputs == 1` so the new optimization-profile loop runs once,
  emitting the same 3 dims as before (the previous Dims4 literal).
- The name-match loop's KataGo cases never fire.
- `d_input_spatial` / `d_input_global` stay nullptr.
- `predict_batch` takes the `else` branch.

MiniGo runs are byte-identical end-to-end. Engine cache key already
includes the model path, so KataGo and MiniGo engines never collide.

---

## 10. Boundary guards — selfplay, benchmark sec5, non-TRT backends

### Files
- `src/main_selfplay.cpp` (early-error guard)
- `src/main_benchmark.cpp` (section 5 if/else skip)
- `src/eigen_compute.cpp`, `src/cuda_compute.cu`,
  `src/opencl_compute.cpp`, `src/metal_compute.mm`,
  `src/rknn_compute.cpp` (handle-creation throw)

### What and why

(HISTORICAL — superseded by the V3 format cleanup: selfplay now accepts KataGo-format models because game records are engine-neutral, and the non-TRT backends accept the format at the interface with placeholder implementations.  Kept for archaeology.)

The original principle "KataGo weights are inference-only, TensorRT-only" was
enforced at every entry point that should reject a KataGo model.
Each guard is the smallest possible diff (4-7 lines) and uses a
clear error message.

### Specific edits

`src/main_selfplay.cpp` — right after `LoadedModel::load`:
```cpp
auto model = LoadedModel::load(model_path);
if (model->format == ModelFormat::KataGo) {
    std::cerr << "ERROR: selfplay does not support KataGo-format models.\n"
              << "  KataGo weights are inference-only in this build.\n"
              << "  Use the play or evaluate binary for KataGo runs.\n";
    return 2;
}
```

`src/main_benchmark.cpp` — section 5 (multi-threaded selfplay) wrapped
in an `if/else`:
```cpp
if (model->format == ModelFormat::KataGo) {
    std::cout << "5. Self-play: SKIPPED for KataGo-format model "
              << "(KataGo runs inference only; selfplay generates "
              << "MiniGo training records).\n";
} else {
    /* existing section 5 body unchanged */
}
```
Sections 1-4 (game engine, single inference, batch inference, MCTS)
are KataGo-aware and run unmodified. Section 1 doesn't use the model,
2-3 use the new evaluator-side encode dispatch, 4 uses MCTS which is
format-agnostic.

`src/eigen_compute.cpp::EigenComputeContext::create_handle` and the
analogous functions in `cuda_compute.cu` / `opencl_compute.cpp` /
`metal_compute.mm` / `rknn_compute.cpp` — all 5 backends grow the
same 4-line throw at the top:
```cpp
if (model->format == ModelFormat::KataGo)
    throw std::runtime_error(
        "KataGo format requires the TensorRT backend. "
        "Rebuild with `cmake -DMINIGO_BACKEND=tensorrt`.");
```
The throw fires on `NNEvaluator` startup (when each server thread
creates its handle), so the user sees the error immediately.

### Invariant

The MiniGo path through every backend is unchanged — the new throw
is unreachable when `format == MiniGo`.

---

## 11. Python tooling — converter / parser / parity test

### Files
- `tools/warm_init_from_katago.py` (additive parser extension only)
- `tools/katago_arch.py` (new — PyTorch nn.Module port of KataGo's
  inference graph, ~330 LoC)
- `tools/katago_to_onnx.py` (new — CLI converter, ~150 LoC)
- `tools/katago_parity_test.py` (new — PyTorch ↔ ONNX Runtime
  parity, ~100 LoC)

### `tools/warm_init_from_katago.py` — parser extension

The pre-existing parser dropped most layers from the policy / value
heads and the trunk-tip BN+act, parsing them only for stream-position
correctness. The inference graph needs all of them. The extension is
strictly additive:

1. `from typing import List, Union` → add `Optional`.
2. `Trunk` dataclass: add `trunk_tip_bn: Optional[BatchNormLayer] = None`
   and `trunk_tip_act: Optional[ActivationLayer] = None`. Populate in
   `_parse_trunk` (which already parsed them, just to discard).
3. `PolicyHead` dataclass: add `Optional[...] = None` fields for
   `g1_conv`, `g1_bn`, `g1_act`, `gpool_to_bias`, `p1_act`,
   `p2_conv`, `gpool_to_pass`, plus three v15+ fields. Populate in
   `_parse_policy_head`.
4. `ValueHead` dataclass: same pattern for `v1_act`, `v2_mul`,
   `v2_bias`, `v2_act`, `v3_mul`, `v3_bias`, `sv3_mul`, `sv3_bias`,
   `v_ownership_conv`. Populate in `_parse_value_head`.

The existing required fields (`p1_conv`, `p1_bn`, `v1_conv`,
`v1_bn`) are unchanged. Warm-init code reads only those, so its
output is byte-identical (verified by re-running warm-init: 5
trunk blocks paired, 6 heads paired, 1.49M params transferred at
44.8% coverage — same numbers as before the extension).

### `tools/katago_arch.py` — PyTorch architecture

Mirrors KataGo's runtime forward graph for **inference**. Key choices:

- **Pre-activation BN**: `nn.BatchNorm2d` placed BEFORE each conv in
  the residual block (opposite of MiniGo's post-act). Running stats
  loaded directly from KataGo's parsed `mean`/`variance`; gamma /
  beta from `scale` / `bias`; `eps` from KataGo's per-layer value.
  Module is in `eval()` mode so BN uses running stats only.

- **Global pool stats**: two distinct pool functions are required
  because the gpool stats and value-head pool stats DIFFER on the
  third statistic. Confirmed against upstream `openclkernels.cpp`:
  ```python
  def _gpool_stats(x):       # used by GPoolBlock + PolicyHead.g1
      mean, max, std0 = mean(x), max(x), mean(x) * (sqrt(N)-14)*0.1
      return cat([mean, std0, max], dim=1)

  def _vhpool_stats(x):      # used by ValueHead.v1
      mean = mean(x)
      a = mean * (sqrt(N)-14) * 0.1
      b = mean * ((sqrt(N)-14)**2 * 0.01 - 0.1)
      return cat([mean, a, b], dim=1)
  ```
  These constants (14.0, 0.1, 0.01) are KataGo-canonical and not
  free parameters.

- **Modules built**: `KataGoStem`, `KataGoOrdinaryBlock`,
  `KataGoGPoolBlock`, `KataGoPolicyHead`, `KataGoValueHead`,
  `KataGoNet` (top-level). Each constructor takes the parsed dataclass
  directly and copies weights via `_load_conv` / `_load_bn` /
  `_load_matmul` / `_load_matbias` helpers.

- **Heads bake post-processing into the graph**:
  - Policy: take `p2_conv` channel 0, flatten to `[N, H*W]`, concat
    `gpool_to_pass` logit at index `H*W` → `[N, H*W+1]`.
  - Value: 3-way softmax then `wld[:,0:1] - wld[:,1:2]` → `[N, 1]`.
  - ScoreMean: `sv3[:, 0:1] * 20.0`.
  - ScoreStdev: `softplus(sv3[:, 1:2]) * 20.0`.
  - Ownership: `(tanh(v_ownership_conv(v1)) + 1) * 0.5`, reshape
    to `[N, H*W]`.

Heads NOT built: lead, vTime, scoreBelief, futurePos, seki — these
are training-time auxiliaries with no MCTS consumer.

### `tools/katago_to_onnx.py` — CLI converter

CLI: `--katago-bin <path>`, `--board <int>`, `--output <path.onnx>`,
optional `--opset` (default 17 — we get opset 18 in practice from
PyTorch's exporter, since opset 17 conversion can fall back).

Pipeline:
1. `parse_katago_model(path)` → `KataGoModel` (uses the extended
   parser from §11 above).
2. `KataGoNet(kmodel)` → PyTorch nn.Module, weights copied during
   construction.
3. Shape-validation forward pass on dummy input.
4. `torch.onnx.export` with explicit `input_names` /
   `output_names` and `dynamic_axes={..: {0: "batch"}}` so TRT
   can build a dynamic-batch engine.
5. **Inline external weights:** PyTorch 2.x writes weights to
   `<output>.data` for large models. We use `onnx.load` +
   `load_external_data_for_model` + `onnx.save(save_as_external_data=False)`
   to inline them, then delete the sidecar. The output is one
   self-contained ~12 MB file for kata1 b10c128.
6. `onnx.checker.check_model` validation.

### `tools/katago_parity_test.py` — validation

PyTorch ↔ ONNX Runtime self-consistency on a random batch (default
N=4, seed 42). Reports max abs diff per output; fails if any exceeds
`--tol` (default 1e-4). Includes an empty-board sanity-check section
that prints the policy top-5, value, score_mean, ownership stats —
useful for spotting architecture-port bugs that wouldn't show up
in self-consistency.

Optional path (not implemented): comparing against upstream `katago
analyze` if it's on PATH. Skipped for now; self-consistency is
enough to validate the export.

### Invariant

These four files don't touch C++ source. The Python training pipeline
(`scripts/train*.py`) is unmodified — it has no path that loads a
KataGo `.bin.gz` directly anyway. `tools/warm_init_from_katago.py`'s
warm-init CLI behavior is unchanged.

---

## 12. Coexistence map — how a single binary handles both formats

The same `play` / `benchmark` / `evaluate` binary, built once with
`-DMINIGO_BACKEND=tensorrt`, handles either format. Here's the
flow for a KataGo run vs a MiniGo run side-by-side, listing each
file and what it does on each path:

| Stage | MiniGo run | KataGo run |
|---|---|---|
| `LoadedModel::load(path)` | parses initializers, sets `format=MiniGo`, fills `model_type`/`board_size`/etc. | calls `parse_onnx_graph_inputs`, finds `state_spatial`+`state_global`, sets `format=KataGo`, fills `input_channels=22`, `input_global_channels=19`, returns early (no per-op weight load needed; TRT parses ONNX) |
| `create_compute_context` | unchanged | unchanged |
| Backend `create_handle` (Eigen / CUDA / OpenCL / Metal / RKNN) | builds backend-specific weights from `LoadedModel` | **throws** `"KataGo format requires the TensorRT backend"` |
| `TensorRTComputeContext::create_handle` | creates `TensorRTComputeHandle` (single-input) | creates `TensorRTComputeHandle` (dual-input) — same class, branches on `model->format` internally |
| TRT engine build | optimization profile sets dims for input 0; engine cache `<name>_<gpu>_b<batch>_<prec>.engine` | same loop iterates over both inputs; engine cache key includes path so KataGo + MiniGo never collide |
| TRT name discovery | finds `state` / `policy_logits` / `value` / etc. | finds `state_spatial` / `state_global` / `policy_logits` / `value` / etc. |
| `MCTS::search` | calls `evaluator_->encode_state(game, state)` → falls into `BatchEvaluator::encode_state` default → `game.encode(out)` (17 channels) | calls `evaluator_->encode_state(game, state)` → falls into `NNEvaluator::encode_state` override → `encode_for_katago(...)` (22*H*W + 19 floats) |
| `NNRequestQueue` push | unchanged — `state_data` / `state_size` semantics preserved | unchanged — KataGo's flat layout still travels as a single byte array |
| `predict_batch` | flat upload to `d_input`, `setInputShape`, `setTensorAddress`, `enqueueV3` | demuxes flat layout into `flat_spatial` + `flat_global`, two `cudaMemcpyAsync`, two `setInputShape`, two `setTensorAddress`, `enqueueV3` |
| Output readback | `cudaMemcpy` from `d_policy` / `d_value` / `d_score` / `d_score_sd` / `d_ownership` | **identical** — by design of our converter, output side is the same |
| `NNOutput` packing | 5-field struct | **identical** — same struct, same field names |
| Server-thread writeback to `NNResultBuf` | copies all 5 fields | **identical** |
| MCTS leaf expansion | reads `policy`, `value`, `score`, `score_sd` | **identical** |
| `AnalysisInfo` | populated from root NN; `root_ownership` now wired | populated from root NN; same fields |

**The only branching points in C++ are:**
1. `LoadedModel::load` — early arm.
2. `NNEvaluator::encode_state` — single `if` on `model_->format`.
3. `TensorRTComputeHandle` — input-side branches in ctor and
   `predict_batch`, gated on `I.format`.
4. Boundary guards — selfplay, benchmark sec 5, non-TRT backends.

Everything else (MCTS, NNEvaluator queue+condvar pattern, AsyncBot,
NNOutput unpacking, analysis HUD) is format-agnostic.

---

## 13. Test surface

Manual / smoke tests run during development:

| Test | Result |
|---|---|
| `python tools/katago_parity_test.py --katago-bin … --onnx … --board 9` | pass — max abs diff 3.2e-5 on policy logits between PyTorch and ONNX Runtime |
| `python tools/warm_init_from_katago.py --katago-bin … --checkpoint /tmp/post.pt` (after parser extension) | identical output to pre-extension run: 5 trunk blocks paired, 6 heads paired, 1,491,720 params transferred (44.8% coverage) |
| `build/benchmark --model models/kata1-b10c128.onnx --max-batch 256 --komi 7.0` | sec 1-4 run; sec 5 emits SKIPPED message; ~65k states/s @ batch=128 on a 2080 Ti |
| `build/evaluate --model1 models/kata1.onnx --model2 models/kata1.onnx --games 1 --komi 7.0` | game completes; M1 wins as Black; 3.9s wall |
| `build/selfplay --model models/kata1-b10c128.onnx …` | exits with `return 2` and the documented error message — guard fires before any encoding |

No unit test harness was added. The plan called for a
`tests/test_katago_inputs.cpp` with reference outputs from upstream
`runtests inputtest` — that's a follow-up if encoder bugs surface.
The manual chain (parity test → benchmark → evaluate) is sufficient
to detect any architecture-port or encode-side regressions.

---

## 13.5. Files NOT modified (worth knowing)

These files were intentionally untouched. If you're hunting for a
KataGo branch in any of them, you won't find one:

- `src/main_play.cpp` — interactive `play` binary. Works for both
  formats because it goes through MCTS, which gets format-aware encode
  via the evaluator. The analysis HUD's `o` ownership overlay reads
  `info.root_ownership` which we now populate.
- `src/main_evaluate.cpp` — match-game `evaluate` binary. Same
  story: format-blind via MCTS. `--model1` and `--model2` can each
  be either format. Cross-format matches (kata1 vs MiniGo) work
  because each MCTS gets its own evaluator.
- `src/async_bot.cpp` — pondering / GTP-like worker. Talks to the
  evaluator polymorphically; never sees `ModelFormat`.
- `src/nn_request_queue.h` — queue protocol unchanged. `state_data`
  / `state_size` carry the encoder's flat output regardless of
  format; the server thread doesn't reinterpret it.
- `include/compute_context.h` — `ComputeHandle` interface unchanged.
- `mcts.cpp` augmentation block (lines 608-668) and
  `self_play_game_impl` (line 670+) — never reached on the KataGo
  path because `main_selfplay` and benchmark section 5 are guarded
  out. Augmentation's `state.size() == input_channels * board_area`
  assumption stays valid because only MiniGo states ever reach it.
- `scripts/*.py` — entire training pipeline (`train.py`,
  `train_continuous.py`, `run_continuous.py`, `selfplay_driver.py`,
  `gatekeeper.py`, `rate.py`, `export_onnx.py`, `model.py`,
  `visualize.py`). No path loads a KataGo binary or KataGo ONNX.
- `training/*` — training output dir layout, no changes.

### Build system

- `CMakeLists.txt` — one line added inside `CORE_SOURCES`:
  ```cmake
  set(CORE_SOURCES
      src/game.cpp
      src/onnx_loader.cpp
      ...
      src/async_bot.cpp
      src/katago_inputs.cpp     # NEW
  )
  ```
  No new target, no new dependency, no per-format compile flag, no
  `#ifdef`. The encoder TU is part of `minigo_core` regardless of
  backend; it's just dead code on a non-TRT build (no caller can
  reach it because non-TRT backends throw at handle creation).

### User-facing docs

- `README.md` — added a one-paragraph blurb in the "Model format"
  paragraph and a full **`### KataGo inference (TensorRT only)`**
  subsection under the existing **`### Benchmark`** quick-start
  block. Documents the 5-step recipe, lists which binaries / Python
  tools accept KataGo weights, calls out the selfplay rejection.
- `KATAGO_INFERENCE.md` — user-facing reference (knobs, presets,
  head kept/dropped table, MCTS gaps, caveats).
- `KATAGO_INTEGRATION_TECH.md` — this file.

---

## 14. Audit

Self-checking the doc against the actual diff. Each row asserts a
piece of "this is correct" that's easy to verify mechanically by
re-reading the commit.

| Claim in this doc | Verified in commit `f8c3411` | Status |
|---|---|---|
| `ModelFormat` enum lives in `loaded_model.h` | yes | ✓ |
| `parse_onnx_graph_inputs` returns `vector<OnnxGraphIO>` | yes — `include/onnx_loader.h` | ✓ |
| `BatchEvaluator::encode_state` default body is `game.encode(out)` | yes — `include/batch_evaluator.h` | ✓ |
| `NNEvaluator::encode_state` is the only `if (model->format == KataGo)` on the encode path | yes — single dispatch in `src/nn_evaluator.cpp` | ✓ |
| `mcts.cpp` renames lines 248 / 340 / 413 only; line 699 stays `game.encode()` | yes — diff shows three `evaluator_->encode_state` replacements; line 699 unchanged | ✓ |
| `recent_actions_` is updated AFTER pass normalization | yes — insert site is just below `if (action == n*n) action = PASS_MOVE;` in `src/game.cpp::play` | ✓ |
| `is_ko_ban` returns false for suicide moves | yes — explicit `if (own_libs == 0) return false; // suicide, not ko` | ✓ |
| TRT optimization profile loops over `getNbInputs()` | yes — `src/tensorrt_compute.cpp` shows the `for (int i = 0; i < nb_inputs; ++i)` loop | ✓ |
| TRT `Impl::format` is set from `model->format` in the ctor | yes — `I.format = model->format;` immediately before `I.board_size = …` | ✓ |
| Selfplay guard returns 2 (not 1) | yes — `return 2;` in `src/main_selfplay.cpp:135` | ✓ |
| Benchmark section 5 emits `"SKIPPED for KataGo-format model"` | yes — verbatim in `src/main_benchmark.cpp:235-238` | ✓ |
| All 5 non-TRT backends throw at `create_handle` | yes — `eigen_compute.cpp`, `cuda_compute.cu`, `opencl_compute.cpp`, `metal_compute.mm`, `rknn_compute.cpp` each have the 4-line throw | ✓ |
| Encoder packs `[22*H*W spatial floats][19 global floats]` in that order | yes — `src/katago_inputs.cpp` writes spatial first via `sp[sp_idx(...)]` then globals via `gl[i]`; `gl = out.data() + 22*HW` | ✓ |
| TRT KataGo `predict_batch` demuxes via `memcpy(flat_spatial, s.data(), sp_per*4)` then `memcpy(flat_global, s.data()+sp_per, gl_per*4)` | yes — verbatim in `src/tensorrt_compute.cpp` | ✓ |
| Plane 5 is "exactly 3 liberties" (not >=3) | yes — `else if (libs == 3)` in `src/katago_inputs.cpp` | ✓ |
| Pool stat formulas match upstream `openclkernels.cpp` | yes — `_gpool_stats` and `_vhpool_stats` in `tools/katago_arch.py` use the documented constants 14.0, 0.1, 0.01 | ✓ |
| `tools/warm_init_from_katago.py` parser change is strictly additive | yes — required fields (`p1_conv`, `p1_bn`, `v1_conv`, `v1_bn`) unchanged; new fields all `Optional[...] = None` | ✓ |
| Warm-init produces byte-identical `training.pt` after the change | yes — re-ran on the same `.bin.gz`, same params transferred, same coverage | ✓ |

### Things this doc may have under-covered

- **Trunk-tip BN handling.** It's parsed in `_parse_trunk` (was
  already happening for stream-position) and now retained in
  `Trunk.trunk_tip_bn` / `Trunk.trunk_tip_act`. `KataGoNet.__init__`
  builds an `nn.BatchNorm2d` and calls `_load_bn` on it. That fact
  is in §11 implicitly but not called out.

- **The `state_data` / `state_size` round-trip.** The encoder writes
  to a `std::vector<float>`, MCTS hands `state.data()` /
  `state.size()` to `NNResultBuf::state_data` / `state_size`, the
  server thread copies them into `all_states[i]` for
  `predict_batch`. The KataGo-shaped state (`22*H*W + 19` floats)
  travels through this protocol unchanged — `state_data` is just a
  byte pointer and `state_size` is the count. Worth a sentence in
  §9 if a maintainer wonders why the queue protocol didn't need
  modification.

- **Why `Config::komi` is plumbed but `Config::input_global_channels`
  isn't.** Komi is consumed by the encoder via `game.komi`; the rest
  of the V7 globals are derived from game state too. There's no
  per-rule-set knob exposed; if one is needed (e.g. switching to
  Chinese rules), the right place is a new `Config::rules` enum
  passed to the encoder. Not in scope for this commit.

- **No mention of how the analysis HUD shows ownership.** It already
  worked for any backend that returned ownership; we just had to
  populate `MCTS::root_nn_ownership_` and copy it into
  `AnalysisInfo`. The `play` binary reads
  `info.root_ownership` (`src/main_play.cpp:574-590`) — no new
  rendering code was needed.

- **`KataGoNet.eval()` is called in the constructor** (last line of
  `__init__`) so `torch.onnx.export` runs in inference mode and BN
  uses running stats. If you ever switch to training-mode export
  the ONNX numerics will diverge.

If you find a claim above that doesn't match the code, treat the
code as authoritative and fix the doc. The diff is the spec; this
file is annotation.
