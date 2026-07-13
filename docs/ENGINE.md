# Engine Internals

Search and evaluation architecture of the C++ engine: the Go board
engine, multi-threaded MCTS, AsyncBot, and the NNEvaluator batching
server.  Backend-specific material lives in [BACKENDS.md](BACKENDS.md).

## Overview

```
            LoadedModel (1, shared, main thread)
                │ CPU weights (ONNX parsed once)
                ▼
          ComputeContext (1, shared, main thread)
          ┌─────┴──────────────────┐
          │                        │
     DeviceState[GPU 0]      DeviceState[GPU 1]
          │                        │
     ┌────┼────┐              ┌────┼────┐
     │         │              │         │
  Handle_0  Handle_1       Handle_2  Handle_3
  (server0) (server1)      (server2) (server3)
  own bufs  own bufs        own bufs  own bufs
     │         │              │         │
     └─────────┴──────────────┴─────────┘
                      │
              NNEvaluator (1 instance)
     ┌────────────────────────────────────┐
     │  Shared Queue (NNResultBuf*)       │ ← search threads push
     │  Competing consumers               │ → N server threads pop
     └────────────────────────────────────┘
                      │
              MCTS Search Threads
     ┌────────────────────────────────────┐
     │  Worker 1 (game 1) ─┐             │
     │  Worker 2 (game 2) ─┤  search()   │
     │  Worker T (game T) ─┘  spawns N   │
     │    threads per move               │
     └────────────────────────────────────┘
                      │
              Self-Play Data (.bin)
                      ▼
          ┌─── Python ───────────┐
          │ train_continuous.py  │
          │  export_onnx.py      │──▶ models/*.onnx
          └──────────────────────┘
```

**All backends use the same NNEvaluator architecture** — even Eigen (CPU).
Each server thread creates its own `ComputeHandle` on its assigned GPU
(KataGo pattern).  The backend is selected at compile time via
`cmake -DMINIGO_BACKEND=...`.

### Multi-Threaded MCTS (KataGo pattern)

Each `MCTS::search()` call spawns `--search-threads` internal threads, all
working on the same tree with virtual loss for synchronization.  Each search
thread pre-allocates one `NNResultBuf` (mutex + condvar, created once,
reused for all evaluations during that thread's lifetime — matching KataGo's
`SearchThread` pattern):

1. **Descend**: walk the tree from root, applying virtual loss at each node.
   Only descend through `EXPANDED` nodes; stop at `UNEVALUATED` leaves.

2. **Evaluate**: call `evaluator->evaluate_with_buf(buf, state)` — pushes
   the pre-allocated `NNResultBuf` to the server queue and **blocks** on
   its condvar until the server processes the batch containing this leaf.

3. **Expand**: claim `UNEVALUATED → EXPANDING` with a single `fetch_or`
   (states encoded so the claimed bit is idempotent on `EXPANDED`), publish
   with a release store.  Only one thread expands each node; others that
   collide revert their virtual losses, `yield()`, and retry from the root.
   No `compare_exchange` anywhere on the search path — everything lowers to
   single AMOs, so it stays cheap on Zaamo-only RISC-V, x86, and ARM alike.

4. **Backprop**: undo virtual loss, increment visit count, update value
   (all via atomics — `std::atomic<int>` `fetch_add` for counts, fixed-point
   `std::atomic<int64_t>` `fetch_add` for the value sum — exact and
   interleaving-independent, unlike float accumulation).

**Batch size adapts naturally**: while the GPU processes batch N, search
threads descend and submit leaves for batch N+1.  Steady-state batch size
≈ total concurrent search threads across all games.

```
total_search_threads = min(games, threads) × search_threads
```

### AsyncBot — persistent worker + tree reuse (KataGo pattern)

The `AsyncBot` class (`include/async_bot.h`) wraps `MCTS + GoGame` and runs
all searches on **one persistent worker thread** that idle-waits on a
condvar between requests — matching KataGo's `internalSearchThreadLoop`
design.  A single `worker_loop()` handles all search modes (GENMOVE for
AI moves, PONDER for background search), so there's only one state
machine to reason about.

**Public API** (single-writer contract):

```cpp
// Callback configuration (persistent; applies to any subsequent search)
void set_callback(AnalysisCallback cb, int interval_ms, int max_pv);
void clear_callback();

// Synchronous operations (block until done)
int  gen_move(Stone color, int sims, float temp, bool noise);
bool play_move(Stone color, int action);

// Asynchronous operations (return immediately)
void start_ponder();                         // background search, no cb
void start_analyze(cb, interval_ms, max_pv); // set_callback + start_ponder
void stop();                                  // interrupt, wait for idle

// Queries (safe from any thread)
MCTS::AnalysisInfo get_analysis(int max_moves);
bool is_searching() const;
```

**Internals**: `pending_mode_` / `current_mode_` ∈ `{IDLE, GENMOVE,
PONDER, SHUTDOWN}` protected by `control_mutex_`.  Callers submit a
request by setting `pending_mode_` and notifying `worker_cv_`, then wait
on `done_cv_` for `current_mode_` to return to IDLE.  The worker spawns
a transient callback thread per iteration when a callback is configured,
joins it at the end of the iteration, then loops back to wait.

**Stop-flag ownership**: `MCTS::should_stop_` is set by `request_stop()`
(called from `stop_locked()`) and checked inside the playout loop.
Critically, it is **NOT** cleared inside `MCTS::search()` — the clear
happens in `AsyncBot::worker_loop` via `mcts_->reset_stop_flag()`
*inside* the same critical section as the `current_mode_` transition.
This avoids a race where a stop signal raised after the worker released
the lock but before it entered `search()` would be silently overwritten
by search()'s own clear.  Non-AsyncBot callers (selfplay, eval,
benchmark) never set the flag, so leaving it default-false is safe.

**Tree reuse**: the `MCTS` instance owns one persistent `root_` that
survives across `search()` calls.  `MCTS::make_move(action)` re-roots
the tree to the played child (preserving its subtree, discarding
siblings outside the lock).  Every search call passes `reuse_tree=true`
so subsequent searches accumulate visits rather than rebuilding.  See
`COMPARISON_WITH_KATAGO.md` §13 for the point-by-point comparison with
KataGo's `Search::makeMove` + `Search::beginSearch`.

**Live analysis flow** (used by the play UI):

1. `bot->set_callback(cb, 500, 10)` — register a callback that pushes
   an `AnalysisInfo` snapshot to a mutex-protected UI struct every 500ms.
2. `bot->start_ponder()` — spawns a GENMOVE or PONDER search via the
   worker; the worker spawns a callback thread for the duration.
3. UI thread reads the latest snapshot each frame under the same mutex
   and redraws.
4. On human move: `bot->play_move(color, action)` internally stops the
   ponder (worker's search returns via `request_stop`), advances game +
   tree, returns.  UI loop restarts ponder at the top of the next
   iteration via `maintain_bot_state`.
5. On AI move: `bot->gen_move(color)` runs a GENMOVE search through the
   same worker — the callback thread fires throughout, so the user sees
   the AI's search tree evolve live.

`MCTS::get_analysis(max_moves)` itself is a simple reader — it holds
`tree_mutex_` briefly, walks `root_->children`, and returns:

```cpp
struct AnalysisInfo {
    vector<MoveInfo> moves;   // top moves sorted by visits
    int   total_visits;       // root visit count
    float root_utility;       // mean Q (blended value + score)
    float root_score;         // NN raw score estimate (points)
};
```

Each `MoveInfo` contains: action, visits, prior (policy), utility
(Q-value).  Visit counts are atomic; priors are stable after the root
is expanded; `nn_score` is stored per-node so re-rooted trees carry
their own correct score without needing re-evaluation.

### KataGo-style NNEvaluator

The `NNEvaluator` (in `include/nn_evaluator.h`) is a server thread that:
1. Waits for at least one `NNResultBuf*` in the shared queue
2. Greedy drains up to `max_batch_size` items (no timeout, no threshold)
3. Runs one `predict_batch()` GPU call
4. Signals each client's condvar with the result

Each search thread's `NNResultBuf` is pre-allocated at the start of the
thread and reused across all evaluations — zero mutex/condvar creation
in the hot path.
