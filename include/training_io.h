#pragma once

// Shared writer for V4 self-play records.  Both selfplay and bootstrap
// binaries emit the same layout; centralizing here keeps the magic +
// version + per-record slot list in one place.

#include "mcts.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minigo {

constexpr std::uint16_t TRAIN_RECORD_MAGIC   = 0x4D47;  // 'MG'
constexpr std::uint16_t TRAIN_RECORD_VERSION = 4;

void write_records(const std::string& path,
                   const std::vector<TrainingRecord>& records,
                   int board_rows,
                   int board_cols);

}  // namespace minigo
