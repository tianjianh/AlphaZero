#include "training_io.h"

#include <fstream>

namespace minigo {

void write_records(const std::string& path,
                   const std::vector<TrainingRecord>& records,
                   int board_rows,
                   int board_cols) {
    std::ofstream out(path, std::ios::binary);

    std::uint16_t magic   = TRAIN_RECORD_MAGIC;
    std::uint16_t version = TRAIN_RECORD_VERSION;
    std::int32_t  n       = static_cast<std::int32_t>(records.size());
    std::int32_t  rows    = board_rows;
    std::int32_t  cols    = board_cols;
    out.write(reinterpret_cast<const char*>(&magic), 2);
    out.write(reinterpret_cast<const char*>(&version), 2);
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(&rows), 4);
    out.write(reinterpret_cast<const char*>(&cols), 4);

    int board_sq = board_rows * board_cols;

    for (auto& rec : records) {
        std::int32_t ss = static_cast<std::int32_t>(rec.state.size());
        std::int32_t ps = static_cast<std::int32_t>(rec.policy.size());
        out.write(reinterpret_cast<const char*>(&ss), 4);
        out.write(reinterpret_cast<const char*>(rec.state.data()), ss * sizeof(float));
        out.write(reinterpret_cast<const char*>(&ps), 4);
        out.write(reinterpret_cast<const char*>(rec.policy.data()), ps * sizeof(float));
        out.write(reinterpret_cast<const char*>(&rec.value), sizeof(float));
        out.write(reinterpret_cast<const char*>(rec.ownership.data()), board_sq * sizeof(float));
        std::int32_t opp = rec.opponent_action;
        out.write(reinterpret_cast<const char*>(&opp), 4);
    }
}

}  // namespace minigo
