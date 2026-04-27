#pragma once

#include "game.h"
#include "loaded_model.h"
#include <vector>

namespace minigo {

// ================================================================
// KataGo V7 input encoder.
//
// Layout contract (as flat std::vector<float>):
//   out[0 .. 22 * H * W)             : spatial planes, row-major [C, H, W]
//   out[22 * H * W .. + 19)          : global feature vector
//   total size = 22 * H * W + 19
//
// The TensorRT KataGo branch (src/tensorrt_compute.cpp) splits this
// flat layout back into the two TRT inputs at predict_batch time.
//
// `model` carries the spatial / global channel counts (22 / 19 for
// kata1) so future format variants don't break this signature.
// ================================================================
void encode_for_katago(const GoGame& game,
                       const LoadedModel* model,
                       std::vector<float>& out);

}  // namespace minigo
