#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Shared minimal ONNX protobuf parser (no external dependency).
// Used by both EigenEngine and OpenCLEngine to load weights.

namespace minigo {
namespace onnx_parser {

struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int data_type = 0;  // 1 = FLOAT

    // Return float data regardless of storage format
    std::vector<float> get_floats() const;

    int64_t total_elements() const {
        if (dims.empty()) return 0;
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    // Internal storage (use get_floats() to read)
    std::vector<float> float_data;
    std::vector<uint8_t> raw_data;
};

// Parse an ONNX model file and return all initializer tensors (weight arrays).
std::vector<OnnxTensor> parse_onnx_file(const std::string& path);

// One graph-level input or output tensor descriptor (name + shape).
// Symbolic shape entries (e.g. dynamic batch) are stored as -1.
struct OnnxGraphIO {
    std::string name;
    std::vector<int64_t> dims;
};

// Enumerate the graph inputs of an ONNX model. Used by LoadedModel to
// distinguish KataGo (two inputs: state_spatial + state_global) from
// MiniGo (one input).
std::vector<OnnxGraphIO> parse_onnx_graph_inputs(const std::string& path);

}  // namespace onnx_parser
}  // namespace minigo
