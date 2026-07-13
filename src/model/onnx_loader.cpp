#include "model/onnx_loader.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace minigo {
namespace onnx_parser {

// ----------------------------------------------------------------
// Minimal protobuf wire-format reader
// ----------------------------------------------------------------
enum WireType { VARINT = 0, FIXED64 = 1, LENGTH_DELIMITED = 2, FIXED32 = 5 };

struct Reader {
    const uint8_t* data;
    const uint8_t* end;

    Reader(const uint8_t* d, size_t len) : data(d), end(d + len) {}

    bool has_data() const { return data < end; }

    uint64_t read_varint() {
        uint64_t result = 0;
        int shift = 0;
        while (data < end) {
            uint8_t b = *data++;
            result |= (uint64_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) return result;
            shift += 7;
        }
        throw std::runtime_error("truncated varint");
    }

    uint32_t read_fixed32() {
        uint32_t v;
        std::memcpy(&v, data, 4);
        data += 4;
        return v;
    }

    uint64_t read_fixed64() {
        uint64_t v;
        std::memcpy(&v, data, 8);
        data += 8;
        return v;
    }

    std::pair<int, int> read_tag() {
        uint64_t t = read_varint();
        return {(int)(t >> 3), (int)(t & 7)};
    }

    Reader read_submessage() {
        uint64_t len = read_varint();
        Reader sub(data, (size_t)len);
        data += len;
        return sub;
    }

    void skip(int wire_type) {
        switch (wire_type) {
            case VARINT: read_varint(); break;
            case FIXED64: data += 8; break;
            case LENGTH_DELIMITED: { auto len = read_varint(); data += len; } break;
            case FIXED32: data += 4; break;
            default: throw std::runtime_error("unknown wire type");
        }
    }
};

// ----------------------------------------------------------------
// OnnxTensor::get_floats()
// ----------------------------------------------------------------
std::vector<float> OnnxTensor::get_floats() const {
    if (!float_data.empty()) return float_data;
    if (!raw_data.empty()) {
        size_t count = raw_data.size() / sizeof(float);
        std::vector<float> result(count);
        std::memcpy(result.data(), raw_data.data(), raw_data.size());
        return result;
    }
    return {};
}

// ----------------------------------------------------------------
// Parse a TensorProto submessage
// ----------------------------------------------------------------
static OnnxTensor parse_tensor(Reader r) {
    OnnxTensor t;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        switch (field) {
            case 1:  // dims (repeated int64)
                if (wire == LENGTH_DELIMITED) {
                    auto sub = r.read_submessage();
                    while (sub.has_data())
                        t.dims.push_back((int64_t)sub.read_varint());
                } else {
                    t.dims.push_back((int64_t)r.read_varint());
                }
                break;
            case 2:  // data_type
                t.data_type = (int)r.read_varint();
                break;
            case 4:  // float_data (packed repeated float)
                if (wire == LENGTH_DELIMITED) {
                    auto sub = r.read_submessage();
                    while (sub.has_data()) {
                        float v;
                        uint32_t bits = sub.read_fixed32();
                        std::memcpy(&v, &bits, 4);
                        t.float_data.push_back(v);
                    }
                } else {
                    float v;
                    uint32_t bits = r.read_fixed32();
                    std::memcpy(&v, &bits, 4);
                    t.float_data.push_back(v);
                }
                break;
            case 8:  // name
                if (wire == LENGTH_DELIMITED) {
                    auto len = r.read_varint();
                    t.name = std::string((const char*)r.data, (size_t)len);
                    r.data += len;
                } else {
                    r.skip(wire);
                }
                break;
            case 9:  // raw_data
            {
                auto len = r.read_varint();
                t.raw_data.assign(r.data, r.data + len);
                r.data += len;
                break;
            }
            default:
                r.skip(wire);
                break;
        }
    }
    return t;
}

// ----------------------------------------------------------------
// Parse GraphProto — collect initializer tensors
// ----------------------------------------------------------------
static std::vector<OnnxTensor> parse_graph(Reader r) {
    std::vector<OnnxTensor> initializers;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 5 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_submessage();
            initializers.push_back(parse_tensor(sub));
        } else {
            r.skip(wire);
        }
    }
    return initializers;
}

// ----------------------------------------------------------------
// Parse ModelProto — extract graph initializers
// ----------------------------------------------------------------
static std::vector<OnnxTensor> parse_model(const uint8_t* data, size_t size) {
    Reader r(data, size);
    std::vector<OnnxTensor> initializers;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 7 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_submessage();
            initializers = parse_graph(sub);
        } else {
            r.skip(wire);
        }
    }
    return initializers;
}

// ----------------------------------------------------------------
// Graph-input enumeration
// ----------------------------------------------------------------
// ONNX wire format:
//   GraphProto.input  is field 11 (repeated ValueInfoProto)
//   ValueInfoProto.name  is field 1 (string)
//   ValueInfoProto.type  is field 2 (TypeProto)
//     TypeProto.tensor_type  is field 1 (TypeProto.Tensor)
//       Tensor.shape  is field 2 (TensorShapeProto)
//         Shape.dim   is field 1 (repeated Dimension)
//           Dimension.dim_value  is field 1 (int64)
//           Dimension.dim_param  is field 2 (string)  → -1
static OnnxGraphIO parse_value_info(Reader r) {
    OnnxGraphIO io;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            // name
            auto len = r.read_varint();
            io.name = std::string((const char*)r.data, (size_t)len);
            r.data += len;
        } else if (field == 2 && wire == LENGTH_DELIMITED) {
            // type → TypeProto
            auto type_msg = r.read_submessage();
            while (type_msg.has_data()) {
                auto [tf, tw] = type_msg.read_tag();
                if (tf == 1 && tw == LENGTH_DELIMITED) {
                    // tensor_type → TypeProto.Tensor
                    auto tensor_msg = type_msg.read_submessage();
                    while (tensor_msg.has_data()) {
                        auto [tf2, tw2] = tensor_msg.read_tag();
                        if (tf2 == 2 && tw2 == LENGTH_DELIMITED) {
                            // shape → TensorShapeProto
                            auto shape_msg = tensor_msg.read_submessage();
                            while (shape_msg.has_data()) {
                                auto [sf, sw] = shape_msg.read_tag();
                                if (sf == 1 && sw == LENGTH_DELIMITED) {
                                    // dim → Dimension
                                    auto dim_msg = shape_msg.read_submessage();
                                    int64_t dim_value = -1;  // symbolic by default
                                    while (dim_msg.has_data()) {
                                        auto [df, dw] = dim_msg.read_tag();
                                        if (df == 1 && dw == VARINT) {
                                            dim_value = (int64_t)dim_msg.read_varint();
                                        } else {
                                            dim_msg.skip(dw);
                                        }
                                    }
                                    io.dims.push_back(dim_value);
                                } else {
                                    shape_msg.skip(sw);
                                }
                            }
                        } else {
                            tensor_msg.skip(tw2);
                        }
                    }
                } else {
                    type_msg.skip(tw);
                }
            }
        } else {
            r.skip(wire);
        }
    }
    return io;
}

static std::vector<OnnxGraphIO> parse_graph_inputs(Reader r) {
    std::vector<OnnxGraphIO> inputs;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 11 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_submessage();
            inputs.push_back(parse_value_info(sub));
        } else {
            r.skip(wire);
        }
    }
    return inputs;
}

static std::vector<OnnxGraphIO> parse_model_inputs(const uint8_t* data, size_t size) {
    Reader r(data, size);
    std::vector<OnnxGraphIO> inputs;
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 7 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_submessage();
            inputs = parse_graph_inputs(sub);
        } else {
            r.skip(wire);
        }
    }
    return inputs;
}

std::vector<OnnxGraphIO> parse_onnx_graph_inputs(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open model file: " + path);
    size_t file_size = (size_t)file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf(file_size);
    file.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)file_size);
    file.close();
    return parse_model_inputs(buf.data(), buf.size());
}

// ----------------------------------------------------------------
// Graph node op_type counting (for activation detection)
// ----------------------------------------------------------------
// GraphProto.node is field 1 (repeated NodeProto);
// NodeProto.op_type is field 4 (string).
static void count_graph_ops(Reader r, int& mish, int& softplus) {
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            auto node = r.read_submessage();
            while (node.has_data()) {
                auto [nf, nw] = node.read_tag();
                if (nf == 4 && nw == LENGTH_DELIMITED) {
                    auto len = node.read_varint();
                    std::string op((const char*)node.data, (size_t)len);
                    node.data += len;
                    if (op == "Mish") mish++;
                    else if (op == "Softplus") softplus++;
                } else {
                    node.skip(nw);
                }
            }
        } else {
            r.skip(wire);
        }
    }
}

bool graph_uses_mish(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open model file: " + path);
    size_t file_size = (size_t)file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf(file_size);
    file.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)file_size);
    file.close();

    int mish = 0, softplus = 0;
    Reader r(buf.data(), buf.size());
    while (r.has_data()) {
        auto [field, wire] = r.read_tag();
        if (field == 7 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_submessage();
            count_graph_ops(sub, mish, softplus);
        } else {
            r.skip(wire);
        }
    }
    // One Softplus is the score_stdev post-processing; a Mish trunk
    // decomposed at opset<=17 contributes one per activation site.
    return mish > 0 || softplus > 2;
}

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------
std::vector<OnnxTensor> parse_onnx_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open model file: " + path);

    size_t file_size = (size_t)file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf(file_size);
    file.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)file_size);
    file.close();

    return parse_model(buf.data(), buf.size());
}

}  // namespace onnx_parser
}  // namespace minigo
