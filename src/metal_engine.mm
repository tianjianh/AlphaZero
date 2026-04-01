#ifdef MINIGO_HAS_METAL

#include "metal_engine.h"
#include "onnx_loader.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace minigo {

// ================================================================
// MPSGraph-based Metal implementation
//
// The entire AlphaZero forward pass is built as a single MPSGraph
// at load_model() time.  predict_batch() feeds input data through
// the pre-compiled graph.
//
// Key advantage over raw Metal compute: MPSGraph handles command
// buffer lifecycle, kernel fusion, and memory planning internally.
// @autoreleasepool around graph.run() matches KataGo's pattern.
// ================================================================

struct MetalEngine::Impl {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> commandQueue = nil;
    MPSGraph*           graph        = nil;

    // Graph input placeholder
    MPSGraphTensor* inputTensor = nil;   // [N, C, H, W]

    // Graph outputs
    MPSGraphTensor* policyOutput = nil;  // [N, action_size] softmaxed
    MPSGraphTensor* valueOutput  = nil;  // [N, 1]

    // Model metadata
    int board_size = 9;
    int input_channels = 17;
    int num_filters = 64;
    int num_res_blocks = 5;

    // ── Graph construction helpers ─────────────────────────────

    MPSGraphTensor* addConv2d(MPSGraphTensor* input,
                               const std::vector<float>& weight_data,
                               int c_out, int c_in, int kH, int kW,
                               const std::string& name) {
        // Weight layout for MPSGraph conv: [C_out, C_in, kH, kW] (OIHW)
        NSUInteger dims[] = { (NSUInteger)c_out, (NSUInteger)c_in,
                              (NSUInteger)kH, (NSUInteger)kW };
        // Create FP32 constant then cast to FP16 (data is provided as float32)
        MPSGraphTensor* weight_fp32 = [graph constantWithData:
            [NSData dataWithBytes:weight_data.data() length:weight_data.size() * sizeof(float)]
            shape:@[@(c_out), @(c_in), @(kH), @(kW)]
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* weight = [graph castTensor:weight_fp32
                                            toType:MPSDataTypeFloat16 name:nil];

        auto* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1
            dilationRateInX:1 dilationRateInY:1
            groups:1
            paddingLeft:(kW > 1 ? 1 : 0) paddingRight:(kW > 1 ? 1 : 0)
            paddingTop:(kH > 1 ? 1 : 0) paddingBottom:(kH > 1 ? 1 : 0)
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        return [graph convolution2DWithSourceTensor:input
                                      weightsTensor:weight
                                         descriptor:desc
                                               name:nil];
    }

    MPSGraphTensor* addBatchNorm(MPSGraphTensor* input,
                                  const std::vector<float>& scale_data,
                                  const std::vector<float>& bias_data,
                                  int channels) {
        // Pre-fused BN: output = scale * input + bias
        // (running_mean/var already folded into scale/bias during load_model)
        MPSGraphTensor* scale_fp32 = [graph constantWithData:
            [NSData dataWithBytes:scale_data.data() length:scale_data.size() * sizeof(float)]
            shape:@[@(channels), @1, @1]
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* scale = [graph castTensor:scale_fp32 toType:MPSDataTypeFloat16 name:nil];
        MPSGraphTensor* bias_fp32 = [graph constantWithData:
            [NSData dataWithBytes:bias_data.data() length:bias_data.size() * sizeof(float)]
            shape:@[@(channels), @1, @1]
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* bias = [graph castTensor:bias_fp32 toType:MPSDataTypeFloat16 name:nil];

        auto* scaled = [graph multiplicationWithPrimaryTensor:input
                                             secondaryTensor:scale
                                                        name:nil];
        return [graph additionWithPrimaryTensor:scaled
                               secondaryTensor:bias
                                          name:nil];
    }

    MPSGraphTensor* addReLU(MPSGraphTensor* input) {
        return [graph reLUWithTensor:input name:nil];
    }

    MPSGraphTensor* addFC(MPSGraphTensor* input,
                           const std::vector<float>& weight_data,
                           const std::vector<float>& bias_data,
                           int out_features, int in_features) {
        // input: [N, in_features], weight: [out_features, in_features]
        // MPSGraph matmul: A × B^T when transposing secondary
        MPSGraphTensor* weight_fp32 = [graph constantWithData:
            [NSData dataWithBytes:weight_data.data() length:weight_data.size() * sizeof(float)]
            shape:@[@(out_features), @(in_features)]
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* weight = [graph castTensor:weight_fp32 toType:MPSDataTypeFloat16 name:nil];
        MPSGraphTensor* bias_fp32 = [graph constantWithData:
            [NSData dataWithBytes:bias_data.data() length:bias_data.size() * sizeof(float)]
            shape:@[@1, @(out_features)]
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* bias = [graph castTensor:bias_fp32 toType:MPSDataTypeFloat16 name:nil];

        // matmul: [N, in] × [in, out] = [N, out]
        // Transpose weight from [out, in] to [in, out]
        MPSGraphTensor* weightT = [graph transposeTensor:weight
                                               dimension:0
                                           withDimension:1
                                                    name:nil];
        auto* mm = [graph matrixMultiplicationWithPrimaryTensor:input
                                               secondaryTensor:weightT
                                                          name:nil];
        return [graph additionWithPrimaryTensor:mm
                               secondaryTensor:bias
                                          name:nil];
    }
};

// ================================================================
// Constructor / Destructor
// ================================================================

MetalEngine::MetalEngine() {
    impl_ = new Impl();
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device)
        throw std::runtime_error("Metal: no GPU device found");
    impl_->commandQueue = [impl_->device newCommandQueue];
    std::cout << "Metal device: " << [impl_->device.name UTF8String] << "\n";
}

MetalEngine::~MetalEngine() {
    delete impl_;
}

// ================================================================
// load_model — parse ONNX and build MPSGraph
// ================================================================

void MetalEngine::load_model(const std::string& path) {
    using namespace onnx_parser;

    auto tensors = parse_onnx_file(path);
    std::unordered_map<std::string, OnnxTensor*> tm;
    for (auto& t : tensors) tm[t.name] = &t;

    auto get = [&](const std::string& name) -> OnnxTensor& {
        auto it = tm.find(name);
        if (it == tm.end())
            throw std::runtime_error("Missing tensor: " + name);
        return *it->second;
    };

    // ── Infer architecture ───────────────────────────────────
    auto& iw = get("input_conv.weight");
    input_channels = (int)iw.dims[1];
    num_filters    = (int)iw.dims[0];
    impl_->input_channels = input_channels;
    impl_->num_filters = num_filters;

    num_res_blocks = 0;
    while (tm.count("res_blocks." + std::to_string(num_res_blocks) + ".conv1.weight"))
        num_res_blocks++;
    impl_->num_res_blocks = num_res_blocks;

    auto& pfw = get("policy_fc.weight");
    int action_size = (int)pfw.dims[0];
    board_size = (int)std::round(std::sqrt((double)(action_size - 1)));
    impl_->board_size = board_size;

    std::cout << "Metal model: board=" << board_size
              << " filters=" << num_filters
              << " blocks=" << num_res_blocks
              << " channels=" << input_channels << "\n";

    const float eps = 1e-5f;

    // Helper: pre-fuse BN into (scale, bias)
    auto fuse_bn = [&](const std::string& prefix)
            -> std::pair<std::vector<float>, std::vector<float>> {
        auto gamma = get(prefix + ".weight").get_floats();
        auto beta  = get(prefix + ".bias").get_floats();
        auto mean  = get(prefix + ".running_mean").get_floats();
        auto var   = get(prefix + ".running_var").get_floats();
        int ch = (int)gamma.size();
        std::vector<float> sc(ch), bi(ch);
        for (int i = 0; i < ch; i++) {
            float inv_std = 1.0f / std::sqrt(var[i] + eps);
            sc[i] = gamma[i] * inv_std;
            bi[i] = beta[i] - gamma[i] * mean[i] * inv_std;
        }
        return { sc, bi };
    };

    // ── Build MPSGraph ───────────────────────────────────────
    impl_->graph = [[MPSGraph alloc] init];
    auto* g = impl_;

    int H = board_size, W = board_size;
    int HW = H * W;

    // Input placeholder: [N, C, H, W] with dynamic batch
    g->inputTensor = [g->graph placeholderWithShape:@[@(-1), @(input_channels), @(H), @(W)]
                                           dataType:MPSDataTypeFloat32
                                               name:@"input"];

    // ── Cast input to FP16 for all compute ─────────────────────
    // Input placeholder is FP32 (CPU data), cast to FP16 for GPU compute
    MPSGraphTensor* x_fp16 = [g->graph castTensor:g->inputTensor
                                           toType:MPSDataTypeFloat16
                                             name:@"input_to_fp16"];

    // ── Input conv + BN + ReLU ───────────────────────────────
    auto input_w = get("input_conv.weight").get_floats();
    auto [input_bn_s, input_bn_b] = fuse_bn("input_bn");

    MPSGraphTensor* x = g->addConv2d(x_fp16, input_w,
                                       num_filters, input_channels, 3, 3, "input_conv");
    x = g->addBatchNorm(x, input_bn_s, input_bn_b, num_filters);
    x = g->addReLU(x);

    // ── Residual blocks ──────────────────────────────────────
    for (int i = 0; i < num_res_blocks; i++) {
        std::string pfx = "res_blocks." + std::to_string(i);
        MPSGraphTensor* residual = x;

        // Conv1 + BN + ReLU
        auto w1 = get(pfx + ".conv1.weight").get_floats();
        auto [s1, b1] = fuse_bn(pfx + ".bn1");
        x = g->addConv2d(x, w1, num_filters, num_filters, 3, 3, "res" + std::to_string(i) + "_conv1");
        x = g->addBatchNorm(x, s1, b1, num_filters);
        x = g->addReLU(x);

        // Conv2 + BN + residual add + ReLU
        auto w2 = get(pfx + ".conv2.weight").get_floats();
        auto [s2, b2] = fuse_bn(pfx + ".bn2");
        x = g->addConv2d(x, w2, num_filters, num_filters, 3, 3, "res" + std::to_string(i) + "_conv2");
        x = g->addBatchNorm(x, s2, b2, num_filters);
        x = [g->graph additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
        x = g->addReLU(x);
    }

    // ── Policy head ──────────────────────────────────────────
    {
        auto pw = get("policy_conv.weight").get_floats();
        auto [ps, pb] = fuse_bn("policy_bn");
        int policy_channels = (int)get("policy_conv.weight").dims[0]; // 2

        MPSGraphTensor* pol = g->addConv2d(x, pw, policy_channels, num_filters, 1, 1, "policy_conv");
        pol = g->addBatchNorm(pol, ps, pb, policy_channels);
        pol = g->addReLU(pol);

        // Reshape [N, 2, H, W] → [N, 2*H*W]
        pol = [g->graph reshapeTensor:pol withShape:@[@(-1), @(policy_channels * HW)] name:nil];

        // FC + softmax
        auto fc_w = get("policy_fc.weight").get_floats();
        auto fc_b = get("policy_fc.bias").get_floats();
        pol = g->addFC(pol, fc_w, fc_b, action_size, policy_channels * HW);
        // Cast to FP32 before softmax (exp() overflows FP16 at inputs > ~11)
        pol = [g->graph castTensor:pol toType:MPSDataTypeFloat32 name:@"policy_to_fp32"];
        g->policyOutput = [g->graph softMaxWithTensor:pol axis:1 name:@"policy_softmax"];
    }

    // ── Value head ───────────────────────────────────────────
    {
        auto vw = get("value_conv.weight").get_floats();
        auto [vs, vb] = fuse_bn("value_bn");
        int value_channels = (int)get("value_conv.weight").dims[0]; // 1

        MPSGraphTensor* val = g->addConv2d(x, vw, value_channels, num_filters, 1, 1, "value_conv");
        val = g->addBatchNorm(val, vs, vb, value_channels);
        val = g->addReLU(val);

        // Reshape [N, 1, H, W] → [N, H*W]
        val = [g->graph reshapeTensor:val withShape:@[@(-1), @(value_channels * HW)] name:nil];

        // FC1 + ReLU
        auto fc1_w = get("value_fc1.weight").get_floats();
        auto fc1_b = get("value_fc1.bias").get_floats();
        int fc1_out = (int)get("value_fc1.weight").dims[0]; // 64
        val = g->addFC(val, fc1_w, fc1_b, fc1_out, value_channels * HW);
        val = g->addReLU(val);

        // FC2 + tanh
        auto fc2_w = get("value_fc2.weight").get_floats();
        auto fc2_b = get("value_fc2.bias").get_floats();
        val = g->addFC(val, fc2_w, fc2_b, 1, fc1_out);
        // Cast to FP32 before tanh (numerical stability for value output)
        val = [g->graph castTensor:val toType:MPSDataTypeFloat32 name:@"value_to_fp32"];
        g->valueOutput = [g->graph tanhWithTensor:val name:@"value_tanh"];
    }

    std::cout << "Metal graph built (MPSGraph)\n";
}

// ================================================================
// predict_batch — feed data through pre-compiled MPSGraph
// ================================================================

std::vector<std::pair<std::vector<float>, float>>
MetalEngine::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    int N = (int)states.size();
    int H = board_size, W = board_size;
    int HW = H * W;
    int action_size = HW + 1;

    // ── Flatten input states into contiguous NCHW buffer ─────
    size_t input_floats = (size_t)N * input_channels * HW;
    std::vector<float> flat_input;
    flat_input.reserve(input_floats);
    for (auto& s : states)
        flat_input.insert(flat_input.end(), s.begin(), s.end());

    // ── Run graph (KataGo pattern: @autoreleasepool per call) ─
    // This is critical for the NNEvaluator server thread which has
    // no implicit autorelease pool.  Without this, Metal objects
    // accumulate and the ObjC runtime hangs.
    __block float* policyPtr = nullptr;
    __block float* valuePtr = nullptr;
    __block NSUInteger policyLen = 0;
    __block NSUInteger valueLen = 0;

    @autoreleasepool {
        // Create input tensor data
        NSData* inputData = [NSData dataWithBytesNoCopy:flat_input.data()
                                                 length:input_floats * sizeof(float)
                                           freeWhenDone:NO];
        MPSGraphTensorData* inputTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:impl_->device]
                      data:inputData
                     shape:@[@(N), @(input_channels), @(H), @(W)]
                  dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            impl_->inputTensor: inputTD
        };

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            [impl_->graph runWithMTLCommandQueue:impl_->commandQueue
                                          feeds:feeds
                                  targetTensors:@[impl_->policyOutput, impl_->valueOutput]
                               targetOperations:nil];

        // Read results
        MPSGraphTensorData* policyTD = results[impl_->policyOutput];
        MPSGraphTensorData* valueTD  = results[impl_->valueOutput];

        // Copy results out before autorelease pool drains
        MPSNDArray* policyArr = policyTD.mpsndarray;
        MPSNDArray* valueArr  = valueTD.mpsndarray;

        policyLen = (NSUInteger)(N * action_size);
        valueLen  = (NSUInteger)N;

        policyPtr = (float*)malloc(policyLen * sizeof(float));
        valuePtr  = (float*)malloc(valueLen * sizeof(float));

        [policyArr readBytes:policyPtr strideBytes:nil];
        [valueArr  readBytes:valuePtr  strideBytes:nil];
    }

    // ── Pack results ─────────────────────────────────────────
    std::vector<std::pair<std::vector<float>, float>> output(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(action_size);
        for (int a = 0; a < action_size; a++)
            pol[a] = policyPtr[n * action_size + a];
        output[n] = { std::move(pol), valuePtr[n] };
    }

    free(policyPtr);
    free(valuePtr);

    return output;
}

std::pair<std::vector<float>, float>
MetalEngine::predict(const std::vector<float>& state) {
    auto results = predict_batch({ state });
    return results[0];
}

}  // namespace minigo
#endif  // MINIGO_HAS_METAL
