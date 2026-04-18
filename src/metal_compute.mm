#ifdef MINIGO_HAS_METAL

#include "metal_compute.h"
#include "loaded_model.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace minigo {

// ================================================================
// MetalComputeContext::Impl - shared Metal device + command queue
// ================================================================

struct MetalComputeContext::Impl {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> commandQueue = nil;
};

MetalComputeContext::MetalComputeContext(const std::vector<int>& /*device_ids*/) {
    impl_ = new Impl();
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device)
        throw std::runtime_error("Metal: no GPU device found");
    impl_->commandQueue = [impl_->device newCommandQueue];
    std::cout << "Metal device: " << [impl_->device.name UTF8String] << "\n";
}

MetalComputeContext::~MetalComputeContext() {
    delete impl_;
}

std::unique_ptr<ComputeHandle>
MetalComputeContext::create_handle(const LoadedModel* model, int /*gpu_id*/, int /*max_batch_size*/) {
    return std::make_unique<MetalComputeHandle>(impl_, model);
}

// ================================================================
// MetalComputeHandle::Impl - per-handle MPSGraph + graph-construction helpers
// ================================================================

struct MetalComputeHandle::Impl {
    MetalComputeContext::Impl* ctx = nil;
    MPSGraph*       graph       = nil;
    MPSGraphTensor* inputTensor = nil;
    MPSGraphTensor* policyOutput = nil;
    MPSGraphTensor* valueOutput  = nil;
    int board_rows = 0, board_cols = 0, input_channels = 0, action_size = 0;
    int value_head_size = 3;

    // ── graph-construction helpers ───────────────────────────────

    MPSGraphTensor* constF16(const std::vector<float>& src, NSArray<NSNumber*>* shape) {
        MPSGraphTensor* fp32 = [graph constantWithData:
            [NSData dataWithBytes:src.data() length:src.size() * sizeof(float)]
            shape:shape dataType:MPSDataTypeFloat32];
        return [graph castTensor:fp32 toType:MPSDataTypeFloat16 name:nil];
    }

    MPSGraphTensor* conv2d(MPSGraphTensor* input, const std::vector<float>& w,
                           int c_out, int c_in, int kH, int kW) {
        MPSGraphTensor* weight = constF16(w, @[@(c_out), @(c_in), @(kH), @(kW)]);
        int pad = (kW > 1 || kH > 1) ? 1 : 0;
        auto* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1
            groups:1
            paddingLeft:pad paddingRight:pad paddingTop:pad paddingBottom:pad
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        return [graph convolution2DWithSourceTensor:input weightsTensor:weight
                                         descriptor:desc name:nil];
    }

    MPSGraphTensor* applyBN(MPSGraphTensor* x,
                            const std::vector<float>& scale,
                            const std::vector<float>& biasv, int ch) {
        MPSGraphTensor* s = constF16(scale, @[@(ch), @1, @1]);
        MPSGraphTensor* b = constF16(biasv, @[@(ch), @1, @1]);
        auto* scaled = [graph multiplicationWithPrimaryTensor:x secondaryTensor:s name:nil];
        return [graph additionWithPrimaryTensor:scaled secondaryTensor:b name:nil];
    }

    MPSGraphTensor* relu(MPSGraphTensor* x) {
        return [graph reLUWithTensor:x name:nil];
    }

    MPSGraphTensor* linear(MPSGraphTensor* input,
                           const std::vector<float>& w,
                           const std::vector<float>& b,
                           int out_features, int in_features) {
        MPSGraphTensor* weight = constF16(w, @[@(out_features), @(in_features)]);
        MPSGraphTensor* bias = constF16(b, @[@1, @(out_features)]);
        MPSGraphTensor* wt = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
        auto* mm = [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:wt name:nil];
        return [graph additionWithPrimaryTensor:mm secondaryTensor:bias name:nil];
    }

    MPSGraphTensor* convBNReLU(MPSGraphTensor* input, const ConvBNWeights& conv) {
        int pad_k = conv.k;
        MPSGraphTensor* x = conv2d(input, conv.weight, conv.c_out, conv.c_in, pad_k, pad_k);
        x = applyBN(x, conv.bn_scale, conv.bn_bias, conv.c_out);
        return relu(x);
    }

    MPSGraphTensor* convBN(MPSGraphTensor* input, const ConvBNWeights& conv) {
        MPSGraphTensor* x = conv2d(input, conv.weight, conv.c_out, conv.c_in, conv.k, conv.k);
        return applyBN(x, conv.bn_scale, conv.bn_bias, conv.c_out);
    }

    // ── Residual block variants ──────────────────────────────────

    MPSGraphTensor* plainBlock(MPSGraphTensor* x, const BlockWeights& blk) {
        MPSGraphTensor* residual = x;
        MPSGraphTensor* y = convBNReLU(x, blk.conv1);
        y = convBN(y, blk.conv2);
        y = [graph additionWithPrimaryTensor:y secondaryTensor:residual name:nil];
        return relu(y);
    }

    MPSGraphTensor* seBlock(MPSGraphTensor* x, const BlockWeights& blk) {
        int C = blk.conv2.c_out;
        MPSGraphTensor* residual = x;
        MPSGraphTensor* y = convBNReLU(x, blk.conv1);
        y = convBN(y, blk.conv2);  // [B, C, H, W], no relu yet

        // Global average pool over spatial dims -> [B, C]
        MPSGraphTensor* pooled = [graph meanOfTensor:y axes:@[@2, @3] name:nil];
        pooled = [graph squeezeTensor:pooled axes:@[@2, @3] name:nil];

        MPSGraphTensor* h = linear(pooled, blk.se_fc1.weight, blk.se_fc1.bias,
                                   blk.se_fc1.out_features, blk.se_fc1.in_features);
        h = relu(h);
        MPSGraphTensor* gate = linear(h, blk.se_fc2.weight, blk.se_fc2.bias,
                                      blk.se_fc2.out_features, blk.se_fc2.in_features);
        gate = [graph sigmoidWithTensor:gate name:nil];
        gate = [graph reshapeTensor:gate withShape:@[@(-1), @(C), @1, @1] name:nil];

        y = [graph multiplicationWithPrimaryTensor:y secondaryTensor:gate name:nil];
        y = [graph additionWithPrimaryTensor:y secondaryTensor:residual name:nil];
        return relu(y);
    }

    MPSGraphTensor* gpoolBlock(MPSGraphTensor* x, const BlockWeights& blk) {
        int C = blk.conv2.c_out;
        MPSGraphTensor* residual = x;

        MPSGraphTensor* y = convBNReLU(x, blk.conv1);

        // Parallel pool branch (from the block input, matches the Python model).
        MPSGraphTensor* p = convBNReLU(x, blk.pool_conv);
        MPSGraphTensor* pool_mean = [graph meanOfTensor:p axes:@[@2, @3] name:nil];
        pool_mean = [graph squeezeTensor:pool_mean axes:@[@2, @3] name:nil];
        MPSGraphTensor* pool_max = [graph reductionMaximumWithTensor:p axes:@[@2, @3] name:nil];
        pool_max = [graph squeezeTensor:pool_max axes:@[@2, @3] name:nil];
        MPSGraphTensor* stats = [graph concatTensor:pool_mean withTensor:pool_max
                                          dimension:1 name:nil];   // [B, 2C]
        MPSGraphTensor* bias = linear(stats, blk.pool_fc.weight, blk.pool_fc.bias,
                                      blk.pool_fc.out_features, blk.pool_fc.in_features);
        bias = [graph reshapeTensor:bias withShape:@[@(-1), @(C), @1, @1] name:nil];
        y = [graph additionWithPrimaryTensor:y secondaryTensor:bias name:nil];

        y = convBN(y, blk.conv2);
        y = [graph additionWithPrimaryTensor:y secondaryTensor:residual name:nil];
        return relu(y);
    }
};

// ================================================================
// Graph construction
// ================================================================

MetalComputeHandle::MetalComputeHandle(MetalComputeContext::Impl* ctx_impl,
                                        const LoadedModel* model) {
    impl_ = new Impl();
    impl_->ctx = ctx_impl;
    impl_->board_rows = model->board_rows;
    impl_->board_cols = model->board_cols;
    impl_->input_channels = model->input_channels;
    impl_->action_size = model->action_size;
    impl_->value_head_size = model->value_head_size;

    auto* g = impl_;
    g->graph = [[MPSGraph alloc] init];

    const int H = model->board_rows, W = model->board_cols, HW = H * W;
    const int nf = model->num_filters;

    // Input placeholder [N, C, H, W] fp32, cast to fp16 for compute.
    g->inputTensor = [g->graph placeholderWithShape:@[@(-1), @(model->input_channels), @(H), @(W)]
                                           dataType:MPSDataTypeFloat32 name:@"input"];
    MPSGraphTensor* x = [g->graph castTensor:g->inputTensor toType:MPSDataTypeFloat16 name:nil];

    // Input conv + BN + ReLU.
    x = g->convBNReLU(x, model->input_conv);

    // Residual tower.
    for (const BlockWeights& blk : model->blocks) {
        switch (blk.kind) {
            case BlockKind::SE:    x = g->seBlock(x, blk); break;
            case BlockKind::GPool: x = g->gpoolBlock(x, blk); break;
            case BlockKind::Plain: default: x = g->plainBlock(x, blk); break;
        }
    }

    // Policy head.
    {
        const int pc = model->policy_conv.c_out;
        MPSGraphTensor* pol = g->convBNReLU(x, model->policy_conv);
        pol = [g->graph reshapeTensor:pol withShape:@[@(-1), @(pc * HW)] name:nil];
        pol = g->linear(pol, model->policy_fc.weight, model->policy_fc.bias,
                        model->action_size, pc * HW);
        g->policyOutput = [g->graph castTensor:pol toType:MPSDataTypeFloat32 name:nil];
    }

    // Value head.  value_fc2 produces `value_head_size` logits; collapse to
    // P(win) - P(loss) for the 3-class WLD head.
    {
        const int vc = model->value_conv.c_out;
        MPSGraphTensor* val = g->convBNReLU(x, model->value_conv);
        val = [g->graph reshapeTensor:val withShape:@[@(-1), @(vc * HW)] name:nil];
        val = g->linear(val, model->value_fc1.weight, model->value_fc1.bias,
                        model->value_fc1.out_features, vc * HW);
        val = g->relu(val);
        val = g->linear(val, model->value_fc2.weight, model->value_fc2.bias,
                        model->value_fc2.out_features, model->value_fc2.in_features);
        val = [g->graph castTensor:val toType:MPSDataTypeFloat32 name:nil];

        if (model->value_head_size == 3) {
            MPSGraphTensor* wdl = [g->graph softMaxWithTensor:val axis:1 name:nil];
            // Slice P(win) and P(loss) along dim=1.
            MPSGraphTensor* pwin = [g->graph sliceTensor:wdl dimension:1 start:0 length:1 name:nil];
            MPSGraphTensor* ploss = [g->graph sliceTensor:wdl dimension:1 start:2 length:1 name:nil];
            g->valueOutput = [g->graph subtractionWithPrimaryTensor:pwin
                                                     secondaryTensor:ploss name:nil];
        } else if (model->value_head_size == 1) {
            g->valueOutput = [g->graph tanhWithTensor:val name:nil];
        } else {
            // Generic fallback: softmax and report (first - last).
            MPSGraphTensor* sm = [g->graph softMaxWithTensor:val axis:1 name:nil];
            MPSGraphTensor* first = [g->graph sliceTensor:sm dimension:1 start:0 length:1 name:nil];
            MPSGraphTensor* last = [g->graph sliceTensor:sm dimension:1
                                                   start:model->value_head_size - 1
                                                  length:1 name:nil];
            g->valueOutput = [g->graph subtractionWithPrimaryTensor:first
                                                     secondaryTensor:last name:nil];
        }
    }
}

MetalComputeHandle::~MetalComputeHandle() {
    delete impl_;
}

// ================================================================
// predict_batch
// ================================================================

std::vector<MetalComputeHandle::Result>
MetalComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    const int N = (int)states.size();
    const int H = impl_->board_rows, W = impl_->board_cols;
    const int HW = H * W;
    const int C = impl_->input_channels;
    const int action_size = impl_->action_size;

    size_t input_floats = (size_t)N * C * HW;
    std::vector<float> flat;
    flat.reserve(input_floats);
    for (const auto& s : states) flat.insert(flat.end(), s.begin(), s.end());

    __block float* polPtr = nullptr;
    __block float* valPtr = nullptr;

    @autoreleasepool {
        NSData* data = [NSData dataWithBytesNoCopy:flat.data()
                                            length:input_floats * sizeof(float)
                                      freeWhenDone:NO];
        MPSGraphTensorData* inputTD = [[MPSGraphTensorData alloc]
            initWithDevice:[MPSGraphDevice deviceWithMTLDevice:impl_->ctx->device]
                      data:data
                     shape:@[@(N), @(C), @(H), @(W)]
                  dataType:MPSDataTypeFloat32];

        auto* results = [impl_->graph runWithMTLCommandQueue:impl_->ctx->commandQueue
                                                      feeds:@{ impl_->inputTensor: inputTD }
                                              targetTensors:@[impl_->policyOutput, impl_->valueOutput]
                                           targetOperations:nil];

        MPSNDArray* polArr = results[impl_->policyOutput].mpsndarray;
        MPSNDArray* valArr = results[impl_->valueOutput].mpsndarray;

        polPtr = (float*)malloc((size_t)N * action_size * sizeof(float));
        valPtr = (float*)malloc((size_t)N * sizeof(float));
        [polArr readBytes:polPtr strideBytes:nil];
        [valArr readBytes:valPtr strideBytes:nil];
    }

    std::vector<Result> output(N);
    for (int n = 0; n < N; ++n) {
        std::vector<float> pol(action_size);
        std::memcpy(pol.data(), polPtr + (size_t)n * action_size,
                    (size_t)action_size * sizeof(float));
        output[n].policy   = std::move(pol);
        output[n].value    = valPtr[n];
        output[n].score    = 0.0f;
        output[n].score_sd = 0.0f;
    }

    free(polPtr);
    free(valPtr);
    return output;
}

}  // namespace minigo
#endif  // MINIGO_HAS_METAL
