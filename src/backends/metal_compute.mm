#ifdef MINIGO_HAS_METAL

#include "backends/metal_compute.h"
#include "model/loaded_model.h"

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
// MetalComputeContext::Impl — shared Metal device + graph template
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
    // Interface contract: all three model formats (MiniGo resnet/vit
    // single-input, KataGo-V7 dual-input) are accepted here; the
    // implementation below is the placeholder that still throws for
    // both until MPSGraph kernels for the current architectures land.
    return std::make_unique<MetalComputeHandle>(impl_, model);
}

// ================================================================
// MetalComputeHandle::Impl — per-thread MPSGraph + tensors
// ================================================================

struct MetalComputeHandle::Impl {
    MetalComputeContext::Impl* ctx = nullptr;
    MPSGraph*           graph       = nil;
    MPSGraphTensor*     inputTensor = nil;
    MPSGraphTensor*     policyOutput = nil;
    MPSGraphTensor*     valueOutput  = nil;
    int board_size = 0, input_channels = 0;

    // Graph construction helpers
    MPSGraphTensor* addConv2d(MPSGraphTensor* input, const std::vector<float>& w,
                               int c_out, int c_in, int kH, int kW) {
        MPSGraphTensor* wfp32 = [graph constantWithData:
            [NSData dataWithBytes:w.data() length:w.size() * sizeof(float)]
            shape:@[@(c_out), @(c_in), @(kH), @(kW)] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* weight = [graph castTensor:wfp32 toType:MPSDataTypeFloat16 name:nil];

        auto* desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1
            groups:1
            paddingLeft:(kW>1?1:0) paddingRight:(kW>1?1:0)
            paddingTop:(kH>1?1:0) paddingBottom:(kH>1?1:0)
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        return [graph convolution2DWithSourceTensor:input weightsTensor:weight
                                         descriptor:desc name:nil];
    }

    MPSGraphTensor* addBN(MPSGraphTensor* input, const std::vector<float>& sc,
                           const std::vector<float>& bi, int ch) {
        MPSGraphTensor* s32 = [graph constantWithData:
            [NSData dataWithBytes:sc.data() length:sc.size()*sizeof(float)]
            shape:@[@(ch),@1,@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* b32 = [graph constantWithData:
            [NSData dataWithBytes:bi.data() length:bi.size()*sizeof(float)]
            shape:@[@(ch),@1,@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* s = [graph castTensor:s32 toType:MPSDataTypeFloat16 name:nil];
        MPSGraphTensor* b = [graph castTensor:b32 toType:MPSDataTypeFloat16 name:nil];
        auto* scaled = [graph multiplicationWithPrimaryTensor:input secondaryTensor:s name:nil];
        return [graph additionWithPrimaryTensor:scaled secondaryTensor:b name:nil];
    }

    MPSGraphTensor* addReLU(MPSGraphTensor* x) { return [graph reLUWithTensor:x name:nil]; }

    MPSGraphTensor* addFC(MPSGraphTensor* input, const std::vector<float>& w,
                           const std::vector<float>& b, int out, int in) {
        MPSGraphTensor* w32 = [graph constantWithData:
            [NSData dataWithBytes:w.data() length:w.size()*sizeof(float)]
            shape:@[@(out),@(in)] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* wt = [graph castTensor:w32 toType:MPSDataTypeFloat16 name:nil];
        MPSGraphTensor* b32 = [graph constantWithData:
            [NSData dataWithBytes:b.data() length:b.size()*sizeof(float)]
            shape:@[@1,@(out)] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* bias = [graph castTensor:b32 toType:MPSDataTypeFloat16 name:nil];
        MPSGraphTensor* wtT = [graph transposeTensor:wt dimension:0 withDimension:1 name:nil];
        auto* mm = [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:wtT name:nil];
        return [graph additionWithPrimaryTensor:mm secondaryTensor:bias name:nil];
    }
};

MetalComputeHandle::MetalComputeHandle(MetalComputeContext::Impl* ctx_impl,
                                         const LoadedModel* model) {
    // TODO(resnet_v2): the new KataGo-style ResNet (alternating SE + GPool
    // residual blocks, global-pool value/score heads) is not yet supported
    // by the Metal/MPSGraph backend.  To re-enable it, build the graph for:
    //   - SEModule (global avg pool + small FC + sigmoid + broadcast mul)
    //   - GPoolResBlock (parallel conv_main + conv_pool, mean+max pool,
    //     FC producing per-channel additive bias for conv_main)
    //   - GPoolHead (1x1 conv + mean+max pool + 2-layer FC)
    // and teach loaded_model.cpp to populate per-block weight slots.
    // Until then, use the TensorRT backend.
    (void)ctx_impl; (void)model;
    throw std::runtime_error(
        "Metal backend: implementation is a placeholder (TODO).  "
        "All three model formats are accepted at the interface "
        "(MiniGo resnet/vit single-input, KataGo-V7 dual-input) but "
        "the kernels for the current architectures are not written "
        "yet — use the TensorRT backend.");

    impl_ = new Impl();
    impl_->ctx = ctx_impl;
    impl_->board_size = model->board_size;
    impl_->input_channels = model->input_channels;

    auto* g = impl_;
    g->graph = [[MPSGraph alloc] init];

    int H = model->board_size, W = model->board_size, HW = H * W;
    int nf = model->num_filters;

    // Input placeholder [N, C, H, W] FP32 → cast to FP16
    g->inputTensor = [g->graph placeholderWithShape:@[@(-1), @(model->input_channels), @(H), @(W)]
                                           dataType:MPSDataTypeFloat32 name:@"input"];
    MPSGraphTensor* x = [g->graph castTensor:g->inputTensor toType:MPSDataTypeFloat16 name:nil];

    // Input conv + BN + ReLU
    x = g->addConv2d(x, model->input_conv.weight, nf, model->input_channels, 3, 3);
    x = g->addBN(x, model->input_conv.bn_scale, model->input_conv.bn_bias, nf);
    x = g->addReLU(x);

    // Residual blocks
    for (int i = 0; i < model->num_res_blocks; i++) {
        MPSGraphTensor* residual = x;
        x = g->addConv2d(x, model->res_conv1[i].weight, nf, nf, 3, 3);
        x = g->addBN(x, model->res_conv1[i].bn_scale, model->res_conv1[i].bn_bias, nf);
        x = g->addReLU(x);
        x = g->addConv2d(x, model->res_conv2[i].weight, nf, nf, 3, 3);
        x = g->addBN(x, model->res_conv2[i].bn_scale, model->res_conv2[i].bn_bias, nf);
        x = [g->graph additionWithPrimaryTensor:x secondaryTensor:residual name:nil];
        x = g->addReLU(x);
    }

    // Policy head
    {
        int pc = model->policy_conv.c_out;
        MPSGraphTensor* pol = g->addConv2d(x, model->policy_conv.weight, pc, nf, 1, 1);
        pol = g->addBN(pol, model->policy_conv.bn_scale, model->policy_conv.bn_bias, pc);
        pol = g->addReLU(pol);
        pol = [g->graph reshapeTensor:pol withShape:@[@(-1), @(pc * HW)] name:nil];
        int as = HW + 1;
        pol = g->addFC(pol, model->policy_fc.weight, model->policy_fc.bias, as, pc * HW);
        pol = [g->graph castTensor:pol toType:MPSDataTypeFloat32 name:nil];
        g->policyOutput = [g->graph softMaxWithTensor:pol axis:1 name:nil];
    }

    // Value head
    {
        int vc = model->value_conv.c_out;
        MPSGraphTensor* val = g->addConv2d(x, model->value_conv.weight, vc, nf, 1, 1);
        val = g->addBN(val, model->value_conv.bn_scale, model->value_conv.bn_bias, vc);
        val = g->addReLU(val);
        val = [g->graph reshapeTensor:val withShape:@[@(-1), @(vc * HW)] name:nil];
        val = g->addFC(val, model->value_fc1.weight, model->value_fc1.bias,
                        model->value_fc1.out_features, vc * HW);
        val = g->addReLU(val);
        val = g->addFC(val, model->value_fc2.weight, model->value_fc2.bias,
                        1, model->value_fc1.out_features);
        val = [g->graph castTensor:val toType:MPSDataTypeFloat32 name:nil];
        g->valueOutput = [g->graph tanhWithTensor:val name:nil];
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

    int N = (int)states.size();
    int H = impl_->board_size, W = H;
    int HW = H * W;
    int C = impl_->input_channels;
    int action_size = HW + 1;

    size_t input_floats = (size_t)N * C * HW;
    std::vector<float> flat;
    flat.reserve(input_floats);
    for (auto& s : states)
        flat.insert(flat.end(), s.begin(), s.end());

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

        polPtr = (float*)malloc(N * action_size * sizeof(float));
        valPtr = (float*)malloc(N * sizeof(float));
        [polArr readBytes:polPtr strideBytes:nil];
        [valArr readBytes:valPtr strideBytes:nil];
    }

    std::vector<Result> output(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(action_size);
        for (int a = 0; a < action_size; a++)
            pol[a] = polPtr[n * action_size + a];
        output[n].policy = std::move(pol);
        output[n].value  = valPtr[n];
        output[n].score  = 0.0f;  // TODO: Metal score head not yet implemented
    }

    free(polPtr);
    free(valPtr);
    return output;
}

}  // namespace minigo
#endif  // MINIGO_HAS_METAL
