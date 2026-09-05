// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <string>

#include "llm_memory/llm_metal_kernels_source.h"

namespace {

// This extra entrypoint exercises the unchanged shared helper, not a workload
// entrypoint or full-backend acceptance. The canonical embedded source is kept
// intact and receives the same decode-contiguous compile selector as production.
constexpr char kHelperProbe[] = R"MSL(
kernel void llm_checksum_helper_probe(device const uint* input [[buffer(0)]],
                                      device uint2* output [[buffer(1)]],
                                      uint id [[thread_position_in_grid]]) {
  uint2 checksum = uint2(0u);
  for (uint i = 0; i < 4u; ++i) {
    mix_checksum_word(checksum, input[id * 4u + i], ulong(i), 12345u + i * 17u);
  }
  output[id] = checksum;
}
)MSL";

}  // namespace

TEST(LlmMetalChecksumHelperIntegrationTest, SharedAffineLanesExposeBoundedContentCollisions) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      GTEST_SKIP() << "metal-device-unavailable";
    }
    std::string source(LlmMetalKernelContract::kSource);
    source += kHelperProbe;
    NSString* source_string = [[NSString alloc] initWithBytes:source.data()
                                                    length:source.size()
                                                  encoding:NSUTF8StringEncoding];
    ASSERT_NE(source_string, nil);
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion2_3;
    options.preprocessorMacros = @{@"LLM_METAL_DECODE_CONTIGUOUS" : @1,
                                  @"LLM_METAL_DECODE_PAGED" : @0,
                                  @"LLM_METAL_PREFILL_CONTIGUOUS" : @0,
                                  @"LLM_METAL_PREFILL_PAGED" : @0};
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source_string options:options error:&error];
    ASSERT_NE(library, nil) << (error == nil ? "no compiler diagnostic" : error.description.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:@"llm_checksum_helper_probe"];
    ASSERT_NE(function, nil);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    ASSERT_NE(pipeline, nil) << (error == nil ? "no pipeline diagnostic" : error.description.UTF8String);
    ASSERT_GE(pipeline.maxTotalThreadsPerThreadgroup, 4U);

    const std::array<uint32_t, 16> input = {11, 23, 37, 43,  // original
                                          43, 37, 23, 11,  // content permutation, same visits
                                          12, 22, 37, 43,  // cancelling +1/-1
                                          10, 23, 37, 43}; // one changed bit, detectable control
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
                                                    length:sizeof(input)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [device newBufferWithLength:8 * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
    ASSERT_NE(input_buffer, nil);
    ASSERT_NE(output_buffer, nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    ASSERT_NE(queue, nil);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    ASSERT_NE(command, nil);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    ASSERT_NE(encoder, nil);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    ASSERT_EQ(command.status, MTLCommandBufferStatusCompleted)
        << (command.error == nil ? "no command diagnostic" : command.error.description.UTF8String);
    const auto* actual = static_cast<const uint32_t*>(output_buffer.contents);
    ASSERT_NE(actual, nullptr);

    // Independent hand-derived mod-2^32 goldens: sum(value)=114 (control 113),
    // sum(index)=6, sum(domain)=4*12345+17*6=49482.
    // a=sum(value)+49482;
    // b=sum(value)*0x9e3779b1+6*0x85ebca77+49482*0x7feb352d mod 2^32.
    // Both affine lanes therefore share the same content-sum collision.
    constexpr std::array<uint32_t, 8> expected = {
        49596, 3847175070U, 49596, 3847175070U,
        49596, 3847175070U, 49595, 1192739309U};
    for (size_t word = 0; word < expected.size(); ++word) {
      EXPECT_EQ(actual[word], expected[word]) << "result word " << word;
    }
    EXPECT_NE(actual[0], actual[6]);
    EXPECT_NE(actual[1], actual[7]);
  }
}
