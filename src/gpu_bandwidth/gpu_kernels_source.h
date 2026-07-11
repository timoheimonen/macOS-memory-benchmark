// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_kernels_source.h
 * @brief Canonical embedded MSL 2.3 source and its CPU-oracle contract
 *
 * This private header is included only by `metal_gpu_backend.mm`. Keeping the
 * source and its revision in one translation unit makes the SHA-256 provenance
 * unambiguous and avoids a generated `.metal` copy.
 */

#ifndef GPU_KERNELS_SOURCE_H
#define GPU_KERNELS_SOURCE_H

#include <cstdint>
#include <string_view>

namespace GpuKernelContract {

inline constexpr char kRevision[] = "gpu-linear-word-mod32-tg-reduce-v2";
inline constexpr uint32_t kMaxSimdgroupsPerThreadgroup = 32U;

// Pattern, per-element, and final-checksum constants are duplicated literally
// in the canonical MSL source. Timed domain weights/tokens are derived on the
// CPU and passed in KernelParams. Unsigned arithmetic intentionally wraps
// modulo 2^32.
inline constexpr uint32_t kPatternSeedHighMultiplier = 0x9e3779b9U;
inline constexpr uint32_t kPatternIndexMultiplier = 0x85ebca6bU;
inline constexpr uint32_t kPatternTagMultiplier = 0xc2b2ae35U;
inline constexpr uint32_t kPatternPassMultiplier = 0x27d4eb2fU;

inline constexpr uint32_t kTimedIndexFirst = 0x165667b1U;
inline constexpr uint32_t kTimedValueSecond = 0x9e3779b1U;
inline constexpr uint32_t kTimedIndexSecond = 0x7f4a7c15U;
inline constexpr uint32_t kTimedKeySeedHighFirst = 0x9e3779b9U;
inline constexpr uint32_t kTimedKeyBufferLowFirst = 0x85ebca6bU;
inline constexpr uint32_t kTimedKeyBufferHighFirst = 0xc2b2ae35U;
inline constexpr uint32_t kTimedKeyPassFirst = 0xd3a2646cU;
inline constexpr uint32_t kTimedKeyOperationFirst = 0xfd7046c5U;
inline constexpr uint32_t kTimedKeyDirectionFirst = 0xb55a4f09U;
inline constexpr uint32_t kTimedKeySeedLowSecond = 0x9e3779b1U;
inline constexpr uint32_t kTimedKeyBufferLowSecond = 0x7f4a7c15U;
inline constexpr uint32_t kTimedKeyBufferHighSecond = 0x94d049bbU;
inline constexpr uint32_t kTimedKeyPassSecond = 0xbf58476dU;
inline constexpr uint32_t kTimedKeyOperationSecond = 0x632be59bU;
inline constexpr uint32_t kTimedKeyDirectionSecond = 0x27d4eb2fU;
inline constexpr uint32_t kTimedElementDomainFirst = 0xa511e9b3U;
inline constexpr uint32_t kTimedElementDomainSecond = 0x63d83595U;
inline constexpr uint32_t kTimedDispatchDomainFirst = 0x243f6a89U;
inline constexpr uint32_t kTimedDispatchDomainSecond = 0xb7e15163U;

inline constexpr uint32_t kChecksumIndexFirst = 0x6d2b79f5U;
inline constexpr uint32_t kChecksumValueSecond = 0x1b873593U;
inline constexpr uint32_t kChecksumIndexSecond = 0x85ebca77U;

inline constexpr uint32_t kReadSourcePatternTag = 0x11U;
inline constexpr uint32_t kCopySourcePatternTag = 0x22U;
inline constexpr uint32_t kPoisonPatternTag = 0x33U;
inline constexpr uint32_t kWritePatternTag = 0x44U;
inline constexpr uint32_t kReadOperationTag = 1U;
inline constexpr uint32_t kWriteOperationTag = 2U;
inline constexpr uint32_t kCopyOperationTag = 3U;

/**
 * Canonical MSL 2.3 bytes hashed for `kernel_source_sha256`.
 *
 * Data is an affine sequence of little-endian 32-bit words. Full 16-byte
 * regions are always processed as consecutive `uint4` values. A final 0-15
 * byte suffix is packed into up to four low-byte words. The timed accumulator
 * applies pass-specific odd domain weights and one token per dispatch to its
 * affine data/index reduction. The separate final checksum remains affine.
 * GPU and CPU use unsigned modulo-2^32 arithmetic; only the GPU consumes the
 * measured buffers.
 */
inline constexpr std::string_view kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct KernelParams {
  ulong vector_count;
  uint seed_low;
  uint seed_high;
  uint pattern_tag;
  uint pattern_pass;
  uint pass_index;
  uint tail_bytes;
  uint timed_element_weight_first;
  uint timed_element_weight_second;
  uint timed_dispatch_token_first;
  uint timed_dispatch_token_second;
};

inline uint pattern_word(ulong word_index, constant KernelParams& params,
                         uint pattern_pass) {
  return params.seed_low + params.seed_high * 0x9e3779b9u +
         uint(word_index) * 0x85ebca6bu +
         params.pattern_tag * 0xc2b2ae35u +
         pattern_pass * 0x27d4eb2fu;
}

inline uint byte_mask(uint byte_count) {
  return byte_count >= 4u ? 0xffffffffu :
         ((1u << (byte_count * 8u)) - 1u);
}

inline uint load_tail_word(const device uchar* bytes, ulong offset,
                           uint byte_count) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < byte_count; ++byte_index) {
    value |= uint(bytes[offset + byte_index]) << (byte_index * 8u);
  }
  return value;
}

inline void store_tail_word(device uchar* bytes, ulong offset,
                            uint byte_count, uint value) {
  for (uint byte_index = 0u; byte_index < byte_count; ++byte_index) {
    bytes[offset + byte_index] = uchar(value >> (byte_index * 8u));
  }
}

inline void accumulate_timed(thread uint2& accumulator, uint value,
                             uint word_index) {
  accumulator.x += value + word_index * 0x165667b1u;
  accumulator.y += value * 0x9e3779b1u +
                   word_index * 0x7f4a7c15u;
}

inline void finalize_timed_accumulator(thread uint2& accumulator,
                                       uint global_id,
                                       constant KernelParams& params) {
  accumulator *= uint2(params.timed_element_weight_first,
                       params.timed_element_weight_second);
  if (global_id == 0u) {
    accumulator += uint2(params.timed_dispatch_token_first,
                         params.timed_dispatch_token_second);
  }
}

inline void accumulate_final(thread uint2& accumulator, uint value,
                             uint word_index) {
  accumulator.x += value + word_index * 0x6d2b79f5u;
  accumulator.y += value * 0x1b873593u +
                   word_index * 0x85ebca77u;
}

inline void commit_threadgroup_reduction(
    uint2 local, uint simd_lane, uint simdgroup_index,
    uint simdgroup_count, uint thread_index,
    threadgroup uint2* partials, device atomic_uint* first,
    device atomic_uint* second) {
  const uint reduced_first = simd_sum(local.x);
  const uint reduced_second = simd_sum(local.y);
  if (simd_lane == 0u && simdgroup_index < 32u) {
    partials[simdgroup_index] = uint2(reduced_first, reduced_second);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0u) {
    uint2 threadgroup_sum = uint2(0u);
    const uint bounded_count = min(simdgroup_count, 32u);
    for (uint group = 0u; group < bounded_count; ++group) {
      threadgroup_sum += partials[group];
    }
    atomic_fetch_add_explicit(first, threadgroup_sum.x,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(second, threadgroup_sum.y,
                              memory_order_relaxed);
  }
}

kernel void gpu_fill_pattern(device uchar* destination [[buffer(0)]],
                             constant KernelParams& params [[buffer(1)]],
                             uint global_id [[thread_position_in_grid]],
                             uint grid_size [[threads_per_grid]]) {
  device uint4* vectors = reinterpret_cast<device uint4*>(destination);
  for (ulong vector_index = global_id; vector_index < params.vector_count;
       vector_index += grid_size) {
    const ulong word_index = vector_index * 4u;
    vectors[vector_index] = uint4(
        pattern_word(word_index, params, params.pattern_pass),
        pattern_word(word_index + 1u, params, params.pattern_pass),
        pattern_word(word_index + 2u, params, params.pattern_pass),
        pattern_word(word_index + 3u, params, params.pattern_pass));
  }

  if (global_id == 0u && params.tail_bytes != 0u) {
    const ulong tail_offset = params.vector_count * 16u;
    const uint tail_word_count = (params.tail_bytes + 3u) / 4u;
    for (uint tail_word = 0u; tail_word < tail_word_count; ++tail_word) {
      const uint consumed = tail_word * 4u;
      const uint count = min(4u, params.tail_bytes - consumed);
      const ulong word_index = params.vector_count * 4u + tail_word;
      store_tail_word(destination, tail_offset + consumed, count,
                      pattern_word(word_index, params, params.pattern_pass));
    }
  }
}

kernel void gpu_read_bandwidth(const device uchar* source [[buffer(0)]],
                               device atomic_uint* status [[buffer(1)]],
                               constant KernelParams& params [[buffer(2)]],
                               uint global_id [[thread_position_in_grid]],
                               uint grid_size [[threads_per_grid]],
                               uint thread_index [[thread_index_in_threadgroup]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                                   [[simdgroup_index_in_threadgroup]],
                               uint simdgroup_count
                                   [[simdgroups_per_threadgroup]]) {
  threadgroup uint2 partials[32];
  const device uint4* vectors =
      reinterpret_cast<const device uint4*>(source);
  uint2 local = uint2(0u);
  for (ulong vector_index = global_id; vector_index < params.vector_count;
       vector_index += grid_size) {
    const uint4 values = vectors[vector_index];
    const uint word_index = uint(vector_index * 4u);
    accumulate_timed(local, values.x, word_index);
    accumulate_timed(local, values.y, word_index + 1u);
    accumulate_timed(local, values.z, word_index + 2u);
    accumulate_timed(local, values.w, word_index + 3u);
  }
  if (global_id == 0u && params.tail_bytes != 0u) {
    const ulong tail_offset = params.vector_count * 16u;
    const uint tail_word_count = (params.tail_bytes + 3u) / 4u;
    for (uint tail_word = 0u; tail_word < tail_word_count; ++tail_word) {
      const uint consumed = tail_word * 4u;
      const uint count = min(4u, params.tail_bytes - consumed);
      accumulate_timed(
          local, load_tail_word(source, tail_offset + consumed, count),
          uint(params.vector_count * 4u + tail_word));
    }
  }
  finalize_timed_accumulator(local, global_id, params);
  commit_threadgroup_reduction(local, simd_lane, simdgroup_index,
                               simdgroup_count, thread_index, partials,
                               &status[0], &status[1]);
}

kernel void gpu_write_bandwidth(device uchar* destination [[buffer(0)]],
                                device atomic_uint* status [[buffer(1)]],
                                constant KernelParams& params [[buffer(2)]],
                                uint global_id [[thread_position_in_grid]],
                                uint grid_size [[threads_per_grid]],
                                uint thread_index [[thread_index_in_threadgroup]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simdgroup_index
                                    [[simdgroup_index_in_threadgroup]],
                                uint simdgroup_count
                                    [[simdgroups_per_threadgroup]]) {
  threadgroup uint2 partials[32];
  device uint4* vectors = reinterpret_cast<device uint4*>(destination);
  uint2 local = uint2(0u);
  for (ulong vector_index = global_id; vector_index < params.vector_count;
       vector_index += grid_size) {
    const ulong base_word = vector_index * 4u;
    const uint4 values = uint4(
        pattern_word(base_word, params, params.pass_index),
        pattern_word(base_word + 1u, params, params.pass_index),
        pattern_word(base_word + 2u, params, params.pass_index),
        pattern_word(base_word + 3u, params, params.pass_index));
    vectors[vector_index] = values;
    const uint word_index = uint(base_word);
    accumulate_timed(local, values.x, word_index);
    accumulate_timed(local, values.y, word_index + 1u);
    accumulate_timed(local, values.z, word_index + 2u);
    accumulate_timed(local, values.w, word_index + 3u);
  }
  if (global_id == 0u && params.tail_bytes != 0u) {
    const ulong tail_offset = params.vector_count * 16u;
    const uint tail_word_count = (params.tail_bytes + 3u) / 4u;
    for (uint tail_word = 0u; tail_word < tail_word_count; ++tail_word) {
      const uint consumed = tail_word * 4u;
      const uint count = min(4u, params.tail_bytes - consumed);
      const ulong word_index = params.vector_count * 4u + tail_word;
      const uint value = pattern_word(word_index, params, params.pass_index) &
                         byte_mask(count);
      store_tail_word(destination, tail_offset + consumed, count, value);
      accumulate_timed(local, value, uint(word_index));
    }
  }
  finalize_timed_accumulator(local, global_id, params);
  commit_threadgroup_reduction(local, simd_lane, simdgroup_index,
                               simdgroup_count, thread_index, partials,
                               &status[0], &status[1]);
}

kernel void gpu_copy_bandwidth(const device uchar* source [[buffer(0)]],
                               device uchar* destination [[buffer(1)]],
                               device atomic_uint* status [[buffer(2)]],
                               constant KernelParams& params [[buffer(3)]],
                               uint global_id [[thread_position_in_grid]],
                               uint grid_size [[threads_per_grid]],
                               uint thread_index [[thread_index_in_threadgroup]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                                   [[simdgroup_index_in_threadgroup]],
                               uint simdgroup_count
                                   [[simdgroups_per_threadgroup]]) {
  threadgroup uint2 partials[32];
  const device uint4* source_vectors =
      reinterpret_cast<const device uint4*>(source);
  device uint4* destination_vectors =
      reinterpret_cast<device uint4*>(destination);
  uint2 local = uint2(0u);
  for (ulong vector_index = global_id; vector_index < params.vector_count;
       vector_index += grid_size) {
    const uint4 values = source_vectors[vector_index];
    destination_vectors[vector_index] = values;
    const uint word_index = uint(vector_index * 4u);
    accumulate_timed(local, values.x, word_index);
    accumulate_timed(local, values.y, word_index + 1u);
    accumulate_timed(local, values.z, word_index + 2u);
    accumulate_timed(local, values.w, word_index + 3u);
  }
  if (global_id == 0u && params.tail_bytes != 0u) {
    const ulong tail_offset = params.vector_count * 16u;
    const uint tail_word_count = (params.tail_bytes + 3u) / 4u;
    for (uint tail_word = 0u; tail_word < tail_word_count; ++tail_word) {
      const uint consumed = tail_word * 4u;
      const uint count = min(4u, params.tail_bytes - consumed);
      const uint value =
          load_tail_word(source, tail_offset + consumed, count);
      store_tail_word(destination, tail_offset + consumed, count, value);
      accumulate_timed(
          local, value, uint(params.vector_count * 4u + tail_word));
    }
  }
  finalize_timed_accumulator(local, global_id, params);
  commit_threadgroup_reduction(local, simd_lane, simdgroup_index,
                               simdgroup_count, thread_index, partials,
                               &status[0], &status[1]);
}

kernel void gpu_checksum(const device uchar* source [[buffer(0)]],
                         device atomic_uint* status [[buffer(1)]],
                         constant KernelParams& params [[buffer(2)]],
                         uint global_id [[thread_position_in_grid]],
                         uint grid_size [[threads_per_grid]],
                         uint thread_index [[thread_index_in_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simdgroup_index
                             [[simdgroup_index_in_threadgroup]],
                         uint simdgroup_count
                             [[simdgroups_per_threadgroup]]) {
  threadgroup uint2 partials[32];
  const device uint4* vectors =
      reinterpret_cast<const device uint4*>(source);
  uint2 local = uint2(0u);
  for (ulong vector_index = global_id; vector_index < params.vector_count;
       vector_index += grid_size) {
    const uint4 values = vectors[vector_index];
    const uint word_index = uint(vector_index * 4u);
    accumulate_final(local, values.x, word_index);
    accumulate_final(local, values.y, word_index + 1u);
    accumulate_final(local, values.z, word_index + 2u);
    accumulate_final(local, values.w, word_index + 3u);
  }
  if (global_id == 0u && params.tail_bytes != 0u) {
    const ulong tail_offset = params.vector_count * 16u;
    const uint tail_word_count = (params.tail_bytes + 3u) / 4u;
    for (uint tail_word = 0u; tail_word < tail_word_count; ++tail_word) {
      const uint consumed = tail_word * 4u;
      const uint count = min(4u, params.tail_bytes - consumed);
      accumulate_final(local,
                       load_tail_word(source, tail_offset + consumed, count),
                       uint(params.vector_count * 4u + tail_word));
    }
  }
  commit_threadgroup_reduction(local, simd_lane, simdgroup_index,
                               simdgroup_count, thread_index, partials,
                               &status[2], &status[3]);
}
)MSL";

}  // namespace GpuKernelContract

#endif  // GPU_KERNELS_SOURCE_H
