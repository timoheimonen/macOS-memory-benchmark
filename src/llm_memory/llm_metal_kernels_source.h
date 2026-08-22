// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//
/**
 * @file llm_metal_kernels_source.h
 * @brief Canonical embedded MSL 2.3 source for the LLM Metal foundation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This private header owns the exact source bytes hashed into Metal execution
 * identity. Phase 8 intentionally exposes only excluded initialization,
 * staged copy, ABI probe, and validation kernels. Timed phase/layout/scenario
 * kernels are added only when their complete public profiles are activated.
 */

#ifndef LLM_METAL_KERNELS_SOURCE_H
#define LLM_METAL_KERNELS_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace LlmMetalKernelContract {

inline constexpr char kRevision[] = "llm-metal-foundation-msl23-v1";
inline constexpr char kParameterAbiRevision[] = "llm-metal-foundation-parameters-v1";
inline constexpr char kResourceTableAbiRevision[] = "llm-metal-resource-table-v1";

inline constexpr char kInitializeBytesEntrypoint[] = "llm_metal_initialize_bytes";
inline constexpr char kCopyBytesEntrypoint[] = "llm_metal_copy_bytes";
inline constexpr char kParameterLayoutProbeEntrypoint[] = "llm_metal_probe_parameter_layout";
inline constexpr char kValidateBytesEntrypoint[] = "llm_metal_validate_bytes";
inline constexpr char kValidateTableEntrypoint[] = "llm_metal_validate_table";

// Direct-buffer indices are part of the foundation pipeline ABI.
inline constexpr uint32_t kInitializeDestinationBufferIndex = 0U;
inline constexpr uint32_t kInitializeParametersBufferIndex = 1U;
inline constexpr uint32_t kCopySourceBufferIndex = 0U;
inline constexpr uint32_t kCopyDestinationBufferIndex = 1U;
inline constexpr uint32_t kCopyParametersBufferIndex = 2U;
inline constexpr uint32_t kByteValidationSourceBufferIndex = 0U;
inline constexpr uint32_t kByteValidationStatusBufferIndex = 1U;
inline constexpr uint32_t kByteValidationParametersBufferIndex = 2U;
inline constexpr uint32_t kTableValidationReferenceBufferIndex = 0U;
inline constexpr uint32_t kTableValidationCandidateBufferIndex = 1U;
inline constexpr uint32_t kTableValidationStatusBufferIndex = 2U;
inline constexpr uint32_t kTableValidationParametersBufferIndex = 3U;
inline constexpr uint32_t kProbeResourcesBufferIndex = 0U;
inline constexpr uint32_t kProbeParametersBufferIndex = 1U;
inline constexpr uint32_t kProbeOutputBufferIndex = 2U;

// The Tier-2 argument-buffer IDs are four non-overlapping 256-slot arrays
// followed by one shared status/checksum resource. Unused slots remain nil.
inline constexpr uint32_t kSegmentSlotCount = 256U;
inline constexpr uint32_t kWeightSegmentBaseId = 0U;
inline constexpr uint32_t kKeySegmentBaseId = 256U;
inline constexpr uint32_t kValueSegmentBaseId = 512U;
inline constexpr uint32_t kTableSegmentBaseId = 768U;
inline constexpr uint32_t kStatusChecksumResourceId = 1024U;
inline constexpr uint32_t kArgumentBufferResourceCount = 1025U;

/**
 * CPU mirror contract for `LlmMetalFoundationParams` in the MSL source.
 *
 * The Objective-C++ boundary defines its own `alignas(8)` aggregate and checks
 * every value below with `static_assert`. Keeping only scalar constants here
 * avoids making a Metal-only transport type part of the public C++ API.
 */
inline constexpr size_t kFoundationParameterAbiSize = 64U;
inline constexpr size_t kFoundationParameterAbiAlignment = 8U;
inline constexpr uint32_t kFoundationParameterAbiVersion = 1U;
inline constexpr uint32_t kFoundationParameterFieldCount = 10U;
inline constexpr size_t kByteCountOffset = 0U;
inline constexpr size_t kSourceOffsetBytesOffset = 8U;
inline constexpr size_t kDestinationOffsetBytesOffset = 16U;
inline constexpr size_t kLogicalBaseBytesOffset = 24U;
inline constexpr size_t kPatternSeedOffset = 32U;
inline constexpr size_t kBlockBytesOffset = 40U;
inline constexpr size_t kPhysicalBlocksPerLayerOffset = 48U;
inline constexpr size_t kPatternKindOffset = 52U;
inline constexpr size_t kProbeResourceKindOffset = 56U;
inline constexpr size_t kProbeResourceSlotOffset = 60U;

inline constexpr uint32_t kContiguousPatternKind = 0U;
inline constexpr uint32_t kPagedPatternKind = 1U;

inline constexpr uint32_t kProbeWeightResourceKind = 0U;
inline constexpr uint32_t kProbeKeyResourceKind = 1U;
inline constexpr uint32_t kProbeValueResourceKind = 2U;
inline constexpr uint32_t kProbeTableResourceKind = 3U;
inline constexpr uint32_t kProbeStatusChecksumResourceKind = 4U;

// `status[0]` is reset by an excluded command before either validation kernel.
// All updates are 32-bit relaxed atomic ORs; no 64-bit atomic support is used.
inline constexpr uint32_t kByteValidationMismatchBit = 1U << 0U;
inline constexpr uint32_t kTableValidationMismatchBit = 1U << 1U;
inline constexpr uint32_t kValidationInvalidParametersBit = 1U << 2U;

// Probe output is an array of 64-bit integer words. The first four words are
// an ABI header, the next ten are compiler-observed offsets, the following ten
// echo the decoded fields, and the final four prove the selected resource slot.
inline constexpr uint32_t kProbeAbiVersionIndex = 0U;
inline constexpr uint32_t kProbeStructSizeIndex = 1U;
inline constexpr uint32_t kProbeStructAlignmentIndex = 2U;
inline constexpr uint32_t kProbeFieldCountIndex = 3U;
inline constexpr uint32_t kProbeFirstFieldOffsetIndex = 4U;
inline constexpr uint32_t kProbeByteCountOffsetIndex = 4U;
inline constexpr uint32_t kProbeSourceOffsetBytesOffsetIndex = 5U;
inline constexpr uint32_t kProbeDestinationOffsetBytesOffsetIndex = 6U;
inline constexpr uint32_t kProbeLogicalBaseBytesOffsetIndex = 7U;
inline constexpr uint32_t kProbePatternSeedOffsetIndex = 8U;
inline constexpr uint32_t kProbeBlockBytesOffsetIndex = 9U;
inline constexpr uint32_t kProbePhysicalBlocksPerLayerOffsetIndex = 10U;
inline constexpr uint32_t kProbePatternKindOffsetIndex = 11U;
inline constexpr uint32_t kProbeResourceKindOffsetIndex = 12U;
inline constexpr uint32_t kProbeResourceSlotOffsetIndex = 13U;
inline constexpr uint32_t kProbeFirstFieldValueIndex = 14U;
inline constexpr uint32_t kProbeByteCountValueIndex = 14U;
inline constexpr uint32_t kProbeSourceOffsetBytesValueIndex = 15U;
inline constexpr uint32_t kProbeDestinationOffsetBytesValueIndex = 16U;
inline constexpr uint32_t kProbeLogicalBaseBytesValueIndex = 17U;
inline constexpr uint32_t kProbePatternSeedValueIndex = 18U;
inline constexpr uint32_t kProbeBlockBytesValueIndex = 19U;
inline constexpr uint32_t kProbePhysicalBlocksPerLayerValueIndex = 20U;
inline constexpr uint32_t kProbePatternKindValueIndex = 21U;
inline constexpr uint32_t kProbeResourceKindValueIndex = 22U;
inline constexpr uint32_t kProbeResourceSlotValueIndex = 23U;
inline constexpr uint32_t kProbeObservedResourceValueIndex = 24U;
inline constexpr uint32_t kProbeObservedResourceKindIndex = 25U;
inline constexpr uint32_t kProbeObservedResourceSlotIndex = 26U;
inline constexpr uint32_t kProbeArgumentBufferResourceCountIndex = 27U;
inline constexpr uint32_t kProbeOutputWordCount = 28U;

// Pattern constants match the current CPU initialization formulas. Unsigned
// 64-bit arithmetic wraps modulo 2^64 on both sides of the ABI.
inline constexpr uint64_t kBufferPatternMultiplier = UINT64_C(0x9E3779B97F4A7C15);
inline constexpr uint64_t kPagedPatternLayerMultiplier = UINT64_C(0xA24BAED4963EE407);
inline constexpr uint64_t kPagedPatternPhysicalMultiplier = UINT64_C(0x9FB21C651E98DF25);
inline constexpr uint64_t kPagedPatternWordMultiplier = UINT64_C(0xC13FA9A902A6328F);

/**
 * Canonical MSL 2.3 bytes hashed for `msl_source_sha256`.
 *
 * Buffer bindings are intentionally explicit:
 *
 * - initialize: destination 0, parameters 1;
 * - copy: source 0, destination 1, parameters 2;
 * - byte validation: source 0, status 1, parameters 2;
 * - table validation: reference 0, candidate 1, status 2, parameters 3;
 * - layout probe: Tier-2 argument buffer 0, parameters 1, output 2.
 *
 * `logical_base_bytes` is the pool-relative byte offset represented by local
 * byte zero. A contiguous pattern uses that absolute byte position. A paged
 * pattern derives layer, physical block, and block-local word from it, so
 * canonical segments can be initialized independently without host mirrors.
 * Copy source/destination ranges must not overlap. Table validation additionally
 * requires 4-byte aligned offsets and a byte count divisible by four. The host
 * validates all additions and ranges before dispatch; kernels use exact vector
 * prefixes with scalar tails and grid-stride iteration.
 *
 * The source is integer-only, uses only 32-bit atomics, and deliberately avoids
 * Metal 4, Apple9-family, GPU-address, and 64-bit-atomic features.
 */
inline constexpr std::string_view kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kSegmentSlotCount = 256u;
constant uint kFoundationParameterAbiVersion = 1u;
constant uint kFoundationParameterFieldCount = 10u;
constant uint kArgumentBufferResourceCount = 1025u;

constant uint kContiguousPatternKind = 0u;
constant uint kPagedPatternKind = 1u;

constant uint kProbeWeightResourceKind = 0u;
constant uint kProbeKeyResourceKind = 1u;
constant uint kProbeValueResourceKind = 2u;
constant uint kProbeTableResourceKind = 3u;
constant uint kProbeStatusChecksumResourceKind = 4u;

constant uint kByteValidationMismatchBit = 1u << 0u;
constant uint kTableValidationMismatchBit = 1u << 1u;
constant uint kValidationInvalidParametersBit = 1u << 2u;

struct LlmMetalFoundationParams {
  ulong byte_count;                    // offset 0
  ulong source_offset_bytes;           // offset 8
  ulong destination_offset_bytes;      // offset 16
  ulong logical_base_bytes;            // offset 24
  ulong pattern_seed;                  // offset 32
  ulong block_bytes;                   // offset 40
  uint physical_blocks_per_layer;      // offset 48
  uint pattern_kind;                   // offset 52
  uint probe_resource_kind;            // offset 56
  uint probe_resource_slot;            // offset 60
};

struct LlmMetalResources {
  array<device uchar*, 256> weight_segments [[id(0)]];
  array<device uchar*, 256> key_segments [[id(256)]];
  array<device uchar*, 256> value_segments [[id(512)]];
  array<device uchar*, 256> table_segments [[id(768)]];
  device atomic_uint* status_checksum [[id(1024)]];
};

inline uchar contiguous_pattern_byte(ulong seed, ulong absolute_byte) {
  const ulong word_index = absolute_byte / 8ul;
  const uint byte_index = uint(absolute_byte % 8ul);
  const ulong word =
      seed + 0x9e3779b97f4a7c15ul * (word_index + 1ul);
  return uchar(word >> (byte_index * 8u));
}

inline uchar paged_pattern_byte(constant LlmMetalFoundationParams& params,
                                ulong absolute_byte) {
  if (params.block_bytes == 0ul || params.physical_blocks_per_layer == 0u) {
    return uchar(0u);
  }
  const ulong global_block = absolute_byte / params.block_bytes;
  const ulong block_byte = absolute_byte % params.block_bytes;
  const ulong layer =
      global_block / ulong(params.physical_blocks_per_layer);
  const ulong physical_block =
      global_block % ulong(params.physical_blocks_per_layer);
  const ulong word_index = block_byte / 8ul;
  const uint byte_index = uint(block_byte % 8ul);
  const ulong word =
      params.pattern_seed +
      0xa24baed4963ee407ul * (layer + 1ul) +
      0x9fb21c651e98df25ul * (physical_block + 1ul) +
      0xc13fa9a902a6328ful * (word_index + 1ul);
  return uchar(word >> (byte_index * 8u));
}

inline uchar expected_pattern_byte(
    constant LlmMetalFoundationParams& params, ulong local_byte) {
  const ulong absolute_byte = params.logical_base_bytes + local_byte;
  return params.pattern_kind == kPagedPatternKind
             ? paged_pattern_byte(params, absolute_byte)
             : contiguous_pattern_byte(params.pattern_seed, absolute_byte);
}

inline uint packed_pattern_word(constant LlmMetalFoundationParams& params,
                                ulong local_byte) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    value |= uint(expected_pattern_byte(params, local_byte + byte_index))
             << (byte_index * 8u);
  }
  return value;
}

inline uint4 pattern_vector(constant LlmMetalFoundationParams& params,
                            ulong local_byte) {
  return uint4(packed_pattern_word(params, local_byte),
               packed_pattern_word(params, local_byte + 4ul),
               packed_pattern_word(params, local_byte + 8ul),
               packed_pattern_word(params, local_byte + 12ul));
}

inline void mark_validation_failure(device atomic_uint* status, uint bit) {
  atomic_fetch_or_explicit(&status[0], bit, memory_order_relaxed);
}

kernel void llm_metal_initialize_bytes(
    device uchar* destination [[buffer(0)]],
    constant LlmMetalFoundationParams& params [[buffer(1)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const bool vector_aligned =
      (params.destination_offset_bytes & 15ul) == 0ul;
  if (!vector_aligned) {
    for (ulong local_byte = ulong(global_id); local_byte < params.byte_count;
         local_byte += ulong(grid_size)) {
      destination[params.destination_offset_bytes + local_byte] =
          expected_pattern_byte(params, local_byte);
    }
    return;
  }

  const ulong vector_count = params.byte_count / 16ul;
  device uint4* vectors = reinterpret_cast<device uint4*>(
      destination + params.destination_offset_bytes);
  for (ulong vector_index = ulong(global_id); vector_index < vector_count;
       vector_index += ulong(grid_size)) {
    const ulong local_byte = vector_index * 16ul;
    vectors[vector_index] = pattern_vector(params, local_byte);
  }
  if (global_id == 0u) {
    for (ulong local_byte = vector_count * 16ul;
         local_byte < params.byte_count; ++local_byte) {
      destination[params.destination_offset_bytes + local_byte] =
          expected_pattern_byte(params, local_byte);
    }
  }
}

kernel void llm_metal_copy_bytes(
    const device uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant LlmMetalFoundationParams& params [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const bool vector_aligned =
      ((params.source_offset_bytes | params.destination_offset_bytes) & 15ul)
      == 0ul;
  if (!vector_aligned) {
    for (ulong local_byte = ulong(global_id); local_byte < params.byte_count;
         local_byte += ulong(grid_size)) {
      destination[params.destination_offset_bytes + local_byte] =
          source[params.source_offset_bytes + local_byte];
    }
    return;
  }

  const ulong vector_count = params.byte_count / 16ul;
  const device uint4* source_vectors =
      reinterpret_cast<const device uint4*>(
          source + params.source_offset_bytes);
  device uint4* destination_vectors = reinterpret_cast<device uint4*>(
      destination + params.destination_offset_bytes);
  for (ulong vector_index = ulong(global_id); vector_index < vector_count;
       vector_index += ulong(grid_size)) {
    destination_vectors[vector_index] = source_vectors[vector_index];
  }
  if (global_id == 0u) {
    for (ulong local_byte = vector_count * 16ul;
         local_byte < params.byte_count; ++local_byte) {
      destination[params.destination_offset_bytes + local_byte] =
          source[params.source_offset_bytes + local_byte];
    }
  }
}

kernel void llm_metal_validate_bytes(
    const device uchar* source [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant LlmMetalFoundationParams& params [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if (params.pattern_kind != kContiguousPatternKind &&
      params.pattern_kind != kPagedPatternKind) {
    if (global_id == 0u) {
      mark_validation_failure(status, kValidationInvalidParametersBit);
    }
    return;
  }
  if (params.pattern_kind == kPagedPatternKind &&
      (params.block_bytes == 0ul || params.physical_blocks_per_layer == 0u)) {
    if (global_id == 0u) {
      mark_validation_failure(status, kValidationInvalidParametersBit);
    }
    return;
  }

  const bool vector_aligned = (params.source_offset_bytes & 15ul) == 0ul;
  if (!vector_aligned) {
    for (ulong local_byte = ulong(global_id); local_byte < params.byte_count;
         local_byte += ulong(grid_size)) {
      if (source[params.source_offset_bytes + local_byte] !=
          expected_pattern_byte(params, local_byte)) {
        mark_validation_failure(status, kByteValidationMismatchBit);
      }
    }
    return;
  }

  const ulong vector_count = params.byte_count / 16ul;
  const device uint4* vectors = reinterpret_cast<const device uint4*>(
      source + params.source_offset_bytes);
  for (ulong vector_index = ulong(global_id); vector_index < vector_count;
       vector_index += ulong(grid_size)) {
    const ulong local_byte = vector_index * 16ul;
    if (any(vectors[vector_index] != pattern_vector(params, local_byte))) {
      mark_validation_failure(status, kByteValidationMismatchBit);
    }
  }
  if (global_id == 0u) {
    for (ulong local_byte = vector_count * 16ul;
         local_byte < params.byte_count; ++local_byte) {
      if (source[params.source_offset_bytes + local_byte] !=
          expected_pattern_byte(params, local_byte)) {
        mark_validation_failure(status, kByteValidationMismatchBit);
      }
    }
  }
}

kernel void llm_metal_validate_table(
    const device uchar* reference_bytes [[buffer(0)]],
    const device uchar* candidate_bytes [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant LlmMetalFoundationParams& params [[buffer(3)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if ((params.byte_count & 3ul) != 0ul ||
      ((params.source_offset_bytes | params.destination_offset_bytes) & 3ul)
          != 0ul) {
    if (global_id == 0u) {
      mark_validation_failure(status, kValidationInvalidParametersBit);
    }
    return;
  }

  const ulong entry_count = params.byte_count / 4ul;
  const device uint* reference = reinterpret_cast<const device uint*>(
      reference_bytes + params.source_offset_bytes);
  const device uint* candidate = reinterpret_cast<const device uint*>(
      candidate_bytes + params.destination_offset_bytes);
  const bool vector_aligned =
      ((params.source_offset_bytes | params.destination_offset_bytes) & 15ul)
      == 0ul;
  if (!vector_aligned) {
    for (ulong entry = ulong(global_id); entry < entry_count;
         entry += ulong(grid_size)) {
      if (reference[entry] != candidate[entry]) {
        mark_validation_failure(status, kTableValidationMismatchBit);
      }
    }
    return;
  }

  const ulong vector_count = entry_count / 4ul;
  const device uint4* reference_vectors =
      reinterpret_cast<const device uint4*>(reference);
  const device uint4* candidate_vectors =
      reinterpret_cast<const device uint4*>(candidate);
  for (ulong vector_index = ulong(global_id); vector_index < vector_count;
       vector_index += ulong(grid_size)) {
    if (any(reference_vectors[vector_index] !=
            candidate_vectors[vector_index])) {
      mark_validation_failure(status, kTableValidationMismatchBit);
    }
  }
  if (global_id == 0u) {
    for (ulong entry = vector_count * 4ul; entry < entry_count; ++entry) {
      if (reference[entry] != candidate[entry]) {
        mark_validation_failure(status, kTableValidationMismatchBit);
      }
    }
  }
}

kernel void llm_metal_probe_parameter_layout(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalFoundationParams& params [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]) {
  if (global_id != 0u) {
    return;
  }

  output[0] = ulong(kFoundationParameterAbiVersion);
  output[1] = ulong(sizeof(LlmMetalFoundationParams));
  output[2] = ulong(alignof(LlmMetalFoundationParams));
  output[3] = ulong(kFoundationParameterFieldCount);
  output[4] = ulong(__builtin_offsetof(LlmMetalFoundationParams, byte_count));
  output[5] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, source_offset_bytes));
  output[6] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, destination_offset_bytes));
  output[7] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, logical_base_bytes));
  output[8] = ulong(__builtin_offsetof(LlmMetalFoundationParams, pattern_seed));
  output[9] = ulong(__builtin_offsetof(LlmMetalFoundationParams, block_bytes));
  output[10] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, physical_blocks_per_layer));
  output[11] = ulong(__builtin_offsetof(LlmMetalFoundationParams, pattern_kind));
  output[12] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, probe_resource_kind));
  output[13] = ulong(__builtin_offsetof(
      LlmMetalFoundationParams, probe_resource_slot));

  output[14] = params.byte_count;
  output[15] = params.source_offset_bytes;
  output[16] = params.destination_offset_bytes;
  output[17] = params.logical_base_bytes;
  output[18] = params.pattern_seed;
  output[19] = params.block_bytes;
  output[20] = ulong(params.physical_blocks_per_layer);
  output[21] = ulong(params.pattern_kind);
  output[22] = ulong(params.probe_resource_kind);
  output[23] = ulong(params.probe_resource_slot);

  ulong observed = 0xfffffffffffffffful;
  if (params.probe_resource_slot < kSegmentSlotCount) {
    const uint slot = params.probe_resource_slot;
    switch (params.probe_resource_kind) {
      case kProbeWeightResourceKind:
        observed = ulong(resources.weight_segments[slot][0]);
        break;
      case kProbeKeyResourceKind:
        observed = ulong(resources.key_segments[slot][0]);
        break;
      case kProbeValueResourceKind:
        observed = ulong(resources.value_segments[slot][0]);
        break;
      case kProbeTableResourceKind:
        observed = ulong(resources.table_segments[slot][0]);
        break;
      default:
        break;
    }
  }
  if (params.probe_resource_kind == kProbeStatusChecksumResourceKind) {
    observed = ulong(atomic_load_explicit(&resources.status_checksum[0],
                                          memory_order_relaxed));
  }
  output[24] = observed;
  output[25] = ulong(params.probe_resource_kind);
  output[26] = ulong(params.probe_resource_slot);
  output[27] = ulong(kArgumentBufferResourceCount);
}
)MSL";

}  // namespace LlmMetalKernelContract

#endif  // LLM_METAL_KERNELS_SOURCE_H
