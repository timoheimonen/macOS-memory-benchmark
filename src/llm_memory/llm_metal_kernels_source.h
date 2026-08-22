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
 * @brief Canonical embedded MSL 2.3 source for Metal LLM workloads
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This private header owns the exact source bytes hashed into Metal execution
 * identity. It contains the shared resource foundation and the three public
 * scenario-specialized decode-contiguous workload entrypoints.
 */

#ifndef LLM_METAL_KERNELS_SOURCE_H
#define LLM_METAL_KERNELS_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace LlmMetalKernelContract {

inline constexpr char kRevision[] = "llm-metal-decode-contiguous-msl23-v1";
inline constexpr char kParameterAbiRevision[] = "llm-metal-foundation-parameters-v1";
inline constexpr char kResourceTableAbiRevision[] = "llm-metal-resource-table-v1";
inline constexpr char kDecodeParameterAbiRevision[] = "llm-metal-decode-contiguous-parameters-v1";
inline constexpr char kChecksumAlgorithmRevision[] = "llm-metal-dual-mod32-v1";

inline constexpr char kInitializeBytesEntrypoint[] = "llm_metal_initialize_bytes";
inline constexpr char kCopyBytesEntrypoint[] = "llm_metal_copy_bytes";
inline constexpr char kParameterLayoutProbeEntrypoint[] = "llm_metal_probe_parameter_layout";
inline constexpr char kDecodeParameterLayoutProbeEntrypoint[] = "llm_metal_probe_decode_parameter_layout";
inline constexpr char kValidateBytesEntrypoint[] = "llm_metal_validate_bytes";
inline constexpr char kValidateTableEntrypoint[] = "llm_metal_validate_table";
inline constexpr char kDecodeWeightsOnlyEntrypoint[] = "llm_metal_decode_contiguous_weights_only";
inline constexpr char kDecodeKvOnlyEntrypoint[] = "llm_metal_decode_contiguous_kv_only";
inline constexpr char kDecodeMixedEntrypoint[] = "llm_metal_decode_contiguous_mixed";
inline constexpr char kValidateDecodeAppendsEntrypoint[] = "llm_metal_validate_decode_contiguous_appends";

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
inline constexpr uint32_t kDecodeProbeParametersBufferIndex = 0U;
inline constexpr uint32_t kDecodeProbeOutputBufferIndex = 1U;
inline constexpr uint32_t kWorkloadResourcesBufferIndex = 0U;
inline constexpr uint32_t kWorkloadParametersBufferIndex = 1U;
inline constexpr uint32_t kWorkloadReductionThreadgroupIndex = 0U;
inline constexpr uint32_t kPostValidationResourcesBufferIndex = 0U;
inline constexpr uint32_t kPostValidationParametersBufferIndex = 1U;

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

inline constexpr size_t kDecodeParameterAbiSize = 120U;
inline constexpr size_t kDecodeParameterAbiAlignment = 8U;
inline constexpr uint32_t kDecodeParameterAbiVersion = 1U;
inline constexpr uint32_t kDecodeParameterFieldCount = 17U;
inline constexpr size_t kDecodeWeightBytesOffset = 0U;
inline constexpr size_t kDecodeKeyBytesOffset = 8U;
inline constexpr size_t kDecodeValueBytesOffset = 16U;
inline constexpr size_t kDecodeSegmentCapacityBytesOffset = 24U;
inline constexpr size_t kDecodeContextTokensOffset = 32U;
inline constexpr size_t kDecodeLayerCountOffset = 40U;
inline constexpr size_t kDecodeBatchSizeOffset = 48U;
inline constexpr size_t kDecodeRecordBytesOffset = 56U;
inline constexpr size_t kDecodeWorkUnitsOffset = 64U;
inline constexpr size_t kDecodeWeightSeedOffset = 72U;
inline constexpr size_t kDecodeKeySeedOffset = 80U;
inline constexpr size_t kDecodeValueSeedOffset = 88U;
inline constexpr size_t kDecodeScenarioSeedOffset = 96U;
inline constexpr size_t kDecodeWeightSegmentCountOffset = 104U;
inline constexpr size_t kDecodeKeySegmentCountOffset = 108U;
inline constexpr size_t kDecodeValueSegmentCountOffset = 112U;
inline constexpr size_t kDecodeReservedZeroOffset = 116U;

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
inline constexpr uint32_t kAppendValidationMismatchBit = 1U << 3U;

inline constexpr uint32_t kStatusFlagsIndex = 0U;
inline constexpr uint32_t kWeightChecksumAIndex = 1U;
inline constexpr uint32_t kWeightChecksumBIndex = 2U;
inline constexpr uint32_t kKeyChecksumAIndex = 3U;
inline constexpr uint32_t kKeyChecksumBIndex = 4U;
inline constexpr uint32_t kValueChecksumAIndex = 5U;
inline constexpr uint32_t kValueChecksumBIndex = 6U;
inline constexpr uint32_t kTimedStatusWordCount = 7U;
inline constexpr uint32_t kReductionLaneCount = 6U;

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

inline constexpr uint32_t kDecodeProbeAbiVersionIndex = 0U;
inline constexpr uint32_t kDecodeProbeStructSizeIndex = 1U;
inline constexpr uint32_t kDecodeProbeStructAlignmentIndex = 2U;
inline constexpr uint32_t kDecodeProbeFieldCountIndex = 3U;
inline constexpr uint32_t kDecodeProbeFirstFieldOffsetIndex = 4U;
inline constexpr uint32_t kDecodeProbeFirstFieldValueIndex = 21U;
inline constexpr uint32_t kDecodeProbeOutputWordCount = 38U;

// Foundation and workload pattern constants. The contiguous public profile
// uses 32-bit affine words so its independent oracle is O(work units * L * B)
// and never rereads a multi-GiB resource. Paged foundation data retains the
// earlier 64-bit physical-layout pattern until that profile is activated.
inline constexpr uint32_t kContiguousPatternWordMultiplier = UINT32_C(0x9E3779B9);
inline constexpr uint64_t kPagedPatternLayerMultiplier = UINT64_C(0xA24BAED4963EE407);
inline constexpr uint64_t kPagedPatternPhysicalMultiplier = UINT64_C(0x9FB21C651E98DF25);
inline constexpr uint64_t kPagedPatternWordMultiplier = UINT64_C(0xC13FA9A902A6328F);
inline constexpr uint32_t kAppendWorkUnitMultiplier = UINT32_C(0x85EBCA6B);
inline constexpr uint32_t kAppendLayerMultiplier = UINT32_C(0xC2B2AE35);
inline constexpr uint32_t kAppendBatchMultiplier = UINT32_C(0x27D4EB2F);
inline constexpr uint32_t kAppendWordMultiplier = UINT32_C(0x165667B1);
inline constexpr uint32_t kAppendKeyDomain = UINT32_C(0x4B455931);
inline constexpr uint32_t kAppendValueDomain = UINT32_C(0x56414C31);
inline constexpr uint32_t kChecksumValueMultiplier = UINT32_C(0x9E3779B1);
inline constexpr uint32_t kChecksumAddressMultiplier = UINT32_C(0x85EBCA77);
inline constexpr uint32_t kChecksumWorkUnitMultiplier = UINT32_C(0xC2B2AE3D);
inline constexpr uint32_t kChecksumLayerMultiplier = UINT32_C(0x27D4EB35);
inline constexpr uint32_t kChecksumBatchMultiplier = UINT32_C(0x165667C5);
inline constexpr uint32_t kChecksumValidMaskMultiplier = UINT32_C(0xD3A2646D);
inline constexpr uint32_t kChecksumMetalDecodeContiguousProfileDomain = UINT32_C(0x4D444331);
inline constexpr uint32_t kChecksumScenarioHighMultiplier = UINT32_C(0xA24BAED5);
inline constexpr uint32_t kChecksumWeightDomain = UINT32_C(0x57474854);
inline constexpr uint32_t kChecksumKeyDomain = UINT32_C(0x4B455943);
inline constexpr uint32_t kChecksumValueDomain = UINT32_C(0x56414C43);
inline constexpr uint32_t kChecksumWeightReadVisit = UINT32_C(0x57524541);
inline constexpr uint32_t kChecksumAppendVisit = UINT32_C(0x41505044);
inline constexpr uint32_t kChecksumKvReadVisit = UINT32_C(0x4B565244);

/**
 * Canonical MSL 2.3 bytes hashed for `msl_source_sha256`.
 *
 * Buffer bindings are intentionally explicit:
 *
 * - initialize: destination 0, parameters 1;
 * - copy: source 0, destination 1, parameters 2;
 * - byte validation: source 0, status 1, parameters 2;
 * - table validation: reference 0, candidate 1, status 2, parameters 3;
 * - layout probe: Tier-2 argument buffer 0, parameters 1, output 2;
 * - timed decode: Tier-2 argument buffer 0, parameters 1, reduction TG 0;
 * - append validation: Tier-2 argument buffer 0, parameters 1.
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
constant uint kAppendValidationMismatchBit = 1u << 3u;

constant uint kStatusFlagsIndex = 0u;
constant uint kWeightChecksumAIndex = 1u;
constant uint kWeightChecksumBIndex = 2u;
constant uint kKeyChecksumAIndex = 3u;
constant uint kKeyChecksumBIndex = 4u;
constant uint kValueChecksumAIndex = 5u;
constant uint kValueChecksumBIndex = 6u;

constant uint kContiguousPatternWordMultiplier = 0x9e3779b9u;
constant uint kAppendWorkUnitMultiplier = 0x85ebca6bu;
constant uint kAppendLayerMultiplier = 0xc2b2ae35u;
constant uint kAppendBatchMultiplier = 0x27d4eb2fu;
constant uint kAppendWordMultiplier = 0x165667b1u;
constant uint kAppendKeyDomain = 0x4b455931u;
constant uint kAppendValueDomain = 0x56414c31u;
constant uint kChecksumValueMultiplier = 0x9e3779b1u;
constant uint kChecksumAddressMultiplier = 0x85ebca77u;
constant uint kChecksumWorkUnitMultiplier = 0xc2b2ae3du;
constant uint kChecksumLayerMultiplier = 0x27d4eb35u;
constant uint kChecksumBatchMultiplier = 0x165667c5u;
constant uint kChecksumValidMaskMultiplier = 0xd3a2646du;
constant uint kChecksumMetalDecodeContiguousProfileDomain = 0x4d444331u;
constant uint kChecksumScenarioHighMultiplier = 0xa24baed5u;
constant uint kChecksumWeightDomain = 0x57474854u;
constant uint kChecksumKeyDomain = 0x4b455943u;
constant uint kChecksumValueDomain = 0x56414c43u;
constant uint kChecksumWeightReadVisit = 0x57524541u;
constant uint kChecksumAppendVisit = 0x41505044u;
constant uint kChecksumKvReadVisit = 0x4b565244u;

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

struct LlmMetalDecodeContiguousParams {
  ulong weight_bytes;                 // offset 0
  ulong k_bytes;                      // offset 8
  ulong v_bytes;                      // offset 16
  ulong segment_capacity_bytes;       // offset 24
  ulong context_tokens;               // offset 32
  ulong layer_count;                  // offset 40
  ulong batch_size;                   // offset 48
  ulong record_bytes;                 // offset 56
  ulong work_units;                   // offset 64
  ulong weight_seed;                  // offset 72
  ulong k_seed;                       // offset 80
  ulong v_seed;                       // offset 88
  ulong scenario_seed;                // offset 96
  uint weight_segment_count;          // offset 104
  uint k_segment_count;               // offset 108
  uint v_segment_count;               // offset 112
  uint reserved_zero;                 // offset 116
};

inline uchar contiguous_pattern_byte(ulong seed, ulong absolute_byte) {
  const ulong word_index = absolute_byte / 4ul;
  const uint byte_index = uint(absolute_byte % 4ul);
  const uint word = uint(seed) +
      kContiguousPatternWordMultiplier * uint(word_index + 1ul);
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

inline uint decode_append_word(ulong scenario_seed, ulong work_unit,
                               ulong layer, ulong batch, ulong word_index,
                               uint pool_domain) {
  return uint(scenario_seed) +
         kAppendWorkUnitMultiplier * uint(work_unit + 1ul) +
         kAppendLayerMultiplier * uint(layer + 1ul) +
         kAppendBatchMultiplier * uint(batch + 1ul) +
         kAppendWordMultiplier * uint(word_index + 1ul) + pool_domain;
}

inline uint checksum_domain(uint pool_domain, uint visit_domain,
                            ulong scenario_seed, ulong work_unit, ulong layer,
                            ulong batch, uint valid_mask) {
  return kChecksumMetalDecodeContiguousProfileDomain +
         uint(scenario_seed) +
         kChecksumScenarioHighMultiplier * uint(scenario_seed >> 32ul) +
         pool_domain + visit_domain +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier * uint(batch + 1ul) +
         kChecksumValidMaskMultiplier * valid_mask;
}

inline void mix_checksum_word(thread uint2& checksum, uint value,
                              ulong word_index, uint domain) {
  checksum.x += value + domain;
  checksum.y += value * kChecksumValueMultiplier +
                uint(word_index) * kChecksumAddressMultiplier +
                domain * 0x7feb352du;
}

inline ulong first_owned_vector(ulong first_vector, uint global_id,
                                uint grid_size) {
  const ulong remainder = first_vector % ulong(grid_size);
  const ulong delta = (ulong(global_id) + ulong(grid_size) - remainder) %
                      ulong(grid_size);
  return first_vector + delta;
}

inline uint range_word_mask(ulong range_start, ulong range_end,
                            ulong word_start) {
  uint mask = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    const ulong absolute_byte = word_start + ulong(byte_index);
    if (absolute_byte >= range_start && absolute_byte < range_end) {
      mask |= 1u << byte_index;
    }
  }
  return mask;
}

inline uchar load_weight_byte(constant LlmMetalResources& resources,
                              constant LlmMetalDecodeContiguousParams& params,
                              ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.weight_segments[segment][local_byte];
}

inline uchar load_key_byte(constant LlmMetalResources& resources,
                           constant LlmMetalDecodeContiguousParams& params,
                           ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.key_segments[segment][local_byte];
}

inline uchar load_value_byte(constant LlmMetalResources& resources,
                             constant LlmMetalDecodeContiguousParams& params,
                             ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.value_segments[segment][local_byte];
}

inline void store_key_byte(constant LlmMetalResources& resources,
                           constant LlmMetalDecodeContiguousParams& params,
                           ulong absolute_byte, uchar value) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  resources.key_segments[segment][local_byte] = value;
}

inline void store_value_byte(constant LlmMetalResources& resources,
                             constant LlmMetalDecodeContiguousParams& params,
                             ulong absolute_byte, uchar value) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  resources.value_segments[segment][local_byte] = value;
}

inline uint packed_weight_word(constant LlmMetalResources& resources,
                               constant LlmMetalDecodeContiguousParams& params,
                               ulong word_start, ulong range_start,
                               ulong range_end, uint mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((mask & (1u << byte_index)) != 0u) {
      value |= uint(load_weight_byte(resources, params,
                                     word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline uint packed_key_word(constant LlmMetalResources& resources,
                            constant LlmMetalDecodeContiguousParams& params,
                            ulong word_start, ulong range_start,
                            ulong range_end, uint mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((mask & (1u << byte_index)) != 0u) {
      value |= uint(load_key_byte(resources, params,
                                  word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline uint packed_value_word(constant LlmMetalResources& resources,
                              constant LlmMetalDecodeContiguousParams& params,
                              ulong word_start, ulong range_start,
                              ulong range_end, uint mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((mask & (1u << byte_index)) != 0u) {
      value |= uint(load_value_byte(resources, params,
                                    word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline void scan_weight_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, uint global_id,
    uint grid_size, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end % 16ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, global_id,
                                                grid_size);
       vector_index < end_vector; vector_index += ulong(grid_size)) {
    const ulong vector_start = vector_index * 16ul;
    const uint segment = uint(vector_start / params.segment_capacity_bytes);
    const ulong local_byte = vector_start % params.segment_capacity_bytes;
    if (vector_start >= range_start && vector_start + 16ul <= range_end) {
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.weight_segments[segment] + local_byte);
      const uint4 values = *source;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        const uint domain = checksum_domain(
            kChecksumWeightDomain, kChecksumWeightReadVisit,
            params.scenario_seed, work_unit, layer, 0ul, 0x0fu);
        mix_checksum_word(checksum, values[lane], word_index, domain);
      }
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask != 0u) {
        const uint value = packed_weight_word(resources, params, word_start,
                                              range_start, range_end, mask);
        const uint domain = checksum_domain(
            kChecksumWeightDomain, kChecksumWeightReadVisit,
            params.scenario_seed, work_unit, layer, 0ul, mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

inline void scan_key_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    uint global_id, uint grid_size, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end % 16ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, global_id,
                                                grid_size);
       vector_index < end_vector; vector_index += ulong(grid_size)) {
    const ulong vector_start = vector_index * 16ul;
    const uint segment = uint(vector_start / params.segment_capacity_bytes);
    const ulong local_byte = vector_start % params.segment_capacity_bytes;
    if (vector_start >= range_start && vector_start + 16ul <= range_end) {
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.key_segments[segment] + local_byte);
      const uint4 values = *source;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        const uint domain = checksum_domain(
            kChecksumKeyDomain, kChecksumKvReadVisit, params.scenario_seed,
            work_unit, layer, batch, 0x0fu);
        mix_checksum_word(checksum, values[lane], word_index, domain);
      }
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask != 0u) {
        const uint value = packed_key_word(resources, params, word_start,
                                           range_start, range_end, mask);
        const uint domain = checksum_domain(
            kChecksumKeyDomain, kChecksumKvReadVisit, params.scenario_seed,
            work_unit, layer, batch, mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

inline void scan_value_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    uint global_id, uint grid_size, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end % 16ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, global_id,
                                                grid_size);
       vector_index < end_vector; vector_index += ulong(grid_size)) {
    const ulong vector_start = vector_index * 16ul;
    const uint segment = uint(vector_start / params.segment_capacity_bytes);
    const ulong local_byte = vector_start % params.segment_capacity_bytes;
    if (vector_start >= range_start && vector_start + 16ul <= range_end) {
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.value_segments[segment] + local_byte);
      const uint4 values = *source;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        const uint domain = checksum_domain(
            kChecksumValueDomain, kChecksumKvReadVisit,
            params.scenario_seed, work_unit, layer, batch, 0x0fu);
        mix_checksum_word(checksum, values[lane], word_index, domain);
      }
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask != 0u) {
        const uint value = packed_value_word(resources, params, word_start,
                                             range_start, range_end, mask);
        const uint domain = checksum_domain(
            kChecksumValueDomain, kChecksumKvReadVisit,
            params.scenario_seed, work_unit, layer, batch, mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

inline void append_key_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    uint global_id, uint grid_size, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end % 16ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, global_id,
                                                grid_size);
       vector_index < end_vector; vector_index += ulong(grid_size)) {
    const ulong vector_start = vector_index * 16ul;
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask == 0u) {
        continue;
      }
      const uint value = decode_append_word(
          params.scenario_seed, work_unit, layer, batch, word_index,
          kAppendKeyDomain);
      uint packed = 0u;
      for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
        if ((mask & (1u << byte_index)) != 0u) {
          const uchar byte_value = uchar(value >> (byte_index * 8u));
          store_key_byte(resources, params, word_start + ulong(byte_index),
                         byte_value);
          packed |= uint(byte_value) << (byte_index * 8u);
        }
      }
      const uint domain = checksum_domain(
          kChecksumKeyDomain, kChecksumAppendVisit, params.scenario_seed,
          work_unit, layer, batch, mask);
      mix_checksum_word(checksum, packed, word_index, domain);
    }
  }
}

inline void append_value_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    uint global_id, uint grid_size, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end % 16ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, global_id,
                                                grid_size);
       vector_index < end_vector; vector_index += ulong(grid_size)) {
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask == 0u) {
        continue;
      }
      const uint value = decode_append_word(
          params.scenario_seed, work_unit, layer, batch, word_index,
          kAppendValueDomain);
      uint packed = 0u;
      for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
        if ((mask & (1u << byte_index)) != 0u) {
          const uchar byte_value = uchar(value >> (byte_index * 8u));
          store_value_byte(resources, params, word_start + ulong(byte_index),
                           byte_value);
          packed |= uint(byte_value) << (byte_index * 8u);
        }
      }
      const uint domain = checksum_domain(
          kChecksumValueDomain, kChecksumAppendVisit, params.scenario_seed,
          work_unit, layer, batch, mask);
      mix_checksum_word(checksum, packed, word_index, domain);
    }
  }
}

inline bool decode_parameters_valid(
    constant LlmMetalDecodeContiguousParams& params) {
  return params.weight_bytes != 0ul && params.k_bytes != 0ul &&
         params.v_bytes != 0ul && params.segment_capacity_bytes != 0ul &&
         params.context_tokens != 0ul && params.layer_count != 0ul &&
         params.batch_size != 0ul && params.record_bytes != 0ul &&
         params.work_units != 0ul && params.weight_segment_count != 0u &&
         params.k_segment_count != 0u && params.v_segment_count != 0u &&
         params.reserved_zero == 0u;
}

inline void publish_task_checksum(
    constant LlmMetalResources& resources, uint2 weight, uint2 key,
    uint2 value, threadgroup uint* reduction, uint thread_index,
    uint threads_per_threadgroup) {
  const uint base = thread_index * 6u;
  reduction[base + 0u] = weight.x;
  reduction[base + 1u] = weight.y;
  reduction[base + 2u] = key.x;
  reduction[base + 3u] = key.y;
  reduction[base + 4u] = value.x;
  reduction[base + 5u] = value.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index != 0u) {
    return;
  }
  uint totals[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  for (uint lane = 0u; lane < threads_per_threadgroup; ++lane) {
    const uint lane_base = lane * 6u;
    for (uint component = 0u; component < 6u; ++component) {
      totals[component] += reduction[lane_base + component];
    }
  }
  for (uint component = 0u; component < 6u; ++component) {
    atomic_fetch_add_explicit(
        &resources.status_checksum[kWeightChecksumAIndex + component],
        totals[component], memory_order_relaxed);
  }
}

inline void run_decode_kv(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodeContiguousParams& params, uint global_id,
    uint grid_size, thread uint2& key, thread uint2& value) {
  const ulong sequence_bytes = params.context_tokens * params.record_bytes;
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
      for (ulong batch = 0ul; batch < params.batch_size; ++batch) {
        const ulong sequence = layer * params.batch_size + batch;
        const ulong sequence_start = sequence * sequence_bytes;
        const ulong append_start = sequence_start +
            (params.context_tokens - 1ul) * params.record_bytes;
        append_key_range(resources, params, append_start, params.record_bytes,
                         work_unit, layer, batch, global_id, grid_size, key);
        append_value_range(resources, params, append_start,
                           params.record_bytes, work_unit, layer, batch,
                           global_id, grid_size, value);
        scan_key_range(resources, params, sequence_start, sequence_bytes,
                       work_unit, layer, batch, global_id, grid_size, key);
        scan_value_range(resources, params, sequence_start, sequence_bytes,
                         work_unit, layer, batch, global_id, grid_size, value);
      }
    }
  }
}

kernel void llm_metal_decode_contiguous_weights_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodeContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!decode_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 weight = uint2(0u);
  const ulong layer_base = params.weight_bytes / params.layer_count;
  const ulong layer_remainder = params.weight_bytes % params.layer_count;
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
      const ulong layer_start = layer * layer_base +
                                min(layer, layer_remainder);
      const ulong layer_bytes = layer_base +
                                (layer < layer_remainder ? 1ul : 0ul);
      scan_weight_range(resources, params, layer_start, layer_bytes,
                        work_unit, layer, global_id, grid_size, weight);
    }
  }
  publish_task_checksum(resources, weight, uint2(0u), uint2(0u), reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_decode_contiguous_kv_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodeContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!decode_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  run_decode_kv(resources, params, global_id, grid_size, key, value);
  publish_task_checksum(resources, uint2(0u), key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_decode_contiguous_mixed(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodeContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!decode_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 weight = uint2(0u);
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  const ulong layer_base = params.weight_bytes / params.layer_count;
  const ulong layer_remainder = params.weight_bytes % params.layer_count;
  const ulong sequence_bytes = params.context_tokens * params.record_bytes;
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
      const ulong layer_start = layer * layer_base +
                                min(layer, layer_remainder);
      const ulong layer_bytes = layer_base +
                                (layer < layer_remainder ? 1ul : 0ul);
      scan_weight_range(resources, params, layer_start, layer_bytes,
                        work_unit, layer, global_id, grid_size, weight);
      for (ulong batch = 0ul; batch < params.batch_size; ++batch) {
        const ulong sequence = layer * params.batch_size + batch;
        const ulong sequence_start = sequence * sequence_bytes;
        const ulong append_start = sequence_start +
            (params.context_tokens - 1ul) * params.record_bytes;
        append_key_range(resources, params, append_start, params.record_bytes,
                         work_unit, layer, batch, global_id, grid_size, key);
        append_value_range(resources, params, append_start,
                           params.record_bytes, work_unit, layer, batch,
                           global_id, grid_size, value);
        scan_key_range(resources, params, sequence_start, sequence_bytes,
                       work_unit, layer, batch, global_id, grid_size, key);
        scan_value_range(resources, params, sequence_start, sequence_bytes,
                         work_unit, layer, batch, global_id, grid_size, value);
      }
    }
  }
  publish_task_checksum(resources, weight, key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_validate_decode_contiguous_appends(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodeContiguousParams& params [[buffer(1)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if (!decode_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  const ulong work_unit = params.work_units - 1ul;
  const ulong sequence_bytes = params.context_tokens * params.record_bytes;
  for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
    for (ulong batch = 0ul; batch < params.batch_size; ++batch) {
      const ulong sequence = layer * params.batch_size + batch;
      const ulong append_start = sequence * sequence_bytes +
          (params.context_tokens - 1ul) * params.record_bytes;
      const ulong append_end = append_start + params.record_bytes;
      for (ulong absolute_byte = append_start + ulong(global_id);
           absolute_byte < append_end; absolute_byte += ulong(grid_size)) {
        const ulong word_index = absolute_byte / 4ul;
        const uint byte_index = uint(absolute_byte % 4ul);
        const uchar expected_key = uchar(decode_append_word(
            params.scenario_seed, work_unit, layer, batch, word_index,
            kAppendKeyDomain) >> (byte_index * 8u));
        const uchar expected_value = uchar(decode_append_word(
            params.scenario_seed, work_unit, layer, batch, word_index,
            kAppendValueDomain) >> (byte_index * 8u));
        if (load_key_byte(resources, params, absolute_byte) != expected_key ||
            load_value_byte(resources, params, absolute_byte) !=
                expected_value) {
          mark_validation_failure(resources.status_checksum,
                                  kAppendValidationMismatchBit);
        }
      }
    }
  }
}

kernel void llm_metal_probe_decode_parameter_layout(
    constant LlmMetalDecodeContiguousParams& params [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]) {
  if (global_id != 0u) {
    return;
  }

  output[0] = 1ul;
  output[1] = ulong(sizeof(LlmMetalDecodeContiguousParams));
  output[2] = ulong(alignof(LlmMetalDecodeContiguousParams));
  output[3] = 17ul;
  output[4] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, weight_bytes));
  output[5] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, k_bytes));
  output[6] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, v_bytes));
  output[7] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, segment_capacity_bytes));
  output[8] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, context_tokens));
  output[9] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, layer_count));
  output[10] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, batch_size));
  output[11] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, record_bytes));
  output[12] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, work_units));
  output[13] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, weight_seed));
  output[14] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, k_seed));
  output[15] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, v_seed));
  output[16] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, scenario_seed));
  output[17] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, weight_segment_count));
  output[18] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, k_segment_count));
  output[19] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, v_segment_count));
  output[20] = ulong(__builtin_offsetof(
      LlmMetalDecodeContiguousParams, reserved_zero));
  output[21] = params.weight_bytes;
  output[22] = params.k_bytes;
  output[23] = params.v_bytes;
  output[24] = params.segment_capacity_bytes;
  output[25] = params.context_tokens;
  output[26] = params.layer_count;
  output[27] = params.batch_size;
  output[28] = params.record_bytes;
  output[29] = params.work_units;
  output[30] = params.weight_seed;
  output[31] = params.k_seed;
  output[32] = params.v_seed;
  output[33] = params.scenario_seed;
  output[34] = ulong(params.weight_segment_count);
  output[35] = ulong(params.k_segment_count);
  output[36] = ulong(params.v_segment_count);
  output[37] = ulong(params.reserved_zero);
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
