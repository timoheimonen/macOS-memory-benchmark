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
 * identity. It contains the shared resource foundation and four public
 * scenario-specialized decode/prefill and contiguous/paged profiles.
 */

#ifndef LLM_METAL_KERNELS_SOURCE_H
#define LLM_METAL_KERNELS_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace LlmMetalKernelContract {

inline constexpr char kRevision[] =
    "llm-metal-decode-prefill-contiguous-paged-msl23-v5";
inline constexpr char kParameterAbiRevision[] = "llm-metal-foundation-parameters-v1";
inline constexpr char kResourceTableAbiRevision[] = "llm-metal-resource-table-v1";
inline constexpr char kDecodeParameterAbiRevision[] = "llm-metal-decode-contiguous-parameters-v1";
inline constexpr char kDecodePagedParameterAbiRevision[] = "llm-metal-decode-paged-parameters-v1";
inline constexpr char kPrefillParameterAbiRevision[] =
    "llm-metal-prefill-contiguous-parameters-v1";
inline constexpr char kPrefillPagedParameterAbiRevision[] =
    "llm-metal-prefill-paged-parameters-v1";
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
inline constexpr char kDecodePagedParameterLayoutProbeEntrypoint[] =
    "llm_metal_probe_decode_paged_parameter_layout";
inline constexpr char kDecodePagedWeightsOnlyEntrypoint[] = "llm_metal_decode_paged_weights_only";
inline constexpr char kDecodePagedKvOnlyEntrypoint[] = "llm_metal_decode_paged_kv_only";
inline constexpr char kDecodePagedMixedEntrypoint[] = "llm_metal_decode_paged_mixed";
inline constexpr char kValidateDecodePagedEntrypoint[] =
    "llm_metal_validate_decode_paged_appends_padding";
inline constexpr char kPrefillParameterLayoutProbeEntrypoint[] =
    "llm_metal_probe_prefill_parameter_layout";
inline constexpr char kPrefillWeightsOnlyEntrypoint[] =
    "llm_metal_prefill_contiguous_weights_only";
inline constexpr char kPrefillKvOnlyEntrypoint[] =
    "llm_metal_prefill_contiguous_kv_only";
inline constexpr char kPrefillMixedEntrypoint[] =
    "llm_metal_prefill_contiguous_mixed";
inline constexpr char kValidatePrefillWritesEntrypoint[] =
    "llm_metal_validate_prefill_contiguous_writes";
inline constexpr char kPrefillPagedParameterLayoutProbeEntrypoint[] =
    "llm_metal_probe_prefill_paged_parameter_layout";
inline constexpr char kPrefillPagedWeightsOnlyEntrypoint[] =
    "llm_metal_prefill_paged_weights_only";
inline constexpr char kPrefillPagedKvOnlyEntrypoint[] =
    "llm_metal_prefill_paged_kv_only";
inline constexpr char kPrefillPagedMixedEntrypoint[] =
    "llm_metal_prefill_paged_mixed";
inline constexpr char kValidatePrefillPagedEntrypoint[] =
    "llm_metal_validate_prefill_paged_writes_padding";

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
inline constexpr uint32_t kWorkloadPhysicalIdThreadgroupIndex = 1U;
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

inline constexpr size_t kDecodePagedParameterAbiSize = 168U;
inline constexpr size_t kDecodePagedParameterAbiAlignment = 8U;
inline constexpr uint32_t kDecodePagedParameterAbiVersion = 1U;
inline constexpr uint32_t kDecodePagedParameterFieldCount = 24U;
inline constexpr size_t kDecodePagedWeightBytesOffset = 0U;
inline constexpr size_t kDecodePagedContextTokensOffset = 8U;
inline constexpr size_t kDecodePagedLayerCountOffset = 16U;
inline constexpr size_t kDecodePagedBatchSizeOffset = 24U;
inline constexpr size_t kDecodePagedRecordBytesOffset = 32U;
inline constexpr size_t kDecodePagedWorkUnitsOffset = 40U;
inline constexpr size_t kDecodePagedBlockBytesOffset = 48U;
inline constexpr size_t kDecodePagedLastBlockValidBytesOffset = 56U;
inline constexpr size_t kDecodePagedAppendOffsetOffset = 64U;
inline constexpr size_t kDecodePagedBlocksPerSequenceOffset = 72U;
inline constexpr size_t kDecodePagedPhysicalBlocksPerLayerOffset = 80U;
inline constexpr size_t kDecodePagedBlocksPerSegmentOffset = 88U;
inline constexpr size_t kDecodePagedTableEntriesPerSegmentOffset = 96U;
inline constexpr size_t kDecodePagedSegmentCapacityBytesOffset = 104U;
inline constexpr size_t kDecodePagedWeightSeedOffset = 112U;
inline constexpr size_t kDecodePagedKeySeedOffset = 120U;
inline constexpr size_t kDecodePagedValueSeedOffset = 128U;
inline constexpr size_t kDecodePagedScenarioSeedOffset = 136U;
inline constexpr size_t kDecodePagedWeightSegmentCountOffset = 144U;
inline constexpr size_t kDecodePagedKeySegmentCountOffset = 148U;
inline constexpr size_t kDecodePagedValueSegmentCountOffset = 152U;
inline constexpr size_t kDecodePagedTableSegmentCountOffset = 156U;
inline constexpr size_t kDecodePagedReservedZeroOffset = 160U;
inline constexpr size_t kDecodePagedPaddingZeroOffset = 164U;

inline constexpr size_t kPrefillParameterAbiSize = 136U;
inline constexpr size_t kPrefillParameterAbiAlignment = 8U;
inline constexpr uint32_t kPrefillParameterAbiVersion = 1U;
inline constexpr uint32_t kPrefillParameterFieldCount = 19U;
inline constexpr size_t kPrefillWeightBytesOffset = 0U;
inline constexpr size_t kPrefillKeyBytesOffset = 8U;
inline constexpr size_t kPrefillValueBytesOffset = 16U;
inline constexpr size_t kPrefillSegmentCapacityBytesOffset = 24U;
inline constexpr size_t kPrefillPromptTokensOffset = 32U;
inline constexpr size_t kPrefillQueryTileTokensOffset = 40U;
inline constexpr size_t kPrefillTileCountOffset = 48U;
inline constexpr size_t kPrefillLayerCountOffset = 56U;
inline constexpr size_t kPrefillBatchSizeOffset = 64U;
inline constexpr size_t kPrefillRecordBytesOffset = 72U;
inline constexpr size_t kPrefillWorkUnitsOffset = 80U;
inline constexpr size_t kPrefillWeightSeedOffset = 88U;
inline constexpr size_t kPrefillKeySeedOffset = 96U;
inline constexpr size_t kPrefillValueSeedOffset = 104U;
inline constexpr size_t kPrefillScenarioSeedOffset = 112U;
inline constexpr size_t kPrefillWeightSegmentCountOffset = 120U;
inline constexpr size_t kPrefillKeySegmentCountOffset = 124U;
inline constexpr size_t kPrefillValueSegmentCountOffset = 128U;
inline constexpr size_t kPrefillReservedZeroOffset = 132U;

inline constexpr size_t kPrefillPagedParameterAbiSize = 184U;
inline constexpr size_t kPrefillPagedParameterAbiAlignment = 8U;
inline constexpr uint32_t kPrefillPagedParameterAbiVersion = 1U;
inline constexpr uint32_t kPrefillPagedParameterFieldCount = 26U;
inline constexpr size_t kPrefillPagedWeightBytesOffset = 0U;
inline constexpr size_t kPrefillPagedPromptTokensOffset = 8U;
inline constexpr size_t kPrefillPagedQueryTileTokensOffset = 16U;
inline constexpr size_t kPrefillPagedTileCountOffset = 24U;
inline constexpr size_t kPrefillPagedLayerCountOffset = 32U;
inline constexpr size_t kPrefillPagedBatchSizeOffset = 40U;
inline constexpr size_t kPrefillPagedRecordBytesOffset = 48U;
inline constexpr size_t kPrefillPagedWorkUnitsOffset = 56U;
inline constexpr size_t kPrefillPagedBlockTokensOffset = 64U;
inline constexpr size_t kPrefillPagedBlockBytesOffset = 72U;
inline constexpr size_t kPrefillPagedLastBlockValidBytesOffset = 80U;
inline constexpr size_t kPrefillPagedBlocksPerSequenceOffset = 88U;
inline constexpr size_t kPrefillPagedPhysicalBlocksPerLayerOffset = 96U;
inline constexpr size_t kPrefillPagedBlocksPerSegmentOffset = 104U;
inline constexpr size_t kPrefillPagedTableEntriesPerSegmentOffset = 112U;
inline constexpr size_t kPrefillPagedSegmentCapacityBytesOffset = 120U;
inline constexpr size_t kPrefillPagedWeightSeedOffset = 128U;
inline constexpr size_t kPrefillPagedKeySeedOffset = 136U;
inline constexpr size_t kPrefillPagedValueSeedOffset = 144U;
inline constexpr size_t kPrefillPagedScenarioSeedOffset = 152U;
inline constexpr size_t kPrefillPagedWeightSegmentCountOffset = 160U;
inline constexpr size_t kPrefillPagedKeySegmentCountOffset = 164U;
inline constexpr size_t kPrefillPagedValueSegmentCountOffset = 168U;
inline constexpr size_t kPrefillPagedTableSegmentCountOffset = 172U;
inline constexpr size_t kPrefillPagedReservedZeroOffset = 176U;
inline constexpr size_t kPrefillPagedPaddingZeroOffset = 180U;

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
inline constexpr uint32_t kKvWriteValidationMismatchBit = 1U << 3U;
inline constexpr uint32_t kPaddingCanaryMismatchBit = 1U << 4U;

inline constexpr uint32_t kStatusFlagsIndex = 0U;
inline constexpr uint32_t kWeightChecksumAIndex = 1U;
inline constexpr uint32_t kWeightChecksumBIndex = 2U;
inline constexpr uint32_t kKeyChecksumAIndex = 3U;
inline constexpr uint32_t kKeyChecksumBIndex = 4U;
inline constexpr uint32_t kValueChecksumAIndex = 5U;
inline constexpr uint32_t kValueChecksumBIndex = 6U;
inline constexpr uint32_t kLayoutMetadataLookupCountIndex = 7U;
inline constexpr uint32_t kTimedStatusWordCount = 8U;
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
inline constexpr uint32_t kDecodePagedProbeAbiVersionIndex = 0U;
inline constexpr uint32_t kDecodePagedProbeStructSizeIndex = 1U;
inline constexpr uint32_t kDecodePagedProbeStructAlignmentIndex = 2U;
inline constexpr uint32_t kDecodePagedProbeFieldCountIndex = 3U;
inline constexpr uint32_t kDecodePagedProbeFirstFieldOffsetIndex = 4U;
inline constexpr uint32_t kDecodePagedProbeFirstFieldValueIndex = 28U;
inline constexpr uint32_t kDecodePagedProbeOutputWordCount = 52U;
inline constexpr uint32_t kPrefillProbeAbiVersionIndex = 0U;
inline constexpr uint32_t kPrefillProbeStructSizeIndex = 1U;
inline constexpr uint32_t kPrefillProbeStructAlignmentIndex = 2U;
inline constexpr uint32_t kPrefillProbeFieldCountIndex = 3U;
inline constexpr uint32_t kPrefillProbeFirstFieldOffsetIndex = 4U;
inline constexpr uint32_t kPrefillProbeFirstFieldValueIndex = 23U;
inline constexpr uint32_t kPrefillProbeOutputWordCount = 42U;
inline constexpr uint32_t kPrefillPagedProbeAbiVersionIndex = 0U;
inline constexpr uint32_t kPrefillPagedProbeStructSizeIndex = 1U;
inline constexpr uint32_t kPrefillPagedProbeStructAlignmentIndex = 2U;
inline constexpr uint32_t kPrefillPagedProbeFieldCountIndex = 3U;
inline constexpr uint32_t kPrefillPagedProbeFirstFieldOffsetIndex = 4U;
inline constexpr uint32_t kPrefillPagedProbeFirstFieldValueIndex = 30U;
inline constexpr uint32_t kPrefillPagedProbeOutputWordCount = 56U;

// Foundation and workload pattern constants. Both active layouts use 32-bit
// affine words. Paged words additionally bind layer, physical block, and
// block-local word so the fixed-size host oracle never rereads device data.
inline constexpr uint32_t kContiguousPatternWordMultiplier = UINT32_C(0x9E3779B9);
inline constexpr uint32_t kPagedPatternLayerMultiplier = UINT32_C(0xA24BAED5);
inline constexpr uint32_t kPagedPatternPhysicalMultiplier = UINT32_C(0x9FB21C65);
inline constexpr uint32_t kPagedPatternWordMultiplier = UINT32_C(0xC13FA9A9);
inline constexpr uint32_t kAppendWorkUnitMultiplier = UINT32_C(0x85EBCA6B);
inline constexpr uint32_t kAppendLayerMultiplier = UINT32_C(0xC2B2AE35);
inline constexpr uint32_t kAppendBatchMultiplier = UINT32_C(0x27D4EB2F);
inline constexpr uint32_t kAppendWordMultiplier = UINT32_C(0x165667B1);
inline constexpr uint32_t kAppendKeyDomain = UINT32_C(0x4B455931);
inline constexpr uint32_t kAppendValueDomain = UINT32_C(0x56414C31);
inline constexpr uint32_t kPrefillWriteKeyDomain = UINT32_C(0x504B5731);
inline constexpr uint32_t kPrefillWriteValueDomain = UINT32_C(0x50565731);
inline constexpr uint32_t kChecksumValueMultiplier = UINT32_C(0x9E3779B1);
inline constexpr uint32_t kChecksumAddressMultiplier = UINT32_C(0x85EBCA77);
inline constexpr uint32_t kChecksumWorkUnitMultiplier = UINT32_C(0xC2B2AE3D);
inline constexpr uint32_t kChecksumLayerMultiplier = UINT32_C(0x27D4EB35);
inline constexpr uint32_t kChecksumBatchMultiplier = UINT32_C(0x165667C5);
inline constexpr uint32_t kChecksumValidMaskMultiplier = UINT32_C(0xD3A2646D);
inline constexpr uint32_t kChecksumMetalDecodeContiguousProfileDomain = UINT32_C(0x4D444331);
inline constexpr uint32_t kChecksumMetalDecodePagedProfileDomain = UINT32_C(0x4D445031);
inline constexpr uint32_t kChecksumMetalPrefillContiguousProfileDomain =
    UINT32_C(0x4D504331);
inline constexpr uint32_t kChecksumMetalPrefillPagedProfileDomain =
    UINT32_C(0x4D505031);
inline constexpr uint32_t kChecksumScenarioHighMultiplier = UINT32_C(0xA24BAED5);
inline constexpr uint32_t kChecksumWeightDomain = UINT32_C(0x57474854);
inline constexpr uint32_t kChecksumKeyDomain = UINT32_C(0x4B455943);
inline constexpr uint32_t kChecksumValueDomain = UINT32_C(0x56414C43);
inline constexpr uint32_t kChecksumWeightReadVisit = UINT32_C(0x57524541);
inline constexpr uint32_t kChecksumAppendVisit = UINT32_C(0x41505044);
inline constexpr uint32_t kChecksumKvReadVisit = UINT32_C(0x4B565244);
inline constexpr uint32_t kChecksumPrefillWriteVisit = UINT32_C(0x50575254);
inline constexpr uint32_t kChecksumTileMultiplier = UINT32_C(0xD1B54A35);
inline constexpr uint32_t kChecksumPagedLogicalMultiplier = UINT32_C(0x7F4A7C15);
inline constexpr uint32_t kChecksumPagedPhysicalMultiplier = UINT32_C(0x94D049BB);
inline constexpr uint32_t kChecksumPagedPairMultiplier = UINT32_C(0x369DEA0F);
inline constexpr uint32_t kChecksumPagedGlobalBlockLowMultiplier =
    UINT32_C(0xDB4F0B91);
inline constexpr uint32_t kChecksumPagedGlobalBlockHighMultiplier =
    UINT32_C(0xBBE05633);
inline constexpr uint32_t kChecksumPagedSegmentMultiplier =
    UINT32_C(0xA0F2EC75);
inline constexpr uint32_t kChecksumPagedLocalBaseLowMultiplier =
    UINT32_C(0x89E18285);
inline constexpr uint32_t kChecksumPagedLocalBaseHighMultiplier =
    UINT32_C(0xC6D1D6C9);
inline constexpr uint32_t kChecksumPagedAbsoluteBaseLowMultiplier =
    UINT32_C(0xB492B66F);
inline constexpr uint32_t kChecksumPagedAbsoluteBaseHighMultiplier =
    UINT32_C(0x9AE16A3B);
inline constexpr uint32_t kChecksumPagedAddressBindingDomain =
    UINT32_C(0x41444452);
inline constexpr uint32_t kChecksumPagedAddressPairMultiplier =
    UINT32_C(0xD6E8FEB9);
inline constexpr uint32_t kChecksumPagedAppendLookupVisit = UINT32_C(0x50414C55);
inline constexpr uint32_t kChecksumPagedKeyLookupVisit = UINT32_C(0x504B4C55);
inline constexpr uint32_t kChecksumPagedValueLookupVisit = UINT32_C(0x50564C55);
inline constexpr uint32_t kChecksumPagedPrefillWriteLookupVisit =
    UINT32_C(0x5050574C);
inline constexpr uint32_t kChecksumPagedPrefillKeyLookupVisit =
    UINT32_C(0x50504B4C);
inline constexpr uint32_t kChecksumPagedPrefillValueLookupVisit =
    UINT32_C(0x5050564C);

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
 * - timed workload: Tier-2 argument buffer 0, parameters 1, reduction TG 0;
 * - KV-write validation: Tier-2 argument buffer 0, parameters 1.
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

#ifndef LLM_METAL_DECODE_CONTIGUOUS
#define LLM_METAL_DECODE_CONTIGUOUS 0
#endif
#ifndef LLM_METAL_DECODE_PAGED
#define LLM_METAL_DECODE_PAGED 0
#endif
#ifndef LLM_METAL_PREFILL_CONTIGUOUS
#define LLM_METAL_PREFILL_CONTIGUOUS 0
#endif
#ifndef LLM_METAL_PREFILL_PAGED
#define LLM_METAL_PREFILL_PAGED 0
#endif
#if (LLM_METAL_DECODE_CONTIGUOUS + LLM_METAL_DECODE_PAGED + \
     LLM_METAL_PREFILL_CONTIGUOUS + LLM_METAL_PREFILL_PAGED) != 1
#error "exactly one LLM Metal profile must be selected"
#endif

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
constant uint kKvWriteValidationMismatchBit = 1u << 3u;
constant uint kPaddingCanaryMismatchBit = 1u << 4u;

constant uint kStatusFlagsIndex = 0u;
constant uint kWeightChecksumAIndex = 1u;
constant uint kWeightChecksumBIndex = 2u;
constant uint kKeyChecksumAIndex = 3u;
constant uint kKeyChecksumBIndex = 4u;
constant uint kValueChecksumAIndex = 5u;
constant uint kValueChecksumBIndex = 6u;
constant uint kLayoutMetadataLookupCountIndex = 7u;

constant uint kContiguousPatternWordMultiplier = 0x9e3779b9u;
constant uint kAppendWorkUnitMultiplier = 0x85ebca6bu;
constant uint kAppendLayerMultiplier = 0xc2b2ae35u;
constant uint kAppendBatchMultiplier = 0x27d4eb2fu;
constant uint kAppendWordMultiplier = 0x165667b1u;
constant uint kAppendKeyDomain = 0x4b455931u;
constant uint kAppendValueDomain = 0x56414c31u;
constant uint kPrefillWriteKeyDomain = 0x504b5731u;
constant uint kPrefillWriteValueDomain = 0x50565731u;
constant uint kChecksumValueMultiplier = 0x9e3779b1u;
constant uint kChecksumAddressMultiplier = 0x85ebca77u;
constant uint kChecksumWorkUnitMultiplier = 0xc2b2ae3du;
constant uint kChecksumLayerMultiplier = 0x27d4eb35u;
constant uint kChecksumBatchMultiplier = 0x165667c5u;
constant uint kChecksumValidMaskMultiplier = 0xd3a2646du;
constant uint kChecksumMetalDecodeContiguousProfileDomain = 0x4d444331u;
constant uint kChecksumMetalDecodePagedProfileDomain = 0x4d445031u;
constant uint kChecksumMetalPrefillContiguousProfileDomain = 0x4d504331u;
constant uint kChecksumMetalPrefillPagedProfileDomain = 0x4d505031u;
constant uint kChecksumScenarioHighMultiplier = 0xa24baed5u;
constant uint kChecksumWeightDomain = 0x57474854u;
constant uint kChecksumKeyDomain = 0x4b455943u;
constant uint kChecksumValueDomain = 0x56414c43u;
constant uint kChecksumWeightReadVisit = 0x57524541u;
constant uint kChecksumAppendVisit = 0x41505044u;
constant uint kChecksumKvReadVisit = 0x4b565244u;
constant uint kChecksumPrefillWriteVisit = 0x50575254u;
constant uint kChecksumTileMultiplier = 0xd1b54a35u;
constant uint kChecksumPagedLogicalMultiplier = 0x7f4a7c15u;
constant uint kChecksumPagedPhysicalMultiplier = 0x94d049bbu;
constant uint kChecksumPagedPairMultiplier = 0x369dea0fu;
constant uint kChecksumPagedGlobalBlockLowMultiplier = 0xdb4f0b91u;
constant uint kChecksumPagedGlobalBlockHighMultiplier = 0xbbe05633u;
constant uint kChecksumPagedSegmentMultiplier = 0xa0f2ec75u;
constant uint kChecksumPagedLocalBaseLowMultiplier = 0x89e18285u;
constant uint kChecksumPagedLocalBaseHighMultiplier = 0xc6d1d6c9u;
constant uint kChecksumPagedAbsoluteBaseLowMultiplier = 0xb492b66fu;
constant uint kChecksumPagedAbsoluteBaseHighMultiplier = 0x9ae16a3bu;
constant uint kChecksumPagedAddressBindingDomain = 0x41444452u;
constant uint kChecksumPagedAddressPairMultiplier = 0xd6e8feb9u;
constant uint kChecksumPagedAppendLookupVisit = 0x50414c55u;
constant uint kChecksumPagedKeyLookupVisit = 0x504b4c55u;
constant uint kChecksumPagedValueLookupVisit = 0x50564c55u;
constant uint kChecksumPagedPrefillWriteLookupVisit = 0x5050574cu;
constant uint kChecksumPagedPrefillKeyLookupVisit = 0x50504b4cu;
constant uint kChecksumPagedPrefillValueLookupVisit = 0x5050564cu;

constant uint kPagedPatternLayerMultiplier = 0xa24baed5u;
constant uint kPagedPatternPhysicalMultiplier = 0x9fb21c65u;
constant uint kPagedPatternWordMultiplier = 0xc13fa9a9u;

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

#if LLM_METAL_DECODE_CONTIGUOUS
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
#endif

#if LLM_METAL_PREFILL_CONTIGUOUS
struct LlmMetalPrefillContiguousParams {
  ulong weight_bytes;                         // offset 0
  ulong k_bytes;                              // offset 8
  ulong v_bytes;                              // offset 16
  ulong segment_capacity_bytes;               // offset 24
  ulong prompt_tokens;                        // offset 32
  ulong attention_query_tile_tokens;          // offset 40
  ulong tile_count;                           // offset 48
  ulong layer_count;                          // offset 56
  ulong batch_size;                           // offset 64
  ulong record_bytes;                         // offset 72
  ulong work_units;                           // offset 80
  ulong weight_seed;                          // offset 88
  ulong k_seed;                               // offset 96
  ulong v_seed;                               // offset 104
  ulong scenario_seed;                        // offset 112
  uint weight_segment_count;                  // offset 120
  uint k_segment_count;                       // offset 124
  uint v_segment_count;                       // offset 128
  uint reserved_zero;                         // offset 132
};
#endif

#if LLM_METAL_DECODE_PAGED
struct LlmMetalDecodePagedParams {
  ulong weight_bytes;                         // offset 0
  ulong context_tokens;                       // offset 8
  ulong layer_count;                          // offset 16
  ulong batch_size;                           // offset 24
  ulong record_bytes;                         // offset 32
  ulong work_units;                           // offset 40
  ulong block_bytes;                          // offset 48
  ulong last_block_valid_bytes;               // offset 56
  ulong append_offset_in_last_block;          // offset 64
  ulong blocks_per_sequence;                  // offset 72
  ulong physical_blocks_per_layer;            // offset 80
  ulong blocks_per_segment;                   // offset 88
  ulong table_entries_per_segment;            // offset 96
  ulong segment_capacity_bytes;               // offset 104
  ulong weight_seed;                          // offset 112
  ulong k_seed;                               // offset 120
  ulong v_seed;                               // offset 128
  ulong scenario_seed;                        // offset 136
  uint weight_segment_count;                  // offset 144
  uint k_segment_count;                       // offset 148
  uint v_segment_count;                       // offset 152
  uint table_segment_count;                   // offset 156
  uint reserved_zero;                         // offset 160
  uint padding_zero;                          // offset 164
};
#endif

#if LLM_METAL_PREFILL_PAGED
struct LlmMetalPrefillPagedParams {
  ulong weight_bytes;                         // offset 0
  ulong prompt_tokens;                        // offset 8
  ulong attention_query_tile_tokens;          // offset 16
  ulong tile_count;                           // offset 24
  ulong layer_count;                          // offset 32
  ulong batch_size;                           // offset 40
  ulong record_bytes;                         // offset 48
  ulong work_units;                           // offset 56
  ulong block_tokens;                         // offset 64
  ulong block_bytes;                          // offset 72
  ulong last_block_valid_bytes;               // offset 80
  ulong blocks_per_sequence;                  // offset 88
  ulong physical_blocks_per_layer;            // offset 96
  ulong blocks_per_segment;                   // offset 104
  ulong table_entries_per_segment;            // offset 112
  ulong segment_capacity_bytes;               // offset 120
  ulong weight_seed;                          // offset 128
  ulong k_seed;                               // offset 136
  ulong v_seed;                               // offset 144
  ulong scenario_seed;                        // offset 152
  uint weight_segment_count;                  // offset 160
  uint k_segment_count;                       // offset 164
  uint v_segment_count;                       // offset 168
  uint table_segment_count;                   // offset 172
  uint reserved_zero;                         // offset 176
  uint padding_zero;                          // offset 180
};
#endif

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
  const ulong word_index = block_byte / 4ul;
  const uint byte_index = uint(block_byte % 4ul);
  const uint word = uint(params.pattern_seed) +
      kPagedPatternLayerMultiplier * uint(layer + 1ul) +
      kPagedPatternPhysicalMultiplier * uint(physical_block + 1ul) +
      kPagedPatternWordMultiplier * uint(word_index + 1ul);
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
                            ulong batch, ulong tile_ordinal,
                            uint valid_mask) {
#if LLM_METAL_PREFILL_CONTIGUOUS
  const uint profile_domain = kChecksumMetalPrefillContiguousProfileDomain;
#else
  const uint profile_domain = kChecksumMetalDecodeContiguousProfileDomain;
#endif
  return profile_domain +
         uint(scenario_seed) +
         kChecksumScenarioHighMultiplier * uint(scenario_seed >> 32ul) +
         pool_domain + visit_domain +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier * uint(batch + 1ul) +
         kChecksumTileMultiplier * uint(tile_ordinal) +
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

#if LLM_METAL_DECODE_CONTIGUOUS || LLM_METAL_PREFILL_CONTIGUOUS
#if LLM_METAL_DECODE_CONTIGUOUS
typedef LlmMetalDecodeContiguousParams LlmMetalContiguousParams;
#else
typedef LlmMetalPrefillContiguousParams LlmMetalContiguousParams;
#endif
inline uchar load_weight_byte(constant LlmMetalResources& resources,
                              constant LlmMetalContiguousParams& params,
                              ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.weight_segments[segment][local_byte];
}

inline uchar load_key_byte(constant LlmMetalResources& resources,
                           constant LlmMetalContiguousParams& params,
                           ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.key_segments[segment][local_byte];
}

inline uchar load_value_byte(constant LlmMetalResources& resources,
                             constant LlmMetalContiguousParams& params,
                             ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.value_segments[segment][local_byte];
}

inline void store_key_byte(constant LlmMetalResources& resources,
                           constant LlmMetalContiguousParams& params,
                           ulong absolute_byte, uchar value) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  resources.key_segments[segment][local_byte] = value;
}

inline void store_value_byte(constant LlmMetalResources& resources,
                             constant LlmMetalContiguousParams& params,
                             ulong absolute_byte, uchar value) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  resources.value_segments[segment][local_byte] = value;
}

inline uint packed_weight_word(constant LlmMetalResources& resources,
                               constant LlmMetalContiguousParams& params,
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
                            constant LlmMetalContiguousParams& params,
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
                              constant LlmMetalContiguousParams& params,
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
    constant LlmMetalContiguousParams& params, ulong range_start,
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
            params.scenario_seed, work_unit, layer, 0ul, 0ul, 0x0fu);
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
            params.scenario_seed, work_unit, layer, 0ul, 0ul, mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

inline void scan_key_range(
    constant LlmMetalResources& resources,
    constant LlmMetalContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    ulong tile_ordinal, uint global_id, uint grid_size,
    thread uint2& checksum) {
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
            work_unit, layer, batch, tile_ordinal, 0x0fu);
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
            work_unit, layer, batch, tile_ordinal, mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

inline void scan_value_range(
    constant LlmMetalResources& resources,
    constant LlmMetalContiguousParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, ulong batch,
    ulong tile_ordinal, uint global_id, uint grid_size,
    thread uint2& checksum) {
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
            params.scenario_seed, work_unit, layer, batch, tile_ordinal,
            0x0fu);
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
            params.scenario_seed, work_unit, layer, batch, tile_ordinal,
            mask);
        mix_checksum_word(checksum, value, word_index, domain);
      }
    }
  }
}

#if LLM_METAL_DECODE_CONTIGUOUS
inline void append_key_range(
    constant LlmMetalResources& resources,
    constant LlmMetalContiguousParams& params, ulong range_start,
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
          work_unit, layer, batch, 0ul, mask);
      mix_checksum_word(checksum, packed, word_index, domain);
    }
  }
}

inline void append_value_range(
    constant LlmMetalResources& resources,
    constant LlmMetalContiguousParams& params, ulong range_start,
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
          work_unit, layer, batch, 0ul, mask);
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
                       work_unit, layer, batch, 0ul, global_id, grid_size,
                       key);
        scan_value_range(resources, params, sequence_start, sequence_bytes,
                         work_unit, layer, batch, 0ul, global_id, grid_size,
                         value);
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
                       work_unit, layer, batch, 0ul, global_id, grid_size,
                       key);
        scan_value_range(resources, params, sequence_start, sequence_bytes,
                         work_unit, layer, batch, 0ul, global_id, grid_size,
                         value);
      }
    }
  }
  publish_task_checksum(resources, weight, key, value, reduction,
                        thread_index, threads_per_threadgroup);
}
#endif  // LLM_METAL_DECODE_CONTIGUOUS

#if LLM_METAL_PREFILL_CONTIGUOUS
inline uint prefill_write_word(ulong scenario_seed, ulong work_unit,
                               ulong layer, ulong batch, ulong word_index,
                               uint pool_domain) {
  return uint(scenario_seed) +
         kAppendWorkUnitMultiplier * uint(work_unit + 1ul) +
         kAppendLayerMultiplier * uint(layer + 1ul) +
         kAppendBatchMultiplier * uint(batch + 1ul) +
         kAppendWordMultiplier * uint(word_index + 1ul) + pool_domain;
}

inline void write_prefill_key_range(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillContiguousParams& params, ulong range_start,
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
      uint4 values;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        values[lane] = prefill_write_word(
            params.scenario_seed, work_unit, layer, batch, word_index,
            kPrefillWriteKeyDomain);
        const uint domain = checksum_domain(
            kChecksumKeyDomain, kChecksumPrefillWriteVisit,
            params.scenario_seed, work_unit, layer, batch, 0ul, 0x0fu);
        mix_checksum_word(checksum, values[lane], word_index, domain);
      }
      device uint4* destination = reinterpret_cast<device uint4*>(
          resources.key_segments[segment] + local_byte);
      *destination = values;
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask == 0u) {
        continue;
      }
      const uint value = prefill_write_word(
          params.scenario_seed, work_unit, layer, batch, word_index,
          kPrefillWriteKeyDomain);
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
          kChecksumKeyDomain, kChecksumPrefillWriteVisit,
          params.scenario_seed, work_unit, layer, batch, 0ul, mask);
      mix_checksum_word(checksum, packed, word_index, domain);
    }
  }
}

inline void write_prefill_value_range(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillContiguousParams& params, ulong range_start,
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
      uint4 values;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        values[lane] = prefill_write_word(
            params.scenario_seed, work_unit, layer, batch, word_index,
            kPrefillWriteValueDomain);
        const uint domain = checksum_domain(
            kChecksumValueDomain, kChecksumPrefillWriteVisit,
            params.scenario_seed, work_unit, layer, batch, 0ul, 0x0fu);
        mix_checksum_word(checksum, values[lane], word_index, domain);
      }
      device uint4* destination = reinterpret_cast<device uint4*>(
          resources.value_segments[segment] + local_byte);
      *destination = values;
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask == 0u) {
        continue;
      }
      const uint value = prefill_write_word(
          params.scenario_seed, work_unit, layer, batch, word_index,
          kPrefillWriteValueDomain);
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
          kChecksumValueDomain, kChecksumPrefillWriteVisit,
          params.scenario_seed, work_unit, layer, batch, 0ul, mask);
      mix_checksum_word(checksum, packed, word_index, domain);
    }
  }
}

inline bool prefill_parameters_valid(
    constant LlmMetalPrefillContiguousParams& params) {
  if (params.attention_query_tile_tokens == 0ul) {
    return false;
  }
  const ulong expected_tiles = params.prompt_tokens /
      params.attention_query_tile_tokens +
      ((params.prompt_tokens % params.attention_query_tile_tokens) != 0ul
           ? 1ul
           : 0ul);
  return params.weight_bytes != 0ul && params.k_bytes != 0ul &&
         params.v_bytes != 0ul && params.segment_capacity_bytes != 0ul &&
         params.prompt_tokens != 0ul &&
         params.attention_query_tile_tokens <= params.prompt_tokens &&
         params.tile_count == expected_tiles && params.layer_count != 0ul &&
         params.batch_size != 0ul && params.record_bytes != 0ul &&
         params.work_units != 0ul && params.weight_segment_count != 0u &&
         params.k_segment_count != 0u && params.v_segment_count != 0u &&
         params.reserved_zero == 0u;
}

inline void run_prefill_kv(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillContiguousParams& params, ulong work_unit,
    ulong layer, ulong batch, uint global_id, uint grid_size,
    thread uint2& key, thread uint2& value) {
  const ulong sequence_bytes = params.prompt_tokens * params.record_bytes;
  const ulong sequence = layer * params.batch_size + batch;
  const ulong sequence_start = sequence * sequence_bytes;
  for (ulong prompt_token = 0ul; prompt_token < params.prompt_tokens;
       ++prompt_token) {
    const ulong token_start =
        sequence_start + prompt_token * params.record_bytes;
    write_prefill_key_range(resources, params, token_start,
                            params.record_bytes, work_unit, layer, batch,
                            global_id, grid_size, key);
    write_prefill_value_range(resources, params, token_start,
                              params.record_bytes, work_unit, layer, batch,
                              global_id, grid_size, value);
  }
  ulong remaining_tokens = params.prompt_tokens;
  ulong prefix_tokens = 0ul;
  ulong tile_ordinal = 0ul;
  while (remaining_tokens != 0ul) {
    const ulong tile_tokens = min(params.attention_query_tile_tokens,
                                  remaining_tokens);
    prefix_tokens += tile_tokens;
    ++tile_ordinal;
    const ulong prefix_bytes = prefix_tokens * params.record_bytes;
    scan_key_range(resources, params, sequence_start, prefix_bytes, work_unit,
                   layer, batch, tile_ordinal, global_id, grid_size, key);
    scan_value_range(resources, params, sequence_start, prefix_bytes,
                     work_unit, layer, batch, tile_ordinal, global_id,
                     grid_size, value);
    remaining_tokens -= tile_tokens;
  }
}

kernel void llm_metal_prefill_contiguous_weights_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!prefill_parameters_valid(params)) {
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

kernel void llm_metal_prefill_contiguous_kv_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!prefill_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
      for (ulong batch = 0ul; batch < params.batch_size; ++batch) {
        run_prefill_kv(resources, params, work_unit, layer, batch, global_id,
                       grid_size, key, value);
      }
    }
  }
  publish_task_checksum(resources, uint2(0u), key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_prefill_contiguous_mixed(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillContiguousParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!prefill_parameters_valid(params)) {
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
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    for (ulong layer = 0ul; layer < params.layer_count; ++layer) {
      const ulong layer_start = layer * layer_base +
                                min(layer, layer_remainder);
      const ulong layer_bytes = layer_base +
                                (layer < layer_remainder ? 1ul : 0ul);
      scan_weight_range(resources, params, layer_start, layer_bytes,
                        work_unit, layer, global_id, grid_size, weight);
      for (ulong batch = 0ul; batch < params.batch_size; ++batch) {
        run_prefill_kv(resources, params, work_unit, layer, batch, global_id,
                       grid_size, key, value);
      }
    }
  }
  publish_task_checksum(resources, weight, key, value, reduction,
                        thread_index, threads_per_threadgroup);
}
#endif  // LLM_METAL_PREFILL_CONTIGUOUS
#endif

#if LLM_METAL_DECODE_PAGED
inline bool decode_paged_parameters_valid(
    constant LlmMetalDecodePagedParams& params) {
  return params.weight_bytes != 0ul && params.context_tokens != 0ul &&
         params.layer_count != 0ul && params.batch_size != 0ul &&
         params.record_bytes != 0ul && params.work_units != 0ul &&
         params.block_bytes != 0ul && params.last_block_valid_bytes != 0ul &&
         params.last_block_valid_bytes <= params.block_bytes &&
         params.append_offset_in_last_block + params.record_bytes ==
             params.last_block_valid_bytes &&
         params.blocks_per_sequence != 0ul &&
         params.physical_blocks_per_layer ==
             params.batch_size * params.blocks_per_sequence &&
         params.blocks_per_segment != 0ul &&
         params.table_entries_per_segment != 0ul &&
         params.segment_capacity_bytes != 0ul &&
         params.weight_segment_count != 0u && params.k_segment_count != 0u &&
         params.v_segment_count != 0u && params.table_segment_count != 0u &&
         params.reserved_zero == 0u && params.padding_zero == 0u;
}

inline uint paged_checksum_domain(
    uint pool_domain, uint visit_domain,
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    ulong batch, ulong logical_table_index, uint physical_id,
    uint valid_mask) {
  const uint logical = uint(logical_table_index + 1ul);
  const uint physical = physical_id + 1u;
  return kChecksumMetalDecodePagedProfileDomain + uint(params.scenario_seed) +
         kChecksumScenarioHighMultiplier * uint(params.scenario_seed >> 32ul) +
         pool_domain + visit_domain +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier * uint(batch + 1ul) +
         kChecksumPagedLogicalMultiplier * logical +
         kChecksumPagedPhysicalMultiplier * physical +
         kChecksumPagedPairMultiplier * logical * physical +
         kChecksumValidMaskMultiplier * valid_mask;
}

inline uint paged_weight_checksum_domain(
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    uint valid_mask) {
  return kChecksumMetalDecodePagedProfileDomain + uint(params.scenario_seed) +
         kChecksumScenarioHighMultiplier * uint(params.scenario_seed >> 32ul) +
         kChecksumWeightDomain + kChecksumWeightReadVisit +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier +
         kChecksumValidMaskMultiplier * valid_mask;
}

inline uint paged_timed_table_lookup(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong logical_table_index,
    threadgroup uint* published_physical_id, uint thread_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0u) {
    const uint table_segment =
        uint(logical_table_index / params.table_entries_per_segment);
    const ulong table_entry =
        logical_table_index % params.table_entries_per_segment;
    device const volatile uint* named_lane_table =
        reinterpret_cast<device const volatile uint*>(
            resources.table_segments[table_segment]);
    published_physical_id[0] = named_lane_table[table_entry];
    atomic_fetch_add_explicit(
        &resources.status_checksum[kLayoutMetadataLookupCountIndex], 1u,
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint physical_id = published_physical_id[0];
  if (ulong(physical_id) >= params.physical_blocks_per_layer) {
    if (thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return 0u;
  }
  return physical_id;
}

inline void mix_paged_lookup(
    thread uint2& checksum, uint physical_id, ulong logical_table_index,
    uint pool_domain, uint visit_domain,
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    ulong batch) {
  const uint domain = paged_checksum_domain(
      pool_domain, visit_domain, params, work_unit, layer, batch,
      logical_table_index, physical_id, 0u);
  mix_checksum_word(checksum, physical_id, logical_table_index, domain);
}

inline ulong paged_physical_block_base(
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, thread uint& segment) {
  const ulong global_block =
      layer * params.physical_blocks_per_layer + ulong(physical_id);
  segment = uint(global_block / params.blocks_per_segment);
  return (global_block % params.blocks_per_segment) * params.block_bytes;
}

inline uchar load_paged_key_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong block_byte) {
  uint segment = 0u;
  const ulong block_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  return resources.key_segments[segment][block_base + block_byte];
}

inline uchar load_paged_value_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong block_byte) {
  uint segment = 0u;
  const ulong block_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  return resources.value_segments[segment][block_base + block_byte];
}

inline void store_paged_key_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong block_byte, uchar value) {
  uint segment = 0u;
  const ulong block_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  resources.key_segments[segment][block_base + block_byte] = value;
}

inline void store_paged_value_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong block_byte, uchar value) {
  uint segment = 0u;
  const ulong block_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  resources.value_segments[segment][block_base + block_byte] = value;
}

inline uint packed_paged_key_word(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong local_word, uint valid_mask) {
  uint value = 0u;
  const ulong word_start = local_word * 4ul;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((valid_mask & (1u << byte_index)) != 0u) {
      value |= uint(load_paged_key_byte(
                   resources, params, layer, physical_id,
                   word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline uint packed_paged_value_word(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong layer,
    uint physical_id, ulong local_word, uint valid_mask) {
  uint value = 0u;
  const ulong word_start = local_word * 4ul;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((valid_mask & (1u << byte_index)) != 0u) {
      value |= uint(load_paged_value_byte(
                   resources, params, layer, physical_id,
                   word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline void scan_paged_key_block(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    ulong batch, ulong logical_table_index, uint physical_id,
    ulong valid_bytes, uint thread_index, uint threads_per_threadgroup,
    thread uint2& checksum) {
  uint segment = 0u;
  const ulong physical_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  const ulong vector_count = valid_bytes / 16ul;
  const ulong words_per_block =
      params.block_bytes / 4ul + ((params.block_bytes & 3ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = ulong(thread_index); vector_index < vector_count;
       vector_index += ulong(threads_per_threadgroup)) {
    const ulong local_byte = vector_index * 16ul;
    uint4 values;
    if (((physical_base + local_byte) & 15ul) == 0ul) {
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.key_segments[segment] + physical_base + local_byte);
      values = *source;
    } else {
      values = uint4(
          packed_paged_key_word(resources, params, layer, physical_id,
                                vector_index * 4ul + 0ul, 0x0fu),
          packed_paged_key_word(resources, params, layer, physical_id,
                                vector_index * 4ul + 1ul, 0x0fu),
          packed_paged_key_word(resources, params, layer, physical_id,
                                vector_index * 4ul + 2ul, 0x0fu),
          packed_paged_key_word(resources, params, layer, physical_id,
                                vector_index * 4ul + 3ul, 0x0fu));
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong local_word = vector_index * 4ul + ulong(lane);
      const ulong word_ordinal = logical_table_index * words_per_block + local_word;
      const uint domain = paged_checksum_domain(
          kChecksumKeyDomain, kChecksumKvReadVisit, params, work_unit, layer,
          batch, logical_table_index, physical_id, 0x0fu);
      mix_checksum_word(checksum, values[lane], word_ordinal, domain);
    }
  }
  const ulong first_tail_word = vector_count * 4ul;
  const ulong word_count = valid_bytes / 4ul +
                           ((valid_bytes & 3ul) != 0ul ? 1ul : 0ul);
  for (ulong local_word = first_tail_word + ulong(thread_index);
       local_word < word_count;
       local_word += ulong(threads_per_threadgroup)) {
    const ulong word_start = local_word * 4ul;
    const uint mask = range_word_mask(0ul, valid_bytes, word_start);
    const uint value = packed_paged_key_word(resources, params, layer,
                                             physical_id, local_word, mask);
    const uint domain = paged_checksum_domain(
        kChecksumKeyDomain, kChecksumKvReadVisit, params, work_unit, layer,
        batch, logical_table_index, physical_id, mask);
    mix_checksum_word(checksum, value,
                      logical_table_index * words_per_block + local_word,
                      domain);
  }
}

inline void scan_paged_value_block(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    ulong batch, ulong logical_table_index, uint physical_id,
    ulong valid_bytes, uint thread_index, uint threads_per_threadgroup,
    thread uint2& checksum) {
  uint segment = 0u;
  const ulong physical_base =
      paged_physical_block_base(params, layer, physical_id, segment);
  const ulong vector_count = valid_bytes / 16ul;
  const ulong words_per_block =
      params.block_bytes / 4ul + ((params.block_bytes & 3ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = ulong(thread_index); vector_index < vector_count;
       vector_index += ulong(threads_per_threadgroup)) {
    const ulong local_byte = vector_index * 16ul;
    uint4 values;
    if (((physical_base + local_byte) & 15ul) == 0ul) {
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.value_segments[segment] + physical_base + local_byte);
      values = *source;
    } else {
      values = uint4(
          packed_paged_value_word(resources, params, layer, physical_id,
                                  vector_index * 4ul + 0ul, 0x0fu),
          packed_paged_value_word(resources, params, layer, physical_id,
                                  vector_index * 4ul + 1ul, 0x0fu),
          packed_paged_value_word(resources, params, layer, physical_id,
                                  vector_index * 4ul + 2ul, 0x0fu),
          packed_paged_value_word(resources, params, layer, physical_id,
                                  vector_index * 4ul + 3ul, 0x0fu));
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong local_word = vector_index * 4ul + ulong(lane);
      const ulong word_ordinal = logical_table_index * words_per_block + local_word;
      const uint domain = paged_checksum_domain(
          kChecksumValueDomain, kChecksumKvReadVisit, params, work_unit,
          layer, batch, logical_table_index, physical_id, 0x0fu);
      mix_checksum_word(checksum, values[lane], word_ordinal, domain);
    }
  }
  const ulong first_tail_word = vector_count * 4ul;
  const ulong word_count = valid_bytes / 4ul +
                           ((valid_bytes & 3ul) != 0ul ? 1ul : 0ul);
  for (ulong local_word = first_tail_word + ulong(thread_index);
       local_word < word_count;
       local_word += ulong(threads_per_threadgroup)) {
    const ulong word_start = local_word * 4ul;
    const uint mask = range_word_mask(0ul, valid_bytes, word_start);
    const uint value = packed_paged_value_word(resources, params, layer,
                                               physical_id, local_word, mask);
    const uint domain = paged_checksum_domain(
        kChecksumValueDomain, kChecksumKvReadVisit, params, work_unit, layer,
        batch, logical_table_index, physical_id, mask);
    mix_checksum_word(checksum, value,
                      logical_table_index * words_per_block + local_word,
                      domain);
  }
}

inline void append_paged_pair(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong work_unit, ulong layer,
    ulong batch, ulong logical_table_index, uint physical_id,
    uint thread_index, uint threads_per_threadgroup, thread uint2& key,
    thread uint2& value) {
  const ulong range_start = params.append_offset_in_last_block;
  const ulong range_end = params.last_block_valid_bytes;
  const ulong first_word = range_start / 4ul;
  const ulong end_word = range_end / 4ul +
                         ((range_end & 3ul) != 0ul ? 1ul : 0ul);
  const ulong words_per_block =
      params.block_bytes / 4ul + ((params.block_bytes & 3ul) != 0ul ? 1ul : 0ul);
  for (ulong local_word = first_word + ulong(thread_index);
       local_word < end_word;
       local_word += ulong(threads_per_threadgroup)) {
    const ulong word_start = local_word * 4ul;
    const uint mask = range_word_mask(range_start, range_end, word_start);
    const uint key_word = decode_append_word(
        params.scenario_seed, work_unit, layer, batch, local_word,
        kAppendKeyDomain);
    const uint value_word = decode_append_word(
        params.scenario_seed, work_unit, layer, batch, local_word,
        kAppendValueDomain);
    uint packed_key = 0u;
    uint packed_value = 0u;
    for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
      if ((mask & (1u << byte_index)) == 0u) {
        continue;
      }
      const ulong block_byte = word_start + ulong(byte_index);
      const uchar key_byte = uchar(key_word >> (byte_index * 8u));
      const uchar value_byte = uchar(value_word >> (byte_index * 8u));
      store_paged_key_byte(resources, params, layer, physical_id, block_byte,
                           key_byte);
      store_paged_value_byte(resources, params, layer, physical_id,
                             block_byte, value_byte);
      packed_key |= uint(key_byte) << (byte_index * 8u);
      packed_value |= uint(value_byte) << (byte_index * 8u);
    }
    const ulong word_ordinal = logical_table_index * words_per_block + local_word;
    mix_checksum_word(
        key, packed_key, word_ordinal,
        paged_checksum_domain(kChecksumKeyDomain, kChecksumAppendVisit,
                              params, work_unit, layer, batch,
                              logical_table_index, physical_id, mask));
    mix_checksum_word(
        value, packed_value, word_ordinal,
        paged_checksum_domain(kChecksumValueDomain, kChecksumAppendVisit,
                              params, work_unit, layer, batch,
                              logical_table_index, physical_id, mask));
  }
  threadgroup_barrier(mem_flags::mem_device);
}

inline uchar load_paged_weight_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.weight_segments[segment][local_byte];
}

inline uint packed_paged_weight_word(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong word_start,
    ulong range_start, ulong range_end, uint mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((mask & (1u << byte_index)) != 0u) {
      value |= uint(load_paged_weight_byte(
                   resources, params, word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline void scan_paged_weight_range(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, uint thread_index,
    uint thread_count, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end & 15ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, thread_index,
                                                thread_count);
       vector_index < end_vector; vector_index += ulong(thread_count)) {
    const ulong vector_start = vector_index * 16ul;
    if (vector_start >= range_start && vector_start + 16ul <= range_end) {
      const uint segment =
          uint(vector_start / params.segment_capacity_bytes);
      const ulong local_byte =
          vector_start % params.segment_capacity_bytes;
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.weight_segments[segment] + local_byte);
      const uint4 values = *source;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        mix_checksum_word(
            checksum, values[lane], word_index,
            paged_weight_checksum_domain(params, work_unit, layer, 0x0fu));
      }
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask != 0u) {
        const uint value = packed_paged_weight_word(
            resources, params, word_start, range_start, range_end, mask);
        mix_checksum_word(
            checksum, value, word_index,
            paged_weight_checksum_domain(params, work_unit, layer, mask));
      }
    }
  }
}

inline void run_decode_paged_kv_owners(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params,
    threadgroup uint* published_physical_id, uint thread_index,
    uint threads_per_threadgroup, uint threadgroup_index,
    uint threadgroup_count, bool include_weight, thread uint2& weight,
    thread uint2& key, thread uint2& value) {
  const ulong owner_count = params.layer_count * params.batch_size *
                            params.blocks_per_sequence;
  const ulong weight_layer_base = params.weight_bytes / params.layer_count;
  const ulong weight_layer_remainder = params.weight_bytes % params.layer_count;
  for (ulong owner = ulong(threadgroup_index); owner < owner_count;
       owner += ulong(threadgroup_count)) {
    const ulong logical_block = owner % params.blocks_per_sequence;
    const ulong sequence_ordinal = owner / params.blocks_per_sequence;
    const ulong batch = sequence_ordinal % params.batch_size;
    const ulong layer = sequence_ordinal / params.batch_size;
    const ulong logical_table_index =
        batch * params.blocks_per_sequence + logical_block;
    const bool terminal = logical_block + 1ul == params.blocks_per_sequence;
    const ulong valid_bytes = terminal ? params.last_block_valid_bytes
                                       : params.block_bytes;
    for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
      if (include_weight && batch == 0ul && logical_block == 0ul) {
        const ulong layer_start = layer * weight_layer_base +
                                  min(layer, weight_layer_remainder);
        const ulong layer_bytes = weight_layer_base +
                                  (layer < weight_layer_remainder ? 1ul : 0ul);
        scan_paged_weight_range(resources, params, layer_start, layer_bytes,
                                work_unit, layer, thread_index,
                                threads_per_threadgroup, weight);
      }
      if (terminal) {
        const uint append_physical = paged_timed_table_lookup(
            resources, params, logical_table_index, published_physical_id,
            thread_index);
        if (thread_index == 0u) {
          mix_paged_lookup(key, append_physical, logical_table_index,
                           kChecksumKeyDomain,
                           kChecksumPagedAppendLookupVisit, params, work_unit,
                           layer, batch);
          mix_paged_lookup(value, append_physical, logical_table_index,
                           kChecksumValueDomain,
                           kChecksumPagedAppendLookupVisit, params, work_unit,
                           layer, batch);
        }
        append_paged_pair(resources, params, work_unit, layer, batch,
                          logical_table_index, append_physical, thread_index,
                          threads_per_threadgroup, key, value);
      }
      const uint key_physical = paged_timed_table_lookup(
          resources, params, logical_table_index, published_physical_id,
          thread_index);
      if (thread_index == 0u) {
        mix_paged_lookup(key, key_physical, logical_table_index,
                         kChecksumKeyDomain, kChecksumPagedKeyLookupVisit,
                         params, work_unit, layer, batch);
      }
      scan_paged_key_block(resources, params, work_unit, layer, batch,
                           logical_table_index, key_physical, valid_bytes,
                           thread_index, threads_per_threadgroup, key);

      const uint value_physical = paged_timed_table_lookup(
          resources, params, logical_table_index, published_physical_id,
          thread_index);
      if (thread_index == 0u) {
        mix_paged_lookup(value, value_physical, logical_table_index,
                         kChecksumValueDomain, kChecksumPagedValueLookupVisit,
                         params, work_unit, layer, batch);
      }
      scan_paged_value_block(resources, params, work_unit, layer, batch,
                             logical_table_index, value_physical, valid_bytes,
                             thread_index, threads_per_threadgroup, value);
    }
  }
}

kernel void llm_metal_decode_paged_weights_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodePagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!decode_paged_parameters_valid(params)) {
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
      scan_paged_weight_range(resources, params, layer_start, layer_bytes,
                              work_unit, layer, global_id, grid_size, weight);
    }
  }
  publish_task_checksum(resources, weight, uint2(0u), uint2(0u), reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_decode_paged_kv_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodePagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    threadgroup uint* published_physical_id [[threadgroup(1)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint threadgroup_count [[threadgroups_per_grid]]) {
  if (!decode_paged_parameters_valid(params)) {
    if (threadgroup_index == 0u && thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  uint2 ignored_weight = uint2(0u);
  run_decode_paged_kv_owners(
      resources, params, published_physical_id, thread_index,
      threads_per_threadgroup, threadgroup_index, threadgroup_count, false,
      ignored_weight, key, value);
  publish_task_checksum(resources, uint2(0u), key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_decode_paged_mixed(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodePagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    threadgroup uint* published_physical_id [[threadgroup(1)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint threadgroup_count [[threadgroups_per_grid]]) {
  if (!decode_paged_parameters_valid(params)) {
    if (threadgroup_index == 0u && thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 weight = uint2(0u);
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  run_decode_paged_kv_owners(
      resources, params, published_physical_id, thread_index,
      threads_per_threadgroup, threadgroup_index, threadgroup_count, true,
      weight, key, value);
  publish_task_checksum(resources, weight, key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

inline uint paged_direct_table_lookup(
    constant LlmMetalResources& resources,
    constant LlmMetalDecodePagedParams& params, ulong logical_table_index) {
  const uint table_segment =
      uint(logical_table_index / params.table_entries_per_segment);
  const ulong table_entry =
      logical_table_index % params.table_entries_per_segment;
  const device uint* table = reinterpret_cast<const device uint*>(
      resources.table_segments[table_segment]);
  return table[table_entry];
}

inline uchar paged_expected_data_byte(
    constant LlmMetalDecodePagedParams& params, ulong seed, ulong layer,
    uint physical_id, ulong block_byte) {
  const ulong local_word = block_byte / 4ul;
  const uint byte_index = uint(block_byte & 3ul);
  const uint word = uint(seed) +
                    kPagedPatternLayerMultiplier * uint(layer + 1ul) +
                    kPagedPatternPhysicalMultiplier * (physical_id + 1u) +
                    kPagedPatternWordMultiplier * uint(local_word + 1ul);
  return uchar(word >> (byte_index * 8u));
}

kernel void llm_metal_validate_decode_paged_appends_padding(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalDecodePagedParams& params [[buffer(1)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if (!decode_paged_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  const ulong final_work_unit = params.work_units - 1ul;
  const ulong sequence_count = params.layer_count * params.batch_size;
  const ulong append_bytes =
      params.last_block_valid_bytes - params.append_offset_in_last_block;
  const ulong append_chunks_per_sequence =
      append_bytes / 16ul + ((append_bytes & 15ul) != 0ul ? 1ul : 0ul);
  const ulong append_chunk_count =
      sequence_count * append_chunks_per_sequence;
  for (ulong append_chunk = ulong(global_id);
       append_chunk < append_chunk_count;
       append_chunk += ulong(grid_size)) {
    const ulong sequence = append_chunk / append_chunks_per_sequence;
    const ulong sequence_append_chunk =
        append_chunk % append_chunks_per_sequence;
    const ulong layer = sequence / params.batch_size;
    const ulong batch = sequence % params.batch_size;
    const ulong logical_table_index =
        batch * params.blocks_per_sequence + params.blocks_per_sequence - 1ul;
    const uint physical_id =
        paged_direct_table_lookup(resources, params, logical_table_index);
    if (ulong(physical_id) >= params.physical_blocks_per_layer) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
      continue;
    }
    const ulong first_append_byte =
        params.append_offset_in_last_block + sequence_append_chunk * 16ul;
    const ulong end_append_byte =
        min(params.last_block_valid_bytes, first_append_byte + 16ul);
    for (ulong block_byte = first_append_byte;
         block_byte < end_append_byte; ++block_byte) {
      const ulong local_word = block_byte / 4ul;
      const uint byte_index = uint(block_byte & 3ul);
      const uchar expected_key = uchar(decode_append_word(
          params.scenario_seed, final_work_unit, layer, batch, local_word,
          kAppendKeyDomain) >> (byte_index * 8u));
      const uchar expected_value = uchar(decode_append_word(
          params.scenario_seed, final_work_unit, layer, batch, local_word,
          kAppendValueDomain) >> (byte_index * 8u));
      if (load_paged_key_byte(resources, params, layer, physical_id,
                              block_byte) != expected_key ||
          load_paged_value_byte(resources, params, layer, physical_id,
                                block_byte) != expected_value) {
        mark_validation_failure(resources.status_checksum,
                                kKvWriteValidationMismatchBit);
      }
    }
  }

  const ulong padding_bytes =
      params.block_bytes - params.last_block_valid_bytes;
  const ulong padding_chunks_per_sequence =
      padding_bytes / 16ul + ((padding_bytes & 15ul) != 0ul ? 1ul : 0ul);
  const ulong padding_chunk_count =
      sequence_count * padding_chunks_per_sequence;
  for (ulong padding_chunk = ulong(global_id);
       padding_chunk < padding_chunk_count;
       padding_chunk += ulong(grid_size)) {
    const ulong sequence =
        padding_chunk / padding_chunks_per_sequence;
    const ulong sequence_padding_chunk =
        padding_chunk % padding_chunks_per_sequence;
    const ulong layer = sequence / params.batch_size;
    const ulong batch = sequence % params.batch_size;
    const ulong logical_table_index =
        batch * params.blocks_per_sequence + params.blocks_per_sequence - 1ul;
    const uint physical_id =
        paged_direct_table_lookup(resources, params, logical_table_index);
    if (ulong(physical_id) >= params.physical_blocks_per_layer) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
      continue;
    }
    const ulong first_padding_byte =
        params.last_block_valid_bytes + sequence_padding_chunk * 16ul;
    const ulong end_padding_byte =
        min(params.block_bytes, first_padding_byte + 16ul);
    for (ulong block_byte = first_padding_byte;
         block_byte < end_padding_byte; ++block_byte) {
      if (load_paged_key_byte(resources, params, layer, physical_id,
                              block_byte) !=
              paged_expected_data_byte(params, params.k_seed, layer,
                                       physical_id, block_byte) ||
          load_paged_value_byte(resources, params, layer, physical_id,
                                block_byte) !=
              paged_expected_data_byte(params, params.v_seed, layer,
                                       physical_id, block_byte)) {
        mark_validation_failure(resources.status_checksum,
                                kPaddingCanaryMismatchBit);
      }
    }
  }
}

#endif  // LLM_METAL_DECODE_PAGED
#if LLM_METAL_PREFILL_PAGED
inline bool prefill_paged_parameters_valid(
    constant LlmMetalPrefillPagedParams& params) {
  if (params.prompt_tokens == 0ul ||
      params.attention_query_tile_tokens == 0ul ||
      params.attention_query_tile_tokens > params.prompt_tokens ||
      params.block_tokens == 0ul) {
    return false;
  }
  const ulong expected_tiles =
      params.prompt_tokens / params.attention_query_tile_tokens +
      ((params.prompt_tokens % params.attention_query_tile_tokens) != 0ul
           ? 1ul
           : 0ul);
  const ulong expected_blocks =
      params.prompt_tokens / params.block_tokens +
      ((params.prompt_tokens % params.block_tokens) != 0ul ? 1ul : 0ul);
  const ulong last_block_tokens =
      params.prompt_tokens - (expected_blocks - 1ul) * params.block_tokens;
  return params.weight_bytes != 0ul && params.tile_count == expected_tiles &&
         params.layer_count != 0ul && params.batch_size != 0ul &&
         params.record_bytes != 0ul && params.work_units != 0ul &&
         params.block_bytes == params.block_tokens * params.record_bytes &&
         params.last_block_valid_bytes ==
             last_block_tokens * params.record_bytes &&
         params.last_block_valid_bytes != 0ul &&
         params.last_block_valid_bytes <= params.block_bytes &&
         params.blocks_per_sequence == expected_blocks &&
         params.physical_blocks_per_layer ==
             params.batch_size * params.blocks_per_sequence &&
         params.blocks_per_segment != 0ul &&
         params.table_entries_per_segment != 0ul &&
         params.segment_capacity_bytes != 0ul &&
         params.weight_segment_count != 0u &&
         params.k_segment_count != 0u && params.v_segment_count != 0u &&
         params.table_segment_count != 0u && params.reserved_zero == 0u &&
         params.padding_zero == 0u;
}

inline uint prefill_paged_write_word(ulong scenario_seed, ulong work_unit,
                                     ulong layer, ulong batch,
                                     ulong logical_word_index,
                                     uint pool_domain) {
  return uint(scenario_seed) +
         kAppendWorkUnitMultiplier * uint(work_unit + 1ul) +
         kAppendLayerMultiplier * uint(layer + 1ul) +
         kAppendBatchMultiplier * uint(batch + 1ul) +
         kAppendWordMultiplier * uint(logical_word_index + 1ul) +
         pool_domain;
}

inline uint prefill_paged_data_checksum_domain(
    uint pool_domain, uint visit_domain,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong tile_ordinal, uint valid_mask) {
  return kChecksumMetalPrefillPagedProfileDomain +
         uint(params.scenario_seed) +
         kChecksumScenarioHighMultiplier *
             uint(params.scenario_seed >> 32ul) +
         pool_domain + visit_domain +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier * uint(batch + 1ul) +
         kChecksumTileMultiplier * uint(tile_ordinal) +
         kChecksumValidMaskMultiplier * valid_mask;
}

inline uint prefill_paged_lookup_checksum_domain(
    uint pool_domain, uint visit_domain,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong tile_ordinal,
    ulong logical_table_index, uint physical_id,
    uint physical_address_token) {
  const uint logical = uint(logical_table_index + 1ul);
  const uint physical = physical_id + 1u;
  return kChecksumMetalPrefillPagedProfileDomain +
         uint(params.scenario_seed) +
         kChecksumScenarioHighMultiplier *
             uint(params.scenario_seed >> 32ul) +
         pool_domain + visit_domain +
         kChecksumWorkUnitMultiplier * uint(work_unit + 1ul) +
         kChecksumLayerMultiplier * uint(layer + 1ul) +
         kChecksumBatchMultiplier * uint(batch + 1ul) +
         kChecksumTileMultiplier * uint(tile_ordinal) +
         kChecksumPagedLogicalMultiplier * logical +
         kChecksumPagedPhysicalMultiplier * physical +
         kChecksumPagedPairMultiplier * logical * physical +
         physical_address_token +
         kChecksumPagedAddressPairMultiplier * logical *
             physical_address_token;
}

inline uint prefill_paged_physical_address_token(
    ulong global_block, uint segment, ulong segment_local_block_base,
    ulong absolute_block_base) {
  return kChecksumPagedAddressBindingDomain +
         kChecksumPagedGlobalBlockLowMultiplier * uint(global_block) +
         kChecksumPagedGlobalBlockHighMultiplier *
             uint(global_block >> 32ul) +
         kChecksumPagedSegmentMultiplier * segment +
         kChecksumPagedLocalBaseLowMultiplier *
             uint(segment_local_block_base) +
         kChecksumPagedLocalBaseHighMultiplier *
             uint(segment_local_block_base >> 32ul) +
         kChecksumPagedAbsoluteBaseLowMultiplier *
             uint(absolute_block_base) +
         kChecksumPagedAbsoluteBaseHighMultiplier *
             uint(absolute_block_base >> 32ul);
}

inline uint prefill_paged_weight_checksum_domain(
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, uint valid_mask) {
  return prefill_paged_data_checksum_domain(
      kChecksumWeightDomain, kChecksumWeightReadVisit, params, work_unit,
      layer, 0ul, 0ul, valid_mask);
}

inline uint prefill_paged_timed_table_lookup(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params,
    ulong logical_table_index, threadgroup uint* published_physical_id,
    uint thread_index) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0u) {
    const uint table_segment =
        uint(logical_table_index / params.table_entries_per_segment);
    const ulong table_entry =
        logical_table_index % params.table_entries_per_segment;
    device const volatile uint* named_lane_table =
        reinterpret_cast<device const volatile uint*>(
            resources.table_segments[table_segment]);
    published_physical_id[0] = named_lane_table[table_entry];
    atomic_fetch_add_explicit(
        &resources.status_checksum[kLayoutMetadataLookupCountIndex], 1u,
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint physical_id = published_physical_id[0];
  if (ulong(physical_id) >= params.physical_blocks_per_layer) {
    if (thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return 0u;
  }
  return physical_id;
}

inline ulong prefill_paged_global_block(
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id) {
  return layer * params.physical_blocks_per_layer + ulong(physical_id);
}

inline ulong prefill_paged_physical_block_base(
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, thread uint& segment) {
  const ulong global_block =
      prefill_paged_global_block(params, layer, physical_id);
  segment = uint(global_block / params.blocks_per_segment);
  return (global_block % params.blocks_per_segment) * params.block_bytes;
}

inline void mix_prefill_paged_lookup(
    thread uint2& checksum, uint physical_id, ulong logical_table_index,
    uint pool_domain, uint visit_domain,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong tile_ordinal) {
  const ulong global_block =
      prefill_paged_global_block(params, layer, physical_id);
  uint segment = 0u;
  const ulong segment_local_block_base =
      prefill_paged_physical_block_base(params, layer, physical_id,
                                        segment);
  const ulong absolute_block_base = global_block * params.block_bytes;
  const uint physical_address_token =
      prefill_paged_physical_address_token(
          global_block, segment, segment_local_block_base,
          absolute_block_base);
  const uint domain = prefill_paged_lookup_checksum_domain(
      pool_domain, visit_domain, params, work_unit, layer, batch,
      tile_ordinal, logical_table_index, physical_id,
      physical_address_token);
  mix_checksum_word(checksum, physical_id, logical_table_index, domain);
}

inline ulong prefill_paged_logical_sequence_start(
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    ulong batch) {
  return (layer * params.batch_size + batch) * params.prompt_tokens *
         params.record_bytes;
}

inline uchar load_prefill_paged_key_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong block_byte) {
  uint segment = 0u;
  const ulong block_base =
      prefill_paged_physical_block_base(params, layer, physical_id, segment);
  return resources.key_segments[segment][block_base + block_byte];
}

inline uchar load_prefill_paged_value_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong block_byte) {
  uint segment = 0u;
  const ulong block_base =
      prefill_paged_physical_block_base(params, layer, physical_id, segment);
  return resources.value_segments[segment][block_base + block_byte];
}

inline void store_prefill_paged_key_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong block_byte, uchar value) {
  uint segment = 0u;
  const ulong block_base =
      prefill_paged_physical_block_base(params, layer, physical_id, segment);
  resources.key_segments[segment][block_base + block_byte] = value;
}

inline void store_prefill_paged_value_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong block_byte, uchar value) {
  uint segment = 0u;
  const ulong block_base =
      prefill_paged_physical_block_base(params, layer, physical_id, segment);
  resources.value_segments[segment][block_base + block_byte] = value;
}

inline uint packed_prefill_paged_key_word(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong logical_range_start, ulong logical_word_start,
    uint valid_mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((valid_mask & (1u << byte_index)) != 0u) {
      const ulong logical_byte = logical_word_start + ulong(byte_index);
      const ulong block_byte = logical_byte - logical_range_start;
      value |= uint(load_prefill_paged_key_byte(
                   resources, params, layer, physical_id, block_byte)) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline uint packed_prefill_paged_value_word(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong layer,
    uint physical_id, ulong logical_range_start, ulong logical_word_start,
    uint valid_mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((valid_mask & (1u << byte_index)) != 0u) {
      const ulong logical_byte = logical_word_start + ulong(byte_index);
      const ulong block_byte = logical_byte - logical_range_start;
      value |= uint(load_prefill_paged_value_byte(
                   resources, params, layer, physical_id, block_byte)) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline void write_prefill_paged_block_pair(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong logical_block, uint physical_id,
    ulong valid_bytes, uint thread_index, uint threads_per_threadgroup,
    thread uint2& key, thread uint2& value) {
  const ulong logical_range_start =
      prefill_paged_logical_sequence_start(params, layer, batch) +
      logical_block * params.block_bytes;
  const ulong logical_range_end = logical_range_start + valid_bytes;
  const ulong first_vector = logical_range_start / 16ul;
  const ulong end_vector = logical_range_end / 16ul +
                           ((logical_range_end & 15ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(
           first_vector, thread_index, threads_per_threadgroup);
       vector_index < end_vector;
       vector_index += ulong(threads_per_threadgroup)) {
    const ulong vector_start = vector_index * 16ul;
    const bool full_vector = vector_start >= logical_range_start &&
                             vector_start + 16ul <= logical_range_end;
    uint4 key_values = uint4(0u);
    uint4 value_values = uint4(0u);
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong logical_word = vector_index * 4ul + ulong(lane);
      const ulong logical_word_start = logical_word * 4ul;
      const uint mask = full_vector
                            ? 0x0fu
                            : range_word_mask(logical_range_start,
                                              logical_range_end,
                                              logical_word_start);
      if (mask == 0u) {
        continue;
      }
      const uint key_word = prefill_paged_write_word(
          params.scenario_seed, work_unit, layer, batch, logical_word,
          kPrefillWriteKeyDomain);
      const uint value_word = prefill_paged_write_word(
          params.scenario_seed, work_unit, layer, batch, logical_word,
          kPrefillWriteValueDomain);
      uint packed_key = 0u;
      uint packed_value = 0u;
      for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
        if ((mask & (1u << byte_index)) == 0u) {
          continue;
        }
        const ulong logical_byte = logical_word_start + ulong(byte_index);
        const ulong block_byte = logical_byte - logical_range_start;
        const uchar key_byte = uchar(key_word >> (byte_index * 8u));
        const uchar value_byte = uchar(value_word >> (byte_index * 8u));
        if (!full_vector) {
          store_prefill_paged_key_byte(resources, params, layer, physical_id,
                                       block_byte, key_byte);
          store_prefill_paged_value_byte(resources, params, layer,
                                         physical_id, block_byte,
                                         value_byte);
        }
        packed_key |= uint(key_byte) << (byte_index * 8u);
        packed_value |= uint(value_byte) << (byte_index * 8u);
      }
      key_values[lane] = packed_key;
      value_values[lane] = packed_value;
      mix_checksum_word(
          key, packed_key, logical_word,
          prefill_paged_data_checksum_domain(
              kChecksumKeyDomain, kChecksumPrefillWriteVisit, params,
              work_unit, layer, batch, 0ul, mask));
      mix_checksum_word(
          value, packed_value, logical_word,
          prefill_paged_data_checksum_domain(
              kChecksumValueDomain, kChecksumPrefillWriteVisit, params,
              work_unit, layer, batch, 0ul, mask));
    }
    if (full_vector) {
      const ulong block_byte = vector_start - logical_range_start;
      uint segment = 0u;
      const ulong physical_base = prefill_paged_physical_block_base(
          params, layer, physical_id, segment);
      if (((physical_base + block_byte) & 15ul) == 0ul) {
        device uint4* key_destination = reinterpret_cast<device uint4*>(
            resources.key_segments[segment] + physical_base + block_byte);
        device uint4* value_destination = reinterpret_cast<device uint4*>(
            resources.value_segments[segment] + physical_base + block_byte);
        *key_destination = key_values;
        *value_destination = value_values;
      } else {
        for (uint lane = 0u; lane < 4u; ++lane) {
          for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
            const ulong byte_offset =
                block_byte + ulong(lane * 4u + byte_index);
            store_prefill_paged_key_byte(
                resources, params, layer, physical_id, byte_offset,
                uchar(key_values[lane] >> (byte_index * 8u)));
            store_prefill_paged_value_byte(
                resources, params, layer, physical_id, byte_offset,
                uchar(value_values[lane] >> (byte_index * 8u)));
          }
        }
      }
    }
  }
}

inline void scan_prefill_paged_key_block(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong tile_ordinal, ulong logical_block,
    uint physical_id, ulong visit_bytes, uint thread_index,
    uint threads_per_threadgroup, thread uint2& checksum) {
  const ulong logical_range_start =
      prefill_paged_logical_sequence_start(params, layer, batch) +
      logical_block * params.block_bytes;
  const ulong logical_range_end = logical_range_start + visit_bytes;
  const ulong first_vector = logical_range_start / 16ul;
  const ulong end_vector = logical_range_end / 16ul +
                           ((logical_range_end & 15ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(
           first_vector, thread_index, threads_per_threadgroup);
       vector_index < end_vector;
       vector_index += ulong(threads_per_threadgroup)) {
    const ulong vector_start = vector_index * 16ul;
    const bool full_vector = vector_start >= logical_range_start &&
                             vector_start + 16ul <= logical_range_end;
    uint4 values = uint4(0u);
    if (full_vector) {
      const ulong block_byte = vector_start - logical_range_start;
      uint segment = 0u;
      const ulong physical_base = prefill_paged_physical_block_base(
          params, layer, physical_id, segment);
      if (((physical_base + block_byte) & 15ul) == 0ul) {
        const device uint4* source = reinterpret_cast<const device uint4*>(
            resources.key_segments[segment] + physical_base + block_byte);
        values = *source;
      } else {
        for (uint lane = 0u; lane < 4u; ++lane) {
          values[lane] = packed_prefill_paged_key_word(
              resources, params, layer, physical_id, logical_range_start,
              vector_start + ulong(lane) * 4ul, 0x0fu);
        }
      }
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong logical_word = vector_index * 4ul + ulong(lane);
      const ulong logical_word_start = logical_word * 4ul;
      const uint mask = full_vector
                            ? 0x0fu
                            : range_word_mask(logical_range_start,
                                              logical_range_end,
                                              logical_word_start);
      if (mask == 0u) {
        continue;
      }
      const uint packed =
          full_vector
              ? values[lane]
              : packed_prefill_paged_key_word(
                    resources, params, layer, physical_id,
                    logical_range_start, logical_word_start, mask);
      mix_checksum_word(
          checksum, packed, logical_word,
          prefill_paged_data_checksum_domain(
              kChecksumKeyDomain, kChecksumKvReadVisit, params, work_unit,
              layer, batch, tile_ordinal, mask));
    }
  }
}

inline void scan_prefill_paged_value_block(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong work_unit,
    ulong layer, ulong batch, ulong tile_ordinal, ulong logical_block,
    uint physical_id, ulong visit_bytes, uint thread_index,
    uint threads_per_threadgroup, thread uint2& checksum) {
  const ulong logical_range_start =
      prefill_paged_logical_sequence_start(params, layer, batch) +
      logical_block * params.block_bytes;
  const ulong logical_range_end = logical_range_start + visit_bytes;
  const ulong first_vector = logical_range_start / 16ul;
  const ulong end_vector = logical_range_end / 16ul +
                           ((logical_range_end & 15ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(
           first_vector, thread_index, threads_per_threadgroup);
       vector_index < end_vector;
       vector_index += ulong(threads_per_threadgroup)) {
    const ulong vector_start = vector_index * 16ul;
    const bool full_vector = vector_start >= logical_range_start &&
                             vector_start + 16ul <= logical_range_end;
    uint4 values = uint4(0u);
    if (full_vector) {
      const ulong block_byte = vector_start - logical_range_start;
      uint segment = 0u;
      const ulong physical_base = prefill_paged_physical_block_base(
          params, layer, physical_id, segment);
      if (((physical_base + block_byte) & 15ul) == 0ul) {
        const device uint4* source = reinterpret_cast<const device uint4*>(
            resources.value_segments[segment] + physical_base + block_byte);
        values = *source;
      } else {
        for (uint lane = 0u; lane < 4u; ++lane) {
          values[lane] = packed_prefill_paged_value_word(
              resources, params, layer, physical_id, logical_range_start,
              vector_start + ulong(lane) * 4ul, 0x0fu);
        }
      }
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong logical_word = vector_index * 4ul + ulong(lane);
      const ulong logical_word_start = logical_word * 4ul;
      const uint mask = full_vector
                            ? 0x0fu
                            : range_word_mask(logical_range_start,
                                              logical_range_end,
                                              logical_word_start);
      if (mask == 0u) {
        continue;
      }
      const uint packed =
          full_vector
              ? values[lane]
              : packed_prefill_paged_value_word(
                    resources, params, layer, physical_id,
                    logical_range_start, logical_word_start, mask);
      mix_checksum_word(
          checksum, packed, logical_word,
          prefill_paged_data_checksum_domain(
              kChecksumValueDomain, kChecksumKvReadVisit, params,
              work_unit, layer, batch, tile_ordinal, mask));
    }
  }
}

inline uchar load_prefill_paged_weight_byte(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong absolute_byte) {
  const uint segment = uint(absolute_byte / params.segment_capacity_bytes);
  const ulong local_byte = absolute_byte % params.segment_capacity_bytes;
  return resources.weight_segments[segment][local_byte];
}

inline uint packed_prefill_paged_weight_word(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong word_start,
    uint mask) {
  uint value = 0u;
  for (uint byte_index = 0u; byte_index < 4u; ++byte_index) {
    if ((mask & (1u << byte_index)) != 0u) {
      value |= uint(load_prefill_paged_weight_byte(
                   resources, params, word_start + ulong(byte_index))) <<
               (byte_index * 8u);
    }
  }
  return value;
}

inline void scan_prefill_paged_weight_range(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params, ulong range_start,
    ulong range_length, ulong work_unit, ulong layer, uint lane_index,
    uint lane_count, thread uint2& checksum) {
  const ulong range_end = range_start + range_length;
  const ulong first_vector = range_start / 16ul;
  const ulong end_vector = range_end / 16ul +
                           ((range_end & 15ul) != 0ul ? 1ul : 0ul);
  for (ulong vector_index = first_owned_vector(first_vector, lane_index,
                                                lane_count);
       vector_index < end_vector; vector_index += ulong(lane_count)) {
    const ulong vector_start = vector_index * 16ul;
    if (vector_start >= range_start && vector_start + 16ul <= range_end) {
      const uint segment =
          uint(vector_start / params.segment_capacity_bytes);
      const ulong local_byte =
          vector_start % params.segment_capacity_bytes;
      const device uint4* source = reinterpret_cast<const device uint4*>(
          resources.weight_segments[segment] + local_byte);
      const uint4 values = *source;
      for (uint lane = 0u; lane < 4u; ++lane) {
        const ulong word_index = vector_index * 4ul + ulong(lane);
        mix_checksum_word(
            checksum, values[lane], word_index,
            prefill_paged_weight_checksum_domain(params, work_unit, layer,
                                                  0x0fu));
      }
      continue;
    }
    for (uint lane = 0u; lane < 4u; ++lane) {
      const ulong word_index = vector_index * 4ul + ulong(lane);
      const ulong word_start = word_index * 4ul;
      const uint mask = range_word_mask(range_start, range_end, word_start);
      if (mask != 0u) {
        const uint packed = packed_prefill_paged_weight_word(
            resources, params, word_start, mask);
        mix_checksum_word(
            checksum, packed, word_index,
            prefill_paged_weight_checksum_domain(params, work_unit, layer,
                                                  mask));
      }
    }
  }
}

inline void run_prefill_paged_kv_owner_rows(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params,
    threadgroup uint* published_physical_id, uint thread_index,
    uint threads_per_threadgroup, uint threadgroup_index,
    uint threadgroup_count, bool include_weight, thread uint2& weight,
    thread uint2& key, thread uint2& value) {
  const ulong owner_count = params.layer_count * params.batch_size *
                            params.blocks_per_sequence;
  const ulong weight_layer_base = params.weight_bytes / params.layer_count;
  const ulong weight_layer_remainder =
      params.weight_bytes % params.layer_count;
  for (ulong work_unit = 0ul; work_unit < params.work_units; ++work_unit) {
    ulong first_owner = ulong(threadgroup_index);
    while (first_owner < owner_count) {
      const ulong row = first_owner / params.blocks_per_sequence;
      const ulong logical_block_first =
          first_owner % params.blocks_per_sequence;
      const ulong batch = row % params.batch_size;
      const ulong layer = row / params.batch_size;
      const ulong logical_table_row = batch * params.blocks_per_sequence;

      if (include_weight && batch == 0ul && logical_block_first == 0ul) {
        const ulong layer_start = layer * weight_layer_base +
                                  min(layer, weight_layer_remainder);
        const ulong layer_bytes =
            weight_layer_base +
            (layer < weight_layer_remainder ? 1ul : 0ul);
        scan_prefill_paged_weight_range(
            resources, params, layer_start, layer_bytes, work_unit, layer,
            thread_index, threads_per_threadgroup, weight);
      }

      // Row write phase: every owned K/V block is populated before any read.
      for (ulong logical_block = logical_block_first;
           logical_block < params.blocks_per_sequence;
           logical_block += ulong(threadgroup_count)) {
        const ulong logical_table_index =
            logical_table_row + logical_block;
        const uint physical_id = prefill_paged_timed_table_lookup(
            resources, params, logical_table_index, published_physical_id,
            thread_index);
        if (thread_index == 0u) {
          mix_prefill_paged_lookup(
              key, physical_id, logical_table_index, kChecksumKeyDomain,
              kChecksumPagedPrefillWriteLookupVisit, params, work_unit,
              layer, batch, 0ul);
          mix_prefill_paged_lookup(
              value, physical_id, logical_table_index,
              kChecksumValueDomain,
              kChecksumPagedPrefillWriteLookupVisit, params, work_unit,
              layer, batch, 0ul);
        }
        const ulong valid_bytes =
            logical_block + 1ul == params.blocks_per_sequence
                ? params.last_block_valid_bytes
                : params.block_bytes;
        write_prefill_paged_block_pair(
            resources, params, work_unit, layer, batch, logical_block,
            physical_id, valid_bytes, thread_index,
            threads_per_threadgroup, key, value);
      }

      // Conservatively publish all same-row writes before tiled read phases.
      threadgroup_barrier(mem_flags::mem_device);

      ulong remaining_tokens = params.prompt_tokens;
      ulong prefix_tokens = 0ul;
      ulong tile_ordinal = 0ul;
      while (remaining_tokens != 0ul) {
        const ulong tile_tokens =
            min(params.attention_query_tile_tokens, remaining_tokens);
        prefix_tokens += tile_tokens;
        ++tile_ordinal;

        // Each K visit performs its own semantic table lookup.
        for (ulong logical_block = logical_block_first;
             logical_block < params.blocks_per_sequence;
             logical_block += ulong(threadgroup_count)) {
          const ulong block_start_token =
              logical_block * params.block_tokens;
          if (block_start_token >= prefix_tokens) {
            break;
          }
          const ulong visit_tokens =
              min(params.block_tokens, prefix_tokens - block_start_token);
          const ulong visit_bytes = visit_tokens * params.record_bytes;
          const ulong logical_table_index =
              logical_table_row + logical_block;
          const uint physical_id = prefill_paged_timed_table_lookup(
              resources, params, logical_table_index,
              published_physical_id, thread_index);
          if (thread_index == 0u) {
            mix_prefill_paged_lookup(
                key, physical_id, logical_table_index, kChecksumKeyDomain,
                kChecksumPagedPrefillKeyLookupVisit, params, work_unit,
                layer, batch, tile_ordinal);
          }
          scan_prefill_paged_key_block(
              resources, params, work_unit, layer, batch, tile_ordinal,
              logical_block, physical_id, visit_bytes, thread_index,
              threads_per_threadgroup, key);
        }

        // The same row prefix is revisited for V after all of its K blocks.
        for (ulong logical_block = logical_block_first;
             logical_block < params.blocks_per_sequence;
             logical_block += ulong(threadgroup_count)) {
          const ulong block_start_token =
              logical_block * params.block_tokens;
          if (block_start_token >= prefix_tokens) {
            break;
          }
          const ulong visit_tokens =
              min(params.block_tokens, prefix_tokens - block_start_token);
          const ulong visit_bytes = visit_tokens * params.record_bytes;
          const ulong logical_table_index =
              logical_table_row + logical_block;
          const uint physical_id = prefill_paged_timed_table_lookup(
              resources, params, logical_table_index,
              published_physical_id, thread_index);
          if (thread_index == 0u) {
            mix_prefill_paged_lookup(
                value, physical_id, logical_table_index,
                kChecksumValueDomain,
                kChecksumPagedPrefillValueLookupVisit, params, work_unit,
                layer, batch, tile_ordinal);
          }
          scan_prefill_paged_value_block(
              resources, params, work_unit, layer, batch, tile_ordinal,
              logical_block, physical_id, visit_bytes, thread_index,
              threads_per_threadgroup, value);
        }
        remaining_tokens -= tile_tokens;
      }

      const ulong remaining_blocks =
          params.blocks_per_sequence - logical_block_first;
      const ulong owner_steps =
          remaining_blocks / ulong(threadgroup_count) +
          ((remaining_blocks % ulong(threadgroup_count)) != 0ul ? 1ul
                                                                : 0ul);
      first_owner += owner_steps * ulong(threadgroup_count);
    }
  }
}

kernel void llm_metal_prefill_paged_weights_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillPagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  if (!prefill_paged_parameters_valid(params)) {
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
      const ulong layer_bytes =
          layer_base + (layer < layer_remainder ? 1ul : 0ul);
      scan_prefill_paged_weight_range(
          resources, params, layer_start, layer_bytes, work_unit, layer,
          global_id, grid_size, weight);
    }
  }
  publish_task_checksum(resources, weight, uint2(0u), uint2(0u), reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_prefill_paged_kv_only(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillPagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    threadgroup uint* published_physical_id [[threadgroup(1)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint threadgroup_count [[threadgroups_per_grid]]) {
  if (!prefill_paged_parameters_valid(params)) {
    if (threadgroup_index == 0u && thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  uint2 ignored_weight = uint2(0u);
  run_prefill_paged_kv_owner_rows(
      resources, params, published_physical_id, thread_index,
      threads_per_threadgroup, threadgroup_index, threadgroup_count, false,
      ignored_weight, key, value);
  publish_task_checksum(resources, uint2(0u), key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

kernel void llm_metal_prefill_paged_mixed(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillPagedParams& params [[buffer(1)]],
    threadgroup uint* reduction [[threadgroup(0)]],
    threadgroup uint* published_physical_id [[threadgroup(1)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint threadgroup_count [[threadgroups_per_grid]]) {
  if (!prefill_paged_parameters_valid(params)) {
    if (threadgroup_index == 0u && thread_index == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  uint2 weight = uint2(0u);
  uint2 key = uint2(0u);
  uint2 value = uint2(0u);
  run_prefill_paged_kv_owner_rows(
      resources, params, published_physical_id, thread_index,
      threads_per_threadgroup, threadgroup_index, threadgroup_count, true,
      weight, key, value);
  publish_task_checksum(resources, weight, key, value, reduction,
                        thread_index, threads_per_threadgroup);
}

inline uint prefill_paged_direct_table_lookup(
    constant LlmMetalResources& resources,
    constant LlmMetalPrefillPagedParams& params,
    ulong logical_table_index) {
  const uint table_segment =
      uint(logical_table_index / params.table_entries_per_segment);
  const ulong table_entry =
      logical_table_index % params.table_entries_per_segment;
  const device uint* table = reinterpret_cast<const device uint*>(
      resources.table_segments[table_segment]);
  return table[table_entry];
}

inline uchar prefill_paged_expected_padding_byte(
    ulong seed, ulong layer, uint physical_id, ulong block_byte) {
  const ulong local_word = block_byte / 4ul;
  const uint byte_index = uint(block_byte & 3ul);
  const uint word = uint(seed) +
                    kPagedPatternLayerMultiplier * uint(layer + 1ul) +
                    kPagedPatternPhysicalMultiplier * (physical_id + 1u) +
                    kPagedPatternWordMultiplier * uint(local_word + 1ul);
  return uchar(word >> (byte_index * 8u));
}

kernel void llm_metal_validate_prefill_paged_writes_padding(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillPagedParams& params [[buffer(1)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if (!prefill_paged_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  const ulong samples_per_block = 6ul;
  const ulong sequence_count = params.layer_count * params.batch_size;
  const ulong sample_count = sequence_count * params.blocks_per_sequence *
                             samples_per_block;
  const ulong final_work_unit = params.work_units - 1ul;
  for (ulong sample = ulong(global_id); sample < sample_count;
       sample += ulong(grid_size)) {
    const ulong block_sample = sample / samples_per_block;
    const ulong sample_in_block = sample % samples_per_block;
    const ulong sequence = block_sample / params.blocks_per_sequence;
    const ulong logical_block =
        block_sample % params.blocks_per_sequence;
    const ulong layer = sequence / params.batch_size;
    const ulong batch = sequence % params.batch_size;
    const ulong logical_table_index =
        batch * params.blocks_per_sequence + logical_block;
    const uint physical_id = prefill_paged_direct_table_lookup(
        resources, params, logical_table_index);
    if (ulong(physical_id) >= params.physical_blocks_per_layer) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
      continue;
    }
    const ulong valid_bytes =
        logical_block + 1ul == params.blocks_per_sequence
            ? params.last_block_valid_bytes
            : params.block_bytes;
    ulong block_byte = 0ul;
    switch (sample_in_block) {
      case 0ul:
        block_byte = 0ul;
        break;
      case 1ul:
        block_byte = min(3ul, valid_bytes - 1ul);
        break;
      case 2ul:
        block_byte = (valid_bytes - 1ul) / 2ul;
        break;
      case 3ul:
        block_byte = valid_bytes / 2ul;
        break;
      case 4ul:
        block_byte = valid_bytes > 4ul ? valid_bytes - 4ul : 0ul;
        break;
      default:
        block_byte = valid_bytes - 1ul;
        break;
    }
    const ulong logical_byte =
        prefill_paged_logical_sequence_start(params, layer, batch) +
        logical_block * params.block_bytes + block_byte;
    const ulong logical_word = logical_byte / 4ul;
    const uint byte_index = uint(logical_byte & 3ul);
    const uchar expected_key = uchar(prefill_paged_write_word(
        params.scenario_seed, final_work_unit, layer, batch, logical_word,
        kPrefillWriteKeyDomain) >> (byte_index * 8u));
    const uchar expected_value = uchar(prefill_paged_write_word(
        params.scenario_seed, final_work_unit, layer, batch, logical_word,
        kPrefillWriteValueDomain) >> (byte_index * 8u));
    if (load_prefill_paged_key_byte(resources, params, layer, physical_id,
                                    block_byte) != expected_key ||
        load_prefill_paged_value_byte(resources, params, layer, physical_id,
                                      block_byte) != expected_value) {
      mark_validation_failure(resources.status_checksum,
                              kKvWriteValidationMismatchBit);
    }
  }

  const ulong padding_bytes =
      params.block_bytes - params.last_block_valid_bytes;
  const ulong padding_chunks_per_sequence =
      padding_bytes / 16ul + ((padding_bytes & 15ul) != 0ul ? 1ul : 0ul);
  const ulong padding_chunk_count =
      sequence_count * padding_chunks_per_sequence;
  for (ulong padding_chunk = ulong(global_id);
       padding_chunk < padding_chunk_count;
       padding_chunk += ulong(grid_size)) {
    const ulong sequence =
        padding_chunk / padding_chunks_per_sequence;
    const ulong sequence_padding_chunk =
        padding_chunk % padding_chunks_per_sequence;
    const ulong layer = sequence / params.batch_size;
    const ulong batch = sequence % params.batch_size;
    const ulong logical_table_index =
        batch * params.blocks_per_sequence +
        params.blocks_per_sequence - 1ul;
    const uint physical_id = prefill_paged_direct_table_lookup(
        resources, params, logical_table_index);
    if (ulong(physical_id) >= params.physical_blocks_per_layer) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
      continue;
    }
    const ulong first_padding_byte =
        params.last_block_valid_bytes + sequence_padding_chunk * 16ul;
    const ulong end_padding_byte =
        min(params.block_bytes, first_padding_byte + 16ul);
    for (ulong block_byte = first_padding_byte;
         block_byte < end_padding_byte; ++block_byte) {
      if (load_prefill_paged_key_byte(resources, params, layer, physical_id,
                                      block_byte) !=
              prefill_paged_expected_padding_byte(
                  params.k_seed, layer, physical_id, block_byte) ||
          load_prefill_paged_value_byte(resources, params, layer,
                                        physical_id, block_byte) !=
              prefill_paged_expected_padding_byte(
                  params.v_seed, layer, physical_id, block_byte)) {
        mark_validation_failure(resources.status_checksum,
                                kPaddingCanaryMismatchBit);
      }
    }
  }
}

kernel void llm_metal_probe_prefill_paged_parameter_layout(
    constant LlmMetalPrefillPagedParams& params [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]) {
  if (global_id != 0u) {
    return;
  }
  output[0] = 1ul;
  output[1] = ulong(sizeof(LlmMetalPrefillPagedParams));
  output[2] = ulong(alignof(LlmMetalPrefillPagedParams));
  output[3] = 26ul;
  output[4] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, weight_bytes));
  output[5] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, prompt_tokens));
  output[6] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, attention_query_tile_tokens));
  output[7] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, tile_count));
  output[8] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, layer_count));
  output[9] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, batch_size));
  output[10] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, record_bytes));
  output[11] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, work_units));
  output[12] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, block_tokens));
  output[13] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, block_bytes));
  output[14] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, last_block_valid_bytes));
  output[15] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, blocks_per_sequence));
  output[16] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, physical_blocks_per_layer));
  output[17] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, blocks_per_segment));
  output[18] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, table_entries_per_segment));
  output[19] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, segment_capacity_bytes));
  output[20] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, weight_seed));
  output[21] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, k_seed));
  output[22] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, v_seed));
  output[23] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, scenario_seed));
  output[24] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, weight_segment_count));
  output[25] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, k_segment_count));
  output[26] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, v_segment_count));
  output[27] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, table_segment_count));
  output[28] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, reserved_zero));
  output[29] = ulong(__builtin_offsetof(
      LlmMetalPrefillPagedParams, padding_zero));
  output[30] = params.weight_bytes;
  output[31] = params.prompt_tokens;
  output[32] = params.attention_query_tile_tokens;
  output[33] = params.tile_count;
  output[34] = params.layer_count;
  output[35] = params.batch_size;
  output[36] = params.record_bytes;
  output[37] = params.work_units;
  output[38] = params.block_tokens;
  output[39] = params.block_bytes;
  output[40] = params.last_block_valid_bytes;
  output[41] = params.blocks_per_sequence;
  output[42] = params.physical_blocks_per_layer;
  output[43] = params.blocks_per_segment;
  output[44] = params.table_entries_per_segment;
  output[45] = params.segment_capacity_bytes;
  output[46] = params.weight_seed;
  output[47] = params.k_seed;
  output[48] = params.v_seed;
  output[49] = params.scenario_seed;
  output[50] = ulong(params.weight_segment_count);
  output[51] = ulong(params.k_segment_count);
  output[52] = ulong(params.v_segment_count);
  output[53] = ulong(params.table_segment_count);
  output[54] = ulong(params.reserved_zero);
  output[55] = ulong(params.padding_zero);
}
#endif  // LLM_METAL_PREFILL_PAGED
#if LLM_METAL_PREFILL_CONTIGUOUS
kernel void llm_metal_validate_prefill_contiguous_writes(
    constant LlmMetalResources& resources [[buffer(0)]],
    constant LlmMetalPrefillContiguousParams& params [[buffer(1)]],
    uint global_id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  if (!prefill_parameters_valid(params)) {
    if (global_id == 0u) {
      mark_validation_failure(resources.status_checksum,
                              kValidationInvalidParametersBit);
    }
    return;
  }
  const ulong samples_per_sequence = 6ul;
  const ulong sequence_bytes = params.prompt_tokens * params.record_bytes;
  const ulong sequence_count = params.layer_count * params.batch_size;
  const ulong sample_count = sequence_count * samples_per_sequence;
  const ulong final_work_unit = params.work_units - 1ul;
  for (ulong sample = ulong(global_id); sample < sample_count;
       sample += ulong(grid_size)) {
    const ulong sequence = sample / samples_per_sequence;
    const ulong sample_in_sequence = sample % samples_per_sequence;
    const ulong layer = sequence / params.batch_size;
    const ulong batch = sequence % params.batch_size;
    ulong sequence_byte = 0ul;
    switch (sample_in_sequence) {
      case 0ul:
        sequence_byte = 0ul;
        break;
      case 1ul:
        sequence_byte = min(3ul, params.record_bytes - 1ul);
        break;
      case 2ul:
        sequence_byte = (params.prompt_tokens / 2ul) * params.record_bytes;
        break;
      case 3ul:
        sequence_byte = (params.prompt_tokens / 2ul) * params.record_bytes +
                        min(3ul, params.record_bytes - 1ul);
        break;
      case 4ul:
        sequence_byte = sequence_bytes - params.record_bytes;
        break;
      default:
        sequence_byte = sequence_bytes - 1ul;
        break;
    }
    const ulong absolute_byte = sequence * sequence_bytes + sequence_byte;
    const ulong word_index = absolute_byte / 4ul;
    const uint byte_index = uint(absolute_byte % 4ul);
    const uchar expected_key = uchar(prefill_write_word(
        params.scenario_seed, final_work_unit, layer, batch, word_index,
        kPrefillWriteKeyDomain) >> (byte_index * 8u));
    const uchar expected_value = uchar(prefill_write_word(
        params.scenario_seed, final_work_unit, layer, batch, word_index,
        kPrefillWriteValueDomain) >> (byte_index * 8u));
    if (load_key_byte(resources, params, absolute_byte) != expected_key ||
        load_value_byte(resources, params, absolute_byte) != expected_value) {
      mark_validation_failure(resources.status_checksum,
                              kKvWriteValidationMismatchBit);
    }
  }
}

kernel void llm_metal_probe_prefill_parameter_layout(
    constant LlmMetalPrefillContiguousParams& params [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]) {
  if (global_id != 0u) {
    return;
  }
  output[0] = 1ul;
  output[1] = ulong(sizeof(LlmMetalPrefillContiguousParams));
  output[2] = ulong(alignof(LlmMetalPrefillContiguousParams));
  output[3] = 19ul;
  output[4] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, weight_bytes));
  output[5] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, k_bytes));
  output[6] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, v_bytes));
  output[7] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, segment_capacity_bytes));
  output[8] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, prompt_tokens));
  output[9] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, attention_query_tile_tokens));
  output[10] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, tile_count));
  output[11] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, layer_count));
  output[12] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, batch_size));
  output[13] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, record_bytes));
  output[14] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, work_units));
  output[15] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, weight_seed));
  output[16] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, k_seed));
  output[17] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, v_seed));
  output[18] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, scenario_seed));
  output[19] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, weight_segment_count));
  output[20] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, k_segment_count));
  output[21] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, v_segment_count));
  output[22] = ulong(__builtin_offsetof(
      LlmMetalPrefillContiguousParams, reserved_zero));
  output[23] = params.weight_bytes;
  output[24] = params.k_bytes;
  output[25] = params.v_bytes;
  output[26] = params.segment_capacity_bytes;
  output[27] = params.prompt_tokens;
  output[28] = params.attention_query_tile_tokens;
  output[29] = params.tile_count;
  output[30] = params.layer_count;
  output[31] = params.batch_size;
  output[32] = params.record_bytes;
  output[33] = params.work_units;
  output[34] = params.weight_seed;
  output[35] = params.k_seed;
  output[36] = params.v_seed;
  output[37] = params.scenario_seed;
  output[38] = ulong(params.weight_segment_count);
  output[39] = ulong(params.k_segment_count);
  output[40] = ulong(params.v_segment_count);
  output[41] = ulong(params.reserved_zero);
}
#endif  // LLM_METAL_PREFILL_CONTIGUOUS
#if LLM_METAL_DECODE_CONTIGUOUS
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
                                  kKvWriteValidationMismatchBit);
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

#endif  // LLM_METAL_DECODE_CONTIGUOUS
#if LLM_METAL_DECODE_PAGED
kernel void llm_metal_probe_decode_paged_parameter_layout(
    constant LlmMetalDecodePagedParams& params [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]) {
  if (global_id != 0u) {
    return;
  }
  output[0] = 1ul;
  output[1] = ulong(sizeof(LlmMetalDecodePagedParams));
  output[2] = ulong(alignof(LlmMetalDecodePagedParams));
  output[3] = 24ul;
  output[4] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, weight_bytes));
  output[5] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, context_tokens));
  output[6] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, layer_count));
  output[7] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, batch_size));
  output[8] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, record_bytes));
  output[9] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, work_units));
  output[10] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, block_bytes));
  output[11] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, last_block_valid_bytes));
  output[12] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, append_offset_in_last_block));
  output[13] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, blocks_per_sequence));
  output[14] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, physical_blocks_per_layer));
  output[15] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, blocks_per_segment));
  output[16] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, table_entries_per_segment));
  output[17] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, segment_capacity_bytes));
  output[18] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, weight_seed));
  output[19] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, k_seed));
  output[20] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, v_seed));
  output[21] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, scenario_seed));
  output[22] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, weight_segment_count));
  output[23] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, k_segment_count));
  output[24] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, v_segment_count));
  output[25] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, table_segment_count));
  output[26] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, reserved_zero));
  output[27] = ulong(__builtin_offsetof(LlmMetalDecodePagedParams, padding_zero));
  output[28] = params.weight_bytes;
  output[29] = params.context_tokens;
  output[30] = params.layer_count;
  output[31] = params.batch_size;
  output[32] = params.record_bytes;
  output[33] = params.work_units;
  output[34] = params.block_bytes;
  output[35] = params.last_block_valid_bytes;
  output[36] = params.append_offset_in_last_block;
  output[37] = params.blocks_per_sequence;
  output[38] = params.physical_blocks_per_layer;
  output[39] = params.blocks_per_segment;
  output[40] = params.table_entries_per_segment;
  output[41] = params.segment_capacity_bytes;
  output[42] = params.weight_seed;
  output[43] = params.k_seed;
  output[44] = params.v_seed;
  output[45] = params.scenario_seed;
  output[46] = ulong(params.weight_segment_count);
  output[47] = ulong(params.k_segment_count);
  output[48] = ulong(params.v_segment_count);
  output[49] = ulong(params.table_segment_count);
  output[50] = ulong(params.reserved_zero);
  output[51] = ulong(params.padding_zero);
}

#endif  // LLM_METAL_DECODE_PAGED
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
