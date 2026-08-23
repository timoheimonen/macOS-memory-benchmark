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

/**
 * @file llm_work_plan.h
 * @brief Overflow-safe logical and backend-specific LLM memory work planning
 */

#ifndef LLM_WORK_PLAN_H
#define LLM_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "core/memory/memory_manager.h"
#include "llm_memory/llm_kv_layout.h"
#include "llm_memory/llm_memory.h"
#include "llm_memory/llm_prefill.h"

/** Stable machine-readable reasons emitted by LLM geometry and planning. */
namespace LlmWorkPlanReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* ACTIVE_WEIGHT_BYTES_ZERO =
    "active-weight-bytes-zero";
inline constexpr const char* LAYER_COUNT_ZERO = "layer-count-zero";
inline constexpr const char* QUERY_HEAD_COUNT_ZERO = "query-head-count-zero";
inline constexpr const char* KV_HEAD_COUNT_ZERO = "kv-head-count-zero";
inline constexpr const char* HEAD_DIMENSION_ZERO = "head-dimension-zero";
inline constexpr const char* INVALID_KV_ELEMENT_BYTES =
    "invalid-kv-element-bytes";
inline constexpr const char* CONTEXT_TOKENS_ZERO = "context-tokens-zero";
inline constexpr const char* CONTEXT_TOKENS_NOT_APPLICABLE =
    "context-tokens-not-applicable";
inline constexpr const char* PROMPT_TOKENS_NOT_APPLICABLE =
    "prompt-tokens-not-applicable";
inline constexpr const char* QUERY_TILE_TOKENS_NOT_APPLICABLE =
    "attention-query-tile-tokens-not-applicable";
inline constexpr const char* KV_BLOCK_TOKENS_NOT_APPLICABLE =
    "kv-block-tokens-not-applicable";
inline constexpr const char* BATCH_SIZE_ZERO = "batch-size-zero";
inline constexpr const char* QUERY_HEADS_BELOW_KV_HEADS =
    "query-heads-below-kv-heads";
inline constexpr const char* QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS =
    "query-heads-not-divisible-by-kv-heads";
inline constexpr const char* KV_VECTOR_BYTES_OVERFLOW =
    "kv-vector-bytes-overflow";
inline constexpr const char* KV_RECORD_BYTES_OVERFLOW =
    "kv-record-bytes-overflow";
inline constexpr const char* KV_LAYER_RECORD_BYTES_OVERFLOW =
    "kv-layer-record-bytes-overflow";
inline constexpr const char* KV_BYTES_PER_TOKEN_OVERFLOW =
    "kv-bytes-per-token-overflow";
inline constexpr const char* KV_SEQUENCE_BYTES_OVERFLOW =
    "kv-sequence-bytes-overflow";
inline constexpr const char* KV_MAPPING_BYTES_OVERFLOW =
    "kv-mapping-bytes-overflow";
inline constexpr const char* KV_CAPACITY_BYTES_OVERFLOW =
    "kv-capacity-bytes-overflow";
inline constexpr const char* KV_WRITE_BYTES_OVERFLOW =
    "kv-write-bytes-overflow";
inline constexpr const char* KV_READ_BYTES_OVERFLOW =
    "kv-read-bytes-overflow";
inline constexpr const char* KV_ONLY_PAYLOAD_OVERFLOW =
    "kv-only-payload-overflow";
inline constexpr const char* MIXED_PAYLOAD_OVERFLOW =
    "mixed-payload-overflow";
inline constexpr const char* TOTAL_DATA_BYTES_OVERFLOW =
    "total-data-bytes-overflow";
inline constexpr const char* REQUESTED_WORKERS_ZERO =
    "requested-workers-zero";
inline constexpr const char* AVAILABLE_WORKERS_ZERO =
    "available-workers-zero";
inline constexpr const char* NO_EXECUTABLE_WORKER =
    "no-executable-worker";
inline constexpr const char* LAYER_SEQUENCE_COUNT_OVERFLOW =
    "layer-sequence-count-overflow";
inline constexpr const char* DESCRIPTOR_COUNT_OVERFLOW =
    "descriptor-count-overflow";
inline constexpr const char* DESCRIPTOR_BYTES_OVERFLOW =
    "descriptor-bytes-overflow";
inline constexpr const char* PLANNER_STORAGE_BYTES_OVERFLOW =
    "planner-storage-bytes-overflow";
inline constexpr const char* MAPPING_GRANULARITY_ZERO =
    "mapping-granularity-zero";
inline constexpr const char* MAPPING_ROUND_UP_OVERFLOW =
    "mapping-round-up-overflow";
inline constexpr const char* AUXILIARY_BYTES_OVERFLOW =
    "auxiliary-bytes-overflow";
inline constexpr const char* AUXILIARY_PREFLIGHT_MISMATCH =
    "auxiliary-preflight-mismatch";
inline constexpr const char* MEMORY_REQUIREMENT_OVERFLOW =
    "memory-requirement-overflow";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW =
    "memory-budget-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED =
    "memory-budget-exceeded";
inline constexpr const char* WITHIN_MEMORY_BUDGET =
    "within-memory-budget";
inline constexpr const char* PLANNER_ALLOCATION_FAILED =
    "planner-allocation-failed";
inline constexpr const char* BLOCK_TABLE_MAPPING_FAILED =
    "block-table-mapping-failed";
inline constexpr const char* BLOCK_TABLE_MATERIALIZATION_FAILED =
    "block-table-materialization-failed";
inline constexpr const char* BLOCK_TABLE_PROTECTION_FAILED =
    "block-table-protection-failed";
inline constexpr const char* INVALID_SCENARIO = "invalid-scenario";
inline constexpr const char* INVALID_MODEL_WORK_PLAN =
    "invalid-model-work-plan";
inline constexpr const char* INVALID_BACKEND = "invalid-backend";
inline constexpr const char* METAL_WORKERS_NOT_APPLICABLE =
    "metal-workers-not-applicable";
inline constexpr const char* INVALID_PHASE = "invalid-phase";
inline constexpr const char* INVALID_KV_LAYOUT = "invalid-kv-layout";
inline constexpr const char* JSON_INTEGER_OUT_OF_RANGE =
    "json-integer-out-of-range";
inline constexpr const char* GUARDRAIL_BELOW_ONE_WORK_UNIT =
    "guardrail-below-one-work-unit";
inline constexpr const char* WORK_UNIT_COUNT_ZERO = "work-unit-count-zero";
inline constexpr const char* WORK_UNIT_CAP_EXCEEDED =
    "work-unit-cap-exceeded";
inline constexpr const char* TASK_ACCOUNTED_BYTES_OVERFLOW =
    "task-accounted-bytes-overflow";
inline constexpr const char* TASK_ACCOUNTED_BYTES_CAP_EXCEEDED =
    "task-accounted-bytes-cap-exceeded";
}  // namespace LlmWorkPlanReason

/** Versioned Phase-9 Metal decode/contiguous planning identities. */
namespace LlmMetalDecodeContiguousVersion {
inline constexpr const char* EXECUTOR =
    "llm-metal-executor-v1-decode-contiguous";
inline constexpr const char* SCHEDULE =
    "llm-metal-decode-contiguous-grid-stride-v1";
inline constexpr const char* TIMER =
    "metal-command-buffer-gpu-start-end-v1";
inline constexpr const char* BUFFER_PATTERN =
    "llm-metal-contiguous-affine32-v1";
inline constexpr const char* WRITE_PATTERN =
    "llm-metal-decode-append-affine32-v1";
inline constexpr const char* CHECKSUM = "llm-metal-dual-mod32-v1";
}  // namespace LlmMetalDecodeContiguousVersion

/** Versioned Phase-10 Metal decode/paged planning identities. */
namespace LlmMetalDecodePagedVersion {
inline constexpr const char* EXECUTOR =
    "llm-metal-executor-v1-decode-paged";
inline constexpr const char* SCHEDULE =
    "llm-metal-decode-paged-block-owner-grid-stride-v1";
inline constexpr const char* TIMER =
    "metal-command-buffer-gpu-start-end-v1";
inline constexpr const char* BUFFER_PATTERN =
    "llm-paged-physical-buffer-pattern-v1";
inline constexpr const char* WRITE_PATTERN =
    "llm-metal-decode-paged-append-affine32-v1";
inline constexpr const char* CHECKSUM =
    "llm-metal-paged-dual-mod32-lookup-mix-v1";
}  // namespace LlmMetalDecodePagedVersion

/** Versioned Phase-11 Metal prefill/contiguous planning identities. */
namespace LlmMetalPrefillContiguousVersion {
inline constexpr const char* EXECUTOR =
    "llm-metal-executor-v1-prefill-contiguous";
inline constexpr const char* SCHEDULE =
    "llm-metal-prefill-contiguous-lane-local-vector-stripe-v1";
inline constexpr const char* TIMER =
    "metal-command-buffer-gpu-start-end-v1";
inline constexpr const char* BUFFER_PATTERN =
    "llm-metal-contiguous-affine32-v1";
inline constexpr const char* WRITE_PATTERN =
    "llm-metal-prefill-contiguous-full-prompt-affine32-v1";
inline constexpr const char* CHECKSUM = "llm-metal-dual-mod32-v1";
}  // namespace LlmMetalPrefillContiguousVersion

/** Versioned Phase-12 Metal prefill/paged planning identities. */
namespace LlmMetalPrefillPagedVersion {
inline constexpr const char* EXECUTOR =
    "llm-metal-executor-v1-prefill-paged";
inline constexpr const char* SCHEDULE =
    "llm-metal-prefill-paged-cyclic-block-owner-grid-stride-v1";
inline constexpr const char* TIMER =
    "metal-command-buffer-gpu-start-end-v1";
inline constexpr const char* BUFFER_PATTERN =
    "llm-paged-physical-buffer-pattern-v1";
inline constexpr const char* WRITE_PATTERN =
    "llm-metal-prefill-paged-full-prompt-affine32-v1";
inline constexpr const char* CHECKSUM =
    "llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1";
}  // namespace LlmMetalPrefillPagedVersion

namespace LlmMetalPlannerAccounting {
inline constexpr size_t RUNTIME_IDENTITY_GROWTH_RESERVE_BYTES =
    64ULL * 1024ULL;
}  // namespace LlmMetalPlannerAccounting

/** Stable reasons emitted by the pure Metal resource and dispatch planner. */
namespace LlmMetalPlanReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* INVALID_GEOMETRY = "invalid-metal-geometry";
inline constexpr const char* PAGED_LAYOUT_REQUIRED =
    "paged-layout-plan-required";
inline constexpr const char* PAGED_LAYOUT_MISMATCH =
    "paged-layout-plan-mismatch";
inline constexpr const char* ARGUMENT_ENCODER_LENGTH_ZERO =
    "metal-argument-encoder-length-zero";
inline constexpr const char* ARGUMENT_ENCODER_ALIGNMENT_INVALID =
    "metal-argument-encoder-alignment-invalid";
inline constexpr const char* ARGUMENT_BUFFER_LAYOUT_INVALID =
    "metal-argument-buffer-layout-invalid";
inline constexpr const char* RESOURCE_LENGTH_OVERFLOW =
    "metal-resource-length-overflow";
inline constexpr const char* RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER =
    "metal-resource-length-exceeds-max-buffer";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW =
    "memory-budget-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED =
    "memory-budget-exceeded";
inline constexpr const char* PIPELINE_WIDTH_ZERO =
    "metal-pipeline-width-zero";
inline constexpr const char* PIPELINE_THREAD_LIMIT_INVALID =
    "metal-pipeline-thread-limit-invalid";
inline constexpr const char* OWNER_COST_COUNT_MISMATCH =
    "metal-owner-cost-count-mismatch";
inline constexpr const char* OWNER_COUNT_OVERFLOW =
    "metal-owner-count-overflow";
inline constexpr const char* OWNER_STRIDE_CAP_EXCEEDED =
    "owner-stride-cap-exceeded";
inline constexpr const char* VECTOR_ITERATION_CAP_EXCEEDED =
    "vector-iteration-cap-exceeded";
inline constexpr const char* SERIAL_RANGE_VISIT_CAP_EXCEEDED =
    "serial-range-visit-cap-exceeded";
inline constexpr const char* SERIAL_RANGE_VISIT_COUNT_OVERFLOW =
    "serial-range-visit-count-overflow";
inline constexpr const char* SEMANTIC_VISIT_CAP_EXCEEDED =
    "semantic-visit-cap-exceeded";
inline constexpr const char* WORK_UNITS_PER_DISPATCH_CAP_EXCEEDED =
    "work-units-per-dispatch-cap-exceeded";
inline constexpr const char* PLANNER_ALLOCATION_FAILED =
    "planner-allocation-failed";
}  // namespace LlmMetalPlanReason

/** Stable domains used to derive independent buffer and scenario seeds. */
enum class LlmSeedDomain : uint8_t {
  WeightBuffer = 0,
  KBuffer,
  VBuffer,
  WeightsOnlyScenario,
  KvOnlyScenario,
  MixedScenario,
};

/** Frozen assembly-facing layer descriptor ABI v1. */
struct alignas(16) LlmLayerDescriptor {
  const uint8_t* weight_ptr;
  uint64_t weight_bytes;
  uint64_t first_sequence_index;
  uint64_t sequence_count;
  uint64_t layer_index;
  uint64_t reserved_zero;
};

/** Frozen assembly-facing per-layer/per-batch KV descriptor ABI v1. */
struct alignas(16) LlmKvSequenceDescriptor {
  const uint8_t* k_visible_ptr;
  uint64_t k_visible_bytes;
  const uint8_t* v_visible_ptr;
  uint64_t v_visible_bytes;
  uint8_t* k_append_ptr;
  uint64_t k_append_bytes;
  uint8_t* v_append_ptr;
  uint64_t v_append_bytes;
  uint64_t batch_sequence_index;
  uint64_t append_record_byte_offset;
};

/** Frozen assembly-facing paged layer descriptor ABI v1. */
struct alignas(16) LlmPagedLayerDescriptor {
  const uint8_t* weight_ptr;
  uint64_t weight_bytes;
  uint64_t first_assignment_index;
  uint64_t assignment_count;
  uint64_t layer_index;
  uint64_t reserved_zero;
};

/** Frozen assembly-facing paged logical-block assignment descriptor ABI v1. */
struct alignas(16) LlmPagedKvAssignmentDescriptor {
  const uint32_t* block_table_row;
  uint8_t* k_layer_pool;
  uint8_t* v_layer_pool;
  uint64_t first_logical_block;
  uint64_t owned_block_count;
  uint64_t blocks_per_sequence;
  uint64_t block_bytes;
  uint64_t last_block_valid_bytes;
  uint64_t decode_append_offset;
  uint64_t append_record_bytes;
  uint64_t layer_index;
  uint64_t batch_sequence_index;
};

/** Frozen assembly-facing contiguous-prefill layer descriptor ABI v1. */
struct alignas(16) LlmPrefillLayerDescriptor {
  const uint8_t* weight_ptr;
  uint64_t weight_bytes;
  uint64_t first_sequence_index;
  uint64_t sequence_count;
  uint64_t layer_index;
  uint64_t reserved_zero;
};

/** Frozen assembly-facing contiguous-prefill owner descriptor ABI v1. */
struct alignas(16) LlmPrefillKvSequenceDescriptor {
  uint8_t* k_owned_ptr;
  uint8_t* v_owned_ptr;
  uint64_t first_token;
  uint64_t owned_token_count;
  uint64_t prompt_tokens;
  uint64_t attention_query_tile_tokens;
  uint64_t record_bytes;
  uint64_t layer_index;
  uint64_t batch_sequence_index;
  uint64_t reserved_zero;
};

/** Frozen assembly-facing paged-prefill layer descriptor ABI v1. */
struct alignas(16) LlmPagedPrefillLayerDescriptor {
  const uint8_t* weight_ptr;
  uint64_t weight_bytes;
  uint64_t first_assignment_index;
  uint64_t assignment_count;
  uint64_t layer_index;
  uint64_t reserved_zero;
};

/** Frozen assembly-facing paged-prefill block-owner descriptor ABI v1. */
struct alignas(16) LlmPagedPrefillKvAssignmentDescriptor {
  const uint32_t* block_table_row;
  uint8_t* k_layer_pool;
  uint8_t* v_layer_pool;
  uint64_t first_logical_block;
  uint64_t owned_block_count;
  uint64_t blocks_per_sequence;
  uint64_t block_tokens;
  uint64_t block_bytes;
  uint64_t last_block_valid_bytes;
  uint64_t prompt_tokens;
  uint64_t attention_query_tile_tokens;
  uint64_t record_bytes;
  uint64_t layer_index;
  uint64_t batch_sequence_index;
};

static_assert(sizeof(void*) == 8,
              "LLM descriptor ABI requires 64-bit ARM64 pointers");
static_assert(std::is_standard_layout_v<LlmLayerDescriptor>);
static_assert(std::is_standard_layout_v<LlmKvSequenceDescriptor>);
static_assert(alignof(LlmLayerDescriptor) == 16);
static_assert(sizeof(LlmLayerDescriptor) == 48);
static_assert(offsetof(LlmLayerDescriptor, weight_ptr) == 0);
static_assert(offsetof(LlmLayerDescriptor, weight_bytes) == 8);
static_assert(offsetof(LlmLayerDescriptor, first_sequence_index) == 16);
static_assert(offsetof(LlmLayerDescriptor, sequence_count) == 24);
static_assert(offsetof(LlmLayerDescriptor, layer_index) == 32);
static_assert(offsetof(LlmLayerDescriptor, reserved_zero) == 40);
static_assert(alignof(LlmKvSequenceDescriptor) == 16);
static_assert(sizeof(LlmKvSequenceDescriptor) == 80);
static_assert(offsetof(LlmKvSequenceDescriptor, k_visible_ptr) == 0);
static_assert(offsetof(LlmKvSequenceDescriptor, k_visible_bytes) == 8);
static_assert(offsetof(LlmKvSequenceDescriptor, v_visible_ptr) == 16);
static_assert(offsetof(LlmKvSequenceDescriptor, v_visible_bytes) == 24);
static_assert(offsetof(LlmKvSequenceDescriptor, k_append_ptr) == 32);
static_assert(offsetof(LlmKvSequenceDescriptor, k_append_bytes) == 40);
static_assert(offsetof(LlmKvSequenceDescriptor, v_append_ptr) == 48);
static_assert(offsetof(LlmKvSequenceDescriptor, v_append_bytes) == 56);
static_assert(offsetof(LlmKvSequenceDescriptor, batch_sequence_index) == 64);
static_assert(
    offsetof(LlmKvSequenceDescriptor, append_record_byte_offset) == 72);
static_assert(std::is_standard_layout_v<LlmPagedLayerDescriptor>);
static_assert(alignof(LlmPagedLayerDescriptor) == 16);
static_assert(sizeof(LlmPagedLayerDescriptor) == 48);
static_assert(offsetof(LlmPagedLayerDescriptor, weight_ptr) == 0);
static_assert(offsetof(LlmPagedLayerDescriptor, weight_bytes) == 8);
static_assert(offsetof(LlmPagedLayerDescriptor, first_assignment_index) == 16);
static_assert(offsetof(LlmPagedLayerDescriptor, assignment_count) == 24);
static_assert(offsetof(LlmPagedLayerDescriptor, layer_index) == 32);
static_assert(offsetof(LlmPagedLayerDescriptor, reserved_zero) == 40);
static_assert(std::is_standard_layout_v<LlmPagedKvAssignmentDescriptor>);
static_assert(alignof(LlmPagedKvAssignmentDescriptor) == 16);
static_assert(sizeof(LlmPagedKvAssignmentDescriptor) == 96);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, block_table_row) == 0);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, k_layer_pool) == 8);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, v_layer_pool) == 16);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, first_logical_block) == 24);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, owned_block_count) == 32);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, blocks_per_sequence) == 40);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, block_bytes) == 48);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, last_block_valid_bytes) == 56);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, decode_append_offset) == 64);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, append_record_bytes) == 72);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, layer_index) == 80);
static_assert(offsetof(LlmPagedKvAssignmentDescriptor, batch_sequence_index) == 88);
static_assert(std::is_standard_layout_v<LlmPrefillLayerDescriptor>);
static_assert(alignof(LlmPrefillLayerDescriptor) == 16);
static_assert(sizeof(LlmPrefillLayerDescriptor) == 48);
static_assert(offsetof(LlmPrefillLayerDescriptor, weight_ptr) == 0);
static_assert(offsetof(LlmPrefillLayerDescriptor, weight_bytes) == 8);
static_assert(offsetof(LlmPrefillLayerDescriptor, first_sequence_index) == 16);
static_assert(offsetof(LlmPrefillLayerDescriptor, sequence_count) == 24);
static_assert(offsetof(LlmPrefillLayerDescriptor, layer_index) == 32);
static_assert(offsetof(LlmPrefillLayerDescriptor, reserved_zero) == 40);
static_assert(std::is_standard_layout_v<LlmPrefillKvSequenceDescriptor>);
static_assert(alignof(LlmPrefillKvSequenceDescriptor) == 16);
static_assert(sizeof(LlmPrefillKvSequenceDescriptor) == 80);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, k_owned_ptr) == 0);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, v_owned_ptr) == 8);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, first_token) == 16);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, owned_token_count) == 24);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, prompt_tokens) == 32);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, attention_query_tile_tokens) == 40);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, record_bytes) == 48);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, layer_index) == 56);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, batch_sequence_index) == 64);
static_assert(offsetof(LlmPrefillKvSequenceDescriptor, reserved_zero) == 72);
static_assert(std::is_standard_layout_v<LlmPagedPrefillLayerDescriptor>);
static_assert(alignof(LlmPagedPrefillLayerDescriptor) == 16);
static_assert(sizeof(LlmPagedPrefillLayerDescriptor) == 48);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, weight_ptr) == 0);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, weight_bytes) == 8);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, first_assignment_index) == 16);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, assignment_count) == 24);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, layer_index) == 32);
static_assert(offsetof(LlmPagedPrefillLayerDescriptor, reserved_zero) == 40);
static_assert(std::is_standard_layout_v<LlmPagedPrefillKvAssignmentDescriptor>);
static_assert(alignof(LlmPagedPrefillKvAssignmentDescriptor) == 16);
static_assert(sizeof(LlmPagedPrefillKvAssignmentDescriptor) == 112);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, block_table_row) == 0);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, k_layer_pool) == 8);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, v_layer_pool) == 16);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, first_logical_block) == 24);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, owned_block_count) == 32);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, blocks_per_sequence) == 40);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, block_tokens) == 48);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, block_bytes) == 56);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, last_block_valid_bytes) == 64);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, prompt_tokens) == 72);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                       attention_query_tile_tokens) == 80);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, record_bytes) == 88);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, layer_index) == 96);
static_assert(offsetof(LlmPagedPrefillKvAssignmentDescriptor, batch_sequence_index) == 104);

/** Raw exact-byte model inputs, independent of CLI unit conversion. */
struct LlmGeometryRequest {
  size_t active_weight_bytes = 0;
  size_t layer_count = 0;
  size_t query_head_count = 0;
  size_t kv_head_count = 0;
  size_t head_dimension = 0;
  size_t kv_element_bytes = Constants::LLM_DEFAULT_KV_ELEMENT_BYTES;
  size_t visible_context_tokens = 0;
  size_t batch_size = Constants::LLM_DEFAULT_BATCH_SIZE;
  size_t kv_block_tokens = 0;
  LlmPhase phase = LlmPhase::Decode;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  size_t prompt_tokens = 0;
  size_t attention_query_tile_tokens = 0;
};

/** Decode-only geometry; present exactly when `phase == decode`. */
struct LlmDecodeGeometry {
  size_t visible_context_tokens = 0;
};

/** Prefill-only checked geometry for one full-prompt operation. */
struct LlmPrefillGeometry {
  size_t prompt_tokens = 0;
  size_t attention_query_tile_tokens = 0;
  size_t tile_count = 0;
  size_t attention_prefix_token_visits_per_sequence = 0;
  size_t causal_token_pairs_per_sequence = 0;
  size_t logical_attention_pairs = 0;
  size_t logical_attention_fma_terms = 0;
  size_t paged_prefix_block_visits_per_sequence = 0;
};

/** Checked common and phase-specific geometry for one synthetic work unit. */
struct LlmGeometry {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO;
  LlmPhase phase = LlmPhase::Decode;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  std::optional<LlmDecodeGeometry> decode;
  std::optional<LlmPrefillGeometry> prefill;
  LlmAttentionKind attention_kind = LlmAttentionKind::Mha;
  size_t active_weight_bytes_per_work_unit = 0;
  size_t layer_count = 0;
  size_t query_head_count = 0;
  size_t kv_head_count = 0;
  size_t query_heads_per_kv_head = 0;
  size_t head_dimension = 0;
  size_t kv_element_bytes = 0;
  size_t batch_size = 0;
  size_t kv_vector_bytes = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
  size_t kv_record_bytes_per_layer = 0;
  size_t kv_bytes_per_visible_token = 0;
  size_t k_or_v_sequence_visible_bytes = 0;
  size_t kv_block_tokens = 0;
  size_t kv_blocks_per_sequence = 0;
  size_t physical_blocks_per_layer = 0;
  size_t total_physical_blocks = 0;
  size_t kv_block_bytes = 0;
  size_t last_block_tokens = 0;
  size_t last_block_valid_bytes = 0;
  size_t decode_append_offset_in_last_block = 0;
  size_t k_logical_bytes = 0;
  size_t v_logical_bytes = 0;
  size_t k_layout_padding_bytes = 0;
  size_t v_layout_padding_bytes = 0;
  size_t block_table_entries = 0;
  size_t block_table_bytes = 0;
  size_t layout_metadata_lookups_per_layer_sequence_per_work_unit = 0;
  size_t k_mapping_bytes = 0;
  size_t v_mapping_bytes = 0;
  size_t kv_capacity_bytes = 0;
  size_t weight_read_bytes_per_work_unit = 0;
  size_t kv_read_bytes_per_work_unit = 0;
  size_t kv_write_bytes_per_work_unit = 0;
  size_t kv_only_effective_model_payload_bytes_per_work_unit = 0;
  size_t mixed_effective_model_payload_bytes_per_work_unit = 0;
  size_t total_data_mapping_bytes = 0;
  size_t traffic_crossover_numerator = 0;
  size_t traffic_crossover_denominator = 0;
  double traffic_crossover_context_tokens = 0.0;
};

/**
 * Exact mapping and auxiliary memory request before budget evaluation.
 * `planner_storage_bytes` records retained vector backing capacities; checksum
 * and orchestration bytes are explicit caller-owned peak estimates.
 */
struct LlmMemoryBudgetRequest {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
  size_t mapping_granularity_bytes = 0;
  size_t requested_weight_mapping_bytes = 0;
  size_t requested_k_mapping_bytes = 0;
  size_t requested_v_mapping_bytes = 0;
  size_t committed_weight_mapping_bytes = 0;
  size_t committed_k_mapping_bytes = 0;
  size_t committed_v_mapping_bytes = 0;
  size_t requested_block_table_mapping_bytes = 0;
  size_t committed_block_table_mapping_bytes = 0;
  size_t requested_data_bytes = 0;
  size_t committed_data_bytes = 0;
  size_t layout_transient_bytes = 0;
  size_t setup_peak_bytes = 0;
  size_t runtime_peak_bytes = 0;
  size_t descriptor_bytes = 0;
  size_t planner_storage_bytes = 0;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  size_t auxiliary_bytes = 0;
  size_t required_total_bytes = 0;
};

/** Deterministic available-memory policy evidence. */
struct LlmMemoryBudget {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
  LlmMemoryBudgetRequest request;
  size_t available_memory_bytes = 0;
  size_t allowed_memory_bytes = 0;
  bool used_fallback = false;
};

/** Offset and exact span within one command-owned mapping. */
struct LlmByteRange {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
};

/** Pointer-free template corresponding to one layer descriptor. */
struct LlmLayerRangeTemplate {
  LlmByteRange weight;
  size_t first_sequence_index = 0;
  size_t sequence_count = 0;
  size_t layer_index = 0;
};

/** Pointer-free template corresponding to one KV sequence descriptor. */
struct LlmKvSequenceRangeTemplate {
  LlmByteRange k_visible;
  LlmByteRange v_visible;
  LlmByteRange k_append;
  LlmByteRange v_append;
  size_t layer_index = 0;
  size_t batch_sequence_index = 0;
  size_t append_record_byte_offset = 0;
};

/** Pointer-free paged logical-block assignment for one layer and batch row. */
struct LlmPagedKvAssignmentTemplate {
  size_t layer_index = 0;
  size_t batch_sequence_index = 0;
  size_t first_logical_block = 0;
  size_t block_count = 0;
};

/** Pointer-free contiguous-prefill token ownership for one layer/batch row. */
struct LlmPrefillKvSequenceRangeTemplate {
  LlmByteRange k_owned;
  LlmByteRange v_owned;
  size_t first_token = 0;
  size_t owned_token_count = 0;
  size_t layer_index = 0;
  size_t batch_sequence_index = 0;
};

/** Pointer-free paged-prefill whole-block owner for one layer/batch row. */
struct LlmPagedPrefillKvAssignmentTemplate {
  size_t layer_index = 0;
  size_t batch_sequence_index = 0;
  size_t first_logical_block = 0;
  size_t block_count = 0;
};

/** Complete immutable pointer-free descriptor set for one worker. */
struct LlmWorkerWorkPlan {
  size_t worker_index = 0;
  std::vector<LlmLayerRangeTemplate> layers;
  std::vector<LlmKvSequenceRangeTemplate> sequences;
  std::vector<LlmPagedKvAssignmentTemplate> paged_assignments;
  /** Three scenario-major sets, each containing layer_count * batch_size rows. */
  std::vector<LlmPrefillKvSequenceRangeTemplate> prefill_sequences;
  /** Three scenario-major paged-prefill block-owner descriptor sets. */
  std::vector<LlmPagedPrefillKvAssignmentTemplate> paged_prefill_assignments;
};

/** Aggregate evidence for one scenario's owner-local prefill map. */
struct LlmPrefillCpuScenarioExecutionPlan {
  LlmScenario scenario = LlmScenario::WeightsOnly;
  std::vector<LlmPrefillCpuOwnershipPlan> ownership_scopes;
  std::vector<size_t> worker_accounted_bytes_per_work_unit;
  size_t minimum_worker_accounted_bytes_per_work_unit = 0;
  size_t maximum_worker_accounted_bytes_per_work_unit = 0;
  size_t worker_accounted_imbalance_bytes_per_work_unit = 0;
  std::string identity;
};

/** Executable prefill scenario partitions and descriptor geometry. */
struct LlmPrefillCpuExecutionPlan {
  size_t sequence_descriptors_per_scenario_per_worker = 0;
  std::array<LlmPrefillCpuScenarioExecutionPlan, kLlmScenarioCount> scenarios;
  std::string identity;
};

/**
 * Command-owned immutable paged layout, table, and execution evidence.
 *
 * `ownership` is populated for decode. Paged prefill retains block-exclusive
 * ownership in `LlmPrefillCpuExecutionPlan`, and `execution_identity` repeats
 * that prefill execution identity. The enclosing model-plan identity binds it
 * beside `layout_identity`, which in turn binds layout and permutation.
 */
struct LlmPagedCpuExecutionPlan {
  LlmKvLayoutPlan layout;
  LlmKvBlockTableValidation table_validation;
  LlmKvPermutationIdentity permutation;
  LlmKvCpuOwnershipPlan ownership;
  MmapPtr block_table_mapping{nullptr, MmapDeleter{0}};
  size_t block_table_logical_bytes = 0;
  size_t block_table_mapping_bytes = 0;
  bool block_table_read_only = false;
  std::string layout_identity;
  std::string execution_identity;

  const uint32_t* block_table() const noexcept {
    return static_cast<const uint32_t*>(block_table_mapping.get());
  }
};

/**
 * CPU-only worker partition and descriptor planning evidence.
 *
 * Prefill retains scenario-major ownership in `prefill`. Paged prefill also
 * retains `paged`, whose immutable block table is shared by all scenarios.
 */
struct LlmCpuExecutionPlan {
  size_t requested_workers = 0;
  size_t available_workers = 0;
  size_t effective_workers = 0;
  size_t layer_descriptors_per_worker = 0;
  size_t sequence_descriptors_per_worker = 0;
  size_t total_layer_descriptors = 0;
  size_t total_sequence_descriptors = 0;
  size_t descriptor_bytes = 0;
  size_t planner_storage_bytes = 0;
  std::vector<LlmWorkerWorkPlan> workers;
  std::optional<LlmPagedCpuExecutionPlan> paged;
  std::optional<LlmPrefillCpuExecutionPlan> prefill;
};

/** Injected caps for deterministic Metal segmentation and dispatch planning. */
struct LlmMetalPlanningLimits {
  size_t segment_capacity_bytes =
      Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  size_t segment_slots_per_pool =
      Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL;
  size_t threads_per_threadgroup_cap =
      Constants::LLM_METAL_THREADS_PER_THREADGROUP_CAP;
  size_t maximum_threadgroups_per_grid =
      Constants::LLM_METAL_MAX_THREADGROUPS_PER_GRID;
  size_t maximum_owner_ordinals_per_threadgroup =
      Constants::LLM_METAL_MAX_OWNER_ORDINALS_PER_THREADGROUP;
  size_t maximum_vector_iterations_per_lane_per_visit =
      Constants::LLM_METAL_MAX_VECTOR_ITERATIONS_PER_LANE_PER_VISIT;
  size_t maximum_serial_range_visits_per_lane_per_task =
      Constants::LLM_METAL_MAX_SERIAL_RANGE_VISITS_PER_LANE_PER_TASK;
  size_t maximum_paged_semantic_lookups_per_task =
      Constants::LLM_METAL_MAX_PAGED_SEMANTIC_LOOKUPS_PER_TASK;
  size_t maximum_work_units_per_dispatch =
      Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH;
};

/** Canonical Tier-2 resource-table slots and exact active resource counts. */
struct LlmMetalArgumentBufferPlan {
  bool valid = false;
  std::string reason_code =
      LlmMetalPlanReason::ARGUMENT_BUFFER_LAYOUT_INVALID;
  size_t weight_slot_base = 0;
  size_t k_slot_base = Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL;
  size_t v_slot_base = 2 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL;
  size_t table_slot_base = 3 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL;
  size_t status_slot = 4 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL;
  size_t encoded_resource_slot_count =
      4 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL + 1;
  size_t weight_segment_count = 0;
  size_t k_segment_count = 0;
  size_t v_segment_count = 0;
  size_t table_segment_count = 0;
  size_t active_resource_count = 0;
  std::string identity;
};

/** Runtime pipeline limits injected into the pure grid resolver. */
struct LlmMetalPipelineCapabilities {
  size_t thread_execution_width = 0;
  size_t max_total_threads_per_threadgroup = 0;
};

/** Exact bounded inputs for one Metal workload dispatch. */
struct LlmMetalGridRequest {
  size_t owner_count = 0;
  size_t visit_bytes = 0;
  size_t work_units = 0;
  size_t paged_semantic_lookups = 0;
  size_t serial_range_visits_per_lane = 0;
  std::vector<size_t> owner_accounted_bytes;
  LlmMetalPipelineCapabilities pipeline;
  LlmMetalPlanningLimits limits;
};

/** Capped grid-stride geometry and optional cyclic owner-cost evidence. */
struct LlmMetalGridPlan {
  bool valid = false;
  std::string reason_code = LlmMetalPlanReason::PIPELINE_WIDTH_ZERO;
  size_t owner_count = 0;
  size_t threads_per_threadgroup = 0;
  size_t actual_threadgroups = 0;
  size_t owner_ordinals_per_threadgroup = 0;
  size_t vector_iterations_per_lane_per_visit = 0;
  size_t work_units = 0;
  size_t paged_semantic_lookups = 0;
  size_t serial_range_visits_per_lane = 0;
  size_t minimum_threadgroup_accounted_bytes = 0;
  size_t maximum_threadgroup_accounted_bytes = 0;
  size_t threadgroup_accounted_imbalance_bytes = 0;
  std::vector<size_t> threadgroup_accounted_bytes;
  std::string identity;
};

/** One exact resource requested by the Metal candidate allocator. */
enum class LlmMetalResourcePool : uint8_t {
  Weight = 0,
  K,
  V,
  BlockTable,
  ArgumentBuffer,
  Status,
  Staging,
};

/** Pure allocation descriptor; storage semantics are fixed by the pool. */
struct LlmMetalPlannedResource {
  LlmMetalResourcePool pool = LlmMetalResourcePool::Weight;
  size_t pool_index = 0;
  size_t length_bytes = 0;
  bool persistent = true;
};

/** Exact pre-allocation resource lengths and host/transient peak evidence. */
struct LlmMetalResourcePlan {
  bool valid = false;
  std::string reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
  LlmMetalPlanningLimits limits;
  LlmKvSegmentPlan weight_segments;
  LlmKvSegmentPlan k_segments;
  LlmKvSegmentPlan v_segments;
  std::optional<LlmKvSegmentPlan> table_segments;
  std::optional<LlmKvLayoutPlan> paged_layout;
  LlmMetalArgumentBufferPlan argument_buffer;
  size_t argument_buffer_encoded_length = 0;
  size_t argument_buffer_alignment = 0;
  size_t max_buffer_length = 0;
  size_t host_mapping_granularity_bytes = 1;
  size_t status_buffer_length = Constants::LLM_METAL_STATUS_BUFFER_BYTES;
  size_t staging_buffer_length = 0;
  size_t host_permutation_mapping_bytes = 0;
  size_t permutation_validation_bitset_bytes = 0;
  size_t additional_owned_bytes = 0;
  size_t persistent_resource_length_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t known_owned_peak_bytes = 0;
  size_t available_memory_bytes = 0;
  size_t admitted_budget_bytes = 0;
  std::vector<LlmMetalPlannedResource> planned_resources;
  std::string identity;
};

/** Metal resource and execution plan retained by the selected backend. */
struct LlmMetalExecutionPlan {
  bool valid = false;
  std::string reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
  LlmMetalResourcePlan resources;
  std::string msl_revision;
  std::string msl_source_sha256;
  std::string identity;
};

/** Exactly one backend-specific execution plan for a logical model plan. */
using LlmBackendExecutionPlan =
    std::variant<LlmCpuExecutionPlan, LlmMetalExecutionPlan>;

/**
 * Inputs for checked layout, descriptor, and memory-budget planning.
 * The caller supplies the page granularity, current available-memory sample,
 * and all caller-supplied checksum/orchestration bytes that will coexist with
 * the three mappings, ABI arrays, and returned pointer-free plan.
 */
struct LlmMemoryWorkPlanRequest {
  LlmGeometryRequest geometry;
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  size_t requested_workers = 0;
  size_t available_workers = 0;
  size_t available_memory_bytes = 0;
  size_t mapping_granularity_bytes = 1;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  uint64_t base_seed = 0;
};

/** Versioned methodology components with explicit applicability. */
struct LlmComponentIdentities {
  std::string logical_profile_version;
  std::string kv_layout_version;
  std::optional<std::string> permutation_version;
  std::string backend_executor_version;
  std::string resource_abi_version;
  std::string schedule_version;
  std::string timer_policy_version;
  std::string buffer_pattern_version;
  std::string write_pattern_version;
  std::string checksum_pattern_version;
  std::optional<std::string> msl_revision;
  std::optional<std::string> msl_source_sha256;
  std::string identity;
};

/**
 * Logical model plan with exactly one tagged backend execution plan.
 * It is move-only because copying retained CPU vectors would invalidate the
 * recorded capacity-based peak budget. Consumers must keep a finalized plan
 * const: changing execution-bound workload evidence or templates requires a
 * rebuilt identity. Admission and allocator-capacity evidence remains
 * separately recorded and is intentionally excluded from workload identity.
 */
struct LlmMemoryWorkPlan {
  LlmMemoryWorkPlan() = default;
  LlmMemoryWorkPlan(const LlmMemoryWorkPlan&) = delete;
  LlmMemoryWorkPlan& operator=(const LlmMemoryWorkPlan&) = delete;
  LlmMemoryWorkPlan(LlmMemoryWorkPlan&& other) noexcept;
  LlmMemoryWorkPlan& operator=(LlmMemoryWorkPlan&& other) noexcept;

  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO;
  LlmGeometry geometry;
  /** Pure logical formula/cost evidence; never an executable prefill plan. */
  std::optional<LlmPrefillPlan> prefill_plan;
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  LlmPhase phase = LlmPhase::Decode;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  size_t weight_passes_per_work_unit = Constants::LLM_WEIGHT_PASSES_PER_WORK_UNIT;
  size_t kv_replay_factor = Constants::LLM_KV_REPLAY_FACTOR;
  uint64_t base_seed = 0;
  uint64_t weight_buffer_seed = 0;
  uint64_t k_buffer_seed = 0;
  uint64_t v_buffer_seed = 0;
  std::array<uint64_t, kLlmScenarioCount> scenario_seeds{};
  LlmMemoryBudget memory_budget;
  std::vector<LlmByteRange> weight_layers;
  std::string methodology_version;
  LlmComponentIdentities component_identities;
  std::string plan_identity;
  LlmBackendExecutionPlan backend_execution_plan;
};

/** Exact allocation and identity-size inputs available before a paged table. */
struct LlmAuxiliaryPreflightView {
  bool valid = false;
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  size_t effective_workers = 0;
  size_t total_layer_descriptors = 0;
  size_t total_sequence_descriptors = 0;
  size_t k_or_v_static_reference_count = 0;
  size_t metal_planned_resource_count = 0;
  size_t metal_persistent_resource_count = 0;
  size_t metal_resolved_execution_plan_backing_bytes = 0;
  size_t metal_resolved_plan_identity_backing_bytes = 0;
  size_t model_plan_identity_bytes = 0;
  std::array<size_t, kLlmScenarioCount>
      maximum_scenario_plan_identity_bytes{};
  size_t frozen_reason_code_bytes = 0;
  size_t frozen_model_plan_identity_bytes = 0;
  size_t frozen_plan_identity_bytes = 0;
  std::array<size_t, kLlmScenarioCount>
      frozen_scenario_reason_code_bytes{};
  std::array<size_t, kLlmScenarioCount>
      frozen_scenario_model_plan_identity_bytes{};
  std::array<size_t, kLlmScenarioCount>
      frozen_scenario_plan_identity_bytes{};
  size_t json_identity_string_bytes = 0;
};

/**
 * Move-only non-executable plan prepared for exact auxiliary admission.
 *
 * `candidate.valid` is always false. The retained pointer-free templates and
 * fixed-size placeholder identities may only be consumed by
 * `finalize_llm_memory_work_plan`; the preflight view is the sole estimator
 * input. No paged table mapping exists while this draft is published.
 */
struct LlmMemoryWorkPlanDraft {
  LlmMemoryWorkPlanDraft() = default;
  LlmMemoryWorkPlanDraft(const LlmMemoryWorkPlanDraft&) = delete;
  LlmMemoryWorkPlanDraft& operator=(const LlmMemoryWorkPlanDraft&) = delete;
  LlmMemoryWorkPlanDraft(LlmMemoryWorkPlanDraft&&) noexcept = default;
  LlmMemoryWorkPlanDraft& operator=(LlmMemoryWorkPlanDraft&&) noexcept =
      default;

  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
  LlmAuxiliaryPreflightView auxiliary_preflight;
  LlmMemoryWorkPlan candidate;
};

/**
 * Return the CPU execution plan only when the declared backend and variant
 * alternative both identify CPU. A backend/variant mismatch returns null.
 */
const LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) noexcept;

/** Non-const overload of the checked CPU execution-plan accessor. */
LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    LlmMemoryWorkPlan& plan) noexcept;

/**
 * Return the Metal execution plan only when the declared backend and variant
 * alternative both identify Metal. A mismatch returns null.
 */
const LlmMetalExecutionPlan* get_llm_metal_execution_plan(
    const LlmMemoryWorkPlan& plan) noexcept;

/** Non-const overload of the checked Metal execution-plan accessor. */
LlmMetalExecutionPlan* get_llm_metal_execution_plan(
    LlmMemoryWorkPlan& plan) noexcept;

/**
 * Stream-validate canonical contiguous or paged prefill ownership and
 * identities.
 *
 * This cold-path validation performs no dynamic allocation and rejects any
 * divergence between scenario scopes, worker descriptors, exact worker costs,
 * aggregate identities, and the model identity before resources are mapped.
 */
bool validate_llm_prefill_cpu_execution_evidence(
    const LlmMemoryWorkPlan& plan) noexcept;

/** Per-scenario payload, layout metadata, and effective work-unit ceiling. */
struct LlmScenarioLimits {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  LlmKvWriteKind kv_write_kind = LlmKvWriteKind::None;
  size_t weight_read_bytes_per_work_unit = 0;
  size_t kv_read_bytes_per_work_unit = 0;
  size_t kv_write_bytes_per_work_unit = 0;
  size_t effective_model_payload_bytes_per_work_unit = 0;
  size_t layout_metadata_lookup_count_per_work_unit = 0;
  size_t layout_metadata_read_bytes_per_work_unit = 0;
  size_t accounted_bytes_per_work_unit = 0;
  size_t maximum_work_units_by_work_unit_cap =
      Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT;
  size_t maximum_work_units_by_guardrail = 0;
  size_t effective_maximum_work_units = 0;
};

/** Fully resolved exact work for one excluded or measured scenario task. */
struct LlmScenarioWorkPlan {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  LlmKvWriteKind kv_write_kind = LlmKvWriteKind::None;
  bool explicit_iterations = false;
  std::string model_plan_identity;
  uint64_t scenario_seed = 0;
  size_t work_units = 0;
  size_t weight_read_bytes_per_work_unit = 0;
  size_t kv_read_bytes_per_work_unit = 0;
  size_t kv_write_bytes_per_work_unit = 0;
  size_t effective_model_payload_bytes_per_work_unit = 0;
  size_t layout_metadata_lookup_count_per_work_unit = 0;
  size_t layout_metadata_read_bytes_per_work_unit = 0;
  size_t accounted_bytes_per_work_unit = 0;
  size_t weight_read_bytes = 0;
  size_t kv_read_bytes = 0;
  size_t kv_write_bytes = 0;
  size_t effective_model_payload_bytes = 0;
  size_t layout_metadata_lookup_count = 0;
  size_t layout_metadata_read_bytes = 0;
  size_t task_accounted_bytes = 0;
  size_t maximum_work_units_by_work_unit_cap = 0;
  size_t maximum_work_units_by_guardrail = 0;
  size_t effective_maximum_work_units = 0;
  std::string plan_identity;
};

/** Three scenario plans frozen together before loop zero. */
struct LlmFrozenScenarioPlans {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::WORK_UNIT_COUNT_ZERO;
  bool explicit_iterations = false;
  std::string model_plan_identity;
  std::array<LlmScenarioWorkPlan, kLlmScenarioCount> scenarios;
  std::string plan_identity;
};

/** Resolve all geometry formulas without allocating descriptor arrays. */
LlmGeometry resolve_llm_geometry(const LlmGeometryRequest& request);

/**
 * Build a checked three-mapping peak-memory request.
 *
 * The three mappings are rounded separately. Auxiliary inputs must describe
 * simultaneous allocations; no category is inferred or silently reduced.
 * @return An invalid request with a stable reason on any overflow.
 */
LlmMemoryBudgetRequest build_llm_memory_budget_request(
    const LlmGeometry& geometry, size_t descriptor_bytes,
    size_t planner_storage_bytes, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    size_t mapping_granularity_bytes,
    size_t block_table_mapping_bytes = 0,
    size_t layout_transient_bytes = 0);

/** Apply the injected 80%-or-fallback project memory policy. */
LlmMemoryBudget evaluate_llm_memory_budget(
    const LlmMemoryBudgetRequest& request, size_t available_memory_bytes);

/** Build retained pointer-free templates without a paged table mapping. */
LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested = {});

/** Validate a config and prepare its non-executable auxiliary-sizing draft. */
LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Attach one runtime-dependent Metal planning outcome to a logical draft.
 *
 * This operation is used after capability initialization. A valid supplied
 * plan rebinds the component/model identities and refreshes the conservative
 * auxiliary preflight without publishing an executable plan. An invalid
 * supplied plan is retained as terminal runtime-failure evidence while the
 * logical draft remains finalizable for output. Callers must inspect the
 * retained alternative before attempting exact execution-plan finalization.
 * A valid supplied plan must describe the draft's activated Metal decode
 * geometry,
 * and its `additional_owned_bytes` must contain the draft's planner storage
 * but no command auxiliary bytes yet.
 */
bool attach_llm_metal_execution_plan(
    LlmMemoryWorkPlanDraft& draft,
    LlmMetalExecutionPlan&& metal_execution_plan) noexcept;

/**
 * Measure the execution-plan capacities retained by the Metal backend's
 * resolved copy and bound its identity string with the conservative 2x
 * policy. `model_plan_identity_bytes` may include a preflight growth reserve.
 */
bool calculate_llm_metal_resolved_plan_backing_bytes(
    const LlmMemoryWorkPlan& plan, size_t model_plan_identity_bytes,
    size_t& execution_plan_backing_bytes,
    size_t& plan_identity_backing_bytes) noexcept;

/**
 * Perform full auxiliary admission, then materialize/protect a paged table.
 *
 * The draft is consumed on every outcome. No table allocation occurs unless
 * the exact final peak, including both supplied auxiliary categories, fits.
 * Contiguous CPU prefill drafts retain their executable descriptor templates.
 * Paged decode and paged prefill drafts materialize the immutable logical-to-
 * physical block table used by their executable descriptor templates.
 */
LlmMemoryWorkPlan finalize_llm_memory_work_plan(
    LlmMemoryWorkPlanDraft&& draft, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Finalize a Metal draft with the exact runtime planning outcome rebuilt after
 * auxiliary sizing. A valid plan's `additional_owned_bytes` must equal the
 * checked sum of planner storage and the two supplied auxiliary categories.
 * The common memory-budget categories remain separate and are not added a
 * second time to the resulting required peak. An invalid outcome is retained
 * in a valid non-executable logical plan so an already initialized backend can
 * produce one terminal runtime-failure result.
 */
LlmMemoryWorkPlan finalize_llm_memory_work_plan(
    LlmMemoryWorkPlanDraft&& draft,
    LlmMetalExecutionPlan&& metal_execution_plan,
    size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Build the logical plan and its tagged backend planning alternative.
 *
 * The active CPU planner reduces effective workers until every standalone
 * scenario has work. Its retained vector capacities are measured after
 * allocation and the budget is re-evaluated before the plan becomes valid.
 * Invalid plans expose no executable templates. Metal decode/contiguous
 * produces a valid logical plan with an unresolved tagged execution
 * alternative; command orchestration must attach the capability-dependent
 * runtime plan before a ready backend can execute it. Both contiguous and
 * paged CPU prefill produce executable descriptor and ownership plans; paged
 * execution retains the immutable block-table identity in the work plan.
 *
 * @param request Fully resolved geometry, environment, and budget inputs.
 * @param stop_requested Optional synchronous predicate polled during paged
 *        block-table preparation.
 */
LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Validate a resolved config and build its pointer-free work plan.
 *
 * @param stop_requested Optional synchronous predicate polled while a paged
 *        block table is initialized, validated, and hashed.
 */
LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t checksum_auxiliary_bytes, size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested = {});

/** Re-evaluate one already-materialized plan with finalized auxiliary bytes. */
bool readmit_llm_memory_work_plan(
    LlmMemoryWorkPlan& plan, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes) noexcept;

/** Build `llm-memory-v1-<backend>-<phase>-<layout>`. */
std::string build_llm_methodology_version(LlmMemoryBackend backend,
                                          LlmPhase phase,
                                          LlmKvLayout layout);

/** Serialize methodology components in the schema-v1 fixed field order. */
std::string serialize_llm_component_identities(
    const LlmComponentIdentities& components);

/** Return the frozen 64-bit constant for a seed domain, or zero. */
uint64_t llm_seed_domain_value(LlmSeedDomain domain);

/**
 * Derive a buffer or scenario seed as SplitMix64(base XOR domain).
 * The function also returns zero for an unrecognized enum, but a valid domain
 * may theoretically derive zero; callers must not use the seed as a validity
 * sentinel.
 */
uint64_t derive_llm_domain_seed(uint64_t base_seed, LlmSeedDomain domain);

/**
 * Calculate exact model payload, accounted bytes, and hard ceilings for one
 * scenario. `work_unit_cap` narrows the common profile cap for a backend ABI.
 */
LlmScenarioLimits calculate_llm_scenario_limits(
    const LlmGeometry& geometry, LlmScenario scenario,
    size_t work_unit_cap = Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);

/** Resolve scenario limits including backend-specific work-unit caps. */
LlmScenarioLimits calculate_llm_scenario_limits(
    const LlmGeometry& geometry, LlmScenario scenario,
    LlmMemoryBackend backend);

/** Count one operation's serial range-helper visits in Metal prefill. */
bool calculate_llm_metal_prefill_serial_range_visits_per_work_unit(
    const LlmGeometry& geometry, LlmScenario scenario,
    size_t& visits) noexcept;

/**
 * Resolve exact component totals for one scenario task.
 * The returned identity and scenario seed are bound to the valid immutable
 * model plan; no mappings or worker state are touched.
 */
LlmScenarioWorkPlan build_llm_scenario_work_plan(
    const LlmMemoryWorkPlan& model_plan, LlmScenario scenario, size_t work_units,
    bool explicit_iterations);

/**
 * Freeze weights-only, KV-only, and mixed tasks in canonical order.
 * All identities bind the supplied model plan's executable geometry,
 * descriptor semantics, templates, and seeds. Admission/environment evidence
 * is separately recorded and does not change a frozen workload identity.
 */
LlmFrozenScenarioPlans freeze_llm_scenario_work_plans(
    const LlmMemoryWorkPlan& model_plan,
    const std::array<size_t, kLlmScenarioCount>& work_units,
    bool explicit_iterations);

/** Select the smallest 8 MiB-floor pilot count within scenario guardrails. */
size_t calculate_llm_pilot_work_units(const LlmScenarioLimits& limits);

/** Scale an excluded attempt toward 150 ms within scenario guardrails. */
size_t calculate_llm_calibrated_work_units(double attempt_duration_seconds,
                                           size_t attempt_work_units,
                                           const LlmScenarioLimits& limits);

/** Return true for a finite duration in the inclusive 100-250 ms window. */
bool llm_duration_in_target_window(double elapsed_seconds);

/** Return a static duration-quality token without mutating a frozen task plan. */
std::string_view classify_llm_duration_quality(
    double elapsed_seconds, size_t work_units,
    const LlmScenarioLimits& limits) noexcept;

/** Build one weights/KV/mixed cyclic rotation for a count-loop index. */
std::array<LlmScenario, kLlmScenarioCount> build_llm_scenario_order(
    size_t loop_index);

#endif  // LLM_WORK_PLAN_H
