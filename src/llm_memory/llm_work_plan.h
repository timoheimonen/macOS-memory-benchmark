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
 * @brief Overflow-safe geometry and immutable CPU LLM memory work planning
 */

#ifndef LLM_WORK_PLAN_H
#define LLM_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "llm_memory/llm_memory.h"

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
inline constexpr const char* KV_APPEND_BYTES_OVERFLOW =
    "kv-append-bytes-overflow";
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
inline constexpr const char* INVALID_SCENARIO = "invalid-scenario";
inline constexpr const char* INVALID_MODEL_WORK_PLAN =
    "invalid-model-work-plan";
inline constexpr const char* PAYLOAD_CAP_BELOW_ONE_STEP =
    "payload-cap-below-one-step";
inline constexpr const char* STEP_COUNT_ZERO = "step-count-zero";
inline constexpr const char* STEP_CAP_EXCEEDED = "step-cap-exceeded";
inline constexpr const char* EXACT_PAYLOAD_OVERFLOW =
    "exact-payload-overflow";
inline constexpr const char* EXACT_PAYLOAD_CAP_EXCEEDED =
    "exact-payload-cap-exceeded";
}  // namespace LlmWorkPlanReason

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
};

/** Checked geometry and exact logical bytes for one batched synthetic step. */
struct LlmGeometry {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO;
  LlmAttentionKind attention_kind = LlmAttentionKind::Mha;
  size_t active_weight_bytes_per_step = 0;
  size_t layer_count = 0;
  size_t query_head_count = 0;
  size_t kv_head_count = 0;
  size_t query_heads_per_kv_head = 0;
  size_t head_dimension = 0;
  size_t kv_element_bytes = 0;
  size_t visible_context_tokens = 0;
  size_t batch_size = 0;
  size_t kv_vector_bytes = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
  size_t kv_record_bytes_per_layer = 0;
  size_t kv_bytes_per_visible_token = 0;
  size_t k_or_v_sequence_visible_bytes = 0;
  size_t k_mapping_bytes = 0;
  size_t v_mapping_bytes = 0;
  size_t kv_capacity_bytes = 0;
  size_t weight_read_bytes_per_step = 0;
  size_t kv_read_bytes_per_step = 0;
  size_t kv_append_write_bytes_per_step = 0;
  size_t kv_only_effective_payload_bytes_per_step = 0;
  size_t mixed_effective_payload_bytes_per_step = 0;
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
  size_t requested_data_bytes = 0;
  size_t committed_data_bytes = 0;
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

/** Complete immutable pointer-free descriptor set for one worker. */
struct LlmWorkerWorkPlan {
  size_t worker_index = 0;
  std::vector<LlmLayerRangeTemplate> layers;
  std::vector<LlmKvSequenceRangeTemplate> sequences;
};

/**
 * Inputs for checked layout, descriptor, and memory-budget planning.
 * The caller supplies the page granularity, current available-memory sample,
 * and all executor-owned checksum/orchestration bytes that will coexist with
 * the three mappings, ABI arrays, and returned pointer-free plan.
 */
struct LlmMemoryWorkPlanRequest {
  LlmGeometryRequest geometry;
  size_t requested_workers = 0;
  size_t available_workers = 0;
  size_t available_memory_bytes = 0;
  size_t mapping_granularity_bytes = 1;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  uint64_t base_seed = 0;
};

/**
 * Valid pointer-free work plan used as the sole later descriptor source.
 * It is move-only because copying retained vectors would invalidate the
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
  size_t requested_workers = 0;
  size_t available_workers = 0;
  size_t effective_workers = 0;
  size_t layer_descriptors_per_worker = 0;
  size_t sequence_descriptors_per_worker = 0;
  size_t total_layer_descriptors = 0;
  size_t total_sequence_descriptors = 0;
  size_t descriptor_bytes = 0;
  size_t planner_storage_bytes = 0;
  uint64_t base_seed = 0;
  uint64_t weight_buffer_seed = 0;
  uint64_t k_buffer_seed = 0;
  uint64_t v_buffer_seed = 0;
  std::array<uint64_t, kLlmScenarioCount> scenario_seeds{};
  LlmMemoryBudget memory_budget;
  std::vector<LlmByteRange> weight_layers;
  std::vector<LlmWorkerWorkPlan> workers;
  std::string descriptor_abi_version;
  std::string backend;
  std::string phase;
  size_t weight_passes_per_step = Constants::LLM_WEIGHT_PASSES_PER_STEP;
  size_t kv_replay_factor = Constants::LLM_KV_REPLAY_FACTOR;
  std::string buffer_pattern_version;
  std::string methodology_version;
  std::string worker_schedule;
  std::string kv_layout;
  std::string plan_identity;
};

/** Per-scenario payload and effective step ceiling. */
struct LlmScenarioLimits {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  size_t weight_read_bytes_per_step = 0;
  size_t kv_read_bytes_per_step = 0;
  size_t kv_append_write_bytes_per_step = 0;
  size_t effective_payload_bytes_per_step = 0;
  size_t maximum_steps_by_step_cap =
      Constants::LLM_MAX_STEPS_PER_MEASUREMENT;
  size_t maximum_steps_by_payload_cap = 0;
  size_t effective_maximum_steps = 0;
};

/** Fully resolved exact work for one excluded or measured scenario task. */
struct LlmScenarioWorkPlan {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  bool explicit_iterations = false;
  std::string model_plan_identity;
  uint64_t scenario_seed = 0;
  size_t steps = 0;
  size_t weight_read_bytes_per_step = 0;
  size_t kv_read_bytes_per_step = 0;
  size_t kv_append_write_bytes_per_step = 0;
  size_t effective_payload_bytes_per_step = 0;
  size_t weight_read_bytes = 0;
  size_t kv_read_bytes = 0;
  size_t kv_append_write_bytes = 0;
  size_t effective_payload_bytes = 0;
  size_t maximum_steps_by_step_cap = 0;
  size_t maximum_steps_by_payload_cap = 0;
  size_t effective_maximum_steps = 0;
  std::string plan_identity;
};

/** Three scenario plans frozen together before loop zero. */
struct LlmFrozenScenarioPlans {
  bool valid = false;
  std::string reason_code = LlmWorkPlanReason::STEP_COUNT_ZERO;
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
    size_t mapping_granularity_bytes);

/** Apply the injected 80%-or-fallback project memory policy. */
LlmMemoryBudget evaluate_llm_memory_budget(
    const LlmMemoryBudgetRequest& request, size_t available_memory_bytes);

/**
 * Build the sole pointer-free source for ABI descriptor materialization.
 *
 * Effective workers are reduced until every standalone scenario has work.
 * Retained vector capacities are measured after allocation and the budget is
 * re-evaluated before the plan becomes valid. Invalid plans expose no
 * executable templates.
 */
LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request);

/** Validate a resolved config and build its pointer-free work plan. */
LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t checksum_auxiliary_bytes, size_t orchestration_auxiliary_bytes);

/** Return the frozen 64-bit constant for a seed domain, or zero. */
uint64_t llm_seed_domain_value(LlmSeedDomain domain);

/**
 * Derive a buffer or scenario seed as SplitMix64(base XOR domain).
 * The function also returns zero for an unrecognized enum, but a valid domain
 * may theoretically derive zero; callers must not use the seed as a validity
 * sentinel.
 */
uint64_t derive_llm_domain_seed(uint64_t base_seed, LlmSeedDomain domain);

/** Calculate the exact payload and hard ceilings for one scenario. */
LlmScenarioLimits calculate_llm_scenario_limits(
    const LlmGeometry& geometry, LlmScenario scenario);

/**
 * Resolve exact component totals for one scenario task.
 * The returned identity and scenario seed are bound to the valid immutable
 * model plan; no mappings or worker state are touched.
 */
LlmScenarioWorkPlan build_llm_scenario_work_plan(
    const LlmMemoryWorkPlan& model_plan, LlmScenario scenario, size_t steps,
    bool explicit_iterations);

/**
 * Freeze weights-only, KV-only, and mixed tasks in canonical order.
 * All identities bind the supplied model plan's executable geometry,
 * descriptor semantics, templates, and seeds. Admission/environment evidence
 * is separately recorded and does not change a frozen workload identity.
 */
LlmFrozenScenarioPlans freeze_llm_scenario_work_plans(
    const LlmMemoryWorkPlan& model_plan,
    const std::array<size_t, kLlmScenarioCount>& steps,
    bool explicit_iterations);

/** Select the smallest 8 MiB-floor pilot count within scenario guardrails. */
size_t calculate_llm_pilot_steps(const LlmScenarioLimits& limits);

/** Scale an excluded attempt toward 150 ms within scenario guardrails. */
size_t calculate_llm_calibrated_steps(double attempt_duration_seconds,
                                      size_t attempt_steps,
                                      const LlmScenarioLimits& limits);

/** Return true for a finite duration in the inclusive 100-250 ms window. */
bool llm_duration_in_target_window(double elapsed_seconds);

/** Return a static duration-quality token without mutating a frozen task plan. */
std::string_view classify_llm_duration_quality(
    double elapsed_seconds, size_t steps,
    const LlmScenarioLimits& limits) noexcept;

/** Build one weights/KV/mixed cyclic rotation for a count-loop index. */
std::array<LlmScenario, kLlmScenarioCount> build_llm_scenario_order(
    size_t loop_index);

#endif  // LLM_WORK_PLAN_H
