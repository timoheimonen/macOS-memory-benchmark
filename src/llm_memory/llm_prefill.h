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
 * @file llm_prefill.h
 * @brief Pure checked planning primitives for synthetic LLM prefill work
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 * @details The API contains no backend resources or mutable shared state.
 * Logical costs are reusable by CPU and Metal execution planners; CPU
 * ownership is a separate, explicitly scenario-tagged result.
 */

#ifndef LLM_PREFILL_H
#define LLM_PREFILL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llm_memory/llm_memory.h"

/** Stable machine-readable reasons emitted by pure prefill planning. */
namespace LlmPrefillReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* ACTIVE_WEIGHT_BYTES_ZERO =
    "active-weight-bytes-zero";
inline constexpr const char* PROMPT_TOKENS_ZERO = "prompt-tokens-zero";
inline constexpr const char* QUERY_TILE_TOKENS_ZERO =
    "attention-query-tile-tokens-zero";
inline constexpr const char* QUERY_TILE_TOKENS_EXCEEDS_PROMPT =
    "attention-query-tile-tokens-exceeds-prompt";
inline constexpr const char* LAYER_COUNT_ZERO = "layer-count-zero";
inline constexpr const char* BATCH_SIZE_ZERO = "batch-size-zero";
inline constexpr const char* QUERY_HEAD_COUNT_ZERO =
    "query-head-count-zero";
inline constexpr const char* HEAD_DIMENSION_ZERO =
    "head-dimension-zero";
inline constexpr const char* KV_RECORD_BYTES_ZERO =
    "kv-record-bytes-zero";
inline constexpr const char* CEIL_DIVIDE_BY_ZERO =
    "ceil-divide-by-zero";
inline constexpr const char* TILE_COUNT_OVERFLOW = "tile-count-overflow";
inline constexpr const char* PREFIX_TOKEN_VISITS_OVERFLOW =
    "prefill-prefix-token-visits-overflow";
inline constexpr const char* CAUSAL_TOKEN_PAIRS_OVERFLOW =
    "prefill-causal-token-pairs-overflow";
inline constexpr const char* LOGICAL_ATTENTION_PAIRS_OVERFLOW =
    "prefill-logical-attention-pairs-overflow";
inline constexpr const char* LOGICAL_ATTENTION_FMA_TERMS_OVERFLOW =
    "prefill-logical-attention-fma-terms-overflow";
inline constexpr const char* KV_DATA_BYTES_OVERFLOW =
    "prefill-kv-data-bytes-overflow";
inline constexpr const char* KV_LOGICAL_BYTES_OVERFLOW =
    "prefill-kv-logical-bytes-overflow";
inline constexpr const char* KV_WRITE_BYTES_OVERFLOW =
    "prefill-kv-write-bytes-overflow";
inline constexpr const char* KV_READ_BYTES_OVERFLOW =
    "prefill-kv-read-bytes-overflow";
inline constexpr const char* KV_ONLY_PAYLOAD_OVERFLOW =
    "prefill-kv-only-payload-overflow";
inline constexpr const char* MIXED_PAYLOAD_OVERFLOW =
    "prefill-mixed-payload-overflow";
inline constexpr const char* PREFIX_BLOCK_VISITS_OVERFLOW =
    "prefill-prefix-block-visits-overflow";
inline constexpr const char* LOOKUP_COUNT_OVERFLOW =
    "prefill-lookup-count-overflow";
inline constexpr const char* LOOKUP_BYTES_OVERFLOW =
    "prefill-lookup-bytes-overflow";
inline constexpr const char* PAGED_PLAN_REQUIRED =
    "prefill-paged-plan-required";
inline constexpr const char* INVALID_LOGICAL_RANGE =
    "prefill-invalid-logical-range";
inline constexpr const char* UNIT_COST_OVERFLOW =
    "prefill-unit-cost-overflow";
inline constexpr const char* INVALID_UNIT_KIND =
    "prefill-invalid-unit-kind";
inline constexpr const char* INVALID_SCENARIO =
    "prefill-invalid-scenario";
inline constexpr const char* WORKER_COUNT_ZERO = "worker-count-zero";
inline constexpr const char* WEIGHT_SHARD_COUNT_MISMATCH =
    "prefill-weight-shard-count-mismatch";
inline constexpr const char* WEIGHT_SHARDS_REQUIRED =
    "prefill-weight-shards-required";
inline constexpr const char* WEIGHT_SHARDS_NOT_APPLICABLE =
    "prefill-weight-shards-not-applicable";
inline constexpr const char* OWNERSHIP_COUNT_OVERFLOW =
    "prefill-ownership-count-overflow";
inline constexpr const char* OWNERSHIP_ACCOUNTING_MISMATCH =
    "prefill-ownership-accounting-mismatch";
inline constexpr const char* SEMANTIC_EVENT_CAP_ZERO =
    "prefill-semantic-event-cap-zero";
inline constexpr const char* SEMANTIC_EVENT_CAP_EXCEEDED =
    "prefill-semantic-event-cap-exceeded";
inline constexpr const char* AFFINE_DOMAIN_INVALID =
    "prefill-affine-domain-invalid";
inline constexpr const char* AFFINE_WORD_RANGE_OVERFLOW =
    "prefill-affine-word-range-overflow";
inline constexpr const char* OPERATION_COUNT_ZERO =
    "prefill-operation-count-zero";
inline constexpr const char* PLANNER_ALLOCATION_FAILED =
    "prefill-planner-allocation-failed";
}  // namespace LlmPrefillReason

/** Frozen methodology identifiers owned by the pure prefill contract. */
namespace LlmPrefillVersion {
inline constexpr const char* PLANNER = "llm-prefill-planner-v1";
inline constexpr const char* CPU_PARTITION =
    "llm-prefill-cpu-accounted-prefix-balanced-v1";
inline constexpr const char* OWNER_LOCAL_SCHEDULE =
    "llm-prefill-owner-local-write-then-tile-k-v-v1";
inline constexpr const char* WRITE_PATTERN =
    "llm-prefill-kv-affine64-v1";
inline constexpr const char* CHECKSUM_ORACLE =
    "llm-prefill-affine64-parity-sum-v1";
}  // namespace LlmPrefillVersion

/** Logical unit whose exact prefix cost is partitioned among CPU workers. */
enum class LlmPrefillPartitionUnitKind : uint8_t {
  ContiguousToken = 0,
  PagedBlock,
};

/** K/V domain coordinate in the frozen prefill affine pattern. */
enum class LlmPrefillKvDomain : uint8_t {
  K = 0,
  V,
};

/** Access phase represented by one backend-neutral semantic trace event. */
enum class LlmPrefillSemanticAccess : uint8_t {
  Write = 0,
  Read,
};

/**
 * Divide upward without evaluating `value + divisor - 1`.
 *
 * @param value Non-negative numerator.
 * @param divisor Required positive denominator.
 * @param output Exact quotient on success; unchanged on failure.
 * @return false only when @p divisor is zero.
 */
bool checked_llm_prefill_ceil_divide(
    size_t value, size_t divisor, size_t& output) noexcept;

/**
 * Calculate `value * (value + 1) / 2` without overflowing `value + 1`.
 *
 * One even factor is halved before the checked multiplication.
 *
 * @param value Last ordinal in the triangular sum.
 * @param output Exact result on success; unchanged on failure.
 * @return true exactly when the result fits in `size_t`.
 */
bool checked_llm_prefill_triangular(
    size_t value, size_t& output) noexcept;

/**
 * Calculate `sum(floor((slope * i + intercept) / denominator))`.
 *
 * The Euclidean reduction is logarithmic in its operands and never loops over
 * @p count. All contributions are checked before publication.
 *
 * @param count Number of terms starting at `i == 0`.
 * @param denominator Required positive denominator.
 * @param slope Linear numerator slope.
 * @param intercept Linear numerator intercept.
 * @param output Exact sum on success; unchanged on failure.
 * @return false for a zero denominator or an unrepresentable exact sum.
 */
bool checked_llm_prefill_floor_sum(
    size_t count, size_t denominator, size_t slope, size_t intercept,
    size_t& output) noexcept;

/** Exact logical inputs for one full-prompt prefill operation. */
struct LlmPrefillPlanRequest {
  size_t active_weight_bytes = 0;
  size_t prompt_tokens = 0;
  size_t attention_query_tile_tokens = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t query_head_count = 0;
  size_t head_dimension = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
  size_t kv_block_tokens = 0;
};

/**
 * Backend-neutral exact prefill geometry, payload, and optional paged costs.
 *
 * `kv_block_tokens == 0` deliberately omits paged fields. A nonzero block
 * size enables mathematical paged accounting; power-of-two and uint32 table
 * constraints remain the shared KV-layout planner's responsibility.
 */
struct LlmPrefillPlan {
  bool valid = false;
  std::string reason_code = LlmPrefillReason::PROMPT_TOKENS_ZERO;
  size_t active_weight_bytes = 0;
  size_t prompt_tokens = 0;
  size_t attention_query_tile_tokens = 0;
  size_t full_query_tile_count = 0;
  size_t final_query_tile_tokens = 0;
  size_t tile_count = 0;
  size_t attention_prefix_token_visits_per_sequence = 0;
  size_t causal_token_pairs_per_sequence = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t query_head_count = 0;
  size_t head_dimension = 0;
  size_t logical_attention_pairs = 0;
  size_t logical_attention_fma_terms = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
  size_t kv_record_bytes_per_layer = 0;
  size_t kv_bytes_per_token = 0;
  size_t k_logical_bytes = 0;
  size_t v_logical_bytes = 0;
  size_t weight_read_bytes_per_work_unit = 0;
  size_t kv_write_bytes_per_work_unit = 0;
  size_t kv_read_bytes_per_work_unit = 0;
  size_t kv_only_payload_bytes_per_work_unit = 0;
  size_t mixed_payload_bytes_per_work_unit = 0;
  bool paged = false;
  size_t kv_block_tokens = 0;
  size_t blocks_per_sequence = 0;
  size_t prefix_block_visits_per_sequence = 0;
  size_t layout_metadata_lookups_per_layer_sequence = 0;
  size_t layout_metadata_lookups_per_work_unit = 0;
  size_t layout_metadata_read_bytes_per_work_unit = 0;
};

/**
 * Resolve all prefill formulas without enumerating query tiles or tokens.
 *
 * Weight bytes are read once per operation. KV payload uses paired K/V record
 * bytes, while K/V logical capacities retain the single-domain record size.
 * Optional paged M and lookup counts use logarithmic `floor_sum`.
 *
 * @return Complete exact evidence, or the first stable validation/overflow
 *         reason. No partial result is marked valid.
 */
LlmPrefillPlan resolve_llm_prefill_plan(
    const LlmPrefillPlanRequest& request);

/** Exact KV-active cost of one contiguous token or paged block range. */
struct LlmPrefillUnitRangeCost {
  bool valid = false;
  std::string reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
  LlmPrefillPartitionUnitKind unit_kind =
      LlmPrefillPartitionUnitKind::ContiguousToken;
  size_t first_unit = 0;
  size_t unit_count = 0;
  size_t valid_token_count = 0;
  size_t data_visit_count = 0;
  size_t layout_metadata_lookup_count = 0;
  size_t model_payload_bytes = 0;
  size_t layout_metadata_read_bytes = 0;
  size_t accounted_bytes = 0;
};

/** Calculate exact paired-K/V cost for a contiguous token range. */
LlmPrefillUnitRangeCost calculate_llm_prefill_contiguous_token_range_cost(
    const LlmPrefillPlan& plan, size_t first_token, size_t token_count);

/** Calculate exact paired-K/V cost for one contiguous token. */
LlmPrefillUnitRangeCost calculate_llm_prefill_contiguous_token_cost(
    const LlmPrefillPlan& plan, size_t token_index);

/**
 * Calculate exact paired-K/V data and uint32 lookup cost for whole blocks.
 *
 * Terminal-block data counts include only valid prompt tokens. The returned
 * lookup count is semantic: one paired write lookup plus separate K and V
 * lookups for every query-tile prefix that reaches the block.
 */
LlmPrefillUnitRangeCost calculate_llm_prefill_paged_block_range_cost(
    const LlmPrefillPlan& plan, size_t first_block, size_t block_count);

/** Calculate exact paired-K/V and metadata cost for one paged block. */
LlmPrefillUnitRangeCost calculate_llm_prefill_paged_block_cost(
    const LlmPrefillPlan& plan, size_t block_index);

/** Inputs for one deterministic CPU partitioning scope. */
struct LlmPrefillCpuOwnershipRequest {
  LlmPrefillPartitionUnitKind unit_kind =
      LlmPrefillPartitionUnitKind::ContiguousToken;
  LlmScenario scenario = LlmScenario::KvOnly;
  size_t worker_count = 0;
  size_t worker_rotation = 0;
  std::vector<size_t> worker_weight_shard_bytes;
};

/** One non-empty, indivisible-unit CPU ownership range. */
struct LlmPrefillCpuAssignment {
  size_t range_rank = 0;
  size_t worker_index = 0;
  size_t first_unit = 0;
  size_t unit_count = 0;
  LlmPrefillUnitRangeCost kv_cost;
};

/**
 * Scenario-tagged exact CPU ownership for one ordered layer/batch scope.
 *
 * The optional weight vector is a caller-defined baseline charged once in
 * this same scope. It is required for weights-only, forbidden for KV-only,
 * and optional for mixed. When supplied for mixed, boundaries minimize the
 * distance of cumulative weight-plus-KV cost from equal rational targets.
 * Non-empty ranges are retained whenever units are at least workers; excess
 * workers own no KV range, though they may retain weight shards. Lower
 * logical boundaries win exact distance ties.
 */
struct LlmPrefillCpuOwnershipPlan {
  bool valid = false;
  std::string reason_code = LlmPrefillReason::WORKER_COUNT_ZERO;
  std::string identity;
  LlmPrefillPartitionUnitKind unit_kind =
      LlmPrefillPartitionUnitKind::ContiguousToken;
  LlmScenario scenario = LlmScenario::KvOnly;
  size_t worker_count = 0;
  size_t worker_rotation = 0;
  size_t logical_unit_count = 0;
  size_t active_worker_count = 0;
  bool weight_shards_included = false;
  size_t total_weight_shard_bytes = 0;
  size_t total_kv_model_payload_bytes = 0;
  size_t total_layout_metadata_lookup_count = 0;
  size_t total_layout_metadata_read_bytes = 0;
  size_t total_kv_accounted_bytes = 0;
  size_t total_scenario_accounted_bytes = 0;
  std::vector<LlmPrefillCpuAssignment> assignments;
  std::vector<size_t> worker_weight_shard_bytes;
  std::vector<size_t> worker_kv_model_payload_bytes;
  std::vector<size_t> worker_layout_metadata_lookup_count;
  std::vector<size_t> worker_layout_metadata_read_bytes;
  std::vector<size_t> worker_scenario_accounted_bytes;
  size_t minimum_worker_accounted_bytes = 0;
  size_t maximum_worker_accounted_bytes = 0;
  size_t worker_accounted_imbalance_bytes = 0;
};

/**
 * Build an exact contiguous-token or whole-block CPU partition.
 *
 * The builder evaluates logarithmic closed-form prefixes and allocates only
 * O(worker_count) result storage; it never materializes a token/block cost
 * array. Worker rotation is normalized modulo the worker count. A valid
 * result includes a fixed-order identity that binds the prefill geometry,
 * scenario, rotation, weight shards, assignment boundaries, and exact
 * per-worker cost evidence. Invalid results publish no assignments, worker
 * vectors, totals, imbalance evidence, or identity.
 *
 * @throws std::bad_alloc If fixed reason storage cannot be allocated before
 *         handled proportional result allocation begins.
 */
LlmPrefillCpuOwnershipPlan build_llm_prefill_cpu_ownership_plan(
    const LlmPrefillPlan& prefill,
    const LlmPrefillCpuOwnershipRequest& request);

/** One exact owner-local write or tile-prefix read event. */
struct LlmPrefillSemanticEvent {
  LlmPrefillSemanticAccess access = LlmPrefillSemanticAccess::Write;
  LlmPrefillKvDomain domain = LlmPrefillKvDomain::K;
  size_t tile_index = 0;
  size_t tile_end_token = 0;
  size_t logical_unit_index = 0;
  size_t visit_token_count = 0;
};

/** Bounded request for materializing the backend-neutral semantic test trace. */
struct LlmPrefillSemanticTraceRequest {
  LlmPrefillPartitionUnitKind unit_kind =
      LlmPrefillPartitionUnitKind::ContiguousToken;
  size_t first_unit = 0;
  size_t unit_count = 0;
  size_t maximum_events = 0;
};

/** Ordered owner-local events; an invalid result publishes no partial trace. */
struct LlmPrefillSemanticTrace {
  bool valid = false;
  std::string reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
  std::vector<LlmPrefillSemanticEvent> events;
};

/**
 * Materialize the write-all -> per-tile K-all -> V-all schedule oracle.
 *
 * Writes traverse logical units in ascending order with K then V for each
 * unit. Each tile then traverses every owned prefix-intersecting unit for K,
 * followed by the same units for V. Tile ends advance by remaining distance,
 * never by `(tile + 1) * Q`. This deliberately bounded test/audit seam may
 * enumerate tiles; production planning remains closed form.
 */
LlmPrefillSemanticTrace build_llm_prefill_semantic_trace(
    const LlmPrefillPlan& plan,
    const LlmPrefillSemanticTraceRequest& request);

/**
 * Return one `llm-prefill-kv-affine64-v1` canonical logical word.
 *
 * Coordinates are one-based inside independent odd-multiplier terms and all
 * arithmetic is deliberately modulo 2^64. The prefill phase has a fixed
 * domain term; @p scenario_seed is expected to be scenario-domain-separated.
 * An invalid enum value returns zero.
 */
uint64_t llm_prefill_affine64_word(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, uint64_t logical_word_index) noexcept;

/** Return one little-endian byte from the frozen affine logical byte stream. */
uint8_t llm_prefill_affine64_byte(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, uint64_t logical_byte_index) noexcept;

/** Closed-form independent checksum evidence for a canonical word range. */
struct LlmPrefillAffine64Checksum {
  bool valid = false;
  std::string reason_code = LlmPrefillReason::AFFINE_DOMAIN_INVALID;
  size_t exact_word_count = 0;
  size_t even_logical_word_count = 0;
  size_t odd_logical_word_count = 0;
  uint64_t even_logical_word_sum = 0;
  uint64_t odd_logical_word_sum = 0;
};

/**
 * Calculate parity-separated affine word sums in O(1).
 *
 * The parity lanes make logical-word placement observable while retaining a
 * bounded scalar oracle. Sums use the checksum version's specified modulo
 * 2^64 arithmetic; only coordinate-range representability is checked.
 */
LlmPrefillAffine64Checksum calculate_llm_prefill_affine64_checksum(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, size_t first_logical_word,
    size_t logical_word_count);

/**
 * Sum one canonical word range across task-local operations `0 ... T - 1`.
 *
 * Counts are exact and checked; lane sums intentionally use modulo-2^64
 * arithmetic. The closed form is O(1) in both @p operation_count and the word
 * count. Excluded final-write validation uses the separate word/byte helpers
 * with terminal ordinal `operation_count - 1`.
 */
LlmPrefillAffine64Checksum calculate_llm_prefill_affine64_task_checksum(
    uint64_t scenario_seed, size_t operation_count, uint64_t layer_index,
    uint64_t batch_sequence_index, LlmPrefillKvDomain domain,
    size_t first_logical_word, size_t logical_word_count);

#endif  // LLM_PREFILL_H
