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
 * @file llm_kv_layout.h
 * @brief Pure checked planning for paged LLM KV layouts
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 * @details The builders use no shared mutable state and are reentrant.
 * Returned plans and tables own their storage; callers must synchronize any
 * later mutation of the same returned object.
 */

#ifndef LLM_KV_LAYOUT_H
#define LLM_KV_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "llm_memory/llm_memory.h"

/** Stable machine-readable reasons emitted by pure paged-KV planning. */
namespace LlmKvLayoutReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* SEQUENCE_TOKENS_ZERO =
    "sequence-tokens-zero";
inline constexpr const char* BLOCK_TOKENS_ZERO = "kv-block-tokens-zero";
inline constexpr const char* BLOCK_TOKENS_NOT_POWER_OF_TWO =
    "kv-block-tokens-not-power-of-two";
inline constexpr const char* BLOCK_TOKENS_EXCEEDS_UINT32 =
    "kv-block-tokens-exceeds-uint32";
inline constexpr const char* LAYER_COUNT_ZERO = "layer-count-zero";
inline constexpr const char* BATCH_SIZE_ZERO = "batch-size-zero";
inline constexpr const char* RECORD_BYTES_ZERO = "kv-record-bytes-zero";
inline constexpr const char* PHYSICAL_BLOCK_COUNT_OVERFLOW =
    "physical-block-count-overflow";
inline constexpr const char* BLOCK_ID_RANGE_EXCEEDED =
    "block-id-range-exceeded";
inline constexpr const char* BLOCK_BYTES_OVERFLOW = "block-bytes-overflow";
inline constexpr const char* LAST_BLOCK_OFFSET_OVERFLOW =
    "last-block-offset-overflow";
inline constexpr const char* LAST_BLOCK_TOKENS_INVALID =
    "last-block-tokens-invalid";
inline constexpr const char* LAST_BLOCK_BYTES_OVERFLOW =
    "last-block-bytes-overflow";
inline constexpr const char* APPEND_OFFSET_OVERFLOW =
    "decode-append-offset-overflow";
inline constexpr const char* LOGICAL_BYTES_OVERFLOW =
    "logical-kv-bytes-overflow";
inline constexpr const char* TOTAL_BLOCKS_OVERFLOW =
    "total-physical-blocks-overflow";
inline constexpr const char* PHYSICAL_BYTES_OVERFLOW =
    "physical-kv-bytes-overflow";
inline constexpr const char* LAYOUT_PADDING_UNDERFLOW =
    "layout-padding-underflow";
inline constexpr const char* BLOCK_TABLE_BYTES_OVERFLOW =
    "block-table-bytes-overflow";
inline constexpr const char* TRANSIENT_BYTES_OVERFLOW =
    "layout-transient-bytes-overflow";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW =
    "layout-memory-budget-overflow";
inline constexpr const char* LOOKUP_COUNT_OVERFLOW =
    "decode-lookup-count-overflow";
inline constexpr const char* LOOKUP_BYTES_OVERFLOW =
    "decode-lookup-bytes-overflow";
inline constexpr const char* WORK_UNIT_COUNT_ZERO = "work-unit-count-zero";
inline constexpr const char* WORK_UNIT_CAP_EXCEEDED =
    "work-unit-cap-exceeded";
inline constexpr const char* TASK_TOTAL_OVERFLOW =
    "decode-task-total-overflow";
inline constexpr const char* ACCOUNTED_BYTES_ZERO =
    "accounted-bytes-per-work-unit-zero";
inline constexpr const char* GUARDRAIL_BELOW_ONE_WORK_UNIT =
    "guardrail-below-one-work-unit";
inline constexpr const char* TASK_ACCOUNTED_BYTES_CAP_EXCEEDED =
    "task-accounted-bytes-cap-exceeded";
inline constexpr const char* INVALID_SCENARIO = "invalid-scenario";
inline constexpr const char* WORKER_COUNT_ZERO = "worker-count-zero";
inline constexpr const char* OWNERSHIP_COUNT_OVERFLOW =
    "block-ownership-count-overflow";
inline constexpr const char* OWNERSHIP_ACCOUNTING_MISMATCH =
    "block-ownership-accounting-mismatch";
inline constexpr const char* SEGMENT_CAPACITY_ZERO =
    "segment-capacity-zero";
inline constexpr const char* SEGMENT_SLOT_CAP_ZERO =
    "segment-slot-cap-zero";
inline constexpr const char* SEGMENT_ELEMENT_BYTES_ZERO =
    "segment-element-bytes-zero";
inline constexpr const char* SEGMENT_ELEMENT_COUNT_ZERO =
    "segment-element-count-zero";
inline constexpr const char* SEGMENT_ELEMENT_EXCEEDS_CAPACITY =
    "segment-element-exceeds-capacity";
inline constexpr const char* SEGMENT_COUNT_EXCEEDS_CAP =
    "segment-count-exceeds-cap";
inline constexpr const char* SEGMENT_ARITHMETIC_OVERFLOW =
    "segment-arithmetic-overflow";
inline constexpr const char* TABLE_ENTRY_COUNT_MISMATCH =
    "block-table-entry-count-mismatch";
inline constexpr const char* TABLE_OUTPUT_NULL =
    "block-table-output-null";
inline constexpr const char* TABLE_INVALID_SENTINEL =
    "block-table-invalid-sentinel";
inline constexpr const char* TABLE_ID_OUT_OF_RANGE =
    "block-table-id-out-of-range";
inline constexpr const char* TABLE_DUPLICATE_ID =
    "block-table-duplicate-id";
inline constexpr const char* TABLE_MISSING_ID = "block-table-missing-id";
inline constexpr const char* HASH_CHUNK_ENTRIES_ZERO =
    "hash-chunk-entries-zero";
inline constexpr const char* PREPARATION_INTERRUPTED =
    "preparation-interrupted";
inline constexpr const char* PLANNER_ALLOCATION_FAILED =
    "planner-allocation-failed";
inline constexpr const char* HASH_FAILED = "block-table-hash-failed";
inline constexpr const char* INVALID_LAYOUT_IDENTITY =
    "invalid-layout-identity";
}  // namespace LlmKvLayoutReason

/** Optional stop source for bounded block-table preparation. */
using LlmKvStopRequested = std::function<bool()>;

/** Exact inputs for paged-KV geometry resolution without a block table. */
struct LlmKvLayoutRequest {
  size_t sequence_tokens = 0;
  size_t kv_block_tokens = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
};

/** Exact logical, physical, padding, table, and preparation-peak bytes. */
struct LlmKvLayoutMemoryBudget {
  size_t k_logical_bytes = 0;
  size_t v_logical_bytes = 0;
  size_t k_physical_bytes = 0;
  size_t v_physical_bytes = 0;
  size_t k_layout_padding_bytes = 0;
  size_t v_layout_padding_bytes = 0;
  size_t block_table_bytes = 0;
  size_t validation_bitset_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t resident_layout_bytes = 0;
  size_t known_owned_peak_bytes = 0;
};

/**
 * Paged geometry shared by CPU and Metal planners before table allocation.
 *
 * `physical_blocks_per_layer` is the block-table entry count and is bounded
 * by `UINT32_MAX`; the sentinel itself is never a physical ID. The plan does
 * not allocate or materialize the table, so the exact boundary remains safe
 * to validate even when the eventual table would be impractically large.
 */
struct LlmKvLayoutPlan {
  bool valid = false;
  std::string reason_code = LlmKvLayoutReason::SEQUENCE_TOKENS_ZERO;
  size_t sequence_tokens = 0;
  size_t kv_block_tokens = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t k_or_v_record_bytes_per_layer = 0;
  size_t blocks_per_sequence = 0;
  size_t physical_blocks_per_layer = 0;
  size_t total_physical_blocks = 0;
  size_t block_bytes = 0;
  size_t last_block_tokens = 0;
  size_t last_block_valid_bytes = 0;
  size_t decode_append_offset_in_last_block = 0;
  size_t block_table_entries = 0;
  size_t permutation_iterations = 0;
  size_t validation_entries = 0;
  size_t hash_entries = 0;
  size_t upload_bytes = 0;
  LlmKvLayoutMemoryBudget memory;
  std::string geometry_identity;
};

/**
 * Resolve paged geometry without allocating proportional-to-entry storage.
 *
 * `kv_block_tokens` must be a positive power of two representable as uint32.
 * All byte/count arithmetic and the `P_b <= UINT32_MAX` ID domain are checked.
 * The returned plan owns its canonical geometry identity and may be freely
 * moved or copied; it borrows no request storage and is safe to read from
 * multiple threads.
 *
 * @param request Positive sequence, block, layer, batch, and record geometry.
 * @return A valid complete geometry, or an invalid plan with a stable reason.
 * @throws std::bad_alloc If canonical identity storage cannot be allocated.
 */
LlmKvLayoutPlan build_llm_kv_layout_plan(
    const LlmKvLayoutRequest& request);

/** Versioned identity of one materialized block-table permutation. */
struct LlmKvPermutationIdentity {
  std::string algorithm_version;
  uint64_t domain = 0;
  std::string domain_uint64_hex;
  uint64_t resolved_seed = 0;
  size_t entry_count = 0;
  std::string sha256;
  std::string identity;
};

/** Validation evidence for a candidate uint32 block-table bijection. */
struct LlmKvBlockTableValidation {
  bool valid = false;
  bool interrupted = false;
  std::string reason_code =
      LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH;
  size_t expected_entries = 0;
  size_t examined_entries = 0;
  size_t validation_bitset_bytes = 0;
};

/** A complete table is published only after generation, validation, and hash. */
struct LlmKvBlockTable {
  bool valid = false;
  bool interrupted = false;
  std::string reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
  std::vector<uint32_t> entries;
  LlmKvBlockTableValidation validation;
  LlmKvPermutationIdentity permutation;
};

/**
 * Completion evidence for materialization into caller-owned uint32 storage.
 *
 * The result deliberately owns no table entries. The caller may publish or
 * protect the supplied storage only after `valid == true`; failed or
 * interrupted preparation may leave a partial permutation in that storage.
 */
struct LlmKvInPlaceBlockTableMaterialization {
  bool valid = false;
  bool interrupted = false;
  std::string reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
  LlmKvBlockTableValidation validation;
  LlmKvPermutationIdentity permutation;
};

/**
 * Return SplitMix64(base_seed XOR KvBlockPermutation-domain).
 *
 * @param base_seed Command-level seed before domain separation.
 * @return Initial state for the frozen stateful permutation stream.
 */
uint64_t derive_llm_kv_permutation_seed(uint64_t base_seed) noexcept;

/**
 * Build canonical permutation identity evidence from an already-known hash.
 *
 * This function does not validate or materialize table storage. It is also
 * used with a fixed 64-character lowercase placeholder while calculating
 * exact identity capacities before the admitted table mapping is created.
 * Consumers must still require separate successful table validation.
 *
 * @param layout Valid paged geometry that fixes the entry count.
 * @param resolved_seed Domain-separated permutation stream seed.
 * @param sha256 Lowercase 64-character SHA-256 hex digest.
 * @return Canonical evidence, or an object with an empty identity on invalid
 *         geometry or digest shape.
 */
LlmKvPermutationIdentity build_llm_kv_permutation_identity(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    std::string sha256);

/**
 * Validate range, sentinel exclusion, uniqueness, and completeness.
 *
 * The checked ceil(entry_count/8) bitset is the only full-domain auxiliary
 * allocation. Exact cardinality plus in-range uniqueness proves completeness,
 * so the entry domain is scanned once. Stop is polled before work, while the
 * bitset is initialized, and at bounded entry intervals.
 *
 * @param expected_entries Required bijection cardinality and exclusive ID cap.
 * @param entries Candidate row-major uint32 table; borrowed for this call.
 * @param stop_requested Optional predicate, called synchronously and at most
 *        once per polling point.
 * @return Validation evidence; allocation and interruption are reason-coded.
 * @note An exception thrown by `stop_requested` propagates to the caller.
 */
LlmKvBlockTableValidation validate_llm_kv_block_table(
    size_t expected_entries, const std::vector<uint32_t>& entries,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Materialize an exact SplitMix64/Fisher-Yates permutation.
 *
 * @pre The caller has admitted `layout.memory.known_owned_peak_bytes`; this
 *      function intentionally has no separate hidden entry or memory cap.
 * @param layout Valid allocation-free geometry.
 * @param resolved_seed Initial mutable SplitMix64 stream state. Passing zero
 *        selects the frozen direct-seed golden; production callers normally
 *        pass `derive_llm_kv_permutation_seed(base_seed)`.
 * @param hash_chunk_entries Requested little-endian entry chunk boundary.
 * @param stop_requested Optional preparation predicate called synchronously
 *        before work, at bounded entry intervals, and at stage boundaries.
 * @return A complete owned table or an invalid result with no published
 *         entries after handled allocation or interruption failures.
 * @note An exception thrown by `stop_requested` propagates to the caller.
 * @throws std::bad_alloc If fixed identity fields cannot be allocated before
 *         handled proportional table preparation begins.
 */
LlmKvBlockTable materialize_llm_kv_block_table(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    size_t hash_chunk_entries = 1024,
    const LlmKvStopRequested& stop_requested = {});

/**
 * Materialize the frozen permutation directly into caller-owned storage.
 *
 * Generation, validation, and incremental little-endian hashing operate on
 * @p entries in place. The implementation allocates only the checked
 * validation bitset and bounded hashing state; it never allocates or copies a
 * second full block table. Stop is polled before mutation, at bounded entry
 * intervals, and at stage boundaries.
 *
 * @param layout Valid allocation-free paged geometry.
 * @param resolved_seed Initial mutable SplitMix64 stream state.
 * @param entries Writable uint32 storage containing exactly @p entry_count
 *        entries. It may be null only when rejected before materialization.
 * @param entry_count Must equal `layout.block_table_entries` exactly.
 * @param hash_chunk_entries Requested little-endian hash chunk boundary.
 * @param stop_requested Optional synchronous preparation predicate.
 * @return Non-owning completion, validation, and permutation evidence.
 * @note Invalid or interrupted calls may leave caller storage partially
 *       initialized; consumers must require `valid == true` before use.
 * @note An exception thrown by `stop_requested` propagates to the caller.
 * @throws std::bad_alloc If fixed identity fields cannot be allocated.
 */
LlmKvInPlaceBlockTableMaterialization
materialize_llm_kv_block_table_in_place(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    uint32_t* entries, size_t entry_count,
    size_t hash_chunk_entries = 1024,
    const LlmKvStopRequested& stop_requested = {});

/** Checked paged metadata accounting for one complete decode task. */
struct LlmPagedDecodeWorkloadPlan {
  bool valid = false;
  std::string reason_code = LlmKvLayoutReason::INVALID_SCENARIO;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  std::string layout_geometry_identity;
  std::string layout_identity;
  size_t work_units = 0;
  size_t effective_model_payload_bytes_per_work_unit = 0;
  size_t layout_metadata_lookup_count_per_layer_sequence = 0;
  size_t layout_metadata_lookup_count_per_work_unit = 0;
  size_t layout_metadata_read_bytes_per_work_unit = 0;
  size_t accounted_bytes_per_work_unit = 0;
  size_t maximum_work_units_by_work_unit_cap = 0;
  size_t maximum_work_units_by_guardrail = 0;
  size_t effective_maximum_work_units = 0;
  size_t effective_model_payload_bytes = 0;
  size_t layout_metadata_lookup_count = 0;
  size_t layout_metadata_read_bytes = 0;
  size_t task_accounted_bytes = 0;
  std::string identity;
};

/**
 * Resolve checked decode lookup and model-plus-metadata task totals.
 *
 * The workload identity is always reconstructed from `layout` and a complete,
 * self-consistent materialized permutation identity; callers cannot substitute
 * an arbitrary layout string. Both the shared work-unit ceiling and the 64 GiB
 * accounted-byte guardrail apply.
 *
 * @param layout Valid geometry used to materialize `permutation`.
 * @param scenario Decode scenario whose KV lookups are being accounted.
 * @param work_units Positive number of complete decode steps in the task.
 * @param effective_model_payload_bytes_per_work_unit Scenario payload without
 *        layout-metadata bytes.
 * @param permutation Frozen materialized table identity matching `layout`.
 * @return A complete canonical task plan, or a stable invalid reason.
 * @throws std::bad_alloc If canonical identity storage cannot be allocated.
 */
LlmPagedDecodeWorkloadPlan build_llm_paged_decode_workload_plan(
    const LlmKvLayoutPlan& layout, LlmScenario scenario, size_t work_units,
    size_t effective_model_payload_bytes_per_work_unit,
    const LlmKvPermutationIdentity& permutation);

/** One contiguous logical-block range owned by exactly one CPU worker. */
struct LlmKvCpuBlockAssignment {
  size_t layer_index = 0;
  size_t batch_sequence_index = 0;
  size_t worker_index = 0;
  size_t first_logical_block = 0;
  size_t block_count = 0;
  size_t valid_token_count = 0;
  size_t model_payload_bytes_per_work_unit = 0;
  size_t layout_metadata_lookup_count_per_work_unit = 0;
  size_t layout_metadata_read_bytes_per_work_unit = 0;
  size_t accounted_bytes_per_work_unit = 0;
};

/**
 * Deterministic block-exclusive decode KV ownership and exact accounting.
 *
 * Worker costs cover only the KV-active portion of decode: paired K/V data,
 * the terminal append, and uint32 block-table reads. Weight-shard costs are
 * intentionally outside this KV-only component and must be combined by a
 * scenario-level execution planner before a mixed-scenario partition is
 * described as cost-balanced.
 */
struct LlmKvCpuOwnershipPlan {
  bool valid = false;
  std::string reason_code = LlmKvLayoutReason::WORKER_COUNT_ZERO;
  size_t worker_count = 0;
  std::string layout_geometry_identity;
  size_t layer_sequence_count = 0;
  size_t total_owned_blocks = 0;
  size_t total_model_payload_bytes_per_work_unit = 0;
  size_t total_layout_metadata_lookup_count_per_work_unit = 0;
  size_t total_layout_metadata_read_bytes_per_work_unit = 0;
  size_t total_accounted_bytes_per_work_unit = 0;
  std::vector<LlmKvCpuBlockAssignment> assignments;
  std::vector<size_t> worker_accounted_bytes_per_work_unit;
  size_t minimum_worker_accounted_bytes_per_work_unit = 0;
  size_t maximum_worker_accounted_bytes_per_work_unit = 0;
  size_t worker_accounted_imbalance_bytes_per_work_unit = 0;
  std::string identity;
};

/**
 * Build decode-KV accounted-cost-balanced contiguous block ranges.
 *
 * Every logical block has exactly one owner. Boundaries minimize distance
 * from equal exact-cost prefixes, with the smaller logical boundary winning
 * ties, and worker assignment rotates by layer/batch ordinal. The active CPU
 * planner caps the admitted team so every effective worker owns KV work.
 * The returned vectors own their storage and publish no partial assignments on
 * allocation or arithmetic failure.
 *
 * @param layout Valid paged geometry.
 * @param worker_count Positive requested worker count. Workers beyond the
 *        number of blocks can be idle for an individual layer/sequence.
 * @param stop_requested Optional synchronous preparation predicate polled at
 *        bounded assignment intervals and stage boundaries.
 * @return Exact KV-only ownership, aggregate cost, and imbalance evidence.
 * @throws std::bad_alloc If canonical identity storage cannot be allocated.
 */
LlmKvCpuOwnershipPlan build_llm_paged_decode_kv_cpu_ownership_plan(
    const LlmKvLayoutPlan& layout, size_t worker_count,
    const LlmKvStopRequested& stop_requested = {});

/** Injected pure limits for whole-element Metal resource segmentation. */
struct LlmKvMetalSegmentLimits {
  size_t segment_capacity_bytes = 0;
  size_t segment_slot_cap = 0;
};

/** Exact segment lengths and maximum addressability for one resource pool. */
struct LlmKvSegmentPlan {
  bool valid = false;
  std::string reason_code =
      LlmKvLayoutReason::SEGMENT_CAPACITY_ZERO;
  size_t element_count = 0;
  size_t element_bytes = 0;
  size_t segment_capacity_bytes = 0;
  size_t segment_slot_cap = 0;
  size_t elements_per_segment = 0;
  size_t segment_count = 0;
  size_t maximum_addressable_elements = 0;
  size_t maximum_addressable_bytes = 0;
  size_t unused_nominal_segment_capacity_bytes = 0;
  size_t total_length_bytes = 0;
  std::vector<size_t> segment_lengths;
  std::string identity;
};

/** Paged K/V pool and uint32 block-table segmentation evidence. */
struct LlmKvMetalSegmentPlan {
  bool valid = false;
  std::string reason_code =
      LlmKvLayoutReason::SEGMENT_CAPACITY_ZERO;
  LlmKvMetalSegmentLimits limits;
  std::string layout_geometry_identity;
  LlmKvSegmentPlan k_or_v_pool;
  LlmKvSegmentPlan block_table;
  std::string identity;
};

/**
 * Build a generic whole-element segment plan using injected limits.
 *
 * No element is split across segments. The result reports both actual lengths
 * and maximum addressability under the same cap/slot pair.
 *
 * @return A complete owned length vector, or a stable invalid reason.
 * @throws std::bad_alloc If canonical identity storage cannot be allocated.
 */
LlmKvSegmentPlan build_llm_kv_segment_plan(
    size_t element_count, size_t element_bytes,
    const LlmKvMetalSegmentLimits& limits);

/**
 * Build paged K/V and table segment plans from one layout.
 *
 * @return Independent whole-block K/V and uint32-table segment evidence.
 * @throws std::bad_alloc If canonical identity storage cannot be allocated.
 */
LlmKvMetalSegmentPlan build_llm_kv_metal_segment_plan(
    const LlmKvLayoutPlan& layout,
    const LlmKvMetalSegmentLimits& limits = {
        Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES,
        Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL});

/**
 * Canonically bind geometry to a self-consistent materialized permutation.
 *
 * @return Length-prefixed identity, or empty when either input is invalid or
 *         their entry domains differ.
 * @throws std::bad_alloc If identity storage cannot be allocated.
 */
std::string serialize_llm_kv_layout_identity(
    const LlmKvLayoutPlan& layout,
    const LlmKvPermutationIdentity& permutation);

/**
 * Canonically bind a frozen workload to CPU ownership semantics.
 *
 * Both typed inputs must be valid, carry canonical component identities, and
 * reference the same geometry identity.
 *
 * @return Length-prefixed identity, or empty for invalid/mismatched inputs.
 * @throws std::bad_alloc If identity storage cannot be allocated.
 */
std::string serialize_llm_kv_cpu_execution_identity(
    const LlmPagedDecodeWorkloadPlan& workload,
    const LlmKvCpuOwnershipPlan& ownership);

/**
 * Canonically bind a frozen workload to Metal segment semantics.
 *
 * Both typed inputs must be valid, carry canonical component identities, and
 * reference the same geometry identity.
 *
 * @return Length-prefixed identity, or empty for invalid/mismatched inputs.
 * @throws std::bad_alloc If identity storage cannot be allocated.
 */
std::string serialize_llm_kv_metal_execution_identity(
    const LlmPagedDecodeWorkloadPlan& workload,
    const LlmKvMetalSegmentPlan& segments);

#endif  // LLM_KV_LAYOUT_H
