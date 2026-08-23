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
 * @file llm_metal_backend.h
 * @brief Pure planning, checksum, and Metal decode/prefill workload contracts
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * The declarations in this file contain no Objective-C types. Pure planners
 * are reentrant and safe for concurrent calls with independent objects. A
 * created backend is command-owned, synchronous, and not safe for concurrent
 * calls. Public Metal workloads are decode and prefill with contiguous or
 * paged KV storage.
 */

#ifndef LLM_METAL_BACKEND_H
#define LLM_METAL_BACKEND_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "llm_memory/llm_backend.h"

/** Canonical CPU mirror of the MSL 2.3 foundation parameter block. */
struct alignas(8) LlmMetalFoundationParams {
  uint64_t byte_count = 0;
  uint64_t source_offset_bytes = 0;
  uint64_t destination_offset_bytes = 0;
  uint64_t logical_base_bytes = 0;
  uint64_t pattern_seed = 0;
  uint64_t block_bytes = 0;
  uint32_t physical_blocks_per_layer = 0;
  uint32_t pattern_kind = 0;
  uint32_t probe_resource_kind = 0;
  uint32_t probe_resource_slot = 0;
};

static_assert(alignof(LlmMetalFoundationParams) == 8);
static_assert(sizeof(LlmMetalFoundationParams) == 64);
static_assert(offsetof(LlmMetalFoundationParams, byte_count) == 0);
static_assert(offsetof(LlmMetalFoundationParams, source_offset_bytes) == 8);
static_assert(offsetof(LlmMetalFoundationParams, destination_offset_bytes) == 16);
static_assert(offsetof(LlmMetalFoundationParams, logical_base_bytes) == 24);
static_assert(offsetof(LlmMetalFoundationParams, pattern_seed) == 32);
static_assert(offsetof(LlmMetalFoundationParams, block_bytes) == 40);
static_assert(offsetof(LlmMetalFoundationParams, physical_blocks_per_layer) == 48);
static_assert(offsetof(LlmMetalFoundationParams, pattern_kind) == 52);
static_assert(offsetof(LlmMetalFoundationParams, probe_resource_kind) == 56);
static_assert(offsetof(LlmMetalFoundationParams, probe_resource_slot) == 60);

inline constexpr size_t kLlmMetalLayoutProbeWordCount = 28;
using LlmMetalLayoutProbeWords = std::array<uint64_t, kLlmMetalLayoutProbeWordCount>;
inline constexpr size_t kLlmMetalDecodeLayoutProbeWordCount = 38;
using LlmMetalDecodeLayoutProbeWords = std::array<uint64_t, kLlmMetalDecodeLayoutProbeWordCount>;
inline constexpr size_t kLlmMetalDecodePagedLayoutProbeWordCount = 52;
using LlmMetalDecodePagedLayoutProbeWords =
    std::array<uint64_t, kLlmMetalDecodePagedLayoutProbeWordCount>;
inline constexpr size_t kLlmMetalPrefillLayoutProbeWordCount = 42;
using LlmMetalPrefillLayoutProbeWords =
    std::array<uint64_t, kLlmMetalPrefillLayoutProbeWordCount>;
inline constexpr size_t kLlmMetalPrefillPagedLayoutProbeWordCount = 56;
using LlmMetalPrefillPagedLayoutProbeWords =
    std::array<uint64_t, kLlmMetalPrefillPagedLayoutProbeWordCount>;

/** Canonical CPU mirror of the MSL decode-contiguous parameter block. */
struct alignas(8) LlmMetalDecodeContiguousParams {
  uint64_t weight_bytes = 0;
  uint64_t k_bytes = 0;
  uint64_t v_bytes = 0;
  uint64_t segment_capacity_bytes = 0;
  uint64_t context_tokens = 0;
  uint64_t layer_count = 0;
  uint64_t batch_size = 0;
  uint64_t record_bytes = 0;
  uint64_t work_units = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  uint64_t scenario_seed = 0;
  uint32_t weight_segment_count = 0;
  uint32_t k_segment_count = 0;
  uint32_t v_segment_count = 0;
  uint32_t reserved_zero = 0;
};

static_assert(alignof(LlmMetalDecodeContiguousParams) == 8);
static_assert(sizeof(LlmMetalDecodeContiguousParams) == 120);
static_assert(offsetof(LlmMetalDecodeContiguousParams, weight_bytes) == 0);
static_assert(offsetof(LlmMetalDecodeContiguousParams, k_bytes) == 8);
static_assert(offsetof(LlmMetalDecodeContiguousParams, v_bytes) == 16);
static_assert(offsetof(LlmMetalDecodeContiguousParams, segment_capacity_bytes) == 24);
static_assert(offsetof(LlmMetalDecodeContiguousParams, context_tokens) == 32);
static_assert(offsetof(LlmMetalDecodeContiguousParams, layer_count) == 40);
static_assert(offsetof(LlmMetalDecodeContiguousParams, batch_size) == 48);
static_assert(offsetof(LlmMetalDecodeContiguousParams, record_bytes) == 56);
static_assert(offsetof(LlmMetalDecodeContiguousParams, work_units) == 64);
static_assert(offsetof(LlmMetalDecodeContiguousParams, weight_seed) == 72);
static_assert(offsetof(LlmMetalDecodeContiguousParams, k_seed) == 80);
static_assert(offsetof(LlmMetalDecodeContiguousParams, v_seed) == 88);
static_assert(offsetof(LlmMetalDecodeContiguousParams, scenario_seed) == 96);
static_assert(offsetof(LlmMetalDecodeContiguousParams, weight_segment_count) == 104);
static_assert(offsetof(LlmMetalDecodeContiguousParams, k_segment_count) == 108);
static_assert(offsetof(LlmMetalDecodeContiguousParams, v_segment_count) == 112);
static_assert(offsetof(LlmMetalDecodeContiguousParams, reserved_zero) == 116);

/** Canonical CPU mirror of the MSL decode-paged parameter block. */
struct alignas(8) LlmMetalDecodePagedParams {
  uint64_t weight_bytes = 0;
  uint64_t context_tokens = 0;
  uint64_t layer_count = 0;
  uint64_t batch_size = 0;
  uint64_t record_bytes = 0;
  uint64_t work_units = 0;
  uint64_t block_bytes = 0;
  uint64_t last_block_valid_bytes = 0;
  uint64_t append_offset_in_last_block = 0;
  uint64_t blocks_per_sequence = 0;
  uint64_t physical_blocks_per_layer = 0;
  uint64_t blocks_per_segment = 0;
  uint64_t table_entries_per_segment = 0;
  uint64_t segment_capacity_bytes = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  uint64_t scenario_seed = 0;
  uint32_t weight_segment_count = 0;
  uint32_t k_segment_count = 0;
  uint32_t v_segment_count = 0;
  uint32_t table_segment_count = 0;
  uint32_t reserved_zero = 0;
  uint32_t padding_zero = 0;
};

static_assert(alignof(LlmMetalDecodePagedParams) == 8);
static_assert(sizeof(LlmMetalDecodePagedParams) == 168);
static_assert(offsetof(LlmMetalDecodePagedParams, weight_bytes) == 0);
static_assert(offsetof(LlmMetalDecodePagedParams, context_tokens) == 8);
static_assert(offsetof(LlmMetalDecodePagedParams, layer_count) == 16);
static_assert(offsetof(LlmMetalDecodePagedParams, batch_size) == 24);
static_assert(offsetof(LlmMetalDecodePagedParams, record_bytes) == 32);
static_assert(offsetof(LlmMetalDecodePagedParams, work_units) == 40);
static_assert(offsetof(LlmMetalDecodePagedParams, block_bytes) == 48);
static_assert(offsetof(LlmMetalDecodePagedParams, last_block_valid_bytes) == 56);
static_assert(offsetof(LlmMetalDecodePagedParams, append_offset_in_last_block) == 64);
static_assert(offsetof(LlmMetalDecodePagedParams, blocks_per_sequence) == 72);
static_assert(offsetof(LlmMetalDecodePagedParams, physical_blocks_per_layer) == 80);
static_assert(offsetof(LlmMetalDecodePagedParams, blocks_per_segment) == 88);
static_assert(offsetof(LlmMetalDecodePagedParams, table_entries_per_segment) == 96);
static_assert(offsetof(LlmMetalDecodePagedParams, segment_capacity_bytes) == 104);
static_assert(offsetof(LlmMetalDecodePagedParams, weight_seed) == 112);
static_assert(offsetof(LlmMetalDecodePagedParams, k_seed) == 120);
static_assert(offsetof(LlmMetalDecodePagedParams, v_seed) == 128);
static_assert(offsetof(LlmMetalDecodePagedParams, scenario_seed) == 136);
static_assert(offsetof(LlmMetalDecodePagedParams, weight_segment_count) == 144);
static_assert(offsetof(LlmMetalDecodePagedParams, k_segment_count) == 148);
static_assert(offsetof(LlmMetalDecodePagedParams, v_segment_count) == 152);
static_assert(offsetof(LlmMetalDecodePagedParams, table_segment_count) == 156);
static_assert(offsetof(LlmMetalDecodePagedParams, reserved_zero) == 160);
static_assert(offsetof(LlmMetalDecodePagedParams, padding_zero) == 164);

/** Canonical CPU mirror of the MSL prefill-contiguous parameter block. */
struct alignas(8) LlmMetalPrefillContiguousParams {
  uint64_t weight_bytes = 0;
  uint64_t k_bytes = 0;
  uint64_t v_bytes = 0;
  uint64_t segment_capacity_bytes = 0;
  uint64_t prompt_tokens = 0;
  uint64_t attention_query_tile_tokens = 0;
  uint64_t tile_count = 0;
  uint64_t layer_count = 0;
  uint64_t batch_size = 0;
  uint64_t record_bytes = 0;
  uint64_t work_units = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  uint64_t scenario_seed = 0;
  uint32_t weight_segment_count = 0;
  uint32_t k_segment_count = 0;
  uint32_t v_segment_count = 0;
  uint32_t reserved_zero = 0;
};

static_assert(alignof(LlmMetalPrefillContiguousParams) == 8);
static_assert(sizeof(LlmMetalPrefillContiguousParams) == 136);
static_assert(offsetof(LlmMetalPrefillContiguousParams, weight_bytes) == 0);
static_assert(offsetof(LlmMetalPrefillContiguousParams, k_bytes) == 8);
static_assert(offsetof(LlmMetalPrefillContiguousParams, v_bytes) == 16);
static_assert(offsetof(LlmMetalPrefillContiguousParams,
                       segment_capacity_bytes) == 24);
static_assert(offsetof(LlmMetalPrefillContiguousParams, prompt_tokens) == 32);
static_assert(offsetof(LlmMetalPrefillContiguousParams,
                       attention_query_tile_tokens) == 40);
static_assert(offsetof(LlmMetalPrefillContiguousParams, tile_count) == 48);
static_assert(offsetof(LlmMetalPrefillContiguousParams, layer_count) == 56);
static_assert(offsetof(LlmMetalPrefillContiguousParams, batch_size) == 64);
static_assert(offsetof(LlmMetalPrefillContiguousParams, record_bytes) == 72);
static_assert(offsetof(LlmMetalPrefillContiguousParams, work_units) == 80);
static_assert(offsetof(LlmMetalPrefillContiguousParams, weight_seed) == 88);
static_assert(offsetof(LlmMetalPrefillContiguousParams, k_seed) == 96);
static_assert(offsetof(LlmMetalPrefillContiguousParams, v_seed) == 104);
static_assert(offsetof(LlmMetalPrefillContiguousParams, scenario_seed) == 112);
static_assert(offsetof(LlmMetalPrefillContiguousParams,
                       weight_segment_count) == 120);
static_assert(offsetof(LlmMetalPrefillContiguousParams, k_segment_count) ==
              124);
static_assert(offsetof(LlmMetalPrefillContiguousParams, v_segment_count) ==
              128);
static_assert(offsetof(LlmMetalPrefillContiguousParams, reserved_zero) == 132);

/** Canonical CPU mirror of the MSL prefill-paged parameter block. */
struct alignas(8) LlmMetalPrefillPagedParams {
  uint64_t weight_bytes = 0;
  uint64_t prompt_tokens = 0;
  uint64_t attention_query_tile_tokens = 0;
  uint64_t tile_count = 0;
  uint64_t layer_count = 0;
  uint64_t batch_size = 0;
  uint64_t record_bytes = 0;
  uint64_t work_units = 0;
  uint64_t block_tokens = 0;
  uint64_t block_bytes = 0;
  uint64_t last_block_valid_bytes = 0;
  uint64_t blocks_per_sequence = 0;
  uint64_t physical_blocks_per_layer = 0;
  uint64_t blocks_per_segment = 0;
  uint64_t table_entries_per_segment = 0;
  uint64_t segment_capacity_bytes = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  uint64_t scenario_seed = 0;
  uint32_t weight_segment_count = 0;
  uint32_t k_segment_count = 0;
  uint32_t v_segment_count = 0;
  uint32_t table_segment_count = 0;
  uint32_t reserved_zero = 0;
  uint32_t padding_zero = 0;
};

static_assert(alignof(LlmMetalPrefillPagedParams) == 8);
static_assert(sizeof(LlmMetalPrefillPagedParams) == 184);
static_assert(offsetof(LlmMetalPrefillPagedParams, weight_bytes) == 0);
static_assert(offsetof(LlmMetalPrefillPagedParams, prompt_tokens) == 8);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       attention_query_tile_tokens) == 16);
static_assert(offsetof(LlmMetalPrefillPagedParams, tile_count) == 24);
static_assert(offsetof(LlmMetalPrefillPagedParams, layer_count) == 32);
static_assert(offsetof(LlmMetalPrefillPagedParams, batch_size) == 40);
static_assert(offsetof(LlmMetalPrefillPagedParams, record_bytes) == 48);
static_assert(offsetof(LlmMetalPrefillPagedParams, work_units) == 56);
static_assert(offsetof(LlmMetalPrefillPagedParams, block_tokens) == 64);
static_assert(offsetof(LlmMetalPrefillPagedParams, block_bytes) == 72);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       last_block_valid_bytes) == 80);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       blocks_per_sequence) == 88);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       physical_blocks_per_layer) == 96);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       blocks_per_segment) == 104);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       table_entries_per_segment) == 112);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       segment_capacity_bytes) == 120);
static_assert(offsetof(LlmMetalPrefillPagedParams, weight_seed) == 128);
static_assert(offsetof(LlmMetalPrefillPagedParams, k_seed) == 136);
static_assert(offsetof(LlmMetalPrefillPagedParams, v_seed) == 144);
static_assert(offsetof(LlmMetalPrefillPagedParams, scenario_seed) == 152);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       weight_segment_count) == 160);
static_assert(offsetof(LlmMetalPrefillPagedParams, k_segment_count) == 164);
static_assert(offsetof(LlmMetalPrefillPagedParams, v_segment_count) == 168);
static_assert(offsetof(LlmMetalPrefillPagedParams,
                       table_segment_count) == 172);
static_assert(offsetof(LlmMetalPrefillPagedParams, reserved_zero) == 176);
static_assert(offsetof(LlmMetalPrefillPagedParams, padding_zero) == 180);

/** Result from the bounded independent Metal checksum oracle. */
struct LlmMetalChecksumOracle {
  bool valid = false;
  std::string_view reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
  LlmMetalDualMod32Checksum checksum;
};

/** Additive table-domain summary retained after the host table is released. */
struct LlmMetalPagedChecksumGroupSummary {
  size_t count = 0;
  uint32_t layer_plus_one_sum = 0;
  uint32_t batch_plus_one_sum = 0;
  uint32_t logical_plus_one_sum = 0;
  uint32_t physical_plus_one_sum = 0;
  uint32_t logical_physical_pair_sum = 0;
  uint32_t physical_address_token_sum = 0;
  uint32_t logical_physical_address_pair_sum = 0;
};

/**
 * Fixed-size paged checksum state built while the canonical host table exists.
 *
 * This summary owns no table entries. It binds the table permutation to the
 * initial K/V scan contribution and to non-separable logical/physical lookup
 * aggregates, allowing every later task oracle to remain allocation-free.
 */
struct LlmMetalPagedChecksumSummary {
  bool valid = false;
  uint64_t base_seed = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t record_bytes = 0;
  size_t blocks_per_sequence = 0;
  size_t physical_blocks_per_layer = 0;
  size_t block_bytes = 0;
  size_t last_block_valid_bytes = 0;
  size_t append_offset_in_last_block = 0;
  size_t words_per_block = 0;
  size_t initial_scan_mix_count = 0;
  uint32_t terminal_boundary_initial_k_value_sum = 0;
  uint32_t terminal_boundary_initial_v_value_sum = 0;
  LlmMetalDualMod32Checksum initial_scan_static_checksum;
  LlmMetalPagedChecksumGroupSummary all_owners;
  LlmMetalPagedChecksumGroupSummary terminal_owners;
};

/**
 * Fixed-size prefill-paged checksum state retained after table upload.
 *
 * Data-word checksums use logical contiguous-pool addresses and therefore do
 * not require a retained table. The two additive groups bind the canonical
 * permutation and its independently resolved physical segment/block address
 * to the paired-write lookups and to every tiled K/V read lookup;
 * `read_tile_ordinal_sum` preserves the prefill tile domain without owning a
 * per-tile or per-entry allocation.
 */
struct LlmMetalPrefillPagedChecksumSummary {
  bool valid = false;
  uint64_t base_seed = 0;
  uint64_t weight_seed = 0;
  uint64_t k_seed = 0;
  uint64_t v_seed = 0;
  size_t prompt_tokens = 0;
  size_t attention_query_tile_tokens = 0;
  size_t tile_count = 0;
  size_t layer_count = 0;
  size_t batch_size = 0;
  size_t record_bytes = 0;
  size_t block_tokens = 0;
  size_t blocks_per_sequence = 0;
  size_t physical_blocks_per_layer = 0;
  size_t block_bytes = 0;
  size_t blocks_per_segment = 0;
  size_t last_block_valid_bytes = 0;
  LlmMetalPagedChecksumGroupSummary write_lookups;
  LlmMetalPagedChecksumGroupSummary read_lookups;
  uint32_t read_tile_ordinal_sum = 0;
};

/** Injectable capability state used by deterministic unit tests. */
struct LlmMetalCapabilityProbe {
  bool device_available = false;
  bool has_unified_memory = false;
  bool apple7_family_supported = false;
  bool argument_buffers_tier2_supported = false;
  size_t max_buffer_length = 0;
  bool command_queue_created = false;
  bool source_compiled = false;
  size_t foundation_pipeline_count = 0;
  size_t workload_pipeline_count = 0;
  bool argument_encoder_created = false;
  size_t argument_buffer_encoded_length = 0;
  size_t argument_buffer_alignment = 0;
};

/** Runtime-dependent inputs completed after capability initialization. */
struct LlmMetalResourcePlanRequest {
  LlmGeometry geometry;
  std::optional<LlmKvLayoutPlan> paged_layout;
  size_t argument_buffer_encoded_length = 0;
  size_t argument_buffer_alignment = 0;
  size_t max_buffer_length = 0;
  size_t available_memory_bytes = 0;
  size_t host_mapping_granularity_bytes = 1;
  size_t additional_owned_bytes = 0;
  LlmMetalPlanningLimits limits;
};

/** Actual length/allocated-size pair fed to second-stage admission. */
struct LlmMetalAllocatedResource {
  LlmMetalResourcePool pool = LlmMetalResourcePool::Weight;
  size_t pool_index = 0;
  size_t length_bytes = 0;
  std::optional<size_t> allocated_size_bytes;
};

/** Exact second-stage committed-memory result. */
struct LlmMetalCommittedAdmission {
  bool valid = false;
  std::string_view reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
  size_t committed_resource_bytes = 0;
  size_t resource_rounding_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t additional_owned_bytes = 0;
  size_t known_owned_peak_bytes = 0;
  size_t admitted_budget_bytes = 0;
};

/** Test-only failure controls copied into a newly created backend. */
struct LlmMetalBackendTestHooks {
  size_t fail_allocation_after = std::numeric_limits<size_t>::max();
  bool force_layout_probe_mismatch = false;
  bool force_initialization_mismatch = false;
  bool force_timed_command_failure = false;
  bool force_invalid_gpu_timestamps = false;
  bool force_timed_checksum_mismatch = false;
  bool force_post_validation_command_failure = false;
  bool force_kv_write_validation_mismatch = false;
  bool force_wrong_paged_table_permutation = false;
  /** Corrupt one real terminal padding byte before GPU post-validation. */
  bool force_padding_canary_mismatch = false;
  std::function<bool()> stop_requested;
};

/** Evaluate capability ordering without creating a Metal device. */
LlmBackendLifecycleResult evaluate_llm_metal_capabilities(const LlmMetalCapabilityProbe& probe) noexcept;

/** Build the canonical Tier-2 slot map for the supplied segment counts. */
LlmMetalArgumentBufferPlan build_llm_metal_argument_buffer_plan(
    size_t weight_segment_count, size_t k_segment_count, size_t v_segment_count, size_t table_segment_count,
    size_t segment_slot_cap = Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL);

/** Resolve a capped grid-stride dispatch from injected pipeline properties. */
LlmMetalGridPlan build_llm_metal_grid_plan(const LlmMetalGridRequest& request);

/**
 * Resolve a weight-vector grid and attach exact actual-threadgroup costs.
 *
 * Every accounted byte is assigned through the same absolute 16-byte vector
 * modulo-grid ownership used by the Metal kernels. The returned vector sums
 * to `weight_bytes * request.work_units`, including an exact final tail.
 */
LlmMetalGridPlan build_llm_metal_weight_grid_plan(
    const LlmMetalGridRequest& request, size_t weight_bytes);

/**
 * Build exact W/K/V/table/argument/status/staging lengths and first admission.
 *
 * Contiguous pools split at exact byte boundaries. Paged K/V pools keep whole
 * blocks, and the table keeps whole uint32 entries. The returned plan owns no
 * Metal resources and performs no device calls.
 */
LlmMetalExecutionPlan build_llm_metal_execution_plan(const LlmMetalResourcePlanRequest& request);

/** Re-evaluate an allocated candidate using one committed value per resource. */
LlmMetalCommittedAdmission evaluate_llm_metal_committed_admission(
    const LlmMetalResourcePlan& plan, const std::vector<LlmMetalAllocatedResource>& allocations) noexcept;

/** Hash the exact canonical embedded MSL source bytes. */
std::string canonical_llm_metal_kernel_source_sha256();

/** Validate the 28-word CPU/MSL parameter-layout probe response. */
bool validate_llm_metal_layout_probe(const LlmMetalFoundationParams& parameters, const LlmMetalLayoutProbeWords& words,
                                     uint64_t expected_observed_resource_value) noexcept;

/** Validate every CPU/MSL field of the decode-contiguous parameter ABI. */
bool validate_llm_metal_decode_layout_probe(const LlmMetalDecodeContiguousParams& parameters,
                                            const LlmMetalDecodeLayoutProbeWords& words) noexcept;

/** Validate every CPU/MSL field of the decode-paged parameter ABI. */
bool validate_llm_metal_decode_paged_layout_probe(
    const LlmMetalDecodePagedParams& parameters,
    const LlmMetalDecodePagedLayoutProbeWords& words) noexcept;

/** Validate every CPU/MSL field of the prefill-contiguous parameter ABI. */
bool validate_llm_metal_prefill_layout_probe(
    const LlmMetalPrefillContiguousParams& parameters,
    const LlmMetalPrefillLayoutProbeWords& words) noexcept;

/** Validate every CPU/MSL field of the prefill-paged parameter ABI. */
bool validate_llm_metal_prefill_paged_layout_probe(
    const LlmMetalPrefillPagedParams& parameters,
    const LlmMetalPrefillPagedLayoutProbeWords& words) noexcept;

/** Return one deterministic Metal contiguous-buffer initialization word. */
uint32_t llm_metal_contiguous_pattern_word(uint64_t seed, uint64_t absolute_word_index) noexcept;

/** Return one deterministic paged physical-block initialization word. */
uint32_t llm_metal_paged_pattern_word(uint64_t seed, uint64_t layer_index,
                                      uint64_t physical_block,
                                      uint64_t block_word_index) noexcept;

/** Return one deterministic decode append word in the pool-address domain. */
uint32_t llm_metal_decode_append_word(uint64_t scenario_seed, uint64_t work_unit, uint64_t layer_index,
                                      uint64_t batch_index, uint64_t absolute_word_index,
                                      LlmMetalResourcePool pool) noexcept;

/** Return one deterministic prefill full-prompt write word. */
uint32_t llm_metal_prefill_write_word(
    uint64_t scenario_seed, uint64_t work_unit, uint64_t layer_index,
    uint64_t batch_index, uint64_t absolute_word_index,
    LlmMetalResourcePool pool) noexcept;

/** Calculate expected timed W/K/V accumulators without reading resource bytes. */
LlmMetalChecksumOracle calculate_llm_metal_decode_contiguous_checksum(
    const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan) noexcept;

/** Calculate expected prefill W/K/V accumulators without reading resources. */
LlmMetalChecksumOracle calculate_llm_metal_prefill_contiguous_checksum(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) noexcept;

/**
 * Count the serial range-helper visits performed by every prefill Metal lane.
 *
 * The checked count includes all work units and is used to reject a task
 * before its independent checksum oracle or GPU dispatch can become
 * unbounded. `visits` is set only on success.
 */
bool calculate_llm_metal_prefill_serial_range_visits_per_lane(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan, size_t& visits) noexcept;

/**
 * Calculate the exact largest aligned vector span of a prefill range.
 *
 * The span covers the 16-byte vectors visited by the selected contiguous or
 * paged prefill weight, KV-write, and KV-read helpers, including unaligned
 * layer, sequence, and block starts. `span_bytes` is set only on success.
 */
bool calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
    const LlmGeometry& geometry, LlmScenario scenario,
    size_t& span_bytes) noexcept;

/**
 * Build the allocation-free task-oracle summary from a validated table.
 *
 * The caller must pass the canonical table before releasing its host mapping.
 * The optional stop predicate is polled at bounded owner intervals.
 */
LlmMetalPagedChecksumSummary build_llm_metal_decode_paged_checksum_summary(
    const LlmMemoryWorkPlan& model_plan, const uint32_t* table_entries,
    size_t entry_count, const std::function<bool()>& stop_requested = {});

/** Calculate the expected paged W/K/V checksum from the fixed-size summary. */
LlmMetalChecksumOracle calculate_llm_metal_decode_paged_checksum(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmMetalPagedChecksumSummary& summary) noexcept;

/** Build the fixed-size prefill-paged oracle summary before table release. */
LlmMetalPrefillPagedChecksumSummary
build_llm_metal_prefill_paged_checksum_summary(
    const LlmMemoryWorkPlan& model_plan, const uint32_t* table_entries,
    size_t entry_count, const std::function<bool()>& stop_requested = {});

/** Calculate the expected prefill-paged W/K/V checksum from the summary. */
LlmMetalChecksumOracle calculate_llm_metal_prefill_paged_checksum(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmMetalPrefillPagedChecksumSummary& summary) noexcept;

/** Compare every lane of two Metal dual-mod32 checksums. */
bool equal_llm_metal_checksum(const LlmMetalDualMod32Checksum& left,
                              const LlmMetalDualMod32Checksum& right) noexcept;

/** Stable label for a planned Metal resource pool. */
const char* llm_metal_resource_pool_to_string(LlmMetalResourcePool pool) noexcept;

/** Create the Metal backend for the experimental decode/prefill preview. */
std::unique_ptr<LlmBackend> create_llm_metal_backend();

/** Create the same backend with deterministic failure injection for tests. */
std::unique_ptr<LlmBackend> create_llm_metal_backend_for_testing(const LlmMetalBackendTestHooks& hooks);

#endif  // LLM_METAL_BACKEND_H
