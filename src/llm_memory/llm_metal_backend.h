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
 * @brief Pure planning contract and inactive Metal resource foundation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * The declarations in this file contain no Objective-C types. Pure planners
 * are reentrant and safe for concurrent calls with independent objects. A
 * created backend is command-owned, synchronous, and not safe for concurrent
 * calls. Phase 8 intentionally exposes no timed Metal task implementation.
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

/** Stable label for a planned Metal resource pool. */
const char* llm_metal_resource_pool_to_string(LlmMetalResourcePool pool) noexcept;

/** Create the inactive production Metal capability/resource backend. */
std::unique_ptr<LlmBackend> create_llm_metal_backend();

/** Create the same backend with deterministic failure injection for tests. */
std::unique_ptr<LlmBackend> create_llm_metal_backend_for_testing(const LlmMetalBackendTestHooks& hooks);

#endif  // LLM_METAL_BACKEND_H
