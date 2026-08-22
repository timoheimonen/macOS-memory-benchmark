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
 * @file llm_metal_backend.mm
 * @brief Metal capability, planning, allocation, and validation foundation
 *
 * Metal and Objective-C ownership remain in this file. The backend is
 * synchronous, command-owned, and deliberately has no timed workload path in
 * Phase 8. Every Objective-C entry boundary catches Objective-C and C++
 * failures, and candidate resources are published only after both admissions,
 * initialization, upload, ABI probing, and excluded validation succeed.
 */

#include "llm_memory/llm_metal_backend.h"

#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "core/system/page_size.h"
#include "llm_memory/llm_metal_kernels_source.h"
#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr size_t kFoundationPipelineCount = 5;

bool is_power_of_two(size_t value) noexcept { return value != 0 && (value & (value - 1)) == 0; }

bool checked_ceil_divide(size_t value, size_t divisor, size_t& result) noexcept {
  if (divisor == 0) {
    return false;
  }
  result = value / divisor + (value % divisor != 0 ? 1U : 0U);
  return true;
}

bool checked_subtract(size_t lhs, size_t rhs, size_t& result) noexcept {
  if (rhs > lhs) {
    return false;
  }
  result = lhs - rhs;
  return true;
}

void append_identity_field(std::string& output, std::string_view key, std::string_view value) {
  output.append(key.data(), key.size());
  output.push_back('=');
  output += std::to_string(value.size());
  output.push_back(':');
  output.append(value.data(), value.size());
  output.push_back(';');
}

void append_identity_field(std::string& output, std::string_view key, size_t value) {
  append_identity_field(output, key, std::to_string(value));
}

std::string ns_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* const utf8 = value.UTF8String;
  return utf8 == nullptr ? std::string{} : std::string(utf8);
}

std::string bounded_diagnostic(std::string value) {
  if (value.size() <= Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES) {
    return value;
  }
  size_t prefix_bytes = Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES;
  while (prefix_bytes > 0 && prefix_bytes < value.size() &&
         (static_cast<unsigned char>(value[prefix_bytes]) & 0xc0U) == 0x80U) {
    --prefix_bytes;
  }
  value.resize(prefix_bytes);
  return value;
}

LlmMetalErrorDiagnostic error_diagnostic(NSError* error) {
  if (error == nil) {
    return {};
  }
  return {bounded_diagnostic(ns_string(error.domain)), static_cast<long long>(error.code),
          bounded_diagnostic(ns_string(error.localizedDescription))};
}

LlmMetalErrorDiagnostic internal_error(std::string description) {
  return {"macos-memory-benchmark.llm-metal", 0, bounded_diagnostic(std::move(description))};
}

std::string version_macro_string(long value) {
  const long major = value / 10000;
  const long minor = (value / 100) % 100;
  const long patch = value % 100;
  std::string result = std::to_string(major) + "." + std::to_string(minor);
  if (patch != 0) {
    result += "." + std::to_string(patch);
  }
  return result;
}

const char* storage_mode_string(MTLStorageMode mode) noexcept {
  switch (mode) {
    case MTLStorageModeShared:
      return "shared";
    case MTLStorageModeManaged:
      return "managed";
    case MTLStorageModePrivate:
      return "private";
    case MTLStorageModeMemoryless:
      return "memoryless";
  }
  return "unknown";
}

const char* cpu_cache_mode_string(MTLCPUCacheMode mode) noexcept {
  switch (mode) {
    case MTLCPUCacheModeDefaultCache:
      return "default-cache";
    case MTLCPUCacheModeWriteCombined:
      return "write-combined";
  }
  return "unknown";
}

const char* hazard_mode_string(MTLHazardTrackingMode mode) noexcept {
  switch (mode) {
    case MTLHazardTrackingModeDefault:
      return "default";
    case MTLHazardTrackingModeUntracked:
      return "untracked";
    case MTLHazardTrackingModeTracked:
      return "tracked";
  }
  return "unknown";
}

bool is_private_pool(LlmMetalResourcePool pool) noexcept {
  return pool == LlmMetalResourcePool::Weight || pool == LlmMetalResourcePool::K || pool == LlmMetalResourcePool::V ||
         pool == LlmMetalResourcePool::BlockTable;
}

size_t argument_resource_id(LlmMetalResourcePool pool, size_t pool_index) noexcept {
  switch (pool) {
    case LlmMetalResourcePool::Weight:
      return LlmMetalKernelContract::kWeightSegmentBaseId + pool_index;
    case LlmMetalResourcePool::K:
      return LlmMetalKernelContract::kKeySegmentBaseId + pool_index;
    case LlmMetalResourcePool::V:
      return LlmMetalKernelContract::kValueSegmentBaseId + pool_index;
    case LlmMetalResourcePool::BlockTable:
      return LlmMetalKernelContract::kTableSegmentBaseId + pool_index;
    case LlmMetalResourcePool::Status:
      return LlmMetalKernelContract::kStatusChecksumResourceId;
    case LlmMetalResourcePool::ArgumentBuffer:
    case LlmMetalResourcePool::Staging:
      break;
  }
  return std::numeric_limits<size_t>::max();
}

bool same_planned_resource(const LlmMetalPlannedResource& planned, const LlmMetalAllocatedResource& actual) noexcept {
  return planned.pool == actual.pool && planned.pool_index == actual.pool_index &&
         planned.length_bytes == actual.length_bytes;
}

bool same_planning_limits(const LlmMetalPlanningLimits& left, const LlmMetalPlanningLimits& right) noexcept {
  return left.segment_capacity_bytes == right.segment_capacity_bytes &&
         left.segment_slots_per_pool == right.segment_slots_per_pool &&
         left.threads_per_threadgroup_cap == right.threads_per_threadgroup_cap &&
         left.maximum_threadgroups_per_grid == right.maximum_threadgroups_per_grid &&
         left.maximum_owner_ordinals_per_threadgroup == right.maximum_owner_ordinals_per_threadgroup &&
         left.maximum_vector_iterations_per_lane_per_visit == right.maximum_vector_iterations_per_lane_per_visit &&
         left.maximum_paged_semantic_lookups_per_task == right.maximum_paged_semantic_lookups_per_task &&
         left.maximum_work_units_per_dispatch == right.maximum_work_units_per_dispatch;
}

bool same_segment_plan(const LlmKvSegmentPlan& left, const LlmKvSegmentPlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.element_count == right.element_count && left.element_bytes == right.element_bytes &&
         left.segment_capacity_bytes == right.segment_capacity_bytes &&
         left.segment_slot_cap == right.segment_slot_cap && left.elements_per_segment == right.elements_per_segment &&
         left.segment_count == right.segment_count &&
         left.maximum_addressable_elements == right.maximum_addressable_elements &&
         left.maximum_addressable_bytes == right.maximum_addressable_bytes &&
         left.unused_nominal_segment_capacity_bytes == right.unused_nominal_segment_capacity_bytes &&
         left.total_length_bytes == right.total_length_bytes && left.segment_lengths == right.segment_lengths &&
         left.identity == right.identity;
}

bool same_layout_memory(const LlmKvLayoutMemoryBudget& left, const LlmKvLayoutMemoryBudget& right) noexcept {
  return left.k_logical_bytes == right.k_logical_bytes && left.v_logical_bytes == right.v_logical_bytes &&
         left.k_physical_bytes == right.k_physical_bytes && left.v_physical_bytes == right.v_physical_bytes &&
         left.k_layout_padding_bytes == right.k_layout_padding_bytes &&
         left.v_layout_padding_bytes == right.v_layout_padding_bytes &&
         left.block_table_bytes == right.block_table_bytes &&
         left.validation_bitset_bytes == right.validation_bitset_bytes &&
         left.transient_peak_bytes == right.transient_peak_bytes &&
         left.resident_layout_bytes == right.resident_layout_bytes &&
         left.known_owned_peak_bytes == right.known_owned_peak_bytes;
}

bool same_layout_plan(const LlmKvLayoutPlan& left, const LlmKvLayoutPlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.sequence_tokens == right.sequence_tokens && left.kv_block_tokens == right.kv_block_tokens &&
         left.layer_count == right.layer_count && left.batch_size == right.batch_size &&
         left.k_or_v_record_bytes_per_layer == right.k_or_v_record_bytes_per_layer &&
         left.blocks_per_sequence == right.blocks_per_sequence &&
         left.physical_blocks_per_layer == right.physical_blocks_per_layer &&
         left.total_physical_blocks == right.total_physical_blocks && left.block_bytes == right.block_bytes &&
         left.last_block_tokens == right.last_block_tokens &&
         left.last_block_valid_bytes == right.last_block_valid_bytes &&
         left.decode_append_offset_in_last_block == right.decode_append_offset_in_last_block &&
         left.block_table_entries == right.block_table_entries &&
         left.permutation_iterations == right.permutation_iterations &&
         left.validation_entries == right.validation_entries && left.hash_entries == right.hash_entries &&
         left.upload_bytes == right.upload_bytes && same_layout_memory(left.memory, right.memory) &&
         left.geometry_identity == right.geometry_identity;
}

bool same_optional_segment_plan(const std::optional<LlmKvSegmentPlan>& left,
                                const std::optional<LlmKvSegmentPlan>& right) noexcept {
  return left.has_value() == right.has_value() && (!left.has_value() || same_segment_plan(*left, *right));
}

bool same_optional_layout_plan(const std::optional<LlmKvLayoutPlan>& left,
                               const std::optional<LlmKvLayoutPlan>& right) noexcept {
  return left.has_value() == right.has_value() && (!left.has_value() || same_layout_plan(*left, *right));
}

bool same_argument_buffer_plan(const LlmMetalArgumentBufferPlan& left, const LlmMetalArgumentBufferPlan& right) {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.weight_slot_base == right.weight_slot_base && left.k_slot_base == right.k_slot_base &&
         left.v_slot_base == right.v_slot_base && left.table_slot_base == right.table_slot_base &&
         left.status_slot == right.status_slot &&
         left.encoded_resource_slot_count == right.encoded_resource_slot_count &&
         left.weight_segment_count == right.weight_segment_count && left.k_segment_count == right.k_segment_count &&
         left.v_segment_count == right.v_segment_count && left.table_segment_count == right.table_segment_count &&
         left.active_resource_count == right.active_resource_count && left.identity == right.identity;
}

bool same_planned_resources(const std::vector<LlmMetalPlannedResource>& left,
                            const std::vector<LlmMetalPlannedResource>& right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t index = 0; index < left.size(); ++index) {
    if (left[index].pool != right[index].pool || left[index].pool_index != right[index].pool_index ||
        left[index].length_bytes != right[index].length_bytes || left[index].persistent != right[index].persistent) {
      return false;
    }
  }
  return true;
}

bool same_resource_plan(const LlmMetalResourcePlan& left, const LlmMetalResourcePlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         same_planning_limits(left.limits, right.limits) &&
         same_segment_plan(left.weight_segments, right.weight_segments) &&
         same_segment_plan(left.k_segments, right.k_segments) && same_segment_plan(left.v_segments, right.v_segments) &&
         same_optional_segment_plan(left.table_segments, right.table_segments) &&
         same_optional_layout_plan(left.paged_layout, right.paged_layout) &&
         same_argument_buffer_plan(left.argument_buffer, right.argument_buffer) &&
         left.argument_buffer_encoded_length == right.argument_buffer_encoded_length &&
         left.argument_buffer_alignment == right.argument_buffer_alignment &&
         left.max_buffer_length == right.max_buffer_length &&
         left.host_mapping_granularity_bytes == right.host_mapping_granularity_bytes &&
         left.status_buffer_length == right.status_buffer_length &&
         left.staging_buffer_length == right.staging_buffer_length &&
         left.host_permutation_mapping_bytes == right.host_permutation_mapping_bytes &&
         left.permutation_validation_bitset_bytes == right.permutation_validation_bitset_bytes &&
         left.additional_owned_bytes == right.additional_owned_bytes &&
         left.persistent_resource_length_bytes == right.persistent_resource_length_bytes &&
         left.transient_peak_bytes == right.transient_peak_bytes &&
         left.known_owned_peak_bytes == right.known_owned_peak_bytes &&
         left.available_memory_bytes == right.available_memory_bytes &&
         left.admitted_budget_bytes == right.admitted_budget_bytes &&
         same_planned_resources(left.planned_resources, right.planned_resources) && left.identity == right.identity;
}

bool calculate_admitted_budget(size_t available_memory_bytes, size_t& admitted_budget_bytes) noexcept {
  if (available_memory_bytes == 0) {
    return NumericUtils::checked_multiply(static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
                                          Constants::BYTES_PER_MB, admitted_budget_bytes);
  }
  const long double scaled = static_cast<long double>(available_memory_bytes) * Constants::MEMORY_LIMIT_FACTOR;
  if (scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  admitted_budget_bytes = static_cast<size_t>(scaled);
  return true;
}

bool append_planned_segments(LlmMetalResourcePool pool, const LlmKvSegmentPlan& segments,
                             std::vector<LlmMetalPlannedResource>& resources) {
  for (size_t index = 0; index < segments.segment_lengths.size(); ++index) {
    resources.push_back({pool, index, segments.segment_lengths[index], true});
  }
  return true;
}

bool segment_plan_matches_length(const LlmKvSegmentPlan& segments, size_t expected_length,
                                 size_t max_buffer_length) noexcept {
  if (!segments.valid || segments.total_length_bytes != expected_length ||
      segments.segment_count != segments.segment_lengths.size()) {
    return false;
  }
  size_t sum = 0;
  for (size_t length : segments.segment_lengths) {
    if (length == 0 || length > max_buffer_length || !NumericUtils::checked_add(sum, length, sum)) {
      return false;
    }
  }
  return sum == expected_length;
}

uint8_t contiguous_pattern_byte(uint64_t seed, uint64_t absolute_byte) noexcept {
  const uint64_t word_index = absolute_byte / sizeof(uint64_t);
  const unsigned byte_index = static_cast<unsigned>(absolute_byte % sizeof(uint64_t));
  const uint64_t word = seed + LlmMetalKernelContract::kBufferPatternMultiplier * (word_index + 1U);
  return static_cast<uint8_t>(word >> (byte_index * 8U));
}

uint8_t paged_pattern_byte(uint64_t seed, uint64_t block_bytes, uint32_t physical_blocks_per_layer,
                           uint64_t absolute_byte) noexcept {
  if (block_bytes == 0 || physical_blocks_per_layer == 0) {
    return 0;
  }
  const uint64_t global_block = absolute_byte / block_bytes;
  const uint64_t block_byte = absolute_byte % block_bytes;
  const uint64_t layer = global_block / static_cast<uint64_t>(physical_blocks_per_layer);
  const uint64_t physical_block = global_block % static_cast<uint64_t>(physical_blocks_per_layer);
  const uint64_t word_index = block_byte / sizeof(uint64_t);
  const unsigned byte_index = static_cast<unsigned>(block_byte % sizeof(uint64_t));
  const uint64_t word = seed + LlmMetalKernelContract::kPagedPatternLayerMultiplier * (layer + 1U) +
                        LlmMetalKernelContract::kPagedPatternPhysicalMultiplier * (physical_block + 1U) +
                        LlmMetalKernelContract::kPagedPatternWordMultiplier * (word_index + 1U);
  return static_cast<uint8_t>(word >> (byte_index * 8U));
}

}  // namespace

LlmBackendLifecycleResult evaluate_llm_metal_capabilities(const LlmMetalCapabilityProbe& probe) noexcept {
  if (!probe.device_available) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::METAL_DEVICE_UNAVAILABLE};
  }
  if (!probe.has_unified_memory) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::UNIFIED_MEMORY_REQUIRED};
  }
  if (!probe.apple7_family_supported) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::APPLE7_FAMILY_REQUIRED};
  }
  if (!probe.argument_buffers_tier2_supported) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED};
  }
  if (probe.max_buffer_length < Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY};
  }
  if (!probe.command_queue_created) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED};
  }
  if (!probe.source_compiled) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
  }
  if (probe.foundation_pipeline_count != kFoundationPipelineCount) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
  }
  if (!probe.argument_encoder_created) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_ENCODER_CREATION_FAILED};
  }
  if (probe.argument_buffer_encoded_length == 0 || !is_power_of_two(probe.argument_buffer_alignment)) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
  }
  return {LlmBackendStatus::Ready, LlmBackendReason::VALID};
}

LlmMetalArgumentBufferPlan build_llm_metal_argument_buffer_plan(size_t weight_segment_count, size_t k_segment_count,
                                                                size_t v_segment_count, size_t table_segment_count,
                                                                size_t segment_slot_cap) {
  LlmMetalArgumentBufferPlan plan;
  if (segment_slot_cap == 0 || weight_segment_count > segment_slot_cap || k_segment_count > segment_slot_cap ||
      v_segment_count > segment_slot_cap || table_segment_count > segment_slot_cap) {
    return plan;
  }
  size_t double_cap = 0;
  size_t triple_cap = 0;
  size_t quadruple_cap = 0;
  if (!NumericUtils::checked_multiply(segment_slot_cap, 2, double_cap) ||
      !NumericUtils::checked_multiply(segment_slot_cap, 3, triple_cap) ||
      !NumericUtils::checked_multiply(segment_slot_cap, 4, quadruple_cap) ||
      !NumericUtils::checked_add(quadruple_cap, 1, plan.encoded_resource_slot_count)) {
    return plan;
  }
  plan.weight_slot_base = 0;
  plan.k_slot_base = segment_slot_cap;
  plan.v_slot_base = double_cap;
  plan.table_slot_base = triple_cap;
  plan.status_slot = quadruple_cap;
  plan.weight_segment_count = weight_segment_count;
  plan.k_segment_count = k_segment_count;
  plan.v_segment_count = v_segment_count;
  plan.table_segment_count = table_segment_count;
  size_t active_without_status = 0;
  if (!NumericUtils::checked_add(weight_segment_count, k_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, v_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, table_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, 1, plan.active_resource_count)) {
    return LlmMetalArgumentBufferPlan{};
  }
  plan.identity = Constants::LLM_METAL_ARGUMENT_BUFFER_ABI_VERSION;
  append_identity_field(plan.identity, "segment_slot_cap", segment_slot_cap);
  append_identity_field(plan.identity, "weight_slot_base", plan.weight_slot_base);
  append_identity_field(plan.identity, "k_slot_base", plan.k_slot_base);
  append_identity_field(plan.identity, "v_slot_base", plan.v_slot_base);
  append_identity_field(plan.identity, "table_slot_base", plan.table_slot_base);
  append_identity_field(plan.identity, "status_slot", plan.status_slot);
  append_identity_field(plan.identity, "encoded_resource_slot_count", plan.encoded_resource_slot_count);
  append_identity_field(plan.identity, "weight_segment_count", weight_segment_count);
  append_identity_field(plan.identity, "k_segment_count", k_segment_count);
  append_identity_field(plan.identity, "v_segment_count", v_segment_count);
  append_identity_field(plan.identity, "table_segment_count", table_segment_count);
  append_identity_field(plan.identity, "active_resource_count", plan.active_resource_count);
  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  return plan;
}

LlmMetalGridPlan build_llm_metal_grid_plan(const LlmMetalGridRequest& request) {
  LlmMetalGridPlan plan;
  plan.owner_count = request.owner_count;
  plan.work_units = request.work_units;
  plan.paged_semantic_lookups = request.paged_semantic_lookups;
  if (request.pipeline.thread_execution_width == 0) {
    return plan;
  }
  const size_t thread_limit =
      std::min(request.pipeline.max_total_threads_per_threadgroup, request.limits.threads_per_threadgroup_cap);
  plan.threads_per_threadgroup =
      (thread_limit / request.pipeline.thread_execution_width) * request.pipeline.thread_execution_width;
  if (plan.threads_per_threadgroup == 0 || request.limits.maximum_threadgroups_per_grid == 0) {
    plan.reason_code = LlmMetalPlanReason::PIPELINE_THREAD_LIMIT_INVALID;
    return plan;
  }
  if (request.work_units > request.limits.maximum_work_units_per_dispatch) {
    plan.reason_code = LlmMetalPlanReason::WORK_UNITS_PER_DISPATCH_CAP_EXCEEDED;
    return plan;
  }
  if (request.paged_semantic_lookups > request.limits.maximum_paged_semantic_lookups_per_task) {
    plan.reason_code = LlmMetalPlanReason::SEMANTIC_VISIT_CAP_EXCEEDED;
    return plan;
  }
  plan.actual_threadgroups =
      request.owner_count == 0 ? 0 : std::min(request.owner_count, request.limits.maximum_threadgroups_per_grid);
  if (plan.actual_threadgroups != 0 &&
      !checked_ceil_divide(request.owner_count, plan.actual_threadgroups, plan.owner_ordinals_per_threadgroup)) {
    plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
    return plan;
  }
  if (plan.owner_ordinals_per_threadgroup > request.limits.maximum_owner_ordinals_per_threadgroup) {
    plan.reason_code = LlmMetalPlanReason::OWNER_STRIDE_CAP_EXCEEDED;
    return plan;
  }
  size_t bytes_per_lane_iteration = 0;
  if (!NumericUtils::checked_multiply(plan.threads_per_threadgroup, Constants::LLM_METAL_VECTOR_WIDTH_BYTES,
                                      bytes_per_lane_iteration) ||
      !checked_ceil_divide(request.visit_bytes, bytes_per_lane_iteration, plan.vector_iterations_per_lane_per_visit)) {
    plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
    return plan;
  }
  if (plan.vector_iterations_per_lane_per_visit > request.limits.maximum_vector_iterations_per_lane_per_visit) {
    plan.reason_code = LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED;
    return plan;
  }
  if (!request.owner_accounted_bytes.empty()) {
    if (request.owner_accounted_bytes.size() != request.owner_count) {
      plan.reason_code = LlmMetalPlanReason::OWNER_COST_COUNT_MISMATCH;
      return plan;
    }
    try {
      plan.threadgroup_accounted_bytes.assign(plan.actual_threadgroups, 0);
    } catch (const std::bad_alloc&) {
      plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
      return plan;
    } catch (const std::length_error&) {
      plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
      return plan;
    }
    for (size_t owner = 0; owner < request.owner_count; ++owner) {
      const size_t threadgroup = owner % plan.actual_threadgroups;
      if (!NumericUtils::checked_add(plan.threadgroup_accounted_bytes[threadgroup],
                                     request.owner_accounted_bytes[owner],
                                     plan.threadgroup_accounted_bytes[threadgroup])) {
        plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
        return plan;
      }
    }
    const auto [minimum, maximum] =
        std::minmax_element(plan.threadgroup_accounted_bytes.begin(), plan.threadgroup_accounted_bytes.end());
    plan.minimum_threadgroup_accounted_bytes = *minimum;
    plan.maximum_threadgroup_accounted_bytes = *maximum;
    if (!checked_subtract(plan.maximum_threadgroup_accounted_bytes, plan.minimum_threadgroup_accounted_bytes,
                          plan.threadgroup_accounted_imbalance_bytes)) {
      plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
      return plan;
    }
  }
  try {
    plan.identity = Constants::LLM_METAL_GRID_PLAN_VERSION;
    append_identity_field(plan.identity, "owner_count", plan.owner_count);
    append_identity_field(plan.identity, "threads_per_threadgroup", plan.threads_per_threadgroup);
    append_identity_field(plan.identity, "actual_threadgroups", plan.actual_threadgroups);
    append_identity_field(plan.identity, "owner_ordinals_per_threadgroup", plan.owner_ordinals_per_threadgroup);
    append_identity_field(plan.identity, "vector_iterations_per_lane_per_visit",
                          plan.vector_iterations_per_lane_per_visit);
    append_identity_field(plan.identity, "work_units", plan.work_units);
    append_identity_field(plan.identity, "paged_semantic_lookups", plan.paged_semantic_lookups);
    append_identity_field(plan.identity, "visit_bytes", request.visit_bytes);
    append_identity_field(plan.identity, "thread_execution_width", request.pipeline.thread_execution_width);
    append_identity_field(plan.identity, "max_total_threads_per_threadgroup",
                          request.pipeline.max_total_threads_per_threadgroup);
    append_identity_field(plan.identity, "threads_per_threadgroup_cap", request.limits.threads_per_threadgroup_cap);
    append_identity_field(plan.identity, "maximum_threadgroups_per_grid", request.limits.maximum_threadgroups_per_grid);
    append_identity_field(plan.identity, "maximum_owner_ordinals_per_threadgroup",
                          request.limits.maximum_owner_ordinals_per_threadgroup);
    append_identity_field(plan.identity, "maximum_vector_iterations_per_lane_per_visit",
                          request.limits.maximum_vector_iterations_per_lane_per_visit);
    append_identity_field(plan.identity, "maximum_paged_semantic_lookups_per_task",
                          request.limits.maximum_paged_semantic_lookups_per_task);
    append_identity_field(plan.identity, "maximum_work_units_per_dispatch",
                          request.limits.maximum_work_units_per_dispatch);
    append_identity_field(plan.identity, "minimum_threadgroup_accounted_bytes",
                          plan.minimum_threadgroup_accounted_bytes);
    append_identity_field(plan.identity, "maximum_threadgroup_accounted_bytes",
                          plan.maximum_threadgroup_accounted_bytes);
    append_identity_field(plan.identity, "threadgroup_accounted_imbalance_bytes",
                          plan.threadgroup_accounted_imbalance_bytes);
    append_identity_field(plan.identity, "threadgroup_accounted_bytes_count", plan.threadgroup_accounted_bytes.size());
    for (size_t accounted_bytes : plan.threadgroup_accounted_bytes) {
      append_identity_field(plan.identity, "threadgroup_accounted_bytes", accounted_bytes);
    }
  } catch (const std::bad_alloc&) {
    plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }
  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  return plan;
}

LlmMetalExecutionPlan build_llm_metal_execution_plan(const LlmMetalResourcePlanRequest& request) {
  LlmMetalExecutionPlan execution;
  LlmMetalResourcePlan& plan = execution.resources;
  plan.limits = request.limits;
  plan.argument_buffer_encoded_length = request.argument_buffer_encoded_length;
  plan.argument_buffer_alignment = request.argument_buffer_alignment;
  plan.max_buffer_length = request.max_buffer_length;
  plan.host_mapping_granularity_bytes = request.host_mapping_granularity_bytes;
  plan.additional_owned_bytes = request.additional_owned_bytes;
  plan.available_memory_bytes = request.available_memory_bytes;
  if (!request.geometry.valid) {
    execution.reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.host_mapping_granularity_bytes == 0) {
    execution.reason_code = LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  const bool decode_geometry = request.geometry.phase == LlmPhase::Decode &&
                               request.geometry.work_unit_kind == LlmWorkUnitKind::DecodeStep &&
                               request.geometry.decode.has_value() && !request.geometry.prefill.has_value();
  const bool prefill_geometry = request.geometry.phase == LlmPhase::Prefill &&
                                request.geometry.work_unit_kind == LlmWorkUnitKind::PrefillOperation &&
                                request.geometry.prefill.has_value() && !request.geometry.decode.has_value();
  if ((!decode_geometry && !prefill_geometry) ||
      (request.geometry.kv_layout != LlmKvLayout::Contiguous && request.geometry.kv_layout != LlmKvLayout::Paged)) {
    execution.reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.argument_buffer_encoded_length == 0) {
    execution.reason_code = LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (!is_power_of_two(request.argument_buffer_alignment)) {
    execution.reason_code = LlmMetalPlanReason::ARGUMENT_ENCODER_ALIGNMENT_INVALID;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.max_buffer_length < request.limits.segment_capacity_bytes ||
      request.argument_buffer_encoded_length > request.max_buffer_length ||
      plan.status_buffer_length > request.max_buffer_length) {
    execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  const LlmKvMetalSegmentLimits segment_limits = {request.limits.segment_capacity_bytes,
                                                  request.limits.segment_slots_per_pool};
  plan.weight_segments =
      build_llm_kv_segment_plan(request.geometry.active_weight_bytes_per_work_unit, 1, segment_limits);
  if (!plan.weight_segments.valid) {
    execution.reason_code = plan.weight_segments.reason_code == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                : plan.weight_segments.reason_code;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  if (request.geometry.kv_layout == LlmKvLayout::Paged) {
    if (!request.paged_layout.has_value()) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_REQUIRED;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    const LlmKvLayoutPlan& paged = *request.paged_layout;
    const size_t sequence_tokens =
        decode_geometry ? request.geometry.decode->visible_context_tokens : request.geometry.prefill->prompt_tokens;
    if (!paged.valid || paged.sequence_tokens != sequence_tokens ||
        paged.kv_block_tokens != request.geometry.kv_block_tokens ||
        paged.layer_count != request.geometry.layer_count || paged.batch_size != request.geometry.batch_size ||
        paged.k_or_v_record_bytes_per_layer != request.geometry.k_or_v_record_bytes_per_layer ||
        paged.total_physical_blocks != request.geometry.total_physical_blocks ||
        paged.block_bytes != request.geometry.kv_block_bytes ||
        paged.memory.k_physical_bytes != request.geometry.k_mapping_bytes ||
        paged.memory.v_physical_bytes != request.geometry.v_mapping_bytes ||
        paged.memory.block_table_bytes != request.geometry.block_table_bytes) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    const LlmKvMetalSegmentPlan paged_segments = build_llm_kv_metal_segment_plan(paged, segment_limits);
    if (!paged_segments.valid) {
      execution.reason_code = paged_segments.reason_code == LlmKvLayoutReason::SEGMENT_ELEMENT_EXCEEDS_CAPACITY
                                  ? LlmBackendReason::PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY
                              : paged_segments.reason_code == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                  ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                  : paged_segments.reason_code;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.k_segments = paged_segments.k_or_v_pool;
    plan.v_segments = paged_segments.k_or_v_pool;
    plan.table_segments = paged_segments.block_table;
    plan.paged_layout = paged;
    if (!NumericUtils::checked_round_up(paged.memory.block_table_bytes, request.host_mapping_granularity_bytes,
                                        plan.host_permutation_mapping_bytes)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.permutation_validation_bitset_bytes = paged.memory.validation_bitset_bytes;
    plan.staging_buffer_length =
        *std::max_element(plan.table_segments->segment_lengths.begin(), plan.table_segments->segment_lengths.end());
  } else {
    if (request.paged_layout.has_value()) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.k_segments = build_llm_kv_segment_plan(request.geometry.k_mapping_bytes, 1, segment_limits);
    plan.v_segments = build_llm_kv_segment_plan(request.geometry.v_mapping_bytes, 1, segment_limits);
    if (!plan.k_segments.valid || !plan.v_segments.valid) {
      const std::string& segment_reason =
          !plan.k_segments.valid ? plan.k_segments.reason_code : plan.v_segments.reason_code;
      execution.reason_code = segment_reason == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                  ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                  : segment_reason;
      plan.reason_code = execution.reason_code;
      return execution;
    }
  }

  if (!segment_plan_matches_length(plan.weight_segments, request.geometry.active_weight_bytes_per_work_unit,
                                   request.max_buffer_length) ||
      !segment_plan_matches_length(plan.k_segments, request.geometry.k_mapping_bytes, request.max_buffer_length) ||
      !segment_plan_matches_length(plan.v_segments, request.geometry.v_mapping_bytes, request.max_buffer_length) ||
      (plan.table_segments.has_value() &&
       !segment_plan_matches_length(*plan.table_segments, request.geometry.block_table_bytes,
                                    request.max_buffer_length)) ||
      plan.staging_buffer_length > request.max_buffer_length) {
    execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  plan.argument_buffer = build_llm_metal_argument_buffer_plan(
      plan.weight_segments.segment_count, plan.k_segments.segment_count, plan.v_segments.segment_count,
      plan.table_segments.has_value() ? plan.table_segments->segment_count : 0, request.limits.segment_slots_per_pool);
  if (!plan.argument_buffer.valid) {
    execution.reason_code = plan.argument_buffer.reason_code;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  try {
    const size_t table_count = plan.table_segments.has_value() ? plan.table_segments->segment_count : 0;
    const size_t staging_count = plan.staging_buffer_length == 0 ? 0 : 1;
    size_t resource_count = 0;
    if (!NumericUtils::checked_add(plan.weight_segments.segment_count, plan.k_segments.segment_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, plan.v_segments.segment_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, table_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, 2 + staging_count, resource_count)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.planned_resources.reserve(resource_count);
    append_planned_segments(LlmMetalResourcePool::Weight, plan.weight_segments, plan.planned_resources);
    append_planned_segments(LlmMetalResourcePool::K, plan.k_segments, plan.planned_resources);
    append_planned_segments(LlmMetalResourcePool::V, plan.v_segments, plan.planned_resources);
    if (plan.table_segments.has_value()) {
      append_planned_segments(LlmMetalResourcePool::BlockTable, *plan.table_segments, plan.planned_resources);
    }
    plan.planned_resources.push_back(
        {LlmMetalResourcePool::ArgumentBuffer, 0, plan.argument_buffer_encoded_length, true});
    plan.planned_resources.push_back({LlmMetalResourcePool::Status, 0, plan.status_buffer_length, true});
    if (plan.staging_buffer_length != 0) {
      plan.planned_resources.push_back({LlmMetalResourcePool::Staging, 0, plan.staging_buffer_length, false});
    }
  } catch (const std::bad_alloc&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  } catch (const std::length_error&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
    if (resource.persistent && !NumericUtils::checked_add(plan.persistent_resource_length_bytes, resource.length_bytes,
                                                          plan.persistent_resource_length_bytes)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
  }
  if (!NumericUtils::checked_add(plan.staging_buffer_length, plan.host_permutation_mapping_bytes,
                                 plan.transient_peak_bytes) ||
      !NumericUtils::checked_add(plan.transient_peak_bytes, plan.permutation_validation_bitset_bytes,
                                 plan.transient_peak_bytes) ||
      !NumericUtils::checked_add(plan.persistent_resource_length_bytes, plan.transient_peak_bytes,
                                 plan.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(plan.known_owned_peak_bytes, plan.additional_owned_bytes,
                                 plan.known_owned_peak_bytes) ||
      !calculate_admitted_budget(plan.available_memory_bytes, plan.admitted_budget_bytes)) {
    execution.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (plan.known_owned_peak_bytes > plan.admitted_budget_bytes) {
    execution.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  try {
    plan.identity = Constants::LLM_METAL_RESOURCE_PLAN_VERSION;
    append_identity_field(plan.identity, "phase", llm_phase_to_string(request.geometry.phase));
    append_identity_field(plan.identity, "kv_layout", llm_kv_layout_to_string(request.geometry.kv_layout));
    append_identity_field(plan.identity, "segment_capacity_bytes", request.limits.segment_capacity_bytes);
    append_identity_field(plan.identity, "segment_slots_per_pool", request.limits.segment_slots_per_pool);
    append_identity_field(plan.identity, "weight_segments", plan.weight_segments.identity);
    append_identity_field(plan.identity, "k_segments", plan.k_segments.identity);
    append_identity_field(plan.identity, "v_segments", plan.v_segments.identity);
    append_identity_field(
        plan.identity, "table_segments",
        plan.table_segments.has_value() ? plan.table_segments->identity : std::string_view{"not-applicable"});
    append_identity_field(
        plan.identity, "paged_layout_geometry",
        plan.paged_layout.has_value() ? plan.paged_layout->geometry_identity : std::string_view{"not-applicable"});
    append_identity_field(plan.identity, "argument_buffer", plan.argument_buffer.identity);
    append_identity_field(plan.identity, "argument_encoded_length", plan.argument_buffer_encoded_length);
    append_identity_field(plan.identity, "argument_alignment", plan.argument_buffer_alignment);
    append_identity_field(plan.identity, "max_buffer_length", plan.max_buffer_length);
    append_identity_field(plan.identity, "host_mapping_granularity_bytes", plan.host_mapping_granularity_bytes);
    append_identity_field(plan.identity, "status_length", plan.status_buffer_length);
    append_identity_field(plan.identity, "staging_length", plan.staging_buffer_length);
    append_identity_field(plan.identity, "persistent_resource_length", plan.persistent_resource_length_bytes);
    append_identity_field(plan.identity, "host_permutation_mapping_bytes", plan.host_permutation_mapping_bytes);
    append_identity_field(plan.identity, "permutation_validation_bitset_bytes",
                          plan.permutation_validation_bitset_bytes);
    append_identity_field(plan.identity, "additional_owned_bytes", plan.additional_owned_bytes);
    append_identity_field(plan.identity, "transient_peak_bytes", plan.transient_peak_bytes);
    append_identity_field(plan.identity, "known_owned_peak_bytes", plan.known_owned_peak_bytes);
    append_identity_field(plan.identity, "available_memory_bytes", plan.available_memory_bytes);
    append_identity_field(plan.identity, "admitted_budget_bytes", plan.admitted_budget_bytes);
    append_identity_field(plan.identity, "planned_resource_count", plan.planned_resources.size());
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      const std::string prefix = "resource_" + std::to_string(index) + "_";
      append_identity_field(plan.identity, prefix + "pool", llm_metal_resource_pool_to_string(resource.pool));
      append_identity_field(plan.identity, prefix + "pool_index", resource.pool_index);
      append_identity_field(plan.identity, prefix + "length_bytes", resource.length_bytes);
      append_identity_field(plan.identity, prefix + "persistent", resource.persistent ? 1U : 0U);
    }
    execution.msl_revision = LlmMetalKernelContract::kRevision;
    execution.msl_source_sha256 = canonical_llm_metal_kernel_source_sha256();
    execution.identity = plan.identity;
    append_identity_field(execution.identity, "msl_revision", execution.msl_revision);
    append_identity_field(execution.identity, "msl_source_sha256", execution.msl_source_sha256);
    append_identity_field(execution.identity, "foundation_parameter_abi",
                          LlmMetalKernelContract::kParameterAbiRevision);
    append_identity_field(execution.identity, "resource_table_abi", LlmMetalKernelContract::kResourceTableAbiRevision);
    append_identity_field(execution.identity, "threads_per_threadgroup_cap",
                          request.limits.threads_per_threadgroup_cap);
    append_identity_field(execution.identity, "maximum_threadgroups_per_grid",
                          request.limits.maximum_threadgroups_per_grid);
    append_identity_field(execution.identity, "maximum_owner_ordinals_per_threadgroup",
                          request.limits.maximum_owner_ordinals_per_threadgroup);
    append_identity_field(execution.identity, "maximum_vector_iterations_per_lane_per_visit",
                          request.limits.maximum_vector_iterations_per_lane_per_visit);
    append_identity_field(execution.identity, "maximum_paged_semantic_lookups_per_task",
                          request.limits.maximum_paged_semantic_lookups_per_task);
    append_identity_field(execution.identity, "maximum_work_units_per_dispatch",
                          request.limits.maximum_work_units_per_dispatch);
  } catch (const std::bad_alloc&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  } catch (const std::length_error&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  execution.valid = true;
  execution.reason_code = LlmMetalPlanReason::VALID;
  return execution;
}

LlmMetalCommittedAdmission evaluate_llm_metal_committed_admission(
    const LlmMetalResourcePlan& plan, const std::vector<LlmMetalAllocatedResource>& allocations) noexcept {
  LlmMetalCommittedAdmission result;
  result.transient_peak_bytes = plan.transient_peak_bytes;
  result.additional_owned_bytes = plan.additional_owned_bytes;
  result.admitted_budget_bytes = plan.admitted_budget_bytes;
  if (!plan.valid || allocations.size() != plan.planned_resources.size()) {
    result.reason_code = LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH;
    return result;
  }
  size_t allocated_resource_bytes = 0;
  size_t planned_resource_bytes = 0;
  for (size_t index = 0; index < allocations.size(); ++index) {
    const LlmMetalPlannedResource& planned = plan.planned_resources[index];
    const LlmMetalAllocatedResource& actual = allocations[index];
    if (!same_planned_resource(planned, actual)) {
      result.reason_code = LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH;
      return result;
    }
    const size_t committed =
        actual.allocated_size_bytes.has_value() ? *actual.allocated_size_bytes : actual.length_bytes;
    if (committed < actual.length_bytes ||
        !NumericUtils::checked_add(allocated_resource_bytes, committed, allocated_resource_bytes) ||
        !NumericUtils::checked_add(planned_resource_bytes, actual.length_bytes, planned_resource_bytes)) {
      result.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      return result;
    }
  }
  if (!checked_subtract(allocated_resource_bytes, planned_resource_bytes, result.resource_rounding_bytes) ||
      !NumericUtils::checked_add(allocated_resource_bytes, plan.host_permutation_mapping_bytes,
                                 result.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(result.known_owned_peak_bytes, plan.permutation_validation_bitset_bytes,
                                 result.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(result.known_owned_peak_bytes, plan.additional_owned_bytes,
                                 result.known_owned_peak_bytes)) {
    result.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_OVERFLOW;
    return result;
  }
  result.committed_resource_bytes = allocated_resource_bytes;
  if (result.known_owned_peak_bytes > result.admitted_budget_bytes) {
    result.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED;
    return result;
  }
  result.valid = true;
  result.reason_code = LlmMetalPlanReason::VALID;
  return result;
}

std::string canonical_llm_metal_kernel_source_sha256() {
  return HashUtils::sha256_hex(LlmMetalKernelContract::kSource);
}

bool validate_llm_metal_layout_probe(const LlmMetalFoundationParams& parameters, const LlmMetalLayoutProbeWords& words,
                                     uint64_t expected_observed_resource_value) noexcept {
  constexpr std::array<uint64_t, 10> kOffsets = {offsetof(LlmMetalFoundationParams, byte_count),
                                                 offsetof(LlmMetalFoundationParams, source_offset_bytes),
                                                 offsetof(LlmMetalFoundationParams, destination_offset_bytes),
                                                 offsetof(LlmMetalFoundationParams, logical_base_bytes),
                                                 offsetof(LlmMetalFoundationParams, pattern_seed),
                                                 offsetof(LlmMetalFoundationParams, block_bytes),
                                                 offsetof(LlmMetalFoundationParams, physical_blocks_per_layer),
                                                 offsetof(LlmMetalFoundationParams, pattern_kind),
                                                 offsetof(LlmMetalFoundationParams, probe_resource_kind),
                                                 offsetof(LlmMetalFoundationParams, probe_resource_slot)};
  const std::array<uint64_t, 10> values = {parameters.byte_count,
                                           parameters.source_offset_bytes,
                                           parameters.destination_offset_bytes,
                                           parameters.logical_base_bytes,
                                           parameters.pattern_seed,
                                           parameters.block_bytes,
                                           parameters.physical_blocks_per_layer,
                                           parameters.pattern_kind,
                                           parameters.probe_resource_kind,
                                           parameters.probe_resource_slot};
  if (words[LlmMetalKernelContract::kProbeAbiVersionIndex] != LlmMetalKernelContract::kFoundationParameterAbiVersion ||
      words[LlmMetalKernelContract::kProbeStructSizeIndex] != sizeof(LlmMetalFoundationParams) ||
      words[LlmMetalKernelContract::kProbeStructAlignmentIndex] != alignof(LlmMetalFoundationParams) ||
      words[LlmMetalKernelContract::kProbeFieldCountIndex] != kOffsets.size()) {
    return false;
  }
  for (size_t index = 0; index < kOffsets.size(); ++index) {
    if (words[LlmMetalKernelContract::kProbeFirstFieldOffsetIndex + index] != kOffsets[index] ||
        words[LlmMetalKernelContract::kProbeFirstFieldValueIndex + index] != values[index]) {
      return false;
    }
  }
  return words[LlmMetalKernelContract::kProbeObservedResourceValueIndex] == expected_observed_resource_value &&
         words[LlmMetalKernelContract::kProbeObservedResourceKindIndex] == parameters.probe_resource_kind &&
         words[LlmMetalKernelContract::kProbeObservedResourceSlotIndex] == parameters.probe_resource_slot &&
         words[LlmMetalKernelContract::kProbeArgumentBufferResourceCountIndex] ==
             LlmMetalKernelContract::kArgumentBufferResourceCount;
}

const char* llm_metal_resource_pool_to_string(LlmMetalResourcePool pool) noexcept {
  switch (pool) {
    case LlmMetalResourcePool::Weight:
      return "weight";
    case LlmMetalResourcePool::K:
      return "k";
    case LlmMetalResourcePool::V:
      return "v";
    case LlmMetalResourcePool::BlockTable:
      return "block_table";
    case LlmMetalResourcePool::ArgumentBuffer:
      return "argument_buffer";
    case LlmMetalResourcePool::Status:
      return "status_checksum";
    case LlmMetalResourcePool::Staging:
      return "staging";
  }
  return "unknown";
}

namespace {

struct FoundationPipelines {
  __strong id<MTLComputePipelineState> initialize = nil;
  __strong id<MTLComputePipelineState> copy = nil;
  __strong id<MTLComputePipelineState> probe = nil;
  __strong id<MTLComputePipelineState> validate_bytes = nil;
  __strong id<MTLComputePipelineState> validate_table = nil;
};

LlmMetalPipelineEvidence pipeline_evidence(id<MTLComputePipelineState> pipeline, const char* label) {
  LlmMetalPipelineEvidence evidence;
  evidence.label = label;
  if (pipeline != nil) {
    evidence.thread_execution_width = static_cast<size_t>(pipeline.threadExecutionWidth);
    evidence.max_total_threads_per_threadgroup = static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
  }
  return evidence;
}

LlmTaskIdentity metal_task_identity(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                    const LlmRunnerTaskContext& context) noexcept {
  LlmTaskIdentity identity;
  identity.backend = model_plan.backend;
  identity.phase = model_plan.phase;
  identity.kv_layout = model_plan.kv_layout;
  identity.work_unit_kind = scenario_plan.work_unit_kind;
  identity.kv_write_kind = scenario_plan.kv_write_kind;
  identity.task_kind = context.kind;
  identity.scenario = context.scenario;
  identity.attempt_index = context.attempt_index;
  identity.loop_index = context.loop_index;
  identity.order_position = context.order_position;
  identity.purpose = context.purpose;
  identity.model_plan_identity = model_plan.plan_identity;
  identity.scenario_plan_identity = scenario_plan.plan_identity;
  return identity;
}

class LlmMetalBackend final : public LlmBackend {
 public:
  explicit LlmMetalBackend(LlmMetalBackendTestHooks hooks) : hooks_(hooks) {
    evidence_.backend = LlmMemoryBackend::Metal;
    evidence_.backend_evidence = LlmMetalBackendEvidence{};
  }

  ~LlmMetalBackend() override {
    @autoreleasepool {
      release_owned_buffers();
      clear_runtime_objects();
    }
  }

  LlmMemoryBackend kind() const noexcept override { return LlmMemoryBackend::Metal; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmAuxiliaryPreflightView& preflight) const noexcept override {
    LlmBackendAuxiliaryEstimate estimate;
    if (preflight.backend != LlmMemoryBackend::Metal) {
      estimate.reason_code = LlmBackendReason::BACKEND_MISMATCH;
      return estimate;
    }
    estimate.reason_code = LlmBackendReason::BACKEND_NOT_ACTIVATED;
    return estimate;
  }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    LlmBackendAuxiliaryEstimate estimate;
    if (model_plan.backend != LlmMemoryBackend::Metal || get_llm_metal_execution_plan(model_plan) == nullptr) {
      estimate.reason_code = LlmBackendReason::BACKEND_MISMATCH;
      return estimate;
    }
    estimate.reason_code = LlmBackendReason::BACKEND_NOT_ACTIVATED;
    return estimate;
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig& config) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return initialize_impl(config);
        } catch (const std::exception& exception) {
          try {
            return fail_initialization(LlmBackendReason::BACKEND_INITIALIZATION_FAILED,
                                       internal_error(exception.what()));
          } catch (...) {
            return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
          }
        } catch (...) {
          return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
        }
      } @catch (NSException* exception) {
        try {
          return fail_initialization(LlmBackendReason::BACKEND_INITIALIZATION_FAILED,
                                     internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
        }
      }
    }
  }

  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan& model_plan) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return resolve_execution_plan_impl(model_plan);
        } catch (const std::exception& exception) {
          try {
            return fail_plan_resolution(LlmBackendReason::EXECUTION_PLAN_MISMATCH, internal_error(exception.what()));
          } catch (...) {
            return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
          }
        } catch (...) {
          return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
        }
      } @catch (NSException* exception) {
        try {
          return fail_plan_resolution(LlmBackendReason::EXECUTION_PLAN_MISMATCH,
                                      internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
        }
      }
    }
  }

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model_plan) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return prepare_resources_impl(model_plan);
        } catch (const std::exception& exception) {
          try {
            return fail_preparation(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED,
                                    internal_error(exception.what()));
          } catch (...) {
            return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
          }
        } catch (...) {
          return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
        }
      } @catch (NSException* exception) {
        try {
          return fail_preparation(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED,
                                  internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
        }
      }
    }
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) noexcept override {
    LlmTaskExecutionResult result{LlmTaskExecutionStatus::NotStarted, std::string{}};
    result.identity = metal_task_identity(model_plan, scenario_plan, context);
    result.completion.planned_work_units = scenario_plan.work_units;
    result.backend_evidence = LlmMetalTaskEvidence{};
    try {
      result.reason_code = LlmBackendReason::TASK_UNSUPPORTED;
      result.status = LlmTaskExecutionStatus::Unsupported;
    } catch (...) {
      result.reason_code.clear();
      result.status = LlmTaskExecutionStatus::Failed;
    }
    return result;
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    @autoreleasepool {
      @try {
        try {
          release_owned_buffers();
          resources_prepared_ = false;
          preparation_interrupted_ = false;
          plan_resolved_ = false;
          resolved_plan_identity_.clear();
          resolved_execution_plan_ = LlmMetalExecutionPlan{};
          LlmMetalBackendEvidence& metal = metal_evidence();
          metal.resources.current_allocated_size_after_release =
              device_ == nil ? 0 : static_cast<uint64_t>(device_.currentAllocatedSize);
          metal.resources.candidate_cleanup_completed = true;
          metal.resources.resources_published = false;
          evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
          return evidence_.release;
        } catch (...) {
          evidence_.release = {LlmBackendStatus::Failed, LlmBackendReason::RESOURCE_RELEASE_FAILED};
          return evidence_.release;
        }
      } @catch (NSException* exception) {
        static_cast<void>(exception);
        evidence_.release = {LlmBackendStatus::Failed, LlmBackendReason::RESOURCE_RELEASE_FAILED};
        return evidence_.release;
      }
    }
  }

 private:
  LlmMetalBackendEvidence& metal_evidence() noexcept {
    return std::get<LlmMetalBackendEvidence>(evidence_.backend_evidence);
  }

  const LlmMetalBackendEvidence& metal_evidence() const noexcept {
    return std::get<LlmMetalBackendEvidence>(evidence_.backend_evidence);
  }

  LlmBackendLifecycleResult fail_initialization(const char* reason, LlmMetalErrorDiagnostic error) {
    LlmMetalBackendEvidence& metal = metal_evidence();
    metal.capability.error = std::move(error);
    initialized_ = true;
    evidence_.initialization = {LlmBackendStatus::Failed, reason};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult fail_plan_resolution(const char* reason, LlmMetalErrorDiagnostic error) {
    metal_evidence().capability.error = std::move(error);
    evidence_.plan_resolution = {LlmBackendStatus::Failed, reason};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult fail_preparation(const char* reason, LlmMetalErrorDiagnostic error) {
    release_owned_buffers();
    LlmMetalResourceEvidence& resources = metal_evidence().resources;
    resources.error = std::move(error);
    resources.current_allocated_size_after_release =
        device_ == nil ? 0 : static_cast<uint64_t>(device_.currentAllocatedSize);
    resources.candidate_cleanup_completed = true;
    resources.resources_published = false;
    resources_prepared_ = false;
    evidence_.preparation = {LlmBackendStatus::Failed, reason};
    return evidence_.preparation;
  }

  static void clear_diagnostic(LlmMetalErrorDiagnostic& error) noexcept {
    error.domain.clear();
    error.code = 0;
    error.description.clear();
  }

  LlmBackendLifecycleResult fail_initialization_without_diagnostic(const char* reason) noexcept {
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->capability.error);
    }
    initialized_ = true;
    plan_resolved_ = false;
    resources_prepared_ = false;
    evidence_.initialization = {LlmBackendStatus::Failed, reason};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult fail_plan_resolution_without_diagnostic(const char* reason) noexcept {
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->capability.error);
    }
    evidence_.plan_resolution = {LlmBackendStatus::Failed, reason};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult fail_preparation_without_diagnostic(const char* reason) noexcept {
    release_owned_buffers();
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->resources.error);
      metal->resources.current_allocated_size_after_release = 0;
      metal->resources.candidate_cleanup_completed = true;
      metal->resources.resources_published = false;
    }
    resources_prepared_ = false;
    evidence_.preparation = {LlmBackendStatus::Failed, reason};
    return evidence_.preparation;
  }

  void clear_runtime_objects() noexcept {
    detach_argument_buffer();
    argument_encoder_ = nil;
    pipelines_ = FoundationPipelines{};
    library_ = nil;
    queue_ = nil;
    device_ = nil;
  }

  void detach_argument_buffer() noexcept {
    if (argument_encoder_ == nil) {
      return;
    }
    @try {
      [argument_encoder_ setArgumentBuffer:nil offset:0];
    } @catch (NSException* exception) {
      static_cast<void>(exception);
    }
  }

  void release_owned_buffers() noexcept {
    detach_argument_buffer();
    owned_buffers_ = nil;
  }

  void append_supported_families(std::vector<std::string>& families) {
    struct FamilyEntry {
      NSInteger raw_value;
      const char* name;
    };
    constexpr FamilyEntry kFamilies[] = {{1001, "apple1"},  {1002, "apple2"},  {1003, "apple3"}, {1004, "apple4"},
                                         {1005, "apple5"},  {1006, "apple6"},  {1007, "apple7"}, {1008, "apple8"},
                                         {1009, "apple9"},  {1010, "apple10"}, {2001, "mac1"},   {2002, "mac2"},
                                         {3001, "common1"}, {3002, "common2"}, {3003, "common3"}};
    for (const FamilyEntry& entry : kFamilies) {
      if ([device_ supportsFamily:static_cast<MTLGPUFamily>(entry.raw_value)]) {
        families.emplace_back(entry.name);
      }
    }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
    if (@available(macOS 13.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal3]) {
        families.emplace_back("metal3");
      }
    }
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal4]) {
        families.emplace_back("metal4");
      }
    }
#endif
  }

  id<MTLComputePipelineState> create_pipeline(const char* entrypoint, const char* label, NSError** error) {
    if (error != nullptr) {
      *error = nil;
    }
    NSString* const function_name = [NSString stringWithUTF8String:entrypoint];
    id<MTLFunction> function = [library_ newFunctionWithName:function_name];
    if (function == nil) {
      if (error != nullptr) {
        *error = [NSError
            errorWithDomain:@"macos-memory-benchmark.llm-metal"
                       code:1
                   userInfo:@{
                     NSLocalizedDescriptionKey : [NSString stringWithFormat:@"Missing function %@", function_name]
                   }];
      }
      return nil;
    }
    function.label = [NSString stringWithUTF8String:label];
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    if (descriptor == nil) {
      if (error != nullptr) {
        *error =
            [NSError errorWithDomain:@"macos-memory-benchmark.llm-metal"
                                code:2
                            userInfo:@{NSLocalizedDescriptionKey : @"MTLComputePipelineDescriptor allocation failed"}];
      }
      return nil;
    }
    descriptor.label = function.label;
    descriptor.computeFunction = function;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    return [device_ newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:error];
  }

  LlmBackendLifecycleResult initialize_impl(const LlmMemoryConfig& config) {
    release_owned_buffers();
    clear_runtime_objects();
    evidence_ = LlmBackendEvidence{};
    evidence_.backend = LlmMemoryBackend::Metal;
    evidence_.backend_evidence = LlmMetalBackendEvidence{};
    initialized_ = false;
    plan_resolved_ = false;
    resources_prepared_ = false;
    preparation_interrupted_ = false;
    resolved_plan_identity_.clear();
    resolved_execution_plan_ = LlmMetalExecutionPlan{};
    table_segment_first_bytes_.fill(0);
    table_segment_first_byte_valid_.fill(false);
    if (config.backend != LlmMemoryBackend::Metal) {
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::BACKEND_MISMATCH};
      return evidence_.initialization;
    }

    LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
    capability.kernel_revision = LlmMetalKernelContract::kRevision;
    capability.kernel_source_sha256 = canonical_llm_metal_kernel_source_sha256();
    capability.compiler_identifier = __clang_version__;
#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
    capability.build_sdk = version_macro_string(__MAC_OS_X_VERSION_MAX_ALLOWED);
#else
    capability.build_sdk = "unknown";
#endif
#ifdef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    capability.deployment_target = version_macro_string(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__);
#else
    capability.deployment_target = "unknown";
#endif

    LlmMetalCapabilityProbe probe;
    device_ = MTLCreateSystemDefaultDevice();
    probe.device_available = device_ != nil;
    if (device_ != nil) {
      capability.device_name = ns_string(device_.name);
      capability.registry_id = device_.registryID;
      capability.has_unified_memory = device_.hasUnifiedMemory;
      capability.required_apple7_family_supported = [device_ supportsFamily:MTLGPUFamilyApple7];
      capability.argument_buffers_tier2_supported = device_.argumentBuffersSupport == MTLArgumentBuffersTier2;
      capability.max_buffer_length = static_cast<size_t>(device_.maxBufferLength);
      capability.recommended_max_working_set_size = device_.recommendedMaxWorkingSetSize;
      append_supported_families(capability.supported_families);
      probe.has_unified_memory = capability.has_unified_memory;
      probe.apple7_family_supported = capability.required_apple7_family_supported;
      probe.argument_buffers_tier2_supported = capability.argument_buffers_tier2_supported;
      probe.max_buffer_length = capability.max_buffer_length;
    }
    LlmBackendLifecycleResult state = evaluate_llm_metal_capabilities(probe);
    if (state.status == LlmBackendStatus::Unsupported) {
      initialized_ = true;
      evidence_.initialization = state;
      return state;
    }

    queue_ = [device_ newCommandQueue];
    probe.command_queue_created = queue_ != nil;
    if (queue_ != nil) {
      queue_.label = @"membenchmark.llm-metal.command-queue";
    }
    state = evaluate_llm_metal_capabilities(probe);
    if (state.reason_code == LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED) {
      initialized_ = true;
      evidence_.initialization = state;
      return state;
    }

    MTLCompileOptions* options = [MTLCompileOptions new];
    if (options == nil) {
      capability.error = internal_error("MTLCompileOptions allocation failed");
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
      return evidence_.initialization;
    }
    options.languageVersion = MTLLanguageVersion2_3;
    options.preprocessorMacros = @{};
    NSString* source = [[NSString alloc] initWithBytes:LlmMetalKernelContract::kSource.data()
                                                length:LlmMetalKernelContract::kSource.size()
                                              encoding:NSUTF8StringEncoding];
    NSError* compile_error = nil;
    if (source != nil) {
      library_ = [device_ newLibraryWithSource:source options:options error:&compile_error];
    }
    if (compile_error != nil) {
      capability.compiler_diagnostics = bounded_diagnostic(ns_string(compile_error.localizedDescription));
    }
    probe.source_compiled = library_ != nil;
    if (!probe.source_compiled) {
      capability.error = error_diagnostic(compile_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
      return evidence_.initialization;
    }
    library_.label = @"membenchmark.llm-metal.runtime-library";

    NSError* pipeline_error = nil;
    pipelines_.initialize = create_pipeline(LlmMetalKernelContract::kInitializeBytesEntrypoint,
                                            "membenchmark.llm-metal.pipeline.initialize", &pipeline_error);
    if (pipelines_.initialize == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.copy = create_pipeline(LlmMetalKernelContract::kCopyBytesEntrypoint,
                                      "membenchmark.llm-metal.pipeline.copy", &pipeline_error);
    if (pipelines_.copy == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.probe = create_pipeline(LlmMetalKernelContract::kParameterLayoutProbeEntrypoint,
                                       "membenchmark.llm-metal.pipeline.layout-probe", &pipeline_error);
    if (pipelines_.probe == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.validate_bytes = create_pipeline(LlmMetalKernelContract::kValidateBytesEntrypoint,
                                                "membenchmark.llm-metal.pipeline.validate-bytes", &pipeline_error);
    if (pipelines_.validate_bytes == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.validate_table = create_pipeline(LlmMetalKernelContract::kValidateTableEntrypoint,
                                                "membenchmark.llm-metal.pipeline.validate-table", &pipeline_error);
    if (pipelines_.validate_table == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    const std::array<id<MTLComputePipelineState>, kFoundationPipelineCount> pipeline_array = {
        pipelines_.initialize, pipelines_.copy, pipelines_.probe, pipelines_.validate_bytes, pipelines_.validate_table};
    probe.foundation_pipeline_count =
        static_cast<size_t>(std::count_if(pipeline_array.begin(), pipeline_array.end(),
                                          [](id<MTLComputePipelineState> pipeline) { return pipeline != nil; }));
    if (probe.foundation_pipeline_count != kFoundationPipelineCount) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    capability.foundation_pipelines = {
        pipeline_evidence(pipelines_.initialize, "membenchmark.llm-metal.pipeline.initialize"),
        pipeline_evidence(pipelines_.copy, "membenchmark.llm-metal.pipeline.copy"),
        pipeline_evidence(pipelines_.probe, "membenchmark.llm-metal.pipeline.layout-probe"),
        pipeline_evidence(pipelines_.validate_bytes, "membenchmark.llm-metal.pipeline.validate-bytes"),
        pipeline_evidence(pipelines_.validate_table, "membenchmark.llm-metal.pipeline.validate-table")};

    id<MTLFunction> probe_function = [library_
        newFunctionWithName:[NSString stringWithUTF8String:LlmMetalKernelContract::kParameterLayoutProbeEntrypoint]];
    argument_encoder_ =
        [probe_function newArgumentEncoderWithBufferIndex:LlmMetalKernelContract::kProbeResourcesBufferIndex];
    probe.argument_encoder_created = argument_encoder_ != nil;
    if (argument_encoder_ != nil) {
      capability.argument_buffer_encoded_length = static_cast<size_t>(argument_encoder_.encodedLength);
      capability.argument_buffer_alignment = static_cast<size_t>(argument_encoder_.alignment);
      probe.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
      probe.argument_buffer_alignment = capability.argument_buffer_alignment;
    }
    state = evaluate_llm_metal_capabilities(probe);
    initialized_ = true;
    evidence_.initialization = state;
    return state;
  }

  LlmBackendLifecycleResult resolve_execution_plan_impl(const LlmMemoryWorkPlan& model_plan) {
    if (!initialized_ || evidence_.initialization.status != LlmBackendStatus::Ready) {
      evidence_.plan_resolution = {evidence_.initialization.status == LlmBackendStatus::Unsupported
                                       ? LlmBackendStatus::Unsupported
                                       : LlmBackendStatus::Failed,
                                   evidence_.initialization.status == LlmBackendStatus::Unsupported
                                       ? evidence_.initialization.reason_code
                                       : std::string_view{LlmBackendReason::NOT_INITIALIZED}};
      return evidence_.plan_resolution;
    }
    if (resources_prepared_) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.plan_resolution;
    }
    plan_resolved_ = false;
    resolved_plan_identity_.clear();
    resolved_execution_plan_ = LlmMetalExecutionPlan{};
    weight_seed_ = 0;
    k_seed_ = 0;
    v_seed_ = 0;
    base_seed_ = 0;
    const auto* metal_plan = get_llm_metal_execution_plan(model_plan);
    const LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
    const size_t runtime_host_mapping_granularity = get_system_page_size_bytes();
    if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal ||
        model_plan.phase != model_plan.geometry.phase || model_plan.kv_layout != model_plan.geometry.kv_layout ||
        model_plan.work_unit_kind != model_plan.geometry.work_unit_kind || metal_plan == nullptr ||
        !metal_plan->valid || metal_plan->identity.empty() || model_plan.plan_identity.empty() ||
        metal_plan->resources.argument_buffer_encoded_length != capability.argument_buffer_encoded_length ||
        metal_plan->resources.argument_buffer_alignment != capability.argument_buffer_alignment ||
        runtime_host_mapping_granularity == 0 ||
        metal_plan->resources.host_mapping_granularity_bytes != runtime_host_mapping_granularity ||
        metal_plan->resources.limits.segment_capacity_bytes != Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES ||
        metal_plan->resources.limits.segment_slots_per_pool != Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.plan_resolution;
    }

    LlmMetalResourcePlanRequest canonical_request;
    canonical_request.geometry = model_plan.geometry;
    if (model_plan.geometry.kv_layout == LlmKvLayout::Paged) {
      size_t sequence_tokens = 0;
      if (model_plan.geometry.phase == LlmPhase::Decode && model_plan.geometry.decode.has_value()) {
        sequence_tokens = model_plan.geometry.decode->visible_context_tokens;
      } else if (model_plan.geometry.phase == LlmPhase::Prefill && model_plan.geometry.prefill.has_value()) {
        sequence_tokens = model_plan.geometry.prefill->prompt_tokens;
      }
      canonical_request.paged_layout = build_llm_kv_layout_plan(
          {sequence_tokens, model_plan.geometry.kv_block_tokens, model_plan.geometry.layer_count,
           model_plan.geometry.batch_size, model_plan.geometry.k_or_v_record_bytes_per_layer});
    }
    canonical_request.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
    canonical_request.argument_buffer_alignment = capability.argument_buffer_alignment;
    canonical_request.max_buffer_length = capability.max_buffer_length;
    canonical_request.available_memory_bytes = metal_plan->resources.available_memory_bytes;
    canonical_request.host_mapping_granularity_bytes = runtime_host_mapping_granularity;
    canonical_request.additional_owned_bytes = metal_plan->resources.additional_owned_bytes;
    canonical_request.limits = metal_plan->resources.limits;
    LlmMetalExecutionPlan canonical = build_llm_metal_execution_plan(canonical_request);
    if (!canonical.valid || canonical.identity != metal_plan->identity ||
        !same_resource_plan(canonical.resources, metal_plan->resources) ||
        canonical.msl_revision != metal_plan->msl_revision ||
        canonical.msl_source_sha256 != metal_plan->msl_source_sha256 ||
        canonical.msl_source_sha256 != capability.kernel_source_sha256 ||
        canonical.msl_revision != capability.kernel_revision) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.plan_resolution;
    }
    resolved_execution_plan_ = std::move(canonical);
    resolved_plan_identity_ = model_plan.plan_identity;
    weight_seed_ = model_plan.weight_buffer_seed;
    k_seed_ = model_plan.k_buffer_seed;
    v_seed_ = model_plan.v_buffer_seed;
    base_seed_ = model_plan.base_seed;
    plan_resolved_ = true;
    evidence_.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.plan_resolution;
  }

  LlmMetalResourceMetadata build_resource_metadata(id<MTLBuffer> buffer, const LlmMetalPlannedResource& planned) const {
    LlmMetalResourceMetadata metadata;
    metadata.label = ns_string(buffer.label);
    metadata.pool = llm_metal_resource_pool_to_string(planned.pool);
    metadata.pool_index = planned.pool_index;
    metadata.storage_mode = storage_mode_string(buffer.storageMode);
    metadata.cpu_cache_mode = cpu_cache_mode_string(buffer.cpuCacheMode);
    metadata.hazard_tracking_mode = hazard_mode_string(buffer.hazardTrackingMode);
    metadata.resource_options = static_cast<uint64_t>(buffer.resourceOptions);
    metadata.length_bytes = static_cast<size_t>(buffer.length);
    if ([buffer respondsToSelector:@selector(allocatedSize)]) {
      metadata.allocated_size_available = true;
      metadata.allocated_size_bytes = static_cast<size_t>(buffer.allocatedSize);
      metadata.committed_bytes = metadata.allocated_size_bytes;
    } else {
      metadata.committed_bytes = metadata.length_bytes;
    }
    if (metadata.committed_bytes >= metadata.length_bytes) {
      metadata.allocation_rounding_bytes = metadata.committed_bytes - metadata.length_bytes;
    }
    return metadata;
  }

  id<MTLBuffer> find_buffer(NSArray<id<MTLBuffer>>* buffers, const std::vector<LlmMetalPlannedResource>& planned,
                            LlmMetalResourcePool pool, size_t pool_index = 0) const noexcept {
    for (size_t index = 0; index < planned.size(); ++index) {
      if (planned[index].pool == pool && planned[index].pool_index == pool_index) {
        return buffers[static_cast<NSUInteger>(index)];
      }
    }
    return nil;
  }

  bool dispatch_for_bytes(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                          size_t byte_count) const noexcept {
    if (encoder == nil || pipeline == nil || byte_count == 0 || pipeline.threadExecutionWidth == 0) {
      return false;
    }
    const size_t width = static_cast<size_t>(pipeline.threadExecutionWidth);
    const size_t thread_limit = std::min(static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup),
                                         Constants::LLM_METAL_THREADS_PER_THREADGROUP_CAP);
    const size_t threads = (thread_limit / width) * width;
    size_t vector_count = 0;
    size_t required_threadgroups = 0;
    if (threads == 0 || !checked_ceil_divide(byte_count, Constants::LLM_METAL_VECTOR_WIDTH_BYTES, vector_count) ||
        !checked_ceil_divide(std::max<size_t>(vector_count, 1), threads, required_threadgroups)) {
      return false;
    }
    const size_t threadgroups =
        std::max<size_t>(1, std::min(required_threadgroups, Constants::LLM_METAL_MAX_THREADGROUPS_PER_GRID));
    [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    return true;
  }

  bool commit_and_wait(id<MTLCommandBuffer> command_buffer, LlmMetalErrorDiagnostic& error) const {
    if (command_buffer == nil) {
      error = internal_error("commandBuffer returned nil");
      return false;
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      error = error_diagnostic(command_buffer.error);
      return false;
    }
    return true;
  }

  bool preparation_stop_requested() const {
    return hooks_.stop_requested ? hooks_.stop_requested() : signal_received();
  }

  bool mark_preparation_interrupted(LlmMetalErrorDiagnostic& error) {
    if (!preparation_stop_requested()) {
      return false;
    }
    preparation_interrupted_ = true;
    error = internal_error(LlmBackendReason::PREPARATION_INTERRUPTED);
    return true;
  }

  uint64_t seed_for_pool(LlmMetalResourcePool pool) const noexcept {
    switch (pool) {
      case LlmMetalResourcePool::Weight:
        return weight_seed_;
      case LlmMetalResourcePool::K:
        return k_seed_;
      case LlmMetalResourcePool::V:
        return v_seed_;
      case LlmMetalResourcePool::BlockTable:
      case LlmMetalResourcePool::ArgumentBuffer:
      case LlmMetalResourcePool::Status:
      case LlmMetalResourcePool::Staging:
        return base_seed_;
    }
    return 0;
  }

  uint8_t expected_data_byte(LlmMetalResourcePool pool, const LlmMetalResourcePlan& plan,
                             uint64_t absolute_byte) const noexcept {
    const uint64_t seed = seed_for_pool(pool);
    if ((pool == LlmMetalResourcePool::K || pool == LlmMetalResourcePool::V) && plan.paged_layout.has_value()) {
      return paged_pattern_byte(seed, plan.paged_layout->block_bytes,
                                static_cast<uint32_t>(plan.paged_layout->physical_blocks_per_layer), absolute_byte);
    }
    return contiguous_pattern_byte(seed, absolute_byte);
  }

  bool encode_argument_buffer(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                              LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> argument = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::ArgumentBuffer);
    if (argument == nil || argument.contents == nullptr || argument.length != plan.argument_buffer_encoded_length) {
      error = internal_error("argument buffer resource is invalid");
      return false;
    }
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      const size_t resource_id = argument_resource_id(resource.pool, resource.pool_index);
      if (resource_id != std::numeric_limits<size_t>::max() &&
          resource_id >= LlmMetalKernelContract::kArgumentBufferResourceCount) {
        error = internal_error("argument resource ID exceeds ABI");
        return false;
      }
    }
    [argument_encoder_ setArgumentBuffer:argument offset:0];
    for (size_t resource_id = 0; resource_id < LlmMetalKernelContract::kArgumentBufferResourceCount; ++resource_id) {
      [argument_encoder_ setBuffer:nil offset:0 atIndex:resource_id];
    }
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      const size_t resource_id = argument_resource_id(resource.pool, resource.pool_index);
      if (resource_id == std::numeric_limits<size_t>::max()) {
        continue;
      }
      [argument_encoder_ setBuffer:candidate[static_cast<NSUInteger>(index)] offset:0 atIndex:resource_id];
    }
    detach_argument_buffer();
    return true;
  }

  bool initialize_and_validate_data(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                                    LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    if (status == nil || status.contents == nullptr || status.length < sizeof(uint32_t)) {
      error = internal_error("status buffer resource is invalid");
      return false;
    }
    std::memset(status.contents, 0, static_cast<size_t>(status.length));
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      error = internal_error("initialization commandBuffer returned nil");
      return false;
    }
    command_buffer.label = @"membenchmark.llm-metal.command.initialize-validate";
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder == nil) {
      error = internal_error("initialization compute encoder returned nil");
      return false;
    }
    encoder.label = @"membenchmark.llm-metal.encoder.initialize-validate";

    std::array<size_t, 3> logical_bases{};
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (mark_preparation_interrupted(error)) {
        [encoder endEncoding];
        return false;
      }
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      size_t logical_base_index = 0;
      if (resource.pool == LlmMetalResourcePool::K) {
        logical_base_index = 1;
      } else if (resource.pool == LlmMetalResourcePool::V) {
        logical_base_index = 2;
      } else if (resource.pool != LlmMetalResourcePool::Weight) {
        continue;
      }
      id<MTLBuffer> buffer = candidate[static_cast<NSUInteger>(index)];
      LlmMetalFoundationParams parameters;
      parameters.byte_count = resource.length_bytes;
      parameters.logical_base_bytes = logical_bases[logical_base_index];
      parameters.pattern_seed = seed_for_pool(resource.pool);
      if (resource.pool != LlmMetalResourcePool::Weight && plan.paged_layout.has_value()) {
        parameters.pattern_kind = LlmMetalKernelContract::kPagedPatternKind;
        parameters.block_bytes = plan.paged_layout->block_bytes;
        parameters.physical_blocks_per_layer = static_cast<uint32_t>(plan.paged_layout->physical_blocks_per_layer);
      } else {
        parameters.pattern_kind = LlmMetalKernelContract::kContiguousPatternKind;
      }
      [encoder setComputePipelineState:pipelines_.initialize];
      [encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kInitializeDestinationBufferIndex];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kInitializeParametersBufferIndex];
      if (!dispatch_for_bytes(encoder, pipelines_.initialize, resource.length_bytes)) {
        [encoder endEncoding];
        error = internal_error("invalid initialization dispatch geometry");
        return false;
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:pipelines_.validate_bytes];
      [encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kByteValidationSourceBufferIndex];
      [encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kByteValidationStatusBufferIndex];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kByteValidationParametersBufferIndex];
      if (!dispatch_for_bytes(encoder, pipelines_.validate_bytes, resource.length_bytes)) {
        [encoder endEncoding];
        error = internal_error("invalid byte-validation dispatch geometry");
        return false;
      }
      if (!NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes,
                                     logical_bases[logical_base_index])) {
        [encoder endEncoding];
        error = internal_error("initialization logical-base overflow");
        return false;
      }
    }
    [encoder endEncoding];
    if (!commit_and_wait(command_buffer, error)) {
      return false;
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }
    const uint32_t status_word = *static_cast<const uint32_t*>(status.contents);
    if (status_word != 0 || hooks_.force_initialization_mismatch) {
      return false;
    }

    size_t data_resource_count = 0;
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      if (resource.pool == LlmMetalResourcePool::Weight || resource.pool == LlmMetalResourcePool::K ||
          resource.pool == LlmMetalResourcePool::V) {
        ++data_resource_count;
      }
    }
    size_t sample_bytes = 0;
    if (!NumericUtils::checked_multiply(data_resource_count, 2, sample_bytes) ||
        sample_bytes > static_cast<size_t>(status.length)) {
      error = internal_error("sample readback exceeds status buffer");
      return false;
    }
    id<MTLCommandBuffer> sample_command = [queue_ commandBuffer];
    if (sample_command == nil) {
      error = internal_error("sample readback commandBuffer returned nil");
      return false;
    }
    sample_command.label = @"membenchmark.llm-metal.command.sample-readback";
    id<MTLComputeCommandEncoder> sample_encoder =
        [sample_command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (sample_encoder == nil) {
      error = internal_error("sample readback compute encoder returned nil");
      return false;
    }
    sample_encoder.label = @"membenchmark.llm-metal.encoder.sample-readback";
    [sample_encoder setComputePipelineState:pipelines_.copy];
    size_t sample_index = 0;
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      if (resource.pool != LlmMetalResourcePool::Weight && resource.pool != LlmMetalResourcePool::K &&
          resource.pool != LlmMetalResourcePool::V) {
        continue;
      }
      if (mark_preparation_interrupted(error)) {
        [sample_encoder endEncoding];
        return false;
      }
      id<MTLBuffer> buffer = candidate[static_cast<NSUInteger>(index)];
      LlmMetalFoundationParams parameters;
      parameters.byte_count = 1;
      parameters.destination_offset_bytes = sample_index * 2;
      [sample_encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kCopySourceBufferIndex];
      [sample_encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kCopyDestinationBufferIndex];
      [sample_encoder setBytes:&parameters
                        length:sizeof(parameters)
                       atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
      if (!dispatch_for_bytes(sample_encoder, pipelines_.copy, parameters.byte_count)) {
        [sample_encoder endEncoding];
        error = internal_error("invalid first-byte sample dispatch geometry");
        return false;
      }
      parameters.source_offset_bytes = resource.length_bytes - 1;
      parameters.destination_offset_bytes = sample_index * 2 + 1;
      [sample_encoder setBytes:&parameters
                        length:sizeof(parameters)
                       atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
      if (!dispatch_for_bytes(sample_encoder, pipelines_.copy, parameters.byte_count)) {
        [sample_encoder endEncoding];
        error = internal_error("invalid last-byte sample dispatch geometry");
        return false;
      }
      ++sample_index;
    }
    [sample_encoder endEncoding];
    if (!commit_and_wait(sample_command, error)) {
      return false;
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }

    const auto* samples = static_cast<const uint8_t*>(status.contents);
    logical_bases = {};
    sample_index = 0;
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      size_t logical_base_index = 0;
      if (resource.pool == LlmMetalResourcePool::K) {
        logical_base_index = 1;
      } else if (resource.pool == LlmMetalResourcePool::V) {
        logical_base_index = 2;
      } else if (resource.pool != LlmMetalResourcePool::Weight) {
        continue;
      }
      size_t last_absolute_byte = 0;
      if (!NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes - 1,
                                     last_absolute_byte) ||
          samples[sample_index * 2] != expected_data_byte(resource.pool, plan, logical_bases[logical_base_index]) ||
          samples[sample_index * 2 + 1] != expected_data_byte(resource.pool, plan, last_absolute_byte) ||
          !NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes,
                                     logical_bases[logical_base_index])) {
        error = internal_error("CPU sample readback mismatch");
        return false;
      }
      ++sample_index;
    }
    return sample_index == data_resource_count;
  }

  bool upload_and_validate_table(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                                 LlmMetalResourceEvidence& evidence, LlmMetalErrorDiagnostic& error) {
    if (!plan.paged_layout.has_value()) {
      evidence.table_upload_completed = true;
      evidence.table_validation_completed = true;
      return true;
    }
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    id<MTLBuffer> staging = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Staging);
    if (status == nil || status.contents == nullptr || staging == nil || staging.contents == nullptr ||
        staging.length != plan.staging_buffer_length) {
      error = internal_error("table upload resources are invalid");
      return false;
    }
    MmapPtr host_table =
        allocate_buffer(plan.host_permutation_mapping_bytes, "LLM Metal paged KV host permutation table");
    if (host_table == nullptr) {
      error = internal_error("host permutation mapping failed");
      return false;
    }
    const LlmKvInPlaceBlockTableMaterialization materialization = materialize_llm_kv_block_table_in_place(
        *plan.paged_layout, derive_llm_kv_permutation_seed(base_seed_), static_cast<uint32_t*>(host_table.get()),
        plan.paged_layout->block_table_entries, Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES,
        [this]() { return preparation_stop_requested(); });
    if (!materialization.valid) {
      preparation_interrupted_ = materialization.reason_code == LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      error = internal_error(materialization.reason_code);
      return false;
    }
    evidence.table_permutation = materialization.permutation;

    const uint8_t* const host_bytes = static_cast<const uint8_t*>(host_table.get());
    constexpr size_t kUploadChunkBytes = Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES * sizeof(uint32_t);
    size_t table_byte_offset = 0;
    for (size_t segment_index = 0; segment_index < plan.table_segments->segment_count; ++segment_index) {
      const size_t length = plan.table_segments->segment_lengths[segment_index];
      id<MTLBuffer> table =
          find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::BlockTable, segment_index);
      if (table == nil || table.length != length || std::min(length, kUploadChunkBytes) > staging.length) {
        error = internal_error("table segment identity mismatch");
        return false;
      }
      table_segment_first_bytes_[segment_index] = host_bytes[table_byte_offset];
      table_segment_first_byte_valid_[segment_index] = true;
      size_t segment_byte_offset = 0;
      while (segment_byte_offset < length) {
        @autoreleasepool {
          if (mark_preparation_interrupted(error)) {
            return false;
          }
          const size_t chunk_bytes = std::min(kUploadChunkBytes, length - segment_byte_offset);
          std::memcpy(staging.contents, host_bytes + table_byte_offset, chunk_bytes);
          std::memset(status.contents, 0, static_cast<size_t>(status.length));
          LlmMetalFoundationParams parameters;
          parameters.byte_count = chunk_bytes;
          parameters.destination_offset_bytes = segment_byte_offset;

          id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
          id<MTLComputeCommandEncoder> encoder =
              [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
          if (command_buffer == nil || encoder == nil) {
            error = internal_error("table upload command creation failed");
            return false;
          }
          command_buffer.label = @"membenchmark.llm-metal.command.table-upload-validate";
          encoder.label = @"membenchmark.llm-metal.encoder.table-upload-validate";
          [encoder setComputePipelineState:pipelines_.copy];
          [encoder setBuffer:staging offset:0 atIndex:LlmMetalKernelContract::kCopySourceBufferIndex];
          [encoder setBuffer:table offset:0 atIndex:LlmMetalKernelContract::kCopyDestinationBufferIndex];
          [encoder setBytes:&parameters
                     length:sizeof(parameters)
                    atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
          if (!dispatch_for_bytes(encoder, pipelines_.copy, chunk_bytes)) {
            [encoder endEncoding];
            error = internal_error("invalid table-copy dispatch geometry");
            return false;
          }
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [encoder setComputePipelineState:pipelines_.validate_table];
          [encoder setBuffer:staging offset:0 atIndex:LlmMetalKernelContract::kTableValidationReferenceBufferIndex];
          [encoder setBuffer:table offset:0 atIndex:LlmMetalKernelContract::kTableValidationCandidateBufferIndex];
          [encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kTableValidationStatusBufferIndex];
          [encoder setBytes:&parameters
                     length:sizeof(parameters)
                    atIndex:LlmMetalKernelContract::kTableValidationParametersBufferIndex];
          if (!dispatch_for_bytes(encoder, pipelines_.validate_table, chunk_bytes)) {
            [encoder endEncoding];
            error = internal_error("invalid table-validation dispatch geometry");
            return false;
          }
          [encoder endEncoding];
          if (!commit_and_wait(command_buffer, error) || *static_cast<const uint32_t*>(status.contents) != 0) {
            if (error.description.empty()) {
              error = internal_error("GPU table validation mismatch");
            }
            return false;
          }

          id<MTLCommandBuffer> readback = [queue_ commandBuffer];
          id<MTLBlitCommandEncoder> blit = [readback blitCommandEncoder];
          if (readback == nil || blit == nil) {
            error = internal_error("table readback command creation failed");
            return false;
          }
          [blit copyFromBuffer:table
                   sourceOffset:segment_byte_offset
                       toBuffer:staging
              destinationOffset:0
                           size:chunk_bytes];
          [blit endEncoding];
          if (!commit_and_wait(readback, error) ||
              std::memcmp(staging.contents, host_bytes + table_byte_offset, chunk_bytes) != 0) {
            if (error.description.empty()) {
              error = internal_error("table readback mismatch");
            }
            return false;
          }
          if (!NumericUtils::checked_add(segment_byte_offset, chunk_bytes, segment_byte_offset) ||
              !NumericUtils::checked_add(table_byte_offset, chunk_bytes, table_byte_offset)) {
            error = internal_error("table upload offset overflow");
            return false;
          }
        }
      }
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }
    if (table_byte_offset != plan.paged_layout->memory.block_table_bytes) {
      error = internal_error("table upload byte count mismatch");
      return false;
    }
    evidence.table_upload_completed = true;
    evidence.table_validation_completed = true;
    return true;
  }

  bool run_layout_probe(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                        LlmMetalCapabilityEvidence& capability, LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> argument = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::ArgumentBuffer);
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    constexpr size_t kProbeOutputOffset = 256;
    if (argument == nil || status == nil || status.contents == nullptr ||
        status.length < kProbeOutputOffset + sizeof(LlmMetalLayoutProbeWords)) {
      error = internal_error("layout-probe resources are invalid");
      return false;
    }
    const size_t probe_width = static_cast<size_t>(pipelines_.probe.threadExecutionWidth);
    if (probe_width == 0 || probe_width > static_cast<size_t>(pipelines_.probe.maxTotalThreadsPerThreadgroup)) {
      error = internal_error("invalid layout-probe dispatch geometry");
      return false;
    }

    struct ProbeCase {
      LlmMetalResourcePool pool;
      size_t pool_index;
      uint32_t resource_kind;
      uint64_t expected_value;
    };
    std::array<ProbeCase, 9> probe_cases{};
    size_t probe_case_count = 0;
    const auto append_case = [&](LlmMetalResourcePool pool, size_t pool_index, uint32_t resource_kind,
                                 uint64_t expected_value) {
      if (probe_case_count >= probe_cases.size()) {
        return false;
      }
      probe_cases[probe_case_count++] = {pool, pool_index, resource_kind, expected_value};
      return true;
    };
    const auto append_data_pool_cases = [&](LlmMetalResourcePool pool, uint32_t resource_kind,
                                            const LlmKvSegmentPlan& segments) {
      if (!segments.valid || segments.segment_count == 0 || segments.segment_lengths.size() != segments.segment_count ||
          !append_case(pool, 0, resource_kind, expected_data_byte(pool, plan, 0))) {
        return false;
      }
      if (segments.segment_count == 1) {
        return true;
      }
      size_t logical_base = 0;
      for (size_t index = 0; index + 1 < segments.segment_count; ++index) {
        if (!NumericUtils::checked_add(logical_base, segments.segment_lengths[index], logical_base)) {
          return false;
        }
      }
      return append_case(pool, segments.segment_count - 1, resource_kind, expected_data_byte(pool, plan, logical_base));
    };
    if (!append_data_pool_cases(LlmMetalResourcePool::Weight, LlmMetalKernelContract::kProbeWeightResourceKind,
                                plan.weight_segments) ||
        !append_data_pool_cases(LlmMetalResourcePool::K, LlmMetalKernelContract::kProbeKeyResourceKind,
                                plan.k_segments) ||
        !append_data_pool_cases(LlmMetalResourcePool::V, LlmMetalKernelContract::kProbeValueResourceKind,
                                plan.v_segments)) {
      error = internal_error("layout-probe segment plan is invalid");
      return false;
    }
    if (plan.table_segments.has_value()) {
      if (!plan.table_segments->valid || plan.table_segments->segment_count == 0 ||
          plan.table_segments->segment_count > table_segment_first_bytes_.size()) {
        error = internal_error("table probe segment plan is invalid");
        return false;
      }
      const size_t last_table_segment = plan.table_segments->segment_count - 1;
      if (!table_segment_first_byte_valid_[0] ||
          !append_case(LlmMetalResourcePool::BlockTable, 0, LlmMetalKernelContract::kProbeTableResourceKind,
                       table_segment_first_bytes_[0])) {
        error = internal_error("table probe value is unavailable");
        return false;
      }
      if (last_table_segment != 0 && (!table_segment_first_byte_valid_[last_table_segment] ||
                                      !append_case(LlmMetalResourcePool::BlockTable, last_table_segment,
                                                   LlmMetalKernelContract::kProbeTableResourceKind,
                                                   table_segment_first_bytes_[last_table_segment]))) {
        error = internal_error("table probe value is unavailable");
        return false;
      }
    }
    if (!append_case(LlmMetalResourcePool::Status, 0, LlmMetalKernelContract::kProbeStatusChecksumResourceKind, 0)) {
      error = internal_error("layout-probe case capacity exceeded");
      return false;
    }

    capability.layout_probe_evaluated = true;
    for (size_t case_index = 0; case_index < probe_case_count; ++case_index) {
      @autoreleasepool {
        const ProbeCase& probe_case = probe_cases[case_index];
        id<MTLBuffer> resource = find_buffer(candidate, plan.planned_resources, probe_case.pool, probe_case.pool_index);
        if (resource == nil) {
          error = internal_error("layout-probe resource is unavailable");
          return false;
        }
        std::memset(status.contents, 0, static_cast<size_t>(status.length));
        LlmMetalFoundationParams parameters;
        parameters.byte_count = UINT64_C(0x0102030405060708);
        parameters.source_offset_bytes = UINT64_C(0x1112131415161718);
        parameters.destination_offset_bytes = UINT64_C(0x2122232425262728);
        parameters.logical_base_bytes = UINT64_C(0x3132333435363738);
        parameters.pattern_seed = UINT64_C(0x4142434445464748);
        parameters.block_bytes = UINT64_C(0x5152535455565758);
        parameters.physical_blocks_per_layer = UINT32_C(0x61626364);
        parameters.pattern_kind = UINT32_C(0x71727374);
        parameters.probe_resource_kind = probe_case.resource_kind;
        parameters.probe_resource_slot = static_cast<uint32_t>(probe_case.pool_index);

        id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
        if (command_buffer == nil) {
          error = internal_error("layout-probe command creation failed");
          return false;
        }
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (encoder == nil) {
          error = internal_error("layout-probe encoder creation failed");
          return false;
        }
        command_buffer.label = @"membenchmark.llm-metal.command.layout-probe";
        encoder.label = @"membenchmark.llm-metal.encoder.layout-probe";
        [encoder setComputePipelineState:pipelines_.probe];
        [encoder setBuffer:argument offset:0 atIndex:LlmMetalKernelContract::kProbeResourcesBufferIndex];
        [encoder setBytes:&parameters
                   length:sizeof(parameters)
                  atIndex:LlmMetalKernelContract::kProbeParametersBufferIndex];
        [encoder setBuffer:status offset:kProbeOutputOffset atIndex:LlmMetalKernelContract::kProbeOutputBufferIndex];
        [encoder useResource:resource usage:MTLResourceUsageRead];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(probe_width, 1, 1)];
        [encoder endEncoding];
        if (!commit_and_wait(command_buffer, error)) {
          return false;
        }
        LlmMetalLayoutProbeWords words{};
        std::memcpy(words.data(), static_cast<const uint8_t*>(status.contents) + kProbeOutputOffset, sizeof(words));
        if (!validate_llm_metal_layout_probe(parameters, words, probe_case.expected_value)) {
          error = internal_error("parameter layout probe mismatch");
          return false;
        }
      }
    }
    capability.layout_probe_valid = !hooks_.force_layout_probe_mismatch;
    capability.layout_probe_resource_count = probe_case_count;
    return capability.layout_probe_valid;
  }

  LlmBackendLifecycleResult prepare_resources_impl(const LlmMemoryWorkPlan& model_plan) {
    if (!initialized_ || !plan_resolved_ || evidence_.initialization.status != LlmBackendStatus::Ready ||
        model_plan.backend != LlmMemoryBackend::Metal || model_plan.plan_identity != resolved_plan_identity_) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.preparation;
    }
    const auto* submitted = get_llm_metal_execution_plan(model_plan);
    if (submitted == nullptr || !submitted->valid || submitted->identity != resolved_execution_plan_.identity ||
        submitted->resources.identity != resolved_execution_plan_.resources.identity) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.preparation;
    }
    if (resources_prepared_) {
      evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
      return evidence_.preparation;
    }
    const LlmMetalResourcePlan& plan = resolved_execution_plan_.resources;
    LlmMetalResourceEvidence resource_evidence;
    preparation_interrupted_ = false;
    metal_evidence().resources = resource_evidence;
    LlmMetalErrorDiagnostic phase_error;
    if (mark_preparation_interrupted(phase_error)) {
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }
    resource_evidence.allocation_attempted = true;
    resource_evidence.persistent_resource_length_bytes = plan.persistent_resource_length_bytes;
    if (plan.paged_layout.has_value() && !NumericUtils::checked_add(plan.paged_layout->memory.k_layout_padding_bytes,
                                                                    plan.paged_layout->memory.v_layout_padding_bytes,
                                                                    resource_evidence.layout_padding_bytes)) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.preparation;
    }
    resource_evidence.transient_peak_bytes = plan.transient_peak_bytes;
    resource_evidence.additional_owned_bytes = plan.additional_owned_bytes;
    resource_evidence.admitted_budget_bytes = plan.admitted_budget_bytes;
    resource_evidence.current_allocated_size_before = static_cast<uint64_t>(device_.currentAllocatedSize);
    resource_evidence.current_allocated_size_peak = resource_evidence.current_allocated_size_before;
    const uint64_t recommended = device_.recommendedMaxWorkingSetSize;
    if (recommended != 0) {
      resource_evidence.recommended_working_set_available = true;
    }
    metal_evidence().resources = resource_evidence;

    __strong NSMutableArray<id<MTLBuffer>>* candidate =
        [[NSMutableArray alloc] initWithCapacity:plan.planned_resources.size()];
    std::vector<LlmMetalAllocatedResource> actual_allocations;
    actual_allocations.reserve(plan.planned_resources.size());
    LlmMetalResourceEvidence& retained = metal_evidence().resources;
    if (candidate == nil) {
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("candidate container allocation failed");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    retained.resources.reserve(plan.planned_resources.size());
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (mark_preparation_interrupted(phase_error)) {
        candidate = nil;
        return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
      }
      if (index == hooks_.fail_allocation_after) {
        candidate = nil;
        retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
        retained.candidate_cleanup_completed = true;
        retained.error = internal_error("injected candidate allocation failure");
        evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
        return evidence_.preparation;
      }
      const LlmMetalPlannedResource& planned = plan.planned_resources[index];
      const MTLResourceOptions options = is_private_pool(planned.pool)
                                             ? MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked
                                             : MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
      id<MTLBuffer> buffer = [device_ newBufferWithLength:planned.length_bytes options:options];
      if (buffer == nil || (!is_private_pool(planned.pool) && buffer.contents == nullptr)) {
        candidate = nil;
        retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
        retained.candidate_cleanup_completed = true;
        retained.error = internal_error("newBufferWithLength returned nil");
        evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
        return evidence_.preparation;
      }
      const std::string label = std::string("membenchmark.llm-metal.") +
                                llm_metal_resource_pool_to_string(planned.pool) + "-" +
                                std::to_string(planned.pool_index);
      buffer.label = [NSString stringWithUTF8String:label.c_str()];
      [candidate addObject:buffer];
      retained.current_allocated_size_peak =
          std::max(retained.current_allocated_size_peak, static_cast<uint64_t>(device_.currentAllocatedSize));
      LlmMetalResourceMetadata metadata = build_resource_metadata(buffer, planned);
      LlmMetalAllocatedResource actual;
      actual.pool = planned.pool;
      actual.pool_index = planned.pool_index;
      actual.length_bytes = metadata.length_bytes;
      if (metadata.allocated_size_available) {
        actual.allocated_size_bytes = metadata.allocated_size_bytes;
      }
      retained.resources.push_back(std::move(metadata));
      actual_allocations.push_back(std::move(actual));
    }
    retained.allocation_completed = true;
    retained.current_allocated_size_peak =
        std::max(retained.current_allocated_size_peak, static_cast<uint64_t>(device_.currentAllocatedSize));
    const LlmMetalCommittedAdmission committed = evaluate_llm_metal_committed_admission(plan, actual_allocations);
    retained.committed_resource_bytes = committed.committed_resource_bytes;
    retained.resource_rounding_bytes = committed.resource_rounding_bytes;
    retained.known_owned_peak_bytes = committed.known_owned_peak_bytes;
    if (retained.recommended_working_set_available) {
      const long double headroom = static_cast<long double>(recommended) -
                                   static_cast<long double>(retained.current_allocated_size_before) -
                                   static_cast<long double>(retained.known_owned_peak_bytes);
      const long double clamped =
          std::max(static_cast<long double>(std::numeric_limits<int64_t>::min()),
                   std::min(headroom, static_cast<long double>(std::numeric_limits<int64_t>::max())));
      retained.recommended_working_set_headroom_bytes = static_cast<int64_t>(clamped);
      retained.recommended_working_set_headroom_fraction =
          static_cast<double>(headroom / static_cast<long double>(recommended));
      retained.exceeds_recommended_working_set = headroom < 0;
    }
    if (!committed.valid) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      evidence_.preparation = {LlmBackendStatus::Failed, committed.reason_code};
      return evidence_.preparation;
    }

    if (!encode_argument_buffer(candidate, plan, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
      return evidence_.preparation;
    }
    if (!upload_and_validate_table(candidate, plan, retained, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, preparation_interrupted_
                                                             ? LlmBackendReason::PREPARATION_INTERRUPTED
                                                             : LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED};
      return evidence_.preparation;
    }
    if (!initialize_and_validate_data(candidate, plan, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error =
          phase_error.description.empty() ? internal_error("private initialization mismatch") : std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, preparation_interrupted_
                                                             ? LlmBackendReason::PREPARATION_INTERRUPTED
                                                             : LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED};
      return evidence_.preparation;
    }
    retained.initialization_completed = true;
    retained.post_validation_completed = true;
    retained.cpu_sample_readback_validation_completed = true;
    if (mark_preparation_interrupted(phase_error)) {
      candidate = nil;
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }
    if (!run_layout_probe(candidate, plan, metal_evidence().capability, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error =
          phase_error.description.empty() ? internal_error("parameter layout probe mismatch") : std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
      return evidence_.preparation;
    }

    if (mark_preparation_interrupted(phase_error)) {
      candidate = nil;
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }

    __strong NSMutableArray<id<MTLBuffer>>* published = [[NSMutableArray alloc] init];
    if (published == nil) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("published container allocation failed");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    size_t persistent_count = 0;
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (plan.planned_resources[index].persistent) {
        [published addObject:candidate[static_cast<NSUInteger>(index)]];
        ++persistent_count;
      }
    }
    if (published.count != persistent_count) {
      published = nil;
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("published resource count mismatch");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    owned_buffers_ = published;
    candidate = nil;
    resources_prepared_ = true;
    retained.resources_published = true;
    retained.candidate_cleanup_completed = true;
    metal_evidence().timed_results_available = false;
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmMetalBackendTestHooks hooks_;
  bool initialized_ = false;
  bool plan_resolved_ = false;
  bool resources_prepared_ = false;
  bool preparation_interrupted_ = false;
  std::string resolved_plan_identity_;
  LlmMetalExecutionPlan resolved_execution_plan_;
  uint64_t weight_seed_ = 0;
  uint64_t k_seed_ = 0;
  uint64_t v_seed_ = 0;
  uint64_t base_seed_ = 0;
  std::array<uint8_t, Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL> table_segment_first_bytes_{};
  std::array<bool, Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL> table_segment_first_byte_valid_{};
  __strong id<MTLDevice> device_ = nil;
  __strong id<MTLCommandQueue> queue_ = nil;
  __strong id<MTLLibrary> library_ = nil;
  FoundationPipelines pipelines_;
  __strong id<MTLArgumentEncoder> argument_encoder_ = nil;
  __strong NSMutableArray<id<MTLBuffer>>* owned_buffers_ = nil;
  LlmBackendEvidence evidence_;
};

}  // namespace

std::unique_ptr<LlmBackend> create_llm_metal_backend() {
  return std::make_unique<LlmMetalBackend>(LlmMetalBackendTestHooks{});
}

std::unique_ptr<LlmBackend> create_llm_metal_backend_for_testing(const LlmMetalBackendTestHooks& hooks) {
  return std::make_unique<LlmMetalBackend>(hooks);
}
