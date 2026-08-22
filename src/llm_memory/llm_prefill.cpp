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
 * @file llm_prefill.cpp
 * @brief Pure checked planning primitives for synthetic LLM prefill work
 */

#include "llm_memory/llm_prefill.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "utils/numeric_utils.h"

namespace {

using WideSize = unsigned __int128;

constexpr size_t kBlockTableEntryBytes = sizeof(uint32_t);
constexpr uint64_t kPrefillPhaseDomain = 0x50524546494C4C31ULL;
constexpr uint64_t kOperationMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kLayerMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kBatchMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kWordMultiplier = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kVDomain = 0x5656565656565656ULL;

bool checked_subtract(size_t lhs, size_t rhs, size_t& output) noexcept {
  if (rhs > lhs) {
    return false;
  }
  output = lhs - rhs;
  return true;
}

bool checked_wide_add(WideSize lhs, WideSize rhs,
                      WideSize& output) noexcept {
  constexpr WideSize maximum = static_cast<WideSize>(-1);
  if (lhs > maximum - rhs) {
    return false;
  }
  output = lhs + rhs;
  return true;
}

bool checked_wide_multiply(WideSize lhs, WideSize rhs,
                           WideSize& output) noexcept {
  constexpr WideSize maximum = static_cast<WideSize>(-1);
  if (lhs != 0 && rhs > maximum / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

WideSize wide_triangular(size_t value) noexcept {
  if ((value & 1U) == 0) {
    return static_cast<WideSize>(value / 2) *
           (static_cast<WideSize>(value) + 1);
  }
  return static_cast<WideSize>(value) *
         (static_cast<WideSize>(value / 2) + 1);
}

/** Euclidean floor-sum reduction with a 128-bit checked accumulator. */
bool checked_wide_floor_sum(size_t count, size_t denominator, size_t slope,
                            size_t intercept, WideSize& output) noexcept {
  if (denominator == 0) {
    return false;
  }

  size_t n = count;
  size_t m = denominator;
  size_t a = slope;
  size_t b = intercept;
  WideSize answer = 0;
  while (true) {
    if (a >= m) {
      const size_t quotient = a / m;
      a %= m;
      WideSize contribution = 0;
      if (n != 0 &&
          (!checked_wide_multiply(wide_triangular(n - 1), quotient,
                                  contribution) ||
           !checked_wide_add(answer, contribution, answer))) {
        return false;
      }
    }
    if (b >= m) {
      const size_t quotient = b / m;
      b %= m;
      WideSize contribution = 0;
      if (!checked_wide_multiply(n, quotient, contribution) ||
          !checked_wide_add(answer, contribution, answer)) {
        return false;
      }
    }

    WideSize maximum_numerator = 0;
    if (!checked_wide_multiply(a, n, maximum_numerator) ||
        !checked_wide_add(maximum_numerator, b, maximum_numerator)) {
      return false;
    }
    if (maximum_numerator < m) {
      output = answer;
      return true;
    }

    const WideSize next_count = maximum_numerator / m;
    if (next_count > std::numeric_limits<size_t>::max()) {
      return false;
    }
    n = static_cast<size_t>(next_count);
    b = static_cast<size_t>(maximum_numerator % m);
    std::swap(m, a);
  }
}

bool valid_affine_domain(LlmPrefillKvDomain domain) noexcept {
  switch (domain) {
    case LlmPrefillKvDomain::K:
    case LlmPrefillKvDomain::V:
      return true;
  }
  return false;
}

uint64_t affine_domain(LlmPrefillKvDomain domain) noexcept {
  switch (domain) {
    case LlmPrefillKvDomain::K:
      return kKDomain;
    case LlmPrefillKvDomain::V:
      return kVDomain;
  }
  return 0;
}

bool valid_scenario(LlmScenario scenario) noexcept {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
    case LlmScenario::KvOnly:
    case LlmScenario::Mixed:
      return true;
  }
  return false;
}

bool valid_unit_kind(LlmPrefillPartitionUnitKind unit_kind) noexcept {
  switch (unit_kind) {
    case LlmPrefillPartitionUnitKind::ContiguousToken:
    case LlmPrefillPartitionUnitKind::PagedBlock:
      return true;
  }
  return false;
}

const char* partition_unit_kind_to_string(
    LlmPrefillPartitionUnitKind unit_kind) noexcept {
  switch (unit_kind) {
    case LlmPrefillPartitionUnitKind::ContiguousToken:
      return "contiguous_token";
    case LlmPrefillPartitionUnitKind::PagedBlock:
      return "paged_block";
  }
  return "invalid";
}

void append_prefill_identity_field(std::string& identity, const char* name,
                                   const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += value;
}

void append_prefill_identity_field(std::string& identity, const char* name,
                                   const char* value) {
  append_prefill_identity_field(identity, name, std::string(value));
}

template <typename Integer>
void append_prefill_identity_field(std::string& identity, const char* name,
                                   Integer value) {
  append_prefill_identity_field(identity, name, std::to_string(value));
}

std::string build_prefill_cpu_ownership_identity(
    const LlmPrefillPlan& prefill,
    const LlmPrefillCpuOwnershipPlan& plan) {
  std::string identity = LlmPrefillVersion::CPU_PARTITION;
  append_prefill_identity_field(identity, "planner_version",
                                LlmPrefillVersion::PLANNER);
  append_prefill_identity_field(identity, "schedule_version",
                                LlmPrefillVersion::OWNER_LOCAL_SCHEDULE);
  append_prefill_identity_field(identity, "unit_kind",
                                partition_unit_kind_to_string(plan.unit_kind));
  append_prefill_identity_field(identity, "scenario",
                                llm_scenario_to_string(plan.scenario));
  append_prefill_identity_field(identity, "active_weight_bytes",
                                prefill.active_weight_bytes);
  append_prefill_identity_field(identity, "prompt_tokens",
                                prefill.prompt_tokens);
  append_prefill_identity_field(identity, "query_tile_tokens",
                                prefill.attention_query_tile_tokens);
  append_prefill_identity_field(identity, "tile_count", prefill.tile_count);
  append_prefill_identity_field(identity, "layer_count", prefill.layer_count);
  append_prefill_identity_field(identity, "batch_size", prefill.batch_size);
  append_prefill_identity_field(identity, "query_head_count",
                                prefill.query_head_count);
  append_prefill_identity_field(identity, "head_dimension",
                                prefill.head_dimension);
  append_prefill_identity_field(identity, "k_or_v_record_bytes_per_layer",
                                prefill.k_or_v_record_bytes_per_layer);
  append_prefill_identity_field(identity, "paged", prefill.paged ? 1 : 0);
  append_prefill_identity_field(identity, "kv_block_tokens",
                                prefill.kv_block_tokens);
  append_prefill_identity_field(identity, "blocks_per_sequence",
                                prefill.blocks_per_sequence);
  append_prefill_identity_field(identity, "worker_count", plan.worker_count);
  append_prefill_identity_field(identity, "worker_rotation",
                                plan.worker_rotation);
  append_prefill_identity_field(identity, "logical_unit_count",
                                plan.logical_unit_count);
  append_prefill_identity_field(identity, "active_worker_count",
                                plan.active_worker_count);
  append_prefill_identity_field(identity, "weight_shards_included",
                                plan.weight_shards_included ? 1 : 0);
  append_prefill_identity_field(identity, "total_weight_shard_bytes",
                                plan.total_weight_shard_bytes);
  append_prefill_identity_field(identity, "total_kv_model_payload_bytes",
                                plan.total_kv_model_payload_bytes);
  append_prefill_identity_field(
      identity, "total_layout_metadata_lookup_count",
      plan.total_layout_metadata_lookup_count);
  append_prefill_identity_field(
      identity, "total_layout_metadata_read_bytes",
      plan.total_layout_metadata_read_bytes);
  append_prefill_identity_field(identity, "total_kv_accounted_bytes",
                                plan.total_kv_accounted_bytes);
  append_prefill_identity_field(identity, "total_scenario_accounted_bytes",
                                plan.total_scenario_accounted_bytes);
  append_prefill_identity_field(identity, "minimum_worker_accounted_bytes",
                                plan.minimum_worker_accounted_bytes);
  append_prefill_identity_field(identity, "maximum_worker_accounted_bytes",
                                plan.maximum_worker_accounted_bytes);
  append_prefill_identity_field(identity, "worker_accounted_imbalance_bytes",
                                plan.worker_accounted_imbalance_bytes);

  append_prefill_identity_field(identity, "worker_vector_count",
                                plan.worker_count);
  for (size_t worker = 0; worker < plan.worker_count; ++worker) {
    const std::string prefix = "worker_" + std::to_string(worker) + '_';
    append_prefill_identity_field(
        identity, (prefix + "weight_shard_bytes").c_str(),
        plan.worker_weight_shard_bytes[worker]);
    append_prefill_identity_field(
        identity, (prefix + "kv_model_payload_bytes").c_str(),
        plan.worker_kv_model_payload_bytes[worker]);
    append_prefill_identity_field(
        identity, (prefix + "layout_metadata_lookup_count").c_str(),
        plan.worker_layout_metadata_lookup_count[worker]);
    append_prefill_identity_field(
        identity, (prefix + "layout_metadata_read_bytes").c_str(),
        plan.worker_layout_metadata_read_bytes[worker]);
    append_prefill_identity_field(
        identity, (prefix + "scenario_accounted_bytes").c_str(),
        plan.worker_scenario_accounted_bytes[worker]);
  }

  append_prefill_identity_field(identity, "assignment_count",
                                plan.assignments.size());
  for (size_t index = 0; index < plan.assignments.size(); ++index) {
    const LlmPrefillCpuAssignment& assignment = plan.assignments[index];
    const std::string prefix = "assignment_" + std::to_string(index) + '_';
    append_prefill_identity_field(identity, (prefix + "range_rank").c_str(),
                                  assignment.range_rank);
    append_prefill_identity_field(identity, (prefix + "worker_index").c_str(),
                                  assignment.worker_index);
    append_prefill_identity_field(identity, (prefix + "first_unit").c_str(),
                                  assignment.first_unit);
    append_prefill_identity_field(identity, (prefix + "unit_count").c_str(),
                                  assignment.unit_count);
    append_prefill_identity_field(
        identity, (prefix + "kv_model_payload_bytes").c_str(),
        assignment.kv_cost.model_payload_bytes);
    append_prefill_identity_field(
        identity, (prefix + "layout_metadata_lookup_count").c_str(),
        assignment.kv_cost.layout_metadata_lookup_count);
    append_prefill_identity_field(
        identity, (prefix + "layout_metadata_read_bytes").c_str(),
        assignment.kv_cost.layout_metadata_read_bytes);
    append_prefill_identity_field(
        identity, (prefix + "kv_accounted_bytes").c_str(),
        assignment.kv_cost.accounted_bytes);
  }
  return identity;
}

LlmPrefillCpuOwnershipPlan invalid_prefill_cpu_ownership_plan(
    const LlmPrefillCpuOwnershipRequest& request,
    const std::string& reason_code) {
  LlmPrefillCpuOwnershipPlan plan;
  plan.reason_code = reason_code;
  plan.unit_kind = request.unit_kind;
  plan.scenario = request.scenario;
  plan.worker_count = request.worker_count;
  if (request.worker_count != 0) {
    plan.worker_rotation = request.worker_rotation % request.worker_count;
  }
  return plan;
}

/** Return exact data visits in the token prefix `[0, end_token)`. */
bool calculate_token_prefix_data_visits(const LlmPrefillPlan& plan,
                                        size_t end_token,
                                        size_t& output) noexcept {
  if (!plan.valid || end_token > plan.prompt_tokens) {
    return false;
  }

  WideSize floor_sum = 0;
  if (!checked_wide_floor_sum(
          end_token, plan.attention_query_tile_tokens, 1, 0, floor_sum)) {
    return false;
  }
  WideSize read_visits =
      static_cast<WideSize>(plan.tile_count) * end_token;
  if (floor_sum > read_visits) {
    return false;
  }
  read_visits -= floor_sum;
  WideSize data_visits = 0;
  if (!checked_wide_add(read_visits, end_token, data_visits) ||
      data_visits > std::numeric_limits<size_t>::max()) {
    return false;
  }
  output = static_cast<size_t>(data_visits);
  return true;
}

/** Return semantic paged lookups in block prefix `[0, end_block)`. */
bool calculate_block_prefix_lookups(const LlmPrefillPlan& plan,
                                    size_t end_block,
                                    size_t& output) noexcept {
  if (!plan.valid || !plan.paged || end_block > plan.blocks_per_sequence) {
    return false;
  }

  WideSize floors = 0;
  if (!checked_wide_floor_sum(
          end_block, plan.attention_query_tile_tokens, plan.kv_block_tokens,
          0, floors)) {
    return false;
  }
  WideSize read_block_visits =
      static_cast<WideSize>(plan.tile_count) * end_block;
  if (floors > read_block_visits) {
    return false;
  }
  read_block_visits -= floors;
  WideSize lookups = 0;
  if (!checked_wide_multiply(read_block_visits, 2, lookups) ||
      !checked_wide_add(lookups, end_block, lookups) ||
      lookups > std::numeric_limits<size_t>::max()) {
    return false;
  }
  output = static_cast<size_t>(lookups);
  return true;
}

LlmPrefillUnitRangeCost invalid_range_cost(
    LlmPrefillPartitionUnitKind unit_kind, const std::string& reason) {
  LlmPrefillUnitRangeCost result;
  result.unit_kind = unit_kind;
  result.reason_code = reason;
  return result;
}

LlmPrefillUnitRangeCost calculate_unit_range_cost(
    const LlmPrefillPlan& plan, LlmPrefillPartitionUnitKind unit_kind,
    size_t first_unit, size_t unit_count) {
  switch (unit_kind) {
    case LlmPrefillPartitionUnitKind::ContiguousToken:
      return calculate_llm_prefill_contiguous_token_range_cost(
          plan, first_unit, unit_count);
    case LlmPrefillPartitionUnitKind::PagedBlock:
      return calculate_llm_prefill_paged_block_range_cost(
          plan, first_unit, unit_count);
  }
  return invalid_range_cost(unit_kind, LlmPrefillReason::INVALID_UNIT_KIND);
}

bool calculate_prefix_accounted_bytes(
    const LlmPrefillPlan& plan, LlmPrefillPartitionUnitKind unit_kind,
    size_t boundary, size_t& output, std::string& failure_reason) {
  const LlmPrefillUnitRangeCost prefix =
      calculate_unit_range_cost(plan, unit_kind, 0, boundary);
  if (!prefix.valid) {
    failure_reason = prefix.reason_code;
    return false;
  }
  output = prefix.accounted_bytes;
  return true;
}

struct RationalTarget {
  size_t whole = 0;
  size_t remainder = 0;
  size_t denominator = 1;
};

/** Calculate rank * total / denominator without overflowing rank * total. */
RationalTarget calculate_partition_target(
    size_t rank, size_t total, size_t denominator,
    size_t preceding_weight_bytes) noexcept {
  const WideSize numerator =
      static_cast<WideSize>(rank) * static_cast<WideSize>(total);
  RationalTarget target;
  target.denominator = denominator;
  target.whole = static_cast<size_t>(numerator / denominator);
  target.remainder = static_cast<size_t>(numerator % denominator);
  if (target.whole < preceding_weight_bytes) {
    target.whole = 0;
    target.remainder = 0;
  } else {
    target.whole -= preceding_weight_bytes;
  }
  return target;
}

bool value_is_at_least_target(size_t value,
                              const RationalTarget& target) noexcept {
  return value > target.whole ||
         (value == target.whole && target.remainder == 0);
}

struct RationalDistance {
  size_t whole = 0;
  size_t remainder = 0;
};

RationalDistance distance_from_target(
    size_t value, const RationalTarget& target) noexcept {
  if (value <= target.whole) {
    return {target.whole - value, target.remainder};
  }
  if (target.remainder == 0) {
    return {value - target.whole, 0};
  }
  return {value - target.whole - 1,
          target.denominator - target.remainder};
}

bool distance_is_less(const RationalDistance& lhs,
                      const RationalDistance& rhs) noexcept {
  return lhs.whole < rhs.whole ||
         (lhs.whole == rhs.whole && lhs.remainder < rhs.remainder);
}

bool choose_partition_boundary(
    const LlmPrefillPlan& plan, LlmPrefillPartitionUnitKind unit_kind,
    size_t minimum_boundary, size_t maximum_boundary,
    const RationalTarget& target, size_t& output,
    std::string& failure_reason) {
  size_t low = minimum_boundary;
  size_t high = maximum_boundary;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    size_t cost = 0;
    if (!calculate_prefix_accounted_bytes(
            plan, unit_kind, middle, cost, failure_reason)) {
      return false;
    }
    if (value_is_at_least_target(cost, target)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }

  const size_t upper_boundary = low;
  const size_t lower_boundary =
      upper_boundary > minimum_boundary ? upper_boundary - 1
                                        : upper_boundary;
  size_t upper_cost = 0;
  size_t lower_cost = 0;
  if (!calculate_prefix_accounted_bytes(
          plan, unit_kind, upper_boundary, upper_cost, failure_reason) ||
      !calculate_prefix_accounted_bytes(
          plan, unit_kind, lower_boundary, lower_cost, failure_reason)) {
    return false;
  }
  const RationalDistance upper_distance =
      distance_from_target(upper_cost, target);
  const RationalDistance lower_distance =
      distance_from_target(lower_cost, target);
  output = distance_is_less(upper_distance, lower_distance)
               ? upper_boundary
               : lower_boundary;
  return true;
}

size_t rotated_worker(size_t rank, size_t rotation,
                      size_t worker_count) noexcept {
  const size_t normalized_rotation = rotation % worker_count;
  return rank >= worker_count - normalized_rotation
             ? rank - (worker_count - normalized_rotation)
             : rank + normalized_rotation;
}

bool add_event(std::vector<LlmPrefillSemanticEvent>& events,
               size_t maximum_events,
               const LlmPrefillSemanticEvent& event) {
  if (events.size() >= maximum_events) {
    return false;
  }
  events.push_back(event);
  return true;
}

bool calculate_unit_valid_tokens(
    const LlmPrefillPlan& plan, LlmPrefillPartitionUnitKind unit_kind,
    size_t unit_index, size_t& output) noexcept {
  if (unit_kind == LlmPrefillPartitionUnitKind::ContiguousToken) {
    if (unit_index >= plan.prompt_tokens) {
      return false;
    }
    output = 1;
    return true;
  }
  if (unit_kind != LlmPrefillPartitionUnitKind::PagedBlock || !plan.paged ||
      unit_index >= plan.blocks_per_sequence) {
    return false;
  }
  size_t block_start = 0;
  if (!NumericUtils::checked_multiply(
          unit_index, plan.kv_block_tokens, block_start) ||
      block_start >= plan.prompt_tokens) {
    return false;
  }
  output = std::min(plan.kv_block_tokens,
                    plan.prompt_tokens - block_start);
  return true;
}

uint64_t triangular_modulo_u64(uint64_t value) noexcept {
  return (value & 1U) == 0
             ? (value / 2) * (value + 1)
             : value * (value / 2 + 1);
}

uint64_t affine_subsequence_sum(uint64_t first_word,
                                size_t word_count) noexcept {
  if (word_count == 0) {
    return 0;
  }
  const uint64_t count = static_cast<uint64_t>(word_count);
  return count * first_word +
         (2 * kWordMultiplier) * triangular_modulo_u64(count - 1);
}

}  // namespace

bool checked_llm_prefill_ceil_divide(
    size_t value, size_t divisor, size_t& output) noexcept {
  if (divisor == 0) {
    return false;
  }
  const size_t quotient = value / divisor;
  const size_t remainder = value % divisor;
  size_t result = quotient;
  if (remainder != 0 &&
      !NumericUtils::checked_add(result, 1, result)) {
    return false;
  }
  output = result;
  return true;
}

bool checked_llm_prefill_triangular(
    size_t value, size_t& output) noexcept {
  size_t result = 0;
  if ((value & 1U) == 0) {
    size_t successor = 0;
    if (!NumericUtils::checked_add(value, 1, successor) ||
        !NumericUtils::checked_multiply(value / 2, successor, result)) {
      return false;
    }
  } else if (!NumericUtils::checked_multiply(
                 value, value / 2 + 1, result)) {
    return false;
  }
  output = result;
  return true;
}

bool checked_llm_prefill_floor_sum(
    size_t count, size_t denominator, size_t slope, size_t intercept,
    size_t& output) noexcept {
  WideSize result = 0;
  if (!checked_wide_floor_sum(
          count, denominator, slope, intercept, result) ||
      result > std::numeric_limits<size_t>::max()) {
    return false;
  }
  output = static_cast<size_t>(result);
  return true;
}

LlmPrefillPlan resolve_llm_prefill_plan(
    const LlmPrefillPlanRequest& request) {
  LlmPrefillPlan plan;
  plan.active_weight_bytes = request.active_weight_bytes;
  plan.prompt_tokens = request.prompt_tokens;
  plan.attention_query_tile_tokens =
      request.attention_query_tile_tokens;
  plan.layer_count = request.layer_count;
  plan.batch_size = request.batch_size;
  plan.query_head_count = request.query_head_count;
  plan.head_dimension = request.head_dimension;
  plan.k_or_v_record_bytes_per_layer =
      request.k_or_v_record_bytes_per_layer;
  plan.kv_block_tokens = request.kv_block_tokens;
  plan.paged = request.kv_block_tokens != 0;

  if (request.active_weight_bytes == 0) {
    plan.reason_code = LlmPrefillReason::ACTIVE_WEIGHT_BYTES_ZERO;
    return plan;
  }
  if (request.prompt_tokens == 0) {
    plan.reason_code = LlmPrefillReason::PROMPT_TOKENS_ZERO;
    return plan;
  }
  if (request.attention_query_tile_tokens == 0) {
    plan.reason_code = LlmPrefillReason::QUERY_TILE_TOKENS_ZERO;
    return plan;
  }
  if (request.attention_query_tile_tokens > request.prompt_tokens) {
    plan.reason_code =
        LlmPrefillReason::QUERY_TILE_TOKENS_EXCEEDS_PROMPT;
    return plan;
  }
  if (request.layer_count == 0) {
    plan.reason_code = LlmPrefillReason::LAYER_COUNT_ZERO;
    return plan;
  }
  if (request.batch_size == 0) {
    plan.reason_code = LlmPrefillReason::BATCH_SIZE_ZERO;
    return plan;
  }
  if (request.query_head_count == 0) {
    plan.reason_code = LlmPrefillReason::QUERY_HEAD_COUNT_ZERO;
    return plan;
  }
  if (request.head_dimension == 0) {
    plan.reason_code = LlmPrefillReason::HEAD_DIMENSION_ZERO;
    return plan;
  }
  if (request.k_or_v_record_bytes_per_layer == 0) {
    plan.reason_code = LlmPrefillReason::KV_RECORD_BYTES_ZERO;
    return plan;
  }

  plan.full_query_tile_count =
      request.prompt_tokens / request.attention_query_tile_tokens;
  plan.final_query_tile_tokens =
      request.prompt_tokens % request.attention_query_tile_tokens;
  plan.tile_count = plan.full_query_tile_count;
  if (plan.final_query_tile_tokens != 0 &&
      !NumericUtils::checked_add(plan.tile_count, 1, plan.tile_count)) {
    plan.reason_code = LlmPrefillReason::TILE_COUNT_OVERFLOW;
    return plan;
  }

  size_t full_tile_triangular = 0;
  size_t full_tile_visits = 0;
  if (!checked_llm_prefill_triangular(
          plan.full_query_tile_count, full_tile_triangular) ||
      !NumericUtils::checked_multiply(
          request.attention_query_tile_tokens, full_tile_triangular,
          full_tile_visits) ||
      (plan.final_query_tile_tokens != 0 &&
       !NumericUtils::checked_add(
           full_tile_visits, request.prompt_tokens, full_tile_visits))) {
    plan.reason_code = LlmPrefillReason::PREFIX_TOKEN_VISITS_OVERFLOW;
    return plan;
  }
  plan.attention_prefix_token_visits_per_sequence = full_tile_visits;

  if (!checked_llm_prefill_triangular(
          request.prompt_tokens,
          plan.causal_token_pairs_per_sequence)) {
    plan.reason_code = LlmPrefillReason::CAUSAL_TOKEN_PAIRS_OVERFLOW;
    return plan;
  }
  size_t layer_batch_count = 0;
  if (!NumericUtils::checked_multiply(
          request.layer_count, request.batch_size, layer_batch_count) ||
      !NumericUtils::checked_multiply(
          layer_batch_count, request.query_head_count,
          plan.logical_attention_pairs) ||
      !NumericUtils::checked_multiply(
          plan.logical_attention_pairs,
          plan.causal_token_pairs_per_sequence,
          plan.logical_attention_pairs)) {
    plan.reason_code =
        LlmPrefillReason::LOGICAL_ATTENTION_PAIRS_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          plan.logical_attention_pairs, request.head_dimension,
          plan.logical_attention_fma_terms)) {
    plan.reason_code =
        LlmPrefillReason::LOGICAL_ATTENTION_FMA_TERMS_OVERFLOW;
    return plan;
  }

  if (!NumericUtils::checked_multiply(
          request.k_or_v_record_bytes_per_layer, 2,
          plan.kv_record_bytes_per_layer) ||
      !NumericUtils::checked_multiply(
          request.layer_count, plan.kv_record_bytes_per_layer,
          plan.kv_bytes_per_token)) {
    plan.reason_code = LlmPrefillReason::KV_DATA_BYTES_OVERFLOW;
    return plan;
  }

  size_t logical_records = 0;
  if (!NumericUtils::checked_multiply(
          layer_batch_count, request.prompt_tokens, logical_records) ||
      !NumericUtils::checked_multiply(
          logical_records, request.k_or_v_record_bytes_per_layer,
          plan.k_logical_bytes)) {
    plan.reason_code = LlmPrefillReason::KV_LOGICAL_BYTES_OVERFLOW;
    return plan;
  }
  plan.v_logical_bytes = plan.k_logical_bytes;

  plan.weight_read_bytes_per_work_unit = request.active_weight_bytes;
  size_t batch_prompt_tokens = 0;
  if (!NumericUtils::checked_multiply(
          request.batch_size, request.prompt_tokens,
          batch_prompt_tokens) ||
      !NumericUtils::checked_multiply(
          batch_prompt_tokens, plan.kv_bytes_per_token,
          plan.kv_write_bytes_per_work_unit)) {
    plan.reason_code = LlmPrefillReason::KV_WRITE_BYTES_OVERFLOW;
    return plan;
  }
  size_t batch_prefix_visits = 0;
  if (!NumericUtils::checked_multiply(
          request.batch_size,
          plan.attention_prefix_token_visits_per_sequence,
          batch_prefix_visits) ||
      !NumericUtils::checked_multiply(
          batch_prefix_visits, plan.kv_bytes_per_token,
          plan.kv_read_bytes_per_work_unit)) {
    plan.reason_code = LlmPrefillReason::KV_READ_BYTES_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_add(
          plan.kv_write_bytes_per_work_unit,
          plan.kv_read_bytes_per_work_unit,
          plan.kv_only_payload_bytes_per_work_unit)) {
    plan.reason_code = LlmPrefillReason::KV_ONLY_PAYLOAD_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_add(
          plan.weight_read_bytes_per_work_unit,
          plan.kv_only_payload_bytes_per_work_unit,
          plan.mixed_payload_bytes_per_work_unit)) {
    plan.reason_code = LlmPrefillReason::MIXED_PAYLOAD_OVERFLOW;
    return plan;
  }

  if (plan.paged) {
    if (!checked_llm_prefill_ceil_divide(
            request.prompt_tokens, request.kv_block_tokens,
            plan.blocks_per_sequence)) {
      plan.reason_code = LlmPrefillReason::CEIL_DIVIDE_BY_ZERO;
      return plan;
    }
    size_t full_tile_block_floor_sum = 0;
    size_t full_tile_block_visits = 0;
    if (!checked_llm_prefill_floor_sum(
            plan.full_query_tile_count, request.kv_block_tokens,
            request.attention_query_tile_tokens,
            request.attention_query_tile_tokens - 1,
            full_tile_block_floor_sum) ||
        !NumericUtils::checked_add(
            plan.full_query_tile_count, full_tile_block_floor_sum,
            full_tile_block_visits) ||
        (plan.final_query_tile_tokens != 0 &&
         !NumericUtils::checked_add(
             full_tile_block_visits, plan.blocks_per_sequence,
             full_tile_block_visits))) {
      plan.reason_code =
          LlmPrefillReason::PREFIX_BLOCK_VISITS_OVERFLOW;
      return plan;
    }
    plan.prefix_block_visits_per_sequence = full_tile_block_visits;
    size_t read_lookups = 0;
    if (!NumericUtils::checked_multiply(
            plan.prefix_block_visits_per_sequence, 2, read_lookups) ||
        !NumericUtils::checked_add(
            plan.blocks_per_sequence, read_lookups,
            plan.layout_metadata_lookups_per_layer_sequence) ||
        !NumericUtils::checked_multiply(
            layer_batch_count,
            plan.layout_metadata_lookups_per_layer_sequence,
            plan.layout_metadata_lookups_per_work_unit)) {
      plan.reason_code = LlmPrefillReason::LOOKUP_COUNT_OVERFLOW;
      return plan;
    }
    if (!NumericUtils::checked_multiply(
            plan.layout_metadata_lookups_per_work_unit,
            kBlockTableEntryBytes,
            plan.layout_metadata_read_bytes_per_work_unit)) {
      plan.reason_code = LlmPrefillReason::LOOKUP_BYTES_OVERFLOW;
      return plan;
    }
  }

  plan.valid = true;
  plan.reason_code = LlmPrefillReason::VALID;
  return plan;
}

LlmPrefillUnitRangeCost calculate_llm_prefill_contiguous_token_range_cost(
    const LlmPrefillPlan& plan, size_t first_token, size_t token_count) {
  LlmPrefillUnitRangeCost result;
  result.unit_kind = LlmPrefillPartitionUnitKind::ContiguousToken;
  result.first_unit = first_token;
  result.unit_count = token_count;
  if (!plan.valid) {
    result.reason_code = plan.reason_code;
    return result;
  }
  size_t end_token = 0;
  if (!NumericUtils::checked_add(first_token, token_count, end_token) ||
      first_token > plan.prompt_tokens || end_token > plan.prompt_tokens) {
    result.reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
    return result;
  }

  size_t first_prefix_visits = 0;
  size_t end_prefix_visits = 0;
  if (!calculate_token_prefix_data_visits(
          plan, first_token, first_prefix_visits) ||
      !calculate_token_prefix_data_visits(
          plan, end_token, end_prefix_visits) ||
      !checked_subtract(end_prefix_visits, first_prefix_visits,
                        result.data_visit_count) ||
      !NumericUtils::checked_multiply(
          result.data_visit_count, plan.kv_record_bytes_per_layer,
          result.model_payload_bytes)) {
    result.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
    return result;
  }
  result.valid_token_count = token_count;
  result.accounted_bytes = result.model_payload_bytes;
  result.valid = true;
  result.reason_code = LlmPrefillReason::VALID;
  return result;
}

LlmPrefillUnitRangeCost calculate_llm_prefill_contiguous_token_cost(
    const LlmPrefillPlan& plan, size_t token_index) {
  return calculate_llm_prefill_contiguous_token_range_cost(
      plan, token_index, 1);
}

LlmPrefillUnitRangeCost calculate_llm_prefill_paged_block_range_cost(
    const LlmPrefillPlan& plan, size_t first_block, size_t block_count) {
  LlmPrefillUnitRangeCost result;
  result.unit_kind = LlmPrefillPartitionUnitKind::PagedBlock;
  result.first_unit = first_block;
  result.unit_count = block_count;
  if (!plan.valid) {
    result.reason_code = plan.reason_code;
    return result;
  }
  if (!plan.paged) {
    result.reason_code = LlmPrefillReason::PAGED_PLAN_REQUIRED;
    return result;
  }
  size_t end_block = 0;
  if (!NumericUtils::checked_add(first_block, block_count, end_block) ||
      first_block > plan.blocks_per_sequence ||
      end_block > plan.blocks_per_sequence) {
    result.reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
    return result;
  }

  size_t first_token = plan.prompt_tokens;
  if (first_block < plan.blocks_per_sequence &&
      !NumericUtils::checked_multiply(
          first_block, plan.kv_block_tokens, first_token)) {
    result.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
    return result;
  }
  size_t end_token = plan.prompt_tokens;
  if (end_block < plan.blocks_per_sequence &&
      !NumericUtils::checked_multiply(
          end_block, plan.kv_block_tokens, end_token)) {
    result.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
    return result;
  }

  size_t first_prefix_visits = 0;
  size_t end_prefix_visits = 0;
  size_t first_prefix_lookups = 0;
  size_t end_prefix_lookups = 0;
  if (!calculate_token_prefix_data_visits(
          plan, first_token, first_prefix_visits) ||
      !calculate_token_prefix_data_visits(
          plan, end_token, end_prefix_visits) ||
      !checked_subtract(end_prefix_visits, first_prefix_visits,
                        result.data_visit_count) ||
      !checked_subtract(end_token, first_token,
                        result.valid_token_count) ||
      !calculate_block_prefix_lookups(
          plan, first_block, first_prefix_lookups) ||
      !calculate_block_prefix_lookups(
          plan, end_block, end_prefix_lookups) ||
      !checked_subtract(end_prefix_lookups, first_prefix_lookups,
                        result.layout_metadata_lookup_count) ||
      !NumericUtils::checked_multiply(
          result.data_visit_count, plan.kv_record_bytes_per_layer,
          result.model_payload_bytes) ||
      !NumericUtils::checked_multiply(
          result.layout_metadata_lookup_count, kBlockTableEntryBytes,
          result.layout_metadata_read_bytes) ||
      !NumericUtils::checked_add(
          result.model_payload_bytes, result.layout_metadata_read_bytes,
          result.accounted_bytes)) {
    result.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
    return result;
  }

  result.valid = true;
  result.reason_code = LlmPrefillReason::VALID;
  return result;
}

LlmPrefillUnitRangeCost calculate_llm_prefill_paged_block_cost(
    const LlmPrefillPlan& plan, size_t block_index) {
  return calculate_llm_prefill_paged_block_range_cost(
      plan, block_index, 1);
}

LlmPrefillCpuOwnershipPlan build_llm_prefill_cpu_ownership_plan(
    const LlmPrefillPlan& prefill,
    const LlmPrefillCpuOwnershipRequest& request) {
  if (!prefill.valid) {
    return invalid_prefill_cpu_ownership_plan(request,
                                              prefill.reason_code);
  }
  if (!valid_unit_kind(request.unit_kind)) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::INVALID_UNIT_KIND);
  }
  if (!valid_scenario(request.scenario)) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::INVALID_SCENARIO);
  }
  if (request.worker_count == 0) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::WORKER_COUNT_ZERO);
  }
  if (request.unit_kind == LlmPrefillPartitionUnitKind::PagedBlock &&
      !prefill.paged) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PAGED_PLAN_REQUIRED);
  }
  if (!request.worker_weight_shard_bytes.empty() &&
      request.worker_weight_shard_bytes.size() != request.worker_count) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::WEIGHT_SHARD_COUNT_MISMATCH);
  }
  if (request.scenario == LlmScenario::WeightsOnly &&
      request.worker_weight_shard_bytes.empty()) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::WEIGHT_SHARDS_REQUIRED);
  }
  if (request.scenario == LlmScenario::KvOnly &&
      !request.worker_weight_shard_bytes.empty()) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::WEIGHT_SHARDS_NOT_APPLICABLE);
  }

  LlmPrefillCpuOwnershipPlan plan;
  plan.unit_kind = request.unit_kind;
  plan.scenario = request.scenario;
  plan.worker_count = request.worker_count;
  plan.worker_rotation = request.worker_rotation % request.worker_count;
  plan.weight_shards_included =
      !request.worker_weight_shard_bytes.empty();
  plan.logical_unit_count =
      request.unit_kind == LlmPrefillPartitionUnitKind::ContiguousToken
          ? prefill.prompt_tokens
          : prefill.blocks_per_sequence;

  try {
    plan.worker_weight_shard_bytes.assign(request.worker_count, 0);
    plan.worker_kv_model_payload_bytes.assign(request.worker_count, 0);
    plan.worker_layout_metadata_lookup_count.assign(request.worker_count, 0);
    plan.worker_layout_metadata_read_bytes.assign(request.worker_count, 0);
    plan.worker_scenario_accounted_bytes.assign(request.worker_count, 0);
    if (plan.weight_shards_included) {
      plan.worker_weight_shard_bytes =
          request.worker_weight_shard_bytes;
    }
  } catch (const std::bad_alloc&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  } catch (const std::length_error&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  }

  for (size_t worker = 0; worker < request.worker_count; ++worker) {
    if (!NumericUtils::checked_add(
            plan.total_weight_shard_bytes,
            plan.worker_weight_shard_bytes[worker],
            plan.total_weight_shard_bytes)) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
    }
    plan.worker_scenario_accounted_bytes[worker] =
        plan.worker_weight_shard_bytes[worker];
  }

  if (request.scenario == LlmScenario::WeightsOnly) {
    plan.logical_unit_count = 0;
    plan.total_scenario_accounted_bytes = plan.total_weight_shard_bytes;
    const auto range = std::minmax_element(
        plan.worker_scenario_accounted_bytes.begin(),
        plan.worker_scenario_accounted_bytes.end());
    plan.minimum_worker_accounted_bytes = *range.first;
    plan.maximum_worker_accounted_bytes = *range.second;
    plan.worker_accounted_imbalance_bytes =
        plan.maximum_worker_accounted_bytes -
        plan.minimum_worker_accounted_bytes;
    try {
      plan.identity = build_prefill_cpu_ownership_identity(prefill, plan);
    } catch (const std::bad_alloc&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    } catch (const std::length_error&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    }
    plan.valid = !plan.identity.empty();
    plan.reason_code = plan.valid ? LlmPrefillReason::VALID
                                  : LlmPrefillReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  const LlmPrefillUnitRangeCost total_kv_cost =
      calculate_unit_range_cost(
          prefill, request.unit_kind, 0, plan.logical_unit_count);
  if (!total_kv_cost.valid) {
    return invalid_prefill_cpu_ownership_plan(
        request, total_kv_cost.reason_code);
  }
  if (!NumericUtils::checked_add(
          plan.total_weight_shard_bytes, total_kv_cost.accounted_bytes,
          plan.total_scenario_accounted_bytes)) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
  }
  plan.active_worker_count =
      std::min(request.worker_count, plan.logical_unit_count);

  size_t active_weight_bytes = 0;
  for (size_t rank = 0; rank < plan.active_worker_count; ++rank) {
    const size_t worker = rotated_worker(
        rank, plan.worker_rotation, request.worker_count);
    if (!NumericUtils::checked_add(
            active_weight_bytes, plan.worker_weight_shard_bytes[worker],
            active_weight_bytes)) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
    }
  }
  size_t partitionable_cost = 0;
  if (!NumericUtils::checked_add(
          total_kv_cost.accounted_bytes, active_weight_bytes,
          partitionable_cost)) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
  }

  std::vector<size_t> boundaries;
  try {
    boundaries.reserve(plan.active_worker_count + 1);
    plan.assignments.reserve(plan.active_worker_count);
    boundaries.push_back(0);
  } catch (const std::bad_alloc&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  } catch (const std::length_error&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  }

  size_t preceding_weight_bytes = 0;
  for (size_t rank = 1; rank < plan.active_worker_count; ++rank) {
    const size_t preceding_worker = rotated_worker(
        rank - 1, plan.worker_rotation, request.worker_count);
    if (!NumericUtils::checked_add(
            preceding_weight_bytes,
            plan.worker_weight_shard_bytes[preceding_worker],
            preceding_weight_bytes)) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
    }
    const RationalTarget target = calculate_partition_target(
        rank, partitionable_cost, plan.active_worker_count,
        preceding_weight_bytes);
    const size_t minimum_boundary = boundaries.back() + 1;
    const size_t remaining_workers = plan.active_worker_count - rank;
    const size_t maximum_boundary =
        plan.logical_unit_count - remaining_workers;
    size_t boundary = 0;
    if (!choose_partition_boundary(
            prefill, request.unit_kind, minimum_boundary,
            maximum_boundary, target, boundary, plan.reason_code)) {
      return invalid_prefill_cpu_ownership_plan(request,
                                                plan.reason_code);
    }
    try {
      boundaries.push_back(boundary);
    } catch (const std::bad_alloc&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    } catch (const std::length_error&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    }
  }
  try {
    boundaries.push_back(plan.logical_unit_count);
  } catch (const std::bad_alloc&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  } catch (const std::length_error&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  }

  for (size_t rank = 0; rank < plan.active_worker_count; ++rank) {
    LlmPrefillCpuAssignment assignment;
    assignment.range_rank = rank;
    assignment.worker_index = rotated_worker(
        rank, plan.worker_rotation, request.worker_count);
    assignment.first_unit = boundaries[rank];
    assignment.unit_count = boundaries[rank + 1] - boundaries[rank];
    assignment.kv_cost = calculate_unit_range_cost(
        prefill, request.unit_kind, assignment.first_unit,
        assignment.unit_count);
    if (!assignment.kv_cost.valid) {
      return invalid_prefill_cpu_ownership_plan(
          request, assignment.kv_cost.reason_code);
    }
    const size_t worker = assignment.worker_index;
    if (!NumericUtils::checked_add(
            plan.worker_kv_model_payload_bytes[worker],
            assignment.kv_cost.model_payload_bytes,
            plan.worker_kv_model_payload_bytes[worker]) ||
        !NumericUtils::checked_add(
            plan.worker_layout_metadata_lookup_count[worker],
            assignment.kv_cost.layout_metadata_lookup_count,
            plan.worker_layout_metadata_lookup_count[worker]) ||
        !NumericUtils::checked_add(
            plan.worker_layout_metadata_read_bytes[worker],
            assignment.kv_cost.layout_metadata_read_bytes,
            plan.worker_layout_metadata_read_bytes[worker]) ||
        !NumericUtils::checked_add(
            plan.worker_scenario_accounted_bytes[worker],
            assignment.kv_cost.accounted_bytes,
            plan.worker_scenario_accounted_bytes[worker]) ||
        !NumericUtils::checked_add(
            plan.total_kv_model_payload_bytes,
            assignment.kv_cost.model_payload_bytes,
            plan.total_kv_model_payload_bytes) ||
        !NumericUtils::checked_add(
            plan.total_layout_metadata_lookup_count,
            assignment.kv_cost.layout_metadata_lookup_count,
            plan.total_layout_metadata_lookup_count) ||
        !NumericUtils::checked_add(
            plan.total_layout_metadata_read_bytes,
            assignment.kv_cost.layout_metadata_read_bytes,
            plan.total_layout_metadata_read_bytes) ||
        !NumericUtils::checked_add(
            plan.total_kv_accounted_bytes,
            assignment.kv_cost.accounted_bytes,
            plan.total_kv_accounted_bytes)) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
    }
    try {
      plan.assignments.push_back(std::move(assignment));
    } catch (const std::bad_alloc&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    } catch (const std::length_error&) {
      return invalid_prefill_cpu_ownership_plan(
          request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
    }
  }

  if (plan.total_kv_model_payload_bytes !=
          total_kv_cost.model_payload_bytes ||
      plan.total_layout_metadata_lookup_count !=
          total_kv_cost.layout_metadata_lookup_count ||
      plan.total_layout_metadata_read_bytes !=
          total_kv_cost.layout_metadata_read_bytes ||
      plan.total_kv_accounted_bytes != total_kv_cost.accounted_bytes ||
      plan.assignments.size() != plan.active_worker_count) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::OWNERSHIP_ACCOUNTING_MISMATCH);
  }

  const auto range = std::minmax_element(
      plan.worker_scenario_accounted_bytes.begin(),
      plan.worker_scenario_accounted_bytes.end());
  plan.minimum_worker_accounted_bytes = *range.first;
  plan.maximum_worker_accounted_bytes = *range.second;
  plan.worker_accounted_imbalance_bytes =
      plan.maximum_worker_accounted_bytes -
      plan.minimum_worker_accounted_bytes;
  try {
    plan.identity = build_prefill_cpu_ownership_identity(prefill, plan);
  } catch (const std::bad_alloc&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  } catch (const std::length_error&) {
    return invalid_prefill_cpu_ownership_plan(
        request, LlmPrefillReason::PLANNER_ALLOCATION_FAILED);
  }
  plan.valid = !plan.identity.empty();
  plan.reason_code = plan.valid ? LlmPrefillReason::VALID
                                : LlmPrefillReason::PLANNER_ALLOCATION_FAILED;
  return plan;
}

LlmPrefillSemanticTrace build_llm_prefill_semantic_trace(
    const LlmPrefillPlan& plan,
    const LlmPrefillSemanticTraceRequest& request) {
  LlmPrefillSemanticTrace trace;
  if (!plan.valid) {
    trace.reason_code = plan.reason_code;
    return trace;
  }
  if (!valid_unit_kind(request.unit_kind)) {
    trace.reason_code = LlmPrefillReason::INVALID_UNIT_KIND;
    return trace;
  }
  if (request.unit_kind == LlmPrefillPartitionUnitKind::PagedBlock &&
      !plan.paged) {
    trace.reason_code = LlmPrefillReason::PAGED_PLAN_REQUIRED;
    return trace;
  }
  if (request.maximum_events == 0) {
    trace.reason_code = LlmPrefillReason::SEMANTIC_EVENT_CAP_ZERO;
    return trace;
  }
  const size_t logical_unit_count =
      request.unit_kind == LlmPrefillPartitionUnitKind::ContiguousToken
          ? plan.prompt_tokens
          : plan.blocks_per_sequence;
  size_t range_end = 0;
  if (!NumericUtils::checked_add(
          request.first_unit, request.unit_count, range_end) ||
      request.first_unit > logical_unit_count ||
      range_end > logical_unit_count) {
    trace.reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
    return trace;
  }

  std::vector<LlmPrefillSemanticEvent> candidate;
  try {
    candidate.reserve(std::min(request.maximum_events,
                               static_cast<size_t>(64)));
    for (size_t unit = request.first_unit; unit < range_end; ++unit) {
      size_t valid_tokens = 0;
      if (!calculate_unit_valid_tokens(
              plan, request.unit_kind, unit, valid_tokens)) {
        trace.reason_code = LlmPrefillReason::INVALID_LOGICAL_RANGE;
        return trace;
      }
      for (const LlmPrefillKvDomain domain :
           {LlmPrefillKvDomain::K, LlmPrefillKvDomain::V}) {
        const LlmPrefillSemanticEvent event{
            LlmPrefillSemanticAccess::Write, domain, 0,
            plan.prompt_tokens, unit, valid_tokens};
        if (!add_event(candidate, request.maximum_events, event)) {
          trace.reason_code =
              LlmPrefillReason::SEMANTIC_EVENT_CAP_EXCEEDED;
          return trace;
        }
      }
    }

    if (request.unit_count != 0) {
      size_t first_logical_token = request.first_unit;
      if (request.unit_kind == LlmPrefillPartitionUnitKind::PagedBlock &&
          !NumericUtils::checked_multiply(
              request.first_unit, plan.kv_block_tokens,
              first_logical_token)) {
        trace.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
        return trace;
      }
      size_t tile_index =
          first_logical_token / plan.attention_query_tile_tokens;
      size_t tile_end = 0;
      if (!NumericUtils::checked_multiply(
              tile_index, plan.attention_query_tile_tokens, tile_end)) {
        trace.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
        return trace;
      }
      while (tile_end < plan.prompt_tokens) {
        const size_t remaining = plan.prompt_tokens - tile_end;
        tile_end += std::min(
            plan.attention_query_tile_tokens, remaining);
        size_t prefix_units = tile_end;
        if (request.unit_kind ==
            LlmPrefillPartitionUnitKind::PagedBlock) {
          if (!checked_llm_prefill_ceil_divide(
                  tile_end, plan.kv_block_tokens, prefix_units)) {
            trace.reason_code = LlmPrefillReason::CEIL_DIVIDE_BY_ZERO;
            return trace;
          }
        }
        const size_t visit_end = std::min(range_end, prefix_units);
        for (const LlmPrefillKvDomain domain :
             {LlmPrefillKvDomain::K, LlmPrefillKvDomain::V}) {
          for (size_t unit = request.first_unit; unit < visit_end; ++unit) {
            size_t unit_start_token = unit;
            if (request.unit_kind ==
                    LlmPrefillPartitionUnitKind::PagedBlock &&
                !NumericUtils::checked_multiply(
                    unit, plan.kv_block_tokens, unit_start_token)) {
              trace.reason_code = LlmPrefillReason::UNIT_COST_OVERFLOW;
              return trace;
            }
            const size_t visit_tokens =
                request.unit_kind ==
                        LlmPrefillPartitionUnitKind::ContiguousToken
                    ? 1
                    : std::min(plan.kv_block_tokens,
                               tile_end - unit_start_token);
            const LlmPrefillSemanticEvent event{
                LlmPrefillSemanticAccess::Read, domain, tile_index,
                tile_end, unit, visit_tokens};
            if (!add_event(candidate, request.maximum_events, event)) {
              trace.reason_code =
                  LlmPrefillReason::SEMANTIC_EVENT_CAP_EXCEEDED;
              return trace;
            }
          }
        }
        ++tile_index;
      }
    }
  } catch (const std::bad_alloc&) {
    trace.reason_code = LlmPrefillReason::PLANNER_ALLOCATION_FAILED;
    return trace;
  } catch (const std::length_error&) {
    trace.reason_code = LlmPrefillReason::PLANNER_ALLOCATION_FAILED;
    return trace;
  }

  trace.events = std::move(candidate);
  trace.valid = true;
  trace.reason_code = LlmPrefillReason::VALID;
  return trace;
}

uint64_t llm_prefill_affine64_word(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, uint64_t logical_word_index) noexcept {
  if (!valid_affine_domain(domain)) {
    return 0;
  }
  return scenario_seed + kPrefillPhaseDomain +
         kOperationMultiplier * (operation_ordinal + 1) +
         kLayerMultiplier * (layer_index + 1) +
         kBatchMultiplier * (batch_sequence_index + 1) +
         affine_domain(domain) +
         kWordMultiplier * (logical_word_index + 1);
}

uint8_t llm_prefill_affine64_byte(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, uint64_t logical_byte_index) noexcept {
  const uint64_t word = llm_prefill_affine64_word(
      scenario_seed, operation_ordinal, layer_index, batch_sequence_index,
      domain, logical_byte_index / sizeof(uint64_t));
  const unsigned int shift = static_cast<unsigned int>(
      (logical_byte_index % sizeof(uint64_t)) * 8);
  return static_cast<uint8_t>(word >> shift);
}

LlmPrefillAffine64Checksum calculate_llm_prefill_affine64_checksum(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, size_t first_logical_word,
    size_t logical_word_count) {
  LlmPrefillAffine64Checksum checksum;
  if (!valid_affine_domain(domain)) {
    checksum.reason_code = LlmPrefillReason::AFFINE_DOMAIN_INVALID;
    return checksum;
  }
  if (logical_word_count != 0 &&
      first_logical_word >
          std::numeric_limits<size_t>::max() -
              (logical_word_count - 1)) {
    checksum.reason_code =
        LlmPrefillReason::AFFINE_WORD_RANGE_OVERFLOW;
    return checksum;
  }

  checksum.exact_word_count = logical_word_count;
  const bool first_is_even = (first_logical_word & 1U) == 0;
  checksum.even_logical_word_count =
      logical_word_count / 2 +
      ((logical_word_count & 1U) != 0 && first_is_even ? 1 : 0);
  checksum.odd_logical_word_count =
      logical_word_count - checksum.even_logical_word_count;

  if (checksum.even_logical_word_count != 0) {
    const size_t first_even =
        first_is_even ? first_logical_word : first_logical_word + 1;
    checksum.even_logical_word_sum = affine_subsequence_sum(
        llm_prefill_affine64_word(
            scenario_seed, operation_ordinal, layer_index,
            batch_sequence_index, domain, first_even),
        checksum.even_logical_word_count);
  }
  if (checksum.odd_logical_word_count != 0) {
    const size_t first_odd =
        first_is_even ? first_logical_word + 1 : first_logical_word;
    checksum.odd_logical_word_sum = affine_subsequence_sum(
        llm_prefill_affine64_word(
            scenario_seed, operation_ordinal, layer_index,
            batch_sequence_index, domain, first_odd),
        checksum.odd_logical_word_count);
  }
  checksum.valid = true;
  checksum.reason_code = LlmPrefillReason::VALID;
  return checksum;
}

LlmPrefillAffine64Checksum calculate_llm_prefill_affine64_task_checksum(
    uint64_t scenario_seed, size_t operation_count, uint64_t layer_index,
    uint64_t batch_sequence_index, LlmPrefillKvDomain domain,
    size_t first_logical_word, size_t logical_word_count) {
  if (operation_count == 0) {
    LlmPrefillAffine64Checksum checksum;
    checksum.reason_code = LlmPrefillReason::OPERATION_COUNT_ZERO;
    return checksum;
  }
  const LlmPrefillAffine64Checksum first_operation =
      calculate_llm_prefill_affine64_checksum(
          scenario_seed, 0, layer_index, batch_sequence_index, domain,
          first_logical_word, logical_word_count);
  if (!first_operation.valid) {
    return first_operation;
  }

  LlmPrefillAffine64Checksum checksum;
  if (!NumericUtils::checked_multiply(
          first_operation.exact_word_count, operation_count,
          checksum.exact_word_count) ||
      !NumericUtils::checked_multiply(
          first_operation.even_logical_word_count, operation_count,
          checksum.even_logical_word_count) ||
      !NumericUtils::checked_multiply(
          first_operation.odd_logical_word_count, operation_count,
          checksum.odd_logical_word_count)) {
    checksum.reason_code = LlmPrefillReason::AFFINE_WORD_RANGE_OVERFLOW;
    return checksum;
  }

  const uint64_t operation_delta_sum =
      kOperationMultiplier * triangular_modulo_u64(
                                 static_cast<uint64_t>(operation_count - 1));
  checksum.even_logical_word_sum =
      static_cast<uint64_t>(operation_count) *
          first_operation.even_logical_word_sum +
      static_cast<uint64_t>(first_operation.even_logical_word_count) *
          operation_delta_sum;
  checksum.odd_logical_word_sum =
      static_cast<uint64_t>(operation_count) *
          first_operation.odd_logical_word_sum +
      static_cast<uint64_t>(first_operation.odd_logical_word_count) *
          operation_delta_sum;
  checksum.valid = true;
  checksum.reason_code = LlmPrefillReason::VALID;
  return checksum;
}
