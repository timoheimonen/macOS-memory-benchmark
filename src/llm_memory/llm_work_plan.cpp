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
 * @file llm_work_plan.cpp
 * @brief Pure checked logical and backend-specific LLM work planning
 */

#include "llm_memory/llm_work_plan.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

constexpr uint64_t kWeightBufferSeedDomain = 0x4C4C4D5745494748ULL;
constexpr uint64_t kKBufferSeedDomain = 0x4C4C4D4B42554631ULL;
constexpr uint64_t kVBufferSeedDomain = 0x4C4C4D5642554631ULL;
constexpr uint64_t kWeightsOnlyScenarioSeedDomain = 0x4C4C4D5357454947ULL;
constexpr uint64_t kKvOnlyScenarioSeedDomain = 0x4C4C4D534B564F4EULL;
constexpr uint64_t kMixedScenarioSeedDomain = 0x4C4C4D534D495845ULL;

bool valid_kv_element_bytes(size_t bytes) {
  return bytes == 1 || bytes == 2 || bytes == 4;
}

bool json_integer_is_safe(size_t value) {
  return value <= Constants::LLM_JSON_MAX_SAFE_INTEGER;
}

LlmAttentionKind classify_attention(size_t query_heads, size_t kv_heads) {
  if (query_heads == kv_heads) {
    return LlmAttentionKind::Mha;
  }
  return kv_heads == 1 ? LlmAttentionKind::Mqa : LlmAttentionKind::Gqa;
}

size_t scenario_index(LlmScenario scenario) {
  return static_cast<size_t>(scenario);
}

bool valid_scenario(LlmScenario scenario) {
  return scenario_index(scenario) < kLlmScenarioCount;
}

/**
 * @brief Split one exact range while prioritizing aligned internal boundaries.
 *
 * Active returned spans are non-empty. If a particular layer or sequence is
 * smaller than the effective team, its exact bytes go to the lowest-indexed
 * workers and the remaining workers receive canonical empty ranges. When any
 * 32-byte-aligned absolute boundary can leave at least one byte for every
 * remaining active worker, the closest such boundary to a quotient/remainder
 * split is selected. The union is exactly the input range.
 */
size_t select_partition_boundary(size_t offset, size_t span,
                                 size_t active_workers, size_t base,
                                 size_t remainder, size_t boundary_index,
                                 size_t previous_local) noexcept {
  const size_t balanced_local =
      base * boundary_index + std::min(boundary_index, remainder);
  const size_t minimum_local = previous_local + 1;
  const size_t remaining_workers = active_workers - boundary_index;
  const size_t maximum_local = span - remaining_workers;
  size_t selected_local =
      std::clamp(balanced_local, minimum_local, maximum_local);

  const size_t minimum_absolute = offset + minimum_local;
  const size_t maximum_absolute = offset + maximum_local;
  const size_t remainder_to_alignment =
      minimum_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
  const size_t alignment_delta =
      remainder_to_alignment == 0
          ? 0
          : Constants::LLM_RANGE_ALIGNMENT_BYTES - remainder_to_alignment;
  if (minimum_absolute <=
      std::numeric_limits<size_t>::max() - alignment_delta) {
    const size_t first_aligned = minimum_absolute + alignment_delta;
    const size_t last_aligned =
        maximum_absolute -
        maximum_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
    if (first_aligned <= last_aligned) {
      const size_t balanced_absolute = offset + selected_local;
      size_t aligned_down =
          balanced_absolute -
          balanced_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
      aligned_down =
          std::clamp(aligned_down, first_aligned, last_aligned);
      size_t aligned_up = aligned_down;
      if (aligned_down < balanced_absolute &&
          aligned_down <=
              std::numeric_limits<size_t>::max() -
                  Constants::LLM_RANGE_ALIGNMENT_BYTES) {
        aligned_up = std::min(
            last_aligned,
            aligned_down + Constants::LLM_RANGE_ALIGNMENT_BYTES);
      }
      const size_t distance_down =
          balanced_absolute >= aligned_down
              ? balanced_absolute - aligned_down
              : aligned_down - balanced_absolute;
      const size_t distance_up =
          balanced_absolute >= aligned_up
              ? balanced_absolute - aligned_up
              : aligned_up - balanced_absolute;
      const size_t selected_absolute =
          distance_down <= distance_up ? aligned_down : aligned_up;
      selected_local = selected_absolute - offset;
    }
  }
  return selected_local;
}

std::vector<LlmByteRange> partition_range(size_t offset, size_t span,
                                          size_t worker_count) {
  std::vector<LlmByteRange> ranges;
  if (worker_count == 0) {
    return ranges;
  }

  ranges.reserve(worker_count);
  const size_t active_workers = std::min(span, worker_count);
  if (active_workers == 0) {
    ranges.resize(worker_count);
    return ranges;
  }
  const size_t base = span / active_workers;
  const size_t remainder = span % active_workers;
  size_t previous_local = 0;
  for (size_t boundary_index = 1; boundary_index < active_workers;
       ++boundary_index) {
    const size_t selected_local = select_partition_boundary(
        offset, span, active_workers, base, remainder, boundary_index,
        previous_local);

    ranges.push_back(
        {offset + previous_local, selected_local - previous_local});
    previous_local = selected_local;
  }
  ranges.push_back({offset + previous_local, span - previous_local});
  ranges.resize(worker_count);
  return ranges;
}

LlmByteRange intersect_ranges(const LlmByteRange& lhs,
                              const LlmByteRange& rhs) {
  const size_t lhs_end = lhs.offset_bytes + lhs.span_bytes;
  const size_t rhs_end = rhs.offset_bytes + rhs.span_bytes;
  const size_t start = std::max(lhs.offset_bytes, rhs.offset_bytes);
  const size_t end = std::min(lhs_end, rhs_end);
  return end > start ? LlmByteRange{start, end - start} : LlmByteRange{};
}

void append_identity_field(std::string& identity, const char* name,
                           const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += value;
}

void append_identity_field(std::string& identity, const char* name,
                           const char* value) {
  append_identity_field(identity, name, std::string(value));
}

template <typename Integer>
void append_identity_field(std::string& identity, const char* name,
                           Integer value) {
  append_identity_field(identity, name, std::to_string(value));
}

void append_component_identity(std::string& identity, const char* name,
                               const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += std::to_string(value.size());
  identity += ':';
  identity += value;
}

void append_component_identity(
    std::string& identity, const char* name,
    const std::optional<std::string>& value) {
  if (value.has_value()) {
    append_component_identity(identity, name, *value);
    return;
  }
  identity += '|';
  identity += name;
  identity += "=null";
}

/** Match a retained pipe-delimited identity without constructing strings. */
class IdentityMatcher {
 public:
  explicit IdentityMatcher(std::string_view retained) noexcept
      : retained_(retained) {}

  bool literal(std::string_view expected) noexcept {
    if (!valid_ || expected.size() > retained_.size() - cursor_ ||
        retained_.compare(cursor_, expected.size(), expected) != 0) {
      valid_ = false;
      return false;
    }
    cursor_ += expected.size();
    return true;
  }

  bool field(std::string_view name, std::string_view value) noexcept {
    return literal("|") && literal(name) && literal("=") && literal(value);
  }

  template <typename Integer>
  bool integer(Integer value) noexcept {
    std::array<char, std::numeric_limits<Integer>::digits10 + 3> encoded{};
    const auto result =
        std::to_chars(encoded.data(), encoded.data() + encoded.size(), value);
    return result.ec == std::errc{} && literal(std::string_view(
                                           encoded.data(),
                                           static_cast<size_t>(result.ptr -
                                                               encoded.data())));
  }

  template <typename Integer>
  bool integer_field(std::string_view name, Integer value) noexcept {
    return literal("|") && literal(name) && literal("=") && integer(value);
  }

  bool sha256_field(std::string_view name,
                    std::string_view value) noexcept {
    std::array<char, 64> encoded{};
    return HashUtils::sha256_hex_noalloc(value, encoded) &&
           field(name, std::string_view(encoded.data(), encoded.size()));
  }

  bool complete() const noexcept {
    return valid_ && cursor_ == retained_.size();
  }

 private:
  std::string_view retained_;
  size_t cursor_ = 0;
  bool valid_ = true;
};

std::string build_model_plan_identity(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    return {};
  }
  const LlmGeometry& geometry = plan.geometry;
  if ((plan.phase == LlmPhase::Decode &&
       (!geometry.decode.has_value() || geometry.prefill.has_value() ||
        plan.prefill_plan.has_value())) ||
      (plan.phase == LlmPhase::Prefill &&
       (geometry.decode.has_value() || !geometry.prefill.has_value() ||
        !plan.prefill_plan.has_value() || !plan.prefill_plan->valid))) {
    return {};
  }
  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "backend",
                        llm_memory_backend_to_string(plan.backend));
  append_identity_field(identity, "phase", llm_phase_to_string(plan.phase));
  append_identity_field(identity, "kv_layout",
                        llm_kv_layout_to_string(plan.kv_layout));
  append_identity_field(identity, "work_unit_kind",
                        llm_work_unit_kind_to_string(plan.work_unit_kind));
  append_identity_field(identity, "methodology", plan.methodology_version);
  append_identity_field(identity, "component_identity_size",
                        plan.component_identities.identity.size());
  append_identity_field(identity, "component_identity",
                        plan.component_identities.identity);
  append_identity_field(identity, "range_alignment",
                        Constants::LLM_RANGE_ALIGNMENT_BYTES);
  append_identity_field(identity, "weight_passes_per_work_unit",
                        plan.weight_passes_per_work_unit);
  append_identity_field(identity, "kv_replay_factor",
                        plan.kv_replay_factor);
  append_identity_field(identity, "weight",
                        geometry.active_weight_bytes_per_work_unit);
  append_identity_field(identity, "layers", geometry.layer_count);
  append_identity_field(identity, "query_heads", geometry.query_head_count);
  append_identity_field(identity, "kv_heads", geometry.kv_head_count);
  append_identity_field(identity, "query_heads_per_kv_head",
                        geometry.query_heads_per_kv_head);
  append_identity_field(identity, "attention_kind",
                        llm_attention_kind_to_string(geometry.attention_kind));
  append_identity_field(identity, "head_dim", geometry.head_dimension);
  append_identity_field(identity, "kv_element_bytes",
                        geometry.kv_element_bytes);
  if (geometry.decode.has_value()) {
    append_identity_field(identity, "decode_context",
                          geometry.decode->visible_context_tokens);
  }
  if (geometry.prefill.has_value()) {
    append_identity_field(identity, "prefill_planner_version",
                          LlmPrefillVersion::PLANNER);
    append_identity_field(identity, "prefill_cpu_partition_version",
                          LlmPrefillVersion::CPU_PARTITION);
    append_identity_field(identity, "prefill_prompt_tokens",
                          geometry.prefill->prompt_tokens);
    append_identity_field(identity, "prefill_query_tile_tokens",
                          geometry.prefill->attention_query_tile_tokens);
    append_identity_field(identity, "prefill_tile_count",
                          geometry.prefill->tile_count);
    append_identity_field(
        identity, "prefill_prefix_token_visits_per_sequence",
        geometry.prefill->attention_prefix_token_visits_per_sequence);
    append_identity_field(identity, "prefill_causal_token_pairs",
                          geometry.prefill->causal_token_pairs_per_sequence);
    append_identity_field(identity, "prefill_logical_attention_pairs",
                          geometry.prefill->logical_attention_pairs);
    append_identity_field(identity, "prefill_logical_attention_fma_terms",
                          geometry.prefill->logical_attention_fma_terms);
    append_identity_field(
        identity, "prefill_prefix_block_visits_per_sequence",
        geometry.prefill->paged_prefix_block_visits_per_sequence);
  }
  append_identity_field(identity, "batch", geometry.batch_size);
  append_identity_field(identity, "kv_vector_bytes",
                        geometry.kv_vector_bytes);
  append_identity_field(identity, "k_or_v_record_bytes_per_layer",
                        geometry.k_or_v_record_bytes_per_layer);
  append_identity_field(identity, "kv_record_bytes_per_layer",
                        geometry.kv_record_bytes_per_layer);
  append_identity_field(identity, "kv_bytes_per_visible_token",
                        geometry.kv_bytes_per_visible_token);
  append_identity_field(identity, "k_or_v_sequence_visible_bytes",
                        geometry.k_or_v_sequence_visible_bytes);
  append_identity_field(identity, "kv_block_tokens",
                        geometry.kv_block_tokens);
  append_identity_field(identity, "kv_blocks_per_sequence",
                        geometry.kv_blocks_per_sequence);
  append_identity_field(identity, "physical_blocks_per_layer",
                        geometry.physical_blocks_per_layer);
  append_identity_field(identity, "total_physical_blocks",
                        geometry.total_physical_blocks);
  append_identity_field(identity, "kv_block_bytes", geometry.kv_block_bytes);
  append_identity_field(identity, "last_block_tokens",
                        geometry.last_block_tokens);
  append_identity_field(identity, "last_block_valid_bytes",
                        geometry.last_block_valid_bytes);
  append_identity_field(identity, "decode_append_offset_in_last_block",
                        geometry.decode_append_offset_in_last_block);
  append_identity_field(identity, "k_logical_bytes", geometry.k_logical_bytes);
  append_identity_field(identity, "v_logical_bytes", geometry.v_logical_bytes);
  append_identity_field(identity, "k_layout_padding_bytes",
                        geometry.k_layout_padding_bytes);
  append_identity_field(identity, "v_layout_padding_bytes",
                        geometry.v_layout_padding_bytes);
  append_identity_field(identity, "block_table_entries",
                        geometry.block_table_entries);
  append_identity_field(identity, "block_table_bytes",
                        geometry.block_table_bytes);
  append_identity_field(
      identity, "layout_metadata_lookups_per_layer_sequence_per_work_unit",
      geometry.layout_metadata_lookups_per_layer_sequence_per_work_unit);
  append_identity_field(identity, "k_mapping_bytes",
                        geometry.k_mapping_bytes);
  append_identity_field(identity, "v_mapping_bytes",
                        geometry.v_mapping_bytes);
  append_identity_field(identity, "kv_capacity_bytes",
                        geometry.kv_capacity_bytes);
  append_identity_field(identity, "weight_read_bytes_per_work_unit",
                        geometry.weight_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_read_bytes_per_work_unit",
                        geometry.kv_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_write_bytes_per_work_unit",
                        geometry.kv_write_bytes_per_work_unit);
  append_identity_field(identity, "kv_only_payload_bytes_per_work_unit",
                        geometry.kv_only_effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity, "mixed_payload_bytes_per_work_unit",
                        geometry.mixed_effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity, "total_data_mapping_bytes",
                        geometry.total_data_mapping_bytes);
  append_identity_field(identity, "traffic_crossover_numerator",
                        geometry.traffic_crossover_numerator);
  append_identity_field(identity, "traffic_crossover_denominator",
                        geometry.traffic_crossover_denominator);
  append_identity_field(identity, "requested_workers",
                        cpu_plan->requested_workers);
  append_identity_field(identity, "effective_workers",
                        cpu_plan->effective_workers);
  append_identity_field(identity, "layer_descriptors_per_worker",
                        cpu_plan->layer_descriptors_per_worker);
  append_identity_field(identity, "sequence_descriptors_per_worker",
                        cpu_plan->sequence_descriptors_per_worker);
  append_identity_field(identity, "total_layer_descriptors",
                        cpu_plan->total_layer_descriptors);
  append_identity_field(identity, "total_sequence_descriptors",
                        cpu_plan->total_sequence_descriptors);
  append_identity_field(identity, "descriptor_bytes",
                        cpu_plan->descriptor_bytes);
  if (cpu_plan->prefill.has_value()) {
    append_identity_field(identity, "prefill_execution_identity_size",
                          cpu_plan->prefill->identity.size());
    append_identity_field(identity, "prefill_execution_identity",
                          cpu_plan->prefill->identity);
  }
  if (cpu_plan->paged.has_value()) {
    append_identity_field(identity, "paged_layout_identity_size",
                          cpu_plan->paged->layout_identity.size());
    append_identity_field(identity, "paged_layout_identity",
                          cpu_plan->paged->layout_identity);
    append_identity_field(identity, "paged_execution_identity_size",
                          cpu_plan->paged->execution_identity.size());
    append_identity_field(identity, "paged_execution_identity",
                          cpu_plan->paged->execution_identity);
  }
  append_identity_field(identity, "base_seed", plan.base_seed);
  append_identity_field(identity, "weight_buffer_seed",
                        plan.weight_buffer_seed);
  append_identity_field(identity, "k_buffer_seed", plan.k_buffer_seed);
  append_identity_field(identity, "v_buffer_seed", plan.v_buffer_seed);
  append_identity_field(identity, "weights_only_scenario_seed",
                        plan.scenario_seeds[0]);
  append_identity_field(identity, "kv_only_scenario_seed",
                        plan.scenario_seeds[1]);
  append_identity_field(identity, "mixed_scenario_seed",
                        plan.scenario_seeds[2]);
  return identity;
}

bool match_prefill_model_plan_identity_noalloc(
    const LlmMemoryWorkPlan& plan,
    const LlmCpuExecutionPlan& cpu_plan) noexcept {
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  if (plan.phase != LlmPhase::Prefill ||
      !plan.geometry.prefill.has_value() ||
      !plan.prefill_plan.has_value() || !cpu_plan.prefill.has_value() ||
      cpu_plan.paged.has_value() != paged_layout) {
    return false;
  }
  const LlmGeometry& geometry = plan.geometry;
  const LlmPrefillGeometry& prefill = *geometry.prefill;
  IdentityMatcher identity(plan.plan_identity);
  if (!(identity.literal(Constants::LLM_WORK_PLAN_IDENTITY_VERSION) &&
         identity.field("backend",
                        llm_memory_backend_to_string(plan.backend)) &&
         identity.field("phase", llm_phase_to_string(plan.phase)) &&
         identity.field("kv_layout",
                        llm_kv_layout_to_string(plan.kv_layout)) &&
         identity.field("work_unit_kind",
                        llm_work_unit_kind_to_string(plan.work_unit_kind)) &&
         identity.field("methodology", plan.methodology_version) &&
         identity.integer_field("component_identity_size",
                                plan.component_identities.identity.size()) &&
         identity.field("component_identity",
                        plan.component_identities.identity) &&
         identity.integer_field("range_alignment",
                                Constants::LLM_RANGE_ALIGNMENT_BYTES) &&
         identity.integer_field("weight_passes_per_work_unit",
                                plan.weight_passes_per_work_unit) &&
         identity.integer_field("kv_replay_factor",
                                plan.kv_replay_factor) &&
         identity.integer_field(
             "weight", geometry.active_weight_bytes_per_work_unit) &&
         identity.integer_field("layers", geometry.layer_count) &&
         identity.integer_field("query_heads",
                                geometry.query_head_count) &&
         identity.integer_field("kv_heads", geometry.kv_head_count) &&
         identity.integer_field("query_heads_per_kv_head",
                                geometry.query_heads_per_kv_head) &&
         identity.field("attention_kind",
                        llm_attention_kind_to_string(
                            geometry.attention_kind)) &&
         identity.integer_field("head_dim", geometry.head_dimension) &&
         identity.integer_field("kv_element_bytes",
                                geometry.kv_element_bytes) &&
         identity.field("prefill_planner_version",
                        LlmPrefillVersion::PLANNER) &&
         identity.field("prefill_cpu_partition_version",
                        LlmPrefillVersion::CPU_PARTITION) &&
         identity.integer_field("prefill_prompt_tokens",
                                prefill.prompt_tokens) &&
         identity.integer_field("prefill_query_tile_tokens",
                                prefill.attention_query_tile_tokens) &&
         identity.integer_field("prefill_tile_count",
                                prefill.tile_count) &&
         identity.integer_field(
             "prefill_prefix_token_visits_per_sequence",
             prefill.attention_prefix_token_visits_per_sequence) &&
         identity.integer_field("prefill_causal_token_pairs",
                                prefill.causal_token_pairs_per_sequence) &&
         identity.integer_field("prefill_logical_attention_pairs",
                                prefill.logical_attention_pairs) &&
         identity.integer_field("prefill_logical_attention_fma_terms",
                                prefill.logical_attention_fma_terms) &&
         identity.integer_field(
             "prefill_prefix_block_visits_per_sequence",
             prefill.paged_prefix_block_visits_per_sequence) &&
         identity.integer_field("batch", geometry.batch_size) &&
         identity.integer_field("kv_vector_bytes",
                                geometry.kv_vector_bytes) &&
         identity.integer_field("k_or_v_record_bytes_per_layer",
                                geometry.k_or_v_record_bytes_per_layer) &&
         identity.integer_field("kv_record_bytes_per_layer",
                                geometry.kv_record_bytes_per_layer) &&
         identity.integer_field("kv_bytes_per_visible_token",
                                geometry.kv_bytes_per_visible_token) &&
         identity.integer_field("k_or_v_sequence_visible_bytes",
                                geometry.k_or_v_sequence_visible_bytes) &&
         identity.integer_field("kv_block_tokens",
                                geometry.kv_block_tokens) &&
         identity.integer_field("kv_blocks_per_sequence",
                                geometry.kv_blocks_per_sequence) &&
         identity.integer_field("physical_blocks_per_layer",
                                geometry.physical_blocks_per_layer) &&
         identity.integer_field("total_physical_blocks",
                                geometry.total_physical_blocks) &&
         identity.integer_field("kv_block_bytes",
                                geometry.kv_block_bytes) &&
         identity.integer_field("last_block_tokens",
                                geometry.last_block_tokens) &&
         identity.integer_field("last_block_valid_bytes",
                                geometry.last_block_valid_bytes) &&
         identity.integer_field("decode_append_offset_in_last_block",
                                geometry.decode_append_offset_in_last_block) &&
         identity.integer_field("k_logical_bytes",
                                geometry.k_logical_bytes) &&
         identity.integer_field("v_logical_bytes",
                                geometry.v_logical_bytes) &&
         identity.integer_field("k_layout_padding_bytes",
                                geometry.k_layout_padding_bytes) &&
         identity.integer_field("v_layout_padding_bytes",
                                geometry.v_layout_padding_bytes) &&
         identity.integer_field("block_table_entries",
                                geometry.block_table_entries) &&
         identity.integer_field("block_table_bytes",
                                geometry.block_table_bytes) &&
         identity.integer_field(
             "layout_metadata_lookups_per_layer_sequence_per_work_unit",
             geometry
                 .layout_metadata_lookups_per_layer_sequence_per_work_unit) &&
         identity.integer_field("k_mapping_bytes",
                                geometry.k_mapping_bytes) &&
         identity.integer_field("v_mapping_bytes",
                                geometry.v_mapping_bytes) &&
         identity.integer_field("kv_capacity_bytes",
                                geometry.kv_capacity_bytes) &&
         identity.integer_field("weight_read_bytes_per_work_unit",
                                geometry.weight_read_bytes_per_work_unit) &&
         identity.integer_field("kv_read_bytes_per_work_unit",
                                geometry.kv_read_bytes_per_work_unit) &&
         identity.integer_field("kv_write_bytes_per_work_unit",
                                geometry.kv_write_bytes_per_work_unit) &&
         identity.integer_field(
             "kv_only_payload_bytes_per_work_unit",
             geometry
                 .kv_only_effective_model_payload_bytes_per_work_unit) &&
         identity.integer_field(
             "mixed_payload_bytes_per_work_unit",
             geometry.mixed_effective_model_payload_bytes_per_work_unit) &&
         identity.integer_field("total_data_mapping_bytes",
                                geometry.total_data_mapping_bytes) &&
         identity.integer_field("traffic_crossover_numerator",
                                geometry.traffic_crossover_numerator) &&
         identity.integer_field("traffic_crossover_denominator",
                                geometry.traffic_crossover_denominator) &&
         identity.integer_field("requested_workers",
                                cpu_plan.requested_workers) &&
         identity.integer_field("effective_workers",
                                cpu_plan.effective_workers) &&
         identity.integer_field("layer_descriptors_per_worker",
                                cpu_plan.layer_descriptors_per_worker) &&
         identity.integer_field("sequence_descriptors_per_worker",
                                cpu_plan.sequence_descriptors_per_worker) &&
         identity.integer_field("total_layer_descriptors",
                                cpu_plan.total_layer_descriptors) &&
         identity.integer_field("total_sequence_descriptors",
                                cpu_plan.total_sequence_descriptors) &&
         identity.integer_field("descriptor_bytes",
                                cpu_plan.descriptor_bytes) &&
         identity.integer_field("prefill_execution_identity_size",
                                cpu_plan.prefill->identity.size()) &&
         identity.field("prefill_execution_identity",
                        cpu_plan.prefill->identity))) {
    return false;
  }
  if (paged_layout) {
    const LlmPagedCpuExecutionPlan& paged = *cpu_plan.paged;
    if (!identity.integer_field("paged_layout_identity_size",
                                paged.layout_identity.size()) ||
        !identity.field("paged_layout_identity", paged.layout_identity) ||
        !identity.integer_field("paged_execution_identity_size",
                                paged.execution_identity.size()) ||
        !identity.field("paged_execution_identity",
                        paged.execution_identity)) {
      return false;
    }
  }
  return identity.integer_field("base_seed", plan.base_seed) &&
         identity.integer_field("weight_buffer_seed",
                                plan.weight_buffer_seed) &&
         identity.integer_field("k_buffer_seed", plan.k_buffer_seed) &&
         identity.integer_field("v_buffer_seed", plan.v_buffer_seed) &&
         identity.integer_field("weights_only_scenario_seed",
                                plan.scenario_seeds[0]) &&
         identity.integer_field("kv_only_scenario_seed",
                                plan.scenario_seeds[1]) &&
         identity.integer_field("mixed_scenario_seed",
                                plan.scenario_seeds[2]) &&
         identity.complete();
}

std::string build_scenario_plan_identity(const LlmScenarioWorkPlan& plan) {
  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "model_plan_identity_size",
                        plan.model_plan_identity.size());
  append_identity_field(identity, "model_plan_identity",
                        plan.model_plan_identity);
  append_identity_field(identity, "scenario",
                        llm_scenario_to_string(plan.scenario));
  append_identity_field(identity, "work_unit_kind",
                        llm_work_unit_kind_to_string(plan.work_unit_kind));
  append_identity_field(identity, "kv_write_kind",
                        llm_kv_write_kind_to_string(plan.kv_write_kind));
  append_identity_field(identity, "scenario_seed", plan.scenario_seed);
  append_identity_field(identity, "explicit",
                        plan.explicit_iterations ? 1 : 0);
  append_identity_field(identity, "work_units", plan.work_units);
  append_identity_field(identity, "weight_read_bytes_per_work_unit",
                        plan.weight_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_read_bytes_per_work_unit",
                        plan.kv_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_write_bytes_per_work_unit",
                        plan.kv_write_bytes_per_work_unit);
  append_identity_field(identity, "effective_model_payload_bytes_per_work_unit",
                        plan.effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity,
                        "layout_metadata_lookup_count_per_work_unit",
                        plan.layout_metadata_lookup_count_per_work_unit);
  append_identity_field(identity,
                        "layout_metadata_read_bytes_per_work_unit",
                        plan.layout_metadata_read_bytes_per_work_unit);
  append_identity_field(identity, "accounted_bytes_per_work_unit",
                        plan.accounted_bytes_per_work_unit);
  append_identity_field(identity, "weight_read_bytes",
                        plan.weight_read_bytes);
  append_identity_field(identity, "kv_read_bytes", plan.kv_read_bytes);
  append_identity_field(identity, "kv_write_bytes",
                        plan.kv_write_bytes);
  append_identity_field(identity, "effective_model_payload_bytes",
                        plan.effective_model_payload_bytes);
  append_identity_field(identity, "layout_metadata_lookup_count",
                        plan.layout_metadata_lookup_count);
  append_identity_field(identity, "layout_metadata_read_bytes",
                        plan.layout_metadata_read_bytes);
  append_identity_field(identity, "task_accounted_bytes",
                        plan.task_accounted_bytes);
  append_identity_field(identity, "maximum_work_units_by_work_unit_cap",
                        plan.maximum_work_units_by_work_unit_cap);
  append_identity_field(identity, "maximum_work_units_by_guardrail",
                        plan.maximum_work_units_by_guardrail);
  append_identity_field(identity, "effective_maximum_work_units",
                        plan.effective_maximum_work_units);
  return identity;
}

std::string build_prefill_cpu_scenario_execution_identity(
    const LlmPrefillCpuScenarioExecutionPlan& scenario) {
  std::string identity = "llm-prefill-cpu-scenario-execution-v1";
  append_identity_field(identity, "scenario",
                        llm_scenario_to_string(scenario.scenario));
  append_identity_field(identity, "scope_count",
                        scenario.ownership_scopes.size());
  for (const LlmPrefillCpuOwnershipPlan& scope :
       scenario.ownership_scopes) {
    append_identity_field(identity, "scope_identity_size",
                          scope.identity.size());
    append_identity_field(identity, "scope_identity_sha256",
                          HashUtils::sha256_hex(scope.identity));
  }
  append_identity_field(
      identity, "worker_count",
      scenario.worker_accounted_bytes_per_work_unit.size());
  for (size_t bytes :
       scenario.worker_accounted_bytes_per_work_unit) {
    append_identity_field(identity, "worker_accounted_bytes", bytes);
  }
  append_identity_field(
      identity, "minimum_worker_accounted_bytes",
      scenario.minimum_worker_accounted_bytes_per_work_unit);
  append_identity_field(
      identity, "maximum_worker_accounted_bytes",
      scenario.maximum_worker_accounted_bytes_per_work_unit);
  append_identity_field(
      identity, "worker_accounted_imbalance_bytes",
      scenario.worker_accounted_imbalance_bytes_per_work_unit);
  return identity;
}

std::string build_prefill_cpu_execution_identity(
    const LlmPrefillCpuExecutionPlan& prefill) {
  std::string identity = "llm-prefill-cpu-execution-v1";
  append_identity_field(identity,
                        "sequence_descriptors_per_scenario_per_worker",
                        prefill.sequence_descriptors_per_scenario_per_worker);
  for (size_t scenario = 0; scenario < prefill.scenarios.size(); ++scenario) {
    const LlmPrefillCpuScenarioExecutionPlan& plan =
        prefill.scenarios[scenario];
    append_identity_field(identity, "scenario_index", scenario);
    append_identity_field(identity, "scenario",
                          llm_scenario_to_string(plan.scenario));
    append_identity_field(identity, "scope_count",
                          plan.ownership_scopes.size());
    for (const LlmPrefillCpuOwnershipPlan& scope : plan.ownership_scopes) {
      append_identity_field(identity, "scope_identity_size",
                            scope.identity.size());
      append_identity_field(identity, "scope_identity_sha256",
                            HashUtils::sha256_hex(scope.identity));
    }
    append_identity_field(identity, "scenario_identity_size",
                          plan.identity.size());
    append_identity_field(identity, "scenario_identity_sha256",
                          HashUtils::sha256_hex(plan.identity));
    append_identity_field(identity, "worker_count",
                          plan.worker_accounted_bytes_per_work_unit.size());
    for (size_t bytes : plan.worker_accounted_bytes_per_work_unit) {
      append_identity_field(identity, "worker_accounted_bytes", bytes);
    }
    append_identity_field(
        identity, "minimum_worker_accounted_bytes",
        plan.minimum_worker_accounted_bytes_per_work_unit);
    append_identity_field(
        identity, "maximum_worker_accounted_bytes",
        plan.maximum_worker_accounted_bytes_per_work_unit);
    append_identity_field(
        identity, "worker_accounted_imbalance_bytes",
        plan.worker_accounted_imbalance_bytes_per_work_unit);
  }
  return identity;
}

struct PrefillRangeEvidence {
  size_t data_visit_count = 0;
  size_t valid_token_count = 0;
  size_t layout_metadata_lookup_count = 0;
  size_t model_payload_bytes = 0;
  size_t layout_metadata_read_bytes = 0;
  size_t accounted_bytes = 0;
};

bool calculate_prefill_prefix_data_visits_noalloc(
    const LlmPrefillPlan& prefill, size_t end_token,
    size_t& output) noexcept {
  if (end_token > prefill.prompt_tokens ||
      prefill.attention_query_tile_tokens == 0) {
    return false;
  }
  size_t floor_sum = 0;
  size_t read_visits = 0;
  if (!checked_llm_prefill_floor_sum(
          end_token, prefill.attention_query_tile_tokens, 1, 0,
          floor_sum) ||
      !NumericUtils::checked_multiply(prefill.tile_count, end_token,
                                      read_visits) ||
      floor_sum > read_visits) {
    return false;
  }
  read_visits -= floor_sum;
  return NumericUtils::checked_add(read_visits, end_token, output);
}

bool calculate_prefill_range_evidence_noalloc(
    const LlmPrefillPlan& prefill,
    LlmPrefillPartitionUnitKind unit_kind, size_t first_unit,
    size_t unit_count, PrefillRangeEvidence& output) noexcept {
  output = {};
  const size_t logical_unit_count =
      unit_kind == LlmPrefillPartitionUnitKind::PagedBlock
          ? prefill.blocks_per_sequence
          : prefill.prompt_tokens;
  size_t end_unit = 0;
  if (!NumericUtils::checked_add(first_unit, unit_count, end_unit) ||
      end_unit > logical_unit_count ||
      (unit_kind == LlmPrefillPartitionUnitKind::PagedBlock &&
       !prefill.paged)) {
    return false;
  }

  size_t first_token = first_unit;
  size_t end_token = end_unit;
  if (unit_kind == LlmPrefillPartitionUnitKind::PagedBlock) {
    first_token = prefill.prompt_tokens;
    end_token = prefill.prompt_tokens;
    if ((first_unit < prefill.blocks_per_sequence &&
         !NumericUtils::checked_multiply(
             first_unit, prefill.kv_block_tokens, first_token)) ||
        (end_unit < prefill.blocks_per_sequence &&
         !NumericUtils::checked_multiply(
             end_unit, prefill.kv_block_tokens, end_token))) {
      return false;
    }
  } else if (unit_kind !=
             LlmPrefillPartitionUnitKind::ContiguousToken) {
    return false;
  }

  size_t first_visits = 0;
  size_t end_visits = 0;
  if (!calculate_prefill_prefix_data_visits_noalloc(
          prefill, first_token, first_visits) ||
      !calculate_prefill_prefix_data_visits_noalloc(
          prefill, end_token, end_visits) ||
      end_visits < first_visits) {
    return false;
  }
  output.data_visit_count = end_visits - first_visits;
  output.valid_token_count = end_token - first_token;
  if (unit_kind == LlmPrefillPartitionUnitKind::PagedBlock) {
    size_t first_floors = 0;
    size_t end_floors = 0;
    size_t first_read_visits = 0;
    size_t end_read_visits = 0;
    size_t first_lookups = 0;
    size_t end_lookups = 0;
    if (!checked_llm_prefill_floor_sum(
            first_unit, prefill.attention_query_tile_tokens,
            prefill.kv_block_tokens, 0, first_floors) ||
        !checked_llm_prefill_floor_sum(
            end_unit, prefill.attention_query_tile_tokens,
            prefill.kv_block_tokens, 0, end_floors) ||
        !NumericUtils::checked_multiply(
            prefill.tile_count, first_unit, first_read_visits) ||
        !NumericUtils::checked_multiply(
            prefill.tile_count, end_unit, end_read_visits) ||
        first_floors > first_read_visits ||
        end_floors > end_read_visits) {
      return false;
    }
    first_read_visits -= first_floors;
    end_read_visits -= end_floors;
    if (!NumericUtils::checked_multiply(first_read_visits, 2,
                                        first_lookups) ||
        !NumericUtils::checked_add(first_lookups, first_unit,
                                   first_lookups) ||
        !NumericUtils::checked_multiply(end_read_visits, 2,
                                        end_lookups) ||
        !NumericUtils::checked_add(end_lookups, end_unit, end_lookups) ||
        end_lookups < first_lookups) {
      return false;
    }
    output.layout_metadata_lookup_count = end_lookups - first_lookups;
  }
  if (!NumericUtils::checked_multiply(
          output.data_visit_count, prefill.kv_record_bytes_per_layer,
          output.model_payload_bytes) ||
      !NumericUtils::checked_multiply(
          output.layout_metadata_lookup_count,
          Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
          output.layout_metadata_read_bytes) ||
      !NumericUtils::checked_add(output.model_payload_bytes,
                                 output.layout_metadata_read_bytes,
                                 output.accounted_bytes)) {
    return false;
  }
  return true;
}

bool prefill_range_cost_matches_noalloc(
    const LlmPrefillUnitRangeCost& retained,
    LlmPrefillPartitionUnitKind unit_kind, size_t first_unit,
    size_t unit_count, const PrefillRangeEvidence& expected) noexcept {
  return retained.valid && retained.reason_code == LlmPrefillReason::VALID &&
         retained.unit_kind == unit_kind &&
         retained.first_unit == first_unit &&
         retained.unit_count == unit_count &&
         retained.valid_token_count == expected.valid_token_count &&
         retained.data_visit_count == expected.data_visit_count &&
         retained.layout_metadata_lookup_count ==
             expected.layout_metadata_lookup_count &&
         retained.model_payload_bytes == expected.model_payload_bytes &&
         retained.layout_metadata_read_bytes ==
             expected.layout_metadata_read_bytes &&
         retained.accounted_bytes == expected.accounted_bytes;
}

using WidePrefillSize = unsigned __int128;

struct PrefillRationalTarget {
  size_t whole = 0;
  size_t remainder = 0;
  size_t denominator = 1;
};

PrefillRationalTarget calculate_prefill_partition_target_noalloc(
    size_t rank, size_t total, size_t denominator,
    size_t preceding_weight_bytes) noexcept {
  const WidePrefillSize numerator =
      static_cast<WidePrefillSize>(rank) * total;
  PrefillRationalTarget target;
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

bool prefill_value_is_at_least_target(
    size_t value, const PrefillRationalTarget& target) noexcept {
  return value > target.whole ||
         (value == target.whole && target.remainder == 0);
}

struct PrefillRationalDistance {
  size_t whole = 0;
  size_t remainder = 0;
};

PrefillRationalDistance prefill_distance_from_target(
    size_t value, const PrefillRationalTarget& target) noexcept {
  if (value <= target.whole) {
    return {target.whole - value, target.remainder};
  }
  if (target.remainder == 0) {
    return {value - target.whole, 0};
  }
  return {value - target.whole - 1,
          target.denominator - target.remainder};
}

bool prefill_distance_is_less(
    const PrefillRationalDistance& lhs,
    const PrefillRationalDistance& rhs) noexcept {
  return lhs.whole < rhs.whole ||
         (lhs.whole == rhs.whole && lhs.remainder < rhs.remainder);
}

bool choose_prefill_partition_boundary_noalloc(
    const LlmPrefillPlan& prefill,
    LlmPrefillPartitionUnitKind unit_kind, size_t minimum_boundary,
    size_t maximum_boundary, const PrefillRationalTarget& target,
    size_t& output) noexcept {
  size_t low = minimum_boundary;
  size_t high = maximum_boundary;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    PrefillRangeEvidence cost;
    if (!calculate_prefill_range_evidence_noalloc(
            prefill, unit_kind, 0, middle, cost)) {
      return false;
    }
    if (prefill_value_is_at_least_target(cost.accounted_bytes, target)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }

  const size_t upper_boundary = low;
  const size_t lower_boundary =
      upper_boundary > minimum_boundary ? upper_boundary - 1
                                        : upper_boundary;
  PrefillRangeEvidence upper_cost;
  PrefillRangeEvidence lower_cost;
  if (!calculate_prefill_range_evidence_noalloc(
          prefill, unit_kind, 0, upper_boundary, upper_cost) ||
      !calculate_prefill_range_evidence_noalloc(
          prefill, unit_kind, 0, lower_boundary, lower_cost)) {
    return false;
  }
  const PrefillRationalDistance upper_distance =
      prefill_distance_from_target(upper_cost.accounted_bytes, target);
  const PrefillRationalDistance lower_distance =
      prefill_distance_from_target(lower_cost.accounted_bytes, target);
  output = prefill_distance_is_less(upper_distance, lower_distance)
               ? upper_boundary
               : lower_boundary;
  return true;
}

size_t rotated_prefill_worker(size_t rank, size_t rotation,
                              size_t worker_count) noexcept {
  const size_t wrap_boundary = worker_count - rotation;
  return rank >= wrap_boundary ? rank - wrap_boundary : rank + rotation;
}

bool match_prefill_ownership_identity_noalloc(
    const LlmPrefillPlan& prefill,
    const LlmPrefillCpuOwnershipPlan& ownership) noexcept {
  const char* const unit_kind =
      ownership.unit_kind == LlmPrefillPartitionUnitKind::PagedBlock
          ? "paged_block"
          : "contiguous_token";
  IdentityMatcher identity(ownership.identity);
  if (!identity.literal(LlmPrefillVersion::CPU_PARTITION) ||
      !identity.field("planner_version", LlmPrefillVersion::PLANNER) ||
      !identity.field("schedule_version",
                      LlmPrefillVersion::OWNER_LOCAL_SCHEDULE) ||
      !identity.field("unit_kind", unit_kind) ||
      !identity.field("scenario",
                      llm_scenario_to_string(ownership.scenario)) ||
      !identity.integer_field("active_weight_bytes",
                              prefill.active_weight_bytes) ||
      !identity.integer_field("prompt_tokens", prefill.prompt_tokens) ||
      !identity.integer_field("query_tile_tokens",
                              prefill.attention_query_tile_tokens) ||
      !identity.integer_field("tile_count", prefill.tile_count) ||
      !identity.integer_field("layer_count", prefill.layer_count) ||
      !identity.integer_field("batch_size", prefill.batch_size) ||
      !identity.integer_field("query_head_count",
                              prefill.query_head_count) ||
      !identity.integer_field("head_dimension", prefill.head_dimension) ||
      !identity.integer_field("k_or_v_record_bytes_per_layer",
                              prefill.k_or_v_record_bytes_per_layer) ||
      !identity.integer_field("paged", prefill.paged ? 1 : 0) ||
      !identity.integer_field("kv_block_tokens", prefill.kv_block_tokens) ||
      !identity.integer_field("blocks_per_sequence",
                              prefill.blocks_per_sequence) ||
      !identity.integer_field("worker_count", ownership.worker_count) ||
      !identity.integer_field("worker_rotation",
                              ownership.worker_rotation) ||
      !identity.integer_field("logical_unit_count",
                              ownership.logical_unit_count) ||
      !identity.integer_field("active_worker_count",
                              ownership.active_worker_count) ||
      !identity.integer_field("weight_shards_included",
                              ownership.weight_shards_included ? 1 : 0) ||
      !identity.integer_field("total_weight_shard_bytes",
                              ownership.total_weight_shard_bytes) ||
      !identity.integer_field("total_kv_model_payload_bytes",
                              ownership.total_kv_model_payload_bytes) ||
      !identity.integer_field(
          "total_layout_metadata_lookup_count",
          ownership.total_layout_metadata_lookup_count) ||
      !identity.integer_field(
          "total_layout_metadata_read_bytes",
          ownership.total_layout_metadata_read_bytes) ||
      !identity.integer_field("total_kv_accounted_bytes",
                              ownership.total_kv_accounted_bytes) ||
      !identity.integer_field("total_scenario_accounted_bytes",
                              ownership.total_scenario_accounted_bytes) ||
      !identity.integer_field("minimum_worker_accounted_bytes",
                              ownership.minimum_worker_accounted_bytes) ||
      !identity.integer_field("maximum_worker_accounted_bytes",
                              ownership.maximum_worker_accounted_bytes) ||
      !identity.integer_field("worker_accounted_imbalance_bytes",
                              ownership.worker_accounted_imbalance_bytes) ||
      !identity.integer_field("worker_vector_count",
                              ownership.worker_count)) {
    return false;
  }
  for (size_t worker = 0; worker < ownership.worker_count; ++worker) {
    if (!identity.literal("|worker_") || !identity.integer(worker) ||
        !identity.literal("_weight_shard_bytes=") ||
        !identity.integer(ownership.worker_weight_shard_bytes[worker]) ||
        !identity.literal("|worker_") || !identity.integer(worker) ||
        !identity.literal("_kv_model_payload_bytes=") ||
        !identity.integer(
            ownership.worker_kv_model_payload_bytes[worker]) ||
        !identity.literal("|worker_") || !identity.integer(worker) ||
        !identity.literal("_layout_metadata_lookup_count=") ||
        !identity.integer(
            ownership.worker_layout_metadata_lookup_count[worker]) ||
        !identity.literal("|worker_") ||
        !identity.integer(worker) ||
        !identity.literal("_layout_metadata_read_bytes=") ||
        !identity.integer(
            ownership.worker_layout_metadata_read_bytes[worker]) ||
        !identity.literal("|worker_") ||
        !identity.integer(worker) ||
        !identity.literal("_scenario_accounted_bytes=") ||
        !identity.integer(
            ownership.worker_scenario_accounted_bytes[worker])) {
      return false;
    }
  }
  if (!identity.integer_field("assignment_count",
                              ownership.assignments.size())) {
    return false;
  }
  for (size_t index = 0; index < ownership.assignments.size(); ++index) {
    const LlmPrefillCpuAssignment& assignment = ownership.assignments[index];
    if (!identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_range_rank=") ||
        !identity.integer(assignment.range_rank) ||
        !identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_worker_index=") ||
        !identity.integer(assignment.worker_index) ||
        !identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_first_unit=") ||
        !identity.integer(assignment.first_unit) ||
        !identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_unit_count=") ||
        !identity.integer(assignment.unit_count) ||
        !identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_kv_model_payload_bytes=") ||
        !identity.integer(assignment.kv_cost.model_payload_bytes) ||
        !identity.literal("|assignment_") || !identity.integer(index) ||
        !identity.literal("_layout_metadata_lookup_count=") ||
        !identity.integer(
            assignment.kv_cost.layout_metadata_lookup_count) ||
        !identity.literal("|assignment_") ||
        !identity.integer(index) ||
        !identity.literal("_layout_metadata_read_bytes=") ||
        !identity.integer(
            assignment.kv_cost.layout_metadata_read_bytes) ||
        !identity.literal("|assignment_") ||
        !identity.integer(index) ||
        !identity.literal("_kv_accounted_bytes=") ||
        !identity.integer(assignment.kv_cost.accounted_bytes)) {
      return false;
    }
  }
  return identity.complete();
}

bool validate_prefill_plan_noalloc(
    const LlmMemoryWorkPlan& plan) noexcept {
  if (!plan.geometry.prefill.has_value() ||
      !plan.prefill_plan.has_value()) {
    return false;
  }
  const LlmGeometry& geometry = plan.geometry;
  const LlmPrefillGeometry& geometry_prefill = *geometry.prefill;
  const LlmPrefillPlan& prefill = *plan.prefill_plan;
  const bool paged_layout = geometry.kv_layout == LlmKvLayout::Paged;
  if (geometry.decode.has_value() ||
      geometry.phase != LlmPhase::Prefill ||
      (geometry.kv_layout != LlmKvLayout::Contiguous && !paged_layout) ||
      geometry.work_unit_kind != LlmWorkUnitKind::PrefillOperation ||
      plan.work_unit_kind != LlmWorkUnitKind::PrefillOperation ||
      !prefill.valid || prefill.reason_code != LlmPrefillReason::VALID ||
      prefill.active_weight_bytes == 0 || prefill.prompt_tokens == 0 ||
      prefill.attention_query_tile_tokens == 0 ||
      prefill.attention_query_tile_tokens > prefill.prompt_tokens ||
      prefill.layer_count == 0 || prefill.batch_size == 0 ||
      prefill.query_head_count == 0 || prefill.head_dimension == 0 ||
      prefill.k_or_v_record_bytes_per_layer == 0 ||
      prefill.paged != paged_layout ||
      (!paged_layout &&
       (prefill.kv_block_tokens != 0 ||
        prefill.blocks_per_sequence != 0 ||
        prefill.prefix_block_visits_per_sequence != 0 ||
        prefill.layout_metadata_lookups_per_layer_sequence != 0 ||
        prefill.layout_metadata_lookups_per_work_unit != 0 ||
        prefill.layout_metadata_read_bytes_per_work_unit != 0)) ||
      prefill.active_weight_bytes !=
          geometry.active_weight_bytes_per_work_unit ||
      prefill.prompt_tokens != geometry_prefill.prompt_tokens ||
      prefill.attention_query_tile_tokens !=
          geometry_prefill.attention_query_tile_tokens ||
      prefill.layer_count != geometry.layer_count ||
      prefill.batch_size != geometry.batch_size ||
      prefill.query_head_count != geometry.query_head_count ||
      prefill.head_dimension != geometry.head_dimension ||
      prefill.k_or_v_record_bytes_per_layer !=
          geometry.k_or_v_record_bytes_per_layer ||
      (paged_layout &&
       (prefill.kv_block_tokens != geometry.kv_block_tokens ||
        prefill.blocks_per_sequence != geometry.kv_blocks_per_sequence))) {
    return false;
  }

  const size_t full_tiles =
      prefill.prompt_tokens / prefill.attention_query_tile_tokens;
  const size_t final_tile =
      prefill.prompt_tokens % prefill.attention_query_tile_tokens;
  size_t tile_count = full_tiles;
  size_t full_tile_triangular = 0;
  size_t prefix_visits = 0;
  size_t causal_pairs = 0;
  size_t layer_batch_count = 0;
  size_t logical_attention_pairs = 0;
  size_t logical_attention_fma_terms = 0;
  size_t kv_record_bytes = 0;
  size_t kv_bytes_per_token = 0;
  size_t logical_records = 0;
  size_t k_logical_bytes = 0;
  size_t batch_prompt_tokens = 0;
  size_t kv_write_bytes = 0;
  size_t batch_prefix_visits = 0;
  size_t kv_read_bytes = 0;
  size_t kv_only_bytes = 0;
  size_t mixed_bytes = 0;
  size_t blocks_per_sequence = 0;
  size_t prefix_block_visits = 0;
  size_t layout_lookups_per_layer_sequence = 0;
  size_t layout_lookups_per_work_unit = 0;
  size_t layout_read_bytes_per_work_unit = 0;
  if ((final_tile != 0 &&
       !NumericUtils::checked_add(tile_count, 1, tile_count)) ||
      !checked_llm_prefill_triangular(full_tiles,
                                      full_tile_triangular) ||
      !NumericUtils::checked_multiply(
          prefill.attention_query_tile_tokens, full_tile_triangular,
          prefix_visits) ||
      (final_tile != 0 &&
       !NumericUtils::checked_add(prefix_visits, prefill.prompt_tokens,
                                  prefix_visits)) ||
      !checked_llm_prefill_triangular(prefill.prompt_tokens,
                                      causal_pairs) ||
      !NumericUtils::checked_multiply(prefill.layer_count,
                                      prefill.batch_size,
                                      layer_batch_count) ||
      !NumericUtils::checked_multiply(layer_batch_count,
                                      prefill.query_head_count,
                                      logical_attention_pairs) ||
      !NumericUtils::checked_multiply(logical_attention_pairs,
                                      causal_pairs,
                                      logical_attention_pairs) ||
      !NumericUtils::checked_multiply(logical_attention_pairs,
                                      prefill.head_dimension,
                                      logical_attention_fma_terms) ||
      !NumericUtils::checked_multiply(
          prefill.k_or_v_record_bytes_per_layer, 2, kv_record_bytes) ||
      !NumericUtils::checked_multiply(prefill.layer_count,
                                      kv_record_bytes,
                                      kv_bytes_per_token) ||
      !NumericUtils::checked_multiply(layer_batch_count,
                                      prefill.prompt_tokens,
                                      logical_records) ||
      !NumericUtils::checked_multiply(
          logical_records, prefill.k_or_v_record_bytes_per_layer,
          k_logical_bytes) ||
      !NumericUtils::checked_multiply(prefill.batch_size,
                                      prefill.prompt_tokens,
                                      batch_prompt_tokens) ||
      !NumericUtils::checked_multiply(batch_prompt_tokens,
                                      kv_bytes_per_token,
                                      kv_write_bytes) ||
      !NumericUtils::checked_multiply(prefill.batch_size,
                                      prefix_visits,
                                      batch_prefix_visits) ||
      !NumericUtils::checked_multiply(batch_prefix_visits,
                                      kv_bytes_per_token,
                                      kv_read_bytes) ||
      !NumericUtils::checked_add(kv_write_bytes, kv_read_bytes,
                                 kv_only_bytes) ||
      !NumericUtils::checked_add(prefill.active_weight_bytes,
                                 kv_only_bytes, mixed_bytes)) {
    return false;
  }
  if (paged_layout) {
    size_t full_tile_block_floor_sum = 0;
    size_t read_lookups = 0;
    if (!checked_llm_prefill_ceil_divide(
            prefill.prompt_tokens, prefill.kv_block_tokens,
            blocks_per_sequence) ||
        !checked_llm_prefill_floor_sum(
            full_tiles, prefill.kv_block_tokens,
            prefill.attention_query_tile_tokens,
            prefill.attention_query_tile_tokens - 1,
            full_tile_block_floor_sum) ||
        !NumericUtils::checked_add(full_tiles,
                                   full_tile_block_floor_sum,
                                   prefix_block_visits) ||
        (final_tile != 0 &&
         !NumericUtils::checked_add(prefix_block_visits,
                                    blocks_per_sequence,
                                    prefix_block_visits)) ||
        !NumericUtils::checked_multiply(prefix_block_visits, 2,
                                        read_lookups) ||
        !NumericUtils::checked_add(blocks_per_sequence, read_lookups,
                                   layout_lookups_per_layer_sequence) ||
        !NumericUtils::checked_multiply(
            layer_batch_count, layout_lookups_per_layer_sequence,
            layout_lookups_per_work_unit) ||
        !NumericUtils::checked_multiply(
            layout_lookups_per_work_unit,
            Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
            layout_read_bytes_per_work_unit)) {
      return false;
    }
  }
  return prefill.full_query_tile_count == full_tiles &&
         prefill.final_query_tile_tokens == final_tile &&
         prefill.tile_count == tile_count &&
         prefill.attention_prefix_token_visits_per_sequence ==
             prefix_visits &&
         prefill.causal_token_pairs_per_sequence == causal_pairs &&
         prefill.logical_attention_pairs == logical_attention_pairs &&
         prefill.logical_attention_fma_terms ==
             logical_attention_fma_terms &&
         prefill.kv_record_bytes_per_layer == kv_record_bytes &&
         prefill.kv_bytes_per_token == kv_bytes_per_token &&
         prefill.k_logical_bytes == k_logical_bytes &&
         prefill.v_logical_bytes == k_logical_bytes &&
         prefill.weight_read_bytes_per_work_unit ==
             prefill.active_weight_bytes &&
         prefill.kv_write_bytes_per_work_unit == kv_write_bytes &&
         prefill.kv_read_bytes_per_work_unit == kv_read_bytes &&
         prefill.kv_only_payload_bytes_per_work_unit == kv_only_bytes &&
         prefill.mixed_payload_bytes_per_work_unit == mixed_bytes &&
         prefill.blocks_per_sequence == blocks_per_sequence &&
         prefill.prefix_block_visits_per_sequence ==
             prefix_block_visits &&
         prefill.layout_metadata_lookups_per_layer_sequence ==
             layout_lookups_per_layer_sequence &&
         prefill.layout_metadata_lookups_per_work_unit ==
             layout_lookups_per_work_unit &&
         prefill.layout_metadata_read_bytes_per_work_unit ==
             layout_read_bytes_per_work_unit &&
         geometry_prefill.tile_count == tile_count &&
         geometry_prefill.attention_prefix_token_visits_per_sequence ==
             prefix_visits &&
         geometry_prefill.causal_token_pairs_per_sequence == causal_pairs &&
         geometry_prefill.logical_attention_pairs ==
             logical_attention_pairs &&
         geometry_prefill.logical_attention_fma_terms ==
             logical_attention_fma_terms &&
         geometry_prefill.paged_prefix_block_visits_per_sequence ==
             prefix_block_visits &&
         geometry.layout_metadata_lookups_per_layer_sequence_per_work_unit ==
             layout_lookups_per_layer_sequence &&
         geometry.k_logical_bytes == k_logical_bytes &&
         geometry.v_logical_bytes == k_logical_bytes &&
         geometry.weight_read_bytes_per_work_unit ==
             prefill.active_weight_bytes &&
         geometry.kv_write_bytes_per_work_unit == kv_write_bytes &&
         geometry.kv_read_bytes_per_work_unit == kv_read_bytes &&
         geometry.kv_only_effective_model_payload_bytes_per_work_unit ==
             kv_only_bytes &&
         geometry.mixed_effective_model_payload_bytes_per_work_unit ==
             mixed_bytes;
}

bool validate_prefill_scope_noalloc(
    const LlmPrefillPlan& prefill,
    const LlmCpuExecutionPlan& cpu_plan, size_t layer_index,
    LlmScenario expected_scenario, size_t expected_rotation,
    bool include_weight_shards,
    const LlmPrefillCpuOwnershipPlan& scope) noexcept {
  const size_t worker_count = cpu_plan.effective_workers;
  if (worker_count == 0 || layer_index >= prefill.layer_count ||
      scope.worker_weight_shard_bytes.size() != worker_count ||
      scope.worker_kv_model_payload_bytes.size() != worker_count ||
      scope.worker_layout_metadata_lookup_count.size() != worker_count ||
      scope.worker_layout_metadata_read_bytes.size() != worker_count ||
      scope.worker_scenario_accounted_bytes.size() != worker_count) {
    return false;
  }
  const bool weights_only = expected_scenario == LlmScenario::WeightsOnly;
  const LlmPrefillPartitionUnitKind unit_kind =
      prefill.paged ? LlmPrefillPartitionUnitKind::PagedBlock
                    : LlmPrefillPartitionUnitKind::ContiguousToken;
  const size_t available_logical_units =
      prefill.paged ? prefill.blocks_per_sequence : prefill.prompt_tokens;
  const size_t active_workers =
      weights_only ? 0
                   : std::min(worker_count, available_logical_units);
  const size_t logical_units = weights_only ? 0 : available_logical_units;
  const size_t expected_assignment_count = active_workers;
  if (!scope.valid || scope.reason_code != LlmPrefillReason::VALID ||
      scope.unit_kind != unit_kind ||
      scope.scenario != expected_scenario ||
      scope.worker_count != worker_count ||
      scope.worker_rotation != expected_rotation ||
      scope.logical_unit_count != logical_units ||
      scope.active_worker_count != active_workers ||
      scope.weight_shards_included != include_weight_shards ||
      scope.assignments.size() != expected_assignment_count) {
    return false;
  }

  size_t total_weight_bytes = 0;
  for (size_t worker = 0; worker < worker_count; ++worker) {
    const size_t expected_weight =
        include_weight_shards
            ? cpu_plan.workers[worker]
                  .layers[layer_index]
                  .weight.span_bytes
            : 0;
    if (scope.worker_weight_shard_bytes[worker] != expected_weight ||
        !NumericUtils::checked_add(total_weight_bytes, expected_weight,
                                   total_weight_bytes)) {
      return false;
    }
  }

  PrefillRangeEvidence total_kv;
  if (!weights_only &&
      !calculate_prefill_range_evidence_noalloc(
          prefill, unit_kind, 0, available_logical_units, total_kv)) {
    return false;
  }
  size_t total_scenario_bytes = 0;
  if (!NumericUtils::checked_add(total_weight_bytes,
                                 total_kv.accounted_bytes,
                                 total_scenario_bytes) ||
      scope.total_weight_shard_bytes != total_weight_bytes ||
      scope.total_kv_model_payload_bytes !=
          total_kv.model_payload_bytes ||
      scope.total_layout_metadata_lookup_count !=
          total_kv.layout_metadata_lookup_count ||
      scope.total_layout_metadata_read_bytes !=
          total_kv.layout_metadata_read_bytes ||
      scope.total_kv_accounted_bytes != total_kv.accounted_bytes ||
      scope.total_scenario_accounted_bytes != total_scenario_bytes) {
    return false;
  }

  if (weights_only) {
    for (size_t worker = 0; worker < worker_count; ++worker) {
      if (scope.worker_kv_model_payload_bytes[worker] != 0 ||
          scope.worker_layout_metadata_lookup_count[worker] != 0 ||
          scope.worker_layout_metadata_read_bytes[worker] != 0 ||
          scope.worker_scenario_accounted_bytes[worker] !=
              scope.worker_weight_shard_bytes[worker]) {
        return false;
      }
    }
  } else {
    size_t active_weight_bytes = 0;
    for (size_t rank = 0; rank < active_workers; ++rank) {
      const size_t worker = rotated_prefill_worker(
          rank, expected_rotation, worker_count);
      if (!NumericUtils::checked_add(
              active_weight_bytes,
              scope.worker_weight_shard_bytes[worker],
              active_weight_bytes)) {
        return false;
      }
    }
    size_t partitionable_cost = 0;
    if (!NumericUtils::checked_add(total_kv.accounted_bytes,
                                   active_weight_bytes,
                                   partitionable_cost)) {
      return false;
    }

    size_t previous_boundary = 0;
    size_t preceding_weight_bytes = 0;
    for (size_t rank = 0; rank < active_workers; ++rank) {
      const size_t worker = rotated_prefill_worker(
          rank, expected_rotation, worker_count);
      size_t next_boundary = available_logical_units;
      if (rank + 1 < active_workers) {
        if (!NumericUtils::checked_add(
                preceding_weight_bytes,
                scope.worker_weight_shard_bytes[worker],
                preceding_weight_bytes)) {
          return false;
        }
        const size_t boundary_rank = rank + 1;
        const PrefillRationalTarget target =
            calculate_prefill_partition_target_noalloc(
                boundary_rank, partitionable_cost, active_workers,
                preceding_weight_bytes);
        const size_t minimum_boundary = previous_boundary + 1;
        const size_t remaining_workers = active_workers - boundary_rank;
        const size_t maximum_boundary =
            available_logical_units - remaining_workers;
        if (!choose_prefill_partition_boundary_noalloc(
                prefill, unit_kind, minimum_boundary, maximum_boundary,
                target, next_boundary)) {
          return false;
        }
      }
      const size_t unit_count = next_boundary - previous_boundary;
      PrefillRangeEvidence expected_cost;
      if (!calculate_prefill_range_evidence_noalloc(
              prefill, unit_kind, previous_boundary, unit_count,
              expected_cost)) {
        return false;
      }
      const LlmPrefillCpuAssignment& assignment = scope.assignments[rank];
      size_t expected_worker_scenario_bytes = 0;
      if (!NumericUtils::checked_add(
              scope.worker_weight_shard_bytes[worker],
              expected_cost.accounted_bytes,
              expected_worker_scenario_bytes) ||
          assignment.range_rank != rank ||
          assignment.worker_index != worker ||
          assignment.first_unit != previous_boundary ||
          assignment.unit_count != unit_count ||
          !prefill_range_cost_matches_noalloc(
              assignment.kv_cost, unit_kind, previous_boundary,
              unit_count, expected_cost) ||
          scope.worker_kv_model_payload_bytes[worker] !=
              expected_cost.model_payload_bytes ||
          scope.worker_layout_metadata_lookup_count[worker] !=
              expected_cost.layout_metadata_lookup_count ||
          scope.worker_layout_metadata_read_bytes[worker] !=
              expected_cost.layout_metadata_read_bytes ||
          scope.worker_scenario_accounted_bytes[worker] !=
              expected_worker_scenario_bytes) {
        return false;
      }
      previous_boundary = next_boundary;
    }
    if (previous_boundary != available_logical_units) {
      return false;
    }
    for (size_t worker = 0; worker < worker_count; ++worker) {
      const size_t rank =
          worker >= expected_rotation
              ? worker - expected_rotation
              : worker + (worker_count - expected_rotation);
      if (rank >= active_workers &&
          (scope.worker_kv_model_payload_bytes[worker] != 0 ||
           scope.worker_layout_metadata_lookup_count[worker] != 0 ||
           scope.worker_layout_metadata_read_bytes[worker] != 0 ||
           scope.worker_scenario_accounted_bytes[worker] !=
               scope.worker_weight_shard_bytes[worker])) {
        return false;
      }
    }
  }

  size_t minimum_worker_bytes = std::numeric_limits<size_t>::max();
  size_t maximum_worker_bytes = 0;
  size_t summed_worker_bytes = 0;
  for (size_t worker = 0; worker < worker_count; ++worker) {
    const size_t bytes = scope.worker_scenario_accounted_bytes[worker];
    minimum_worker_bytes = std::min(minimum_worker_bytes, bytes);
    maximum_worker_bytes = std::max(maximum_worker_bytes, bytes);
    if (!NumericUtils::checked_add(summed_worker_bytes, bytes,
                                   summed_worker_bytes)) {
      return false;
    }
  }
  return summed_worker_bytes == total_scenario_bytes &&
         scope.minimum_worker_accounted_bytes == minimum_worker_bytes &&
         scope.maximum_worker_accounted_bytes == maximum_worker_bytes &&
         scope.worker_accounted_imbalance_bytes ==
             maximum_worker_bytes - minimum_worker_bytes &&
         match_prefill_ownership_identity_noalloc(prefill, scope);
}

bool match_prefill_scenario_identity_noalloc(
    const LlmPrefillCpuScenarioExecutionPlan& scenario) noexcept {
  IdentityMatcher identity(scenario.identity);
  if (!identity.literal("llm-prefill-cpu-scenario-execution-v1") ||
      !identity.field("scenario",
                      llm_scenario_to_string(scenario.scenario)) ||
      !identity.integer_field("scope_count",
                              scenario.ownership_scopes.size())) {
    return false;
  }
  for (const LlmPrefillCpuOwnershipPlan& scope :
       scenario.ownership_scopes) {
    if (!identity.integer_field("scope_identity_size",
                                scope.identity.size()) ||
        !identity.sha256_field("scope_identity_sha256", scope.identity)) {
      return false;
    }
  }
  if (!identity.integer_field(
          "worker_count",
          scenario.worker_accounted_bytes_per_work_unit.size())) {
    return false;
  }
  for (size_t bytes : scenario.worker_accounted_bytes_per_work_unit) {
    if (!identity.integer_field("worker_accounted_bytes", bytes)) {
      return false;
    }
  }
  return identity.integer_field(
             "minimum_worker_accounted_bytes",
             scenario.minimum_worker_accounted_bytes_per_work_unit) &&
         identity.integer_field(
             "maximum_worker_accounted_bytes",
             scenario.maximum_worker_accounted_bytes_per_work_unit) &&
         identity.integer_field(
             "worker_accounted_imbalance_bytes",
             scenario.worker_accounted_imbalance_bytes_per_work_unit) &&
         identity.complete();
}

bool match_prefill_execution_identity_noalloc(
    const LlmPrefillCpuExecutionPlan& prefill) noexcept {
  IdentityMatcher identity(prefill.identity);
  if (!identity.literal("llm-prefill-cpu-execution-v1") ||
      !identity.integer_field(
          "sequence_descriptors_per_scenario_per_worker",
          prefill.sequence_descriptors_per_scenario_per_worker)) {
    return false;
  }
  for (size_t scenario_index_value = 0;
       scenario_index_value < prefill.scenarios.size();
       ++scenario_index_value) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        prefill.scenarios[scenario_index_value];
    if (!identity.integer_field("scenario_index",
                                scenario_index_value) ||
        !identity.field("scenario",
                        llm_scenario_to_string(scenario.scenario)) ||
        !identity.integer_field("scope_count",
                                scenario.ownership_scopes.size())) {
      return false;
    }
    for (const LlmPrefillCpuOwnershipPlan& scope :
         scenario.ownership_scopes) {
      if (!identity.integer_field("scope_identity_size",
                                  scope.identity.size()) ||
          !identity.sha256_field("scope_identity_sha256",
                                 scope.identity)) {
        return false;
      }
    }
    if (!identity.integer_field("scenario_identity_size",
                                scenario.identity.size()) ||
        !identity.sha256_field("scenario_identity_sha256",
                               scenario.identity) ||
        !identity.integer_field(
            "worker_count",
            scenario.worker_accounted_bytes_per_work_unit.size())) {
      return false;
    }
    for (size_t bytes : scenario.worker_accounted_bytes_per_work_unit) {
      if (!identity.integer_field("worker_accounted_bytes", bytes)) {
        return false;
      }
    }
    if (!identity.integer_field(
            "minimum_worker_accounted_bytes",
            scenario.minimum_worker_accounted_bytes_per_work_unit) ||
        !identity.integer_field(
            "maximum_worker_accounted_bytes",
            scenario.maximum_worker_accounted_bytes_per_work_unit) ||
        !identity.integer_field(
            "worker_accounted_imbalance_bytes",
            scenario.worker_accounted_imbalance_bytes_per_work_unit)) {
      return false;
    }
  }
  return identity.complete();
}

/**
 * Conservative closed-form admission bound for retained prefill identities.
 *
 * Scope identities serialize fixed geometry plus five worker and eight
 * assignment fields whose decimal coordinates are bounded by `size_t`.
 * Aggregate identities retain only SHA-256 references to those scope strings.
 * The slack also covers implementation string-capacity rounding, so a large
 * L*B*worker request is rejected before proportional ownership allocation.
 */
bool calculate_prefill_identity_preflight_bytes(
    size_t scope_count, size_t assignment_count, size_t worker_count,
    size_t& identity_bytes) {
  constexpr size_t kScopeFixedBytes = 4096;
  constexpr size_t kScopeWorkerBytes = 768;
  constexpr size_t kScopeAssignmentBytes = 1024;
  constexpr size_t kAggregateFixedBytes = 4096;
  constexpr size_t kAggregateScopeDigestBytes = 512;
  constexpr size_t kAggregateWorkerBytes = 512;
  size_t scope_fixed_bytes = 0;
  size_t scope_worker_count = 0;
  size_t scope_worker_bytes = 0;
  size_t scope_assignment_bytes = 0;
  size_t aggregate_scope_bytes = 0;
  size_t aggregate_worker_bytes = 0;
  identity_bytes = 0;
  if (!NumericUtils::checked_multiply(scope_count, kScopeFixedBytes,
                                      scope_fixed_bytes) ||
      !NumericUtils::checked_multiply(scope_count, worker_count,
                                      scope_worker_count) ||
      !NumericUtils::checked_multiply(scope_worker_count,
                                      kScopeWorkerBytes,
                                      scope_worker_bytes) ||
      !NumericUtils::checked_multiply(assignment_count,
                                      kScopeAssignmentBytes,
                                      scope_assignment_bytes) ||
      !NumericUtils::checked_multiply(scope_count,
                                      kAggregateScopeDigestBytes,
                                      aggregate_scope_bytes) ||
      !NumericUtils::checked_multiply(worker_count,
                                      kAggregateWorkerBytes,
                                      aggregate_worker_bytes) ||
      !NumericUtils::checked_add(scope_fixed_bytes, scope_worker_bytes,
                                 identity_bytes) ||
      !NumericUtils::checked_add(identity_bytes, scope_assignment_bytes,
                                 identity_bytes) ||
      !NumericUtils::checked_add(identity_bytes, kAggregateFixedBytes,
                                 identity_bytes) ||
      !NumericUtils::checked_add(identity_bytes, aggregate_scope_bytes,
                                 identity_bytes) ||
      !NumericUtils::checked_add(identity_bytes, aggregate_worker_bytes,
                                 identity_bytes)) {
    return false;
  }
  return true;
}

bool calculate_planner_storage_bytes(size_t layer_count,
                                     size_t sequence_count_per_worker,
                                     size_t worker_count,
                                     bool paged_layout,
                                     size_t ownership_assignment_count,
                                     bool prefill_phase,
                                     size_t prefill_scope_count,
                                     size_t prefill_assignment_count,
                                     size_t& planner_storage_bytes) {
  size_t weight_layer_bytes = 0;
  size_t worker_object_bytes = 0;
  size_t layer_template_count = 0;
  size_t layer_template_bytes = 0;
  size_t sequence_template_count = 0;
  size_t sequence_template_bytes = 0;
  size_t ownership_assignment_bytes = 0;
  size_t ownership_worker_bytes = 0;
  size_t prefill_scope_bytes = 0;
  size_t prefill_assignment_bytes = 0;
  size_t prefill_scope_worker_values = 0;
  size_t prefill_scope_worker_bytes = 0;
  size_t prefill_aggregate_worker_values = 0;
  size_t prefill_aggregate_worker_bytes = 0;
  size_t prefill_identity_bytes = 0;
  if (!NumericUtils::checked_multiply(layer_count, sizeof(LlmByteRange),
                                      weight_layer_bytes) ||
      !NumericUtils::checked_multiply(worker_count,
                                      sizeof(LlmWorkerWorkPlan),
                                      worker_object_bytes) ||
      !NumericUtils::checked_multiply(worker_count, layer_count,
                                      layer_template_count) ||
      !NumericUtils::checked_multiply(layer_template_count,
                                      sizeof(LlmLayerRangeTemplate),
                                      layer_template_bytes) ||
      !NumericUtils::checked_multiply(worker_count,
                                      sequence_count_per_worker,
                                      sequence_template_count) ||
      !NumericUtils::checked_multiply(
          sequence_template_count,
          prefill_phase
              ? paged_layout
                    ? sizeof(LlmPagedPrefillKvAssignmentTemplate)
                    : sizeof(LlmPrefillKvSequenceRangeTemplate)
          : paged_layout ? sizeof(LlmPagedKvAssignmentTemplate)
                       : sizeof(LlmKvSequenceRangeTemplate),
          sequence_template_bytes) ||
      !NumericUtils::checked_multiply(
          ownership_assignment_count, sizeof(LlmKvCpuBlockAssignment),
          ownership_assignment_bytes) ||
      !NumericUtils::checked_multiply(
          paged_layout && !prefill_phase ? worker_count : 0,
          sizeof(size_t),
          ownership_worker_bytes) ||
      !NumericUtils::checked_multiply(
          prefill_scope_count, sizeof(LlmPrefillCpuOwnershipPlan),
          prefill_scope_bytes) ||
      !NumericUtils::checked_multiply(
          prefill_assignment_count, sizeof(LlmPrefillCpuAssignment),
          prefill_assignment_bytes) ||
      !NumericUtils::checked_multiply(prefill_scope_count, worker_count,
                                      prefill_scope_worker_values) ||
      !NumericUtils::checked_multiply(prefill_scope_worker_values, 5,
                                      prefill_scope_worker_values) ||
      !NumericUtils::checked_multiply(prefill_scope_worker_values,
                                      sizeof(size_t),
                                      prefill_scope_worker_bytes) ||
      !NumericUtils::checked_multiply(
          prefill_phase ? kLlmScenarioCount : 0, worker_count,
          prefill_aggregate_worker_values) ||
      !NumericUtils::checked_multiply(prefill_aggregate_worker_values,
                                      sizeof(size_t),
                                      prefill_aggregate_worker_bytes) ||
      (prefill_phase &&
       !calculate_prefill_identity_preflight_bytes(
           prefill_scope_count, prefill_assignment_count, worker_count,
           prefill_identity_bytes))) {
    return false;
  }

  size_t outer_storage_bytes = 0;
  size_t template_storage_bytes = 0;
  size_t ownership_storage_bytes = 0;
  size_t prefill_storage_bytes = 0;
  return NumericUtils::checked_add(weight_layer_bytes, worker_object_bytes,
                                   outer_storage_bytes) &&
         NumericUtils::checked_add(layer_template_bytes,
                                   sequence_template_bytes,
                                   template_storage_bytes) &&
         NumericUtils::checked_add(ownership_assignment_bytes,
                                   ownership_worker_bytes,
                                   ownership_storage_bytes) &&
         NumericUtils::checked_add(prefill_scope_bytes,
                                   prefill_assignment_bytes,
                                   prefill_storage_bytes) &&
         NumericUtils::checked_add(prefill_storage_bytes,
                                   prefill_scope_worker_bytes,
                                   prefill_storage_bytes) &&
         NumericUtils::checked_add(prefill_storage_bytes,
                                   prefill_aggregate_worker_bytes,
                                   prefill_storage_bytes) &&
         NumericUtils::checked_add(prefill_storage_bytes,
                                   prefill_identity_bytes,
                                   prefill_storage_bytes) &&
         NumericUtils::checked_add(outer_storage_bytes,
                                   template_storage_bytes,
                                   planner_storage_bytes) &&
         NumericUtils::checked_add(planner_storage_bytes,
                                   ownership_storage_bytes,
                                   planner_storage_bytes) &&
         NumericUtils::checked_add(planner_storage_bytes,
                                   prefill_storage_bytes,
                                   planner_storage_bytes);
}

bool add_allocation_capacity(size_t capacity, size_t element_bytes,
                             size_t& total_bytes) {
  size_t allocation_bytes = 0;
  return NumericUtils::checked_multiply(capacity, element_bytes,
                                        allocation_bytes) &&
         NumericUtils::checked_add(total_bytes, allocation_bytes,
                                   total_bytes);
}

/** Account only external string storage; inline small-string bytes live in the
 * containing planner object that is already retained by its owning vector. */
bool add_external_string_capacity(const std::string& value,
                                  size_t& total_bytes) {
  const std::uintptr_t data =
      reinterpret_cast<std::uintptr_t>(value.data());
  const std::uintptr_t object =
      reinterpret_cast<std::uintptr_t>(&value);
  const bool inline_storage =
      data >= object && data < object + sizeof(value);
  if (inline_storage || value.capacity() == 0) {
    return true;
  }
  size_t allocation_bytes = 0;
  return NumericUtils::checked_add(value.capacity(), size_t{1},
                                   allocation_bytes) &&
         NumericUtils::checked_add(total_bytes, allocation_bytes,
                                   total_bytes);
}

bool calculate_actual_planner_storage_bytes(
    const LlmMemoryWorkPlan& plan, size_t& planner_storage_bytes) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    return false;
  }
  planner_storage_bytes = 0;
  if (!add_allocation_capacity(plan.weight_layers.capacity(),
                               sizeof(LlmByteRange),
                               planner_storage_bytes) ||
      !add_allocation_capacity(cpu_plan->workers.capacity(),
                               sizeof(LlmWorkerWorkPlan),
                               planner_storage_bytes)) {
    return false;
  }
  for (const LlmWorkerWorkPlan& worker : cpu_plan->workers) {
    if (!add_allocation_capacity(worker.layers.capacity(),
                                 sizeof(LlmLayerRangeTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(worker.sequences.capacity(),
                                 sizeof(LlmKvSequenceRangeTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(
            worker.paged_assignments.capacity(),
            sizeof(LlmPagedKvAssignmentTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(
            worker.prefill_sequences.capacity(),
            sizeof(LlmPrefillKvSequenceRangeTemplate),
            planner_storage_bytes) ||
        !add_allocation_capacity(
            worker.paged_prefill_assignments.capacity(),
            sizeof(LlmPagedPrefillKvAssignmentTemplate),
            planner_storage_bytes)) {
      return false;
    }
  }
  if (cpu_plan->paged.has_value() && plan.phase == LlmPhase::Decode &&
      (!add_allocation_capacity(
           cpu_plan->paged->ownership.assignments.capacity(),
           sizeof(LlmKvCpuBlockAssignment), planner_storage_bytes) ||
       !add_allocation_capacity(
           cpu_plan->paged->ownership.worker_accounted_bytes_per_work_unit.capacity(),
           sizeof(size_t), planner_storage_bytes))) {
    return false;
  }
  if (cpu_plan->prefill.has_value()) {
    if (!add_external_string_capacity(cpu_plan->prefill->identity,
                                      planner_storage_bytes)) {
      return false;
    }
    for (const LlmPrefillCpuScenarioExecutionPlan& scenario :
         cpu_plan->prefill->scenarios) {
      if (!add_allocation_capacity(
              scenario.ownership_scopes.capacity(),
              sizeof(LlmPrefillCpuOwnershipPlan), planner_storage_bytes) ||
          !add_allocation_capacity(
              scenario.worker_accounted_bytes_per_work_unit.capacity(),
              sizeof(size_t), planner_storage_bytes) ||
          !add_external_string_capacity(scenario.identity,
                                        planner_storage_bytes)) {
        return false;
      }
      for (const LlmPrefillCpuOwnershipPlan& scope :
           scenario.ownership_scopes) {
        if (!add_allocation_capacity(scope.assignments.capacity(),
                                     sizeof(LlmPrefillCpuAssignment),
                                     planner_storage_bytes) ||
            !add_allocation_capacity(scope.worker_weight_shard_bytes.capacity(),
                                     sizeof(size_t), planner_storage_bytes) ||
            !add_allocation_capacity(scope.worker_kv_model_payload_bytes.capacity(),
                                     sizeof(size_t), planner_storage_bytes) ||
            !add_allocation_capacity(scope.worker_layout_metadata_lookup_count.capacity(),
                                     sizeof(size_t), planner_storage_bytes) ||
            !add_allocation_capacity(scope.worker_layout_metadata_read_bytes.capacity(),
                                     sizeof(size_t), planner_storage_bytes) ||
            !add_allocation_capacity(scope.worker_scenario_accounted_bytes.capacity(),
                                     sizeof(size_t), planner_storage_bytes)) {
          return false;
        }
        if (!add_external_string_capacity(scope.reason_code,
                                          planner_storage_bytes) ||
            !add_external_string_capacity(scope.identity,
                                          planner_storage_bytes)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool finalize_plan_identities(LlmMemoryWorkPlan& plan) {
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  const bool prefill = plan.phase == LlmPhase::Prefill;
  plan.methodology_version =
      build_llm_methodology_version(plan.backend, plan.phase,
                                   plan.kv_layout);
  plan.component_identities.logical_profile_version =
      prefill ? Constants::LLM_PREFILL_LOGICAL_PROFILE_VERSION
              : Constants::LLM_LOGICAL_PROFILE_VERSION;
  plan.component_identities.kv_layout_version =
      paged_layout ? Constants::LLM_PAGED_KV_LAYOUT_VERSION
                   : Constants::LLM_CONTIGUOUS_KV_LAYOUT_VERSION;
  if (paged_layout) {
    plan.component_identities.permutation_version =
        Constants::LLM_KV_BLOCK_PERMUTATION_VERSION;
  } else {
    plan.component_identities.permutation_version.reset();
  }
  const LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  const bool executable_prefill =
      prefill && cpu_plan != nullptr && cpu_plan->prefill.has_value();
  plan.component_identities.backend_executor_version =
      executable_prefill
          ? paged_layout
                ? Constants::LLM_PREFILL_PAGED_CPU_EXECUTOR_VERSION
                : Constants::LLM_PREFILL_CPU_EXECUTOR_VERSION
          : prefill ? Constants::LLM_PREFILL_PLANNED_EXECUTOR_VERSION
              : paged_layout ? Constants::LLM_PAGED_CPU_EXECUTOR_VERSION
                             : Constants::LLM_CPU_EXECUTOR_VERSION;
  plan.component_identities.resource_abi_version =
      executable_prefill
          ? paged_layout
                ? Constants::LLM_PREFILL_PAGED_DESCRIPTOR_ABI_VERSION
                : Constants::LLM_PREFILL_DESCRIPTOR_ABI_VERSION
          : prefill ? Constants::LLM_PREFILL_RESOURCE_PLAN_VERSION
              : paged_layout ? Constants::LLM_PAGED_DESCRIPTOR_ABI_VERSION
                             : Constants::LLM_DESCRIPTOR_ABI_VERSION;
  plan.component_identities.schedule_version =
      prefill ? LlmPrefillVersion::OWNER_LOCAL_SCHEDULE
              : paged_layout ? Constants::LLM_PAGED_CPU_SCHEDULE_VERSION
                             : Constants::LLM_CPU_SCHEDULE_VERSION;
  plan.component_identities.timer_policy_version =
      Constants::LLM_CPU_TIMER_POLICY_VERSION;
  plan.component_identities.buffer_pattern_version =
      paged_layout ? Constants::LLM_PAGED_BUFFER_PATTERN_VERSION
                   : Constants::LLM_BUFFER_PATTERN_VERSION;
  plan.component_identities.write_pattern_version =
      prefill ? LlmPrefillVersion::WRITE_PATTERN
              : Constants::LLM_APPEND_PATTERN_VERSION;
  plan.component_identities.checksum_pattern_version =
      prefill ? paged_layout ? LlmPrefillVersion::PAGED_CHECKSUM_ORACLE
                               : LlmPrefillVersion::CHECKSUM_ORACLE
              : paged_layout ? Constants::LLM_PAGED_READ_CHECKSUM_VERSION
                             : Constants::LLM_READ_CHECKSUM_VERSION;
  plan.component_identities.msl_revision.reset();
  plan.component_identities.msl_source_sha256.reset();
  plan.component_identities.identity =
      serialize_llm_component_identities(plan.component_identities);
  plan.plan_identity = build_model_plan_identity(plan);
  return !plan.methodology_version.empty() &&
         !plan.component_identities.identity.empty() &&
         !plan.plan_identity.empty();
}

bool prepare_paged_identity_shape(LlmMemoryWorkPlan& plan) {
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !cpu_plan->paged.has_value()) {
    return false;
  }
  LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
  paged.block_table_logical_bytes = paged.layout.memory.block_table_bytes;
  paged.block_table_mapping_bytes =
      plan.memory_budget.request.committed_block_table_mapping_bytes;
  paged.block_table_read_only = false;
  paged.table_validation.valid = true;
  paged.table_validation.interrupted = false;
  paged.table_validation.reason_code = LlmKvLayoutReason::VALID;
  paged.table_validation.expected_entries =
      paged.layout.block_table_entries;
  paged.table_validation.examined_entries =
      paged.layout.block_table_entries;
  paged.table_validation.validation_bitset_bytes =
      paged.layout.memory.validation_bitset_bytes;
  paged.permutation = build_llm_kv_permutation_identity(
      paged.layout, derive_llm_kv_permutation_seed(plan.base_seed),
      std::string(64, '0'));
  paged.layout_identity =
      serialize_llm_kv_layout_identity(paged.layout, paged.permutation);
  paged.execution_identity =
      plan.phase == LlmPhase::Prefill
          ? cpu_plan->prefill.has_value() ? cpu_plan->prefill->identity
                                          : std::string{}
          : paged.ownership.identity;
  return !paged.permutation.identity.empty() &&
         !paged.layout_identity.empty() &&
         !paged.execution_identity.empty();
}

bool add_identity_bytes(size_t bytes, size_t& total) {
  return NumericUtils::checked_add(total, bytes, total);
}

bool build_auxiliary_preflight_view(
    LlmMemoryWorkPlan& plan, LlmAuxiliaryPreflightView& view) {
  view = {};
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !plan.memory_budget.valid ||
      plan.plan_identity.empty() || cpu_plan->effective_workers == 0) {
    return false;
  }

  const bool original_valid = plan.valid;
  plan.valid = true;
  view.backend = plan.backend;
  view.kv_layout = plan.kv_layout;
  view.effective_workers = cpu_plan->effective_workers;
  view.total_layer_descriptors = cpu_plan->total_layer_descriptors;
  view.total_sequence_descriptors = cpu_plan->total_sequence_descriptors;
  view.k_or_v_static_reference_count =
      plan.kv_layout == LlmKvLayout::Paged
          ? cpu_plan->paged.has_value()
                ? cpu_plan->paged->layout.total_physical_blocks
                : plan.geometry.total_physical_blocks
          : plan.phase == LlmPhase::Prefill
                ? 0
                : cpu_plan->total_sequence_descriptors;
  view.model_plan_identity_bytes = plan.plan_identity.size();

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly,
      LlmScenario::Mixed};
  std::array<size_t, kLlmScenarioCount> maximum_work_units{};
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioLimits limits =
        calculate_llm_scenario_limits(plan.geometry, kScenarios[index]);
    if (!limits.valid) {
      plan.valid = original_valid;
      return false;
    }
    maximum_work_units[index] = limits.effective_maximum_work_units;
    const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
        plan, kScenarios[index], maximum_work_units[index], false);
    if (!scenario.valid) {
      plan.valid = original_valid;
      return false;
    }
    view.maximum_scenario_plan_identity_bytes[index] =
        scenario.plan_identity.size();
  }
  const LlmFrozenScenarioPlans frozen =
      freeze_llm_scenario_work_plans(plan, maximum_work_units, false);
  if (!frozen.valid) {
    plan.valid = original_valid;
    return false;
  }
  view.frozen_reason_code_bytes = frozen.reason_code.size();
  view.frozen_model_plan_identity_bytes =
      frozen.model_plan_identity.size();
  view.frozen_plan_identity_bytes = frozen.plan_identity.size();
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    view.frozen_scenario_reason_code_bytes[index] =
        frozen.scenarios[index].reason_code.size();
    view.frozen_scenario_model_plan_identity_bytes[index] =
        frozen.scenarios[index].model_plan_identity.size();
    view.frozen_scenario_plan_identity_bytes[index] =
        frozen.scenarios[index].plan_identity.size();
  }

  size_t json_identity_bytes = 0;
  const LlmComponentIdentities& components = plan.component_identities;
  const auto add_string = [&](const std::string& value) {
    return add_identity_bytes(value.size(), json_identity_bytes);
  };
  const auto add_optional = [&](const std::optional<std::string>& value) {
    return !value.has_value() || add_string(*value);
  };
  bool json_valid =
      add_string(plan.plan_identity) &&
      add_string(plan.methodology_version) &&
      add_string(components.logical_profile_version) &&
      add_string(components.kv_layout_version) &&
      add_optional(components.permutation_version) &&
      add_string(components.backend_executor_version) &&
      add_string(components.resource_abi_version) &&
      add_string(components.schedule_version) &&
      add_string(components.timer_policy_version) &&
      add_string(components.buffer_pattern_version) &&
      add_string(components.write_pattern_version) &&
      add_string(components.checksum_pattern_version) &&
      add_optional(components.msl_revision) &&
      add_optional(components.msl_source_sha256) &&
      add_string(components.identity);
  if (json_valid && cpu_plan->paged.has_value()) {
    const LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
    json_valid =
        add_string(paged.layout.geometry_identity) &&
        add_string(paged.layout_identity) &&
        add_string(paged.execution_identity) &&
        add_string(paged.table_validation.reason_code) &&
        add_string(paged.permutation.algorithm_version) &&
        add_string(paged.permutation.domain_uint64_hex) &&
        add_string(paged.permutation.sha256) &&
        add_string(paged.permutation.identity);
    if (json_valid && plan.phase == LlmPhase::Decode) {
      json_valid = add_string(paged.ownership.reason_code) &&
                   add_string(paged.ownership.layout_geometry_identity) &&
                   add_string(paged.ownership.identity);
    }
  }
  if (json_valid && cpu_plan->prefill.has_value()) {
    json_valid = add_string(cpu_plan->prefill->identity);
    for (const LlmPrefillCpuScenarioExecutionPlan& scenario :
         cpu_plan->prefill->scenarios) {
      json_valid = json_valid && add_string(scenario.identity);
      for (const LlmPrefillCpuOwnershipPlan& scope :
           scenario.ownership_scopes) {
        json_valid = json_valid && add_string(scope.reason_code) &&
                     add_string(scope.identity);
      }
    }
  }
  plan.valid = original_valid;
  if (!json_valid) {
    return false;
  }
  view.json_identity_string_bytes = json_identity_bytes;
  view.valid = true;
  return true;
}

bool auxiliary_preflight_views_match(
    const LlmAuxiliaryPreflightView& lhs,
    const LlmAuxiliaryPreflightView& rhs) {
  return lhs.valid && rhs.valid && lhs.backend == rhs.backend &&
         lhs.kv_layout == rhs.kv_layout &&
         lhs.effective_workers == rhs.effective_workers &&
         lhs.total_layer_descriptors == rhs.total_layer_descriptors &&
         lhs.total_sequence_descriptors == rhs.total_sequence_descriptors &&
         lhs.k_or_v_static_reference_count ==
             rhs.k_or_v_static_reference_count &&
         lhs.model_plan_identity_bytes == rhs.model_plan_identity_bytes &&
         lhs.maximum_scenario_plan_identity_bytes ==
             rhs.maximum_scenario_plan_identity_bytes &&
         lhs.frozen_reason_code_bytes == rhs.frozen_reason_code_bytes &&
         lhs.frozen_model_plan_identity_bytes ==
             rhs.frozen_model_plan_identity_bytes &&
         lhs.frozen_plan_identity_bytes == rhs.frozen_plan_identity_bytes &&
         lhs.frozen_scenario_reason_code_bytes ==
             rhs.frozen_scenario_reason_code_bytes &&
         lhs.frozen_scenario_model_plan_identity_bytes ==
             rhs.frozen_scenario_model_plan_identity_bytes &&
         lhs.frozen_scenario_plan_identity_bytes ==
             rhs.frozen_scenario_plan_identity_bytes &&
         lhs.json_identity_string_bytes == rhs.json_identity_string_bytes;
}

void discard_executable_templates(LlmMemoryWorkPlan& plan) {
  std::vector<LlmByteRange>().swap(plan.weight_layers);
  plan.prefill_plan.reset();
  plan.methodology_version.clear();
  plan.component_identities = {};
  plan.plan_identity.clear();
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan != nullptr) {
    std::vector<LlmWorkerWorkPlan>().swap(cpu_plan->workers);
    cpu_plan->paged.reset();
    cpu_plan->prefill.reset();
    cpu_plan->effective_workers = 0;
  }
}

LlmMemoryWorkPlan invalid_config_plan(const std::string& reason_code) {
  LlmMemoryWorkPlan plan;
  plan.reason_code = reason_code;
  return plan;
}

}  // namespace

const LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Cpu) {
    return nullptr;
  }
  return std::get_if<LlmCpuExecutionPlan>(&plan.backend_execution_plan);
}

LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Cpu) {
    return nullptr;
  }
  return std::get_if<LlmCpuExecutionPlan>(&plan.backend_execution_plan);
}

const LlmMetalExecutionPlan* get_llm_metal_execution_plan(
    const LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Metal) {
    return nullptr;
  }
  return std::get_if<LlmMetalExecutionPlan>(&plan.backend_execution_plan);
}

LlmMetalExecutionPlan* get_llm_metal_execution_plan(
    LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Metal) {
    return nullptr;
  }
  return std::get_if<LlmMetalExecutionPlan>(&plan.backend_execution_plan);
}

bool validate_llm_prefill_cpu_execution_evidence(
    const LlmMemoryWorkPlan& plan) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  if (cpu_plan == nullptr || plan.phase != LlmPhase::Prefill ||
      (plan.kv_layout != LlmKvLayout::Contiguous && !paged_layout) ||
      !cpu_plan->prefill.has_value() ||
      cpu_plan->paged.has_value() != paged_layout ||
      cpu_plan->effective_workers == 0 ||
      cpu_plan->workers.size() != cpu_plan->effective_workers ||
      plan.weight_layers.size() != plan.geometry.layer_count ||
      !validate_prefill_plan_noalloc(plan)) {
    return false;
  }
  size_t rows = 0;
  size_t descriptors_per_worker = 0;
  if (!NumericUtils::checked_multiply(
          plan.geometry.layer_count, plan.geometry.batch_size, rows) ||
      !NumericUtils::checked_multiply(rows, kLlmScenarioCount,
                                      descriptors_per_worker) ||
      cpu_plan->prefill
              ->sequence_descriptors_per_scenario_per_worker != rows ||
      cpu_plan->sequence_descriptors_per_worker !=
          descriptors_per_worker) {
    return false;
  }
  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    if (worker.worker_index != worker_index ||
        worker.layers.size() != plan.geometry.layer_count ||
        !worker.sequences.empty() || !worker.paged_assignments.empty() ||
        (paged_layout
             ? !worker.prefill_sequences.empty() ||
                   worker.paged_prefill_assignments.size() !=
                       descriptors_per_worker
             : worker.prefill_sequences.size() != descriptors_per_worker ||
                   !worker.paged_prefill_assignments.empty())) {
      return false;
    }
  }

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly,
      LlmScenario::Mixed};
  const LlmPrefillCpuExecutionPlan& prefill = *cpu_plan->prefill;
  for (size_t scenario_index_value = 0;
       scenario_index_value < kScenarios.size();
       ++scenario_index_value) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        prefill.scenarios[scenario_index_value];
    const size_t expected_scope_count =
        scenario_index_value ==
                static_cast<size_t>(LlmScenario::WeightsOnly)
            ? plan.geometry.layer_count
            : rows;
    if (scenario.scenario != kScenarios[scenario_index_value] ||
        scenario.ownership_scopes.size() != expected_scope_count ||
        scenario.worker_accounted_bytes_per_work_unit.size() !=
            cpu_plan->effective_workers) {
      return false;
    }
  }

  const LlmPrefillPlan& logical_prefill = *plan.prefill_plan;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    const size_t active_weight_workers = std::min(
        layer_range.span_bytes, cpu_plan->effective_workers);
    const size_t base =
        active_weight_workers == 0
            ? 0
            : layer_range.span_bytes / active_weight_workers;
    const size_t remainder =
        active_weight_workers == 0
            ? 0
            : layer_range.span_bytes % active_weight_workers;
    size_t previous_local = 0;
    for (size_t worker = 0; worker < cpu_plan->effective_workers;
         ++worker) {
      LlmByteRange expected;
      if (worker < active_weight_workers) {
        const size_t next_local =
            worker + 1 == active_weight_workers
                ? layer_range.span_bytes
                : select_partition_boundary(
                      layer_range.offset_bytes, layer_range.span_bytes,
                      active_weight_workers, base, remainder, worker + 1,
                      previous_local);
        size_t expected_offset = 0;
        if (!NumericUtils::checked_add(layer_range.offset_bytes,
                                       previous_local,
                                       expected_offset)) {
          return false;
        }
        expected = {expected_offset, next_local - previous_local};
        previous_local = next_local;
      }
      const LlmLayerRangeTemplate& retained =
          cpu_plan->workers[worker].layers[layer];
      if (retained.weight.offset_bytes != expected.offset_bytes ||
          retained.weight.span_bytes != expected.span_bytes ||
          retained.first_sequence_index !=
              layer * plan.geometry.batch_size ||
          retained.sequence_count != plan.geometry.batch_size ||
          retained.layer_index != layer) {
        return false;
      }
    }
    if (previous_local != layer_range.span_bytes) {
      return false;
    }

    const LlmPrefillCpuOwnershipPlan& weight_scope =
        prefill.scenarios[static_cast<size_t>(LlmScenario::WeightsOnly)]
            .ownership_scopes[layer];
    if (!validate_prefill_scope_noalloc(
            logical_prefill, *cpu_plan, layer,
            LlmScenario::WeightsOnly,
            layer % cpu_plan->effective_workers, true, weight_scope)) {
      return false;
    }

    for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
      const size_t row = layer * plan.geometry.batch_size + batch;
      for (size_t worker = 0; worker < cpu_plan->effective_workers;
           ++worker) {
        if (paged_layout) {
          const LlmPagedPrefillKvAssignmentTemplate& weights_descriptor =
              cpu_plan->workers[worker].paged_prefill_assignments[row];
          if (weights_descriptor.first_logical_block != 0 ||
              weights_descriptor.block_count != 0 ||
              weights_descriptor.layer_index != layer ||
              weights_descriptor.batch_sequence_index != batch) {
            return false;
          }
        } else {
          const LlmPrefillKvSequenceRangeTemplate& weights_descriptor =
              cpu_plan->workers[worker].prefill_sequences[row];
          if (weights_descriptor.first_token != 0 ||
              weights_descriptor.owned_token_count != 0 ||
              weights_descriptor.k_owned.offset_bytes != 0 ||
              weights_descriptor.k_owned.span_bytes != 0 ||
              weights_descriptor.v_owned.offset_bytes != 0 ||
              weights_descriptor.v_owned.span_bytes != 0 ||
              weights_descriptor.layer_index != layer ||
              weights_descriptor.batch_sequence_index != batch) {
            return false;
          }
        }
      }

      const size_t rotation = row % cpu_plan->effective_workers;
      size_t row_offset = 0;
      if (!paged_layout && !NumericUtils::checked_multiply(
              row, plan.geometry.k_or_v_sequence_visible_bytes,
              row_offset)) {
        return false;
      }
      for (LlmScenario scenario_kind :
           {LlmScenario::KvOnly, LlmScenario::Mixed}) {
        const size_t scenario_index_value =
            static_cast<size_t>(scenario_kind);
        const LlmPrefillCpuOwnershipPlan& scope =
            prefill.scenarios[scenario_index_value]
                .ownership_scopes[row];
        const bool include_weight_shards =
            scenario_kind == LlmScenario::Mixed && batch == 0;
        if (!validate_prefill_scope_noalloc(
                logical_prefill, *cpu_plan, layer, scenario_kind,
                rotation, include_weight_shards, scope)) {
          return false;
        }
        const size_t scenario_base = scenario_index_value * rows;
        for (size_t worker = 0; worker < cpu_plan->effective_workers;
             ++worker) {
          const size_t rank =
              worker >= rotation
                  ? worker - rotation
                  : worker + (cpu_plan->effective_workers - rotation);
          size_t expected_first_unit = 0;
          size_t expected_unit_count = 0;
          if (rank < scope.active_worker_count) {
            const LlmPrefillCpuAssignment& assignment =
                scope.assignments[rank];
            expected_first_unit = assignment.first_unit;
            expected_unit_count = assignment.unit_count;
          }
          if (paged_layout) {
            const LlmPagedPrefillKvAssignmentTemplate& retained =
                cpu_plan->workers[worker].paged_prefill_assignments[
                    scenario_base + row];
            if (retained.first_logical_block != expected_first_unit ||
                retained.block_count != expected_unit_count ||
                retained.layer_index != layer ||
                retained.batch_sequence_index != batch) {
              return false;
            }
            continue;
          }

          LlmByteRange expected_owned;
          if (rank < scope.active_worker_count) {
            size_t token_offset = 0;
            if (!NumericUtils::checked_multiply(
                    expected_first_unit,
                    plan.geometry.k_or_v_record_bytes_per_layer,
                    token_offset) ||
                !NumericUtils::checked_add(
                    row_offset, token_offset,
                    expected_owned.offset_bytes) ||
                !NumericUtils::checked_multiply(
                    expected_unit_count,
                    plan.geometry.k_or_v_record_bytes_per_layer,
                    expected_owned.span_bytes)) {
              return false;
            }
          }
          const LlmPrefillKvSequenceRangeTemplate& retained =
              cpu_plan->workers[worker]
                  .prefill_sequences[scenario_base + row];
          if (retained.first_token != expected_first_unit ||
              retained.owned_token_count != expected_unit_count ||
              retained.k_owned.offset_bytes !=
                  expected_owned.offset_bytes ||
              retained.k_owned.span_bytes != expected_owned.span_bytes ||
              retained.v_owned.offset_bytes !=
                  expected_owned.offset_bytes ||
              retained.v_owned.span_bytes != expected_owned.span_bytes ||
              retained.layer_index != layer ||
              retained.batch_sequence_index != batch) {
            return false;
          }
        }
      }
    }
  }

  for (size_t scenario_index_value = 0;
       scenario_index_value < kScenarios.size();
       ++scenario_index_value) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        prefill.scenarios[scenario_index_value];
    size_t minimum = std::numeric_limits<size_t>::max();
    size_t maximum = 0;
    size_t scenario_total = 0;
    for (size_t worker = 0; worker < cpu_plan->effective_workers;
         ++worker) {
      size_t total = 0;
      for (const LlmPrefillCpuOwnershipPlan& scope :
           scenario.ownership_scopes) {
        if (!NumericUtils::checked_add(
                total, scope.worker_scenario_accounted_bytes[worker],
                total)) {
          return false;
        }
      }
      if (scenario.worker_accounted_bytes_per_work_unit[worker] != total) {
        return false;
      }
      if (!NumericUtils::checked_add(scenario_total, total,
                                     scenario_total)) {
        return false;
      }
      minimum = std::min(minimum, total);
      maximum = std::max(maximum, total);
    }
    const size_t expected_scenario_total =
        scenario.scenario == LlmScenario::WeightsOnly
            ? logical_prefill.weight_read_bytes_per_work_unit
            : scenario.scenario == LlmScenario::KvOnly
                  ? logical_prefill.kv_only_payload_bytes_per_work_unit +
                        logical_prefill
                            .layout_metadata_read_bytes_per_work_unit
                  : logical_prefill.mixed_payload_bytes_per_work_unit +
                        logical_prefill
                            .layout_metadata_read_bytes_per_work_unit;
    if (scenario_total != expected_scenario_total ||
        scenario.minimum_worker_accounted_bytes_per_work_unit != minimum ||
        scenario.maximum_worker_accounted_bytes_per_work_unit != maximum ||
        scenario.worker_accounted_imbalance_bytes_per_work_unit !=
            maximum - minimum ||
        !match_prefill_scenario_identity_noalloc(scenario)) {
      return false;
    }
  }
  if (paged_layout) {
    const LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
    const LlmKvLayoutPlan& layout = paged.layout;
    if (!layout.valid ||
        layout.sequence_tokens != logical_prefill.prompt_tokens ||
        layout.kv_block_tokens != logical_prefill.kv_block_tokens ||
        layout.layer_count != plan.geometry.layer_count ||
        layout.batch_size != plan.geometry.batch_size ||
        layout.k_or_v_record_bytes_per_layer !=
            plan.geometry.k_or_v_record_bytes_per_layer ||
        layout.blocks_per_sequence != logical_prefill.blocks_per_sequence ||
        layout.total_physical_blocks !=
            plan.geometry.total_physical_blocks ||
        layout.block_bytes != plan.geometry.kv_block_bytes ||
        layout.last_block_valid_bytes !=
            plan.geometry.last_block_valid_bytes ||
        !validate_llm_kv_layout_identity(
            layout, paged.permutation, paged.layout_identity) ||
        paged.execution_identity != prefill.identity ||
        paged.permutation.entry_count != layout.block_table_entries) {
      return false;
    }
  }
  return match_prefill_execution_identity_noalloc(prefill) &&
         match_prefill_model_plan_identity_noalloc(plan, *cpu_plan);
}

std::string build_llm_methodology_version(LlmMemoryBackend backend,
                                          LlmPhase phase,
                                          LlmKvLayout layout) {
  std::string methodology = "llm-memory-v1-";
  methodology += llm_memory_backend_to_string(backend);
  methodology += '-';
  methodology += llm_phase_to_string(phase);
  methodology += '-';
  methodology += llm_kv_layout_to_string(layout);
  return methodology;
}

std::string serialize_llm_component_identities(
    const LlmComponentIdentities& components) {
  std::string identity = Constants::LLM_COMPONENT_IDENTITY_VERSION;
  append_component_identity(identity, "logical_profile_version",
                            components.logical_profile_version);
  append_component_identity(identity, "kv_layout_version",
                            components.kv_layout_version);
  append_component_identity(identity, "permutation_version",
                            components.permutation_version);
  append_component_identity(identity, "backend_executor_version",
                            components.backend_executor_version);
  append_component_identity(identity, "resource_abi_version",
                            components.resource_abi_version);
  append_component_identity(identity, "schedule_version",
                            components.schedule_version);
  append_component_identity(identity, "timer_policy_version",
                            components.timer_policy_version);
  append_component_identity(identity, "buffer_pattern_version",
                            components.buffer_pattern_version);
  append_component_identity(identity, "write_pattern_version",
                            components.write_pattern_version);
  append_component_identity(identity, "checksum_pattern_version",
                            components.checksum_pattern_version);
  append_component_identity(identity, "msl_revision",
                            components.msl_revision);
  append_component_identity(identity, "msl_source_sha256",
                            components.msl_source_sha256);
  return identity;
}

LlmMemoryWorkPlan::LlmMemoryWorkPlan(LlmMemoryWorkPlan&& other) noexcept {
  *this = std::move(other);
}

LlmMemoryWorkPlan& LlmMemoryWorkPlan::operator=(
    LlmMemoryWorkPlan&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  valid = other.valid;
  reason_code = std::move(other.reason_code);
  geometry = std::move(other.geometry);
  prefill_plan = std::move(other.prefill_plan);
  backend = other.backend;
  phase = other.phase;
  kv_layout = other.kv_layout;
  work_unit_kind = other.work_unit_kind;
  weight_passes_per_work_unit = other.weight_passes_per_work_unit;
  kv_replay_factor = other.kv_replay_factor;
  base_seed = other.base_seed;
  weight_buffer_seed = other.weight_buffer_seed;
  k_buffer_seed = other.k_buffer_seed;
  v_buffer_seed = other.v_buffer_seed;
  scenario_seeds = other.scenario_seeds;
  memory_budget = std::move(other.memory_budget);
  weight_layers = std::move(other.weight_layers);
  methodology_version = std::move(other.methodology_version);
  component_identities = std::move(other.component_identities);
  plan_identity = std::move(other.plan_identity);
  backend_execution_plan = std::move(other.backend_execution_plan);

  other.valid = false;
  other.reason_code.clear();
  other.geometry.valid = false;
  other.prefill_plan.reset();
  other.base_seed = 0;
  other.weight_buffer_seed = 0;
  other.k_buffer_seed = 0;
  other.v_buffer_seed = 0;
  other.scenario_seeds = {};
  other.memory_budget.valid = false;
  other.memory_budget.request.valid = false;
  other.weight_layers.clear();
  other.backend = LlmMemoryBackend::Cpu;
  other.phase = LlmPhase::Decode;
  other.kv_layout = LlmKvLayout::Contiguous;
  other.work_unit_kind = LlmWorkUnitKind::DecodeStep;
  other.methodology_version.clear();
  other.component_identities = {};
  other.plan_identity.clear();
  other.backend_execution_plan = LlmCpuExecutionPlan{};
  return *this;
}

uint64_t llm_seed_domain_value(LlmSeedDomain domain) {
  switch (domain) {
    case LlmSeedDomain::WeightBuffer:
      return kWeightBufferSeedDomain;
    case LlmSeedDomain::KBuffer:
      return kKBufferSeedDomain;
    case LlmSeedDomain::VBuffer:
      return kVBufferSeedDomain;
    case LlmSeedDomain::WeightsOnlyScenario:
      return kWeightsOnlyScenarioSeedDomain;
    case LlmSeedDomain::KvOnlyScenario:
      return kKvOnlyScenarioSeedDomain;
    case LlmSeedDomain::MixedScenario:
      return kMixedScenarioSeedDomain;
  }
  return 0;
}

uint64_t derive_llm_domain_seed(uint64_t base_seed, LlmSeedDomain domain) {
  const uint64_t domain_value = llm_seed_domain_value(domain);
  return domain_value == 0 ? 0
                           : SeedUtils::splitmix64(base_seed ^ domain_value);
}

LlmGeometry resolve_llm_geometry(const LlmGeometryRequest& request) {
  LlmGeometry geometry;
  geometry.phase = request.phase;
  geometry.kv_layout = request.kv_layout;
  geometry.work_unit_kind = llm_work_unit_kind_for_phase(request.phase);
  geometry.active_weight_bytes_per_work_unit = request.active_weight_bytes;
  geometry.layer_count = request.layer_count;
  geometry.query_head_count = request.query_head_count;
  geometry.kv_head_count = request.kv_head_count;
  geometry.head_dimension = request.head_dimension;
  geometry.kv_element_bytes = request.kv_element_bytes;
  geometry.batch_size = request.batch_size;
  geometry.kv_block_tokens = request.kv_block_tokens;

  size_t sequence_tokens = 0;
  switch (request.phase) {
    case LlmPhase::Decode:
      if (request.visible_context_tokens == 0) {
        geometry.reason_code = LlmWorkPlanReason::CONTEXT_TOKENS_ZERO;
        return geometry;
      }
      if (request.prompt_tokens != 0) {
        geometry.reason_code =
            LlmWorkPlanReason::PROMPT_TOKENS_NOT_APPLICABLE;
        return geometry;
      }
      if (request.attention_query_tile_tokens != 0) {
        geometry.reason_code =
            LlmWorkPlanReason::QUERY_TILE_TOKENS_NOT_APPLICABLE;
        return geometry;
      }
      sequence_tokens = request.visible_context_tokens;
      break;
    case LlmPhase::Prefill:
      if (request.visible_context_tokens != 0) {
        geometry.reason_code =
            LlmWorkPlanReason::CONTEXT_TOKENS_NOT_APPLICABLE;
        return geometry;
      }
      sequence_tokens = request.prompt_tokens;
      break;
    default:
      geometry.reason_code = LlmWorkPlanReason::PHASE_NOT_ACTIVATED;
      return geometry;
  }
  if (request.active_weight_bytes == 0) {
    geometry.reason_code = LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO;
    return geometry;
  }
  if (request.layer_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::LAYER_COUNT_ZERO;
    return geometry;
  }
  if (request.query_head_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::QUERY_HEAD_COUNT_ZERO;
    return geometry;
  }
  if (request.kv_head_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::KV_HEAD_COUNT_ZERO;
    return geometry;
  }
  if (request.head_dimension == 0) {
    geometry.reason_code = LlmWorkPlanReason::HEAD_DIMENSION_ZERO;
    return geometry;
  }
  if (!valid_kv_element_bytes(request.kv_element_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES;
    return geometry;
  }
  if (request.phase == LlmPhase::Prefill) {
    if (request.prompt_tokens == 0) {
      geometry.reason_code = LlmPrefillReason::PROMPT_TOKENS_ZERO;
      return geometry;
    }
    if (request.attention_query_tile_tokens == 0) {
      geometry.reason_code = LlmPrefillReason::QUERY_TILE_TOKENS_ZERO;
      return geometry;
    }
    if (request.attention_query_tile_tokens > request.prompt_tokens) {
      geometry.reason_code =
          LlmPrefillReason::QUERY_TILE_TOKENS_EXCEEDS_PROMPT;
      return geometry;
    }
  }
  if (request.kv_layout == LlmKvLayout::Contiguous &&
      request.kv_block_tokens != 0) {
    geometry.reason_code =
        LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE;
    return geometry;
  }
  if (request.batch_size == 0) {
    geometry.reason_code = LlmWorkPlanReason::BATCH_SIZE_ZERO;
    return geometry;
  }
  if (request.query_head_count < request.kv_head_count) {
    geometry.reason_code = LlmWorkPlanReason::QUERY_HEADS_BELOW_KV_HEADS;
    return geometry;
  }
  if (request.query_head_count % request.kv_head_count != 0) {
    geometry.reason_code =
        LlmWorkPlanReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS;
    return geometry;
  }

  geometry.query_heads_per_kv_head =
      request.query_head_count / request.kv_head_count;
  geometry.attention_kind =
      classify_attention(request.query_head_count, request.kv_head_count);
  if (!NumericUtils::checked_multiply(
          request.head_dimension, request.kv_element_bytes,
          geometry.kv_vector_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_VECTOR_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.kv_head_count, geometry.kv_vector_bytes,
          geometry.k_or_v_record_bytes_per_layer)) {
    geometry.reason_code = LlmWorkPlanReason::KV_RECORD_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          geometry.k_or_v_record_bytes_per_layer, 2,
          geometry.kv_record_bytes_per_layer)) {
    geometry.reason_code =
        LlmWorkPlanReason::KV_LAYER_RECORD_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.layer_count, geometry.kv_record_bytes_per_layer,
          geometry.kv_bytes_per_visible_token)) {
    geometry.reason_code = LlmWorkPlanReason::KV_BYTES_PER_TOKEN_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          sequence_tokens,
          geometry.k_or_v_record_bytes_per_layer,
          geometry.k_or_v_sequence_visible_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_SEQUENCE_BYTES_OVERFLOW;
    return geometry;
  }

  size_t layer_batch_count = 0;
  if (!NumericUtils::checked_multiply(request.layer_count, request.batch_size,
                                      layer_batch_count) ||
      !NumericUtils::checked_multiply(
          layer_batch_count, geometry.k_or_v_sequence_visible_bytes,
          geometry.k_logical_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_MAPPING_BYTES_OVERFLOW;
    return geometry;
  }
  geometry.v_logical_bytes = geometry.k_logical_bytes;
  if (request.kv_layout == LlmKvLayout::Paged) {
    const LlmKvLayoutPlan paged_layout = build_llm_kv_layout_plan(
        {sequence_tokens, request.kv_block_tokens,
         request.layer_count, request.batch_size,
         geometry.k_or_v_record_bytes_per_layer});
    if (!paged_layout.valid) {
      geometry.reason_code = paged_layout.reason_code;
      return geometry;
    }
    geometry.kv_blocks_per_sequence = paged_layout.blocks_per_sequence;
    geometry.physical_blocks_per_layer =
        paged_layout.physical_blocks_per_layer;
    geometry.total_physical_blocks = paged_layout.total_physical_blocks;
    geometry.kv_block_bytes = paged_layout.block_bytes;
    geometry.last_block_tokens = paged_layout.last_block_tokens;
    geometry.last_block_valid_bytes = paged_layout.last_block_valid_bytes;
    geometry.decode_append_offset_in_last_block =
        paged_layout.decode_append_offset_in_last_block;
    geometry.k_layout_padding_bytes =
        paged_layout.memory.k_layout_padding_bytes;
    geometry.v_layout_padding_bytes =
        paged_layout.memory.v_layout_padding_bytes;
    geometry.block_table_entries = paged_layout.block_table_entries;
    geometry.block_table_bytes = paged_layout.memory.block_table_bytes;
    geometry.k_mapping_bytes = paged_layout.memory.k_physical_bytes;
    geometry.v_mapping_bytes = paged_layout.memory.v_physical_bytes;
  } else if (request.kv_layout == LlmKvLayout::Contiguous) {
    geometry.k_mapping_bytes = geometry.k_logical_bytes;
    geometry.v_mapping_bytes = geometry.v_logical_bytes;
  } else {
    geometry.reason_code = LlmWorkPlanReason::KV_LAYOUT_NOT_ACTIVATED;
    return geometry;
  }
  if (!NumericUtils::checked_add(geometry.k_mapping_bytes,
                                 geometry.v_mapping_bytes,
                                 geometry.kv_capacity_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_CAPACITY_BYTES_OVERFLOW;
    return geometry;
  }

  geometry.weight_read_bytes_per_work_unit = request.active_weight_bytes;
  if (!NumericUtils::checked_add(request.active_weight_bytes,
                                 geometry.kv_capacity_bytes,
                                 geometry.total_data_mapping_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::TOTAL_DATA_BYTES_OVERFLOW;
    return geometry;
  }
  if (request.phase == LlmPhase::Decode) {
    if (!NumericUtils::checked_multiply(
            request.batch_size, geometry.kv_bytes_per_visible_token,
            geometry.kv_write_bytes_per_work_unit)) {
      geometry.reason_code = LlmWorkPlanReason::KV_WRITE_BYTES_OVERFLOW;
      return geometry;
    }
    if (!NumericUtils::checked_multiply(
            request.visible_context_tokens,
            geometry.kv_write_bytes_per_work_unit,
            geometry.kv_read_bytes_per_work_unit)) {
      geometry.reason_code = LlmWorkPlanReason::KV_READ_BYTES_OVERFLOW;
      return geometry;
    }
    if (request.kv_layout == LlmKvLayout::Paged &&
        (!NumericUtils::checked_multiply(
             geometry.kv_blocks_per_sequence, static_cast<size_t>(2),
             geometry
                 .layout_metadata_lookups_per_layer_sequence_per_work_unit) ||
         !NumericUtils::checked_add(
             geometry.layout_metadata_lookups_per_layer_sequence_per_work_unit,
             static_cast<size_t>(1),
             geometry
                 .layout_metadata_lookups_per_layer_sequence_per_work_unit))) {
      geometry.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
      return geometry;
    }
    geometry.traffic_crossover_numerator = request.active_weight_bytes;
    geometry.traffic_crossover_denominator =
        geometry.kv_write_bytes_per_work_unit;
    geometry.traffic_crossover_context_tokens = static_cast<double>(
        static_cast<long double>(geometry.traffic_crossover_numerator) /
        static_cast<long double>(geometry.traffic_crossover_denominator));
    geometry.decode = LlmDecodeGeometry{request.visible_context_tokens};
    geometry.prefill.reset();
  } else {
    const LlmPrefillPlan prefill = resolve_llm_prefill_plan(
        {request.active_weight_bytes,
         request.prompt_tokens,
         request.attention_query_tile_tokens,
         request.layer_count,
         request.batch_size,
         request.query_head_count,
         request.head_dimension,
         geometry.k_or_v_record_bytes_per_layer,
         request.kv_layout == LlmKvLayout::Paged ? request.kv_block_tokens
                                                 : size_t{0}});
    if (!prefill.valid) {
      geometry.reason_code = prefill.reason_code;
      return geometry;
    }
    if (prefill.k_logical_bytes != geometry.k_logical_bytes ||
        prefill.v_logical_bytes != geometry.v_logical_bytes ||
        prefill.kv_bytes_per_token != geometry.kv_bytes_per_visible_token) {
      geometry.reason_code = LlmPrefillReason::KV_LOGICAL_BYTES_OVERFLOW;
      return geometry;
    }
    geometry.kv_write_bytes_per_work_unit =
        prefill.kv_write_bytes_per_work_unit;
    geometry.kv_read_bytes_per_work_unit =
        prefill.kv_read_bytes_per_work_unit;
    geometry.kv_only_effective_model_payload_bytes_per_work_unit =
        prefill.kv_only_payload_bytes_per_work_unit;
    geometry.mixed_effective_model_payload_bytes_per_work_unit =
        prefill.mixed_payload_bytes_per_work_unit;
    geometry.layout_metadata_lookups_per_layer_sequence_per_work_unit =
        prefill.layout_metadata_lookups_per_layer_sequence;
    geometry.decode_append_offset_in_last_block = 0;
    geometry.decode.reset();
    geometry.prefill = LlmPrefillGeometry{
        prefill.prompt_tokens,
        prefill.attention_query_tile_tokens,
        prefill.tile_count,
        prefill.attention_prefix_token_visits_per_sequence,
        prefill.causal_token_pairs_per_sequence,
        prefill.logical_attention_pairs,
        prefill.logical_attention_fma_terms,
        prefill.prefix_block_visits_per_sequence};
  }
  if (!NumericUtils::checked_add(
          geometry.kv_read_bytes_per_work_unit,
          geometry.kv_write_bytes_per_work_unit,
          geometry.kv_only_effective_model_payload_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::KV_ONLY_PAYLOAD_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_add(
          geometry.weight_read_bytes_per_work_unit,
          geometry.kv_only_effective_model_payload_bytes_per_work_unit,
          geometry.mixed_effective_model_payload_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::MIXED_PAYLOAD_OVERFLOW;
    return geometry;
  }
  geometry.valid = true;
  geometry.reason_code = LlmWorkPlanReason::VALID;
  return geometry;
}

LlmMemoryBudgetRequest build_llm_memory_budget_request(
    const LlmGeometry& geometry, size_t descriptor_bytes,
    size_t planner_storage_bytes,
    size_t checksum_auxiliary_bytes, size_t orchestration_auxiliary_bytes,
    size_t mapping_granularity_bytes, size_t block_table_mapping_bytes,
    size_t layout_transient_bytes) {
  LlmMemoryBudgetRequest request;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.descriptor_bytes = descriptor_bytes;
  request.planner_storage_bytes = planner_storage_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  request.requested_block_table_mapping_bytes = block_table_mapping_bytes;
  request.layout_transient_bytes = layout_transient_bytes;
  if (!geometry.valid) {
    request.reason_code = geometry.reason_code;
    return request;
  }
  if (mapping_granularity_bytes == 0) {
    request.reason_code = LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO;
    return request;
  }

  request.requested_weight_mapping_bytes =
      geometry.active_weight_bytes_per_work_unit;
  request.requested_k_mapping_bytes = geometry.k_mapping_bytes;
  request.requested_v_mapping_bytes = geometry.v_mapping_bytes;
  request.requested_data_bytes = geometry.total_data_mapping_bytes;
  if (!NumericUtils::checked_round_up(
          request.requested_weight_mapping_bytes, mapping_granularity_bytes,
          request.committed_weight_mapping_bytes) ||
      !NumericUtils::checked_round_up(
          request.requested_k_mapping_bytes, mapping_granularity_bytes,
          request.committed_k_mapping_bytes) ||
      !NumericUtils::checked_round_up(
          request.requested_v_mapping_bytes, mapping_granularity_bytes,
          request.committed_v_mapping_bytes) ||
      (block_table_mapping_bytes != 0 &&
       !NumericUtils::checked_round_up(
           block_table_mapping_bytes, mapping_granularity_bytes,
           request.committed_block_table_mapping_bytes))) {
    request.reason_code = LlmWorkPlanReason::MAPPING_ROUND_UP_OVERFLOW;
    return request;
  }

  size_t committed_kv_bytes = 0;
  if (!NumericUtils::checked_add(request.committed_k_mapping_bytes,
                                 request.committed_v_mapping_bytes,
                                 committed_kv_bytes) ||
      !NumericUtils::checked_add(request.committed_weight_mapping_bytes,
                                 committed_kv_bytes,
                                 request.committed_data_bytes)) {
    request.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return request;
  }
  size_t descriptor_and_planner = 0;
  size_t checksum_and_orchestration = 0;
  if (!NumericUtils::checked_add(descriptor_bytes, planner_storage_bytes,
                                 descriptor_and_planner) ||
      !NumericUtils::checked_add(checksum_auxiliary_bytes,
                                 orchestration_auxiliary_bytes,
                                 checksum_and_orchestration) ||
      !NumericUtils::checked_add(descriptor_and_planner,
                                 checksum_and_orchestration,
                                 request.auxiliary_bytes)) {
    request.reason_code = LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW;
    return request;
  }
  size_t resident_runtime_bytes = 0;
  if (!NumericUtils::checked_add(
          request.committed_data_bytes,
          request.committed_block_table_mapping_bytes,
          resident_runtime_bytes) ||
      !NumericUtils::checked_add(resident_runtime_bytes,
                                 request.auxiliary_bytes,
                                 request.runtime_peak_bytes) ||
      !NumericUtils::checked_add(
          request.committed_block_table_mapping_bytes,
          request.layout_transient_bytes, request.setup_peak_bytes) ||
      !NumericUtils::checked_add(request.setup_peak_bytes,
                                 request.planner_storage_bytes,
                                 request.setup_peak_bytes)) {
    request.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return request;
  }
  request.required_total_bytes =
      std::max(request.setup_peak_bytes, request.runtime_peak_bytes);

  request.valid = true;
  request.reason_code = LlmWorkPlanReason::VALID;
  return request;
}

LlmMemoryBudget evaluate_llm_memory_budget(
    const LlmMemoryBudgetRequest& request, size_t available_memory_bytes) {
  LlmMemoryBudget budget;
  budget.request = request;
  budget.available_memory_bytes = available_memory_bytes;
  if (!request.valid) {
    budget.reason_code = request.reason_code;
    return budget;
  }

  if (available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.allowed_memory_bytes)) {
      budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return budget;
    }
  } else {
    const long double scaled =
        static_cast<long double>(available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled >
        static_cast<long double>(std::numeric_limits<size_t>::max())) {
      budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return budget;
    }
    budget.allowed_memory_bytes = static_cast<size_t>(scaled);
  }

  if (request.required_total_bytes > budget.allowed_memory_bytes) {
    budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return budget;
  }
  budget.valid = true;
  budget.reason_code = LlmWorkPlanReason::WITHIN_MEMORY_BUDGET;
  return budget;
}

LlmMemoryWorkPlan build_llm_memory_work_plan_candidate(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlan plan;
  plan.backend = request.backend;
  if (request.backend == LlmMemoryBackend::Metal) {
    plan.backend_execution_plan = LlmMetalExecutionPlan{};
  }
  plan.phase = request.geometry.phase;
  plan.kv_layout = request.geometry.kv_layout;
  plan.work_unit_kind = llm_work_unit_kind_for_phase(plan.phase);
  plan.base_seed = request.base_seed;
  plan.weight_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::WeightBuffer);
  plan.k_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::KBuffer);
  plan.v_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::VBuffer);
  plan.scenario_seeds = {
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::WeightsOnlyScenario),
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::KvOnlyScenario),
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::MixedScenario)};
  if (request.backend != LlmMemoryBackend::Cpu) {
    plan.reason_code = LlmWorkPlanReason::BACKEND_NOT_ACTIVATED;
    return plan;
  }
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    plan.reason_code = LlmWorkPlanReason::BACKEND_NOT_ACTIVATED;
    return plan;
  }
  cpu_plan->requested_workers = request.requested_workers;
  cpu_plan->available_workers = request.available_workers;
  plan.geometry = resolve_llm_geometry(request.geometry);
  if (!plan.geometry.valid) {
    plan.reason_code = plan.geometry.reason_code;
    return plan;
  }
  const bool prefill_phase = plan.phase == LlmPhase::Prefill;
  if (prefill_phase) {
    plan.prefill_plan = resolve_llm_prefill_plan(
        {request.geometry.active_weight_bytes,
         request.geometry.prompt_tokens,
         request.geometry.attention_query_tile_tokens,
         request.geometry.layer_count,
         request.geometry.batch_size,
         request.geometry.query_head_count,
         request.geometry.head_dimension,
         plan.geometry.k_or_v_record_bytes_per_layer,
         plan.kv_layout == LlmKvLayout::Paged
             ? request.geometry.kv_block_tokens
             : size_t{0}});
    if (!plan.prefill_plan->valid) {
      plan.reason_code = plan.prefill_plan->reason_code;
      return plan;
    }
  }
  if (!json_integer_is_safe(request.requested_workers) ||
      !json_integer_is_safe(request.available_workers)) {
    plan.reason_code = LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }
  if (request.requested_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::REQUESTED_WORKERS_ZERO;
    return plan;
  }
  if (request.available_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::AVAILABLE_WORKERS_ZERO;
    return plan;
  }

  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  const size_t sequence_tokens =
      prefill_phase ? request.geometry.prompt_tokens
                    : request.geometry.visible_context_tokens;
  LlmKvLayoutPlan paged_geometry;
  size_t paged_layer_sequence_count = 0;
  size_t paged_worker_coverage_limit = 0;
  if (paged_layout) {
    paged_geometry = build_llm_kv_layout_plan(
        {sequence_tokens,
         request.geometry.kv_block_tokens, request.geometry.layer_count,
         request.geometry.batch_size,
         plan.geometry.k_or_v_record_bytes_per_layer});
    if (!paged_geometry.valid) {
      plan.reason_code = paged_geometry.reason_code;
      return plan;
    }
    if (!NumericUtils::checked_multiply(
            paged_geometry.layer_count, paged_geometry.batch_size,
            paged_layer_sequence_count) ||
        !NumericUtils::checked_add(
            paged_layer_sequence_count,
            paged_geometry.blocks_per_sequence - 1,
            paged_worker_coverage_limit)) {
      plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
      return plan;
    }
  }

  const size_t weight_layer_base =
      request.geometry.active_weight_bytes / request.geometry.layer_count;
  const size_t weight_layer_remainder =
      request.geometry.active_weight_bytes % request.geometry.layer_count;
  const size_t maximum_weight_layer_bytes =
      weight_layer_base + (weight_layer_remainder != 0 ? 1 : 0);
  const size_t maximum_kv_workers =
      paged_layout
          ? prefill_phase
                ? paged_geometry.blocks_per_sequence
                : std::min(paged_geometry.total_physical_blocks,
                           paged_worker_coverage_limit)
          : prefill_phase ? sequence_tokens
                          : plan.geometry.k_or_v_sequence_visible_bytes;
  const size_t maximum_shared_worker_count =
      std::min(maximum_weight_layer_bytes, maximum_kv_workers);
  cpu_plan->effective_workers =
      std::min({request.requested_workers, request.available_workers,
                maximum_shared_worker_count});
  if (cpu_plan->effective_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::NO_EXECUTABLE_WORKER;
    return plan;
  }

  size_t ownership_assignment_count = 0;
  if (paged_layout && !prefill_phase) {
    const size_t active_workers = std::min(
        cpu_plan->effective_workers, paged_geometry.blocks_per_sequence);
    if (!NumericUtils::checked_multiply(
            paged_layer_sequence_count, active_workers,
            ownership_assignment_count)) {
      plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
      return plan;
    }
  }

  const bool executable_prefill = prefill_phase;
  size_t prefill_sequences_per_scenario = 0;
  size_t prefill_scope_count = 0;
  size_t prefill_assignment_count = 0;
  if (executable_prefill) {
    const size_t active_prefill_workers =
        std::min(cpu_plan->effective_workers,
                 paged_layout ? paged_geometry.blocks_per_sequence
                              : sequence_tokens);
    size_t kv_and_mixed_scope_count = 0;
    if (!NumericUtils::checked_multiply(
            plan.geometry.layer_count, plan.geometry.batch_size,
            prefill_sequences_per_scenario) ||
        !NumericUtils::checked_multiply(prefill_sequences_per_scenario, 2,
                                        kv_and_mixed_scope_count) ||
        !NumericUtils::checked_add(plan.geometry.layer_count,
                                   kv_and_mixed_scope_count,
                                   prefill_scope_count) ||
        !NumericUtils::checked_multiply(kv_and_mixed_scope_count,
                                        active_prefill_workers,
                                        prefill_assignment_count)) {
      plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
      return plan;
    }
  }

  cpu_plan->layer_descriptors_per_worker = plan.geometry.layer_count;
  if (executable_prefill) {
    if (!NumericUtils::checked_multiply(
            prefill_sequences_per_scenario, kLlmScenarioCount,
            cpu_plan->sequence_descriptors_per_worker)) {
      plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
      return plan;
    }
  } else if (!prefill_phase) {
    if (!NumericUtils::checked_multiply(
            plan.geometry.layer_count, plan.geometry.batch_size,
            cpu_plan->sequence_descriptors_per_worker)) {
      plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
      return plan;
    }
  }
  if (!NumericUtils::checked_multiply(
          cpu_plan->layer_descriptors_per_worker,
          cpu_plan->effective_workers, cpu_plan->total_layer_descriptors) ||
      !NumericUtils::checked_multiply(
          cpu_plan->sequence_descriptors_per_worker,
          cpu_plan->effective_workers,
          cpu_plan->total_sequence_descriptors)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_COUNT_OVERFLOW;
    return plan;
  }
  size_t layer_descriptor_bytes = 0;
  size_t sequence_descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(
          cpu_plan->total_layer_descriptors,
          executable_prefill
              ? paged_layout ? sizeof(LlmPagedPrefillLayerDescriptor)
                             : sizeof(LlmPrefillLayerDescriptor)
          : paged_layout ? sizeof(LlmPagedLayerDescriptor)
                       : sizeof(LlmLayerDescriptor),
          layer_descriptor_bytes) ||
      !NumericUtils::checked_multiply(
          cpu_plan->total_sequence_descriptors,
          executable_prefill
              ? paged_layout
                    ? sizeof(LlmPagedPrefillKvAssignmentDescriptor)
                    : sizeof(LlmPrefillKvSequenceDescriptor)
          : paged_layout ? sizeof(LlmPagedKvAssignmentDescriptor)
                       : sizeof(LlmKvSequenceDescriptor),
          sequence_descriptor_bytes) ||
      !NumericUtils::checked_add(layer_descriptor_bytes,
                                 sequence_descriptor_bytes,
                                 cpu_plan->descriptor_bytes)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_BYTES_OVERFLOW;
    return plan;
  }

  if (!calculate_planner_storage_bytes(
          plan.geometry.layer_count,
          cpu_plan->sequence_descriptors_per_worker,
          cpu_plan->effective_workers, paged_layout,
          ownership_assignment_count, executable_prefill,
          prefill_scope_count, prefill_assignment_count,
          cpu_plan->planner_storage_bytes)) {
    plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
    return plan;
  }
  if (!json_integer_is_safe(request.geometry.layer_count) ||
      !json_integer_is_safe(request.geometry.query_head_count) ||
      !json_integer_is_safe(request.geometry.kv_head_count) ||
      !json_integer_is_safe(request.geometry.head_dimension) ||
      !json_integer_is_safe(request.geometry.visible_context_tokens) ||
      !json_integer_is_safe(request.geometry.prompt_tokens) ||
      !json_integer_is_safe(
          request.geometry.attention_query_tile_tokens) ||
      !json_integer_is_safe(request.geometry.kv_block_tokens) ||
      !json_integer_is_safe(request.geometry.batch_size) ||
      !json_integer_is_safe(cpu_plan->layer_descriptors_per_worker) ||
      !json_integer_is_safe(cpu_plan->sequence_descriptors_per_worker) ||
      !json_integer_is_safe(cpu_plan->total_layer_descriptors) ||
      !json_integer_is_safe(cpu_plan->total_sequence_descriptors)) {
    plan.reason_code = LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }

  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes,
      paged_layout ? paged_geometry.memory.block_table_bytes : 0,
      paged_layout
          ? paged_geometry.memory.validation_bitset_bytes
          : 0);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  LlmKvCpuOwnershipPlan paged_decode_ownership;
  if (paged_layout && !prefill_phase) {
    paged_decode_ownership =
        build_llm_paged_decode_kv_cpu_ownership_plan(
            paged_geometry, cpu_plan->effective_workers, stop_requested);
    if (!paged_decode_ownership.valid) {
      plan.reason_code = paged_decode_ownership.reason_code;
      return plan;
    }
  }
  if (paged_layout) {
    cpu_plan->paged.emplace();
    cpu_plan->paged->layout = std::move(paged_geometry);
    if (!prefill_phase) {
      cpu_plan->paged->ownership = std::move(paged_decode_ownership);
    }
  }
  if (executable_prefill) {
    cpu_plan->prefill.emplace();
    cpu_plan->prefill->sequence_descriptors_per_scenario_per_worker =
        prefill_sequences_per_scenario;
    constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
        LlmScenario::WeightsOnly, LlmScenario::KvOnly,
        LlmScenario::Mixed};
    for (size_t index = 0; index < kScenarios.size(); ++index) {
      cpu_plan->prefill->scenarios[index].scenario = kScenarios[index];
    }
  }

  try {
    plan.weight_layers.reserve(plan.geometry.layer_count);
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      const size_t layer_bytes =
          weight_layer_base + (layer < weight_layer_remainder ? 1 : 0);
      plan.weight_layers.push_back({weight_offset, layer_bytes});
      weight_offset += layer_bytes;
    }

    cpu_plan->workers.resize(cpu_plan->effective_workers);
    for (size_t worker = 0; worker < cpu_plan->effective_workers; ++worker) {
      cpu_plan->workers[worker].worker_index = worker;
      cpu_plan->workers[worker].layers.reserve(plan.geometry.layer_count);
      if (prefill_phase && paged_layout) {
        cpu_plan->workers[worker].paged_prefill_assignments.resize(
            cpu_plan->sequence_descriptors_per_worker);
        for (size_t scenario = 0; scenario < kLlmScenarioCount; ++scenario) {
          const size_t scenario_base =
              scenario * prefill_sequences_per_scenario;
          for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
            for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
              LlmPagedPrefillKvAssignmentTemplate& assignment =
                  cpu_plan->workers[worker].paged_prefill_assignments[
                      scenario_base + layer * plan.geometry.batch_size +
                      batch];
              assignment.layer_index = layer;
              assignment.batch_sequence_index = batch;
            }
          }
        }
      } else if (paged_layout) {
        cpu_plan->workers[worker].paged_assignments.resize(
            cpu_plan->sequence_descriptors_per_worker);
        for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
          for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
            LlmPagedKvAssignmentTemplate& assignment =
                cpu_plan->workers[worker]
                    .paged_assignments[layer * plan.geometry.batch_size +
                                       batch];
            assignment.layer_index = layer;
            assignment.batch_sequence_index = batch;
          }
        }
      } else if (!prefill_phase) {
        cpu_plan->workers[worker].sequences.reserve(
            cpu_plan->sequence_descriptors_per_worker);
      } else {
        cpu_plan->workers[worker].prefill_sequences.resize(
            cpu_plan->sequence_descriptors_per_worker);
        for (size_t scenario = 0; scenario < kLlmScenarioCount; ++scenario) {
          const size_t scenario_base =
              scenario * prefill_sequences_per_scenario;
          for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
            for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
              LlmPrefillKvSequenceRangeTemplate& sequence =
                  cpu_plan->workers[worker].prefill_sequences[
                      scenario_base + layer * plan.geometry.batch_size +
                      batch];
              sequence.layer_index = layer;
              sequence.batch_sequence_index = batch;
            }
          }
        }
      }
    }

    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      const std::vector<LlmByteRange> weight_ranges = partition_range(
          plan.weight_layers[layer].offset_bytes,
          plan.weight_layers[layer].span_bytes, cpu_plan->effective_workers);
      const size_t first_sequence_index = layer * plan.geometry.batch_size;
      for (size_t worker = 0; worker < cpu_plan->effective_workers; ++worker) {
        const LlmByteRange weight =
            worker < weight_ranges.size() ? weight_ranges[worker]
                                          : LlmByteRange{};
        cpu_plan->workers[worker].layers.push_back(
            {weight, first_sequence_index, plan.geometry.batch_size, layer});
      }

      if (paged_layout || prefill_phase) {
        continue;
      }
      for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
        const size_t sequence_index = first_sequence_index + batch;
        const size_t visible_offset =
            sequence_index * plan.geometry.k_or_v_sequence_visible_bytes;
        const LlmByteRange visible_record{
            visible_offset, plan.geometry.k_or_v_sequence_visible_bytes};
        const size_t append_offset =
            visible_offset +
            (plan.geometry.decode->visible_context_tokens - 1) *
                plan.geometry.k_or_v_record_bytes_per_layer;
        const LlmByteRange append_record{
            append_offset, plan.geometry.k_or_v_record_bytes_per_layer};
        const std::vector<LlmByteRange> visible_ranges = partition_range(
            visible_record.offset_bytes, visible_record.span_bytes,
            cpu_plan->effective_workers);
        for (size_t worker = 0; worker < cpu_plan->effective_workers;
             ++worker) {
          const LlmByteRange visible =
              worker < visible_ranges.size() ? visible_ranges[worker]
                                             : LlmByteRange{};
          const LlmByteRange append = intersect_ranges(visible, append_record);
          const size_t append_record_byte_offset =
              append.span_bytes == 0 ? 0 : append.offset_bytes - append_offset;
          cpu_plan->workers[worker].sequences.push_back(
              {visible, visible, append, append, layer, batch,
               append_record_byte_offset});
        }
      }
    }
    if (executable_prefill) {
      LlmPrefillCpuExecutionPlan& prefill = *cpu_plan->prefill;
      const LlmPrefillPartitionUnitKind partition_unit_kind =
          paged_layout ? LlmPrefillPartitionUnitKind::PagedBlock
                       : LlmPrefillPartitionUnitKind::ContiguousToken;
      for (LlmPrefillCpuScenarioExecutionPlan& scenario :
           prefill.scenarios) {
        const size_t scope_count =
            scenario.scenario == LlmScenario::WeightsOnly
                ? plan.geometry.layer_count
                : prefill_sequences_per_scenario;
        scenario.ownership_scopes.reserve(scope_count);
        scenario.worker_accounted_bytes_per_work_unit.assign(
            cpu_plan->effective_workers, 0);
      }

      for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
        std::vector<size_t> weight_shards(cpu_plan->effective_workers, 0);
        for (size_t worker = 0; worker < cpu_plan->effective_workers;
             ++worker) {
          weight_shards[worker] =
              cpu_plan->workers[worker].layers[layer].weight.span_bytes;
        }

        LlmPrefillCpuOwnershipPlan weights =
            build_llm_prefill_cpu_ownership_plan(
                *plan.prefill_plan,
                {partition_unit_kind,
                 LlmScenario::WeightsOnly, cpu_plan->effective_workers,
                 layer % cpu_plan->effective_workers, weight_shards});
        if (!weights.valid) {
          throw std::length_error("invalid prefill weights ownership");
        }
        LlmPrefillCpuScenarioExecutionPlan& weights_scenario =
            prefill.scenarios[scenario_index(LlmScenario::WeightsOnly)];
        for (size_t worker = 0; worker < cpu_plan->effective_workers;
             ++worker) {
          if (!NumericUtils::checked_add(
                  weights_scenario.worker_accounted_bytes_per_work_unit[worker],
                  weights.worker_scenario_accounted_bytes[worker],
                  weights_scenario.worker_accounted_bytes_per_work_unit[worker])) {
            throw std::length_error("prefill weights accounting overflow");
          }
        }
        weights_scenario.ownership_scopes.push_back(std::move(weights));

        for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
          const size_t sequence_index =
              layer * plan.geometry.batch_size + batch;
          const size_t rotation =
              sequence_index % cpu_plan->effective_workers;
          for (const LlmScenario scenario_kind :
               {LlmScenario::KvOnly, LlmScenario::Mixed}) {
            const std::vector<size_t> mixed_weights =
                scenario_kind == LlmScenario::Mixed && batch == 0
                    ? weight_shards
                    : std::vector<size_t>{};
            LlmPrefillCpuOwnershipPlan ownership =
                build_llm_prefill_cpu_ownership_plan(
                    *plan.prefill_plan,
                    {partition_unit_kind, scenario_kind,
                     cpu_plan->effective_workers, rotation, mixed_weights});
            if (!ownership.valid) {
              throw std::length_error("invalid prefill KV ownership");
            }
            LlmPrefillCpuScenarioExecutionPlan& scenario =
                prefill.scenarios[scenario_index(scenario_kind)];
            for (size_t worker = 0; worker < cpu_plan->effective_workers;
                 ++worker) {
              if (!NumericUtils::checked_add(
                      scenario.worker_accounted_bytes_per_work_unit[worker],
                      ownership.worker_scenario_accounted_bytes[worker],
                      scenario.worker_accounted_bytes_per_work_unit[worker])) {
                throw std::length_error("prefill accounting overflow");
              }
            }
            const size_t scenario_base =
                scenario_index(scenario_kind) *
                prefill_sequences_per_scenario;
            for (const LlmPrefillCpuAssignment& assignment :
                 ownership.assignments) {
              const size_t worker = assignment.worker_index;
              if (worker >= cpu_plan->effective_workers) {
                throw std::length_error("prefill descriptor overflow");
              }
              if (paged_layout) {
                LlmPagedPrefillKvAssignmentTemplate& destination =
                    cpu_plan->workers[worker].paged_prefill_assignments[
                        scenario_base + sequence_index];
                destination.first_logical_block = assignment.first_unit;
                destination.block_count = assignment.unit_count;
                continue;
              }
              size_t row_offset = 0;
              size_t owned_offset = 0;
              size_t owned_bytes = 0;
              if (!NumericUtils::checked_multiply(
                      sequence_index,
                      plan.geometry.k_or_v_sequence_visible_bytes,
                      row_offset) ||
                  !NumericUtils::checked_multiply(
                      assignment.first_unit,
                      plan.geometry.k_or_v_record_bytes_per_layer,
                      owned_offset) ||
                  !NumericUtils::checked_add(row_offset, owned_offset,
                                             owned_offset) ||
                  !NumericUtils::checked_multiply(
                      assignment.unit_count,
                      plan.geometry.k_or_v_record_bytes_per_layer,
                      owned_bytes)) {
                throw std::length_error("prefill descriptor overflow");
              }
              LlmPrefillKvSequenceRangeTemplate& destination =
                  cpu_plan->workers[worker].prefill_sequences[
                      scenario_base + sequence_index];
              destination.k_owned = {owned_offset, owned_bytes};
              destination.v_owned = destination.k_owned;
              destination.first_token = assignment.first_unit;
              destination.owned_token_count = assignment.unit_count;
            }
            scenario.ownership_scopes.push_back(std::move(ownership));
          }
        }
      }

      for (size_t scenario_index_value = 0;
           scenario_index_value < prefill.scenarios.size();
           ++scenario_index_value) {
        LlmPrefillCpuScenarioExecutionPlan& scenario =
            prefill.scenarios[scenario_index_value];
        const LlmScenarioLimits limits = calculate_llm_scenario_limits(
            plan.geometry, scenario.scenario);
        if (!limits.valid) {
          throw std::length_error("invalid prefill scenario limits");
        }
        size_t total = 0;
        for (size_t worker_bytes :
             scenario.worker_accounted_bytes_per_work_unit) {
          if (!NumericUtils::checked_add(total, worker_bytes, total)) {
            throw std::length_error("prefill aggregate accounting overflow");
          }
        }
        if (total != limits.accounted_bytes_per_work_unit) {
          throw std::length_error("prefill aggregate accounting mismatch");
        }
        const auto range = std::minmax_element(
            scenario.worker_accounted_bytes_per_work_unit.begin(),
            scenario.worker_accounted_bytes_per_work_unit.end());
        scenario.minimum_worker_accounted_bytes_per_work_unit = *range.first;
        scenario.maximum_worker_accounted_bytes_per_work_unit = *range.second;
        scenario.worker_accounted_imbalance_bytes_per_work_unit =
            *range.second - *range.first;
        scenario.identity =
            build_prefill_cpu_scenario_execution_identity(scenario);
      }
      prefill.identity = build_prefill_cpu_execution_identity(prefill);
      if (prefill.identity.empty()) {
        throw std::length_error("empty prefill execution identity");
      }
    }
    if (paged_layout && !prefill_phase) {
      for (const LlmKvCpuBlockAssignment& source :
           cpu_plan->paged->ownership.assignments) {
        if (source.worker_index >= cpu_plan->effective_workers ||
            source.layer_index >= plan.geometry.layer_count ||
            source.batch_sequence_index >= plan.geometry.batch_size) {
          throw std::length_error("invalid paged ownership");
        }
        const size_t assignment_index =
            source.layer_index * plan.geometry.batch_size +
            source.batch_sequence_index;
        LlmPagedKvAssignmentTemplate& destination =
            cpu_plan->workers[source.worker_index]
                .paged_assignments[assignment_index];
        if (destination.block_count != 0 || source.block_count == 0) {
          throw std::length_error("duplicate paged ownership");
        }
        destination.first_logical_block = source.first_logical_block;
        destination.block_count = source.block_count;
      }
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  size_t actual_planner_storage_bytes = 0;
  if (!calculate_actual_planner_storage_bytes(
          plan, actual_planner_storage_bytes)) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
    return plan;
  }
  cpu_plan->planner_storage_bytes = actual_planner_storage_bytes;
  const size_t block_table_bytes =
      paged_layout ? cpu_plan->paged->layout.memory.block_table_bytes : 0;
  const size_t layout_transient_bytes =
      paged_layout
          ? cpu_plan->paged->layout.memory.validation_bitset_bytes
          : 0;
  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes, block_table_bytes,
      layout_transient_bytes);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    discard_executable_templates(plan);
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  try {
    if ((paged_layout && !prepare_paged_identity_shape(plan)) ||
        !finalize_plan_identities(plan)) {
      discard_executable_templates(plan);
      plan.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      return plan;
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  plan.valid = false;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlanDraft draft;
  draft.candidate =
      build_llm_memory_work_plan_candidate(request, stop_requested);
  draft.reason_code = draft.candidate.reason_code;
  if (draft.candidate.reason_code != LlmWorkPlanReason::VALID) {
    return draft;
  }
  try {
    if (!build_auxiliary_preflight_view(
            draft.candidate, draft.auxiliary_preflight)) {
      discard_executable_templates(draft.candidate);
      draft.candidate.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      draft.reason_code = draft.candidate.reason_code;
      return draft;
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(draft.candidate);
    draft.candidate.reason_code =
        LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    draft.reason_code = draft.candidate.reason_code;
    return draft;
  } catch (const std::length_error&) {
    discard_executable_templates(draft.candidate);
    draft.candidate.reason_code =
        LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    draft.reason_code = draft.candidate.reason_code;
    return draft;
  }
  draft.valid = true;
  draft.reason_code = LlmWorkPlanReason::VALID;
  return draft;
}

LlmMemoryWorkPlan finalize_llm_memory_work_plan(
    LlmMemoryWorkPlanDraft&& draft, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlan plan = std::move(draft.candidate);
  if (!draft.valid || plan.reason_code != LlmWorkPlanReason::VALID ||
      plan.valid) {
    if (plan.reason_code == LlmWorkPlanReason::VALID) {
      plan.reason_code = draft.reason_code;
    }
    return plan;
  }
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || cpu_plan->effective_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return plan;
  }
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  const size_t block_table_bytes =
      paged_layout ? plan.geometry.block_table_bytes : 0;
  const size_t validation_bitset_bytes =
      paged_layout
          ? cpu_plan->paged->layout.memory.validation_bitset_bytes
          : 0;
  const size_t mapping_granularity_bytes =
      plan.memory_budget.request.mapping_granularity_bytes;
  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, checksum_auxiliary_bytes,
      orchestration_auxiliary_bytes, mapping_granularity_bytes,
      block_table_bytes, validation_bitset_bytes);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, plan.memory_budget.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    discard_executable_templates(plan);
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  try {
    if (paged_layout) {
      if (!cpu_plan->paged.has_value()) {
        plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
        discard_executable_templates(plan);
        return plan;
      }
      LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
      paged.block_table_logical_bytes =
          paged.layout.memory.block_table_bytes;
      paged.block_table_mapping_bytes =
          plan.memory_budget.request.committed_block_table_mapping_bytes;
      paged.block_table_mapping = allocate_buffer(
          paged.block_table_mapping_bytes, "LLM paged KV block table");
      if (paged.block_table_mapping == nullptr) {
        plan.reason_code = LlmWorkPlanReason::BLOCK_TABLE_MAPPING_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      const LlmKvInPlaceBlockTableMaterialization materialization =
          materialize_llm_kv_block_table_in_place(
              paged.layout, derive_llm_kv_permutation_seed(plan.base_seed),
              static_cast<uint32_t*>(paged.block_table_mapping.get()),
              paged.layout.block_table_entries,
              Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES,
              stop_requested);
      if (!materialization.valid) {
        plan.reason_code = materialization.reason_code;
        discard_executable_templates(plan);
        return plan;
      }
      paged.table_validation = materialization.validation;
      paged.permutation = materialization.permutation;
      paged.layout_identity =
          serialize_llm_kv_layout_identity(paged.layout,
                                           paged.permutation);
      paged.execution_identity =
          plan.phase == LlmPhase::Prefill
              ? cpu_plan->prefill.has_value() ? cpu_plan->prefill->identity
                                              : std::string{}
              : paged.ownership.identity;
      if (paged.layout_identity.empty() ||
          paged.execution_identity.empty()) {
        plan.reason_code =
            LlmWorkPlanReason::BLOCK_TABLE_MATERIALIZATION_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      if (!protect_buffer_read_only(paged.block_table_mapping.get(),
                                    paged.block_table_mapping_bytes)) {
        plan.reason_code =
            LlmWorkPlanReason::BLOCK_TABLE_PROTECTION_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      paged.block_table_read_only = true;
    }

    if (!finalize_plan_identities(plan)) {
      plan.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      discard_executable_templates(plan);
      return plan;
    }
    plan.valid = true;
    if (draft.auxiliary_preflight.valid) {
      LlmAuxiliaryPreflightView finalized_view;
      if (!build_auxiliary_preflight_view(plan, finalized_view) ||
          !auxiliary_preflight_views_match(
              draft.auxiliary_preflight, finalized_view)) {
        plan.valid = false;
        plan.reason_code =
            LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
        discard_executable_templates(plan);
        return plan;
      }
    }
  } catch (const std::bad_alloc&) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    discard_executable_templates(plan);
    return plan;
  } catch (const std::length_error&) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    discard_executable_templates(plan);
    return plan;
  }

  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlanDraft draft;
  draft.candidate =
      build_llm_memory_work_plan_candidate(request, stop_requested);
  draft.valid =
      draft.candidate.reason_code == LlmWorkPlanReason::VALID;
  draft.reason_code = draft.candidate.reason_code;
  return finalize_llm_memory_work_plan(
      std::move(draft), request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes, stop_requested);
}

LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    const LlmKvStopRequested& stop_requested) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    LlmMemoryWorkPlanDraft draft;
    draft.reason_code = validation.reason_code;
    draft.candidate = invalid_config_plan(validation.reason_code);
    return draft;
  }

  LlmMemoryWorkPlanRequest request;
  request.geometry = {validation.active_weight_bytes,
                      config.layer_count,
                      config.query_head_count,
                      config.kv_head_count,
                      config.head_dimension,
                      config.kv_element_bytes,
                      config.visible_context_tokens,
                      config.batch_size,
                      config.kv_block_tokens,
                      config.phase,
                      config.kv_layout};
  request.geometry.prompt_tokens = config.prompt_tokens;
  request.geometry.attention_query_tile_tokens =
      config.attention_query_tile_tokens;
  request.backend = config.backend;
  request.requested_workers = config.requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = available_memory_bytes;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.base_seed = config.seed;
  return prepare_llm_memory_work_plan(request, stop_requested);
}

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    return invalid_config_plan(validation.reason_code);
  }

  LlmMemoryWorkPlanRequest request;
  request.geometry = {validation.active_weight_bytes,
                      config.layer_count,
                      config.query_head_count,
                      config.kv_head_count,
                      config.head_dimension,
                      config.kv_element_bytes,
                      config.visible_context_tokens,
                      config.batch_size,
                      config.kv_block_tokens,
                      config.phase,
                      config.kv_layout};
  request.geometry.prompt_tokens = config.prompt_tokens;
  request.geometry.attention_query_tile_tokens =
      config.attention_query_tile_tokens;
  request.backend = config.backend;
  request.requested_workers = config.requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = available_memory_bytes;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  request.base_seed = config.seed;
  return build_llm_memory_work_plan(request, stop_requested);
}

bool readmit_llm_memory_work_plan(
    LlmMemoryWorkPlan& plan, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes) noexcept {
  try {
    LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
    if (!plan.valid || cpu_plan == nullptr) {
      return false;
    }
    const size_t block_table_bytes =
        plan.kv_layout == LlmKvLayout::Paged
            ? plan.geometry.block_table_bytes
            : 0;
    const size_t transient_bytes =
        cpu_plan->paged.has_value()
            ? cpu_plan->paged->layout.memory.validation_bitset_bytes
            : plan.memory_budget.request.layout_transient_bytes;
    const LlmMemoryBudgetRequest request = build_llm_memory_budget_request(
        plan.geometry, cpu_plan->descriptor_bytes,
        cpu_plan->planner_storage_bytes, checksum_auxiliary_bytes,
        orchestration_auxiliary_bytes,
        plan.memory_budget.request.mapping_granularity_bytes,
        block_table_bytes, transient_bytes);
    plan.memory_budget = evaluate_llm_memory_budget(
        request, plan.memory_budget.available_memory_bytes);
    if (!plan.memory_budget.valid) {
      plan.valid = false;
      plan.reason_code = plan.memory_budget.reason_code;
      return false;
    }
    plan.reason_code = LlmWorkPlanReason::VALID;
    return true;
  } catch (...) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return false;
  }
}

LlmScenarioLimits calculate_llm_scenario_limits(
    const LlmGeometry& geometry, LlmScenario scenario) {
  LlmScenarioLimits limits;
  limits.scenario = scenario;
  if (!geometry.valid) {
    limits.reason_code = geometry.reason_code;
    return limits;
  }
  if (!valid_scenario(scenario)) {
    limits.reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
    return limits;
  }
  limits.work_unit_kind = geometry.work_unit_kind;
  limits.kv_write_kind = llm_kv_write_kind_for(geometry.phase, scenario);

  switch (scenario) {
    case LlmScenario::WeightsOnly:
      limits.weight_read_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      break;
    case LlmScenario::KvOnly:
      limits.kv_read_bytes_per_work_unit = geometry.kv_read_bytes_per_work_unit;
      limits.kv_write_bytes_per_work_unit =
          geometry.kv_write_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.kv_only_effective_model_payload_bytes_per_work_unit;
      break;
    case LlmScenario::Mixed:
      limits.weight_read_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      limits.kv_read_bytes_per_work_unit = geometry.kv_read_bytes_per_work_unit;
      limits.kv_write_bytes_per_work_unit =
          geometry.kv_write_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.mixed_effective_model_payload_bytes_per_work_unit;
      break;
  }

  limits.layout_metadata_lookup_count_per_work_unit = 0;
  limits.layout_metadata_read_bytes_per_work_unit = 0;
  if (geometry.kv_layout == LlmKvLayout::Paged &&
      scenario != LlmScenario::WeightsOnly) {
    size_t layer_sequence_count = 0;
    if (!NumericUtils::checked_multiply(
            geometry.layer_count, geometry.batch_size,
            layer_sequence_count) ||
        !NumericUtils::checked_multiply(
            layer_sequence_count,
            geometry
                .layout_metadata_lookups_per_layer_sequence_per_work_unit,
            limits.layout_metadata_lookup_count_per_work_unit) ||
        !NumericUtils::checked_multiply(
            limits.layout_metadata_lookup_count_per_work_unit,
            Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
            limits.layout_metadata_read_bytes_per_work_unit)) {
      limits.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
      return limits;
    }
  }
  if (!NumericUtils::checked_add(
          limits.effective_model_payload_bytes_per_work_unit,
          limits.layout_metadata_read_bytes_per_work_unit,
          limits.accounted_bytes_per_work_unit)) {
    limits.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
    return limits;
  }
  limits.maximum_work_units_by_guardrail =
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
      limits.accounted_bytes_per_work_unit;
  limits.effective_maximum_work_units =
      std::min(limits.maximum_work_units_by_work_unit_cap,
               limits.maximum_work_units_by_guardrail);
  if (limits.effective_maximum_work_units == 0) {
    limits.reason_code = LlmWorkPlanReason::GUARDRAIL_BELOW_ONE_WORK_UNIT;
    return limits;
  }
  limits.valid = true;
  limits.reason_code = LlmWorkPlanReason::VALID;
  return limits;
}

LlmScenarioWorkPlan build_llm_scenario_work_plan(
    const LlmMemoryWorkPlan& model_plan, LlmScenario scenario, size_t work_units,
    bool explicit_iterations) {
  LlmScenarioWorkPlan plan;
  plan.scenario = scenario;
  plan.explicit_iterations = explicit_iterations;
  if (!model_plan.valid || model_plan.plan_identity.empty()) {
    plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return plan;
  }
  plan.model_plan_identity = model_plan.plan_identity;
  if (valid_scenario(scenario)) {
    plan.scenario_seed = model_plan.scenario_seeds[scenario_index(scenario)];
  }
  const LlmScenarioLimits limits =
      calculate_llm_scenario_limits(model_plan.geometry, scenario);
  if (!limits.valid) {
    plan.reason_code = limits.reason_code;
    return plan;
  }
  plan.maximum_work_units_by_work_unit_cap = limits.maximum_work_units_by_work_unit_cap;
  plan.maximum_work_units_by_guardrail = limits.maximum_work_units_by_guardrail;
  plan.effective_maximum_work_units = limits.effective_maximum_work_units;
  plan.work_unit_kind = limits.work_unit_kind;
  plan.kv_write_kind = limits.kv_write_kind;
  if (work_units == 0) {
    plan.reason_code = LlmWorkPlanReason::WORK_UNIT_COUNT_ZERO;
    return plan;
  }
  if (work_units > limits.maximum_work_units_by_work_unit_cap) {
    plan.reason_code = LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED;
    return plan;
  }

  plan.work_units = work_units;
  plan.weight_read_bytes_per_work_unit = limits.weight_read_bytes_per_work_unit;
  plan.kv_read_bytes_per_work_unit = limits.kv_read_bytes_per_work_unit;
  plan.kv_write_bytes_per_work_unit =
      limits.kv_write_bytes_per_work_unit;
  plan.effective_model_payload_bytes_per_work_unit =
      limits.effective_model_payload_bytes_per_work_unit;
  plan.layout_metadata_lookup_count_per_work_unit =
      limits.layout_metadata_lookup_count_per_work_unit;
  plan.layout_metadata_read_bytes_per_work_unit =
      limits.layout_metadata_read_bytes_per_work_unit;
  plan.accounted_bytes_per_work_unit = limits.accounted_bytes_per_work_unit;
  if (!NumericUtils::checked_multiply(plan.weight_read_bytes_per_work_unit, work_units,
                                      plan.weight_read_bytes) ||
      !NumericUtils::checked_multiply(plan.kv_read_bytes_per_work_unit, work_units,
                                      plan.kv_read_bytes) ||
      !NumericUtils::checked_multiply(
          plan.kv_write_bytes_per_work_unit, work_units,
          plan.kv_write_bytes) ||
      !NumericUtils::checked_multiply(
          plan.effective_model_payload_bytes_per_work_unit, work_units,
          plan.effective_model_payload_bytes) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_lookup_count_per_work_unit, work_units,
          plan.layout_metadata_lookup_count) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_read_bytes_per_work_unit, work_units,
          plan.layout_metadata_read_bytes) ||
      !NumericUtils::checked_multiply(plan.accounted_bytes_per_work_unit,
                                      work_units,
                                      plan.task_accounted_bytes)) {
    plan.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
    return plan;
  }
  if (plan.task_accounted_bytes >
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK) {
    plan.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED;
    return plan;
  }

  plan.plan_identity = build_scenario_plan_identity(plan);
  plan.valid = true;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmFrozenScenarioPlans freeze_llm_scenario_work_plans(
    const LlmMemoryWorkPlan& model_plan,
    const std::array<size_t, kLlmScenarioCount>& work_units,
    bool explicit_iterations) {
  LlmFrozenScenarioPlans frozen;
  frozen.explicit_iterations = explicit_iterations;
  if (!model_plan.valid || model_plan.plan_identity.empty()) {
    frozen.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return frozen;
  }
  frozen.model_plan_identity = model_plan.plan_identity;
  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    frozen.scenarios[index] = build_llm_scenario_work_plan(
        model_plan, kScenarios[index], work_units[index], explicit_iterations);
    if (!frozen.scenarios[index].valid) {
      frozen.reason_code = frozen.scenarios[index].reason_code;
      frozen.scenarios = {};
      frozen.model_plan_identity.clear();
      return frozen;
    }
  }

  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "kind", "frozen_scenarios");
  append_identity_field(identity, "explicit",
                        frozen.explicit_iterations ? 1 : 0);
  append_identity_field(identity, "model_plan_identity_size",
                        frozen.model_plan_identity.size());
  append_identity_field(identity, "model_plan_identity",
                        frozen.model_plan_identity);
  constexpr std::array<const char*, kLlmScenarioCount> kIdentitySizeNames = {
      "weights_only_identity_size", "kv_only_identity_size",
      "mixed_identity_size"};
  constexpr std::array<const char*, kLlmScenarioCount> kIdentityNames = {
      "weights_only_identity", "kv_only_identity", "mixed_identity"};
  for (size_t index = 0; index < frozen.scenarios.size(); ++index) {
    append_identity_field(identity, kIdentitySizeNames[index],
                          frozen.scenarios[index].plan_identity.size());
    append_identity_field(identity, kIdentityNames[index],
                          frozen.scenarios[index].plan_identity);
  }
  frozen.plan_identity = std::move(identity);
  frozen.valid = true;
  frozen.reason_code = LlmWorkPlanReason::VALID;
  return frozen;
}

size_t calculate_llm_pilot_work_units(const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_minimum_pilot_count(
      limits.accounted_bytes_per_work_unit,
      Constants::LLM_CALIBRATION_MIN_PILOT_BYTES,
      limits.effective_maximum_work_units);
}

size_t calculate_llm_calibrated_work_units(double attempt_duration_seconds,
                                           size_t attempt_work_units,
                                           const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_duration_scaled_count(
      attempt_duration_seconds, attempt_work_units,
      Constants::LLM_CALIBRATION_TARGET_SECONDS, 1,
      limits.effective_maximum_work_units);
}

bool llm_duration_in_target_window(double elapsed_seconds) {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::LLM_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::LLM_CALIBRATION_MAX_SECONDS;
}

std::string_view classify_llm_duration_quality(
    double elapsed_seconds, size_t work_units,
    const LlmScenarioLimits& limits) noexcept {
  if (!limits.valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0 || work_units == 0 ||
      work_units > limits.effective_maximum_work_units) {
    return "invalid-duration";
  }
  if (llm_duration_in_target_window(elapsed_seconds)) {
    return "within-target-window";
  }
  if (work_units == 1 &&
      elapsed_seconds > Constants::LLM_CALIBRATION_MAX_SECONDS) {
    return "above-target-single-work-unit";
  }
  if (work_units == limits.effective_maximum_work_units &&
      elapsed_seconds < Constants::LLM_CALIBRATION_MIN_SECONDS) {
    return "guardrail-limited-below-target";
  }
  return elapsed_seconds < Constants::LLM_CALIBRATION_MIN_SECONDS
             ? "below-target-window"
             : "above-target-window";
}

std::array<LlmScenario, kLlmScenarioCount> build_llm_scenario_order(
    size_t loop_index) {
  constexpr std::array<LlmScenario, kLlmScenarioCount> kBaseOrder = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  std::array<LlmScenario, kLlmScenarioCount> order{};
  const size_t rotation = loop_index % kLlmScenarioCount;
  for (size_t position = 0; position < kLlmScenarioCount; ++position) {
    order[position] =
        kBaseOrder[(rotation + position) % kLlmScenarioCount];
  }
  return order;
}
