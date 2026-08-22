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
 * @file llm_executor.cpp
 * @brief CPU LLM resource preparation and synchronized scenario execution
 */

#include "llm_memory/llm_executor.h"

#include <mach/mach.h>
#include <pthread/qos.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>

#include "core/timing/timer.h"
#include "utils/numeric_utils.h"

namespace {

constexpr uint64_t kBufferPatternMultiplier = 0x9E3779B97F4A7C15ULL;

constexpr uint64_t kAppendStepMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kAppendLayerMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kAppendBatchMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kAppendWordMultiplier = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kAppendKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kAppendVDomain = 0x5656565656565656ULL;

constexpr uint64_t kChecksumInitialA = 0x243F6A8885A308D3ULL;
constexpr uint64_t kChecksumInitialB = 0x13198A2E03707344ULL;
constexpr uint64_t kChecksumWeightDomain = 0x5745494748545F31ULL;
constexpr uint64_t kChecksumKDomain = 0x4B5F524541445F31ULL;
constexpr uint64_t kChecksumVDomain = 0x565F524541445F31ULL;

constexpr uint64_t kRunInitialA = 0x6A09E667F3BCC909ULL;
constexpr uint64_t kRunInitialB = 0xBB67AE8584CAA73BULL;

uint64_t rotate_left(uint64_t value, unsigned int shift) noexcept {
  return (value << shift) | (value >> (64U - shift));
}

uint64_t checksum_domain(LlmChecksumComponent component) noexcept {
  switch (component) {
    case LlmChecksumComponent::Weight:
      return kChecksumWeightDomain;
    case LlmChecksumComponent::K:
      return kChecksumKDomain;
    case LlmChecksumComponent::V:
      return kChecksumVDomain;
  }
  return 0;
}

uint64_t append_domain(LlmChecksumComponent component) noexcept {
  switch (component) {
    case LlmChecksumComponent::K:
      return kAppendKDomain;
    case LlmChecksumComponent::V:
      return kAppendVDomain;
    case LlmChecksumComponent::Weight:
      return 0;
  }
  return 0;
}

bool checked_range_end(const LlmByteRange& range, size_t limit, size_t& end) noexcept {
  if (range.span_bytes == 0) {
    return range.offset_bytes == 0;
  }
  return NumericUtils::checked_add(range.offset_bytes, range.span_bytes, end) && end <= limit;
}

bool equal_ranges(const LlmByteRange& lhs, const LlmByteRange& rhs) noexcept {
  return lhs.offset_bytes == rhs.offset_bytes && lhs.span_bytes == rhs.span_bytes;
}

LlmByteRange intersect_ranges(const LlmByteRange& lhs, const LlmByteRange& rhs) noexcept {
  size_t lhs_end = 0;
  size_t rhs_end = 0;
  if (lhs.span_bytes == 0 || rhs.span_bytes == 0 ||
      !NumericUtils::checked_add(lhs.offset_bytes, lhs.span_bytes, lhs_end) ||
      !NumericUtils::checked_add(rhs.offset_bytes, rhs.span_bytes, rhs_end)) {
    return {};
  }
  const size_t start = std::max(lhs.offset_bytes, rhs.offset_bytes);
  const size_t end = std::min(lhs_end, rhs_end);
  return end > start ? LlmByteRange{start, end - start} : LlmByteRange{};
}

/**
 * Validate the complete pointer-free layout before any mapping is created.
 *
 * This deliberately rechecks exact worker unions rather than trusting `valid`:
 * later descriptor materialization must never turn a corrupted offset into an
 * out-of-bounds pointer. The validation is allocation-free and does not mutate
 * the finalized work plan.
 */
bool validate_work_plan_layout(const LlmMemoryWorkPlan& plan) noexcept {
  size_t weight_and_k_bytes = 0;
  size_t expected_total_data_bytes = 0;
  if (!plan.valid || !plan.geometry.valid || !plan.memory_budget.valid || plan.effective_workers == 0 ||
      plan.geometry.layer_count == 0 || plan.geometry.batch_size == 0 || plan.geometry.k_mapping_bytes == 0 ||
      plan.geometry.v_mapping_bytes == 0 || plan.geometry.k_mapping_bytes != plan.geometry.v_mapping_bytes ||
      !NumericUtils::checked_add(plan.geometry.active_weight_bytes_per_step, plan.geometry.k_mapping_bytes,
                                 weight_and_k_bytes) ||
      !NumericUtils::checked_add(weight_and_k_bytes, plan.geometry.v_mapping_bytes, expected_total_data_bytes) ||
      expected_total_data_bytes != plan.geometry.total_data_mapping_bytes ||
      plan.workers.size() != plan.effective_workers || plan.weight_layers.size() != plan.geometry.layer_count ||
      plan.layer_descriptors_per_worker != plan.geometry.layer_count) {
    return false;
  }

  size_t expected_sequences_per_worker = 0;
  size_t expected_total_layers = 0;
  size_t expected_total_sequences = 0;
  size_t layer_descriptor_bytes = 0;
  size_t sequence_descriptor_bytes = 0;
  size_t expected_descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(plan.geometry.layer_count, plan.geometry.batch_size,
                                      expected_sequences_per_worker) ||
      expected_sequences_per_worker != plan.sequence_descriptors_per_worker ||
      !NumericUtils::checked_multiply(plan.effective_workers, plan.layer_descriptors_per_worker,
                                      expected_total_layers) ||
      !NumericUtils::checked_multiply(plan.effective_workers, plan.sequence_descriptors_per_worker,
                                      expected_total_sequences) ||
      expected_total_layers != plan.total_layer_descriptors ||
      expected_total_sequences != plan.total_sequence_descriptors ||
      !NumericUtils::checked_multiply(expected_total_layers, sizeof(LlmLayerDescriptor), layer_descriptor_bytes) ||
      !NumericUtils::checked_multiply(expected_total_sequences, sizeof(LlmKvSequenceDescriptor),
                                      sequence_descriptor_bytes) ||
      !NumericUtils::checked_add(layer_descriptor_bytes, sequence_descriptor_bytes, expected_descriptor_bytes) ||
      expected_descriptor_bytes != plan.descriptor_bytes) {
    return false;
  }

  size_t weight_mapping_cursor = 0;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    size_t layer_end = 0;
    if (layer_range.span_bytes == 0 || layer_range.offset_bytes != weight_mapping_cursor ||
        !checked_range_end(layer_range, plan.geometry.active_weight_bytes_per_step, layer_end)) {
      return false;
    }

    size_t worker_cursor = layer_range.offset_bytes;
    const size_t expected_first_sequence = layer * plan.geometry.batch_size;
    for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = plan.workers[worker_index];
      if (worker.worker_index != worker_index || worker.layers.size() != plan.layer_descriptors_per_worker ||
          worker.sequences.size() != plan.sequence_descriptors_per_worker) {
        return false;
      }
      const LlmLayerRangeTemplate& worker_layer = worker.layers[layer];
      if (worker_layer.first_sequence_index != expected_first_sequence ||
          worker_layer.sequence_count != plan.geometry.batch_size || worker_layer.layer_index != layer) {
        return false;
      }
      size_t worker_end = 0;
      if (!checked_range_end(worker_layer.weight, plan.geometry.active_weight_bytes_per_step, worker_end)) {
        return false;
      }
      if (worker_layer.weight.span_bytes != 0) {
        if (worker_layer.weight.offset_bytes != worker_cursor || worker_end > layer_end) {
          return false;
        }
        worker_cursor = worker_end;
      }
    }
    if (worker_cursor != layer_end) {
      return false;
    }
    weight_mapping_cursor = layer_end;
  }
  if (weight_mapping_cursor != plan.geometry.active_weight_bytes_per_step) {
    return false;
  }

  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
      const size_t sequence_index = layer * plan.geometry.batch_size + batch;
      size_t visible_offset = 0;
      size_t visible_end = 0;
      size_t append_token_offset = 0;
      size_t append_offset = 0;
      if (!NumericUtils::checked_multiply(sequence_index, plan.geometry.k_or_v_sequence_visible_bytes,
                                          visible_offset) ||
          !NumericUtils::checked_add(visible_offset, plan.geometry.k_or_v_sequence_visible_bytes, visible_end) ||
          visible_end > plan.geometry.k_mapping_bytes ||
          !NumericUtils::checked_multiply(plan.geometry.visible_context_tokens - 1,
                                          plan.geometry.k_or_v_record_bytes_per_layer, append_token_offset) ||
          !NumericUtils::checked_add(visible_offset, append_token_offset, append_offset)) {
        return false;
      }
      const LlmByteRange append_record{append_offset, plan.geometry.k_or_v_record_bytes_per_layer};

      size_t worker_cursor = visible_offset;
      for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
        const LlmKvSequenceRangeTemplate& sequence = plan.workers[worker_index].sequences[sequence_index];
        if (sequence.layer_index != layer || sequence.batch_sequence_index != batch ||
            !equal_ranges(sequence.k_visible, sequence.v_visible) ||
            !equal_ranges(sequence.k_append, sequence.v_append)) {
          return false;
        }

        size_t visible_worker_end = 0;
        size_t v_visible_worker_end = 0;
        if (!checked_range_end(sequence.k_visible, plan.geometry.k_mapping_bytes, visible_worker_end) ||
            !checked_range_end(sequence.v_visible, plan.geometry.v_mapping_bytes, v_visible_worker_end) ||
            visible_worker_end != v_visible_worker_end) {
          return false;
        }
        if (sequence.k_visible.span_bytes != 0) {
          if (sequence.k_visible.offset_bytes != worker_cursor || visible_worker_end > visible_end) {
            return false;
          }
          worker_cursor = visible_worker_end;
        }

        const LlmByteRange expected_append = intersect_ranges(sequence.k_visible, append_record);
        if (!equal_ranges(sequence.k_append, expected_append)) {
          return false;
        }
        if (expected_append.span_bytes == 0) {
          if (sequence.append_record_byte_offset != 0) {
            return false;
          }
        } else if (sequence.append_record_byte_offset != expected_append.offset_bytes - append_offset) {
          return false;
        }
      }
      if (worker_cursor != visible_end) {
        return false;
      }
    }
  }
  return true;
}

bool checked_add_to(size_t value, size_t& total) noexcept {
  size_t updated = 0;
  if (!NumericUtils::checked_add(total, value, updated)) {
    return false;
  }
  total = updated;
  return true;
}

bool buffer_output_is_empty(const LlmBufferSet& output) noexcept {
  return output.weight == nullptr && output.k == nullptr && output.v == nullptr;
}

bool resource_output_is_empty(const LlmExecutionResources& output) noexcept {
  return !output.valid && output.model_plan_identity.empty() && buffer_output_is_empty(output.buffers) &&
         output.layer_descriptors == nullptr && output.sequence_descriptors == nullptr &&
         output.weight_references == nullptr && output.k_references == nullptr && output.v_references == nullptr &&
         output.worker_count == 0 && output.layer_descriptors_per_worker == 0 &&
         output.sequence_descriptors_per_worker == 0 && output.total_layer_descriptors == 0 &&
         output.total_sequence_descriptors == 0;
}

bool calculate_budget_with_executor_auxiliary(const LlmMemoryWorkPlan& plan,
                                              const LlmExecutorAuxiliaryEstimate& auxiliary,
                                              LlmMemoryBudget& budget) noexcept {
  if (!auxiliary.valid) {
    return false;
  }
  // Admission is evidence, not a permission to silently expand the plan after
  // it was frozen. A caller must rebuild the work plan with at least these
  // executor-owned bytes before any mapping can be attempted.
  if (plan.memory_budget.request.checksum_auxiliary_bytes < auxiliary.checksum_auxiliary_bytes ||
      plan.memory_budget.request.orchestration_auxiliary_bytes < auxiliary.orchestration_auxiliary_bytes) {
    budget = plan.memory_budget;
    budget.valid = false;
    budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return false;
  }
  const size_t checksum_bytes = plan.memory_budget.request.checksum_auxiliary_bytes;
  const size_t orchestration_bytes = plan.memory_budget.request.orchestration_auxiliary_bytes;
  const LlmMemoryBudgetRequest request =
      build_llm_memory_budget_request(plan.geometry, plan.descriptor_bytes, plan.planner_storage_bytes, checksum_bytes,
                                      orchestration_bytes, plan.memory_budget.request.mapping_granularity_bytes);
  budget = evaluate_llm_memory_budget(request, plan.memory_budget.available_memory_bytes);
  return budget.valid;
}

uint64_t low_byte_mask(size_t byte_count) noexcept {
  return byte_count >= sizeof(uint64_t) ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << (byte_count * 8U)) - 1;
}

/** Generate at most eight pattern bytes beginning at an arbitrary byte. */
uint64_t pattern_byte_word(uint64_t seed, size_t absolute_byte_offset, size_t byte_count) noexcept {
  const uint64_t word_index = static_cast<uint64_t>(absolute_byte_offset / sizeof(uint64_t));
  const size_t byte_in_word = absolute_byte_offset % sizeof(uint64_t);
  uint64_t value = llm_buffer_pattern_word(seed, word_index) >> (byte_in_word * 8U);
  const size_t first_word_bytes = sizeof(uint64_t) - byte_in_word;
  if (byte_count > first_word_bytes) {
    value |= llm_buffer_pattern_word(seed, word_index + 1) << (first_word_bytes * 8U);
  }
  return value & low_byte_mask(byte_count);
}

/**
 * Initialize one exact finalized span and accumulate its read reference.
 *
 * Local checksum-word parity starts at zero even when the mapping offset is
 * unaligned. `memcpy` keeps every 1-7 byte tail exact and avoids undefined
 * unaligned integer stores.
 */
bool initialize_span(uint8_t* mapping, size_t mapping_bytes, uint64_t seed, size_t offset, size_t span_bytes,
                     LlmStaticSpanReference& reference) noexcept {
  reference = {};
  if (span_bytes == 0) {
    return offset == 0;
  }
  size_t end = 0;
  if (mapping == nullptr || !NumericUtils::checked_add(offset, span_bytes, end) || end > mapping_bytes) {
    return false;
  }

  reference.span_bytes = static_cast<uint64_t>(span_bytes);
  size_t local_offset = 0;
  size_t word_index = 0;
  while (local_offset < span_bytes) {
    const size_t word_bytes = std::min(sizeof(uint64_t), span_bytes - local_offset);
    const uint64_t word = pattern_byte_word(seed, offset + local_offset, word_bytes);
    std::memcpy(mapping + offset + local_offset, &word, word_bytes);
    if ((word_index & 1U) == 0) {
      reference.span_even += word;
    } else {
      reference.span_odd += word;
    }
    local_offset += word_bytes;
    ++word_index;
  }
  return true;
}

bool add_initialized_span(const LlmStaticSpanReference& reference, size_t& initialized_bytes,
                          size_t& non_empty_spans) noexcept {
  if (reference.span_bytes == 0) {
    return true;
  }
  if (reference.span_bytes > std::numeric_limits<size_t>::max() ||
      !checked_add_to(static_cast<size_t>(reference.span_bytes), initialized_bytes) ||
      non_empty_spans == std::numeric_limits<size_t>::max()) {
    return false;
  }
  ++non_empty_spans;
  return true;
}

bool materialize_descriptors(const LlmMemoryWorkPlan& plan, LlmExecutionResources& resources) noexcept {
  uint8_t* const weight = static_cast<uint8_t*>(resources.buffers.weight.get());
  uint8_t* const k = static_cast<uint8_t*>(resources.buffers.k.get());
  uint8_t* const v = static_cast<uint8_t*>(resources.buffers.v.get());
  if (weight == nullptr || k == nullptr || v == nullptr) {
    return false;
  }

  for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = plan.workers[worker_index];
    const size_t layer_base = worker_index * plan.layer_descriptors_per_worker;
    const size_t sequence_base = worker_index * plan.sequence_descriptors_per_worker;
    for (size_t layer = 0; layer < plan.layer_descriptors_per_worker; ++layer) {
      const LlmLayerRangeTemplate& source = worker.layers[layer];
      LlmLayerDescriptor& destination = resources.layer_descriptors[layer_base + layer];
      destination.weight_ptr = source.weight.span_bytes == 0 ? nullptr : weight + source.weight.offset_bytes;
      destination.weight_bytes = source.weight.span_bytes;
      destination.first_sequence_index = source.first_sequence_index;
      destination.sequence_count = source.sequence_count;
      destination.layer_index = source.layer_index;
      destination.reserved_zero = 0;
    }
    for (size_t sequence_index = 0; sequence_index < plan.sequence_descriptors_per_worker; ++sequence_index) {
      const LlmKvSequenceRangeTemplate& source = worker.sequences[sequence_index];
      LlmKvSequenceDescriptor& destination = resources.sequence_descriptors[sequence_base + sequence_index];
      destination.k_visible_ptr = source.k_visible.span_bytes == 0 ? nullptr : k + source.k_visible.offset_bytes;
      destination.k_visible_bytes = source.k_visible.span_bytes;
      destination.v_visible_ptr = source.v_visible.span_bytes == 0 ? nullptr : v + source.v_visible.offset_bytes;
      destination.v_visible_bytes = source.v_visible.span_bytes;
      destination.k_append_ptr = source.k_append.span_bytes == 0 ? nullptr : k + source.k_append.offset_bytes;
      destination.k_append_bytes = source.k_append.span_bytes;
      destination.v_append_ptr = source.v_append.span_bytes == 0 ? nullptr : v + source.v_append.offset_bytes;
      destination.v_append_bytes = source.v_append.span_bytes;
      destination.batch_sequence_index = source.batch_sequence_index;
      destination.append_record_byte_offset = source.k_append.span_bytes == 0 ? 0 : source.append_record_byte_offset;
    }
  }
  return true;
}

bool initialize_resources(const LlmMemoryWorkPlan& plan, LlmExecutionResources& resources,
                          LlmInitializationEvidence& evidence) noexcept {
  uint8_t* const weight = static_cast<uint8_t*>(resources.buffers.weight.get());
  uint8_t* const k = static_cast<uint8_t*>(resources.buffers.k.get());
  uint8_t* const v = static_cast<uint8_t*>(resources.buffers.v.get());

  for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = plan.workers[worker_index];
    const size_t layer_base = worker_index * plan.layer_descriptors_per_worker;
    const size_t sequence_base = worker_index * plan.sequence_descriptors_per_worker;
    for (size_t layer = 0; layer < plan.layer_descriptors_per_worker; ++layer) {
      LlmStaticSpanReference& reference = resources.weight_references[layer_base + layer];
      const LlmByteRange& range = worker.layers[layer].weight;
      if (!initialize_span(weight, plan.geometry.active_weight_bytes_per_step, plan.weight_buffer_seed,
                           range.offset_bytes, range.span_bytes, reference) ||
          !add_initialized_span(reference, evidence.weight_bytes, evidence.non_empty_weight_spans)) {
        return false;
      }
    }
    for (size_t sequence_index = 0; sequence_index < plan.sequence_descriptors_per_worker; ++sequence_index) {
      const LlmKvSequenceRangeTemplate& sequence = worker.sequences[sequence_index];
      LlmStaticSpanReference& k_reference = resources.k_references[sequence_base + sequence_index];
      LlmStaticSpanReference& v_reference = resources.v_references[sequence_base + sequence_index];
      if (!initialize_span(k, plan.geometry.k_mapping_bytes, plan.k_buffer_seed, sequence.k_visible.offset_bytes,
                           sequence.k_visible.span_bytes, k_reference) ||
          !add_initialized_span(k_reference, evidence.k_bytes, evidence.non_empty_k_spans) ||
          !initialize_span(v, plan.geometry.v_mapping_bytes, plan.v_buffer_seed, sequence.v_visible.offset_bytes,
                           sequence.v_visible.span_bytes, v_reference) ||
          !add_initialized_span(v_reference, evidence.v_bytes, evidence.non_empty_v_spans)) {
        return false;
      }
    }
  }

  if (evidence.weight_bytes != plan.geometry.active_weight_bytes_per_step ||
      evidence.k_bytes != plan.geometry.k_mapping_bytes || evidence.v_bytes != plan.geometry.v_mapping_bytes ||
      !NumericUtils::checked_add(evidence.weight_bytes, evidence.k_bytes, evidence.total_bytes) ||
      !checked_add_to(evidence.v_bytes, evidence.total_bytes) ||
      evidence.total_bytes != plan.geometry.total_data_mapping_bytes) {
    return false;
  }
  evidence.complete = true;
  return true;
}

bool materialized_resources_match_plan(const LlmMemoryWorkPlan& plan, const LlmExecutionResources& resources) noexcept {
  if (!validate_work_plan_layout(plan) || !resources.valid || resources.model_plan_identity != plan.plan_identity ||
      !resources.buffers.complete() || resources.layer_descriptors == nullptr ||
      resources.sequence_descriptors == nullptr || resources.weight_references == nullptr ||
      resources.k_references == nullptr || resources.v_references == nullptr ||
      resources.worker_count != plan.effective_workers ||
      resources.layer_descriptors_per_worker != plan.layer_descriptors_per_worker ||
      resources.sequence_descriptors_per_worker != plan.sequence_descriptors_per_worker ||
      resources.total_layer_descriptors != plan.total_layer_descriptors ||
      resources.total_sequence_descriptors != plan.total_sequence_descriptors || !resources.initialization.complete ||
      resources.initialization.weight_bytes != plan.geometry.active_weight_bytes_per_step ||
      resources.initialization.k_bytes != plan.geometry.k_mapping_bytes ||
      resources.initialization.v_bytes != plan.geometry.v_mapping_bytes) {
    return false;
  }

  const uint8_t* const weight = static_cast<const uint8_t*>(resources.buffers.weight.get());
  const uint8_t* const k = static_cast<const uint8_t*>(resources.buffers.k.get());
  const uint8_t* const v = static_cast<const uint8_t*>(resources.buffers.v.get());
  for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = plan.workers[worker_index];
    const size_t layer_base = worker_index * plan.layer_descriptors_per_worker;
    const size_t sequence_base = worker_index * plan.sequence_descriptors_per_worker;
    for (size_t layer = 0; layer < plan.layer_descriptors_per_worker; ++layer) {
      const LlmLayerRangeTemplate& source = worker.layers[layer];
      const LlmLayerDescriptor& descriptor = resources.layer_descriptors[layer_base + layer];
      const LlmStaticSpanReference& reference = resources.weight_references[layer_base + layer];
      const uint8_t* expected_pointer = source.weight.span_bytes == 0 ? nullptr : weight + source.weight.offset_bytes;
      if (descriptor.weight_ptr != expected_pointer || descriptor.weight_bytes != source.weight.span_bytes ||
          descriptor.first_sequence_index != source.first_sequence_index ||
          descriptor.sequence_count != source.sequence_count || descriptor.layer_index != source.layer_index ||
          descriptor.reserved_zero != 0 || reference.span_bytes != source.weight.span_bytes) {
        return false;
      }
    }
    for (size_t sequence_index = 0; sequence_index < plan.sequence_descriptors_per_worker; ++sequence_index) {
      const LlmKvSequenceRangeTemplate& source = worker.sequences[sequence_index];
      const LlmKvSequenceDescriptor& descriptor = resources.sequence_descriptors[sequence_base + sequence_index];
      const uint8_t* expected_k_visible =
          source.k_visible.span_bytes == 0 ? nullptr : k + source.k_visible.offset_bytes;
      const uint8_t* expected_v_visible =
          source.v_visible.span_bytes == 0 ? nullptr : v + source.v_visible.offset_bytes;
      uint8_t* expected_k_append =
          source.k_append.span_bytes == 0 ? nullptr : const_cast<uint8_t*>(k) + source.k_append.offset_bytes;
      uint8_t* expected_v_append =
          source.v_append.span_bytes == 0 ? nullptr : const_cast<uint8_t*>(v) + source.v_append.offset_bytes;
      const uint64_t expected_record_offset = source.k_append.span_bytes == 0 ? 0 : source.append_record_byte_offset;
      if (descriptor.k_visible_ptr != expected_k_visible || descriptor.k_visible_bytes != source.k_visible.span_bytes ||
          descriptor.v_visible_ptr != expected_v_visible || descriptor.v_visible_bytes != source.v_visible.span_bytes ||
          descriptor.k_append_ptr != expected_k_append || descriptor.k_append_bytes != source.k_append.span_bytes ||
          descriptor.v_append_ptr != expected_v_append || descriptor.v_append_bytes != source.v_append.span_bytes ||
          descriptor.batch_sequence_index != source.batch_sequence_index ||
          descriptor.append_record_byte_offset != expected_record_offset ||
          resources.k_references[sequence_base + sequence_index].span_bytes != source.k_visible.span_bytes ||
          resources.v_references[sequence_base + sequence_index].span_bytes != source.v_visible.span_bytes) {
        return false;
      }
    }
  }
  return true;
}

bool scenario_plan_matches_model(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan) {
  if (!scenario_plan.valid || scenario_plan.steps == 0 || llm_scenario_flags(scenario_plan.scenario) == 0 ||
      scenario_plan.model_plan_identity != model_plan.plan_identity) {
    return false;
  }
  const size_t scenario_index = static_cast<size_t>(scenario_plan.scenario);
  if (scenario_index >= kLlmScenarioCount || scenario_plan.scenario_seed != model_plan.scenario_seeds[scenario_index]) {
    return false;
  }
  const LlmScenarioWorkPlan rebuilt = build_llm_scenario_work_plan(
      model_plan, scenario_plan.scenario, scenario_plan.steps, scenario_plan.explicit_iterations);
  return rebuilt.valid && rebuilt.plan_identity == scenario_plan.plan_identity &&
         rebuilt.weight_read_bytes == scenario_plan.weight_read_bytes &&
         rebuilt.kv_read_bytes == scenario_plan.kv_read_bytes &&
         rebuilt.kv_append_write_bytes == scenario_plan.kv_append_write_bytes &&
         rebuilt.effective_payload_bytes == scenario_plan.effective_payload_bytes;
}

uint8_t append_byte(uint64_t scenario_seed, uint64_t task_local_step, uint64_t layer_index,
                    uint64_t batch_sequence_index, size_t record_byte_offset, LlmChecksumComponent component) noexcept {
  const uint64_t word = llm_append_word(scenario_seed, task_local_step, layer_index, batch_sequence_index,
                                        static_cast<uint64_t>(record_byte_offset / sizeof(uint64_t)), component);
  return static_cast<uint8_t>(word >> ((record_byte_offset % sizeof(uint64_t)) * 8U));
}

/**
 * Replace an initialized span's append suffix in sum space.
 *
 * Only affected local checksum words are regenerated. Prefix bytes in a
 * mid-word intersection come from the versioned buffer formula, and suffix
 * bytes come from canonical append-record offsets. This handles worker splits
 * whose read-word and append-word alignments differ without rereading memory.
 */
bool append_adjusted_reference(const LlmStaticSpanReference& static_reference,
                               const LlmKvSequenceRangeTemplate& sequence, uint64_t buffer_seed, uint64_t scenario_seed,
                               uint64_t task_local_step, LlmChecksumComponent component,
                               LlmStaticSpanReference& adjusted) noexcept {
  adjusted = static_reference;
  const LlmByteRange& visible = component == LlmChecksumComponent::K ? sequence.k_visible : sequence.v_visible;
  const LlmByteRange& append = component == LlmChecksumComponent::K ? sequence.k_append : sequence.v_append;
  if (static_reference.span_bytes != visible.span_bytes) {
    return false;
  }
  if (append.span_bytes == 0) {
    return sequence.append_record_byte_offset == 0;
  }

  size_t visible_end = 0;
  size_t append_end = 0;
  if (visible.span_bytes == 0 || append.offset_bytes < visible.offset_bytes ||
      !NumericUtils::checked_add(visible.offset_bytes, visible.span_bytes, visible_end) ||
      !NumericUtils::checked_add(append.offset_bytes, append.span_bytes, append_end) || append_end != visible_end) {
    return false;
  }
  const size_t append_local_start = append.offset_bytes - visible.offset_bytes;
  const size_t first_affected_word = append_local_start / sizeof(uint64_t);
  const size_t total_words = (visible.span_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  for (size_t word_index = first_affected_word; word_index < total_words; ++word_index) {
    const size_t local_word_start = word_index * sizeof(uint64_t);
    const size_t word_bytes = std::min(sizeof(uint64_t), visible.span_bytes - local_word_start);
    const uint64_t initialized_word =
        pattern_byte_word(buffer_seed, visible.offset_bytes + local_word_start, word_bytes);
    uint64_t appended_word = 0;
    for (size_t byte_index = 0; byte_index < word_bytes; ++byte_index) {
      const size_t local_byte = local_word_start + byte_index;
      uint8_t value = 0;
      if (local_byte < append_local_start) {
        value = static_cast<uint8_t>(initialized_word >> (byte_index * 8U));
      } else {
        const size_t record_byte = sequence.append_record_byte_offset + local_byte - append_local_start;
        value = append_byte(scenario_seed, task_local_step, sequence.layer_index, sequence.batch_sequence_index,
                            record_byte, component);
      }
      appended_word |= static_cast<uint64_t>(value) << (byte_index * 8U);
    }
    if ((word_index & 1U) == 0) {
      adjusted.span_even = adjusted.span_even - initialized_word + appended_word;
    } else {
      adjusted.span_odd = adjusted.span_odd - initialized_word + appended_word;
    }
  }
  return true;
}

bool absorb_reference(LlmReadChecksumComponent& checksum, const LlmStaticSpanReference& reference) noexcept {
  if (reference.span_bytes == 0) {
    return true;
  }
  if (checksum.exact_bytes_read > std::numeric_limits<uint64_t>::max() - reference.span_bytes ||
      checksum.span_count == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  const uint64_t ordinal = checksum.span_count;
  checksum.state_a = rotate_left(checksum.state_a + reference.span_even + kAppendStepMultiplier * (ordinal + 1), 17);
  checksum.state_b = rotate_left(
      checksum.state_b + reference.span_odd + reference.span_bytes + kAppendWordMultiplier * (ordinal + 1), 29);
  checksum.exact_bytes_read += reference.span_bytes;
  ++checksum.span_count;
  return true;
}

bool add_checksum_bytes(uint64_t bytes, uint64_t& total) noexcept {
  if (total > std::numeric_limits<uint64_t>::max() - bytes) {
    return false;
  }
  total += bytes;
  return true;
}

bool equal_component(const LlmReadChecksumComponent& lhs, const LlmReadChecksumComponent& rhs) noexcept {
  return lhs.state_a == rhs.state_a && lhs.state_b == rhs.state_b && lhs.exact_bytes_read == rhs.exact_bytes_read &&
         lhs.span_count == rhs.span_count;
}

bool equal_worker_checksum(const LlmWorkerChecksum& lhs, const LlmWorkerChecksum& rhs) noexcept {
  return equal_component(lhs.weight, rhs.weight) && equal_component(lhs.k, rhs.k) && equal_component(lhs.v, rhs.v);
}

bool production_kernel_invoke(void*, const LlmKernelInvocation& invocation) noexcept {
  if (invocation.output == nullptr) {
    return false;
  }
  llm_decode_memory_asm(invocation.layers, invocation.sequences, invocation.layer_count, invocation.step_count,
                        invocation.scenario_flags, invocation.scenario_seed, invocation.output);
  return true;
}

void observe_event(const LlmExecutorTestControl* control, LlmExecutorEvent event, size_t worker_index) noexcept {
  if (control != nullptr && control->observe_event != nullptr) {
    control->observe_event(control->event_context, event, worker_index);
  }
}

int set_worker_qos(const LlmExecutorTestControl* control, size_t worker_index) noexcept {
  if (control != nullptr && control->set_worker_qos != nullptr) {
    return control->set_worker_qos(control->qos_context, worker_index);
  }
  return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

void join_threads(std::vector<std::thread>& threads) noexcept {
  for (std::thread& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

}  // namespace

bool LlmBufferSet::complete() const noexcept { return weight != nullptr && k != nullptr && v != nullptr; }

const LlmLayerDescriptor* LlmExecutionResources::worker_layers(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || layer_descriptors == nullptr) {
    return nullptr;
  }
  return layer_descriptors.get() + worker_index * layer_descriptors_per_worker;
}

const LlmKvSequenceDescriptor* LlmExecutionResources::worker_sequences(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || sequence_descriptors == nullptr) {
    return nullptr;
  }
  return sequence_descriptors.get() + worker_index * sequence_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_weight_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || weight_references == nullptr) {
    return nullptr;
  }
  return weight_references.get() + worker_index * layer_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_k_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || k_references == nullptr) {
    return nullptr;
  }
  return k_references.get() + worker_index * sequence_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_v_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || v_references == nullptr) {
    return nullptr;
  }
  return v_references.get() + worker_index * sequence_descriptors_per_worker;
}

uint64_t llm_scenario_flags(LlmScenario scenario) noexcept {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return kLlmScenarioFlagWeight;
    case LlmScenario::KvOnly:
      return kLlmScenarioFlagKv;
    case LlmScenario::Mixed:
      return kLlmScenarioFlagMixed;
  }
  return 0;
}

uint64_t llm_buffer_pattern_word(uint64_t buffer_domain_seed, uint64_t absolute_mapping_word_index) noexcept {
  return buffer_domain_seed + kBufferPatternMultiplier * (absolute_mapping_word_index + 1);
}

uint64_t llm_append_word(uint64_t scenario_seed, uint64_t task_local_step, uint64_t layer_index,
                         uint64_t batch_sequence_index, uint64_t record_word_index,
                         LlmChecksumComponent component) noexcept {
  return scenario_seed + kAppendStepMultiplier * (task_local_step + 1) + kAppendLayerMultiplier * (layer_index + 1) +
         kAppendBatchMultiplier * (batch_sequence_index + 1) + kAppendWordMultiplier * (record_word_index + 1) +
         append_domain(component);
}

LlmReadChecksumComponent initial_llm_read_checksum(LlmChecksumComponent component) noexcept {
  const uint64_t domain = checksum_domain(component);
  return {kChecksumInitialA ^ domain, kChecksumInitialB + domain, 0, 0};
}

LlmRunChecksum fold_llm_worker_checksums(const LlmWorkerChecksum* workers, size_t worker_count) noexcept {
  LlmRunChecksum result{kRunInitialA, kRunInitialB};
  if (workers == nullptr) {
    return result;
  }
  uint64_t tuple_ordinal = 0;
  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    const LlmReadChecksumComponent* const components[] = {&workers[worker_index].weight, &workers[worker_index].k,
                                                          &workers[worker_index].v};
    for (const LlmReadChecksumComponent* component : components) {
      result.state_a =
          rotate_left(result.state_a + component->state_a + kAppendStepMultiplier * (tuple_ordinal + 1), 23);
      result.state_b =
          rotate_left(result.state_b + component->state_b + component->exact_bytes_read +
                          kAppendWordMultiplier * component->span_count + kAppendBatchMultiplier * (tuple_ordinal + 1),
                      41);
      ++tuple_ordinal;
    }
  }
  return result;
}

LlmExecutorAuxiliaryEstimate calculate_llm_executor_auxiliary_estimate(const LlmMemoryWorkPlan& plan) noexcept {
  LlmExecutorAuxiliaryEstimate estimate;
  if (!validate_work_plan_layout(plan)) {
    estimate.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
    return estimate;
  }

  size_t sequence_reference_count = 0;
  size_t reference_count = 0;
  if (!NumericUtils::checked_multiply(plan.total_sequence_descriptors, 2, sequence_reference_count) ||
      !NumericUtils::checked_add(plan.total_layer_descriptors, sequence_reference_count, reference_count) ||
      !NumericUtils::checked_multiply(reference_count, sizeof(LlmStaticSpanReference),
                                      estimate.static_reference_bytes) ||
      !NumericUtils::checked_multiply(plan.effective_workers, sizeof(LlmWorkerChecksum),
                                      estimate.expected_checksum_bytes)) {
    return estimate;
  }
  estimate.actual_checksum_bytes = estimate.expected_checksum_bytes;
  if (!NumericUtils::checked_multiply(2, sizeof(LlmRunChecksum), estimate.run_checksum_bytes) ||
      !NumericUtils::checked_multiply(plan.effective_workers, sizeof(uint8_t), estimate.worker_status_bytes) ||
      !NumericUtils::checked_multiply(plan.effective_workers, sizeof(std::thread), estimate.thread_handle_bytes)) {
    return estimate;
  }

  if (!checked_add_to(estimate.static_reference_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.expected_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.actual_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.run_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.worker_status_bytes, estimate.orchestration_auxiliary_bytes) ||
      !checked_add_to(estimate.thread_handle_bytes, estimate.orchestration_auxiliary_bytes) ||
      !NumericUtils::checked_add(estimate.checksum_auxiliary_bytes, estimate.orchestration_auxiliary_bytes,
                                 estimate.total_auxiliary_bytes)) {
    estimate = {};
    estimate.reason_code = LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
    return estimate;
  }
  estimate.valid = true;
  estimate.reason_code = LlmExecutorReason::VALID;
  return estimate;
}

LlmBufferAllocationResult allocate_llm_buffers(const LlmMemoryWorkPlan& plan, LlmBufferSet& output) noexcept {
  LlmBufferAllocationResult result;
  try {
    if (!buffer_output_is_empty(output)) {
      result.reason_code = LlmExecutorReason::OUTPUT_NOT_EMPTY;
      return result;
    }
    if (!validate_work_plan_layout(plan)) {
      result.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
      return result;
    }
    result.auxiliary = calculate_llm_executor_auxiliary_estimate(plan);
    if (!result.auxiliary.valid) {
      result.reason_code = result.auxiliary.reason_code;
      return result;
    }
    if (!calculate_budget_with_executor_auxiliary(plan, result.auxiliary, result.memory_budget)) {
      result.reason_code = result.memory_budget.request.valid ? LlmExecutorReason::MEMORY_BUDGET_EXCEEDED
                                                              : LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
      return result;
    }

    LlmBufferSet candidate;
    candidate.weight = allocate_buffer(plan.geometry.active_weight_bytes_per_step, "LLM weight buffer");
    if (candidate.weight == nullptr) {
      result.reason_code = LlmExecutorReason::WEIGHT_MAPPING_FAILED;
      return result;
    }
    candidate.k = allocate_buffer(plan.geometry.k_mapping_bytes, "LLM K buffer");
    if (candidate.k == nullptr) {
      result.reason_code = LlmExecutorReason::K_MAPPING_FAILED;
      return result;
    }
    candidate.v = allocate_buffer(plan.geometry.v_mapping_bytes, "LLM V buffer");
    if (candidate.v == nullptr) {
      result.reason_code = LlmExecutorReason::V_MAPPING_FAILED;
      return result;
    }

    output = std::move(candidate);
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (...) {
    result.valid = false;
    if (result.reason_code == LlmExecutorReason::VALID) {
      result.reason_code = LlmExecutorReason::WEIGHT_MAPPING_FAILED;
    }
    return result;
  }
}

LlmResourcePreparationResult prepare_llm_execution_resources(const LlmMemoryWorkPlan& plan,
                                                             LlmExecutionResources& output) noexcept {
  LlmResourcePreparationResult result;
  try {
    if (!resource_output_is_empty(output)) {
      result.reason_code = LlmExecutorReason::OUTPUT_NOT_EMPTY;
      return result;
    }
    if (!validate_work_plan_layout(plan)) {
      result.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
      return result;
    }

    LlmExecutionResources candidate;
    const LlmBufferAllocationResult allocation = allocate_llm_buffers(plan, candidate.buffers);
    result.auxiliary = allocation.auxiliary;
    result.memory_budget = allocation.memory_budget;
    if (!allocation.valid) {
      result.reason_code = allocation.reason_code;
      return result;
    }

    candidate.layer_descriptors.reset(new (std::nothrow) LlmLayerDescriptor[plan.total_layer_descriptors]);
    candidate.sequence_descriptors.reset(new (std::nothrow) LlmKvSequenceDescriptor[plan.total_sequence_descriptors]);
    candidate.weight_references.reset(new (std::nothrow) LlmStaticSpanReference[plan.total_layer_descriptors]);
    candidate.k_references.reset(new (std::nothrow) LlmStaticSpanReference[plan.total_sequence_descriptors]);
    candidate.v_references.reset(new (std::nothrow) LlmStaticSpanReference[plan.total_sequence_descriptors]);
    if (candidate.layer_descriptors == nullptr || candidate.sequence_descriptors == nullptr ||
        candidate.weight_references == nullptr || candidate.k_references == nullptr ||
        candidate.v_references == nullptr) {
      result.reason_code = LlmExecutorReason::DESCRIPTOR_ALLOCATION_FAILED;
      return result;
    }

    candidate.model_plan_identity = plan.plan_identity;
    candidate.worker_count = plan.effective_workers;
    candidate.layer_descriptors_per_worker = plan.layer_descriptors_per_worker;
    candidate.sequence_descriptors_per_worker = plan.sequence_descriptors_per_worker;
    candidate.total_layer_descriptors = plan.total_layer_descriptors;
    candidate.total_sequence_descriptors = plan.total_sequence_descriptors;
    candidate.auxiliary = allocation.auxiliary;
    candidate.memory_budget = allocation.memory_budget;
    if (!materialize_descriptors(plan, candidate)) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      return result;
    }
    if (!initialize_resources(plan, candidate, candidate.initialization)) {
      result.reason_code = LlmExecutorReason::INITIALIZATION_FAILED;
      return result;
    }
    candidate.valid = true;
    if (!materialized_resources_match_plan(plan, candidate)) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      return result;
    }

    result.initialization = candidate.initialization;
    output = std::move(candidate);
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::DESCRIPTOR_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::INITIALIZATION_FAILED;
    return result;
  }
}

LlmExpectedChecksumResult calculate_llm_expected_checksums(const LlmMemoryWorkPlan& model_plan,
                                                           const LlmScenarioWorkPlan& scenario_plan,
                                                           const LlmExecutionResources& resources) noexcept {
  LlmExpectedChecksumResult result;
  try {
    if (!materialized_resources_match_plan(model_plan, resources)) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      return result;
    }
    if (!scenario_plan_matches_model(model_plan, scenario_plan)) {
      result.reason_code = LlmExecutorReason::SCENARIO_PLAN_MISMATCH;
      return result;
    }

    result.workers.resize(model_plan.effective_workers);
    const bool reads_weight = (llm_scenario_flags(scenario_plan.scenario) & kLlmScenarioFlagWeight) != 0;
    const bool reads_kv = (llm_scenario_flags(scenario_plan.scenario) & kLlmScenarioFlagKv) != 0;
    uint64_t total_weight_bytes = 0;
    uint64_t total_k_bytes = 0;
    uint64_t total_v_bytes = 0;

    for (size_t worker_index = 0; worker_index < model_plan.effective_workers; ++worker_index) {
      LlmWorkerChecksum& checksum = result.workers[worker_index];
      checksum.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
      checksum.k = initial_llm_read_checksum(LlmChecksumComponent::K);
      checksum.v = initial_llm_read_checksum(LlmChecksumComponent::V);
      const LlmWorkerWorkPlan& worker = model_plan.workers[worker_index];
      const LlmStaticSpanReference* const weight_references = resources.worker_weight_references(worker_index);
      const LlmStaticSpanReference* const k_references = resources.worker_k_references(worker_index);
      const LlmStaticSpanReference* const v_references = resources.worker_v_references(worker_index);
      if (weight_references == nullptr || k_references == nullptr || v_references == nullptr) {
        result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
        result.workers.clear();
        return result;
      }

      for (size_t step = 0; step < scenario_plan.steps; ++step) {
        for (size_t layer = 0; layer < model_plan.geometry.layer_count; ++layer) {
          if (reads_weight && !absorb_reference(checksum.weight, weight_references[layer])) {
            result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
            result.workers.clear();
            return result;
          }
          if (!reads_kv) {
            continue;
          }
          const LlmLayerRangeTemplate& layer_descriptor = worker.layers[layer];
          for (size_t batch_offset = 0; batch_offset < layer_descriptor.sequence_count; ++batch_offset) {
            const size_t sequence_index = layer_descriptor.first_sequence_index + batch_offset;
            if (sequence_index >= worker.sequences.size()) {
              result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
              result.workers.clear();
              return result;
            }
            const LlmKvSequenceRangeTemplate& sequence = worker.sequences[sequence_index];
            LlmStaticSpanReference k_adjusted;
            LlmStaticSpanReference v_adjusted;
            if (!append_adjusted_reference(k_references[sequence_index], sequence, model_plan.k_buffer_seed,
                                           scenario_plan.scenario_seed, static_cast<uint64_t>(step),
                                           LlmChecksumComponent::K, k_adjusted) ||
                !append_adjusted_reference(v_references[sequence_index], sequence, model_plan.v_buffer_seed,
                                           scenario_plan.scenario_seed, static_cast<uint64_t>(step),
                                           LlmChecksumComponent::V, v_adjusted) ||
                !absorb_reference(checksum.k, k_adjusted) || !absorb_reference(checksum.v, v_adjusted)) {
              result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
              result.workers.clear();
              return result;
            }
          }
        }
      }

      if (!add_checksum_bytes(checksum.weight.exact_bytes_read, total_weight_bytes) ||
          !add_checksum_bytes(checksum.k.exact_bytes_read, total_k_bytes) ||
          !add_checksum_bytes(checksum.v.exact_bytes_read, total_v_bytes)) {
        result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
        result.workers.clear();
        return result;
      }
    }

    uint64_t total_kv_bytes = 0;
    if (!add_checksum_bytes(total_k_bytes, total_kv_bytes) || !add_checksum_bytes(total_v_bytes, total_kv_bytes) ||
        total_weight_bytes != scenario_plan.weight_read_bytes || total_kv_bytes != scenario_plan.kv_read_bytes) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      result.workers.clear();
      return result;
    }
    result.run_checksum = fold_llm_worker_checksums(result.workers.data(), result.workers.size());
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_ALLOCATION_FAILED;
    result.workers.clear();
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::INVALID_SCENARIO_PLAN;
    result.workers.clear();
    return result;
  }
}

LlmKernelAdapter production_llm_kernel_adapter() noexcept { return {production_kernel_invoke, nullptr}; }

LlmExecutorResult execute_llm_scenario(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                       const LlmExecutionResources& resources, HighResTimer& timer,
                                       LlmKernelAdapter kernel, const LlmExecutorTestControl* test_control) noexcept {
  LlmExecutorResult result;
  result.requested_workers = model_plan.effective_workers;
  try {
    if (!materialized_resources_match_plan(model_plan, resources)) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      return result;
    }
    if (!scenario_plan_matches_model(model_plan, scenario_plan)) {
      result.reason_code = LlmExecutorReason::SCENARIO_PLAN_MISMATCH;
      return result;
    }
    if (kernel.invoke == nullptr) {
      result.reason_code = LlmExecutorReason::KERNEL_FAILED;
      return result;
    }

    LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(model_plan, scenario_plan, resources);
    if (!expected.valid) {
      result.reason_code = expected.reason_code;
      return result;
    }
    result.expected_checksums = std::move(expected.workers);
    result.expected_run_checksum = expected.run_checksum;
    result.actual_checksums.resize(model_plan.effective_workers);

    std::vector<uint8_t> worker_succeeded;
    std::vector<std::thread> threads;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    size_t ready_workers = 0;
    bool start_workers = false;
    bool cancel_workers = false;
    bool measurement_complete = false;
    bool timer_stop_succeeded = false;
    double measured_duration = 0.0;
    std::atomic<size_t> remaining_workers{model_plan.effective_workers};
    std::atomic<size_t> completed_workers{0};
    std::atomic<size_t> qos_successful_workers{0};
    std::atomic<size_t> qos_failed_workers{0};

    try {
      worker_succeeded.resize(model_plan.effective_workers, 0);
      threads.reserve(model_plan.effective_workers);
      for (size_t worker_index = 0; worker_index < model_plan.effective_workers; ++worker_index) {
        if (test_control != nullptr && test_control->fail_before_worker_index >= 0 &&
            worker_index == static_cast<size_t>(test_control->fail_before_worker_index)) {
          throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        threads.emplace_back([&, worker_index] {
          const int qos_result = set_worker_qos(test_control, worker_index);
          if (qos_result == KERN_SUCCESS) {
            qos_successful_workers.fetch_add(1, std::memory_order_relaxed);
          } else {
            qos_failed_workers.fetch_add(1, std::memory_order_relaxed);
          }

          {
            std::unique_lock<std::mutex> lock(state_mutex);
            ++ready_workers;
            observe_event(test_control, LlmExecutorEvent::WorkerReady, worker_index);
            state_cv.notify_all();
            state_cv.wait(lock, [&] { return start_workers || cancel_workers; });
            if (cancel_workers) {
              lock.unlock();
              observe_event(test_control, LlmExecutorEvent::WorkerCancelled, worker_index);
              return;
            }
          }

          const LlmKernelInvocation invocation{resources.worker_layers(worker_index),
                                               resources.worker_sequences(worker_index),
                                               static_cast<uint64_t>(model_plan.layer_descriptors_per_worker),
                                               static_cast<uint64_t>(scenario_plan.steps),
                                               llm_scenario_flags(scenario_plan.scenario),
                                               scenario_plan.scenario_seed,
                                               &result.actual_checksums[worker_index],
                                               worker_index};
          observe_event(test_control, LlmExecutorEvent::KernelStarted, worker_index);
          bool succeeded = false;
          try {
            succeeded = kernel.invoke(kernel.context, invocation);
          } catch (...) {
            succeeded = false;
          }
          worker_succeeded[worker_index] = succeeded ? 1 : 0;
          observe_event(test_control, LlmExecutorEvent::KernelCompleted, worker_index);

          asm volatile("dsb ish" ::: "memory");
          completed_workers.fetch_add(1, std::memory_order_relaxed);
          if (remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            double duration = 0.0;
            bool stopped = false;
            try {
              duration = timer.stop();
              stopped = true;
            } catch (...) {
              stopped = false;
            }
            if (stopped) {
              observe_event(test_control, LlmExecutorEvent::TimerStopped, worker_index);
            }
            {
              std::lock_guard<std::mutex> lock(state_mutex);
              measured_duration = duration;
              timer_stop_succeeded = stopped;
              measurement_complete = true;
            }
            state_cv.notify_one();
          }
        });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        cancel_workers = true;
      }
      state_cv.notify_all();
      join_threads(threads);
      result.created_workers = threads.size();
      result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
      result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
      result.worker_startup_failed = true;
      result.reason_code = LlmExecutorReason::WORKER_STARTUP_FAILED;
      return result;
    }

    try {
      {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_cv.wait(lock, [&] { return ready_workers == model_plan.effective_workers; });
        timer.start();
        result.timer_started = true;
        observe_event(test_control, LlmExecutorEvent::TimerStarted, model_plan.effective_workers);
        start_workers = true;
      }
      state_cv.notify_all();

      {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_cv.wait(lock, [&] { return measurement_complete; });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!start_workers) {
          cancel_workers = true;
        }
      }
      state_cv.notify_all();
      join_threads(threads);
      result.created_workers = threads.size();
      result.completed_workers = completed_workers.load(std::memory_order_relaxed);
      result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
      result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
      result.timer_stopped = timer_stop_succeeded;
      result.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }
    join_threads(threads);
    result.created_workers = threads.size();
    result.completed_workers = completed_workers.load(std::memory_order_relaxed);
    result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
    result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
    result.elapsed_seconds = measured_duration;
    result.timer_stopped = timer_stop_succeeded;

    result.kernel_succeeded =
        std::all_of(worker_succeeded.begin(), worker_succeeded.end(), [](uint8_t succeeded) { return succeeded != 0; });
    if (!result.kernel_succeeded) {
      result.reason_code = LlmExecutorReason::KERNEL_FAILED;
      return result;
    }
    if (!result.timer_stopped || !std::isfinite(result.elapsed_seconds) || result.elapsed_seconds <= 0.0) {
      result.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }

    observe_event(test_control, LlmExecutorEvent::ChecksumValidationStarted, model_plan.effective_workers);
    result.actual_run_checksum =
        fold_llm_worker_checksums(result.actual_checksums.data(), result.actual_checksums.size());
    result.checksum_valid = result.expected_checksums.size() == result.actual_checksums.size();
    for (size_t worker_index = 0; result.checksum_valid && worker_index < result.expected_checksums.size();
         ++worker_index) {
      result.checksum_valid =
          equal_worker_checksum(result.expected_checksums[worker_index], result.actual_checksums[worker_index]);
    }
    result.checksum_evaluated = true;
    if (!result.checksum_valid) {
      result.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
      return result;
    }

    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::KERNEL_FAILED;
    return result;
  }
}
