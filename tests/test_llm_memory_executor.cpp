// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_cpu_backend.h"
#include "llm_memory/llm_executor.h"
#include "llm_memory/llm_work_plan.h"
#include "test_memory_system_calls.h"
#include "test_timer_system_calls.h"

namespace {

constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kAppendLayer = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kAppendBatch = 0x94D049BB133111EBULL;
constexpr uint64_t kAppendWord = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kAppendKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kAppendVDomain = 0x5656565656565656ULL;
constexpr uint64_t kPagedLayer = 0xA24BAED4963EE407ULL;
constexpr uint64_t kPagedPhysical = 0x9FB21C651E98DF25ULL;
constexpr uint64_t kPagedWord = 0xC13FA9A902A6328FULL;

const LlmCpuExecutionPlan& cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    throw std::logic_error("expected CPU execution plan");
  }
  return *cpu_plan;
}

LlmMemoryWorkPlanRequest plan_request(const LlmGeometryRequest& geometry, size_t workers = 2) {
  LlmMemoryWorkPlanRequest request;
  request.geometry = geometry;
  request.requested_workers = workers;
  request.available_workers = workers;
  request.available_memory_bytes = 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = 1;
  request.base_seed = 42;
  return request;
}

LlmMemoryWorkPlan build_executor_ready_plan(const LlmGeometryRequest& geometry, size_t workers = 2) {
  LlmMemoryWorkPlanRequest request = plan_request(geometry, workers);
  LlmMemoryWorkPlan plan = build_llm_memory_work_plan(request);
  if (!plan.valid) {
    return plan;
  }
  const LlmExecutorAuxiliaryEstimate auxiliary =
      calculate_llm_executor_auxiliary_estimate(plan);
  if (!auxiliary.valid) {
    return plan;
  }
  readmit_llm_memory_work_plan(plan, auxiliary.checksum_auxiliary_bytes,
                              auxiliary.orchestration_auxiliary_bytes);
  return plan;
}

LlmGeometryRequest paged_geometry(size_t visible_tokens,
                                  size_t block_tokens,
                                  size_t record_bytes = 33,
                                  size_t layers = 2,
                                  size_t batch = 2) {
  LlmGeometryRequest geometry{257, layers, 1, 1, record_bytes, 1,
                              visible_tokens, batch};
  geometry.kv_block_tokens = block_tokens;
  geometry.kv_layout = LlmKvLayout::Paged;
  return geometry;
}

LlmGeometryRequest prefill_geometry(size_t prompt_tokens = 7,
                                    size_t query_tile_tokens = 3,
                                    size_t record_bytes = 33,
                                    size_t layers = 2,
                                    size_t batch = 2) {
  LlmGeometryRequest geometry{257, layers, 1, 1, record_bytes, 1, 0,
                              batch};
  geometry.phase = LlmPhase::Prefill;
  geometry.prompt_tokens = prompt_tokens;
  geometry.attention_query_tile_tokens = query_tile_tokens;
  return geometry;
}

uint8_t expected_buffer_byte(uint64_t seed, size_t absolute_byte) {
  const uint64_t word = seed + kGolden * (static_cast<uint64_t>(absolute_byte / 8) + 1);
  return static_cast<uint8_t>(word >> (8 * (absolute_byte % 8)));
}

uint8_t expected_paged_buffer_byte(uint64_t seed, size_t layer,
                                   uint32_t physical_block,
                                   size_t block_byte) {
  const uint64_t word =
      seed + kPagedLayer * (static_cast<uint64_t>(layer) + 1) +
      kPagedPhysical * (static_cast<uint64_t>(physical_block) + 1) +
      kPagedWord * (static_cast<uint64_t>(block_byte / 8) + 1);
  return static_cast<uint8_t>(word >> (8 * (block_byte % 8)));
}

uint64_t load_partial(const uint8_t* bytes, size_t count) {
  uint64_t word = 0;
  for (size_t index = 0; index < count; ++index) {
    word |= static_cast<uint64_t>(bytes[index]) << (8 * index);
  }
  return word;
}

LlmStaticSpanReference reference_span(const uint8_t* bytes, size_t count) {
  LlmStaticSpanReference reference{};
  reference.span_bytes = count;
  for (size_t offset = 0, word_index = 0; offset < count; offset += 8, ++word_index) {
    const size_t word_bytes = std::min<size_t>(8, count - offset);
    const uint64_t word = load_partial(bytes + offset, word_bytes);
    if ((word_index % 2) == 0) {
      reference.span_even += word;
    } else {
      reference.span_odd += word;
    }
  }
  return reference;
}

uint64_t rotate_left(uint64_t value, unsigned int amount) { return (value << amount) | (value >> (64 - amount)); }

void absorb(LlmReadChecksumComponent& state, const uint8_t* bytes, size_t count) {
  if (count == 0) {
    return;
  }
  const LlmStaticSpanReference reference = reference_span(bytes, count);
  const uint64_t ordinal = state.span_count;
  state.state_a = rotate_left(state.state_a + reference.span_even + kGolden * (ordinal + 1), 17);
  state.state_b =
      rotate_left(state.state_b + reference.span_odd + static_cast<uint64_t>(count) + kAppendWord * (ordinal + 1), 29);
  state.exact_bytes_read += count;
  ++state.span_count;
}

uint64_t scalar_append_word(uint64_t seed, uint64_t step, uint64_t layer, uint64_t batch, uint64_t word_index,
                            LlmChecksumComponent component) {
  const uint64_t domain = component == LlmChecksumComponent::K ? kAppendKDomain : kAppendVDomain;
  return seed + kGolden * (step + 1) + kAppendLayer * (layer + 1) + kAppendBatch * (batch + 1) +
         kAppendWord * (word_index + 1) + domain;
}

void apply_append(std::vector<uint8_t>& visible, const uint8_t* visible_pointer, const uint8_t* append_pointer,
                  size_t append_bytes, size_t record_byte_offset, uint64_t seed, uint64_t step, uint64_t layer,
                  uint64_t batch, LlmChecksumComponent component) {
  if (append_bytes == 0) {
    return;
  }
  const size_t local_offset = static_cast<size_t>(append_pointer - visible_pointer);
  ASSERT_LE(local_offset + append_bytes, visible.size());
  for (size_t byte = 0; byte < append_bytes; ++byte) {
    const size_t canonical_byte = record_byte_offset + byte;
    const uint64_t word = scalar_append_word(seed, step, layer, batch, canonical_byte / 8, component);
    visible[local_offset + byte] = static_cast<uint8_t>(word >> (8 * (canonical_byte % 8)));
  }
}

std::vector<LlmWorkerChecksum> scalar_expected(const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan& scenario,
                                               const LlmExecutionResources& resources) {
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(model);
  std::vector<LlmWorkerChecksum> expected(cpu_plan.effective_workers);
  for (size_t worker = 0; worker < cpu_plan.effective_workers; ++worker) {
    expected[worker].weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
    expected[worker].k = initial_llm_read_checksum(LlmChecksumComponent::K);
    expected[worker].v = initial_llm_read_checksum(LlmChecksumComponent::V);
    const LlmLayerDescriptor* layers = resources.worker_layers(worker);
    const LlmKvSequenceDescriptor* sequences = resources.worker_sequences(worker);
    for (size_t step = 0; step < scenario.work_units; ++step) {
      for (size_t layer = 0; layer < model.geometry.layer_count; ++layer) {
        const LlmLayerDescriptor& layer_descriptor = layers[layer];
        if ((llm_scenario_flags(scenario.scenario) & kLlmScenarioFlagWeight) != 0) {
          absorb(expected[worker].weight, layer_descriptor.weight_ptr, layer_descriptor.weight_bytes);
        }
        if ((llm_scenario_flags(scenario.scenario) & kLlmScenarioFlagKv) == 0) {
          continue;
        }
        for (size_t local = 0; local < layer_descriptor.sequence_count; ++local) {
          const LlmKvSequenceDescriptor& sequence = sequences[layer_descriptor.first_sequence_index + local];
          std::vector<uint8_t> k_visible(sequence.k_visible_ptr, sequence.k_visible_ptr + sequence.k_visible_bytes);
          std::vector<uint8_t> v_visible(sequence.v_visible_ptr, sequence.v_visible_ptr + sequence.v_visible_bytes);
          apply_append(k_visible, sequence.k_visible_ptr, sequence.k_append_ptr, sequence.k_append_bytes,
                       sequence.append_record_byte_offset, scenario.scenario_seed, step, layer_descriptor.layer_index,
                       sequence.batch_sequence_index, LlmChecksumComponent::K);
          apply_append(v_visible, sequence.v_visible_ptr, sequence.v_append_ptr, sequence.v_append_bytes,
                       sequence.append_record_byte_offset, scenario.scenario_seed, step, layer_descriptor.layer_index,
                       sequence.batch_sequence_index, LlmChecksumComponent::V);
          absorb(expected[worker].k, k_visible.data(), k_visible.size());
          absorb(expected[worker].v, v_visible.data(), v_visible.size());
        }
      }
    }
  }
  return expected;
}

void expect_component_equal(const LlmReadChecksumComponent& actual, const LlmReadChecksumComponent& expected) {
  EXPECT_EQ(actual.state_a, expected.state_a);
  EXPECT_EQ(actual.state_b, expected.state_b);
  EXPECT_EQ(actual.exact_bytes_read, expected.exact_bytes_read);
  EXPECT_EQ(actual.span_count, expected.span_count);
}

void expect_worker_equal(const LlmWorkerChecksum& actual, const LlmWorkerChecksum& expected) {
  expect_component_equal(actual.weight, expected.weight);
  expect_component_equal(actual.k, expected.k);
  expect_component_equal(actual.v, expected.v);
}

uint64_t deterministic_tick = 0;

uint64_t deterministic_absolute_time() {
  const uint64_t current = deterministic_tick;
  deterministic_tick += 100;
  return current;
}

void reset_deterministic_tick() { deterministic_tick = 0; }

using ScopedExecutorTimer =
    test_timer_system_calls::ScopedTimerSystemCalls<deterministic_absolute_time, reset_deterministic_tick>;

uint64_t stationary_absolute_time() { return 7; }

using ScopedStationaryExecutorTimer = test_timer_system_calls::ScopedTimerSystemCalls<stationary_absolute_time>;

uint64_t throwing_start_absolute_time() { throw std::runtime_error("injected timer start exception"); }

size_t stop_exception_clock_calls = 0;

void reset_stop_exception_clock() { stop_exception_clock_calls = 0; }

uint64_t throwing_stop_absolute_time() {
  ++stop_exception_clock_calls;
  if (stop_exception_clock_calls == 2) {
    throw std::runtime_error("injected timer stop exception");
  }
  return 100;
}

using ScopedThrowingStartTimer = test_timer_system_calls::ScopedTimerSystemCalls<throwing_start_absolute_time>;
using ScopedThrowingStopTimer =
    test_timer_system_calls::ScopedTimerSystemCalls<throwing_stop_absolute_time, reset_stop_exception_clock>;

struct FakeKernelContext {
  const std::vector<LlmWorkerChecksum>* expected = nullptr;
  std::vector<LlmKernelInvocation> invocations;
  std::vector<LlmExecutorEvent> events;
  std::mutex mutex;
  std::atomic<size_t> calls{0};
  int fail_worker = -1;
  int corrupt_worker = -1;
};

bool fake_kernel(void* opaque, const LlmKernelInvocation& invocation) {
  auto& context = *static_cast<FakeKernelContext*>(opaque);
  {
    std::lock_guard<std::mutex> lock(context.mutex);
    context.invocations.push_back(invocation);
  }
  context.calls.fetch_add(1, std::memory_order_relaxed);
  if (static_cast<int>(invocation.worker_index) == context.fail_worker) {
    return false;
  }
  *invocation.output = (*context.expected)[invocation.worker_index];
  if (static_cast<int>(invocation.worker_index) == context.corrupt_worker) {
    ++invocation.output->k.state_b;
  }
  return true;
}

bool throwing_kernel(void*, const LlmKernelInvocation&) { throw std::runtime_error("injected kernel exception"); }

enum class PagedCorruption {
  Append,
  Padding,
};

struct PagedCorruptionContext {
  PagedCorruption kind = PagedCorruption::Append;
  bool corrupted = false;
};

struct PrefillCorruptionContext {
  std::atomic<bool> corrupted{false};
};

bool prefill_corrupting_kernel(void* opaque,
                               const LlmKernelInvocation& invocation) {
  auto& context = *static_cast<PrefillCorruptionContext*>(opaque);
  if (invocation.phase != LlmPhase::Prefill ||
      invocation.prefill_layers == nullptr ||
      invocation.prefill_sequences == nullptr || invocation.output == nullptr) {
    return false;
  }
  llm_prefill_memory_asm(
      invocation.prefill_layers, invocation.prefill_sequences,
      invocation.layer_count, invocation.work_unit_count,
      invocation.scenario_flags, invocation.scenario_seed,
      invocation.output);
  bool expected = false;
  if (context.corrupted.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    for (size_t layer = 0; layer < invocation.layer_count; ++layer) {
      const LlmPrefillLayerDescriptor& layer_descriptor =
          invocation.prefill_layers[layer];
      for (size_t local = 0; local < layer_descriptor.sequence_count;
           ++local) {
        const LlmPrefillKvSequenceDescriptor& sequence =
            invocation.prefill_sequences[
                layer_descriptor.first_sequence_index + local];
        if (sequence.owned_token_count != 0) {
          sequence.k_owned_ptr[0] ^= 0x80;
          return true;
        }
      }
    }
    return false;
  }
  return true;
}

bool paged_corrupting_kernel(void* opaque,
                             const LlmKernelInvocation& invocation) {
  auto& context = *static_cast<PagedCorruptionContext*>(opaque);
  if (invocation.kv_layout != LlmKvLayout::Paged ||
      invocation.paged_layers == nullptr ||
      invocation.paged_assignments == nullptr || invocation.output == nullptr) {
    return false;
  }
  llm_decode_memory_paged_asm(
      invocation.paged_layers, invocation.paged_assignments,
      invocation.layer_count, invocation.work_unit_count,
      invocation.scenario_flags, invocation.scenario_seed,
      invocation.output);
  if (context.corrupted) {
    return true;
  }
  for (size_t layer = 0; layer < invocation.layer_count; ++layer) {
    const LlmPagedLayerDescriptor& layer_descriptor =
        invocation.paged_layers[layer];
    for (size_t local = 0; local < layer_descriptor.assignment_count;
         ++local) {
      const LlmPagedKvAssignmentDescriptor& assignment =
          invocation.paged_assignments[
              layer_descriptor.first_assignment_index + local];
      if (assignment.owned_block_count == 0 ||
          assignment.first_logical_block + assignment.owned_block_count !=
              assignment.blocks_per_sequence) {
        continue;
      }
      const uint32_t physical_id =
          assignment.block_table_row[assignment.blocks_per_sequence - 1];
      uint8_t* const block =
          assignment.k_layer_pool + physical_id * assignment.block_bytes;
      const size_t byte_offset =
          context.kind == PagedCorruption::Append
              ? assignment.decode_append_offset
              : assignment.last_block_valid_bytes;
      if (byte_offset >= assignment.block_bytes) {
        return false;
      }
      block[byte_offset] ^= 0x80;
      context.corrupted = true;
      return true;
    }
  }
  return false;
}

void observe_executor_event(void* opaque, LlmExecutorEvent event, size_t) noexcept {
  auto& context = *static_cast<FakeKernelContext*>(opaque);
  std::lock_guard<std::mutex> lock(context.mutex);
  context.events.push_back(event);
}

int fake_worker_qos(void*, size_t worker_index) noexcept { return worker_index == 0 ? 0 : 1; }

size_t first_event(const std::vector<LlmExecutorEvent>& events, LlmExecutorEvent event) {
  const auto found = std::find(events.begin(), events.end(), event);
  return found == events.end() ? std::numeric_limits<size_t>::max() : static_cast<size_t>(found - events.begin());
}

size_t last_event(const std::vector<LlmExecutorEvent>& events, LlmExecutorEvent event) {
  for (size_t index = events.size(); index > 0; --index) {
    if (events[index - 1] == event) {
      return index - 1;
    }
  }
  return std::numeric_limits<size_t>::max();
}

class LlmMemoryExecutorTest : public FakeMemorySystemCallsTest {};

}  // namespace

TEST_F(LlmMemoryExecutorTest, BufferAndAppendGeneratorsMatchIndependentGoldenVectors) {
  EXPECT_EQ(llm_buffer_pattern_word(0, 0), 0x9E3779B97F4A7C15ULL);
  EXPECT_EQ(llm_buffer_pattern_word(0, 1), 0x3C6EF372FE94F82AULL);
  EXPECT_EQ(llm_buffer_pattern_word(std::numeric_limits<uint64_t>::max(), 0), 0x9E3779B97F4A7C14ULL);
  EXPECT_EQ(llm_buffer_pattern_word(0x0123456789ABCDEFULL, 5), 0xB6701FC0856AB66DULL);

  EXPECT_EQ(llm_append_word(0, 0, 0, 0, 0, LlmChecksumComponent::K), 0x149454E56105BC97ULL);
  EXPECT_EQ(llm_append_word(0, 0, 0, 0, 0, LlmChecksumComponent::V), 0x1F9F5FF06C10C7A2ULL);
  EXPECT_EQ(llm_append_word(0x0123456789ABCDEFULL, 2, 3, 4, 5, LlmChecksumComponent::K), 0x15FD848D8C7B6F66ULL);

  EXPECT_EQ(llm_scenario_flags(LlmScenario::WeightsOnly), 1u);
  EXPECT_EQ(llm_scenario_flags(LlmScenario::KvOnly), 2u);
  EXPECT_EQ(llm_scenario_flags(LlmScenario::Mixed), 3u);
  EXPECT_EQ(llm_scenario_flags(static_cast<LlmScenario>(255)), 0u);
}

TEST_F(LlmMemoryExecutorTest, AuxiliaryAdmissionFailsBeforeMmapAndExactEstimateCanBeReplanned) {
  const LlmGeometryRequest geometry = {64, 1, 1, 1, 19, 1, 2, 1};
  LlmMemoryWorkPlanRequest request = plan_request(geometry, 2);
  const LlmMemoryWorkPlan under_budgeted = build_llm_memory_work_plan(request);
  ASSERT_TRUE(under_budgeted.valid) << under_budgeted.reason_code;
  const LlmExecutorAuxiliaryEstimate estimate = calculate_llm_executor_auxiliary_estimate(under_budgeted);
  ASSERT_TRUE(estimate.valid) << estimate.reason_code;
  EXPECT_GT(estimate.checksum_auxiliary_bytes, 0u);
  EXPECT_GT(estimate.orchestration_auxiliary_bytes, 0u);
  EXPECT_EQ(estimate.total_auxiliary_bytes, estimate.checksum_auxiliary_bytes + estimate.orchestration_auxiliary_bytes);

  LlmBufferSet buffers;
  const LlmBufferAllocationResult rejected = allocate_llm_buffers(under_budgeted, buffers);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code, LlmExecutorReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_FALSE(buffers.complete());
  EXPECT_EQ(state.map_calls, 0u);

  request.checksum_auxiliary_bytes = estimate.checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = estimate.orchestration_auxiliary_bytes;
  const LlmMemoryWorkPlan admitted = build_llm_memory_work_plan(request);
  ASSERT_TRUE(admitted.valid) << admitted.reason_code;
  const LlmBufferAllocationResult allocated = allocate_llm_buffers(admitted, buffers);
  ASSERT_TRUE(allocated.valid) << allocated.reason_code;
  EXPECT_TRUE(buffers.complete());
  EXPECT_EQ(state.map_calls, 3u);
  EXPECT_EQ(state.advise_calls, 3u);
  EXPECT_EQ(state.last_advice, MADV_WILLNEED);
  EXPECT_EQ(allocated.memory_budget.request.checksum_auxiliary_bytes, estimate.checksum_auxiliary_bytes);
}

TEST_F(LlmMemoryExecutorTest, ThreeMappingAllocationIsAtomicForEveryFailurePosition) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({64, 1, 1, 1, 19, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  LlmBufferSet retained;
  ASSERT_TRUE(allocate_llm_buffers(plan, retained).valid);
  const std::array<void*, 3> retained_pointers = {retained.weight.get(), retained.k.get(), retained.v.get()};

  const size_t maps_before_replacement = state.map_calls;
  const LlmBufferAllocationResult replacement = allocate_llm_buffers(plan, retained);
  EXPECT_FALSE(replacement.valid);
  EXPECT_EQ(replacement.reason_code, LlmExecutorReason::OUTPUT_NOT_EMPTY);
  EXPECT_EQ(retained.weight.get(), retained_pointers[0]);
  EXPECT_EQ(retained.k.get(), retained_pointers[1]);
  EXPECT_EQ(retained.v.get(), retained_pointers[2]);
  EXPECT_EQ(state.map_calls, maps_before_replacement);

  for (size_t failure_position = 1; failure_position <= 3; ++failure_position) {
    LlmBufferSet candidate;
    state.fail_map_on_call = state.map_calls + failure_position;
    const size_t unmaps_before = state.unmap_calls;
    testing::internal::CaptureStderr();
    const LlmBufferAllocationResult result = allocate_llm_buffers(plan, candidate);
    static_cast<void>(testing::internal::GetCapturedStderr());
    EXPECT_FALSE(result.valid) << failure_position;
    EXPECT_FALSE(candidate.complete()) << failure_position;
    EXPECT_EQ(state.unmap_calls, unmaps_before + failure_position - 1) << failure_position;
    state.fail_map_on_call = 0;
  }
}

TEST_F(LlmMemoryExecutorTest, PreparationMaterializesExactDescriptorsPatternsAndStaticSums) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({64, 1, 1, 1, 19, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared = prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  ASSERT_TRUE(resources.valid);
  EXPECT_TRUE(resources.initialization.complete);
  EXPECT_EQ(resources.initialization.weight_bytes, 64u);
  EXPECT_EQ(resources.initialization.k_bytes, 38u);
  EXPECT_EQ(resources.initialization.v_bytes, 38u);
  EXPECT_EQ(resources.initialization.total_bytes, 140u);
  EXPECT_EQ(resources.worker_count, 2u);

  const auto* weight = static_cast<const uint8_t*>(resources.buffers.weight.get());
  const auto* k = static_cast<const uint8_t*>(resources.buffers.k.get());
  const auto* v = static_cast<const uint8_t*>(resources.buffers.v.get());
  for (size_t byte = 0; byte < 64; ++byte) {
    EXPECT_EQ(weight[byte], expected_buffer_byte(plan.weight_buffer_seed, byte));
  }
  for (size_t byte = 0; byte < 38; ++byte) {
    EXPECT_EQ(k[byte], expected_buffer_byte(plan.k_buffer_seed, byte));
    EXPECT_EQ(v[byte], expected_buffer_byte(plan.v_buffer_seed, byte));
  }

  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  for (size_t worker = 0; worker < cpu_plan.effective_workers; ++worker) {
    const LlmLayerDescriptor& layer = resources.worker_layers(worker)[0];
    const LlmKvSequenceDescriptor& sequence = resources.worker_sequences(worker)[0];
    const LlmWorkerWorkPlan& planned = cpu_plan.workers[worker];
    EXPECT_EQ(layer.weight_ptr,
              planned.layers[0].weight.span_bytes == 0 ? nullptr : weight + planned.layers[0].weight.offset_bytes);
    EXPECT_EQ(layer.weight_bytes, planned.layers[0].weight.span_bytes);
    EXPECT_EQ(layer.first_sequence_index, 0u);
    EXPECT_EQ(layer.sequence_count, 1u);
    EXPECT_EQ(layer.layer_index, 0u);
    EXPECT_EQ(layer.reserved_zero, 0u);
    EXPECT_EQ(sequence.k_visible_ptr, k + planned.sequences[0].k_visible.offset_bytes);
    EXPECT_EQ(sequence.v_visible_ptr, v + planned.sequences[0].v_visible.offset_bytes);
    EXPECT_EQ(sequence.k_visible_bytes, planned.sequences[0].k_visible.span_bytes);
    EXPECT_EQ(sequence.v_visible_bytes, planned.sequences[0].v_visible.span_bytes);
    EXPECT_EQ(sequence.k_append_ptr, k + planned.sequences[0].k_append.offset_bytes);
    EXPECT_EQ(sequence.v_append_ptr, v + planned.sequences[0].v_append.offset_bytes);
    EXPECT_EQ(sequence.append_record_byte_offset, planned.sequences[0].append_record_byte_offset);

    const LlmStaticSpanReference weight_reference = reference_span(layer.weight_ptr, layer.weight_bytes);
    const LlmStaticSpanReference k_reference = reference_span(sequence.k_visible_ptr, sequence.k_visible_bytes);
    const LlmStaticSpanReference v_reference = reference_span(sequence.v_visible_ptr, sequence.v_visible_bytes);
    const LlmStaticSpanReference& actual_weight = resources.worker_weight_references(worker)[0];
    const LlmStaticSpanReference& actual_k = resources.worker_k_references(worker)[0];
    const LlmStaticSpanReference& actual_v = resources.worker_v_references(worker)[0];
    EXPECT_EQ(actual_weight.span_even, weight_reference.span_even);
    EXPECT_EQ(actual_weight.span_odd, weight_reference.span_odd);
    EXPECT_EQ(actual_weight.span_bytes, weight_reference.span_bytes);
    EXPECT_EQ(actual_k.span_even, k_reference.span_even);
    EXPECT_EQ(actual_k.span_odd, k_reference.span_odd);
    EXPECT_EQ(actual_k.span_bytes, k_reference.span_bytes);
    EXPECT_EQ(actual_v.span_even, v_reference.span_even);
    EXPECT_EQ(actual_v.span_odd, v_reference.span_odd);
    EXPECT_EQ(actual_v.span_bytes, v_reference.span_bytes);
  }

  EXPECT_EQ(cpu_plan.workers[0].sequences[0].append_record_byte_offset, 0u);
  EXPECT_EQ(cpu_plan.workers[1].sequences[0].append_record_byte_offset,
            13u);
  EXPECT_EQ(resources.worker_sequences(1)[0].k_append_bytes, 6u);

  const size_t weight_slot = FakeMemorySystemCallState::kSlotSize;
  EXPECT_TRUE(std::all_of(weight + 64, weight + weight_slot, [](uint8_t byte) { return byte == 0; }));

  const size_t maps_before_reprepare = state.map_calls;
  const LlmResourcePreparationResult reprepare = prepare_llm_execution_resources(plan, resources);
  EXPECT_FALSE(reprepare.valid);
  EXPECT_EQ(reprepare.reason_code, LlmExecutorReason::OUTPUT_NOT_EMPTY);
  EXPECT_EQ(state.map_calls, maps_before_reprepare);
  EXPECT_EQ(resources.buffers.weight.get(), weight);
  EXPECT_TRUE(resources.valid);
}

TEST_F(LlmMemoryExecutorTest,
       PrefillPreparationMaterializesScenarioSetsAndWeightsOnlyUsesNullKv) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan(prefill_geometry(), 3);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu_plan.prefill.has_value());
  const size_t rows =
      plan.geometry.layer_count * plan.geometry.batch_size;
  ASSERT_EQ(cpu_plan.sequence_descriptors_per_worker,
            rows * kLlmScenarioCount);

  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared =
      prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  EXPECT_EQ(resources.layer_descriptors, nullptr);
  EXPECT_EQ(resources.sequence_descriptors, nullptr);
  EXPECT_EQ(resources.paged_layer_descriptors, nullptr);
  EXPECT_EQ(resources.paged_assignment_descriptors, nullptr);
  ASSERT_NE(resources.prefill_layer_descriptors, nullptr);
  ASSERT_NE(resources.prefill_sequence_descriptors, nullptr);
  EXPECT_NE(resources.weight_references, nullptr);
  EXPECT_EQ(resources.k_references, nullptr);
  EXPECT_EQ(resources.v_references, nullptr);
  EXPECT_EQ(resources.initialization.k_bytes,
            plan.geometry.k_mapping_bytes);
  EXPECT_EQ(resources.initialization.v_bytes,
            plan.geometry.v_mapping_bytes);

  const auto* const k =
      static_cast<const uint8_t*>(resources.buffers.k.get());
  const auto* const v =
      static_cast<const uint8_t*>(resources.buffers.v.get());
  for (size_t worker = 0; worker < cpu_plan.effective_workers; ++worker) {
    const LlmPrefillLayerDescriptor* const layers =
        resources.worker_prefill_layers(worker);
    ASSERT_NE(layers, nullptr);
    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      EXPECT_EQ(layers[layer].first_sequence_index,
                layer * plan.geometry.batch_size);
      EXPECT_EQ(layers[layer].sequence_count,
                plan.geometry.batch_size);
      EXPECT_EQ(layers[layer].layer_index, layer);
      EXPECT_EQ(layers[layer].reserved_zero, 0u);
    }
    const LlmPrefillKvSequenceDescriptor* const weights =
        resources.worker_prefill_sequences(worker,
                                           LlmScenario::WeightsOnly);
    const LlmPrefillKvSequenceDescriptor* const kv =
        resources.worker_prefill_sequences(worker, LlmScenario::KvOnly);
    const LlmPrefillKvSequenceDescriptor* const mixed =
        resources.worker_prefill_sequences(worker, LlmScenario::Mixed);
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(kv, nullptr);
    ASSERT_NE(mixed, nullptr);
    EXPECT_EQ(kv - weights, static_cast<ptrdiff_t>(rows));
    EXPECT_EQ(mixed - kv, static_cast<ptrdiff_t>(rows));
    for (size_t row = 0; row < rows; ++row) {
      EXPECT_EQ(weights[row].k_owned_ptr, nullptr);
      EXPECT_EQ(weights[row].v_owned_ptr, nullptr);
      EXPECT_EQ(weights[row].owned_token_count, 0u);
      for (const LlmPrefillKvSequenceDescriptor* descriptor :
           {kv + row, mixed + row}) {
        const size_t descriptor_end =
            descriptor->first_token + descriptor->owned_token_count;
        EXPECT_LE(descriptor_end,
                  plan.geometry.prefill->prompt_tokens);
        EXPECT_EQ(descriptor->prompt_tokens,
                  plan.geometry.prefill->prompt_tokens);
        EXPECT_EQ(descriptor->attention_query_tile_tokens,
                  plan.geometry.prefill->attention_query_tile_tokens);
        EXPECT_EQ(descriptor->record_bytes,
                  plan.geometry.k_or_v_record_bytes_per_layer);
        EXPECT_EQ(descriptor->reserved_zero, 0u);
        if (descriptor->owned_token_count != 0) {
          EXPECT_GE(descriptor->k_owned_ptr, k);
          EXPECT_LT(descriptor->k_owned_ptr,
                    k + plan.geometry.k_mapping_bytes);
          EXPECT_GE(descriptor->v_owned_ptr, v);
          EXPECT_LT(descriptor->v_owned_ptr,
                    v + plan.geometry.v_mapping_bytes);
        }
      }
    }
  }

  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
      plan, LlmScenario::WeightsOnly, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const LlmExpectedChecksumResult expected =
      calculate_llm_expected_checksums(plan, scenario, resources);
  ASSERT_TRUE(expected.valid) << expected.reason_code;
  ScopedExecutorTimer timer_scope;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  FakeKernelContext context;
  context.expected = &expected.workers;
  const LlmExecutorResult result = execute_llm_scenario(
      plan, scenario, resources, *timer, {fake_kernel, &context});
  ASSERT_TRUE(result.valid) << result.reason_code;
  ASSERT_EQ(context.invocations.size(), cpu_plan.effective_workers);
  for (const LlmKernelInvocation& invocation : context.invocations) {
    EXPECT_EQ(invocation.phase, LlmPhase::Prefill);
    EXPECT_EQ(invocation.layers, nullptr);
    EXPECT_EQ(invocation.sequences, nullptr);
    EXPECT_EQ(invocation.paged_layers, nullptr);
    EXPECT_EQ(invocation.paged_assignments, nullptr);
    EXPECT_EQ(invocation.prefill_layers,
              resources.worker_prefill_layers(invocation.worker_index));
    EXPECT_EQ(invocation.prefill_sequences, nullptr);
    EXPECT_EQ(invocation.work_unit_count, 2u);
    EXPECT_EQ(invocation.scenario_flags, kLlmScenarioFlagWeight);
  }
}

TEST_F(LlmMemoryExecutorTest,
       PrefillNoallocCanonicalValidatorAcceptsExactTileBoundaries) {
  struct Case {
    size_t prompt_tokens;
    size_t query_tile_tokens;
    size_t record_bytes;
    size_t workers;
  };
  constexpr std::array<Case, 3> kCases = {
      {{4, 1, 31, 1}, {3, 3, 32, 3}, {5, 2, 33, 3}}};
  for (const Case& test_case : kCases) {
    SCOPED_TRACE(::testing::Message()
                 << "P=" << test_case.prompt_tokens
                 << " Q=" << test_case.query_tile_tokens
                 << " R=" << test_case.record_bytes
                 << " workers=" << test_case.workers);
    const LlmMemoryWorkPlan plan = build_executor_ready_plan(
        prefill_geometry(test_case.prompt_tokens,
                         test_case.query_tile_tokens,
                         test_case.record_bytes),
        test_case.workers);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_TRUE(validate_llm_prefill_cpu_execution_evidence(plan));

    LlmExecutionResources resources;
    const LlmResourcePreparationResult result =
        prepare_llm_execution_resources(plan, resources);
    EXPECT_TRUE(result.valid) << result.reason_code;
  }
}

TEST_F(LlmMemoryExecutorTest,
       PrefillStructuralValidationRejectsShortScenarioDescriptorSet) {
  LlmMemoryWorkPlan plan =
      build_executor_ready_plan(prefill_geometry(), 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  ASSERT_NE(cpu_plan, nullptr);
  ASSERT_TRUE(cpu_plan->prefill.has_value());
  ASSERT_FALSE(cpu_plan->workers[0].prefill_sequences.empty());
  cpu_plan->workers[0].prefill_sequences.pop_back();
  const size_t map_calls_before = state.map_calls;

  LlmExecutionResources resources;
  const LlmResourcePreparationResult result =
      prepare_llm_execution_resources(plan, resources);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);
  EXPECT_EQ(state.map_calls, map_calls_before);
  EXPECT_FALSE(resources.valid);
  EXPECT_EQ(resources.prefill_layer_descriptors, nullptr);
  EXPECT_EQ(resources.prefill_sequence_descriptors, nullptr);
}

TEST_F(LlmMemoryExecutorTest,
       PrefillCanonicalOwnershipAndIdentityTamperingIsRejectedBeforeMmap) {
  for (size_t tamper = 0; tamper < 8; ++tamper) {
    SCOPED_TRACE(::testing::Message() << "tamper=" << tamper);
    LlmMemoryWorkPlan plan =
        build_executor_ready_plan(prefill_geometry(), 2);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    LlmCpuExecutionPlan* const cpu_plan =
        get_llm_cpu_execution_plan(plan);
    ASSERT_NE(cpu_plan, nullptr);
    ASSERT_TRUE(cpu_plan->prefill.has_value());
    const size_t rows =
        plan.geometry.layer_count * plan.geometry.batch_size;
    LlmPrefillCpuScenarioExecutionPlan& kv_scenario =
        cpu_plan->prefill->scenarios[
            static_cast<size_t>(LlmScenario::KvOnly)];

    if (tamper == 0) {
      const size_t scenario_base =
          static_cast<size_t>(LlmScenario::KvOnly) * rows;
      std::array<size_t, 2> ordered_workers = {0, 1};
      std::sort(ordered_workers.begin(), ordered_workers.end(),
                [&](size_t lhs, size_t rhs) {
                  return cpu_plan->workers[lhs]
                             .prefill_sequences[scenario_base]
                             .first_token <
                         cpu_plan->workers[rhs]
                             .prefill_sequences[scenario_base]
                             .first_token;
                });
      LlmPrefillKvSequenceRangeTemplate& left =
          cpu_plan->workers[ordered_workers[0]]
              .prefill_sequences[scenario_base];
      LlmPrefillKvSequenceRangeTemplate& right =
          cpu_plan->workers[ordered_workers[1]]
              .prefill_sequences[scenario_base];
      ASSERT_GT(left.owned_token_count, 0u);
      ASSERT_GT(right.owned_token_count, 0u);
      const size_t record_bytes =
          plan.geometry.k_or_v_record_bytes_per_layer;
      if (left.owned_token_count > 1) {
        --left.owned_token_count;
        left.k_owned.span_bytes -= record_bytes;
        left.v_owned = left.k_owned;
        --right.first_token;
        ++right.owned_token_count;
        right.k_owned.offset_bytes -= record_bytes;
        right.k_owned.span_bytes += record_bytes;
        right.v_owned = right.k_owned;
      } else {
        ASSERT_GT(right.owned_token_count, 1u);
        ++left.owned_token_count;
        left.k_owned.span_bytes += record_bytes;
        left.v_owned = left.k_owned;
        ++right.first_token;
        --right.owned_token_count;
        right.k_owned.offset_bytes += record_bytes;
        right.k_owned.span_bytes -= record_bytes;
        right.v_owned = right.k_owned;
      }
    } else if (tamper == 1) {
      ASSERT_GT(kv_scenario.ownership_scopes.size(), 1u);
      std::swap(kv_scenario.ownership_scopes[0],
                kv_scenario.ownership_scopes[1]);
    } else if (tamper == 2) {
      ASSERT_FALSE(kv_scenario.ownership_scopes.empty());
      kv_scenario.ownership_scopes[0].identity += "|tampered=1";
    } else if (tamper == 3) {
      kv_scenario.identity += "|tampered=1";
    } else if (tamper == 4) {
      cpu_plan->prefill->identity += "|tampered=1";
    } else if (tamper == 5) {
      plan.plan_identity += "|tampered=1";
    } else {
      const size_t scenario_base =
          static_cast<size_t>(LlmScenario::KvOnly) * rows;
      LlmPrefillKvSequenceRangeTemplate& descriptor =
          cpu_plan->workers[0].prefill_sequences[scenario_base];
      ASSERT_GT(descriptor.owned_token_count, 0u);
      if (tamper == 6) {
        ++descriptor.v_owned.offset_bytes;
      } else {
        ++descriptor.layer_index;
      }
    }

    const size_t map_calls_before = state.map_calls;
    LlmExecutionResources resources;
    const LlmResourcePreparationResult result =
        prepare_llm_execution_resources(plan, resources);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);
    EXPECT_EQ(state.map_calls, map_calls_before);
    EXPECT_FALSE(resources.valid);
  }
}

TEST_F(LlmMemoryExecutorTest,
       PrefillBoundaryCorruptionFailsExcludedPostValidationIntegration) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan(prefill_geometry(), 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared =
      prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
      plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  PrefillCorruptionContext context;
  ScopedExecutorTimer timer_scope;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  const LlmExecutorResult result = execute_llm_scenario(
      plan, scenario, resources, *timer,
      {prefill_corrupting_kernel, &context});
  EXPECT_TRUE(context.corrupted.load(std::memory_order_acquire));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code,
            LlmExecutorReason::PREFILL_POST_VALIDATION_FAILED);
  EXPECT_TRUE(result.kernel_succeeded);
  EXPECT_TRUE(result.checksum_evaluated);
  EXPECT_TRUE(result.checksum_valid);
  EXPECT_TRUE(result.post_validation_evaluated);
  EXPECT_FALSE(result.post_validation_valid);
}

TEST_F(LlmMemoryExecutorTest,
       PagedPreparationMaterializesPhysicalPoolsTableAndDistinctDescriptors) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan(paged_geometry(5, 2), 4);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu_plan.paged.has_value());
  const LlmPagedCpuExecutionPlan& paged = *cpu_plan.paged;

  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared =
      prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  ASSERT_TRUE(resources.valid);
  EXPECT_EQ(resources.layer_descriptors, nullptr);
  EXPECT_EQ(resources.sequence_descriptors, nullptr);
  ASSERT_NE(resources.paged_layer_descriptors, nullptr);
  ASSERT_NE(resources.paged_assignment_descriptors, nullptr);
  EXPECT_EQ(resources.block_table, paged.block_table());
  EXPECT_EQ(resources.block_table_entries,
            paged.layout.block_table_entries);
  EXPECT_EQ(resources.paged_block_reference_count,
            paged.layout.total_physical_blocks);
  EXPECT_EQ(resources.initialization.weight_bytes,
            plan.geometry.active_weight_bytes_per_work_unit);
  EXPECT_EQ(resources.initialization.k_bytes,
            paged.layout.memory.k_physical_bytes);
  EXPECT_EQ(resources.initialization.v_bytes,
            paged.layout.memory.v_physical_bytes);
  EXPECT_EQ(resources.initialization.block_table_logical_bytes,
            paged.block_table_logical_bytes);
  EXPECT_EQ(resources.initialization.block_table_mapping_bytes,
            paged.block_table_mapping_bytes);
  EXPECT_TRUE(resources.initialization.block_table_read_only);
  EXPECT_EQ(resources.initialization.k_layout_padding_bytes,
            paged.layout.memory.k_layout_padding_bytes);
  EXPECT_EQ(resources.initialization.v_layout_padding_bytes,
            paged.layout.memory.v_layout_padding_bytes);
  EXPECT_EQ(resources.initialization.non_empty_k_spans,
            paged.layout.total_physical_blocks);
  EXPECT_EQ(resources.initialization.non_empty_v_spans,
            paged.layout.total_physical_blocks);

  const auto* const weight =
      static_cast<const uint8_t*>(resources.buffers.weight.get());
  const auto* const k =
      static_cast<const uint8_t*>(resources.buffers.k.get());
  const auto* const v =
      static_cast<const uint8_t*>(resources.buffers.v.get());
  const size_t layer_pool_bytes =
      paged.layout.physical_blocks_per_layer * paged.layout.block_bytes;
  for (size_t layer = 0; layer < paged.layout.layer_count; ++layer) {
    for (uint32_t physical_id = 0;
         physical_id < paged.layout.physical_blocks_per_layer;
         ++physical_id) {
      const size_t block_offset =
          layer * layer_pool_bytes +
          static_cast<size_t>(physical_id) * paged.layout.block_bytes;
      for (size_t byte = 0; byte < paged.layout.block_bytes; ++byte) {
        EXPECT_EQ(k[block_offset + byte],
                  expected_paged_buffer_byte(plan.k_buffer_seed, layer,
                                             physical_id, byte));
        EXPECT_EQ(v[block_offset + byte],
                  expected_paged_buffer_byte(plan.v_buffer_seed, layer,
                                             physical_id, byte));
      }
    }
  }

  size_t zero_block_assignments = 0;
  for (size_t worker_index = 0;
       worker_index < cpu_plan.effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = cpu_plan.workers[worker_index];
    const LlmPagedLayerDescriptor* const layers =
        resources.worker_paged_layers(worker_index);
    const LlmPagedKvAssignmentDescriptor* const assignments =
        resources.worker_paged_assignments(worker_index);
    ASSERT_NE(layers, nullptr);
    ASSERT_NE(assignments, nullptr);
    EXPECT_EQ(resources.worker_layers(worker_index), nullptr);
    EXPECT_EQ(resources.worker_sequences(worker_index), nullptr);
    for (size_t layer = 0; layer < paged.layout.layer_count; ++layer) {
      const LlmLayerRangeTemplate& source = worker.layers[layer];
      EXPECT_EQ(layers[layer].weight_ptr,
                source.weight.span_bytes == 0
                    ? nullptr
                    : weight + source.weight.offset_bytes);
      EXPECT_EQ(layers[layer].weight_bytes, source.weight.span_bytes);
      EXPECT_EQ(layers[layer].first_assignment_index,
                source.first_sequence_index);
      EXPECT_EQ(layers[layer].assignment_count, source.sequence_count);
      EXPECT_EQ(layers[layer].layer_index, source.layer_index);
      EXPECT_EQ(layers[layer].reserved_zero, 0u);
    }
    for (size_t index = 0;
         index < cpu_plan.sequence_descriptors_per_worker; ++index) {
      const LlmPagedKvAssignmentTemplate& source =
          worker.paged_assignments[index];
      const LlmPagedKvAssignmentDescriptor& descriptor = assignments[index];
      EXPECT_EQ(descriptor.block_table_row,
                resources.block_table +
                    source.batch_sequence_index *
                        paged.layout.blocks_per_sequence);
      EXPECT_EQ(descriptor.k_layer_pool,
                k + source.layer_index * layer_pool_bytes);
      EXPECT_EQ(descriptor.v_layer_pool,
                v + source.layer_index * layer_pool_bytes);
      EXPECT_EQ(descriptor.first_logical_block,
                source.first_logical_block);
      EXPECT_EQ(descriptor.owned_block_count, source.block_count);
      EXPECT_EQ(descriptor.blocks_per_sequence,
                paged.layout.blocks_per_sequence);
      EXPECT_EQ(descriptor.block_bytes, paged.layout.block_bytes);
      EXPECT_EQ(descriptor.last_block_valid_bytes,
                paged.layout.last_block_valid_bytes);
      EXPECT_EQ(descriptor.decode_append_offset,
                paged.layout.decode_append_offset_in_last_block);
      EXPECT_EQ(descriptor.append_record_bytes,
                paged.layout.k_or_v_record_bytes_per_layer);
      EXPECT_EQ(descriptor.layer_index, source.layer_index);
      EXPECT_EQ(descriptor.batch_sequence_index,
                source.batch_sequence_index);
      zero_block_assignments += source.block_count == 0 ? 1 : 0;
    }
  }
  EXPECT_GT(zero_block_assignments, 0u);
}

TEST_F(LlmMemoryExecutorTest,
       PagedProductionDispatchPassesAllScenariosAndExactByteTails) {
  for (size_t tail_bytes : {31u, 32u, 33u}) {
    SCOPED_TRACE(::testing::Message() << "tail bytes " << tail_bytes);
    const LlmMemoryWorkPlan plan = build_executor_ready_plan(
        paged_geometry(5, 4, tail_bytes, 2, 2), 3);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_EQ(plan.geometry.last_block_valid_bytes, tail_bytes);
    LlmExecutionResources resources;
    const LlmResourcePreparationResult prepared =
        prepare_llm_execution_resources(plan, resources);
    ASSERT_TRUE(prepared.valid) << prepared.reason_code;

    for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                                 LlmScenario::KvOnly,
                                 LlmScenario::Mixed,
                                 LlmScenario::WeightsOnly}) {
      const LlmScenarioWorkPlan scenario_plan =
          build_llm_scenario_work_plan(plan, scenario, 2, true);
      ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
      ScopedExecutorTimer timer_scope;
      auto timer = HighResTimer::create();
      ASSERT_TRUE(timer.has_value());
      const LlmExecutorResult result = execute_llm_scenario(
          plan, scenario_plan, resources, *timer);
      EXPECT_TRUE(result.valid)
          << llm_scenario_to_string(scenario) << ": " << result.reason_code;
      EXPECT_EQ(result.reason_code, LlmExecutorReason::VALID);
      EXPECT_TRUE(result.checksum_evaluated);
      EXPECT_TRUE(result.checksum_valid);
      EXPECT_TRUE(result.post_validation_evaluated);
      EXPECT_TRUE(result.post_validation_valid);
    }
  }
}

TEST_F(LlmMemoryExecutorTest,
       PagedExecutorRejectsTamperedMetadataAndAccountedBytePlans) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan(paged_geometry(5, 4), 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared =
      prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  const LlmScenarioWorkPlan canonical = build_llm_scenario_work_plan(
      plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(canonical.valid) << canonical.reason_code;

  const std::array<size_t LlmScenarioWorkPlan::*, 6> fields = {
      &LlmScenarioWorkPlan::layout_metadata_lookup_count_per_work_unit,
      &LlmScenarioWorkPlan::layout_metadata_read_bytes_per_work_unit,
      &LlmScenarioWorkPlan::accounted_bytes_per_work_unit,
      &LlmScenarioWorkPlan::layout_metadata_lookup_count,
      &LlmScenarioWorkPlan::layout_metadata_read_bytes,
      &LlmScenarioWorkPlan::task_accounted_bytes,
  };
  for (size_t LlmScenarioWorkPlan::*field : fields) {
    LlmScenarioWorkPlan tampered = canonical;
    ASSERT_GT(tampered.*field, 0u);
    --(tampered.*field);

    const LlmExpectedChecksumResult result =
        calculate_llm_expected_checksums(plan, tampered, resources);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code,
              LlmExecutorReason::SCENARIO_PLAN_MISMATCH);
  }
}

TEST_F(LlmMemoryExecutorTest,
       PagedAppendAndPaddingCorruptionFailExcludedPostValidation) {
  for (PagedCorruption corruption : {PagedCorruption::Append,
                                     PagedCorruption::Padding}) {
    SCOPED_TRACE(corruption == PagedCorruption::Append ? "append"
                                                       : "padding");
    const LlmMemoryWorkPlan plan =
        build_executor_ready_plan(paged_geometry(5, 4), 1);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_LT(plan.geometry.last_block_valid_bytes,
              plan.geometry.kv_block_bytes);
    LlmExecutionResources resources;
    const LlmResourcePreparationResult prepared =
        prepare_llm_execution_resources(plan, resources);
    ASSERT_TRUE(prepared.valid) << prepared.reason_code;
    const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
        plan, LlmScenario::KvOnly, 2, true);
    ASSERT_TRUE(scenario.valid) << scenario.reason_code;
    PagedCorruptionContext context{corruption, false};
    ScopedExecutorTimer timer_scope;
    auto timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());
    const LlmExecutorResult result = execute_llm_scenario(
        plan, scenario, resources, *timer,
        {paged_corrupting_kernel, &context});
    EXPECT_TRUE(context.corrupted);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code,
              LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
    EXPECT_TRUE(result.checksum_evaluated);
    EXPECT_TRUE(result.checksum_valid);
    EXPECT_TRUE(result.post_validation_evaluated);
    EXPECT_FALSE(result.post_validation_valid);
  }
}

TEST_F(LlmMemoryExecutorTest,
       PagedWeightsOnlyRejectsUnexpectedValidKvWrite) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan(paged_geometry(5, 4), 1);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared =
      prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
      plan, LlmScenario::WeightsOnly, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  PagedCorruptionContext context{PagedCorruption::Append, false};
  ScopedExecutorTimer timer_scope;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  const LlmExecutorResult result = execute_llm_scenario(
      plan, scenario, resources, *timer,
      {paged_corrupting_kernel, &context});

  EXPECT_TRUE(context.corrupted);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code,
            LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_TRUE(result.checksum_evaluated);
  EXPECT_TRUE(result.checksum_valid);
  EXPECT_TRUE(result.post_validation_evaluated);
  EXPECT_FALSE(result.post_validation_valid);
}

TEST_F(LlmMemoryExecutorTest, StructuralValidationRejectsDivergentVMappingBeforeMmap) {
  LlmMemoryWorkPlan plan = build_executor_ready_plan({64, 1, 1, 1, 19, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_GT(plan.geometry.v_mapping_bytes, 1u);
  --plan.geometry.v_mapping_bytes;
  --plan.geometry.total_data_mapping_bytes;

  LlmBufferSet buffers;
  const LlmBufferAllocationResult result = allocate_llm_buffers(plan, buffers);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_FALSE(buffers.complete());
}

TEST_F(LlmMemoryExecutorTest,
       PagedStructuralValidationRejectsShortReferenceGeometryBeforeMmap) {
  LlmMemoryWorkPlan plan =
      build_executor_ready_plan(paged_geometry(5, 4), 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmCpuExecutionPlan* const cpu = get_llm_cpu_execution_plan(plan);
  ASSERT_NE(cpu, nullptr);
  ASSERT_TRUE(cpu->paged.has_value());
  ASSERT_GT(cpu->paged->layout.total_physical_blocks, 1u);
  --cpu->paged->layout.total_physical_blocks;
  const size_t map_calls_before_preparation = state.map_calls;

  LlmExecutionResources resources;
  const LlmResourcePreparationResult result =
      prepare_llm_execution_resources(plan, resources);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);
  EXPECT_EQ(state.map_calls, map_calls_before_preparation);
  EXPECT_FALSE(resources.valid);
}

TEST_F(LlmMemoryExecutorTest,
       BackendVariantMismatchIsRejectedBeforeResourceAllocation) {
  LlmMemoryWorkPlan plan =
      build_executor_ready_plan({64, 1, 1, 1, 19, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  plan.backend_execution_plan = LlmMetalExecutionPlan{};

  const LlmExecutorAuxiliaryEstimate auxiliary =
      calculate_llm_executor_auxiliary_estimate(plan);
  EXPECT_FALSE(auxiliary.valid);
  EXPECT_EQ(auxiliary.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);

  LlmBufferSet buffers;
  const LlmBufferAllocationResult allocation =
      allocate_llm_buffers(plan, buffers);
  EXPECT_FALSE(allocation.valid);
  EXPECT_EQ(allocation.reason_code, LlmExecutorReason::INVALID_WORK_PLAN);
  EXPECT_FALSE(buffers.complete());
  EXPECT_EQ(state.map_calls, 0u);
}

TEST_F(LlmMemoryExecutorTest, ExpectedChecksumsMatchIndependentAllScenarioMultiStepOracle) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({257, 2, 4, 2, 7, 1, 3, 2}, 3);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared = prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;

  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const LlmScenarioWorkPlan scenario_plan = build_llm_scenario_work_plan(plan, scenario, 2, true);
    ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
    const LlmExpectedChecksumResult actual = calculate_llm_expected_checksums(plan, scenario_plan, resources);
    ASSERT_TRUE(actual.valid) << actual.reason_code;
    const std::vector<LlmWorkerChecksum> expected = scalar_expected(plan, scenario_plan, resources);
    ASSERT_EQ(actual.workers.size(), expected.size());
    for (size_t worker = 0; worker < expected.size(); ++worker) {
      SCOPED_TRACE(::testing::Message() << llm_scenario_to_string(scenario) << " worker " << worker);
      expect_worker_equal(actual.workers[worker], expected[worker]);
    }
    const LlmRunChecksum folded = fold_llm_worker_checksums(expected.data(), expected.size());
    EXPECT_EQ(actual.run_checksum.state_a, folded.state_a);
    EXPECT_EQ(actual.run_checksum.state_b, folded.state_b);
  }
}

TEST_F(LlmMemoryExecutorTest, RunFoldRetainsEmptyComponentsInCanonicalOrderGolden) {
  LlmWorkerChecksum worker;
  worker.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
  worker.k = initial_llm_read_checksum(LlmChecksumComponent::K);
  worker.v = initial_llm_read_checksum(LlmChecksumComponent::V);
  std::array<uint8_t, 19> span{};
  for (size_t index = 0; index < span.size(); ++index) {
    span[index] = static_cast<uint8_t>(index);
  }
  absorb(worker.weight, span.data(), span.size());
  const LlmRunChecksum folded = fold_llm_worker_checksums(&worker, 1);
  EXPECT_EQ(folded.state_a, 0xBCA46801BE6585DBULL);
  EXPECT_EQ(folded.state_b, 0xEAAC493C97E06CF7ULL);
}

TEST_F(LlmMemoryExecutorTest, FakeKernelExecutorHonorsInvocationTimingQosAndChecksumBoundary) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({257, 2, 4, 2, 7, 1, 3, 2}, 3);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  ASSERT_TRUE(prepare_llm_execution_resources(plan, resources).valid);
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(plan, scenario, resources);
  ASSERT_TRUE(expected.valid) << expected.reason_code;

  ScopedExecutorTimer timer_calls;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  FakeKernelContext context;
  context.expected = &expected.workers;
  LlmExecutorTestControl control;
  control.set_worker_qos = fake_worker_qos;
  control.observe_event = observe_executor_event;
  control.event_context = &context;
  const LlmExecutorResult result =
      execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &context}, &control);

  ASSERT_TRUE(result.valid) << result.reason_code;
  const size_t effective_workers =
      cpu_execution_plan(plan).effective_workers;
  EXPECT_EQ(result.reason_code, LlmExecutorReason::VALID);
  EXPECT_DOUBLE_EQ(result.elapsed_seconds, 100.0 / 1e9);
  EXPECT_EQ(result.requested_workers, effective_workers);
  EXPECT_EQ(result.created_workers, effective_workers);
  EXPECT_EQ(result.completed_workers, effective_workers);
  EXPECT_EQ(result.qos_successful_workers, 1u);
  EXPECT_EQ(result.qos_failed_workers, effective_workers - 1);
  EXPECT_TRUE(result.timer_started);
  EXPECT_TRUE(result.timer_stopped);
  EXPECT_TRUE(result.kernel_succeeded);
  EXPECT_TRUE(result.checksum_evaluated);
  EXPECT_TRUE(result.checksum_valid);
  EXPECT_EQ(context.calls.load(std::memory_order_relaxed),
            effective_workers);

  ASSERT_EQ(context.invocations.size(), effective_workers);
  for (const LlmKernelInvocation& invocation : context.invocations) {
    EXPECT_EQ(invocation.layers, resources.worker_layers(invocation.worker_index));
    EXPECT_EQ(invocation.sequences, resources.worker_sequences(invocation.worker_index));
    EXPECT_EQ(invocation.layer_count, plan.geometry.layer_count);
    EXPECT_EQ(invocation.work_unit_count, scenario.work_units);
    EXPECT_EQ(invocation.scenario_flags, 3u);
    EXPECT_EQ(invocation.scenario_seed, scenario.scenario_seed);
  }
  const size_t timer_started = first_event(context.events, LlmExecutorEvent::TimerStarted);
  const size_t first_kernel = first_event(context.events, LlmExecutorEvent::KernelStarted);
  const size_t last_kernel = last_event(context.events, LlmExecutorEvent::KernelCompleted);
  const size_t timer_stopped = first_event(context.events, LlmExecutorEvent::TimerStopped);
  const size_t validation = first_event(context.events, LlmExecutorEvent::ChecksumValidationStarted);
  EXPECT_LT(timer_started, first_kernel);
  EXPECT_LT(last_kernel, timer_stopped);
  EXPECT_LT(timer_stopped, validation);
}

TEST_F(LlmMemoryExecutorTest, ExecutorRejectsComponentMismatchKernelFailureAndPartialStartup) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({128, 1, 1, 1, 16, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  ASSERT_TRUE(prepare_llm_execution_resources(plan, resources).valid);
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 1, true);
  ASSERT_TRUE(scenario.valid);
  const LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(plan, scenario, resources);
  ASSERT_TRUE(expected.valid);

  ScopedExecutorTimer timer_calls;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  FakeKernelContext mismatch;
  mismatch.expected = &expected.workers;
  mismatch.corrupt_worker = 0;
  LlmExecutorResult result = execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &mismatch});
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_TRUE(result.checksum_evaluated);
  EXPECT_FALSE(result.checksum_valid);
  EXPECT_TRUE(result.post_validation_evaluated);
  EXPECT_TRUE(result.post_validation_valid);
  EXPECT_TRUE(result.timer_stopped);

  FakeKernelContext kernel_failure;
  kernel_failure.expected = &expected.workers;
  kernel_failure.fail_worker = 0;
  result = execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &kernel_failure});
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_FALSE(result.kernel_succeeded);
  EXPECT_TRUE(result.timer_stopped);

  FakeKernelContext startup_failure;
  startup_failure.expected = &expected.workers;
  LlmExecutorTestControl control;
  control.fail_before_worker_index = 1;
  control.observe_event = observe_executor_event;
  control.event_context = &startup_failure;
  result = execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &startup_failure}, &control);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::WORKER_STARTUP_FAILED);
  EXPECT_TRUE(result.worker_startup_failed);
  EXPECT_FALSE(result.timer_started);
  EXPECT_FALSE(result.timer_stopped);
  EXPECT_EQ(startup_failure.calls.load(std::memory_order_relaxed), 0u);
  EXPECT_NE(first_event(startup_failure.events, LlmExecutorEvent::WorkerCancelled), std::numeric_limits<size_t>::max());
}

TEST_F(LlmMemoryExecutorTest, ExecutorRejectsZeroElapsedTimeAndContainsKernelExceptions) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({128, 1, 1, 1, 16, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  ASSERT_TRUE(prepare_llm_execution_resources(plan, resources).valid);
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 1, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(plan, scenario, resources);
  ASSERT_TRUE(expected.valid) << expected.reason_code;

  {
    ScopedStationaryExecutorTimer timer_calls;
    auto timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());
    FakeKernelContext context;
    context.expected = &expected.workers;
    const LlmExecutorResult result = execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &context});
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_ELAPSED_TIME);
    EXPECT_TRUE(result.kernel_succeeded);
    EXPECT_TRUE(result.timer_started);
    EXPECT_TRUE(result.timer_stopped);
    EXPECT_DOUBLE_EQ(result.elapsed_seconds, 0.0);
    EXPECT_FALSE(result.checksum_evaluated);
  }

  {
    ScopedExecutorTimer timer_calls;
    auto timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());
    const LlmExecutorResult result =
        execute_llm_scenario(plan, scenario, resources, *timer, {throwing_kernel, nullptr});
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::KERNEL_FAILED);
    EXPECT_FALSE(result.kernel_succeeded);
    EXPECT_TRUE(result.timer_stopped);
  }
}

TEST_F(LlmMemoryExecutorTest, ExecutorCancelsOrFinalizesWorkersWhenTimerHooksThrow) {
  const LlmMemoryWorkPlan plan = build_executor_ready_plan({128, 1, 1, 1, 16, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmExecutionResources resources;
  ASSERT_TRUE(prepare_llm_execution_resources(plan, resources).valid);
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 1, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(plan, scenario, resources);
  ASSERT_TRUE(expected.valid) << expected.reason_code;
  const size_t effective_workers =
      cpu_execution_plan(plan).effective_workers;

  {
    ScopedThrowingStartTimer timer_calls;
    auto timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());
    FakeKernelContext context;
    context.expected = &expected.workers;
    LlmExecutorTestControl control;
    control.observe_event = observe_executor_event;
    control.event_context = &context;
    const LlmExecutorResult result =
        execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &context}, &control);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_ELAPSED_TIME);
    EXPECT_EQ(result.created_workers, effective_workers);
    EXPECT_EQ(result.completed_workers, 0u);
    EXPECT_FALSE(result.worker_startup_failed);
    EXPECT_FALSE(result.timer_started);
    EXPECT_FALSE(result.timer_stopped);
    EXPECT_EQ(context.calls.load(std::memory_order_relaxed), 0u);
    EXPECT_NE(first_event(context.events, LlmExecutorEvent::WorkerCancelled), std::numeric_limits<size_t>::max());
  }

  {
    ScopedThrowingStopTimer timer_calls;
    auto timer = HighResTimer::create();
    ASSERT_TRUE(timer.has_value());
    FakeKernelContext context;
    context.expected = &expected.workers;
    const LlmExecutorResult result = execute_llm_scenario(plan, scenario, resources, *timer, {fake_kernel, &context});
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_ELAPSED_TIME);
    EXPECT_EQ(result.created_workers, effective_workers);
    EXPECT_EQ(result.completed_workers, effective_workers);
    EXPECT_TRUE(result.kernel_succeeded);
    EXPECT_TRUE(result.timer_started);
    EXPECT_FALSE(result.timer_stopped);
    EXPECT_EQ(context.calls.load(std::memory_order_relaxed),
              effective_workers);
  }
}

TEST_F(LlmMemoryExecutorTest, CpuAdapterOwnsCpuLifecycleAcceptanceAndPreservesTaggedEvidence) {
  const LlmMemoryWorkPlan plan =
      build_executor_ready_plan({128, 1, 1, 1, 16, 1, 2, 1}, 2);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  const LlmScenarioWorkPlan scenario =
      build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 3, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  LlmRunnerTaskContext context;
  context.kind = LlmRunnerTaskKind::Measurement;
  context.purpose = "measurement";
  context.scenario = scenario.scenario;
  context.loop_index = 4;
  context.order_position = 2;

  const auto successful_cpu_evidence = [&]() {
    LlmExecutorResult execution;
    execution.valid = true;
    execution.reason_code = LlmExecutorReason::VALID;
    execution.elapsed_seconds = 0.25;
    execution.requested_workers = cpu_plan.effective_workers;
    execution.created_workers = cpu_plan.effective_workers;
    execution.completed_workers = cpu_plan.effective_workers;
    execution.qos_successful_workers = cpu_plan.effective_workers;
    execution.kernel_succeeded = true;
    execution.timer_started = true;
    execution.timer_stopped = true;
    execution.checksum_evaluated = true;
    execution.checksum_valid = true;
    execution.post_validation_evaluated = true;
    execution.post_validation_valid = true;
    execution.expected_checksums.resize(cpu_plan.effective_workers);
    execution.actual_checksums = execution.expected_checksums;
    execution.expected_run_checksum = {11, 22};
    execution.actual_run_checksum = execution.expected_run_checksum;
    return execution;
  };

  LlmTaskExecutionResult result = adapt_llm_cpu_executor_result(
      plan, scenario, context, successful_cpu_evidence());
  EXPECT_EQ(result.status, LlmTaskExecutionStatus::Complete);
  EXPECT_EQ(result.reason_code, LlmBackendReason::VALID);
  EXPECT_EQ(result.identity.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(result.identity.scenario, LlmScenario::Mixed);
  EXPECT_EQ(result.identity.loop_index, 4u);
  EXPECT_EQ(result.identity.order_position, 2u);
  EXPECT_EQ(result.identity.model_plan_identity, plan.plan_identity);
  EXPECT_EQ(result.identity.scenario_plan_identity, scenario.plan_identity);
  EXPECT_TRUE(result.timing.evaluated);
  EXPECT_TRUE(result.timing.valid);
  EXPECT_DOUBLE_EQ(result.timing.elapsed_seconds, 0.25);
  EXPECT_EQ(result.completion.completed_work_units, scenario.work_units);
  EXPECT_EQ(result.completion.completed_effective_model_payload_bytes,
            scenario.effective_model_payload_bytes);
  EXPECT_EQ(result.completion.completed_layout_metadata_lookup_count,
            scenario.layout_metadata_lookup_count);
  EXPECT_EQ(result.completion.completed_layout_metadata_read_bytes,
            scenario.layout_metadata_read_bytes);
  EXPECT_EQ(result.completion.completed_task_accounted_bytes,
            scenario.task_accounted_bytes);
  EXPECT_TRUE(result.validation.evaluated);
  EXPECT_TRUE(result.validation.valid);
  const LlmExecutorResult* retained = get_llm_cpu_task_evidence(result);
  ASSERT_NE(retained, nullptr);
  EXPECT_EQ(retained->expected_checksums.size(), cpu_plan.effective_workers);
  EXPECT_EQ(retained->actual_checksums.size(), cpu_plan.effective_workers);

  LlmExecutorResult malformed = successful_cpu_evidence();
  malformed.actual_checksums.pop_back();
  result = adapt_llm_cpu_executor_result(plan, scenario, context,
                                         std::move(malformed));
  EXPECT_EQ(result.status, LlmTaskExecutionStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_RESOURCES);
  EXPECT_FALSE(result.validation.evaluated);

  LlmExecutorResult checksum_mismatch = successful_cpu_evidence();
  checksum_mismatch.valid = false;
  checksum_mismatch.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
  checksum_mismatch.checksum_valid = false;
  result = adapt_llm_cpu_executor_result(
      plan, scenario, context, std::move(checksum_mismatch));
  EXPECT_EQ(result.status, LlmTaskExecutionStatus::Invalid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_TRUE(result.validation.evaluated);
  EXPECT_FALSE(result.validation.valid);

  LlmExecutorResult invalid_timer = successful_cpu_evidence();
  invalid_timer.valid = false;
  invalid_timer.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
  invalid_timer.elapsed_seconds = 0.0;
  result = adapt_llm_cpu_executor_result(plan, scenario, context,
                                         std::move(invalid_timer));
  EXPECT_EQ(result.status, LlmTaskExecutionStatus::Invalid);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_TRUE(result.timing.evaluated);
  EXPECT_FALSE(result.timing.valid);
}
