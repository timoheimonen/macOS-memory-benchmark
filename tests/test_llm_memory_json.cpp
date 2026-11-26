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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "llm_memory/llm_cpu_backend.h"
#include "llm_memory/llm_json.h"
#include "llm_memory/llm_metal_backend.h"
#include "utils/numeric_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr uint64_t kAboveJsonExactInteger = 9007199254740993ULL;

const LlmCpuExecutionPlan& cpu_execution_plan(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) throw std::logic_error("expected CPU execution plan");
  return *cpu_plan;
}

LlmMemoryConfig explicit_config(size_t loop_count = 3) {
  LlmMemoryConfig config;
  config.weight_size_mb = 1;
  config.layer_count = 2;
  config.query_head_count = 4;
  config.kv_head_count = 2;
  config.head_dimension = 8;
  config.kv_element_bytes = 2;
  config.visible_context_tokens = 3;
  config.batch_size = 1;
  config.requested_workers = 2;
  config.available_workers = 2;
  config.iterations = 4;
  config.loop_count = loop_count;
  config.seed = std::numeric_limits<uint64_t>::max();
  config.user_specified_iterations = true;
  config.user_specified_seed = true;
  config.user_specified_workers = true;
  config.output_file = "--literal-output-name.json";
  config.argv = {"memory_benchmark", "--llm-memory", "--output", "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig paged_config(size_t loop_count = 1) {
  LlmMemoryConfig config = explicit_config(loop_count);
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  config.argv = {"memory_benchmark", "--llm-memory", "--kv-layout", "paged", "--kv-block-tokens", "2", "--output",
                 "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig prefill_config(size_t loop_count = 3) {
  LlmMemoryConfig config = explicit_config(loop_count);
  config.phase = LlmPhase::Prefill;
  config.visible_context_tokens = 0;
  config.prompt_tokens = 5;
  config.attention_query_tile_tokens = 2;
  config.user_specified_phase = true;
  config.user_specified_context_tokens = false;
  config.user_specified_prompt_tokens = true;
  config.user_specified_attention_query_tile_tokens = true;
  config.argv = {"memory_benchmark", "--llm-memory", "--phase", "prefill", "--prompt-tokens", "5",
                 "--attention-query-tile-tokens", "2", "--output", "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig paged_prefill_config(size_t loop_count = 3) {
  LlmMemoryConfig config = prefill_config(loop_count);
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  config.argv = {"memory_benchmark", "--llm-memory", "--phase", "prefill", "--prompt-tokens", "5",
                 "--attention-query-tile-tokens", "2", "--kv-layout", "paged", "--kv-block-tokens", "2", "--output",
                 "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig make_metal_variant(LlmMemoryConfig config) {
  config.backend = LlmMemoryBackend::Metal;
  config.requested_workers = 0;
  config.available_workers = 0;
  config.user_specified_backend = true;
  config.user_specified_workers = false;
  return config;
}

LlmMemoryConfig explicit_metal_config() {
  LlmMemoryConfig config = make_metal_variant(explicit_config(1));
  config.weight_size_mb = 257;
  config.argv = {"memory_benchmark", "--llm-memory", "--llm-memory-backend", "metal", "--weight-size-mb", "257",
                 "--output", "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig explicit_metal_paged_config() {
  LlmMemoryConfig config = explicit_metal_config();
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  config.argv.insert(config.argv.end() - 2, {"--kv-layout", "paged", "--kv-block-tokens", "2"});
  return config;
}

LlmMemoryConfig explicit_metal_prefill_config() {
  LlmMemoryConfig config = make_metal_variant(prefill_config(1));
  config.argv.insert(config.argv.begin() + 2, {"--llm-memory-backend", "metal"});
  return config;
}

LlmMemoryConfig explicit_metal_paged_prefill_config() {
  LlmMemoryConfig config = make_metal_variant(paged_prefill_config(1));
  config.argv.insert(config.argv.begin() + 2, {"--llm-memory-backend", "metal"});
  return config;
}

LlmMemoryWorkPlanRequest plan_request(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request;
  request.geometry.active_weight_bytes = config.weight_size_mb * Constants::BYTES_PER_MB;
  request.geometry.layer_count = config.layer_count;
  request.geometry.query_head_count = config.query_head_count;
  request.geometry.kv_head_count = config.kv_head_count;
  request.geometry.head_dimension = config.head_dimension;
  request.geometry.kv_element_bytes = config.kv_element_bytes;
  request.geometry.visible_context_tokens = config.visible_context_tokens;
  request.geometry.prompt_tokens = config.prompt_tokens;
  request.geometry.attention_query_tile_tokens = config.attention_query_tile_tokens;
  request.geometry.batch_size = config.batch_size;
  request.geometry.kv_block_tokens = config.kv_block_tokens;
  request.geometry.phase = config.phase;
  request.geometry.kv_layout = config.kv_layout;
  request.requested_workers = config.requested_workers;
  request.available_workers = config.available_workers;
  request.available_memory_bytes = 8ULL * 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = 4096;
  request.base_seed = config.seed;
  return request;
}

LlmMemoryWorkPlan admitted_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = plan_request(config);
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(request);
  if (!draft.valid) return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  const auto executor = calculate_llm_executor_auxiliary_estimate(draft.auxiliary_preflight);
  const auto runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  const auto json_peak = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
  size_t runner_and_executor_orchestration = 0;
  if (!executor.valid || !runner.valid || !json_peak.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 runner_and_executor_orchestration) ||
      !NumericUtils::checked_add(runner_and_executor_orchestration, json_peak.total_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  return finalize_llm_memory_work_plan(std::move(draft), request.checksum_auxiliary_bytes,
                                       request.orchestration_auxiliary_bytes);
}

LlmMemoryWorkPlan admitted_metal_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = plan_request(config);
  request.backend = LlmMemoryBackend::Metal;
  request.requested_workers = 0;
  request.available_workers = 0;
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(request);
  if (!draft.valid) return finalize_llm_memory_work_plan(std::move(draft), 0, 0);

  const auto resource_request = [&](size_t additional_owned_bytes) {
    LlmMetalResourcePlanRequest metal_request;
    metal_request.geometry = draft.candidate.geometry;
    if (draft.candidate.kv_layout == LlmKvLayout::Paged) {
      const size_t tokens = draft.candidate.phase == LlmPhase::Prefill
                                ? draft.candidate.geometry.prefill->prompt_tokens
                                : draft.candidate.geometry.decode->visible_context_tokens;
      metal_request.paged_layout = build_llm_kv_layout_plan(
          {tokens, draft.candidate.geometry.kv_block_tokens, draft.candidate.geometry.layer_count,
           draft.candidate.geometry.batch_size, draft.candidate.geometry.k_or_v_record_bytes_per_layer});
    }
    metal_request.argument_buffer_encoded_length = 8192;
    metal_request.argument_buffer_alignment = 256;
    metal_request.max_buffer_length = 2 * Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
    metal_request.available_memory_bytes = request.available_memory_bytes;
    metal_request.host_mapping_granularity_bytes = request.mapping_granularity_bytes;
    metal_request.additional_owned_bytes = additional_owned_bytes;
    return metal_request;
  };

  LlmMetalExecutionPlan provisional =
      build_llm_metal_execution_plan(resource_request(draft.candidate.memory_budget.request.planner_storage_bytes));
  if (!provisional.valid || !attach_llm_metal_execution_plan(draft, std::move(provisional))) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }

  std::unique_ptr<LlmBackend> backend = create_llm_backend(LlmMemoryBackend::Metal);
  const auto backend_aux =
      backend ? backend->calculate_auxiliary_estimate(draft.auxiliary_preflight) : LlmBackendAuxiliaryEstimate{};
  const auto runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  const auto json_peak = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
  size_t checksum_aux = 0, runner_and_backend_orch = 0, orch_aux = 0, cmd_aux = 0, additional_owned = 0;
  if (!backend_aux.valid || !runner.valid || !json_peak.valid ||
      !NumericUtils::checked_add(backend_aux.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes, checksum_aux) ||
      !NumericUtils::checked_add(backend_aux.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 runner_and_backend_orch) ||
      !NumericUtils::checked_add(runner_and_backend_orch, json_peak.total_bytes, orch_aux) ||
      !NumericUtils::checked_add(checksum_aux, orch_aux, cmd_aux) ||
      !NumericUtils::checked_add(draft.candidate.memory_budget.request.planner_storage_bytes, cmd_aux,
                                 additional_owned)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  return finalize_llm_memory_work_plan(std::move(draft),
                                       build_llm_metal_execution_plan(resource_request(additional_owned)), checksum_aux,
                                       orch_aux);
}

std::string metal_pipeline_prefix(const LlmMemoryWorkPlan& plan) {
  return std::string("membenchmark.llm-metal.pipeline.") + (plan.phase == LlmPhase::Prefill ? "prefill-" : "decode-") +
         (plan.kv_layout == LlmKvLayout::Paged ? "paged." : "contiguous.");
}

LlmMetalTaskEvidence complete_metal_task_evidence(const LlmMemoryWorkPlan& plan) {
  LlmMetalTaskEvidence task;
  task.timed_pipeline_available = true;
  task.pipeline_label = metal_pipeline_prefix(plan) + "mixed";
  task.pipeline_thread_execution_width = 32;
  task.pipeline_max_total_threads_per_threadgroup = 1024;
  task.grid_plan_available = true;
  task.grid_plan.valid = true;
  task.grid_plan.reason_code = LlmMetalPlanReason::VALID;
  task.grid_plan.owner_count = 2;
  task.grid_plan.threads_per_threadgroup = 128;
  task.grid_plan.actual_threadgroups = 2;
  task.grid_plan.owner_ordinals_per_threadgroup = 1;
  task.grid_plan.vector_iterations_per_lane_per_visit = 17;
  task.grid_plan.work_units = 1;
  if (const auto limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::Mixed); limits.valid) {
    task.grid_plan.paged_semantic_lookups = limits.layout_metadata_lookup_count_per_work_unit;
  }
  if (plan.phase == LlmPhase::Prefill && plan.kv_layout == LlmKvLayout::Contiguous) {
    const auto scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 1, true);
    size_t serial_visits = 0;
    if (scenario.valid && calculate_llm_metal_prefill_serial_range_visits_per_lane(plan, scenario, serial_visits)) {
      task.grid_plan.serial_range_visits_per_lane = serial_visits;
    }
  }
  task.grid_plan.identity = "metal-grid-v1-test";
  if (plan.kv_layout == LlmKvLayout::Paged) {
    task.grid_plan.minimum_threadgroup_accounted_bytes = 1024;
    task.grid_plan.maximum_threadgroup_accounted_bytes = 2048;
    task.grid_plan.threadgroup_accounted_imbalance_bytes = 1024;
    task.grid_plan.threadgroup_accounted_bytes = {1024, 2048};
  }
  if (plan.phase == LlmPhase::Prefill && plan.kv_layout == LlmKvLayout::Paged) {
    task.grid_plan.owner_count = 6;
    task.grid_plan.owner_ordinals_per_threadgroup = 3;
    task.grid_plan.vector_iterations_per_lane_per_visit = 256;
    task.grid_plan.minimum_threadgroup_accounted_bytes = 525372;
    task.grid_plan.maximum_threadgroup_accounted_bytes = 525372;
    task.grid_plan.threadgroup_accounted_imbalance_bytes = 0;
    task.grid_plan.threadgroup_accounted_bytes = {525372, 525372};
    task.grid_plan.identity = "metal-grid-v1-test;prefill-paged-cyclic-owner-cost-v1";
  }
  task.timing_evaluated = true;
  task.timing_valid = true;
  task.gpu_start_seconds = 10.0;
  task.gpu_end_seconds = 10.0025;
  task.gpu_elapsed_seconds = 0.0025;
  task.host_timing_evaluated = true;
  task.host_submit_to_completion_seconds = 0.003;
  task.host_wait_seconds = 0.00275;
  task.queue_delay_available = true;
  task.queue_delay_seconds = 0.0005;
  task.reset_command_buffer_count = 1;
  task.timed_command_buffer_count = 1;
  task.post_validation_command_buffer_count = 1;
  task.timed_compute_encoder_count = 1;
  task.timed_workload_dispatch_count = 1;
  task.reset_command_status = "completed";
  task.timed_command_status = "completed";
  task.post_validation_command_status = "completed";
  task.checksum_evaluated = true;
  task.checksum_valid = true;
  task.checksum_algorithm_version = plan.component_identities.checksum_pattern_version;
  task.expected_checksum = {{1, 2}, {3, 4}, {5, 6}};
  task.actual_checksum = task.expected_checksum;
  task.kv_write_validation_evaluated = true;
  task.kv_write_validation_valid = true;
  task.padding_canary_applicable = plan.kv_layout == LlmKvLayout::Paged;
  task.padding_canary_evaluated = task.padding_canary_applicable;
  task.padding_canary_valid = task.padding_canary_applicable;
  task.post_validation_evaluated = true;
  task.post_validation_valid = true;
  return task;
}

LlmBackendEvidence complete_metal_backend_evidence(const LlmMemoryWorkPlan& plan) {
  LlmBackendEvidence backend;
  backend.backend = LlmMemoryBackend::Metal;
  backend.initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  backend.plan_resolution = backend.initialization;
  backend.preparation = backend.initialization;
  backend.release = backend.initialization;

  LlmMetalBackendEvidence metal;
  metal.capability.device_name = "Test Metal Device";
  metal.capability.registry_id = kAboveJsonExactInteger;
  metal.capability.has_unified_memory = true;
  metal.capability.required_apple7_family_supported = true;
  metal.capability.argument_buffers_tier2_supported = true;
  metal.capability.supported_families = {"apple7", "apple8"};
  metal.capability.max_buffer_length = 2 * Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  metal.capability.recommended_max_working_set_size = 6ULL * 1024ULL * Constants::BYTES_PER_MB;
  metal.capability.kernel_revision =
      plan.component_identities.msl_revision.value_or("llm-metal-decode-prefill-contiguous-paged-msl23-v5");
  metal.capability.kernel_source_sha256 = canonical_llm_metal_kernel_source_sha256();
  metal.capability.argument_buffer_encoded_length = 8192;
  metal.capability.argument_buffer_alignment = 256;
  metal.capability.layout_probe_evaluated = true;
  metal.capability.layout_probe_valid = true;
  metal.capability.layout_probe_resource_count = 7;
  metal.capability.foundation_pipelines = {{"membenchmark.llm-metal.foundation.fill", 32, 1024}};
  const std::string prefix = metal_pipeline_prefix(plan);
  metal.workload_pipelines = {
      {prefix + "weights-only", 32, 1024}, {prefix + "kv-only", 32, 1024}, {prefix + "mixed", 32, 1024}};
  metal.timed_results_available = true;

  if (const auto* execution = get_llm_metal_execution_plan(plan); execution != nullptr) {
    metal.resources.allocation_attempted = true;
    metal.resources.allocation_completed = true;
    metal.resources.initialization_completed = true;
    if (execution->resources.paged_layout.has_value()) {
      const LlmKvBlockTable table = materialize_llm_kv_block_table(
          *execution->resources.paged_layout, derive_llm_kv_permutation_seed(plan.base_seed),
          Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
      if (table.valid) {
        metal.resources.table_upload_completed = true;
        metal.resources.table_validation_completed = true;
        metal.resources.table_permutation = table.permutation;
      }
      metal.resources.layout_padding_bytes = execution->resources.paged_layout->memory.k_layout_padding_bytes +
                                             execution->resources.paged_layout->memory.v_layout_padding_bytes;
    }
    metal.resources.post_validation_completed = true;
    metal.resources.resources_published = true;
    metal.resources.persistent_resource_length_bytes = execution->resources.persistent_resource_length_bytes;
    metal.resources.committed_resource_bytes = execution->resources.persistent_resource_length_bytes + 4096;
    metal.resources.resource_rounding_bytes = 4096;
    metal.resources.transient_peak_bytes = execution->resources.transient_peak_bytes;
    metal.resources.known_owned_peak_bytes = execution->resources.known_owned_peak_bytes + 4096;
    metal.resources.admitted_budget_bytes = execution->resources.admitted_budget_bytes;
  }
  backend.backend_evidence = std::move(metal);
  return backend;
}

LlmMemoryResult metal_result_with_measurement_and_calibration(const LlmMemoryWorkPlan& plan) {
  LlmMemoryResult result;
  result.initialized = true;
  result.status = LlmRunStatus::Complete;
  result.reason_code = LlmBackendReason::VALID;
  result.results_complete = true;
  result.run_accepted = true;
  const auto scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 1, true);
  // Controlled serialization witnesses, not independently validated GPU results.
  LlmCanonicalScenarioPlan canonical;
  canonical.plan = scenario;
  canonical.expected.available = true;
  canonical.expected.reason_code = "valid";
  canonical.expected.metal_run = {{1, 2}, {3, 4}, {5, 6}};
  result.scenario_plans.push_back(std::move(canonical));
  result.frozen_plan_handles[static_cast<size_t>(LlmScenario::Mixed)] = 0;

  LlmMetalTaskEvidence task = complete_metal_task_evidence(plan);
  task.cold_checks = required_llm_cold_checks(plan, LlmScenario::Mixed);
  for (size_t slot = 0; slot < task.cold_checks.size(); ++slot) resolve_llm_cold_check(task.cold_checks, slot, true);

  LlmMeasurementState measurement;
  measurement.plan_handle = 0;
  measurement.scenario = LlmScenario::Mixed;
  measurement.status = LlmMeasurementStatus::Measured;
  measurement.reason_code = LlmBackendReason::VALID;
  measurement.attempted = true;
  measurement.execution_evidence_available = true;
  measurement.execution.status = LlmTaskExecutionStatus::Complete;
  measurement.execution.reason_code = LlmBackendReason::VALID;
  measurement.execution.timing = {true, true, 0.0025};
  measurement.execution.validation = {true, true};
  measurement.execution.backend_evidence = static_cast<const LlmMetalRuntimeEvidence&>(task);
  measurement.execution.completion = {scenario.work_units, scenario.effective_model_payload_bytes,
                                      scenario.layout_metadata_lookup_count, scenario.layout_metadata_read_bytes,
                                      scenario.task_accounted_bytes};
  result.measurements.push_back(std::move(measurement));

  task.queue_delay_available = false;
  LlmCalibrationAttempt attempt;
  attempt.plan_handle = 0;
  attempt.scenario = LlmScenario::Mixed;
  attempt.reason_code = LlmBackendReason::VALID;
  attempt.terminal = true;
  attempt.valid = true;
  attempt.execution.available = true;
  attempt.execution.valid = true;
  attempt.execution.reason_code = LlmBackendReason::VALID;
  attempt.execution.status = LlmTaskExecutionStatus::Complete;
  attempt.execution.elapsed_seconds = 0.0025;
  attempt.execution.timing_evaluated = true;
  attempt.execution.timing_valid = true;
  attempt.execution.validation_evaluated = true;
  attempt.execution.validation_valid = true;
  attempt.execution.metal_evidence_available = true;
  attempt.execution.metal = static_cast<const LlmMetalRuntimeEvidence&>(task);
  attempt.execution.completion = result.measurements.front().execution.completion;
  result.calibration_attempts[static_cast<size_t>(LlmScenario::Mixed)].push_back(std::move(attempt));
  prepare_llm_result_snapshot(result);
  return result;
}

LlmReadChecksumComponent checksum_component(uint64_t offset) {
  return {std::numeric_limits<uint64_t>::max() - offset, kAboveJsonExactInteger + offset,
          kAboveJsonExactInteger + 100 + offset, 17 + offset};
}

LlmExecutorResult successful_execution(const LlmMemoryWorkPlan& plan, double elapsed_seconds = 0.150) {
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  LlmExecutorResult execution;
  execution.valid = true;
  execution.reason_code = LlmExecutorReason::VALID;
  execution.elapsed_seconds = elapsed_seconds;
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
  for (size_t i = 0; i < cpu_plan.effective_workers; ++i) {
    execution.expected_checksums[i] = {checksum_component(i * 3), checksum_component(i * 3 + 1),
                                       checksum_component(i * 3 + 2)};
  }
  execution.actual_checksums = execution.expected_checksums;
  execution.expected_run_checksum = {std::numeric_limits<uint64_t>::max(), kAboveJsonExactInteger};
  execution.actual_run_checksum = execution.expected_run_checksum;
  return execution;
}

LlmResourcePreparationResult preparation_for(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  LlmResourcePreparationResult preparation;
  preparation.valid = true;
  preparation.reason_code = LlmExecutorReason::VALID;
  preparation.auxiliary = calculate_llm_executor_auxiliary_estimate(plan);
  preparation.memory_budget = plan.memory_budget;
  preparation.initialization.complete = true;
  preparation.initialization.weight_bytes = plan.geometry.active_weight_bytes_per_work_unit;
  preparation.initialization.k_bytes = plan.geometry.k_mapping_bytes;
  preparation.initialization.v_bytes = plan.geometry.v_mapping_bytes;
  preparation.initialization.total_bytes = plan.geometry.total_data_mapping_bytes;
  preparation.initialization.non_empty_weight_spans = cpu_plan.total_layer_descriptors;
  preparation.initialization.non_empty_k_spans = cpu_plan.total_sequence_descriptors;
  preparation.initialization.non_empty_v_spans = cpu_plan.total_sequence_descriptors;
  preparation.initialization.k_layout_padding_bytes = plan.geometry.k_layout_padding_bytes;
  preparation.initialization.v_layout_padding_bytes = plan.geometry.v_layout_padding_bytes;
  if (cpu_plan.paged.has_value()) {
    preparation.initialization.block_table_logical_bytes = cpu_plan.paged->block_table_logical_bytes;
    preparation.initialization.block_table_mapping_bytes = cpu_plan.paged->block_table_mapping_bytes;
    preparation.initialization.block_table_read_only = cpu_plan.paged->block_table_read_only;
  }
  return preparation;
}

class FakeLlmBackend final : public LlmBackend {
 public:
  using CpuTaskExecutor = std::function<LlmExecutorResult(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                                                          const LlmRunnerTaskContext&)>;

  explicit FakeLlmBackend(CpuTaskExecutor executor = {}) : executor_(std::move(executor)) {
    evidence_.backend = LlmMemoryBackend::Cpu;
  }

  LlmMemoryBackend kind() const noexcept override { return LlmMemoryBackend::Cpu; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(const LlmMemoryWorkPlan& model) const noexcept override {
    const auto cpu = calculate_llm_executor_auxiliary_estimate(model);
    return {cpu.valid, cpu.reason_code, cpu.checksum_auxiliary_bytes, cpu.orchestration_auxiliary_bytes,
            cpu.total_auxiliary_bytes, cpu};
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig&) noexcept override {
    evidence_ = LlmBackendEvidence{};
    evidence_.backend = LlmMemoryBackend::Cpu;
    evidence_.initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan&) noexcept override {
    evidence_.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model) noexcept override {
    evidence_.backend_evidence = LlmCpuBackendEvidence{preparation_for(model)};
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmExpectedChecksumResult expected_cpu_checksum(const LlmMemoryWorkPlan& model,
                                                  const LlmScenarioWorkPlan&) const noexcept override {
    const auto controlled = successful_execution(model);
    return {true, LlmExecutorReason::VALID, controlled.expected_checksums, controlled.expected_run_checksum};
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) override {
    LlmExecutorResult execution =
        executor_ ? executor_(model_plan, scenario_plan, context) : successful_execution(model_plan);
    execution.cold_checks = required_llm_cold_checks(model_plan, scenario_plan.scenario);
    if (execution.post_validation_evaluated) {
      for (size_t slot = 0; slot < execution.cold_checks.size(); ++slot) {
        resolve_llm_cold_check(execution.cold_checks, slot, execution.post_validation_valid,
                               execution.post_validation_valid
                                   ? std::string_view("valid")
                                   : canonicalize_llm_result_reason_code(execution.reason_code));
      }
    }
    execution.kv_write_validation_applicable = scenario_plan.scenario != LlmScenario::WeightsOnly;
    execution.kv_write_validation_evaluated =
        execution.kv_write_validation_applicable && execution.post_validation_evaluated;
    execution.kv_write_validation_valid = execution.kv_write_validation_evaluated && execution.post_validation_valid;
    return adapt_llm_cpu_executor_result(model_plan, scenario_plan, context, std::move(execution));
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.release;
  }

 private:
  CpuTaskExecutor executor_;
  LlmBackendEvidence evidence_;
};

LlmMemoryResult complete_result(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  FakeLlmBackend backend;
  LlmMemoryResult result;
  EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_SUCCESS);
  return result;
}

LlmResultMetadata fixed_metadata(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  LlmResultMetadata metadata;
  metadata.timestamp = "2000-01-01T00:00:00Z";
  metadata.processor_name = "Test Apple Silicon";
  metadata.macos_version = "test-macos";
  metadata.performance_core_count = 4;
  metadata.efficiency_core_count = 2;
  metadata.logical_core_count = 6;
  metadata.page_size_bytes = 16384;
  metadata.l1_data_cache_bytes = 128 * 1024;
  metadata.l2_data_cache_bytes = 0;
  metadata.available_memory_bytes = 4ULL * 1024ULL * Constants::BYTES_PER_MB;
  metadata.available_memory_source = "test-provider";
  metadata.json_peak_estimate = calculate_llm_json_peak_estimate(config, plan);
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_start.low_power_mode_available = true;
  metadata.environment_start.low_power_mode_enabled = false;
  metadata.environment_start.physical_memory_bytes = std::numeric_limits<uint64_t>::max();
  metadata.environment_end = metadata.environment_start;
  return metadata;
}

OrderedJson build_cpu_doc(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan, const LlmMemoryResult& result) {
  return build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
}

OrderedJson run_cpu_doc(const LlmMemoryConfig& config, FakeLlmBackend::CpuTaskExecutor executor, int expected_exit,
                        const LlmRunnerHooks& hooks = {}) {
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  EXPECT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(std::move(executor));
  LlmMemoryResult result;
  EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks), expected_exit);
  return build_cpu_doc(config, plan, result);
}

std::vector<std::string> json_string_array(const OrderedJson& array) {
  std::vector<std::string> output;
  for (const OrderedJson& value : array) output.push_back(value.get<std::string>());
  return output;
}

const OrderedJson& referenced_plan(const OrderedJson& document, const OrderedJson& record) {
  const auto& reference = record.at("plan_ref");
  if (!reference.is_number_unsigned() && !reference.is_number_integer())
    throw std::invalid_argument("expected integer plan_ref");
  const auto index = reference.get<int64_t>();
  if (index < 0 || static_cast<size_t>(index) >= document.at("resolved_plan").at("scenario_plans").size())
    throw std::out_of_range("plan_ref");
  return document.at("resolved_plan").at("scenario_plans").at(static_cast<size_t>(index));
}

const OrderedJson& named_check(const OrderedJson& execution, const char* kind) {
  for (const auto& check : execution.at("validation").at("checks")) {
    if (check.at("kind") == kind) return check;
  }
  throw std::invalid_argument("missing named check");
}

void expect_named_check(const OrderedJson& execution, const char* kind, bool applicable, std::optional<bool> evaluated,
                        std::optional<bool> valid, const char* reason = nullptr) {
  const auto& check = named_check(execution, kind);
  EXPECT_EQ(check["applicable"].get<bool>(), applicable) << kind;
  if (evaluated.has_value()) EXPECT_EQ(check["evaluated"].get<bool>(), *evaluated) << kind;
  else EXPECT_TRUE(check["evaluated"].is_null()) << kind;
  if (valid.has_value()) EXPECT_EQ(check["valid"].get<bool>(), *valid) << kind;
  else EXPECT_TRUE(check["valid"].is_null()) << kind;
  if (reason != nullptr) EXPECT_EQ(check["reason_code"], reason) << kind;
}

void expect_exact_keys(const OrderedJson& object, std::initializer_list<const char*> keys) {
  ASSERT_TRUE(object.is_object());
  EXPECT_EQ(object.size(), keys.size());
  for (const char* key : keys) EXPECT_TRUE(object.contains(key)) << key;
}

void expect_null_fields(const OrderedJson& object, std::initializer_list<const char*> keys) {
  for (const char* key : keys) EXPECT_TRUE(object.at(key).is_null()) << key;
}

void expect_string_fields(const OrderedJson& object, std::initializer_list<const char*> keys) {
  for (const char* key : keys) EXPECT_TRUE(object.at(key).is_string()) << key;
}

void expect_string_values(const OrderedJson& object,
                          std::initializer_list<std::pair<const char*, std::string_view>> pairs) {
  for (const auto& [key, value] : pairs) EXPECT_EQ(object.at(key), value) << key;
}

void expect_not_evaluated_checksum(const OrderedJson& checksum, const char* reason, bool check_workers_null = false) {
  EXPECT_EQ(checksum["status"], "not_evaluated");
  EXPECT_EQ(checksum["reason_code"], reason);
  EXPECT_TRUE(checksum["checksum_valid"].is_null());
  EXPECT_TRUE(checksum["actual_run_checksum"].is_null());
  if (check_workers_null) EXPECT_TRUE(checksum["actual_worker_checksums"].is_null());
}

}  // namespace

TEST(LlmMemoryJsonTest, MetalDecodeAndPrefillPagedDocumentsPublishCompleteEvidence) {
  {
    SCOPED_TRACE("metal_decode_contiguous");
    const LlmMemoryConfig config = explicit_metal_config();
    const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmMetalExecutionPlan* const metal_plan = get_llm_metal_execution_plan(plan);
    ASSERT_NE(metal_plan, nullptr);
    ASSERT_TRUE(metal_plan->valid) << metal_plan->reason_code;
    ASSERT_EQ(metal_plan->resources.weight_segments.segment_count, 2U);

    const OrderedJson doc = build_llm_memory_json(config, plan, complete_metal_backend_evidence(plan),
                                                  fixed_metadata(config, plan),
                                                  metal_result_with_measurement_and_calibration(plan));

    expect_string_values(doc, {{"backend", "metal"}, {"methodology_version", "llm-memory-v2-metal-decode-contiguous"}});
    EXPECT_EQ(doc["configuration"]["resolved_sources"]["backend"], "explicit");
    expect_null_fields(doc["configuration"], {"requested_workers", "available_workers", "worker_source"});
    EXPECT_TRUE(doc["configuration"]["resolved_sources"]["workers"].is_null());
    EXPECT_TRUE(doc["resolved_plan"]["model_work_plan"]["effective_workers"].is_null());

    const OrderedJson& resources = doc["resolved_plan"]["resources"]["metal"];
    ASSERT_TRUE(resources.is_object());
    EXPECT_EQ(resources["segment_capacity_bytes"], "268435456");
    EXPECT_EQ(resources["weight_segments"]["segment_count"], 2U);
    EXPECT_EQ(resources["weight_segments"]["total_length_bytes"], "269484032");
    EXPECT_EQ(resources["weight_segments"]["segment_lengths_bytes"], (OrderedJson::array({"268435456", "1048576"})));
    EXPECT_EQ(resources["argument_buffer"]["encoded_length_bytes"], "8192");
    EXPECT_TRUE(resources["table_segments"].is_null());

    EXPECT_TRUE(doc["backend_evidence"]["cpu"].is_null());
    const OrderedJson& metal_backend = doc["backend_evidence"]["metal"];
    ASSERT_TRUE(metal_backend.is_object());
    EXPECT_FALSE(metal_backend["workers_applicable"].get<bool>());
    EXPECT_FALSE(metal_backend["worker_qos_applicable"].get<bool>());
    EXPECT_EQ(metal_backend["lifecycle"]["initialization"]["status"], "ready");
    expect_string_values(metal_backend["capability"],
                         {{"device_name", "Test Metal Device"}, {"registry_id_uint64_decimal", "9007199254740993"}});
    EXPECT_EQ(metal_backend["workload_pipelines"].size(), 3U);
    EXPECT_EQ(metal_backend["workload_pipelines"][2]["label"],
              "membenchmark.llm-metal.pipeline.decode-contiguous.mixed");
    EXPECT_TRUE(metal_backend["timed_results_available"].get<bool>());

    const OrderedJson& methodology = doc["resolved_plan"]["methodology"];
    expect_string_values(
        methodology,
        {{"timing_policy", "gpu-start-to-gpu-end-per-command-buffer-task"},
         {"cache_policy", "metal-private-and-shared-cacheable-no-explicit-flush-between-scenarios"}});
    EXPECT_EQ(methodology["maximum_work_units_per_measurement"], Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH);
    const std::string transient = std::to_string(metal_plan->resources.transient_peak_bytes);
    const std::string peak = std::to_string(metal_plan->resources.known_owned_peak_bytes + 4096);
    const std::string budget = std::to_string(metal_plan->resources.admitted_budget_bytes);
    expect_string_values(doc["memory_budget"], {{"resource_rounding_bytes", "4096"},
                                                {"transient_peak_bytes", transient},
                                                {"known_owned_peak_bytes", peak},
                                                {"admitted_budget_bytes", budget}});

    ASSERT_EQ(doc["measurements"].size(), 1U);
    const OrderedJson& execution = doc["measurements"][0]["execution"];
    expect_null_fields(execution, {"requested_workers", "qos_successful_workers", "qos_failed_workers"});
    EXPECT_DOUBLE_EQ(doc["measurements"][0]["elapsed_seconds"].get<double>(), 0.0025);
    const OrderedJson& task = execution["metal"];
    EXPECT_EQ(task["pipeline"]["label"], "membenchmark.llm-metal.pipeline.decode-contiguous.mixed");
    EXPECT_EQ(task["grid"]["actual_threadgroups"], 2U);
    expect_null_fields(task["grid"], {"cost_unit", "minimum_threadgroup_accounted_bytes",
                                      "maximum_threadgroup_accounted_bytes", "threadgroup_accounted_imbalance_bytes"});
    EXPECT_TRUE(task["grid"]["threadgroup_accounted_bytes"].empty());
    EXPECT_TRUE(task["timing"]["evaluated"].get<bool>());
    EXPECT_TRUE(task["timing"]["valid"].get<bool>());
    EXPECT_DOUBLE_EQ(task["timing"]["queue_delay_seconds"].get<double>(), 0.0005);
    expect_string_values(task["commands"], {{"reset_status", "completed"},
                                            {"timed_status", "completed"},
                                            {"post_validation_status", "completed"}});
    EXPECT_EQ(task["commands"]["timed_workload_dispatches"], 1U);

    const OrderedJson& checksum = doc["measurements"][0]["checksum"];
    EXPECT_EQ(checksum["status"], "valid");
    EXPECT_EQ(checksum["actual_run_checksum"]["weight"]["a_uint32_decimal"], "1");
    EXPECT_EQ(checksum["actual_run_checksum"]["v"]["b_uint32_decimal"], "6");

    const OrderedJson& compact = doc["calibration"]["attempts"]["mixed"][0]["execution"];
    EXPECT_TRUE(compact["requested_workers"].is_null());
    EXPECT_EQ(doc["resolved_plan"]["scenario_plans"][0]["expected_checksum"]["expected_run_checksum"]["k"]
                 ["b_uint32_decimal"],
              "4");
    EXPECT_EQ(compact["metal"]["commands"]["timed_status"], "completed");
    EXPECT_TRUE(compact["metal"]["timing"]["queue_delay_seconds"].is_null());
    expect_named_check(execution, "kv-append-final", true, true, true);
    expect_named_check(execution, "kv-padding-canary", false, std::nullopt, std::nullopt);
  }
  {
    SCOPED_TRACE("metal_prefill_paged");
    const LlmMemoryConfig config = explicit_metal_paged_prefill_config();
    const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmMetalExecutionPlan* const metal_plan = get_llm_metal_execution_plan(plan);
    ASSERT_NE(metal_plan, nullptr);
    ASSERT_TRUE(metal_plan->valid) << metal_plan->reason_code;
    ASSERT_TRUE(metal_plan->resources.paged_layout.has_value());
    ASSERT_TRUE(metal_plan->resources.table_segments.has_value());

    const OrderedJson doc = build_llm_memory_json(config, plan, complete_metal_backend_evidence(plan),
                                                  fixed_metadata(config, plan),
                                                  metal_result_with_measurement_and_calibration(plan));

    expect_string_values(doc, {{"backend", "metal"},
                               {"phase", "prefill"},
                               {"kv_layout", "paged"},
                               {"methodology_version", "llm-memory-v2-metal-prefill-paged"}});
    EXPECT_EQ(doc["resolved_plan"]["work_unit_kind"], "prefill_operation");
    EXPECT_EQ(doc["resolved_plan"]["methodology"]["maximum_serial_range_visits_per_lane_per_task"],
              Constants::LLM_METAL_MAX_SERIAL_RANGE_VISITS_PER_LANE_PER_TASK);

    expect_string_values(
        doc["resolved_plan"]["component_identities"],
        {{"logical_profile_version", Constants::LLM_PREFILL_LOGICAL_PROFILE_VERSION},
         {"kv_layout_version", Constants::LLM_PAGED_KV_LAYOUT_VERSION},
         {"permutation_version", Constants::LLM_KV_BLOCK_PERMUTATION_VERSION},
         {"backend_executor_version", LlmMetalPrefillPagedVersion::EXECUTOR},
         {"resource_abi_version", Constants::LLM_METAL_ARGUMENT_BUFFER_ABI_VERSION},
         {"schedule_version", LlmMetalPrefillPagedVersion::SCHEDULE},
         {"timer_policy_version", LlmMetalPrefillPagedVersion::TIMER},
         {"buffer_pattern_version", LlmMetalPrefillPagedVersion::BUFFER_PATTERN},
         {"write_pattern_version", LlmMetalPrefillPagedVersion::WRITE_PATTERN},
         {"checksum_pattern_version", LlmMetalPrefillPagedVersion::CHECKSUM},
         {"msl_revision", "llm-metal-decode-prefill-contiguous-paged-msl23-v5"},
         {"msl_source_sha256", "2fb49f1f8045aa01a74329d7b9d6b04134502d845e9f70c53b8f6fd0baf8fdd2"}});

    EXPECT_EQ(doc["configuration"]["prompt_tokens"], 5U);
    EXPECT_EQ(doc["configuration"]["attention_query_tile_tokens"], 2U);
    expect_string_values(doc["configuration"]["resolved_sources"],
                         {{"kv_layout", "explicit"}, {"kv_block_tokens", "explicit"}});
    const OrderedJson& geometry = doc["resolved_plan"]["geometry"];
    EXPECT_TRUE(geometry["decode"].is_null());
    ASSERT_TRUE(geometry["prefill"].is_object());
    EXPECT_EQ(geometry["prefill"]["prompt_tokens"], 5U);
    EXPECT_EQ(geometry["prefill"]["attention_query_tile_tokens"], 2U);
    expect_string_values(geometry["prefill"],
                         {{"tile_count", "3"}, {"attention_prefix_token_visits_per_sequence", "11"}});
    expect_string_values(geometry, {{"kv_read_bytes_per_work_unit", "1408"}, {"kv_write_bytes_per_work_unit", "640"}});

    const OrderedJson& layout = doc["resolved_plan"]["layout"];
    EXPECT_EQ(layout["kv_block_tokens"], 2U);
    EXPECT_TRUE(layout["decode_append_offset_in_last_block"].is_null());
    expect_string_values(layout, {{"kv_layout", "paged"},
                                  {"blocks_per_sequence", "3"},
                                  {"physical_blocks_per_layer", "3"},
                                  {"total_physical_blocks", "6"},
                                  {"block_bytes", "64"},
                                  {"last_block_tokens", "1"},
                                  {"last_block_valid_bytes", "32"},
                                  {"block_table_entries", "3"},
                                  {"block_table_bytes", "12"},
                                  {"permutation_algorithm_version", Constants::LLM_KV_BLOCK_PERMUTATION_VERSION},
                                  {"permutation_entry_count", "3"}});
    EXPECT_FALSE(layout["layout_geometry_identity"].is_null());
    EXPECT_FALSE(layout["layout_identity"].is_null());
    EXPECT_FALSE(layout["permutation_identity"].is_null());
    ASSERT_TRUE(layout["permutation_sha256"].is_string());
    EXPECT_EQ(layout["permutation_sha256"].get<std::string>().size(), 64U);

    const OrderedJson& resources = doc["resolved_plan"]["resources"]["metal"];
    EXPECT_EQ(resources["weight_segments"]["segment_count"], 1U);
    expect_string_values(resources["k_segments"],
                         {{"element_bytes", "64"}, {"element_count", "6"}, {"total_length_bytes", "384"}});
    EXPECT_EQ(resources["v_segments"]["total_length_bytes"], "384");
    expect_string_values(resources["table_segments"],
                         {{"element_bytes", "4"}, {"element_count", "3"}, {"total_length_bytes", "12"}});
    EXPECT_EQ(resources["staging_buffer_length_bytes"], "12");

    const OrderedJson& metal = doc["backend_evidence"]["metal"];
    expect_string_values(
        metal["capability"],
        {{"kernel_revision", "llm-metal-decode-prefill-contiguous-paged-msl23-v5"},
         {"kernel_source_sha256", "2fb49f1f8045aa01a74329d7b9d6b04134502d845e9f70c53b8f6fd0baf8fdd2"}});
    EXPECT_EQ(metal["workload_pipelines"][2]["label"], "membenchmark.llm-metal.pipeline.prefill-paged.mixed");
    const OrderedJson& metal_resources = metal["resources"];
    EXPECT_TRUE(metal_resources["table_upload_completed"].get<bool>());
    EXPECT_TRUE(metal_resources["table_validation_completed"].get<bool>());
    EXPECT_EQ(metal_resources["layout_padding_bytes"], "128");
    ASSERT_TRUE(metal_resources["table_permutation"].is_object());
    EXPECT_EQ(metal_resources["table_permutation"]["entry_count"], "3");
    EXPECT_EQ(metal_resources["table_permutation"]["sha256"], layout["permutation_sha256"]);
    EXPECT_EQ(metal_resources["table_permutation"]["identity"], layout["permutation_identity"]);

    ASSERT_EQ(doc["measurements"].size(), 1U);
    const OrderedJson& measurement = doc["measurements"][0];
    const OrderedJson& scenario_plan = referenced_plan(doc, measurement);
    EXPECT_EQ(measurement["scenario"], "mixed");
    EXPECT_EQ(scenario_plan["work_units"], 1U);
    EXPECT_EQ(measurement["completed_work_units"], 1U);
    expect_string_values(scenario_plan, {{"work_unit_kind", "prefill_operation"},
                                         {"kv_write_kind", "full_prompt_population"},
                                         {"weight_read_bytes_per_work_unit", "1048576"},
                                         {"kv_read_bytes_per_work_unit", "1408"},
                                         {"kv_write_bytes_per_work_unit", "640"},
                                         {"effective_model_payload_bytes_per_work_unit", "1050624"},
                                         {"layout_metadata_lookup_count_per_work_unit", "30"},
                                         {"layout_metadata_read_bytes_per_work_unit", "120"},
                                         {"accounted_bytes_per_work_unit", "1050744"}});

    const OrderedJson& task = measurement["execution"]["metal"];
    EXPECT_EQ(task["pipeline"]["label"], "membenchmark.llm-metal.pipeline.prefill-paged.mixed");
    expect_exact_keys(task["grid"],
                      {"valid", "reason_code", "owner_count", "threads_per_threadgroup", "actual_threadgroups",
                       "owner_ordinals_per_threadgroup", "vector_iterations_per_lane_per_visit", "work_units",
                       "paged_semantic_lookups", "serial_range_visits_per_lane", "cost_unit",
                       "minimum_threadgroup_accounted_bytes", "maximum_threadgroup_accounted_bytes",
                       "threadgroup_accounted_imbalance_bytes", "threadgroup_accounted_bytes", "identity"});
    EXPECT_EQ(task["grid"]["actual_threadgroups"], 2U);
    expect_string_values(task["grid"], {{"owner_count", "6"},
                                        {"owner_ordinals_per_threadgroup", "3"},
                                        {"paged_semantic_lookups", "30"},
                                        {"serial_range_visits_per_lane", "0"},
                                        {"cost_unit", "actual-threadgroup-cost"},
                                        {"minimum_threadgroup_accounted_bytes", "525372"},
                                        {"maximum_threadgroup_accounted_bytes", "525372"},
                                        {"threadgroup_accounted_imbalance_bytes", "0"}});
    EXPECT_EQ(task["grid"]["threadgroup_accounted_bytes"], (OrderedJson::array({"525372", "525372"})));
    EXPECT_EQ(doc["calibration"]["attempts"]["mixed"][0]["execution"]["checksum"]["actual_run_checksum"]["k"]
                 ["b_uint32_decimal"],
              "4");
    const std::string serialized = doc.dump();
    EXPECT_EQ(serialized.find("tokens/s"), std::string::npos);
    EXPECT_EQ(serialized.find("tokens_per_second"), std::string::npos);
    expect_named_check(measurement["execution"], "kv-prefill-final-samples", true, true, true);
    expect_named_check(measurement["execution"], "kv-padding-canary", true, true, true);
    EXPECT_EQ(measurement["checksum"]["actual_run_checksum"]["weight"]["a_uint32_decimal"], "1");
  }
}

TEST(LlmMemoryJsonTest, MetalMemoryBudgetUsesActualValuesOnlyAfterCompleteAllocationAndRejectsCommittedOverage) {
  const LlmMemoryConfig config = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = metal_result_with_measurement_and_calibration(plan);
  const LlmResultMetadata metadata = fixed_metadata(config, plan);

  LlmBackendEvidence partial_backend = complete_metal_backend_evidence(plan);
  auto* partial_metal = std::get_if<LlmMetalBackendEvidence>(&partial_backend.backend_evidence);
  ASSERT_NE(partial_metal, nullptr);
  partial_metal->resources.allocation_completed = false;
  partial_metal->resources.resource_rounding_bytes = 0;
  partial_metal->resources.transient_peak_bytes = 0;
  partial_metal->resources.known_owned_peak_bytes = 0;
  partial_metal->resources.admitted_budget_bytes = 0;
  const OrderedJson partial_doc = build_llm_memory_json(config, plan, partial_backend, metadata, result);
  EXPECT_EQ(partial_doc["memory_budget"]["known_owned_peak_bytes"],
            std::to_string(plan.memory_budget.request.required_total_bytes));
  EXPECT_EQ(partial_doc["memory_budget"]["admitted_budget_bytes"],
            std::to_string(plan.memory_budget.allowed_memory_bytes));
  EXPECT_TRUE(partial_doc["memory_budget"]["valid"].get<bool>());
  EXPECT_EQ(partial_doc["memory_budget"]["reason_code"], plan.memory_budget.reason_code);

  LlmBackendEvidence overage_backend = complete_metal_backend_evidence(plan);
  auto* overage_metal = std::get_if<LlmMetalBackendEvidence>(&overage_backend.backend_evidence);
  ASSERT_NE(overage_metal, nullptr);
  overage_metal->resources.known_owned_peak_bytes = overage_metal->resources.admitted_budget_bytes + 1;
  const OrderedJson overage_doc = build_llm_memory_json(config, plan, overage_backend, metadata, result);
  EXPECT_FALSE(overage_doc["memory_budget"]["valid"].get<bool>());
  EXPECT_EQ(overage_doc["memory_budget"]["reason_code"], LlmBackendReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_EQ(overage_doc["memory_budget"]["known_owned_peak_bytes"],
            std::to_string(overage_metal->resources.known_owned_peak_bytes));
}

TEST(LlmMemoryJsonTest, TerminalMetalSetupAndUnsupportedPagedDocuments) {
  {
    SCOPED_TRACE("unsupported_metal_paged");
    const LlmMemoryConfig config = explicit_metal_paged_config();
    const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;

    LlmBackendEvidence backend;
    backend.backend = LlmMemoryBackend::Metal;
    backend.initialization = {LlmBackendStatus::Unsupported, LlmBackendReason::APPLE7_FAMILY_REQUIRED};
    LlmMetalBackendEvidence metal;
    metal.capability.required_apple7_family_supported = false;
    backend.backend_evidence = std::move(metal);

    LlmMemoryResult result;
    result.initialized = true;
    result.status = LlmRunStatus::Unsupported;
    result.reason_code = LlmBackendReason::APPLE7_FAMILY_REQUIRED;
    const OrderedJson doc = build_llm_memory_json(config, plan, backend, fixed_metadata(config, plan), result);

    EXPECT_EQ(doc["status"], "unsupported");
    const OrderedJson& layout = doc["resolved_plan"]["layout"];
    EXPECT_EQ(layout["kv_block_tokens"], 2U);
    expect_string_values(layout, {{"kv_layout", "paged"}, {"blocks_per_sequence", "2"}, {"block_table_entries", "2"}});
    EXPECT_FALSE(layout["layout_geometry_identity"].is_null());
    expect_null_fields(layout, {"layout_identity", "permutation_seed_uint64_decimal", "permutation_sha256",
                                "permutation_identity"});
    EXPECT_TRUE(doc["backend_evidence"]["metal"]["resources"]["table_permutation"].is_null());
    EXPECT_TRUE(doc["measurements"].empty());
  }
  {
    SCOPED_TRACE("terminal_metal_setup");
    const auto config = explicit_metal_config();
    const auto plan = admitted_metal_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    for (bool unsupported : {false, true}) {
      SCOPED_TRACE(unsupported);
      auto backend = complete_metal_backend_evidence(plan);
      const auto* reason =
          unsupported ? LlmBackendReason::APPLE7_FAMILY_REQUIRED : LlmBackendReason::METAL_PIPELINE_CREATION_FAILED;
      const char* status = unsupported ? "unsupported" : "failed";
      const char* phase = unsupported ? "initialization" : "plan_resolution";
      const LlmBackendLifecycleResult terminal = {
          unsupported ? LlmBackendStatus::Unsupported : LlmBackendStatus::Failed, reason};
      if (unsupported) {
        backend.initialization = terminal;
        backend.plan_resolution = {};
      } else {
        backend.plan_resolution = terminal;
      }
      backend.preparation = {};
      backend.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
      auto* metal = std::get_if<LlmMetalBackendEvidence>(&backend.backend_evidence);
      ASSERT_NE(metal, nullptr);
      metal->timed_results_available = false;
      if (unsupported) metal->capability.required_apple7_family_supported = false;

      LlmMemoryResult result;
      result.initialized = true;
      result.status = unsupported ? LlmRunStatus::Unsupported : LlmRunStatus::Failed;
      result.reason_code = reason;
      const auto doc = build_llm_memory_json(config, plan, backend, fixed_metadata(config, plan), result);
      EXPECT_EQ(doc["schema_version"], 2);
      expect_string_values(doc, {{"mode", "llm_memory"}, {"backend", "metal"},
                                 {"status", status}, {"reason_code", reason}});
      EXPECT_FALSE(doc["results_complete"].get<bool>());
      EXPECT_FALSE(doc["run_accepted"].get<bool>());
      EXPECT_TRUE(doc["backend_evidence"]["cpu"].is_null());
      expect_string_values(doc["backend_evidence"]["metal"]["lifecycle"][phase],
                           {{"status", status}, {"reason_code", reason}});
      if (unsupported) {
        EXPECT_FALSE(doc["backend_evidence"]["metal"]["capability"]["required_apple7_family_supported"].get<bool>());
      }
    }
  }
}

TEST(LlmMemoryJsonTest, CompleteCpuDecodeDocumentSchemaAndDecimalIntegers) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete);
  const LlmResultMetadata metadata = fixed_metadata(config, plan);
  const OrderedJson doc = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);

  expect_exact_keys(doc, {"schema_version", "mode", "backend", "phase", "kv_layout", "methodology_version", "software",
                          "configuration", "resolved_plan", "backend_evidence", "memory_budget", "calibration",
                          "measurements", "aggregates", "status", "reason_code", "results_complete", "run_accepted",
                          "interpretation", "diagnostic", "interruption_requested", "scenario_order_balance_complete",
                          "seeds", "counters", "checkpoint_lifecycle", "loop_records", "environment",
                          "quality_warnings", "build_manifest"});

  EXPECT_TRUE(doc["resolved_plan"]["methodology"]["maximum_serial_range_visits_per_lane_per_task"].is_null());
  ASSERT_TRUE(doc["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(doc["backend_evidence"]["metal"].is_null());
  expect_null_fields(doc["backend_evidence"]["cpu"], {"prefill", "paged"});
  const OrderedJson& cpu_resources = doc["backend_evidence"]["cpu"]["resources"];
  EXPECT_TRUE(cpu_resources["mappings"]["block_table"].is_null());
  expect_null_fields(cpu_resources["initialization"],
                     {"block_table_logical_bytes", "block_table_page_rounded_bytes", "block_table_read_only"});
  EXPECT_TRUE(doc["resolved_plan"]["geometry"]["decode"].is_object());
  EXPECT_TRUE(doc["resolved_plan"]["geometry"]["prefill"].is_null());

  expect_exact_keys(doc["configuration"],
                    {"backend", "phase", "kv_layout", "kv_block_tokens", "weight_size_mb", "layer_count",
                     "query_head_count", "kv_head_count", "head_dimension", "kv_element_bytes",
                     "visible_context_tokens", "prompt_tokens", "attention_query_tile_tokens", "batch_size",
                     "requested_workers", "available_workers", "worker_source", "iterations", "work_policy",
                     "loop_count", "base_seed_uint64_decimal", "seed_source", "output_file", "argv",
                     "resolved_sources"});
  expect_exact_keys(doc["configuration"]["resolved_sources"],
                    {"backend", "phase", "kv_layout", "kv_block_tokens", "visible_context_tokens", "prompt_tokens",
                     "attention_query_tile_tokens", "workers", "iterations", "seed"});
  expect_exact_keys(doc["resolved_plan"]["geometry"],
                    {"valid", "reason_code", "phase", "work_unit_kind", "decode", "prefill", "attention_kind",
                     "active_weight_bytes_per_work_unit", "layer_count", "query_head_count", "kv_head_count",
                     "query_heads_per_kv_head", "head_dimension", "kv_element_bytes", "batch_size", "kv_vector_bytes",
                     "k_or_v_record_bytes_per_layer", "kv_record_bytes_per_layer", "kv_bytes_per_visible_token",
                     "k_or_v_sequence_visible_bytes", "k_mapping_bytes", "v_mapping_bytes", "kv_capacity_bytes",
                     "weight_read_bytes_per_work_unit", "kv_read_bytes_per_work_unit", "kv_write_bytes_per_work_unit",
                     "kv_only_effective_model_payload_bytes_per_work_unit",
                     "mixed_effective_model_payload_bytes_per_work_unit", "total_data_mapping_bytes",
                     "traffic_crossover_numerator", "traffic_crossover_denominator",
                     "traffic_crossover_context_tokens"});
  expect_exact_keys(cpu_resources["mappings"], {"policy", "full_size_physical_mappings", "weight", "k", "v",
                                                "block_table", "requested_data_bytes", "committed_data_bytes"});
  expect_exact_keys(cpu_resources["mappings"]["weight"], {"requested_bytes", "committed_bytes"});
  expect_exact_keys(cpu_resources["json_output_peak_estimate"],
                    {"enabled", "valid", "reason_code", "policy", "fixed_schema_bytes", "input_string_bytes",
                     "measurement_record_bytes", "worker_checksum_bytes", "total_bytes"});
  expect_exact_keys(
      cpu_resources["initialization"],
      {"complete", "pattern_version", "pre_touch_policy", "separate_reference_read_pass",
       "static_references_accumulated_during_initialization", "weight_bytes", "k_bytes", "v_bytes", "total_bytes",
       "non_empty_weight_spans", "non_empty_k_spans", "non_empty_v_spans", "block_table_logical_bytes",
       "block_table_page_rounded_bytes", "block_table_read_only", "k_layout_padding_bytes", "v_layout_padding_bytes"});
  expect_exact_keys(doc["seeds"],
                    {"base_seed_uint64_decimal", "source", "buffer_domain_seeds", "scenario_domain_seeds"});
  expect_exact_keys(doc["seeds"]["buffer_domain_seeds"],
                    {"weight_uint64_decimal", "k_uint64_decimal", "v_uint64_decimal"});
  expect_exact_keys(doc["seeds"]["scenario_domain_seeds"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(doc["resolved_plan"]["geometry"]["decode"], {"visible_context_tokens"});
  expect_exact_keys(doc["resolved_plan"]["layout"],
                    {"kv_layout", "kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer",
                     "total_physical_blocks", "block_bytes", "last_block_tokens", "last_block_valid_bytes",
                     "decode_append_offset_in_last_block", "block_table_entries", "block_table_bytes",
                     "layout_geometry_identity", "layout_identity", "permutation_domain_uint64_hex",
                     "permutation_seed_uint64_decimal", "permutation_algorithm_version", "permutation_entry_count",
                     "permutation_sha256", "permutation_identity"});
  expect_exact_keys(doc["resolved_plan"]["resources"],
                    {"weight_logical_bytes", "k_logical_bytes", "v_logical_bytes", "k_physical_length_bytes",
                     "v_physical_length_bytes", "k_layout_padding_bytes", "v_layout_padding_bytes",
                     "block_table_bytes"});
  expect_exact_keys(doc["environment"],
                    {"processor_name", "macos_version", "performance_core_count", "efficiency_core_count",
                     "logical_core_count", "page_size_bytes", "l1_data_cache_bytes", "l2_data_cache_bytes",
                     "available_memory_bytes", "available_memory_source", "main_thread_qos", "start", "end"});
  expect_exact_keys(doc["environment"]["main_thread_qos"], {"requested", "applied", "code"});
  expect_exact_keys(doc["environment"]["start"],
                    {"thermal_state", "low_power_mode_available", "low_power_mode_enabled", "physical_memory_bytes"});
  expect_exact_keys(
      doc["interpretation"],
      {"result_scope", "reported_rate", "backend", "phase", "work_unit_kind", "transformer_math_included",
       "framework_scheduler_and_dispatch_included", "compute_memory_overlap_included", "physical_dram_traffic_measured",
       "dram_residency", "cache_residency", "fixed_context_includes_current_token_slot", "kv_layout",
       "payload_semantics", "layout_metadata_timed_but_excluded_from_effective_model_payload_gb_s",
       "prefill_transformer_compute_or_ttft_prediction_included", "private_metal_storage_implies_separate_vram",
       "cross_backend_performance_distributions_combined", "traffic_classification_semantics",
       "full_size_working_set_reduces_but_does_not_prove_dram_residency", "comparability_requires"});

  EXPECT_EQ(doc["schema_version"], 2);
  expect_string_values(doc, {{"mode", "llm_memory"},
                             {"backend", "cpu"},
                             {"phase", "decode"},
                             {"kv_layout", "contiguous"},
                             {"methodology_version", Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION},
                             {"status", "complete"}});
  expect_string_values(doc["software"], {{"version", SOFTVERSION}, {"timestamp", metadata.timestamp}});
  EXPECT_TRUE(doc["results_complete"]);
  EXPECT_TRUE(doc["run_accepted"]);
  EXPECT_TRUE(doc["scenario_order_balance_complete"]);
  EXPECT_TRUE(doc["diagnostic"].is_null());

  expect_string_values(doc["configuration"], {{"output_file", "--literal-output-name.json"},
                                              {"work_policy", "explicit_fixed_work"},
                                              {"kv_element_bytes", "2"},
                                              {"base_seed_uint64_decimal", "18446744073709551615"}});
  EXPECT_EQ(doc["configuration"]["argv"], config.argv);
  EXPECT_EQ(doc["configuration"]["visible_context_tokens"], 3);
  expect_null_fields(doc["configuration"], {"prompt_tokens", "attention_query_tile_tokens", "kv_block_tokens"});
  expect_string_values(doc["configuration"]["resolved_sources"], {{"phase", "default"}, {"kv_layout", "default"}});
  expect_null_fields(doc["configuration"]["resolved_sources"],
                     {"kv_block_tokens", "prompt_tokens", "attention_query_tile_tokens"});

  const std::string active_weights = std::to_string(plan.geometry.active_weight_bytes_per_work_unit);
  const std::string kv_cap = std::to_string(plan.geometry.kv_capacity_bytes);
  expect_string_values(doc["resolved_plan"]["geometry"], {{"active_weight_bytes_per_work_unit", active_weights},
                                                          {"kv_capacity_bytes", kv_cap},
                                                          {"kv_element_bytes", "2"}});
  EXPECT_TRUE(doc["resolved_plan"]["geometry"]["active_weight_bytes_per_work_unit"].is_string());
  EXPECT_EQ(cpu_resources["model_ref"], "resolved_plan");
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["enabled"]);
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["valid"]);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["reason_code"], LlmJsonReason::VALID);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["total_bytes"],
            std::to_string(metadata.json_peak_estimate.total_bytes));
  EXPECT_EQ(cpu_resources["initialization"]["pattern_version"], Constants::LLM_BUFFER_PATTERN_VERSION);
  EXPECT_FALSE(cpu_resources["initialization"]["separate_reference_read_pass"]);
  EXPECT_EQ(doc["resolved_plan"]["plan_identity"], plan.plan_identity);
  EXPECT_EQ(doc["resolved_plan"]["scenario_plans"].size(), kLlmScenarioCount);
  EXPECT_EQ(doc["calibration"]["attempts"]["weights_only"].size(), 1u);
  EXPECT_EQ(doc["loop_records"].size(), 3u);
  EXPECT_EQ(doc["measurements"].size(), 9u);
  EXPECT_EQ(doc["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"]["sample_count"], 3u);
  EXPECT_EQ(doc["environment"]["start"]["physical_memory_bytes"], "18446744073709551615");
  EXPECT_EQ(doc["interpretation"]["reported_rate"], "synthetic_memory_work_units_per_second");
  EXPECT_FALSE(doc["interpretation"]["transformer_math_included"]);
  EXPECT_FALSE(doc["interpretation"]["physical_dram_traffic_measured"]);

  // Decimal-string and contiguous-null invariants on measurement, layout, budget, checksum, and counters.
  const OrderedJson& measurement = doc["measurements"][0];
  const OrderedJson& scenario = referenced_plan(doc, measurement);
  expect_string_values(scenario, {{"work_unit_kind", "decode_step"},
                                  {"kv_write_kind", "none"},
                                  {"layout_metadata_lookup_count_per_work_unit", "0"},
                                  {"layout_metadata_read_bytes_per_work_unit", "0"}});
  EXPECT_TRUE(scenario["work_units"].is_number_unsigned());
  EXPECT_TRUE(measurement["completed_work_units"].is_number_unsigned());
  expect_string_fields(measurement,
                       {"completed_effective_model_payload_bytes", "completed_layout_metadata_lookup_count",
                        "completed_layout_metadata_read_bytes", "completed_task_accounted_bytes"});
  expect_string_fields(scenario, {"weight_read_bytes_per_work_unit", "kv_read_bytes_per_work_unit",
                                  "kv_write_bytes_per_work_unit", "effective_model_payload_bytes",
                                  "layout_metadata_lookup_count", "layout_metadata_read_bytes",
                                  "task_accounted_bytes"});
  EXPECT_TRUE(measurement["synthetic_work_unit_latency_seconds"].is_number());
  EXPECT_TRUE(measurement["synthetic_memory_work_units_per_second"].is_number());
  EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_number());
  expect_null_fields(doc["resolved_plan"]["layout"],
                     {"kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer", "total_physical_blocks",
                      "block_bytes", "last_block_tokens", "last_block_valid_bytes",
                      "decode_append_offset_in_last_block", "block_table_entries", "block_table_bytes",
                      "layout_geometry_identity", "layout_identity", "permutation_domain_uint64_hex",
                      "permutation_seed_uint64_decimal", "permutation_algorithm_version", "permutation_entry_count",
                      "permutation_sha256", "permutation_identity"});
  EXPECT_TRUE(doc["resolved_plan"]["resources"]["weight_logical_bytes"].is_string());
  EXPECT_TRUE(doc["resolved_plan"]["resources"]["block_table_bytes"].is_null());
  expect_null_fields(doc["resolved_plan"]["component_identities"],
                     {"permutation_version", "msl_revision", "msl_source_sha256"});
  EXPECT_TRUE(doc["memory_budget"]["layout_transient_bytes"].is_null());
  expect_null_fields(doc["memory_budget"]["request"],
                     {"requested_block_table_mapping_bytes", "committed_block_table_mapping_bytes",
                      "layout_transient_bytes"});
  expect_string_fields(doc["memory_budget"], {"resource_rounding_bytes", "transient_peak_bytes",
                                              "known_owned_peak_bytes", "setup_peak_bytes", "runtime_peak_bytes",
                                              "admitted_budget_bytes"});
  EXPECT_EQ(doc["seeds"]["base_seed_uint64_decimal"], "18446744073709551615");
  ASSERT_EQ(measurement["checksum"]["status"], "valid");
  ASSERT_TRUE(measurement["checksum"]["checksum_valid"]);
  const OrderedJson& worker0_weight = scenario["expected_checksum"]["expected_worker_checksums"][0]["weight"];
  expect_string_values(worker0_weight, {{"state_a_uint64_decimal", "18446744073709551615"},
                                        {"state_b_uint64_decimal", "9007199254740993"},
                                        {"exact_bytes_read", "9007199254741093"}});
  EXPECT_TRUE(worker0_weight["span_count_uint64_decimal"].is_string());
  EXPECT_EQ(measurement["checksum"]["actual_run_checksum"]["state_a_uint64_decimal"], "18446744073709551615");
  expect_string_fields(doc["counters"], {"planned_work_units", "completed_effective_model_payload_bytes"});

  LlmResultMetadata unavailable_mem = metadata;
  unavailable_mem.available_memory_bytes = 0;
  unavailable_mem.available_memory_source = "unavailable";
  const OrderedJson unavailable_doc =
      build_llm_memory_json(config, plan, preparation_for(plan), unavailable_mem, result);
  EXPECT_TRUE(unavailable_doc["environment"]["available_memory_bytes"].is_null());
  EXPECT_EQ(unavailable_doc["environment"]["available_memory_source"], "unavailable");
}

TEST(LlmMemoryJsonTest, CanonicalPlansRemappedHandlesAndCalibrationWitnesses) {
  {
    SCOPED_TRACE("canonical_plans_and_witnesses");
    const auto config = explicit_config();
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid);
    const auto doc = build_cpu_doc(config, plan, complete_result(config, plan));
    const auto& resolved = doc.at("resolved_plan");
    expect_exact_keys(resolved,
                      {"valid", "reason_code", "plan_identity", "backend", "phase", "kv_layout", "work_unit_kind",
                       "geometry", "model_context", "layout", "resources", "component_identities", "methodology",
                       "model_work_plan", "scenario_plans", "frozen_plan_refs"});
    EXPECT_EQ(resolved["component_identities"]["run_policy_version"], "llm-run-policy-bounded-loop-snapshots-v1");
    ASSERT_EQ(resolved["scenario_plans"].size(), 3u);
    expect_exact_keys(resolved["frozen_plan_refs"], {"weights_only", "kv_only", "mixed"});
    for (size_t i = 0; i < 3; ++i) {
      const auto& canonical = resolved["scenario_plans"][i];
      EXPECT_EQ(canonical["work_units"], 4u);
      expect_string_values(canonical, {{"model_ref", "resolved_plan"}, {"work_policy", "explicit_fixed_work"}});
      const auto& expected = canonical["expected_checksum"];
      expect_exact_keys(expected, {"status", "reason_code", "expected_worker_checksums", "expected_run_checksum"});
      EXPECT_EQ(expected["status"], "available");
      ASSERT_EQ(expected["expected_worker_checksums"].size(), 2u);
      expect_exact_keys(expected["expected_worker_checksums"][0], {"worker_index", "weight", "k", "v"});
      EXPECT_EQ(expected["expected_worker_checksums"][0]["weight"]["state_b_uint64_decimal"], "9007199254740993");
      EXPECT_EQ(expected["expected_run_checksum"]["state_a_uint64_decimal"], "18446744073709551615");
    }
    for (size_t id = 0; id < doc["measurements"].size(); ++id) {
      const auto& measurement = doc["measurements"][id];
      EXPECT_EQ(measurement["measurement_id"], id);
      const auto& canonical = referenced_plan(doc, measurement);
      EXPECT_EQ(canonical["scenario"], measurement["scenario"]);
      EXPECT_EQ(measurement["completion_derivation"], "accepted-plan-derived");
      EXPECT_EQ(measurement["requested_workers"], 2u);
      EXPECT_EQ(measurement["effective_workers"], 2u);
      EXPECT_EQ(measurement["qos_successful_workers"], 2u);
      EXPECT_EQ(measurement["qos_failed_workers"], 0u);
      EXPECT_EQ(measurement["calibration_attempt_count"], 1u);
      EXPECT_EQ(measurement["calibration_attempt_indexes"], OrderedJson::array({0}));
      EXPECT_EQ(measurement["working_set"]["bytes"], "1048960");
      if (measurement["scenario"] == "mixed") {
        EXPECT_DOUBLE_EQ(measurement["weight_payload_fraction"].get<double>(), 1048576.0 / 1049088.0);
        EXPECT_DOUBLE_EQ(measurement["kv_read_payload_fraction"].get<double>(), 384.0 / 1049088.0);
        EXPECT_DOUBLE_EQ(measurement["kv_write_payload_fraction"].get<double>(), 128.0 / 1049088.0);
      } else {
        EXPECT_TRUE(measurement["weight_payload_fraction"].is_null());
      }
      expect_exact_keys(measurement["checksum"],
                        {"status", "reason_code", "checksum_valid", "actual_worker_checksums", "actual_run_checksum"});
      EXPECT_EQ(measurement["checksum"]["actual_run_checksum"],
                canonical["expected_checksum"]["expected_run_checksum"]);
      for (const auto& check : measurement["execution"]["validation"]["checks"]) {
        expect_exact_keys(check, {"kind", "applicable", "evaluated", "valid", "reason_code"});
        if (check["applicable"] == true) {
          EXPECT_TRUE(check["evaluated"]);
          EXPECT_TRUE(check["valid"]);
        } else {
          expect_null_fields(check, {"evaluated", "valid"});
        }
      }
    }
    for (const char* scenario : {"weights_only", "kv_only", "mixed"}) {
      const auto& attempt = doc["calibration"]["attempts"][scenario][0];
      EXPECT_EQ(referenced_plan(doc, attempt)["scenario"], scenario);
      EXPECT_EQ(attempt["execution"]["checksum"]["actual_run_checksum"],
                referenced_plan(doc, attempt)["expected_checksum"]["expected_run_checksum"]);
      ASSERT_EQ(attempt["execution"]["validation"]["checks"].size(), 3u);
    }
    const auto& manifest = doc["build_manifest"];
    expect_exact_keys(manifest, {"manifest_version", "status", "reason_code", "binary_sha256", "git_commit",
                                 "git_dirty", "compiler", "compile_flags", "link_flags", "target_arch", "sdk",
                                 "min_os"});
    EXPECT_EQ(manifest["manifest_version"], 1);
    expect_string_values(manifest, {{"status", "unavailable"}, {"reason_code", "build-provenance-not-provided"}});
    expect_null_fields(manifest, {"binary_sha256", "git_commit", "git_dirty", "compiler", "compile_flags", "link_flags",
                                  "target_arch", "sdk", "min_os"});
    EXPECT_TRUE(resolved["model_context"]["prefill"].is_null());
  }
  {
    SCOPED_TRACE("sorted_snapshot_handle_remap");
    const auto config = explicit_config(1);
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid);
    FakeLlmBackend backend;
    LlmMemoryResult result;
    const auto t8 = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 8, false);
    const auto t2 = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 2, false);
    const auto handle8 = register_llm_scenario_plan(result, plan, t8, backend);
    const auto handle2 = register_llm_scenario_plan(result, plan, t2, backend);
    EXPECT_EQ(register_llm_scenario_plan(result, plan, t8, backend), handle8);
    result.frozen_plan_handles[static_cast<size_t>(LlmScenario::Mixed)] = handle8;
    LlmMeasurementState measurement;
    measurement.scenario = LlmScenario::Mixed;
    measurement.plan_handle = handle8;
    result.measurements.push_back(std::move(measurement));
    LlmCalibrationAttempt attempt;
    attempt.scenario = LlmScenario::Mixed;
    attempt.plan_handle = handle2;
    result.calibration_attempts[static_cast<size_t>(LlmScenario::Mixed)].push_back(std::move(attempt));
    prepare_llm_result_snapshot(result);
    const auto doc = build_cpu_doc(config, plan, result);
    ASSERT_EQ(doc["resolved_plan"]["scenario_plans"].size(), 2u);
    EXPECT_EQ(doc["resolved_plan"]["scenario_plans"][0]["work_units"], 2u);
    EXPECT_EQ(doc["resolved_plan"]["scenario_plans"][1]["work_units"], 8u);
    EXPECT_EQ(doc["resolved_plan"]["frozen_plan_refs"]["mixed"], 1u);
    EXPECT_EQ(doc["measurements"][0]["plan_ref"], 1u);
    EXPECT_EQ(doc["calibration"]["attempts"]["mixed"][0]["plan_ref"], 0u);
    EXPECT_EQ(result.measurements[0].plan_handle, handle8);
    EXPECT_EQ(result.calibration_attempts[static_cast<size_t>(LlmScenario::Mixed)][0].plan_handle, handle2);
  }
  {
    SCOPED_TRACE("automatic_calibration_witnesses");
    auto config = explicit_config(1);
    config.user_specified_iterations = false;
    config.iterations = 0;
    const auto doc = run_cpu_doc(
        config,
        [](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan& task, const LlmRunnerTaskContext&) {
          return successful_execution(model, static_cast<double>(task.work_units) * 0.01);
        },
        EXIT_SUCCESS);
    ASSERT_GT(doc["resolved_plan"]["scenario_plans"].size(), 3u);
    for (const char* scenario : {"weights_only", "kv_only", "mixed"}) {
      const auto& attempts = doc["calibration"]["attempts"][scenario];
      ASSERT_GT(attempts.size(), 1u);
      for (const auto& attempt : attempts) {
        const auto& canonical = referenced_plan(doc, attempt);
        expect_string_values(canonical, {{"scenario", scenario}, {"work_policy", "automatic_calibration"}});
        EXPECT_EQ(canonical["expected_checksum"]["status"], "available");
        ASSERT_EQ(canonical["expected_checksum"]["expected_worker_checksums"].size(), 2u);
        EXPECT_EQ(canonical["expected_checksum"]["expected_run_checksum"],
                  attempt["execution"]["checksum"]["actual_run_checksum"]);
      }
    }
  }
}

TEST(LlmMemoryJsonTest, UnknownAndContradictorySnapshotReferencesAreRejected) {
  const auto config = explicit_config(1);
  const auto plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid);
  auto result = complete_result(config, plan);
  const auto saved = result.measurements[0].plan_handle;
  result.measurements[0].plan_handle = result.scenario_plans.size();
  EXPECT_THROW(build_cpu_doc(config, plan, result), std::invalid_argument);
  result.measurements[0].plan_handle = saved;
  result.snapshot_plan_refs[saved] = result.snapshot_plan_order.size();
  EXPECT_THROW(build_cpu_doc(config, plan, result), std::invalid_argument);
  prepare_llm_result_snapshot(result);
  auto swapped = result;
  std::swap(swapped.frozen_plan_handles[0], swapped.frozen_plan_handles[1]);
  EXPECT_THROW(build_cpu_doc(config, plan, swapped), std::invalid_argument);
  swapped = result;
  swapped.calibration_attempts[0][0].plan_handle = swapped.frozen_plan_handles[1];
  EXPECT_THROW(build_cpu_doc(config, plan, swapped), std::invalid_argument);
  swapped = result;
  swapped.measurements[0].plan_handle = swapped.frozen_plan_handles[1];
  EXPECT_THROW(build_cpu_doc(config, plan, swapped), std::invalid_argument);
  result.calibration_attempts[0][0].plan_handle = result.scenario_plans.size();
  EXPECT_THROW(build_cpu_doc(config, plan, result), std::invalid_argument);
}

TEST(LlmMemoryJsonTest, SnapshotAggregatesOneLoopAcceptanceAndLoopRecordPrefix) {
  {
    SCOPED_TRACE("one_loop_acceptance_and_realized_prefix");
    const auto config = explicit_config(1);
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid);
    auto result = complete_result(config, plan);
    const auto doc = build_cpu_doc(config, plan, result);
    EXPECT_TRUE(doc["run_accepted"]);
    EXPECT_FALSE(doc["scenario_order_balance_complete"]);
    EXPECT_EQ(doc["aggregates"]["scenarios"]["weights_only"]["observed_cv_classification"], "insufficient-samples");
    EXPECT_EQ(doc["aggregates"]["scenarios"]["weights_only"]["accepted_measurement_ids"], OrderedJson::array({0}));
    for (const char* metric : {"synthetic_work_unit_latency_seconds", "synthetic_memory_work_units_per_second",
                               "effective_model_payload_gb_s"}) {
      const auto& stats = doc["aggregates"]["scenarios"]["weights_only"][metric]["statistics"];
      EXPECT_EQ(stats["sample_count"], 1u);
      EXPECT_DOUBLE_EQ(stats["stddev"].get<double>(), 0.0);
      EXPECT_DOUBLE_EQ(stats["coefficient_of_variation_pct"].get<double>(), 0.0);
      EXPECT_DOUBLE_EQ(stats["median_absolute_deviation"].get<double>(), 0.0);
    }

    result.loops[0].realized_order_count = 1;
    const auto prefix_doc = build_cpu_doc(config, plan, result);
    const OrderedJson& loop = prefix_doc["loop_records"][0];
    EXPECT_EQ(loop["planned_order"].size(), kLlmScenarioCount);
    EXPECT_EQ(loop["realized_order"], OrderedJson::array({"weights_only"}));
    EXPECT_EQ(loop["measurement_indexes"], (OrderedJson::array({0, 1, 2})));
  }
  {
    SCOPED_TRACE("snapshot_statistics_accepted_ids_and_median");
    const auto config = explicit_config(3);
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid);
    auto result = complete_result(config, plan);
    ASSERT_EQ(result.measurements[0].scenario, LlmScenario::WeightsOnly);
    ASSERT_EQ(result.measurements[5].scenario, LlmScenario::WeightsOnly);
    ASSERT_EQ(result.measurements[7].scenario, LlmScenario::WeightsOnly);
    result.measurements[0].execution.timing.elapsed_seconds = 1.0;
    result.measurements[5].execution.timing.elapsed_seconds = 3.0;
    result.measurements[7].status = LlmMeasurementStatus::Invalid;
    prepare_llm_result_snapshot(result);
    const auto doc = build_cpu_doc(config, plan, result);
    const auto& aggregate = doc["aggregates"]["scenarios"]["weights_only"];
    EXPECT_EQ(aggregate["accepted_measurement_ids"], OrderedJson::array({0, 5}));
    for (const char* metric : {"synthetic_work_unit_latency_seconds", "synthetic_memory_work_units_per_second",
                               "effective_model_payload_gb_s"}) {
      EXPECT_EQ(aggregate[metric]["sample_count"], 2u);
    }
    EXPECT_DOUBLE_EQ(aggregate["synthetic_memory_work_units_per_second"]["statistics"]["median"].get<double>(),
                     (4.0 + 4.0 / 3.0) / 2.0);
    EXPECT_TRUE(doc["measurements"][7]["effective_model_payload_gb_s"].is_null());
  }
}

TEST(LlmMemoryJsonTest, CpuPagedAndPrefillDocumentsPublishLayoutAndPartitionProvenance) {
  {
    SCOPED_TRACE("cpu_prefill_contiguous");
    const LlmMemoryConfig config = prefill_config();
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_TRUE(cpu_execution_plan(plan).prefill.has_value());
    const LlmMemoryResult result = complete_result(config, plan);
    ASSERT_TRUE(result.results_complete);
    const OrderedJson doc = build_cpu_doc(config, plan, result);

    EXPECT_TRUE(doc["results_complete"]);
    EXPECT_TRUE(doc["run_accepted"]);
    expect_string_values(doc, {{"status", "complete"},
                               {"phase", "prefill"},
                               {"kv_layout", "contiguous"},
                               {"methodology_version", "llm-memory-v2-cpu-prefill-contiguous"}});
    EXPECT_EQ(doc["configuration"]["phase"], "prefill");
    EXPECT_TRUE(doc["configuration"]["visible_context_tokens"].is_null());
    EXPECT_EQ(doc["configuration"]["prompt_tokens"], 5u);
    EXPECT_EQ(doc["configuration"]["attention_query_tile_tokens"], 2u);
    expect_string_values(doc["configuration"]["resolved_sources"],
                         {{"phase", "explicit"}, {"prompt_tokens", "explicit"},
                          {"attention_query_tile_tokens", "explicit"}});
    EXPECT_TRUE(doc["configuration"]["resolved_sources"]["visible_context_tokens"].is_null());

    const OrderedJson& geometry = doc["resolved_plan"]["geometry"];
    EXPECT_TRUE(geometry["decode"].is_null());
    expect_exact_keys(
        geometry["prefill"],
        {"prompt_tokens", "attention_query_tile_tokens", "tile_count", "attention_prefix_token_visits_per_sequence"});
    EXPECT_EQ(geometry["prefill"]["prompt_tokens"], 5u);
    EXPECT_EQ(geometry["prefill"]["attention_query_tile_tokens"], 2u);
    expect_string_values(geometry["prefill"],
                         {{"tile_count", "3"}, {"attention_prefix_token_visits_per_sequence", "11"}});
    expect_string_values(doc["resolved_plan"]["model_context"]["prefill"], {{"causal_token_pairs_per_sequence", "15"},
                                                                            {"logical_attention_pairs", "120"},
                                                                            {"logical_attention_fma_terms", "960"}});
    expect_string_values(geometry, {{"k_mapping_bytes", "320"},
                                    {"v_mapping_bytes", "320"},
                                    {"kv_capacity_bytes", "640"},
                                    {"weight_read_bytes_per_work_unit", "1048576"},
                                    {"kv_read_bytes_per_work_unit", "1408"},
                                    {"kv_write_bytes_per_work_unit", "640"},
                                    {"kv_only_effective_model_payload_bytes_per_work_unit", "2048"},
                                    {"mixed_effective_model_payload_bytes_per_work_unit", "1050624"}});
    expect_null_fields(geometry, {"traffic_crossover_numerator", "traffic_crossover_denominator",
                                  "traffic_crossover_context_tokens"});
    expect_null_fields(doc["aggregates"]["traffic_diagnostics"],
                       {"traffic_crossover_numerator", "traffic_crossover_denominator",
                        "traffic_crossover_context_tokens", "current_visible_context_tokens",
                        "current_weight_to_kv_read_payload_ratio", "current_context_classification"});
    expect_string_values(doc["resolved_plan"]["component_identities"],
                         {{"logical_profile_version", Constants::LLM_PREFILL_LOGICAL_PROFILE_VERSION},
                          {"backend_executor_version", Constants::LLM_PREFILL_CPU_EXECUTOR_VERSION},
                          {"resource_abi_version", Constants::LLM_PREFILL_DESCRIPTOR_ABI_VERSION},
                          {"write_pattern_version", "llm-prefill-kv-affine64-v1"}});

    const OrderedJson& prefill = doc["backend_evidence"]["cpu"]["prefill"];
    expect_exact_keys(prefill, {"cost_unit", "sequence_descriptors_per_scenario_per_worker", "scenarios", "identity"});
    EXPECT_EQ(prefill["cost_unit"], "worker-cost");
    EXPECT_EQ(prefill["sequence_descriptors_per_scenario_per_worker"], 2u);
    EXPECT_FALSE(prefill["identity"].get<std::string>().empty());
    ASSERT_EQ(prefill["scenarios"].size(), 3u);
    const std::array<std::pair<const char*, const char*>, 3> expected_costs = {
        {{"weights_only", "524288"}, {"kv_only", "1024"}, {"mixed", "525312"}}};
    for (size_t i = 0; i < expected_costs.size(); ++i) {
      const OrderedJson& scenario = prefill["scenarios"][i];
      expect_exact_keys(
          scenario, {"scenario", "cost_unit", "scope_count", "scope_identities", "worker_accounted_bytes_per_work_unit",
                     "minimum_worker_accounted_bytes_per_work_unit", "maximum_worker_accounted_bytes_per_work_unit",
                     "worker_accounted_imbalance_bytes_per_work_unit", "identity"});
      expect_string_values(scenario, {{"scenario", expected_costs[i].first},
                                      {"cost_unit", "worker-cost"},
                                      {"scope_count", "2"},
                                      {"minimum_worker_accounted_bytes_per_work_unit", expected_costs[i].second},
                                      {"maximum_worker_accounted_bytes_per_work_unit", expected_costs[i].second},
                                      {"worker_accounted_imbalance_bytes_per_work_unit", "0"}});
      ASSERT_EQ(scenario["scope_identities"].size(), 2u);
      EXPECT_FALSE(scenario["scope_identities"][0].get<std::string>().empty());
      EXPECT_EQ(scenario["worker_accounted_bytes_per_work_unit"],
                (OrderedJson::array({expected_costs[i].second, expected_costs[i].second})));
      EXPECT_FALSE(scenario["identity"].get<std::string>().empty());
    }
    EXPECT_TRUE(doc["backend_evidence"]["cpu"]["paged"].is_null());

    const OrderedJson& scenarios = doc["resolved_plan"]["scenario_plans"];
    ASSERT_EQ(scenarios.size(), 3u);
    expect_string_values(scenarios[0], {{"work_unit_kind", "prefill_operation"}, {"kv_write_kind", "none"}});
    expect_string_values(
        scenarios[1],
        {{"kv_write_kind", "full_prompt_population"}, {"effective_model_payload_bytes_per_work_unit", "2048"}});
    expect_string_values(
        scenarios[2],
        {{"kv_write_kind", "full_prompt_population"}, {"effective_model_payload_bytes_per_work_unit", "1050624"}});

    ASSERT_EQ(doc["measurements"].size(), 9u);
    for (const OrderedJson& measurement : doc["measurements"]) {
      EXPECT_EQ(measurement["status"], "measured");
      EXPECT_EQ(referenced_plan(doc, measurement)["work_unit_kind"], "prefill_operation");
      expect_null_fields(measurement["working_set"], {"fixed_visible_context_tokens", "current_token_slot_included"});
      EXPECT_EQ(measurement["checksum"]["status"], "valid");
      EXPECT_TRUE(measurement["checksum"]["checksum_valid"]);
    }
    EXPECT_TRUE(doc["interpretation"]["fixed_context_includes_current_token_slot"].is_null());
    EXPECT_FALSE(doc["interpretation"]["prefill_transformer_compute_or_ttft_prediction_included"]);
  }
  {
    SCOPED_TRACE("cpu_prefill_paged");
    const LlmMemoryConfig config = paged_prefill_config();
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
    ASSERT_TRUE(cpu.paged.has_value());
    ASSERT_TRUE(cpu.prefill.has_value());
    const LlmPagedCpuExecutionPlan& paged = *cpu.paged;
    const LlmPrefillCpuExecutionPlan& prefill_plan = *cpu.prefill;
    const LlmMemoryResult result = complete_result(config, plan);
    ASSERT_TRUE(result.results_complete) << result.reason_code;
    const OrderedJson doc = build_cpu_doc(config, plan, result);

    EXPECT_EQ(doc["schema_version"], 2);
    expect_string_values(doc, {{"status", "complete"},
                               {"phase", "prefill"},
                               {"kv_layout", "paged"},
                               {"methodology_version", Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION}});
    expect_string_values(doc["resolved_plan"]["component_identities"],
                         {{"backend_executor_version", Constants::LLM_PREFILL_PAGED_CPU_EXECUTOR_VERSION},
                          {"resource_abi_version", Constants::LLM_PREFILL_PAGED_DESCRIPTOR_ABI_VERSION},
                          {"buffer_pattern_version", Constants::LLM_PAGED_BUFFER_PATTERN_VERSION},
                          {"write_pattern_version", LlmPrefillVersion::WRITE_PATTERN}});

    const OrderedJson& layout = doc["resolved_plan"]["layout"];
    EXPECT_EQ(layout["kv_block_tokens"], 2u);
    EXPECT_TRUE(layout["decode_append_offset_in_last_block"].is_null());
    expect_string_values(layout, {{"kv_layout", "paged"},
                                  {"layout_identity", paged.layout_identity},
                                  {"permutation_identity", paged.permutation.identity}});

    const OrderedJson& paged_ev = doc["backend_evidence"]["cpu"]["paged"];
    expect_exact_keys(paged_ev, {"layout_identity", "execution_identity", "block_table_logical_bytes",
                                 "block_table_page_rounded_bytes", "block_table_read_only", "table_validation",
                                 "permutation", "ownership"});
    expect_string_values(paged_ev, {{"layout_identity", paged.layout_identity},
                                    {"execution_identity", prefill_plan.identity}});
    EXPECT_TRUE(paged_ev["ownership"].is_null());

    const OrderedJson& prefill_ev = doc["backend_evidence"]["cpu"]["prefill"];
    expect_exact_keys(prefill_ev,
                      {"cost_unit", "sequence_descriptors_per_scenario_per_worker", "scenarios", "identity"});
    EXPECT_EQ(prefill_ev["identity"], prefill_plan.identity);
    ASSERT_EQ(prefill_ev["scenarios"].size(), kLlmScenarioCount);
    EXPECT_EQ(prefill_ev["scenarios"][1]["scenario"], "kv_only");
    EXPECT_EQ(prefill_ev["scenarios"][2]["scenario"], "mixed");

    const LlmScenarioLimits kv_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::KvOnly);
    ASSERT_TRUE(kv_limits.valid) << kv_limits.reason_code;
    ASSERT_GE(doc["measurements"].size(), 2u);
    const std::string lookups = std::to_string(kv_limits.layout_metadata_lookup_count_per_work_unit);
    const std::string lookup_bytes = std::to_string(kv_limits.layout_metadata_read_bytes_per_work_unit);
    const std::string accounted = std::to_string(kv_limits.accounted_bytes_per_work_unit);
    expect_string_values(referenced_plan(doc, doc["measurements"][1]),
                         {{"work_unit_kind", "prefill_operation"},
                          {"kv_write_kind", "full_prompt_population"},
                          {"layout_metadata_lookup_count_per_work_unit", lookups},
                          {"layout_metadata_read_bytes_per_work_unit", lookup_bytes},
                          {"accounted_bytes_per_work_unit", accounted}});
  }
  {
    SCOPED_TRACE("cpu_decode_paged");
    const LlmMemoryConfig config = paged_config();
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
    ASSERT_TRUE(cpu.paged.has_value());
    const LlmPagedCpuExecutionPlan& paged = *cpu.paged;
    const LlmMemoryResult result = complete_result(config, plan);
    ASSERT_TRUE(result.results_complete) << result.reason_code;
    const OrderedJson doc = build_cpu_doc(config, plan, result);

    EXPECT_EQ(doc["schema_version"], 2);
    EXPECT_EQ(doc["kv_layout"], "paged");
    EXPECT_EQ(doc["configuration"]["kv_layout"], "paged");
    EXPECT_EQ(doc["configuration"]["kv_block_tokens"], 2u);
    expect_string_values(doc["configuration"]["resolved_sources"],
                         {{"kv_layout", "explicit"}, {"kv_block_tokens", "explicit"}});

    const std::string perm_seed = std::to_string(paged.permutation.resolved_seed);
    EXPECT_EQ(doc["resolved_plan"]["layout"]["kv_block_tokens"], 2u);
    expect_string_values(doc["resolved_plan"]["layout"],
                         {{"kv_layout", "paged"},
                          {"blocks_per_sequence", "2"},
                          {"physical_blocks_per_layer", "2"},
                          {"total_physical_blocks", "4"},
                          {"block_bytes", "64"},
                          {"last_block_tokens", "1"},
                          {"last_block_valid_bytes", "32"},
                          {"decode_append_offset_in_last_block", "0"},
                          {"block_table_entries", "2"},
                          {"block_table_bytes", "8"},
                          {"layout_geometry_identity", paged.layout.geometry_identity},
                          {"layout_identity", paged.layout_identity},
                          {"permutation_domain_uint64_hex", "0x4c4c4d4b56504731"},
                          {"permutation_seed_uint64_decimal", perm_seed},
                          {"permutation_algorithm_version", Constants::LLM_KV_BLOCK_PERMUTATION_VERSION},
                          {"permutation_entry_count", "2"},
                          {"permutation_sha256", paged.permutation.sha256},
                          {"permutation_identity", paged.permutation.identity}});
    expect_string_values(doc["resolved_plan"]["resources"], {{"k_logical_bytes", "192"},
                                                             {"v_logical_bytes", "192"},
                                                             {"k_physical_length_bytes", "256"},
                                                             {"v_physical_length_bytes", "256"},
                                                             {"k_layout_padding_bytes", "64"},
                                                             {"v_layout_padding_bytes", "64"},
                                                             {"block_table_bytes", "8"}});

    const OrderedJson& paged_ev = doc["backend_evidence"]["cpu"]["paged"];
    expect_exact_keys(paged_ev, {"layout_identity", "execution_identity", "block_table_logical_bytes",
                                 "block_table_page_rounded_bytes", "block_table_read_only", "table_validation",
                                 "permutation", "ownership"});
    expect_string_values(paged_ev, {{"layout_identity", paged.layout_identity},
                                    {"execution_identity", paged.execution_identity},
                                    {"block_table_logical_bytes", "8"},
                                    {"block_table_page_rounded_bytes", "4096"}});
    EXPECT_TRUE(paged_ev["block_table_read_only"]);
    expect_exact_keys(paged_ev["table_validation"], {"valid", "interrupted", "reason_code", "expected_entries",
                                                     "examined_entries", "validation_bitset_bytes"});
    EXPECT_TRUE(paged_ev["table_validation"]["valid"]);
    expect_string_values(paged_ev["table_validation"],
                         {{"expected_entries", "2"}, {"examined_entries", "2"}, {"validation_bitset_bytes", "1"}});
    expect_exact_keys(paged_ev["permutation"], {"algorithm_version", "domain_uint64_hex",
                                                "resolved_seed_uint64_decimal", "entry_count", "sha256", "identity"});
    expect_string_values(paged_ev["permutation"],
                         {{"sha256", paged.permutation.sha256}, {"identity", paged.permutation.identity}});
    expect_exact_keys(
        paged_ev["ownership"],
        {"valid", "reason_code", "worker_count", "layout_geometry_identity", "layer_sequence_count",
         "total_owned_blocks", "total_model_payload_bytes_per_work_unit",
         "total_layout_metadata_lookup_count_per_work_unit", "total_layout_metadata_read_bytes_per_work_unit",
         "total_accounted_bytes_per_work_unit", "minimum_worker_accounted_bytes_per_work_unit",
         "maximum_worker_accounted_bytes_per_work_unit", "worker_accounted_imbalance_bytes_per_work_unit",
         "assignment_count", "identity"});
    EXPECT_TRUE(paged_ev["ownership"]["valid"]);
    expect_string_values(paged_ev["ownership"], {{"total_owned_blocks", "4"},
                                                 {"total_layout_metadata_lookup_count_per_work_unit", "10"},
                                                 {"total_layout_metadata_read_bytes_per_work_unit", "40"},
                                                 {"identity", paged.ownership.identity}});

    const OrderedJson& cpu_res = doc["backend_evidence"]["cpu"]["resources"];
    expect_exact_keys(cpu_res["mappings"]["block_table"], {"requested_bytes", "committed_bytes"});
    expect_string_values(cpu_res["mappings"]["block_table"], {{"requested_bytes", "8"}, {"committed_bytes", "4096"}});
    expect_string_values(cpu_res["initialization"], {{"block_table_logical_bytes", "8"},
                                                     {"block_table_page_rounded_bytes", "4096"},
                                                     {"k_layout_padding_bytes", "64"},
                                                     {"v_layout_padding_bytes", "64"}});
    EXPECT_TRUE(cpu_res["initialization"]["block_table_read_only"]);

    const std::string setup_peak = std::to_string(plan.memory_budget.request.setup_peak_bytes);
    const std::string runtime_peak = std::to_string(plan.memory_budget.request.runtime_peak_bytes);
    expect_string_values(doc["memory_budget"], {{"layout_transient_bytes", "1"},
                                                {"setup_peak_bytes", setup_peak},
                                                {"runtime_peak_bytes", runtime_peak}});
    expect_string_values(doc["memory_budget"]["request"], {{"requested_block_table_mapping_bytes", "8"},
                                                           {"committed_block_table_mapping_bytes", "4096"},
                                                           {"layout_transient_bytes", "1"}});

    ASSERT_GE(doc["measurements"].size(), 2u);
    ASSERT_EQ(doc["measurements"][1]["scenario"], "kv_only");
    expect_string_values(referenced_plan(doc, doc["measurements"][1]),
                         {{"layout_metadata_lookup_count_per_work_unit", "10"},
                          {"layout_metadata_read_bytes_per_work_unit", "40"},
                          {"effective_model_payload_bytes_per_work_unit", "512"},
                          {"accounted_bytes_per_work_unit", "552"}});
    expect_string_values(doc["resolved_plan"]["component_identities"],
                         {{"buffer_pattern_version", Constants::LLM_PAGED_BUFFER_PATTERN_VERSION},
                          {"checksum_pattern_version", Constants::LLM_PAGED_READ_CHECKSUM_VERSION}});
    ASSERT_FALSE(doc["calibration"]["attempts"]["weights_only"].empty());
  }
}

TEST(LlmMemoryJsonTest, UnsupportedAndUnresolvedPlanNullSerialization) {
  {
    SCOPED_TRACE("unsupported_before_preparation");
    const LlmMemoryConfig config = explicit_config();
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_TRUE(plan.memory_budget.valid) << plan.memory_budget.reason_code;
    LlmMemoryResult result;
    result.initialized = true;
    result.status = LlmRunStatus::Unsupported;
    result.reason_code = LlmBackendReason::TASK_UNSUPPORTED;
    LlmBackendEvidence backend;
    backend.backend = LlmMemoryBackend::Cpu;
    backend.initialization = {LlmBackendStatus::Unsupported, LlmBackendReason::TASK_UNSUPPORTED};

    const OrderedJson doc = build_llm_memory_json(config, plan, backend, fixed_metadata(config, plan), result);
    expect_string_values(doc, {{"status", "unsupported"}, {"reason_code", LlmBackendReason::TASK_UNSUPPORTED}});
    EXPECT_TRUE(doc["memory_budget"]["valid"].get<bool>());
    EXPECT_EQ(doc["memory_budget"]["reason_code"], plan.memory_budget.reason_code);
    EXPECT_EQ(doc["memory_budget"]["admitted_budget_bytes"], std::to_string(plan.memory_budget.allowed_memory_bytes));
    EXPECT_FALSE(doc["backend_evidence"]["cpu"]["resources"]["valid"].get<bool>());
  }
  {
    SCOPED_TRACE("unresolved_plan_and_expected_witness");
    const auto config = explicit_config(1);
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid);
    LlmMemoryResult result;
    LlmCanonicalScenarioPlan entry;
    entry.plan = build_llm_scenario_work_plan(plan, LlmScenario::KvOnly, 4, true);
    result.scenario_plans.push_back(std::move(entry));
    LlmMeasurementState measurement;
    measurement.scenario = LlmScenario::KvOnly;
    result.measurements.push_back(std::move(measurement));
    prepare_llm_result_snapshot(result);
    const auto doc = build_cpu_doc(config, plan, result);
    const auto& expected = doc["resolved_plan"]["scenario_plans"][0]["expected_checksum"];
    expect_string_values(expected, {{"status", "unavailable"}, {"reason_code", "not-evaluated"}});
    expect_null_fields(expected, {"expected_worker_checksums", "expected_run_checksum"});
    EXPECT_TRUE(doc["measurements"][0]["plan_ref"].is_null());
    EXPECT_TRUE(doc["resolved_plan"]["frozen_plan_refs"]["kv_only"].is_null());
    expect_named_check(doc["measurements"][0]["execution"], "kv-append-final", true, false, std::nullopt);
    EXPECT_FALSE(doc["run_accepted"]);
  }
}

TEST(LlmMemoryJsonTest, TrafficClassificationAndPrefillTheoreticalOverflow) {
  {
    SCOPED_TRACE("traffic_classification");
    LlmGeometryRequest request{8, 1, 1, 1, 1, 1, 4, 1};
    const std::array<std::pair<size_t, const char*>, 3> cases = {
        {{8, "near_crossover"}, {9, "weight_payload_dominant"}, {7, "kv_read_payload_dominant"}}};
    for (const auto& [weights, label] : cases) {
      request.active_weight_bytes = weights;
      const LlmGeometry geometry = resolve_llm_geometry(request);
      ASSERT_TRUE(geometry.valid) << geometry.reason_code;
      if (weights == 8) ASSERT_EQ(geometry.weight_read_bytes_per_work_unit, geometry.kv_read_bytes_per_work_unit);
      EXPECT_STREQ(classify_llm_traffic_payload(geometry), label);
    }
    EXPECT_STREQ(classify_llm_traffic_payload(LlmGeometry{}), "unavailable");
  }
  {
    SCOPED_TRACE("prefill_theoretical_overflow");
    auto config = prefill_config(1);
    config.query_head_count = size_t{1} << 52;
    config.head_dimension = 256;
    const auto plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const auto doc = build_cpu_doc(config, plan, LlmMemoryResult{});
    const auto& context = doc["resolved_plan"]["model_context"]["prefill"];
    expect_exact_keys(context, {"causal_token_pairs_per_sequence", "causal_token_pairs_per_sequence_reason_code",
                                "logical_attention_pairs", "logical_attention_pairs_reason_code",
                                "logical_attention_fma_terms", "logical_attention_fma_terms_reason_code"});
    expect_string_values(context, {{"causal_token_pairs_per_sequence", "15"},
                                   {"logical_attention_pairs", "135107988821114880"},
                                   {"logical_attention_pairs_reason_code", "valid"},
                                   {"logical_attention_fma_terms_reason_code", "arithmetic-overflow"}});
    EXPECT_TRUE(context["logical_attention_fma_terms"].is_null());
    EXPECT_EQ(doc["resolved_plan"]["geometry"]["prefill"]["attention_prefix_token_visits_per_sequence"], "11");
    EXPECT_EQ(doc["resolved_plan"]["geometry"]["kv_read_bytes_per_work_unit"], "45056");
  }
}

TEST(LlmMemoryJsonTest, CollectJsonForSweepEnablesJsonPeakEstimateAndEvidenceWithEmptyOutputFile) {
  LlmMemoryConfig disabled = explicit_config(1);
  disabled.output_file.clear();
  disabled.collect_json_for_sweep = false;

  const auto disabled_empty = calculate_llm_json_peak_estimate(disabled, LlmMemoryWorkPlan{});
  EXPECT_TRUE(disabled_empty.valid);
  EXPECT_EQ(disabled_empty.reason_code, LlmJsonReason::VALID);
  EXPECT_EQ(std::tie(disabled_empty.fixed_schema_bytes, disabled_empty.input_string_bytes,
                     disabled_empty.measurement_record_bytes, disabled_empty.worker_checksum_bytes,
                     disabled_empty.total_bytes),
            std::make_tuple(size_t{0}, size_t{0}, size_t{0}, size_t{0}, size_t{0}));

  LlmMemoryConfig sweep_collect = disabled;
  sweep_collect.collect_json_for_sweep = true;

  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(sweep_collect));
  ASSERT_TRUE(draft.valid) << draft.reason_code;

  const auto disabled_preflight = calculate_llm_json_peak_estimate(disabled, draft.auxiliary_preflight);
  EXPECT_TRUE(disabled_preflight.valid);
  EXPECT_EQ(disabled_preflight.total_bytes, 0u);

  const auto collect_preflight = calculate_llm_json_peak_estimate(sweep_collect, draft.auxiliary_preflight);
  ASSERT_TRUE(collect_preflight.valid);
  EXPECT_GT(collect_preflight.fixed_schema_bytes, 0u);
  EXPECT_GT(collect_preflight.measurement_record_bytes, 0u);
  EXPECT_GT(collect_preflight.worker_checksum_bytes, 0u);
  EXPECT_GT(collect_preflight.total_bytes, 0u);

  const auto invalid_preflight = calculate_llm_json_peak_estimate(sweep_collect, LlmAuxiliaryPreflightView{});
  EXPECT_FALSE(invalid_preflight.valid);
  EXPECT_EQ(invalid_preflight.reason_code, LlmJsonReason::INVALID_MODEL_WORK_PLAN);

  const LlmMemoryWorkPlan plan = admitted_plan(sweep_collect);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const auto collect_final = calculate_llm_json_peak_estimate(sweep_collect, plan);
  ASSERT_TRUE(collect_final.valid);
  EXPECT_EQ(collect_final.total_bytes, collect_preflight.total_bytes);

  const auto invalid_plan = calculate_llm_json_peak_estimate(sweep_collect, LlmMemoryWorkPlan{});
  EXPECT_FALSE(invalid_plan.valid);
  EXPECT_EQ(invalid_plan.reason_code, LlmJsonReason::INVALID_MODEL_WORK_PLAN);

  const LlmMemoryResult result = complete_result(sweep_collect, plan);
  const OrderedJson collect_doc = build_cpu_doc(sweep_collect, plan, result);
  const OrderedJson& collect_peak = collect_doc["backend_evidence"]["cpu"]["resources"]["json_output_peak_estimate"];
  EXPECT_TRUE(collect_peak["enabled"].get<bool>());
  EXPECT_TRUE(collect_peak["valid"].get<bool>());
  EXPECT_EQ(collect_peak["total_bytes"], std::to_string(collect_final.total_bytes));

  const OrderedJson disabled_doc = build_cpu_doc(disabled, plan, result);
  const OrderedJson& disabled_peak = disabled_doc["backend_evidence"]["cpu"]["resources"]["json_output_peak_estimate"];
  EXPECT_FALSE(disabled_peak["enabled"].get<bool>());
  EXPECT_EQ(disabled_peak["total_bytes"], "0");
}

TEST(LlmMemoryJsonTest, PreflightAuxiliaryEstimatesExactlyMatchFinalizedPlanForAllActiveCpuProfiles) {
  const auto executor_fields = [](const LlmExecutorAuxiliaryEstimate& e) {
    return std::tie(e.valid, e.reason_code, e.static_reference_bytes, e.expected_checksum_bytes,
                    e.actual_checksum_bytes, e.run_checksum_bytes, e.worker_status_bytes, e.thread_handle_bytes,
                    e.checksum_auxiliary_bytes, e.orchestration_auxiliary_bytes, e.total_auxiliary_bytes);
  };
  const auto runner_fields = [](const LlmRunnerAuxiliaryEstimate& r) {
    return std::tie(r.valid, r.reason_code, r.measurement_record_bytes, r.loop_record_bytes, r.calibration_record_bytes,
                    r.calibration_identity_bytes, r.aggregate_value_bytes, r.statistics_workspace_bytes,
                    r.warning_record_bytes, r.fixed_metadata_bytes, r.retained_checksum_bytes,
                    r.checksum_auxiliary_bytes, r.orchestration_auxiliary_bytes, r.total_auxiliary_bytes);
  };
  const auto json_fields = [](const LlmJsonPeakEstimate& j) {
    return std::tie(j.valid, j.reason_code, j.fixed_schema_bytes, j.input_string_bytes, j.measurement_record_bytes,
                    j.worker_checksum_bytes, j.total_bytes);
  };

  for (const LlmMemoryConfig& config :
       {explicit_config(3), paged_config(3), prefill_config(3), paged_prefill_config(3)}) {
    SCOPED_TRACE(llm_kv_layout_to_string(config.kv_layout));
    LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(config));
    ASSERT_TRUE(draft.valid) << draft.reason_code;
    const auto pre_exec = calculate_llm_executor_auxiliary_estimate(draft.auxiliary_preflight);
    const auto pre_run = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
    const auto pre_json = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
    ASSERT_TRUE(pre_exec.valid && pre_run.valid && pre_json.valid);
    EXPECT_GT(pre_json.fixed_schema_bytes, 0u);
    EXPECT_GT(pre_json.input_string_bytes, 0u);
    EXPECT_GT(pre_json.measurement_record_bytes, 0u);
    EXPECT_GT(pre_json.worker_checksum_bytes, pre_json.measurement_record_bytes);
    EXPECT_EQ(pre_json.total_bytes, pre_json.fixed_schema_bytes + pre_json.input_string_bytes +
                                        pre_json.measurement_record_bytes + pre_json.worker_checksum_bytes);

    size_t checksum_aux = 0, orch_aux = 0;
    ASSERT_TRUE(NumericUtils::checked_add(pre_exec.checksum_auxiliary_bytes, pre_run.checksum_auxiliary_bytes,
                                          checksum_aux));
    ASSERT_TRUE(NumericUtils::checked_add(pre_exec.orchestration_auxiliary_bytes, pre_run.orchestration_auxiliary_bytes,
                                          orch_aux));
    ASSERT_TRUE(NumericUtils::checked_add(orch_aux, pre_json.total_bytes, orch_aux));
    const LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(std::move(draft), checksum_aux, orch_aux);
    ASSERT_TRUE(plan.valid) << plan.reason_code;

    EXPECT_EQ(executor_fields(pre_exec), executor_fields(calculate_llm_executor_auxiliary_estimate(plan)));
    EXPECT_EQ(runner_fields(pre_run), runner_fields(calculate_llm_runner_auxiliary_estimate(config, plan)));
    EXPECT_EQ(json_fields(pre_json), json_fields(calculate_llm_json_peak_estimate(config, plan)));
    EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, checksum_aux);
    EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, orch_aux);
  }
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateRejectsIdentityAndCountOverflowBeforeDom) {
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  for (size_t overflow = 0; overflow < 5; ++overflow) {
    SCOPED_TRACE(overflow);
    auto config = overflow == 3 ? prefill_config(1) : explicit_config(2);
    auto draft = prepare_llm_memory_work_plan(plan_request(config));
    ASSERT_TRUE(draft.valid) << draft.reason_code;
    auto& preflight = draft.auxiliary_preflight;
    switch (overflow) {
      case 0:
        preflight.maximum_scenario_plan_identity_bytes[0] = maximum / 2 + 1;
        break;
      case 1:
        config.loop_count = 1;
        config.user_specified_iterations = false;
        preflight.maximum_scenario_plan_identity_bytes[1] =
            maximum / (4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS) + 1;
        break;
      case 2:
        preflight.frozen_reason_code_bytes = maximum;
        preflight.frozen_model_plan_identity_bytes = 1;
        break;
      case 3:
        preflight.json_identity_string_bytes = maximum;
        break;
      case 4:
        config.loop_count = maximum;
        break;
    }
    const auto estimate = calculate_llm_json_peak_estimate(config, preflight);
    EXPECT_FALSE(estimate.valid);
    EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
    EXPECT_EQ(estimate.total_bytes, 0u);
  }
}

TEST(LlmMemoryJsonTest, InterruptedAndExceptionMeasurementsSerializeUnavailableEvidence) {
  const auto config = explicit_config(1);
  {
    SCOPED_TRACE("interrupted");
    LlmRunnerHooks hooks;
    hooks.stop_requested = []() { return true; };
    const OrderedJson doc = run_cpu_doc(config, {}, EXIT_SUCCESS, hooks);
    EXPECT_EQ(doc["status"], "interrupted");
    EXPECT_FALSE(doc["results_complete"]);
    EXPECT_FALSE(doc["run_accepted"]);
    ASSERT_FALSE(doc["measurements"].empty());
    const OrderedJson& m = doc["measurements"][0];
    expect_string_values(m, {{"status", "interrupted"},
                             {"reason_code", LlmRunnerReason::INTERRUPTION_BEFORE_TASK},
                             {"completed_effective_model_payload_bytes", "0"}});
    EXPECT_FALSE(m["attempted"]);
    EXPECT_EQ(m["completed_work_units"], 0u);
    expect_null_fields(m, {"elapsed_seconds", "synthetic_work_unit_latency_seconds",
                           "synthetic_memory_work_units_per_second", "effective_model_payload_gb_s"});
    EXPECT_EQ(m["execution"]["status"], "not_run");
    expect_null_fields(m["execution"], {"valid", "qos_successful_workers", "qos_failed_workers"});
    expect_not_evaluated_checksum(m["checksum"], LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
    EXPECT_TRUE(referenced_plan(doc, m)["effective_model_payload_bytes"].is_string());
  }
  {
    SCOPED_TRACE("executor_exception");
    for (auto phase : {LlmRunnerTaskKind::Measurement, LlmRunnerTaskKind::Warmup}) {
      SCOPED_TRACE(static_cast<size_t>(phase));
      const auto doc = run_cpu_doc(
          config,
          [phase](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
            if (context.kind == phase) throw std::runtime_error("injected JSON exception path");
            return successful_execution(model);
          },
          EXIT_FAILURE);
      expect_string_values(doc, {{"status", "failed"},
                                 {"reason_code", LlmRunnerReason::RUNNER_EXCEPTION},
                                 {"diagnostic", "injected JSON exception path"}});
      const bool measured = phase == LlmRunnerTaskKind::Measurement;
      const auto& entry = measured ? doc["measurements"][0] : doc["calibration"]["attempts"]["weights_only"][0];
      const auto& execution = entry["execution"];
      const auto& checksum = measured ? entry["checksum"] : execution["checksum"];
      if (measured) {
        EXPECT_TRUE(entry["attempted"]);
        expect_string_values(entry, {{"status", "failed"}, {"reason_code", LlmRunnerReason::RUNNER_EXCEPTION}});
        expect_null_fields(execution, {"qos_successful_workers", "qos_failed_workers"});
      }
      expect_string_values(execution, {{"status", "unavailable"}, {"reason_code", LlmRunnerReason::RUNNER_EXCEPTION}});
      expect_null_fields(execution, {"valid", "requested_workers", "kernel_succeeded"});
      EXPECT_EQ(checksum["status"], "not_evaluated");
      EXPECT_EQ(checksum["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
      EXPECT_TRUE(checksum["checksum_valid"].is_null());
    }
  }
}

TEST(LlmMemoryJsonTest, InvalidElapsedCardinalityAndChecksumMismatchSerialization) {
  const LlmMemoryConfig config = explicit_config(1);
  {
    SCOPED_TRACE("invalid_elapsed_measurement");
    const OrderedJson doc = run_cpu_doc(
        config,
        [](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
          auto execution = successful_execution(model);
          if (context.kind == LlmRunnerTaskKind::Measurement) {
            execution.valid = false;
            execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
            execution.elapsed_seconds = 0.0;
            execution.checksum_evaluated = false;
            execution.checksum_valid = false;
          }
          return execution;
        },
        EXIT_FAILURE);
    const OrderedJson& m = doc["measurements"][0];
    expect_string_values(m, {{"status", "invalid"}, {"reason_code", LlmExecutorReason::INVALID_ELAPSED_TIME}});
    expect_string_values(m["execution"],
                         {{"status", "invalid"}, {"reason_code", LlmExecutorReason::INVALID_ELAPSED_TIME}});
    EXPECT_FALSE(m["execution"]["valid"]);
    expect_not_evaluated_checksum(m["checksum"], LlmExecutorReason::INVALID_ELAPSED_TIME, true);
  }
  {
    SCOPED_TRACE("malformed_excluded_checksum_cardinality");
    for (bool invalid_elapsed : {false, true}) {
      SCOPED_TRACE(invalid_elapsed);
      const OrderedJson doc = run_cpu_doc(
          config,
          [invalid_elapsed](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan&,
                            const LlmRunnerTaskContext& context) {
            auto execution = successful_execution(model);
            if (context.kind == LlmRunnerTaskKind::Warmup) {
              if (invalid_elapsed) {
                execution.valid = false;
                execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
                execution.elapsed_seconds = 0.0;
                execution.checksum_evaluated = false;
                execution.checksum_valid = false;
              } else {
                execution.expected_checksums.resize(1);
                execution.actual_checksums.resize(1);
              }
            }
            return execution;
          },
          EXIT_FAILURE);
      const char* reason =
          invalid_elapsed ? LlmExecutorReason::INVALID_ELAPSED_TIME : LlmExecutorReason::INVALID_RESOURCES;
      const OrderedJson& attempt = doc["calibration"]["attempts"]["weights_only"][0];
      EXPECT_FALSE(attempt["valid"]);
      EXPECT_EQ(attempt["reason_code"], reason);
      expect_string_values(attempt["execution"], {{"status", "invalid"}, {"reason_code", reason}});
      EXPECT_FALSE(attempt["execution"]["valid"]);
      expect_not_evaluated_checksum(attempt["execution"]["checksum"], reason);
    }
  }
  {
    SCOPED_TRACE("checksum_mismatch");
    const OrderedJson doc = run_cpu_doc(
        config,
        [](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
          auto execution = successful_execution(model);
          if (context.kind == LlmRunnerTaskKind::Measurement) execution.checksum_valid = false;
          return execution;
        },
        EXIT_FAILURE);
    const OrderedJson& m = doc["measurements"][0];
    expect_string_values(m, {{"status", "invalid"}, {"reason_code", LlmExecutorReason::CHECKSUM_MISMATCH}});
    expect_string_values(m["checksum"], {{"status", "invalid"}, {"reason_code", LlmExecutorReason::CHECKSUM_MISMATCH}});
    EXPECT_FALSE(m["checksum"]["checksum_valid"]);
    EXPECT_TRUE(referenced_plan(doc, m)["expected_checksum"]["expected_worker_checksums"].is_array());
    EXPECT_TRUE(m["checksum"]["actual_worker_checksums"].is_array());
  }
}

TEST(LlmMemoryJsonTest, PostValidationAndAcceptanceFailureSerialization) {
  {
    SCOPED_TRACE("post_validation_failure");
    for (bool paged : {false, true}) {
      SCOPED_TRACE(paged);
      const auto config = paged ? paged_config() : explicit_config(1);
      const auto* reason =
          paged ? LlmExecutorReason::PAGED_POST_VALIDATION_FAILED : LlmExecutorReason::DECODE_POST_VALIDATION_FAILED;
      const auto doc = run_cpu_doc(
          config,
          [paged, reason](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan& scenario,
                          const LlmRunnerTaskContext& context) {
            auto execution = successful_execution(model);
            if (context.kind == LlmRunnerTaskKind::Measurement && (paged || scenario.scenario == LlmScenario::KvOnly)) {
              execution.valid = false;
              execution.reason_code = reason;
              execution.post_validation_valid = false;
            }
            return execution;
          },
          EXIT_FAILURE);
      const auto& m = doc["measurements"][paged ? 0 : 1];
      expect_string_values(m, {{"status", "invalid"}, {"reason_code", reason}});
      const std::vector<const char*> kinds =
          paged ? std::vector<const char*>{"post-validation-structure", "kv-append-unchanged", "kv-padding-canary"}
                : std::vector<const char*>{"kv-append-final"};
      for (const char* kind : kinds) expect_named_check(m["execution"], kind, true, true, false, reason);
      if (!paged) {
        expect_named_check(doc["measurements"][0]["execution"], "kv-append-unchanged", false, std::nullopt,
                           std::nullopt);
        EXPECT_EQ(doc["aggregates"]["scenarios"]["kv_only"]["effective_model_payload_gb_s"]["sample_count"], 0u);
      }
      expect_null_fields(m, {"elapsed_seconds", "synthetic_memory_work_units_per_second",
                             "effective_model_payload_gb_s"});
      EXPECT_DOUBLE_EQ(m["execution"]["timing"]["diagnostic_elapsed_seconds"].get<double>(), 0.150);
      EXPECT_TRUE(m["checksum"]["checksum_valid"]);
    }
  }
  {
    SCOPED_TRACE("common_acceptance_failure");
    const LlmMemoryConfig config = explicit_config(1);
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    LlmMemoryResult result = complete_result(config, plan);
    ASSERT_FALSE(result.measurements.empty());
    LlmMeasurementState& rejected = result.measurements.front();
    ASSERT_EQ(rejected.execution.status, LlmTaskExecutionStatus::Complete);
    rejected.status = LlmMeasurementStatus::Failed;
    rejected.reason_code = LlmBackendReason::TASK_COMPLETION_MISMATCH;
    rejected.execution.reason_code = LlmBackendReason::TASK_COMPLETION_MISMATCH;

    const OrderedJson doc = build_cpu_doc(config, plan, result);
    const OrderedJson& execution = doc["measurements"][0]["execution"];
    expect_string_values(execution,
                         {{"status", "invalid"}, {"reason_code", LlmBackendReason::TASK_COMPLETION_MISMATCH}});
    EXPECT_FALSE(execution["valid"]);
  }
  {
    SCOPED_TRACE("required_write_validation_not_run");
    const auto doc = run_cpu_doc(
        explicit_config(1),
        [](const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan& scenario, const LlmRunnerTaskContext& context) {
          auto execution = successful_execution(model);
          if (context.kind == LlmRunnerTaskKind::Measurement && scenario.scenario == LlmScenario::KvOnly) {
            execution.post_validation_evaluated = false;
          }
          return execution;
        },
        EXIT_FAILURE);
    const auto& m = doc["measurements"][1];
    expect_named_check(m["execution"], "kv-append-final", true, false, std::nullopt);
    EXPECT_EQ(m["status"], "failed");
    expect_null_fields(m, {"synthetic_memory_work_units_per_second", "effective_model_payload_gb_s"});
  }
}

TEST(LlmMemoryJsonTest, LoopSnapshotSerializesPartialStatusAndUnavailableTailContract) {
  for (bool failed : {false, true}) {
    SCOPED_TRACE(failed);
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend backend;
    LlmMemoryResult partial_snapshot;
    bool captured = false;
    LlmRunnerHooks hooks;
    hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
      if (!captured && kind == LlmCheckpointKind::MeasurementTerminal) {
        partial_snapshot = snapshot;
        captured = true;
      }
      return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    };
    LlmMemoryResult result;
    ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks), failed ? EXIT_FAILURE : EXIT_SUCCESS);
    ASSERT_TRUE(captured);

    if (failed) partial_snapshot = result;
    partial_snapshot.prior_file_writer_attempts = 3;
    partial_snapshot.prior_successful_file_writes = 2;
    const OrderedJson doc = build_cpu_doc(config, plan, partial_snapshot);
    expect_string_values(
        doc, {{"status", failed ? "failed" : "partial"},
              {"reason_code", failed ? LlmRunnerReason::CHECKPOINT_WRITE_FAILED : LlmRunnerReason::PARTIAL_RESULTS}});
    EXPECT_FALSE(doc["interruption_requested"]);
    EXPECT_FALSE(doc["results_complete"]);
    EXPECT_FALSE(doc["run_accepted"]);
    EXPECT_FALSE(doc["scenario_order_balance_complete"]);
    EXPECT_EQ(doc["counters"]["planned_measurements"], 6u);
    EXPECT_EQ(doc["counters"]["attempted_measurements"], 3u);
    EXPECT_EQ(doc["counters"]["terminal_measurements"], failed ? 6u : 3u);
    EXPECT_EQ(doc["counters"]["measured_measurements"], 3u);
    const OrderedJson& lc = doc["checkpoint_lifecycle"];
    if (!failed) EXPECT_EQ(lc["current_request"], "progress");
    EXPECT_EQ(lc["checkpoint_failed"], failed);
    expect_string_values(lc, {{"observation_point", "before-current-snapshot-preparation"},
                              {"checkpoint_policy", "bounded-loop-snapshots"}});
    EXPECT_EQ(lc["prior_file_writer_attempts"], 3u);
    EXPECT_EQ(lc["prior_successful_file_writes"], 2u);
    EXPECT_EQ(lc["snapshot_interval_loops"], 1u);
    EXPECT_TRUE(lc["current_persistence_success"].is_null());

    size_t measured_count = 0, not_run_count = 0;
    for (const OrderedJson& m : doc["measurements"]) {
      if (m["status"] == "measured") {
        ++measured_count;
        EXPECT_EQ(m["reason_code"], "measured");
        EXPECT_TRUE(m["attempted"]);
        EXPECT_TRUE(m["completed_work_units"].is_number_unsigned());
        EXPECT_TRUE(m["elapsed_seconds"].is_number());
        EXPECT_EQ(m["execution"]["status"], "valid");
        EXPECT_EQ(m["checksum"]["status"], "valid");
        EXPECT_TRUE(m["checksum"]["checksum_valid"]);
        continue;
      }
      ASSERT_EQ(m["status"], failed ? "failed" : "not_run");
      ++not_run_count;
      if (!failed) EXPECT_EQ(m["reason_code"], "not-run");
      EXPECT_FALSE(m["attempted"]);
      EXPECT_EQ(m["completed_work_units"], 0u);
      EXPECT_EQ(m["completed_effective_model_payload_bytes"], "0");
      expect_null_fields(m, {"elapsed_seconds", "effective_model_payload_gb_s"});
      EXPECT_EQ(m["execution"]["status"], "not_run");
      if (!failed) EXPECT_EQ(m["execution"]["reason_code"], "not-run");
      expect_null_fields(m["execution"], {"valid", "requested_workers", "qos_successful_workers"});
      EXPECT_EQ(m["checksum"]["status"], "not_evaluated");
      if (!failed) EXPECT_EQ(m["checksum"]["reason_code"], "not-run");
      EXPECT_TRUE(m["checksum"]["checksum_valid"].is_null());
    }
    EXPECT_EQ(measured_count, 3u);
    EXPECT_EQ(not_run_count, 3u);
  }
}

TEST(LlmMemoryJsonTest, QualityWarningsMergeAndDeduplicateInStableConsoleAgreementOrder) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMemoryResult result = complete_result(config, plan);
  result.quality_warnings = {"mixed-high-cv", "environment-not-nominal", "mixed-high-cv"};
  for (LlmMeasurementState& m : result.measurements) m.duration_quality = "above-target-single-work-unit";
  std::get<LlmCpuRetainedEvidence>(result.measurements[0].execution.backend_evidence).executor.qos_failed_workers = 1;

  LlmResultMetadata metadata = fixed_metadata(config, plan);
  metadata.environment_end.thermal_state = "serious";
  metadata.main_thread_qos = {true, false, 1};
  metadata.l2_data_cache_bytes = 2 * Constants::BYTES_PER_MB;

  const OrderedJson doc = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);
  EXPECT_EQ(json_string_array(doc["quality_warnings"]),
            (std::vector<std::string>{
                "mixed-high-cv", "environment-not-nominal", "main-thread-qos-not-applied", "worker-qos-not-applied",
                "weight-working-set-cache-dominant", "kv-working-set-cache-dominant",
                "weights_only-duration-above-target-single-work-unit", "kv_only-duration-above-target-single-work-unit",
                "mixed-duration-above-target-single-work-unit"}));
}

TEST(LlmMemoryJsonTest, AllEightProfilesPublishExactSchemaTwoSelectorsAndNamedKinds) {
  using ConfigFactory = LlmMemoryConfig (*)();
  const std::array<std::pair<ConfigFactory, const char*>, 8> profiles = {{
      {+[]() { return explicit_config(1); }, "llm-memory-v2-cpu-decode-contiguous"},
      {+[]() { return paged_config(1); }, "llm-memory-v2-cpu-decode-paged"},
      {+[]() { return prefill_config(1); }, "llm-memory-v2-cpu-prefill-contiguous"},
      {+[]() { return paged_prefill_config(1); }, "llm-memory-v2-cpu-prefill-paged"},
      {explicit_metal_config, "llm-memory-v2-metal-decode-contiguous"},
      {explicit_metal_paged_config, "llm-memory-v2-metal-decode-paged"},
      {explicit_metal_prefill_config, "llm-memory-v2-metal-prefill-contiguous"},
      {explicit_metal_paged_prefill_config, "llm-memory-v2-metal-prefill-paged"},
  }};
  for (const auto& [factory, selector] : profiles) {
    SCOPED_TRACE(selector);
    const auto config = factory();
    const bool metal = config.backend == LlmMemoryBackend::Metal;
    const auto plan = metal ? admitted_metal_plan(config) : admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const auto result = metal ? metal_result_with_measurement_and_calibration(plan) : complete_result(config, plan);
    const auto doc = metal ? build_llm_memory_json(config, plan, complete_metal_backend_evidence(plan),
                                                   fixed_metadata(config, plan), result)
                           : build_cpu_doc(config, plan, result);
    EXPECT_EQ(doc["schema_version"], 2);
    EXPECT_EQ(doc["methodology_version"], selector);
    EXPECT_EQ(doc["resolved_plan"]["component_identities"]["run_policy_version"],
              "llm-run-policy-bounded-loop-snapshots-v1");
    const auto& measurement = doc["measurements"][metal ? 0 : 1];
    const char* write_kind = config.phase == LlmPhase::Decode ? "kv-append-final" : "kv-prefill-final-samples";
    EXPECT_TRUE(named_check(measurement["execution"], write_kind)["valid"]);
    EXPECT_TRUE(named_check(measurement["execution"], "post-validation-structure")["valid"]);
    const auto& expected = referenced_plan(doc, measurement)["expected_checksum"];
    EXPECT_EQ(expected["status"], "available");
    if (metal) EXPECT_TRUE(expected["expected_worker_checksums"].is_null());
    else ASSERT_EQ(expected["expected_worker_checksums"].size(), 2u);
    if (!metal && config.kv_layout == LlmKvLayout::Paged && config.phase == LlmPhase::Decode) {
      expect_named_check(doc["measurements"][0]["execution"], "kv-append-unchanged", true, true, true);
    }
  }
}

TEST(LlmMemoryJsonTest, OriginalCpuTimingAndCommandManifestSurviveProjection) {
  const auto config = explicit_config(1);
  const auto plan = admitted_plan(config);
  auto result = complete_result(config, plan);
  auto metadata = fixed_metadata(config, plan);
  metadata.build_manifest["binary_sha256"] = std::string(64, 'a');
  metadata.build_manifest["status"] = "partial";
  const MachTimingSnapshot raw{100, 103, 3, 125, 3};
  result.measurements[0].execution.timing.cpu_raw = raw;
  result.calibration_attempts[0][0].execution.cpu_raw = raw;
  const auto doc = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);
  EXPECT_EQ(doc["build_manifest"], metadata.build_manifest);
  const auto& captured = doc["measurements"][0]["execution"]["timing"]["cpu_raw"];
  expect_string_values(captured, {{"start_ticks", "100"}, {"stop_ticks", "103"}, {"delta_ticks", "3"}});
  EXPECT_EQ(captured["timebase_numer"], 125);
  EXPECT_EQ(captured["timebase_denom"], 3);
  EXPECT_EQ(doc["calibration"]["attempts"]["weights_only"][0]["execution"]["timing"]["cpu_raw"], captured);
  EXPECT_TRUE(doc["measurements"][1]["execution"]["timing"]["cpu_raw"].is_null());
}
