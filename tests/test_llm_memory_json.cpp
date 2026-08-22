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
#include <stdexcept>
#include <string>
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
  if (cpu_plan == nullptr) {
    throw std::logic_error("expected CPU execution plan");
  }
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
  config.argv = {
      "memory_benchmark",          "--llm-memory", "--kv-layout", "paged", "--kv-block-tokens", "2", "--output",
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
  config.argv = {"memory_benchmark",
                 "--llm-memory",
                 "--phase",
                 "prefill",
                 "--prompt-tokens",
                 "5",
                 "--attention-query-tile-tokens",
                 "2",
                 "--output",
                 "--literal-output-name.json"};
  return config;
}

LlmMemoryConfig paged_prefill_config(size_t loop_count = 3) {
  LlmMemoryConfig config = prefill_config(loop_count);
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  config.argv = {"memory_benchmark",
                 "--llm-memory",
                 "--phase",
                 "prefill",
                 "--prompt-tokens",
                 "5",
                 "--attention-query-tile-tokens",
                 "2",
                 "--kv-layout",
                 "paged",
                 "--kv-block-tokens",
                 "2",
                 "--output",
                 "--literal-output-name.json"};
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
  if (!draft.valid) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }

  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(draft.auxiliary_preflight);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  const LlmJsonPeakEstimate json_peak = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
  size_t runner_and_executor_orchestration_bytes = 0;
  if (!executor.valid || !runner.valid || !json_peak.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 runner_and_executor_orchestration_bytes) ||
      !NumericUtils::checked_add(runner_and_executor_orchestration_bytes, json_peak.total_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  return finalize_llm_memory_work_plan(std::move(draft), request.checksum_auxiliary_bytes,
                                       request.orchestration_auxiliary_bytes);
}

LlmMemoryConfig explicit_metal_config() {
  LlmMemoryConfig config = explicit_config(1);
  config.backend = LlmMemoryBackend::Metal;
  config.weight_size_mb = 257;
  config.requested_workers = 0;
  config.available_workers = 0;
  config.user_specified_backend = true;
  config.user_specified_workers = false;
  config.argv = {"memory_benchmark",
                 "--llm-memory",
                 "--llm-memory-backend",
                 "metal",
                 "--weight-size-mb",
                 "257",
                 "--output",
                 "--literal-output-name.json"};
  return config;
}

LlmMemoryWorkPlan admitted_metal_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = plan_request(config);
  request.backend = LlmMemoryBackend::Metal;
  request.requested_workers = 0;
  request.available_workers = 0;
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(request);
  if (!draft.valid) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }

  const auto resource_request = [&](size_t additional_owned_bytes) {
    LlmMetalResourcePlanRequest metal_request;
    metal_request.geometry = draft.candidate.geometry;
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
  const LlmBackendAuxiliaryEstimate backend_auxiliary =
      backend == nullptr ? LlmBackendAuxiliaryEstimate{}
                         : backend->calculate_auxiliary_estimate(draft.auxiliary_preflight);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  const LlmJsonPeakEstimate json_peak = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
  size_t checksum_auxiliary_bytes = 0;
  size_t runner_and_backend_orchestration_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  size_t command_auxiliary_bytes = 0;
  size_t additional_owned_bytes = 0;
  if (!backend_auxiliary.valid || !runner.valid || !json_peak.valid ||
      !NumericUtils::checked_add(backend_auxiliary.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(backend_auxiliary.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 runner_and_backend_orchestration_bytes) ||
      !NumericUtils::checked_add(runner_and_backend_orchestration_bytes, json_peak.total_bytes,
                                 orchestration_auxiliary_bytes) ||
      !NumericUtils::checked_add(checksum_auxiliary_bytes, orchestration_auxiliary_bytes, command_auxiliary_bytes) ||
      !NumericUtils::checked_add(draft.candidate.memory_budget.request.planner_storage_bytes, command_auxiliary_bytes,
                                 additional_owned_bytes)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }

  LlmMetalExecutionPlan execution = build_llm_metal_execution_plan(resource_request(additional_owned_bytes));
  return finalize_llm_memory_work_plan(std::move(draft), std::move(execution), checksum_auxiliary_bytes,
                                       orchestration_auxiliary_bytes);
}

LlmMetalTaskEvidence complete_metal_task_evidence() {
  LlmMetalTaskEvidence task;
  task.timed_pipeline_available = true;
  task.pipeline_label = "membenchmark.llm-metal.pipeline.decode-contiguous.mixed";
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
  task.grid_plan.minimum_threadgroup_accounted_bytes = 1024;
  task.grid_plan.maximum_threadgroup_accounted_bytes = 2048;
  task.grid_plan.threadgroup_accounted_imbalance_bytes = 1024;
  task.grid_plan.threadgroup_accounted_bytes = {1024, 2048};
  task.grid_plan.identity = "metal-grid-v1-test";
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
  task.expected_checksum = {{1, 2}, {3, 4}, {5, 6}};
  task.actual_checksum = task.expected_checksum;
  task.append_validation_evaluated = true;
  task.append_validation_valid = true;
  task.padding_canary_applicable = false;
  task.post_validation_evaluated = true;
  task.post_validation_valid = true;
  return task;
}

LlmBackendEvidence complete_metal_backend_evidence(const LlmMemoryWorkPlan& plan) {
  LlmBackendEvidence backend;
  backend.backend = LlmMemoryBackend::Metal;
  backend.initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  backend.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  backend.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  backend.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};

  LlmMetalBackendEvidence metal;
  metal.capability.device_name = "Test Metal Device";
  metal.capability.registry_id = kAboveJsonExactInteger;
  metal.capability.has_unified_memory = true;
  metal.capability.required_apple7_family_supported = true;
  metal.capability.argument_buffers_tier2_supported = true;
  metal.capability.supported_families = {"apple7", "apple8"};
  metal.capability.max_buffer_length = 2 * Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  metal.capability.recommended_max_working_set_size = 6ULL * 1024ULL * Constants::BYTES_PER_MB;
  metal.capability.kernel_revision = "llm-metal-decode-contiguous-msl23-v1";
  metal.capability.kernel_source_sha256 = canonical_llm_metal_kernel_source_sha256();
  metal.capability.argument_buffer_encoded_length = 8192;
  metal.capability.argument_buffer_alignment = 256;
  metal.capability.layout_probe_evaluated = true;
  metal.capability.layout_probe_valid = true;
  metal.capability.layout_probe_resource_count = 7;
  metal.capability.foundation_pipelines = {{"membenchmark.llm-metal.foundation.fill", 32, 1024}};
  metal.workload_pipelines = {{"membenchmark.llm-metal.pipeline.decode-contiguous.weights-only", 32, 1024},
                              {"membenchmark.llm-metal.pipeline.decode-contiguous.kv-only", 32, 1024},
                              {"membenchmark.llm-metal.pipeline.decode-contiguous.mixed", 32, 1024}};
  metal.timed_results_available = true;

  const LlmMetalExecutionPlan* const execution = get_llm_metal_execution_plan(plan);
  if (execution != nullptr) {
    metal.resources.allocation_attempted = true;
    metal.resources.allocation_completed = true;
    metal.resources.initialization_completed = true;
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

LlmMemoryResult metal_result_with_measurement_and_calibration() {
  LlmMemoryResult result;
  result.initialized = true;
  result.status = LlmRunStatus::Complete;
  result.reason_code = LlmBackendReason::VALID;
  result.results_complete = true;
  result.conclusions_valid = true;

  LlmMetalTaskEvidence task = complete_metal_task_evidence();
  LlmMeasurementState measurement;
  measurement.scenario = LlmScenario::Mixed;
  measurement.work_unit_kind = LlmWorkUnitKind::DecodeStep;
  measurement.kv_write_kind = LlmKvWriteKind::CurrentTokenAppend;
  measurement.status = LlmMeasurementStatus::Measured;
  measurement.reason_code = LlmBackendReason::VALID;
  measurement.attempted = true;
  measurement.execution_evidence_available = true;
  measurement.execution.status = LlmTaskExecutionStatus::Complete;
  measurement.execution.reason_code = LlmBackendReason::VALID;
  measurement.execution.timing = {true, true, 0.0025};
  measurement.execution.validation = {true, true};
  measurement.execution.backend_evidence = task;
  measurement.elapsed_seconds = 0.0025;
  measurement.checksum_valid = true;
  result.measurements.push_back(std::move(measurement));

  task.queue_delay_available = false;
  LlmCalibrationAttempt attempt;
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
  attempt.execution.metal = std::move(task);
  result.calibration_attempts[static_cast<size_t>(LlmScenario::Mixed)].push_back(std::move(attempt));
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
  for (size_t worker_index = 0; worker_index < cpu_plan.effective_workers; ++worker_index) {
    execution.expected_checksums[worker_index] = {checksum_component(worker_index * 3),
                                                  checksum_component(worker_index * 3 + 1),
                                                  checksum_component(worker_index * 3 + 2)};
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

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    const LlmExecutorAuxiliaryEstimate cpu = calculate_llm_executor_auxiliary_estimate(model_plan);
    LlmBackendAuxiliaryEstimate estimate;
    estimate.valid = cpu.valid;
    estimate.reason_code = cpu.reason_code;
    estimate.checksum_auxiliary_bytes = cpu.checksum_auxiliary_bytes;
    estimate.orchestration_auxiliary_bytes = cpu.orchestration_auxiliary_bytes;
    estimate.total_auxiliary_bytes = cpu.total_auxiliary_bytes;
    estimate.backend_evidence = cpu;
    return estimate;
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

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model_plan) noexcept override {
    evidence_.backend_evidence = LlmCpuBackendEvidence{preparation_for(model_plan)};
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) override {
    LlmExecutorResult execution =
        executor_ ? executor_(model_plan, scenario_plan, context) : successful_execution(model_plan);
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

std::vector<std::string> json_string_array(const OrderedJson& array) {
  std::vector<std::string> output;
  for (const OrderedJson& value : array) {
    output.push_back(value.get<std::string>());
  }
  return output;
}

void expect_exact_keys(const OrderedJson& object, std::initializer_list<const char*> keys) {
  ASSERT_TRUE(object.is_object());
  EXPECT_EQ(object.size(), keys.size());
  for (const char* key : keys) {
    EXPECT_TRUE(object.contains(key)) << key;
  }
}

}  // namespace

TEST(LlmMemoryJsonTest, MetalDocumentPublishesSegmentationBackendTaskAndCompactEvidence) {
  const LlmMemoryConfig config = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMetalExecutionPlan* const metal_plan = get_llm_metal_execution_plan(plan);
  ASSERT_NE(metal_plan, nullptr);
  ASSERT_TRUE(metal_plan->valid) << metal_plan->reason_code;
  ASSERT_EQ(metal_plan->resources.weight_segments.segment_count, 2U);

  const LlmBackendEvidence backend = complete_metal_backend_evidence(plan);
  const LlmMemoryResult result = metal_result_with_measurement_and_calibration();
  const LlmResultMetadata metadata = fixed_metadata(config, plan);
  const OrderedJson document = build_llm_memory_json(config, plan, backend, metadata, result);

  EXPECT_EQ(document["backend"], "metal");
  EXPECT_EQ(document["methodology_version"], "llm-memory-v1-metal-decode-contiguous");
  EXPECT_EQ(document["configuration"]["resolved_sources"]["backend"], "explicit");
  EXPECT_TRUE(document["configuration"]["requested_workers"].is_null());
  EXPECT_TRUE(document["configuration"]["available_workers"].is_null());
  EXPECT_TRUE(document["configuration"]["worker_source"].is_null());
  EXPECT_TRUE(document["configuration"]["resolved_sources"]["workers"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["model_work_plan"]["effective_workers"].is_null());

  const OrderedJson& resources = document["resolved_plan"]["resources"]["metal"];
  ASSERT_TRUE(resources.is_object());
  EXPECT_EQ(resources["segment_capacity_bytes"], "268435456");
  EXPECT_EQ(resources["weight_segments"]["segment_count"], 2U);
  EXPECT_EQ(resources["weight_segments"]["total_length_bytes"], "269484032");
  ASSERT_EQ(resources["weight_segments"]["segment_lengths_bytes"].size(), 2U);
  EXPECT_EQ(resources["weight_segments"]["segment_lengths_bytes"][0], "268435456");
  EXPECT_EQ(resources["weight_segments"]["segment_lengths_bytes"][1], "1048576");
  EXPECT_EQ(resources["argument_buffer"]["encoded_length_bytes"], "8192");
  EXPECT_TRUE(resources["table_segments"].is_null());

  EXPECT_TRUE(document["backend_evidence"]["cpu"].is_null());
  const OrderedJson& metal_backend = document["backend_evidence"]["metal"];
  ASSERT_TRUE(metal_backend.is_object());
  EXPECT_FALSE(metal_backend["workers_applicable"].get<bool>());
  EXPECT_FALSE(metal_backend["worker_qos_applicable"].get<bool>());
  EXPECT_EQ(metal_backend["lifecycle"]["initialization"]["status"], "ready");
  EXPECT_EQ(metal_backend["capability"]["device_name"], "Test Metal Device");
  EXPECT_EQ(metal_backend["capability"]["registry_id_uint64_decimal"], "9007199254740993");
  EXPECT_EQ(metal_backend["workload_pipelines"].size(), 3U);
  EXPECT_EQ(metal_backend["workload_pipelines"][2]["label"], "membenchmark.llm-metal.pipeline.decode-contiguous.mixed");
  EXPECT_TRUE(metal_backend["timed_results_available"].get<bool>());

  const OrderedJson& methodology = document["resolved_plan"]["methodology"];
  EXPECT_EQ(methodology["timing_policy"],
            "gpu-start-to-gpu-end-per-command-buffer-task");
  EXPECT_EQ(methodology["cache_policy"],
            "metal-private-and-shared-cacheable-no-explicit-flush-between-scenarios");
  EXPECT_EQ(methodology["maximum_work_units_per_measurement"],
            Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH);
  EXPECT_EQ(document["memory_budget"]["resource_rounding_bytes"], "4096");
  EXPECT_EQ(document["memory_budget"]["transient_peak_bytes"],
            std::to_string(metal_plan->resources.transient_peak_bytes));
  EXPECT_EQ(document["memory_budget"]["known_owned_peak_bytes"],
            std::to_string(metal_plan->resources.known_owned_peak_bytes +
                           4096));
  EXPECT_EQ(document["memory_budget"]["admitted_budget_bytes"],
            std::to_string(metal_plan->resources.admitted_budget_bytes));

  ASSERT_EQ(document["measurements"].size(), 1U);
  EXPECT_TRUE(document["measurements"][0]["requested_workers"].is_null());
  EXPECT_TRUE(document["measurements"][0]["effective_workers"].is_null());
  EXPECT_TRUE(document["measurements"][0]["qos_successful_workers"].is_null());
  EXPECT_TRUE(document["measurements"][0]["qos_failed_workers"].is_null());
  const OrderedJson& execution = document["measurements"][0]["execution"];
  EXPECT_TRUE(execution["requested_workers"].is_null());
  EXPECT_TRUE(execution["qos_successful_workers"].is_null());
  EXPECT_DOUBLE_EQ(execution["elapsed_seconds"].get<double>(), 0.0025);
  const OrderedJson& task = execution["metal"];
  EXPECT_EQ(task["pipeline"]["label"], "membenchmark.llm-metal.pipeline.decode-contiguous.mixed");
  EXPECT_EQ(task["grid"]["actual_threadgroups"], 2U);
  EXPECT_DOUBLE_EQ(task["timing"]["gpu_elapsed_seconds"].get<double>(), 0.0025);
  EXPECT_DOUBLE_EQ(task["timing"]["queue_delay_seconds"].get<double>(), 0.0005);
  EXPECT_EQ(task["commands"]["reset_status"], "completed");
  EXPECT_EQ(task["commands"]["timed_status"], "completed");
  EXPECT_EQ(task["commands"]["post_validation_status"], "completed");
  EXPECT_EQ(task["commands"]["timed_workload_dispatches"], 1U);
  EXPECT_EQ(task["checksum"]["algorithm_version"], "llm-metal-dual-mod32-v1");
  EXPECT_EQ(task["checksum"]["expected"]["weight"]["a_uint32_decimal"], "1");
  EXPECT_TRUE(task["validation"]["append_valid"].get<bool>());
  EXPECT_FALSE(task["validation"]["padding_canary_applicable"].get<bool>());
  EXPECT_TRUE(task["validation"]["padding_canary_evaluated"].is_null());

  const OrderedJson& checksum = document["measurements"][0]["checksum"];
  EXPECT_EQ(checksum["status"], "valid");
  EXPECT_EQ(checksum["checksum_pattern_version"], "llm-metal-dual-mod32-v1");
  EXPECT_TRUE(checksum["expected_worker_checksums"].is_null());
  EXPECT_EQ(checksum["actual_run_checksum"]["v"]["b_uint32_decimal"], "6");

  const OrderedJson& compact = document["calibration"]["attempts"]["mixed"][0]["execution"];
  EXPECT_TRUE(compact["requested_workers"].is_null());
  EXPECT_EQ(compact["checksum"]["algorithm_version"], "llm-metal-dual-mod32-v1");
  EXPECT_EQ(compact["checksum"]["expected_run_checksum"]["k"]["b_uint32_decimal"], "4");
  EXPECT_EQ(compact["metal"]["commands"]["timed_status"], "completed");
  EXPECT_TRUE(compact["metal"]["timing"]["queue_delay_seconds"].is_null());
}

TEST(LlmMemoryJsonTest,
     MetalMemoryBudgetUsesActualValuesOnlyAfterCompleteAllocationAndRejectsCommittedOverage) {
  const LlmMemoryConfig config = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = metal_result_with_measurement_and_calibration();
  const LlmResultMetadata metadata = fixed_metadata(config, plan);

  LlmBackendEvidence partial_backend = complete_metal_backend_evidence(plan);
  auto* partial_metal =
      std::get_if<LlmMetalBackendEvidence>(&partial_backend.backend_evidence);
  ASSERT_NE(partial_metal, nullptr);
  partial_metal->resources.allocation_completed = false;
  partial_metal->resources.resource_rounding_bytes = 0;
  partial_metal->resources.transient_peak_bytes = 0;
  partial_metal->resources.known_owned_peak_bytes = 0;
  partial_metal->resources.admitted_budget_bytes = 0;
  const OrderedJson partial_document =
      build_llm_memory_json(config, plan, partial_backend, metadata, result);
  EXPECT_EQ(partial_document["memory_budget"]["known_owned_peak_bytes"],
            std::to_string(plan.memory_budget.request.required_total_bytes));
  EXPECT_EQ(partial_document["memory_budget"]["admitted_budget_bytes"],
            std::to_string(plan.memory_budget.allowed_memory_bytes));
  EXPECT_TRUE(partial_document["memory_budget"]["valid"].get<bool>());
  EXPECT_EQ(partial_document["memory_budget"]["reason_code"],
            plan.memory_budget.reason_code);

  LlmBackendEvidence overage_backend = complete_metal_backend_evidence(plan);
  auto* overage_metal =
      std::get_if<LlmMetalBackendEvidence>(&overage_backend.backend_evidence);
  ASSERT_NE(overage_metal, nullptr);
  overage_metal->resources.known_owned_peak_bytes =
      overage_metal->resources.admitted_budget_bytes + 1;
  const OrderedJson overage_document =
      build_llm_memory_json(config, plan, overage_backend, metadata, result);
  EXPECT_FALSE(overage_document["memory_budget"]["valid"].get<bool>());
  EXPECT_EQ(overage_document["memory_budget"]["reason_code"],
            LlmBackendReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_EQ(overage_document["memory_budget"]["known_owned_peak_bytes"],
            std::to_string(overage_metal->resources.known_owned_peak_bytes));
}

TEST(LlmMemoryJsonTest, UnsupportedMetalCapabilityRetainsTerminalLifecycleWithoutCpuEvidence) {
  const LlmMemoryConfig config = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmBackendEvidence backend = complete_metal_backend_evidence(plan);
  backend.initialization = {LlmBackendStatus::Unsupported, LlmBackendReason::APPLE7_FAMILY_REQUIRED};
  backend.plan_resolution = {};
  backend.preparation = {};
  LlmMetalBackendEvidence* const metal = std::get_if<LlmMetalBackendEvidence>(&backend.backend_evidence);
  ASSERT_NE(metal, nullptr);
  metal->capability.required_apple7_family_supported = false;
  metal->timed_results_available = false;

  LlmMemoryResult result;
  result.initialized = true;
  result.status = LlmRunStatus::Unsupported;
  result.reason_code = LlmBackendReason::APPLE7_FAMILY_REQUIRED;
  const OrderedJson document = build_llm_memory_json(config, plan, backend, fixed_metadata(config, plan), result);

  EXPECT_EQ(document["status"], "unsupported");
  EXPECT_EQ(document["reason_code"], LlmBackendReason::APPLE7_FAMILY_REQUIRED);
  EXPECT_TRUE(document["backend_evidence"]["cpu"].is_null());
  EXPECT_EQ(document["backend_evidence"]["metal"]["lifecycle"]["initialization"]["status"], "unsupported");
  EXPECT_EQ(document["backend_evidence"]["metal"]["lifecycle"]["initialization"]["reason_code"],
            LlmBackendReason::APPLE7_FAMILY_REQUIRED);
  EXPECT_FALSE(document["backend_evidence"]["metal"]["capability"]["required_apple7_family_supported"].get<bool>());
}

TEST(LlmMemoryJsonTest, FailedMetalRuntimeSetupRetainsTerminalSchemaWithoutCpuFallback) {
  const LlmMemoryConfig config = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmBackendEvidence backend = complete_metal_backend_evidence(plan);
  backend.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
  backend.preparation = {};
  backend.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  LlmMetalBackendEvidence* const metal = std::get_if<LlmMetalBackendEvidence>(&backend.backend_evidence);
  ASSERT_NE(metal, nullptr);
  metal->timed_results_available = false;

  LlmMemoryResult result;
  result.initialized = true;
  result.status = LlmRunStatus::Failed;
  result.reason_code = LlmBackendReason::METAL_PIPELINE_CREATION_FAILED;
  const OrderedJson document = build_llm_memory_json(config, plan, backend, fixed_metadata(config, plan), result);

  EXPECT_EQ(document["schema_version"], 1);
  EXPECT_EQ(document["mode"], "llm_memory");
  EXPECT_EQ(document["backend"], "metal");
  EXPECT_EQ(document["status"], "failed");
  EXPECT_EQ(document["reason_code"], LlmBackendReason::METAL_PIPELINE_CREATION_FAILED);
  EXPECT_FALSE(document["results_complete"].get<bool>());
  EXPECT_FALSE(document["conclusions_valid"].get<bool>());
  EXPECT_TRUE(document["backend_evidence"]["cpu"].is_null());
  EXPECT_EQ(document["backend_evidence"]["metal"]["lifecycle"]["plan_resolution"]["status"], "failed");
  EXPECT_EQ(document["backend_evidence"]["metal"]["lifecycle"]["plan_resolution"]["reason_code"],
            LlmBackendReason::METAL_PIPELINE_CREATION_FAILED);
}

TEST(LlmMemoryJsonTest, CompleteDocumentHasExactTopLevelIdentityAndAuditableNestedEvidence) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete);
  const LlmResourcePreparationResult preparation = preparation_for(plan);
  const LlmResultMetadata metadata = fixed_metadata(config, plan);

  const OrderedJson document = build_llm_memory_json(config, plan, preparation, metadata, result);

  const std::array<const char*, 28> top_level_keys = {"schema_version",
                                                      "mode",
                                                      "backend",
                                                      "phase",
                                                      "kv_layout",
                                                      "methodology_version",
                                                      "software",
                                                      "configuration",
                                                      "resolved_plan",
                                                      "backend_evidence",
                                                      "memory_budget",
                                                      "calibration",
                                                      "measurements",
                                                      "aggregates",
                                                      "status",
                                                      "reason_code",
                                                      "results_complete",
                                                      "conclusions_valid",
                                                      "interpretation",
                                                      "diagnostic",
                                                      "interruption_requested",
                                                      "scenario_order_balance_complete",
                                                      "seeds",
                                                      "counters",
                                                      "checkpoint_lifecycle",
                                                      "loop_records",
                                                      "environment",
                                                      "quality_warnings"};
  ASSERT_EQ(document.size(), top_level_keys.size());
  for (const char* key : top_level_keys) {
    EXPECT_TRUE(document.contains(key)) << key;
  }

  expect_exact_keys(document["configuration"], {"backend",
                                                "phase",
                                                "kv_layout",
                                                "kv_block_tokens",
                                                "weight_size_mb",
                                                "layer_count",
                                                "query_head_count",
                                                "kv_head_count",
                                                "head_dimension",
                                                "kv_element_bytes",
                                                "visible_context_tokens",
                                                "prompt_tokens",
                                                "attention_query_tile_tokens",
                                                "batch_size",
                                                "requested_workers",
                                                "available_workers",
                                                "worker_source",
                                                "iterations",
                                                "work_policy",
                                                "loop_count",
                                                "base_seed_uint64_decimal",
                                                "seed_source",
                                                "output_file",
                                                "argv",
                                                "resolved_sources"});
  expect_exact_keys(document["configuration"]["resolved_sources"],
                    {"backend", "phase", "kv_layout", "kv_block_tokens", "visible_context_tokens", "prompt_tokens",
                     "attention_query_tile_tokens", "workers", "iterations", "seed"});
  expect_exact_keys(document["resolved_plan"]["methodology"], {"methodology_version",
                                                               "backend",
                                                               "phase",
                                                               "kv_layout",
                                                               "work_unit_kind",
                                                               "weight_passes_per_work_unit",
                                                               "kv_replay_factor",
                                                               "schedule_version",
                                                               "warmup_policy",
                                                               "context_policy",
                                                               "scenario_order_policy",
                                                               "timing_policy",
                                                               "cache_policy",
                                                               "calibration_policy",
                                                               "calibration_target_seconds",
                                                               "calibration_min_seconds",
                                                               "calibration_max_seconds",
                                                               "calibration_max_corrections",
                                                               "calibration_min_pilot_accounted_bytes",
                                                               "maximum_work_units_per_measurement",
                                                               "maximum_accounted_bytes_per_task",
                                                               "repeatability_cv_warning_threshold_pct",
                                                               "calibration_excluded_from_results",
                                                               "timed_region_exclusions",
                                                               "resource_abi_version",
                                                               "buffer_pattern_version",
                                                               "write_pattern_version",
                                                               "checksum_pattern_version"});
  expect_exact_keys(document["resolved_plan"]["geometry"], {"valid",
                                                            "reason_code",
                                                            "phase",
                                                            "work_unit_kind",
                                                            "decode",
                                                            "prefill",
                                                            "attention_kind",
                                                            "active_weight_bytes_per_work_unit",
                                                            "layer_count",
                                                            "query_head_count",
                                                            "kv_head_count",
                                                            "query_heads_per_kv_head",
                                                            "head_dimension",
                                                            "kv_element_bytes",
                                                            "batch_size",
                                                            "kv_vector_bytes",
                                                            "k_or_v_record_bytes_per_layer",
                                                            "kv_record_bytes_per_layer",
                                                            "kv_bytes_per_visible_token",
                                                            "k_or_v_sequence_visible_bytes",
                                                            "k_mapping_bytes",
                                                            "v_mapping_bytes",
                                                            "kv_capacity_bytes",
                                                            "weight_read_bytes_per_work_unit",
                                                            "kv_read_bytes_per_work_unit",
                                                            "kv_write_bytes_per_work_unit",
                                                            "kv_only_effective_model_payload_bytes_per_work_unit",
                                                            "mixed_effective_model_payload_bytes_per_work_unit",
                                                            "total_data_mapping_bytes",
                                                            "traffic_crossover_numerator",
                                                            "traffic_crossover_denominator",
                                                            "traffic_crossover_context_tokens"});
  expect_exact_keys(document["aggregates"]["traffic_diagnostics"],
                    {"classification_version", "traffic_crossover_numerator", "traffic_crossover_denominator",
                     "traffic_crossover_context_tokens", "current_visible_context_tokens",
                     "current_weight_read_payload_bytes_per_work_unit", "current_kv_read_payload_bytes_per_work_unit",
                     "current_weight_to_kv_read_payload_ratio", "current_context_classification",
                     "classification_is_payload_only", "scenario_headlines"});
  expect_exact_keys(document["aggregates"]["traffic_diagnostics"]["scenario_headlines"]["mixed"],
                    {"synthetic_work_unit_latency_seconds", "synthetic_memory_work_units_per_second",
                     "effective_model_payload_gb_s"});
  expect_exact_keys(document["memory_budget"],
                    {"resource_rounding_bytes", "transient_peak_bytes", "layout_transient_bytes", "setup_peak_bytes",
                     "runtime_peak_bytes", "known_owned_peak_bytes", "admitted_budget_bytes", "valid", "reason_code",
                     "request", "available_memory_bytes", "allowed_memory_bytes", "used_fallback"});
  expect_exact_keys(document["memory_budget"]["request"], {"valid",
                                                           "reason_code",
                                                           "mapping_granularity_bytes",
                                                           "requested_weight_mapping_bytes",
                                                           "requested_k_mapping_bytes",
                                                           "requested_v_mapping_bytes",
                                                           "committed_weight_mapping_bytes",
                                                           "committed_k_mapping_bytes",
                                                           "committed_v_mapping_bytes",
                                                           "requested_block_table_mapping_bytes",
                                                           "committed_block_table_mapping_bytes",
                                                           "requested_data_bytes",
                                                           "committed_data_bytes",
                                                           "layout_transient_bytes",
                                                           "setup_peak_bytes",
                                                           "runtime_peak_bytes",
                                                           "descriptor_bytes",
                                                           "planner_storage_bytes",
                                                           "checksum_auxiliary_bytes",
                                                           "orchestration_auxiliary_bytes",
                                                           "auxiliary_bytes",
                                                           "required_total_bytes"});
  expect_exact_keys(document["backend_evidence"], {"cpu", "metal"});
  ASSERT_TRUE(document["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(document["backend_evidence"]["metal"].is_null());
  expect_exact_keys(document["backend_evidence"]["cpu"],
                    {"requested_workers", "available_workers", "effective_workers", "resource_abi_version",
                     "schedule_version", "timer_policy_version", "prefill", "paged", "resources"});
  EXPECT_TRUE(document["backend_evidence"]["cpu"]["prefill"].is_null());
  EXPECT_TRUE(document["backend_evidence"]["cpu"]["paged"].is_null());
  const OrderedJson& cpu_resources = document["backend_evidence"]["cpu"]["resources"];
  expect_exact_keys(cpu_resources,
                    {"valid", "reason_code", "model_plan_identity", "mappings", "descriptors", "executor_auxiliary",
                     "json_output_peak_estimate", "allocation_memory_budget", "initialization"});
  expect_exact_keys(cpu_resources["mappings"], {"policy", "full_size_physical_mappings", "weight", "k", "v",
                                                "block_table", "requested_data_bytes", "committed_data_bytes"});
  EXPECT_TRUE(cpu_resources["mappings"]["block_table"].is_null());
  expect_exact_keys(cpu_resources["mappings"]["weight"], {"requested_bytes", "committed_bytes"});
  expect_exact_keys(cpu_resources["descriptors"],
                    {"abi_version", "layer_descriptors_per_worker", "sequence_descriptors_per_worker",
                     "total_layer_descriptors", "total_sequence_descriptors", "descriptor_bytes"});
  expect_exact_keys(cpu_resources["executor_auxiliary"],
                    {"valid", "reason_code", "static_reference_bytes", "expected_checksum_bytes",
                     "actual_checksum_bytes", "run_checksum_bytes", "worker_status_bytes", "thread_handle_bytes",
                     "checksum_auxiliary_bytes", "orchestration_auxiliary_bytes", "total_auxiliary_bytes"});
  expect_exact_keys(cpu_resources["json_output_peak_estimate"],
                    {"enabled", "valid", "reason_code", "policy", "fixed_schema_bytes", "input_string_bytes",
                     "measurement_record_bytes", "worker_checksum_bytes", "total_bytes"});
  expect_exact_keys(
      cpu_resources["initialization"],
      {"complete", "pattern_version", "pre_touch_policy", "separate_reference_read_pass",
       "static_references_accumulated_during_initialization", "weight_bytes", "k_bytes", "v_bytes", "total_bytes",
       "non_empty_weight_spans", "non_empty_k_spans", "non_empty_v_spans", "block_table_logical_bytes",
       "block_table_page_rounded_bytes", "block_table_read_only", "k_layout_padding_bytes", "v_layout_padding_bytes"});
  EXPECT_TRUE(cpu_resources["initialization"]["block_table_logical_bytes"].is_null());
  EXPECT_TRUE(cpu_resources["initialization"]["block_table_page_rounded_bytes"].is_null());
  EXPECT_TRUE(cpu_resources["initialization"]["block_table_read_only"].is_null());
  expect_exact_keys(document["seeds"],
                    {"base_seed_uint64_decimal", "source", "buffer_domain_seeds", "scenario_domain_seeds"});
  expect_exact_keys(document["seeds"]["buffer_domain_seeds"],
                    {"weight_uint64_decimal", "k_uint64_decimal", "v_uint64_decimal"});
  expect_exact_keys(document["seeds"]["scenario_domain_seeds"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(document["resolved_plan"],
                    {"valid", "reason_code", "plan_identity", "methodology_version", "backend", "phase", "kv_layout",
                     "work_unit_kind", "geometry", "layout", "resources", "component_identities", "methodology",
                     "model_work_plan", "frozen_scenario_work_plans"});
  expect_exact_keys(document["resolved_plan"]["geometry"]["decode"], {"visible_context_tokens"});
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["prefill"].is_null());
  expect_exact_keys(
      document["resolved_plan"]["layout"],
      {"kv_layout", "kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer", "total_physical_blocks",
       "block_bytes", "last_block_tokens", "last_block_valid_bytes", "decode_append_offset_in_last_block",
       "block_table_entries", "block_table_bytes", "layout_geometry_identity", "layout_identity",
       "permutation_domain_uint64_hex", "permutation_seed_uint64_decimal", "permutation_algorithm_version",
       "permutation_entry_count", "permutation_sha256", "permutation_identity"});
  expect_exact_keys(
      document["resolved_plan"]["resources"],
      {"weight_logical_bytes", "k_logical_bytes", "v_logical_bytes", "k_physical_length_bytes",
       "v_physical_length_bytes", "k_layout_padding_bytes", "v_layout_padding_bytes", "block_table_bytes"});
  expect_exact_keys(
      document["resolved_plan"]["component_identities"],
      {"logical_profile_version", "kv_layout_version", "permutation_version", "backend_executor_version",
       "resource_abi_version", "schedule_version", "timer_policy_version", "buffer_pattern_version",
       "write_pattern_version", "checksum_pattern_version", "msl_revision", "msl_source_sha256", "identity"});
  expect_exact_keys(document["resolved_plan"]["model_work_plan"], {"valid",
                                                                   "reason_code",
                                                                   "plan_identity",
                                                                   "methodology_version",
                                                                   "backend",
                                                                   "phase",
                                                                   "kv_layout",
                                                                   "work_unit_kind",
                                                                   "component_identity",
                                                                   "weight_passes_per_work_unit",
                                                                   "kv_replay_factor",
                                                                   "requested_workers",
                                                                   "available_workers",
                                                                   "effective_workers",
                                                                   "worker_plan_count",
                                                                   "weight_layer_count",
                                                                   "layer_descriptors_per_worker",
                                                                   "sequence_descriptors_per_worker",
                                                                   "total_layer_descriptors",
                                                                   "total_sequence_descriptors",
                                                                   "descriptor_bytes",
                                                                   "planner_storage_bytes"});
  expect_exact_keys(
      document["resolved_plan"]["frozen_scenario_work_plans"],
      {"valid", "reason_code", "explicit_iterations", "model_plan_identity", "plan_identity", "scenarios"});
  expect_exact_keys(document["resolved_plan"]["frozen_scenario_work_plans"]["scenarios"][0],
                    {"valid",
                     "reason_code",
                     "scenario",
                     "work_unit_kind",
                     "kv_write_kind",
                     "explicit_iterations",
                     "model_plan_identity",
                     "scenario_seed_uint64_decimal",
                     "work_units",
                     "weight_read_bytes_per_work_unit",
                     "kv_read_bytes_per_work_unit",
                     "kv_write_bytes_per_work_unit",
                     "effective_model_payload_bytes_per_work_unit",
                     "layout_metadata_lookup_count_per_work_unit",
                     "layout_metadata_read_bytes_per_work_unit",
                     "accounted_bytes_per_work_unit",
                     "weight_read_bytes",
                     "kv_read_bytes",
                     "kv_write_bytes",
                     "effective_model_payload_bytes",
                     "layout_metadata_lookup_count",
                     "layout_metadata_read_bytes",
                     "task_accounted_bytes",
                     "maximum_work_units_by_work_unit_cap",
                     "maximum_work_units_by_guardrail",
                     "effective_maximum_work_units",
                     "plan_identity"});
  expect_exact_keys(document["calibration"], {"excluded_from_results", "attempts"});
  expect_exact_keys(document["calibration"]["attempts"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(document["calibration"]["attempts"]["weights_only"][0], {"attempt_index",
                                                                             "scenario",
                                                                             "work_unit_kind",
                                                                             "kv_write_kind",
                                                                             "purpose",
                                                                             "explicit_iterations",
                                                                             "work_units",
                                                                             "weight_read_bytes",
                                                                             "kv_read_bytes",
                                                                             "kv_write_bytes",
                                                                             "effective_model_payload_bytes",
                                                                             "layout_metadata_lookup_count",
                                                                             "layout_metadata_read_bytes",
                                                                             "task_accounted_bytes",
                                                                             "work_plan_identity",
                                                                             "duration_quality",
                                                                             "terminal",
                                                                             "valid",
                                                                             "reason_code",
                                                                             "execution"});
  expect_exact_keys(document["calibration"]["attempts"]["weights_only"][0]["execution"],
                    {"status", "reason_code", "valid", "elapsed_seconds", "requested_workers", "created_workers",
                     "completed_workers", "qos_successful_workers", "qos_failed_workers", "worker_startup_failed",
                     "kernel_succeeded", "timer_started", "timer_stopped", "checksum"});
  expect_exact_keys(
      document["calibration"]["attempts"]["weights_only"][0]["execution"]["checksum"],
      {"status", "reason_code", "algorithm_version", "checksum_valid", "expected_run_checksum", "actual_run_checksum"});
  expect_exact_keys(
      document["counters"],
      {"planned_loops", "attempted_loops", "completed_loops", "planned_measurements", "attempted_measurements",
       "terminal_measurements", "measured_measurements", "planned_work_units", "completed_work_units",
       "planned_effective_model_payload_bytes", "completed_effective_model_payload_bytes",
       "planned_layout_metadata_lookup_count", "completed_layout_metadata_lookup_count",
       "planned_layout_metadata_read_bytes", "completed_layout_metadata_read_bytes", "planned_task_accounted_bytes",
       "completed_task_accounted_bytes", "runner_auxiliary"});
  expect_exact_keys(
      document["counters"]["runner_auxiliary"],
      {"valid", "reason_code", "measurement_record_bytes", "loop_record_bytes", "calibration_record_bytes",
       "calibration_identity_bytes", "aggregate_value_bytes", "statistics_workspace_bytes", "warning_record_bytes",
       "fixed_metadata_bytes", "retained_checksum_bytes", "checksum_auxiliary_bytes", "orchestration_auxiliary_bytes",
       "total_auxiliary_bytes"});
  expect_exact_keys(
      document["checkpoint_lifecycle"],
      {"checkpoint_failed", "logical_checkpoint_attempts", "successful_logical_checkpoints",
       "terminal_checkpoint_attempted", "terminal_checkpoint_completed", "checkpoint_policy",
       "file_checkpoint_failure_is_terminal_and_not_retried", "stdout_intermediate_checkpoints_are_lazy"});
  expect_exact_keys(document["loop_records"][0],
                    {"loop_index", "planned_order", "realized_order", "realized_order_count", "measurement_indexes"});
  expect_exact_keys(document["measurements"][0], {"scenario",
                                                  "work_unit_kind",
                                                  "kv_write_kind",
                                                  "loop_index",
                                                  "order_position",
                                                  "status",
                                                  "reason_code",
                                                  "attempted",
                                                  "requested_workers",
                                                  "effective_workers",
                                                  "qos_successful_workers",
                                                  "qos_failed_workers",
                                                  "frozen_plan_index",
                                                  "frozen_work_plan_identity",
                                                  "scenario_seed_uint64_decimal",
                                                  "explicit_iterations",
                                                  "work_policy",
                                                  "duration_quality",
                                                  "calibration_attempt_count",
                                                  "calibration_attempt_indexes",
                                                  "planned_work_units",
                                                  "completed_work_units",
                                                  "weight_read_bytes_per_work_unit",
                                                  "kv_read_bytes_per_work_unit",
                                                  "kv_write_bytes_per_work_unit",
                                                  "effective_model_payload_bytes_per_work_unit",
                                                  "layout_metadata_lookup_count_per_work_unit",
                                                  "layout_metadata_read_bytes_per_work_unit",
                                                  "accounted_bytes_per_work_unit",
                                                  "planned_weight_read_bytes",
                                                  "planned_kv_read_bytes",
                                                  "planned_kv_write_bytes",
                                                  "planned_effective_model_payload_bytes",
                                                  "completed_effective_model_payload_bytes",
                                                  "planned_layout_metadata_lookup_count",
                                                  "completed_layout_metadata_lookup_count",
                                                  "planned_layout_metadata_read_bytes",
                                                  "completed_layout_metadata_read_bytes",
                                                  "planned_task_accounted_bytes",
                                                  "completed_task_accounted_bytes",
                                                  "elapsed_seconds",
                                                  "synthetic_work_unit_latency_seconds",
                                                  "synthetic_memory_work_units_per_second",
                                                  "effective_model_payload_gb_s",
                                                  "weight_payload_fraction",
                                                  "kv_read_payload_fraction",
                                                  "kv_write_payload_fraction",
                                                  "working_set",
                                                  "execution",
                                                  "checksum"});
  expect_exact_keys(document["measurements"][0]["working_set"],
                    {"bytes", "full_size_physical_mappings", "cacheable", "kv_layout", "fixed_visible_context_tokens",
                     "current_token_slot_included"});
  expect_exact_keys(
      document["measurements"][0]["execution"],
      {"status", "reason_code", "valid", "elapsed_seconds", "requested_workers", "created_workers", "completed_workers",
       "qos_successful_workers", "qos_failed_workers", "worker_startup_failed", "kernel_succeeded", "timer_started",
       "timer_stopped", "post_validation_evaluated", "post_validation_valid"});
  expect_exact_keys(document["measurements"][0]["checksum"],
                    {"status", "reason_code", "initialization_pattern_version", "write_pattern_version",
                     "checksum_pattern_version", "checksum_valid", "expected_worker_checksums",
                     "actual_worker_checksums", "expected_run_checksum", "actual_run_checksum"});
  expect_exact_keys(document["measurements"][0]["checksum"]["expected_worker_checksums"][0],
                    {"worker_index", "weight", "k", "v"});
  expect_exact_keys(
      document["measurements"][0]["checksum"]["expected_worker_checksums"][0]["weight"],
      {"state_a_uint64_decimal", "state_b_uint64_decimal", "exact_bytes_read", "span_count_uint64_decimal"});
  expect_exact_keys(document["measurements"][0]["checksum"]["expected_run_checksum"],
                    {"state_a_uint64_decimal", "state_b_uint64_decimal"});
  expect_exact_keys(document["aggregates"], {"scenarios", "traffic_diagnostics"});
  expect_exact_keys(document["aggregates"]["scenarios"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(
      document["aggregates"]["scenarios"]["mixed"],
      {"scenario", "status", "stability_quality", "cv_warning_threshold_pct", "synthetic_work_unit_latency_seconds",
       "synthetic_memory_work_units_per_second", "effective_model_payload_gb_s"});
  expect_exact_keys(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"],
                    {"units", "sample_count", "headline_semantics", "headline", "values", "statistics"});
  expect_exact_keys(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"]["statistics"],
                    {"sample_count", "average", "min", "max", "median", "p90", "p95", "p99", "stddev",
                     "coefficient_of_variation_pct", "median_absolute_deviation"});
  expect_exact_keys(document["environment"],
                    {"processor_name", "macos_version", "performance_core_count", "efficiency_core_count",
                     "logical_core_count", "page_size_bytes", "l1_data_cache_bytes", "l2_data_cache_bytes",
                     "available_memory_bytes", "available_memory_source", "main_thread_qos", "start", "end"});
  expect_exact_keys(document["environment"]["main_thread_qos"], {"requested", "applied", "code"});
  expect_exact_keys(document["environment"]["start"],
                    {"thermal_state", "low_power_mode_available", "low_power_mode_enabled", "physical_memory_bytes"});
  expect_exact_keys(document["interpretation"], {"result_scope",
                                                 "reported_rate",
                                                 "backend",
                                                 "phase",
                                                 "work_unit_kind",
                                                 "transformer_math_included",
                                                 "framework_scheduler_and_dispatch_included",
                                                 "compute_memory_overlap_included",
                                                 "physical_dram_traffic_measured",
                                                 "dram_residency",
                                                 "cache_residency",
                                                 "fixed_context_includes_current_token_slot",
                                                 "kv_layout",
                                                 "payload_semantics",
                                                 "layout_metadata_timed_but_excluded_from_effective_model_payload_gb_s",
                                                 "prefill_transformer_compute_or_ttft_prediction_included",
                                                 "private_metal_storage_implies_separate_vram",
                                                 "cross_backend_performance_distributions_combined",
                                                 "traffic_classification_semantics",
                                                 "full_size_working_set_reduces_but_does_not_prove_dram_residency",
                                                 "comparability_requires"});

  EXPECT_EQ(document["software"]["version"], SOFTVERSION);
  EXPECT_EQ(document["software"]["timestamp"], metadata.timestamp);
  EXPECT_EQ(document["schema_version"], 1);
  EXPECT_EQ(document["mode"], "llm_memory");
  EXPECT_EQ(document["backend"], "cpu");
  EXPECT_EQ(document["phase"], "decode");
  EXPECT_EQ(document["kv_layout"], "contiguous");
  EXPECT_EQ(document["methodology_version"], Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION);
  EXPECT_EQ(document["status"], "complete");
  EXPECT_TRUE(document["results_complete"]);
  EXPECT_TRUE(document["conclusions_valid"]);
  EXPECT_TRUE(document["scenario_order_balance_complete"]);
  EXPECT_TRUE(document["diagnostic"].is_null());

  EXPECT_EQ(document["configuration"]["output_file"], "--literal-output-name.json");
  EXPECT_EQ(document["configuration"]["argv"], config.argv);
  EXPECT_EQ(document["configuration"]["work_policy"], "explicit_fixed_work");
  EXPECT_EQ(document["configuration"]["kv_element_bytes"], "2");
  EXPECT_EQ(document["configuration"]["visible_context_tokens"], 3);
  EXPECT_TRUE(document["configuration"]["prompt_tokens"].is_null());
  EXPECT_TRUE(document["configuration"]["attention_query_tile_tokens"].is_null());
  EXPECT_EQ(document["configuration"]["resolved_sources"]["phase"], "default");
  EXPECT_TRUE(document["configuration"]["resolved_sources"]["prompt_tokens"].is_null());
  EXPECT_TRUE(document["configuration"]["resolved_sources"]["attention_query_tile_tokens"].is_null());
  EXPECT_EQ(document["configuration"]["base_seed_uint64_decimal"], "18446744073709551615");
  EXPECT_EQ(document["resolved_plan"]["geometry"]["active_weight_bytes_per_work_unit"],
            std::to_string(plan.geometry.active_weight_bytes_per_work_unit));
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["active_weight_bytes_per_work_unit"].is_string());
  EXPECT_EQ(document["resolved_plan"]["geometry"]["kv_capacity_bytes"],
            std::to_string(plan.geometry.kv_capacity_bytes));
  EXPECT_EQ(document["resolved_plan"]["geometry"]["kv_element_bytes"], "2");
  EXPECT_EQ(cpu_resources["model_plan_identity"], plan.plan_identity);
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["enabled"]);
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["valid"]);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["reason_code"], LlmJsonReason::VALID);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["total_bytes"],
            std::to_string(metadata.json_peak_estimate.total_bytes));
  EXPECT_EQ(cpu_resources["initialization"]["pattern_version"], Constants::LLM_BUFFER_PATTERN_VERSION);
  EXPECT_FALSE(cpu_resources["initialization"]["separate_reference_read_pass"]);
  EXPECT_EQ(document["resolved_plan"]["model_work_plan"]["plan_identity"], plan.plan_identity);
  EXPECT_EQ(document["resolved_plan"]["frozen_scenario_work_plans"]["scenarios"].size(), kLlmScenarioCount);
  EXPECT_EQ(document["calibration"]["attempts"]["weights_only"].size(), 1u);
  EXPECT_EQ(document["loop_records"].size(), 3u);
  EXPECT_EQ(document["measurements"].size(), 9u);
  EXPECT_EQ(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"]["sample_count"], 3u);
  EXPECT_EQ(document["environment"]["start"]["physical_memory_bytes"], "18446744073709551615");
  EXPECT_EQ(document["interpretation"]["reported_rate"], "synthetic_memory_work_units_per_second");
  EXPECT_FALSE(document["interpretation"]["transformer_math_included"]);
  EXPECT_FALSE(document["interpretation"]["physical_dram_traffic_measured"]);
}

TEST(LlmMemoryJsonTest, CompleteContiguousPrefillDocumentHasExactGeometryAndPartitionEvidence) {
  const LlmMemoryConfig config = prefill_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu.prefill.has_value());
  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete);
  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);

  EXPECT_EQ(document["status"], "complete");
  EXPECT_TRUE(document["results_complete"]);
  EXPECT_TRUE(document["conclusions_valid"]);
  EXPECT_EQ(document["phase"], "prefill");
  EXPECT_EQ(document["kv_layout"], "contiguous");
  EXPECT_EQ(document["methodology_version"], "llm-memory-v1-cpu-prefill-contiguous");

  const OrderedJson& configuration = document["configuration"];
  EXPECT_EQ(configuration["phase"], "prefill");
  EXPECT_TRUE(configuration["visible_context_tokens"].is_null());
  EXPECT_EQ(configuration["prompt_tokens"], 5u);
  EXPECT_EQ(configuration["attention_query_tile_tokens"], 2u);
  EXPECT_EQ(configuration["resolved_sources"]["phase"], "explicit");
  EXPECT_TRUE(configuration["resolved_sources"]["visible_context_tokens"].is_null());
  EXPECT_EQ(configuration["resolved_sources"]["prompt_tokens"], "explicit");
  EXPECT_EQ(configuration["resolved_sources"]["attention_query_tile_tokens"], "explicit");

  const OrderedJson& geometry = document["resolved_plan"]["geometry"];
  EXPECT_TRUE(geometry["decode"].is_null());
  expect_exact_keys(
      geometry["prefill"],
      {"prompt_tokens", "attention_query_tile_tokens", "tile_count", "attention_prefix_token_visits_per_sequence",
       "causal_token_pairs_per_sequence", "logical_attention_pairs", "logical_attention_fma_terms"});
  EXPECT_EQ(geometry["prefill"]["prompt_tokens"], 5u);
  EXPECT_EQ(geometry["prefill"]["attention_query_tile_tokens"], 2u);
  EXPECT_EQ(geometry["prefill"]["tile_count"], "3");
  EXPECT_EQ(geometry["prefill"]["attention_prefix_token_visits_per_sequence"], "11");
  EXPECT_EQ(geometry["prefill"]["causal_token_pairs_per_sequence"], "15");
  EXPECT_EQ(geometry["prefill"]["logical_attention_pairs"], "120");
  EXPECT_EQ(geometry["prefill"]["logical_attention_fma_terms"], "960");
  EXPECT_EQ(geometry["k_mapping_bytes"], "320");
  EXPECT_EQ(geometry["v_mapping_bytes"], "320");
  EXPECT_EQ(geometry["kv_capacity_bytes"], "640");
  EXPECT_EQ(geometry["weight_read_bytes_per_work_unit"], "1048576");
  EXPECT_EQ(geometry["kv_read_bytes_per_work_unit"], "1408");
  EXPECT_EQ(geometry["kv_write_bytes_per_work_unit"], "640");
  EXPECT_EQ(geometry["kv_only_effective_model_payload_bytes_per_work_unit"], "2048");
  EXPECT_EQ(geometry["mixed_effective_model_payload_bytes_per_work_unit"], "1050624");
  EXPECT_TRUE(geometry["traffic_crossover_numerator"].is_null());
  EXPECT_TRUE(geometry["traffic_crossover_denominator"].is_null());
  EXPECT_TRUE(geometry["traffic_crossover_context_tokens"].is_null());

  const OrderedJson& traffic = document["aggregates"]["traffic_diagnostics"];
  EXPECT_TRUE(traffic["traffic_crossover_numerator"].is_null());
  EXPECT_TRUE(traffic["traffic_crossover_denominator"].is_null());
  EXPECT_TRUE(traffic["traffic_crossover_context_tokens"].is_null());
  EXPECT_TRUE(traffic["current_visible_context_tokens"].is_null());
  EXPECT_TRUE(traffic["current_weight_to_kv_read_payload_ratio"].is_null());
  EXPECT_TRUE(traffic["current_context_classification"].is_null());

  const OrderedJson& components = document["resolved_plan"]["component_identities"];
  EXPECT_EQ(components["logical_profile_version"], Constants::LLM_PREFILL_LOGICAL_PROFILE_VERSION);
  EXPECT_EQ(components["backend_executor_version"], Constants::LLM_PREFILL_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(components["resource_abi_version"], Constants::LLM_PREFILL_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(components["write_pattern_version"], "llm-prefill-kv-affine64-v1");
  EXPECT_EQ(components["checksum_pattern_version"], "llm-prefill-affine64-parity-sum-v1");

  const OrderedJson& prefill = document["backend_evidence"]["cpu"]["prefill"];
  expect_exact_keys(prefill, {"cost_unit", "sequence_descriptors_per_scenario_per_worker", "scenarios", "identity"});
  EXPECT_EQ(prefill["cost_unit"], "worker-cost");
  EXPECT_EQ(prefill["sequence_descriptors_per_scenario_per_worker"], 2u);
  EXPECT_FALSE(prefill["identity"].get<std::string>().empty());
  ASSERT_EQ(prefill["scenarios"].size(), 3u);
  const std::array<const char*, 3> scenario_names = {"weights_only", "kv_only", "mixed"};
  const std::array<const char*, 3> worker_costs = {"524288", "1024", "525312"};
  for (size_t index = 0; index < scenario_names.size(); ++index) {
    const OrderedJson& scenario = prefill["scenarios"][index];
    expect_exact_keys(
        scenario, {"scenario", "cost_unit", "scope_count", "scope_identities", "worker_accounted_bytes_per_work_unit",
                   "minimum_worker_accounted_bytes_per_work_unit", "maximum_worker_accounted_bytes_per_work_unit",
                   "worker_accounted_imbalance_bytes_per_work_unit", "identity"});
    EXPECT_EQ(scenario["scenario"], scenario_names[index]);
    EXPECT_EQ(scenario["cost_unit"], "worker-cost");
    EXPECT_EQ(scenario["scope_count"], "2");
    ASSERT_EQ(scenario["scope_identities"].size(), 2u);
    EXPECT_FALSE(scenario["scope_identities"][0].get<std::string>().empty());
    ASSERT_EQ(scenario["worker_accounted_bytes_per_work_unit"].size(), 2u);
    EXPECT_EQ(scenario["worker_accounted_bytes_per_work_unit"][0], worker_costs[index]);
    EXPECT_EQ(scenario["worker_accounted_bytes_per_work_unit"][1], worker_costs[index]);
    EXPECT_EQ(scenario["minimum_worker_accounted_bytes_per_work_unit"], worker_costs[index]);
    EXPECT_EQ(scenario["maximum_worker_accounted_bytes_per_work_unit"], worker_costs[index]);
    EXPECT_EQ(scenario["worker_accounted_imbalance_bytes_per_work_unit"], "0");
    EXPECT_FALSE(scenario["identity"].get<std::string>().empty());
  }
  EXPECT_TRUE(document["backend_evidence"]["cpu"]["paged"].is_null());

  const OrderedJson& scenarios = document["resolved_plan"]["frozen_scenario_work_plans"]["scenarios"];
  ASSERT_EQ(scenarios.size(), 3u);
  EXPECT_EQ(scenarios[0]["work_unit_kind"], "prefill_operation");
  EXPECT_EQ(scenarios[0]["kv_write_kind"], "none");
  EXPECT_EQ(scenarios[1]["kv_write_kind"], "full_prompt_population");
  EXPECT_EQ(scenarios[1]["effective_model_payload_bytes_per_work_unit"], "2048");
  EXPECT_EQ(scenarios[2]["kv_write_kind"], "full_prompt_population");
  EXPECT_EQ(scenarios[2]["effective_model_payload_bytes_per_work_unit"], "1050624");

  ASSERT_EQ(document["measurements"].size(), 9u);
  for (const OrderedJson& measurement : document["measurements"]) {
    EXPECT_EQ(measurement["status"], "measured");
    EXPECT_EQ(measurement["work_unit_kind"], "prefill_operation");
    EXPECT_TRUE(measurement["working_set"]["fixed_visible_context_tokens"].is_null());
    EXPECT_TRUE(measurement["working_set"]["current_token_slot_included"].is_null());
    EXPECT_EQ(measurement["checksum"]["status"], "valid");
    EXPECT_TRUE(measurement["checksum"]["checksum_valid"]);
    EXPECT_FALSE(measurement["checksum"].contains("append_pattern_version"));
    EXPECT_FALSE(measurement["checksum"].contains("read_checksum_version"));
  }
  EXPECT_TRUE(document["interpretation"]["fixed_context_includes_current_token_slot"].is_null());
  EXPECT_FALSE(document["interpretation"]["prefill_transformer_compute_or_ttft_prediction_included"]);
}

TEST(LlmMemoryJsonTest, CompletePagedPrefillPublishesLayoutAndPrefillEvidenceWithoutDecodeOwnership) {
  const LlmMemoryConfig config = paged_prefill_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu.paged.has_value());
  ASSERT_TRUE(cpu.prefill.has_value());
  const LlmPagedCpuExecutionPlan& paged = *cpu.paged;
  const LlmPrefillCpuExecutionPlan& prefill_plan = *cpu.prefill;
  EXPECT_EQ(paged.execution_identity, prefill_plan.identity);
  for (size_t scenario_index = 1; scenario_index < kLlmScenarioCount; ++scenario_index) {
    for (const LlmPrefillCpuOwnershipPlan& scope : prefill_plan.scenarios[scenario_index].ownership_scopes) {
      EXPECT_EQ(scope.unit_kind, LlmPrefillPartitionUnitKind::PagedBlock);
    }
  }

  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete) << result.reason_code;
  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);

  EXPECT_EQ(document["schema_version"], 1);
  EXPECT_EQ(document["status"], "complete");
  EXPECT_EQ(document["phase"], "prefill");
  EXPECT_EQ(document["kv_layout"], "paged");
  EXPECT_EQ(document["methodology_version"], Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION);
  EXPECT_EQ(document["resolved_plan"]["component_identities"]["backend_executor_version"],
            Constants::LLM_PREFILL_PAGED_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(document["resolved_plan"]["component_identities"]["resource_abi_version"],
            Constants::LLM_PREFILL_PAGED_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(document["resolved_plan"]["component_identities"]["checksum_pattern_version"],
            LlmPrefillVersion::PAGED_CHECKSUM_ORACLE);

  const OrderedJson& layout = document["resolved_plan"]["layout"];
  EXPECT_EQ(layout["kv_layout"], "paged");
  EXPECT_EQ(layout["kv_block_tokens"], 2u);
  EXPECT_TRUE(layout["decode_append_offset_in_last_block"].is_null());
  EXPECT_EQ(layout["layout_identity"], paged.layout_identity);
  EXPECT_EQ(layout["permutation_identity"], paged.permutation.identity);

  const OrderedJson& paged_evidence = document["backend_evidence"]["cpu"]["paged"];
  expect_exact_keys(paged_evidence, {"layout_identity", "execution_identity", "block_table_logical_bytes",
                                     "block_table_page_rounded_bytes", "block_table_read_only", "table_validation",
                                     "permutation", "ownership"});
  EXPECT_EQ(paged_evidence["layout_identity"], paged.layout_identity);
  EXPECT_EQ(paged_evidence["execution_identity"], prefill_plan.identity);
  EXPECT_TRUE(paged_evidence["ownership"].is_null());

  const OrderedJson& prefill_evidence = document["backend_evidence"]["cpu"]["prefill"];
  expect_exact_keys(prefill_evidence,
                    {"cost_unit", "sequence_descriptors_per_scenario_per_worker", "scenarios", "identity"});
  EXPECT_EQ(prefill_evidence["identity"], prefill_plan.identity);
  ASSERT_EQ(prefill_evidence["scenarios"].size(), kLlmScenarioCount);
  EXPECT_EQ(prefill_evidence["scenarios"][1]["scenario"], "kv_only");
  EXPECT_EQ(prefill_evidence["scenarios"][2]["scenario"], "mixed");

  const LlmScenarioLimits kv_only_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::KvOnly);
  ASSERT_TRUE(kv_only_limits.valid) << kv_only_limits.reason_code;
  ASSERT_GE(document["measurements"].size(), 2u);
  const OrderedJson& kv_only = document["measurements"][1];
  EXPECT_EQ(kv_only["work_unit_kind"], "prefill_operation");
  EXPECT_EQ(kv_only["kv_write_kind"], "full_prompt_population");
  EXPECT_EQ(kv_only["layout_metadata_lookup_count_per_work_unit"],
            std::to_string(kv_only_limits.layout_metadata_lookup_count_per_work_unit));
  EXPECT_EQ(kv_only["layout_metadata_read_bytes_per_work_unit"],
            std::to_string(kv_only_limits.layout_metadata_read_bytes_per_work_unit));
  EXPECT_EQ(kv_only["accounted_bytes_per_work_unit"], std::to_string(kv_only_limits.accounted_bytes_per_work_unit));
  EXPECT_EQ(kv_only["checksum"]["initialization_pattern_version"], Constants::LLM_PAGED_BUFFER_PATTERN_VERSION);
  EXPECT_EQ(kv_only["checksum"]["write_pattern_version"], LlmPrefillVersion::WRITE_PATTERN);
  EXPECT_EQ(kv_only["checksum"]["checksum_pattern_version"], LlmPrefillVersion::PAGED_CHECKSUM_ORACLE);
}

TEST(LlmMemoryJsonTest, PagedDocumentPublishesExactLayoutResourceAndOwnershipProvenance) {
  const LlmMemoryConfig config = paged_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu.paged.has_value());
  const LlmPagedCpuExecutionPlan& paged = *cpu.paged;
  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete) << result.reason_code;
  const LlmResourcePreparationResult preparation = preparation_for(plan);
  const OrderedJson document = build_llm_memory_json(config, plan, preparation, fixed_metadata(config, plan), result);

  EXPECT_EQ(document["schema_version"], 1);
  EXPECT_EQ(document["kv_layout"], "paged");
  EXPECT_EQ(document["configuration"]["kv_layout"], "paged");
  EXPECT_EQ(document["configuration"]["kv_block_tokens"], 2u);
  EXPECT_EQ(document["configuration"]["resolved_sources"]["kv_layout"], "explicit");
  EXPECT_EQ(document["configuration"]["resolved_sources"]["kv_block_tokens"], "explicit");

  const OrderedJson& layout = document["resolved_plan"]["layout"];
  expect_exact_keys(layout, {"kv_layout", "kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer",
                             "total_physical_blocks", "block_bytes", "last_block_tokens", "last_block_valid_bytes",
                             "decode_append_offset_in_last_block", "block_table_entries", "block_table_bytes",
                             "layout_geometry_identity", "layout_identity", "permutation_domain_uint64_hex",
                             "permutation_seed_uint64_decimal", "permutation_algorithm_version",
                             "permutation_entry_count", "permutation_sha256", "permutation_identity"});
  EXPECT_EQ(layout["kv_layout"], "paged");
  EXPECT_EQ(layout["kv_block_tokens"], 2u);
  EXPECT_EQ(layout["blocks_per_sequence"], "2");
  EXPECT_EQ(layout["physical_blocks_per_layer"], "2");
  EXPECT_EQ(layout["total_physical_blocks"], "4");
  EXPECT_EQ(layout["block_bytes"], "64");
  EXPECT_EQ(layout["last_block_tokens"], "1");
  EXPECT_EQ(layout["last_block_valid_bytes"], "32");
  EXPECT_EQ(layout["decode_append_offset_in_last_block"], "0");
  EXPECT_EQ(layout["block_table_entries"], "2");
  EXPECT_EQ(layout["block_table_bytes"], "8");
  EXPECT_EQ(layout["layout_geometry_identity"], paged.layout.geometry_identity);
  EXPECT_EQ(layout["layout_identity"], paged.layout_identity);
  EXPECT_EQ(layout["permutation_domain_uint64_hex"], "0x4c4c4d4b56504731");
  EXPECT_EQ(layout["permutation_seed_uint64_decimal"], std::to_string(paged.permutation.resolved_seed));
  EXPECT_EQ(layout["permutation_algorithm_version"], Constants::LLM_KV_BLOCK_PERMUTATION_VERSION);
  EXPECT_EQ(layout["permutation_entry_count"], "2");
  EXPECT_EQ(layout["permutation_sha256"], paged.permutation.sha256);
  EXPECT_EQ(layout["permutation_identity"], paged.permutation.identity);

  const OrderedJson& resources = document["resolved_plan"]["resources"];
  EXPECT_EQ(resources["k_logical_bytes"], "192");
  EXPECT_EQ(resources["v_logical_bytes"], "192");
  EXPECT_EQ(resources["k_physical_length_bytes"], "256");
  EXPECT_EQ(resources["v_physical_length_bytes"], "256");
  EXPECT_EQ(resources["k_layout_padding_bytes"], "64");
  EXPECT_EQ(resources["v_layout_padding_bytes"], "64");
  EXPECT_EQ(resources["block_table_bytes"], "8");

  const OrderedJson& paged_evidence = document["backend_evidence"]["cpu"]["paged"];
  expect_exact_keys(paged_evidence, {"layout_identity", "execution_identity", "block_table_logical_bytes",
                                     "block_table_page_rounded_bytes", "block_table_read_only", "table_validation",
                                     "permutation", "ownership"});
  EXPECT_EQ(paged_evidence["layout_identity"], paged.layout_identity);
  EXPECT_EQ(paged_evidence["execution_identity"], paged.execution_identity);
  EXPECT_EQ(paged_evidence["block_table_logical_bytes"], "8");
  EXPECT_EQ(paged_evidence["block_table_page_rounded_bytes"], "4096");
  EXPECT_TRUE(paged_evidence["block_table_read_only"]);
  expect_exact_keys(paged_evidence["table_validation"], {"valid", "interrupted", "reason_code", "expected_entries",
                                                         "examined_entries", "validation_bitset_bytes"});
  EXPECT_TRUE(paged_evidence["table_validation"]["valid"]);
  EXPECT_EQ(paged_evidence["table_validation"]["expected_entries"], "2");
  EXPECT_EQ(paged_evidence["table_validation"]["examined_entries"], "2");
  EXPECT_EQ(paged_evidence["table_validation"]["validation_bitset_bytes"], "1");
  expect_exact_keys(
      paged_evidence["permutation"],
      {"algorithm_version", "domain_uint64_hex", "resolved_seed_uint64_decimal", "entry_count", "sha256", "identity"});
  EXPECT_EQ(paged_evidence["permutation"]["sha256"], paged.permutation.sha256);
  EXPECT_EQ(paged_evidence["permutation"]["identity"], paged.permutation.identity);
  expect_exact_keys(
      paged_evidence["ownership"],
      {"valid", "reason_code", "worker_count", "layout_geometry_identity", "layer_sequence_count", "total_owned_blocks",
       "total_model_payload_bytes_per_work_unit", "total_layout_metadata_lookup_count_per_work_unit",
       "total_layout_metadata_read_bytes_per_work_unit", "total_accounted_bytes_per_work_unit",
       "minimum_worker_accounted_bytes_per_work_unit", "maximum_worker_accounted_bytes_per_work_unit",
       "worker_accounted_imbalance_bytes_per_work_unit", "assignment_count", "identity"});
  EXPECT_TRUE(paged_evidence["ownership"]["valid"]);
  EXPECT_EQ(paged_evidence["ownership"]["total_owned_blocks"], "4");
  EXPECT_EQ(paged_evidence["ownership"]["total_layout_metadata_lookup_count_per_work_unit"], "10");
  EXPECT_EQ(paged_evidence["ownership"]["total_layout_metadata_read_bytes_per_work_unit"], "40");
  EXPECT_EQ(paged_evidence["ownership"]["identity"], paged.ownership.identity);

  const OrderedJson& cpu_resources = document["backend_evidence"]["cpu"]["resources"];
  expect_exact_keys(cpu_resources["mappings"]["block_table"], {"requested_bytes", "committed_bytes"});
  EXPECT_EQ(cpu_resources["mappings"]["block_table"]["requested_bytes"], "8");
  EXPECT_EQ(cpu_resources["mappings"]["block_table"]["committed_bytes"], "4096");
  EXPECT_EQ(cpu_resources["initialization"]["block_table_logical_bytes"], "8");
  EXPECT_EQ(cpu_resources["initialization"]["block_table_page_rounded_bytes"], "4096");
  EXPECT_TRUE(cpu_resources["initialization"]["block_table_read_only"]);
  EXPECT_EQ(cpu_resources["initialization"]["k_layout_padding_bytes"], "64");
  EXPECT_EQ(cpu_resources["initialization"]["v_layout_padding_bytes"], "64");

  const OrderedJson& budget = document["memory_budget"];
  EXPECT_EQ(budget["layout_transient_bytes"], "1");
  EXPECT_EQ(budget["setup_peak_bytes"], std::to_string(plan.memory_budget.request.setup_peak_bytes));
  EXPECT_EQ(budget["runtime_peak_bytes"], std::to_string(plan.memory_budget.request.runtime_peak_bytes));
  EXPECT_EQ(budget["request"]["requested_block_table_mapping_bytes"], "8");
  EXPECT_EQ(budget["request"]["committed_block_table_mapping_bytes"], "4096");
  EXPECT_EQ(budget["request"]["layout_transient_bytes"], "1");

  ASSERT_GE(document["measurements"].size(), 2u);
  const OrderedJson& kv_only = document["measurements"][1];
  ASSERT_EQ(kv_only["scenario"], "kv_only");
  EXPECT_EQ(kv_only["layout_metadata_lookup_count_per_work_unit"], "10");
  EXPECT_EQ(kv_only["layout_metadata_read_bytes_per_work_unit"], "40");
  EXPECT_EQ(kv_only["effective_model_payload_bytes_per_work_unit"], "512");
  EXPECT_EQ(kv_only["accounted_bytes_per_work_unit"], "552");
  EXPECT_TRUE(kv_only["execution"]["post_validation_evaluated"]);
  EXPECT_TRUE(kv_only["execution"]["post_validation_valid"]);
  EXPECT_EQ(kv_only["checksum"]["initialization_pattern_version"], Constants::LLM_PAGED_BUFFER_PATTERN_VERSION);
  EXPECT_EQ(kv_only["checksum"]["checksum_pattern_version"], Constants::LLM_PAGED_READ_CHECKSUM_VERSION);
  ASSERT_FALSE(document["calibration"]["attempts"]["weights_only"].empty());
  EXPECT_EQ(document["calibration"]["attempts"]["weights_only"][0]["execution"]["checksum"]["algorithm_version"],
            Constants::LLM_PAGED_READ_CHECKSUM_VERSION);
}

TEST(LlmMemoryJsonTest, UnsupportedBeforePreparationRetainsAdmittedTopLevelMemoryBudget) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_TRUE(plan.memory_budget.valid) << plan.memory_budget.reason_code;
  LlmMemoryResult result;
  result.initialized = true;
  result.status = LlmRunStatus::Unsupported;
  result.reason_code = LlmBackendReason::TASK_UNSUPPORTED;
  LlmBackendEvidence backend_evidence;
  backend_evidence.backend = LlmMemoryBackend::Cpu;
  backend_evidence.initialization = {LlmBackendStatus::Unsupported, LlmBackendReason::TASK_UNSUPPORTED};

  const OrderedJson document =
      build_llm_memory_json(config, plan, backend_evidence, fixed_metadata(config, plan), result);

  EXPECT_EQ(document["status"], "unsupported");
  EXPECT_EQ(document["reason_code"], LlmBackendReason::TASK_UNSUPPORTED);
  EXPECT_TRUE(document["memory_budget"]["valid"].get<bool>());
  EXPECT_EQ(document["memory_budget"]["reason_code"], plan.memory_budget.reason_code);
  EXPECT_EQ(document["memory_budget"]["admitted_budget_bytes"],
            std::to_string(plan.memory_budget.allowed_memory_bytes));
  EXPECT_FALSE(document["backend_evidence"]["cpu"]["resources"]["valid"].get<bool>());
}

TEST(LlmMemoryJsonTest, TrafficClassificationUsesExactPayloadEqualityForNearCrossover) {
  LlmGeometryRequest request{8, 1, 1, 1, 1, 1, 4, 1};
  LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_EQ(geometry.weight_read_bytes_per_work_unit, geometry.kv_read_bytes_per_work_unit);
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "near_crossover");

  request.active_weight_bytes = 9;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "weight_payload_dominant");

  request.active_weight_bytes = 7;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "kv_read_payload_dominant");

  EXPECT_STREQ(classify_llm_traffic_payload(LlmGeometry{}), "unavailable");
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateIsZeroWhenOutputIsDisabled) {
  LlmMemoryConfig config = explicit_config(1);
  config.output_file.clear();
  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, LlmMemoryWorkPlan{});

  EXPECT_TRUE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::VALID);
  EXPECT_EQ(estimate.fixed_schema_bytes, 0u);
  EXPECT_EQ(estimate.input_string_bytes, 0u);
  EXPECT_EQ(estimate.measurement_record_bytes, 0u);
  EXPECT_EQ(estimate.worker_checksum_bytes, 0u);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, PreflightAuxiliaryEstimatesExactlyMatchFinalizedPlanForAllActiveCpuProfiles) {
  const std::array<LlmMemoryConfig, 4> configs = {explicit_config(3), paged_config(3), prefill_config(3),
                                                  paged_prefill_config(3)};
  for (const LlmMemoryConfig& config : configs) {
    SCOPED_TRACE(llm_kv_layout_to_string(config.kv_layout));
    LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(config));
    ASSERT_TRUE(draft.valid) << draft.reason_code;
    const LlmExecutorAuxiliaryEstimate preflight_executor =
        calculate_llm_executor_auxiliary_estimate(draft.auxiliary_preflight);
    const LlmRunnerAuxiliaryEstimate preflight_runner =
        calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
    const LlmJsonPeakEstimate preflight_json = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
    ASSERT_TRUE(preflight_executor.valid);
    ASSERT_TRUE(preflight_runner.valid);
    ASSERT_TRUE(preflight_json.valid);

    size_t checksum_auxiliary_bytes = 0;
    size_t orchestration_auxiliary_bytes = 0;
    ASSERT_TRUE(NumericUtils::checked_add(preflight_executor.checksum_auxiliary_bytes,
                                          preflight_runner.checksum_auxiliary_bytes, checksum_auxiliary_bytes));
    ASSERT_TRUE(NumericUtils::checked_add(preflight_executor.orchestration_auxiliary_bytes,
                                          preflight_runner.orchestration_auxiliary_bytes,
                                          orchestration_auxiliary_bytes));
    ASSERT_TRUE(NumericUtils::checked_add(orchestration_auxiliary_bytes, preflight_json.total_bytes,
                                          orchestration_auxiliary_bytes));
    const LlmMemoryWorkPlan plan =
        finalize_llm_memory_work_plan(std::move(draft), checksum_auxiliary_bytes, orchestration_auxiliary_bytes);
    ASSERT_TRUE(plan.valid) << plan.reason_code;

    const LlmExecutorAuxiliaryEstimate final_executor = calculate_llm_executor_auxiliary_estimate(plan);
    const LlmRunnerAuxiliaryEstimate final_runner = calculate_llm_runner_auxiliary_estimate(config, plan);
    const LlmJsonPeakEstimate final_json = calculate_llm_json_peak_estimate(config, plan);

    EXPECT_EQ(preflight_executor.valid, final_executor.valid);
    EXPECT_EQ(preflight_executor.reason_code, final_executor.reason_code);
    EXPECT_EQ(preflight_executor.static_reference_bytes, final_executor.static_reference_bytes);
    EXPECT_EQ(preflight_executor.expected_checksum_bytes, final_executor.expected_checksum_bytes);
    EXPECT_EQ(preflight_executor.actual_checksum_bytes, final_executor.actual_checksum_bytes);
    EXPECT_EQ(preflight_executor.run_checksum_bytes, final_executor.run_checksum_bytes);
    EXPECT_EQ(preflight_executor.worker_status_bytes, final_executor.worker_status_bytes);
    EXPECT_EQ(preflight_executor.thread_handle_bytes, final_executor.thread_handle_bytes);
    EXPECT_EQ(preflight_executor.checksum_auxiliary_bytes, final_executor.checksum_auxiliary_bytes);
    EXPECT_EQ(preflight_executor.orchestration_auxiliary_bytes, final_executor.orchestration_auxiliary_bytes);
    EXPECT_EQ(preflight_executor.total_auxiliary_bytes, final_executor.total_auxiliary_bytes);

    EXPECT_EQ(preflight_runner.valid, final_runner.valid);
    EXPECT_EQ(preflight_runner.reason_code, final_runner.reason_code);
    EXPECT_EQ(preflight_runner.measurement_record_bytes, final_runner.measurement_record_bytes);
    EXPECT_EQ(preflight_runner.loop_record_bytes, final_runner.loop_record_bytes);
    EXPECT_EQ(preflight_runner.calibration_record_bytes, final_runner.calibration_record_bytes);
    EXPECT_EQ(preflight_runner.calibration_identity_bytes, final_runner.calibration_identity_bytes);
    EXPECT_EQ(preflight_runner.aggregate_value_bytes, final_runner.aggregate_value_bytes);
    EXPECT_EQ(preflight_runner.statistics_workspace_bytes, final_runner.statistics_workspace_bytes);
    EXPECT_EQ(preflight_runner.warning_record_bytes, final_runner.warning_record_bytes);
    EXPECT_EQ(preflight_runner.fixed_metadata_bytes, final_runner.fixed_metadata_bytes);
    EXPECT_EQ(preflight_runner.retained_checksum_bytes, final_runner.retained_checksum_bytes);
    EXPECT_EQ(preflight_runner.checksum_auxiliary_bytes, final_runner.checksum_auxiliary_bytes);
    EXPECT_EQ(preflight_runner.orchestration_auxiliary_bytes, final_runner.orchestration_auxiliary_bytes);
    EXPECT_EQ(preflight_runner.total_auxiliary_bytes, final_runner.total_auxiliary_bytes);

    EXPECT_EQ(preflight_json.valid, final_json.valid);
    EXPECT_EQ(preflight_json.reason_code, final_json.reason_code);
    EXPECT_EQ(preflight_json.fixed_schema_bytes, final_json.fixed_schema_bytes);
    EXPECT_EQ(preflight_json.input_string_bytes, final_json.input_string_bytes);
    EXPECT_EQ(preflight_json.measurement_record_bytes, final_json.measurement_record_bytes);
    EXPECT_EQ(preflight_json.worker_checksum_bytes, final_json.worker_checksum_bytes);
    EXPECT_EQ(preflight_json.total_bytes, final_json.total_bytes);

    EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, checksum_auxiliary_bytes);
    EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, orchestration_auxiliary_bytes);
  }
}

TEST(LlmMemoryJsonTest, SyntheticPreflightIdentityCopiesScaleByLoopAndCalibrationCapacity) {
  LlmAuxiliaryPreflightView preflight;
  preflight.valid = true;
  preflight.backend = LlmMemoryBackend::Cpu;
  preflight.effective_workers = 1;
  preflight.json_identity_string_bytes = 53;
  preflight.maximum_scenario_plan_identity_bytes = {2, 3, 5};
  preflight.frozen_reason_code_bytes = 7;
  preflight.frozen_model_plan_identity_bytes = 11;
  preflight.frozen_plan_identity_bytes = 13;
  preflight.frozen_scenario_reason_code_bytes = {17, 19, 23};
  preflight.frozen_scenario_model_plan_identity_bytes = {29, 31, 37};
  preflight.frozen_scenario_plan_identity_bytes = {41, 43, 47};

  constexpr size_t kDomTransportExpansionFactor = 16;
  const auto expected_input_string_bytes = [&](const LlmMemoryConfig& config) {
    size_t argv_bytes = 0;
    for (const std::string& argument : config.argv) {
      argv_bytes += argument.size();
    }
    size_t scenario_identity_bytes = 0;
    size_t frozen_identity_bytes = preflight.frozen_reason_code_bytes + preflight.frozen_model_plan_identity_bytes +
                                   preflight.frozen_plan_identity_bytes;
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      scenario_identity_bytes += preflight.maximum_scenario_plan_identity_bytes[index];
      frozen_identity_bytes += preflight.frozen_scenario_reason_code_bytes[index] +
                               preflight.frozen_scenario_model_plan_identity_bytes[index] +
                               preflight.frozen_scenario_plan_identity_bytes[index];
    }
    const size_t calibration_attempts =
        config.user_specified_iterations ? 1 : 4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
    const size_t raw_identity_and_input_bytes =
        config.output_file.size() + argv_bytes + preflight.json_identity_string_bytes + frozen_identity_bytes +
        config.loop_count * scenario_identity_bytes + calibration_attempts * scenario_identity_bytes;
    return raw_identity_and_input_bytes * kDomTransportExpansionFactor;
  };

  const LlmMemoryConfig one_loop = explicit_config(1);
  const LlmJsonPeakEstimate one_loop_estimate = calculate_llm_json_peak_estimate(one_loop, preflight);
  ASSERT_TRUE(one_loop_estimate.valid) << one_loop_estimate.reason_code;
  EXPECT_EQ(one_loop_estimate.input_string_bytes, expected_input_string_bytes(one_loop));

  LlmMemoryConfig four_loops = one_loop;
  four_loops.loop_count = 4;
  const LlmJsonPeakEstimate four_loop_estimate = calculate_llm_json_peak_estimate(four_loops, preflight);
  ASSERT_TRUE(four_loop_estimate.valid) << four_loop_estimate.reason_code;
  EXPECT_EQ(four_loop_estimate.input_string_bytes, expected_input_string_bytes(four_loops));
  constexpr size_t kScenarioIdentityBytes = 2 + 3 + 5;
  EXPECT_EQ(four_loop_estimate.input_string_bytes - one_loop_estimate.input_string_bytes,
            3 * kScenarioIdentityBytes * kDomTransportExpansionFactor);

  LlmMemoryConfig automatic = four_loops;
  automatic.user_specified_iterations = false;
  const LlmJsonPeakEstimate automatic_estimate = calculate_llm_json_peak_estimate(automatic, preflight);
  ASSERT_TRUE(automatic_estimate.valid) << automatic_estimate.reason_code;
  EXPECT_EQ(automatic_estimate.input_string_bytes, expected_input_string_bytes(automatic));
  EXPECT_EQ(automatic_estimate.input_string_bytes - four_loop_estimate.input_string_bytes,
            (3 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS) * kScenarioIdentityBytes * kDomTransportExpansionFactor);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateRejectsSyntheticIdentityReplicationOverflowBeforeDom) {
  LlmMemoryConfig config = explicit_config(2);
  LlmAuxiliaryPreflightView preflight;
  preflight.valid = true;
  preflight.backend = LlmMemoryBackend::Cpu;
  preflight.effective_workers = 1;

  preflight.maximum_scenario_plan_identity_bytes[0] = std::numeric_limits<size_t>::max() / 2 + 1;
  LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, preflight);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);

  config.loop_count = 1;
  config.user_specified_iterations = false;
  preflight.maximum_scenario_plan_identity_bytes = {};
  const size_t calibration_attempts = 4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
  preflight.maximum_scenario_plan_identity_bytes[1] = std::numeric_limits<size_t>::max() / calibration_attempts + 1;
  estimate = calculate_llm_json_peak_estimate(config, preflight);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);

  config.user_specified_iterations = true;
  preflight.maximum_scenario_plan_identity_bytes = {};
  preflight.frozen_reason_code_bytes = std::numeric_limits<size_t>::max();
  preflight.frozen_model_plan_identity_bytes = 1;
  estimate = calculate_llm_json_peak_estimate(config, preflight);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateAccountsForEveryPrefillPartitionIdentity) {
  const LlmMemoryConfig config = prefill_config(1);
  LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmCpuExecutionPlan* const cpu = get_llm_cpu_execution_plan(plan);
  ASSERT_NE(cpu, nullptr);
  ASSERT_TRUE(cpu->prefill.has_value());

  const LlmJsonPeakEstimate baseline = calculate_llm_json_peak_estimate(config, plan);
  ASSERT_TRUE(baseline.valid) << baseline.reason_code;

  constexpr size_t kAggregateSuffixBytes = 1;
  constexpr size_t kScenarioSuffixBytes = 2;
  constexpr size_t kScopeSuffixBytes = 3;
  size_t scope_count = 0;
  cpu->prefill->identity.append(kAggregateSuffixBytes, 'a');
  for (LlmPrefillCpuScenarioExecutionPlan& scenario : cpu->prefill->scenarios) {
    scenario.identity.append(kScenarioSuffixBytes, 'b');
    for (LlmPrefillCpuOwnershipPlan& scope : scenario.ownership_scopes) {
      scope.identity.append(kScopeSuffixBytes, 'c');
      ++scope_count;
    }
  }
  ASSERT_GT(scope_count, 0u);

  const LlmJsonPeakEstimate expanded = calculate_llm_json_peak_estimate(config, plan);
  ASSERT_TRUE(expanded.valid) << expanded.reason_code;
  const size_t added_identity_bytes =
      kAggregateSuffixBytes + kLlmScenarioCount * kScenarioSuffixBytes + scope_count * kScopeSuffixBytes;
  constexpr size_t kLiveDomAndSerializedTransportFactor = 16;
  EXPECT_EQ(expanded.input_string_bytes - baseline.input_string_bytes,
            added_identity_bytes * kLiveDomAndSerializedTransportFactor);
  EXPECT_EQ(expanded.total_bytes - baseline.total_bytes, added_identity_bytes * kLiveDomAndSerializedTransportFactor);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateRejectsPrefillIdentityByteOverflowBeforeFinalization) {
  const LlmMemoryConfig config = prefill_config(1);
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(config));
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  draft.auxiliary_preflight.json_identity_string_bytes = std::numeric_limits<size_t>::max();

  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, draft.auxiliary_preflight);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateScalesWithMeasurementsAndWorkersAndIsAdmitted) {
  const LlmMemoryConfig config = explicit_config(3);
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  ASSERT_TRUE(preliminary.valid) << preliminary.reason_code;
  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, preliminary);
  ASSERT_TRUE(executor.valid);
  ASSERT_TRUE(runner.valid);
  ASSERT_TRUE(estimate.valid);

  EXPECT_GT(estimate.fixed_schema_bytes, 0u);
  EXPECT_GT(estimate.input_string_bytes, 0u);
  EXPECT_GT(estimate.measurement_record_bytes, 0u);
  EXPECT_GT(estimate.worker_checksum_bytes, estimate.measurement_record_bytes);
  EXPECT_EQ(estimate.total_bytes, estimate.fixed_schema_bytes + estimate.input_string_bytes +
                                      estimate.measurement_record_bytes + estimate.worker_checksum_bytes);

  const LlmMemoryWorkPlan admitted = admitted_plan(config);
  ASSERT_TRUE(admitted.valid) << admitted.reason_code;
  EXPECT_EQ(admitted.memory_budget.request.orchestration_auxiliary_bytes,
            executor.orchestration_auxiliary_bytes + runner.orchestration_auxiliary_bytes + estimate.total_bytes);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateAccountsForRetainedMetalMeasurementAndCalibrationTaskEvidence) {
  const LlmMemoryConfig one_loop = explicit_metal_config();
  const LlmMemoryWorkPlan plan = admitted_metal_plan(one_loop);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  const LlmJsonPeakEstimate one_loop_estimate = calculate_llm_json_peak_estimate(one_loop, plan);
  ASSERT_TRUE(one_loop_estimate.valid) << one_loop_estimate.reason_code;
  constexpr size_t kExplicitTaskRecords = 2 * kLlmScenarioCount;
  ASSERT_EQ(one_loop_estimate.measurement_record_bytes % kExplicitTaskRecords, 0u);
  const size_t metal_task_record_bytes = one_loop_estimate.measurement_record_bytes / kExplicitTaskRecords;
  EXPECT_GT(metal_task_record_bytes, 64u * 1024u);

  LlmMemoryConfig four_loops = one_loop;
  four_loops.loop_count = 4;
  const LlmJsonPeakEstimate four_loop_estimate = calculate_llm_json_peak_estimate(four_loops, plan);
  ASSERT_TRUE(four_loop_estimate.valid) << four_loop_estimate.reason_code;
  EXPECT_EQ(four_loop_estimate.measurement_record_bytes - one_loop_estimate.measurement_record_bytes,
            3 * kLlmScenarioCount * metal_task_record_bytes);

  LlmMemoryConfig automatic = four_loops;
  automatic.user_specified_iterations = false;
  const LlmJsonPeakEstimate automatic_estimate = calculate_llm_json_peak_estimate(automatic, plan);
  ASSERT_TRUE(automatic_estimate.valid) << automatic_estimate.reason_code;
  EXPECT_EQ(automatic_estimate.measurement_record_bytes - four_loop_estimate.measurement_record_bytes,
            (3 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS) * kLlmScenarioCount * metal_task_record_bytes);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateRejectsCountArithmeticOverflow) {
  LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  config.loop_count = std::numeric_limits<size_t>::max();

  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, plan);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, ExactByteSeedAndChecksumIntegersAreCanonicalDecimalStrings) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = complete_result(config, plan);
  LlmResultMetadata metadata = fixed_metadata(config, plan);
  metadata.available_memory_bytes = 0;
  metadata.available_memory_source = "unavailable";
  const OrderedJson document = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);

  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["work_unit_kind"], "decode_step");
  EXPECT_EQ(measurement["kv_write_kind"], "none");
  EXPECT_TRUE(measurement["planned_work_units"].is_number_unsigned());
  EXPECT_TRUE(measurement["completed_work_units"].is_number_unsigned());
  const std::array<const char*, 15> decimal_measurement_fields = {"weight_read_bytes_per_work_unit",
                                                                  "kv_read_bytes_per_work_unit",
                                                                  "kv_write_bytes_per_work_unit",
                                                                  "effective_model_payload_bytes_per_work_unit",
                                                                  "layout_metadata_lookup_count_per_work_unit",
                                                                  "layout_metadata_read_bytes_per_work_unit",
                                                                  "accounted_bytes_per_work_unit",
                                                                  "planned_effective_model_payload_bytes",
                                                                  "completed_effective_model_payload_bytes",
                                                                  "planned_layout_metadata_lookup_count",
                                                                  "completed_layout_metadata_lookup_count",
                                                                  "planned_layout_metadata_read_bytes",
                                                                  "completed_layout_metadata_read_bytes",
                                                                  "planned_task_accounted_bytes",
                                                                  "completed_task_accounted_bytes"};
  for (const char* field : decimal_measurement_fields) {
    EXPECT_TRUE(measurement[field].is_string()) << field;
  }
  EXPECT_EQ(measurement["layout_metadata_lookup_count_per_work_unit"], "0");
  EXPECT_EQ(measurement["layout_metadata_read_bytes_per_work_unit"], "0");
  EXPECT_TRUE(measurement["synthetic_work_unit_latency_seconds"].is_number());
  EXPECT_TRUE(measurement["synthetic_memory_work_units_per_second"].is_number());
  EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_number());

  EXPECT_TRUE(document["resolved_plan"]["geometry"]["decode"].is_object());
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["prefill"].is_null());
  EXPECT_TRUE(document["configuration"]["kv_block_tokens"].is_null());
  EXPECT_EQ(document["configuration"]["resolved_sources"]["kv_layout"], "default");
  EXPECT_TRUE(document["configuration"]["resolved_sources"]["kv_block_tokens"].is_null());
  for (const char* field :
       {"kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer", "total_physical_blocks", "block_bytes",
        "last_block_tokens", "last_block_valid_bytes", "decode_append_offset_in_last_block", "block_table_entries",
        "block_table_bytes", "layout_geometry_identity", "layout_identity", "permutation_domain_uint64_hex",
        "permutation_seed_uint64_decimal", "permutation_algorithm_version", "permutation_entry_count",
        "permutation_sha256", "permutation_identity"}) {
    EXPECT_TRUE(document["resolved_plan"]["layout"][field].is_null()) << field;
  }
  EXPECT_TRUE(document["resolved_plan"]["resources"]["weight_logical_bytes"].is_string());
  EXPECT_TRUE(document["resolved_plan"]["resources"]["block_table_bytes"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["permutation_version"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["msl_revision"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["msl_source_sha256"].is_null());
  EXPECT_TRUE(document["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(document["backend_evidence"]["cpu"]["paged"].is_null());
  EXPECT_TRUE(document["backend_evidence"]["metal"].is_null());
  EXPECT_TRUE(document["memory_budget"]["layout_transient_bytes"].is_null());
  EXPECT_TRUE(document["memory_budget"]["request"]["requested_block_table_mapping_bytes"].is_null());
  EXPECT_TRUE(document["memory_budget"]["request"]["committed_block_table_mapping_bytes"].is_null());
  EXPECT_TRUE(document["memory_budget"]["request"]["layout_transient_bytes"].is_null());
  for (const char* field : {"resource_rounding_bytes", "transient_peak_bytes", "known_owned_peak_bytes",
                            "setup_peak_bytes", "runtime_peak_bytes", "admitted_budget_bytes"}) {
    EXPECT_TRUE(document["memory_budget"][field].is_string()) << field;
  }

  EXPECT_EQ(document["seeds"]["base_seed_uint64_decimal"], "18446744073709551615");
  const OrderedJson& checksum = document["measurements"][0]["checksum"];
  ASSERT_EQ(checksum["status"], "valid");
  ASSERT_TRUE(checksum["checksum_valid"]);
  EXPECT_TRUE(document["measurements"][0]["execution"]["post_validation_evaluated"]);
  EXPECT_TRUE(document["measurements"][0]["execution"]["post_validation_valid"]);
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["state_a_uint64_decimal"], "18446744073709551615");
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["state_b_uint64_decimal"], "9007199254740993");
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["exact_bytes_read"], "9007199254741093");
  EXPECT_TRUE(checksum["expected_worker_checksums"][0]["weight"]["span_count_uint64_decimal"].is_string());
  EXPECT_EQ(checksum["actual_run_checksum"]["state_a_uint64_decimal"], "18446744073709551615");
  EXPECT_TRUE(document["counters"]["planned_work_units"].is_string());
  EXPECT_TRUE(document["counters"]["completed_effective_model_payload_bytes"].is_string());
  EXPECT_TRUE(document["environment"]["available_memory_bytes"].is_null());
  EXPECT_EQ(document["environment"]["available_memory_source"], "unavailable");
}

TEST(LlmMemoryJsonTest, InterruptedRunnerSerializesUnavailableMetricsExecutionQosAndChecksumAsNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return true; };
  LlmMemoryResult result;
  FakeLlmBackend backend;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks), EXIT_SUCCESS);
  ASSERT_EQ(result.status, LlmRunStatus::Interrupted);
  ASSERT_FALSE(result.measurements.empty());

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "interrupted");
  EXPECT_FALSE(document["results_complete"]);
  EXPECT_FALSE(document["conclusions_valid"]);
  const OrderedJson& serialized = document["measurements"][0];
  EXPECT_EQ(serialized["status"], "interrupted");
  EXPECT_EQ(serialized["reason_code"], LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
  EXPECT_FALSE(serialized["attempted"]);
  EXPECT_TRUE(serialized["qos_successful_workers"].is_null());
  EXPECT_TRUE(serialized["qos_failed_workers"].is_null());
  EXPECT_EQ(serialized["completed_work_units"], 0u);
  EXPECT_EQ(serialized["completed_effective_model_payload_bytes"], "0");
  EXPECT_TRUE(serialized["elapsed_seconds"].is_null());
  EXPECT_TRUE(serialized["synthetic_work_unit_latency_seconds"].is_null());
  EXPECT_TRUE(serialized["synthetic_memory_work_units_per_second"].is_null());
  EXPECT_TRUE(serialized["effective_model_payload_gb_s"].is_null());
  EXPECT_EQ(serialized["execution"]["status"], "not_run");
  EXPECT_TRUE(serialized["execution"]["valid"].is_null());
  EXPECT_TRUE(serialized["execution"]["post_validation_evaluated"].is_null());
  EXPECT_TRUE(serialized["execution"]["post_validation_valid"].is_null());
  EXPECT_EQ(serialized["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(serialized["checksum"]["reason_code"], LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
  EXPECT_TRUE(serialized["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(serialized["checksum"]["expected_worker_checksums"].is_null());
  EXPECT_TRUE(serialized["checksum"]["actual_run_checksum"].is_null());
  // The immutable frozen plan remains available despite the interruption.
  EXPECT_TRUE(serialized["planned_effective_model_payload_bytes"].is_string());
}

TEST(LlmMemoryJsonTest, ExecutorExceptionUsesRunnerReasonAndNullUnavailableExecutionEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        if (context.kind == LlmRunnerTaskKind::Measurement) {
          throw std::runtime_error("injected JSON exception path");
        }
        return successful_execution(model_plan);
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);
  ASSERT_EQ(result.status, LlmRunStatus::Failed);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "failed");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(document["diagnostic"], "injected JSON exception path");
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_TRUE(measurement["attempted"]);
  EXPECT_EQ(measurement["status"], "failed");
  EXPECT_EQ(measurement["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["qos_successful_workers"].is_null());
  EXPECT_TRUE(measurement["qos_failed_workers"].is_null());
  EXPECT_EQ(measurement["execution"]["status"], "unavailable");
  EXPECT_EQ(measurement["execution"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["execution"]["valid"].is_null());
  EXPECT_TRUE(measurement["execution"]["requested_workers"].is_null());
  EXPECT_TRUE(measurement["execution"]["kernel_succeeded"].is_null());
  EXPECT_TRUE(measurement["execution"]["post_validation_evaluated"].is_null());
  EXPECT_TRUE(measurement["execution"]["post_validation_valid"].is_null());
  EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, ExcludedExecutorExceptionUsesRunnerReasonAndNullUnavailableExecutionEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        if (context.kind == LlmRunnerTaskKind::Warmup) {
          throw std::runtime_error("injected excluded JSON exception path");
        }
        return successful_execution(model_plan);
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& execution = document["calibration"]["attempts"]["weights_only"][0]["execution"];
  EXPECT_EQ(execution["status"], "unavailable");
  EXPECT_EQ(execution["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(execution["valid"].is_null());
  EXPECT_TRUE(execution["requested_workers"].is_null());
  EXPECT_TRUE(execution["kernel_succeeded"].is_null());
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, InvalidElapsedMeasurementLeavesChecksumEvidenceNotEvaluatedAndNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        LlmExecutorResult execution = successful_execution(model_plan);
        if (context.kind == LlmRunnerTaskKind::Measurement) {
          execution.valid = false;
          execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
          execution.elapsed_seconds = 0.0;
          execution.checksum_evaluated = false;
          execution.checksum_valid = false;
        }
        return execution;
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["status"], "invalid");
  EXPECT_EQ(measurement["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_EQ(measurement["execution"]["status"], "invalid");
  EXPECT_EQ(measurement["execution"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_FALSE(measurement["execution"]["valid"]);
  EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(measurement["checksum"]["expected_worker_checksums"].is_null());
  EXPECT_TRUE(measurement["checksum"]["actual_worker_checksums"].is_null());
  EXPECT_TRUE(measurement["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(measurement["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, InvalidElapsedExcludedTaskLeavesCompactChecksumEvidenceNotEvaluatedAndNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        LlmExecutorResult execution = successful_execution(model_plan);
        if (context.kind == LlmRunnerTaskKind::Warmup) {
          execution.valid = false;
          execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
          execution.elapsed_seconds = 0.0;
          execution.checksum_evaluated = false;
          execution.checksum_valid = false;
        }
        return execution;
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& execution = document["calibration"]["attempts"]["weights_only"][0]["execution"];
  EXPECT_EQ(execution["status"], "invalid");
  EXPECT_EQ(execution["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_FALSE(execution["valid"]);
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(execution["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(execution["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, MalformedExcludedChecksumCardinalityDoesNotPublishCompactChecksumEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_GE(cpu_execution_plan(plan).effective_workers, 2u);
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        LlmExecutorResult execution = successful_execution(model_plan);
        if (context.kind == LlmRunnerTaskKind::Warmup) {
          execution.expected_checksums.resize(1);
          execution.actual_checksums.resize(1);
        }
        return execution;
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& attempt = document["calibration"]["attempts"]["weights_only"][0];
  EXPECT_FALSE(attempt["valid"]);
  EXPECT_EQ(attempt["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  const OrderedJson& execution = attempt["execution"];
  EXPECT_EQ(execution["status"], "invalid");
  EXPECT_EQ(execution["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  EXPECT_FALSE(execution["valid"]);
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(execution["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(execution["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, ChecksumMismatchSerializesEvaluatedFalseInsteadOfMissingNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        LlmExecutorResult execution = successful_execution(model_plan);
        if (context.kind == LlmRunnerTaskKind::Measurement) {
          execution.checksum_valid = false;
        }
        return execution;
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["status"], "invalid");
  EXPECT_EQ(measurement["reason_code"], LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_EQ(measurement["checksum"]["status"], "invalid");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_FALSE(measurement["checksum"]["checksum_valid"]);
  EXPECT_TRUE(measurement["checksum"]["expected_worker_checksums"].is_array());
  EXPECT_TRUE(measurement["checksum"]["actual_worker_checksums"].is_array());
}

TEST(LlmMemoryJsonTest, PagedPostValidationFailureSerializesEvaluatedInvalidEvidence) {
  const LlmMemoryConfig config = paged_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend(
      [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context) {
        LlmExecutorResult execution = successful_execution(model_plan);
        if (context.kind == LlmRunnerTaskKind::Measurement) {
          execution.valid = false;
          execution.reason_code = LlmExecutorReason::PAGED_POST_VALIDATION_FAILED;
          execution.post_validation_valid = false;
        }
        return execution;
      });
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["status"], "invalid");
  EXPECT_EQ(measurement["reason_code"], LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_TRUE(measurement["execution"]["post_validation_evaluated"]);
  EXPECT_FALSE(measurement["execution"]["post_validation_valid"]);
}

TEST(LlmMemoryJsonTest, CommonAcceptanceFailureCannotSerializeCompleteBackendEvidenceAsValid) {
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

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& execution = document["measurements"][0]["execution"];
  EXPECT_EQ(execution["status"], "invalid");
  EXPECT_EQ(execution["reason_code"], LlmBackendReason::TASK_COMPLETION_MISMATCH);
  EXPECT_FALSE(execution["valid"]);
}

TEST(LlmMemoryJsonTest, MeasurementCheckpointSerializesPartialStatusAndUnavailableTailContract) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend;
  LlmMemoryResult partial_snapshot;
  bool captured_partial_snapshot = false;
  LlmRunnerHooks hooks;
  hooks.checkpoint = [&](const LlmMemoryResult& checkpoint_result, LlmCheckpointKind kind) {
    if (!captured_partial_snapshot && kind == LlmCheckpointKind::MeasurementTerminal) {
      partial_snapshot = checkpoint_result;
      captured_partial_snapshot = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks), EXIT_SUCCESS);
  ASSERT_TRUE(captured_partial_snapshot);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), partial_snapshot);
  EXPECT_EQ(document["status"], "partial");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::PARTIAL_RESULTS);
  EXPECT_FALSE(document["interruption_requested"]);
  EXPECT_FALSE(document["results_complete"]);
  EXPECT_FALSE(document["conclusions_valid"]);
  EXPECT_FALSE(document["scenario_order_balance_complete"]);
  EXPECT_EQ(document["counters"]["planned_measurements"], 3u);
  EXPECT_EQ(document["counters"]["attempted_measurements"], 1u);
  EXPECT_EQ(document["counters"]["terminal_measurements"], 1u);
  EXPECT_EQ(document["counters"]["measured_measurements"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["logical_checkpoint_attempts"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["successful_logical_checkpoints"], 1u);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_attempted"]);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_completed"]);

  size_t measured_count = 0;
  size_t not_run_count = 0;
  for (const OrderedJson& measurement : document["measurements"]) {
    if (measurement["status"] == "measured") {
      ++measured_count;
      EXPECT_EQ(measurement["reason_code"], "measured");
      EXPECT_TRUE(measurement["attempted"]);
      EXPECT_TRUE(measurement["completed_work_units"].is_number_unsigned());
      EXPECT_TRUE(measurement["elapsed_seconds"].is_number());
      EXPECT_EQ(measurement["execution"]["status"], "valid");
      EXPECT_EQ(measurement["checksum"]["status"], "valid");
      EXPECT_TRUE(measurement["checksum"]["checksum_valid"]);
      continue;
    }

    ASSERT_EQ(measurement["status"], "not_run");
    ++not_run_count;
    EXPECT_EQ(measurement["reason_code"], "not-run");
    EXPECT_FALSE(measurement["attempted"]);
    EXPECT_TRUE(measurement["qos_successful_workers"].is_null());
    EXPECT_EQ(measurement["completed_work_units"], 0u);
    EXPECT_EQ(measurement["completed_effective_model_payload_bytes"], "0");
    EXPECT_TRUE(measurement["elapsed_seconds"].is_null());
    EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_null());
    EXPECT_EQ(measurement["execution"]["status"], "not_run");
    EXPECT_EQ(measurement["execution"]["reason_code"], "not-run");
    EXPECT_TRUE(measurement["execution"]["valid"].is_null());
    EXPECT_TRUE(measurement["execution"]["requested_workers"].is_null());
    EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
    EXPECT_EQ(measurement["checksum"]["reason_code"], "not-run");
    EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
  }
  EXPECT_EQ(measured_count, 1u);
  EXPECT_EQ(not_run_count, 2u);
}

TEST(LlmMemoryJsonTest, CheckpointFailureRetainsMeasuredPrefixAndNullFailedTailWithoutRetry) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend backend;
  LlmRunnerHooks hooks;
  hooks.checkpoint = [](const LlmMemoryResult&, LlmCheckpointKind) { return EXIT_FAILURE; };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "failed");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(document["checkpoint_lifecycle"]["checkpoint_failed"]);
  EXPECT_EQ(document["checkpoint_lifecycle"]["logical_checkpoint_attempts"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["successful_logical_checkpoints"], 0u);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_attempted"]);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_completed"]);
  EXPECT_EQ(document["measurements"][0]["status"], "measured");
  EXPECT_EQ(document["measurements"][1]["status"], "failed");
  EXPECT_TRUE(document["measurements"][1]["qos_successful_workers"].is_null());
  EXPECT_TRUE(document["measurements"][1]["effective_model_payload_gb_s"].is_null());
  EXPECT_TRUE(document["measurements"][1]["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, QualityWarningsMergeAndDeduplicateInStableConsoleAgreementOrder) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMemoryResult result = complete_result(config, plan);
  result.quality_warnings = {"mixed-high-cv", "environment-not-nominal", "mixed-high-cv"};
  for (LlmMeasurementState& measurement : result.measurements) {
    measurement.duration_quality = "above-target-single-work-unit";
  }
  result.measurements[0].qos_failed_workers = 1;

  LlmResultMetadata metadata = fixed_metadata(config, plan);
  metadata.environment_end.thermal_state = "serious";
  metadata.main_thread_qos = {true, false, 1};
  metadata.l2_data_cache_bytes = 2 * Constants::BYTES_PER_MB;

  const OrderedJson document = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);
  EXPECT_EQ(json_string_array(document["quality_warnings"]),
            (std::vector<std::string>{
                "mixed-high-cv", "environment-not-nominal", "main-thread-qos-not-applied", "worker-qos-not-applied",
                "weight-working-set-cache-dominant", "kv-working-set-cache-dominant",
                "weights_only-duration-above-target-single-work-unit", "kv_only-duration-above-target-single-work-unit",
                "mixed-duration-above-target-single-work-unit"}));
}

TEST(LlmMemoryJsonTest, LoopRecordsExposeOnlyRealizedPrefixAndAllMeasurementIndexes) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMemoryResult result = complete_result(config, plan);
  result.loops[0].realized_order_count = 1;

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& loop = document["loop_records"][0];
  EXPECT_EQ(loop["planned_order"].size(), kLlmScenarioCount);
  EXPECT_EQ(loop["realized_order"].size(), 1u);
  EXPECT_EQ(loop["realized_order"][0], "weights_only");
  EXPECT_EQ(loop["measurement_indexes"].size(), kLlmScenarioCount);
  EXPECT_EQ(loop["measurement_indexes"], (OrderedJson::array({0, 1, 2})));
}
