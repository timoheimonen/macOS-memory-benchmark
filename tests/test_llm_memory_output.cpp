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

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/config/constants.h"
#include "llm_memory/llm_cpu_backend.h"
#include "llm_memory/llm_json.h"
#include "llm_memory/llm_output.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"

namespace {

const LlmCpuExecutionPlan& cpu_execution_plan(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    throw std::logic_error("expected CPU execution plan");
  }
  return *cpu_plan;
}

void set_headline(LlmMemoryResult& result, LlmScenario scenario, double latency_seconds, double work_units_per_second,
                  double bandwidth_gb_s) {
  LlmScenarioAggregate& aggregate = result.aggregates[static_cast<size_t>(scenario)];
  aggregate.scenario = scenario;
  aggregate.work_unit_latency_seconds.headline = latency_seconds;
  aggregate.synthetic_memory_work_units_per_second.headline = work_units_per_second;
  aggregate.effective_model_payload_gb_s.headline = bandwidth_gb_s;
}

LlmMemoryWorkPlan make_console_plan() {
  LlmMemoryWorkPlan plan;
  plan.valid = true;
  plan.backend = LlmMemoryBackend::Cpu;
  plan.phase = LlmPhase::Decode;
  plan.kv_layout = LlmKvLayout::Contiguous;
  plan.work_unit_kind = LlmWorkUnitKind::DecodeStep;
  plan.geometry.valid = true;
  plan.geometry.phase = LlmPhase::Decode;
  plan.geometry.kv_layout = LlmKvLayout::Contiguous;
  plan.geometry.work_unit_kind = LlmWorkUnitKind::DecodeStep;
  plan.geometry.decode = LlmDecodeGeometry{8};
  plan.geometry.active_weight_bytes_per_work_unit = 1024;
  plan.geometry.kv_read_bytes_per_work_unit = 768;
  plan.geometry.kv_write_bytes_per_work_unit = 256;
  plan.geometry.kv_capacity_bytes = 2048;
  plan.geometry.traffic_crossover_context_tokens = 4.0;
  return plan;
}

LlmMemoryWorkPlan make_paged_console_plan() {
  LlmMemoryWorkPlan plan = make_console_plan();
  plan.kv_layout = LlmKvLayout::Paged;
  plan.geometry.kv_layout = LlmKvLayout::Paged;
  plan.geometry.kv_block_tokens = 4;
  plan.geometry.kv_blocks_per_sequence = 2;
  plan.geometry.physical_blocks_per_layer = 2;
  plan.geometry.total_physical_blocks = 4;
  plan.geometry.kv_block_bytes = 128;
  plan.geometry.last_block_tokens = 4;
  plan.geometry.last_block_valid_bytes = 128;
  plan.geometry.k_logical_bytes = 384;
  plan.geometry.v_logical_bytes = 384;
  plan.geometry.k_mapping_bytes = 512;
  plan.geometry.v_mapping_bytes = 512;
  plan.geometry.k_layout_padding_bytes = 128;
  plan.geometry.v_layout_padding_bytes = 128;
  LlmCpuExecutionPlan* const cpu = get_llm_cpu_execution_plan(plan);
  if (cpu == nullptr) {
    throw std::logic_error("expected mutable CPU execution plan");
  }
  cpu->paged.emplace();
  cpu->paged->layout.valid = true;
  cpu->paged->layout.kv_block_tokens = 4;
  cpu->paged->layout.blocks_per_sequence = 2;
  cpu->paged->layout.physical_blocks_per_layer = 2;
  cpu->paged->layout.total_physical_blocks = 4;
  cpu->paged->layout.block_bytes = 128;
  cpu->paged->layout.last_block_tokens = 4;
  cpu->paged->layout.last_block_valid_bytes = 128;
  cpu->paged->layout.block_table_entries = 2;
  cpu->paged->block_table_logical_bytes = 8;
  cpu->paged->block_table_mapping_bytes = 4096;
  cpu->paged->permutation.algorithm_version = "permutation-v1";
  cpu->paged->permutation.resolved_seed = 99;
  cpu->paged->permutation.sha256 = "0123456789abcdef";
  cpu->paged->permutation.identity = "paged-permutation-identity";
  cpu->paged->ownership.valid = true;
  cpu->paged->ownership
      .total_layout_metadata_lookup_count_per_work_unit = 20;
  cpu->paged->ownership
      .total_layout_metadata_read_bytes_per_work_unit = 80;
  cpu->paged->ownership.total_accounted_bytes_per_work_unit = 1104;
  return plan;
}

LlmMemoryWorkPlan make_prefill_console_plan() {
  LlmMemoryWorkPlan plan = make_console_plan();
  plan.phase = LlmPhase::Prefill;
  plan.work_unit_kind = LlmWorkUnitKind::PrefillOperation;
  plan.geometry.phase = LlmPhase::Prefill;
  plan.geometry.work_unit_kind = LlmWorkUnitKind::PrefillOperation;
  plan.geometry.decode.reset();
  plan.geometry.prefill = LlmPrefillGeometry{5, 2, 3, 11, 15, 60, 480, 0};
  plan.geometry.traffic_crossover_context_tokens = 3.5;
  return plan;
}

LlmMemoryWorkPlan make_paged_prefill_console_plan() {
  LlmMemoryWorkPlan plan = make_prefill_console_plan();
  plan.kv_layout = LlmKvLayout::Paged;
  plan.geometry.kv_layout = LlmKvLayout::Paged;
  plan.geometry.layer_count = 2;
  plan.geometry.batch_size = 1;
  plan.geometry.kv_block_tokens = 2;
  plan.geometry.kv_blocks_per_sequence = 3;
  plan.geometry.physical_blocks_per_layer = 3;
  plan.geometry.total_physical_blocks = 6;
  plan.geometry.kv_block_bytes = 128;
  plan.geometry.last_block_tokens = 1;
  plan.geometry.last_block_valid_bytes = 64;
  plan.geometry.k_logical_bytes = 640;
  plan.geometry.v_logical_bytes = 640;
  plan.geometry.k_mapping_bytes = 768;
  plan.geometry.v_mapping_bytes = 768;
  plan.geometry.k_layout_padding_bytes = 128;
  plan.geometry.v_layout_padding_bytes = 128;
  plan.geometry.block_table_entries = 3;
  plan.geometry.block_table_bytes = 12;
  plan.geometry.layout_metadata_lookups_per_layer_sequence_per_work_unit = 15;
  plan.geometry.kv_read_bytes_per_work_unit = 2816;
  plan.geometry.kv_write_bytes_per_work_unit = 1280;
  plan.geometry.kv_only_effective_model_payload_bytes_per_work_unit = 4096;
  plan.geometry.mixed_effective_model_payload_bytes_per_work_unit = 5120;

  LlmCpuExecutionPlan* const cpu = get_llm_cpu_execution_plan(plan);
  if (cpu == nullptr) {
    throw std::logic_error("expected mutable CPU execution plan");
  }
  cpu->paged.emplace();
  cpu->paged->layout.valid = true;
  cpu->paged->layout.kv_block_tokens = 2;
  cpu->paged->layout.blocks_per_sequence = 3;
  cpu->paged->layout.physical_blocks_per_layer = 3;
  cpu->paged->layout.total_physical_blocks = 6;
  cpu->paged->layout.block_bytes = 128;
  cpu->paged->layout.last_block_tokens = 1;
  cpu->paged->layout.last_block_valid_bytes = 64;
  cpu->paged->layout.block_table_entries = 3;
  cpu->paged->block_table_logical_bytes = 12;
  cpu->paged->block_table_mapping_bytes = 4096;
  cpu->paged->permutation.algorithm_version = "permutation-v1";
  cpu->paged->permutation.resolved_seed = 101;
  cpu->paged->permutation.sha256 = "fedcba9876543210";
  cpu->paged->permutation.identity = "paged-prefill-permutation-identity";
  return plan;
}

LlmMemoryConfig fake_runner_config() {
  LlmMemoryConfig config;
  config.weight_size_mb = 1;
  config.layer_count = 1;
  config.query_head_count = 1;
  config.kv_head_count = 1;
  config.head_dimension = 16;
  config.kv_element_bytes = 1;
  config.visible_context_tokens = 2;
  config.batch_size = 1;
  config.requested_workers = 1;
  config.available_workers = 1;
  config.iterations = 4;
  config.loop_count = 3;
  config.seed = 42;
  config.user_specified_iterations = true;
  config.user_specified_seed = true;
  config.user_specified_workers = true;
  return config;
}

LlmMemoryWorkPlanRequest fake_runner_plan_request(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request;
  request.geometry.active_weight_bytes = config.weight_size_mb * Constants::BYTES_PER_MB;
  request.geometry.layer_count = config.layer_count;
  request.geometry.query_head_count = config.query_head_count;
  request.geometry.kv_head_count = config.kv_head_count;
  request.geometry.head_dimension = config.head_dimension;
  request.geometry.kv_element_bytes = config.kv_element_bytes;
  request.geometry.visible_context_tokens = config.visible_context_tokens;
  request.geometry.batch_size = config.batch_size;
  request.requested_workers = config.requested_workers;
  request.available_workers = config.available_workers;
  request.available_memory_bytes = 8ULL * 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = 1;
  request.base_seed = config.seed;
  return request;
}

LlmMemoryWorkPlan make_fake_runner_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = fake_runner_plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  if (!preliminary.valid) {
    return build_llm_memory_work_plan(request);
  }

  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  if (!executor.valid || !runner.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return build_llm_memory_work_plan(request);
  }
  return build_llm_memory_work_plan(request);
}

LlmExecutorResult successful_fake_execution(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  LlmExecutorResult execution;
  execution.valid = true;
  execution.reason_code = LlmExecutorReason::VALID;
  execution.elapsed_seconds = 0.150;
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
}

class FakeLlmBackend final : public LlmBackend {
 public:
  FakeLlmBackend() { evidence_.backend = LlmMemoryBackend::Cpu; }

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

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan&) noexcept override {
    evidence_.backend_evidence = LlmCpuBackendEvidence{};
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan,
                                      const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) override {
    return adapt_llm_cpu_executor_result(model_plan, scenario_plan, context,
                                         successful_fake_execution(model_plan));
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.release;
  }

 private:
  LlmBackendEvidence evidence_;
};

size_t count_substrings(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t position = 0;
  while (!needle.empty() && (position = text.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

}  // namespace

TEST(LlmMemoryOutputTest, PrintsExactPayloadHeadlinesAndInterpretation) {
  LlmMemoryWorkPlan plan = make_console_plan();
  LlmResultMetadata metadata;
  metadata.l2_data_cache_bytes = 512;
  metadata.main_thread_qos.requested = true;
  metadata.main_thread_qos.applied = true;
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end.thermal_state = "nominal";

  LlmMemoryResult result;
  set_headline(result, LlmScenario::WeightsOnly, 0.00125, 800.0, 100.5);
  set_headline(result, LlmScenario::KvOnly, 0.0025, 400.0, 50.25);
  set_headline(result, LlmScenario::Mixed, 0.00375, 266.666, 80.25);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(errors.empty());
  EXPECT_EQ(output,
            "Synthetic LLM memory profile (backend=cpu, phase=decode, "
            "work_unit=decode_step, kv_layout=contiguous, warm/cacheable)\n"
            "  Visible context tokens:     8\n"
            "  Traffic crossover:          4.00 visible context tokens\n"
            "  Active weight bytes / decode step: 1024\n"
            "  KV read bytes / decode step:       768\n"
            "  KV write bytes / decode step:      256\n\n"
            "  Weights only: 1.250 ms/decode step, "
            "100.50 GB/s effective model payload\n"
            "  KV only:      2.500 ms/decode step, "
            "50.25 GB/s effective model payload\n"
            "  Mixed:        3.750 ms/decode step, 266.67 synthetic decode steps/s, "
            "80.25 GB/s effective model payload\n"
            "  Interpretation: each decode step is synthetic memory-only work, not an inference "
            "token; effective model payload is logical, not physical DRAM-counter traffic.\n"
            "  Phase/layout: phase=decode, kv_layout=contiguous; the visible context includes the "
            "current-token slot; KV uses layer/batch/token/head/dimension order.\n"
            "  Crossover: logical weight/KV-read payload equality is not a proven hardware "
            "bottleneck transition.\n"
            "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
            "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence.\n");
}

TEST(LlmMemoryOutputTest,
     PagedReportIdentifiesBlockGeometryAndTimedMetadataOutsidePayload) {
  LlmMemoryWorkPlan plan = make_paged_console_plan();
  LlmResultMetadata metadata;
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end = metadata.environment_start;
  LlmMemoryResult result;

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(errors.empty());
  EXPECT_EQ(count_substrings(output, "kv_layout=paged"), 2u);
  EXPECT_EQ(count_substrings(output, "  Paged KV block tokens (G): 4\n"),
            1u);
  EXPECT_EQ(count_substrings(output, "  Blocks per sequence (N):   2\n"),
            1u);
  EXPECT_EQ(
      count_substrings(output, "  Physical blocks/layer (P_b): 2\n"),
      1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Physical block geometry: total_blocks=4, "
                "block_bytes=128\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output, "  Terminal block: tokens=4, valid_bytes=128\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  K bytes (logical/physical/padding): 384/512/128\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  V bytes (logical/physical/padding): 384/512/128\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Block table: 2 uint32 entries, 8 bytes, 4096 "
                "page-rounded bytes\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Permutation: version=permutation-v1, seed=99, "
                "sha256=0123456789abcdef\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Permutation identity: paged-permutation-identity\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Timed block-table metadata / KV-active decode step: "
                "20 lookups, 80 bytes\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Accounted bytes / KV-active decode step: 1104\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output,
                "  Effective model payload excludes timed block-table "
                "metadata bytes.\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output, "  KV read bytes / decode step:       768\n"),
            1u);
  EXPECT_EQ(count_substrings(
                output, "  KV write bytes / decode step:      256\n"),
            1u);
}

TEST(LlmMemoryOutputTest, PrefillReportIdentifiesFullPromptWorkAndTiledCausalGeometry) {
  LlmMemoryWorkPlan plan = make_prefill_console_plan();
  LlmResultMetadata metadata;
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end = metadata.environment_start;
  LlmMemoryResult result;
  set_headline(result, LlmScenario::Mixed, 0.004, 250.0, 75.0);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(errors.empty());
  EXPECT_NE(output.find("Synthetic LLM memory profile (backend=cpu, phase=prefill, "
                        "work_unit=prefill_operation, kv_layout=contiguous"),
            std::string::npos);
  EXPECT_NE(output.find("  Prompt tokens (P):                 5\n"), std::string::npos);
  EXPECT_NE(output.find("  Attention query tile tokens (Q):  2\n"), std::string::npos);
  EXPECT_NE(output.find("  Attention query tiles (C):        3\n"), std::string::npos);
  EXPECT_NE(output.find("  Prefix token visits / sequence:   11\n"), std::string::npos);
  EXPECT_NE(output.find("  Causal token pairs / sequence:    15\n"), std::string::npos);
  EXPECT_NE(output.find("  Logical attention pairs:          60\n"), std::string::npos);
  EXPECT_NE(output.find("  Logical attention FMA terms:      480\n"), std::string::npos);
  EXPECT_NE(output.find("  Mixed:        4.000 ms/prefill operation, 250.00 synthetic "
                        "prefill operations/s, 75.00 GB/s effective model payload\n"),
            std::string::npos);
  EXPECT_NE(output.find("prefill performs no Transformer compute and does "
                        "not predict TTFT"),
            std::string::npos);
  EXPECT_EQ(output.find("Crossover:"), std::string::npos);
  EXPECT_EQ(output.find("tokens/s"), std::string::npos);
}

TEST(LlmMemoryOutputTest, PagedPrefillReportUsesKvActivePrefillAccountingWithoutDecodeOwnership) {
  const LlmMemoryWorkPlan plan = make_paged_prefill_console_plan();
  LlmResultMetadata metadata;
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end = metadata.environment_start;
  LlmMemoryResult result;

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(errors.empty());
  EXPECT_NE(output.find("Synthetic LLM memory profile (backend=cpu, phase=prefill, "
                        "work_unit=prefill_operation, kv_layout=paged"),
            std::string::npos);
  EXPECT_EQ(count_substrings(output, "  Paged KV block tokens (G): 2\n"), 1u);
  EXPECT_EQ(count_substrings(output, "  Blocks per sequence (N):   3\n"), 1u);
  EXPECT_EQ(count_substrings(output,
                             "  Timed block-table metadata / KV-active prefill operation: "
                             "30 lookups, 120 bytes\n"),
            1u);
  EXPECT_EQ(count_substrings(output, "  Accounted bytes / KV-active prefill operation: 4216\n"), 1u);
  EXPECT_EQ(count_substrings(output,
                             "  Effective model payload excludes timed block-table metadata bytes.\n"),
            1u);
}

TEST(LlmMemoryOutputTest, EmitsDeduplicatedWarningsInContractOrder) {
  LlmMemoryWorkPlan plan = make_console_plan();
  LlmResultMetadata metadata;
  metadata.l2_data_cache_bytes = 4096;
  metadata.main_thread_qos.requested = true;
  metadata.main_thread_qos.applied = false;
  metadata.main_thread_qos.code = 7;
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end.thermal_state = "serious";

  LlmMemoryResult result;
  result.aggregates[static_cast<size_t>(LlmScenario::KvOnly)]
      .effective_model_payload_gb_s.statistics.coefficient_of_variation_pct = 6.25;
  result.quality_warnings = {"kv_only-high-cv", "kv_only-high-cv", "scenario-order-not-balanced"};

  LlmMeasurementState mixed;
  mixed.scenario = LlmScenario::Mixed;
  mixed.status = LlmMeasurementStatus::Measured;
  mixed.qos_failed_workers = 1;
  mixed.duration_quality = "above-target-single-work-unit";
  result.measurements.push_back(mixed);
  result.measurements.push_back(mixed);

  LlmMeasurementState kv;
  kv.scenario = LlmScenario::KvOnly;
  kv.status = LlmMeasurementStatus::Measured;
  kv.duration_quality = "above-target-window";
  result.measurements.push_back(kv);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(errors,
            "Warning: LLM KV only repeatability CV 6.25% exceeds 5.00%\n"
            "Warning: LLM scenario order is not fully balanced across completed loops\n"
            "Warning: LLM result environment is not reference-eligible "
            "(thermal state or Low Power Mode)\n"
            "Warning: LLM main-thread QoS request was not applied (code: 7)\n"
            "Warning: One or more LLM worker QoS requests were not applied\n"
            "Warning: LLM weight working set (1024 bytes) does not exceed reported L2 cache "
            "(4096 bytes); the result may be cache-dominant\n"
            "Warning: LLM KV working set (2048 bytes) does not exceed reported L2 cache "
            "(4096 bytes); the result may be cache-dominant\n"
            "Warning: LLM Mixed duration quality is above-target-single-work-unit\n"
            "Warning: LLM KV only duration quality is above-target-window\n");
}

TEST(LlmMemoryOutputTest, ConsoleHeadlinesAgreeExactlyWithJsonFromSameFakeRunnerResult) {
  const LlmMemoryConfig config = fake_runner_config();
  const LlmMemoryWorkPlan plan = make_fake_runner_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  FakeLlmBackend backend;
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_SUCCESS);
  ASSERT_TRUE(result.results_complete);

  LlmResultMetadata metadata;
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_end = metadata.environment_start;
  const nlohmann::ordered_json document =
      build_llm_memory_json(config, plan, LlmResourcePreparationResult{}, metadata, result);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_llm_memory_console_report(plan, metadata, result);
  const std::string errors = testing::internal::GetCapturedStderr();
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(errors.empty()) << errors;

  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const size_t index = static_cast<size_t>(scenario);
    const LlmScenarioAggregate& aggregate = result.aggregates[index];
    ASSERT_TRUE(aggregate.work_unit_latency_seconds.headline.has_value());
    ASSERT_TRUE(aggregate.synthetic_memory_work_units_per_second.headline.has_value());
    ASSERT_TRUE(aggregate.effective_model_payload_gb_s.headline.has_value());

    const std::string scenario_token = llm_scenario_to_string(scenario);
    const nlohmann::ordered_json& json_aggregate =
        document["aggregates"]["scenarios"][scenario_token];
    const double json_latency = json_aggregate["synthetic_work_unit_latency_seconds"]["headline"].get<double>();
    const double json_work_units_per_second =
        json_aggregate["synthetic_memory_work_units_per_second"]["headline"]
            .get<double>();
    const double json_bandwidth = json_aggregate["effective_model_payload_gb_s"]["headline"].get<double>();
    EXPECT_DOUBLE_EQ(json_latency, *aggregate.work_unit_latency_seconds.headline);
    EXPECT_DOUBLE_EQ(json_work_units_per_second, *aggregate.synthetic_memory_work_units_per_second.headline);
    EXPECT_DOUBLE_EQ(json_bandwidth, *aggregate.effective_model_payload_gb_s.headline);

    const std::string expected_line = Messages::report_llm_memory_scenario_headline(
        Messages::report_llm_memory_scenario_name(scenario_token), "decode step",
        "decode steps", json_latency * 1000.0, json_work_units_per_second,
        json_bandwidth, scenario == LlmScenario::Mixed);
    EXPECT_EQ(count_substrings(output, expected_line), 1u) << expected_line << "\n" << output;
  }
}
