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
#include <string>

#include "core/config/constants.h"
#include "llm_memory/llm_json.h"
#include "llm_memory/llm_output.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"

namespace {

void set_headline(LlmMemoryResult& result, LlmScenario scenario, double latency_seconds, double steps_per_second,
                  double bandwidth_gb_s) {
  LlmScenarioAggregate& aggregate = result.aggregates[static_cast<size_t>(scenario)];
  aggregate.scenario = scenario;
  aggregate.step_latency_seconds.headline = latency_seconds;
  aggregate.synthetic_memory_steps_per_second.headline = steps_per_second;
  aggregate.effective_payload_gb_s.headline = bandwidth_gb_s;
}

LlmMemoryWorkPlan make_console_plan() {
  LlmMemoryWorkPlan plan;
  plan.valid = true;
  plan.geometry.valid = true;
  plan.geometry.active_weight_bytes_per_step = 1024;
  plan.geometry.kv_read_bytes_per_step = 768;
  plan.geometry.kv_append_write_bytes_per_step = 256;
  plan.geometry.kv_capacity_bytes = 2048;
  plan.geometry.traffic_crossover_context_tokens = 4.0;
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
  LlmExecutorResult execution;
  execution.valid = true;
  execution.reason_code = LlmExecutorReason::VALID;
  execution.elapsed_seconds = 0.150;
  execution.requested_workers = plan.effective_workers;
  execution.created_workers = plan.effective_workers;
  execution.completed_workers = plan.effective_workers;
  execution.qos_successful_workers = plan.effective_workers;
  execution.kernel_succeeded = true;
  execution.timer_started = true;
  execution.timer_stopped = true;
  execution.checksum_evaluated = true;
  execution.checksum_valid = true;
  execution.expected_checksums.resize(plan.effective_workers);
  execution.actual_checksums = execution.expected_checksums;
  execution.expected_run_checksum = {11, 22};
  execution.actual_run_checksum = execution.expected_run_checksum;
  return execution;
}

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
            "Synthetic LLM decode memory profile (CPU, fixed context, warm/cacheable)\n"
            "  Active weight bytes / step: 1024\n"
            "  KV read bytes / step:       768\n"
            "  KV append bytes / step:     256\n"
            "  Traffic crossover:          4.00 visible context tokens\n\n"
            "  Weights only: 1.250 ms/step, 100.50 GB/s effective payload\n"
            "  KV only:      2.500 ms/step, 50.25 GB/s effective payload\n"
            "  Mixed:        3.750 ms/step, 266.67 synthetic memory steps/s, "
            "80.25 GB/s effective payload\n"
            "  Interpretation: each step is synthetic memory-only work, not an inference token; "
            "effective payload is logical, not physical DRAM-counter traffic.\n"
            "  Context/layout: the fixed visible context includes the current-token slot; KV uses "
            "contiguous layer/batch/token/head/dimension layout.\n"
            "  Crossover: logical weight/KV-read payload equality is not a proven hardware "
            "bottleneck transition.\n"
            "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
            "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence.\n");
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
      .effective_payload_gb_s.statistics.coefficient_of_variation_pct = 6.25;
  result.quality_warnings = {"kv_only-high-cv", "kv_only-high-cv", "scenario-order-not-balanced"};

  LlmMeasurementState mixed;
  mixed.scenario = LlmScenario::Mixed;
  mixed.status = LlmMeasurementStatus::Measured;
  mixed.qos_failed_workers = 1;
  mixed.duration_quality = "single-step-over-target";
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
            "Warning: LLM Mixed duration quality is single-step-over-target\n"
            "Warning: LLM KV only duration quality is above-target-window\n");
}

TEST(LlmMemoryOutputTest, ConsoleHeadlinesAgreeExactlyWithJsonFromSameFakeRunnerResult) {
  const LlmMemoryConfig config = fake_runner_config();
  const LlmMemoryWorkPlan plan = make_fake_runner_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  LlmMemoryResult result;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) { return successful_fake_execution(model_plan); };
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
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
    ASSERT_TRUE(aggregate.step_latency_seconds.headline.has_value());
    ASSERT_TRUE(aggregate.synthetic_memory_steps_per_second.headline.has_value());
    ASSERT_TRUE(aggregate.effective_payload_gb_s.headline.has_value());

    const std::string scenario_token = llm_scenario_to_string(scenario);
    const nlohmann::ordered_json& json_aggregate = document["scenario_aggregates"][scenario_token];
    const double json_latency = json_aggregate["synthetic_step_latency_seconds"]["headline"].get<double>();
    const double json_steps_per_second = json_aggregate["synthetic_memory_steps_per_second"]["headline"].get<double>();
    const double json_bandwidth = json_aggregate["effective_payload_gb_s"]["headline"].get<double>();
    EXPECT_DOUBLE_EQ(json_latency, *aggregate.step_latency_seconds.headline);
    EXPECT_DOUBLE_EQ(json_steps_per_second, *aggregate.synthetic_memory_steps_per_second.headline);
    EXPECT_DOUBLE_EQ(json_bandwidth, *aggregate.effective_payload_gb_s.headline);

    const std::string expected_line = Messages::report_llm_memory_scenario_headline(
        Messages::report_llm_memory_scenario_name(scenario_token), json_latency * 1000.0, json_steps_per_second,
        json_bandwidth, scenario == LlmScenario::Mixed);
    EXPECT_EQ(count_substrings(output, expected_line), 1u) << expected_line << "\n" << output;
  }
}
