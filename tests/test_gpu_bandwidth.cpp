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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "gpu_bandwidth/gpu_backend.h"
#include "gpu_bandwidth/gpu_bandwidth.h"
#include "gpu_bandwidth/gpu_json.h"
#include "gpu_bandwidth/gpu_runner.h"

namespace {

using Json = nlohmann::ordered_json;

size_t operation_index(GpuOperation operation) {
  return static_cast<size_t>(operation);
}

struct PhaseCall {
  std::string phase;
  size_t occurrence = 0;
  GpuBackendAttemptRequest request;
};

class FakeGpuBackend final : public GpuBackend {
 public:
  FakeGpuBackend() {
    initialization.status = GpuBackendStatus::Success;
    initialization.reason_code = "initialized";
    initialization.device.macos_product_version = "15.5";
    initialization.device.macos_build = "24F74";
    initialization.device.hardware_model = "Mac16,10";
    initialization.device.physical_memory_bytes =
        16ULL * 1024ULL * Constants::BYTES_PER_MB;
    initialization.device.device_name = "Fake Apple GPU";
    initialization.device.registry_id = 18446744073709551615ULL;
    initialization.device.has_unified_memory = true;
    initialization.device.required_apple7_family_supported = true;
    initialization.device.supported_families = {"apple7", "apple9"};
    initialization.device.max_buffer_length =
        4ULL * 1024ULL * Constants::BYTES_PER_MB;
    initialization.device.recommended_max_working_set_size =
        8ULL * 1024ULL * Constants::BYTES_PER_MB;
    initialization.device.available_memory_bytes =
        4ULL * 1024ULL * Constants::BYTES_PER_MB;
    initialization.device.available_memory_source = "fake-provider";
    for (GpuPipelineMetadata* pipeline :
         {&initialization.device.read_pipeline,
          &initialization.device.write_pipeline,
          &initialization.device.copy_pipeline,
          &initialization.device.initialization_pipeline,
          &initialization.device.validation_pipeline}) {
      pipeline->thread_execution_width = 32;
      pipeline->max_total_threads_per_threadgroup = 1024;
    }
    initialization.device.read_pipeline.label = "gpu.read";
    initialization.device.write_pipeline.label = "gpu.write";
    initialization.device.copy_pipeline.label = "gpu.copy";
    initialization.device.initialization_pipeline.label = "gpu.init";
    initialization.device.validation_pipeline.label = "gpu.validation";
    initialization.compilation.kernel_revision = "fake-kernel-v1";
    initialization.compilation.kernel_source_sha256 =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    initialization.compilation.compiler_identifier = "fake-clang";
    initialization.compilation.build_sdk = "macosx26.0";
    initialization.compilation.deployment_target = "11.0";

    allocation.status = GpuBackendStatus::Success;
    allocation.reason_code = "allocated";
    allocation.buffer_a = make_resource("gpu.buffer.a");
    allocation.buffer_b = make_resource("gpu.buffer.b");
    allocation.status_buffer = make_resource("gpu.status");

    release.status = GpuBackendStatus::Success;
    release.reason_code = "released";
    environment.thermal_state = "nominal";
    environment.low_power_mode_available = true;
  }

  GpuBackendInitialization initialize() noexcept override {
    lifecycle_log.push_back("initialize");
    return initialization;
  }

  GpuAllocationResult allocate_resources(
      const GpuAllocationRequest& request) noexcept override {
    lifecycle_log.push_back("allocate");
    last_allocation_request = request;
    GpuAllocationResult output = allocation;
    output.requested_buffer_size_bytes = request.buffer_size_bytes;
    output.auxiliary_bytes = request.auxiliary_bytes;
    output.memory_budget_bytes = request.memory_budget_bytes;
    output.required_total_bytes =
        request.buffer_size_bytes * 2U + request.auxiliary_bytes;
    output.recommended_working_set_available = true;
    output.recommended_working_set_headroom_bytes =
        static_cast<int64_t>(
            initialization.device.recommended_max_working_set_size) -
        static_cast<int64_t>(output.required_total_bytes);
    output.recommended_working_set_headroom_fraction =
        static_cast<double>(output.recommended_working_set_headroom_bytes) /
        static_cast<double>(
            initialization.device.recommended_max_working_set_size);
    output.buffer_a.length_bytes = request.buffer_size_bytes;
    output.buffer_b.length_bytes = request.buffer_size_bytes;
    output.status_buffer.length_bytes = request.auxiliary_bytes;
    if (output.status == GpuBackendStatus::Success) {
      resources_allocated = true;
      environment.current_allocated_size = output.required_total_bytes;
      output.current_allocated_size_peak = output.required_total_bytes;
    }
    return output;
  }

  GpuEnvironmentSnapshot snapshot_environment() noexcept override {
    const size_t occurrence = ++environment_snapshot_calls;
    return environment_provider ? environment_provider(occurrence)
                                : environment;
  }

  GpuBackendPhaseResult run_warmup(
      const GpuBackendAttemptRequest& request) noexcept override {
    const size_t occurrence = ++warmup_calls;
    GpuBackendPhaseResult output = successful_phase("warmup-complete");
    output.data_initialization_dispatch_count =
        request.operation == GpuOperation::Copy ? 2 : 1;
    output.benchmark_operation_dispatch_count = 1;
    output.status_reset_count = 1;
    output.dispatch_count =
        output.data_initialization_dispatch_count + 1;
    record_phase("warmup", occurrence, request);
    invoke_phase_hook("warmup", occurrence, request);
    return output;
  }

  GpuBackendPhaseResult run_precondition(
      const GpuBackendAttemptRequest& request) noexcept override {
    const size_t occurrence = ++precondition_calls;
    GpuBackendPhaseResult output =
        successful_phase("precondition-complete");
    output.data_initialization_dispatch_count =
        request.operation == GpuOperation::Copy ? 2 : 1;
    output.status_reset_count = 1;
    output.dispatch_count = output.data_initialization_dispatch_count;
    record_phase("precondition", occurrence, request);
    invoke_phase_hook("precondition", occurrence, request);
    return output;
  }

  GpuTimedResult run_timed(
      const GpuBackendAttemptRequest& request) noexcept override {
    const size_t occurrence = ++timed_calls;
    const size_t operation_occurrence =
        ++timed_calls_by_operation[operation_index(request.operation)];
    GpuTimedResult output;
    static_cast<GpuBackendPhaseResult&>(output) =
        successful_phase("timed-complete");
    output.dispatch_count = request.passes;
    output.benchmark_operation_dispatch_count = request.passes;
    const double elapsed = duration_provider
                               ? duration_provider(request,
                                                   operation_occurrence)
                               : 0.150;
    output.gpu_start_seconds = 1.0 + static_cast<double>(occurrence);
    output.gpu_elapsed_seconds = elapsed;
    output.gpu_end_seconds = output.gpu_start_seconds + elapsed;
    output.host_submit_seconds = output.gpu_start_seconds - 0.001;
    output.host_wait_end_seconds = output.gpu_end_seconds + 0.001;
    output.host_wall_seconds = elapsed + 0.002;
    output.queue_delay_available = true;
    output.queue_delay_seconds = 0.001;
    output.expected_accumulator = {0x12345678U, 0x9abcdef0U};
    output.actual_accumulator = output.expected_accumulator;
    if (timed_mutator) {
      timed_mutator(request, occurrence, output);
    }
    record_phase("timed", occurrence, request);
    invoke_phase_hook("timed", occurrence, request);
    return output;
  }

  GpuValidationResult run_validation(
      const GpuBackendAttemptRequest& request,
      const GpuTimedResult&) noexcept override {
    const size_t occurrence = ++validation_calls;
    GpuValidationResult output;
    output.status = GpuBackendStatus::Success;
    output.reason_code = "validation-passed";
    output.validation_status = GpuValidationStatus::Passed;
    output.expected_final_checksum = {0x11111111U, 0x22222222U};
    output.actual_final_checksum = output.expected_final_checksum;
    if (request.operation == GpuOperation::Read) {
      output.command_status = GpuCommandStatus::NotRun;
    } else {
      output.final_checksum_algorithm = "gpu-dual-mod32-v1";
      output.command_status = GpuCommandStatus::Completed;
      output.command_buffer_count = 1;
      output.compute_encoder_count = 1;
      output.dispatch_count = 1;
      output.validation_dispatch_count = 1;
      output.status_reset_count = 1;
    }
    if (validation_mutator) {
      validation_mutator(request, occurrence, output);
    }
    record_phase("validation", occurrence, request);
    invoke_phase_hook("validation", occurrence, request);
    return output;
  }

  GpuBackendPhaseResult readback_last_output(
      std::vector<uint8_t>& output) noexcept override {
    output.clear();
    return successful_phase("readback-complete");
  }

  GpuAllocationResult release_resources() noexcept override {
    lifecycle_log.push_back("release");
    resources_allocated = false;
    environment.current_allocated_size = 0;
    release.current_allocated_size_after_release = 0;
    return release;
  }

  GpuBackendInitialization initialization;
  GpuAllocationResult allocation;
  GpuAllocationResult release;
  GpuEnvironmentSnapshot environment;
  std::function<GpuEnvironmentSnapshot(size_t)> environment_provider;
  GpuAllocationRequest last_allocation_request;
  std::vector<std::string> lifecycle_log;
  std::vector<PhaseCall> phase_calls;
  std::function<double(const GpuBackendAttemptRequest&, size_t)>
      duration_provider;
  std::function<void(const GpuBackendAttemptRequest&, size_t,
                     GpuTimedResult&)>
      timed_mutator;
  std::function<void(const GpuBackendAttemptRequest&, size_t,
                     GpuValidationResult&)>
      validation_mutator;
  std::function<void(const std::string&, size_t,
                     const GpuBackendAttemptRequest&)>
      phase_hook;
  std::array<size_t, kGpuOperationCount> timed_calls_by_operation{};
  size_t environment_snapshot_calls = 0;
  size_t warmup_calls = 0;
  size_t precondition_calls = 0;
  size_t timed_calls = 0;
  size_t validation_calls = 0;
  bool resources_allocated = false;

 private:
  static GpuResourceMetadata make_resource(const std::string& label) {
    GpuResourceMetadata resource;
    resource.label = label;
    resource.storage_mode = label == "gpu.status" ? "shared" : "private";
    resource.cpu_cache_mode = "default-cache";
    resource.hazard_tracking_mode = "tracked";
    resource.resource_options = 18446744073709551615ULL;
    return resource;
  }

  static GpuBackendPhaseResult successful_phase(
      const std::string& reason_code) {
    GpuBackendPhaseResult output;
    output.status = GpuBackendStatus::Success;
    output.reason_code = reason_code;
    output.command_status = GpuCommandStatus::Completed;
    output.command_buffer_count = 1;
    output.compute_encoder_count = 1;
    return output;
  }

  void record_phase(const std::string& phase, size_t occurrence,
                    const GpuBackendAttemptRequest& request) {
    phase_calls.push_back({phase, occurrence, request});
  }

  void invoke_phase_hook(const std::string& phase, size_t occurrence,
                         const GpuBackendAttemptRequest& request) {
    if (phase_hook) {
      phase_hook(phase, occurrence, request);
    }
  }
};

GpuBandwidthConfig explicit_config(size_t loop_count = 3,
                                   size_t passes = 2) {
  GpuBandwidthConfig config;
  config.buffer_size_mb = 64;
  config.buffer_size_bytes = 64 * Constants::BYTES_PER_MB;
  config.iterations = passes;
  config.loop_count = loop_count;
  config.seed = 18446744073709551615ULL;
  config.user_specified_iterations = true;
  config.user_specified_seed = true;
  config.argv = {"memory_benchmark", "--gpu-bandwidth", "--buffer-size",
                 "64", "--iterations", std::to_string(passes), "--count",
                 std::to_string(loop_count), "--seed",
                 "18446744073709551615"};
  return config;
}

GpuBandwidthConfig automatic_config(size_t loop_count = 3) {
  GpuBandwidthConfig config = explicit_config(loop_count, 0);
  config.iterations = 0;
  config.user_specified_iterations = false;
  return config;
}

int parse_gpu_arguments(std::vector<std::string> arguments,
                        GpuBandwidthConfig& config) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  return parse_gpu_bandwidth_arguments(static_cast<int>(argv.size()),
                                       argv.data(), config);
}

int parse_gpu_arguments_silently(std::vector<std::string> arguments,
                                 GpuBandwidthConfig& config) {
  testing::internal::CaptureStderr();
  const int status = parse_gpu_arguments(std::move(arguments), config);
  static_cast<void>(testing::internal::GetCapturedStderr());
  return status;
}

void expect_interrupted_tail(const GpuRunResult& result,
                             size_t measured_prefix) {
  ASSERT_EQ(result.measurements.size(), result.counters.planned_measurements);
  for (size_t index = 0; index < result.measurements.size(); ++index) {
    const GpuMeasurement& measurement = result.measurements[index];
    if (index < measured_prefix) {
      EXPECT_EQ(measurement.status, GpuMeasurementStatus::Measured)
          << index;
      EXPECT_TRUE(measurement.value_gb_s.has_value()) << index;
    } else {
      EXPECT_EQ(measurement.status, GpuMeasurementStatus::Interrupted)
          << index;
      EXPECT_EQ(measurement.reason_code, "interruption-before-task")
          << index;
      EXPECT_FALSE(measurement.value_gb_s.has_value()) << index;
      EXPECT_FALSE(measurement.attempted) << index;
    }
  }
}

void expect_unstarted_failed_tail(const GpuRunResult& result,
                                  size_t first_unstarted) {
  ASSERT_LE(first_unstarted, result.measurements.size());
  for (size_t index = first_unstarted;
       index < result.measurements.size(); ++index) {
    const GpuMeasurement& measurement = result.measurements[index];
    EXPECT_EQ(measurement.status, GpuMeasurementStatus::Failed) << index;
    EXPECT_EQ(measurement.reason_code,
              "not-run-after-runtime-failure")
        << index;
    EXPECT_FALSE(measurement.value_gb_s.has_value()) << index;
    EXPECT_FALSE(measurement.attempted) << index;
  }
}

void expect_unstarted_interrupted_tail(const GpuRunResult& result,
                                       size_t first_unstarted) {
  ASSERT_LE(first_unstarted, result.measurements.size());
  for (size_t index = first_unstarted;
       index < result.measurements.size(); ++index) {
    const GpuMeasurement& measurement = result.measurements[index];
    EXPECT_EQ(measurement.status, GpuMeasurementStatus::Interrupted)
        << index;
    EXPECT_EQ(measurement.reason_code, "interruption-before-task")
        << index;
    EXPECT_FALSE(measurement.value_gb_s.has_value()) << index;
    EXPECT_FALSE(measurement.attempted) << index;
  }
}

void expect_unresolved_json_operation_identity(const Json& output) {
  const std::array<std::string, kGpuOperationCount> operations = {
      "read", "write", "copy"};
  ASSERT_EQ(output["work_plans"].size(), operations.size());
  for (size_t index = 0; index < operations.size(); ++index) {
    EXPECT_FALSE(output["work_plans"][index]["valid"].get<bool>());
    EXPECT_EQ(output["work_plans"][index]["operation"],
              operations[index]);
  }
  for (const Json& measurement : output["measurements"]) {
    EXPECT_EQ(measurement["status"], "not-run");
    EXPECT_EQ(measurement["work_policy"], "not-resolved");
    EXPECT_EQ(measurement["work_plan"]["operation"],
              measurement["operation"]);
  }
}

class GpuBandwidthParserTest : public ::testing::Test {
 protected:
  void TearDown() override {
    set_gpu_bandwidth_parser_test_hooks(nullptr);
  }
};

TEST_F(GpuBandwidthParserTest, DefaultsUseGeneratedSeedOnce) {
  GpuBandwidthParserTestHooks parser_hooks;
  parser_hooks.generated_seed = 0x123456789abcdef0ULL;
  set_gpu_bandwidth_parser_test_hooks(&parser_hooks);
  GpuBandwidthConfig config;

  ASSERT_EQ(parse_gpu_arguments(
                {"memory_benchmark", "--gpu-bandwidth"}, config),
            EXIT_SUCCESS);
  EXPECT_EQ(config.buffer_size_mb, 512U);
  EXPECT_EQ(config.buffer_size_bytes,
            512U * Constants::BYTES_PER_MB);
  EXPECT_EQ(config.loop_count, 3U);
  EXPECT_EQ(config.iterations, 0U);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_EQ(config.seed, 0x123456789abcdef0ULL);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_EQ(config.argv,
            (std::vector<std::string>{"memory_benchmark",
                                      "--gpu-bandwidth"}));
}

TEST_F(GpuBandwidthParserTest, UserValuesAndMaximumSeedAreExact) {
  GpuBandwidthConfig config;
  ASSERT_EQ(parse_gpu_arguments(
                {"memory_benchmark", "-G", "-b", "64", "-i", "7",
                 "-r", "4", "-o", "/tmp/gpu.json", "--seed",
                 "18446744073709551615"},
                config),
            EXIT_SUCCESS);
  EXPECT_EQ(config.buffer_size_bytes, 64U * Constants::BYTES_PER_MB);
  EXPECT_EQ(config.iterations, 7U);
  EXPECT_EQ(config.loop_count, 4U);
  EXPECT_EQ(config.output_file, "/tmp/gpu.json");
  EXPECT_EQ(config.seed, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(config.user_specified_iterations);
  EXPECT_TRUE(config.user_specified_seed);
}

TEST_F(GpuBandwidthParserTest,
       StrictDuplicateAndIncompatibleOptionsAreRejected) {
  const std::vector<std::vector<std::string>> invalid_arguments = {
      {"memory_benchmark", "-G", "--buffer-size", "64x"},
      {"memory_benchmark", "-G", "--count", "3x"},
      {"memory_benchmark", "-G", "--count", "2147483648"},
      {"memory_benchmark", "-G", "-b", "64", "--buffer-size", "64"},
      {"memory_benchmark", "-G", "--seed", "1", "--seed", "2"},
      {"memory_benchmark", "-G", "--analyze-core2core"},
      {"memory_benchmark", "-G", "--unknown"},
      {"memory_benchmark", "-G", "--iterations"},
      {"memory_benchmark", "-G", "--threads", "1"},
      {"memory_benchmark", "-G", "--cache-size", "1"},
      {"memory_benchmark", "-G", "--latency-samples", "1"},
      {"memory_benchmark", "-G", "--latency-stride-bytes", "256"},
      {"memory_benchmark", "-G", "--latency-chain-mode", "auto"},
      {"memory_benchmark", "-G", "--latency-tlb-locality-kb", "1024"},
      {"memory_benchmark", "-G", "--tlb-density", "high"},
      {"memory_benchmark", "-G", "--only-bandwidth"},
      {"memory_benchmark", "-G", "--only-latency"},
      {"memory_benchmark", "-G", "--non-cacheable"},
      {"memory_benchmark", "-G", "--benchmark"},
      {"memory_benchmark", "-G", "--patterns"},
      {"memory_benchmark", "-G", "--analyze-tlb"},
      {"memory_benchmark", "-G", "--sweep", "buffer-size=64"},
      {"memory_benchmark", "-G", "--sweep-max-runs", "1"},
      {"memory_benchmark", "-G", "-G"},
      {"memory_benchmark", "-G", "--help", "--help"},
  };
  for (const std::vector<std::string>& arguments : invalid_arguments) {
    GpuBandwidthConfig config;
    EXPECT_EQ(parse_gpu_arguments_silently(arguments, config), EXIT_FAILURE)
        << ::testing::PrintToString(arguments);
  }
}

TEST(GpuRunnerTest, PlannedMeasurementCountOverflowFailsBeforeBackend) {
  GpuBandwidthConfig config = explicit_config(
      std::numeric_limits<size_t>::max() / kGpuOperationCount + 1U, 1);
  FakeGpuBackend backend;
  GpuRunResult result;

  EXPECT_EQ(run_gpu_bandwidth_suite(config, backend, result), EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "planned-measurement-count-overflow");
  EXPECT_EQ(result.counters.planned_loops, config.loop_count);
  EXPECT_EQ(result.counters.planned_measurements, 0U);
  EXPECT_TRUE(result.loops.empty());
  EXPECT_TRUE(result.measurements.empty());
  EXPECT_TRUE(backend.lifecycle_log.empty());
  EXPECT_EQ(result.work_plans[0].operation, GpuOperation::Read);
  EXPECT_EQ(result.work_plans[1].operation, GpuOperation::Write);
  EXPECT_EQ(result.work_plans[2].operation, GpuOperation::Copy);
}

TEST_F(GpuBandwidthParserTest,
       MinimumBufferAndExplicitCopyPayloadCapFailBeforeRun) {
  GpuBandwidthConfig config;
  EXPECT_EQ(parse_gpu_arguments_silently(
                {"memory_benchmark", "-G", "-b", "63"}, config),
            EXIT_FAILURE);
  EXPECT_EQ(config.seed, 0U);

  EXPECT_EQ(parse_gpu_arguments_silently(
                {"memory_benchmark", "-G", "-b", "64", "-i", "513"},
                config),
            EXIT_FAILURE);
  EXPECT_EQ(parse_gpu_arguments(
                {"memory_benchmark", "-G", "-b", "64", "-i", "512",
                 "--seed", "1"},
                config),
            EXIT_SUCCESS);
  EXPECT_EQ(config.iterations, 512U);
}

TEST(GpuMemoryBudgetTest, DetectsOverflowAndExceededBudget) {
  const GpuMemoryBudget multiply_overflow = calculate_gpu_memory_budget(
      std::numeric_limits<size_t>::max() / 2U + 1U,
      std::numeric_limits<size_t>::max(), 0);
  EXPECT_FALSE(multiply_overflow.valid);
  EXPECT_EQ(multiply_overflow.reason_code, "memory-requirement-overflow");

  const GpuMemoryBudget add_overflow = calculate_gpu_memory_budget(
      std::numeric_limits<size_t>::max() / 2U,
      std::numeric_limits<size_t>::max(), 2);
  EXPECT_FALSE(add_overflow.valid);
  EXPECT_EQ(add_overflow.reason_code, "memory-requirement-overflow");

  const GpuMemoryBudget exceeded = calculate_gpu_memory_budget(
      512U * Constants::BYTES_PER_MB, 1024U * Constants::BYTES_PER_MB,
      Constants::GPU_AUXILIARY_BUFFER_BYTES);
  EXPECT_FALSE(exceeded.valid);
  EXPECT_EQ(exceeded.reason_code, "memory-budget-exceeded");
  EXPECT_EQ(exceeded.memory_budget_bytes,
            static_cast<size_t>(1024U * Constants::BYTES_PER_MB *
                                Constants::MEMORY_LIMIT_FACTOR));
}

TEST(GpuMemoryBudgetTest, ZeroAvailableMemoryUsesReportedFallback) {
  const GpuMemoryBudget budget = calculate_gpu_memory_budget(
      64U * Constants::BYTES_PER_MB, 0,
      Constants::GPU_AUXILIARY_BUFFER_BYTES);
  ASSERT_TRUE(budget.valid) << budget.reason_code;
  EXPECT_TRUE(budget.used_fallback);
  EXPECT_EQ(budget.available_memory_bytes, 0U);
  EXPECT_EQ(budget.required_total_bytes,
            128U * Constants::BYTES_PER_MB +
                Constants::GPU_AUXILIARY_BUFFER_BYTES);
  EXPECT_EQ(budget.memory_budget_bytes,
            Constants::FALLBACK_TOTAL_LIMIT_MB * Constants::BYTES_PER_MB);
}

TEST(GpuRunnerTest, UnsupportedPreRunCheckpointKeepsNotRunOperationIdentity) {
  FakeGpuBackend backend;
  backend.initialization.status = GpuBackendStatus::Unsupported;
  backend.initialization.reason_code =
      "required-gpu-family-unsupported";
  const GpuBandwidthConfig config = explicit_config();
  GpuRunResult result;
  Json checkpoint;
  size_t checkpoint_count = 0;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    ++checkpoint_count;
    checkpoint = build_gpu_bandwidth_json(config, snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Unsupported);
  EXPECT_EQ(result.reason_code, "required-gpu-family-unsupported");
  EXPECT_EQ(result.counters.terminal_measurements, 0U);
  EXPECT_EQ(checkpoint_count, 1U);
  EXPECT_EQ(checkpoint["status"], "unsupported");
  expect_unresolved_json_operation_identity(checkpoint);
  EXPECT_EQ(backend.lifecycle_log,
            (std::vector<std::string>{"initialize"}));
}

TEST(GpuRunnerTest,
     MaximumBufferPreRunCheckpointKeepsNotRunOperationIdentity) {
  FakeGpuBackend backend;
  backend.initialization.device.max_buffer_length =
      32U * Constants::BYTES_PER_MB;
  const GpuBandwidthConfig config = explicit_config();
  GpuRunResult result;
  Json checkpoint;
  size_t checkpoint_count = 0;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    ++checkpoint_count;
    checkpoint = build_gpu_bandwidth_json(config, snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "max-buffer-length-exceeded");
  EXPECT_EQ(result.counters.terminal_measurements, 0U);
  EXPECT_EQ(checkpoint_count, 1U);
  EXPECT_EQ(checkpoint["status"], "failed");
  expect_unresolved_json_operation_identity(checkpoint);
  EXPECT_EQ(backend.lifecycle_log,
            (std::vector<std::string>{"initialize"}));
}

TEST(GpuRunnerTest,
     AllocationPreRunCheckpointKeepsNotRunOperationIdentity) {
  FakeGpuBackend backend;
  backend.allocation.status = GpuBackendStatus::Failed;
  backend.allocation.reason_code = "private-buffer-allocation-failed";
  const GpuBandwidthConfig config = explicit_config();
  GpuRunResult result;
  Json checkpoint;
  size_t checkpoint_count = 0;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    ++checkpoint_count;
    checkpoint = build_gpu_bandwidth_json(config, snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "private-buffer-allocation-failed");
  EXPECT_EQ(result.counters.terminal_measurements, 0U);
  EXPECT_EQ(checkpoint_count, 1U);
  EXPECT_EQ(checkpoint["status"], "failed");
  expect_unresolved_json_operation_identity(checkpoint);
  EXPECT_EQ(backend.lifecycle_log,
            (std::vector<std::string>{"initialize", "allocate"}));
}

TEST(GpuRunnerTest, ExplicitThreeLoopRunHasExactOrderWorkAndCounters) {
  FakeGpuBackend backend;
  const GpuBandwidthConfig config = explicit_config();
  GpuRunResult result;
  std::vector<GpuRunResult> checkpoints;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    checkpoints.push_back(snapshot);
    return EXIT_SUCCESS;
  };

  ASSERT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_SUCCESS);
  EXPECT_EQ(result.status, GpuRunStatus::Complete);
  EXPECT_EQ(result.reason_code, "complete");
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.conclusions_valid);
  EXPECT_TRUE(result.operation_order_balance_complete);
  EXPECT_EQ(result.counters.planned_loops, 3U);
  EXPECT_EQ(result.counters.attempted_loops, 3U);
  EXPECT_EQ(result.counters.completed_loops, 3U);
  EXPECT_EQ(result.counters.planned_measurements, 9U);
  EXPECT_EQ(result.counters.attempted_measurements, 9U);
  EXPECT_EQ(result.counters.terminal_measurements, 9U);
  EXPECT_EQ(result.counters.completed_measurements, 9U);
  EXPECT_EQ(result.counters.validated_measurements, 9U);

  ASSERT_EQ(result.loops.size(), 3U);
  EXPECT_EQ(result.loops[0].planned_order,
            (std::vector<GpuOperation>{GpuOperation::Read,
                                       GpuOperation::Write,
                                       GpuOperation::Copy}));
  EXPECT_EQ(result.loops[1].planned_order,
            (std::vector<GpuOperation>{GpuOperation::Write,
                                       GpuOperation::Copy,
                                       GpuOperation::Read}));
  EXPECT_EQ(result.loops[2].planned_order,
            (std::vector<GpuOperation>{GpuOperation::Copy,
                                       GpuOperation::Read,
                                       GpuOperation::Write}));
  for (const GpuLoopRecord& loop : result.loops) {
    EXPECT_EQ(loop.realized_order, loop.planned_order);
  }

  ASSERT_EQ(result.measurements.size(), 9U);
  for (const GpuMeasurement& measurement : result.measurements) {
    ASSERT_EQ(measurement.status, GpuMeasurementStatus::Measured);
    ASSERT_TRUE(measurement.value_gb_s.has_value());
    EXPECT_EQ(measurement.work_plan.passes, 2U);
    EXPECT_EQ(measurement.work_plan.dispatch_count, 2U);
    EXPECT_EQ(measurement.work_plan.measured_command_buffer_count, 1U);
    EXPECT_EQ(measurement.work_plan.measured_compute_encoder_count, 1U);
    EXPECT_EQ(measurement.timed.command_buffer_count, 1U);
    EXPECT_EQ(measurement.timed.compute_encoder_count, 1U);
    EXPECT_EQ(measurement.timed.dispatch_count, 2U);
    const size_t multiplier = measurement.operation == GpuOperation::Copy
                                  ? 2U
                                  : 1U;
    EXPECT_EQ(measurement.work_plan.exact_payload_bytes,
              config.buffer_size_bytes * 2U * multiplier);
  }

  for (const GpuOperationAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.values_gb_s.size(), 3U);
    EXPECT_EQ(aggregate.status, "complete");
    EXPECT_EQ(aggregate.stability_quality, "stable");
  }
  ASSERT_TRUE(result.aggregates[0].headline_gb_s.has_value());
  ASSERT_TRUE(result.aggregates[2].headline_gb_s.has_value());
  EXPECT_DOUBLE_EQ(*result.aggregates[2].headline_gb_s,
                   *result.aggregates[0].headline_gb_s * 2.0);

  EXPECT_EQ(backend.phase_calls.size(), 9U * 4U);
  EXPECT_EQ(checkpoints.size(), 9U);
  EXPECT_EQ(checkpoints.front().status, GpuRunStatus::Partial);
  EXPECT_EQ(checkpoints.back().status, GpuRunStatus::Complete);
  EXPECT_TRUE(checkpoints.back().results_complete);
  EXPECT_EQ(backend.lifecycle_log,
            (std::vector<std::string>{"initialize", "allocate", "release"}));
}

TEST(GpuRunnerTest, EnvironmentWarningScansEveryMeasurementSnapshot) {
  struct WarningCase {
    size_t snapshot_occurrence;
    bool low_power;
  };
  for (const WarningCase& warning_case :
       {WarningCase{18, false}, WarningCase{19, true}}) {
    FakeGpuBackend backend;
    const GpuEnvironmentSnapshot nominal = backend.environment;
    backend.environment_provider =
        [nominal, warning_case](size_t occurrence) {
          GpuEnvironmentSnapshot snapshot = nominal;
          if (occurrence == warning_case.snapshot_occurrence) {
            if (warning_case.low_power) {
              snapshot.low_power_mode_enabled = true;
            } else {
              snapshot.thermal_state = "serious";
            }
          }
          return snapshot;
        };
    GpuRunResult result;
    GpuRunnerTestHooks hooks;
    hooks.stop_requested = []() { return false; };

    ASSERT_EQ(run_gpu_bandwidth_suite(explicit_config(), backend, result,
                                      hooks),
              EXIT_SUCCESS);
    EXPECT_EQ(result.environment_start.thermal_state, "nominal");
    EXPECT_FALSE(result.environment_start.low_power_mode_enabled);
    EXPECT_EQ(result.environment_end.thermal_state, "nominal");
    EXPECT_FALSE(result.environment_end.low_power_mode_enabled);
    ASSERT_EQ(result.measurements.size(), 9U);
    if (warning_case.low_power) {
      EXPECT_TRUE(
          result.measurements.back().environment_after.low_power_mode_enabled);
    } else {
      EXPECT_EQ(result.measurements.back().environment_before.thermal_state,
                "serious");
    }
    EXPECT_EQ(std::count(result.quality_warnings.begin(),
                         result.quality_warnings.end(),
                         "environment-not-nominal"),
              1);
  }
}

TEST(GpuRunnerTest, StreamingCvThresholdIsInclusiveOnlyAtFivePercent) {
  struct CvCase {
    double symmetric_delta;
    bool noisy;
  };
  for (const CvCase& cv_case :
       {CvCase{5.0, false}, CvCase{5.1, true}}) {
    FakeGpuBackend backend;
    backend.duration_provider =
        [cv_case](const GpuBackendAttemptRequest& request,
                  size_t operation_occurrence) {
          double desired_gb_s = 100.0;
          if (operation_occurrence == 1) {
            desired_gb_s -= cv_case.symmetric_delta;
          } else if (operation_occurrence == 3) {
            desired_gb_s += cv_case.symmetric_delta;
          }
          const size_t payload_multiplier =
              request.operation == GpuOperation::Copy ? 2U : 1U;
          const long double payload_bytes =
              static_cast<long double>(request.buffer_size_bytes) *
              static_cast<long double>(request.passes) *
              static_cast<long double>(payload_multiplier);
          return static_cast<double>(
              payload_bytes /
              (static_cast<long double>(desired_gb_s) * 1.0e9L));
        };
    GpuRunResult result;
    GpuRunnerTestHooks hooks;
    hooks.stop_requested = []() { return false; };

    ASSERT_EQ(run_gpu_bandwidth_suite(explicit_config(), backend, result,
                                      hooks),
              EXIT_SUCCESS);
    for (const GpuOperationAggregate& aggregate : result.aggregates) {
      EXPECT_NEAR(aggregate.statistics.coefficient_of_variation_pct,
                  cv_case.symmetric_delta, 1e-10);
      EXPECT_EQ(aggregate.stability_quality,
                cv_case.noisy ? "noisy" : "stable");
    }
    EXPECT_EQ(result.quality_warnings.size(),
              cv_case.noisy ? kGpuOperationCount : 0U);
  }
}

TEST(GpuRunnerTest, AutomaticPilotAndTrialAreExcludedAndPlanIsFrozen) {
  FakeGpuBackend backend;
  backend.duration_provider = [](const GpuBackendAttemptRequest&,
                                 size_t operation_occurrence) {
    return operation_occurrence == 1 ? 0.010 : 0.150;
  };
  const GpuBandwidthConfig config = automatic_config();
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };

  ASSERT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_SUCCESS);
  ASSERT_EQ(result.status, GpuRunStatus::Complete);
  for (size_t index = 0; index < kGpuOperationCount; ++index) {
    ASSERT_EQ(result.calibration_attempts[index].size(), 2U);
    EXPECT_EQ(result.calibration_attempts[index][0].purpose, "pilot");
    EXPECT_EQ(result.calibration_attempts[index][0].passes, 1U);
    EXPECT_EQ(result.calibration_attempts[index][1].purpose,
              "duration-trial");
    EXPECT_EQ(result.calibration_attempts[index][1].passes, 15U);
    ASSERT_TRUE(result.work_plans[index].valid);
    EXPECT_EQ(result.work_plans[index].passes, 15U);
    EXPECT_EQ(backend.timed_calls_by_operation[index], 5U);
  }
  for (const GpuMeasurement& measurement : result.measurements) {
    EXPECT_EQ(measurement.work_plan.passes, 15U);
    EXPECT_EQ(measurement.work_plan.plan_identity,
              result.work_plans[operation_index(measurement.operation)]
                  .plan_identity);
  }
}

TEST(GpuRunnerTest, AutomaticCalibrationUsesBothCorrectionTrialsAtMost) {
  FakeGpuBackend backend;
  backend.duration_provider = [](const GpuBackendAttemptRequest&,
                                 size_t operation_occurrence) {
    switch (operation_occurrence) {
      case 1:
        return 0.010;
      case 2:
        return 0.050;
      case 3:
      case 4:
        return 0.300;
      default:
        return 0.150;
    }
  };
  const GpuBandwidthConfig config = automatic_config();
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };

  ASSERT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_SUCCESS);
  ASSERT_EQ(result.status, GpuRunStatus::Complete);
  for (GpuOperation operation : {GpuOperation::Read, GpuOperation::Write,
                                 GpuOperation::Copy}) {
    const size_t index = operation_index(operation);
    const GpuPassLimits limits =
        calculate_gpu_pass_limits(config.buffer_size_bytes, operation);
    const size_t pilot = calculate_gpu_pilot_passes(limits);
    const size_t duration_trial =
        calculate_gpu_calibrated_passes(0.010, pilot, limits);
    const size_t correction_one =
        calculate_gpu_calibrated_passes(0.050, duration_trial, limits);
    const size_t correction_two =
        calculate_gpu_calibrated_passes(0.300, correction_one, limits);
    const std::array<size_t, 4> expected_passes = {
        pilot, duration_trial, correction_one, correction_two};
    const std::array<std::string, 4> expected_purposes = {
        "pilot", "duration-trial", "correction-trial-1",
        "correction-trial-2"};

    ASSERT_EQ(result.calibration_attempts[index].size(),
              expected_passes.size());
    for (size_t attempt_index = 0;
         attempt_index < expected_passes.size(); ++attempt_index) {
      EXPECT_EQ(result.calibration_attempts[index][attempt_index].purpose,
                expected_purposes[attempt_index]);
      EXPECT_EQ(result.calibration_attempts[index][attempt_index].passes,
                expected_passes[attempt_index]);
    }
    EXPECT_EQ(result.calibration_attempts[index].back().duration_quality,
              "above-target-window");
    EXPECT_EQ(result.work_plans[index].passes, correction_two);
    EXPECT_EQ(backend.timed_calls_by_operation[index], 7U);
  }
}

TEST(GpuRunnerTest, PayloadCapDurationQualitySurvivesFrozenMeasurements) {
  FakeGpuBackend backend;
  backend.duration_provider = [](const GpuBackendAttemptRequest&,
                                 size_t operation_occurrence) {
    return operation_occurrence == 1 ? 0.0001 : 0.050;
  };
  const GpuBandwidthConfig config = automatic_config();
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };

  ASSERT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_SUCCESS);
  ASSERT_EQ(result.status, GpuRunStatus::Complete);
  for (GpuOperation operation : {GpuOperation::Read, GpuOperation::Write,
                                 GpuOperation::Copy}) {
    const size_t index = operation_index(operation);
    const GpuPassLimits limits =
        calculate_gpu_pass_limits(config.buffer_size_bytes, operation);
    ASSERT_TRUE(limits.payload_cap_is_limiting);
    ASSERT_EQ(result.calibration_attempts[index].size(), 2U);
    EXPECT_EQ(result.calibration_attempts[index].back().passes,
              limits.effective_maximum_passes);
    EXPECT_EQ(result.calibration_attempts[index].back().duration_quality,
              "payload-cap-below-target");
    EXPECT_EQ(result.work_plans[index].passes,
              limits.effective_maximum_passes);
  }
  for (const GpuMeasurement& measurement : result.measurements) {
    EXPECT_EQ(measurement.duration_quality,
              "payload-cap-below-target");
  }
}

TEST(GpuRunnerTest, CalibrationRuntimeFailureFinalizesUnstartedSlotsFailed) {
  FakeGpuBackend backend;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.status = GpuBackendStatus::Failed;
    timed.command_status = GpuCommandStatus::Error;
    timed.reason_code = "calibration-timed-command-failed";
  };
  const GpuBandwidthConfig config = automatic_config();
  GpuRunResult result;
  Json checkpoint;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    checkpoint = build_gpu_bandwidth_json(config, snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "calibration-timed-command-failed");
  EXPECT_EQ(result.counters.attempted_measurements, 0U);
  EXPECT_EQ(result.counters.completed_measurements, 0U);
  EXPECT_EQ(result.counters.terminal_measurements,
            result.counters.planned_measurements);
  expect_unstarted_failed_tail(result, 0);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1U);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  for (const Json& measurement : checkpoint["measurements"]) {
    EXPECT_EQ(measurement["status"], "failed");
    EXPECT_TRUE(measurement["value_gb_s"].is_null());
  }
}

TEST(GpuRunnerTest, ProductionFinalSaveFailureWinsCalibrationFailure) {
  FakeGpuBackend backend;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.status = GpuBackendStatus::Failed;
    timed.command_status = GpuCommandStatus::Error;
    timed.reason_code = "calibration-timed-command-failed";
  };
  GpuBandwidthConfig config = automatic_config();
  config.output_file =
      "/tmp/" + std::string(5000, 'x') + "/gpu.json";
  GpuRunResult result;

  testing::internal::CaptureStderr();
  const int status = run_gpu_bandwidth_suite(config, backend, result);
  const std::string error = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, EXIT_FAILURE);
  EXPECT_NE(error.find("Failed to write file"), std::string::npos);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
}

TEST(GpuRunnerTest, ProductionFinalSaveFailureWinsRunnerException) {
  FakeGpuBackend backend;
  GpuBandwidthConfig config = explicit_config(1);
  config.output_file =
      "/tmp/" + std::string(5000, 'x') + "/gpu.json";
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() -> bool {
    throw std::runtime_error("injected-stop-failure");
  };

  testing::internal::CaptureStderr();
  const int status =
      run_gpu_bandwidth_suite(config, backend, result, hooks);
  const std::string error = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, EXIT_FAILURE);
  EXPECT_NE(error.find("Failed to write file"), std::string::npos);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
}

TEST(GpuRunnerTest, CalibrationFailureWithPendingStopKeepsInterruptedTail) {
  FakeGpuBackend backend;
  bool stop = false;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.status = GpuBackendStatus::Failed;
    timed.command_status = GpuCommandStatus::Error;
    timed.reason_code = "calibration-timed-command-failed";
  };
  backend.phase_hook = [&](const std::string& phase, size_t occurrence,
                           const GpuBackendAttemptRequest&) {
    if (phase == "timed" && occurrence == 1) {
      stop = true;
    }
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [](const GpuRunResult&) { return EXIT_SUCCESS; };

  EXPECT_EQ(run_gpu_bandwidth_suite(automatic_config(2), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "calibration-timed-command-failed");
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.counters.attempted_measurements, 0U);
  EXPECT_EQ(result.counters.completed_measurements, 0U);
  EXPECT_EQ(result.counters.terminal_measurements,
            result.counters.planned_measurements);
  expect_unstarted_interrupted_tail(result, 0);
}

TEST(GpuRunnerTest, InvalidTimerStopsBeforeValidationAndProducesNullValue) {
  FakeGpuBackend backend;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.gpu_elapsed_seconds = 0.0;
    timed.gpu_end_seconds = timed.gpu_start_seconds;
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                    hooks),
            EXIT_FAILURE);
  ASSERT_EQ(result.status, GpuRunStatus::Failed);
  ASSERT_EQ(result.measurements[0].status, GpuMeasurementStatus::Invalid);
  EXPECT_EQ(result.measurements[0].reason_code, "invalid-gpu-timestamp");
  EXPECT_FALSE(result.measurements[0].value_gb_s.has_value());
  EXPECT_EQ(result.measurements[0].validation.validation_status,
            GpuValidationStatus::NotRunTimerInvalid);
  EXPECT_EQ(backend.validation_calls, 0U);
  EXPECT_EQ(result.counters.completed_measurements, 1U);
  EXPECT_EQ(result.counters.validated_measurements, 0U);
  EXPECT_EQ(result.counters.attempted_measurements, 1U);
  EXPECT_EQ(result.counters.terminal_measurements,
            result.counters.planned_measurements);
  expect_unstarted_failed_tail(result, 1);
}

TEST(GpuRunnerTest, TimedAccumulatorMismatchIsInvalidAfterValidation) {
  FakeGpuBackend backend;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    ++timed.actual_accumulator.first;
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.measurements[0].status, GpuMeasurementStatus::Invalid);
  EXPECT_EQ(result.measurements[0].reason_code,
            "timed-accumulator-mismatch");
  EXPECT_EQ(backend.validation_calls, 1U);
  EXPECT_FALSE(result.measurements[0].value_gb_s.has_value());
}

TEST(GpuRunnerTest, ValidationMismatchAndErrorRemainDistinct) {
  {
    FakeGpuBackend backend;
    backend.validation_mutator =
        [](const GpuBackendAttemptRequest&, size_t,
           GpuValidationResult& validation) {
          validation.validation_status = GpuValidationStatus::Mismatch;
          validation.reason_code = "final-checksum-mismatch";
          ++validation.actual_final_checksum.second;
        };
    GpuRunResult result;
    GpuRunnerTestHooks hooks;
    hooks.stop_requested = []() { return false; };
    EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                      hooks),
              EXIT_FAILURE);
    EXPECT_EQ(result.measurements[0].status, GpuMeasurementStatus::Invalid);
    EXPECT_EQ(result.measurements[0].reason_code,
              "final-checksum-mismatch");
  }
  {
    FakeGpuBackend backend;
    backend.validation_mutator =
        [](const GpuBackendAttemptRequest&, size_t,
           GpuValidationResult& validation) {
          validation.status = GpuBackendStatus::Failed;
          validation.command_status = GpuCommandStatus::Error;
          validation.validation_status = GpuValidationStatus::Error;
          validation.reason_code = "validation-command-failed";
          validation.error = {"MTLCommandBufferErrorDomain", 4,
                              "localized GPU fault"};
        };
    GpuRunResult result;
    GpuRunnerTestHooks hooks;
    hooks.stop_requested = []() { return false; };
    EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                      hooks),
              EXIT_FAILURE);
    EXPECT_EQ(result.measurements[0].status, GpuMeasurementStatus::Failed);
    EXPECT_EQ(result.measurements[0].reason_code,
              "validation-command-failed");
    EXPECT_EQ(result.measurements[0].validation.error.description,
              "localized GPU fault");
  }
}

TEST(GpuRunnerTest, StopBeforeTaskStartsNoBackendPhaseAndInterruptsAllSlots) {
  FakeGpuBackend backend;
  GpuRunResult result;
  std::vector<GpuRunResult> checkpoints;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return true; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    checkpoints.push_back(snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_SUCCESS);
  EXPECT_EQ(result.status, GpuRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_TRUE(backend.phase_calls.empty());
  EXPECT_EQ(result.counters.attempted_measurements, 0U);
  EXPECT_EQ(result.counters.terminal_measurements, 6U);
  expect_interrupted_tail(result, 0);
  ASSERT_EQ(checkpoints.size(), 1U);
  EXPECT_EQ(checkpoints[0].status, GpuRunStatus::Interrupted);
}

TEST(GpuRunnerTest, StopInsideStartedPhaseUsesCompletionWins) {
  for (const std::string& stop_phase : {"warmup", "timed", "validation"}) {
    FakeGpuBackend backend;
    bool stop = false;
    backend.phase_hook = [&](const std::string& phase, size_t occurrence,
                             const GpuBackendAttemptRequest&) {
      EXPECT_TRUE(backend.resources_allocated) << phase;
      if (phase == stop_phase && occurrence == 1) {
        stop = true;
      }
    };
    GpuRunResult result;
    std::vector<GpuRunResult> checkpoints;
    GpuRunnerTestHooks hooks;
    hooks.stop_requested = [&]() { return stop; };
    hooks.checkpoint = [&](const GpuRunResult& snapshot) {
      checkpoints.push_back(snapshot);
      return EXIT_SUCCESS;
    };

    EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                      hooks),
              EXIT_SUCCESS)
        << stop_phase;
    EXPECT_EQ(result.status, GpuRunStatus::Interrupted) << stop_phase;
    EXPECT_TRUE(result.interruption_requested) << stop_phase;
    EXPECT_EQ(backend.phase_calls.size(), 4U) << stop_phase;
    EXPECT_EQ(backend.validation_calls, 1U) << stop_phase;
    EXPECT_FALSE(backend.resources_allocated) << stop_phase;
    expect_interrupted_tail(result, 1);
    EXPECT_EQ(result.counters.attempted_measurements, 1U) << stop_phase;
    EXPECT_EQ(result.counters.completed_measurements, 1U) << stop_phase;
    EXPECT_EQ(result.counters.validated_measurements, 1U) << stop_phase;
    ASSERT_EQ(checkpoints.size(), 1U) << stop_phase;
    EXPECT_EQ(checkpoints[0].measurements[0].status,
              GpuMeasurementStatus::Measured)
        << stop_phase;
  }
}

TEST(GpuRunnerTest, FailureWinsOverStopAndPreventsValidationAfterTimedError) {
  FakeGpuBackend backend;
  bool stop = false;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.status = GpuBackendStatus::Failed;
    timed.command_status = GpuCommandStatus::Error;
    timed.reason_code = "timed-command-failed";
  };
  backend.phase_hook = [&](const std::string& phase, size_t occurrence,
                           const GpuBackendAttemptRequest&) {
    if (phase == "timed" && occurrence == 1) {
      stop = true;
    }
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "timed-command-failed");
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.measurements[0].status, GpuMeasurementStatus::Failed);
  EXPECT_EQ(backend.validation_calls, 0U);
  expect_unstarted_interrupted_tail(result, 1);
}

TEST(GpuRunnerTest, ValidationMismatchWinsOverSimultaneousStop) {
  FakeGpuBackend backend;
  bool stop = false;
  backend.validation_mutator =
      [](const GpuBackendAttemptRequest&, size_t,
         GpuValidationResult& validation) {
        validation.validation_status = GpuValidationStatus::Mismatch;
        validation.reason_code = "final-checksum-mismatch";
        ++validation.actual_final_checksum.first;
      };
  backend.phase_hook = [&](const std::string& phase, size_t occurrence,
                           const GpuBackendAttemptRequest&) {
    if (phase == "validation" && occurrence == 1) {
      stop = true;
    }
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "final-checksum-mismatch");
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.measurements[0].status,
            GpuMeasurementStatus::Invalid);
  EXPECT_EQ(result.measurements[0].reason_code,
            "final-checksum-mismatch");
  EXPECT_FALSE(result.measurements[0].value_gb_s.has_value());
  EXPECT_EQ(backend.timed_calls, 1U);
  EXPECT_EQ(backend.validation_calls, 1U);
  EXPECT_EQ(result.counters.attempted_measurements, 1U);
  EXPECT_EQ(result.counters.completed_measurements, 1U);
  expect_unstarted_interrupted_tail(result, 1);
}

TEST(GpuRunnerTest, StopInLastTaskKeepsAllMeasurementsButInvalidatesRun) {
  FakeGpuBackend backend;
  bool stop = false;
  backend.phase_hook = [&](const std::string& phase, size_t occurrence,
                           const GpuBackendAttemptRequest&) {
    if (phase == "validation" && occurrence == 9) {
      stop = true;
    }
  };
  GpuRunResult result;
  std::vector<GpuRunResult> checkpoints;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    checkpoints.push_back(snapshot);
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(), backend, result,
                                    hooks),
            EXIT_SUCCESS);
  EXPECT_EQ(result.status, GpuRunStatus::Interrupted);
  EXPECT_EQ(result.counters.completed_measurements, 9U);
  EXPECT_EQ(result.counters.validated_measurements, 9U);
  EXPECT_EQ(result.counters.completed_loops, 3U);
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_TRUE(result.operation_order_balance_complete);
  for (const GpuMeasurement& measurement : result.measurements) {
    EXPECT_EQ(measurement.status, GpuMeasurementStatus::Measured);
    EXPECT_TRUE(measurement.value_gb_s.has_value());
  }
  ASSERT_EQ(checkpoints.size(), 9U);
  EXPECT_EQ(checkpoints.back().status, GpuRunStatus::Interrupted);
}

TEST(GpuRunnerTest, StopFirstSeenInCheckpointWritesAtMostOneExtraSnapshot) {
  FakeGpuBackend backend;
  bool stop = false;
  GpuRunResult result;
  std::vector<GpuRunResult> checkpoints;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const GpuRunResult& snapshot) {
    checkpoints.push_back(snapshot);
    if (checkpoints.size() == 1) {
      stop = true;
    }
    return EXIT_SUCCESS;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_SUCCESS);
  ASSERT_EQ(checkpoints.size(), 2U);
  EXPECT_EQ(checkpoints[0].status, GpuRunStatus::Partial);
  EXPECT_EQ(checkpoints[0].measurements[0].status,
            GpuMeasurementStatus::Measured);
  EXPECT_EQ(checkpoints[0].measurements[1].status,
            GpuMeasurementStatus::NotRun);
  EXPECT_EQ(checkpoints[1].status, GpuRunStatus::Interrupted);
  expect_interrupted_tail(checkpoints[1], 1);
  EXPECT_EQ(backend.timed_calls, 1U);
  EXPECT_EQ(result.status, GpuRunStatus::Interrupted);
}

TEST(GpuRunnerTest, CheckpointFailureWinsAndNeverStartsNextTask) {
  FakeGpuBackend backend;
  bool stop = false;
  GpuRunResult result;
  size_t checkpoint_calls = 0;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const GpuRunResult&) {
    ++checkpoint_calls;
    stop = true;
    return EXIT_FAILURE;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(checkpoint_calls, 1U);
  EXPECT_EQ(backend.timed_calls, 1U);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_EQ(result.counters.attempted_measurements, 1U);
  EXPECT_EQ(result.counters.completed_measurements, 1U);
  EXPECT_EQ(result.counters.terminal_measurements,
            result.counters.planned_measurements);
  expect_unstarted_interrupted_tail(result, 1);
}

TEST(GpuRunnerTest, CheckpointFailureWithoutStopFinalizesFailedTail) {
  FakeGpuBackend backend;
  GpuRunResult result;
  size_t checkpoint_calls = 0;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const GpuRunResult&) {
    ++checkpoint_calls;
    return EXIT_FAILURE;
  };

  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(2), backend, result,
                                    hooks),
            EXIT_FAILURE);
  EXPECT_EQ(checkpoint_calls, 1U);
  EXPECT_EQ(backend.timed_calls, 1U);
  EXPECT_EQ(result.status, GpuRunStatus::Failed);
  EXPECT_EQ(result.reason_code, "checkpoint-write-failed");
  EXPECT_FALSE(result.interruption_requested);
  EXPECT_EQ(result.counters.attempted_measurements, 1U);
  EXPECT_EQ(result.counters.completed_measurements, 1U);
  EXPECT_EQ(result.counters.terminal_measurements,
            result.counters.planned_measurements);
  expect_unstarted_failed_tail(result, 1);
}

TEST(GpuJsonTest, SchemaV1UsesExactStringsAndMeasuredOnlyValues) {
  FakeGpuBackend backend;
  GpuBandwidthConfig config = explicit_config();
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  ASSERT_EQ(run_gpu_bandwidth_suite(config, backend, result, hooks),
            EXIT_SUCCESS);
  const Json output = build_gpu_bandwidth_json(config, result);

  EXPECT_EQ(output["schema_version"], 1);
  EXPECT_EQ(output["mode"], "gpu_bandwidth");
  EXPECT_EQ(output["methodology_version"],
            Constants::GPU_METHODOLOGY_VERSION);
  EXPECT_EQ(output["methodology"]["timed_observable_reduction"],
            "dual-mod32-threadgroup-reduction-v2");
  EXPECT_EQ(output["status"], "complete");
  EXPECT_TRUE(output["results_complete"].get<bool>());
  EXPECT_EQ(output["configuration"]["base_seed_uint64_decimal"],
            "18446744073709551615");
  EXPECT_EQ(output["backend"]["device"]["registry_id_uint64_decimal"],
            "18446744073709551615");
  EXPECT_TRUE(output["backend"]["allocation"]
                    ["recommended_working_set_available"]
                        .get<bool>());
  EXPECT_TRUE(output["backend"]["allocation"]
                    ["recommended_working_set_headroom_bytes"]
                        .is_string());
  EXPECT_TRUE(output["backend"]["allocation"]
                    ["recommended_working_set_headroom_fraction"]
                        .is_number());
  ASSERT_EQ(output["work_plans"].size(), 3U);
  const std::array<std::string, kGpuOperationCount> operations = {
      "read", "write", "copy"};
  for (size_t index = 0; index < operations.size(); ++index) {
    const Json& plan = output["work_plans"][index];
    EXPECT_EQ(plan["operation"], operations[index]);
    EXPECT_TRUE(plan["exact_payload_bytes"].is_string());
    EXPECT_TRUE(plan["operation_seed_uint64_decimal"].is_string());
    EXPECT_EQ(plan["resource_options"]["data_uint64_decimal"],
              "18446744073709551615");
    EXPECT_EQ(plan["resource_options"]["status_uint64_decimal"],
              "18446744073709551615");
    EXPECT_EQ(plan["resource_options"]["data_storage_mode"], "private");
    EXPECT_EQ(plan["resource_options"]["data_hazard_tracking_mode"],
              "tracked");
    EXPECT_EQ(plan["resource_options"]["status_storage_mode"], "shared");
    EXPECT_EQ(plan["resource_options"]["status_hazard_tracking_mode"],
              "tracked");
    EXPECT_EQ(plan["kernel_revision"], "fake-kernel-v1");
    EXPECT_EQ(plan["kernel_source_sha256"],
              "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    EXPECT_EQ(plan["msl_language_version"], "2.3");
    EXPECT_FALSE(plan["plan_identity"].get<std::string>().empty());
  }
  for (size_t index = 0; index < output["measurements"].size(); ++index) {
    const Json& measurement = output["measurements"][index];
    const size_t plan_index =
        operation_index(result.measurements[index].operation);
    EXPECT_EQ(measurement["work_plan"]["operation"],
              measurement["operation"]);
    EXPECT_EQ(measurement["work_plan"]["plan_identity"],
              output["work_plans"][plan_index]["plan_identity"]);
    EXPECT_FALSE(measurement["validation"].contains("algorithm"));
    EXPECT_EQ(measurement["validation"]
                         ["timed_accumulator_algorithm"],
              "gpu-dual-mod32-v2");
    EXPECT_EQ(measurement["validation"]
                         ["final_checksum_algorithm"],
              measurement["operation"] == "read"
                  ? "not-applicable"
                  : "gpu-dual-mod32-v1");
    EXPECT_EQ(measurement["validation"]["status_reset_count"],
              measurement["operation"] == "read" ? 0U : 1U);
  }
  EXPECT_TRUE(output["measurements"][0]["value_gb_s"].is_number());
  EXPECT_EQ(output["measurements"][0]["timed"]["command_buffer_count"],
            1U);
  EXPECT_EQ(output["measurements"][0]["timed"]["compute_encoder_count"],
            1U);
  EXPECT_EQ(output["measurements"][0]["warmup"]
                  ["data_initialization_dispatch_count"],
            1U);
  EXPECT_EQ(output["measurements"][0]["warmup"]
                  ["benchmark_operation_dispatch_count"],
            1U);
  EXPECT_EQ(output["measurements"][0]["precondition"]
                  ["status_reset_count"],
            1U);
  EXPECT_TRUE(output["measurements"][0]["timed"]
                  ["expected_timed_accumulator"]["first_uint32_decimal"]
                      .is_string());
  EXPECT_EQ(output["aggregates"]["copy"]["sample_count"], 3U);
  EXPECT_EQ(output["copy_payload_semantics"], "aggregate-read-plus-write");
  EXPECT_EQ(output["dram_residency"], "unverified");
}

TEST(GpuJsonTest, MissingAndNonFiniteValuesAreNull) {
  FakeGpuBackend backend;
  backend.timed_mutator = [](const GpuBackendAttemptRequest&, size_t,
                             GpuTimedResult& timed) {
    timed.gpu_elapsed_seconds =
        std::numeric_limits<double>::quiet_NaN();
  };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                    hooks),
            EXIT_FAILURE);
  const Json output = build_gpu_bandwidth_json(explicit_config(1), result);
  EXPECT_EQ(output["measurements"][0]["status"], "invalid");
  EXPECT_TRUE(output["measurements"][0]["value_gb_s"].is_null());
  EXPECT_TRUE(output["measurements"][0]["timed"]["gpu_elapsed_seconds"]
                  .is_null());
  for (size_t index = 1; index < output["measurements"].size(); ++index) {
    EXPECT_EQ(output["measurements"][index]["status"], "failed");
    EXPECT_EQ(output["measurements"][index]["reason_code"],
              "not-run-after-runtime-failure");
    EXPECT_TRUE(output["measurements"][index]["value_gb_s"].is_null());
    EXPECT_EQ(output["measurements"][index]["warmup"]["status"],
              "not-run");
    EXPECT_EQ(output["measurements"][index]["warmup"]
                    ["command_buffer_status"],
              "not-run");
  }
  EXPECT_TRUE(output["aggregates"]["read"]["headline_gb_s"].is_null());
  EXPECT_TRUE(output["aggregates"]["read"]["statistics"].is_null());
}

TEST(GpuJsonTest, StableReasonCodeAndRawNSErrorStaySeparate) {
  FakeGpuBackend backend;
  backend.validation_mutator =
      [](const GpuBackendAttemptRequest&, size_t,
         GpuValidationResult& validation) {
        validation.status = GpuBackendStatus::Failed;
        validation.command_status = GpuCommandStatus::Error;
        validation.validation_status = GpuValidationStatus::Error;
        validation.reason_code = "validation-command-failed";
        validation.error = {"MTLCommandBufferErrorDomain", 9,
                            "localized and unstable description"};
      };
  GpuRunResult result;
  GpuRunnerTestHooks hooks;
  hooks.stop_requested = []() { return false; };
  EXPECT_EQ(run_gpu_bandwidth_suite(explicit_config(1), backend, result,
                                    hooks),
            EXIT_FAILURE);
  const Json output = build_gpu_bandwidth_json(explicit_config(1), result);
  const Json& measurement = output["measurements"][0];
  EXPECT_EQ(output["reason_code"], "validation-command-failed");
  EXPECT_EQ(measurement["reason_code"], "validation-command-failed");
  EXPECT_EQ(measurement["validation"]["error"]["domain"],
            "MTLCommandBufferErrorDomain");
  EXPECT_EQ(measurement["validation"]["error"]["code"], 9);
  EXPECT_EQ(measurement["validation"]["error"]["description"],
            "localized and unstable description");
  EXPECT_NE(measurement["reason_code"],
            measurement["validation"]["error"]["description"]);
}

}  // namespace
