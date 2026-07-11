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
//

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include <thread>
#include <unistd.h>

#include "asm/asm_functions.h"
#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_tests.h"
#include "benchmark/benchmark_work_plan.h"
#include "benchmark/parallel_test_framework.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/memory/memory_utils.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "test_config_helpers.h"
#include "test_timer_system_calls.h"

namespace {

uint64_t deterministic_timer_tick = 0;

uint64_t deterministic_timer_ticks() {
  deterministic_timer_tick += 100;
  return deterministic_timer_tick;
}

void reset_deterministic_timer_ticks() { deterministic_timer_tick = 0; }

using ScopedDeterministicTimerSystemCalls =
    test_timer_system_calls::ScopedTimerSystemCalls<
        deterministic_timer_ticks, reset_deterministic_timer_ticks>;

BenchmarkConfig build_base_config() {
  BenchmarkConfig config;
  initialize_system_info(config);

  config.buffer_size = static_cast<size_t>(getpagesize());
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 1;
  config.use_custom_cache_size = false;

  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  return config;
}

BenchmarkConfig build_injected_failure_config() {
  BenchmarkConfig config;
  config.buffer_size = 16 * Constants::BYTES_PER_KB;
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 1;
  config.num_threads = 1;
  config.l1_buffer_size = 16 * Constants::BYTES_PER_KB;
  config.l2_buffer_size = 32 * Constants::BYTES_PER_KB;
  config.lat_num_accesses = 64;
  config.l1_num_accesses = 64;
  config.l2_num_accesses = 64;
  config.benchmark_seed = 1;
  return config;
}

struct DelayedThreadExit {
  ~DelayedThreadExit() {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
};

}  // namespace

TEST(BenchmarkExecutorTest, WriteTestCoversEntireBufferForUnevenThreadSplitsIntegration) {
  const size_t buffer_size = (1024 * 1024) + 123;
  std::vector<unsigned char> buffer(buffer_size, 0xAB);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  double duration = run_write_test(buffer.data(), buffer.size(), 1, 10, *timer_opt);
  EXPECT_GT(duration, 0.0);

  bool all_zero = std::all_of(buffer.begin(), buffer.end(), [](unsigned char value) {
    return value == 0;
  });
  EXPECT_TRUE(all_zero);
}

TEST(BenchmarkExecutorTest, WriteTestReturnsZeroForInvalidThreadCountsIntegration) {
  std::vector<unsigned char> buffer(4096, 0xCD);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  double zero_thread_duration = run_write_test(buffer.data(), buffer.size(), 1, 0, *timer_opt);
  double negative_thread_duration = run_write_test(buffer.data(), buffer.size(), 1, -1, *timer_opt);

  EXPECT_EQ(zero_thread_duration, 0.0);
  EXPECT_EQ(negative_thread_duration, 0.0);

  bool unchanged = std::all_of(buffer.begin(), buffer.end(), [](unsigned char value) {
    return value == 0xCD;
  });
  EXPECT_TRUE(unchanged);
}

TEST(BenchmarkExecutorTest, ParallelTimingStopsBeforeWorkerTeardownIntegration) {
  std::array<unsigned char, 4096> buffer{};
  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  const auto wall_start = std::chrono::steady_clock::now();
  const double measured_duration = run_parallel_test(
      buffer.data(), buffer.size(), 1, 1, *timer_opt,
      [](char* /* chunk_start */, size_t /* chunk_size */, int /* iterations */) {
        thread_local DelayedThreadExit delayed_exit;
        static_cast<void>(delayed_exit);
      },
      "timing_test");
  const double wall_duration =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

  EXPECT_GE(wall_duration, 0.035);
  EXPECT_GT(measured_duration, 0.0);
  EXPECT_LT(measured_duration, wall_duration - 0.020);
}

TEST(BenchmarkExecutorTest, ReadLoopChecksumFoldsUpperVectorLaneIntegration) {
  alignas(64) std::array<std::uint8_t, 32> buffer{};
  buffer[8] = 0x5A;

  EXPECT_EQ(memory_read_loop_asm(buffer.data(), buffer.size()), 0x5AULL);
}

TEST(BenchmarkExecutorTest, LatencySamplingClampsToAccessCountIntegration) {
  const size_t buffer_size = static_cast<size_t>(getpagesize());
  std::vector<unsigned char> buffer(buffer_size, 0);

  ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, Constants::LATENCY_STRIDE_BYTES), EXIT_SUCCESS);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  const size_t num_accesses = 7;
  const int requested_samples = 100;
  std::vector<double> samples;

  double total_duration_ns = run_latency_test(buffer.data(), num_accesses, *timer_opt, &samples, requested_samples);
  EXPECT_GE(total_duration_ns, 0.0);
  EXPECT_FALSE(std::isnan(total_duration_ns));
  EXPECT_FALSE(std::isinf(total_duration_ns));
  EXPECT_EQ(samples.size(), num_accesses);

  for (double sample : samples) {
    EXPECT_GE(sample, 0.0);
    EXPECT_FALSE(std::isnan(sample));
    EXPECT_FALSE(std::isinf(sample));
  }
}

TEST(BenchmarkExecutorTest, LatencySamplingContinuesFromPriorTerminalPointer) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  std::array<uintptr_t, 4> nodes{};
  std::vector<uintptr_t*> observed_starts;
  std::vector<size_t> observed_access_counts;
  size_t chase_calls = 0;
  LatencyMeasurementTestHooks hooks;
  hooks.chase = [&](uintptr_t* start, size_t access_count) {
    observed_starts.push_back(start);
    observed_access_counts.push_back(access_count);
    ++chase_calls;
    return &nodes[chase_calls];
  };

  std::vector<double> samples;
  const double total_duration_ns =
      run_latency_test(&nodes[0], 7, *timer, &samples, 3, &hooks);

  EXPECT_DOUBLE_EQ(total_duration_ns, 300.0);
  EXPECT_EQ(samples, (std::vector<double>{100.0 / 3.0, 50.0, 50.0}));
  EXPECT_EQ(observed_starts, (std::vector<uintptr_t*>{&nodes[0], &nodes[1], &nodes[2]}));
  EXPECT_EQ(observed_access_counts, (std::vector<size_t>{3, 2, 2}));
}

TEST(BenchmarkExecutorTest, ActiveLatencyPathReportsAndReusesAuditableWorkIntegration) {
  BenchmarkConfig config = build_base_config();
  config.only_latency = true;
  config.only_bandwidth = false;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = 0;
  config.custom_buffer_size = 0;
  config.user_specified_latency_tlb_locality = true;
  config.latency_sample_count = 0;

  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  BenchmarkExecutionState execution_state;

  const BenchmarkResults first = run_single_benchmark_loop(
      config, 0, *timer, &execution_state);
  ASSERT_EQ(first.status, BenchmarkRunStatus::Complete);
  EXPECT_EQ(first.planned_phases, 1u);
  EXPECT_EQ(first.completed_phases, 1u);
  EXPECT_EQ(first.planned_measurements, 1u);
  EXPECT_EQ(first.completed_measurements, 1u);
  EXPECT_EQ(first.planned_phase_order,
            (std::vector<std::string>{"main-latency"}));
  EXPECT_EQ(first.realized_phase_order, first.planned_phase_order);

  const BenchmarkMeasurement& first_latency = first.main_latency;
  ASSERT_TRUE(first_latency.is_measured());
  EXPECT_EQ(first_latency.target, "main-memory");
  EXPECT_EQ(first_latency.operation, "latency");
  EXPECT_EQ(first_latency.work_policy, "automatic-duration-calibration");
  EXPECT_TRUE(first_latency.automatic_calibration);
  EXPECT_GT(first_latency.pilot_elapsed_seconds, 0.0);
  EXPECT_GT(first_latency.access_count, 0u);
  EXPECT_GT(first_latency.chain_node_count, 1u);
  EXPECT_GE(first_latency.complete_chain_cycles,
            Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES);
  EXPECT_EQ(first_latency.requested_threads, 1);
  EXPECT_EQ(first_latency.effective_threads, 1);
  EXPECT_EQ(first_latency.created_workers, 1);
  EXPECT_EQ(first.locality_16k_latency.status,
            BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(first.global_random_latency.status,
            BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(first.locality_latency_delta.status,
            BenchmarkMeasurementStatus::NotRun);

  ASSERT_TRUE(execution_state.latency[
                  benchmark_latency_state_index(BenchmarkTarget::MainMemory)]
                  .initialized);
  const size_t resolved_access_count = first_latency.access_count;
  const double pilot_elapsed_seconds = first_latency.pilot_elapsed_seconds;

  const BenchmarkResults second = run_single_benchmark_loop(
      config, 1, *timer, &execution_state);
  ASSERT_EQ(second.status, BenchmarkRunStatus::Complete);
  ASSERT_TRUE(second.main_latency.is_measured());
  EXPECT_EQ(second.loop_index, 1u);
  EXPECT_EQ(second.main_latency.access_count, resolved_access_count);
  EXPECT_DOUBLE_EQ(second.main_latency.pilot_elapsed_seconds,
                   pilot_elapsed_seconds);
  EXPECT_EQ(second.main_latency.seed, first_latency.seed);
}

TEST(BenchmarkExecutorTest, InjectedPreparationFailureCoversEveryPhaseBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config = build_injected_failure_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  const std::array<const char*, 4> phases = {
      "main-bandwidth", "cache-bandwidth", "cache-latency", "main-latency"};

  for (size_t phase_index = 0; phase_index < phases.size(); ++phase_index) {
    const std::string failing_phase = phases[phase_index];
    BenchmarkExecutorTestHooks hooks;
    hooks.fail_phase_preparation = [&](const std::string& phase_name) {
      return phase_name == failing_phase;
    };
    const std::string expected_reason =
        Messages::benchmark_reason_prepare_failed(failing_phase);
    std::string caught_reason;
    testing::internal::CaptureStderr();
    try {
      static_cast<void>(run_single_benchmark_loop(
          config, static_cast<int>(phase_index), *timer, nullptr, &hooks));
    } catch (const std::runtime_error& error) {
      caught_reason = error.what();
    }
    const std::string error_output =
        testing::internal::GetCapturedStderr();
    EXPECT_EQ(caught_reason, expected_reason) << failing_phase;
    EXPECT_NE(error_output.find(expected_reason), std::string::npos)
        << failing_phase;
  }
}

TEST(BenchmarkExecutorTest, InjectedLatencyChainFailureCoversCacheAndMainPhases) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config = build_injected_failure_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  for (const auto& phase :
       std::array<std::pair<const char*, int>, 2>{
           std::pair<const char*, int>{"cache-latency", 2},
           std::pair<const char*, int>{"main-latency", 3}}) {
    BenchmarkExecutorTestHooks hooks;
    hooks.fail_latency_chain_setup = [&](const std::string& phase_name) {
      return phase_name == phase.first;
    };
    const std::string expected_reason =
        Messages::benchmark_reason_latency_chain_setup_failed(phase.first);
    std::string caught_reason;
    testing::internal::CaptureStderr();
    try {
      static_cast<void>(run_single_benchmark_loop(
          config, phase.second, *timer, nullptr, &hooks));
    } catch (const std::runtime_error& error) {
      caught_reason = error.what();
    }
    const std::string error_output =
        testing::internal::GetCapturedStderr();
    EXPECT_EQ(caught_reason, expected_reason) << phase.first;
    EXPECT_NE(error_output.find(expected_reason), std::string::npos)
        << phase.first;
  }
}
