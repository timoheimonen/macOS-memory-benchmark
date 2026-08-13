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
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_tests.h"
#include "benchmark/benchmark_work_plan.h"
#include "benchmark/parallel_test_framework.h"
#include "core/config/config.h"
#include "core/config/constants.h"
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

struct ThreadExitTickProbe {
  std::atomic<uint64_t>* observed_tick;

  ~ThreadExitTickProbe() {
    observed_tick->store(deterministic_timer_tick, std::memory_order_relaxed);
  }
};

using ScopedDeterministicTimerSystemCalls =
    test_timer_system_calls::ScopedTimerSystemCalls<deterministic_timer_ticks, reset_deterministic_timer_ticks>;

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

}  // namespace

TEST(BenchmarkExecutorTest, ParallelExecutorConsumesAlignedUnevenChunksExactly) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  alignas(Constants::CACHE_LINE_SIZE_BYTES) std::array<unsigned char, 2049> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  std::array<size_t, 10> observed_offsets{};
  std::array<size_t, 10> observed_sizes{};
  std::array<std::atomic<bool>, 10> executed;
  for (std::atomic<bool>& flag : executed) {
    flag.store(false, std::memory_order_relaxed);
  }
  auto make_work = [&](size_t offset, size_t size, int iterations, size_t worker_index) {
    observed_offsets[worker_index] = offset;
    observed_sizes[worker_index] = size;
    EXPECT_EQ(iterations, 1);
    return [&, worker_index] { executed[worker_index].store(true, std::memory_order_relaxed); };
  };
  ParallelExecutionMetadata metadata;
  const double elapsed = run_parallel_test_common(buffer.data(), buffer.size(), 1, 10, *timer, "partition-test",
                                                  make_work, nullptr, &metadata);

  EXPECT_DOUBLE_EQ(elapsed, 100.0 / 1e9);
  EXPECT_EQ(metadata.created_workers, 10);
  EXPECT_FALSE(metadata.worker_startup_failed);

  size_t covered = 0;
  for (size_t index = 0; index < observed_offsets.size(); ++index) {
    EXPECT_EQ(observed_offsets[index], covered);
    EXPECT_GT(observed_sizes[index], 0u);
    EXPECT_TRUE(executed[index].load(std::memory_order_relaxed));
    covered += observed_sizes[index];
    if (index + 1 < observed_offsets.size()) {
      const uintptr_t boundary_address = reinterpret_cast<uintptr_t>(buffer.data()) + covered;
      EXPECT_EQ(boundary_address % Constants::CACHE_LINE_SIZE_BYTES, 0u);
    }
  }
  EXPECT_EQ(covered, buffer.size());
}

TEST(BenchmarkExecutorTest, WriteTestRejectsZeroThreadCountWithoutModifyingBuffer) {
  std::array<unsigned char, 4096> buffer{};
  buffer.fill(0xCD);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  EXPECT_EQ(run_write_test(buffer.data(), buffer.size(), 1, 0, *timer_opt), 0.0);

  const bool unchanged = std::all_of(buffer.begin(), buffer.end(), [](unsigned char value) { return value == 0xCD; });
  EXPECT_TRUE(unchanged);
}

TEST(BenchmarkExecutorTest, ParallelTimingStopsBeforeWorkerTeardown) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  std::array<unsigned char, 4096> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  uint64_t tick_during_work = 0;
  std::atomic<uint64_t> tick_at_worker_teardown{0};
  auto make_work = [&](size_t, size_t, int, size_t) {
    return [&] {
      thread_local ThreadExitTickProbe probe{&tick_at_worker_teardown};
      static_cast<void>(probe);
      tick_during_work = deterministic_timer_tick;
    };
  };

  const double measured_duration = run_parallel_test_common(buffer.data(), buffer.size(), 1, 1, *timer, "timing_test",
                                                            make_work);

  EXPECT_DOUBLE_EQ(measured_duration, 100.0 / 1e9);
  EXPECT_EQ(tick_during_work, 100u);
  EXPECT_EQ(tick_at_worker_teardown.load(std::memory_order_relaxed), 200u);
  EXPECT_EQ(deterministic_timer_tick, 200u);
}

TEST(BenchmarkExecutorTest, LatencySamplingClampsAndContinuesFromPriorTerminalPointer) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  auto verify_sampling = [&](int requested_samples, const std::vector<size_t>& expected_access_counts,
                             const std::vector<double>& expected_samples) {
    reset_deterministic_timer_ticks();
    std::array<uintptr_t, 8> nodes{};
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
    const double total_duration_ns = run_latency_test(&nodes[0], 7, *timer, &samples, requested_samples, &hooks);

    std::vector<uintptr_t*> expected_starts;
    for (size_t index = 0; index < expected_access_counts.size(); ++index) {
      expected_starts.push_back(&nodes[index]);
    }
    EXPECT_DOUBLE_EQ(total_duration_ns, 100.0 * static_cast<double>(expected_access_counts.size()));
    EXPECT_EQ(samples, expected_samples);
    EXPECT_EQ(observed_starts, expected_starts);
    EXPECT_EQ(observed_access_counts, expected_access_counts);
  };

  verify_sampling(3, {3, 2, 2}, {100.0 / 3.0, 50.0, 50.0});
  verify_sampling(100, {1, 1, 1, 1, 1, 1, 1}, {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0});
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

  const BenchmarkResults first = run_single_benchmark_loop(config, 0, *timer, &execution_state);
  ASSERT_EQ(first.status, BenchmarkRunStatus::Complete);
  EXPECT_EQ(first.planned_phases, 1u);
  EXPECT_EQ(first.completed_phases, 1u);
  EXPECT_EQ(first.planned_measurements, 1u);
  EXPECT_EQ(first.completed_measurements, 1u);
  EXPECT_EQ(first.planned_phase_order, (std::vector<std::string>{"main-latency"}));
  EXPECT_EQ(first.realized_phase_order, first.planned_phase_order);

  const BenchmarkMeasurement& first_latency = first.main_latency;
  ASSERT_TRUE(first_latency.is_measured());
  EXPECT_EQ(first_latency.target, "main-memory");
  EXPECT_EQ(first_latency.operation, "latency");
  EXPECT_EQ(first_latency.work_policy, "automatic-duration-calibration");
  EXPECT_TRUE(first_latency.automatic_calibration);
  EXPECT_GT(first_latency.access_count, 0u);
  EXPECT_GT(first_latency.chain_node_count, 1u);
  EXPECT_GE(first_latency.complete_chain_cycles, Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES);
  EXPECT_EQ(first_latency.requested_threads, 1);
  EXPECT_EQ(first_latency.effective_threads, 1);
  EXPECT_EQ(first_latency.created_workers, 1);
  EXPECT_EQ(first.locality_16k_latency.status, BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(first.global_random_latency.status, BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(first.locality_latency_delta.status, BenchmarkMeasurementStatus::NotRun);

  ASSERT_TRUE(execution_state.latency[benchmark_latency_state_index(BenchmarkTarget::MainMemory)].initialized);
  const size_t resolved_access_count = first_latency.access_count;
  const double pilot_elapsed_seconds = first_latency.pilot_elapsed_seconds;

  const BenchmarkResults second = run_single_benchmark_loop(config, 1, *timer, &execution_state);
  ASSERT_EQ(second.status, BenchmarkRunStatus::Complete);
  ASSERT_TRUE(second.main_latency.is_measured());
  EXPECT_EQ(second.loop_index, 1u);
  EXPECT_EQ(second.main_latency.access_count, resolved_access_count);
  EXPECT_DOUBLE_EQ(second.main_latency.pilot_elapsed_seconds, pilot_elapsed_seconds);
  EXPECT_EQ(second.main_latency.seed, first_latency.seed);
}

TEST(BenchmarkExecutorTest, InjectedPreparationFailureCoversEveryPhaseBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config = build_injected_failure_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  const std::array<const char*, 4> phases = {"main-bandwidth", "cache-bandwidth", "cache-latency", "main-latency"};

  for (size_t phase_index = 0; phase_index < phases.size(); ++phase_index) {
    const std::string failing_phase = phases[phase_index];
    BenchmarkExecutorTestHooks hooks;
    hooks.fail_phase_preparation = [&](const std::string& phase_name) { return phase_name == failing_phase; };
    const std::string expected_reason = Messages::benchmark_reason_prepare_failed(failing_phase);
    std::string caught_reason;
    testing::internal::CaptureStderr();
    try {
      static_cast<void>(run_single_benchmark_loop(config, static_cast<int>(phase_index), *timer, nullptr, &hooks));
    } catch (const std::runtime_error& error) {
      caught_reason = error.what();
    }
    const std::string error_output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(caught_reason, expected_reason) << failing_phase;
    EXPECT_NE(error_output.find(expected_reason), std::string::npos) << failing_phase;
  }
}

TEST(BenchmarkExecutorTest, InjectedLatencyChainFailureCoversCacheAndMainPhases) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config = build_injected_failure_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  for (const auto& phase : std::array<std::pair<const char*, int>, 2>{std::pair<const char*, int>{"cache-latency", 2},
                                                                      std::pair<const char*, int>{"main-latency", 3}}) {
    BenchmarkExecutorTestHooks hooks;
    hooks.fail_latency_chain_setup = [&](const std::string& phase_name) { return phase_name == phase.first; };
    const std::string expected_reason = Messages::benchmark_reason_latency_chain_setup_failed(phase.first);
    std::string caught_reason;
    testing::internal::CaptureStderr();
    try {
      static_cast<void>(run_single_benchmark_loop(config, phase.second, *timer, nullptr, &hooks));
    } catch (const std::runtime_error& error) {
      caught_reason = error.what();
    }
    const std::string error_output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(caught_reason, expected_reason) << phase.first;
    EXPECT_NE(error_output.find(expected_reason), std::string::npos) << phase.first;
  }
}
