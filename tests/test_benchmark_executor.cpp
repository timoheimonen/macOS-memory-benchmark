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
#include "benchmark/parallel_test_framework.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/memory/buffer_manager.h"
#include "core/memory/memory_utils.h"
#include "core/timing/timer.h"
#include "test_config_helpers.h"

namespace {

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

TEST(BenchmarkExecutorTest, MainMemoryLatencyCollectsSamplesWhenConfiguredIntegration) {
  BenchmarkConfig config = build_base_config();
  config.only_latency = true;
  config.latency_sample_count = 17;

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  TimingResults timings;
  BenchmarkResults results;
  run_main_memory_latency_test(buffers, config, timings, results, *timer_opt);

  EXPECT_GT(timings.total_lat_time_ns, 0.0);
  EXPECT_EQ(results.main_latency.samples.size(),
            static_cast<size_t>(config.latency_sample_count));
  ASSERT_TRUE(results.locality_16k_latency.is_measured());
  ASSERT_TRUE(results.global_random_latency.is_measured());
  EXPECT_GT(*results.locality_16k_latency.value, 0.0);
  EXPECT_GT(*results.global_random_latency.value, 0.0);
}

TEST(BenchmarkExecutorTest, MainMemoryLatencySkipsAutoTlbBreakdownWhenLocalitySpecifiedIntegration) {
  BenchmarkConfig config = build_base_config();
  config.only_latency = true;
  config.user_specified_latency_tlb_locality = true;
  config.latency_tlb_locality_bytes = static_cast<size_t>(16) * Constants::BYTES_PER_KB;

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  TimingResults timings;
  BenchmarkResults results;
  run_main_memory_latency_test(buffers, config, timings, results, *timer_opt);

  EXPECT_GT(timings.total_lat_time_ns, 0.0);
  EXPECT_EQ(results.locality_16k_latency.status,
            BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(results.global_random_latency.status,
            BenchmarkMeasurementStatus::NotRun);
  EXPECT_EQ(results.locality_latency_delta.status,
            BenchmarkMeasurementStatus::NotRun);
  EXPECT_FALSE(results.locality_16k_latency.value.has_value());
  EXPECT_FALSE(results.global_random_latency.value.has_value());
  EXPECT_FALSE(results.locality_latency_delta.value.has_value());
}

TEST(BenchmarkExecutorTest, MainMemoryLatencySkipsWhenMainLatencyDisabledIntegration) {
  BenchmarkConfig config = build_base_config();
  config.only_latency = true;
  config.buffer_size_mb = 0;
  config.buffer_size = 0;
  config.custom_cache_size_kb_ll = 8096;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = static_cast<size_t>(8096) * Constants::BYTES_PER_KB;

  calculate_buffer_sizes(config);
  calculate_access_counts(config);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  TimingResults timings;
  BenchmarkResults results;
  run_main_memory_latency_test(buffers, config, timings, results, *timer_opt);

  EXPECT_EQ(timings.total_lat_time_ns, 0.0);
  EXPECT_TRUE(results.main_latency.samples.empty());
  EXPECT_EQ(results.main_latency.status, BenchmarkMeasurementStatus::NotRun);
}

TEST(BenchmarkExecutorTest, InjectedPreparationFailureCoversEveryPhaseBoundary) {
  BenchmarkConfig config = build_base_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  BenchmarkBuffers unused_buffers;
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
    testing::internal::CaptureStderr();
    EXPECT_THROW(run_single_benchmark_loop(
                     unused_buffers, config, static_cast<int>(phase_index),
                     *timer, nullptr, &hooks),
                 std::runtime_error)
        << failing_phase;
    static_cast<void>(testing::internal::GetCapturedStderr());
  }
}

TEST(BenchmarkExecutorTest, InjectedLatencyChainFailureCoversCacheAndMainPhases) {
  BenchmarkConfig config = build_base_config();
  config.only_bandwidth = false;
  config.only_latency = false;
  BenchmarkBuffers unused_buffers;
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
    testing::internal::CaptureStderr();
    EXPECT_THROW(run_single_benchmark_loop(unused_buffers, config, phase.second,
                                           *timer, nullptr, &hooks),
                 std::runtime_error)
        << phase.first;
    static_cast<void>(testing::internal::GetCapturedStderr());
  }
}
