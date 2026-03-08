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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unistd.h>

#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_tests.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/memory/buffer_manager.h"
#include "core/memory/memory_utils.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"

namespace {

BenchmarkConfig build_base_config() {
  BenchmarkConfig config;

  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();

  config.buffer_size = static_cast<size_t>(getpagesize());
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 1;
  config.use_custom_cache_size = false;

  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  return config;
}

}  // namespace

TEST(BenchmarkExecutorTest, WriteTestCoversEntireBufferForUnevenThreadSplits) {
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

TEST(BenchmarkExecutorTest, WriteTestReturnsZeroForInvalidThreadCounts) {
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

TEST(BenchmarkExecutorTest, LatencySamplingClampsToAccessCount) {
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

TEST(BenchmarkExecutorTest, MainMemoryLatencyCollectsSamplesWhenConfigured) {
  BenchmarkConfig config = build_base_config();
  config.only_latency = true;
  config.latency_sample_count = 17;

  BenchmarkBuffers buffers;
  ASSERT_EQ(allocate_all_buffers(config, buffers), EXIT_SUCCESS);
  ASSERT_EQ(initialize_all_buffers(buffers, config), EXIT_SUCCESS);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  TimingResults timings;
  BenchmarkResults results;
  run_main_memory_latency_test(buffers, config, timings, results, *timer_opt);

  EXPECT_GT(timings.total_lat_time_ns, 0.0);
  EXPECT_EQ(results.latency_samples.size(), static_cast<size_t>(config.latency_sample_count));
}

TEST(BenchmarkExecutorTest, MainMemoryLatencySkipsWhenMainLatencyDisabled) {
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
  ASSERT_EQ(allocate_all_buffers(config, buffers), EXIT_SUCCESS);
  ASSERT_EQ(initialize_all_buffers(buffers, config), EXIT_SUCCESS);

  auto timer_opt = HighResTimer::create();
  ASSERT_TRUE(timer_opt.has_value());

  TimingResults timings;
  BenchmarkResults results;
  run_main_memory_latency_test(buffers, config, timings, results, *timer_opt);

  EXPECT_EQ(timings.total_lat_time_ns, 0.0);
  EXPECT_TRUE(results.latency_samples.empty());
}
