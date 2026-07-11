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
#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_statistics_collector.h"
#include "core/config/config.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "test_statistics_helpers.h"
#include "test_timer_system_calls.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

uint64_t deterministic_timer_ticks() { return 100; }

using ScopedDeterministicTimerSystemCalls =
    test_timer_system_calls::ScopedTimerSystemCalls<deterministic_timer_ticks>;

void inject_deterministic_elapsed(BenchmarkRunnerTestHooks& hooks) {
  hooks.elapsed_seconds = []() { return 1.0; };
}

void expect_double_vector_eq(const std::vector<double>& actual, const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "index " << i;
  }
}

BenchmarkConfig make_collector_config() {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.latency_sample_count = 3;
  config.buffer_size = 4096;
  config.lat_num_accesses = 16;
  config.l1_buffer_size = 1024;
  config.l2_buffer_size = 2048;
  return config;
}

BenchmarkResults make_collector_results() {
  BenchmarkResults results;
  results.status = BenchmarkRunStatus::Complete;
  results.planned_measurements = 12;
  results.completed_measurements = 12;
  set_measurement_value(results.main_read_bandwidth, 10.0, 1.0);
  set_measurement_value(results.main_write_bandwidth, 20.0, 1.0);
  set_measurement_value(results.main_copy_bandwidth, 30.0, 1.0);
  set_measurement_value(results.main_latency, 40.0, 1.0);
  set_measurement_value(results.locality_16k_latency, 41.0, 1.0);
  set_measurement_value(results.global_random_latency, 90.0, 1.0);
  set_measurement_value(results.locality_latency_delta, 49.0, 1.0);
  results.main_latency.samples = {40.1, 40.2};

  set_measurement_value(results.l1_latency, 5.0, 1.0);
  set_measurement_value(results.l1_read_bandwidth, 50.0, 1.0);
  set_measurement_value(results.l1_write_bandwidth, 60.0, 1.0);
  set_measurement_value(results.l1_copy_bandwidth, 70.0, 1.0);
  results.l1_latency.samples = {5.1, 5.2};

  set_measurement_value(results.l2_latency, 8.0, 1.0);
  set_measurement_value(results.l2_read_bandwidth, 80.0, 1.0);
  set_measurement_value(results.l2_write_bandwidth, 90.0, 1.0);
  set_measurement_value(results.l2_copy_bandwidth, 100.0, 1.0);
  results.l2_latency.samples = {8.1, 8.2};

  set_measurement_value(results.custom_latency, 12.0, 1.0);
  set_measurement_value(results.custom_read_bandwidth, 110.0, 1.0);
  set_measurement_value(results.custom_write_bandwidth, 120.0, 1.0);
  set_measurement_value(results.custom_copy_bandwidth, 130.0, 1.0);
  results.custom_latency.samples = {12.1, 12.2};

  return results;
}

}  // namespace

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsAggregatesMainAndDetectedCacheMetrics) {
  BenchmarkConfig config = make_collector_config();
  BenchmarkStatistics stats;
  stats.all_read_bw_gb_s = {999.0};
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  expect_double_vector_eq(stats.all_write_bw_gb_s, {20.0});
  expect_double_vector_eq(stats.all_copy_bw_gb_s, {30.0});
  expect_double_vector_eq(stats.all_average_latency_ns, {40.0});
  expect_double_vector_eq(stats.all_tlb_hit_latency_ns, {41.0});
  expect_double_vector_eq(stats.all_tlb_miss_latency_ns, {90.0});
  expect_double_vector_eq(stats.all_page_walk_penalty_ns, {49.0});
  expect_double_vector_eq(stats.all_main_mem_latency_samples, {40.1, 40.2});

  expect_double_vector_eq(stats.all_l1_latency_ns, {5.0});
  expect_double_vector_eq(stats.all_l1_read_bw_gb_s, {50.0});
  expect_double_vector_eq(stats.all_l1_write_bw_gb_s, {60.0});
  expect_double_vector_eq(stats.all_l1_copy_bw_gb_s, {70.0});
  expect_double_vector_eq(stats.all_l1_latency_samples, {5.1, 5.2});

  expect_double_vector_eq(stats.all_l2_latency_ns, {8.0});
  expect_double_vector_eq(stats.all_l2_read_bw_gb_s, {80.0});
  expect_double_vector_eq(stats.all_l2_write_bw_gb_s, {90.0});
  expect_double_vector_eq(stats.all_l2_copy_bw_gb_s, {100.0});
  expect_double_vector_eq(stats.all_l2_latency_samples, {8.1, 8.2});
}

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsUsesCustomCacheInsteadOfDetectedCaches) {
  BenchmarkConfig config = make_collector_config();
  config.use_custom_cache_size = true;
  config.custom_buffer_size = 4096;

  BenchmarkStatistics stats;
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_custom_latency_ns, {12.0});
  expect_double_vector_eq(stats.all_custom_read_bw_gb_s, {110.0});
  expect_double_vector_eq(stats.all_custom_write_bw_gb_s, {120.0});
  expect_double_vector_eq(stats.all_custom_copy_bw_gb_s, {130.0});
  expect_double_vector_eq(stats.all_custom_latency_samples, {12.1, 12.2});

  EXPECT_TRUE(stats.all_l1_latency_ns.empty());
  EXPECT_TRUE(stats.all_l1_read_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_l2_latency_ns.empty());
  EXPECT_TRUE(stats.all_l2_read_bw_gb_s.empty());
}

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsSkipsMainLatencyWhenDisabled) {
  BenchmarkConfig config = make_collector_config();
  config.only_bandwidth = true;

  BenchmarkStatistics stats;
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  EXPECT_TRUE(stats.all_average_latency_ns.empty());
  EXPECT_TRUE(stats.all_tlb_hit_latency_ns.empty());
  EXPECT_TRUE(stats.all_tlb_miss_latency_ns.empty());
  EXPECT_TRUE(stats.all_page_walk_penalty_ns.empty());
  EXPECT_TRUE(stats.all_main_mem_latency_samples.empty());
}

TEST(BenchmarkStatisticsCollectorTest, InterruptedMeasurementsNeverEnterAggregates) {
  BenchmarkConfig config = make_collector_config();
  BenchmarkStatistics stats;
  initialize_statistics(stats, config);

  BenchmarkResults results = make_collector_results();
  results.status = BenchmarkRunStatus::Interrupted;
  results.planned_measurements = 12;
  results.completed_measurements = 10;
  set_measurement_unavailable(results.main_write_bandwidth,
                              BenchmarkMeasurementStatus::Interrupted,
                              "interrupted during measured operation");
  set_measurement_unavailable(results.main_latency,
                              BenchmarkMeasurementStatus::Invalid,
                              "invalid latency duration");

  collect_loop_results(stats, results, config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  EXPECT_TRUE(stats.all_write_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_average_latency_ns.empty());
  EXPECT_TRUE(stats.all_main_mem_latency_samples.empty());
  EXPECT_EQ(stats.completed_measurements, 10u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(stats.loop_results[0].main_write_bandwidth.status,
            BenchmarkMeasurementStatus::Interrupted);
  EXPECT_FALSE(stats.loop_results[0].main_write_bandwidth.value.has_value());
}

TEST(BenchmarkStatisticsCollectorTest, InitializationResetsStateAndReservesExactPopulations) {
  BenchmarkConfig config;
  config.loop_count = 4;
  config.latency_sample_count = 3;
  config.l1_buffer_size = 1024;
  config.l2_buffer_size = 2048;
  BenchmarkStatistics stats;
  stats.status = BenchmarkRunStatus::Failed;
  stats.status_reason = "stale";
  stats.planned_loops = 99;
  stats.completed_loops = 98;
  stats.planned_measurements = 97;
  stats.completed_measurements = 96;
  stats.loop_results.push_back(BenchmarkResults{});
  stats.all_read_bw_gb_s = {1.0};
  stats.all_l1_latency_ns = {2.0};
  stats.all_main_mem_latency_samples = {3.0};

  initialize_statistics(stats, config);

  EXPECT_EQ(stats.status, BenchmarkRunStatus::NotStarted);
  EXPECT_TRUE(stats.status_reason.empty());
  EXPECT_EQ(stats.planned_loops, 4u);
  EXPECT_EQ(stats.completed_loops, 0u);
  EXPECT_EQ(stats.planned_measurements, 0u);
  EXPECT_EQ(stats.completed_measurements, 0u);
  EXPECT_TRUE(stats.loop_results.empty());
  EXPECT_GE(stats.loop_results.capacity(), 4u);
  EXPECT_TRUE(stats.all_read_bw_gb_s.empty());
  EXPECT_GE(stats.all_read_bw_gb_s.capacity(), 4u);
  EXPECT_TRUE(stats.all_l1_latency_ns.empty());
  EXPECT_GE(stats.all_l1_latency_ns.capacity(), 4u);
  EXPECT_TRUE(stats.all_main_mem_latency_samples.empty());
  EXPECT_GE(stats.all_main_mem_latency_samples.capacity(), 12u);
  EXPECT_GE(stats.all_l1_latency_samples.capacity(), 12u);
  EXPECT_GE(stats.all_l2_latency_samples.capacity(), 12u);
}

TEST(BenchmarkRunnerTest, InjectedTimerCreationFailureIsReportedAndCheckpointed) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  BenchmarkStatistics stats;
  size_t checkpoints = 0;
  BenchmarkRunnerTestHooks hooks;
  hooks.force_timer_creation_failure = true;
  inject_deterministic_elapsed(hooks);
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics&,
                         double, bool) {
    ++checkpoints;
    return EXIT_SUCCESS;
  };

  testing::internal::CaptureStderr();
  const int result = run_all_benchmarks(config, stats, &hooks);
  const std::string error_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::error_timer_creation_failed());
  EXPECT_EQ(checkpoints, 1u);
  EXPECT_NE(error_output.find(stats.status_reason), std::string::npos);
}

TEST(BenchmarkRunnerTest, InjectedLoopExceptionIsFailedAndCheckpointedWithExactReason) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  BenchmarkStatistics stats;
  size_t checkpoints = 0;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.execute_loop = [](BenchmarkConfig&, int, HighResTimer&,
                          BenchmarkExecutionState*)
      -> BenchmarkResults { throw std::runtime_error("injected loop failure"); };
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics& snapshot,
                         double, bool) {
    ++checkpoints;
    EXPECT_EQ(snapshot.status, BenchmarkRunStatus::Failed);
    EXPECT_EQ(snapshot.status_reason, "injected loop failure");
    return EXIT_SUCCESS;
  };

  testing::internal::CaptureStderr();
  const int result = run_all_benchmarks(config, stats, &hooks);
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, "injected loop failure");
  EXPECT_EQ(stats.completed_loops, 0u);
  EXPECT_TRUE(stats.loop_results.empty());
  EXPECT_EQ(checkpoints, 1u);
  EXPECT_EQ(error, Messages::error_benchmark_loop(0, "injected loop failure") +
                       "\n");
}

TEST(BenchmarkRunnerTest, UnknownLoopExceptionIsContainedWithCentralizedReason) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  hooks.execute_loop = [](BenchmarkConfig&, int, HighResTimer&,
                          BenchmarkExecutionState*)
      -> BenchmarkResults { throw 7; };

  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(
      result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_unknown_loop_exception());
  EXPECT_EQ(error,
            Messages::error_benchmark_loop(0, stats.status_reason) + "\n");
}

TEST(BenchmarkRunnerTest, StopHookExceptionIsContainedAtCoordinatorBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  hooks.stop_requested = []() -> bool {
    throw std::runtime_error("injected stop failure");
  };

  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(
      result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_coordinator_exception(
                "injected stop failure"));
  EXPECT_EQ(error,
            Messages::error_prefix() + stats.status_reason + "\n");
}

TEST(BenchmarkRunnerTest,
     UnknownCheckpointExceptionPreservesLoopAtCoordinatorBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&,
                          BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [](const BenchmarkConfig&, const BenchmarkStatistics&,
                        double, bool) -> int { throw 11; };

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(
      result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_unknown_coordinator_exception());
  EXPECT_EQ(stats.completed_loops, 1u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(error,
            Messages::error_prefix() + stats.status_reason + "\n");
}

TEST(BenchmarkRunnerTest, InjectedCheckpointFailurePreservesCompletedLoopButFailsCommand) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&,
                          BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [](const BenchmarkConfig&, const BenchmarkStatistics&,
                        double, bool) { return EXIT_FAILURE; };

  testing::internal::CaptureStdout();
  const int result = run_all_benchmarks(config, stats, &hooks);
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_checkpoint_failed());
  EXPECT_EQ(stats.completed_loops, 1u);
}

TEST(BenchmarkRunnerTest, InjectedStopBetweenLoopsPreservesCompletedLoop) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 3;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  size_t stop_checks = 0;
  size_t checkpoints = 0;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.stop_requested = [&] { return stop_checks++ >= 1; };
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&,
                          BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics&,
                         double, bool) {
    ++checkpoints;
    return EXIT_SUCCESS;
  };

  testing::internal::CaptureStdout();
  const int result = run_all_benchmarks(config, stats, &hooks);
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Interrupted);
  EXPECT_EQ(stats.completed_loops, 1u);
  EXPECT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(checkpoints, 2u);
}

TEST(BenchmarkRunnerTest, StatisticsPrintsPairedLocalityMetrics) {
  const std::vector<double> all_main_mem_latency = {15.0, 16.0};
  const std::vector<double> all_tlb_hit_latency = {14.0, 15.0};
  const std::vector<double> all_tlb_miss_latency = {90.0, 92.0};
  const std::vector<double> all_page_walk_penalty = {76.0, 77.0};

  const std::string output = test_statistics_helpers::capture_auto_tlb_breakdown(
      all_main_mem_latency, all_tlb_hit_latency, all_tlb_miss_latency, all_page_walk_penalty);

  EXPECT_NE(output.find("16 KiB Locality Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("Global-Random Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("Locality Latency Delta, Global - 16 KiB (ns):"),
            std::string::npos);
}
