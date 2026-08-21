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

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_statistics_collector.h"
#include "core/config/config.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"
#include "output/json/json_output/json_output_session.h"
#include "test_timer_system_calls.h"

namespace {

uint64_t deterministic_timer_ticks() { return 100; }

using ScopedDeterministicTimerSystemCalls = test_timer_system_calls::ScopedTimerSystemCalls<deterministic_timer_ticks>;

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

class TemporaryBenchmarkJsonFile {
 public:
  TemporaryBenchmarkJsonFile() {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_runner_session_" + std::to_string(::getpid()) +
             "_" + std::to_string(sequence.fetch_add(1)) + ".json");
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_.string() + ".tmp", ignored);
  }

  ~TemporaryBenchmarkJsonFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_.string() + ".tmp", ignored);
  }

  TemporaryBenchmarkJsonFile(const TemporaryBenchmarkJsonFile&) = delete;
  TemporaryBenchmarkJsonFile& operator=(
      const TemporaryBenchmarkJsonFile&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

nlohmann::ordered_json read_ordered_json_file(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  nlohmann::ordered_json output;
  input >> output;
  return output;
}

}  // namespace

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsAggregatesMainAndDetectedCacheMetrics) {
  BenchmarkConfig config = make_collector_config();
  BenchmarkStatistics stats;
  stats.status = BenchmarkRunStatus::Failed;
  stats.status_reason = "stale";
  stats.planned_loops = 99;
  stats.completed_loops = 98;
  stats.planned_measurements = 97;
  stats.completed_measurements = 96;
  stats.loop_results.push_back(BenchmarkResults{});
  stats.all_read_bw_gb_s = {999.0};
  stats.all_l1_latency_ns = {998.0};
  stats.all_main_mem_latency_samples = {997.0};
  initialize_statistics(stats, config);

  EXPECT_EQ(stats.status, BenchmarkRunStatus::NotStarted);
  EXPECT_TRUE(stats.status_reason.empty());
  EXPECT_EQ(stats.planned_loops, 2u);
  EXPECT_EQ(stats.completed_loops, 0u);
  EXPECT_EQ(stats.planned_measurements, 0u);
  EXPECT_EQ(stats.completed_measurements, 0u);
  EXPECT_TRUE(stats.loop_results.empty());

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
  set_measurement_unavailable(results.main_write_bandwidth, BenchmarkMeasurementStatus::Interrupted,
                              "interrupted during measured operation");
  set_measurement_unavailable(results.main_latency, BenchmarkMeasurementStatus::Invalid, "invalid latency duration");

  collect_loop_results(stats, results, config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  EXPECT_TRUE(stats.all_write_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_average_latency_ns.empty());
  EXPECT_TRUE(stats.all_main_mem_latency_samples.empty());
  EXPECT_EQ(stats.completed_measurements, 10u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(stats.loop_results[0].main_write_bandwidth.status, BenchmarkMeasurementStatus::Interrupted);
  EXPECT_FALSE(stats.loop_results[0].main_write_bandwidth.value.has_value());
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
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics&, double, bool) {
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
  hooks.execute_loop = [](BenchmarkConfig&, int, HighResTimer&, BenchmarkExecutionState*) -> BenchmarkResults {
    throw std::runtime_error("injected loop failure");
  };
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics& snapshot, double, bool) {
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
  EXPECT_EQ(error, Messages::error_benchmark_loop(0, "injected loop failure") + "\n");
}

TEST(BenchmarkRunnerTest, UnknownLoopExceptionIsContainedWithCentralizedReason) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  hooks.execute_loop = [](BenchmarkConfig&, int, HighResTimer&, BenchmarkExecutionState*) -> BenchmarkResults {
    throw 7;
  };

  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::benchmark_reason_unknown_loop_exception());
  EXPECT_EQ(error, Messages::error_benchmark_loop(0, stats.status_reason) + "\n");
}

TEST(BenchmarkRunnerTest, StopHookExceptionIsContainedAtCoordinatorBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  hooks.stop_requested = []() -> bool { throw std::runtime_error("injected stop failure"); };

  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::benchmark_reason_coordinator_exception("injected stop failure"));
  EXPECT_EQ(error, Messages::error_prefix() + stats.status_reason + "\n");
}

TEST(BenchmarkRunnerTest, UnknownCheckpointExceptionPreservesLoopAtCoordinatorBoundary) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&, BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [](const BenchmarkConfig&, const BenchmarkStatistics&, double, bool) -> int { throw 11; };

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(result = run_all_benchmarks(config, stats, &hooks));
  const std::string error = testing::internal::GetCapturedStderr();
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::benchmark_reason_unknown_coordinator_exception());
  EXPECT_EQ(stats.completed_loops, 1u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(error, Messages::error_prefix() + stats.status_reason + "\n");
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
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&, BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [](const BenchmarkConfig&, const BenchmarkStatistics&, double, bool) { return EXIT_FAILURE; };

  testing::internal::CaptureStdout();
  const int result = run_all_benchmarks(config, stats, &hooks);
  static_cast<void>(testing::internal::GetCapturedStdout());

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::benchmark_reason_checkpoint_failed());
  EXPECT_EQ(stats.completed_loops, 1u);
}

TEST(BenchmarkRunnerTest, StopBeforeFirstLoopCheckpointFailureTakesPrecedence) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 2;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  size_t executed_loops = 0;
  size_t checkpoints = 0;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.stop_requested = [] { return true; };
  hooks.execute_loop = [&](BenchmarkConfig&, int, HighResTimer&,
                           BenchmarkExecutionState*) {
    ++executed_loops;
    return BenchmarkResults{};
  };
  hooks.checkpoint = [&](const BenchmarkConfig&,
                         const BenchmarkStatistics& snapshot, double, bool) {
    ++checkpoints;
    EXPECT_EQ(snapshot.status, BenchmarkRunStatus::Interrupted);
    EXPECT_EQ(snapshot.status_reason, Messages::msg_interrupted_by_user());
    EXPECT_EQ(snapshot.completed_loops, 0u);
    EXPECT_TRUE(snapshot.loop_results.empty());
    return EXIT_FAILURE;
  };

  testing::internal::CaptureStdout();
  const int result = run_all_benchmarks(config, stats, &hooks);
  const std::string human_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_checkpoint_failed());
  EXPECT_EQ(stats.planned_loops, 2u);
  EXPECT_EQ(stats.completed_loops, 0u);
  EXPECT_TRUE(stats.loop_results.empty());
  EXPECT_EQ(executed_loops, 0u);
  EXPECT_EQ(checkpoints, 1u);
  EXPECT_TRUE(human_output.empty());
  EXPECT_FALSE(should_write_standard_final_json(JsonOutputKind::File, result));
}

TEST(BenchmarkRunnerTest,
     CommandBoundaryFinalizationNeverRetriesFailedFileCheckpoint) {
  EXPECT_FALSE(should_write_standard_final_json(JsonOutputKind::Disabled,
                                                EXIT_SUCCESS));
  EXPECT_TRUE(should_write_standard_final_json(JsonOutputKind::File,
                                               EXIT_SUCCESS));
  EXPECT_FALSE(should_write_standard_final_json(JsonOutputKind::File,
                                                EXIT_FAILURE));
  EXPECT_TRUE(should_write_standard_final_json(JsonOutputKind::Stdout,
                                               EXIT_SUCCESS));
  EXPECT_TRUE(should_write_standard_final_json(JsonOutputKind::Stdout,
                                               EXIT_FAILURE));
}

TEST(BenchmarkRunnerTest,
     InterruptionCheckpointFailureAfterOneLoopRetainsEvidence) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 3;
  config.output_file = "/tmp/benchmark-runner-hook-unused.json";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  size_t stop_checks = 0;
  size_t executed_loops = 0;
  size_t checkpoints = 0;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.stop_requested = [&] { return stop_checks++ >= 1; };
  hooks.execute_loop = [&](BenchmarkConfig&, int loop, HighResTimer&,
                           BenchmarkExecutionState*) {
    ++executed_loops;
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    results.planned_phases = 1;
    results.completed_phases = 1;
    results.planned_measurements = 1;
    results.completed_measurements = 1;
    return results;
  };
  hooks.checkpoint = [&](const BenchmarkConfig&,
                         const BenchmarkStatistics& snapshot, double, bool) {
    ++checkpoints;
    EXPECT_EQ(snapshot.completed_loops, 1u);
    EXPECT_EQ(snapshot.loop_results.size(), 1u);
    if (checkpoints == 1) {
      EXPECT_EQ(snapshot.status, BenchmarkRunStatus::Partial);
      return EXIT_SUCCESS;
    }
    EXPECT_EQ(snapshot.status, BenchmarkRunStatus::Interrupted);
    EXPECT_EQ(snapshot.status_reason, Messages::msg_interrupted_by_user());
    return EXIT_FAILURE;
  };

  testing::internal::CaptureStdout();
  const int result = run_all_benchmarks(config, stats, &hooks);
  const std::string human_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason,
            Messages::benchmark_reason_checkpoint_failed());
  EXPECT_EQ(stats.planned_loops, 3u);
  EXPECT_EQ(stats.completed_loops, 1u);
  EXPECT_EQ(stats.planned_measurements, 1u);
  EXPECT_EQ(stats.completed_measurements, 1u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(stats.loop_results[0].loop_index, 0u);
  EXPECT_EQ(stats.loop_results[0].status, BenchmarkRunStatus::Complete);
  EXPECT_EQ(executed_loops, 1u);
  EXPECT_EQ(stop_checks, 2u);
  EXPECT_EQ(checkpoints, 2u);
  EXPECT_EQ(human_output.find(Messages::msg_interrupted_by_user()),
            std::string::npos);
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
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&, BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    return results;
  };
  hooks.checkpoint = [&](const BenchmarkConfig&, const BenchmarkStatistics&, double, bool) {
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

TEST(BenchmarkRunnerTest,
     StdoutInterruptionRetainsOneLoopAndEmitsOneSchema2TerminalDocument) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  BenchmarkConfig config;
  config.loop_count = 2;
  config.output_file = "-";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  size_t stop_checks = 0;
  size_t executed_loops = 0;
  size_t logical_checkpoints = 0;
  size_t checkpoint_payload_builds = 0;
  BenchmarkRunnerTestHooks hooks;
  hooks.stop_requested = [&] { return stop_checks++ >= 1; };
  hooks.elapsed_seconds = [&] {
    ++logical_checkpoints;
    return 1.0;
  };
  hooks.checkpoint_payload_build = [&] { ++checkpoint_payload_builds; };
  hooks.execute_loop = [&](BenchmarkConfig&, int loop, HighResTimer&,
                           BenchmarkExecutionState*) {
    ++executed_loops;
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    results.planned_phases = 1;
    results.completed_phases = 1;
    results.planned_measurements = 1;
    results.completed_measurements = 1;
    return results;
  };

  int run_result = EXIT_FAILURE;
  int output_result = EXIT_FAILURE;
  nlohmann::ordered_json terminal_payload;
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  {
    JsonOutputSession session(make_json_output_target(config.output_file));
    run_result = run_all_benchmarks(config, stats, &hooks, &session);
    terminal_payload = build_results_json(config, stats, 1.0);
    output_result = session.write_final(terminal_payload);
  }
  const std::string human_output = testing::internal::GetCapturedStderr();
  const std::string json_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(run_result, EXIT_SUCCESS);
  EXPECT_EQ(output_result, EXIT_SUCCESS);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Interrupted);
  EXPECT_EQ(stats.status_reason, Messages::msg_interrupted_by_user());
  EXPECT_EQ(stats.planned_loops, 2u);
  EXPECT_EQ(stats.completed_loops, 1u);
  EXPECT_EQ(stats.planned_measurements, 1u);
  EXPECT_EQ(stats.completed_measurements, 1u);
  ASSERT_EQ(stats.loop_results.size(), 1u);
  EXPECT_EQ(stats.loop_results[0].status, BenchmarkRunStatus::Complete);
  EXPECT_EQ(executed_loops, 1u);
  EXPECT_EQ(stop_checks, 2u);
  EXPECT_EQ(logical_checkpoints, 2u);
  EXPECT_EQ(checkpoint_payload_builds, 0u);

  EXPECT_EQ(json_output, terminal_payload.dump(2) + "\n");
  nlohmann::ordered_json parsed_payload;
  ASSERT_NO_THROW(parsed_payload =
                      nlohmann::ordered_json::parse(json_output));
  EXPECT_EQ(parsed_payload, terminal_payload);
  ASSERT_TRUE(parsed_payload.contains("configuration"));
  EXPECT_EQ(parsed_payload["configuration"]["benchmark_schema_version"], 2);
  EXPECT_EQ(parsed_payload["status"], "interrupted");
  EXPECT_FALSE(parsed_payload["results_complete"].get<bool>());
  EXPECT_EQ(parsed_payload["planned_loops"], 2);
  EXPECT_EQ(parsed_payload["completed_loops"], 1);
  EXPECT_EQ(parsed_payload["planned_measurements"], 1);
  EXPECT_EQ(parsed_payload["completed_measurements"], 1);
  ASSERT_TRUE(parsed_payload["loops"].is_array());
  ASSERT_EQ(parsed_payload["loops"].size(), 1u);
  EXPECT_EQ(parsed_payload["loops"][0]["status"], "complete");
  EXPECT_NE(human_output.find("Interrupted by user"), std::string::npos);
}

TEST(BenchmarkRunnerTest,
     StdoutSessionSuppressesFailureCheckpointAndEmitsOneTerminalPayload) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = "-";
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  hooks.force_timer_creation_failure = true;
  inject_deterministic_elapsed(hooks);

  int run_result = EXIT_SUCCESS;
  int output_result = EXIT_FAILURE;
  nlohmann::ordered_json terminal_payload;
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  {
    JsonOutputSession session(make_json_output_target(config.output_file));
    run_result = run_all_benchmarks(config, stats, &hooks, &session);
    terminal_payload = build_results_json(config, stats, 1.0);
    output_result = session.write_final(terminal_payload);
  }
  const std::string error_output = testing::internal::GetCapturedStderr();
  const std::string json_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(run_result, EXIT_FAILURE);
  EXPECT_EQ(output_result, EXIT_SUCCESS);
  EXPECT_EQ(stats.status, BenchmarkRunStatus::Failed);
  EXPECT_EQ(stats.status_reason, Messages::error_timer_creation_failed());
  EXPECT_EQ(json_output, terminal_payload.dump(2) + "\n");
  EXPECT_EQ(nlohmann::ordered_json::parse(json_output), terminal_payload);
  EXPECT_EQ(terminal_payload["status"], "failed");
  EXPECT_FALSE(terminal_payload["results_complete"].get<bool>());
  EXPECT_NE(error_output.find(Messages::error_timer_creation_failed()),
            std::string::npos);
}

TEST(BenchmarkRunnerTest,
     FileSessionPersistsTheExistingCompletedLoopCheckpoint) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  TemporaryBenchmarkJsonFile output_file;
  BenchmarkConfig config;
  config.loop_count = 1;
  config.output_file = output_file.path().string();
  config.only_bandwidth = true;
  BenchmarkStatistics stats;
  BenchmarkRunnerTestHooks hooks;
  inject_deterministic_elapsed(hooks);
  hooks.execute_loop = [](BenchmarkConfig&, int loop, HighResTimer&,
                          BenchmarkExecutionState*) {
    BenchmarkResults results;
    results.status = BenchmarkRunStatus::Complete;
    results.loop_index = static_cast<size_t>(loop);
    results.planned_measurements = 1;
    results.completed_measurements = 1;
    return results;
  };

  testing::internal::CaptureStdout();
  int run_result = EXIT_FAILURE;
  {
    JsonOutputSession session(make_json_output_target(config.output_file));
    run_result = run_all_benchmarks(config, stats, &hooks, &session);
  }
  static_cast<void>(testing::internal::GetCapturedStdout());

  ASSERT_EQ(run_result, EXIT_SUCCESS);
  ASSERT_TRUE(std::filesystem::exists(output_file.path()));
  const nlohmann::ordered_json checkpoint =
      read_ordered_json_file(output_file.path());
  EXPECT_EQ(checkpoint["status"], "complete");
  EXPECT_TRUE(checkpoint["results_complete"].get<bool>());
  EXPECT_EQ(checkpoint["completed_loops"], 1);
  EXPECT_EQ(checkpoint["completed_measurements"], 1);
  EXPECT_FALSE(
      std::filesystem::exists(output_file.path().string() + ".tmp"));
}
