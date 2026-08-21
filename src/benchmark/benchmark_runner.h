// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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

/**
 * @file benchmark_runner.h
 * @brief Benchmark runner and results structures
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header defines structures for storing benchmark results and provides
 * functions to run all benchmark loops and collect statistics.
 */
 
#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include <vector>
#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <functional>
#include <string>

#include "benchmark/benchmark_measurement.h"

// Forward declarations to avoid including headers
struct BenchmarkConfig;
struct BenchmarkExecutionState;
struct HighResTimer;
class JsonOutputSession;
enum class JsonOutputKind;

/**
 * @struct BenchmarkResults
 * @brief Structure containing results from a single benchmark loop
 *
 * Stores all performance measurements from one complete benchmark execution,
 * including bandwidth (GB/s) and latency (nanoseconds) for main memory and caches.
 */
struct BenchmarkResults {
  BenchmarkRunStatus status = BenchmarkRunStatus::NotStarted;
  std::string status_reason;
  size_t loop_index = 0;
  size_t planned_phases = 0;
  size_t completed_phases = 0;
  size_t planned_measurements = 0;
  size_t completed_measurements = 0;
  size_t phase_order_index = 0;
  size_t operation_order_index = 0;
  std::vector<std::string> planned_phase_order;
  std::vector<std::string> realized_phase_order;

  BenchmarkMeasurement main_read_bandwidth;
  BenchmarkMeasurement main_write_bandwidth;
  BenchmarkMeasurement main_copy_bandwidth;
  BenchmarkMeasurement main_latency;
  BenchmarkMeasurement locality_16k_latency;
  BenchmarkMeasurement global_random_latency;
  BenchmarkMeasurement locality_latency_delta;

  BenchmarkMeasurement l1_read_bandwidth;
  BenchmarkMeasurement l1_write_bandwidth;
  BenchmarkMeasurement l1_copy_bandwidth;
  BenchmarkMeasurement l1_latency;
  BenchmarkMeasurement l2_read_bandwidth;
  BenchmarkMeasurement l2_write_bandwidth;
  BenchmarkMeasurement l2_copy_bandwidth;
  BenchmarkMeasurement l2_latency;
  BenchmarkMeasurement custom_read_bandwidth;
  BenchmarkMeasurement custom_write_bandwidth;
  BenchmarkMeasurement custom_copy_bandwidth;
  BenchmarkMeasurement custom_latency;
};

/**
 * @struct BenchmarkStatistics
 * @brief Structure containing aggregated statistics across all benchmark loops
 *
 * Collects results from multiple benchmark loops to enable statistical analysis,
 * including mean, min, max, and percentile calculations.
 */
struct BenchmarkStatistics {
  BenchmarkRunStatus status = BenchmarkRunStatus::NotStarted;
  std::string status_reason;
  size_t planned_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t completed_measurements = 0;
  std::vector<BenchmarkResults> loop_results;

  // Vectors storing results from each loop
  std::vector<double> all_read_bw_gb_s;        ///< Read bandwidth from each loop (GB/s)
  std::vector<double> all_write_bw_gb_s;      ///< Write bandwidth from each loop (GB/s)
  std::vector<double> all_copy_bw_gb_s;       ///< Copy bandwidth from each loop (GB/s)
  std::vector<double> all_l1_latency_ns;      ///< L1 latency from each loop (nanoseconds)
  std::vector<double> all_l2_latency_ns;      ///< L2 latency from each loop (nanoseconds)
  std::vector<double> all_average_latency_ns; ///< Main memory latency from each loop (nanoseconds)
  std::vector<double> all_tlb_hit_latency_ns; ///< Legacy name for 16 KiB locality latency from each loop
  std::vector<double> all_tlb_miss_latency_ns; ///< Legacy name for global-random latency from each loop
  std::vector<double> all_page_walk_penalty_ns; ///< Legacy name for paired global-random minus 16 KiB locality deltas
  std::vector<double> all_l1_read_bw_gb_s;    ///< L1 read bandwidth from each loop (GB/s)
  std::vector<double> all_l1_write_bw_gb_s;   ///< L1 write bandwidth from each loop (GB/s)
  std::vector<double> all_l1_copy_bw_gb_s;   ///< L1 copy bandwidth from each loop (GB/s)
  std::vector<double> all_l2_read_bw_gb_s;   ///< L2 read bandwidth from each loop (GB/s)
  std::vector<double> all_l2_write_bw_gb_s;  ///< L2 write bandwidth from each loop (GB/s)
  std::vector<double> all_l2_copy_bw_gb_s;   ///< L2 copy bandwidth from each loop (GB/s)
  std::vector<double> all_custom_latency_ns;  ///< Custom cache latency from each loop (nanoseconds)
  std::vector<double> all_custom_read_bw_gb_s;   ///< Custom cache read bandwidth from each loop (GB/s)
  std::vector<double> all_custom_write_bw_gb_s;  ///< Custom cache write bandwidth from each loop (GB/s)
  std::vector<double> all_custom_copy_bw_gb_s;  ///< Custom cache copy bandwidth from each loop (GB/s)
  
  // Full sample distributions for percentile calculations
  std::vector<double> all_main_mem_latency_samples;  ///< Concatenated latency samples from all loops (nanoseconds)
  std::vector<double> all_l1_latency_samples;  ///< Concatenated L1 latency samples from all loops (nanoseconds)
  std::vector<double> all_l2_latency_samples;  ///< Concatenated L2 latency samples from all loops (nanoseconds)
  std::vector<double> all_custom_latency_samples;  ///< Concatenated custom cache latency samples from all loops (nanoseconds)
};

/**
 * @brief Optional dependency seams for deterministic coordinator failure tests.
 *
 * Production callers leave this null. Tests can fail timer/checkpoint creation,
 * inject a kernel-free loop result, request a stop between loops, or throw from
 * a dependency callback to verify the coordinator exception boundary. They can
 * also observe whether a session-backed checkpoint actually requests lazy JSON
 * payload construction.
 */
struct BenchmarkRunnerTestHooks {
  bool force_timer_creation_failure = false;
  std::function<bool()> stop_requested;
  std::function<double()> elapsed_seconds;
  std::function<void()> checkpoint_payload_build;
  std::function<BenchmarkResults(BenchmarkConfig&, int, HighResTimer&,
                                 BenchmarkExecutionState*)>
      execute_loop;
  std::function<int(const BenchmarkConfig&, const BenchmarkStatistics&, double,
                    bool)>
      checkpoint;
};

/**
 * @brief Decide whether the command boundary may write standard final JSON.
 *
 * Stdout owns one terminal snapshot even when the runner returns failure and
 * retained evidence is available. A file runner failure may represent a failed
 * checkpoint and therefore forbids an outer retry. Disabled output never
 * writes.
 *
 * @param output_kind Classified command output transport.
 * @param runner_status Return code from `run_all_benchmarks`.
 * @return `true` only when the command boundary owns a final write.
 */
bool should_write_standard_final_json(JsonOutputKind output_kind,
                                      int runner_status) noexcept;

/**
 * @brief Run configured standard benchmark loops and retain their evidence.
 *
 * Reinitializes @p stats, executes up to `config.loop_count` loops, aggregates
 * completed measurements, and offers each result transition to the configured
 * checkpoint transport. A stop observed between loops is a graceful
 * interruption only when its logical checkpoint succeeds. Checkpoint failure
 * takes precedence, returns failure, and leaves all earlier loop evidence and
 * counters available in @p stats.
 *
 * @param[in,out] config Benchmark configuration used by loop execution and JSON
 *        construction. Per-phase buffers are owned below this API.
 * @param[out] stats Reinitialized statistics and retained per-loop evidence.
 * @param[in] test_hooks Optional synchronous deterministic dependency seams;
 *        null in production and never retained beyond the call.
 * @param[in] output_session Optional command-owned JSON transport. File targets
 *        persist checkpoints through the shared atomic writer; stdout targets
 *        perform lazy successful checkpoint no-ops. The session is not retained.
 * @return `EXIT_SUCCESS` after complete execution or graceful interruption whose
 *         checkpoint succeeded. Inspect `stats.status` for completeness.
 * @return `EXIT_FAILURE` after timer, loop, coordinator, or checkpoint failure;
 *         representable evidence remains in @p stats.
 * @note Exceptions from execution and injected dependencies are converted to a
 *       failed status and do not propagate across this coordinator boundary.
 * @warning When @p output_session is null, a non-empty `config.output_file` is
 *          handled by the legacy file-only checkpoint adapter and must not be
 *          the exact stdout sentinel `-`. Command code must classify targets and
 *          pass a session.
 */
int run_all_benchmarks(BenchmarkConfig& config, BenchmarkStatistics& stats,
                       const BenchmarkRunnerTestHooks* test_hooks = nullptr,
                       JsonOutputSession* output_session = nullptr);

#endif // BENCHMARK_RUNNER_H
