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

/**
 * @file benchmark_executor.cpp
 * @brief Single benchmark loop execution
 *
 * Implements the execution logic for a single benchmark loop, coordinating all test
 * types (bandwidth and latency for main memory and cache). Handles warmup operations,
 * test execution, and result calculation.
 *
 * Key features:
 * - Modular test execution (main memory, cache bandwidth, cache latency)
 * - Automatic warmup before each test to stabilize caches
 * - Conditional execution based on configuration flags
 * - Progress indication for user feedback
 * - Exception handling with re-throw for caller handling
 *
 * Test execution order:
 * 1. Main memory bandwidth (read, write, copy)
 * 2. Cache bandwidth (L1/L2 or custom)
 * 3. Cache latency tests
 * 4. Main memory latency test
 *
 * Conditional execution:
 * - only_bandwidth: Skip latency tests
 * - only_latency: Skip bandwidth tests
 * - use_custom_cache_size: Use custom cache instead of L1/L2
 */
 
#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_work_plan.h"
#include "benchmark/parallel_test_framework.h"
#include "core/memory/buffer_manager.h"  // BenchmarkBuffers
#include "core/config/config.h"           // BenchmarkConfig
#include "core/memory/memory_manager.h"   // allocate_buffer, allocate_buffer_non_cacheable
#include "core/memory/memory_utils.h"     // initialize_buffers, setup_latency_chain
#include "utils/benchmark.h"        // All benchmark functions and print functions
#include "benchmark/benchmark_runner.h" // BenchmarkResults
#include "output/console/messages/messages_api.h"             // Centralized messages
#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kAutoTlbComparisonRepeats = 3;
constexpr uint64_t kSeedDomainMainLatency = 0x1001;
constexpr uint64_t kSeedDomainL1Latency = 0x1002;
constexpr uint64_t kSeedDomainL2Latency = 0x1003;
constexpr uint64_t kSeedDomainCustomLatency = 0x1004;
constexpr uint64_t kSeedDomainLocality16k = 0x2001;
constexpr uint64_t kSeedDomainGlobalRandom = 0x2002;

std::vector<BenchmarkMeasurement*> expected_measurements(BenchmarkResults& results,
                                                         const BenchmarkConfig& config) {
  std::vector<BenchmarkMeasurement*> measurements;
  if (!config.only_latency) {
    if (config.buffer_size > 0) {
      measurements.insert(measurements.end(),
                          {&results.main_read_bandwidth,
                           &results.main_write_bandwidth,
                           &results.main_copy_bandwidth});
    }
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        measurements.insert(measurements.end(),
                            {&results.custom_read_bandwidth,
                             &results.custom_write_bandwidth,
                             &results.custom_copy_bandwidth});
      }
    } else {
      if (config.l1_buffer_size > 0) {
        measurements.insert(measurements.end(),
                            {&results.l1_read_bandwidth,
                             &results.l1_write_bandwidth,
                             &results.l1_copy_bandwidth});
      }
      if (config.l2_buffer_size > 0) {
        measurements.insert(measurements.end(),
                            {&results.l2_read_bandwidth,
                             &results.l2_write_bandwidth,
                             &results.l2_copy_bandwidth});
      }
    }
  }

  if (!config.only_bandwidth) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        measurements.push_back(&results.custom_latency);
      }
    } else {
      if (config.l1_buffer_size > 0) {
        measurements.push_back(&results.l1_latency);
      }
      if (config.l2_buffer_size > 0) {
        measurements.push_back(&results.l2_latency);
      }
    }
    if (config.buffer_size > 0 && config.lat_num_accesses > 0) {
      measurements.push_back(&results.main_latency);
      if (!config.user_specified_latency_tlb_locality) {
        measurements.insert(measurements.end(),
                            {&results.locality_16k_latency,
                             &results.global_random_latency,
                             &results.locality_latency_delta});
      }
    }
  }
  return measurements;
}

void finalize_loop_results(BenchmarkResults& results,
                           const BenchmarkConfig& config,
                           bool interrupted) {
  std::vector<BenchmarkMeasurement*> measurements = expected_measurements(results, config);
  results.planned_measurements = measurements.size();
  results.completed_measurements = 0;
  for (BenchmarkMeasurement* measurement : measurements) {
    if (interrupted && measurement->status == BenchmarkMeasurementStatus::NotRun) {
      set_measurement_unavailable(*measurement,
                                  BenchmarkMeasurementStatus::Interrupted,
                                  Messages::benchmark_reason_interrupted_before_measurement());
    }
    if (measurement->is_measured()) {
      ++results.completed_measurements;
    }
  }

  if (interrupted) {
    results.status = BenchmarkRunStatus::Interrupted;
    results.status_reason = Messages::benchmark_reason_interrupted_by_user();
  } else if (results.completed_measurements == results.planned_measurements) {
    results.status = BenchmarkRunStatus::Complete;
    results.status_reason.clear();
  } else {
    results.status = BenchmarkRunStatus::Partial;
    results.status_reason =
        Messages::benchmark_reason_planned_measurements_unavailable();
  }
}

/**
 * @brief Allocates a phase-local buffer using the active cacheability policy.
 *
 * Standard benchmark execution allocates buffers immediately before a phase and
 * releases them when the local `BenchmarkBuffers` owner goes out of scope.
 * This helper centralizes the mode switch between regular mappings and the
 * best-effort cache-discouraging allocation path.
 */
MmapPtr allocate_phase_buffer(const BenchmarkConfig& config, size_t size, const char* buffer_name) {
  if (config.use_non_cacheable) {
    return allocate_buffer_non_cacheable(size, buffer_name);
  }
  return allocate_buffer(size, buffer_name);
}

/**
 * @brief Prepares main-memory bandwidth buffers for one phase.
 *
 * Allocates source and destination buffers and initializes deterministic data
 * before timing starts. Copy/read/write kernels depend on both buffers being
 * present at the same time, so this phase intentionally uses a 2x main-buffer
 * footprint.
 */
int prepare_main_memory_bandwidth_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.buffer_size == 0) {
    return EXIT_SUCCESS;
  }

  buffers.src_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "src_buffer");
  if (!buffers.src_buffer_ptr) {
    return EXIT_FAILURE;
  }

  buffers.dst_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "dst_buffer");
  if (!buffers.dst_buffer_ptr) {
    return EXIT_FAILURE;
  }

  return initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(), config.buffer_size);
}

/**
 * @brief Prepares cache-bandwidth buffers for one phase.
 *
 * In custom-cache mode this prepares one src/dst pair. In auto-cache mode this
 * prepares L1 and L2 src/dst pairs. All initialization happens before measured
 * cache bandwidth kernels run.
 */
int prepare_cache_bandwidth_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size == 0) {
      return EXIT_SUCCESS;
    }

    buffers.custom_bw_src_ptr =
        allocate_phase_buffer(config, config.custom_buffer_size, "custom_bw_src_buffer");
    if (!buffers.custom_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.custom_bw_dst_ptr =
        allocate_phase_buffer(config, config.custom_buffer_size, "custom_bw_dst_buffer");
    if (!buffers.custom_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    return initialize_buffers(buffers.custom_bw_src(), buffers.custom_bw_dst(),
                              config.custom_buffer_size);
  }

  if (config.l1_buffer_size > 0) {
    buffers.l1_bw_src_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_bw_src_buffer");
    if (!buffers.l1_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.l1_bw_dst_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_bw_dst_buffer");
    if (!buffers.l1_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    if (initialize_buffers(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size) !=
        EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  if (config.l2_buffer_size > 0) {
    buffers.l2_bw_src_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_bw_src_buffer");
    if (!buffers.l2_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.l2_bw_dst_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_bw_dst_buffer");
    if (!buffers.l2_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    if (initialize_buffers(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size) !=
        EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Prepares cache-latency buffers and pointer chains for one phase.
 *
 * Builds latency chains using current stride/locality settings.
 */
int prepare_cache_latency_buffers(BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size == 0) {
      return EXIT_SUCCESS;
    }

    buffers.custom_buffer_ptr = allocate_phase_buffer(config, config.custom_buffer_size, "custom_buffer");
    if (!buffers.custom_buffer_ptr) {
      return EXIT_FAILURE;
    }

    return setup_latency_chain(buffers.custom_buffer(),
                               config.custom_buffer_size,
                               config.latency_stride_bytes,
                               config.latency_tlb_locality_bytes,
                               nullptr,
                               config.latency_chain_mode,
                               derive_benchmark_seed(config.benchmark_seed,
                                                     kSeedDomainCustomLatency));
  }

  if (config.l1_buffer_size > 0) {
    buffers.l1_buffer_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_buffer");
    if (!buffers.l1_buffer_ptr) {
      return EXIT_FAILURE;
    }

    if (setup_latency_chain(buffers.l1_buffer(),
                            config.l1_buffer_size,
                            config.latency_stride_bytes,
                            config.latency_tlb_locality_bytes,
                            nullptr,
                            config.latency_chain_mode,
                            derive_benchmark_seed(config.benchmark_seed,
                                                  kSeedDomainL1Latency)) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  if (config.l2_buffer_size > 0) {
    buffers.l2_buffer_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_buffer");
    if (!buffers.l2_buffer_ptr) {
      return EXIT_FAILURE;
    }

    if (setup_latency_chain(buffers.l2_buffer(),
                            config.l2_buffer_size,
                            config.latency_stride_bytes,
                            config.latency_tlb_locality_bytes,
                            nullptr,
                            config.latency_chain_mode,
                            derive_benchmark_seed(config.benchmark_seed,
                                                  kSeedDomainL2Latency)) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Prepares the main-memory latency buffer and pointer chain.
 *
 * This path is skipped when main latency is disabled (zero main buffer or zero
 * configured accesses). Chain setup is completed before timing starts.
 */
int prepare_main_memory_latency_buffer(BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.buffer_size == 0 || config.lat_num_accesses == 0) {
    return EXIT_SUCCESS;
  }

  buffers.lat_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "lat_buffer");
  if (!buffers.lat_buffer_ptr) {
    return EXIT_FAILURE;
  }

  return setup_latency_chain(buffers.lat_buffer(),
                             config.buffer_size,
                             config.latency_stride_bytes,
                             config.latency_tlb_locality_bytes,
                             nullptr,
                             config.latency_chain_mode,
                             derive_benchmark_seed(config.benchmark_seed,
                                                   kSeedDomainMainLatency));
}

/**
 * @brief Returns the median value from a small latency sample set.
 *
 * Used for the automatic paired locality comparison where one inflated pass can
 * otherwise dominate a three-round point estimate. The input is intentionally
 * copied because the caller owns a very small vector.
 */
double median_latency(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

/**
 * @brief Measure one auto-TLB comparison point with a small P50 noise guard.
 *
 * Each repeat rebuilds the chain, warms it, and performs one continuous
 * pointer-chase timing pass. Taking P50 across complete passes rejects a single
 * IRQ-inflated outlier without deriving the result from segmented sample
 * windows.
 */
void run_paired_locality_comparison(void* buffer,
                                    size_t buffer_size,
                                    size_t stride_bytes,
                                    size_t num_accesses,
                                    uint64_t base_seed,
                                    bool main_thread_qos_requested,
                                    bool main_thread_qos_applied,
                                    HighResTimer& timer,
                                    BenchmarkResults& results,
                                    size_t phase_order_index) {
  constexpr size_t kLocality16kBytes = 16 * Constants::BYTES_PER_KB;
  auto& locality = results.locality_16k_latency;
  auto& global = results.global_random_latency;
  auto& delta = results.locality_latency_delta;
  for (BenchmarkMeasurement* measurement : {&locality, &global, &delta}) {
    measurement->target = "main-memory";
    measurement->operation = "locality-comparison";
    measurement->buffer_size_bytes = buffer_size;
    measurement->access_count = num_accesses;
    measurement->requested_threads = 1;
    measurement->effective_threads = 1;
    measurement->created_workers = 1;
    measurement->qos_successful_workers = main_thread_qos_applied ? 1 : 0;
    measurement->qos_failed_workers =
        main_thread_qos_requested && !main_thread_qos_applied ? 1 : 0;
    measurement->qos_outcome = main_thread_qos_applied
                                   ? "main-thread-qos-applied"
                               : main_thread_qos_requested
                                   ? "main-thread-qos-failed"
                                   : "main-thread-qos-not-requested";
    measurement->phase_order_index = phase_order_index;
    measurement->work_policy = "paired-alternating-rounds";
  }
  locality.seed = derive_benchmark_seed(base_seed, kSeedDomainLocality16k);
  global.seed = derive_benchmark_seed(base_seed, kSeedDomainGlobalRandom);

  if (buffer == nullptr || buffer_size == 0 || stride_bytes == 0 ||
      num_accesses == 0) {
    for (BenchmarkMeasurement* measurement : {&locality, &global, &delta}) {
      set_measurement_unavailable(*measurement,
                                  BenchmarkMeasurementStatus::Invalid,
                                  Messages::benchmark_reason_invalid_locality_work());
    }
    return;
  }

  double total_elapsed_seconds = 0.0;
  auto measure_layout = [&](bool use_locality, int round) -> bool {
    BenchmarkMeasurement& measurement = use_locality ? locality : global;
    const uint64_t domain = use_locality ? kSeedDomainLocality16k
                                         : kSeedDomainGlobalRandom;
    const uint64_t seed = derive_benchmark_seed(
        base_seed, domain ^ static_cast<uint64_t>(round + 1));
    const size_t locality_bytes = use_locality ? kLocality16kBytes : 0;
    const LatencyChainMode mode = use_locality
                                      ? LatencyChainMode::RandomInBoxRandomBox
                                      : LatencyChainMode::GlobalRandom;
    if (setup_latency_chain(buffer, buffer_size, stride_bytes, locality_bytes,
                            nullptr, mode, seed) != EXIT_SUCCESS) {
      return false;
    }
    show_progress();
    warmup_latency(buffer, buffer_size);
    const double elapsed_ns =
        run_latency_test(buffer, num_accesses, timer, nullptr, 0);
    if (signal_received() || !benchmark_elapsed_is_valid(elapsed_ns)) {
      return false;
    }
    measurement.samples.push_back(
        elapsed_ns / static_cast<double>(num_accesses));
    measurement.sample_seeds.push_back(seed);
    total_elapsed_seconds += elapsed_ns / Constants::NANOSECONDS_PER_SECOND;
    return true;
  };

  for (int round = 0; round < kAutoTlbComparisonRepeats; ++round) {
    const bool locality_first = (round % 2) == 0;
    if (!measure_layout(locality_first, round) ||
        !measure_layout(!locality_first, round)) {
      const BenchmarkMeasurementStatus status =
          signal_received() ? BenchmarkMeasurementStatus::Interrupted
                            : BenchmarkMeasurementStatus::Invalid;
      for (BenchmarkMeasurement* measurement : {&locality, &global, &delta}) {
        set_measurement_unavailable(*measurement, status,
                                    Messages::benchmark_reason_locality_comparison_unavailable());
      }
      return;
    }
    delta.samples.push_back(global.samples.back() - locality.samples.back());
  }

  set_measurement_value(locality, median_latency(locality.samples),
                        total_elapsed_seconds * 0.5);
  set_measurement_value(global, median_latency(global.samples),
                        total_elapsed_seconds * 0.5);
  set_measurement_value(delta, median_latency(delta.samples),
                        total_elapsed_seconds);
}

void warmup_bandwidth_operation(void* src_buffer,
                                void* dst_buffer,
                                const BenchmarkWorkPlan& plan) {
  const bool cache_target = plan.target != BenchmarkTarget::MainMemory;
  switch (plan.operation) {
    case BenchmarkOperation::Read: {
      std::atomic<uint64_t> checksum{0};
      if (cache_target) {
        warmup_cache_read(src_buffer, plan.buffer_size_bytes,
                          plan.effective_threads, checksum);
      } else {
        warmup_read(src_buffer, plan.buffer_size_bytes,
                    plan.effective_threads, checksum);
      }
      break;
    }
    case BenchmarkOperation::Write:
      if (cache_target) {
        warmup_cache_write(dst_buffer, plan.buffer_size_bytes,
                           plan.effective_threads);
      } else {
        warmup_write(dst_buffer, plan.buffer_size_bytes,
                     plan.effective_threads);
      }
      break;
    case BenchmarkOperation::Copy:
      if (cache_target) {
        warmup_cache_copy(dst_buffer, src_buffer, plan.buffer_size_bytes,
                          plan.effective_threads);
      } else {
        warmup_copy(dst_buffer, src_buffer, plan.buffer_size_bytes,
                    plan.effective_threads);
      }
      break;
    case BenchmarkOperation::Latency:
      break;
  }
}

double execute_bandwidth_plan(void* src_buffer,
                              void* dst_buffer,
                              const BenchmarkWorkPlan& plan,
                              HighResTimer& timer,
                              ParallelExecutionMetadata* execution_metadata) {
  const bool cache_target = plan.target != BenchmarkTarget::MainMemory;
  switch (plan.operation) {
    case BenchmarkOperation::Read: {
      uint64_t checksum = 0;
      return run_read_test_with_plan(
          src_buffer, plan, checksum, timer,
          cache_target ? memory_read_cache_loop_asm : memory_read_loop_asm,
          execution_metadata);
    }
    case BenchmarkOperation::Write:
      return run_write_test_with_plan(
          dst_buffer, plan, timer,
          cache_target ? memory_write_cache_loop_asm : memory_write_loop_asm,
          execution_metadata);
    case BenchmarkOperation::Copy:
      return run_copy_test_with_plan(
          dst_buffer, src_buffer, plan, timer,
          cache_target ? memory_copy_cache_loop_asm : memory_copy_loop_asm,
          execution_metadata);
    case BenchmarkOperation::Latency:
      return 0.0;
  }
  return 0.0;
}

void populate_parallel_execution_metadata(
    BenchmarkMeasurement& measurement,
    const ParallelExecutionMetadata& execution_metadata) {
  measurement.qos_successful_workers =
      execution_metadata.qos_successful_workers;
  measurement.qos_failed_workers = execution_metadata.qos_failed_workers;
  measurement.created_workers = execution_metadata.created_workers;
  measurement.worker_startup_failed =
      execution_metadata.worker_startup_failed;
  if (execution_metadata.worker_startup_failed) {
    measurement.qos_outcome = "worker-startup-failed";
  } else if (execution_metadata.qos_failed_workers == 0 &&
             execution_metadata.qos_successful_workers > 0) {
    measurement.qos_outcome = "applied-to-all-workers";
  } else if (execution_metadata.qos_successful_workers > 0) {
    measurement.qos_outcome = "partially-applied";
  } else {
    measurement.qos_outcome = "failed-for-all-workers";
  }
}

void populate_bandwidth_metadata(BenchmarkMeasurement& measurement,
                                 const BenchmarkBandwidthExecutionState& state,
                                 size_t phase_order_index,
                                 size_t operation_order_index) {
  const BenchmarkWorkPlan& plan = state.plan;
  measurement.target = benchmark_target_to_string(plan.target);
  measurement.operation = benchmark_operation_to_string(plan.operation);
  measurement.buffer_size_bytes = plan.buffer_size_bytes;
  measurement.passes = plan.passes;
  measurement.exact_payload_bytes = plan.total_payload_bytes;
  measurement.requested_threads = plan.requested_threads;
  measurement.effective_threads = plan.effective_threads;
  measurement.pilot_elapsed_seconds = state.pilot_elapsed_seconds;
  measurement.calibration_corrections = state.calibration_corrections;
  measurement.duration_quality = state.duration_quality;
  measurement.phase_order_index = phase_order_index;
  measurement.operation_order_index = operation_order_index;
}

void run_calibrated_bandwidth_measurement(
    void* src_buffer, void* dst_buffer, size_t buffer_size, int requested_threads,
    BenchmarkTarget target, BenchmarkOperation operation,
    bool explicit_iterations, size_t explicit_passes,
    BenchmarkBandwidthExecutionState& state, BenchmarkMeasurement& measurement,
    HighResTimer& timer, size_t phase_order_index, size_t operation_order_index) {
  const bool first_execution = !state.initialized;
  measurement.automatic_calibration = !explicit_iterations;
  measurement.work_policy = explicit_iterations ? "explicit-iterations"
                                                : "automatic-duration-calibration";

  if (first_execution) {
    size_t initial_passes = explicit_passes;
    if (!explicit_iterations) {
      const size_t bytes_per_pass = operation == BenchmarkOperation::Copy
                                        ? buffer_size * Constants::COPY_OPERATION_MULTIPLIER
                                        : buffer_size;
      initial_passes = calculate_benchmark_pilot_passes(
          bytes_per_pass, Constants::BENCHMARK_CALIBRATION_MIN_PILOT_BYTES,
          Constants::BENCHMARK_CALIBRATION_MAX_PASSES);
    }
    BenchmarkWorkPlan initial_plan = build_benchmark_bandwidth_work_plan(
        buffer_size, requested_threads, initial_passes, target, operation);
    if (initial_plan.status != BenchmarkMeasurementStatus::Measured) {
      set_measurement_unavailable(measurement, initial_plan.status,
                                  initial_plan.status_reason);
      return;
    }

    if (!explicit_iterations) {
      warmup_bandwidth_operation(src_buffer, dst_buffer, initial_plan);
      ParallelExecutionMetadata pilot_metadata;
      state.pilot_elapsed_seconds =
          execute_bandwidth_plan(src_buffer, dst_buffer, initial_plan, timer,
                                 &pilot_metadata);
      if (signal_received()) {
        set_measurement_unavailable(measurement,
                                    BenchmarkMeasurementStatus::Interrupted,
                                    Messages::benchmark_reason_interrupted_calibration_pilot());
        return;
      }
      const size_t calibrated_passes = calculate_benchmark_calibrated_count(
          state.pilot_elapsed_seconds, initial_plan.passes,
          Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS, 1,
          Constants::BENCHMARK_CALIBRATION_MAX_PASSES);
      if (calibrated_passes == 0) {
        set_measurement_unavailable(measurement,
                                    BenchmarkMeasurementStatus::Invalid,
                                    Messages::benchmark_reason_invalid_calibration_pilot());
        return;
      }
      state.plan = build_benchmark_bandwidth_work_plan(
          buffer_size, requested_threads, calibrated_passes, target, operation);
    } else {
      state.plan = std::move(initial_plan);
      state.duration_quality = "explicit-work-policy";
    }
    if (state.plan.status != BenchmarkMeasurementStatus::Measured) {
      set_measurement_unavailable(measurement, state.plan.status,
                                  state.plan.status_reason);
      return;
    }
  }

  double elapsed_seconds = 0.0;
  ParallelExecutionMetadata execution_metadata;
  for (size_t attempt = 0;; ++attempt) {
    show_progress();
    warmup_bandwidth_operation(src_buffer, dst_buffer, state.plan);
    elapsed_seconds = execute_bandwidth_plan(src_buffer, dst_buffer, state.plan,
                                             timer, &execution_metadata);
    if (signal_received()) {
      populate_bandwidth_metadata(measurement, state, phase_order_index,
                                  operation_order_index);
      populate_parallel_execution_metadata(measurement, execution_metadata);
      set_measurement_unavailable(measurement,
                                  BenchmarkMeasurementStatus::Interrupted,
                                  Messages::benchmark_reason_interrupted_measured_operation());
      return;
    }
    if (explicit_iterations || !first_execution ||
        benchmark_duration_in_window(
            elapsed_seconds, Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
            Constants::BENCHMARK_CALIBRATION_MAX_SECONDS) ||
        (state.plan.passes == 1 &&
         elapsed_seconds > Constants::BENCHMARK_CALIBRATION_MAX_SECONDS) ||
        attempt >= Constants::BENCHMARK_CALIBRATION_MAX_CORRECTIONS) {
      break;
    }
    const size_t corrected_passes = calculate_benchmark_calibrated_count(
        elapsed_seconds, state.plan.passes,
        Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS, 1,
        Constants::BENCHMARK_CALIBRATION_MAX_PASSES);
    if (corrected_passes == 0 || corrected_passes == state.plan.passes ||
        !set_benchmark_work_plan_passes(state.plan, corrected_passes)) {
      break;
    }
    ++state.calibration_corrections;
  }

  state.duration_quality = explicit_iterations
                               ? "explicit-work-policy"
                               : classify_benchmark_duration_quality(
                                     elapsed_seconds, state.plan.passes,
                                     Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
                                     Constants::BENCHMARK_CALIBRATION_MAX_SECONDS);
  state.initialized = true;
  populate_bandwidth_metadata(measurement, state, phase_order_index,
                              operation_order_index);
  populate_parallel_execution_metadata(measurement, execution_metadata);
  measurement.duration_within_target =
      explicit_iterations ||
      benchmark_duration_in_window(
          elapsed_seconds, Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
          Constants::BENCHMARK_CALIBRATION_MAX_SECONDS);
  if (!benchmark_elapsed_is_valid(elapsed_seconds)) {
    set_measurement_unavailable(measurement,
                                BenchmarkMeasurementStatus::Invalid,
                                Messages::benchmark_reason_invalid_bandwidth_duration());
    return;
  }
  const double bandwidth_gb_s =
      static_cast<double>(state.plan.total_payload_bytes) /
      elapsed_seconds / Constants::NANOSECONDS_PER_SECOND;
  if (!benchmark_elapsed_is_valid(bandwidth_gb_s)) {
    set_measurement_unavailable(measurement,
                                BenchmarkMeasurementStatus::Invalid,
                                Messages::benchmark_reason_invalid_bandwidth_value());
    return;
  }
  set_measurement_value(measurement, bandwidth_gb_s, elapsed_seconds);
}

void populate_latency_metadata(BenchmarkMeasurement& measurement,
                               const BenchmarkLatencyExecutionState& state,
                               size_t phase_order_index) {
  const BenchmarkLatencyWorkPlan& plan = state.plan;
  measurement.target = benchmark_target_to_string(plan.target);
  measurement.operation = "latency";
  measurement.buffer_size_bytes = plan.buffer_size_bytes;
  measurement.access_count = plan.access_count;
  measurement.chain_node_count = plan.chain_node_count;
  measurement.complete_chain_cycles = plan.complete_chain_cycles;
  measurement.seed = plan.seed;
  measurement.requested_threads = 1;
  measurement.effective_threads = 1;
  measurement.automatic_calibration = true;
  measurement.work_policy = "automatic-duration-calibration";
  measurement.pilot_elapsed_seconds = state.pilot_elapsed_seconds;
  measurement.calibration_corrections = state.calibration_corrections;
  measurement.duration_quality = state.duration_quality;
  measurement.phase_order_index = phase_order_index;
}

void run_calibrated_latency_measurement(
    void* buffer, size_t buffer_size, size_t stride_bytes,
    size_t fallback_access_count, BenchmarkTarget target, uint64_t seed,
    BenchmarkLatencyExecutionState& state, BenchmarkMeasurement& measurement,
    HighResTimer& timer, int sample_count, size_t phase_order_index) {
  const bool first_execution = !state.initialized;
  const size_t node_count = stride_bytes == 0 ? 0 : buffer_size / stride_bytes;
  if (first_execution) {
    BenchmarkLatencyWorkPlan pilot_plan = build_benchmark_latency_work_plan(
        buffer_size, stride_bytes, std::max<size_t>(node_count, 1), 1,
        Constants::BENCHMARK_LATENCY_MAX_ACCESSES, target, seed);
    if (pilot_plan.status != BenchmarkMeasurementStatus::Measured) {
      set_measurement_unavailable(measurement, pilot_plan.status,
                                  pilot_plan.status_reason);
      return;
    }
    warmup_latency(buffer, buffer_size);
    const double pilot_ns = run_latency_test(
        buffer, pilot_plan.access_count, timer, nullptr, 0);
    state.pilot_elapsed_seconds =
        pilot_ns / Constants::NANOSECONDS_PER_SECOND;
    if (signal_received()) {
      set_measurement_unavailable(measurement,
                                  BenchmarkMeasurementStatus::Interrupted,
                                  Messages::benchmark_reason_interrupted_latency_pilot());
      return;
    }
    const size_t minimum_accesses =
        node_count <= Constants::BENCHMARK_LATENCY_MAX_ACCESSES /
                          Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES
            ? node_count * Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES
            : 0;
    const size_t calibrated_accesses = calculate_benchmark_calibrated_count(
        state.pilot_elapsed_seconds, pilot_plan.access_count,
        Constants::BENCHMARK_LATENCY_TARGET_SECONDS,
        std::max<size_t>(minimum_accesses, 1),
        Constants::BENCHMARK_LATENCY_MAX_ACCESSES,
        std::max<size_t>(node_count, 1));
    const size_t desired_accesses = calibrated_accesses != 0
                                        ? calibrated_accesses
                                        : fallback_access_count;
    state.plan = build_benchmark_latency_work_plan(
        buffer_size, stride_bytes, desired_accesses,
        Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES,
        Constants::BENCHMARK_LATENCY_MAX_ACCESSES, target, seed);
    if (state.plan.status != BenchmarkMeasurementStatus::Measured) {
      set_measurement_unavailable(measurement, state.plan.status,
                                  state.plan.status_reason);
      return;
    }
  }

  double elapsed_ns = 0.0;
  for (size_t attempt = 0;; ++attempt) {
    show_progress();
    warmup_latency(buffer, buffer_size);
    elapsed_ns = run_latency_test(buffer, state.plan.access_count, timer,
                                  nullptr, 0);
    const double elapsed_seconds = elapsed_ns / Constants::NANOSECONDS_PER_SECOND;
    if (signal_received()) {
      populate_latency_metadata(measurement, state, phase_order_index);
      set_measurement_unavailable(measurement,
                                  BenchmarkMeasurementStatus::Interrupted,
                                  Messages::benchmark_reason_interrupted_latency_measurement());
      return;
    }
    if (!first_execution ||
        benchmark_duration_in_window(
            elapsed_seconds,
            Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS,
            Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS) ||
        attempt >= Constants::BENCHMARK_CALIBRATION_MAX_CORRECTIONS) {
      break;
    }
    const size_t corrected_accesses = calculate_benchmark_calibrated_count(
        elapsed_seconds, state.plan.access_count,
        Constants::BENCHMARK_LATENCY_TARGET_SECONDS,
        state.plan.chain_node_count *
            Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES,
        Constants::BENCHMARK_LATENCY_MAX_ACCESSES,
        state.plan.chain_node_count);
    BenchmarkLatencyWorkPlan corrected_plan = build_benchmark_latency_work_plan(
        buffer_size, stride_bytes, corrected_accesses,
        Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES,
        Constants::BENCHMARK_LATENCY_MAX_ACCESSES, target, seed);
    if (corrected_plan.status != BenchmarkMeasurementStatus::Measured ||
        corrected_plan.access_count == state.plan.access_count) {
      break;
    }
    state.plan = std::move(corrected_plan);
    ++state.calibration_corrections;
  }

  const double elapsed_seconds = elapsed_ns / Constants::NANOSECONDS_PER_SECOND;
  const size_t minimum_accesses =
      state.plan.chain_node_count *
      Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES;
  const bool minimum_cycles_limit_duration =
      state.plan.access_count == minimum_accesses &&
      elapsed_seconds > Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS;
  state.duration_quality = classify_benchmark_duration_quality(
      elapsed_seconds, state.plan.complete_chain_cycles,
      Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS,
      Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS,
      minimum_cycles_limit_duration);
  state.initialized = true;
  populate_latency_metadata(measurement, state, phase_order_index);
  measurement.duration_within_target = benchmark_duration_in_window(
      elapsed_seconds, Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS,
      Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS);
  if (!benchmark_elapsed_is_valid(elapsed_ns) ||
      state.plan.access_count == 0) {
    set_measurement_unavailable(measurement,
                                BenchmarkMeasurementStatus::Invalid,
                                Messages::benchmark_reason_invalid_latency_measurement());
    return;
  }
  set_measurement_value(measurement,
                        elapsed_ns / static_cast<double>(state.plan.access_count),
                        elapsed_seconds);
  if (sample_count > 0) {
    (void)run_latency_test(buffer, state.plan.access_count, timer,
                           &measurement.samples, sample_count);
  }
}

}  // namespace

/**
 * @brief Run a single benchmark loop and return results.
 *
 * Orchestrates execution of all configured tests for one benchmark loop:
 * 1. Executes tests based on configuration flags
 * 2. Calculates bandwidth results from timing data
 * 3. Calculates main memory latency if applicable
 * 4. Returns complete results for the loop
 *
 * Conditional execution modes:
 * - only_bandwidth: Execute only bandwidth tests (main + cache)
 * - only_latency: Execute only latency tests (cache + main)
 * - Default: Execute all tests
 *
 * Exception handling:
 * - Catches std::exception during test execution
 * - Logs error message to stderr
 * - Re-throws exception for caller to handle
 *
 * @param[in]     config      Benchmark configuration (sizes, counts, flags)
 * @param[in]     loop        Zero-based loop index used for cyclic phase/operation order and diagnostics
 * @param[in,out] test_timer  High-resolution timer for measurements
 * @param[in,out] execution_state Optional cross-loop calibration state; a local state is used when null
 * @param[in]     test_hooks  Optional deterministic failure seams used by tests
 *
 * @return BenchmarkResults structure with all calculated results for this loop
 *
 * @throws std::exception Re-thrown from test execution failures
 *
 * @note Results include both raw timing data and calculated bandwidth values.
 * @note Main memory latency is calculated as total_time / num_accesses.
 * @note Phase-local `BenchmarkBuffers` objects free mappings on scope exit (RAII).
 *
 * @see run_calibrated_bandwidth_measurement() for bandwidth execution
 * @see run_calibrated_latency_measurement() for latency execution
 */
BenchmarkResults run_single_benchmark_loop(BenchmarkConfig& config,
                                           int loop,
                                           HighResTimer& test_timer,
                                           BenchmarkExecutionState* execution_state,
                                           const BenchmarkExecutorTestHooks* test_hooks) {
  BenchmarkResults results;
  BenchmarkExecutionState local_execution_state;
  BenchmarkExecutionState& state = execution_state != nullptr
                                       ? *execution_state
                                       : local_execution_state;

  results.status = BenchmarkRunStatus::Partial;
  results.loop_index = loop >= 0 ? static_cast<size_t>(loop) : 0;
  results.operation_order_index = results.loop_index % 3;

  enum class Phase {
    MainBandwidth,
    CacheBandwidth,
    CacheLatency,
    MainLatency,
  };
  struct EnabledPhase {
    Phase phase;
    const char* name;
  };
  std::vector<EnabledPhase> enabled_phases;
  if (!config.only_latency && config.buffer_size > 0) {
    enabled_phases.push_back({Phase::MainBandwidth, "main-bandwidth"});
  }
  const bool has_cache = config.use_custom_cache_size
                             ? config.custom_buffer_size > 0
                             : (config.l1_buffer_size > 0 || config.l2_buffer_size > 0);
  if (!config.only_latency && has_cache) {
    enabled_phases.push_back({Phase::CacheBandwidth, "cache-bandwidth"});
  }
  if (!config.only_bandwidth && has_cache) {
    enabled_phases.push_back({Phase::CacheLatency, "cache-latency"});
  }
  if (!config.only_bandwidth && config.buffer_size > 0 &&
      config.lat_num_accesses > 0) {
    enabled_phases.push_back({Phase::MainLatency, "main-latency"});
  }
  results.planned_phases = enabled_phases.size();
  results.phase_order_index = results.planned_phases == 0
                                  ? 0
                                  : results.loop_index % results.planned_phases;
  const std::vector<size_t> phase_order = build_benchmark_cyclic_order(
      enabled_phases.size(), results.loop_index);
  for (const size_t phase_index : phase_order) {
    results.planned_phase_order.push_back(enabled_phases[phase_index].name);
  }

  const std::array<BenchmarkOperation, 3> operations = {
      BenchmarkOperation::Read, BenchmarkOperation::Write,
      BenchmarkOperation::Copy};
  const std::vector<size_t> operation_order =
      build_benchmark_cyclic_order(operations.size(), results.loop_index);

  auto measurement_for_operation = [](BenchmarkOperation operation,
                                      BenchmarkMeasurement& read,
                                      BenchmarkMeasurement& write,
                                      BenchmarkMeasurement& copy)
      -> BenchmarkMeasurement& {
    if (operation == BenchmarkOperation::Read) return read;
    if (operation == BenchmarkOperation::Write) return write;
    return copy;
  };

  auto run_bandwidth_target = [&](void* src_buffer, void* dst_buffer,
                                  size_t buffer_size, int requested_threads,
                                  BenchmarkTarget target,
                                  BenchmarkMeasurement& read,
                                  BenchmarkMeasurement& write,
                                  BenchmarkMeasurement& copy,
                                  size_t phase_position) {
    for (size_t operation_position = 0;
         operation_position < operation_order.size(); ++operation_position) {
      const BenchmarkOperation operation =
          operations[operation_order[operation_position]];
      BenchmarkBandwidthExecutionState& operation_state = state.bandwidth[
          benchmark_bandwidth_state_index(target, operation)];
      BenchmarkMeasurement& measurement = measurement_for_operation(
          operation, read, write, copy);
      run_calibrated_bandwidth_measurement(
          src_buffer, dst_buffer, buffer_size, requested_threads, target,
          operation, config.user_specified_iterations,
          static_cast<size_t>(config.iterations), operation_state, measurement,
          test_timer, phase_position, operation_position);
      if (signal_received()) return;
    }
  };

  auto run_latency_target = [&](void* buffer, size_t buffer_size,
                                size_t fallback_access_count,
                                BenchmarkTarget target,
                                BenchmarkMeasurement& measurement,
                                size_t phase_position) {
    const uint64_t domain = target == BenchmarkTarget::MainMemory
                                ? kSeedDomainMainLatency
                            : target == BenchmarkTarget::L1
                                ? kSeedDomainL1Latency
                            : target == BenchmarkTarget::L2
                                ? kSeedDomainL2Latency
                                : kSeedDomainCustomLatency;
    BenchmarkLatencyExecutionState& latency_state =
        state.latency[benchmark_latency_state_index(target)];
    run_calibrated_latency_measurement(
        buffer, buffer_size, config.latency_stride_bytes,
        fallback_access_count, target,
        derive_benchmark_seed(config.benchmark_seed, domain), latency_state,
        measurement, test_timer, config.latency_sample_count, phase_position);
    measurement.qos_successful_workers = config.main_thread_qos_applied ? 1 : 0;
    measurement.qos_failed_workers =
        config.main_thread_qos_requested && !config.main_thread_qos_applied ? 1 : 0;
    measurement.created_workers = 1;
    measurement.qos_outcome = config.main_thread_qos_applied
                                  ? "main-thread-qos-applied"
                              : config.main_thread_qos_requested
                                  ? "main-thread-qos-failed"
                                  : "main-thread-qos-not-requested";
  };

  bool phase_execution_interrupted = false;
  try {
    const BenchmarkPhaseExecutionResult phase_execution =
        execute_benchmark_phase_schedule(
            phase_order, [] { return signal_received(); },
            [&](size_t phase_position, size_t phase_index) {
      const EnabledPhase& enabled_phase = enabled_phases[phase_index];
      results.realized_phase_order.push_back(enabled_phase.name);
      if (test_hooks != nullptr && test_hooks->fail_phase_preparation &&
          test_hooks->fail_phase_preparation(enabled_phase.name)) {
        throw std::runtime_error(
            Messages::benchmark_reason_prepare_failed(enabled_phase.name));
      }
      if (test_hooks != nullptr && test_hooks->fail_latency_chain_setup &&
          (enabled_phase.phase == Phase::CacheLatency ||
           enabled_phase.phase == Phase::MainLatency) &&
          test_hooks->fail_latency_chain_setup(enabled_phase.name)) {
        throw std::runtime_error(
            Messages::benchmark_reason_latency_chain_setup_failed(
                enabled_phase.name));
      }
      switch (enabled_phase.phase) {
        case Phase::MainBandwidth: {
          BenchmarkBuffers phase_buffers;
          if (prepare_main_memory_bandwidth_buffers(config, phase_buffers) !=
              EXIT_SUCCESS) {
            throw std::runtime_error(
                Messages::benchmark_reason_prepare_failed(
                    "main-memory bandwidth"));
          }
          run_bandwidth_target(
              phase_buffers.src_buffer(), phase_buffers.dst_buffer(),
              config.buffer_size, config.num_threads,
              BenchmarkTarget::MainMemory, results.main_read_bandwidth,
              results.main_write_bandwidth, results.main_copy_bandwidth,
              phase_position);
          break;
        }
        case Phase::CacheBandwidth: {
          BenchmarkBuffers phase_buffers;
          if (prepare_cache_bandwidth_buffers(config, phase_buffers) !=
              EXIT_SUCCESS) {
            throw std::runtime_error(Messages::benchmark_reason_prepare_failed(
                "cache bandwidth"));
          }
          const int cache_threads = config.user_specified_threads
                                        ? config.num_threads
                                        : Constants::SINGLE_THREAD;
          if (config.use_custom_cache_size) {
            run_bandwidth_target(
                phase_buffers.custom_bw_src(), phase_buffers.custom_bw_dst(),
                config.custom_buffer_size, cache_threads,
                BenchmarkTarget::Custom, results.custom_read_bandwidth,
                results.custom_write_bandwidth, results.custom_copy_bandwidth,
                phase_position);
          } else {
            if (config.l1_buffer_size > 0) {
              run_bandwidth_target(
                  phase_buffers.l1_bw_src(), phase_buffers.l1_bw_dst(),
                  config.l1_buffer_size, cache_threads, BenchmarkTarget::L1,
                  results.l1_read_bandwidth, results.l1_write_bandwidth,
                  results.l1_copy_bandwidth, phase_position);
            }
            if (!signal_received() && config.l2_buffer_size > 0) {
              run_bandwidth_target(
                  phase_buffers.l2_bw_src(), phase_buffers.l2_bw_dst(),
                  config.l2_buffer_size, cache_threads, BenchmarkTarget::L2,
                  results.l2_read_bandwidth, results.l2_write_bandwidth,
                  results.l2_copy_bandwidth, phase_position);
            }
          }
          break;
        }
        case Phase::CacheLatency: {
          BenchmarkBuffers phase_buffers;
          if (prepare_cache_latency_buffers(config, phase_buffers) !=
              EXIT_SUCCESS) {
            throw std::runtime_error(Messages::benchmark_reason_prepare_failed(
                "cache latency"));
          }
          if (config.use_custom_cache_size) {
            run_latency_target(
                phase_buffers.custom_buffer(), config.custom_buffer_size,
                config.custom_num_accesses, BenchmarkTarget::Custom,
                results.custom_latency, phase_position);
          } else {
            if (config.l1_buffer_size > 0) {
              run_latency_target(phase_buffers.l1_buffer(),
                                 config.l1_buffer_size,
                                 config.l1_num_accesses, BenchmarkTarget::L1,
                                 results.l1_latency, phase_position);
            }
            if (!signal_received() && config.l2_buffer_size > 0) {
              run_latency_target(phase_buffers.l2_buffer(),
                                 config.l2_buffer_size,
                                 config.l2_num_accesses, BenchmarkTarget::L2,
                                 results.l2_latency, phase_position);
            }
          }
          break;
        }
        case Phase::MainLatency: {
          BenchmarkBuffers phase_buffers;
          if (prepare_main_memory_latency_buffer(config, phase_buffers) !=
              EXIT_SUCCESS) {
            throw std::runtime_error(
                Messages::benchmark_reason_prepare_failed(
                    "main-memory latency"));
          }
          run_latency_target(phase_buffers.lat_buffer(), config.buffer_size,
                             config.lat_num_accesses,
                             BenchmarkTarget::MainMemory,
                             results.main_latency, phase_position);
          if (!signal_received() && results.main_latency.is_measured() &&
              !config.user_specified_latency_tlb_locality) {
            run_paired_locality_comparison(
                phase_buffers.lat_buffer(), config.buffer_size,
                config.latency_stride_bytes,
                results.main_latency.access_count, config.benchmark_seed,
                config.main_thread_qos_requested,
                config.main_thread_qos_applied,
                test_timer, results, phase_position);
          }
          break;
        }
      }
    });
    results.completed_phases = phase_execution.completed_phases;
    phase_execution_interrupted = phase_execution.interrupted;
  } catch (const std::exception &e) {
    std::cerr << Messages::error_benchmark_tests(e.what()) << std::endl;
    throw;  // Re-throw to be handled by caller
  }

  finalize_loop_results(results, config,
                        phase_execution_interrupted || signal_received());

  return results;
}
