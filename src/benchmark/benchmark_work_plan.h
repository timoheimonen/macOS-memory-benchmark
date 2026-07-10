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
 * @file benchmark_work_plan.h
 * @brief Deterministic work planning and calibration for --benchmark
 */
#ifndef BENCHMARK_WORK_PLAN_H
#define BENCHMARK_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "benchmark/benchmark_measurement.h"

enum class BenchmarkTarget {
  MainMemory,
  L1,
  L2,
  Custom,
};

enum class BenchmarkOperation {
  Read,
  Write,
  Copy,
  Latency,
};

struct BenchmarkWorkerRange {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
};

struct BenchmarkWorkPlan {
  BenchmarkMeasurementStatus status = BenchmarkMeasurementStatus::Invalid;
  std::string status_reason;
  BenchmarkTarget target = BenchmarkTarget::MainMemory;
  BenchmarkOperation operation = BenchmarkOperation::Read;
  size_t buffer_size_bytes = 0;
  int requested_threads = 0;
  int effective_threads = 0;
  size_t passes = 0;
  size_t payload_bytes_per_pass = 0;
  size_t total_payload_bytes = 0;
  std::vector<size_t> boundaries;
  std::vector<BenchmarkWorkerRange> workers;
};

struct BenchmarkLatencyWorkPlan {
  BenchmarkMeasurementStatus status = BenchmarkMeasurementStatus::Invalid;
  std::string status_reason;
  BenchmarkTarget target = BenchmarkTarget::MainMemory;
  size_t buffer_size_bytes = 0;
  size_t stride_bytes = 0;
  size_t chain_node_count = 0;
  size_t access_count = 0;
  size_t complete_chain_cycles = 0;
  uint64_t seed = 0;
};

struct BenchmarkBandwidthExecutionState {
  bool initialized = false;
  BenchmarkWorkPlan plan;
  double pilot_elapsed_seconds = 0.0;
  size_t calibration_corrections = 0;
  std::string duration_quality;
};

struct BenchmarkLatencyExecutionState {
  bool initialized = false;
  BenchmarkLatencyWorkPlan plan;
  double pilot_elapsed_seconds = 0.0;
  size_t calibration_corrections = 0;
  std::string duration_quality;
};

/** @brief Per-command calibrated work reused by every --count loop. */
struct BenchmarkExecutionState {
  std::array<BenchmarkBandwidthExecutionState, 12> bandwidth;
  std::array<BenchmarkLatencyExecutionState, 4> latency;
};

/** @brief Result of executing a cold-path phase schedule with stop checks. */
struct BenchmarkPhaseExecutionResult {
  size_t completed_phases = 0;
  bool interrupted = false;
  std::vector<size_t> realized_phase_indexes;
};

size_t benchmark_bandwidth_state_index(BenchmarkTarget target,
                                       BenchmarkOperation operation);
size_t benchmark_latency_state_index(BenchmarkTarget target);

BenchmarkWorkPlan build_benchmark_bandwidth_work_plan(
    size_t buffer_size_bytes, int requested_threads, size_t passes,
    BenchmarkTarget target, BenchmarkOperation operation);

bool set_benchmark_work_plan_passes(BenchmarkWorkPlan& plan, size_t passes);

BenchmarkLatencyWorkPlan build_benchmark_latency_work_plan(
    size_t buffer_size_bytes, size_t stride_bytes, size_t desired_access_count,
    size_t minimum_complete_cycles, size_t maximum_access_count,
    BenchmarkTarget target, uint64_t seed);

size_t calculate_benchmark_pilot_passes(size_t payload_bytes_per_pass,
                                        size_t minimum_pilot_payload_bytes,
                                        size_t maximum_passes);

size_t calculate_benchmark_calibrated_count(double pilot_duration_seconds,
                                            size_t pilot_count,
                                            double target_duration_seconds,
                                            size_t minimum_count,
                                            size_t maximum_count,
                                            size_t quantum = 1);

std::vector<size_t> build_benchmark_cyclic_order(size_t item_count,
                                                 size_t loop_index);

BenchmarkPhaseExecutionResult execute_benchmark_phase_schedule(
    const std::vector<size_t>& phase_order,
    const std::function<bool()>& stop_requested,
    const std::function<void(size_t, size_t)>& execute_phase);

bool benchmark_elapsed_is_valid(double elapsed);

bool benchmark_duration_in_window(double elapsed_seconds,
                                  double minimum_seconds,
                                  double maximum_seconds);

std::string classify_benchmark_duration_quality(
    double elapsed_seconds, size_t count, double minimum_seconds,
    double maximum_seconds, bool minimum_work_limited = false);

uint64_t derive_benchmark_seed(uint64_t base_seed, uint64_t domain);

const char* benchmark_target_to_string(BenchmarkTarget target);
const char* benchmark_operation_to_string(BenchmarkOperation operation);

#endif  // BENCHMARK_WORK_PLAN_H
