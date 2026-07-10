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
 * @file benchmark_executor.h
 * @brief Benchmark execution functions for memory and cache tests
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides functions to execute various memory and cache benchmarks,
 * including bandwidth and latency tests for main memory and different cache levels.
 */
#ifndef BENCHMARK_EXECUTOR_H
#define BENCHMARK_EXECUTOR_H

#include <cstddef>  // size_t
#include <functional>
#include <string>

// Forward declarations
struct BenchmarkBuffers;
struct BenchmarkConfig;
struct BenchmarkResults;
struct HighResTimer;
struct BenchmarkExecutionState;

/** @brief Optional kernel-free fault seams for phase preparation tests. */
struct BenchmarkExecutorTestHooks {
  std::function<bool(const std::string&)> fail_phase_preparation;
  std::function<bool(const std::string&)> fail_latency_chain_setup;
};

/**
 * @brief Run a single benchmark loop and return results
 * @param buffers Reference to benchmark buffers structure (unused in phase-local allocation mode)
 * @param config Reference to benchmark configuration (updated for per-loop diagnostics)
 * @param loop Zero-based loop index used for cyclic phase/operation order and diagnostics
 * @param test_timer Reference to high-resolution timer for measurements
 * @param execution_state Optional cross-loop calibration state; a local state is used when null
 * @param test_hooks Optional deterministic failure seams used by tests
 * @return BenchmarkResults structure containing all results from the loop
 *
 * Executes one complete benchmark loop, running all configured tests
 * (bandwidth and/or latency) and calculating results.
 */
BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, BenchmarkConfig& config, int loop,
                                           HighResTimer& test_timer,
                                           BenchmarkExecutionState* execution_state = nullptr,
                                           const BenchmarkExecutorTestHooks* test_hooks = nullptr);

#endif // BENCHMARK_EXECUTOR_H
