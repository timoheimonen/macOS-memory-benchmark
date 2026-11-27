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

/**
 * @file cpu_latency_sweep_runner.h
 * @brief Cartesian sweep planner, preflight, and schema-1 envelope runner
 */

#ifndef CPU_LATENCY_SWEEP_RUNNER_H
#define CPU_LATENCY_SWEEP_RUNNER_H

#include <cstddef>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "cpu_latency/cpu_latency.h"
#include "cpu_latency/cpu_latency_work_plan.h"
#include "output/json/json_output/json_output_session.h"

/** Compute total Cartesian run count for `-L` `--sweep` specs. */
size_t calculate_cpu_latency_sweep_run_count(
    const CpuLatencyConfig& config) noexcept;

/** Build isolated per-run `CpuLatencyConfig` objects in Cartesian order. */
std::vector<CpuLatencyConfig> build_cpu_latency_sweep_run_configs(
    const CpuLatencyConfig& base_config);

/**
 * Preflight ALL Cartesian sweep combinations before run 0 starts.
 *
 * Rejects any combination that disables all targets, violates locality/stride/
 * compare-locality constraints, or fails `memory_budget` preflight.
 */
bool preflight_cpu_latency_sweep(const CpuLatencyConfig& base_config,
                                 const CpuLatencySystemInfo& sys_info);

/** Execute a validated `-L` `--sweep` through the command's JSON session. */
SweepExecutionResult run_cpu_latency_sweep(
    const CpuLatencyConfig& base_config,
    JsonOutputSession& output_session);

#endif  // CPU_LATENCY_SWEEP_RUNNER_H
