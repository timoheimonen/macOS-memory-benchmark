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
 * @file patterns_sweep_runner.h
 * @brief Cartesian sweep planner, preflight, and schema-1 envelope runner
 */

#ifndef PATTERNS_SWEEP_RUNNER_H
#define PATTERNS_SWEEP_RUNNER_H

#include <cstddef>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "output/json/json_output/json_output_session.h"
#include "pattern_benchmark/patterns.h"
#include "pattern_benchmark/patterns_work_plan.h"

/** Compute total Cartesian run count for `-P` `--sweep` specs. */
size_t calculate_patterns_sweep_run_count(const PatternsConfig& config) noexcept;

/** Build isolated per-run `PatternsConfig` objects in Cartesian order. */
std::vector<PatternsConfig> build_patterns_sweep_run_configs(
    const PatternsConfig& base_config);

/**
 * Preflight ALL Cartesian sweep combinations before run 0 starts.
 *
 * Rejects any combination that violates work-plan or memory-budget bounds.
 */
bool preflight_patterns_sweep(const PatternsConfig& base_config,
                              const PatternsSystemInfo& sys_info);

/** Execute a validated `-P` `--sweep` through the command's JSON session. */
SweepExecutionResult run_patterns_sweep(const PatternsConfig& base_config,
                                        JsonOutputSession& output_session);

#endif  // PATTERNS_SWEEP_RUNNER_H
