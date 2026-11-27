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
 * @file cpu_latency_output.h
 * @brief Console help and human-readable summary output for `--cpu-latency`
 */

#ifndef CPU_LATENCY_OUTPUT_H
#define CPU_LATENCY_OUTPUT_H

#include <ostream>

#include "cpu_latency/cpu_latency_runner.h"

/** Print `-L` / `--cpu-latency` usage and whitelisted options to @p os. */
void print_cpu_latency_help(const char* prog_name, std::ostream& os);

/** Print resolved configuration & target geometry summary to `std::cout`. */
void print_cpu_latency_configuration(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResultMetadata& metadata);

/** Print per-target latency statistics, locality comparison, and warnings. */
void print_cpu_latency_console_summary(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResourceSet& resources,
    const CpuLatencyResultMetadata& metadata,
    const CpuLatencyResult& result);

#endif  // CPU_LATENCY_OUTPUT_H
