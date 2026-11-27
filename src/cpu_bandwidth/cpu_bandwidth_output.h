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
 * @file cpu_bandwidth_output.h
 * @brief Console help and human-readable summary output for `--cpu-bandwidth`
 */

#ifndef CPU_BANDWIDTH_OUTPUT_H
#define CPU_BANDWIDTH_OUTPUT_H

#include <ostream>

#include "cpu_bandwidth/cpu_bandwidth_runner.h"

/** Print `-B` / `--cpu-bandwidth` usage and whitelisted options to @p os. */
void print_cpu_bandwidth_help(const char* prog_name, std::ostream& os);

/** Print resolved configuration & target geometry summary to `std::cout`. */
void print_cpu_bandwidth_configuration(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResultMetadata& metadata);

/** Print per-loop measurements, aggregate statistics, and quality warnings. */
void print_cpu_bandwidth_console_summary(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result);

#endif  // CPU_BANDWIDTH_OUTPUT_H
