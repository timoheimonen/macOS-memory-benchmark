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
 * @file patterns_output.h
 * @brief Console help and human-readable summary output for `-P` / `--patterns`
 */

#ifndef PATTERNS_OUTPUT_H
#define PATTERNS_OUTPUT_H

#include <ostream>

#include "pattern_benchmark/patterns_runner.h"

/** Print `-P` / `--patterns` usage and exact 11 whitelisted options to @p os. */
void print_patterns_help(const char* prog_name, std::ostream& os);

/** Print resolved configuration & 7-pattern geometry summary to `std::cout`. */
void print_patterns_configuration(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResultMetadata& metadata);

/** Print per-pattern aggregate statistics, skipped statuses, and stderr quality warnings. */
void print_patterns_console_summary(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result);

#endif  // PATTERNS_OUTPUT_H
