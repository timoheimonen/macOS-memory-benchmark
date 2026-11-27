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
 * @file patterns_json.h
 * @brief Schema v1 JSON serializer, quality warning collector, and DOM peak helper
 */

#ifndef PATTERNS_JSON_H
#define PATTERNS_JSON_H

#include <string>
#include <vector>

#include "pattern_benchmark/patterns.h"
#include "pattern_benchmark/patterns_runner.h"
#include "third_party/nlohmann/json.hpp"

/** Return conservative JSON DOM + transport peak estimate from a work plan. */
PatternsJsonPeakEstimate calculate_patterns_json_peak_estimate(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan) noexcept;

/** Collect ordered, deduplicated quality warning tokens for console & JSON. */
std::vector<std::string> collect_patterns_quality_warning_tokens(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result);

/** Serialize a complete or partial `patterns` schema v1 JSON document. */
nlohmann::ordered_json build_patterns_json(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result);

#endif  // PATTERNS_JSON_H
