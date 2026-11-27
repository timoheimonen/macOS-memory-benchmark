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
 * @file cpu_bandwidth_json.h
 * @brief Schema v1 JSON serializer, quality warning collector, and DOM peak helper
 */

#ifndef CPU_BANDWIDTH_JSON_H
#define CPU_BANDWIDTH_JSON_H

#include <string>
#include <vector>

#include "cpu_bandwidth/cpu_bandwidth_runner.h"
#include "third_party/nlohmann/json.hpp"

/** Return conservative JSON DOM + transport peak estimate from a work plan. */
CpuBandwidthJsonPeakEstimate calculate_cpu_bandwidth_json_peak_estimate(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan) noexcept;

/** Collect ordered, deduplicated quality warning tokens for console & JSON. */
std::vector<std::string> collect_cpu_bandwidth_quality_warning_tokens(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result);

/** Serialize a complete or partial `cpu_bandwidth` schema v1 JSON document. */
nlohmann::ordered_json build_cpu_bandwidth_json(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result);

#endif  // CPU_BANDWIDTH_JSON_H
