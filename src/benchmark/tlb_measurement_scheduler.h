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
 * @file tlb_measurement_scheduler.h
 * @brief Deterministic balanced measurement scheduling for TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef TLB_MEASUREMENT_SCHEDULER_H
#define TLB_MEASUREMENT_SCHEDULER_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "benchmark/tlb_sweep_planner.h"

enum class TlbMeasurementPass {
  Base = 0,
  Refinement,
  LargeLocality,
};

/** One deterministic measurement action in execution order. */
struct TlbMeasurementTask {
  TlbMeasurementPass pass = TlbMeasurementPass::Base;
  size_t point_index = 0;
  size_t locality_bytes = 0;
  size_t round_index = 0;
  size_t order_index = 0;
  uint64_t seed = 0;
};

/** Completed task metadata and measured latency. */
struct TlbMeasurementRecord {
  TlbMeasurementPass pass = TlbMeasurementPass::Base;
  size_t point_index = 0;
  size_t locality_bytes = 0;
  size_t round_index = 0;
  size_t order_index = 0;
  uint64_t seed = 0;
  double latency_ns = 0.0;
};

enum class TlbTaskMeasureStatus {
  Success = 0,
  Error,
};

enum class TlbScheduleExecutionStatus {
  Complete = 0,
  Interrupted,
  Error,
};

struct TlbScheduleExecutionResult {
  TlbScheduleExecutionStatus status = TlbScheduleExecutionStatus::Complete;
  std::vector<TlbMeasurementRecord> records;
};

using TlbStopRequested = std::function<bool()>;
using TlbTaskMeasureFunction =
    std::function<TlbTaskMeasureStatus(const TlbMeasurementTask&, double&)>;

const char* tlb_measurement_pass_to_string(TlbMeasurementPass pass);

uint64_t derive_tlb_measurement_seed(uint64_t base_seed,
                                     TlbMeasurementPass pass,
                                     size_t round_index,
                                     size_t point_index);

/** Build seeded cyclic Latin rounds in which every point occurs once per round. */
std::vector<TlbMeasurementTask> build_tlb_measurement_schedule(
    const std::vector<TlbSweepPoint>& points,
    size_t round_count,
    uint64_t base_seed,
    TlbMeasurementPass pass);

/** Execute tasks through an injected measurement callback and non-blocking stop predicate. */
TlbScheduleExecutionResult execute_tlb_measurement_schedule(
    const std::vector<TlbMeasurementTask>& schedule,
    const TlbStopRequested& stop_requested,
    const TlbTaskMeasureFunction& measure_task);

#endif  // TLB_MEASUREMENT_SCHEDULER_H
