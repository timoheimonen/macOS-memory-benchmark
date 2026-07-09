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
 * @file tlb_measurement_scheduler.cpp
 * @brief Deterministic balanced measurement scheduling for TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "benchmark/tlb_measurement_scheduler.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

namespace {

uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

}  // namespace

const char* tlb_measurement_pass_to_string(TlbMeasurementPass pass) {
  switch (pass) {
    case TlbMeasurementPass::Base:
      return "base";
    case TlbMeasurementPass::Refinement:
      return "refinement";
    case TlbMeasurementPass::LargeLocality:
      return "large-locality";
  }
  return "base";
}

uint64_t derive_tlb_measurement_seed(uint64_t base_seed,
                                     TlbMeasurementPass pass,
                                     size_t round_index,
                                     size_t point_index) {
  uint64_t value = splitmix64(base_seed ^ static_cast<uint64_t>(pass));
  value = splitmix64(value ^ static_cast<uint64_t>(round_index));
  return splitmix64(value ^ static_cast<uint64_t>(point_index));
}

std::vector<TlbMeasurementTask> build_tlb_measurement_schedule(
    const std::vector<TlbSweepPoint>& points,
    size_t round_count,
    uint64_t base_seed,
    TlbMeasurementPass pass) {
  std::vector<TlbMeasurementTask> schedule;
  if (points.empty() || round_count == 0) {
    return schedule;
  }
  if (round_count > std::numeric_limits<size_t>::max() / points.size()) {
    return schedule;
  }

  std::vector<size_t> seeded_point_order(points.size());
  std::iota(seeded_point_order.begin(), seeded_point_order.end(), 0);
  std::mt19937_64 rng(derive_tlb_measurement_seed(base_seed, pass, 0, points.size()));
  std::shuffle(seeded_point_order.begin(), seeded_point_order.end(), rng);

  schedule.reserve(points.size() * round_count);
  for (size_t round_index = 0; round_index < round_count; ++round_index) {
    for (size_t order_index = 0; order_index < points.size(); ++order_index) {
      const size_t rotated_index = (order_index + round_index) % points.size();
      const size_t local_point_index = seeded_point_order[rotated_index];
      const size_t point_index = points[local_point_index].point_index;
      schedule.push_back(TlbMeasurementTask{
          pass,
          point_index,
          points[local_point_index].locality_bytes,
          round_index,
          order_index,
          derive_tlb_measurement_seed(base_seed, pass, round_index, point_index),
      });
    }
  }
  return schedule;
}

TlbScheduleExecutionResult execute_tlb_measurement_schedule(
    const std::vector<TlbMeasurementTask>& schedule,
    const TlbStopRequested& stop_requested,
    const TlbTaskMeasureFunction& measure_task) {
  TlbScheduleExecutionResult result;
  result.records.reserve(schedule.size());

  for (const TlbMeasurementTask& task : schedule) {
    if (stop_requested && stop_requested()) {
      result.status = TlbScheduleExecutionStatus::Interrupted;
      return result;
    }

    double latency_ns = 0.0;
    if (!measure_task || measure_task(task, latency_ns) != TlbTaskMeasureStatus::Success) {
      result.status = TlbScheduleExecutionStatus::Error;
      return result;
    }
    result.records.push_back(TlbMeasurementRecord{
        task.pass,
        task.point_index,
        task.locality_bytes,
        task.round_index,
        task.order_index,
        task.seed,
        latency_ns,
    });

    if (stop_requested && stop_requested()) {
      result.status = TlbScheduleExecutionStatus::Interrupted;
      return result;
    }
  }

  return result;
}
