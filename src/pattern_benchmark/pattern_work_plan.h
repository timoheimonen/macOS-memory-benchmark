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
 * @file pattern_work_plan.h
 * @brief Deterministic work planning for pattern benchmarks
 */
#ifndef PATTERN_WORK_PLAN_H
#define PATTERN_WORK_PLAN_H

#include <cstddef>
#include <string>
#include <vector>

enum class PatternMeasurementStatus {
  Measured,
  Skipped,
  Interrupted,
  Invalid,
};

struct PatternWorkerRange {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
  size_t accesses_per_pass = 0;
  size_t payload_bytes_per_pass = 0;
};

struct PatternWorkPlan {
  PatternMeasurementStatus status = PatternMeasurementStatus::Invalid;
  std::string status_reason;
  size_t stride_bytes = 0;
  size_t access_size_bytes = 0;
  int requested_threads = 0;
  int effective_threads = 0;
  size_t accesses_per_pass = 0;
  size_t payload_bytes_per_pass = 0;
  size_t passes = 0;
  size_t total_accesses = 0;
  size_t total_payload_bytes = 0;
  std::vector<PatternWorkerRange> workers;
};

struct PatternRandomWorkerIndices {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
  std::vector<size_t> indices;
};

/**
 * @brief Build an exact strided work plan from finalized worker chunks.
 *
 * Active workers are reduced until every worker has at least two valid
 * addresses and therefore performs at least one genuine stride transition.
 * Internal chunk boundaries are cache-line aligned exactly like the parallel
 * benchmark runner for page-aligned benchmark buffers.
 */
PatternWorkPlan build_strided_pattern_work_plan(size_t buffer_size, size_t stride, size_t access_size,
                                                int requested_threads, int base_passes,
                                                size_t minimum_total_payload_bytes);

/**
 * @brief Partition global random offsets into finalized per-worker arrays.
 *
 * Returned indices are relative to the worker's chunk and include an access
 * ending exactly at a chunk boundary. Invalid or boundary-crossing accesses are
 * omitted. Preparation is deterministic and intended to run outside timing.
 */
std::vector<PatternRandomWorkerIndices> build_random_worker_indices(
    size_t buffer_size, size_t access_size, int requested_threads,
    const std::vector<size_t>& global_indices);

const char* pattern_measurement_status_to_string(PatternMeasurementStatus status);

#endif  // PATTERN_WORK_PLAN_H
