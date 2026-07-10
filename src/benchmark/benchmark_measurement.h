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
 * @file benchmark_measurement.h
 * @brief Status-bearing measurements for the standard benchmark
 */
#ifndef BENCHMARK_MEASUREMENT_H
#define BENCHMARK_MEASUREMENT_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class BenchmarkMeasurementStatus {
  NotRun,
  Measured,
  Skipped,
  Interrupted,
  Invalid,
  Failed,
};

enum class BenchmarkRunStatus {
  NotStarted,
  Complete,
  Partial,
  Interrupted,
  Failed,
};

inline const char* benchmark_measurement_status_to_string(BenchmarkMeasurementStatus status) {
  switch (status) {
    case BenchmarkMeasurementStatus::NotRun:
      return "not-run";
    case BenchmarkMeasurementStatus::Measured:
      return "measured";
    case BenchmarkMeasurementStatus::Skipped:
      return "skipped";
    case BenchmarkMeasurementStatus::Interrupted:
      return "interrupted";
    case BenchmarkMeasurementStatus::Invalid:
      return "invalid";
    case BenchmarkMeasurementStatus::Failed:
      return "failed";
  }
  return "invalid";
}

inline const char* benchmark_run_status_to_string(BenchmarkRunStatus status) {
  switch (status) {
    case BenchmarkRunStatus::NotStarted:
      return "not-started";
    case BenchmarkRunStatus::Complete:
      return "complete";
    case BenchmarkRunStatus::Partial:
      return "partial";
    case BenchmarkRunStatus::Interrupted:
      return "interrupted";
    case BenchmarkRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

/**
 * @brief One standard benchmark metric and the exact work that produced it.
 *
 * A numeric value is present only when status is Measured. Work and calibration
 * fields are populated progressively by the planner and executor; keeping them
 * on the measurement makes partial-run JSON auditable without sentinel values.
 */
struct BenchmarkMeasurement {
  BenchmarkMeasurementStatus status = BenchmarkMeasurementStatus::NotRun;
  std::string status_reason;
  std::optional<double> value;
  double elapsed_seconds = 0.0;
  double pilot_elapsed_seconds = 0.0;
  bool automatic_calibration = false;
  bool duration_within_target = false;
  std::string duration_quality;
  std::string work_policy;
  std::string target;
  std::string operation;
  std::string qos_outcome = "best-effort-request-not-observed";
  int created_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
  size_t calibration_corrections = 0;
  size_t buffer_size_bytes = 0;
  size_t passes = 0;
  size_t access_count = 0;
  size_t chain_node_count = 0;
  size_t complete_chain_cycles = 0;
  size_t exact_payload_bytes = 0;
  int requested_threads = 0;
  int effective_threads = 0;
  uint64_t seed = 0;
  size_t phase_order_index = 0;
  size_t operation_order_index = 0;
  std::vector<double> samples;
  std::vector<uint64_t> sample_seeds;

  bool is_measured() const {
    return status == BenchmarkMeasurementStatus::Measured && value.has_value();
  }
};

inline void set_measurement_unavailable(BenchmarkMeasurement& measurement,
                                        BenchmarkMeasurementStatus status,
                                        const std::string& reason) {
  measurement.status = status;
  measurement.status_reason = reason;
  measurement.value.reset();
}

inline void set_measurement_value(BenchmarkMeasurement& measurement,
                                  double value,
                                  double elapsed_seconds) {
  measurement.status = BenchmarkMeasurementStatus::Measured;
  measurement.status_reason.clear();
  measurement.value = value;
  measurement.elapsed_seconds = elapsed_seconds;
}

#endif  // BENCHMARK_MEASUREMENT_H
