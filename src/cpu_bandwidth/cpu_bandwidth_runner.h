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
 * @file cpu_bandwidth_runner.h
 * @brief Excluded calibration, atomic plan freezing, Latin-square schedule, and checkpoints
 */

#ifndef CPU_BANDWIDTH_RUNNER_H
#define CPU_BANDWIDTH_RUNNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/system/benchmark_qos.h"
#include "cpu_bandwidth/cpu_bandwidth_executor.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"
#include "utils/descriptive_statistics.h"

/** Terminal state of one measured row. */
enum class CpuBandwidthMeasurementStatus : uint8_t {
  NotRun = 0,
  Measured,
  Invalid,
  Failed,
  Interrupted,
};

/** Terminal state of the overall `--cpu-bandwidth` run. */
enum class CpuBandwidthRunStatus : uint8_t {
  NotStarted = 0,
  Complete,
  Partial,
  Interrupted,
  Unsupported,
  Failed,
};

/** Logical checkpoint boundary kind. */
enum class CpuBandwidthCheckpointKind : uint8_t {
  Progress = 0,
  CommandTerminal,
};

/** One excluded calibration or warmup attempt. */
struct CpuBandwidthCalibrationAttempt {
  size_t attempt_index = 0;
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  CpuBandwidthOperation operation = CpuBandwidthOperation::Read;
  std::optional<size_t> plan_ref;
  std::string purpose;
  std::string duration_quality = "unavailable";
  bool terminal = false;
  bool valid = false;
  std::string reason_code = "not-started";
  CpuBandwidthOperationExecution execution;
};

/** One scheduled measurement row in `measurements[]`. */
struct CpuBandwidthMeasurementRow {
  size_t measurement_id = 0;
  std::optional<size_t> plan_ref;
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  CpuBandwidthOperation operation = CpuBandwidthOperation::Read;
  size_t loop_index = 0;
  size_t target_order_position = 0;
  size_t operation_order_position = 0;
  CpuBandwidthMeasurementStatus status = CpuBandwidthMeasurementStatus::NotRun;
  std::string reason_code = "not-run";
  bool attempted = false;
  size_t completed_passes = 0;
  size_t completed_exact_payload_bytes = 0;
  std::optional<double> bandwidth_gb_s;
  std::string duration_quality = "unavailable";
  CpuBandwidthOperationExecution execution;
};

/** Ordered `(target, operation)` pair inside `loop_records[]`. */
struct CpuBandwidthLoopStep {
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  CpuBandwidthOperation operation = CpuBandwidthOperation::Read;
};

/** Per-loop planned/realized order and measurement index references. */
struct CpuBandwidthLoopRecord {
  size_t loop_index = 0;
  std::vector<CpuBandwidthLoopStep> planned_order;
  std::vector<CpuBandwidthLoopStep> realized_order;
  std::vector<size_t> measurement_indexes;
};

/** Aggregated statistics and quality flags for one `(target, operation)`. */
struct CpuBandwidthTargetOpAggregate {
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  CpuBandwidthOperation operation = CpuBandwidthOperation::Read;
  std::vector<size_t> accepted_measurement_ids;
  std::optional<double> value;
  DescriptiveStatistics statistics;
  bool repeatability_assessment_available = false;
  bool run_complete = false;
  double cv_warning_threshold_pct = kCpuBandwidthCvWarningPct;
  bool cv_warning = false;
};

/** Run-level counters serialized under `counters`. */
struct CpuBandwidthCounters {
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t measured_measurements = 0;
  uint64_t planned_passes = 0;
  uint64_t completed_passes = 0;
  uint64_t planned_exact_payload_bytes = 0;
  uint64_t completed_exact_payload_bytes = 0;
};

/** Host and provenance metadata captured across command lifetime. */
struct CpuBandwidthResultMetadata {
  nlohmann::ordered_json build_manifest = unavailable_build_manifest_json();
  std::string timestamp;
  std::string processor_name;
  std::string macos_version;
  int performance_core_count = 0;
  int efficiency_core_count = 0;
  int logical_core_count = 0;
  size_t page_size_bytes = 0;
  size_t l1_data_cache_bytes = 0;
  size_t l2_data_cache_bytes = 0;
  size_t available_memory_bytes = 0;
  std::string available_memory_source = "mach-free-plus-inactive-rounded-mib";
  MainThreadQosResult main_thread_qos;
  AuditHostEnvironmentSnapshot environment_start;
  AuditHostEnvironmentSnapshot environment_end;
};

/** Complete auditable result state for a `--cpu-bandwidth` run. */
struct CpuBandwidthResult {
  bool initialized = false;
  CpuBandwidthRunStatus status = CpuBandwidthRunStatus::NotStarted;
  std::string reason_code = "not-started";
  bool results_complete = false;
  bool run_accepted = false;
  std::string diagnostic;
  bool interruption_requested = false;
  bool operation_order_balance_complete = false;
  bool target_order_balance_complete = false;

  std::vector<CpuBandwidthOperationPlan> operation_plans;
  std::array<std::array<std::optional<size_t>, kCpuBandwidthOperationCount>,
             kCpuBandwidthMaxTargets>
      frozen_plan_refs;
  std::array<std::array<std::vector<CpuBandwidthCalibrationAttempt>,
                        kCpuBandwidthOperationCount>,
             kCpuBandwidthMaxTargets>
      calibration_attempts;

  std::vector<CpuBandwidthMeasurementRow> measurements;
  std::vector<CpuBandwidthLoopRecord> loops;
  std::array<std::array<CpuBandwidthTargetOpAggregate,
                        kCpuBandwidthOperationCount>,
             kCpuBandwidthMaxTargets>
      aggregates;
  CpuBandwidthCounters counters;

  bool checkpoint_failed = false;
  size_t prior_file_writer_attempts = 0;
  size_t prior_successful_file_writes = 0;
  std::string snapshot_request = "terminal";
  size_t snapshot_interval_loops = 1;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;

  uint64_t write_attempt_ordinal = 0;
  uint64_t copy_attempt_ordinal = 0;
  std::unordered_set<uint64_t> seen_write_patterns;
  std::vector<std::string> quality_warnings;
};

/** Side-effect & test hooks for `run_cpu_bandwidth_suite`. */
struct CpuBandwidthRunnerHooks {
  std::function<bool()> stop_requested;
  bool progress_snapshots = false;
  std::function<std::pair<size_t, size_t>()> observe_file_writes;
  std::function<int(const CpuBandwidthResult&, CpuBandwidthCheckpointKind)>
      checkpoint;
  const CpuBandwidthExecutorTestControl* executor_test_control = nullptr;
  std::function<void(size_t loop_index, size_t within_loop)>
      before_measurement_hook;
};

/** Return frozen string token for `CpuBandwidthMeasurementStatus`. */
const char* cpu_bandwidth_measurement_status_to_string(
    CpuBandwidthMeasurementStatus status) noexcept;

/** Return frozen string token for `CpuBandwidthRunStatus`. */
const char* cpu_bandwidth_run_status_to_string(
    CpuBandwidthRunStatus status) noexcept;

/** Classify elapsed duration against `[0.100s, 0.250s]`. */
std::string classify_cpu_bandwidth_duration_quality(
    double elapsed_seconds, bool timing_valid) noexcept;

/** Initialize empty failed/interrupted result for preflight/setup failure serialization. */
void initialize_cpu_bandwidth_setup_failure_result(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    std::string_view reason_code,
    CpuBandwidthResult& result);

/** Initialize empty failed/interrupted result with explicit run status & interruption flag. */
void initialize_cpu_bandwidth_setup_failure_result(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    CpuBandwidthRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    CpuBandwidthResult& result);

/**
 * Run calibration (when auto), freeze operation plans atomically before loop 0,
 * execute cyclic Latin-square loops with bounded checkpoints, and finalize state.
 */
int run_cpu_bandwidth_suite(const CpuBandwidthConfig& config,
                            const CpuBandwidthWorkPlan& work_plan,
                            CpuBandwidthResourceSet& resources,
                            CpuBandwidthResult& result,
                            const CpuBandwidthRunnerHooks& hooks);

#endif  // CPU_BANDWIDTH_RUNNER_H
