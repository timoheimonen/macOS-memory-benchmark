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
 * @file patterns_runner.h
 * @brief Excluded calibration, atomic plan freezing, 2-axis Latin-square schedule, and checkpoints
 */

#ifndef PATTERNS_RUNNER_H
#define PATTERNS_RUNNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/system/benchmark_qos.h"
#include "pattern_benchmark/patterns_executor.h"
#include "pattern_benchmark/patterns_work_plan.h"
#include "utils/descriptive_statistics.h"

/** Terminal state of one measured row. */
enum class PatternsMeasurementStatus : uint8_t {
  NotRun = 0,
  Measured,
  Invalid,
  Failed,
  Interrupted,
};

/** Terminal state of the overall `-P` / `--patterns` run. */
enum class PatternsRunStatus : uint8_t {
  NotStarted = 0,
  Complete,
  Partial,
  Interrupted,
  Unsupported,
  Failed,
};

/** Aggregate publication status for each canonical `(pattern, operation)`. */
enum class PatternsAggregateStatus : uint8_t {
  Measured = 0,
  Skipped,
  Unavailable,
};

/** Logical checkpoint boundary kind. */
enum class PatternsCheckpointKind : uint8_t {
  Progress = 0,
  CommandTerminal,
};

/** One excluded calibration or warmup attempt. */
struct PatternsCalibrationAttempt {
  size_t attempt_index = 0;
  PatternKind pattern = PatternKind::SequentialForward;
  PatternOperation operation = PatternOperation::Read;
  std::optional<size_t> plan_ref;
  std::string purpose;
  std::string duration_quality = "unavailable";
  bool terminal = false;
  bool valid = false;
  std::string reason_code = "not-started";
  PatternsOperationExecution execution;
};

/** One scheduled measurement row in `measurements[]`. */
struct PatternsMeasurementRow {
  size_t measurement_id = 0;
  std::optional<size_t> plan_ref;
  PatternKind pattern = PatternKind::SequentialForward;
  PatternOperation operation = PatternOperation::Read;
  size_t loop_index = 0;
  size_t pattern_order_position = 0;
  size_t operation_order_position = 0;
  PatternsMeasurementStatus status = PatternsMeasurementStatus::NotRun;
  std::string reason_code = "not-run";
  bool attempted = false;
  size_t completed_passes = 0;
  size_t completed_exact_payload_bytes = 0;
  std::optional<double> bandwidth_gb_s;
  std::string duration_quality = "unavailable";
  PatternsOperationExecution execution;
};

/** Ordered `(pattern, operation)` pair inside `loop_records[]`. */
struct PatternsLoopStep {
  PatternKind pattern = PatternKind::SequentialForward;
  PatternOperation operation = PatternOperation::Read;
};

/** Per-loop planned/realized order and measurement index references. */
struct PatternsLoopRecord {
  size_t loop_index = 0;
  std::vector<PatternsLoopStep> planned_order;
  std::vector<PatternsLoopStep> realized_order;
  std::vector<size_t> measurement_indexes;
};

/** Aggregated statistics and quality flags for one `(pattern, operation)`. */
struct PatternsPatternOpAggregate {
  PatternKind pattern = PatternKind::SequentialForward;
  PatternOperation operation = PatternOperation::Read;
  PatternsAggregateStatus status = PatternsAggregateStatus::Unavailable;
  std::string reason_code = "not-started";
  std::vector<size_t> accepted_measurement_ids;
  std::optional<double> value;
  std::optional<double> relative_to_forward_pct;
  DescriptiveStatistics statistics;
  bool repeatability_assessment_available = false;
  bool run_complete = false;
  double cv_warning_threshold_pct = kPatternsStreamingCvWarningPct;
  bool cv_warning = false;
};

/** Run-level counters serialized under `counters`. */
struct PatternsCounters {
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
struct PatternsResultMetadata {
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
  bool user_specified_seed = false;
  MainThreadQosResult main_thread_qos;
  AuditHostEnvironmentSnapshot environment_start;
  AuditHostEnvironmentSnapshot environment_end;
};

/** Complete auditable result state for a `-P` / `--patterns` run. */
struct PatternsResult {
  bool initialized = false;
  PatternsRunStatus status = PatternsRunStatus::NotStarted;
  std::string reason_code = "not-started";
  bool results_complete = false;
  bool run_accepted = false;
  std::string diagnostic;
  bool interruption_requested = false;
  bool operation_order_balance_complete = false;
  bool pattern_order_balance_complete = false;

  std::vector<PatternsOperationPlan> operation_plans;
  std::array<std::array<std::optional<size_t>, kPatternsOperationCount>,
             kPatternsMaxKinds>
      frozen_plan_refs;
  std::array<
      std::array<std::vector<PatternsCalibrationAttempt>, kPatternsOperationCount>,
      kPatternsMaxKinds>
      calibration_attempts;

  std::vector<PatternsMeasurementRow> measurements;
  std::vector<PatternsLoopRecord> loops;
  std::array<std::array<PatternsPatternOpAggregate, kPatternsOperationCount>,
             kPatternsMaxKinds>
      aggregates;
  PatternsCounters counters;

  bool checkpoint_failed = false;
  size_t prior_file_writer_attempts = 0;
  size_t prior_successful_file_writes = 0;
  std::string snapshot_request = "terminal";
  size_t snapshot_interval_loops = 1;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;

  uint64_t write_attempt_ordinal = 0;
  uint64_t sentinel_attempt_ordinal = 0;
  std::unordered_set<uint64_t> seen_write_patterns;
  std::vector<std::string> quality_warnings;
};

/** Side-effect & test hooks for `run_patterns_suite`. */
struct PatternsRunnerHooks {
  std::function<bool()> stop_requested;
  bool progress_snapshots = false;
  std::function<std::pair<size_t, size_t>()> observe_file_writes;
  std::function<int(const PatternsResult&, PatternsCheckpointKind)> checkpoint;
  const PatternsExecutorTestControl* executor_test_control = nullptr;
  std::function<void(size_t loop_index, size_t within_loop)>
      before_measurement_hook;
};

/** Return frozen string token for `PatternsMeasurementStatus`. */
const char* patterns_measurement_status_to_string(
    PatternsMeasurementStatus status) noexcept;

/** Return frozen string token for `PatternsRunStatus`. */
const char* patterns_run_status_to_string(PatternsRunStatus status) noexcept;

/** Return frozen string token for `PatternsAggregateStatus`. */
const char* patterns_aggregate_status_to_string(
    PatternsAggregateStatus status) noexcept;

/** Classify elapsed duration against `[0.100s, 0.250s]`. */
std::string classify_patterns_duration_quality(double elapsed_seconds,
                                               bool timing_valid) noexcept;

/** Collect deduplicated quality warning tokens for console & JSON output. */
std::vector<std::string> collect_patterns_quality_warning_tokens(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result);

/** Initialize empty failed/interrupted result for preflight/setup failure serialization. */
void initialize_patterns_setup_failure_result(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    std::string_view reason_code,
    PatternsResult& result);

/** Initialize empty failed/interrupted result with explicit run status & interruption flag. */
void initialize_patterns_setup_failure_result(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    PatternsRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    PatternsResult& result);

/**
 * Run calibration (when auto), freeze operation plans atomically before loop 0,
 * execute 2-axis cyclic Latin-square loops with bounded checkpoints, and finalize state.
 */
int run_patterns_suite(const PatternsWorkPlanConfig& config,
                       const PatternsWorkPlan& work_plan,
                       PatternsResourceSet& resources,
                       PatternsResult& result,
                       const PatternsRunnerHooks& hooks);

#endif  // PATTERNS_RUNNER_H
