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
 * @file cpu_latency_runner.h
 * @brief Excluded calibration, atomic plan freezing, cyclic schedule, and checkpoints
 */

#ifndef CPU_LATENCY_RUNNER_H
#define CPU_LATENCY_RUNNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/system/benchmark_qos.h"
#include "cpu_latency/cpu_latency_executor.h"
#include "cpu_latency/cpu_latency_work_plan.h"
#include "utils/descriptive_statistics.h"

/** Terminal state of one measured row. */
enum class CpuLatencyMeasurementStatus : uint8_t {
  NotRun = 0,
  Measured,
  Invalid,
  Failed,
  Interrupted,
};

/** Terminal state of the overall `--cpu-latency` run. */
enum class CpuLatencyRunStatus : uint8_t {
  NotStarted = 0,
  Complete,
  Partial,
  Interrupted,
  Unsupported,
  Failed,
};

/** Logical checkpoint boundary kind. */
enum class CpuLatencyCheckpointKind : uint8_t {
  Progress = 0,
  CommandTerminal,
};

/** One excluded calibration or warmup attempt. */
struct CpuLatencyCalibrationAttempt {
  size_t attempt_index = 0;
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  std::optional<size_t> plan_ref;
  std::string purpose;
  std::string duration_quality = "unavailable";
  bool terminal = false;
  bool valid = false;
  std::string reason_code = "not-started";
  CpuLatencyAttemptExecution execution;
};

/** One scheduled measurement row in `measurements[]`. */
struct CpuLatencyMeasurementRow {
  size_t measurement_id = 0;
  std::optional<size_t> plan_ref;
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  size_t loop_index = 0;
  size_t target_order_position = 0;
  CpuLatencyMeasurementStatus status = CpuLatencyMeasurementStatus::NotRun;
  std::string reason_code = "not-run";
  bool attempted = false;
  size_t completed_accesses = 0;
  size_t completed_chain_cycles = 0;
  std::optional<double> latency_ns;
  std::string duration_quality = "unavailable";
  CpuLatencyAttemptExecution execution;
};

/** Per-loop planned/realized target order and measurement index references. */
struct CpuLatencyLoopRecord {
  size_t loop_index = 0;
  std::vector<CpuLatencyTarget> planned_order;
  std::vector<CpuLatencyTarget> realized_order;
  std::vector<size_t> measurement_indexes;
};

/** One paired `--compare-locality` round (`16 KiB random-box` vs `global-random`). */
struct CpuLatencyLocalityRound {
  size_t loop_index = 0;
  size_t round_index = 0;
  CpuLatencyLocalitySubrunResult locality_16k;
  CpuLatencyLocalitySubrunResult locality_global;
};

/** Aggregated statistics, quality flags, and pooled samples for one target. */
struct CpuLatencyTargetAggregate {
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  std::vector<size_t> accepted_measurement_ids;
  std::optional<double> value;
  DescriptiveStatistics statistics;
  bool repeatability_assessment_available = false;
  bool run_complete = false;
  double cv_warning_threshold_pct = kCpuLatencyCvWarningPct;
  bool cv_warning = false;
  bool pooled_applicable = false;
  std::vector<double> pooled_samples_ns;
  DescriptiveStatistics pooled_statistics;
};

/** Aggregate comparison summary for `--compare-locality`. */
struct CpuLatencyLocalityAggregate {
  std::optional<double> locality_16k_latency_ns;
  std::optional<double> global_random_latency_ns;
  std::optional<double> locality_latency_delta_ns;
};

/** Run-level counters serialized under `counters`. */
struct CpuLatencyCounters {
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t measured_measurements = 0;
  uint64_t planned_accesses = 0;
  uint64_t completed_accesses = 0;
};

/** Host and provenance metadata captured across command lifetime. */
struct CpuLatencyResultMetadata {
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

/** Complete auditable result state for a `--cpu-latency` run. */
struct CpuLatencyResult {
  bool initialized = false;
  CpuLatencyRunStatus status = CpuLatencyRunStatus::NotStarted;
  std::string reason_code = "not-started";
  bool results_complete = false;
  bool run_accepted = false;
  std::string diagnostic;
  bool interruption_requested = false;
  bool target_order_balance_complete = false;

  std::vector<CpuLatencyTargetExecutionPlan> target_plans;
  std::array<std::optional<size_t>, kCpuLatencyMaxTargets> frozen_plan_refs;
  std::array<std::vector<CpuLatencyCalibrationAttempt>, kCpuLatencyMaxTargets>
      calibration_attempts;

  std::vector<CpuLatencyMeasurementRow> measurements;
  std::vector<CpuLatencyLoopRecord> loops;
  std::vector<CpuLatencyLocalityRound> locality_rounds;
  std::array<CpuLatencyTargetAggregate, kCpuLatencyMaxTargets> aggregates;
  CpuLatencyLocalityAggregate locality_aggregate;
  CpuLatencyCounters counters;

  bool checkpoint_failed = false;
  size_t prior_file_writer_attempts = 0;
  size_t prior_successful_file_writes = 0;
  std::string snapshot_request = "terminal";
  size_t snapshot_interval_loops = 1;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;

  std::vector<std::string> quality_warnings;
};

/** Side-effect & test hooks for `run_cpu_latency_suite`. */
struct CpuLatencyRunnerHooks {
  std::function<bool()> stop_requested;
  bool progress_snapshots = false;
  std::function<std::pair<size_t, size_t>()> observe_file_writes;
  std::function<int(const CpuLatencyResult&, CpuLatencyCheckpointKind)>
      checkpoint;
  const CpuLatencyExecutorTestControl* executor_test_control = nullptr;
  std::function<void(size_t loop_index, size_t target_pos)>
      before_measurement_hook;
  std::function<void(size_t loop_index, size_t round_index)>
      before_locality_round_hook;
};

/** Return frozen string token for `CpuLatencyMeasurementStatus`. */
const char* cpu_latency_measurement_status_to_string(
    CpuLatencyMeasurementStatus status) noexcept;

/** Return frozen string token for `CpuLatencyRunStatus`. */
const char* cpu_latency_run_status_to_string(
    CpuLatencyRunStatus status) noexcept;

/** Classify elapsed duration against `[0.100s, 0.300s]`. */
std::string classify_cpu_latency_duration_quality(
    double elapsed_seconds, bool timing_valid) noexcept;

/** Initialize empty failed/interrupted result for setup failure serialization. */
void initialize_cpu_latency_setup_failure_result(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    CpuLatencyRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    CpuLatencyResult& result);

/**
 * Run calibration (when auto), freeze target plans atomically before loop 0,
 * execute cyclic loops with optional sample windows and `--compare-locality`,
 * perform bounded checkpoints, and finalize state.
 */
int run_cpu_latency_suite(const CpuLatencyConfig& config,
                          const CpuLatencyWorkPlan& work_plan,
                          CpuLatencyResourceSet& resources,
                          CpuLatencyResult& result,
                          const CpuLatencyRunnerHooks& hooks);

#endif  // CPU_LATENCY_RUNNER_H
