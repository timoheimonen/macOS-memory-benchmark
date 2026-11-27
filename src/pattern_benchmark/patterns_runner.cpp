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
 * @file patterns_runner.cpp
 * @brief Excluded calibration, atomic plan freezing, 2-axis Latin-square schedule, and checkpoints
 */

#include "pattern_benchmark/patterns_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

#include "core/audit/checkpoint_policy.h"
#include "core/config/constants.h"
#include "utils/cyclic_order.h"
#include "utils/numeric_utils.h"

namespace {

constexpr std::array<PatternOperation, kPatternsOperationCount> kOperations = {
    PatternOperation::Read,
    PatternOperation::Write,
    PatternOperation::Copy,
};

bool duration_in_target_window(double elapsed_seconds) noexcept {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::PATTERN_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::PATTERN_CALIBRATION_MAX_SECONDS;
}

bool stop_requested(const PatternsRunnerHooks& hooks) {
  return hooks.stop_requested && hooks.stop_requested();
}

void append_unique_warning(std::vector<std::string>& warnings,
                           const std::string& token) {
  if (!token.empty() &&
      std::find(warnings.begin(), warnings.end(), token) == warnings.end()) {
    warnings.push_back(token);
  }
}

bool environment_is_non_nominal(const AuditHostEnvironmentSnapshot& env) noexcept {
  const bool thermal_non_nominal =
      env.thermal_state != "nominal" && env.thermal_state != "unavailable";
  const bool low_power =
      env.low_power_mode_available && env.low_power_mode_enabled;
  return thermal_non_nominal || low_power;
}

size_t register_operation_plan(PatternsResult& result,
                               const PatternsWorkPlan& work_plan,
                               const PatternPlanEntry& pattern_plan,
                               const PatternsResourceSet& resources,
                               PatternOperation op,
                               size_t passes,
                               bool explicit_iterations) {
  for (size_t i = 0; i < result.operation_plans.size(); ++i) {
    const auto& existing = result.operation_plans[i];
    if (existing.pattern == pattern_plan.kind &&
        existing.operation == op &&
        existing.passes == passes &&
        existing.explicit_iterations == explicit_iterations) {
      return i;
    }
  }
  PatternsOperationPlan op_plan = build_patterns_operation_plan(
      pattern_plan, work_plan.src_seed, op, passes, explicit_iterations);
  if (pattern_plan.kind == PatternKind::Random &&
      op_plan.logical_working_set_bytes == 0 &&
      resources.random_workload.logical_working_set_bytes > 0) {
    op_plan.logical_working_set_bytes =
        resources.random_workload.logical_working_set_bytes;
  }
  op_plan.plan_ref = result.operation_plans.size();
  if (op_plan.valid && op == PatternOperation::Read) {
    op_plan.expected_checksum = compute_patterns_expected_read_checksum(
        work_plan.src_seed, op_plan.op_seed, pattern_plan.kind,
        pattern_plan.stride_bytes, static_cast<uint64_t>(passes),
        pattern_plan.worker_spans,
        pattern_plan.kind == PatternKind::Random ? &resources.random_workload
                                                 : nullptr);
  } else {
    op_plan.expected_checksum.applicable = false;
    op_plan.expected_checksum.available = false;
    op_plan.expected_checksum.reason_code = "not-applicable";
  }
  result.operation_plans.push_back(std::move(op_plan));
  return result.operation_plans.size() - 1;
}

std::pair<uint64_t, uint64_t> allocate_unique_write_pattern(
    PatternsResult& result, uint64_t op_seed) {
  uint64_t attempt_seed = 0;
  uint64_t pattern = 0;
  do {
    ++result.write_attempt_ordinal;
    attempt_seed = derive_patterns_write_attempt_seed(
        op_seed, result.write_attempt_ordinal);
    pattern = derive_patterns_write_pattern(attempt_seed);
  } while (pattern == 0 || (pattern & 1ULL) == 0 ||
           result.seen_write_patterns.count(pattern) > 0);
  result.seen_write_patterns.insert(pattern);
  return {attempt_seed, pattern};
}

uint64_t allocate_pre_op_sentinel(
    PatternsResult& result,
    uint64_t pre_op_sentinel_seed,
    std::optional<uint64_t> avoid_write_pattern) noexcept {
  uint64_t sentinel = 0;
  do {
    ++result.sentinel_attempt_ordinal;
    sentinel = derive_patterns_pre_op_sentinel(pre_op_sentinel_seed,
                                               result.sentinel_attempt_ordinal);
  } while (sentinel == 0 || (sentinel & 1ULL) == 0 ||
           (avoid_write_pattern.has_value() &&
            sentinel == *avoid_write_pattern));
  return sentinel;
}

void populate_loop_schedule(const PatternsWorkPlanConfig& config,
                            const PatternsWorkPlan& work_plan,
                            PatternsResult& result) {
  const size_t num_patterns =
      work_plan.valid ? work_plan.active_patterns.size() : 0;
  const size_t per_loop = num_patterns * kPatternsOperationCount;
  result.counters.planned_loops = config.loop_count;
  result.counters.planned_measurements = config.loop_count * per_loop;
  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);

  for (size_t loop_idx = 0; loop_idx < config.loop_count; ++loop_idx) {
    PatternsLoopRecord loop_rec;
    loop_rec.loop_index = loop_idx;
    const std::vector<size_t> p_order =
        build_cyclic_order(num_patterns, loop_idx);
    const std::vector<size_t> o_order =
        build_cyclic_order(kPatternsOperationCount, loop_idx);

    for (size_t p_pos = 0; p_pos < num_patterns; ++p_pos) {
      const PatternKind kind = work_plan.active_patterns[p_order[p_pos]];
      const auto& pplan = work_plan.patterns[patterns_kind_index(kind)];
      size_t default_touched_bytes = 0;
      const bool default_untouched =
          patterns_kind_is_strided(kind) ||
          (kind == PatternKind::Random &&
           NumericUtils::checked_multiply(
               pplan.random_workload.selected_accesses_per_pass,
               kPatternsAccessSizeBytes, default_touched_bytes) &&
           default_touched_bytes < pplan.buffer_bytes);
      for (size_t o_pos = 0; o_pos < kPatternsOperationCount; ++o_pos) {
        const PatternOperation op = kOperations[o_order[o_pos]];
        loop_rec.planned_order.push_back({kind, op});

        PatternsMeasurementRow row;
        row.measurement_id = result.measurements.size();
        row.pattern = kind;
        row.operation = op;
        row.loop_index = loop_idx;
        row.pattern_order_position = p_pos;
        row.operation_order_position = o_pos;
        row.execution.validation_checks =
            make_default_patterns_checks(op, default_untouched);
        row.execution.checksum_applicable = (op == PatternOperation::Read);
        loop_rec.measurement_indexes.push_back(row.measurement_id);
        result.measurements.push_back(std::move(row));
      }
    }
    result.loops.push_back(std::move(loop_rec));
  }
}

void assign_frozen_plans_to_measurements(const PatternsWorkPlanConfig& config,
                                         const PatternsWorkPlan& work_plan,
                                         PatternsResult& result) {
  uint64_t planned_passes_per_loop = 0;
  uint64_t planned_bytes_per_loop = 0;
  bool all_frozen = !work_plan.active_patterns.empty();

  for (PatternKind kind : work_plan.active_patterns) {
    const size_t p_idx = patterns_kind_index(kind);
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto& ref = result.frozen_plan_refs[p_idx][op_i];
      if (!ref.has_value()) {
        all_frozen = false;
        break;
      }
      const auto& op_plan = result.operation_plans.at(*ref);
      planned_passes_per_loop += static_cast<uint64_t>(op_plan.passes);
      planned_bytes_per_loop += static_cast<uint64_t>(op_plan.exact_payload_bytes);
    }
    if (!all_frozen) {
      break;
    }
  }

  for (auto& row : result.measurements) {
    const size_t p_idx = patterns_kind_index(row.pattern);
    const size_t op_i = patterns_operation_index(row.operation);
    row.plan_ref = result.frozen_plan_refs[p_idx][op_i];
    if (row.plan_ref.has_value()) {
      const auto& op_plan = result.operation_plans.at(*row.plan_ref);
      row.execution.validation_checks = make_default_patterns_checks(
          row.operation, op_plan.has_untouched_dst_bytes);
    }
  }

  if (all_frozen) {
    result.counters.planned_passes =
        planned_passes_per_loop * static_cast<uint64_t>(config.loop_count);
    result.counters.planned_exact_payload_bytes =
        planned_bytes_per_loop * static_cast<uint64_t>(config.loop_count);
  } else {
    result.counters.planned_passes = 0;
    result.counters.planned_exact_payload_bytes = 0;
  }
}

void finalize_remaining_rows(PatternsResult& result,
                             PatternsMeasurementStatus terminal_status,
                             const std::string& reason_code) {
  for (auto& row : result.measurements) {
    if (!row.attempted && row.status == PatternsMeasurementStatus::NotRun) {
      row.status = terminal_status;
      row.reason_code = reason_code;
      if (terminal_status != PatternsMeasurementStatus::NotRun) {
        ++result.counters.terminal_measurements;
      }
    }
  }
}

void recompute_aggregates_and_completion(const PatternsWorkPlanConfig& config,
                                         const PatternsWorkPlan& work_plan,
                                         PatternsResult& result) {
  // 1. Populate all 7 canonical patterns x 3 operations.
  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
    const double cv_threshold = patterns_cv_warning_threshold_pct(kind);

    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto op = kOperations[op_i];
      PatternsPatternOpAggregate& agg = result.aggregates[p_idx][op_i];
      agg = PatternsPatternOpAggregate{};
      agg.pattern = kind;
      agg.operation = op;
      agg.cv_warning_threshold_pct = cv_threshold;

      if (!work_plan.valid) {
        agg.status = PatternsAggregateStatus::Unavailable;
        agg.reason_code = !work_plan.reason_code.empty()
                              ? work_plan.reason_code
                              : (!result.reason_code.empty()
                                     ? result.reason_code
                                     : "unavailable");
        continue;
      }

      if (!pplan.enabled && pplan.buffer_bytes > 0) {
        agg.status = PatternsAggregateStatus::Skipped;
        agg.reason_code =
            pplan.reason_code.empty()
                ? std::string(PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL)
                : pplan.reason_code;
        continue;
      }

      if (!pplan.enabled) {
        agg.status = PatternsAggregateStatus::Unavailable;
        agg.reason_code = !result.reason_code.empty()
                              ? result.reason_code
                              : "unavailable";
        continue;
      }

      std::vector<double> values;
      std::string first_row_reason;
      for (const auto& row : result.measurements) {
        if (row.pattern == kind && row.operation == op) {
          if (first_row_reason.empty() && !row.reason_code.empty() &&
              row.reason_code != "not-run") {
            first_row_reason = row.reason_code;
          }
          if (row.status == PatternsMeasurementStatus::Measured &&
              row.bandwidth_gb_s.has_value()) {
            agg.accepted_measurement_ids.push_back(row.measurement_id);
            values.push_back(*row.bandwidth_gb_s);
          }
        }
      }

      const size_t n = agg.accepted_measurement_ids.size();
      agg.repeatability_assessment_available = (n >= 2);
      agg.run_complete = (n == config.loop_count && config.loop_count > 0);
      if (!values.empty()) {
        agg.status = PatternsAggregateStatus::Measured;
        agg.reason_code = "measured";
        agg.statistics = calculate_descriptive_statistics(values);
        agg.value = agg.statistics.median;
        agg.cv_warning =
            (n >= 3 && agg.statistics.coefficient_of_variation_defined &&
             agg.statistics.coefficient_of_variation_pct > cv_threshold);
      } else {
        agg.status = PatternsAggregateStatus::Unavailable;
        if (!first_row_reason.empty()) {
          agg.reason_code = first_row_reason;
        } else if (!result.reason_code.empty() &&
                   result.reason_code != "not-started") {
          agg.reason_code = result.reason_code;
        } else {
          agg.reason_code = "unavailable";
        }
      }
    }
  }

  // 2. Compute relative_to_forward_pct = 100.0 * agg.value / fwd_val per op.
  const size_t fwd_idx = patterns_kind_index(PatternKind::SequentialForward);
  for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
    const std::optional<double> fwd_val = result.aggregates[fwd_idx][op_i].value;
    if (fwd_val.has_value() && std::isfinite(*fwd_val) && *fwd_val > 0.0) {
      for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
        auto& agg = result.aggregates[p_idx][op_i];
        if (agg.value.has_value() && std::isfinite(*agg.value)) {
          agg.relative_to_forward_pct = 100.0 * (*agg.value) / (*fwd_val);
        }
      }
    }
  }

  // 3. Recompute loop-level counters accurately.
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  for (const auto& loop : result.loops) {
    if (!loop.realized_order.empty()) {
      ++attempted_loops;
    }
    bool all_measured = !loop.measurement_indexes.empty();
    for (size_t idx : loop.measurement_indexes) {
      if (result.measurements[idx].status !=
          PatternsMeasurementStatus::Measured) {
        all_measured = false;
        break;
      }
    }
    if (all_measured) {
      ++completed_loops;
    }
  }
  result.counters.attempted_loops = attempted_loops;
  result.counters.completed_loops = completed_loops;

  result.results_complete =
      result.counters.planned_measurements > 0 &&
      result.counters.measured_measurements ==
          result.counters.planned_measurements;
  const size_t num_active = work_plan.active_patterns.size();
  result.operation_order_balance_complete =
      result.results_complete &&
      (config.loop_count % kPatternsOperationCount == 0);
  result.pattern_order_balance_complete =
      result.results_complete && num_active > 0 &&
      (config.loop_count % num_active == 0);

  if (result.status != PatternsRunStatus::Failed &&
      result.status != PatternsRunStatus::Unsupported) {
    if (result.results_complete) {
      result.status = PatternsRunStatus::Complete;
      result.reason_code = "complete";
    } else if (result.interruption_requested) {
      result.status = PatternsRunStatus::Interrupted;
      result.reason_code = "interruption-requested";
    } else if (result.counters.terminal_measurements > 0 ||
               result.counters.attempted_measurements > 0) {
      result.status = PatternsRunStatus::Partial;
      result.reason_code = "partial-results";
    } else {
      result.status = PatternsRunStatus::NotStarted;
      result.reason_code = "not-started";
    }
  }

  result.run_accepted = result.status == PatternsRunStatus::Complete &&
                        result.results_complete &&
                        !result.checkpoint_failed;
}

bool invoke_checkpoint(PatternsResult& result,
                       PatternsCheckpointKind kind,
                       const PatternsRunnerHooks& hooks) {
  result.snapshot_request =
      (kind == PatternsCheckpointKind::CommandTerminal) ? "terminal"
                                                        : "progress";
  if (kind == PatternsCheckpointKind::CommandTerminal) {
    result.terminal_checkpoint_attempted = true;
    result.terminal_checkpoint_completed = false;
  }

  if (hooks.observe_file_writes) {
    const auto observed = hooks.observe_file_writes();
    result.prior_file_writer_attempts = observed.first;
    result.prior_successful_file_writes = observed.second;
  }

  int status = EXIT_SUCCESS;
  try {
    if (hooks.checkpoint) {
      status = hooks.checkpoint(result, kind);
    }
  } catch (const std::exception& err) {
    status = EXIT_FAILURE;
    result.diagnostic = err.what();
  } catch (...) {
    status = EXIT_FAILURE;
  }

  if (status != EXIT_SUCCESS) {
    result.checkpoint_failed = true;
    result.status = PatternsRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
    result.run_accepted = false;
    return false;
  }

  if (kind == PatternsCheckpointKind::CommandTerminal) {
    result.terminal_checkpoint_completed = true;
  }
  return true;
}

bool execute_excluded_attempt(const PatternsWorkPlanConfig& config,
                              const PatternsWorkPlan& work_plan,
                              const PatternPlanEntry& pattern_plan,
                              size_t plan_ref,
                              const std::string& purpose,
                              PatternsResourceSet& resources,
                              HighResTimer& timer,
                              PatternsResult& result,
                              const PatternsRunnerHooks& hooks,
                              double& out_elapsed) {
  const PatternsOperationPlan& op_plan = result.operation_plans.at(plan_ref);
  const size_t p_idx = patterns_kind_index(op_plan.pattern);
  const size_t op_i = patterns_operation_index(op_plan.operation);
  auto& attempts_vec = result.calibration_attempts[p_idx][op_i];

  std::optional<uint64_t> write_seed;
  std::optional<uint64_t> write_pattern;
  if (op_plan.operation == PatternOperation::Write) {
    auto pair = allocate_unique_write_pattern(result, op_plan.op_seed);
    write_seed = pair.first;
    write_pattern = pair.second;
  }

  std::optional<uint64_t> sentinel_pattern;
  if (op_plan.operation == PatternOperation::Copy ||
      (op_plan.operation == PatternOperation::Write &&
       op_plan.has_untouched_dst_bytes)) {
    sentinel_pattern = allocate_pre_op_sentinel(
        result, work_plan.pre_op_sentinel_seed, write_pattern);
  }

  PatternsCalibrationAttempt attempt;
  attempt.attempt_index = attempts_vec.size();
  attempt.pattern = op_plan.pattern;
  attempt.operation = op_plan.operation;
  attempt.plan_ref = plan_ref;
  attempt.purpose = purpose;
  attempt.execution = execute_patterns_operation(
      config, pattern_plan, op_plan, resources, timer, write_seed,
      write_pattern, sentinel_pattern, hooks.executor_test_control);
  attempt.duration_quality = classify_patterns_duration_quality(
      attempt.execution.elapsed_seconds, attempt.execution.timing_valid);
  attempt.terminal = true;
  attempt.valid = attempt.execution.valid;
  attempt.reason_code = attempt.execution.reason_code;
  out_elapsed = attempt.execution.elapsed_seconds;

  attempts_vec.push_back(std::move(attempt));
  return attempts_vec.back().valid;
}

}  // namespace

const char* patterns_measurement_status_to_string(
    PatternsMeasurementStatus status) noexcept {
  switch (status) {
    case PatternsMeasurementStatus::NotRun:
      return "not_run";
    case PatternsMeasurementStatus::Measured:
      return "measured";
    case PatternsMeasurementStatus::Invalid:
      return "invalid";
    case PatternsMeasurementStatus::Failed:
      return "failed";
    case PatternsMeasurementStatus::Interrupted:
      return "interrupted";
  }
  return "not_run";
}

const char* patterns_run_status_to_string(PatternsRunStatus status) noexcept {
  switch (status) {
    case PatternsRunStatus::NotStarted:
      return "not_started";
    case PatternsRunStatus::Complete:
      return "complete";
    case PatternsRunStatus::Partial:
      return "partial";
    case PatternsRunStatus::Interrupted:
      return "interrupted";
    case PatternsRunStatus::Unsupported:
      return "unsupported";
    case PatternsRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

const char* patterns_aggregate_status_to_string(
    PatternsAggregateStatus status) noexcept {
  switch (status) {
    case PatternsAggregateStatus::Measured:
      return "measured";
    case PatternsAggregateStatus::Skipped:
      return "skipped";
    case PatternsAggregateStatus::Unavailable:
      return "unavailable";
  }
  return "unavailable";
}

std::string classify_patterns_duration_quality(double elapsed_seconds,
                                               bool timing_valid) noexcept {
  if (!timing_valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0) {
    return "unavailable";
  }
  if (elapsed_seconds < Constants::PATTERN_CALIBRATION_MIN_SECONDS) {
    return "below-target-window";
  }
  if (elapsed_seconds > Constants::PATTERN_CALIBRATION_MAX_SECONDS) {
    return "above-target-window";
  }
  return "within-target-window";
}

std::vector<std::string> collect_patterns_quality_warning_tokens(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result) {
  std::vector<std::string> warnings;
  for (const auto& w : result.quality_warnings) {
    append_unique_warning(warnings, w);
  }
  if (environment_is_non_nominal(metadata.environment_start) ||
      environment_is_non_nominal(metadata.environment_end)) {
    append_unique_warning(warnings, "environment-not-nominal");
  }
  if (metadata.main_thread_qos.requested && !metadata.main_thread_qos.applied) {
    append_unique_warning(warnings, "main-thread-qos-not-applied");
  }
  if (std::any_of(result.measurements.begin(), result.measurements.end(),
                  [](const PatternsMeasurementRow& row) {
                    return row.attempted && row.execution.qos_failed_workers > 0;
                  })) {
    append_unique_warning(warnings, "worker-qos-not-applied");
  }
  if (config.non_cacheable &&
      (resources.src.mapped_bytes > 0 || resources.dst.mapped_bytes > 0) &&
      (!resources.src.madvise_applied || !resources.dst.madvise_applied)) {
    append_unique_warning(warnings, "madvise-random-not-applied");
  }
  if (!config.iterations.has_value()) {
    for (const auto& row : result.measurements) {
      if (row.status == PatternsMeasurementStatus::Measured &&
          row.duration_quality != "within-target-window") {
        append_unique_warning(
            warnings,
            std::string(patterns_kind_to_string(row.pattern)) + "-" +
                patterns_operation_to_string(row.operation) + "-duration-" +
                row.duration_quality);
      }
    }
  }
  for (PatternKind kind : work_plan.active_patterns) {
    const size_t p_idx = patterns_kind_index(kind);
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto& agg = result.aggregates[p_idx][op_i];
      if (agg.cv_warning) {
        append_unique_warning(
            warnings,
            std::string(patterns_kind_to_string(kind)) + "-" +
                patterns_operation_to_string(kOperations[op_i]) + "-high-cv");
      }
    }
  }
  return warnings;
}

void initialize_patterns_setup_failure_result(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    PatternsRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    PatternsResult& result) {
  result = PatternsResult{};
  result.status = status;
  result.reason_code = std::string(reason_code);
  result.interruption_requested = interruption_requested;
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);
  populate_loop_schedule(config, work_plan, result);
  const PatternsMeasurementStatus row_status =
      interruption_requested ? PatternsMeasurementStatus::Interrupted
                             : PatternsMeasurementStatus::Failed;
  finalize_remaining_rows(result, row_status, result.reason_code);
  recompute_aggregates_and_completion(config, work_plan, result);
  result.snapshot_request = "terminal";
  result.initialized = true;
}

void initialize_patterns_setup_failure_result(
    const PatternsWorkPlanConfig& config,
    const PatternsWorkPlan& work_plan,
    std::string_view reason_code,
    PatternsResult& result) {
  initialize_patterns_setup_failure_result(
      config, work_plan, PatternsRunStatus::Failed, reason_code, false, result);
}

int run_patterns_suite(const PatternsWorkPlanConfig& config,
                       const PatternsWorkPlan& work_plan,
                       PatternsResourceSet& resources,
                       PatternsResult& result,
                       const PatternsRunnerHooks& hooks) {
  result = PatternsResult{};
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);

  if (!work_plan.valid || !resources.valid || config.loop_count == 0 ||
      work_plan.active_patterns.empty()) {
    const std::string fail_reason =
        !work_plan.valid
            ? work_plan.reason_code
            : (!resources.valid
                   ? resources.reason_code
                   : (config.loop_count == 0
                          ? std::string(PatternsWorkPlanReason::INVALID_LOOP_COUNT)
                          : std::string("no-active-patterns")));
    initialize_patterns_setup_failure_result(config, work_plan, fail_reason,
                                             result);
    invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal, hooks);
    return EXIT_FAILURE;
  }

  auto timer_opt = HighResTimer::create();
  if (!timer_opt.has_value()) {
    initialize_patterns_setup_failure_result(config, work_plan,
                                             "timer-creation-failed", result);
    invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal, hooks);
    return EXIT_FAILURE;
  }
  HighResTimer& timer = *timer_opt;

  populate_loop_schedule(config, work_plan, result);
  result.initialized = true;

  // Phase 1: Calibration & excluded warmup.
  const bool explicit_iterations = config.iterations.has_value();
  if (explicit_iterations) {
    for (PatternKind kind : work_plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        const auto op = kOperations[op_i];
        const size_t plan_ref = register_operation_plan(
            result, work_plan, pplan, resources, op, *config.iterations, true);
        if (!result.operation_plans.at(plan_ref).valid) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code = result.operation_plans.at(plan_ref).reason_code;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
        result.frozen_plan_refs[p_idx][op_i] = plan_ref;
      }
    }
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (PatternKind kind : work_plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, PatternsCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        const size_t plan_ref = *result.frozen_plan_refs[p_idx][op_i];
        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, pplan, plan_ref, "frozen_measurement_warmup",
                resources, timer, result, hooks, elapsed)) {
          const auto& last_attempt =
              result.calibration_attempts[p_idx][op_i].back();
          result.status = PatternsRunStatus::Failed;
          result.reason_code = last_attempt.reason_code;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }
    }
  } else {
    // Automatic excluded calibration per active (pattern, op), then atomic freeze.
    for (PatternKind kind : work_plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        const auto op = kOperations[op_i];
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, PatternsCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        const size_t max_passes = calculate_patterns_max_passes(pplan, op);
        const PatternsOperationPlan op_one = build_patterns_operation_plan(
            pplan, work_plan.src_seed, op, 1, false);
        if (max_passes == 0 || !op_one.valid ||
            op_one.exact_payload_bytes == 0) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code =
              !op_one.valid ? op_one.reason_code
                            : std::string(PatternsWorkPlanReason::WORK_CAP_EXCEEDED);
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        size_t passes = NumericUtils::calculate_minimum_pilot_count(
            op_one.exact_payload_bytes,
            Constants::PATTERN_CALIBRATION_MIN_PILOT_BYTES, max_passes);
        if (passes == 0) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code = "calibration-scaling-failed";
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        size_t latest_plan_ref = register_operation_plan(
            result, work_plan, pplan, resources, op, passes, false);
        if (!result.operation_plans.at(latest_plan_ref).valid) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code =
              result.operation_plans.at(latest_plan_ref).reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, pplan, latest_plan_ref,
                "calibration_shape_warmup", resources, timer, result, hooks,
                elapsed)) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[p_idx][op_i].back().reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, PatternsCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        if (!execute_excluded_attempt(config, work_plan, pplan, latest_plan_ref,
                                      "pilot", resources, timer, result, hooks,
                                      elapsed)) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[p_idx][op_i].back().reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        bool single_unit_warmed = (passes == 1);
        for (size_t corr = 0;
             !duration_in_target_window(elapsed) &&
             corr < Constants::PATTERN_CALIBRATION_MAX_CORRECTIONS;
             ++corr) {
          if (passes == 1 &&
              elapsed > Constants::PATTERN_CALIBRATION_MAX_SECONDS) {
            break;
          }
          const size_t corrected =
              NumericUtils::calculate_duration_scaled_count(
                  elapsed, passes,
                  Constants::PATTERN_CALIBRATION_TARGET_SECONDS, 1, max_passes,
                  1);
          if (corrected == 0) {
            result.status = PatternsRunStatus::Failed;
            result.reason_code = "calibration-scaling-failed";
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }
          if (corrected == passes) {
            break;
          }
          passes = corrected;
          latest_plan_ref = register_operation_plan(
              result, work_plan, pplan, resources, op, passes, false);
          if (!result.operation_plans.at(latest_plan_ref).valid) {
            result.status = PatternsRunStatus::Failed;
            result.reason_code =
                result.operation_plans.at(latest_plan_ref).reason_code;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }

          if (stop_requested(hooks)) {
            result.interruption_requested = true;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result,
                                    PatternsMeasurementStatus::Interrupted,
                                    "interruption-before-task");
            recompute_aggregates_and_completion(config, work_plan, result);
            const bool ck_ok = invoke_checkpoint(
                result, PatternsCheckpointKind::CommandTerminal, hooks);
            return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
          }

          const bool single_unit = (passes == 1);
          if (single_unit && !single_unit_warmed) {
            if (!execute_excluded_attempt(
                    config, work_plan, pplan, latest_plan_ref,
                    "single_unit_confirmation_warmup", resources, timer, result,
                    hooks, elapsed)) {
              result.status = PatternsRunStatus::Failed;
              result.reason_code =
                  result.calibration_attempts[p_idx][op_i].back().reason_code;
              assign_frozen_plans_to_measurements(config, work_plan, result);
              finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                      result.reason_code);
              recompute_aggregates_and_completion(config, work_plan, result);
              invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                                hooks);
              return EXIT_FAILURE;
            }
            single_unit_warmed = true;
            if (stop_requested(hooks)) {
              result.interruption_requested = true;
              assign_frozen_plans_to_measurements(config, work_plan, result);
              finalize_remaining_rows(result,
                                      PatternsMeasurementStatus::Interrupted,
                                      "interruption-before-task");
              recompute_aggregates_and_completion(config, work_plan, result);
              const bool ck_ok = invoke_checkpoint(
                  result, PatternsCheckpointKind::CommandTerminal, hooks);
              return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
            }
          }

          const std::string purpose =
              single_unit ? "single_unit_confirmation" : "correction";
          if (!execute_excluded_attempt(config, work_plan, pplan,
                                        latest_plan_ref, purpose, resources,
                                        timer, result, hooks, elapsed)) {
            result.status = PatternsRunStatus::Failed;
            result.reason_code =
                result.calibration_attempts[p_idx][op_i].back().reason_code;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }
        }

        result.frozen_plan_refs[p_idx][op_i] = latest_plan_ref;
      }
    }

    // Freeze plans atomically before loop 0 and run frozen_measurement_warmup.
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (PatternKind kind : work_plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, PatternsCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        const size_t plan_ref = *result.frozen_plan_refs[p_idx][op_i];
        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, pplan, plan_ref, "frozen_measurement_warmup",
                resources, timer, result, hooks, elapsed)) {
          result.status = PatternsRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[p_idx][op_i].back().reason_code;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }
    }
  }

  // Phase 2: Timed measurement loops in 2-axis cyclic Latin-square order.
  for (auto& loop : result.loops) {
    for (size_t pos = 0; pos < loop.measurement_indexes.size(); ++pos) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, PatternsCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      if (hooks.before_measurement_hook) {
        hooks.before_measurement_hook(loop.loop_index, pos);
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, PatternsCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
      }

      PatternsMeasurementRow& row =
          result.measurements[loop.measurement_indexes[pos]];
      const size_t p_idx = patterns_kind_index(row.pattern);
      const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
      const PatternsOperationPlan& op_plan =
          result.operation_plans.at(*row.plan_ref);

      std::optional<uint64_t> write_seed;
      std::optional<uint64_t> write_pattern;
      if (row.operation == PatternOperation::Write) {
        auto pair = allocate_unique_write_pattern(result, op_plan.op_seed);
        write_seed = pair.first;
        write_pattern = pair.second;
      }

      std::optional<uint64_t> sentinel_pattern;
      if (row.operation == PatternOperation::Copy ||
          (row.operation == PatternOperation::Write &&
           op_plan.has_untouched_dst_bytes)) {
        sentinel_pattern = allocate_pre_op_sentinel(
            result, work_plan.pre_op_sentinel_seed, write_pattern);
      }

      row.attempted = true;
      ++result.counters.attempted_measurements;
      loop.realized_order.push_back({row.pattern, row.operation});

      row.execution = execute_patterns_operation(
          config, pplan, op_plan, resources, timer, write_seed, write_pattern,
          sentinel_pattern, hooks.executor_test_control);
      row.duration_quality = classify_patterns_duration_quality(
          row.execution.elapsed_seconds, row.execution.timing_valid);

      ++result.counters.terminal_measurements;
      if (row.execution.valid) {
        const long double exact_bytes =
            static_cast<long double>(op_plan.exact_payload_bytes);
        const long double elapsed =
            static_cast<long double>(row.execution.elapsed_seconds);
        const double bw = static_cast<double>(exact_bytes / elapsed / 1.0e9L);
        if (std::isfinite(bw) && bw > 0.0) {
          row.status = PatternsMeasurementStatus::Measured;
          row.reason_code = "measured";
          row.completed_passes = op_plan.passes;
          row.completed_exact_payload_bytes = op_plan.exact_payload_bytes;
          row.bandwidth_gb_s = bw;
          ++result.counters.measured_measurements;
          result.counters.completed_passes +=
              static_cast<uint64_t>(op_plan.passes);
          result.counters.completed_exact_payload_bytes +=
              static_cast<uint64_t>(op_plan.exact_payload_bytes);
        } else {
          row.status = PatternsMeasurementStatus::Invalid;
          row.reason_code = "invalid-derived-metric";
        }
      } else {
        row.status = PatternsMeasurementStatus::Invalid;
        row.reason_code = row.execution.reason_code;
      }

      if (row.status != PatternsMeasurementStatus::Measured) {
        result.status = PatternsRunStatus::Failed;
        result.reason_code = row.reason_code;
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                  "interruption-before-task");
        } else {
          finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                  "not-run-after-runtime-failure");
        }
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      const bool has_remaining_work =
          (pos + 1 < loop.measurement_indexes.size()) ||
          (loop.loop_index + 1 < config.loop_count);
      if (has_remaining_work && stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result, PatternsMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, PatternsCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    }

    // End of loop bounded checkpoint (for intermediate loops when file target).
    recompute_aggregates_and_completion(config, work_plan, result);
    if (hooks.progress_snapshots &&
        ((loop.loop_index + 1) % result.snapshot_interval_loops == 0) &&
        (loop.loop_index + 1 < config.loop_count)) {
      if (!invoke_checkpoint(result, PatternsCheckpointKind::Progress, hooks)) {
        finalize_remaining_rows(result, PatternsMeasurementStatus::Failed,
                                "not-run-after-runtime-failure");
        recompute_aggregates_and_completion(config, work_plan, result);
        return EXIT_FAILURE;
      }
    }
  }

  recompute_aggregates_and_completion(config, work_plan, result);
  if (!invoke_checkpoint(result, PatternsCheckpointKind::CommandTerminal,
                         hooks)) {
    recompute_aggregates_and_completion(config, work_plan, result);
    return EXIT_FAILURE;
  }

  return (result.status == PatternsRunStatus::Failed ||
          result.status == PatternsRunStatus::Unsupported)
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
