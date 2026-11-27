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
 * @file cpu_bandwidth_runner.cpp
 * @brief Excluded calibration, atomic plan freezing, Latin-square schedule, and checkpoints
 */

#include "cpu_bandwidth/cpu_bandwidth_runner.h"

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

constexpr std::array<CpuBandwidthOperation, kCpuBandwidthOperationCount>
    kOperations = {
        CpuBandwidthOperation::Read,
        CpuBandwidthOperation::Write,
        CpuBandwidthOperation::Copy,
};

bool duration_in_target_window(double elapsed_seconds) noexcept {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS;
}

bool stop_requested(const CpuBandwidthRunnerHooks& hooks) {
  return hooks.stop_requested && hooks.stop_requested();
}

size_t register_operation_plan(CpuBandwidthResult& result,
                               const CpuBandwidthTargetPlan& target_plan,
                               CpuBandwidthOperation op,
                               size_t passes,
                               bool explicit_iterations) {
  for (size_t i = 0; i < result.operation_plans.size(); ++i) {
    const auto& existing = result.operation_plans[i];
    if (existing.target == target_plan.target &&
        existing.operation == op &&
        existing.passes == passes &&
        existing.explicit_iterations == explicit_iterations) {
      return i;
    }
  }
  CpuBandwidthOperationPlan op_plan = build_cpu_bandwidth_operation_plan(
      target_plan, op, passes, explicit_iterations);
  op_plan.plan_ref = result.operation_plans.size();
  if (op_plan.valid && op == CpuBandwidthOperation::Read) {
    op_plan.expected_checksum = compute_cpu_bandwidth_expected_read_checksum(
        target_plan.src_seed, op_plan.op_seed, static_cast<uint64_t>(passes),
        target_plan.worker_spans);
  } else {
    op_plan.expected_checksum.applicable = false;
    op_plan.expected_checksum.available = false;
    op_plan.expected_checksum.reason_code = "not-applicable";
  }
  result.operation_plans.push_back(std::move(op_plan));
  return result.operation_plans.size() - 1;
}

std::pair<uint64_t, uint64_t> allocate_unique_write_pattern(
    CpuBandwidthResult& result, uint64_t op_seed) {
  uint64_t attempt_seed = 0;
  uint64_t pattern = 0;
  do {
    ++result.write_attempt_ordinal;
    attempt_seed = derive_cpu_bandwidth_write_attempt_seed(
        op_seed, result.write_attempt_ordinal);
    pattern = derive_cpu_bandwidth_write_pattern(attempt_seed);
  } while (pattern == 0 || (pattern & 1ULL) == 0 ||
           result.seen_write_patterns.count(pattern) > 0);
  result.seen_write_patterns.insert(pattern);
  return {attempt_seed, pattern};
}

uint64_t allocate_copy_sentinel(CpuBandwidthResult& result,
                                uint64_t pre_copy_sentinel_seed) noexcept {
  ++result.copy_attempt_ordinal;
  return derive_cpu_bandwidth_pre_copy_sentinel_pattern(
      pre_copy_sentinel_seed, result.copy_attempt_ordinal);
}

void populate_loop_schedule(const CpuBandwidthConfig& config,
                            const CpuBandwidthWorkPlan& work_plan,
                            CpuBandwidthResult& result) {
  const size_t num_targets = work_plan.active_targets.size();
  const size_t per_loop = num_targets * kCpuBandwidthOperationCount;
  result.counters.planned_loops = config.loop_count;
  result.counters.planned_measurements = config.loop_count * per_loop;
  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);

  for (size_t loop_idx = 0; loop_idx < config.loop_count; ++loop_idx) {
    CpuBandwidthLoopRecord loop_rec;
    loop_rec.loop_index = loop_idx;
    const std::vector<size_t> t_order =
        build_cyclic_order(num_targets, loop_idx);
    const std::vector<size_t> o_order =
        build_cyclic_order(kCpuBandwidthOperationCount, loop_idx);

    for (size_t t_pos = 0; t_pos < num_targets; ++t_pos) {
      const CpuBandwidthTarget target = work_plan.active_targets[t_order[t_pos]];
      for (size_t o_pos = 0; o_pos < kCpuBandwidthOperationCount; ++o_pos) {
        const CpuBandwidthOperation op = kOperations[o_order[o_pos]];
        loop_rec.planned_order.push_back({target, op});

        CpuBandwidthMeasurementRow row;
        row.measurement_id = result.measurements.size();
        row.target = target;
        row.operation = op;
        row.loop_index = loop_idx;
        row.target_order_position = t_pos;
        row.operation_order_position = o_pos;
        row.execution.validation_checks = make_default_cpu_bandwidth_checks(op);
        row.execution.checksum_applicable = (op == CpuBandwidthOperation::Read);
        loop_rec.measurement_indexes.push_back(row.measurement_id);
        result.measurements.push_back(std::move(row));
      }
    }
    result.loops.push_back(std::move(loop_rec));
  }
}

void assign_frozen_plans_to_measurements(const CpuBandwidthConfig& config,
                                         const CpuBandwidthWorkPlan& work_plan,
                                         CpuBandwidthResult& result) {
  uint64_t planned_passes_per_loop = 0;
  uint64_t planned_bytes_per_loop = 0;
  bool all_frozen = !work_plan.active_targets.empty();

  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto& ref = result.frozen_plan_refs[t_idx][op_i];
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
    const size_t t_idx = cpu_bandwidth_target_index(row.target);
    const size_t op_i = cpu_bandwidth_operation_index(row.operation);
    row.plan_ref = result.frozen_plan_refs[t_idx][op_i];
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

void finalize_remaining_rows(CpuBandwidthResult& result,
                             CpuBandwidthMeasurementStatus terminal_status,
                             const std::string& reason_code) {
  for (auto& row : result.measurements) {
    if (!row.attempted && row.status == CpuBandwidthMeasurementStatus::NotRun) {
      row.status = terminal_status;
      row.reason_code = reason_code;
      if (terminal_status != CpuBandwidthMeasurementStatus::NotRun) {
        ++result.counters.terminal_measurements;
      }
    }
  }
}

void recompute_aggregates_and_completion(const CpuBandwidthConfig& config,
                                         const CpuBandwidthWorkPlan& work_plan,
                                         CpuBandwidthResult& result) {
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto op = static_cast<CpuBandwidthOperation>(op_i);
      CpuBandwidthTargetOpAggregate& agg = result.aggregates[t_idx][op_i];
      agg = CpuBandwidthTargetOpAggregate{};
      agg.target = target;
      agg.operation = op;
      agg.cv_warning_threshold_pct = kCpuBandwidthCvWarningPct;

      std::vector<double> values;
      for (const auto& row : result.measurements) {
        if (row.target == target && row.operation == op &&
            row.status == CpuBandwidthMeasurementStatus::Measured &&
            row.bandwidth_gb_s.has_value()) {
          agg.accepted_measurement_ids.push_back(row.measurement_id);
          values.push_back(*row.bandwidth_gb_s);
        }
      }
      const size_t n = agg.accepted_measurement_ids.size();
      agg.repeatability_assessment_available = (n >= 2);
      agg.run_complete = (n == config.loop_count);
      if (!values.empty()) {
        agg.statistics = calculate_descriptive_statistics(values);
        agg.value = agg.statistics.median;
        agg.cv_warning =
            (n >= 3 && agg.statistics.coefficient_of_variation_defined &&
             agg.statistics.coefficient_of_variation_pct >
                 kCpuBandwidthCvWarningPct);
      }
    }
  }

  // Recompute loop-level counters accurately.
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  for (const auto& loop : result.loops) {
    if (!loop.realized_order.empty()) {
      ++attempted_loops;
    }
    bool all_measured = !loop.measurement_indexes.empty();
    for (size_t idx : loop.measurement_indexes) {
      if (result.measurements[idx].status !=
          CpuBandwidthMeasurementStatus::Measured) {
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
  const size_t num_targets = work_plan.active_targets.size();
  result.operation_order_balance_complete =
      result.results_complete &&
      (config.loop_count % kCpuBandwidthOperationCount == 0);
  result.target_order_balance_complete =
      result.results_complete && num_targets > 0 &&
      (config.loop_count % num_targets == 0);

  if (result.status != CpuBandwidthRunStatus::Failed &&
      result.status != CpuBandwidthRunStatus::Unsupported) {
    if (result.results_complete) {
      result.status = CpuBandwidthRunStatus::Complete;
      result.reason_code = "complete";
    } else if (result.interruption_requested) {
      result.status = CpuBandwidthRunStatus::Interrupted;
      result.reason_code = "interruption-requested";
    } else if (result.counters.terminal_measurements > 0 ||
               result.counters.attempted_measurements > 0) {
      result.status = CpuBandwidthRunStatus::Partial;
      result.reason_code = "partial-results";
    } else {
      result.status = CpuBandwidthRunStatus::NotStarted;
      result.reason_code = "not-started";
    }
  }

  result.run_accepted = result.status == CpuBandwidthRunStatus::Complete &&
                        result.results_complete &&
                        !result.checkpoint_failed;
}

bool invoke_checkpoint(CpuBandwidthResult& result,
                       CpuBandwidthCheckpointKind kind,
                       const CpuBandwidthRunnerHooks& hooks) {
  result.snapshot_request =
      (kind == CpuBandwidthCheckpointKind::CommandTerminal) ? "terminal"
                                                            : "progress";
  if (kind == CpuBandwidthCheckpointKind::CommandTerminal) {
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
    result.status = CpuBandwidthRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
    result.run_accepted = false;
    return false;
  }

  if (kind == CpuBandwidthCheckpointKind::CommandTerminal) {
    result.terminal_checkpoint_completed = true;
  }
  return true;
}

bool execute_excluded_attempt(const CpuBandwidthConfig& config,
                              const CpuBandwidthWorkPlan& work_plan,
                              const CpuBandwidthTargetPlan& target_plan,
                              size_t plan_ref,
                              const std::string& purpose,
                              CpuBandwidthResourceSet& resources,
                              HighResTimer& timer,
                              CpuBandwidthResult& result,
                              const CpuBandwidthRunnerHooks& hooks,
                              double& out_elapsed) {
  const CpuBandwidthOperationPlan& op_plan = result.operation_plans.at(plan_ref);
  const size_t t_idx = cpu_bandwidth_target_index(op_plan.target);
  const size_t op_i = cpu_bandwidth_operation_index(op_plan.operation);
  auto& attempts_vec = result.calibration_attempts[t_idx][op_i];

  std::optional<uint64_t> write_seed;
  std::optional<uint64_t> write_pattern;
  if (op_plan.operation == CpuBandwidthOperation::Write) {
    auto pair = allocate_unique_write_pattern(result, op_plan.op_seed);
    write_seed = pair.first;
    write_pattern = pair.second;
  }
  const uint64_t sentinel_pattern =
      op_plan.operation == CpuBandwidthOperation::Copy
          ? allocate_copy_sentinel(result, work_plan.pre_copy_sentinel_seed)
          : 0ULL;

  CpuBandwidthCalibrationAttempt attempt;
  attempt.attempt_index = attempts_vec.size();
  attempt.target = op_plan.target;
  attempt.operation = op_plan.operation;
  attempt.plan_ref = plan_ref;
  attempt.purpose = purpose;
  attempt.execution = execute_cpu_bandwidth_operation(
      config, target_plan, op_plan, resources.targets[t_idx], timer, write_seed,
      write_pattern, sentinel_pattern, hooks.executor_test_control);
  attempt.duration_quality = classify_cpu_bandwidth_duration_quality(
      attempt.execution.elapsed_seconds, attempt.execution.timing_valid);
  attempt.terminal = true;
  attempt.valid = attempt.execution.valid;
  attempt.reason_code = attempt.execution.reason_code;
  out_elapsed = attempt.execution.elapsed_seconds;

  attempts_vec.push_back(std::move(attempt));
  return attempts_vec.back().valid;
}

}  // namespace

const char* cpu_bandwidth_measurement_status_to_string(
    CpuBandwidthMeasurementStatus status) noexcept {
  switch (status) {
    case CpuBandwidthMeasurementStatus::NotRun:
      return "not_run";
    case CpuBandwidthMeasurementStatus::Measured:
      return "measured";
    case CpuBandwidthMeasurementStatus::Invalid:
      return "invalid";
    case CpuBandwidthMeasurementStatus::Failed:
      return "failed";
    case CpuBandwidthMeasurementStatus::Interrupted:
      return "interrupted";
  }
  return "not_run";
}

const char* cpu_bandwidth_run_status_to_string(
    CpuBandwidthRunStatus status) noexcept {
  switch (status) {
    case CpuBandwidthRunStatus::NotStarted:
      return "not_started";
    case CpuBandwidthRunStatus::Complete:
      return "complete";
    case CpuBandwidthRunStatus::Partial:
      return "partial";
    case CpuBandwidthRunStatus::Interrupted:
      return "interrupted";
    case CpuBandwidthRunStatus::Unsupported:
      return "unsupported";
    case CpuBandwidthRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

std::string classify_cpu_bandwidth_duration_quality(
    double elapsed_seconds, bool timing_valid) noexcept {
  if (!timing_valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0) {
    return "unavailable";
  }
  if (elapsed_seconds < Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS) {
    return "below-target-window";
  }
  if (elapsed_seconds > Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS) {
    return "above-target-window";
  }
  return "within-target-window";
}

void initialize_cpu_bandwidth_setup_failure_result(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    CpuBandwidthRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    CpuBandwidthResult& result) {
  result = CpuBandwidthResult{};
  result.status = status;
  result.reason_code = std::string(reason_code);
  result.interruption_requested = interruption_requested;
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);
  populate_loop_schedule(config, work_plan, result);
  const CpuBandwidthMeasurementStatus row_status =
      interruption_requested ? CpuBandwidthMeasurementStatus::Interrupted
                             : CpuBandwidthMeasurementStatus::Failed;
  finalize_remaining_rows(result, row_status, result.reason_code);
  recompute_aggregates_and_completion(config, work_plan, result);
  result.snapshot_request = "terminal";
  result.initialized = true;
}

void initialize_cpu_bandwidth_setup_failure_result(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    std::string_view reason_code,
    CpuBandwidthResult& result) {
  initialize_cpu_bandwidth_setup_failure_result(
      config, work_plan, CpuBandwidthRunStatus::Failed, reason_code, false,
      result);
}

int run_cpu_bandwidth_suite(const CpuBandwidthConfig& config,
                            const CpuBandwidthWorkPlan& work_plan,
                            CpuBandwidthResourceSet& resources,
                            CpuBandwidthResult& result,
                            const CpuBandwidthRunnerHooks& hooks) {
  result = CpuBandwidthResult{};
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);

  if (!work_plan.valid || !resources.valid) {
    initialize_cpu_bandwidth_setup_failure_result(
        config, work_plan,
        !work_plan.valid ? work_plan.reason_code : resources.reason_code,
        result);
    invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                      hooks);
    return EXIT_FAILURE;
  }

  auto timer_opt = HighResTimer::create();
  if (!timer_opt.has_value()) {
    initialize_cpu_bandwidth_setup_failure_result(
        config, work_plan, "timer-creation-failed", result);
    invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                      hooks);
    return EXIT_FAILURE;
  }
  HighResTimer& timer = *timer_opt;

  populate_loop_schedule(config, work_plan, result);
  result.initialized = true;

  // Phase 1: Calibration & excluded warmup.
  const bool explicit_iterations = config.iterations.has_value();
  if (explicit_iterations) {
    for (CpuBandwidthTarget target : work_plan.active_targets) {
      const size_t t_idx = cpu_bandwidth_target_index(target);
      const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
      for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
        const auto op = kOperations[op_i];
        const size_t plan_ref = register_operation_plan(
            result, tplan, op, *config.iterations, true);
        if (!result.operation_plans.at(plan_ref).valid) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code = result.operation_plans.at(plan_ref).reason_code;
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
        result.frozen_plan_refs[t_idx][op_i] = plan_ref;
      }
    }
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (CpuBandwidthTarget target : work_plan.active_targets) {
      const size_t t_idx = cpu_bandwidth_target_index(target);
      const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
      for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        const size_t plan_ref = *result.frozen_plan_refs[t_idx][op_i];
        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, tplan, plan_ref, "frozen_measurement_warmup",
                resources, timer, result, hooks, elapsed)) {
          const auto& last_attempt =
              result.calibration_attempts[t_idx][op_i].back();
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code = last_attempt.reason_code;
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }
    }
  } else {
    // Automatic calibration per (target, op), then freeze and run frozen warmup.
    for (CpuBandwidthTarget target : work_plan.active_targets) {
      const size_t t_idx = cpu_bandwidth_target_index(target);
      const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
      for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
        const auto op = kOperations[op_i];
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        const size_t max_passes =
            calculate_cpu_bandwidth_max_passes(tplan.bytes, op);
        const size_t mult = (op == CpuBandwidthOperation::Copy)
                                ? Constants::COPY_OPERATION_MULTIPLIER
                                : 1;
        size_t bytes_per_pass = 0;
        if (!NumericUtils::checked_multiply(tplan.bytes, mult, bytes_per_pass) ||
            max_passes == 0) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code = CpuBandwidthWorkPlanReason::WORK_CAP_EXCEEDED;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        size_t passes = NumericUtils::calculate_minimum_pilot_count(
            bytes_per_pass, Constants::BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES,
            max_passes);
        if (passes == 0) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code = "calibration-scaling-failed";
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        size_t latest_plan_ref =
            register_operation_plan(result, tplan, op, passes, false);
        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, tplan, latest_plan_ref,
                "calibration_shape_warmup", resources, timer, result, hooks,
                elapsed)) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[t_idx][op_i].back().reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        if (!execute_excluded_attempt(config, work_plan, tplan, latest_plan_ref,
                                      "pilot", resources, timer, result, hooks,
                                      elapsed)) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[t_idx][op_i].back().reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        bool single_unit_warmed = (passes == 1);
        for (size_t corr = 0;
             !duration_in_target_window(elapsed) &&
             corr < Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;
             ++corr) {
          if (passes == 1 &&
              elapsed > Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS) {
            break;
          }
          const size_t corrected =
              NumericUtils::calculate_duration_scaled_count(
                  elapsed, passes,
                  Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS, 1,
                  max_passes, 1);
          if (corrected == 0) {
            result.status = CpuBandwidthRunStatus::Failed;
            result.reason_code = "calibration-scaling-failed";
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result,
                                    CpuBandwidthMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result,
                              CpuBandwidthCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }
          if (corrected == passes) {
            break;
          }
          passes = corrected;
          latest_plan_ref =
              register_operation_plan(result, tplan, op, passes, false);

          if (stop_requested(hooks)) {
            result.interruption_requested = true;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(
                result, CpuBandwidthMeasurementStatus::Interrupted,
                "interruption-before-task");
            recompute_aggregates_and_completion(config, work_plan, result);
            const bool ck_ok = invoke_checkpoint(
                result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
            return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
          }

          const bool single_unit = (passes == 1);
          if (single_unit && !single_unit_warmed) {
            if (!execute_excluded_attempt(
                    config, work_plan, tplan, latest_plan_ref,
                    "single_unit_confirmation_warmup", resources, timer, result,
                    hooks, elapsed)) {
              result.status = CpuBandwidthRunStatus::Failed;
              result.reason_code =
                  result.calibration_attempts[t_idx][op_i].back().reason_code;
              assign_frozen_plans_to_measurements(config, work_plan, result);
              finalize_remaining_rows(result,
                                      CpuBandwidthMeasurementStatus::Failed,
                                      result.reason_code);
              recompute_aggregates_and_completion(config, work_plan, result);
              invoke_checkpoint(result,
                                CpuBandwidthCheckpointKind::CommandTerminal,
                                hooks);
              return EXIT_FAILURE;
            }
            single_unit_warmed = true;
            if (stop_requested(hooks)) {
              result.interruption_requested = true;
              assign_frozen_plans_to_measurements(config, work_plan, result);
              finalize_remaining_rows(
                  result, CpuBandwidthMeasurementStatus::Interrupted,
                  "interruption-before-task");
              recompute_aggregates_and_completion(config, work_plan, result);
              const bool ck_ok = invoke_checkpoint(
                  result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
              return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
            }
          }

          const std::string purpose =
              single_unit ? "single_unit_confirmation" : "correction";
          if (!execute_excluded_attempt(config, work_plan, tplan,
                                        latest_plan_ref, purpose, resources,
                                        timer, result, hooks, elapsed)) {
            result.status = CpuBandwidthRunStatus::Failed;
            result.reason_code =
                result.calibration_attempts[t_idx][op_i].back().reason_code;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result,
                                    CpuBandwidthMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result,
                              CpuBandwidthCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }
        }

        result.frozen_plan_refs[t_idx][op_i] = latest_plan_ref;
      }
    }

    // Freeze plans atomically before loop 0 and run frozen_measurement_warmup.
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (CpuBandwidthTarget target : work_plan.active_targets) {
      const size_t t_idx = cpu_bandwidth_target_index(target);
      const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
      for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        const size_t plan_ref = *result.frozen_plan_refs[t_idx][op_i];
        double elapsed = 0.0;
        if (!execute_excluded_attempt(
                config, work_plan, tplan, plan_ref, "frozen_measurement_warmup",
                resources, timer, result, hooks, elapsed)) {
          result.status = CpuBandwidthRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[t_idx][op_i].back().reason_code;
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }
    }
  }

  // Phase 2: Timed measurement loops in cyclic Latin-square order.
  for (auto& loop : result.loops) {
    for (size_t pos = 0; pos < loop.measurement_indexes.size(); ++pos) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(
            result, CpuBandwidthMeasurementStatus::Interrupted,
            "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      if (hooks.before_measurement_hook) {
        hooks.before_measurement_hook(loop.loop_index, pos);
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
      }

      CpuBandwidthMeasurementRow& row =
          result.measurements[loop.measurement_indexes[pos]];
      const size_t t_idx = cpu_bandwidth_target_index(row.target);
      const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
      const CpuBandwidthOperationPlan& op_plan =
          result.operation_plans.at(*row.plan_ref);

      std::optional<uint64_t> write_seed;
      std::optional<uint64_t> write_pattern;
      if (row.operation == CpuBandwidthOperation::Write) {
        auto pair = allocate_unique_write_pattern(result, op_plan.op_seed);
        write_seed = pair.first;
        write_pattern = pair.second;
      }
      const uint64_t sentinel_pattern =
          row.operation == CpuBandwidthOperation::Copy
              ? allocate_copy_sentinel(result, work_plan.pre_copy_sentinel_seed)
              : 0ULL;

      row.attempted = true;
      ++result.counters.attempted_measurements;
      loop.realized_order.push_back({row.target, row.operation});

      row.execution = execute_cpu_bandwidth_operation(
          config, tplan, op_plan, resources.targets[t_idx], timer, write_seed,
          write_pattern, sentinel_pattern, hooks.executor_test_control);
      row.duration_quality = classify_cpu_bandwidth_duration_quality(
          row.execution.elapsed_seconds, row.execution.timing_valid);

      ++result.counters.terminal_measurements;
      if (row.execution.valid) {
        const long double exact_bytes =
            static_cast<long double>(op_plan.exact_payload_bytes);
        const long double elapsed =
            static_cast<long double>(row.execution.elapsed_seconds);
        const double bw = static_cast<double>(exact_bytes / elapsed / 1.0e9L);
        if (std::isfinite(bw) && bw > 0.0) {
          row.status = CpuBandwidthMeasurementStatus::Measured;
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
          row.status = CpuBandwidthMeasurementStatus::Invalid;
          row.reason_code = "invalid-derived-metric";
        }
      } else {
        row.status = CpuBandwidthMeasurementStatus::Invalid;
        row.reason_code = row.execution.reason_code;
      }

      if (row.status != CpuBandwidthMeasurementStatus::Measured) {
        result.status = CpuBandwidthRunStatus::Failed;
        result.reason_code = row.reason_code;
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(
              result, CpuBandwidthMeasurementStatus::Interrupted,
              "interruption-before-task");
        } else {
          finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                  "not-run-after-runtime-failure");
        }
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      const bool has_remaining_work =
          (pos + 1 < loop.measurement_indexes.size()) ||
          (loop.loop_index + 1 < config.loop_count);
      if (has_remaining_work && stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(
            result, CpuBandwidthMeasurementStatus::Interrupted,
            "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuBandwidthCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    }

    // End of loop bounded checkpoint (for intermediate loops when file target).
    recompute_aggregates_and_completion(config, work_plan, result);
    if (hooks.progress_snapshots &&
        ((loop.loop_index + 1) % result.snapshot_interval_loops == 0) &&
        (loop.loop_index + 1 < config.loop_count)) {
      if (!invoke_checkpoint(result, CpuBandwidthCheckpointKind::Progress,
                             hooks)) {
        finalize_remaining_rows(result, CpuBandwidthMeasurementStatus::Failed,
                                "not-run-after-runtime-failure");
        recompute_aggregates_and_completion(config, work_plan, result);
        return EXIT_FAILURE;
      }
    }
  }

  recompute_aggregates_and_completion(config, work_plan, result);
  if (!invoke_checkpoint(result, CpuBandwidthCheckpointKind::CommandTerminal,
                         hooks)) {
    recompute_aggregates_and_completion(config, work_plan, result);
    return EXIT_FAILURE;
  }

  return (result.status == CpuBandwidthRunStatus::Failed ||
          result.status == CpuBandwidthRunStatus::Unsupported)
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
