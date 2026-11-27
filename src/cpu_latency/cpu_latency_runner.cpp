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
 * @file cpu_latency_runner.cpp
 * @brief Excluded calibration, atomic plan freezing, cyclic schedule, and checkpoints
 */

#include "cpu_latency/cpu_latency_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "core/audit/checkpoint_policy.h"
#include "core/config/constants.h"
#include "utils/cyclic_order.h"
#include "utils/numeric_utils.h"

namespace {

constexpr size_t kMinPilotAccesses = 16384;

bool duration_in_target_window(double elapsed_seconds) noexcept {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS;
}

bool stop_requested(const CpuLatencyRunnerHooks& hooks) {
  return hooks.stop_requested && hooks.stop_requested();
}

size_t register_target_plan(CpuLatencyResult& result,
                            const CpuLatencyTargetPlan& target_plan,
                            size_t complete_chain_cycles,
                            bool explicit_iterations) {
  for (size_t i = 0; i < result.target_plans.size(); ++i) {
    const auto& existing = result.target_plans[i];
    if (existing.target == target_plan.target &&
        existing.complete_chain_cycles == complete_chain_cycles &&
        existing.explicit_iterations == explicit_iterations) {
      return i;
    }
  }
  CpuLatencyTargetExecutionPlan exec_plan =
      build_cpu_latency_target_execution_plan(
          target_plan, complete_chain_cycles, explicit_iterations);
  exec_plan.plan_ref = result.target_plans.size();
  result.target_plans.push_back(std::move(exec_plan));
  return result.target_plans.size() - 1;
}

void populate_loop_schedule(const CpuLatencyConfig& config,
                            const CpuLatencyWorkPlan& work_plan,
                            CpuLatencyResult& result) {
  const size_t num_targets = work_plan.active_targets.size();
  result.counters.planned_loops = config.loop_count;
  result.counters.planned_measurements = config.loop_count * num_targets;
  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);

  const bool samples_applicable = config.latency_samples > 0;

  for (size_t loop_idx = 0; loop_idx < config.loop_count; ++loop_idx) {
    CpuLatencyLoopRecord loop_rec;
    loop_rec.loop_index = loop_idx;
    const std::vector<size_t> t_order =
        build_cyclic_order(num_targets, loop_idx);

    for (size_t t_pos = 0; t_pos < num_targets; ++t_pos) {
      const CpuLatencyTarget target = work_plan.active_targets[t_order[t_pos]];
      loop_rec.planned_order.push_back(target);

      CpuLatencyMeasurementRow row;
      row.measurement_id = result.measurements.size();
      row.target = target;
      row.loop_index = loop_idx;
      row.target_order_position = t_pos;
      row.execution.validation_checks =
          make_default_cpu_latency_checks(samples_applicable);
      loop_rec.measurement_indexes.push_back(row.measurement_id);
      result.measurements.push_back(std::move(row));
    }
    result.loops.push_back(std::move(loop_rec));
  }
}

void assign_frozen_plans_to_measurements(const CpuLatencyConfig& config,
                                         const CpuLatencyWorkPlan& work_plan,
                                         CpuLatencyResult& result) {
  uint64_t planned_accesses_per_loop = 0;
  bool all_frozen = !work_plan.active_targets.empty();

  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    const auto& ref = result.frozen_plan_refs[t_idx];
    if (!ref.has_value()) {
      all_frozen = false;
      break;
    }
    const auto& exec_plan = result.target_plans.at(*ref);
    planned_accesses_per_loop += static_cast<uint64_t>(exec_plan.access_count);
  }

  for (auto& row : result.measurements) {
    const size_t t_idx = cpu_latency_target_index(row.target);
    row.plan_ref = result.frozen_plan_refs[t_idx];
  }

  if (all_frozen) {
    result.counters.planned_accesses =
        planned_accesses_per_loop * static_cast<uint64_t>(config.loop_count);
  } else {
    result.counters.planned_accesses = 0;
  }
}

void finalize_remaining_rows(CpuLatencyResult& result,
                             CpuLatencyMeasurementStatus terminal_status,
                             const std::string& reason_code) {
  for (auto& row : result.measurements) {
    if (!row.attempted && row.status == CpuLatencyMeasurementStatus::NotRun) {
      row.status = terminal_status;
      row.reason_code = reason_code;
      if (terminal_status != CpuLatencyMeasurementStatus::NotRun) {
        ++result.counters.terminal_measurements;
      }
    }
  }
}

void recompute_aggregates_and_completion(const CpuLatencyConfig& config,
                                         const CpuLatencyWorkPlan& work_plan,
                                         CpuLatencyResult& result) {
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    CpuLatencyTargetAggregate& agg = result.aggregates[t_idx];
    agg = CpuLatencyTargetAggregate{};
    agg.target = target;
    agg.cv_warning_threshold_pct = kCpuLatencyCvWarningPct;
    agg.pooled_applicable = (config.latency_samples > 0);

    std::vector<double> values;
    for (const auto& row : result.measurements) {
      if (row.target == target &&
          row.status == CpuLatencyMeasurementStatus::Measured &&
          row.latency_ns.has_value()) {
        agg.accepted_measurement_ids.push_back(row.measurement_id);
        values.push_back(*row.latency_ns);
        if (agg.pooled_applicable && row.execution.sample_windows.has_value()) {
          const auto& sns = row.execution.sample_windows->samples_ns;
          agg.pooled_samples_ns.insert(agg.pooled_samples_ns.end(), sns.begin(),
                                       sns.end());
        }
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
               kCpuLatencyCvWarningPct);
    }
    if (agg.pooled_applicable && !agg.pooled_samples_ns.empty()) {
      agg.pooled_statistics =
          calculate_descriptive_statistics(agg.pooled_samples_ns);
    }
  }

  result.locality_aggregate = CpuLatencyLocalityAggregate{};
  if (config.compare_locality) {
    std::vector<double> loc16k;
    std::vector<double> loc_global;
    for (const auto& rnd : result.locality_rounds) {
      if (rnd.locality_16k.status == "measured" &&
          rnd.locality_16k.latency_ns.has_value()) {
        loc16k.push_back(*rnd.locality_16k.latency_ns);
      }
      if (rnd.locality_global.status == "measured" &&
          rnd.locality_global.latency_ns.has_value()) {
        loc_global.push_back(*rnd.locality_global.latency_ns);
      }
    }
    if (!loc16k.empty() && !loc_global.empty()) {
      const DescriptiveStatistics s16 = calculate_descriptive_statistics(loc16k);
      const DescriptiveStatistics sg =
          calculate_descriptive_statistics(loc_global);
      result.locality_aggregate.locality_16k_latency_ns = s16.median;
      result.locality_aggregate.global_random_latency_ns = sg.median;
      result.locality_aggregate.locality_latency_delta_ns =
          sg.median - s16.median;
    }
  }

  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  for (const auto& loop : result.loops) {
    if (!loop.realized_order.empty()) {
      ++attempted_loops;
    }
    bool all_measured = !loop.measurement_indexes.empty();
    for (size_t idx : loop.measurement_indexes) {
      if (result.measurements[idx].status !=
          CpuLatencyMeasurementStatus::Measured) {
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

  bool locality_complete = true;
  if (config.compare_locality) {
    const size_t expected_rounds =
        config.loop_count * kCpuLatencyCompareLocalityRoundsPerLoop;
    locality_complete = (result.locality_rounds.size() == expected_rounds);
    if (locality_complete) {
      for (const auto& rnd : result.locality_rounds) {
        if (rnd.locality_16k.status != "measured" ||
            rnd.locality_global.status != "measured") {
          locality_complete = false;
          break;
        }
      }
    }
  }

  result.results_complete =
      result.counters.planned_measurements > 0 &&
      result.counters.measured_measurements ==
          result.counters.planned_measurements;

  const size_t num_targets = work_plan.active_targets.size();
  result.target_order_balance_complete =
      result.results_complete && num_targets > 0 &&
      (config.loop_count % num_targets == 0);

  if (result.status != CpuLatencyRunStatus::Failed &&
      result.status != CpuLatencyRunStatus::Unsupported) {
    if (result.results_complete && locality_complete) {
      result.status = CpuLatencyRunStatus::Complete;
      result.reason_code = "complete";
    } else if (result.interruption_requested) {
      result.status = CpuLatencyRunStatus::Interrupted;
      result.reason_code = "interruption-requested";
    } else if (result.counters.terminal_measurements > 0 ||
               result.counters.attempted_measurements > 0) {
      result.status = CpuLatencyRunStatus::Partial;
      result.reason_code = "partial-results";
    } else {
      result.status = CpuLatencyRunStatus::NotStarted;
      result.reason_code = "not-started";
    }
  }

  result.run_accepted = result.status == CpuLatencyRunStatus::Complete &&
                        result.results_complete &&
                        !result.checkpoint_failed;
}

bool invoke_checkpoint(CpuLatencyResult& result,
                       CpuLatencyCheckpointKind kind,
                       const CpuLatencyRunnerHooks& hooks) {
  result.snapshot_request =
      (kind == CpuLatencyCheckpointKind::CommandTerminal) ? "terminal"
                                                          : "progress";
  if (kind == CpuLatencyCheckpointKind::CommandTerminal) {
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
    result.status = CpuLatencyRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
    result.run_accepted = false;
    return false;
  }

  if (kind == CpuLatencyCheckpointKind::CommandTerminal) {
    result.terminal_checkpoint_completed = true;
  }
  return true;
}

bool execute_excluded_attempt(const CpuLatencyTargetPlan& target_plan,
                              size_t plan_ref,
                              const std::string& purpose,
                              CpuLatencyResourceSet& resources,
                              HighResTimer& timer,
                              CpuLatencyResult& result,
                              const CpuLatencyRunnerHooks& hooks,
                              double& out_elapsed) {
  const CpuLatencyTargetExecutionPlan& exec_plan =
      result.target_plans.at(plan_ref);
  const size_t t_idx = cpu_latency_target_index(exec_plan.target);
  auto& attempts_vec = result.calibration_attempts[t_idx];

  CpuLatencyCalibrationAttempt attempt;
  attempt.attempt_index = attempts_vec.size();
  attempt.target = exec_plan.target;
  attempt.plan_ref = plan_ref;
  attempt.purpose = purpose;
  attempt.execution = execute_cpu_latency_attempt(
      target_plan, exec_plan, resources.targets[t_idx], timer,
      /*requested_samples=*/0, hooks.executor_test_control);
  attempt.duration_quality = classify_cpu_latency_duration_quality(
      attempt.execution.elapsed_seconds, attempt.execution.timing_valid);
  attempt.terminal = true;
  attempt.valid = attempt.execution.valid;
  attempt.reason_code = attempt.execution.reason_code;
  out_elapsed = attempt.execution.elapsed_seconds;

  attempts_vec.push_back(std::move(attempt));
  return attempts_vec.back().valid;
}

}  // namespace

const char* cpu_latency_measurement_status_to_string(
    CpuLatencyMeasurementStatus status) noexcept {
  switch (status) {
    case CpuLatencyMeasurementStatus::NotRun:
      return "not_run";
    case CpuLatencyMeasurementStatus::Measured:
      return "measured";
    case CpuLatencyMeasurementStatus::Invalid:
      return "invalid";
    case CpuLatencyMeasurementStatus::Failed:
      return "failed";
    case CpuLatencyMeasurementStatus::Interrupted:
      return "interrupted";
  }
  return "not_run";
}

const char* cpu_latency_run_status_to_string(
    CpuLatencyRunStatus status) noexcept {
  switch (status) {
    case CpuLatencyRunStatus::NotStarted:
      return "not_started";
    case CpuLatencyRunStatus::Complete:
      return "complete";
    case CpuLatencyRunStatus::Partial:
      return "partial";
    case CpuLatencyRunStatus::Interrupted:
      return "interrupted";
    case CpuLatencyRunStatus::Unsupported:
      return "unsupported";
    case CpuLatencyRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

std::string classify_cpu_latency_duration_quality(double elapsed_seconds,
                                                  bool timing_valid) noexcept {
  if (!timing_valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0) {
    return "unavailable";
  }
  if (elapsed_seconds < Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS) {
    return "below-target-window";
  }
  if (elapsed_seconds > Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS) {
    return "above-target-window";
  }
  return "within-target-window";
}

void initialize_cpu_latency_setup_failure_result(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    CpuLatencyRunStatus status,
    std::string_view reason_code,
    bool interruption_requested,
    CpuLatencyResult& result) {
  result = CpuLatencyResult{};
  result.status = status;
  result.reason_code = std::string(reason_code);
  result.interruption_requested = interruption_requested;
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);
  populate_loop_schedule(config, work_plan, result);
  const CpuLatencyMeasurementStatus row_status =
      interruption_requested ? CpuLatencyMeasurementStatus::Interrupted
                             : CpuLatencyMeasurementStatus::Failed;
  finalize_remaining_rows(result, row_status, result.reason_code);
  recompute_aggregates_and_completion(config, work_plan, result);
  result.snapshot_request = "terminal";
  result.initialized = true;
}

int run_cpu_latency_suite(const CpuLatencyConfig& config,
                          const CpuLatencyWorkPlan& work_plan,
                          CpuLatencyResourceSet& resources,
                          CpuLatencyResult& result,
                          const CpuLatencyRunnerHooks& hooks) {
  result = CpuLatencyResult{};
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);

  if (!work_plan.valid || !resources.valid) {
    initialize_cpu_latency_setup_failure_result(
        config, work_plan, CpuLatencyRunStatus::Failed,
        !work_plan.valid ? work_plan.reason_code : resources.reason_code, false,
        result);
    invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
    return EXIT_FAILURE;
  }

  auto timer_opt = HighResTimer::create();
  if (!timer_opt.has_value()) {
    initialize_cpu_latency_setup_failure_result(
        config, work_plan, CpuLatencyRunStatus::Failed, "timer-creation-failed",
        false, result);
    invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
    return EXIT_FAILURE;
  }
  HighResTimer& timer = *timer_opt;

  populate_loop_schedule(config, work_plan, result);
  result.initialized = true;

  // Phase 1: Calibration & excluded warmup.
  const bool explicit_iterations = config.iterations.has_value();
  if (explicit_iterations) {
    for (CpuLatencyTarget target : work_plan.active_targets) {
      const size_t t_idx = cpu_latency_target_index(target);
      const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
      const size_t plan_ref =
          register_target_plan(result, tplan, *config.iterations, true);
      if (!result.target_plans.at(plan_ref).valid) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code = result.target_plans.at(plan_ref).reason_code;
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }
      result.frozen_plan_refs[t_idx] = plan_ref;
    }
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (CpuLatencyTarget target : work_plan.active_targets) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }
      const size_t t_idx = cpu_latency_target_index(target);
      const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
      const size_t plan_ref = *result.frozen_plan_refs[t_idx];
      double elapsed = 0.0;
      if (!execute_excluded_attempt(tplan, plan_ref, "frozen_measurement_warmup",
                                    resources, timer, result, hooks, elapsed)) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code =
            result.calibration_attempts[t_idx].back().reason_code;
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }
    }
  } else {
    // Automatic calibration per active target, then atomic freeze & warmup.
    for (CpuLatencyTarget target : work_plan.active_targets) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      const size_t t_idx = cpu_latency_target_index(target);
      const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
      const size_t max_cycles =
          calculate_cpu_latency_max_cycles(tplan.node_count);
      if (max_cycles == 0) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code = CpuLatencyWorkPlanReason::WORK_CAP_EXCEEDED;
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      const size_t min_cycles =
          std::min<size_t>(kCpuLatencyMinCompleteCycles, max_cycles);
      size_t cycles = NumericUtils::calculate_minimum_pilot_count(
          tplan.node_count, kMinPilotAccesses, max_cycles);
      if (cycles == 0) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code = "calibration-scaling-failed";
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      size_t latest_plan_ref =
          register_target_plan(result, tplan, cycles, false);
      double elapsed = 0.0;
      if (!execute_excluded_attempt(tplan, latest_plan_ref,
                                    "calibration_shape_warmup", resources,
                                    timer, result, hooks, elapsed)) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code =
            result.calibration_attempts[t_idx].back().reason_code;
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      if (!execute_excluded_attempt(tplan, latest_plan_ref, "pilot", resources,
                                    timer, result, hooks, elapsed)) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code =
            result.calibration_attempts[t_idx].back().reason_code;
        assign_frozen_plans_to_measurements(config, work_plan, result);
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      bool min_cycles_warmed = (cycles == min_cycles);
      for (size_t corr = 0;
           corr < Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS; ++corr) {
        if (duration_in_target_window(elapsed) && cycles >= min_cycles) {
          break;
        }
        if (cycles == min_cycles &&
            elapsed > Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS) {
          break;
        }
        const size_t corrected =
            NumericUtils::calculate_duration_scaled_count(
                elapsed, cycles, Constants::BENCHMARK_LATENCY_TARGET_SECONDS,
                min_cycles, max_cycles, 1);
        if (corrected == 0) {
          result.status = CpuLatencyRunStatus::Failed;
          result.reason_code = "calibration-scaling-failed";
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
        if (corrected == cycles) {
          break;
        }
        cycles = corrected;
        latest_plan_ref = register_target_plan(result, tplan, cycles, false);

        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result,
                                  CpuLatencyMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        const bool at_min_cycles = (cycles == min_cycles);
        if (at_min_cycles && !min_cycles_warmed) {
          if (!execute_excluded_attempt(
                  tplan, latest_plan_ref, "single_unit_confirmation_warmup",
                  resources, timer, result, hooks, elapsed)) {
            result.status = CpuLatencyRunStatus::Failed;
            result.reason_code =
                result.calibration_attempts[t_idx].back().reason_code;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                    result.reason_code);
            recompute_aggregates_and_completion(config, work_plan, result);
            invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                              hooks);
            return EXIT_FAILURE;
          }
          min_cycles_warmed = true;
          if (stop_requested(hooks)) {
            result.interruption_requested = true;
            assign_frozen_plans_to_measurements(config, work_plan, result);
            finalize_remaining_rows(result,
                                    CpuLatencyMeasurementStatus::Interrupted,
                                    "interruption-before-task");
            recompute_aggregates_and_completion(config, work_plan, result);
            const bool ck_ok = invoke_checkpoint(
                result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
            return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
          }
        }

        const std::string purpose =
            at_min_cycles ? "single_unit_confirmation" : "correction";
        if (!execute_excluded_attempt(tplan, latest_plan_ref, purpose,
                                      resources, timer, result, hooks,
                                      elapsed)) {
          result.status = CpuLatencyRunStatus::Failed;
          result.reason_code =
              result.calibration_attempts[t_idx].back().reason_code;
          assign_frozen_plans_to_measurements(config, work_plan, result);
          finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                  result.reason_code);
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }

      result.frozen_plan_refs[t_idx] = latest_plan_ref;
    }

    // Freeze plans atomically before loop 0 and run frozen_measurement_warmup.
    assign_frozen_plans_to_measurements(config, work_plan, result);

    for (CpuLatencyTarget target : work_plan.active_targets) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }
      const size_t t_idx = cpu_latency_target_index(target);
      const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
      const size_t plan_ref = *result.frozen_plan_refs[t_idx];
      double elapsed = 0.0;
      if (!execute_excluded_attempt(tplan, plan_ref, "frozen_measurement_warmup",
                                    resources, timer, result, hooks, elapsed)) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code =
            result.calibration_attempts[t_idx].back().reason_code;
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                result.reason_code);
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }
    }
  }

  // Phase 2: Timed measurement loops in cyclic target order.
  for (auto& loop : result.loops) {
    if (config.compare_locality) {
      const size_t main_idx =
          cpu_latency_target_index(CpuLatencyTarget::MainMemory);
      const CpuLatencyTargetExecutionPlan& main_exec_plan =
          result.target_plans.at(*result.frozen_plan_refs[main_idx]);
      for (size_t r_idx = 0; r_idx < kCpuLatencyCompareLocalityRoundsPerLoop;
           ++r_idx) {
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result,
                                  CpuLatencyMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (hooks.before_locality_round_hook) {
          hooks.before_locality_round_hook(loop.loop_index, r_idx);
          if (stop_requested(hooks)) {
            result.interruption_requested = true;
            finalize_remaining_rows(result,
                                    CpuLatencyMeasurementStatus::Interrupted,
                                    "interruption-before-task");
            recompute_aggregates_and_completion(config, work_plan, result);
            const bool ck_ok = invoke_checkpoint(
                result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
            return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
          }
        }

        CpuLatencyLocalityRound rnd;
        rnd.loop_index = loop.loop_index;
        rnd.round_index = r_idx;
        rnd.locality_16k = execute_cpu_latency_locality_subrun(
            *work_plan.locality_16k_plan, resources.locality_16k_identity,
            resources.locality_16k_indices,
            main_exec_plan.complete_chain_cycles, resources, timer,
            hooks.executor_test_control);
        if (rnd.locality_16k.status != "measured") {
          result.locality_rounds.push_back(std::move(rnd));
          result.status = CpuLatencyRunStatus::Failed;
          result.reason_code = result.locality_rounds.back().locality_16k.reason_code;
          finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                  "not-run-after-runtime-failure");
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }

        if (stop_requested(hooks)) {
          rnd.locality_global.status = "interrupted";
          rnd.locality_global.attempted = false;
          rnd.locality_global.reason_code = "interruption-before-task";
          result.locality_rounds.push_back(std::move(rnd));
          result.interruption_requested = true;
          finalize_remaining_rows(result,
                                  CpuLatencyMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        rnd.locality_global = execute_cpu_latency_locality_subrun(
            *work_plan.locality_global_plan, resources.locality_global_identity,
            resources.locality_global_indices,
            main_exec_plan.complete_chain_cycles, resources, timer,
            hooks.executor_test_control);
        const bool global_ok = (rnd.locality_global.status == "measured");
        const std::string global_reason = rnd.locality_global.reason_code;
        result.locality_rounds.push_back(std::move(rnd));
        if (!global_ok) {
          result.status = CpuLatencyRunStatus::Failed;
          result.reason_code = global_reason;
          finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                  "not-run-after-runtime-failure");
          recompute_aggregates_and_completion(config, work_plan, result);
          invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                            hooks);
          return EXIT_FAILURE;
        }
      }
    }

    for (size_t pos = 0; pos < loop.measurement_indexes.size(); ++pos) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      if (hooks.before_measurement_hook) {
        hooks.before_measurement_hook(loop.loop_index, pos);
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result,
                                  CpuLatencyMeasurementStatus::Interrupted,
                                  "interruption-before-task");
          recompute_aggregates_and_completion(config, work_plan, result);
          const bool ck_ok = invoke_checkpoint(
              result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
          return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        }
      }

      CpuLatencyMeasurementRow& row =
          result.measurements[loop.measurement_indexes[pos]];
      const size_t t_idx = cpu_latency_target_index(row.target);
      const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
      const CpuLatencyTargetExecutionPlan& exec_plan =
          result.target_plans.at(*row.plan_ref);

      row.attempted = true;
      ++result.counters.attempted_measurements;
      loop.realized_order.push_back(row.target);

      row.execution = execute_cpu_latency_attempt(
          tplan, exec_plan, resources.targets[t_idx], timer,
          config.latency_samples, hooks.executor_test_control);
      row.duration_quality = classify_cpu_latency_duration_quality(
          row.execution.elapsed_seconds, row.execution.timing_valid);

      ++result.counters.terminal_measurements;
      if (row.execution.valid) {
        const long double accesses =
            static_cast<long double>(exec_plan.access_count);
        const long double elapsed =
            static_cast<long double>(row.execution.elapsed_seconds);
        const double lat_ns = static_cast<double>((elapsed * 1.0e9L) / accesses);
        if (std::isfinite(lat_ns) && lat_ns > 0.0) {
          row.status = CpuLatencyMeasurementStatus::Measured;
          row.reason_code = "measured";
          row.completed_accesses = exec_plan.access_count;
          row.completed_chain_cycles = exec_plan.complete_chain_cycles;
          row.latency_ns = lat_ns;
          ++result.counters.measured_measurements;
          result.counters.completed_accesses +=
              static_cast<uint64_t>(exec_plan.access_count);
        } else {
          row.status = CpuLatencyMeasurementStatus::Invalid;
          row.reason_code = "invalid-derived-metric";
        }
      } else {
        row.status = CpuLatencyMeasurementStatus::Invalid;
        row.reason_code = row.execution.reason_code;
      }

      if (row.status != CpuLatencyMeasurementStatus::Measured) {
        result.status = CpuLatencyRunStatus::Failed;
        result.reason_code = row.reason_code;
        if (stop_requested(hooks)) {
          result.interruption_requested = true;
          finalize_remaining_rows(result,
                                  CpuLatencyMeasurementStatus::Interrupted,
                                  "interruption-before-task");
        } else {
          finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                  "not-run-after-runtime-failure");
        }
        recompute_aggregates_and_completion(config, work_plan, result);
        invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                          hooks);
        return EXIT_FAILURE;
      }

      const bool has_remaining_work =
          (pos + 1 < loop.measurement_indexes.size()) ||
          (loop.loop_index + 1 < config.loop_count);
      if (has_remaining_work && stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_rows(result,
                                CpuLatencyMeasurementStatus::Interrupted,
                                "interruption-before-task");
        recompute_aggregates_and_completion(config, work_plan, result);
        const bool ck_ok = invoke_checkpoint(
            result, CpuLatencyCheckpointKind::CommandTerminal, hooks);
        return ck_ok ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    }

    recompute_aggregates_and_completion(config, work_plan, result);
    if (hooks.progress_snapshots &&
        ((loop.loop_index + 1) % result.snapshot_interval_loops == 0) &&
        (loop.loop_index + 1 < config.loop_count)) {
      if (!invoke_checkpoint(result, CpuLatencyCheckpointKind::Progress,
                             hooks)) {
        finalize_remaining_rows(result, CpuLatencyMeasurementStatus::Failed,
                                "not-run-after-runtime-failure");
        recompute_aggregates_and_completion(config, work_plan, result);
        return EXIT_FAILURE;
      }
    }
  }

  recompute_aggregates_and_completion(config, work_plan, result);
  if (!invoke_checkpoint(result, CpuLatencyCheckpointKind::CommandTerminal,
                         hooks)) {
    recompute_aggregates_and_completion(config, work_plan, result);
    return EXIT_FAILURE;
  }

  return (result.status == CpuLatencyRunStatus::Failed ||
          result.status == CpuLatencyRunStatus::Unsupported)
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
