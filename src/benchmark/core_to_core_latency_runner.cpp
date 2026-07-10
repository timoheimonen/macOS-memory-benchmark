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
 * @file core_to_core_latency_runner.cpp
 * @brief Calibrated and balanced standalone core-to-core handoff benchmark
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "benchmark/core_to_core_latency.h"
#include "benchmark/core_to_core_latency_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <pthread/qos.h>

#include "asm/asm_functions.h"
#include "benchmark/core_to_core_latency_json.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/signal/signal_handler.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"

namespace {

struct SummaryStats {
  double average = 0.0;
  double min = 0.0;
  double max = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double stddev = 0.0;
  double coefficient_of_variation_pct = 0.0;
  double median_absolute_deviation = 0.0;
};

struct alignas(Constants::CACHE_LINE_SIZE_BYTES) SharedTurn {
  uint32_t value = Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE;
};

struct alignas(Constants::CACHE_LINE_SIZE_BYTES) SharedFlags {
  std::atomic<bool> start{false};
  std::atomic<bool> cancel{false};
  std::atomic<int> ready_threads{0};
};

struct SharedPingPongState {
  SharedTurn turn;
  SharedFlags flags;
};

double calculate_average(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }

  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

SummaryStats calculate_summary_stats(const std::vector<double>& values) {
  SummaryStats stats;
  if (values.empty()) {
    return stats;
  }

  stats.average = calculate_average(values);
  stats.min = *std::min_element(values.begin(), values.end());
  stats.max = *std::max_element(values.begin(), values.end());

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const size_t count = sorted.size();
  const auto percentile = [&sorted, count](double percentile_value) {
    if (count == 1) {
      return sorted[0];
    }
    const double index = percentile_value * static_cast<double>(count - 1);
    const size_t lower = static_cast<size_t>(index);
    const size_t upper = lower + 1;
    if (upper >= count) {
      return sorted[count - 1];
    }
    const double weight = index - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
  };

  stats.median = percentile(0.50);
  stats.p90 = percentile(0.90);
  stats.p95 = percentile(0.95);
  stats.p99 = percentile(0.99);

  std::vector<double> absolute_deviations;
  absolute_deviations.reserve(values.size());
  for (double value : values) {
    absolute_deviations.push_back(std::abs(value - stats.median));
  }
  std::sort(absolute_deviations.begin(), absolute_deviations.end());
  if (absolute_deviations.size() % 2 == 0) {
    const size_t upper = absolute_deviations.size() / 2;
    stats.median_absolute_deviation = (absolute_deviations[upper - 1] + absolute_deviations[upper]) * 0.5;
  } else {
    stats.median_absolute_deviation = absolute_deviations[absolute_deviations.size() / 2];
  }

  if (values.size() > 1) {
    double variance_sum = 0.0;
    for (double value : values) {
      const double difference = value - stats.average;
      variance_sum += difference * difference;
    }
    stats.stddev = std::sqrt(variance_sum / static_cast<double>(values.size() - 1));
    if (stats.average != 0.0) {
      stats.coefficient_of_variation_pct = std::abs(stats.stddev / stats.average) * 100.0;
    }
  }

  return stats;
}

bool positive_finite(double value) { return value > 0.0 && std::isfinite(value); }

std::string classify_duration_quality(double elapsed_seconds) {
  if (!positive_finite(elapsed_seconds)) {
    return "invalid-elapsed";
  }
  if (elapsed_seconds < Constants::CORE_TO_CORE_HEADLINE_MIN_SECONDS) {
    return "below-target-window";
  }
  if (elapsed_seconds > Constants::CORE_TO_CORE_HEADLINE_MAX_SECONDS) {
    return "above-target-window";
  }
  return "within-target-window";
}

ThreadHintStatus apply_thread_hints(bool request_affinity, int affinity_tag) {
  ThreadHintStatus status;

  const int qos_result = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  status.qos_applied = (qos_result == KERN_SUCCESS);
  status.qos_code = qos_result;

  status.affinity_requested = request_affinity;
  status.affinity_tag = affinity_tag;
  if (!request_affinity) {
    return status;
  }

  thread_affinity_policy_data_t affinity_policy = {affinity_tag};
  const thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
  const kern_return_t affinity_result =
      thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&affinity_policy),
                        THREAD_AFFINITY_POLICY_COUNT);
  status.affinity_applied = (affinity_result == KERN_SUCCESS);
  status.affinity_code = affinity_result;
  return status;
}

bool wait_for_start_signal(const SharedPingPongState& state) {
  while (!state.flags.start.load(std::memory_order_acquire)) {
    if (state.flags.cancel.load(std::memory_order_acquire)) {
      return false;
    }
  }
  return !state.flags.cancel.load(std::memory_order_acquire);
}

bool calculate_total_responder_round_trips(const CoreToCoreWorkPlan& work_plan, int sample_count,
                                           size_t& out_round_trips) {
  if (sample_count < 0 || work_plan.warmup_round_trips == 0 || work_plan.headline_round_trips == 0 ||
      (sample_count > 0 && work_plan.sample_window_round_trips == 0)) {
    return false;
  }

  const size_t maximum = std::numeric_limits<size_t>::max();
  if (work_plan.warmup_round_trips > maximum - work_plan.headline_round_trips) {
    return false;
  }
  size_t total = work_plan.warmup_round_trips + work_plan.headline_round_trips;
  const size_t sample_count_size = static_cast<size_t>(sample_count);
  if (sample_count_size > 0 && work_plan.sample_window_round_trips > (maximum - total) / sample_count_size) {
    return false;
  }
  total += sample_count_size * work_plan.sample_window_round_trips;
  out_round_trips = total;
  return true;
}

void run_round_trip_batch(uint32_t* turn_ptr, size_t round_trips) {
  core_to_core_initiator_round_trips_asm(turn_ptr, round_trips, Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE,
                                         Constants::CORE_TO_CORE_RESPONDER_TURN_VALUE);
}

void run_responder_round_trip_batch(uint32_t* turn_ptr, size_t round_trips) {
  core_to_core_responder_round_trips_asm(turn_ptr, round_trips, Constants::CORE_TO_CORE_RESPONDER_TURN_VALUE,
                                         Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE);
}

std::vector<ScenarioDescriptor> build_scenarios() {
  return {
      {Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY, Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
       Constants::CORE_TO_CORE_AFFINITY_TAG_NONE, Constants::CORE_TO_CORE_AFFINITY_TAG_NONE},
      {Constants::CORE_TO_CORE_SCENARIO_SAME_AFFINITY, Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
       Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY, Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY},
      {Constants::CORE_TO_CORE_SCENARIO_DIFFERENT_AFFINITY, Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
       Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY, Constants::CORE_TO_CORE_AFFINITY_TAG_SECONDARY},
  };
}

CoreToCoreWorkPlan build_calibration_pilot_plan() {
  CoreToCoreWorkPlan plan;
  plan.warmup_round_trips = Constants::CORE_TO_CORE_CALIBRATION_WARMUP_ROUND_TRIPS;
  plan.headline_round_trips = Constants::CORE_TO_CORE_CALIBRATION_ROUND_TRIPS;
  plan.sample_window_round_trips = Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS;
  return plan;
}

void print_statistics(const SummaryStats& stats) {
  std::cout << "    " << Messages::statistics_average(stats.average, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_median_p50(stats.median, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_p90(stats.p90, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_p95(stats.p95, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_p99(stats.p99, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_stddev(stats.stddev, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_coefficient_of_variation(stats.coefficient_of_variation_pct) << std::endl;
  std::cout << "    "
            << Messages::statistics_median_absolute_deviation(stats.median_absolute_deviation,
                                                              Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_min(stats.min, Constants::LATENCY_PRECISION) << std::endl;
  std::cout << "    " << Messages::statistics_max(stats.max, Constants::LATENCY_PRECISION) << std::endl;
}

void print_scenario_report(const CoreToCoreLatencyScenarioResult& scenario_result) {
  std::cout << std::endl;
  std::cout << Messages::report_core_to_core_scenario_title(scenario_result.scenario_name) << std::endl;
  std::cout << Messages::report_core_to_core_measurement_status(
                   core_to_core_measurement_status_to_string(scenario_result.status), scenario_result.status_reason,
                   scenario_result.completed_loops, scenario_result.planned_loops)
            << std::endl;

  if (scenario_result.work_plan.calibrated) {
    std::cout << Messages::report_core_to_core_work_plan(scenario_result.work_plan.calibration_round_trips,
                                                         scenario_result.work_plan.calibration_round_trip_ns,
                                                         scenario_result.work_plan.warmup_round_trips,
                                                         scenario_result.work_plan.headline_round_trips,
                                                         scenario_result.work_plan.sample_window_round_trips)
              << std::endl;
  }

  if (scenario_result.loop_round_trip_ns.empty()) {
    return;
  }

  const SummaryStats headline_stats = calculate_summary_stats(scenario_result.loop_round_trip_ns);
  std::cout << Messages::report_core_to_core_round_trip(headline_stats.median) << std::endl;
  std::cout << Messages::report_core_to_core_one_way_estimate(headline_stats.median * 0.5) << std::endl;
  std::cout << Messages::report_core_to_core_headline_statistics(scenario_result.loop_round_trip_ns.size())
            << std::endl;
  print_statistics(headline_stats);

  if (headline_stats.coefficient_of_variation_pct > Constants::CORE_TO_CORE_CV_WARNING_PCT) {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_benchmark_high_cv(scenario_result.scenario_name,
                                                     headline_stats.coefficient_of_variation_pct,
                                                     Constants::CORE_TO_CORE_CV_WARNING_PCT)
              << std::endl;
  }

  if (!scenario_result.sample_round_trip_ns.empty()) {
    const SummaryStats sample_stats = calculate_summary_stats(scenario_result.sample_round_trip_ns);
    std::cout << Messages::report_core_to_core_sample_statistics(scenario_result.sample_round_trip_ns.size())
              << std::endl;
    print_statistics(sample_stats);
  }

  std::cout << Messages::report_core_to_core_hint_status(
                   "Initiator", scenario_result.initiator_hint.qos_applied, scenario_result.initiator_hint.qos_code,
                   scenario_result.initiator_hint.affinity_requested, scenario_result.initiator_hint.affinity_applied,
                   scenario_result.initiator_hint.affinity_code, scenario_result.initiator_hint.affinity_tag)
            << std::endl;
  std::cout << Messages::report_core_to_core_hint_status(
                   "Responder", scenario_result.responder_hint.qos_applied, scenario_result.responder_hint.qos_code,
                   scenario_result.responder_hint.affinity_requested, scenario_result.responder_hint.affinity_applied,
                   scenario_result.responder_hint.affinity_code, scenario_result.responder_hint.affinity_tag)
            << std::endl;
}

}  // namespace

size_t calculate_core_to_core_calibrated_round_trips(double pilot_elapsed_seconds, size_t pilot_round_trips,
                                                     double target_duration_seconds, size_t minimum_round_trips,
                                                     size_t maximum_round_trips) {
  if (!positive_finite(pilot_elapsed_seconds) || pilot_round_trips == 0 ||
      !positive_finite(target_duration_seconds) || minimum_round_trips == 0 ||
      maximum_round_trips < minimum_round_trips) {
    return 0;
  }

  const long double scaled =
      static_cast<long double>(pilot_round_trips) * target_duration_seconds / pilot_elapsed_seconds;
  size_t result = scaled >= static_cast<long double>(maximum_round_trips) ? maximum_round_trips
                                                                          : static_cast<size_t>(std::ceil(scaled));
  result = std::max(result, minimum_round_trips);
  return std::min(result, maximum_round_trips);
}

std::vector<size_t> build_core_to_core_scenario_order(size_t scenario_count, size_t loop_index) {
  std::vector<size_t> order;
  order.reserve(scenario_count);
  if (scenario_count == 0) {
    return order;
  }
  const size_t first = loop_index % scenario_count;
  for (size_t position = 0; position < scenario_count; ++position) {
    order.push_back((first + position) % scenario_count);
  }
  return order;
}

bool build_core_to_core_work_plan(double pilot_elapsed_seconds, CoreToCoreWorkPlan& out_plan) {
  CoreToCoreWorkPlan plan;
  plan.calibration_round_trips = Constants::CORE_TO_CORE_CALIBRATION_ROUND_TRIPS;
  plan.calibration_elapsed_seconds = pilot_elapsed_seconds;
  if (!positive_finite(pilot_elapsed_seconds)) {
    return false;
  }
  plan.calibration_round_trip_ns = pilot_elapsed_seconds * 1e9 / static_cast<double>(plan.calibration_round_trips);
  plan.warmup_round_trips = calculate_core_to_core_calibrated_round_trips(
      pilot_elapsed_seconds, plan.calibration_round_trips, Constants::CORE_TO_CORE_WARMUP_TARGET_SECONDS,
      Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS, Constants::CORE_TO_CORE_MAX_ROUND_TRIPS);
  plan.headline_round_trips = calculate_core_to_core_calibrated_round_trips(
      pilot_elapsed_seconds, plan.calibration_round_trips, Constants::CORE_TO_CORE_HEADLINE_TARGET_SECONDS,
      Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS, Constants::CORE_TO_CORE_MAX_ROUND_TRIPS);
  plan.sample_window_round_trips = calculate_core_to_core_calibrated_round_trips(
      pilot_elapsed_seconds, plan.calibration_round_trips, Constants::CORE_TO_CORE_SAMPLE_TARGET_SECONDS,
      Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS, Constants::CORE_TO_CORE_MAX_ROUND_TRIPS);
  plan.calibrated = plan.warmup_round_trips > 0 && plan.headline_round_trips > 0 &&
                    plan.sample_window_round_trips > 0 && positive_finite(plan.calibration_round_trip_ns);
  if (!plan.calibrated) {
    return false;
  }
  out_plan = plan;
  return true;
}

bool execute_single_scenario(const ScenarioDescriptor& scenario, const CoreToCoreWorkPlan& work_plan, int sample_count,
                             ScenarioMeasurement& out_measurement,
                             const CoreToCoreFailureInjection* failure_injection) {
  out_measurement = ScenarioMeasurement{};

  if (failure_injection != nullptr && failure_injection->fail_timer_creation) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "timer-creation-failed";
    return false;
  }
  auto timer_optional = HighResTimer::create();
  if (!timer_optional) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "timer-creation-failed";
    return false;
  }

  size_t responder_round_trips = 0;
  if (!calculate_total_responder_round_trips(work_plan, sample_count, responder_round_trips)) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "invalid-or-overflowing-work-plan";
    return false;
  }

  try {
    out_measurement.samples_ns.reserve(static_cast<size_t>(sample_count));
  } catch (const std::exception&) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "sample-storage-allocation-failed";
    return false;
  }

  SharedPingPongState state;
  std::thread responder_thread;
  std::thread initiator_thread;

  if (failure_injection != nullptr && failure_injection->fail_responder_startup) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "responder-thread-startup-failed";
    return false;
  }
  try {
    responder_thread = std::thread([&state, &scenario, &out_measurement, responder_round_trips]() {
      out_measurement.responder_hint = apply_thread_hints(scenario.apply_affinity, scenario.responder_affinity_tag);
      state.flags.ready_threads.fetch_add(1, std::memory_order_release);
      if (!wait_for_start_signal(state)) {
        return;
      }
      run_responder_round_trip_batch(&state.turn.value, responder_round_trips);
    });
  } catch (const std::system_error&) {
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "responder-thread-startup-failed";
    return false;
  }

  if (failure_injection != nullptr && failure_injection->fail_initiator_startup) {
    state.flags.cancel.store(true, std::memory_order_release);
    state.flags.start.store(true, std::memory_order_release);
    responder_thread.join();
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "initiator-thread-startup-failed";
    return false;
  }
  try {
    initiator_thread = std::thread([&state, &scenario, &out_measurement, sample_count, work_plan, &timer_optional]() {
      out_measurement.initiator_hint = apply_thread_hints(scenario.apply_affinity, scenario.initiator_affinity_tag);
      state.flags.ready_threads.fetch_add(1, std::memory_order_release);
      if (!wait_for_start_signal(state)) {
        return;
      }

      run_round_trip_batch(&state.turn.value, work_plan.warmup_round_trips);

      auto& timer = *timer_optional;
      timer.start();
      run_round_trip_batch(&state.turn.value, work_plan.headline_round_trips);
      const double headline_total_ns = timer.stop_ns();
      out_measurement.headline_elapsed_seconds = headline_total_ns / 1e9;
      out_measurement.round_trip_ns = headline_total_ns / static_cast<double>(work_plan.headline_round_trips);
      out_measurement.duration_quality = classify_duration_quality(out_measurement.headline_elapsed_seconds);

      for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        timer.start();
        run_round_trip_batch(&state.turn.value, work_plan.sample_window_round_trips);
        const double sample_total_ns = timer.stop_ns();
        out_measurement.samples_ns.push_back(sample_total_ns /
                                             static_cast<double>(work_plan.sample_window_round_trips));
      }
    });
  } catch (const std::system_error&) {
    state.flags.cancel.store(true, std::memory_order_release);
    state.flags.start.store(true, std::memory_order_release);
    responder_thread.join();
    out_measurement.status = CoreToCoreMeasurementStatus::Failed;
    out_measurement.status_reason = "initiator-thread-startup-failed";
    return false;
  }

  while (state.flags.ready_threads.load(std::memory_order_acquire) < Constants::CORE_TO_CORE_READY_THREADS_TARGET) {
  }
  state.turn.value = Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE;
  state.flags.start.store(true, std::memory_order_release);

  initiator_thread.join();
  responder_thread.join();

  if (!positive_finite(out_measurement.headline_elapsed_seconds) ||
      !positive_finite(out_measurement.round_trip_ns)) {
    out_measurement.status = CoreToCoreMeasurementStatus::Invalid;
    out_measurement.status_reason = "invalid-headline-elapsed";
    return true;
  }
  for (double sample : out_measurement.samples_ns) {
    if (!positive_finite(sample)) {
      out_measurement.status = CoreToCoreMeasurementStatus::Invalid;
      out_measurement.status_reason = "invalid-sample-elapsed";
      return true;
    }
  }

  out_measurement.status = CoreToCoreMeasurementStatus::Measured;
  return true;
}

int run_core_to_core_latency_collect(const CoreToCoreLatencyConfig& config, nlohmann::ordered_json& result_json) {
  const auto analysis_start = std::chrono::steady_clock::now();

  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_core_to_core_analysis() << std::endl;

  const std::string cpu_name = get_processor_name();
  const int performance_cores = get_performance_cores();
  const int efficiency_cores = get_efficiency_cores();

  const std::vector<ScenarioDescriptor> scenarios = build_scenarios();
  std::vector<CoreToCoreLatencyScenarioResult> scenario_results;
  scenario_results.reserve(scenarios.size());
  for (const ScenarioDescriptor& scenario : scenarios) {
    CoreToCoreLatencyScenarioResult result;
    result.scenario_name = scenario.name;
    result.planned_loops = static_cast<size_t>(config.loop_count);
    scenario_results.push_back(std::move(result));
  }

  bool run_failed = false;
  bool interrupted = false;
  const CoreToCoreWorkPlan pilot_plan = build_calibration_pilot_plan();
  for (size_t scenario_index = 0; scenario_index < scenarios.size(); ++scenario_index) {
    if (signal_received()) {
      interrupted = true;
      break;
    }

    ScenarioMeasurement pilot;
    CoreToCoreLatencyScenarioResult& scenario_result = scenario_results[scenario_index];
    if (!execute_single_scenario(scenarios[scenario_index], pilot_plan, 0, pilot) ||
        pilot.status != CoreToCoreMeasurementStatus::Measured ||
        !build_core_to_core_work_plan(pilot.headline_elapsed_seconds, scenario_result.work_plan)) {
      scenario_result.status = pilot.status == CoreToCoreMeasurementStatus::Invalid
                                   ? CoreToCoreMeasurementStatus::Invalid
                                   : CoreToCoreMeasurementStatus::Failed;
      scenario_result.status_reason = pilot.status_reason.empty() ? "calibration-failed" : pilot.status_reason;
      run_failed = true;
      break;
    }
  }

  if (!run_failed && !interrupted) {
    for (int loop_index = 0; loop_index < config.loop_count; ++loop_index) {
      const std::vector<size_t> scenario_order =
          build_core_to_core_scenario_order(scenarios.size(), static_cast<size_t>(loop_index));
      for (size_t order_position = 0; order_position < scenario_order.size(); ++order_position) {
        if (signal_received()) {
          interrupted = true;
          break;
        }

        const size_t scenario_index = scenario_order[order_position];
        const ScenarioDescriptor& scenario = scenarios[scenario_index];
        CoreToCoreLatencyScenarioResult& scenario_result = scenario_results[scenario_index];
        std::cout << Messages::msg_core_to_core_scenario_progress(static_cast<size_t>(loop_index + 1),
                                                                  static_cast<size_t>(config.loop_count), scenario.name)
                  << std::endl;

        ScenarioMeasurement measurement;
        const bool execution_succeeded =
            execute_single_scenario(scenario, scenario_result.work_plan, config.latency_sample_count, measurement);

        CoreToCoreLoopRecord record;
        record.loop_index = static_cast<size_t>(loop_index);
        record.order_position = order_position;
        record.status = measurement.status;
        record.status_reason = measurement.status_reason;
        record.sample_start_index = scenario_result.sample_round_trip_ns.size();
        record.completed_sample_windows = measurement.samples_ns.size();
        record.initiator_hint = measurement.initiator_hint;
        record.responder_hint = measurement.responder_hint;
        if (measurement.status == CoreToCoreMeasurementStatus::Measured) {
          record.round_trip_ns = measurement.round_trip_ns;
          record.headline_elapsed_seconds = measurement.headline_elapsed_seconds;
          record.duration_quality = measurement.duration_quality;
          scenario_result.loop_round_trip_ns.push_back(measurement.round_trip_ns);
          scenario_result.sample_round_trip_ns.insert(scenario_result.sample_round_trip_ns.end(),
                                                      measurement.samples_ns.begin(), measurement.samples_ns.end());
          if (scenario_result.completed_loops == 0) {
            scenario_result.initiator_hint = measurement.initiator_hint;
            scenario_result.responder_hint = measurement.responder_hint;
          }
          ++scenario_result.completed_loops;
        }
        scenario_result.loop_records.push_back(std::move(record));

        if (!execution_succeeded || measurement.status == CoreToCoreMeasurementStatus::Failed ||
            measurement.status == CoreToCoreMeasurementStatus::Invalid) {
          scenario_result.status = measurement.status;
          scenario_result.status_reason = measurement.status_reason;
          std::cerr << Messages::error_prefix()
                    << Messages::error_core_to_core_measurement_failed(measurement.status_reason) << std::endl;
          run_failed = true;
          break;
        }
      }

      if (run_failed || interrupted) {
        break;
      }
    }
  }

  if (signal_received()) {
    interrupted = true;
  }
  for (CoreToCoreLatencyScenarioResult& scenario_result : scenario_results) {
    if (scenario_result.status == CoreToCoreMeasurementStatus::Failed ||
        scenario_result.status == CoreToCoreMeasurementStatus::Invalid) {
      continue;
    }
    if (!run_failed && !interrupted && scenario_result.completed_loops == scenario_result.planned_loops) {
      scenario_result.status = CoreToCoreMeasurementStatus::Measured;
      scenario_result.status_reason.clear();
    } else {
      scenario_result.status = CoreToCoreMeasurementStatus::Interrupted;
      scenario_result.status_reason = "command-incomplete";
    }
  }

  if (interrupted) {
    std::cout << std::endl << Messages::msg_interrupted_by_user() << std::endl;
  }

  std::cout << std::endl;
  std::cout << Messages::report_core_to_core_header() << std::endl;
  std::cout << Messages::report_core_to_core_scheduler_note() << std::endl;
  std::cout << Messages::report_core_to_core_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_core_to_core_cores(performance_cores, efficiency_cores) << std::endl;
  std::cout << Messages::report_core_to_core_loop_config(
                   config.loop_count, config.latency_sample_count, Constants::CORE_TO_CORE_HEADLINE_TARGET_SECONDS,
                   Constants::CORE_TO_CORE_HEADLINE_MIN_SECONDS, Constants::CORE_TO_CORE_HEADLINE_MAX_SECONDS,
                   Constants::CORE_TO_CORE_SAMPLE_TARGET_SECONDS)
            << std::endl;

  for (const CoreToCoreLatencyScenarioResult& scenario_result : scenario_results) {
    print_scenario_report(scenario_result);
  }

  const auto analysis_end = std::chrono::steady_clock::now();
  const double total_execution_time_seconds = std::chrono::duration<double>(analysis_end - analysis_start).count();
  size_t completed_measurements = 0;
  for (const CoreToCoreLatencyScenarioResult& scenario_result : scenario_results) {
    completed_measurements += scenario_result.completed_loops;
  }
  const size_t planned_measurements = scenarios.size() * static_cast<size_t>(config.loop_count);
  const std::string status = run_failed ? "failed" : (interrupted ? "interrupted" : "complete");

  const CoreToCoreLatencyJsonContext json_context = {
      config,
      cpu_name,
      performance_cores,
      efficiency_cores,
      Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS,
      Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS,
      Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS,
      scenario_results,
      total_execution_time_seconds,
      status,
      planned_measurements,
      completed_measurements,
  };
  result_json = build_core_to_core_latency_json(json_context);
  return run_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

int run_core_to_core_latency(const CoreToCoreLatencyConfig& config) {
  nlohmann::ordered_json result_json;
  const int run_result = run_core_to_core_latency_collect(config, result_json);

  if (!config.output_file.empty() && !result_json.empty()) {
    std::filesystem::path file_path(config.output_file);
    if (file_path.is_relative()) {
      file_path = std::filesystem::current_path() / file_path;
    }
    if (write_json_to_file(file_path, result_json) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  return run_result;
}
