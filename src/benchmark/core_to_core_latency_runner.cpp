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
 * @brief Standalone core-to-core cache-line handoff benchmark implementation (@author Timo Heimonen <timo.heimonen@proton.me>, @date 2026)
 */

#include "benchmark/core_to_core_latency.h"
#include "benchmark/core_to_core_latency_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
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
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"

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
};

struct alignas(Constants::CACHE_LINE_SIZE_BYTES) SharedTurn {
  uint32_t value = Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE;
};

struct alignas(Constants::CACHE_LINE_SIZE_BYTES) SharedFlags {
  std::atomic<bool> start{false};
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
  for (double v : values) {
    sum += v;
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
  const size_t n = sorted.size();

  const auto percentile = [&sorted, n](double p) {
    if (n == 1) {
      return sorted[0];
    }
    const double idx = p * static_cast<double>(n - 1);
    const size_t lower = static_cast<size_t>(idx);
    const size_t upper = lower + 1;
    if (upper >= n) {
      return sorted[n - 1];
    }
    const double weight = idx - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
  };

  stats.median = percentile(0.50);
  stats.p90 = percentile(0.90);
  stats.p95 = percentile(0.95);
  stats.p99 = percentile(0.99);

  if (values.size() > 1) {
    double variance_sum = 0.0;
    for (double v : values) {
      const double diff = v - stats.average;
      variance_sum += diff * diff;
    }
    stats.stddev = std::sqrt(variance_sum / static_cast<double>(values.size() - 1));
  }

  return stats;
}

ThreadHintStatus apply_thread_hints(bool request_affinity, int affinity_tag) {
  ThreadHintStatus status;

  const int qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  status.qos_applied = (qos_ret == KERN_SUCCESS);
  status.qos_code = qos_ret;

  status.affinity_requested = request_affinity;
  status.affinity_tag = affinity_tag;
  if (!request_affinity) {
    return status;
  }

  thread_affinity_policy_data_t affinity_policy = {affinity_tag};
  const thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
  const kern_return_t affinity_ret =
      thread_policy_set(mach_thread,
                        THREAD_AFFINITY_POLICY,
                        reinterpret_cast<thread_policy_t>(&affinity_policy),
                        THREAD_AFFINITY_POLICY_COUNT);
  status.affinity_applied = (affinity_ret == KERN_SUCCESS);
  status.affinity_code = affinity_ret;
  return status;
}

// Busy-wait start barrier keeps thread creation outside timed region.
void wait_for_start_signal(const SharedPingPongState& state) {
  while (!state.flags.start.load(std::memory_order_acquire)) {
  }
}

size_t calculate_total_responder_round_trips(int sample_count) {
  return Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS +
         Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS +
         (static_cast<size_t>(sample_count) * Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS);
}

// One round trip: initiator hands token to responder and waits to get it back.
void run_round_trip_batch(uint32_t* turn_ptr, size_t round_trips) {
  core_to_core_initiator_round_trips_asm(turn_ptr,
                                         round_trips,
                                         Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE,
                                         Constants::CORE_TO_CORE_RESPONDER_TURN_VALUE);
}

void run_responder_round_trip_batch(uint32_t* turn_ptr, size_t round_trips) {
  core_to_core_responder_round_trips_asm(turn_ptr,
                                         round_trips,
                                         Constants::CORE_TO_CORE_RESPONDER_TURN_VALUE,
                                         Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE);
}

// Scenarios are interpreted as scheduler-hint requests, not hard pinning contracts.
std::vector<ScenarioDescriptor> build_scenarios() {
  std::vector<ScenarioDescriptor> scenarios;
  scenarios.push_back({Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
                       Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_NONE});
  scenarios.push_back({Constants::CORE_TO_CORE_SCENARIO_SAME_AFFINITY,
                       Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY});
  scenarios.push_back({Constants::CORE_TO_CORE_SCENARIO_DIFFERENT_AFFINITY,
                       Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY,
                       Constants::CORE_TO_CORE_AFFINITY_TAG_SECONDARY});
  return scenarios;
}

void print_scenario_report(const CoreToCoreLatencyScenarioResult& scenario_result) {
  const double avg_round_trip_ns = calculate_average(scenario_result.loop_round_trip_ns);
  const double one_way_estimate_ns = avg_round_trip_ns * 0.5;
  const SummaryStats sample_stats = calculate_summary_stats(scenario_result.sample_round_trip_ns);

  std::cout << std::endl;
  std::cout << Messages::report_core_to_core_scenario_title(scenario_result.scenario_name) << std::endl;
  std::cout << Messages::report_core_to_core_round_trip(avg_round_trip_ns) << std::endl;
  std::cout << Messages::report_core_to_core_one_way_estimate(one_way_estimate_ns) << std::endl;
  std::cout << Messages::report_core_to_core_samples(scenario_result.sample_round_trip_ns.size())
            << std::endl;
  std::cout << "    "
            << Messages::statistics_median_p50(sample_stats.median, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_p90(sample_stats.p90, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_p95(sample_stats.p95, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_p99(sample_stats.p99, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    "
            << Messages::statistics_stddev(sample_stats.stddev, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_min(sample_stats.min, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << "    " << Messages::statistics_max(sample_stats.max, Constants::LATENCY_PRECISION)
            << std::endl;
  std::cout << Messages::report_core_to_core_hint_status("Initiator",
                                                         scenario_result.initiator_hint.qos_applied,
                                                         scenario_result.initiator_hint.qos_code,
                                                         scenario_result.initiator_hint.affinity_requested,
                                                         scenario_result.initiator_hint.affinity_applied,
                                                         scenario_result.initiator_hint.affinity_code,
                                                         scenario_result.initiator_hint.affinity_tag)
            << std::endl;
  std::cout << Messages::report_core_to_core_hint_status("Responder",
                                                         scenario_result.responder_hint.qos_applied,
                                                         scenario_result.responder_hint.qos_code,
                                                         scenario_result.responder_hint.affinity_requested,
                                                         scenario_result.responder_hint.affinity_applied,
                                                         scenario_result.responder_hint.affinity_code,
                                                         scenario_result.responder_hint.affinity_tag)
            << std::endl;
}

}  // namespace

bool execute_single_scenario(const ScenarioDescriptor& scenario,
                             int sample_count,
                             ScenarioMeasurement& out_measurement) {
  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    return false;
  }

  const size_t responder_round_trips = calculate_total_responder_round_trips(sample_count);

  SharedPingPongState state;

  std::thread responder_thread([&state, &scenario, &out_measurement, responder_round_trips]() {
    out_measurement.responder_hint =
        apply_thread_hints(scenario.apply_affinity, scenario.responder_affinity_tag);
    state.flags.ready_threads.fetch_add(1, std::memory_order_release);
    wait_for_start_signal(state);
    run_responder_round_trip_batch(&state.turn.value, responder_round_trips);
  });

  std::thread initiator_thread([&state, &scenario, &out_measurement, sample_count, &timer_opt]() {
    out_measurement.initiator_hint =
        apply_thread_hints(scenario.apply_affinity, scenario.initiator_affinity_tag);
    state.flags.ready_threads.fetch_add(1, std::memory_order_release);
    wait_for_start_signal(state);

    run_round_trip_batch(&state.turn.value, Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS);

    auto& timer = *timer_opt;
    timer.start();
    run_round_trip_batch(&state.turn.value, Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS);
    const double headline_total_ns = timer.stop_ns();
    out_measurement.round_trip_ns =
        headline_total_ns / static_cast<double>(Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS);

    out_measurement.samples_ns.reserve(static_cast<size_t>(sample_count));
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
      timer.start();
      run_round_trip_batch(&state.turn.value, Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS);
      const double sample_total_ns = timer.stop_ns();
      out_measurement.samples_ns.push_back(
          sample_total_ns / static_cast<double>(Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS));
    }
  });

  // Ensure both worker threads have applied hints before starting handoff loop.
  while (state.flags.ready_threads.load(std::memory_order_acquire) <
         Constants::CORE_TO_CORE_READY_THREADS_TARGET) {
  }
  state.turn.value = Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE;
  state.flags.start.store(true, std::memory_order_release);

  initiator_thread.join();
  responder_thread.join();

  return true;
}

int run_core_to_core_latency(const CoreToCoreLatencyConfig& config) {
  const auto analysis_start = std::chrono::steady_clock::now();

  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_core_to_core_analysis() << std::endl;

  const std::string cpu_name = get_processor_name();
  const int perf_cores = get_performance_cores();
  const int eff_cores = get_efficiency_cores();

  const std::vector<ScenarioDescriptor> scenarios = build_scenarios();
  std::vector<CoreToCoreLatencyScenarioResult> scenario_results;
  scenario_results.reserve(scenarios.size());
  for (const ScenarioDescriptor& scenario : scenarios) {
    CoreToCoreLatencyScenarioResult result;
    result.scenario_name = scenario.name;
    scenario_results.push_back(std::move(result));
  }

  for (int loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    for (size_t scenario_index = 0; scenario_index < scenarios.size(); ++scenario_index) {
      const ScenarioDescriptor& scenario = scenarios[scenario_index];
      std::cout << Messages::msg_core_to_core_scenario_progress(
                       static_cast<size_t>(loop_index + 1),
                       static_cast<size_t>(config.loop_count),
                       scenario.name)
                << std::endl;

      ScenarioMeasurement measurement;
      if (!execute_single_scenario(scenario, config.latency_sample_count, measurement)) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_core_to_core_timer_creation_failed()
                  << std::endl;
        return EXIT_FAILURE;
      }

      CoreToCoreLatencyScenarioResult& scenario_result = scenario_results[scenario_index];
      scenario_result.loop_round_trip_ns.push_back(measurement.round_trip_ns);
      scenario_result.sample_round_trip_ns.insert(scenario_result.sample_round_trip_ns.end(),
                                                  measurement.samples_ns.begin(),
                                                  measurement.samples_ns.end());
      if (loop_index == 0) {
        scenario_result.initiator_hint = measurement.initiator_hint;
        scenario_result.responder_hint = measurement.responder_hint;
      }
    }
  }

  std::cout << std::endl;
  std::cout << Messages::report_core_to_core_header() << std::endl;
  std::cout << Messages::report_core_to_core_scheduler_note() << std::endl;
  std::cout << Messages::report_core_to_core_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_core_to_core_cores(perf_cores, eff_cores) << std::endl;
  std::cout << Messages::report_core_to_core_loop_config(config.loop_count,
                                                         config.latency_sample_count,
                                                         Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS,
                                                         Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS)
            << std::endl;

  for (const CoreToCoreLatencyScenarioResult& scenario_result : scenario_results) {
    print_scenario_report(scenario_result);
  }

  const auto analysis_end = std::chrono::steady_clock::now();
  const double total_execution_time_sec =
      std::chrono::duration<double>(analysis_end - analysis_start).count();

  const CoreToCoreLatencyJsonContext json_context = {
      config,
      cpu_name,
      perf_cores,
      eff_cores,
      Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS,
      Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS,
      Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS,
      scenario_results,
      total_execution_time_sec,
  };
  if (save_core_to_core_latency_to_json(json_context) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
