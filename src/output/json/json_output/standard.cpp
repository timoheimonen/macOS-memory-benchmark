// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file standard.cpp
 * @brief Auditable schema-v2 JSON for the standard benchmark
 */

#include "output/json/json_output/json_output_api.h"

#include <cmath>
#include <string>
#include <vector>

#include "benchmark/benchmark_runner.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "utils/json_utils.h"

namespace {

using MeasurementMember = BenchmarkMeasurement BenchmarkResults::*;

nlohmann::json measurement_json(const BenchmarkMeasurement& measurement,
                                size_t loop_index) {
  nlohmann::json json;
  json["benchmark_loop_index"] = loop_index;
  json["status"] =
      benchmark_measurement_status_to_string(measurement.status);
  json["reason"] = measurement.status_reason;
  if (measurement.value.has_value()) {
    json["value"] = *measurement.value;
  } else {
    json["value"] = nullptr;
  }
  json["target"] = measurement.target;
  json["operation"] = measurement.operation;
  json["work_policy"] = measurement.work_policy;
  json["buffer_size_bytes"] = measurement.buffer_size_bytes;
  json["passes"] = measurement.passes;
  json["access_count"] = measurement.access_count;
  json["chain_node_count"] = measurement.chain_node_count;
  json["complete_chain_cycles"] = measurement.complete_chain_cycles;
  json["exact_payload_bytes"] = measurement.exact_payload_bytes;
  json["requested_threads"] = measurement.requested_threads;
  json["effective_threads"] = measurement.effective_threads;
  json["qos_outcome"] = measurement.qos_outcome;
  json["seed"] = std::to_string(measurement.seed);
  json["seed_encoding"] = "uint64-decimal-string";
  json["pilot_elapsed_seconds"] = measurement.pilot_elapsed_seconds;
  json["elapsed_seconds"] = measurement.elapsed_seconds;
  json["automatic_calibration"] = measurement.automatic_calibration;
  json["calibration_corrections"] = measurement.calibration_corrections;
  json["duration_within_target"] = measurement.duration_within_target;
  json["duration_quality"] = measurement.duration_quality;
  json["phase_order_position"] = measurement.phase_order_index;
  json["operation_order_position"] = measurement.operation_order_index;
  if (!measurement.samples.empty()) {
    json["samples_ns"] = measurement.samples;
  }
  if (!measurement.sample_seeds.empty()) {
    json["sample_seeds"] = nlohmann::json::array();
    for (uint64_t seed : measurement.sample_seeds) {
      json["sample_seeds"].push_back(std::to_string(seed));
    }
  }
  return json;
}

std::vector<double> measured_values(const BenchmarkStatistics& stats,
                                    MeasurementMember member,
                                    const std::vector<double>& fallback) {
  if (stats.loop_results.empty()) return fallback;
  std::vector<double> values;
  values.reserve(stats.loop_results.size());
  for (const BenchmarkResults& loop : stats.loop_results) {
    const BenchmarkMeasurement& measurement = loop.*member;
    if (measurement.is_measured()) values.push_back(*measurement.value);
  }
  return values;
}

nlohmann::json build_pooled_samples(const BenchmarkStatistics& stats,
                                    MeasurementMember member) {
  nlohmann::json pooled;
  std::vector<double> values;
  nlohmann::json ranges = nlohmann::json::array();
  for (const BenchmarkResults& loop : stats.loop_results) {
    const BenchmarkMeasurement& measurement = loop.*member;
    if (!measurement.is_measured() || measurement.samples.empty()) continue;
    const size_t start = values.size();
    values.insert(values.end(), measurement.samples.begin(),
                  measurement.samples.end());
    ranges.push_back({{"benchmark_loop_index", loop.loop_index},
                      {"start_index", start},
                      {"sample_count", measurement.samples.size()}});
  }
  if (values.empty()) return nullptr;
  pooled["semantics"] = "pooled-separate-sample-window-distribution";
  pooled["values_ns"] = values;
  pooled["loop_ranges"] = ranges;
  pooled["statistics"] = calculate_json_statistics(values);
  return pooled;
}

nlohmann::json aggregate_json(const BenchmarkStatistics& stats,
                              MeasurementMember member,
                              const std::vector<double>& fallback,
                              const char* unit,
                              bool include_pooled_samples = false) {
  const std::vector<double> values = measured_values(stats, member, fallback);
  nlohmann::json aggregate;
  aggregate["unit"] = unit;
  aggregate["requested_loop_count"] = stats.planned_loops;
  aggregate["measured_loop_count"] = values.size();
  aggregate["values"] = values;
  aggregate["measurements"] = nlohmann::json::array();

  size_t expected_measurements = 0;
  bool every_duration_in_window = true;
  bool has_automatic_measurement = false;
  for (const BenchmarkResults& loop : stats.loop_results) {
    const BenchmarkMeasurement& measurement = loop.*member;
    if (measurement.status != BenchmarkMeasurementStatus::NotRun) {
      ++expected_measurements;
      aggregate["measurements"].push_back(
          measurement_json(measurement, loop.loop_index));
    }
    if (measurement.is_measured() && measurement.automatic_calibration) {
      has_automatic_measurement = true;
      every_duration_in_window &= measurement.duration_within_target;
    }
  }

  if (values.empty()) {
    aggregate["status"] = expected_measurements == 0 ? "not-run" : "unavailable";
    aggregate["value"] = nullptr;
    aggregate["statistics"] = nullptr;
  } else {
    const nlohmann::json statistics = calculate_json_statistics(values);
    aggregate["status"] = values.size() == expected_measurements ||
                                  expected_measurements == 0
                              ? "measured"
                              : "partial";
    aggregate["value"] = statistics["median"];
    aggregate["headline_semantics"] = values.size() == 1
                                           ? "single-measured-loop"
                                           : "median-p50-across-loop-headlines";
    aggregate["statistics"] = statistics;
  }

  nlohmann::json quality;
  quality["repeatability_assessment_available"] = values.size() >= 2;
  quality["automatic_duration_window_all_met"] =
      has_automatic_measurement ? nlohmann::json(every_duration_in_window)
                                : nlohmann::json(nullptr);
  quality["run_complete"] = stats.status == BenchmarkRunStatus::Complete;
  quality["cv_warning_threshold_pct"] = Constants::BENCHMARK_CV_WARNING_PCT;
  if (values.size() >= 2) {
    const nlohmann::json statistics = calculate_json_statistics(values);
    const nlohmann::json& cv = statistics["coefficient_of_variation_pct"];
    quality["cv_warning"] =
        !cv.is_null() && cv.get<double>() > Constants::BENCHMARK_CV_WARNING_PCT;
  } else {
    quality["cv_warning"] = nullptr;
  }
  aggregate["quality"] = quality;

  if (include_pooled_samples) {
    const nlohmann::json pooled = build_pooled_samples(stats, member);
    if (!pooled.is_null()) aggregate["pooled_sample_distribution"] = pooled;
  }
  return aggregate;
}

nlohmann::json bandwidth_json(const BenchmarkStatistics& stats,
                              MeasurementMember read_member,
                              MeasurementMember write_member,
                              MeasurementMember copy_member,
                              const std::vector<double>& read_fallback,
                              const std::vector<double>& write_fallback,
                              const std::vector<double>& copy_fallback) {
  return {{"read_gb_s", aggregate_json(stats, read_member, read_fallback, "GB/s")},
          {"write_gb_s", aggregate_json(stats, write_member, write_fallback, "GB/s")},
          {"copy_gb_s", aggregate_json(stats, copy_member, copy_fallback, "GB/s")}};
}

nlohmann::json loop_json(const BenchmarkResults& loop,
                         const BenchmarkConfig& config) {
  nlohmann::json json;
  json["benchmark_loop_index"] = loop.loop_index;
  json["status"] = benchmark_run_status_to_string(loop.status);
  json["reason"] = loop.status_reason;
  json["planned_phases"] = loop.planned_phases;
  json["completed_phases"] = loop.completed_phases;
  json["planned_measurements"] = loop.planned_measurements;
  json["completed_measurements"] = loop.completed_measurements;
  json["phase_order_index"] = loop.phase_order_index;
  json["operation_order_index"] = loop.operation_order_index;
  json["planned_phase_order"] = loop.planned_phase_order;
  json["realized_phase_order"] = loop.realized_phase_order;
  nlohmann::json measurements;
  auto add = [&](const char* name, MeasurementMember member) {
    const BenchmarkMeasurement& measurement = loop.*member;
    if (measurement.status != BenchmarkMeasurementStatus::NotRun) {
      measurements[name] = measurement_json(measurement, loop.loop_index);
    }
  };
  if (!config.only_latency) {
    add("main_read_bandwidth", &BenchmarkResults::main_read_bandwidth);
    add("main_write_bandwidth", &BenchmarkResults::main_write_bandwidth);
    add("main_copy_bandwidth", &BenchmarkResults::main_copy_bandwidth);
    if (config.use_custom_cache_size) {
      add("custom_read_bandwidth", &BenchmarkResults::custom_read_bandwidth);
      add("custom_write_bandwidth", &BenchmarkResults::custom_write_bandwidth);
      add("custom_copy_bandwidth", &BenchmarkResults::custom_copy_bandwidth);
    } else {
      add("l1_read_bandwidth", &BenchmarkResults::l1_read_bandwidth);
      add("l1_write_bandwidth", &BenchmarkResults::l1_write_bandwidth);
      add("l1_copy_bandwidth", &BenchmarkResults::l1_copy_bandwidth);
      add("l2_read_bandwidth", &BenchmarkResults::l2_read_bandwidth);
      add("l2_write_bandwidth", &BenchmarkResults::l2_write_bandwidth);
      add("l2_copy_bandwidth", &BenchmarkResults::l2_copy_bandwidth);
    }
  }
  if (!config.only_bandwidth) {
    add("main_latency", &BenchmarkResults::main_latency);
    add("locality_16k_latency", &BenchmarkResults::locality_16k_latency);
    add("global_random_latency", &BenchmarkResults::global_random_latency);
    add("locality_latency_delta", &BenchmarkResults::locality_latency_delta);
    if (config.use_custom_cache_size) {
      add("custom_latency", &BenchmarkResults::custom_latency);
    } else {
      add("l1_latency", &BenchmarkResults::l1_latency);
      add("l2_latency", &BenchmarkResults::l2_latency);
    }
  }
  json["measurements"] = measurements;
  return json;
}

}  // namespace

void add_standard_benchmark_results(nlohmann::ordered_json& output,
                                    const BenchmarkConfig& config,
                                    const BenchmarkStatistics& stats) {
  output["status"] = benchmark_run_status_to_string(stats.status);
  output["status_reason"] = stats.status_reason;
  output["planned_loops"] = stats.planned_loops;
  output["completed_loops"] = stats.completed_loops;
  output["planned_measurements"] = stats.planned_measurements;
  output["completed_measurements"] = stats.completed_measurements;
  output["results_complete"] =
      stats.status == BenchmarkRunStatus::Complete &&
      stats.completed_loops == stats.planned_loops &&
      stats.completed_measurements == stats.planned_measurements;

  output["loops"] = nlohmann::json::array();
  for (const BenchmarkResults& loop : stats.loop_results) {
    output["loops"][output["loops"].size()] = loop_json(loop, config);
  }

  if (!config.only_latency) {
    nlohmann::json main;
    main["bandwidth"] = bandwidth_json(
        stats, &BenchmarkResults::main_read_bandwidth,
        &BenchmarkResults::main_write_bandwidth,
        &BenchmarkResults::main_copy_bandwidth, stats.all_read_bw_gb_s,
        stats.all_write_bw_gb_s, stats.all_copy_bw_gb_s);
    output[JsonKeys::MAIN_MEMORY] = main;
  }
  if (!config.only_bandwidth) {
    if (!output.contains(JsonKeys::MAIN_MEMORY)) {
      output[JsonKeys::MAIN_MEMORY] = nlohmann::json::object();
    }
    nlohmann::json latency;
    latency["headline_ns"] = aggregate_json(
        stats, &BenchmarkResults::main_latency,
        stats.all_average_latency_ns, "ns", true);
    latency["automatic_locality_comparison"] = {
        {"locality_16k_latency_ns",
         aggregate_json(stats, &BenchmarkResults::locality_16k_latency,
                        stats.all_tlb_hit_latency_ns, "ns")},
        {"global_random_latency_ns",
         aggregate_json(stats, &BenchmarkResults::global_random_latency,
                        stats.all_tlb_miss_latency_ns, "ns")},
        {"locality_latency_delta_ns",
         aggregate_json(stats, &BenchmarkResults::locality_latency_delta,
                        stats.all_page_walk_penalty_ns, "ns")}};
    output[JsonKeys::MAIN_MEMORY]["latency"] = latency;
  }

  if (config.only_latency && config.only_bandwidth) return;
  nlohmann::json cache;
  auto add_cache = [&](const char* key, MeasurementMember read,
                       MeasurementMember write, MeasurementMember copy,
                       MeasurementMember latency,
                       const std::vector<double>& read_values,
                       const std::vector<double>& write_values,
                       const std::vector<double>& copy_values,
                       const std::vector<double>& latency_values) {
    nlohmann::json target;
    if (!config.only_latency) {
      target["bandwidth"] = bandwidth_json(
          stats, read, write, copy, read_values, write_values, copy_values);
    }
    if (!config.only_bandwidth) {
      target["latency"] = {{"headline_ns", aggregate_json(
          stats, latency, latency_values, "ns", true)}};
    }
    cache[key] = target;
  };
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      add_cache("custom", &BenchmarkResults::custom_read_bandwidth,
                &BenchmarkResults::custom_write_bandwidth,
                &BenchmarkResults::custom_copy_bandwidth,
                &BenchmarkResults::custom_latency,
                stats.all_custom_read_bw_gb_s,
                stats.all_custom_write_bw_gb_s,
                stats.all_custom_copy_bw_gb_s,
                stats.all_custom_latency_ns);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      add_cache("l1", &BenchmarkResults::l1_read_bandwidth,
                &BenchmarkResults::l1_write_bandwidth,
                &BenchmarkResults::l1_copy_bandwidth,
                &BenchmarkResults::l1_latency, stats.all_l1_read_bw_gb_s,
                stats.all_l1_write_bw_gb_s, stats.all_l1_copy_bw_gb_s,
                stats.all_l1_latency_ns);
    }
    if (config.l2_buffer_size > 0) {
      add_cache("l2", &BenchmarkResults::l2_read_bandwidth,
                &BenchmarkResults::l2_write_bandwidth,
                &BenchmarkResults::l2_copy_bandwidth,
                &BenchmarkResults::l2_latency, stats.all_l2_read_bw_gb_s,
                stats.all_l2_write_bw_gb_s, stats.all_l2_copy_bw_gb_s,
                stats.all_l2_latency_ns);
    }
  }
  if (!cache.empty()) output[JsonKeys::CACHE] = cache;
}
