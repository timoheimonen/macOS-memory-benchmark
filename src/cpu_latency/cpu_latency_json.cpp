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
 * @file cpu_latency_json.cpp
 * @brief Schema v1 JSON serializer, quality warning collector, and DOM peak helper
 */

#include "cpu_latency/cpu_latency_json.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/audit/checkpoint_policy.h"
#include "core/config/version.h"
#include "utils/json_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

template <typename Integer>
OrderedJson decimal_string(Integer value) {
  return OrderedJson(std::to_string(value));
}

OrderedJson finite_or_null(double value) {
  return std::isfinite(value) ? OrderedJson(value) : OrderedJson(nullptr);
}

OrderedJson optional_finite_or_null(const std::optional<double>& value) {
  return (value.has_value() && std::isfinite(*value)) ? OrderedJson(*value)
                                                      : OrderedJson(nullptr);
}

OrderedJson cpu_raw_json(const std::optional<MachTimingSnapshot>& raw) {
  if (!raw.has_value()) {
    return nullptr;
  }
  return OrderedJson{{"start_ticks", decimal_string(raw->start_ticks)},
                     {"stop_ticks", decimal_string(raw->stop_ticks)},
                     {"delta_ticks", decimal_string(raw->delta_ticks)},
                     {"timebase_numer", raw->numer},
                     {"timebase_denom", raw->denom}};
}

OrderedJson validation_checks_json(
    const std::array<CpuLatencyValidationCheck, 5>& checks) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& ck : checks) {
    OrderedJson item;
    item["kind"] = ck.kind;
    item["applicable"] = ck.applicable;
    item["evaluated"] = ck.evaluated.has_value() ? OrderedJson(*ck.evaluated)
                                                 : OrderedJson(nullptr);
    item["valid"] =
        ck.valid.has_value() ? OrderedJson(*ck.valid) : OrderedJson(nullptr);
    item["reason_code"] = ck.reason_code;
    arr.push_back(std::move(item));
  }
  return arr;
}

OrderedJson buffer_evidence_json(const AllocatedBufferEvidence& ev) {
  OrderedJson out;
  out["requested_bytes"] = decimal_string(ev.requested_bytes);
  out["mapped_bytes"] = decimal_string(ev.mapped_bytes);
  out["non_cacheable_requested"] = ev.non_cacheable_requested;
  out["madvise_advice"] = ev.madvise_advice;
  out["madvise_applied"] = ev.madvise_applied;
  out["madvise_errno"] = ev.madvise_errno;
  out["read_only_protected"] = ev.read_only_protected;
  out["mprotect_errno"] = ev.mprotect_errno;
  return out;
}

OrderedJson chain_identity_json(const CpuLatencyChainIdentity& id) {
  OrderedJson out;
  out["node_count"] = id.node_count;
  out["stride_bytes"] = id.stride_bytes;
  out["chain_mode"] = latency_chain_mode_to_string(id.chain_mode);
  out["tlb_locality_kb"] = id.tlb_locality_kb;
  out["chain_seed_uint64_decimal"] = decimal_string(id.chain_seed);
  out["chain_permutation_sha256"] = id.permutation_sha256;
  return out;
}

OrderedJson final_pointer_json(const CpuLatencyFinalPointerVerification& fpv,
                               bool attempted) {
  OrderedJson out;
  out["expected_node"] = fpv.expected_node;
  out["actual_node"] = (attempted && fpv.actual_node.has_value())
                           ? OrderedJson(*fpv.actual_node)
                           : OrderedJson(nullptr);
  out["valid"] = (attempted && fpv.valid.has_value()) ? OrderedJson(*fpv.valid)
                                                      : OrderedJson(nullptr);
  return out;
}

OrderedJson sample_windows_json(
    const std::optional<CpuLatencySampleWindows>& sw_opt) {
  if (!sw_opt.has_value()) {
    return nullptr;
  }
  const CpuLatencySampleWindows& sw = *sw_opt;
  OrderedJson delta_ticks = OrderedJson::array();
  for (uint64_t dt : sw.sample_delta_ticks) {
    delta_ticks.push_back(decimal_string(dt));
  }
  OrderedJson samples = OrderedJson::array();
  for (double ns : sw.samples_ns) {
    samples.push_back(finite_or_null(ns));
  }
  OrderedJson end_nodes = OrderedJson::array();
  for (const auto& idx : sw.end_node_indices) {
    end_nodes.push_back(idx.has_value() ? OrderedJson(*idx)
                                        : OrderedJson(nullptr));
  }
  OrderedJson out;
  out["timebase"] = OrderedJson{{"timebase_numer", sw.timebase_numer},
                                {"timebase_denom", sw.timebase_denom}};
  out["sample_count"] = sw.sample_count;
  out["sample_delta_ticks"] = std::move(delta_ticks);
  out["samples_ns"] = std::move(samples);
  out["validation"] =
      OrderedJson{{"valid", sw.validation_valid},
                  {"all_end_pointers_valid", sw.all_end_pointers_valid},
                  {"end_node_indices", std::move(end_nodes)}};
  return out;
}

OrderedJson configuration_json(const CpuLatencyConfig& config) {
  OrderedJson resolved_sources;
  resolved_sources["buffer_size"] =
      config.swept_buffer_size
          ? "sweep"
          : (config.user_specified_buffer_size ? "explicit" : "default");
  resolved_sources["cache_size"] =
      config.swept_cache_size
          ? "sweep"
          : (config.user_specified_cache_size ? "explicit" : "default");
  resolved_sources["iterations"] =
      config.user_specified_iterations ? "explicit" : "auto";
  resolved_sources["latency_samples"] =
      config.user_specified_latency_samples ? "explicit" : "default";
  resolved_sources["latency_stride_bytes"] =
      config.swept_stride_bytes
          ? "sweep"
          : (config.user_specified_stride_bytes ? "explicit" : "default");
  resolved_sources["latency_chain_mode"] =
      config.swept_chain_mode
          ? "sweep"
          : (config.user_specified_chain_mode ? "explicit" : "default");
  resolved_sources["latency_tlb_locality_kb"] =
      config.swept_tlb_locality_kb
          ? "sweep"
          : (config.user_specified_tlb_locality_kb ? "explicit" : "default");
  resolved_sources["seed"] =
      config.user_specified_seed ? "explicit" : "generated";

  OrderedJson out;
  out["buffer_size_mb"] = config.buffer_size_mb;
  out["custom_cache_size_kb"] = config.custom_cache_size_kb.has_value()
                                    ? OrderedJson(*config.custom_cache_size_kb)
                                    : OrderedJson(nullptr);
  out["iterations"] = config.iterations.has_value()
                          ? OrderedJson(*config.iterations)
                          : OrderedJson(nullptr);
  out["work_policy"] = config.iterations.has_value() ? "explicit_fixed_work"
                                                     : "automatic_calibration";
  out["loop_count"] = config.loop_count;
  out["latency_samples"] = config.latency_samples;
  out["latency_stride_bytes"] = config.latency_stride_bytes;
  out["latency_chain_mode"] =
      latency_chain_mode_to_string(config.latency_chain_mode);
  out["latency_tlb_locality_kb"] = config.latency_tlb_locality_kb;
  out["compare_locality"] = config.compare_locality;
  out["non_cacheable"] = config.non_cacheable;
  out["base_seed_uint64_decimal"] = decimal_string(config.seed);
  out["seed_source"] = config.user_specified_seed ? "user" : "generated";
  out["output_file"] = config.output_file;
  out["argv"] = config.argv;
  out["resolved_sources"] = std::move(resolved_sources);
  return out;
}

OrderedJson resolved_plan_json(const CpuLatencyWorkPlan& work_plan,
                               const CpuLatencyResourceSet& resources,
                               const CpuLatencyResult& result) {
  OrderedJson comp;
  comp["version"] = work_plan.component_identities.version;
  comp["work_plan_version"] = work_plan.component_identities.work_plan_version;
  comp["chain_generator_version"] =
      work_plan.component_identities.chain_generator_version;
  comp["chain_version"] = work_plan.component_identities.chain_version;
  comp["run_policy_version"] =
      work_plan.component_identities.run_policy_version;

  OrderedJson targets_json = OrderedJson::object();
  OrderedJson resources_json = OrderedJson::object();
  OrderedJson chain_identities_json = OrderedJson::object();
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    const CpuLatencyTargetPlan& tplan = work_plan.targets[t_idx];
    const char* tname = cpu_latency_target_to_string(target);
    targets_json[tname] = OrderedJson{{"enabled", tplan.enabled},
                                      {"bytes", decimal_string(tplan.bytes)}};

    const CpuLatencyTargetBuffer& tbuf = resources.targets[t_idx];
    if (tbuf.active) {
      resources_json[tname] =
          OrderedJson{{"buffer", buffer_evidence_json(tbuf.buffer)}};
    }
    if (tbuf.chain_identity.active) {
      chain_identities_json[tname] = chain_identity_json(tbuf.chain_identity);
    }
  }
  if (work_plan.compare_locality) {
    if (resources.compare_locality_active) {
      resources_json["compare_buffer"] =
          buffer_evidence_json(resources.compare_buffer);
    }
    if (resources.locality_16k_identity.active) {
      chain_identities_json["locality_16k"] =
          chain_identity_json(resources.locality_16k_identity);
    }
    if (resources.locality_global_identity.active) {
      chain_identities_json["locality_global"] =
          chain_identity_json(resources.locality_global_identity);
    }
  }

  OrderedJson target_plans_json = OrderedJson::array();
  for (const auto& tp : result.target_plans) {
    OrderedJson item;
    item["plan_ref"] = tp.plan_ref;
    item["target"] = cpu_latency_target_to_string(tp.target);
    item["complete_chain_cycles"] = tp.complete_chain_cycles;
    item["access_count"] = tp.access_count;
    item["explicit_iterations"] = tp.explicit_iterations;
    target_plans_json.push_back(std::move(item));
  }

  OrderedJson frozen_refs = OrderedJson::object();
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    const auto& ref = result.frozen_plan_refs[t_idx];
    frozen_refs[cpu_latency_target_to_string(target)] =
        ref.has_value() ? OrderedJson(*ref) : OrderedJson(nullptr);
  }

  OrderedJson out;
  out["valid"] = work_plan.valid;
  out["reason_code"] = work_plan.reason_code;
  out["plan_identity"] =
      work_plan.plan_identity.empty()
          ? std::string(kCpuLatencyWorkPlanVersion) + "|invalid"
          : work_plan.plan_identity;
  out["component_identities"] = std::move(comp);
  out["targets"] = std::move(targets_json);
  out["resources"] = std::move(resources_json);
  out["chain_identities"] = std::move(chain_identities_json);
  out["target_plans"] = std::move(target_plans_json);
  out["frozen_plan_refs"] = std::move(frozen_refs);
  return out;
}

OrderedJson memory_budget_json(const CpuLatencyMemoryBudget& budget) {
  OrderedJson out;
  out["resource_rounding_bytes"] =
      decimal_string(budget.resource_rounding_bytes);
  out["transient_peak_bytes"] = decimal_string(budget.transient_peak_bytes);
  out["setup_peak_bytes"] = decimal_string(budget.setup_peak_bytes);
  out["runtime_peak_bytes"] = decimal_string(budget.runtime_peak_bytes);
  out["known_owned_peak_bytes"] = decimal_string(budget.known_owned_peak_bytes);
  out["admitted_budget_bytes"] = decimal_string(budget.admitted_budget_bytes);
  out["available_memory_bytes"] = decimal_string(budget.available_memory_bytes);
  out["allowed_memory_bytes"] = decimal_string(budget.allowed_memory_bytes);
  out["used_fallback"] = budget.used_fallback;
  out["valid"] = budget.valid;
  out["reason_code"] = budget.reason_code;
  return out;
}

OrderedJson calibration_json(const CpuLatencyWorkPlan& work_plan,
                             const CpuLatencyResult& result) {
  OrderedJson attempts_by_target = OrderedJson::object();
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    OrderedJson arr = OrderedJson::array();
    for (const auto& att : result.calibration_attempts[t_idx]) {
      const auto& ex = att.execution;
      OrderedJson ex_json;
      ex_json["status"] =
          !ex.available ? "unavailable" : (ex.valid ? "valid" : "invalid");
      ex_json["reason_code"] = ex.reason_code;
      ex_json["valid"] =
          ex.available ? OrderedJson(ex.valid) : OrderedJson(nullptr);
      ex_json["elapsed_seconds"] =
          (ex.available && ex.timing_evaluated && ex.timing_valid)
              ? finite_or_null(ex.elapsed_seconds)
              : OrderedJson(nullptr);
      ex_json["timing"] = OrderedJson{{"cpu_raw", cpu_raw_json(ex.cpu_raw)}};
      ex_json["validation"] =
          OrderedJson{{"checks", validation_checks_json(ex.validation_checks)}};

      OrderedJson att_json;
      att_json["attempt_index"] = att.attempt_index;
      att_json["target"] = cpu_latency_target_to_string(att.target);
      att_json["plan_ref"] = att.plan_ref.has_value()
                                 ? OrderedJson(*att.plan_ref)
                                 : OrderedJson(nullptr);
      att_json["purpose"] = att.purpose;
      att_json["duration_quality"] = att.duration_quality;
      att_json["terminal"] = att.terminal;
      att_json["valid"] = att.valid;
      att_json["reason_code"] = att.reason_code;
      att_json["execution"] = std::move(ex_json);
      arr.push_back(std::move(att_json));
    }
    attempts_by_target[cpu_latency_target_to_string(target)] = std::move(arr);
  }
  return OrderedJson{{"excluded_from_results", true},
                     {"attempts", std::move(attempts_by_target)}};
}

OrderedJson measurements_json(const CpuLatencyResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& row : result.measurements) {
    const auto& ex = row.execution;
    const bool measured = (row.status == CpuLatencyMeasurementStatus::Measured);

    OrderedJson timing_json;
    timing_json["evaluated"] = ex.timing_evaluated;
    timing_json["valid"] = ex.timing_evaluated ? OrderedJson(ex.timing_valid)
                                               : OrderedJson(nullptr);
    timing_json["elapsed_seconds"] =
        measured ? finite_or_null(ex.elapsed_seconds) : OrderedJson(nullptr);
    timing_json["diagnostic_elapsed_seconds"] =
        (!measured && row.attempted && ex.timing_evaluated)
            ? finite_or_null(ex.elapsed_seconds)
            : OrderedJson(nullptr);
    timing_json["cpu_raw"] =
        row.attempted ? cpu_raw_json(ex.cpu_raw) : OrderedJson(nullptr);

    OrderedJson ex_json;
    ex_json["status"] =
        !row.attempted ? "not_run" : (ex.valid ? "valid" : "invalid");
    ex_json["reason_code"] = row.attempted ? ex.reason_code : row.reason_code;
    ex_json["valid"] =
        row.attempted ? OrderedJson(ex.valid) : OrderedJson(nullptr);
    ex_json["timing"] = std::move(timing_json);
    ex_json["validation"] =
        OrderedJson{{"checks", validation_checks_json(ex.validation_checks)}};

    OrderedJson row_json;
    row_json["measurement_id"] = row.measurement_id;
    row_json["plan_ref"] = row.plan_ref.has_value()
                               ? OrderedJson(*row.plan_ref)
                               : OrderedJson(nullptr);
    row_json["target"] = cpu_latency_target_to_string(row.target);
    row_json["loop_index"] = row.loop_index;
    row_json["target_order_position"] = row.target_order_position;
    row_json["status"] = cpu_latency_measurement_status_to_string(row.status);
    row_json["reason_code"] = row.reason_code;
    row_json["attempted"] = row.attempted;
    row_json["completed_accesses"] = row.completed_accesses;
    row_json["completed_chain_cycles"] = row.completed_chain_cycles;
    row_json["latency_ns"] = optional_finite_or_null(row.latency_ns);
    row_json["final_pointer_verification"] =
        final_pointer_json(ex.final_pointer, row.attempted);
    row_json["sample_windows"] =
        row.attempted ? sample_windows_json(ex.sample_windows)
                      : OrderedJson(nullptr);
    row_json["execution"] = std::move(ex_json);
    arr.push_back(std::move(row_json));
  }
  return arr;
}

OrderedJson locality_subrun_json(const CpuLatencyLocalitySubrunResult& sub) {
  const bool measured = (sub.status == "measured");
  OrderedJson timing;
  timing["elapsed_seconds"] =
      measured ? finite_or_null(sub.elapsed_seconds) : OrderedJson(nullptr);
  timing["diagnostic_elapsed_seconds"] =
      (!measured && sub.attempted && sub.timing_evaluated)
          ? finite_or_null(sub.elapsed_seconds)
          : OrderedJson(nullptr);
  timing["cpu_raw"] =
      sub.attempted ? cpu_raw_json(sub.cpu_raw) : OrderedJson(nullptr);

  OrderedJson out;
  out["status"] = sub.status;
  out["reason_code"] = sub.reason_code;
  out["attempted"] = sub.attempted;
  out["completed_accesses"] = sub.completed_accesses;
  out["latency_ns"] = optional_finite_or_null(sub.latency_ns);
  out["final_pointer_verification"] =
      final_pointer_json(sub.final_pointer, sub.attempted);
  out["execution"] = OrderedJson{{"timing", std::move(timing)}};
  return out;
}

OrderedJson locality_comparison_json(const CpuLatencyConfig& config,
                                     const CpuLatencyResult& result) {
  if (!config.compare_locality) {
    return nullptr;
  }
  OrderedJson rounds = OrderedJson::array();
  for (const auto& rnd : result.locality_rounds) {
    rounds.push_back(OrderedJson{
        {"loop_index", rnd.loop_index},
        {"round_index", rnd.round_index},
        {"locality_16k", locality_subrun_json(rnd.locality_16k)},
        {"locality_global", locality_subrun_json(rnd.locality_global)}});
  }
  return OrderedJson{{"rounds", std::move(rounds)}};
}

OrderedJson statistics_json(const DescriptiveStatistics& stats, bool available) {
  if (!available) {
    return nullptr;
  }
  OrderedJson out;
  out["sample_count"] = stats.sample_count;
  out["average"] = finite_or_null(stats.average);
  out["min"] = finite_or_null(stats.min);
  out["max"] = finite_or_null(stats.max);
  out["median"] = finite_or_null(stats.median);
  out["p90"] = finite_or_null(stats.p90);
  out["p95"] = finite_or_null(stats.p95);
  out["p99"] = finite_or_null(stats.p99);
  out["stddev"] = finite_or_null(stats.stddev);
  out["coefficient_of_variation_pct"] =
      stats.coefficient_of_variation_defined
          ? finite_or_null(stats.coefficient_of_variation_pct)
          : OrderedJson(nullptr);
  out["median_absolute_deviation"] =
      finite_or_null(stats.median_absolute_deviation);
  return out;
}

OrderedJson aggregates_json(const CpuLatencyConfig& config,
                            const CpuLatencyWorkPlan& work_plan,
                            const CpuLatencyResult& result) {
  OrderedJson targets_json = OrderedJson::object();
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    const auto& agg = result.aggregates[t_idx];
    const bool available = !agg.accepted_measurement_ids.empty();

    OrderedJson pooled_json = nullptr;
    if (config.latency_samples > 0) {
      OrderedJson vals = OrderedJson::array();
      for (double v : agg.pooled_samples_ns) {
        vals.push_back(finite_or_null(v));
      }
      pooled_json = OrderedJson{
          {"semantics", "pooled-separate-sample-window-distribution"},
          {"values_ns", std::move(vals)},
          {"statistics",
           statistics_json(agg.pooled_statistics,
                           !agg.pooled_samples_ns.empty())}};
    }

    OrderedJson a;
    a["accepted_measurement_ids"] = agg.accepted_measurement_ids;
    a["value"] = optional_finite_or_null(agg.value);
    a["statistics"] = statistics_json(agg.statistics, available);
    a["quality"] = OrderedJson{
        {"repeatability_assessment_available",
         agg.repeatability_assessment_available},
        {"run_complete", agg.run_complete},
        {"cv_warning_threshold_pct", agg.cv_warning_threshold_pct},
        {"cv_warning", agg.cv_warning}};
    a["pooled_sample_distribution"] = std::move(pooled_json);
    targets_json[cpu_latency_target_to_string(target)] = std::move(a);
  }

  OrderedJson loc_agg = nullptr;
  if (config.compare_locality) {
    loc_agg = OrderedJson{
        {"locality_16k_latency_ns",
         optional_finite_or_null(
             result.locality_aggregate.locality_16k_latency_ns)},
        {"global_random_latency_ns",
         optional_finite_or_null(
             result.locality_aggregate.global_random_latency_ns)},
        {"locality_latency_delta_ns",
         optional_finite_or_null(
             result.locality_aggregate.locality_latency_delta_ns)}};
  }

  return OrderedJson{{"targets", std::move(targets_json)},
                     {"locality_comparison", std::move(loc_agg)}};
}

OrderedJson interpretation_json(const CpuLatencyConfig& config) {
  OrderedJson comparability = OrderedJson::array();
  comparability.push_back("same-methodology-version");
  comparability.push_back("same-frozen-work-plan-and-targets");
  comparability.push_back("same-software-version");
  comparability.push_back("sufficiently-similar-hardware");
  comparability.push_back("sufficiently-similar-thermal-and-power-state");

  OrderedJson out;
  out["dram_residency"] = "unverified";
  out["cache_residency"] = "unverified";
  out["non_cacheable_requested"] = config.non_cacheable;
  out["non_cacheable_is_best_effort_madvise"] = true;
  out["sample_windows_run_as_separate_pass_after_main_measurement"] = true;
  out["comparability_requires"] = std::move(comparability);
  return out;
}

OrderedJson seeds_json(const CpuLatencyConfig& config,
                       const CpuLatencyWorkPlan& work_plan) {
  OrderedJson domain_seeds = OrderedJson::object();
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_latency_target_index(target);
    domain_seeds[cpu_latency_target_to_string(target)] =
        decimal_string(work_plan.targets[t_idx].chain_seed);
  }
  if (work_plan.compare_locality) {
    if (work_plan.locality_16k_plan.has_value()) {
      domain_seeds["locality_16k"] =
          decimal_string(work_plan.locality_16k_plan->chain_seed);
    }
    if (work_plan.locality_global_plan.has_value()) {
      domain_seeds["locality_global"] =
          decimal_string(work_plan.locality_global_plan->chain_seed);
    }
  }
  return OrderedJson{
      {"base_seed_uint64_decimal", decimal_string(work_plan.base_seed)},
      {"source", config.user_specified_seed ? "user" : "generated"},
      {"target_domain_seeds", std::move(domain_seeds)}};
}

OrderedJson counters_json(const CpuLatencyCounters& c) {
  OrderedJson out;
  out["planned_loops"] = c.planned_loops;
  out["attempted_loops"] = c.attempted_loops;
  out["completed_loops"] = c.completed_loops;
  out["planned_measurements"] = c.planned_measurements;
  out["attempted_measurements"] = c.attempted_measurements;
  out["terminal_measurements"] = c.terminal_measurements;
  out["measured_measurements"] = c.measured_measurements;
  out["planned_accesses"] = decimal_string(c.planned_accesses);
  out["completed_accesses"] = decimal_string(c.completed_accesses);
  return out;
}

OrderedJson loop_records_json(const CpuLatencyResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& loop : result.loops) {
    OrderedJson planned = OrderedJson::array();
    for (CpuLatencyTarget t : loop.planned_order) {
      planned.push_back(cpu_latency_target_to_string(t));
    }
    OrderedJson realized = OrderedJson::array();
    for (CpuLatencyTarget t : loop.realized_order) {
      realized.push_back(cpu_latency_target_to_string(t));
    }
    arr.push_back(OrderedJson{
        {"loop_index", loop.loop_index},
        {"planned_order", std::move(planned)},
        {"realized_order", std::move(realized)},
        {"realized_order_count", loop.realized_order.size()},
        {"measurement_indexes", loop.measurement_indexes}});
  }
  return arr;
}

OrderedJson environment_snapshot_json(
    const AuditHostEnvironmentSnapshot& snap) {
  OrderedJson out;
  out["thermal_state"] = snap.thermal_state;
  out["low_power_mode_available"] = snap.low_power_mode_available;
  out["low_power_mode_enabled"] =
      snap.low_power_mode_available ? OrderedJson(snap.low_power_mode_enabled)
                                    : OrderedJson(nullptr);
  out["physical_memory_bytes"] =
      snap.physical_memory_bytes == 0
          ? OrderedJson(nullptr)
          : decimal_string(snap.physical_memory_bytes);
  return out;
}

OrderedJson environment_json(const CpuLatencyResultMetadata& metadata) {
  OrderedJson out;
  out["processor_name"] = metadata.processor_name;
  out["macos_version"] = metadata.macos_version;
  out["performance_core_count"] = metadata.performance_core_count;
  out["efficiency_core_count"] = metadata.efficiency_core_count;
  out["logical_core_count"] = metadata.logical_core_count;
  out["page_size_bytes"] = decimal_string(metadata.page_size_bytes);
  out["l1_data_cache_bytes"] = decimal_string(metadata.l1_data_cache_bytes);
  out["l2_data_cache_bytes"] = decimal_string(metadata.l2_data_cache_bytes);
  out["available_memory_bytes"] =
      metadata.available_memory_bytes == 0
          ? OrderedJson(nullptr)
          : decimal_string(metadata.available_memory_bytes);
  out["available_memory_source"] = metadata.available_memory_source;
  out["main_thread_qos"] =
      OrderedJson{{"requested", metadata.main_thread_qos.requested},
                  {"applied", metadata.main_thread_qos.applied},
                  {"code", metadata.main_thread_qos.code}};
  out["start"] = environment_snapshot_json(metadata.environment_start);
  out["end"] = environment_snapshot_json(metadata.environment_end);
  return out;
}

bool environment_is_non_nominal(
    const AuditHostEnvironmentSnapshot& snap) noexcept {
  const bool thermal_non_nominal =
      snap.thermal_state != "nominal" && snap.thermal_state != "unavailable";
  const bool low_power =
      snap.low_power_mode_available && snap.low_power_mode_enabled;
  return thermal_non_nominal || low_power;
}

void append_unique_warning(std::vector<std::string>& warnings,
                           std::string token) {
  if (std::find(warnings.begin(), warnings.end(), token) == warnings.end()) {
    warnings.push_back(std::move(token));
  }
}

}  // namespace

CpuLatencyJsonPeakEstimate calculate_cpu_latency_json_peak_estimate(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan) noexcept {
  if (config.output_file.empty() && !config.collect_json_for_sweep) {
    CpuLatencyJsonPeakEstimate empty_est;
    empty_est.valid = true;
    empty_est.reason_code = "valid";
    return empty_est;
  }
  return work_plan.json_peak_estimate;
}

std::vector<std::string> collect_cpu_latency_quality_warning_tokens(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResourceSet& resources,
    const CpuLatencyResultMetadata& metadata,
    const CpuLatencyResult& result) {
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
  if (config.non_cacheable) {
    bool madvise_missed = false;
    for (CpuLatencyTarget target : work_plan.active_targets) {
      const auto& tbuf = resources.targets[cpu_latency_target_index(target)];
      if (tbuf.active && !tbuf.buffer.madvise_applied) {
        madvise_missed = true;
        break;
      }
    }
    if (resources.compare_locality_active &&
        !resources.compare_buffer.madvise_applied) {
      madvise_missed = true;
    }
    if (madvise_missed) {
      append_unique_warning(warnings, "madvise-random-not-applied");
    }
  }
  if (!config.iterations.has_value()) {
    for (const auto& row : result.measurements) {
      if (row.status == CpuLatencyMeasurementStatus::Measured &&
          row.duration_quality != "within-target-window") {
        append_unique_warning(
            warnings, std::string(cpu_latency_target_to_string(row.target)) +
                          "-duration-" + row.duration_quality);
      }
    }
  }
  for (CpuLatencyTarget target : work_plan.active_targets) {
    const auto& agg = result.aggregates[cpu_latency_target_index(target)];
    if (agg.cv_warning) {
      append_unique_warning(
          warnings,
          std::string(cpu_latency_target_to_string(target)) + "-high-cv");
    }
  }
  return warnings;
}

nlohmann::ordered_json build_cpu_latency_json(
    const CpuLatencyConfig& config,
    const CpuLatencyWorkPlan& work_plan,
    const CpuLatencyResourceSet& resources,
    const CpuLatencyResultMetadata& metadata,
    const CpuLatencyResult& result) {
  OrderedJson out;
  out["schema_version"] = kCpuLatencyJsonSchemaVersion;
  out["mode"] = kCpuLatencyJsonModeName;
  out["methodology_version"] = kCpuLatencyMethodologyVersion;
  out["software"] = OrderedJson{
      {"version", SOFTVERSION},
      {"timestamp", metadata.timestamp.empty() ? build_utc_timestamp()
                                             : metadata.timestamp}};
  out["build_manifest"] = metadata.build_manifest;
  out["configuration"] = configuration_json(config);
  out["resolved_plan"] = resolved_plan_json(work_plan, resources, result);
  out["memory_budget"] = memory_budget_json(work_plan.memory_budget);
  out["calibration"] = calibration_json(work_plan, result);
  out["measurements"] = measurements_json(result);
  out["locality_comparison"] = locality_comparison_json(config, result);
  out["aggregates"] = aggregates_json(config, work_plan, result);
  out["status"] = cpu_latency_run_status_to_string(result.status);
  out["reason_code"] = result.reason_code;
  out["results_complete"] = result.results_complete;
  out["run_accepted"] = result.run_accepted;
  out["interpretation"] = interpretation_json(config);
  out["diagnostic"] =
      result.diagnostic.empty() ? OrderedJson(nullptr)
                                : OrderedJson(result.diagnostic);
  out["interruption_requested"] = result.interruption_requested;
  out["target_order_balance_complete"] = result.target_order_balance_complete;
  out["seeds"] = seeds_json(config, work_plan);
  out["counters"] = counters_json(result.counters);
  out["checkpoint_lifecycle"] =
      build_checkpoint_lifecycle_json(CheckpointLifecycleState{
          result.checkpoint_failed,
          result.prior_file_writer_attempts,
          result.prior_successful_file_writes,
          result.snapshot_request,
          result.snapshot_interval_loops,
      });
  out["loop_records"] = loop_records_json(result);
  out["environment"] = environment_json(metadata);

  OrderedJson warnings_json = OrderedJson::array();
  for (const auto& w : collect_cpu_latency_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    warnings_json.push_back(w);
  }
  out["quality_warnings"] = std::move(warnings_json);
  return out;
}
