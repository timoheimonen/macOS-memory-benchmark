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
 * @file cpu_bandwidth_json.cpp
 * @brief Schema v1 JSON serializer, quality warning collector, and DOM peak helper
 */

#include "cpu_bandwidth/cpu_bandwidth_json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "core/audit/checkpoint_policy.h"
#include "core/config/version.h"
#include "utils/json_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr std::array<CpuBandwidthOperation, kCpuBandwidthOperationCount>
    kOperations = {
        CpuBandwidthOperation::Read,
        CpuBandwidthOperation::Write,
        CpuBandwidthOperation::Copy,
};

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

OrderedJson run_checksum_json(const CpuBandwidthRunChecksum& checksum) {
  return OrderedJson{
      {"state_a_uint64_decimal", decimal_string(checksum.state_a)},
      {"state_b_uint64_decimal", decimal_string(checksum.state_b)}};
}

OrderedJson worker_checksums_json(
    const std::vector<CpuBandwidthExpectedWorkerChecksum>& workers) {
  OrderedJson out = OrderedJson::array();
  for (const auto& w : workers) {
    out.push_back(OrderedJson{
        {"worker_index", w.worker_index},
        {"state_a_uint64_decimal", decimal_string(w.state_a)},
        {"state_b_uint64_decimal", decimal_string(w.state_b)},
        {"exact_bytes_read", decimal_string(w.exact_bytes_read)},
        {"pass_count_uint64_decimal", decimal_string(w.pass_count)}});
  }
  return out;
}

OrderedJson validation_checks_json(
    const std::array<CpuBandwidthValidationCheck, 5>& checks) {
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

OrderedJson configuration_json(const CpuBandwidthConfig& config) {
  OrderedJson resolved_sources;
  resolved_sources["buffer_size"] =
      config.swept_buffer_size
          ? "sweep"
          : (config.user_specified_buffer_size ? "explicit" : "default");
  resolved_sources["cache_size"] =
      config.swept_cache_size
          ? "sweep"
          : (config.user_specified_cache_size ? "explicit" : "default");
  resolved_sources["workers"] =
      config.swept_threads
          ? "sweep"
          : (config.user_specified_workers ? "explicit" : "auto");
  resolved_sources["iterations"] =
      config.user_specified_iterations ? "explicit" : "auto";
  resolved_sources["seed"] =
      config.user_specified_seed ? "explicit" : "generated";

  OrderedJson out;
  out["buffer_size_mb"] = config.buffer_size_mb;
  out["custom_cache_size_kb"] = config.custom_cache_size_kb.has_value()
                                    ? OrderedJson(*config.custom_cache_size_kb)
                                    : OrderedJson(nullptr);
  out["requested_workers"] = config.requested_workers;
  out["available_workers"] = config.available_workers;
  out["worker_source"] = config.user_specified_workers ? "user" : "auto";
  out["iterations"] = config.iterations.has_value()
                          ? OrderedJson(*config.iterations)
                          : OrderedJson(nullptr);
  out["work_policy"] = config.iterations.has_value() ? "explicit_fixed_work"
                                                     : "automatic_calibration";
  out["loop_count"] = config.loop_count;
  out["non_cacheable"] = config.non_cacheable;
  out["base_seed_uint64_decimal"] = decimal_string(config.seed);
  out["seed_source"] = config.user_specified_seed ? "user" : "generated";
  out["output_file"] = config.output_file;
  out["argv"] = config.argv;
  out["resolved_sources"] = std::move(resolved_sources);
  return out;
}

OrderedJson resolved_plan_json(const CpuBandwidthWorkPlan& work_plan,
                               const CpuBandwidthResourceSet& resources,
                               const CpuBandwidthResult& result) {
  OrderedJson comp;
  comp["version"] = work_plan.component_identities.version;
  comp["work_plan_version"] = work_plan.component_identities.work_plan_version;
  comp["buffer_pattern_version"] =
      work_plan.component_identities.buffer_pattern_version;
  comp["read_checksum_version"] =
      work_plan.component_identities.read_checksum_version;
  comp["run_policy_version"] = work_plan.component_identities.run_policy_version;

  OrderedJson targets_json = OrderedJson::object();
  OrderedJson resources_json = OrderedJson::object();
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    const CpuBandwidthTargetPlan& tplan = work_plan.targets[t_idx];
    OrderedJson spans_json = OrderedJson::array();
    for (const auto& sp : tplan.worker_spans) {
      spans_json.push_back(OrderedJson{
          {"offset_bytes", decimal_string(sp.offset_bytes)},
          {"span_bytes", decimal_string(sp.span_bytes)}});
    }
    const char* tname = cpu_bandwidth_target_to_string(target);
    targets_json[tname] = OrderedJson{
        {"enabled", tplan.enabled},
        {"bytes", decimal_string(tplan.bytes)},
        {"effective_workers", tplan.effective_workers},
        {"worker_spans", std::move(spans_json)}};

    const CpuBandwidthTargetBuffers& tbuf = resources.targets[t_idx];
    if (tbuf.active) {
      resources_json[tname] = OrderedJson{
          {"src", buffer_evidence_json(tbuf.src)},
          {"dst", buffer_evidence_json(tbuf.dst)}};
    }
  }

  OrderedJson op_plans_json = OrderedJson::array();
  for (const auto& op_plan : result.operation_plans) {
    OrderedJson exp_ck;
    if (op_plan.operation == CpuBandwidthOperation::Read &&
        op_plan.expected_checksum.available) {
      exp_ck["status"] = "available";
      exp_ck["reason_code"] = op_plan.expected_checksum.reason_code;
      exp_ck["expected_worker_checksums"] =
          worker_checksums_json(op_plan.expected_checksum.workers);
      exp_ck["expected_run_checksum"] =
          run_checksum_json(op_plan.expected_checksum.run_checksum);
    } else {
      exp_ck["status"] = "not_applicable";
      exp_ck["reason_code"] = "not-applicable";
      exp_ck["expected_worker_checksums"] = nullptr;
      exp_ck["expected_run_checksum"] = nullptr;
    }

    OrderedJson item;
    item["plan_ref"] = op_plan.plan_ref;
    item["target"] = cpu_bandwidth_target_to_string(op_plan.target);
    item["operation"] = cpu_bandwidth_operation_to_string(op_plan.operation);
    item["passes"] = op_plan.passes;
    item["explicit_iterations"] = op_plan.explicit_iterations;
    item["exact_payload_bytes"] = decimal_string(op_plan.exact_payload_bytes);
    item["op_seed_uint64_decimal"] = decimal_string(op_plan.op_seed);
    item["expected_checksum"] = std::move(exp_ck);
    op_plans_json.push_back(std::move(item));
  }

  OrderedJson frozen_refs = OrderedJson::object();
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    OrderedJson t_refs = OrderedJson::object();
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto& ref = result.frozen_plan_refs[t_idx][op_i];
      t_refs[cpu_bandwidth_operation_to_string(kOperations[op_i])] =
          ref.has_value() ? OrderedJson(*ref) : OrderedJson(nullptr);
    }
    frozen_refs[cpu_bandwidth_target_to_string(target)] = std::move(t_refs);
  }

  OrderedJson out;
  out["valid"] = work_plan.valid;
  out["reason_code"] = work_plan.reason_code;
  out["plan_identity"] =
      work_plan.plan_identity.empty()
          ? std::string(kCpuBandwidthWorkPlanVersion) + "|invalid"
          : work_plan.plan_identity;
  out["component_identities"] = std::move(comp);
  out["targets"] = std::move(targets_json);
  out["resources"] = std::move(resources_json);
  out["operation_plans"] = std::move(op_plans_json);
  out["frozen_plan_refs"] = std::move(frozen_refs);
  return out;
}

OrderedJson memory_budget_json(const CpuBandwidthMemoryBudget& budget) {
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

OrderedJson compact_checksum_json(CpuBandwidthOperation op,
                                  const CpuBandwidthOperationExecution& ex) {
  OrderedJson ck;
  if (op != CpuBandwidthOperation::Read) {
    ck["status"] = "not_applicable";
    ck["reason_code"] = "not-applicable";
    ck["checksum_valid"] = nullptr;
    ck["actual_run_checksum"] = nullptr;
    return ck;
  }
  if (!ex.checksum_evaluated) {
    ck["status"] = "not_evaluated";
    ck["reason_code"] = ex.reason_code;
    ck["checksum_valid"] = nullptr;
    ck["actual_run_checksum"] = nullptr;
    return ck;
  }
  ck["status"] = ex.checksum_valid ? "valid" : "invalid";
  ck["reason_code"] = ex.checksum_valid ? "valid" : ex.reason_code;
  ck["checksum_valid"] = ex.checksum_valid;
  ck["actual_run_checksum"] = run_checksum_json(ex.actual_run_checksum);
  return ck;
}

OrderedJson calibration_json(const CpuBandwidthWorkPlan& work_plan,
                             const CpuBandwidthResult& result) {
  OrderedJson attempts_by_target = OrderedJson::object();
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    OrderedJson per_op = OrderedJson::object();
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto op = kOperations[op_i];
      OrderedJson arr = OrderedJson::array();
      for (const auto& att : result.calibration_attempts[t_idx][op_i]) {
        const auto& ex = att.execution;
        OrderedJson ex_json;
        ex_json["status"] = !ex.available ? "unavailable"
                                          : (ex.valid ? "valid" : "invalid");
        ex_json["reason_code"] = ex.reason_code;
        ex_json["valid"] =
            ex.available ? OrderedJson(ex.valid) : OrderedJson(nullptr);
        ex_json["elapsed_seconds"] =
            (ex.available && ex.timing_evaluated && ex.timing_valid)
                ? finite_or_null(ex.elapsed_seconds)
                : OrderedJson(nullptr);
        ex_json["timing"] = OrderedJson{{"cpu_raw", cpu_raw_json(ex.cpu_raw)}};
        ex_json["requested_workers"] = ex.requested_workers;
        ex_json["created_workers"] = ex.created_workers;
        ex_json["completed_workers"] = ex.completed_workers;
        ex_json["write_pattern_uint64_decimal"] =
            (op == CpuBandwidthOperation::Write && ex.write_pattern.has_value())
                ? decimal_string(*ex.write_pattern)
                : OrderedJson(nullptr);
        if (op == CpuBandwidthOperation::Write && ex.attempt_seed.has_value()) {
          ex_json["attempt_seed_uint64_decimal"] =
              decimal_string(*ex.attempt_seed);
        }
        ex_json["checksum"] = compact_checksum_json(op, ex);
        ex_json["validation"] =
            OrderedJson{{"checks", validation_checks_json(ex.validation_checks)}};

        OrderedJson att_json;
        att_json["attempt_index"] = att.attempt_index;
        att_json["target"] = cpu_bandwidth_target_to_string(att.target);
        att_json["operation"] = cpu_bandwidth_operation_to_string(att.operation);
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
      per_op[cpu_bandwidth_operation_to_string(op)] = std::move(arr);
    }
    attempts_by_target[cpu_bandwidth_target_to_string(target)] =
        std::move(per_op);
  }

  return OrderedJson{{"excluded_from_results", true},
                     {"attempts", std::move(attempts_by_target)}};
}

OrderedJson measurement_checksum_json(const CpuBandwidthMeasurementRow& row) {
  OrderedJson ck;
  const auto& ex = row.execution;
  if (row.operation != CpuBandwidthOperation::Read) {
    ck["status"] = "not_applicable";
    ck["reason_code"] = "not-applicable";
    ck["checksum_valid"] = nullptr;
    ck["actual_worker_checksums"] = nullptr;
    ck["actual_run_checksum"] = nullptr;
    return ck;
  }
  if (!row.attempted || !ex.checksum_evaluated) {
    ck["status"] = "not_evaluated";
    ck["reason_code"] = row.attempted ? ex.reason_code : row.reason_code;
    ck["checksum_valid"] = nullptr;
    ck["actual_worker_checksums"] = nullptr;
    ck["actual_run_checksum"] = nullptr;
    return ck;
  }
  ck["status"] = ex.checksum_valid ? "valid" : "invalid";
  ck["reason_code"] = ex.checksum_valid ? "valid" : ex.reason_code;
  ck["checksum_valid"] = ex.checksum_valid;
  ck["actual_worker_checksums"] =
      worker_checksums_json(ex.actual_worker_checksums);
  ck["actual_run_checksum"] = run_checksum_json(ex.actual_run_checksum);
  return ck;
}

OrderedJson measurements_json(const CpuBandwidthResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& row : result.measurements) {
    const auto& ex = row.execution;
    const bool measured = (row.status == CpuBandwidthMeasurementStatus::Measured);

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
    ex_json["requested_workers"] = ex.requested_workers;
    ex_json["created_workers"] = ex.created_workers;
    ex_json["completed_workers"] = ex.completed_workers;
    ex_json["write_pattern_uint64_decimal"] =
        (row.attempted && row.operation == CpuBandwidthOperation::Write &&
         ex.write_pattern.has_value())
            ? decimal_string(*ex.write_pattern)
            : OrderedJson(nullptr);
    ex_json["timing"] = std::move(timing_json);
    ex_json["validation"] =
        OrderedJson{{"checks", validation_checks_json(ex.validation_checks)}};
    if (row.attempted && row.operation == CpuBandwidthOperation::Write &&
        ex.attempt_seed.has_value()) {
      ex_json["attempt_seed_uint64_decimal"] =
          decimal_string(*ex.attempt_seed);
    }

    OrderedJson row_json;
    row_json["measurement_id"] = row.measurement_id;
    row_json["plan_ref"] = row.plan_ref.has_value()
                               ? OrderedJson(*row.plan_ref)
                               : OrderedJson(nullptr);
    row_json["target"] = cpu_bandwidth_target_to_string(row.target);
    row_json["operation"] = cpu_bandwidth_operation_to_string(row.operation);
    row_json["loop_index"] = row.loop_index;
    row_json["target_order_position"] = row.target_order_position;
    row_json["operation_order_position"] = row.operation_order_position;
    row_json["status"] = cpu_bandwidth_measurement_status_to_string(row.status);
    row_json["reason_code"] = row.reason_code;
    row_json["attempted"] = row.attempted;
    row_json["completed_passes"] = row.completed_passes;
    row_json["completed_exact_payload_bytes"] =
        decimal_string(row.completed_exact_payload_bytes);
    row_json["bandwidth_gb_s"] = optional_finite_or_null(row.bandwidth_gb_s);
    row_json["execution"] = std::move(ex_json);
    row_json["checksum"] = measurement_checksum_json(row);
    arr.push_back(std::move(row_json));
  }
  return arr;
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

OrderedJson aggregates_json(const CpuBandwidthWorkPlan& work_plan,
                            const CpuBandwidthResult& result) {
  OrderedJson targets_json = OrderedJson::object();
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    OrderedJson per_op = OrderedJson::object();
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto& agg = result.aggregates[t_idx][op_i];
      const bool available = !agg.accepted_measurement_ids.empty();
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
      per_op[cpu_bandwidth_operation_to_string(kOperations[op_i])] =
          std::move(a);
    }
    targets_json[cpu_bandwidth_target_to_string(target)] = std::move(per_op);
  }
  return OrderedJson{{"targets", std::move(targets_json)}};
}

OrderedJson interpretation_json(const CpuBandwidthConfig& config) {
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
  out["single_pass_xor_collision_caveat"] =
      "Per-pass read XOR detects single-bit and slice-address errors, but even-parity identical flips within one pass can cancel in 64-bit XOR.";
  out["write_copy_post_validation_verifies_final_state_not_intermediate_passes"] =
      true;
  out["comparability_requires"] = std::move(comparability);
  return out;
}

OrderedJson seeds_json(const CpuBandwidthConfig& config,
                       const CpuBandwidthWorkPlan& work_plan) {
  OrderedJson domain_seeds = OrderedJson::object();
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    domain_seeds[cpu_bandwidth_target_to_string(target)] =
        decimal_string(work_plan.targets[t_idx].src_seed);
  }
  domain_seeds["pre_copy_sentinel_uint64_decimal"] =
      decimal_string(work_plan.pre_copy_sentinel_seed);

  return OrderedJson{
      {"base_seed_uint64_decimal", decimal_string(work_plan.base_seed)},
      {"source", config.user_specified_seed ? "user" : "generated"},
      {"buffer_domain_seeds", std::move(domain_seeds)}};
}

OrderedJson counters_json(const CpuBandwidthCounters& c) {
  OrderedJson out;
  out["planned_loops"] = c.planned_loops;
  out["attempted_loops"] = c.attempted_loops;
  out["completed_loops"] = c.completed_loops;
  out["planned_measurements"] = c.planned_measurements;
  out["attempted_measurements"] = c.attempted_measurements;
  out["terminal_measurements"] = c.terminal_measurements;
  out["measured_measurements"] = c.measured_measurements;
  out["planned_passes"] = decimal_string(c.planned_passes);
  out["completed_passes"] = decimal_string(c.completed_passes);
  out["planned_exact_payload_bytes"] =
      decimal_string(c.planned_exact_payload_bytes);
  out["completed_exact_payload_bytes"] =
      decimal_string(c.completed_exact_payload_bytes);
  return out;
}

OrderedJson loop_records_json(const CpuBandwidthResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& loop : result.loops) {
    OrderedJson planned = OrderedJson::array();
    for (const auto& step : loop.planned_order) {
      planned.push_back(OrderedJson{
          {"target", cpu_bandwidth_target_to_string(step.target)},
          {"operation", cpu_bandwidth_operation_to_string(step.operation)}});
    }
    OrderedJson realized = OrderedJson::array();
    for (const auto& step : loop.realized_order) {
      realized.push_back(OrderedJson{
          {"target", cpu_bandwidth_target_to_string(step.target)},
          {"operation", cpu_bandwidth_operation_to_string(step.operation)}});
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

OrderedJson environment_json(const CpuBandwidthResultMetadata& metadata) {
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

CpuBandwidthJsonPeakEstimate calculate_cpu_bandwidth_json_peak_estimate(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan) noexcept {
  if (config.output_file.empty() && !config.collect_json_for_sweep) {
    CpuBandwidthJsonPeakEstimate empty_est;
    empty_est.valid = true;
    empty_est.reason_code = "valid";
    return empty_est;
  }
  return work_plan.json_peak_estimate;
}

std::vector<std::string> collect_cpu_bandwidth_quality_warning_tokens(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result) {
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
                  [](const CpuBandwidthMeasurementRow& row) {
                    return row.attempted && row.execution.qos_failed_workers > 0;
                  })) {
    append_unique_warning(warnings, "worker-qos-not-applied");
  }
  if (config.non_cacheable) {
    for (CpuBandwidthTarget target : work_plan.active_targets) {
      const auto& tbuf = resources.targets[cpu_bandwidth_target_index(target)];
      if (tbuf.active && (!tbuf.src.madvise_applied || !tbuf.dst.madvise_applied)) {
        append_unique_warning(warnings, "madvise-random-not-applied");
        break;
      }
    }
  }
  if (!config.iterations.has_value()) {
    for (const auto& row : result.measurements) {
      if (row.status == CpuBandwidthMeasurementStatus::Measured &&
          row.duration_quality != "within-target-window") {
        append_unique_warning(
            warnings,
            std::string(cpu_bandwidth_target_to_string(row.target)) + "-" +
                cpu_bandwidth_operation_to_string(row.operation) + "-duration-" +
                row.duration_quality);
      }
    }
  }
  for (CpuBandwidthTarget target : work_plan.active_targets) {
    const size_t t_idx = cpu_bandwidth_target_index(target);
    for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
      const auto& agg = result.aggregates[t_idx][op_i];
      if (agg.cv_warning) {
        append_unique_warning(
            warnings,
            std::string(cpu_bandwidth_target_to_string(target)) + "-" +
                cpu_bandwidth_operation_to_string(kOperations[op_i]) +
                "-high-cv");
      }
    }
  }
  return warnings;
}

nlohmann::ordered_json build_cpu_bandwidth_json(
    const CpuBandwidthConfig& config,
    const CpuBandwidthWorkPlan& work_plan,
    const CpuBandwidthResourceSet& resources,
    const CpuBandwidthResultMetadata& metadata,
    const CpuBandwidthResult& result) {
  OrderedJson out;
  out["schema_version"] = kCpuBandwidthJsonSchemaVersion;
  out["mode"] = kCpuBandwidthJsonModeName;
  out["methodology_version"] = kCpuBandwidthMethodologyVersion;
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
  out["aggregates"] = aggregates_json(work_plan, result);
  out["status"] = cpu_bandwidth_run_status_to_string(result.status);
  out["reason_code"] = result.reason_code;
  out["results_complete"] = result.results_complete;
  out["run_accepted"] = result.run_accepted;
  out["interpretation"] = interpretation_json(config);
  out["diagnostic"] =
      result.diagnostic.empty() ? OrderedJson(nullptr)
                                : OrderedJson(result.diagnostic);
  out["interruption_requested"] = result.interruption_requested;
  out["operation_order_balance_complete"] =
      result.operation_order_balance_complete;
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
  for (const auto& w : collect_cpu_bandwidth_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    warnings_json.push_back(w);
  }
  out["quality_warnings"] = std::move(warnings_json);
  return out;
}
