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
 * @file patterns_json.cpp
 * @brief Schema v1 JSON serializer, quality warning collector, and DOM peak helper
 */

#include "pattern_benchmark/patterns_json.h"

#include <array>
#include <cmath>
#include <string>
#include <utility>

#include "core/audit/checkpoint_policy.h"
#include "core/config/version.h"
#include "utils/json_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr std::array<PatternOperation, kPatternsOperationCount> kOperations = {
    PatternOperation::Read,
    PatternOperation::Write,
    PatternOperation::Copy,
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

OrderedJson run_checksum_json(const PatternsRunChecksum& checksum) {
  return OrderedJson{
      {"state_a_uint64_decimal", decimal_string(checksum.state_a)},
      {"state_b_uint64_decimal", decimal_string(checksum.state_b)}};
}

OrderedJson worker_checksums_json(
    const std::vector<PatternsExpectedWorkerChecksum>& workers) {
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
    const std::array<PatternsValidationCheck, kPatternsValidationCheckCount>&
        checks) {
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

OrderedJson configuration_json(const PatternsConfig& config) {
  OrderedJson resolved_sources;
  resolved_sources["buffer_size"] =
      config.swept_buffer_size
          ? "sweep"
          : (config.user_specified_buffer_size ? "explicit" : "default");
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

OrderedJson resolved_plan_json(const PatternsWorkPlan& work_plan,
                               const PatternsResourceSet& resources,
                               const PatternsResult& result) {
  OrderedJson comp;
  comp["version"] = work_plan.component_identities.version;
  comp["work_plan_version"] = work_plan.component_identities.work_plan_version;
  comp["buffer_pattern_version"] =
      work_plan.component_identities.buffer_pattern_version;
  comp["read_checksum_version"] =
      work_plan.component_identities.read_checksum_version;
  comp["random_index_generator_version"] =
      work_plan.component_identities.random_index_generator_version;
  comp["run_policy_version"] = work_plan.component_identities.run_policy_version;

  OrderedJson patterns_json = OrderedJson::object();
  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    const PatternPlanEntry& pplan = work_plan.patterns[p_idx];
    const char* pname = patterns_kind_to_string(kind);

    // Normalize invalid work plans (`!work_plan.valid`) and early failures
    // (`buffer_bytes == 0`) so unavailable entries never masquerade as `active`
    // or `buffer-too-small-for-stride-transition`.
    const bool resolved_enabled = work_plan.valid && pplan.enabled;
    const std::string resolved_status =
        !work_plan.valid
            ? std::string("unavailable")
            : (pplan.enabled
                   ? std::string("active")
                   : (pplan.buffer_bytes > 0 ? pplan.status
                                             : std::string("unavailable")));
    const std::string resolved_reason =
        !work_plan.valid
            ? (work_plan.reason_code.empty() ? std::string("unavailable")
                                             : work_plan.reason_code)
            : (pplan.enabled
                   ? pplan.reason_code
                   : (pplan.buffer_bytes > 0
                          ? pplan.reason_code
                          : (work_plan.reason_code.empty()
                                 ? std::string("unavailable")
                                 : work_plan.reason_code)));

    OrderedJson spans_json = OrderedJson::array();
    for (const auto& sp : pplan.worker_spans) {
      spans_json.push_back(OrderedJson{
          {"offset_bytes", decimal_string(sp.offset_bytes)},
          {"span_bytes", decimal_string(sp.span_bytes)},
          {"accesses_per_pass", decimal_string(sp.accesses_per_pass)},
          {"random_access_count", sp.random_access_count}});
    }

    OrderedJson entry_json;
    entry_json["enabled"] = resolved_enabled;
    entry_json["status"] = resolved_status;
    entry_json["reason_code"] = resolved_reason;
    entry_json["access_size_bytes"] = pplan.access_size_bytes;
    entry_json["stride_bytes"] = pplan.stride_bytes;
    entry_json["phase_period_passes"] = pplan.phase_period_passes;
    entry_json["native_page_size_bytes"] = pplan.native_page_size_bytes;
    entry_json["stride_equals_native_page_size"] =
        pplan.stride_equals_native_page_size;
    entry_json["large_page_backing_verified"] =
        pplan.large_page_backing_verified;
    entry_json["large_page_backing_status"] = pplan.large_page_backing_status;
    entry_json["effective_workers"] = pplan.effective_workers;
    entry_json["worker_spans"] = std::move(spans_json);
    if (kind == PatternKind::Random) {
      entry_json["random_workload"] = OrderedJson{
          {"random_seed_uint64_decimal",
           decimal_string(pplan.random_workload.random_seed)},
          {"selected_accesses_per_pass",
           pplan.random_workload.selected_accesses_per_pass},
          {"distinct_address_count",
           pplan.random_workload.distinct_address_count},
          {"logical_working_set_bytes",
           decimal_string(pplan.random_workload.logical_working_set_bytes)},
          {"random_indices_sha256",
           pplan.random_workload.random_indices_sha256.empty()
               ? OrderedJson(nullptr)
               : OrderedJson(pplan.random_workload.random_indices_sha256)}};
    }
    patterns_json[pname] = std::move(entry_json);
  }

  OrderedJson resources_json = OrderedJson::object();
  if (resources.valid || resources.src.ptr != nullptr ||
      resources.dst.ptr != nullptr) {
    resources_json["src"] = buffer_evidence_json(resources.src);
    resources_json["dst"] = buffer_evidence_json(resources.dst);
    resources_json["random_index_table_bytes"] =
        decimal_string(resources.random_index_table_bytes);
    resources_json["random_validation_index_bytes"] =
        decimal_string(resources.random_validation_index_bytes);
  }

  OrderedJson op_plans_json = OrderedJson::array();
  for (const auto& op_plan : result.operation_plans) {
    OrderedJson exp_ck;
    if (op_plan.operation == PatternOperation::Read &&
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
    item["pattern"] = patterns_kind_to_string(op_plan.pattern);
    item["operation"] = patterns_operation_to_string(op_plan.operation);
    item["passes"] = op_plan.passes;
    item["explicit_iterations"] = op_plan.explicit_iterations;
    item["total_accesses"] = decimal_string(op_plan.total_accesses);
    item["exact_payload_bytes"] = decimal_string(op_plan.exact_payload_bytes);
    item["accesses_per_pass"] = decimal_string(op_plan.accesses_per_pass);
    item["min_accesses_per_pass"] =
        decimal_string(op_plan.min_accesses_per_pass);
    item["max_accesses_per_pass"] =
        decimal_string(op_plan.max_accesses_per_pass);
    item["distinct_address_count"] =
        decimal_string(op_plan.distinct_address_count);
    item["logical_working_set_bytes"] =
        decimal_string(op_plan.logical_working_set_bytes);
    item["completed_phase_cycles"] = op_plan.completed_phase_cycles;
    item["has_untouched_dst_bytes"] = op_plan.has_untouched_dst_bytes;
    item["op_seed_uint64_decimal"] = decimal_string(op_plan.op_seed);
    item["expected_checksum"] = std::move(exp_ck);
    op_plans_json.push_back(std::move(item));
  }

  OrderedJson frozen_refs = OrderedJson::object();
  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    OrderedJson p_refs = OrderedJson::object();
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto& ref = result.frozen_plan_refs[p_idx][op_i];
      p_refs[patterns_operation_to_string(kOperations[op_i])] =
          ref.has_value() ? OrderedJson(*ref) : OrderedJson(nullptr);
    }
    frozen_refs[patterns_kind_to_string(kind)] = std::move(p_refs);
  }

  OrderedJson out;
  out["valid"] = work_plan.valid;
  out["reason_code"] = work_plan.reason_code;
  out["plan_identity"] =
      work_plan.plan_identity.empty()
          ? std::string(kPatternsWorkPlanVersion) + "|invalid"
          : work_plan.plan_identity;
  out["component_identities"] = std::move(comp);
  out["patterns"] = std::move(patterns_json);
  out["resources"] = std::move(resources_json);
  out["operation_plans"] = std::move(op_plans_json);
  out["frozen_plan_refs"] = std::move(frozen_refs);
  return out;
}

OrderedJson memory_budget_json(const PatternsMemoryBudget& budget) {
  OrderedJson out;
  out["requested_data_bytes"] = decimal_string(budget.requested_data_bytes);
  out["committed_data_bytes"] = decimal_string(budget.committed_data_bytes);
  out["resource_rounding_bytes"] =
      decimal_string(budget.resource_rounding_bytes);
  out["index_structure_bytes"] = decimal_string(budget.index_structure_bytes);
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

OrderedJson compact_checksum_json(PatternOperation op,
                                  const PatternsOperationExecution& ex) {
  OrderedJson ck;
  if (op != PatternOperation::Read) {
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

OrderedJson calibration_json(const PatternsWorkPlan& work_plan,
                             const PatternsResult& result) {
  OrderedJson attempts_by_pattern = OrderedJson::object();
  if (work_plan.valid) {
    for (PatternKind kind : work_plan.active_patterns) {
      const size_t p_idx = patterns_kind_index(kind);
      OrderedJson per_op = OrderedJson::object();
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        const auto op = kOperations[op_i];
        OrderedJson arr = OrderedJson::array();
        for (const auto& att : result.calibration_attempts[p_idx][op_i]) {
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
              (op == PatternOperation::Write && ex.write_pattern.has_value())
                  ? decimal_string(*ex.write_pattern)
                  : OrderedJson(nullptr);
          if (op == PatternOperation::Write && ex.attempt_seed.has_value()) {
            ex_json["attempt_seed_uint64_decimal"] =
                decimal_string(*ex.attempt_seed);
          }
          ex_json["pre_op_sentinel_uint64_decimal"] =
              ex.pre_op_sentinel.has_value()
                  ? decimal_string(*ex.pre_op_sentinel)
                  : OrderedJson(nullptr);
          ex_json["checksum"] = compact_checksum_json(op, ex);
          ex_json["validation"] = OrderedJson{
              {"checks", validation_checks_json(ex.validation_checks)}};

          OrderedJson att_json;
          att_json["attempt_index"] = att.attempt_index;
          att_json["pattern"] = patterns_kind_to_string(att.pattern);
          att_json["operation"] = patterns_operation_to_string(att.operation);
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
        per_op[patterns_operation_to_string(op)] = std::move(arr);
      }
      attempts_by_pattern[patterns_kind_to_string(kind)] = std::move(per_op);
    }
  }

  return OrderedJson{{"excluded_from_results", true},
                     {"attempts", std::move(attempts_by_pattern)}};
}

OrderedJson measurement_checksum_json(const PatternsMeasurementRow& row) {
  OrderedJson ck;
  const auto& ex = row.execution;
  if (row.operation != PatternOperation::Read) {
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

OrderedJson measurements_json(const PatternsResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& row : result.measurements) {
    const auto& ex = row.execution;
    const bool measured = (row.status == PatternsMeasurementStatus::Measured);

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
        (row.attempted && row.operation == PatternOperation::Write &&
         ex.write_pattern.has_value())
            ? decimal_string(*ex.write_pattern)
            : OrderedJson(nullptr);
    ex_json["pre_op_sentinel_uint64_decimal"] =
        (row.attempted && ex.pre_op_sentinel.has_value())
            ? decimal_string(*ex.pre_op_sentinel)
            : OrderedJson(nullptr);
    ex_json["timing"] = std::move(timing_json);
    ex_json["validation"] =
        OrderedJson{{"checks", validation_checks_json(ex.validation_checks)}};
    if (row.attempted && row.operation == PatternOperation::Write &&
        ex.attempt_seed.has_value()) {
      ex_json["attempt_seed_uint64_decimal"] =
          decimal_string(*ex.attempt_seed);
    }

    OrderedJson row_json;
    row_json["measurement_id"] = row.measurement_id;
    row_json["plan_ref"] = row.plan_ref.has_value()
                               ? OrderedJson(*row.plan_ref)
                               : OrderedJson(nullptr);
    row_json["pattern"] = patterns_kind_to_string(row.pattern);
    row_json["operation"] = patterns_operation_to_string(row.operation);
    row_json["loop_index"] = row.loop_index;
    row_json["pattern_order_position"] = row.pattern_order_position;
    row_json["operation_order_position"] = row.operation_order_position;
    row_json["status"] = patterns_measurement_status_to_string(row.status);
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

OrderedJson aggregates_json(const PatternsResult& result) {
  OrderedJson patterns_json = OrderedJson::object();
  for (size_t p_idx = 0; p_idx < kPatternsMaxKinds; ++p_idx) {
    const auto kind = static_cast<PatternKind>(p_idx);
    OrderedJson per_op = OrderedJson::object();
    for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
      const auto& agg = result.aggregates[p_idx][op_i];
      const bool available = !agg.accepted_measurement_ids.empty();
      OrderedJson a;
      a["status"] = patterns_aggregate_status_to_string(agg.status);
      a["reason_code"] = agg.reason_code;
      a["accepted_measurement_ids"] = agg.accepted_measurement_ids;
      a["value"] = optional_finite_or_null(agg.value);
      a["relative_to_forward_pct"] =
          optional_finite_or_null(agg.relative_to_forward_pct);
      a["statistics"] = statistics_json(agg.statistics, available);
      a["quality"] = OrderedJson{
          {"repeatability_assessment_available",
           agg.repeatability_assessment_available},
          {"run_complete", agg.run_complete},
          {"cv_warning_threshold_pct", agg.cv_warning_threshold_pct},
          {"cv_warning", agg.cv_warning}};
      per_op[patterns_operation_to_string(kOperations[op_i])] = std::move(a);
    }
    patterns_json[patterns_kind_to_string(kind)] = std::move(per_op);
  }
  return OrderedJson{{"patterns", std::move(patterns_json)}};
}

OrderedJson interpretation_json(const PatternsConfig& config) {
  OrderedJson comparability = OrderedJson::array();
  comparability.push_back("same-methodology-version");
  comparability.push_back("same-frozen-work-plan-and-patterns");
  comparability.push_back("same-software-version");
  comparability.push_back("sufficiently-similar-hardware");
  comparability.push_back("sufficiently-similar-thermal-and-power-state");

  OrderedJson out;
  out["dram_residency"] = "unverified";
  out["cache_residency"] = "unverified";
  out["non_cacheable_requested"] = config.non_cacheable;
  out["non_cacheable_is_best_effort_madvise"] = true;
  out["large_page_backing_verified"] = false;
  out["single_pass_xor_collision_caveat"] =
      "Per-pass read XOR detects single-bit and slice-address errors, but even-parity identical flips within one pass can cancel in 64-bit XOR.";
  out["write_copy_post_validation_verifies_final_state_not_intermediate_passes"] =
      true;
  out["comparability_requires"] = std::move(comparability);
  return out;
}

OrderedJson seeds_json(const PatternsConfig& config,
                       const PatternsWorkPlan& work_plan) {
  OrderedJson domain_seeds = OrderedJson::object();
  domain_seeds["src_uint64_decimal"] = decimal_string(work_plan.src_seed);
  domain_seeds["random_uint64_decimal"] =
      decimal_string(work_plan.random_domain_seed);
  domain_seeds["pre_op_sentinel_uint64_decimal"] =
      decimal_string(work_plan.pre_op_sentinel_seed);

  return OrderedJson{
      {"base_seed_uint64_decimal", decimal_string(work_plan.base_seed)},
      {"source", config.user_specified_seed ? "user" : "generated"},
      {"domain_seeds", std::move(domain_seeds)}};
}

OrderedJson counters_json(const PatternsCounters& c) {
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

OrderedJson loop_records_json(const PatternsResult& result) {
  OrderedJson arr = OrderedJson::array();
  for (const auto& loop : result.loops) {
    OrderedJson planned = OrderedJson::array();
    for (const auto& step : loop.planned_order) {
      planned.push_back(OrderedJson{
          {"pattern", patterns_kind_to_string(step.pattern)},
          {"operation", patterns_operation_to_string(step.operation)}});
    }
    OrderedJson realized = OrderedJson::array();
    for (const auto& step : loop.realized_order) {
      realized.push_back(OrderedJson{
          {"pattern", patterns_kind_to_string(step.pattern)},
          {"operation", patterns_operation_to_string(step.operation)}});
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

OrderedJson environment_json(const PatternsResultMetadata& metadata) {
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

}  // namespace

PatternsJsonPeakEstimate calculate_patterns_json_peak_estimate(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan) noexcept {
  if (config.output_file.empty() && !config.collect_json_for_sweep) {
    PatternsJsonPeakEstimate empty_est;
    empty_est.valid = true;
    empty_est.reason_code = PatternsWorkPlanReason::VALID;
    return empty_est;
  }
  return work_plan.json_peak_estimate;
}

std::vector<std::string> collect_patterns_quality_warning_tokens(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result) {
  return ::collect_patterns_quality_warning_tokens(
      config.to_work_plan_config(), work_plan, resources, metadata, result);
}

nlohmann::ordered_json build_patterns_json(
    const PatternsConfig& config,
    const PatternsWorkPlan& work_plan,
    const PatternsResourceSet& resources,
    const PatternsResultMetadata& metadata,
    const PatternsResult& result) {
  OrderedJson out;
  out["schema_version"] = kPatternsSchemaVersion;
  out["mode"] = kPatternsModeName;
  out["methodology_version"] = kPatternsMethodologyVersion;
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
  out["aggregates"] = aggregates_json(result);
  out["status"] = patterns_run_status_to_string(result.status);
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
  out["pattern_order_balance_complete"] = result.pattern_order_balance_complete;
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
  for (const auto& w : collect_patterns_quality_warning_tokens(
           config, work_plan, resources, metadata, result)) {
    warnings_json.push_back(w);
  }
  out["quality_warnings"] = std::move(warnings_json);
  return out;
}
