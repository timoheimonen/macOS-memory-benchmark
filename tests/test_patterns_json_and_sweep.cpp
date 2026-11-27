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

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "core/timing/timer.h"
#include "output/json/json_output/json_output_session.h"
#include "pattern_benchmark/patterns.h"
#include "pattern_benchmark/patterns_executor.h"
#include "pattern_benchmark/patterns_json.h"
#include "pattern_benchmark/patterns_runner.h"
#include "pattern_benchmark/patterns_sweep_runner.h"
#include "pattern_benchmark/patterns_work_plan.h"

namespace {

int run_python_verifier(const std::string& json_path,
                        const std::string& extra_flags,
                        const std::string& verdict_out_path) {
  const std::string cmd =
      "python3 script-examples/verify_patterns_result.py \"" + json_path +
      "\" " + extra_flags + " > \"" + verdict_out_path + "\"";
  const int raw = std::system(cmd.c_str());
  if (raw == -1 || !WIFEXITED(raw)) {
    return -1;
  }
  return WEXITSTATUS(raw);
}

struct TempDirGuard {
  std::filesystem::path dir;
  explicit TempDirGuard(const std::string& prefix) {
    dir = std::filesystem::temp_directory_path() /
          (prefix + "_" + std::to_string(::getpid()) + "_gtest");
    std::filesystem::create_directories(dir);
  }
  ~TempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

struct ArgvBuilder {
  std::vector<std::string> storage;
  std::vector<char*> ptrs;

  explicit ArgvBuilder(std::initializer_list<const char*> args) {
    for (const char* a : args) {
      storage.emplace_back(a);
    }
    ptrs.reserve(storage.size());
    for (auto& s : storage) {
      ptrs.push_back(s.data());
    }
  }

  int argc() const { return static_cast<int>(ptrs.size()); }
  char** argv() { return ptrs.data(); }
};

constexpr std::array<const char*, 26> kExpectedTopLevelKeys = {
    "schema_version",
    "mode",
    "methodology_version",
    "software",
    "build_manifest",
    "configuration",
    "resolved_plan",
    "memory_budget",
    "calibration",
    "measurements",
    "aggregates",
    "status",
    "reason_code",
    "results_complete",
    "run_accepted",
    "interpretation",
    "diagnostic",
    "interruption_requested",
    "operation_order_balance_complete",
    "pattern_order_balance_complete",
    "seeds",
    "counters",
    "checkpoint_lifecycle",
    "loop_records",
    "environment",
    "quality_warnings",
};

constexpr std::array<const char*, 7> kCanonicalPatterns = {
    "sequential_forward",
    "sequential_reverse",
    "strided_64",
    "strided_4096",
    "strided_16384",
    "strided_2mb",
    "random",
};

TEST(PatternsJsonAndSweepTest, ClassifyPatternsCompletionStates) {
  nlohmann::ordered_json complete = {
      {"schema_version", 1},
      {"mode", "patterns"},
      {"status", "complete"},
      {"reason_code", "complete"},
      {"results_complete", true},
      {"run_accepted", true},
      {"measurements", {{{"status", "measured"}}}},
  };
  const auto c_res =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, complete);
  EXPECT_EQ(c_res.status, SweepAttemptStatus::Complete);

  nlohmann::ordered_json interrupted = complete;
  interrupted["status"] = "interrupted";
  interrupted["reason_code"] = "interruption-requested";
  interrupted["results_complete"] = false;
  interrupted["run_accepted"] = false;
  const auto i_res =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, interrupted);
  EXPECT_EQ(i_res.status, SweepAttemptStatus::Interrupted);
  EXPECT_EQ(i_res.reason, "interruption-requested");

  nlohmann::ordered_json failed = complete;
  failed["status"] = "failed";
  failed["reason_code"] = "memory-budget-exceeded";
  failed["results_complete"] = false;
  failed["run_accepted"] = false;
  const auto f_res =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, failed);
  EXPECT_EQ(f_res.status, SweepAttemptStatus::Failed);
  EXPECT_EQ(f_res.reason, "memory-budget-exceeded");

  nlohmann::ordered_json bad_mode = complete;
  bad_mode["mode"] = "cpu_bandwidth";
  EXPECT_EQ(
      classify_sweep_nested_completion(SweepNestedMode::Patterns, bad_mode)
          .reason,
      "invalid-patterns-schema-contract");

  nlohmann::ordered_json bad_schema = complete;
  bad_schema["schema_version"] = 3;
  EXPECT_EQ(
      classify_sweep_nested_completion(SweepNestedMode::Patterns, bad_schema)
          .reason,
      "unsupported-patterns-schema-version");

  nlohmann::ordered_json empty_meas = complete;
  empty_meas["measurements"] = nlohmann::ordered_json::array();
  EXPECT_EQ(
      classify_sweep_nested_completion(SweepNestedMode::Patterns, empty_meas)
          .reason,
      "invalid-patterns-schema-contract");
}

TEST(PatternsJsonAndSweepTest, SweepRunConfigsShareSeedAndEmptyOutputFile) {
  PatternsConfig base;
  base.buffer_size_mb = 512;
  base.requested_workers = 2;
  base.available_workers = 4;
  base.user_specified_workers = true;
  base.seed = 424242ULL;
  base.user_specified_seed = false;
  base.swept_buffer_size = true;
  base.swept_threads = true;
  base.run_sweep = true;
  base.output_file = "sweep.json";
  base.sweep_specs = {
      {PatternsSweepParameter::BufferSizeMb,
       "buffer-size",
       {{"1", 1}, {"3", 3}}},
      {PatternsSweepParameter::Threads, "threads", {{"1", 1}, {"2", 2}}},
  };

  EXPECT_EQ(calculate_patterns_sweep_run_count(base), 4u);
  const auto runs = build_patterns_sweep_run_configs(base);
  ASSERT_EQ(runs.size(), 4u);
  for (const auto& r : runs) {
    EXPECT_FALSE(r.run_sweep);
    EXPECT_TRUE(r.output_file.empty());
    EXPECT_TRUE(r.collect_json_for_sweep);
    EXPECT_TRUE(r.swept_buffer_size);
    EXPECT_TRUE(r.swept_threads);
    EXPECT_EQ(r.seed, 424242ULL);
    EXPECT_FALSE(r.user_specified_seed);
  }
  EXPECT_EQ(runs[0].buffer_size_mb, 1u);
  EXPECT_EQ(runs[0].requested_workers, 1u);
  EXPECT_EQ(runs[3].buffer_size_mb, 3u);
  EXPECT_EQ(runs[3].requested_workers, 2u);
}

TEST(PatternsJsonAndSweepTest,
     PeakEstimateWarningsAndBuildJsonAcrossAllStates) {
  PatternsConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.requested_workers = 2;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.loop_count = 2;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  cfg.argv = {"memory_benchmark", "-P", "-b", "1", "-t", "2", "-r", "2", "-u",
              "--seed", "424242"};

  PatternsSystemInfo sys_info;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  const auto plan =
      build_patterns_work_plan(cfg.to_work_plan_config(), sys_info);
  ASSERT_TRUE(plan.valid);

  // 1. Peak estimate helper: empty output_file without sweep vs active output.
  {
    const auto empty_est = calculate_patterns_json_peak_estimate(cfg, plan);
    EXPECT_TRUE(empty_est.valid);
    EXPECT_EQ(empty_est.total_bytes, 0u);

    auto cfg_out = cfg;
    cfg_out.output_file = "out.json";
    const auto plan_out =
        build_patterns_work_plan(cfg_out.to_work_plan_config(), sys_info);
    ASSERT_TRUE(plan_out.valid);
    const auto active_est =
        calculate_patterns_json_peak_estimate(cfg_out, plan_out);
    EXPECT_TRUE(active_est.valid);
    EXPECT_GT(active_est.total_bytes, 0u);
  }

  // 2. Quality warning tokens across all conditions.
  {
    PatternsResourceSet res;
    res.src.mapped_bytes = plan.buffer_bytes;
    res.dst.mapped_bytes = plan.buffer_bytes;
    res.src.madvise_applied = false;
    res.dst.madvise_applied = true;

    PatternsResultMetadata meta;
    meta.main_thread_qos = {true, false, 1};
    meta.environment_start.thermal_state = "fair";
    meta.environment_end.thermal_state = "nominal";
    meta.environment_end.low_power_mode_available = true;
    meta.environment_end.low_power_mode_enabled = true;

    PatternsResult res_state;
    res_state.quality_warnings = {"custom-extra", "custom-extra"};
    PatternsMeasurementRow row;
    row.pattern = PatternKind::SequentialForward;
    row.operation = PatternOperation::Read;
    row.status = PatternsMeasurementStatus::Measured;
    row.attempted = true;
    row.duration_quality = "below-target-window";
    row.execution.qos_failed_workers = 1;
    res_state.measurements.push_back(row);
    res_state.aggregates[patterns_kind_index(PatternKind::SequentialForward)][0]
        .cv_warning = true;

    const auto tokens =
        collect_patterns_quality_warning_tokens(cfg, plan, res, meta, res_state);
    auto has = [&](const std::string& tok) {
      return std::find(tokens.begin(), tokens.end(), tok) != tokens.end();
    };
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), "custom-extra"), 1);
    EXPECT_TRUE(has("environment-not-nominal"));
    EXPECT_TRUE(has("main-thread-qos-not-applied"));
    EXPECT_TRUE(has("worker-qos-not-applied"));
    EXPECT_TRUE(has("madvise-random-not-applied"));
    EXPECT_TRUE(has("sequential_forward-read-duration-below-target-window"));
    EXPECT_TRUE(has("sequential_forward-read-high-cv"));
  }

  // 3. Complete -b 1 run: strided_2mb skipped in plan/aggregates/frozen_refs, absent in calibration.attempts.
  {
    auto cfg_1mb = cfg;
    cfg_1mb.iterations = 1;
    cfg_1mb.user_specified_iterations = true;
    auto plan_1mb =
        build_patterns_work_plan(cfg_1mb.to_work_plan_config(), sys_info);
    ASSERT_TRUE(plan_1mb.valid);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan_1mb, true, res));
    PatternsResult complete_res;
    PatternsRunnerHooks hooks;
    ASSERT_EQ(run_patterns_suite(cfg_1mb.to_work_plan_config(), plan_1mb, res,
                                 complete_res, hooks),
              EXIT_SUCCESS);

    PatternsResultMetadata meta;
    meta.processor_name = "Test Apple Silicon";
    meta.macos_version = "26.0";
    meta.logical_core_count = 4;
    meta.page_size_bytes = 16384;
    meta.user_specified_seed = cfg_1mb.user_specified_seed;
    meta.available_memory_bytes = 0;
    meta.available_memory_source = "unavailable";
    meta.environment_start.thermal_state = "nominal";
    meta.environment_start.low_power_mode_available = false;
    meta.environment_start.physical_memory_bytes = 0;
    meta.environment_end = meta.environment_start;

    const auto doc =
        build_patterns_json(cfg_1mb, plan_1mb, res, meta, complete_res);

    // Exact 26 top-level keys in canonical order.
    std::vector<std::string> top_keys;
    for (const auto& item : doc.items()) {
      top_keys.push_back(item.key());
    }
    ASSERT_EQ(top_keys.size(), kExpectedTopLevelKeys.size());
    for (size_t i = 0; i < kExpectedTopLevelKeys.size(); ++i) {
      EXPECT_EQ(top_keys[i], kExpectedTopLevelKeys[i]);
    }

    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["mode"], "patterns");
    EXPECT_EQ(doc["methodology_version"], kPatternsMethodologyVersion);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());

    // All 7 canonical patterns in resolved_plan.patterns, frozen_plan_refs, aggregates.patterns.
    std::vector<std::string> plan_keys;
    for (const auto& item : doc["resolved_plan"]["patterns"].items()) {
      plan_keys.push_back(item.key());
    }
    ASSERT_EQ(plan_keys.size(), kCanonicalPatterns.size());
    for (size_t i = 0; i < kCanonicalPatterns.size(); ++i) {
      EXPECT_EQ(plan_keys[i], kCanonicalPatterns[i]);
    }
    EXPECT_EQ(doc["resolved_plan"]["patterns"]["strided_2mb"]["status"],
              "skipped");
    EXPECT_EQ(doc["resolved_plan"]["patterns"]["strided_2mb"]["reason_code"],
              "buffer-too-small-for-stride-transition");
    EXPECT_TRUE(
        doc["resolved_plan"]["frozen_plan_refs"]["strided_2mb"]["read"].is_null());
    EXPECT_EQ(doc["aggregates"]["patterns"]["strided_2mb"]["read"]["status"],
              "skipped");
    EXPECT_FALSE(doc["calibration"]["attempts"].contains("strided_2mb"));

    EXPECT_TRUE(doc["environment"]["available_memory_bytes"].is_null());
    EXPECT_TRUE(
        doc["environment"]["start"]["low_power_mode_enabled"].is_null());
    EXPECT_TRUE(doc["environment"]["start"]["physical_memory_bytes"].is_null());
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["src"]["read_only_protected"]
                    .get<bool>());
    release_patterns_resources(res);
  }

  // 4. Complete -b 3 run: all 7 active, random SHA-256, 6 validation checks, CV thresholds, relative_to_forward_pct.
  {
    auto cfg_3mb = cfg;
    cfg_3mb.buffer_size_mb = 3;
    cfg_3mb.iterations = 1;
    cfg_3mb.user_specified_iterations = true;
    auto plan_3mb =
        build_patterns_work_plan(cfg_3mb.to_work_plan_config(), sys_info);
    ASSERT_TRUE(plan_3mb.valid);

    PatternsResourceSet res;
    ASSERT_TRUE(prepare_patterns_resources(plan_3mb, false, res));
    PatternsResult complete_res;
    PatternsRunnerHooks hooks;
    ASSERT_EQ(run_patterns_suite(cfg_3mb.to_work_plan_config(), plan_3mb, res,
                                 complete_res, hooks),
              EXIT_SUCCESS);

    PatternsResultMetadata meta;
    meta.environment_start.thermal_state = "nominal";
    meta.environment_end = meta.environment_start;
    const auto doc =
        build_patterns_json(cfg_3mb, plan_3mb, res, meta, complete_res);
    EXPECT_EQ(doc["measurements"].size(), 2u * 7u * 3u);
    EXPECT_FALSE(
        doc["resolved_plan"]["patterns"]["random"]["random_workload"]
           ["random_indices_sha256"]
               .is_null());
    EXPECT_EQ(
        doc["measurements"][0]["execution"]["validation"]["checks"].size(), 6u);
    EXPECT_DOUBLE_EQ(
        doc["aggregates"]["patterns"]["sequential_forward"]["read"]
           ["relative_to_forward_pct"]
               .get<double>(),
        100.0);
    EXPECT_DOUBLE_EQ(
        doc["aggregates"]["patterns"]["sequential_forward"]["read"]["quality"]
           ["cv_warning_threshold_pct"]
               .get<double>(),
        5.0);
    EXPECT_DOUBLE_EQ(
        doc["aggregates"]["patterns"]["strided_4096"]["read"]["quality"]
           ["cv_warning_threshold_pct"]
               .get<double>(),
        10.0);
    release_patterns_resources(res);
  }

  // 5. Partial & early work-plan failure JSON states (unavailable normalization when buffer_bytes == 0).
  {
    PatternsResourceSet empty_res;
    PatternsResultMetadata meta;
    meta.environment_start.thermal_state = "nominal";
    meta.environment_end = meta.environment_start;

    PatternsResult partial_res;
    initialize_patterns_setup_failure_result(
        cfg.to_work_plan_config(), plan, PatternsRunStatus::Partial,
        "partial-results", false, partial_res);
    partial_res.diagnostic = "partial-diagnostic";
    partial_res.measurements[0].attempted = true;
    partial_res.measurements[0].status = PatternsMeasurementStatus::Invalid;
    partial_res.measurements[0].reason_code = "invalid-elapsed-time";
    partial_res.measurements[0].execution.timing_evaluated = true;
    partial_res.measurements[0].execution.timing_valid = false;
    partial_res.measurements[0].execution.elapsed_seconds = 0.0;
    const auto partial_doc =
        build_patterns_json(cfg, plan, empty_res, meta, partial_res);
    EXPECT_EQ(partial_doc["status"], "partial");
    EXPECT_EQ(partial_doc["diagnostic"], "partial-diagnostic");
    EXPECT_EQ(
        partial_doc["measurements"][0]["execution"]["timing"]
                   ["diagnostic_elapsed_seconds"],
        0.0);
    EXPECT_TRUE(partial_doc["resolved_plan"]["resources"].empty());

    PatternsWorkPlan invalid_plan;
    invalid_plan.valid = false;
    invalid_plan.reason_code = "memory-budget-exceeded";
    PatternsResult fail_res;
    initialize_patterns_setup_failure_result(
        cfg.to_work_plan_config(), invalid_plan, "memory-budget-exceeded",
        fail_res);
    const auto fail_doc =
        build_patterns_json(cfg, invalid_plan, empty_res, meta, fail_res);
    EXPECT_EQ(fail_doc["status"], "failed");
    EXPECT_FALSE(fail_doc["resolved_plan"]["valid"].get<bool>());
    EXPECT_EQ(fail_doc["resolved_plan"]["plan_identity"],
              "patterns-work-plan-v1|invalid");
    for (const char* pname : kCanonicalPatterns) {
      EXPECT_FALSE(fail_doc["resolved_plan"]["patterns"][pname]["enabled"]
                       .get<bool>());
      EXPECT_EQ(fail_doc["resolved_plan"]["patterns"][pname]["status"],
                "unavailable");
      EXPECT_EQ(fail_doc["resolved_plan"]["patterns"][pname]["reason_code"],
                "memory-budget-exceeded");
      for (const char* op_name : {"read", "write", "copy"}) {
        EXPECT_EQ(fail_doc["aggregates"]["patterns"][pname][op_name]["status"],
                  "unavailable");
        EXPECT_EQ(
            fail_doc["aggregates"]["patterns"][pname][op_name]["reason_code"],
            "memory-budget-exceeded");
      }
    }

    // Post-geometry work-plan rejections (`buffer_bytes > 0`) must also
    // normalize ALL 7 patterns (including `strided_2mb` on `-b 1`) to
    // `unavailable` with `work_plan.reason_code`.
    const auto check_post_geometry_rejection =
        [&](const PatternsConfig& bad_cfg, const std::string& expected_reason) {
          const PatternsWorkPlan bad_plan =
              build_patterns_work_plan(bad_cfg.to_work_plan_config(), sys_info);
          ASSERT_FALSE(bad_plan.valid);
          ASSERT_EQ(bad_plan.reason_code, expected_reason);
          ASSERT_GT(bad_plan.patterns[0].buffer_bytes, 0u);
          ASSERT_FALSE(bad_plan.active_patterns.empty());

          PatternsResult post_fail_res;
          initialize_patterns_setup_failure_result(
              bad_cfg.to_work_plan_config(), bad_plan, expected_reason,
              post_fail_res);
          const auto post_doc = build_patterns_json(bad_cfg, bad_plan, empty_res,
                                                    meta, post_fail_res);
          EXPECT_EQ(post_doc["status"], "failed");
          EXPECT_EQ(post_doc["reason_code"], expected_reason);
          EXPECT_FALSE(post_doc["run_accepted"].get<bool>());
          EXPECT_FALSE(post_doc["resolved_plan"]["valid"].get<bool>());
          EXPECT_TRUE(post_doc["calibration"]["attempts"].empty());
          EXPECT_TRUE(post_doc["measurements"].empty());
          EXPECT_EQ(post_doc["counters"]["planned_measurements"], 0u);
          for (const char* pname : kCanonicalPatterns) {
            EXPECT_FALSE(post_doc["resolved_plan"]["patterns"][pname]["enabled"]
                             .get<bool>())
                << expected_reason << " " << pname;
            EXPECT_EQ(post_doc["resolved_plan"]["patterns"][pname]["status"],
                      "unavailable")
                << expected_reason << " " << pname;
            EXPECT_EQ(
                post_doc["resolved_plan"]["patterns"][pname]["reason_code"],
                expected_reason)
                << expected_reason << " " << pname;
            for (const char* op_name : {"read", "write", "copy"}) {
              EXPECT_EQ(
                  post_doc["aggregates"]["patterns"][pname][op_name]["status"],
                  "unavailable")
                  << expected_reason << " " << pname << " " << op_name;
              EXPECT_EQ(post_doc["aggregates"]["patterns"][pname][op_name]
                                ["reason_code"],
                        expected_reason)
                  << expected_reason << " " << pname << " " << op_name;
            }
          }
        };

    auto oom_cfg = cfg;
    oom_cfg.buffer_size_mb = 999999999ULL;
    oom_cfg.iterations = 1;
    oom_cfg.loop_count = 1;
    check_post_geometry_rejection(oom_cfg, "memory-budget-exceeded");

    auto work_cap_cfg = cfg;
    work_cap_cfg.buffer_size_mb = 1;
    work_cap_cfg.iterations = 1000000000ULL;
    work_cap_cfg.user_specified_iterations = true;
    work_cap_cfg.loop_count = 1;
    check_post_geometry_rejection(work_cap_cfg, "work-cap-exceeded");

    auto meas_limit_cfg = cfg;
    meas_limit_cfg.buffer_size_mb = 1;
    meas_limit_cfg.iterations = 1;
    meas_limit_cfg.user_specified_iterations = true;
    meas_limit_cfg.loop_count = 100000ULL;
    check_post_geometry_rejection(meas_limit_cfg, "measurement-limit-exceeded");
  }
}

TEST(PatternsJsonAndSweepTest, SweepPreflightAndRunnerLifecycleUnitPaths) {
  TempDirGuard tmp("pat_sweep_unit");

  PatternsSystemInfo sys_info;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  // 1. Preflight rejections & valid sweep.
  {
    PatternsConfig cfg;
    EXPECT_FALSE(preflight_patterns_sweep(cfg, sys_info));  // !run_sweep
    cfg.run_sweep = true;
    EXPECT_FALSE(preflight_patterns_sweep(cfg, sys_info));  // empty specs

    cfg.sweep_specs = {
        {PatternsSweepParameter::BufferSizeMb,
         "buffer-size",
         {{"1", 1}, {"3", 3}}}};
    cfg.sweep_max_runs = 1;
    EXPECT_FALSE(preflight_patterns_sweep(cfg, sys_info));  // > sweep_max_runs

    cfg.sweep_max_runs = 16;
    EXPECT_TRUE(preflight_patterns_sweep(cfg, sys_info));

    std::ostringstream err;
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    cfg.sweep_specs = {{PatternsSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"999999999", 999999999ULL}}}};
    EXPECT_FALSE(preflight_patterns_sweep(cfg, sys_info));  // budget exceeded
    std::cerr.rdbuf(old_err);
  }

  // 2. Multi-point run_patterns_sweep with File session & nested failure.
  {
    const auto sweep_out = (tmp.dir / "unit_sweep.json").string();
    PatternsConfig cfg;
    cfg.buffer_size_mb = 1;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    cfg.seed = 42ULL;
    cfg.user_specified_seed = true;
    cfg.run_sweep = true;
    cfg.output_file = sweep_out;
    cfg.sweep_specs = {{PatternsSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"1", 1}, {"3", 3}}}};

    JsonOutputSession session(make_json_output_target(sweep_out));
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const SweepExecutionResult exec = run_patterns_sweep(cfg, session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(exec.output_json["status"], "complete");
    EXPECT_EQ(exec.output_json["completed_runs"], 2u);
    EXPECT_TRUE(exec.output_json["conclusions_valid"].get<bool>());
    EXPECT_EQ(exec.output_json["configuration"]["base_mode"], "patterns");

    // Timer failure path in run_patterns_sweep.
    TimerSystemCalls bad_timer;
    bad_timer.timebase_info = [](mach_timebase_info_t) -> kern_return_t {
      return KERN_FAILURE;
    };
    set_timer_system_calls_for_testing(bad_timer);
    JsonOutputSession disabled_session(make_json_output_target(""));
    std::ostringstream out2, err2;
    old_out = std::cout.rdbuf(out2.rdbuf());
    old_err = std::cerr.rdbuf(err2.rdbuf());
    const SweepExecutionResult timer_fail =
        run_patterns_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    reset_timer_system_calls_for_testing();
    EXPECT_NE(timer_fail.exit_code, EXIT_SUCCESS);

    // Nested run failure in sweep (bypassing preflight with huge buffer size).
    cfg.sweep_specs = {{PatternsSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"999999999", 999999999ULL}}}};
    cfg.output_file.clear();
    std::ostringstream out3, err3;
    old_out = std::cout.rdbuf(out3.rdbuf());
    old_err = std::cerr.rdbuf(err3.rdbuf());
    const SweepExecutionResult fail_exec =
        run_patterns_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(fail_exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(fail_exec.output_json["status"], "failed");
  }
}

TEST(PatternsModeIntegrationTest,
     EndToEndDirectStdoutAndSweepProduceValidSchemaV1) {
  TempDirGuard tmp("pat_e2e");

  // 1. Direct file output with -b 3 (all 7 active), explicit iterations & seed.
  const auto direct_path = (tmp.dir / "direct_3mb.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "3", "-t", "2", "-i", "2",
                      "-r", "3", "--seed", "424242", "-o",
                      direct_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_patterns_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
  }
  {
    std::ifstream ifs(direct_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["mode"], "patterns");
    EXPECT_EQ(doc["methodology_version"], kPatternsMethodologyVersion);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["results_complete"].get<bool>());
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_TRUE(doc["operation_order_balance_complete"].get<bool>());
    EXPECT_EQ(doc["measurements"].size(), 63u);
    EXPECT_EQ(doc["resolved_plan"]["patterns"]["strided_2mb"]["status"],
              "active");
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["src"]["read_only_protected"]
                    .get<bool>());
  }

  // 2. Stdout JSON transport (`-o -`) with -b 1 (strided_2mb skipped).
  const auto stdout_captured = (tmp.dir / "stdout_1mb.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "1", "-t", "2", "-i", "2",
                      "-r", "1", "--seed", "424242", "-o", "-"});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_patterns_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    const auto doc = nlohmann::ordered_json::parse(out.str());
    EXPECT_EQ(doc["configuration"]["output_file"], "-");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_EQ(doc["resolved_plan"]["patterns"]["strided_2mb"]["status"],
              "skipped");
    EXPECT_EQ(doc["resolved_plan"]["patterns"]["strided_2mb"]["reason_code"],
              "buffer-too-small-for-stride-transition");
    EXPECT_FALSE(err.str().empty());
    std::ofstream ofs(stdout_captured);
    ofs << out.str();
  }

  // 3. Budget overflow writes failed JSON with memory_budget.valid == false.
  const auto budget_path = (tmp.dir / "budget_fail.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "999999999", "-i", "1",
                      "-r", "1", "-o", budget_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_patterns_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(rc, EXIT_SUCCESS);
    std::ifstream ifs(budget_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_EQ(doc["reason_code"], "memory-budget-exceeded");
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
    EXPECT_FALSE(doc["resolved_plan"]["valid"].get<bool>());
    EXPECT_FALSE(doc["memory_budget"]["valid"].get<bool>());
    EXPECT_TRUE(doc["calibration"]["attempts"].empty());
    EXPECT_TRUE(doc["measurements"].empty());
    EXPECT_EQ(doc["counters"]["planned_measurements"], 0u);
    EXPECT_EQ(doc["counters"]["planned_passes"], "0");
    for (const char* pname : kCanonicalPatterns) {
      EXPECT_FALSE(doc["resolved_plan"]["patterns"][pname]["enabled"].get<bool>())
          << pname;
      EXPECT_EQ(doc["resolved_plan"]["patterns"][pname]["status"],
                "unavailable")
          << pname;
      EXPECT_EQ(doc["resolved_plan"]["patterns"][pname]["reason_code"],
                "memory-budget-exceeded")
          << pname;
      for (const char* op_name : {"read", "write", "copy"}) {
        EXPECT_EQ(doc["aggregates"]["patterns"][pname][op_name]["status"],
                  "unavailable")
            << pname << " " << op_name;
        EXPECT_EQ(doc["aggregates"]["patterns"][pname][op_name]["reason_code"],
                  "memory-budget-exceeded")
            << pname << " " << op_name;
      }
    }
  }

  // 4. Sweep mode (`-S buffer-size=1,3`).
  const auto sweep_path = (tmp.dir / "sweep.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-P", "-i", "1", "-r", "1", "-S",
                      "buffer-size=1,3", "-o", sweep_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_patterns_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    std::ifstream ifs(sweep_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["configuration"]["mode"], "sweep");
    EXPECT_EQ(doc["configuration"]["base_mode"], "patterns");
    EXPECT_EQ(doc["configuration"]["sweep_schema_version"], 1);
    EXPECT_EQ(doc["planned_runs"], 2u);
    EXPECT_EQ(doc["completed_runs"], 2u);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["conclusions_valid"].get<bool>());
  }

  if (std::filesystem::exists("script-examples/verify_patterns_result.py")) {
    const auto v_out = (tmp.dir / "verdict.json").string();
    EXPECT_EQ(run_python_verifier(direct_path, "--require-raw-timing", v_out),
              0);
    EXPECT_EQ(
        run_python_verifier(stdout_captured, "--require-raw-timing", v_out), 0);
    EXPECT_EQ(run_python_verifier(sweep_path, "--require-raw-timing", v_out),
              0);
    EXPECT_EQ(run_python_verifier(budget_path, "", v_out), 1);
  }
}

}  // namespace
