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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "core/timing/timer.h"
#include "cpu_bandwidth/cpu_bandwidth.h"
#include "cpu_bandwidth/cpu_bandwidth_executor.h"
#include "cpu_bandwidth/cpu_bandwidth_json.h"
#include "cpu_bandwidth/cpu_bandwidth_runner.h"
#include "cpu_bandwidth/cpu_bandwidth_sweep_runner.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"
#include "output/json/json_output/json_output_session.h"

namespace {

int run_python_verifier(const std::string& json_path,
                        const std::string& extra_flags,
                        const std::string& verdict_out_path) {
  const std::string cmd =
      "python3 script-examples/verify_cpu_bandwidth_result.py \"" + json_path +
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
    dir = std::filesystem::temp_directory_path() / (prefix + "_gtest");
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

TEST(CpuBandwidthJsonAndSweepTest, ClassifyCpuBandwidthCompletionStates) {
  nlohmann::ordered_json complete = {
      {"schema_version", 1},
      {"mode", "cpu_bandwidth"},
      {"status", "complete"},
      {"reason_code", "complete"},
      {"results_complete", true},
      {"run_accepted", true},
      {"measurements", {{{"status", "measured"}}}},
  };
  const auto c_res = classify_sweep_nested_completion(
      SweepNestedMode::CpuBandwidth, complete);
  EXPECT_EQ(c_res.status, SweepAttemptStatus::Complete);

  nlohmann::ordered_json interrupted = complete;
  interrupted["status"] = "interrupted";
  interrupted["reason_code"] = "interruption-requested";
  interrupted["results_complete"] = false;
  interrupted["run_accepted"] = false;
  const auto i_res = classify_sweep_nested_completion(
      SweepNestedMode::CpuBandwidth, interrupted);
  EXPECT_EQ(i_res.status, SweepAttemptStatus::Interrupted);

  nlohmann::ordered_json failed = complete;
  failed["status"] = "failed";
  failed["reason_code"] = "memory-budget-exceeded";
  failed["results_complete"] = false;
  failed["run_accepted"] = false;
  const auto f_res = classify_sweep_nested_completion(
      SweepNestedMode::CpuBandwidth, failed);
  EXPECT_EQ(f_res.status, SweepAttemptStatus::Failed);
  EXPECT_EQ(f_res.reason, "memory-budget-exceeded");
}

TEST(CpuBandwidthJsonAndSweepTest, SweepRunConfigsKeepFixedKeysAndEmptyOutputFile) {
  CpuBandwidthConfig base;
  base.buffer_size_mb = 512;
  base.custom_cache_size_kb = 0;
  base.requested_workers = 2;
  base.available_workers = 4;
  base.user_specified_workers = true;
  base.swept_buffer_size = true;
  base.swept_threads = true;
  base.run_sweep = true;
  base.output_file = "sweep.json";
  base.sweep_specs = {
      {CpuBandwidthSweepParameter::BufferSizeMb,
       "buffer-size",
       {{"1", 1}, {"2", 2}}},
      {CpuBandwidthSweepParameter::Threads,
       "threads",
       {{"1", 1}, {"2", 2}}},
  };

  EXPECT_EQ(calculate_cpu_bandwidth_sweep_run_count(base), 4u);
  const auto runs = build_cpu_bandwidth_sweep_run_configs(base);
  ASSERT_EQ(runs.size(), 4u);
  for (const auto& r : runs) {
    EXPECT_FALSE(r.run_sweep);
    EXPECT_TRUE(r.output_file.empty());
    EXPECT_TRUE(r.collect_json_for_sweep);
    EXPECT_TRUE(r.swept_buffer_size);
    EXPECT_TRUE(r.swept_threads);
  }
  EXPECT_EQ(runs[0].buffer_size_mb, 1u);
  EXPECT_EQ(runs[0].requested_workers, 1u);
  EXPECT_EQ(runs[3].buffer_size_mb, 2u);
  EXPECT_EQ(runs[3].requested_workers, 2u);
}

TEST(CpuBandwidthJsonAndSweepTest,
     PeakEstimateWarningsAndBuildJsonAcrossAllStates) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 1;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.loop_count = 2;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  cfg.argv = {"memory_benchmark", "-B", "-b", "0", "-k", "16", "-t", "1", "-r", "2", "-u", "--seed", "424242"};

  CpuBandwidthSystemInfo sys_info;
  sys_info.l1_cache_bytes = 128ULL * 1024ULL;
  sys_info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  const auto plan = build_cpu_bandwidth_work_plan(cfg, sys_info);
  ASSERT_TRUE(plan.valid);

  // 1. Peak estimate helper: empty output_file without sweep vs active output.
  {
    const auto empty_est = calculate_cpu_bandwidth_json_peak_estimate(cfg, plan);
    EXPECT_TRUE(empty_est.valid);
    EXPECT_EQ(empty_est.total_bytes, 0u);
    auto cfg_out = cfg;
    cfg_out.output_file = "out.json";
    const auto plan_out = build_cpu_bandwidth_work_plan(cfg_out, sys_info);
    ASSERT_TRUE(plan_out.valid);
    const auto active_est =
        calculate_cpu_bandwidth_json_peak_estimate(cfg_out, plan_out);
    EXPECT_TRUE(active_est.valid);
    EXPECT_GT(active_est.total_bytes, 0u);
  }

  // 2. Quality warning tokens across all conditions.
  {
    CpuBandwidthResourceSet res;
    const size_t c_idx = cpu_bandwidth_target_index(CpuBandwidthTarget::Custom);
    res.targets[c_idx].active = true;
    res.targets[c_idx].src.madvise_applied = false;
    res.targets[c_idx].dst.madvise_applied = true;

    CpuBandwidthResultMetadata meta;
    meta.main_thread_qos = {true, false, 1};
    meta.environment_start.thermal_state = "fair";
    meta.environment_end.thermal_state = "nominal";
    meta.environment_end.low_power_mode_available = true;
    meta.environment_end.low_power_mode_enabled = true;

    CpuBandwidthResult res_state;
    res_state.quality_warnings = {"custom-extra", "custom-extra"};
    CpuBandwidthMeasurementRow row;
    row.target = CpuBandwidthTarget::Custom;
    row.operation = CpuBandwidthOperation::Read;
    row.status = CpuBandwidthMeasurementStatus::Measured;
    row.attempted = true;
    row.duration_quality = "below-target-window";
    row.execution.qos_failed_workers = 1;
    res_state.measurements.push_back(row);
    res_state.aggregates[c_idx][0].cv_warning = true;

    const auto tokens = collect_cpu_bandwidth_quality_warning_tokens(
        cfg, plan, res, meta, res_state);
    auto has = [&](const std::string& tok) {
      return std::find(tokens.begin(), tokens.end(), tok) != tokens.end();
    };
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), "custom-extra"), 1);
    EXPECT_TRUE(has("environment-not-nominal"));
    EXPECT_TRUE(has("main-thread-qos-not-applied"));
    EXPECT_TRUE(has("worker-qos-not-applied"));
    EXPECT_TRUE(has("madvise-random-not-applied"));
    EXPECT_TRUE(has("custom-read-duration-below-target-window"));
    EXPECT_TRUE(has("custom-read-high-cv"));
  }

  // 3. Complete run JSON (auto-calibrated, Write/Read/Copy, MachTimingSnapshot, calibration attempts).
  {
    CpuBandwidthResourceSet res;
    ASSERT_TRUE(prepare_cpu_bandwidth_resources(plan, true, res));
    CpuBandwidthResult complete_res;
    CpuBandwidthRunnerHooks hooks;
    auto cfg_exp = cfg;
    cfg_exp.iterations = 1;
    cfg_exp.user_specified_iterations = true;
    ASSERT_EQ(run_cpu_bandwidth_suite(cfg_exp, plan, res, complete_res, hooks),
              EXIT_SUCCESS);

    // Add a synthetic calibration placeholder & invalid calibration attempt to exercise serialization branches.
    const size_t c_idx = cpu_bandwidth_target_index(CpuBandwidthTarget::Custom);
    CpuBandwidthCalibrationAttempt placeholder;
    placeholder.attempt_index = complete_res.calibration_attempts[c_idx][0].size();
    placeholder.target = CpuBandwidthTarget::Custom;
    placeholder.operation = CpuBandwidthOperation::Read;
    placeholder.purpose = "placeholder";
    complete_res.calibration_attempts[c_idx][0].push_back(placeholder);

    CpuBandwidthResultMetadata meta;
    meta.processor_name = "Test Apple Silicon";
    meta.macos_version = "26.0";
    meta.logical_core_count = 4;
    meta.page_size_bytes = 16384;
    meta.available_memory_bytes = 0;
    meta.available_memory_source = "unavailable";
    meta.environment_start.thermal_state = "nominal";
    meta.environment_start.low_power_mode_available = false;
    meta.environment_start.physical_memory_bytes = 0;
    meta.environment_end = meta.environment_start;

    const auto doc = build_cpu_bandwidth_json(cfg_exp, plan, res, meta, complete_res);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_TRUE(doc["environment"]["available_memory_bytes"].is_null());
    EXPECT_TRUE(doc["environment"]["start"]["low_power_mode_enabled"].is_null());
    EXPECT_TRUE(doc["environment"]["start"]["physical_memory_bytes"].is_null());
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["custom"]["src"]["read_only_protected"].get<bool>());
    EXPECT_FALSE(doc["calibration"]["attempts"]["custom"]["read"].back()["execution"]["valid"].is_boolean());
    release_cpu_bandwidth_resources(res);
  }

  // 4. Partial, Interrupted, and Failed JSON states (including empty resources & invalid work_plan).
  {
    CpuBandwidthResourceSet empty_res;
    CpuBandwidthResultMetadata meta;
    meta.environment_start.thermal_state = "nominal";
    meta.environment_end = meta.environment_start;

    CpuBandwidthResult partial_res;
    initialize_cpu_bandwidth_setup_failure_result(
        cfg, plan, CpuBandwidthRunStatus::Partial, "partial-results", false,
        partial_res);
    partial_res.diagnostic = "partial-diagnostic";
    partial_res.measurements[0].attempted = true;
    partial_res.measurements[0].status = CpuBandwidthMeasurementStatus::Invalid;
    partial_res.measurements[0].reason_code = "invalid-elapsed-time";
    partial_res.measurements[0].execution.timing_evaluated = true;
    partial_res.measurements[0].execution.timing_valid = false;
    partial_res.measurements[0].execution.elapsed_seconds = 0.0;
    const auto partial_doc =
        build_cpu_bandwidth_json(cfg, plan, empty_res, meta, partial_res);
    EXPECT_EQ(partial_doc["status"], "partial");
    EXPECT_EQ(partial_doc["diagnostic"], "partial-diagnostic");
    EXPECT_EQ(partial_doc["measurements"][0]["execution"]["timing"]["diagnostic_elapsed_seconds"], 0.0);
    EXPECT_TRUE(partial_doc["resolved_plan"]["resources"].empty());

    CpuBandwidthWorkPlan invalid_plan;
    invalid_plan.valid = false;
    invalid_plan.reason_code = "memory-budget-exceeded";
    CpuBandwidthResult fail_res;
    initialize_cpu_bandwidth_setup_failure_result(
        cfg, invalid_plan, "memory-budget-exceeded", fail_res);
    const auto fail_doc =
        build_cpu_bandwidth_json(cfg, invalid_plan, empty_res, meta, fail_res);
    EXPECT_EQ(fail_doc["status"], "failed");
    EXPECT_FALSE(fail_doc["resolved_plan"]["valid"].get<bool>());
    EXPECT_EQ(fail_doc["resolved_plan"]["plan_identity"],
              "cpu-bandwidth-work-plan-v1|invalid");
  }
}

TEST(CpuBandwidthJsonAndSweepTest, SweepPreflightAndRunnerLifecycleUnitPaths) {
  TempDirGuard tmp("cpu_bw_sweep_unit");

  CpuBandwidthSystemInfo sys_info;
  sys_info.l1_cache_bytes = 128ULL * 1024ULL;
  sys_info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  // 1. Preflight rejections & valid sweep.
  {
    CpuBandwidthConfig cfg;
    EXPECT_FALSE(preflight_cpu_bandwidth_sweep(cfg, sys_info));  // !run_sweep
    cfg.run_sweep = true;
    EXPECT_FALSE(preflight_cpu_bandwidth_sweep(cfg, sys_info));  // empty specs

    cfg.sweep_specs = {{CpuBandwidthSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"0", 0}, {"1", 1}}}};
    cfg.sweep_max_runs = 1;
    EXPECT_FALSE(preflight_cpu_bandwidth_sweep(cfg, sys_info));  // > sweep_max_runs

    cfg.sweep_max_runs = 16;
    cfg.custom_cache_size_kb = 0;
    std::ostringstream err;
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    EXPECT_FALSE(preflight_cpu_bandwidth_sweep(cfg, sys_info));  // (-b 0, -k 0) omission
    cfg.sweep_specs = {{CpuBandwidthSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"999999999", 999999999ULL}}}};
    EXPECT_FALSE(preflight_cpu_bandwidth_sweep(cfg, sys_info));  // budget exceeded
    std::cerr.rdbuf(old_err);

    cfg.sweep_specs = {{CpuBandwidthSweepParameter::CacheSizeKb,
                        "cache-size",
                        {{"16", 16}, {"32", 32}}}};
    cfg.buffer_size_mb = 0;
    EXPECT_TRUE(preflight_cpu_bandwidth_sweep(cfg, sys_info));
  }

  // 2. Multi-point run_cpu_bandwidth_sweep with File session & nested failure.
  {
    const auto sweep_out = (tmp.dir / "unit_sweep.json").string();
    CpuBandwidthConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    cfg.seed = 42ULL;
    cfg.user_specified_seed = true;
    cfg.run_sweep = true;
    cfg.output_file = sweep_out;
    cfg.sweep_specs = {{CpuBandwidthSweepParameter::CacheSizeKb,
                        "cache-size",
                        {{"16", 16}, {"32", 32}}}};

    JsonOutputSession session(make_json_output_target(sweep_out));
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const SweepExecutionResult exec = run_cpu_bandwidth_sweep(cfg, session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(exec.output_json["status"], "complete");
    EXPECT_EQ(exec.output_json["completed_runs"], 2u);
    EXPECT_TRUE(exec.output_json["conclusions_valid"].get<bool>());

    // Timer failure path in run_cpu_bandwidth_sweep.
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
        run_cpu_bandwidth_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    reset_timer_system_calls_for_testing();
    EXPECT_NE(timer_fail.exit_code, EXIT_SUCCESS);

    // Nested run failure in sweep (bypassing preflight with huge buffer size).
    cfg.sweep_specs = {{CpuBandwidthSweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"999999999", 999999999ULL}}}};
    cfg.custom_cache_size_kb = 0;
    cfg.output_file.clear();
    std::ostringstream out3, err3;
    old_out = std::cout.rdbuf(out3.rdbuf());
    old_err = std::cerr.rdbuf(err3.rdbuf());
    const SweepExecutionResult fail_exec =
        run_cpu_bandwidth_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(fail_exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(fail_exec.output_json["status"], "failed");
  }
}

TEST(CpuBandwidthModeIntegrationTest, EndToEndDirectStdoutAndSweepProduceValidSchemaV1) {
  TempDirGuard tmp("cpu_bw_e2e");

  // 1. Direct file output with explicit iterations & seed.
  const auto direct_path = (tmp.dir / "direct.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "1", "-k", "16", "-t", "2",
                      "-i", "2", "-r", "3", "--seed", "424242", "-o",
                      direct_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_bandwidth_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
  }
  {
    std::ifstream ifs(direct_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["mode"], "cpu_bandwidth");
    EXPECT_EQ(doc["methodology_version"],
              "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced");
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["results_complete"].get<bool>());
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_TRUE(doc["operation_order_balance_complete"].get<bool>());
    EXPECT_EQ(doc["measurements"].size(), 18u);
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["main_memory"]["src"]
                   ["read_only_protected"]
                       .get<bool>());
  }

  // 2. Stdout JSON transport (`-o -`).
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "1", "-k", "0", "-i", "1",
                      "-r", "1", "-o", "-"});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_bandwidth_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    const auto doc = nlohmann::ordered_json::parse(out.str());
    EXPECT_EQ(doc["configuration"]["output_file"], "-");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_FALSE(err.str().empty());
  }

  // 3. Budget overflow writes failed JSON with memory_budget.valid == false.
  const auto budget_path = (tmp.dir / "budget_fail.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "999999999", "-k", "0",
                      "-i", "1", "-r", "1", "-o", budget_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_bandwidth_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(rc, EXIT_SUCCESS);
    std::ifstream ifs(budget_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
    EXPECT_FALSE(doc["memory_budget"]["valid"].get<bool>());
    EXPECT_EQ(doc["counters"]["planned_passes"], "0");
  }

  // 4. Sweep mode (`-S buffer-size=1,2`).
  const auto sweep_path = (tmp.dir / "sweep.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-k", "0", "-i", "1", "-r", "1",
                      "-S", "buffer-size=1,2", "-o", sweep_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_bandwidth_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    std::ifstream ifs(sweep_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["configuration"]["mode"], "sweep");
    EXPECT_EQ(doc["configuration"]["base_mode"], "cpu_bandwidth");
    EXPECT_EQ(doc["configuration"]["sweep_schema_version"], 1);
    EXPECT_EQ(doc["planned_runs"], 2u);
    EXPECT_EQ(doc["completed_runs"], 2u);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["conclusions_valid"].get<bool>());
  }

  // 5. Auto-calibrated run (default L1+L2+main with -b 8 -r 1) + Python verifier check.
  const auto autocal_path = (tmp.dir / "autocal.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "8", "-r", "1", "-o",
                      autocal_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_bandwidth_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
  }

  if (std::filesystem::exists("script-examples/verify_cpu_bandwidth_result.py")) {
    const auto v_out = (tmp.dir / "verdict.json").string();
    EXPECT_EQ(run_python_verifier(direct_path, "--require-raw-timing", v_out),
              0);
    EXPECT_EQ(run_python_verifier(autocal_path, "--require-raw-timing", v_out),
              0);
    EXPECT_EQ(run_python_verifier(sweep_path, "--require-raw-timing", v_out),
              0);
    EXPECT_EQ(run_python_verifier(budget_path, "", v_out), 1);
    std::ifstream vifs(v_out);
    ASSERT_TRUE(vifs.is_open());
    const auto verdict = nlohmann::ordered_json::parse(vifs);
    EXPECT_TRUE(verdict["artifact_consistent"].get<bool>());
    EXPECT_FALSE(verdict["run_accepted"].get<bool>());
  }
}

}  // namespace
