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

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "asm/asm_functions.h"
#include "benchmark/sweep_runner.h"
#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/system/benchmark_qos.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "cpu_latency/cpu_latency.h"
#include "cpu_latency/cpu_latency_executor.h"
#include "cpu_latency/cpu_latency_json.h"
#include "cpu_latency/cpu_latency_runner.h"
#include "cpu_latency/cpu_latency_sweep_runner.h"
#include "cpu_latency/cpu_latency_work_plan.h"
#include "output/json/json_output/json_output_session.h"
#include "utils/json_utils.h"

namespace {

int run_python_verifier(const std::string& json_path,
                        const std::string& extra_flags,
                        const std::string& verdict_out_path) {
  const std::string cmd =
      "python3 script-examples/verify_cpu_latency_result.py \"" + json_path +
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

TEST(CpuLatencyJsonAndSweepTest, ClassifyCpuLatencyCompletionStates) {
  nlohmann::ordered_json complete = {
      {"schema_version", 1},
      {"mode", "cpu_latency"},
      {"status", "complete"},
      {"reason_code", "complete"},
      {"results_complete", true},
      {"run_accepted", true},
      {"measurements", {{{"status", "measured"}}}},
  };
  const auto c_res =
      classify_sweep_nested_completion(SweepNestedMode::CpuLatency, complete);
  EXPECT_EQ(c_res.status, SweepAttemptStatus::Complete);

  nlohmann::ordered_json interrupted = complete;
  interrupted["status"] = "interrupted";
  interrupted["reason_code"] = "interruption-requested";
  interrupted["results_complete"] = false;
  interrupted["run_accepted"] = false;
  const auto i_res = classify_sweep_nested_completion(
      SweepNestedMode::CpuLatency, interrupted);
  EXPECT_EQ(i_res.status, SweepAttemptStatus::Interrupted);

  nlohmann::ordered_json failed = complete;
  failed["status"] = "failed";
  failed["reason_code"] = "memory-budget-exceeded";
  failed["results_complete"] = false;
  failed["run_accepted"] = false;
  const auto f_res =
      classify_sweep_nested_completion(SweepNestedMode::CpuLatency, failed);
  EXPECT_EQ(f_res.status, SweepAttemptStatus::Failed);
  EXPECT_EQ(f_res.reason, "memory-budget-exceeded");
}

TEST(CpuLatencyJsonAndSweepTest, SweepRunConfigsKeepFixedKeysAndEmptyOutputFile) {
  CpuLatencyConfig base;
  base.buffer_size_mb = 512;
  base.custom_cache_size_kb = 0;
  base.swept_buffer_size = true;
  base.swept_chain_mode = true;
  base.run_sweep = true;
  base.output_file = "sweep.json";
  base.sweep_specs = {
      {CpuLatencySweepParameter::BufferSizeMb,
       "buffer-size",
       {{"1", 1, LatencyChainMode::Auto}, {"2", 2, LatencyChainMode::Auto}}},
      {CpuLatencySweepParameter::LatencyChainMode,
       "latency-chain-mode",
       {{"global-random", 0, LatencyChainMode::GlobalRandom},
        {"random-box", 0, LatencyChainMode::RandomInBoxRandomBox}}},
  };

  EXPECT_EQ(calculate_cpu_latency_sweep_run_count(base), 4u);
  const auto runs = build_cpu_latency_sweep_run_configs(base);
  ASSERT_EQ(runs.size(), 4u);
  for (const auto& r : runs) {
    EXPECT_FALSE(r.run_sweep);
    EXPECT_TRUE(r.output_file.empty());
    EXPECT_TRUE(r.collect_json_for_sweep);
    EXPECT_TRUE(r.swept_buffer_size);
    EXPECT_TRUE(r.swept_chain_mode);
  }
  EXPECT_EQ(runs[0].buffer_size_mb, 1u);
  EXPECT_EQ(runs[0].latency_chain_mode, LatencyChainMode::GlobalRandom);
  EXPECT_EQ(runs[3].buffer_size_mb, 2u);
  EXPECT_EQ(runs[3].latency_chain_mode,
            LatencyChainMode::RandomInBoxRandomBox);
}

TEST(CpuLatencyJsonAndSweepTest,
     PeakEstimateWarningsAndBuildJsonAcrossAllStates) {
  CpuLatencyConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.custom_cache_size_kb = 16;
  cfg.iterations = 2;
  cfg.user_specified_iterations = true;
  cfg.loop_count = 2;
  cfg.latency_samples = 2;
  cfg.latency_stride_bytes = 256;
  cfg.latency_chain_mode = LatencyChainMode::Auto;
  cfg.latency_tlb_locality_kb = 16;
  cfg.compare_locality = true;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  cfg.argv = {"memory_benchmark", "-L", "-b", "1", "-k", "16", "-i", "2",
              "-r", "2", "-n", "2", "--compare-locality", "-u", "--seed",
              "424242"};

  CpuLatencySystemInfo sys_info;
  sys_info.l1_cache_bytes = 128ULL * 1024ULL;
  sys_info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  const auto plan = build_cpu_latency_work_plan(cfg, sys_info);
  ASSERT_TRUE(plan.valid);

  // 1. Peak estimate helper.
  {
    const auto empty_est = calculate_cpu_latency_json_peak_estimate(cfg, plan);
    EXPECT_TRUE(empty_est.valid);
    EXPECT_EQ(empty_est.total_bytes, 0u);
    auto cfg_out = cfg;
    cfg_out.output_file = "lat.json";
    const auto plan_out = build_cpu_latency_work_plan(cfg_out, sys_info);
    ASSERT_TRUE(plan_out.valid);
    const auto active_est =
        calculate_cpu_latency_json_peak_estimate(cfg_out, plan_out);
    EXPECT_TRUE(active_est.valid);
    EXPECT_GT(active_est.total_bytes, 0u);
  }

  // 2. Quality warning tokens across all branches (including compare_buffer & auto-duration).
  {
    CpuLatencyResourceSet res;
    res.compare_locality_active = true;
    res.compare_buffer.madvise_applied = false;

    CpuLatencyResultMetadata meta;
    meta.main_thread_qos = {true, false, 1};
    meta.environment_start.thermal_state = "critical";
    meta.environment_end.thermal_state = "nominal";
    meta.environment_end.low_power_mode_available = true;
    meta.environment_end.low_power_mode_enabled = true;

    CpuLatencyResult res_state;
    res_state.quality_warnings = {"custom-extra", "custom-extra"};
    CpuLatencyMeasurementRow row;
    row.target = CpuLatencyTarget::Custom;
    row.status = CpuLatencyMeasurementStatus::Measured;
    row.duration_quality = "above-target-window";
    res_state.measurements.push_back(row);
    res_state.aggregates[cpu_latency_target_index(CpuLatencyTarget::Custom)].cv_warning = true;

    auto auto_cfg = cfg;
    auto_cfg.iterations = std::nullopt;
    const auto tokens = collect_cpu_latency_quality_warning_tokens(
        auto_cfg, plan, res, meta, res_state);
    auto has = [&](const std::string& tok) {
      return std::find(tokens.begin(), tokens.end(), tok) != tokens.end();
    };
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), "custom-extra"), 1);
    EXPECT_TRUE(has("environment-not-nominal"));
    EXPECT_TRUE(has("main-thread-qos-not-applied"));
    EXPECT_TRUE(has("madvise-random-not-applied"));
    EXPECT_TRUE(has("custom-duration-above-target-window"));
    EXPECT_TRUE(has("custom-high-cv"));
  }

  // 3. Complete run JSON with sample_windows & compare_locality.
  {
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, true, res));
    CpuLatencyResult complete_res;
    ASSERT_EQ(run_cpu_latency_suite(cfg, plan, res, complete_res, {}),
              EXIT_SUCCESS);

    const size_t c_idx = cpu_latency_target_index(CpuLatencyTarget::Custom);
    CpuLatencyCalibrationAttempt placeholder;
    placeholder.attempt_index = complete_res.calibration_attempts[c_idx].size();
    placeholder.target = CpuLatencyTarget::Custom;
    placeholder.purpose = "placeholder";
    complete_res.calibration_attempts[c_idx].push_back(placeholder);

    CpuLatencyResultMetadata meta;
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

    const auto doc = build_cpu_latency_json(cfg, plan, res, meta, complete_res);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["compare_buffer"]["read_only_protected"].get<bool>());
    EXPECT_EQ(doc["locality_comparison"]["rounds"].size(), 6u);
    EXPECT_FALSE(doc["aggregates"]["locality_comparison"]["locality_latency_delta_ns"].is_null());
    EXPECT_FALSE(doc["measurements"][0]["sample_windows"].is_null());
    EXPECT_TRUE(doc["environment"]["available_memory_bytes"].is_null());
    EXPECT_TRUE(doc["environment"]["start"]["low_power_mode_enabled"].is_null());
    release_cpu_latency_resources(res);
  }

  // 4. Partial, Interrupted, and Failed JSON states (including invalid work_plan & empty resources).
  {
    CpuLatencyResourceSet empty_res;
    CpuLatencyResultMetadata meta;
    meta.environment_start.thermal_state = "nominal";
    meta.environment_end = meta.environment_start;

    CpuLatencyResult int_res;
    initialize_cpu_latency_setup_failure_result(
        cfg, plan, CpuLatencyRunStatus::Interrupted, "interruption-requested",
        true, int_res);
    int_res.diagnostic = "interrupted-diagnostic";
    const auto int_doc =
        build_cpu_latency_json(cfg, plan, empty_res, meta, int_res);
    EXPECT_EQ(int_doc["status"], "interrupted");
    EXPECT_TRUE(int_doc["interruption_requested"].get<bool>());
    EXPECT_EQ(int_doc["diagnostic"], "interrupted-diagnostic");
    EXPECT_TRUE(int_doc["resolved_plan"]["resources"].empty());

    CpuLatencyWorkPlan invalid_plan;
    invalid_plan.valid = false;
    invalid_plan.reason_code = "work-cap-exceeded";
    CpuLatencyResult fail_res;
    initialize_cpu_latency_setup_failure_result(
        cfg, invalid_plan, CpuLatencyRunStatus::Failed, "work-cap-exceeded",
        false, fail_res);
    const auto fail_doc =
        build_cpu_latency_json(cfg, invalid_plan, empty_res, meta, fail_res);
    EXPECT_EQ(fail_doc["status"], "failed");
    EXPECT_FALSE(fail_doc["resolved_plan"]["valid"].get<bool>());
    EXPECT_EQ(fail_doc["resolved_plan"]["plan_identity"],
              "cpu-latency-work-plan-v1|invalid");
  }
}

TEST(CpuLatencyJsonAndSweepTest, SweepPreflightAndRunnerLifecycleUnitPaths) {
  TempDirGuard tmp("cpu_lat_sweep_unit");

  CpuLatencySystemInfo sys_info;
  sys_info.l1_cache_bytes = 128ULL * 1024ULL;
  sys_info.l2_cache_bytes = 16ULL * 1024ULL * 1024ULL;
  sys_info.page_size_bytes = 16384ULL;
  sys_info.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  // 1. Preflight rejections & valid sweep across all 5 axes.
  {
    CpuLatencyConfig cfg;
    EXPECT_FALSE(preflight_cpu_latency_sweep(cfg, sys_info));  // !run_sweep
    cfg.run_sweep = true;
    EXPECT_FALSE(preflight_cpu_latency_sweep(cfg, sys_info));  // empty specs

    cfg.sweep_specs = {{CpuLatencySweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"0", 0}, {"1", 1}}}};
    cfg.sweep_max_runs = 1;
    EXPECT_FALSE(preflight_cpu_latency_sweep(cfg, sys_info));  // > sweep_max_runs

    cfg.sweep_max_runs = 64;
    cfg.custom_cache_size_kb = 0;
    std::ostringstream err;
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    EXPECT_FALSE(preflight_cpu_latency_sweep(cfg, sys_info));  // target omission

    cfg.sweep_specs = {{CpuLatencySweepParameter::LatencyTlbLocalityKb,
                        "latency-tlb-locality-kb",
                        {{"1", 1}}}};
    cfg.buffer_size_mb = 1;
    EXPECT_FALSE(preflight_cpu_latency_sweep(cfg, sys_info));  // non-page-multiple locality
    std::cerr.rdbuf(old_err);

    cfg.buffer_size_mb = 0;
    cfg.sweep_specs = {
        {CpuLatencySweepParameter::CacheSizeKb, "cache-size", {{"16", 16}}},
        {CpuLatencySweepParameter::LatencyStrideBytes, "latency-stride-bytes", {{"128", 128}}},
        {CpuLatencySweepParameter::LatencyTlbLocalityKb, "latency-tlb-locality-kb", {{"16", 16}}},
        {CpuLatencySweepParameter::LatencyChainMode,
         "latency-chain-mode",
         {{"global-random", 0, LatencyChainMode::GlobalRandom},
          {"random-box", 0, LatencyChainMode::RandomInBoxRandomBox}}},
    };
    EXPECT_TRUE(preflight_cpu_latency_sweep(cfg, sys_info));
  }

  // 2. Multi-point run_cpu_latency_sweep with File session, timer failure, and nested failure.
  {
    const auto sweep_out = (tmp.dir / "unit_lat_sweep.json").string();
    CpuLatencyConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.latency_samples = 0;
    cfg.seed = 42ULL;
    cfg.user_specified_seed = true;
    cfg.run_sweep = true;
    cfg.output_file = sweep_out;
    cfg.sweep_specs = {
        {CpuLatencySweepParameter::CacheSizeKb, "cache-size", {{"16", 16}}},
        {CpuLatencySweepParameter::LatencyStrideBytes, "latency-stride-bytes", {{"128", 128}, {"256", 256}}},
    };

    JsonOutputSession session(make_json_output_target(sweep_out));
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const SweepExecutionResult exec = run_cpu_latency_sweep(cfg, session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(exec.output_json["status"], "complete");
    EXPECT_EQ(exec.output_json["completed_runs"], 2u);
    EXPECT_TRUE(exec.output_json["conclusions_valid"].get<bool>());

    // Timer failure path.
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
        run_cpu_latency_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    reset_timer_system_calls_for_testing();
    EXPECT_NE(timer_fail.exit_code, EXIT_SUCCESS);

    // Nested run failure in sweep (bypassing preflight with work-cap-exceeded).
    cfg.sweep_specs = {{CpuLatencySweepParameter::BufferSizeMb,
                        "buffer-size",
                        {{"999999999", 999999999ULL}}}};
    cfg.custom_cache_size_kb = 0;
    cfg.output_file.clear();
    std::ostringstream out3, err3;
    old_out = std::cout.rdbuf(out3.rdbuf());
    old_err = std::cerr.rdbuf(err3.rdbuf());
    const SweepExecutionResult fail_exec =
        run_cpu_latency_sweep(cfg, disabled_session);
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(fail_exec.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(fail_exec.output_json["status"], "failed");
  }
}

TEST(CpuLatencyModeIntegrationTest, EndToEndDirectStdoutSamplesLocalityAndSweepValid) {
  TempDirGuard tmp("cpu_lat_e2e");

  // 1. Direct file output with explicit iterations, seed & sample windows.
  const auto direct_path = (tmp.dir / "direct.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "1", "-k", "16", "-i",
                      "2", "-r", "3", "-n", "10", "--seed", "424242", "-o",
                      direct_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
  }
  {
    std::ifstream ifs(direct_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["mode"], "cpu_latency");
    EXPECT_EQ(doc["methodology_version"],
              "cpu-latency-v1-arm64-calibrated-seeded-balanced");
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["results_complete"].get<bool>());
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_EQ(doc["measurements"].size(), 6u);
    EXPECT_TRUE(doc["resolved_plan"]["resources"]["main_memory"]["buffer"]
                   ["read_only_protected"]
                       .get<bool>());
  }

  // 2. Stdout JSON transport (`-o -`).
  const auto stdout_path = (tmp.dir / "stdout_capture.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "1", "-k", "0", "-i",
                      "1", "-r", "1", "-n", "5", "-o", "-"});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    const auto doc = nlohmann::ordered_json::parse(out.str());
    EXPECT_EQ(doc["configuration"]["output_file"], "-");
    EXPECT_TRUE(doc["run_accepted"].get<bool>());
    EXPECT_FALSE(err.str().empty());
    std::ofstream ofs(stdout_path);
    ofs << out.str();
  }

  // 3. `-n 0` disables sample windows and pooled_sample_distribution.
  const auto n0_path = (tmp.dir / "n0.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "1", "-k", "0", "-i",
                      "2", "-r", "2", "-n", "0", "-o", n0_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    std::ifstream ifs(n0_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["configuration"]["latency_samples"], 0);
    EXPECT_TRUE(doc["measurements"][0]["sample_windows"].is_null());
    EXPECT_TRUE(doc["aggregates"]["targets"]["main_memory"]
                   ["pooled_sample_distribution"]
                       .is_null());
  }

  // 4. `--compare-locality` runs 3 paired rounds per loop.
  const auto loc_path = (tmp.dir / "compare_locality.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "2", "-k", "0", "-i",
                      "1", "-r", "2", "-n", "5", "--compare-locality", "-o",
                      loc_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    std::ifstream ifs(loc_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_TRUE(doc["configuration"]["compare_locality"].get<bool>());
    EXPECT_EQ(doc["locality_comparison"]["rounds"].size(), 6u);
    EXPECT_FALSE(doc["aggregates"]["locality_comparison"].is_null());
  }

  // 5. Auto-calibrated run (default L1+L2+main with -b 4 -r 1 -n 10).
  const auto autocal_path = (tmp.dir / "autocal.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "4", "-r", "1", "-n",
                      "10", "-o", autocal_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
  }

  // 6. Budget overflow rejects cleanly without silent capping.
  const auto budget_path = (tmp.dir / "budget_fail.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "999999999", "-k", "0",
                      "-i", "1", "-r", "1", "-o", budget_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_NE(rc, EXIT_SUCCESS);
    EXPECT_FALSE(err.str().empty());
  }

  // 7. Multi-axis sweep (`-S buffer-size=1,2 -S latency-chain-mode=global-random,random-box`).
  const auto sweep_path = (tmp.dir / "sweep.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-k", "0", "-l", "16", "-i",
                      "1", "-r", "1", "-n", "5", "-S", "buffer-size=1,2", "-S",
                      "latency-chain-mode=global-random,random-box", "-o",
                      sweep_path.c_str()});
    std::ostringstream out, err;
    auto* old_out = std::cout.rdbuf(out.rdbuf());
    auto* old_err = std::cerr.rdbuf(err.rdbuf());
    const int rc = run_cpu_latency_mode(args.argc(), args.argv());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    EXPECT_EQ(rc, EXIT_SUCCESS) << err.str();
    std::ifstream ifs(sweep_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["configuration"]["mode"], "sweep");
    EXPECT_EQ(doc["configuration"]["base_mode"], "cpu_latency");
    EXPECT_EQ(doc["configuration"]["sweep_schema_version"], 1);
    EXPECT_EQ(doc["planned_runs"], 4u);
    EXPECT_EQ(doc["completed_runs"], 4u);
    EXPECT_EQ(doc["status"], "complete");
    EXPECT_TRUE(doc["conclusions_valid"].get<bool>());
  }

  if (std::filesystem::exists("script-examples/verify_cpu_latency_result.py")) {
    const auto v_out = (tmp.dir / "verdict.json").string();
    for (const std::string& path :
         {direct_path, stdout_path, n0_path, loc_path, autocal_path,
          sweep_path}) {
      const int v_rc = run_python_verifier(path, "--require-raw-timing", v_out);
      std::ifstream vifs(v_out);
      ASSERT_TRUE(vifs.is_open());
      const auto verdict = nlohmann::ordered_json::parse(vifs);
      EXPECT_EQ(v_rc, 0) << "Verifier failed for " << path << ": "
                         << verdict.dump();
      EXPECT_TRUE(verdict["artifact_consistent"].get<bool>())
          << path << ": " << verdict.dump();
      EXPECT_TRUE(verdict["run_accepted"].get<bool>())
          << path << ": " << verdict.dump();
    }
  }
}

TEST(CpuLatencyModeIntegrationTest,
     BoundaryLocalityAndFinalMeasurementInterruptionsPassVerifier) {
  ASSERT_TRUE(
      std::filesystem::exists("script-examples/verify_cpu_latency_result.py"));
  TempDirGuard tmp("cpu_lat_boundary_e2e");

  auto make_boundary_config = [&](bool compare_locality,
                                  const std::string& out_path) {
    CpuLatencyConfig cfg;
    cfg.buffer_size_mb = 1;
    cfg.user_specified_buffer_size = true;
    cfg.custom_cache_size_kb = 0;
    cfg.user_specified_cache_size = true;
    cfg.iterations = 2;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 2;
    cfg.latency_samples = 2;
    cfg.user_specified_latency_samples = true;
    cfg.latency_stride_bytes = 256;
    cfg.latency_chain_mode = LatencyChainMode::Auto;
    cfg.latency_tlb_locality_kb = 16;
    cfg.compare_locality = compare_locality;
    cfg.seed = 424242ULL;
    cfg.user_specified_seed = true;
    cfg.output_file = out_path;
    cfg.argv = {"memory_benchmark", "-L", "-b", "1", "-k", "0", "-i", "2",
                "-r", "2", "-n", "2", "--seed", "424242", "-o", out_path};
    if (compare_locality) {
      cfg.argv.push_back("--compare-locality");
    }
    return cfg;
  };

  auto capture_metadata = [&]() {
    CpuLatencyResultMetadata meta;
    meta.build_manifest =
        build_runtime_provenance_manifest_json(capture_binary_sha256());
    meta.timestamp = build_utc_timestamp();
    meta.processor_name = get_processor_name();
    meta.macos_version = get_macos_version();
    meta.performance_core_count = get_performance_cores();
    meta.efficiency_core_count = get_efficiency_cores();
    meta.logical_core_count = get_total_logical_cores();
    meta.page_size_bytes = get_system_page_size_bytes();
    meta.l1_data_cache_bytes = get_l1_cache_size();
    meta.l2_data_cache_bytes = get_l2_cache_size();
    const unsigned long avail_mb = get_available_memory_mb();
    meta.available_memory_bytes =
        static_cast<size_t>(avail_mb) * Constants::BYTES_PER_MB;
    meta.available_memory_source =
        meta.available_memory_bytes == 0
            ? "unavailable"
            : "mach-free-plus-inactive-rounded-mib";
    meta.main_thread_qos =
        prepare_main_thread_benchmark_qos(MainThreadQosSetter{}, false);
    meta.environment_start = capture_host_environment();
    meta.environment_end = meta.environment_start;
    return meta;
  };

  auto run_and_verify = [&](const std::string& filename,
                            bool compare_locality,
                            int expected_run_rc,
                            int expected_verifier_rc,
                            bool expected_run_accepted,
                            const std::string& expected_verifier_reason,
                            auto configure_hooks) {
    const std::string artifact_path = (tmp.dir / filename).string();
    const std::string verdict_path = (tmp.dir / (filename + ".verdict")).string();
    const CpuLatencyConfig cfg =
        make_boundary_config(compare_locality, artifact_path);
    CpuLatencyResultMetadata meta = capture_metadata();

    CpuLatencySystemInfo sys_info;
    sys_info.l1_cache_bytes = meta.l1_data_cache_bytes;
    sys_info.l2_cache_bytes = meta.l2_data_cache_bytes;
    sys_info.page_size_bytes = meta.page_size_bytes;
    sys_info.available_memory_bytes = meta.available_memory_bytes;

    const CpuLatencyWorkPlan plan = build_cpu_latency_work_plan(cfg, sys_info);
    ASSERT_TRUE(plan.valid) << filename;
    CpuLatencyResourceSet res;
    ASSERT_TRUE(prepare_cpu_latency_resources(plan, false, res)) << filename;

    CpuLatencyResult result;
    CpuLatencyRunnerHooks hooks;
    configure_hooks(plan, hooks);
    const int rc = run_cpu_latency_suite(cfg, plan, res, result, hooks);
    EXPECT_EQ(rc, expected_run_rc) << filename;
    meta.environment_end = capture_host_environment();

    const auto doc = build_cpu_latency_json(cfg, plan, res, meta, result);
    {
      std::ofstream ofs(artifact_path);
      ASSERT_TRUE(ofs.is_open()) << artifact_path;
      ofs << doc.dump(2);
    }
    release_cpu_latency_resources(res);

    const int v_rc =
        run_python_verifier(artifact_path, "--require-raw-timing", verdict_path);
    std::ifstream vifs(verdict_path);
    ASSERT_TRUE(vifs.is_open()) << verdict_path;
    const auto verdict = nlohmann::ordered_json::parse(vifs);
    EXPECT_EQ(v_rc, expected_verifier_rc) << filename << ": " << verdict.dump();
    EXPECT_TRUE(verdict["artifact_consistent"].get<bool>())
        << filename << ": " << verdict.dump();
    EXPECT_EQ(verdict["run_accepted"].get<bool>(), expected_run_accepted)
        << filename << ": " << verdict.dump();
    EXPECT_EQ(verdict["reason_code"], expected_verifier_reason)
        << filename << ": " << verdict.dump();
  };

  // 1a. Loop 0 --compare-locality interruption at before_locality_round_hook(0, 1).
  {
    std::atomic<bool> stop{false};
    run_and_verify("loop0_loc_hook_stop.json", true, EXIT_SUCCESS, 1, false,
                   "run-not-accepted",
                   [&](const CpuLatencyWorkPlan&, CpuLatencyRunnerHooks& hooks) {
                     stop.store(false);
                     hooks.stop_requested = [&]() { return stop.load(); };
                     hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                            size_t r_idx) {
                       if (loop_idx == 0 && r_idx == 1) {
                         stop.store(true);
                       }
                     };
                   });
  }

  // 1b. Loop 0 --compare-locality mid-round interruption (between locality_16k and locality_global).
  {
    std::atomic<bool> armed{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (armed.exchange(false)) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };
    run_and_verify("loop0_loc_mid_round_stop.json", true, EXIT_SUCCESS, 1,
                   false, "run-not-accepted",
                   [&](const CpuLatencyWorkPlan&, CpuLatencyRunnerHooks& hooks) {
                     armed.store(false);
                     stop.store(false);
                     hooks.executor_test_control = &ctrl;
                     hooks.stop_requested = [&]() { return stop.load(); };
                     hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                            size_t r_idx) {
                       if (loop_idx == 0 && r_idx == 0) {
                         armed.store(true);
                       }
                     };
                   });
  }

  // 2a. Final-loop --compare-locality interruption (Challenger stop_mid_locality at (1, 1)).
  {
    std::atomic<bool> stop{false};
    run_and_verify("final_loop_loc_hook_stop.json", true, EXIT_SUCCESS, 1,
                   false, "run-not-accepted",
                   [&](const CpuLatencyWorkPlan&, CpuLatencyRunnerHooks& hooks) {
                     stop.store(false);
                     hooks.stop_requested = [&]() { return stop.load(); };
                     hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                            size_t r_idx) {
                       if (loop_idx == 1 && r_idx == 1) {
                         stop.store(true);
                       }
                     };
                   });
  }

  // 2b. Final-loop --compare-locality mid-round interruption (between locality_16k and locality_global).
  {
    std::atomic<bool> armed{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (armed.exchange(false)) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };
    run_and_verify("final_loop_loc_mid_round_stop.json", true, EXIT_SUCCESS, 1,
                   false, "run-not-accepted",
                   [&](const CpuLatencyWorkPlan&, CpuLatencyRunnerHooks& hooks) {
                     armed.store(false);
                     stop.store(false);
                     hooks.executor_test_control = &ctrl;
                     hooks.stop_requested = [&]() { return stop.load(); };
                     hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                            size_t r_idx) {
                       if (loop_idx == 1 && r_idx == 1) {
                         armed.store(true);
                       }
                     };
                   });
  }

  // 3a. Final-loop --compare-locality failure (Hamiltonian fail at (1, 1)).
  {
    CpuLatencyExecutorTestControl ctrl;
    run_and_verify("final_loop_loc_hamiltonian_fail.json", true, EXIT_FAILURE,
                   1, false, "run-not-accepted",
                   [&](const CpuLatencyWorkPlan&, CpuLatencyRunnerHooks& hooks) {
                     ctrl.force_hamiltonian_fail = false;
                     hooks.executor_test_control = &ctrl;
                     hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                            size_t r_idx) {
                       if (loop_idx == 1 && r_idx == 1) {
                         ctrl.force_hamiltonian_fail = true;
                       }
                     };
                   });
  }

  // 3b. Final-loop --compare-locality failure (pointer corruption at (1, 1)).
  {
    std::atomic<bool> corrupt{false};
    CpuLatencyExecutorTestControl ctrl;
    run_and_verify(
        "final_loop_loc_pointer_corrupt.json", true, EXIT_FAILURE, 1, false,
        "run-not-accepted",
        [&](const CpuLatencyWorkPlan& plan, CpuLatencyRunnerHooks& hooks) {
          corrupt.store(false);
          ctrl.chase_override = [&, stride = plan.locality_16k_plan->stride_bytes](
                                    uintptr_t* ptr, size_t steps) -> uintptr_t* {
            if (corrupt.load()) {
              (void)memory_latency_chase_asm(ptr, steps);
              return chain_node_to_pointer(ptr, stride, 1);
            }
            return memory_latency_chase_asm(ptr, steps);
          };
          hooks.executor_test_control = &ctrl;
          hooks.before_locality_round_hook = [&](size_t loop_idx,
                                                 size_t r_idx) {
            if (loop_idx == 1 && r_idx == 1) {
              corrupt.store(true);
            }
          };
        });
  }

  // 4a. Interruption arriving during measurements[last] (compare_locality = false).
  // 4b. Interruption arriving during measurements[last] (compare_locality = true).
  for (const bool compare_locality : {false, true}) {
    std::atomic<bool> in_final{false};
    std::atomic<bool> stop{false};
    CpuLatencyExecutorTestControl ctrl;
    ctrl.chase_override = [&](uintptr_t* ptr, size_t steps) -> uintptr_t* {
      if (in_final.load()) {
        stop.store(true);
      }
      return memory_latency_chase_asm(ptr, steps);
    };
    const std::string fname = compare_locality
                                  ? "stop_during_last_meas_loc_true.json"
                                  : "stop_during_last_meas_loc_false.json";
    run_and_verify(
        fname, compare_locality, EXIT_SUCCESS, 0, true, "valid",
        [&](const CpuLatencyWorkPlan& plan, CpuLatencyRunnerHooks& hooks) {
          in_final.store(false);
          stop.store(false);
          hooks.executor_test_control = &ctrl;
          hooks.stop_requested = [&]() { return stop.load(); };
          hooks.before_measurement_hook = [&](size_t loop_idx, size_t pos) {
            if (loop_idx == 1 && pos + 1 == plan.active_targets.size()) {
              in_final.store(true);
            }
          };
        });
  }
}

}  // namespace
