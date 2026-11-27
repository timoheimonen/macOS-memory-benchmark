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

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <vector>

#include "core/memory/memory_manager.h"
#include "cpu_bandwidth/cpu_bandwidth.h"
#include "cpu_bandwidth/cpu_bandwidth_executor.h"
#include "cpu_bandwidth/cpu_bandwidth_output.h"
#include "cpu_bandwidth/cpu_bandwidth_runner.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"
#include "output/json/json_output/json_output_session.h"

namespace {

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

struct StreamCapture {
  std::ostringstream out;
  std::ostringstream err;
  std::streambuf* old_out = nullptr;
  std::streambuf* old_err = nullptr;

  StreamCapture() {
    old_out = std::cout.rdbuf(out.rdbuf());
    old_err = std::cerr.rdbuf(err.rdbuf());
  }
  ~StreamCapture() {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
  }
};

class CpuBandwidthConfigTest : public ::testing::Test {
 protected:
  void TearDown() override {
    set_cpu_bandwidth_parser_test_hooks(nullptr);
    set_cpu_bandwidth_command_test_hooks(nullptr);
    reset_memory_system_calls_for_testing();
  }
};

struct ConfigTempDirGuard {
  std::filesystem::path dir;
  explicit ConfigTempDirGuard(const std::string& prefix) {
    dir = std::filesystem::temp_directory_path() / (prefix + "_gtest");
    std::filesystem::create_directories(dir);
  }
  ~ConfigTempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

void* bw_fail_mmap(void*, size_t, int, int, int, off_t) {
  errno = ENOMEM;
  return MAP_FAILED;
}

int bw_fail_mprotect(void*, size_t, int) {
  errno = EACCES;
  return -1;
}

TEST_F(CpuBandwidthConfigTest, ParsesDefaultsAndAllWhitelistedOptions) {
  ArgvBuilder args({"memory_benchmark", "--cpu-bandwidth"});
  CpuBandwidthCliOptions opts;
  StreamCapture cap;
  EXPECT_EQ(parse_cpu_bandwidth_cli(args.argc(), args.argv(), opts),
            EXIT_SUCCESS);
  EXPECT_EQ(opts.buffer_size_mb, 512u);
  EXPECT_FALSE(opts.custom_cache_size_kb.has_value());
  EXPECT_FALSE(opts.threads.has_value());
  EXPECT_FALSE(opts.iterations.has_value());
  EXPECT_EQ(opts.loop_count, 3u);
  EXPECT_FALSE(opts.non_cacheable);
  EXPECT_FALSE(opts.seed.has_value());
  EXPECT_TRUE(opts.output_file.empty());
  EXPECT_FALSE(opts.run_sweep);
  EXPECT_EQ(opts.sweep_max_runs, 256u);
  EXPECT_FALSE(opts.help_requested);

  ArgvBuilder full(
      {"memory_benchmark", "-B", "--buffer-size=16", "--cache-size=64",
       "--threads=2", "--iterations=4", "--count=5", "--seed=0", "-u",
       "--output=out.json", "-X", "12"});
  CpuBandwidthCliOptions opts2;
  EXPECT_EQ(parse_cpu_bandwidth_cli(full.argc(), full.argv(), opts2),
            EXIT_SUCCESS);
  EXPECT_EQ(opts2.buffer_size_mb, 16u);
  ASSERT_TRUE(opts2.custom_cache_size_kb.has_value());
  EXPECT_EQ(*opts2.custom_cache_size_kb, 64u);
  ASSERT_TRUE(opts2.threads.has_value());
  EXPECT_EQ(*opts2.threads, 2u);
  ASSERT_TRUE(opts2.iterations.has_value());
  EXPECT_EQ(*opts2.iterations, 4u);
  EXPECT_EQ(opts2.loop_count, 5u);
  EXPECT_TRUE(opts2.non_cacheable);
  ASSERT_TRUE(opts2.seed.has_value());
  EXPECT_EQ(*opts2.seed, 0ULL);
  EXPECT_EQ(opts2.output_file, "out.json");
  EXPECT_EQ(opts2.sweep_max_runs, 12u);
}

TEST_F(CpuBandwidthConfigTest, HelpExitsZeroBeforeOsQueriesAndDoesNotCreateDash) {
  const std::vector<std::vector<const char*>> cases = {
      {"memory_benchmark", "-B", "-h"},
      {"memory_benchmark", "--cpu-bandwidth", "--help"},
      {"memory_benchmark", "-B", "-h", "-o", "-"},
      {"memory_benchmark", "--cpu-bandwidth", "--output=-", "--help"},
  };
  for (const auto& vec : cases) {
    ArgvBuilder args({});
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.ptrs.clear();
    for (auto& s : args.storage) {
      args.ptrs.push_back(s.data());
    }
    CpuBandwidthCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_cpu_bandwidth_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS);
    EXPECT_TRUE(opts.help_requested);
    EXPECT_TRUE(cap.err.str().empty());
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);
    EXPECT_NE(cap.out.str().find("--cpu-bandwidth"), std::string::npos);
    EXPECT_EQ(cap.out.str().find("Running"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists("-"));
  }
}

TEST_F(CpuBandwidthConfigTest, RejectsSyntaxUnknownCrossModeAndLegacyFlags) {
  const std::vector<std::vector<const char*>> bad_cases = {
      {"memory_benchmark", "-B", "-b=16"},
      {"memory_benchmark", "-B", "-b", "16", "-b", "16"},
      {"memory_benchmark", "-B", "--unknown-option"},
      {"memory_benchmark", "-B", "--help=1"},
      {"memory_benchmark", "-B", "-b"},
      {"memory_benchmark", "-B", "-o"},
      {"memory_benchmark", "-B", "positional"},
      {"memory_benchmark", "-B", "-n", "10"},
      {"memory_benchmark", "-B", "-s", "64"},
      {"memory_benchmark", "-B", "-m", "auto"},
      {"memory_benchmark", "-B", "-l", "64"},
      {"memory_benchmark", "-B", "--compare-locality"},
      {"memory_benchmark", "-B", "-W"},
      {"memory_benchmark", "-B", "--benchmark"},
      {"memory_benchmark", "-B", "--only-bandwidth"},
      {"memory_benchmark", "-B", "--only-latency"},
  };
  for (const auto& vec : bad_cases) {
    ArgvBuilder args({});
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.ptrs.clear();
    for (auto& s : args.storage) {
      args.ptrs.push_back(s.data());
    }
    CpuBandwidthCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_cpu_bandwidth_cli(args.argc(), args.argv(), opts),
              EXIT_FAILURE);
    EXPECT_FALSE(cap.err.str().empty());
  }
}

TEST_F(CpuBandwidthConfigTest, RejectsInvalidNumericBoundsAndTargetOmission) {
  const std::vector<std::vector<const char*>> bad_cases = {
      {"memory_benchmark", "-B", "-b", "-1"},
      {"memory_benchmark", "-B", "-b", "abc"},
      {"memory_benchmark", "-B", "-b", "9007199254740992"},
      {"memory_benchmark", "-B", "-k", "15"},
      {"memory_benchmark", "-B", "-k", "1048577"},
      {"memory_benchmark", "-B", "-i", "0"},
      {"memory_benchmark", "-B", "-i", "1000000001"},
      {"memory_benchmark", "-B", "-r", "0"},
      {"memory_benchmark", "-B", "-t", "0"},
      {"memory_benchmark", "-B", "--seed", "-1"},
      {"memory_benchmark", "-B", "--seed", "+1"},
      {"memory_benchmark", "-B", "-X", "0"},
      {"memory_benchmark", "-B", "-b", "0", "-k", "0"},
  };
  for (const auto& vec : bad_cases) {
    ArgvBuilder args({});
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.ptrs.clear();
    for (auto& s : args.storage) {
      args.ptrs.push_back(s.data());
    }
    CpuBandwidthCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_cpu_bandwidth_cli(args.argc(), args.argv(), opts),
              EXIT_FAILURE);
    EXPECT_FALSE(cap.err.str().empty());
  }
}

TEST_F(CpuBandwidthConfigTest, RejectsInvalidSweepFlagsAndCartesianTargetOmission) {
  const std::vector<std::vector<const char*>> bad_parse = {
      {"memory_benchmark", "-B", "-S", "buffer-size=8,16"},
      {"memory_benchmark", "-B", "-b", "8", "-S", "buffer-size=8,16", "-o", "out.json"},
      {"memory_benchmark", "-B", "-k", "64", "-S", "cache-size=32,64", "-o", "out.json"},
      {"memory_benchmark", "-B", "-t", "2", "-S", "threads=1,2", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "buffer-size=8", "-S", "buffer-size=16", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "buffer-size=8,16", "-S", "threads=1,2", "-X", "3", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "buffer-size=", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "buffer-size=8,,16", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "latency-stride-bytes=64,256", "-o", "out.json"},
      {"memory_benchmark", "-B", "-S", "tlb-density=low,high", "-o", "out.json"},
  };
  for (const auto& vec : bad_parse) {
    ArgvBuilder args({});
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.ptrs.clear();
    for (auto& s : args.storage) {
      args.ptrs.push_back(s.data());
    }
    CpuBandwidthCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_cpu_bandwidth_cli(args.argc(), args.argv(), opts),
              EXIT_FAILURE);
  }

  // Cartesian combination yielding (-b 0, -k 0) must be rejected at config build.
  CpuBandwidthParserTestHooks hooks;
  hooks.detected_workers = 4;
  hooks.generated_seed = 987654321ULL;
  hooks.use_memory_override = true;
  hooks.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  set_cpu_bandwidth_parser_test_hooks(&hooks);

  ArgvBuilder omit_sweep({"memory_benchmark", "-B", "-k", "0", "-S",
                          "buffer-size=0,16", "-o", "out.json"});
  CpuBandwidthCliOptions opts;
  StreamCapture cap;
  ASSERT_EQ(parse_cpu_bandwidth_cli(omit_sweep.argc(), omit_sweep.argv(), opts),
            EXIT_SUCCESS);
  CpuBandwidthConfig cfg;
  EXPECT_EQ(build_cpu_bandwidth_config(opts, cfg), EXIT_FAILURE);
}

TEST_F(CpuBandwidthConfigTest, ResolvesSeedWorkersAndSweepFlagsDeterministically) {
  CpuBandwidthParserTestHooks hooks;
  hooks.detected_workers = 8;
  hooks.generated_seed = 11223344ULL;
  hooks.use_memory_override = true;
  hooks.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  set_cpu_bandwidth_parser_test_hooks(&hooks);

  ArgvBuilder auto_args({"memory_benchmark", "-B", "-b", "16"});
  CpuBandwidthCliOptions opts;
  ASSERT_EQ(parse_cpu_bandwidth_cli(auto_args.argc(), auto_args.argv(), opts),
            EXIT_SUCCESS);
  CpuBandwidthConfig cfg;
  ASSERT_EQ(build_cpu_bandwidth_config(opts, cfg), EXIT_SUCCESS);
  EXPECT_EQ(cfg.available_workers, 8u);
  EXPECT_EQ(cfg.requested_workers, 8u);
  EXPECT_FALSE(cfg.user_specified_workers);
  EXPECT_FALSE(cfg.user_specified_seed);
  EXPECT_EQ(cfg.seed, 11223344ULL);

  ArgvBuilder zero_seed({"memory_benchmark", "-B", "-b", "16", "--seed", "0"});
  ASSERT_EQ(parse_cpu_bandwidth_cli(zero_seed.argc(), zero_seed.argv(), opts),
            EXIT_SUCCESS);
  ASSERT_EQ(build_cpu_bandwidth_config(opts, cfg), EXIT_SUCCESS);
  EXPECT_TRUE(cfg.user_specified_seed);
  EXPECT_EQ(cfg.seed, 0ULL);
}

TEST_F(CpuBandwidthConfigTest, DirectAndCollectPreparationInterruptionHandling) {
  ConfigTempDirGuard tmp("cpu_bw_prep_int");
  CpuBandwidthCommandTestHooks cmd_hooks;
  cmd_hooks.stop_requested = []() { return true; };
  set_cpu_bandwidth_command_test_hooks(&cmd_hooks);

  // 1. Direct file output (`-o <file>`) must emit status="interrupted", exit 0.
  const std::string file_path = (tmp.dir / "prep_int.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "0", "-k", "16", "-i",
                      "1", "-r", "1", "-o", file_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(args.argc(), args.argv()), EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Interrupted by user"), std::string::npos);
    std::ifstream ifs(file_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "interrupted");
    EXPECT_EQ(doc["reason_code"], "interruption-requested");
    EXPECT_TRUE(doc["interruption_requested"].get<bool>());
    EXPECT_FALSE(doc["results_complete"].get<bool>());
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
  }

  // 2. Direct stdout output (`-o -`) must emit JSON with status="interrupted", exit 0.
  {
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "0", "-k", "16", "-i",
                      "1", "-r", "1", "-o", "-"});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(args.argc(), args.argv()), EXIT_SUCCESS);
    EXPECT_NE(cap.err.str().find("Interrupted by user"), std::string::npos);
    const auto doc = nlohmann::ordered_json::parse(cap.out.str());
    EXPECT_EQ(doc["status"], "interrupted");
    EXPECT_EQ(doc["reason_code"], "interruption-requested");
    EXPECT_TRUE(doc["interruption_requested"].get<bool>());
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
  }

  // 3. Collect mode (`run_cpu_bandwidth_collect`) sets preparation_interrupted=true, exit 0.
  {
    CpuBandwidthConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    cfg.seed = 42ULL;
    cfg.collect_json_for_sweep = true;

    nlohmann::ordered_json collected;
    std::string failure_reason;
    bool prep_interrupted = false;
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_collect(cfg, collected, &failure_reason,
                                        &prep_interrupted),
              EXIT_SUCCESS);
    EXPECT_TRUE(prep_interrupted);
    EXPECT_EQ(failure_reason, "preparation-interrupted");
    EXPECT_TRUE(collected.is_null());
  }
}

TEST_F(CpuBandwidthConfigTest, ExecuteRunSetupFailuresAndCommandPaths) {
  ConfigTempDirGuard tmp("cpu_bw_cmd_paths");

  // 1. Help and invalid CLI via run_cpu_bandwidth_mode.
  {
    ArgvBuilder help_args({"memory_benchmark", "-B", "-h"});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(help_args.argc(), help_args.argv()),
              EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);

    ArgvBuilder bad_args({"memory_benchmark", "-B", "--unknown-flag"});
    StreamCapture bad_cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(bad_args.argc(), bad_args.argv()),
              EXIT_FAILURE);
    EXPECT_FALSE(bad_cap.err.str().empty());
  }

  // 2. Work plan rejection (`-b 999999999`) via collect & mode.
  {
    CpuBandwidthConfig cfg;
    cfg.buffer_size_mb = 999999999ULL;
    cfg.custom_cache_size_kb = 0;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "memory-budget-exceeded");
    EXPECT_EQ(collected["status"], "failed");
    EXPECT_FALSE(collected["run_accepted"].get<bool>());
  }

  // 3. Buffer allocation & mprotect failure in run_cpu_bandwidth_mode.
  {
    MemorySystemCalls mmap_fail;
    mmap_fail.map = bw_fail_mmap;
    set_memory_system_calls_for_testing(mmap_fail);
    const std::string mmap_path = (tmp.dir / "mmap_fail.json").string();
    ArgvBuilder args({"memory_benchmark", "-B", "-b", "0", "-k", "16", "-i",
                      "1", "-r", "1", "-o", mmap_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(args.argc(), args.argv()), EXIT_FAILURE);
    std::ifstream ifs(mmap_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_EQ(doc["reason_code"], "buffer-allocation-failed");
    reset_memory_system_calls_for_testing();

    MemorySystemCalls prot_fail;
    prot_fail.protect = bw_fail_mprotect;
    set_memory_system_calls_for_testing(prot_fail);
    const std::string prot_path = (tmp.dir / "prot_fail.json").string();
    ArgvBuilder prot_args({"memory_benchmark", "-B", "-b", "0", "-k", "16", "-i",
                           "1", "-r", "1", "-o", prot_path.c_str()});
    StreamCapture prot_cap;
    EXPECT_EQ(run_cpu_bandwidth_mode(prot_args.argc(), prot_args.argv()),
              EXIT_FAILURE);
    std::ifstream pifs(prot_path);
    ASSERT_TRUE(pifs.is_open());
    const auto pdoc = nlohmann::ordered_json::parse(pifs);
    EXPECT_EQ(pdoc["reason_code"],
              CpuBandwidthExecutorReason::SRC_READONLY_FAILED);
    reset_memory_system_calls_for_testing();
  }

  // 4. Command hooks (before_runner throwing exception & after_initialized_runner).
  {
    CpuBandwidthCommandTestHooks hooks;
    bool after_called = false;
    hooks.before_runner = [](CpuBandwidthWorkPlan&) {
      throw std::runtime_error("injected-before-runner");
    };
    hooks.after_initialized_runner = [&]() { after_called = true; };
    set_cpu_bandwidth_command_test_hooks(&hooks);

    CpuBandwidthConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_cpu_bandwidth_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "runner-exception");
    EXPECT_FALSE(after_called);
  }
}

TEST_F(CpuBandwidthConfigTest, ConsoleOutputRendersConfigSummaryAndWarnings) {
  CpuBandwidthConfig cfg;
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.requested_workers = 2;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.loop_count = 1;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;

  CpuBandwidthWorkPlan plan;
  plan.valid = true;
  plan.active_targets = {CpuBandwidthTarget::Custom};
  auto& tplan = plan.targets[cpu_bandwidth_target_index(CpuBandwidthTarget::Custom)];
  tplan.target = CpuBandwidthTarget::Custom;
  tplan.enabled = true;
  tplan.bytes = 16384;
  tplan.effective_workers = 2;

  CpuBandwidthResourceSet res;
  auto& tbuf = res.targets[cpu_bandwidth_target_index(CpuBandwidthTarget::Custom)];
  tbuf.active = true;
  tbuf.src.madvise_applied = false;

  CpuBandwidthResultMetadata meta;
  meta.processor_name = "Test Apple Silicon";
  meta.performance_core_count = 4;
  meta.efficiency_core_count = 4;
  meta.logical_core_count = 8;
  meta.main_thread_qos = {true, false, 1};
  meta.environment_start.thermal_state = "serious";
  meta.environment_end = meta.environment_start;

  CpuBandwidthResult result;
  result.status = CpuBandwidthRunStatus::Partial;
  result.counters.planned_measurements = 3;
  result.counters.measured_measurements = 1;
  auto& read_agg =
      result.aggregates[cpu_bandwidth_target_index(CpuBandwidthTarget::Custom)][0];
  read_agg.value = 50.25;
  read_agg.statistics.min = 49.0;
  read_agg.statistics.max = 51.5;
  read_agg.statistics.coefficient_of_variation_defined = true;
  read_agg.statistics.coefficient_of_variation_pct = 2.5;
  read_agg.statistics.median_absolute_deviation = 0.5;
  read_agg.statistics.sample_count = 2;

  std::ostringstream help_os;
  print_cpu_bandwidth_help(nullptr, help_os);
  EXPECT_NE(help_os.str().find("memory_benchmark -B"), std::string::npos);

  StreamCapture cap;
  print_cpu_bandwidth_configuration(cfg, plan, meta);
  print_cpu_bandwidth_console_summary(cfg, plan, res, meta, result);
  EXPECT_NE(cap.out.str().find("automatic_calibration"), std::string::npos);
  EXPECT_NE(cap.out.str().find("Custom Cache"), std::string::npos);
  EXPECT_NE(cap.out.str().find("P50=50.25 GB/s"), std::string::npos);
  EXPECT_NE(cap.out.str().find("Write: unavailable"), std::string::npos);
  EXPECT_NE(cap.err.str().find("environment-not-nominal"), std::string::npos);
  EXPECT_NE(cap.err.str().find("main-thread-qos-not-applied"), std::string::npos);
  EXPECT_NE(cap.err.str().find("madvise-random-not-applied"), std::string::npos);
}

}  // namespace
