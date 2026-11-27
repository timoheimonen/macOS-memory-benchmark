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
#include "cpu_latency/cpu_latency.h"
#include "cpu_latency/cpu_latency_executor.h"
#include "cpu_latency/cpu_latency_output.h"
#include "cpu_latency/cpu_latency_runner.h"
#include "cpu_latency/cpu_latency_work_plan.h"

namespace {

struct ArgvBuilder {
  std::vector<std::string> storage;
  std::vector<char*> ptrs;

  ArgvBuilder() = default;
  explicit ArgvBuilder(std::initializer_list<const char*> args) {
    for (const char* a : args) {
      storage.emplace_back(a);
    }
    rebuild();
  }

  void rebuild() {
    ptrs.clear();
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

class CpuLatencyConfigTest : public ::testing::Test {
 protected:
  void TearDown() override {
    set_cpu_latency_parser_test_hooks(nullptr);
    set_cpu_latency_command_test_hooks(nullptr);
    reset_memory_system_calls_for_testing();
  }
};

struct LatencyConfigTempDirGuard {
  std::filesystem::path dir;
  explicit LatencyConfigTempDirGuard(const std::string& prefix) {
    dir = std::filesystem::temp_directory_path() / (prefix + "_gtest");
    std::filesystem::create_directories(dir);
  }
  ~LatencyConfigTempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

void* lat_fail_mmap(void*, size_t, int, int, int, off_t) {
  errno = ENOMEM;
  return MAP_FAILED;
}

int lat_fail_mprotect(void*, size_t, int) {
  errno = EACCES;
  return -1;
}

TEST_F(CpuLatencyConfigTest, ParsesDefaultsAndAllWhitelistedOptions) {
  ArgvBuilder args({"memory_benchmark", "--cpu-latency"});
  CpuLatencyCliOptions opts;
  StreamCapture cap;
  EXPECT_EQ(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
            EXIT_SUCCESS);
  EXPECT_EQ(opts.buffer_size_mb, 512u);
  EXPECT_FALSE(opts.custom_cache_size_kb.has_value());
  EXPECT_FALSE(opts.iterations.has_value());
  EXPECT_EQ(opts.loop_count, 3u);
  EXPECT_EQ(opts.latency_samples, 1000u);
  EXPECT_EQ(opts.latency_stride_bytes, 256u);
  EXPECT_EQ(opts.latency_chain_mode, LatencyChainMode::Auto);
  EXPECT_EQ(opts.latency_tlb_locality_kb, 1024u);
  EXPECT_FALSE(opts.compare_locality);
  EXPECT_FALSE(opts.non_cacheable);
  EXPECT_FALSE(opts.seed.has_value());
  EXPECT_TRUE(opts.output_file.empty());
  EXPECT_FALSE(opts.run_sweep);
  EXPECT_EQ(opts.sweep_max_runs, 256u);
  EXPECT_FALSE(opts.help_requested);

  ArgvBuilder full(
      {"memory_benchmark", "-L", "--buffer-size=16", "--cache-size=64",
       "--iterations=4", "--count=5", "--latency-samples=0",
       "--latency-stride-bytes=64", "--latency-chain-mode=same-random-in-box",
       "--latency-tlb-locality-kb=16", "--compare-locality", "--seed=0", "-u",
       "--output=out.json", "-X", "12"});
  CpuLatencyCliOptions opts2;
  EXPECT_EQ(parse_cpu_latency_cli(full.argc(), full.argv(), opts2),
            EXIT_SUCCESS);
  EXPECT_EQ(opts2.buffer_size_mb, 16u);
  ASSERT_TRUE(opts2.custom_cache_size_kb.has_value());
  EXPECT_EQ(*opts2.custom_cache_size_kb, 64u);
  ASSERT_TRUE(opts2.iterations.has_value());
  EXPECT_EQ(*opts2.iterations, 4u);
  EXPECT_EQ(opts2.loop_count, 5u);
  EXPECT_EQ(opts2.latency_samples, 0u);
  EXPECT_EQ(opts2.latency_stride_bytes, 64u);
  EXPECT_EQ(opts2.latency_chain_mode,
            LatencyChainMode::SameRandomInBoxIncreasingBox);
  EXPECT_EQ(opts2.latency_tlb_locality_kb, 16u);
  EXPECT_TRUE(opts2.compare_locality);
  EXPECT_TRUE(opts2.non_cacheable);
  ASSERT_TRUE(opts2.seed.has_value());
  EXPECT_EQ(*opts2.seed, 0ULL);
  EXPECT_EQ(opts2.output_file, "out.json");
  EXPECT_EQ(opts2.sweep_max_runs, 12u);
}

TEST_F(CpuLatencyConfigTest, HelpExitsZeroBeforeOsQueriesAndDoesNotCreateDash) {
  const std::vector<std::vector<const char*>> cases = {
      {"memory_benchmark", "-L", "-h"},
      {"memory_benchmark", "--cpu-latency", "--help"},
      {"memory_benchmark", "-L", "-h", "-o", "-"},
      {"memory_benchmark", "--cpu-latency", "--output=-", "--help"},
      {"memory_benchmark", "-L", "-b", "0", "-k", "0", "-h"},
  };
  for (const auto& vec : cases) {
    ArgvBuilder args;
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.rebuild();

    CpuLatencyCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS);
    EXPECT_TRUE(opts.help_requested);
    EXPECT_TRUE(cap.err.str().empty());
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);
    EXPECT_NE(cap.out.str().find("--cpu-latency"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists("-"));
  }
}

TEST_F(CpuLatencyConfigTest, RejectsInvalidSyntaxCrossModeBoundsAndLocality) {
  const std::vector<std::vector<const char*>> bad_cli = {
      {"memory_benchmark", "-L", "-t", "2"},
      {"memory_benchmark", "-L", "--threads", "2"},
      {"memory_benchmark", "-L", "--benchmark"},
      {"memory_benchmark", "-L", "-W"},
      {"memory_benchmark", "-L", "--only-latency"},
      {"memory_benchmark", "-L", "--unknown-option"},
      {"memory_benchmark", "-L", "-b=16"},
      {"memory_benchmark", "-L", "-b", "16", "-b", "16"},
      {"memory_benchmark", "-L", "-n", "10", "-n", "20"},
      {"memory_benchmark", "-L", "-s"},
      {"memory_benchmark", "-L", "-b", "-1"},
      {"memory_benchmark", "-L", "-b", "abc"},
      {"memory_benchmark", "-L", "-k", "15"},
      {"memory_benchmark", "-L", "-k", "1048577"},
      {"memory_benchmark", "-L", "-i", "0"},
      {"memory_benchmark", "-L", "-i", "1000000001"},
      {"memory_benchmark", "-L", "-r", "0"},
      {"memory_benchmark", "-L", "--seed", "-1"},
      {"memory_benchmark", "-L", "--seed", "+1"},
      {"memory_benchmark", "-L", "-n", "-1"},
      {"memory_benchmark", "-L", "-s", "0"},
      {"memory_benchmark", "-L", "-s", "65"},
      {"memory_benchmark", "-L", "-m", "nonexistent-mode"},
      {"memory_benchmark", "-L", "-m", "random-box", "-l", "0"},
      {"memory_benchmark", "-L", "-l", "16", "-s", "16384"},
      {"memory_benchmark", "-L", "-b", "0", "-k", "0"},
      {"memory_benchmark", "-L", "--compare-locality", "-b", "0", "-k", "64"},
      {"memory_benchmark", "-L", "--compare-locality", "-b", "16", "-s",
       "16384"},
  };

  for (const auto& vec : bad_cli) {
    ArgvBuilder args;
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.rebuild();

    CpuLatencyCliOptions opts;
    StreamCapture cap;
    EXPECT_NE(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS)
        << "Expected failure for " << vec[2];
    EXPECT_FALSE(cap.err.str().empty());
  }
}

TEST_F(CpuLatencyConfigTest, BuildConfigRejectsNonPageMultipleLocality) {
  CpuLatencyParserTestHooks hooks;
  hooks.page_size_bytes = 16384;
  set_cpu_latency_parser_test_hooks(&hooks);

  ArgvBuilder args({"memory_benchmark", "-L", "-l", "1", "-b", "16"});
  CpuLatencyCliOptions opts;
  StreamCapture cap;
  ASSERT_EQ(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
            EXIT_SUCCESS);

  CpuLatencyConfig cfg;
  EXPECT_NE(build_cpu_latency_config(opts, cfg), EXIT_SUCCESS);
  EXPECT_FALSE(cap.err.str().empty());
}

TEST_F(CpuLatencyConfigTest, ParsesAllFiveSweepKeysAndRejectsSweepErrors) {
  CpuLatencyParserTestHooks hooks;
  hooks.page_size_bytes = 16384;
  hooks.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  hooks.use_memory_override = true;
  set_cpu_latency_parser_test_hooks(&hooks);

  {
    ArgvBuilder args(
        {"memory_benchmark", "-L", "-o", "sweep.json", "-S", "buffer-size=8,16",
         "-S", "cache-size=32,64", "-S", "latency-tlb-locality-kb=16,32", "-S",
         "latency-stride-bytes=64,256", "-S",
         "latency-chain-mode=global-random,random-box", "-X", "64"});
    CpuLatencyCliOptions opts;
    StreamCapture cap;
    ASSERT_EQ(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS)
        << cap.err.str();
    EXPECT_EQ(opts.sweep_specs.size(), 5u);

    CpuLatencyConfig cfg;
    EXPECT_EQ(build_cpu_latency_config(opts, cfg), EXIT_SUCCESS)
        << cap.err.str();
    EXPECT_TRUE(cfg.swept_buffer_size);
    EXPECT_TRUE(cfg.swept_cache_size);
    EXPECT_TRUE(cfg.swept_tlb_locality_kb);
    EXPECT_TRUE(cfg.swept_stride_bytes);
    EXPECT_TRUE(cfg.swept_chain_mode);
  }

  const std::vector<std::vector<const char*>> bad_sweeps = {
      {"memory_benchmark", "-L", "-S", "buffer-size=8,16"},
      {"memory_benchmark", "-L", "-s", "64", "-S", "latency-stride-bytes=64,256",
       "-o", "s.json"},
      {"memory_benchmark", "-L", "-S", "cache-size=32", "-S", "cache-size=64",
       "-o", "s.json"},
      {"memory_benchmark", "-L", "-S", "buffer-size=8,16", "-S",
       "cache-size=32,64", "-X", "2", "-o", "s.json"},
      {"memory_benchmark", "-L", "-S", "cache-size=,32", "-o", "s.json"},
      {"memory_benchmark", "-L", "-S", "threads=1,2", "-o", "s.json"},
  };
  for (const auto& vec : bad_sweeps) {
    ArgvBuilder args;
    for (const char* s : vec) {
      args.storage.emplace_back(s);
    }
    args.rebuild();
    CpuLatencyCliOptions opts;
    StreamCapture cap;
    EXPECT_NE(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS);
  }

  // Cartesian target omission (`-b 0 -S cache-size=0,64`) rejected in build_cpu_latency_config.
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "0", "-S",
                      "cache-size=0,64", "-o", "s.json"});
    CpuLatencyCliOptions opts;
    StreamCapture cap;
    ASSERT_EQ(parse_cpu_latency_cli(args.argc(), args.argv(), opts),
              EXIT_SUCCESS);
    CpuLatencyConfig cfg;
    EXPECT_NE(build_cpu_latency_config(opts, cfg), EXIT_SUCCESS);
  }
}

TEST_F(CpuLatencyConfigTest, InvalidWorkPlanEmitsInspectableFailedJson) {
  LatencyConfigTempDirGuard tmp("cpu_lat_preflight_json");
  const std::string file_path = (tmp.dir / "invalid_plan.json").string();

  // 1. Direct file output (`-o <file>`) on !work_plan.valid emits status="failed" JSON & returns EXIT_FAILURE.
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "999999999", "-k", "0",
                      "-i", "1", "-r", "1", "-o", file_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_mode(args.argc(), args.argv()), EXIT_FAILURE);
    EXPECT_NE(cap.err.str().find("work-cap-exceeded"), std::string::npos);
    std::ifstream ifs(file_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["schema_version"], 1);
    EXPECT_EQ(doc["mode"], "cpu_latency");
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_EQ(doc["reason_code"], "work-cap-exceeded");
    EXPECT_FALSE(doc["results_complete"].get<bool>());
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
    EXPECT_FALSE(doc["resolved_plan"]["valid"].get<bool>());
    EXPECT_FALSE(doc["memory_budget"]["valid"].get<bool>());
  }

  // 2. Direct stdout output (`-o -`) on !work_plan.valid emits JSON to stdout.
  {
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "999999999", "-k", "0",
                      "-i", "1", "-r", "1", "-o", "-"});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_mode(args.argc(), args.argv()), EXIT_FAILURE);
    const auto doc = nlohmann::ordered_json::parse(cap.out.str());
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_EQ(doc["reason_code"], "work-cap-exceeded");
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
  }

  // 3. Collect mode on !work_plan.valid returns EXIT_FAILURE and populates collected_json.
  {
    CpuLatencyConfig cfg;
    cfg.buffer_size_mb = 999999999ULL;
    cfg.custom_cache_size_kb = 0;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.collect_json_for_sweep = true;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "work-cap-exceeded");
    EXPECT_EQ(collected["status"], "failed");
    EXPECT_FALSE(collected["run_accepted"].get<bool>());
  }
}

TEST_F(CpuLatencyConfigTest,
       DirectPreparationInterruptionAndFailurePathsInCommandMode) {
  LatencyConfigTempDirGuard tmp("cpu_lat_cmd_paths");

  // 1. Help and invalid CLI via run_cpu_latency_mode.
  {
    ArgvBuilder help_args({"memory_benchmark", "-L", "-h"});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_mode(help_args.argc(), help_args.argv()),
              EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);

    ArgvBuilder bad_args({"memory_benchmark", "-L", "--unknown-flag"});
    StreamCapture bad_cap;
    EXPECT_EQ(run_cpu_latency_mode(bad_args.argc(), bad_args.argv()),
              EXIT_FAILURE);
    EXPECT_FALSE(bad_cap.err.str().empty());
  }

  // 2. Direct & collect preparation interruption.
  {
    CpuLatencyCommandTestHooks hooks;
    hooks.stop_requested = []() { return true; };
    set_cpu_latency_command_test_hooks(&hooks);

    const std::string int_path = (tmp.dir / "prep_int.json").string();
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "0", "-k", "16", "-i",
                      "1", "-r", "1", "-n", "0", "-o", int_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_mode(args.argc(), args.argv()), EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Interrupted by user"), std::string::npos);
    std::ifstream ifs(int_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "interrupted");
    EXPECT_EQ(doc["reason_code"], "interruption-requested");
    EXPECT_TRUE(doc["interruption_requested"].get<bool>());

    CpuLatencyConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.loop_count = 1;
    cfg.latency_samples = 0;
    nlohmann::ordered_json collected;
    std::string reason;
    bool prep_int = false;
    EXPECT_EQ(run_cpu_latency_collect(cfg, collected, &reason, &prep_int),
              EXIT_SUCCESS);
    EXPECT_TRUE(prep_int);
    EXPECT_EQ(reason, "preparation-interrupted");
    set_cpu_latency_command_test_hooks(nullptr);
  }

  // 3. Allocation and mprotect failures.
  {
    MemorySystemCalls mmap_fail;
    mmap_fail.map = lat_fail_mmap;
    set_memory_system_calls_for_testing(mmap_fail);
    const std::string mmap_path = (tmp.dir / "mmap_fail.json").string();
    ArgvBuilder args({"memory_benchmark", "-L", "-b", "0", "-k", "16", "-i",
                      "1", "-r", "1", "-n", "0", "-o", mmap_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_mode(args.argc(), args.argv()), EXIT_FAILURE);
    std::ifstream ifs(mmap_path);
    ASSERT_TRUE(ifs.is_open());
    EXPECT_EQ(nlohmann::ordered_json::parse(ifs)["reason_code"],
              "buffer-allocation-failed");
    reset_memory_system_calls_for_testing();

    MemorySystemCalls prot_fail;
    prot_fail.protect = lat_fail_mprotect;
    set_memory_system_calls_for_testing(prot_fail);
    const std::string prot_path = (tmp.dir / "prot_fail.json").string();
    ArgvBuilder p_args({"memory_benchmark", "-L", "-b", "0", "-k", "16", "-i",
                        "1", "-r", "1", "-n", "0", "-o", prot_path.c_str()});
    StreamCapture p_cap;
    EXPECT_EQ(run_cpu_latency_mode(p_args.argc(), p_args.argv()), EXIT_FAILURE);
    std::ifstream pifs(prot_path);
    ASSERT_TRUE(pifs.is_open());
    EXPECT_EQ(nlohmann::ordered_json::parse(pifs)["reason_code"],
              CpuLatencyExecutorReason::CHAIN_READONLY_FAILED);
    reset_memory_system_calls_for_testing();
  }

  // 4. Command hook exception caught cleanly.
  {
    CpuLatencyCommandTestHooks hooks;
    hooks.before_runner = [](CpuLatencyWorkPlan&) {
      throw std::runtime_error("injected-lat-err");
    };
    set_cpu_latency_command_test_hooks(&hooks);
    CpuLatencyConfig cfg;
    cfg.buffer_size_mb = 0;
    cfg.custom_cache_size_kb = 16;
    cfg.iterations = 1;
    cfg.loop_count = 1;
    cfg.latency_samples = 0;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_cpu_latency_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "runner-exception");
  }
}

TEST_F(CpuLatencyConfigTest, ConsoleOutputRendersConfigSummaryAndWarnings) {
  CpuLatencyConfig cfg;
  cfg.buffer_size_mb = 0;
  cfg.custom_cache_size_kb = 16;
  cfg.loop_count = 1;
  cfg.latency_samples = 2;
  cfg.compare_locality = true;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;

  CpuLatencyWorkPlan plan;
  plan.valid = true;
  plan.effective_chain_mode = LatencyChainMode::RandomInBoxRandomBox;
  plan.active_targets = {CpuLatencyTarget::Custom, CpuLatencyTarget::MainMemory};
  auto& c_plan = plan.targets[cpu_latency_target_index(CpuLatencyTarget::Custom)];
  c_plan.target = CpuLatencyTarget::Custom;
  c_plan.enabled = true;
  c_plan.bytes = 16384;
  c_plan.node_count = 64;

  CpuLatencyResourceSet res;
  auto& c_buf = res.targets[cpu_latency_target_index(CpuLatencyTarget::Custom)];
  c_buf.active = true;
  c_buf.buffer.madvise_applied = false;

  CpuLatencyResultMetadata meta;
  meta.processor_name = "Test Apple Silicon";
  meta.performance_core_count = 4;
  meta.efficiency_core_count = 4;
  meta.logical_core_count = 8;
  meta.page_size_bytes = 16384;
  meta.main_thread_qos = {true, false, 1};
  meta.environment_start.thermal_state = "critical";
  meta.environment_end = meta.environment_start;

  CpuLatencyResult result;
  result.status = CpuLatencyRunStatus::Partial;
  result.counters.planned_measurements = 2;
  result.counters.measured_measurements = 1;
  auto& c_agg = result.aggregates[cpu_latency_target_index(CpuLatencyTarget::Custom)];
  c_agg.value = 3.5;
  c_agg.statistics.min = 3.2;
  c_agg.statistics.max = 3.8;
  c_agg.statistics.coefficient_of_variation_defined = true;
  c_agg.statistics.coefficient_of_variation_pct = 8.0;
  c_agg.statistics.median_absolute_deviation = 0.3;
  c_agg.statistics.sample_count = 2;
  c_agg.pooled_applicable = true;
  c_agg.pooled_samples_ns = {3.2, 3.8};
  c_agg.pooled_statistics.median = 3.5;
  c_agg.pooled_statistics.p90 = 3.74;
  c_agg.pooled_statistics.p99 = 3.79;
  c_agg.pooled_statistics.sample_count = 2;

  std::ostringstream help_os;
  print_cpu_latency_help(nullptr, help_os);
  EXPECT_NE(help_os.str().find("memory_benchmark -L"), std::string::npos);

  StreamCapture cap;
  print_cpu_latency_configuration(cfg, plan, meta);
  print_cpu_latency_console_summary(cfg, plan, res, meta, result);
  EXPECT_NE(cap.out.str().find("automatic_calibration"), std::string::npos);
  EXPECT_NE(cap.out.str().find("PooledSamples P50=3.50 ns"), std::string::npos);
  EXPECT_NE(cap.out.str().find("Main Memory : unavailable"), std::string::npos);
  EXPECT_NE(cap.out.str().find("Locality Comparison: unavailable"), std::string::npos);
  EXPECT_NE(cap.err.str().find("environment-not-nominal"), std::string::npos);
  EXPECT_NE(cap.err.str().find("main-thread-qos-not-applied"), std::string::npos);
  EXPECT_NE(cap.err.str().find("madvise-random-not-applied"), std::string::npos);

  result.locality_aggregate.locality_16k_latency_ns = 12.0;
  result.locality_aggregate.global_random_latency_ns = 45.0;
  result.locality_aggregate.locality_latency_delta_ns = 33.0;
  StreamCapture cap2;
  print_cpu_latency_console_summary(cfg, plan, res, meta, result);
  EXPECT_NE(cap2.out.str().find("16KB P50=12.00 ns"), std::string::npos);
}

}  // namespace
