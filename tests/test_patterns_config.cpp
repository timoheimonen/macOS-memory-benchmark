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
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/memory/memory_manager.h"
#include "output/json/json_output/json_output_session.h"
#include "pattern_benchmark/patterns.h"
#include "pattern_benchmark/patterns_executor.h"
#include "pattern_benchmark/patterns_output.h"
#include "pattern_benchmark/patterns_runner.h"
#include "pattern_benchmark/patterns_work_plan.h"

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

  void reset(const std::vector<const char*>& vec) {
    storage.clear();
    ptrs.clear();
    for (const char* s : vec) {
      storage.emplace_back(s);
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

class PatternsConfigTest : public ::testing::Test {
 protected:
  void TearDown() override {
    set_patterns_parser_test_hooks(nullptr);
    set_patterns_command_test_hooks(nullptr);
    reset_memory_system_calls_for_testing();
  }
};

struct ConfigTempDirGuard {
  std::filesystem::path dir;
  explicit ConfigTempDirGuard(const std::string& prefix) {
    dir = std::filesystem::temp_directory_path() /
          (prefix + "_" + std::to_string(::getpid()) + "_gtest");
    std::filesystem::create_directories(dir);
  }
  ~ConfigTempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

void* pat_fail_mmap(void*, size_t, int, int, int, off_t) {
  errno = ENOMEM;
  return MAP_FAILED;
}

int pat_fail_mprotect(void*, size_t, int) {
  errno = EACCES;
  return -1;
}

TEST_F(PatternsConfigTest, ParsesDefaultsAndAllWhitelistedOptions) {
  ArgvBuilder args({"memory_benchmark", "--patterns"});
  PatternsCliOptions opts;
  StreamCapture cap;
  EXPECT_EQ(parse_patterns_cli(args.argc(), args.argv(), opts), EXIT_SUCCESS);
  EXPECT_EQ(opts.buffer_size_mb, 512u);
  EXPECT_FALSE(opts.threads.has_value());
  EXPECT_FALSE(opts.iterations.has_value());
  EXPECT_EQ(opts.loop_count, 3u);
  EXPECT_FALSE(opts.non_cacheable);
  EXPECT_FALSE(opts.seed.has_value());
  EXPECT_TRUE(opts.output_file.empty());
  EXPECT_FALSE(opts.run_sweep);
  EXPECT_EQ(opts.sweep_max_runs, 256u);
  EXPECT_FALSE(opts.help_requested);
  EXPECT_FALSE(opts.user_specified_buffer_size);
  EXPECT_FALSE(opts.user_specified_threads);
  EXPECT_FALSE(opts.user_specified_iterations);
  EXPECT_FALSE(opts.user_specified_seed);

  ArgvBuilder full(
      {"memory_benchmark", "-P", "--buffer-size=16", "--threads=2",
       "--iterations=4", "--count=5", "--seed=0", "-u", "--output=out.json",
       "-X", "12"});
  PatternsCliOptions opts2;
  EXPECT_EQ(parse_patterns_cli(full.argc(), full.argv(), opts2), EXIT_SUCCESS);
  EXPECT_EQ(opts2.buffer_size_mb, 16u);
  EXPECT_TRUE(opts2.user_specified_buffer_size);
  ASSERT_TRUE(opts2.threads.has_value());
  EXPECT_EQ(*opts2.threads, 2u);
  EXPECT_TRUE(opts2.user_specified_threads);
  ASSERT_TRUE(opts2.iterations.has_value());
  EXPECT_EQ(*opts2.iterations, 4u);
  EXPECT_TRUE(opts2.user_specified_iterations);
  EXPECT_EQ(opts2.loop_count, 5u);
  EXPECT_TRUE(opts2.non_cacheable);
  ASSERT_TRUE(opts2.seed.has_value());
  EXPECT_EQ(*opts2.seed, 0ULL);
  EXPECT_TRUE(opts2.user_specified_seed);
  EXPECT_EQ(opts2.output_file, "out.json");
  EXPECT_EQ(opts2.sweep_max_runs, 12u);
}

TEST_F(PatternsConfigTest, HelpExitsZeroBeforeOsQueriesAndDoesNotCreateDash) {
  const std::vector<std::vector<const char*>> cases = {
      {"memory_benchmark", "-P", "-h"},
      {"memory_benchmark", "--patterns", "--help"},
      {"memory_benchmark", "-P", "-h", "-o", "-"},
      {"memory_benchmark", "--patterns", "--output=-", "--help"},
      {"memory_benchmark", "-P", "-h", "-S", "buffer-size=1,3"},
      {"memory_benchmark", "-P", "-h", "-b", "999999999"},
  };
  for (const auto& vec : cases) {
    ArgvBuilder args({});
    args.reset(vec);
    PatternsCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_patterns_cli(args.argc(), args.argv(), opts), EXIT_SUCCESS);
    EXPECT_TRUE(opts.help_requested);
    EXPECT_TRUE(cap.err.str().empty());
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);
    EXPECT_NE(cap.out.str().find("--patterns"), std::string::npos);
    EXPECT_EQ(cap.out.str().find("Running"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists("-"));
  }
}

TEST_F(PatternsConfigTest, RejectsSyntaxUnknownCrossModeAndLegacyFlags) {
  const std::vector<std::vector<const char*>> bad_cases = {
      {"memory_benchmark", "-P", "-b=16"},
      {"memory_benchmark", "-P", "-b", "16", "-b", "16"},
      {"memory_benchmark", "-P", "--unknown-option"},
      {"memory_benchmark", "-P", "--help=1"},
      {"memory_benchmark", "-P", "-b"},
      {"memory_benchmark", "-P", "-o"},
      {"memory_benchmark", "-P", "positional"},
      {"memory_benchmark", "-P", "-k", "64"},
      {"memory_benchmark", "-P", "-n", "10"},
      {"memory_benchmark", "-P", "-s", "64"},
      {"memory_benchmark", "-P", "-m", "auto"},
      {"memory_benchmark", "-P", "-l", "16"},
      {"memory_benchmark", "-P", "--compare-locality"},
      {"memory_benchmark", "-P", "-D", "low"},
      {"memory_benchmark", "-P", "--weight-size-mb", "1"},
      {"memory_benchmark", "-P", "-W"},
      {"memory_benchmark", "-P", "--benchmark"},
      {"memory_benchmark", "-P", "--only-bandwidth"},
      {"memory_benchmark", "-P", "--only-latency"},
  };
  for (const auto& vec : bad_cases) {
    ArgvBuilder args({});
    args.reset(vec);
    PatternsCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_patterns_cli(args.argc(), args.argv(), opts), EXIT_FAILURE);
    EXPECT_FALSE(cap.err.str().empty());
  }
}

TEST_F(PatternsConfigTest, RejectsInvalidNumericBounds) {
  const std::vector<std::vector<const char*>> bad_cases = {
      {"memory_benchmark", "-P", "-b", "0"},
      {"memory_benchmark", "-P", "--buffer-size", "0"},
      {"memory_benchmark", "-P", "--buffer-size=0"},
      {"memory_benchmark", "-P", "-b", "3="},
      {"memory_benchmark", "-P", "--buffer-size=3="},
      {"memory_benchmark", "-P", "-b", "-1"},
      {"memory_benchmark", "-P", "-b", "abc"},
      {"memory_benchmark", "-P", "-b", "9007199254740992"},
      {"memory_benchmark", "-P", "-i", "0"},
      {"memory_benchmark", "-P", "-i", "1000000001"},
      {"memory_benchmark", "-P", "-r", "0"},
      {"memory_benchmark", "-P", "-t", "0"},
      {"memory_benchmark", "-P", "-t", "4294967296"},
      {"memory_benchmark", "-P", "--seed", "-1"},
      {"memory_benchmark", "-P", "--seed", "+1"},
      {"memory_benchmark", "-P", "--seed", "18446744073709551616"},
      {"memory_benchmark", "-P", "-X", "0"},
      {"memory_benchmark", "-P", "-X", "abc"},
      {"memory_benchmark", "-P", "-X", "9007199254740992"},
  };
  for (const auto& vec : bad_cases) {
    ArgvBuilder args({});
    args.reset(vec);
    PatternsCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_patterns_cli(args.argc(), args.argv(), opts), EXIT_FAILURE);
    EXPECT_FALSE(cap.err.str().empty());
  }
}

TEST_F(PatternsConfigTest, RejectsInvalidSweepFlagsAndCartesianBudgetPreflight) {
  const std::vector<std::vector<const char*>> bad_parse = {
      {"memory_benchmark", "-P", "-S", "buffer-size=1,3"},
      {"memory_benchmark", "-P", "-S", "buffer-size=1,3", "-o", ""},
      {"memory_benchmark", "-P", "-b", "3", "-S", "buffer-size=1,3", "-o", "out.json"},
      {"memory_benchmark", "-P", "-t", "2", "-S", "threads=1,2", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=1", "-S", "buffer-size=3", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=1,3", "-S", "threads=1,2", "-X", "3", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=1,,3", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=1,2=", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "cache-size=32,64", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "latency-stride-bytes=64,256", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "tlb-density=low,high", "-o", "out.json"},
      {"memory_benchmark", "-P", "-S", "buffer-size=0,3", "-o", "out.json"},
  };
  for (const auto& vec : bad_parse) {
    ArgvBuilder args({});
    args.reset(vec);
    PatternsCliOptions opts;
    StreamCapture cap;
    EXPECT_EQ(parse_patterns_cli(args.argc(), args.argv(), opts), EXIT_FAILURE);
    EXPECT_FALSE(cap.err.str().empty());
  }

  // Cartesian combination exceeding memory budget must fail at build_patterns_config preflight.
  PatternsParserTestHooks hooks;
  hooks.detected_workers = 4;
  hooks.generated_seed = 987654321ULL;
  hooks.use_memory_override = true;
  hooks.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  set_patterns_parser_test_hooks(&hooks);

  ArgvBuilder overflow_sweep(
      {"memory_benchmark", "-P", "-S", "buffer-size=1,999999999", "-o",
       "out.json"});
  PatternsCliOptions opts;
  StreamCapture cap;
  ASSERT_EQ(
      parse_patterns_cli(overflow_sweep.argc(), overflow_sweep.argv(), opts),
      EXIT_SUCCESS);
  PatternsConfig cfg;
  EXPECT_EQ(build_patterns_config(opts, cfg), EXIT_FAILURE);
  EXPECT_NE(cap.err.str().find("memory-budget-exceeded"), std::string::npos);
}

TEST_F(PatternsConfigTest, ResolvesSeedWorkersAndSweepFlagsDeterministically) {
  PatternsParserTestHooks hooks;
  hooks.detected_workers = 8;
  hooks.generated_seed = 11223344ULL;
  hooks.use_memory_override = true;
  hooks.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  set_patterns_parser_test_hooks(&hooks);

  ArgvBuilder auto_args({"memory_benchmark", "-P", "-b", "16"});
  PatternsCliOptions opts;
  ASSERT_EQ(parse_patterns_cli(auto_args.argc(), auto_args.argv(), opts),
            EXIT_SUCCESS);
  PatternsConfig cfg;
  ASSERT_EQ(build_patterns_config(opts, cfg), EXIT_SUCCESS);
  EXPECT_EQ(cfg.available_workers, 8u);
  EXPECT_EQ(cfg.requested_workers, 8u);
  EXPECT_FALSE(cfg.user_specified_workers);
  EXPECT_FALSE(cfg.user_specified_seed);
  EXPECT_EQ(cfg.seed, 11223344ULL);

  const PatternsWorkPlanConfig wp_cfg = cfg.to_work_plan_config();
  EXPECT_EQ(wp_cfg.buffer_size_mb, 16u);
  EXPECT_EQ(wp_cfg.requested_workers, 8u);
  EXPECT_EQ(wp_cfg.available_workers, 8u);
  EXPECT_FALSE(wp_cfg.user_specified_workers);
  EXPECT_EQ(wp_cfg.seed, 11223344ULL);
  EXPECT_EQ(wp_cfg.loop_count, 3u);

  ArgvBuilder zero_seed(
      {"memory_benchmark", "-P", "-b", "16", "--seed", "0", "-S",
       "buffer-size=1,3", "-S", "threads=1,2", "-o", "out.json"});
  // Note: -b 16 conflicts with -S buffer-size, so test --seed 0 and sweep separately.
  ArgvBuilder sweep_zero_seed(
      {"memory_benchmark", "-P", "--seed", "0", "-S", "buffer-size=1,3", "-S",
       "threads=1,2", "-o", "out.json"});
  ASSERT_EQ(
      parse_patterns_cli(sweep_zero_seed.argc(), sweep_zero_seed.argv(), opts),
      EXIT_SUCCESS);
  StreamCapture cap;
  ASSERT_EQ(build_patterns_config(opts, cfg), EXIT_SUCCESS);
  EXPECT_TRUE(cfg.user_specified_seed);
  EXPECT_EQ(cfg.seed, 0ULL);
  EXPECT_TRUE(cfg.swept_buffer_size);
  EXPECT_TRUE(cfg.swept_threads);
  EXPECT_TRUE(cfg.user_specified_workers);
}

TEST_F(PatternsConfigTest, DirectAndCollectPreparationInterruptionHandling) {
  ConfigTempDirGuard tmp("pat_prep_int");
  PatternsCommandTestHooks cmd_hooks;
  cmd_hooks.stop_requested = []() { return true; };
  set_patterns_command_test_hooks(&cmd_hooks);

  // 1. Direct file output (`-o <file>`) must emit status="interrupted", exit 0.
  const std::string file_path = (tmp.dir / "prep_int.json").string();
  {
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "1", "-t", "1", "-i", "1",
                      "-r", "1", "-o", file_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_patterns_mode(args.argc(), args.argv()), EXIT_SUCCESS);
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
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "1", "-t", "1", "-i", "1",
                      "-r", "1", "-o", "-"});
    StreamCapture cap;
    EXPECT_EQ(run_patterns_mode(args.argc(), args.argv()), EXIT_SUCCESS);
    EXPECT_NE(cap.err.str().find("Interrupted by user"), std::string::npos);
    const auto doc = nlohmann::ordered_json::parse(cap.out.str());
    EXPECT_EQ(doc["status"], "interrupted");
    EXPECT_EQ(doc["reason_code"], "interruption-requested");
    EXPECT_TRUE(doc["interruption_requested"].get<bool>());
    EXPECT_FALSE(doc["run_accepted"].get<bool>());
  }

  // 3. Collect mode (`run_patterns_collect`) sets preparation_interrupted=true, exit 0.
  {
    PatternsConfig cfg;
    cfg.buffer_size_mb = 1;
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
    EXPECT_EQ(run_patterns_collect(cfg, collected, &failure_reason,
                                   &prep_interrupted),
              EXIT_SUCCESS);
    EXPECT_TRUE(prep_interrupted);
    EXPECT_EQ(failure_reason, "preparation-interrupted");
    EXPECT_TRUE(collected.is_null());
  }
}

TEST_F(PatternsConfigTest, ExecuteRunSetupFailuresAndCommandPaths) {
  ConfigTempDirGuard tmp("pat_cmd_paths");

  // 1. Help and invalid CLI via run_patterns_mode.
  {
    ArgvBuilder help_args({"memory_benchmark", "-P", "-h"});
    StreamCapture cap;
    EXPECT_EQ(run_patterns_mode(help_args.argc(), help_args.argv()),
              EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Usage:"), std::string::npos);

    ArgvBuilder bad_args({"memory_benchmark", "-P", "--unknown-flag"});
    StreamCapture bad_cap;
    EXPECT_EQ(run_patterns_mode(bad_args.argc(), bad_args.argv()),
              EXIT_FAILURE);
    EXPECT_FALSE(bad_cap.err.str().empty());
  }

  // 2. Work plan rejection (`-b 999999999`) via collect & mode.
  {
    PatternsConfig cfg;
    cfg.buffer_size_mb = 999999999ULL;
    cfg.iterations = 1;
    cfg.user_specified_iterations = true;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_patterns_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "memory-budget-exceeded");
    EXPECT_EQ(collected["status"], "failed");
    EXPECT_FALSE(collected["run_accepted"].get<bool>());
    EXPECT_FALSE(collected["memory_budget"]["valid"].get<bool>());
  }

  // 3. Buffer allocation & mprotect failure in run_patterns_mode.
  {
    MemorySystemCalls mmap_fail;
    mmap_fail.map = pat_fail_mmap;
    set_memory_system_calls_for_testing(mmap_fail);
    const std::string mmap_path = (tmp.dir / "mmap_fail.json").string();
    ArgvBuilder args({"memory_benchmark", "-P", "-b", "1", "-t", "1", "-i", "1",
                      "-r", "1", "-o", mmap_path.c_str()});
    StreamCapture cap;
    EXPECT_EQ(run_patterns_mode(args.argc(), args.argv()), EXIT_FAILURE);
    std::ifstream ifs(mmap_path);
    ASSERT_TRUE(ifs.is_open());
    const auto doc = nlohmann::ordered_json::parse(ifs);
    EXPECT_EQ(doc["status"], "failed");
    EXPECT_EQ(doc["reason_code"], "buffer-allocation-failed");
    reset_memory_system_calls_for_testing();

    MemorySystemCalls prot_fail;
    prot_fail.protect = pat_fail_mprotect;
    set_memory_system_calls_for_testing(prot_fail);
    const std::string prot_path = (tmp.dir / "prot_fail.json").string();
    ArgvBuilder prot_args({"memory_benchmark", "-P", "-b", "1", "-t", "1", "-i",
                           "1", "-r", "1", "-o", prot_path.c_str()});
    StreamCapture prot_cap;
    EXPECT_EQ(run_patterns_mode(prot_args.argc(), prot_args.argv()),
              EXIT_FAILURE);
    std::ifstream pifs(prot_path);
    ASSERT_TRUE(pifs.is_open());
    const auto pdoc = nlohmann::ordered_json::parse(pifs);
    EXPECT_EQ(pdoc["reason_code"], PatternsExecutorReason::SRC_READONLY_FAILED);
    reset_memory_system_calls_for_testing();
  }

  // 4. Command hooks (before_runner throwing exception & after_initialized_runner).
  {
    PatternsCommandTestHooks hooks;
    bool after_called = false;
    hooks.before_runner = [](PatternsWorkPlan&) {
      throw std::runtime_error("injected-before-runner");
    };
    hooks.after_initialized_runner = [&]() { after_called = true; };
    set_patterns_command_test_hooks(&hooks);

    PatternsConfig cfg;
    cfg.buffer_size_mb = 1;
    cfg.iterations = 1;
    cfg.loop_count = 1;
    cfg.requested_workers = 1;
    cfg.available_workers = 1;
    nlohmann::ordered_json collected;
    std::string reason;
    StreamCapture cap;
    EXPECT_EQ(run_patterns_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, "runner-exception");
    EXPECT_FALSE(after_called);
    set_patterns_command_test_hooks(nullptr);
  }

  // 5. Direct `-P -o ""` disables JSON output cleanly (exit 0), while `-b 9007199254740991`
  //    parses cleanly at CLI and rejects at work-plan with `buffer-bytes-overflow`.
  {
    ArgvBuilder empty_out({"memory_benchmark", "-P", "-b", "1", "-t", "1", "-i",
                           "1", "-r", "1", "-o", ""});
    StreamCapture cap;
    EXPECT_EQ(run_patterns_mode(empty_out.argc(), empty_out.argv()),
              EXIT_SUCCESS);
    EXPECT_NE(cap.out.str().find("Sequential Forward"), std::string::npos);
    EXPECT_EQ(cap.out.str().find("\"schema_version\""), std::string::npos);

    ArgvBuilder max_safe({"memory_benchmark", "-P", "-b", "9007199254740991",
                          "-i", "1", "-r", "1"});
    PatternsCliOptions opts;
    StreamCapture parse_cap;
    ASSERT_EQ(parse_patterns_cli(max_safe.argc(), max_safe.argv(), opts),
              EXIT_SUCCESS);
    EXPECT_EQ(opts.buffer_size_mb, 9007199254740991ULL);

    PatternsConfig cfg;
    ASSERT_EQ(build_patterns_config(opts, cfg), EXIT_SUCCESS);
    nlohmann::ordered_json collected;
    std::string reason;
    EXPECT_EQ(run_patterns_collect(cfg, collected, &reason), EXIT_FAILURE);
    EXPECT_EQ(reason, PatternsWorkPlanReason::BUFFER_BYTES_OVERFLOW);
    EXPECT_EQ(collected["status"], "failed");
    EXPECT_EQ(collected["reason_code"],
              PatternsWorkPlanReason::BUFFER_BYTES_OVERFLOW);
    EXPECT_FALSE(collected["resolved_plan"]["valid"].get<bool>());
    EXPECT_FALSE(collected["run_accepted"].get<bool>());
  }
}

TEST_F(PatternsConfigTest, ConsoleOutputRendersConfigSummaryAndWarnings) {
  PatternsConfig cfg;
  cfg.buffer_size_mb = 1;
  cfg.requested_workers = 2;
  cfg.available_workers = 4;
  cfg.user_specified_workers = true;
  cfg.loop_count = 1;
  cfg.non_cacheable = true;
  cfg.seed = 424242ULL;
  cfg.user_specified_seed = true;
  const PatternsWorkPlanConfig wp_cfg = cfg.to_work_plan_config();

  PatternsWorkPlan plan;
  plan.valid = true;
  plan.buffer_bytes = 1024ULL * 1024ULL;
  plan.active_patterns = {PatternKind::SequentialForward};
  auto& fwd_plan =
      plan.patterns[patterns_kind_index(PatternKind::SequentialForward)];
  fwd_plan.kind = PatternKind::SequentialForward;
  fwd_plan.enabled = true;
  fwd_plan.status = "active";
  fwd_plan.effective_workers = 2;
  auto& s2m_plan = plan.patterns[patterns_kind_index(PatternKind::Strided2MiB)];
  s2m_plan.kind = PatternKind::Strided2MiB;
  s2m_plan.enabled = false;
  s2m_plan.status = "skipped";
  s2m_plan.reason_code = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;
  s2m_plan.buffer_bytes = plan.buffer_bytes;

  PatternsResourceSet res;
  res.src.mapped_bytes = plan.buffer_bytes;
  res.dst.mapped_bytes = plan.buffer_bytes;
  res.src.madvise_applied = false;
  res.dst.madvise_applied = true;

  PatternsResultMetadata meta;
  meta.processor_name = "Test Apple Silicon";
  meta.performance_core_count = 4;
  meta.efficiency_core_count = 4;
  meta.logical_core_count = 8;
  meta.user_specified_seed = cfg.user_specified_seed;
  meta.main_thread_qos = {true, false, 1};
  meta.environment_start.thermal_state = "serious";
  meta.environment_end = meta.environment_start;

  PatternsResult result;
  result.status = PatternsRunStatus::Partial;
  result.counters.planned_measurements = 18;
  result.counters.measured_measurements = 1;
  auto& fwd_read =
      result.aggregates[patterns_kind_index(PatternKind::SequentialForward)][0];
  fwd_read.status = PatternsAggregateStatus::Measured;
  fwd_read.value = 50.25;
  fwd_read.relative_to_forward_pct = 100.0;
  fwd_read.statistics.min = 49.0;
  fwd_read.statistics.max = 51.5;
  fwd_read.statistics.coefficient_of_variation_defined = true;
  fwd_read.statistics.coefficient_of_variation_pct = 2.5;
  fwd_read.statistics.median_absolute_deviation = 0.5;
  fwd_read.statistics.sample_count = 2;

  auto& s2m_read =
      result.aggregates[patterns_kind_index(PatternKind::Strided2MiB)][0];
  s2m_read.status = PatternsAggregateStatus::Skipped;
  s2m_read.reason_code = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;

  std::ostringstream help_os;
  print_patterns_help(nullptr, help_os);
  EXPECT_NE(help_os.str().find("memory_benchmark -P"), std::string::npos);

  StreamCapture cap;
  print_patterns_configuration(wp_cfg, plan, meta);
  print_patterns_console_summary(wp_cfg, plan, res, meta, result);
  EXPECT_NE(cap.out.str().find("automatic_calibration"), std::string::npos);
  EXPECT_NE(cap.out.str().find("424242 (user)"), std::string::npos);
  EXPECT_NE(cap.out.str().find("skipped (buffer-too-small-for-stride-transition)"),
            std::string::npos);
  EXPECT_NE(cap.out.str().find("P50=50.25 GB/s (100.0% vs forward)"),
            std::string::npos);
  EXPECT_NE(cap.out.str().find("Write: unavailable"), std::string::npos);
  EXPECT_NE(cap.err.str().find("environment-not-nominal"), std::string::npos);
  EXPECT_NE(cap.err.str().find("main-thread-qos-not-applied"), std::string::npos);
  EXPECT_NE(cap.err.str().find("madvise-random-not-applied"), std::string::npos);
}

}  // namespace
