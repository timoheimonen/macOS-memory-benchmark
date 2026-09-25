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
//
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

namespace {

class ScopedConfigTestHooks {
 public:
  ScopedConfigTestHooks() {
    hooks_.use_system_info = true;
    hooks_.cpu_name = "Injected Apple CPU";
    hooks_.macos_version = "15.5.1";
    hooks_.performance_cores = 6;
    hooks_.efficiency_cores = 4;
    hooks_.total_logical_cores = 10;
    hooks_.l1_cache_size = 128 * Constants::BYTES_PER_KB;
    hooks_.l2_cache_size = 4 * Constants::BYTES_PER_MB;
    hooks_.generated_seed = 0x123456789abcdef0ULL;
    hooks_.page_size_bytes = 16 * Constants::BYTES_PER_KB;
    set_config_test_hooks(&hooks_);
  }

  ~ScopedConfigTestHooks() { set_config_test_hooks(nullptr); }

  size_t page_size_bytes() const { return hooks_.page_size_bytes; }
  int total_logical_cores() const { return hooks_.total_logical_cores; }
  uint64_t generated_seed() const { return hooks_.generated_seed; }

 private:
  ConfigTestHooks hooks_;
};

ScopedConfigTestHooks scoped_config_test_hooks;

struct CapturedParseResult {
  int result = EXIT_FAILURE;
  std::string stderr_output;
};

CapturedParseResult parse_capturing_stderr(const std::vector<std::string>& arguments, BenchmarkConfig& config) {
  std::vector<std::string> mutable_arguments = arguments;
  std::vector<char*> argv;
  argv.reserve(mutable_arguments.size());
  for (std::string& argument : mutable_arguments) {
    argv.push_back(argument.data());
  }

  testing::internal::CaptureStderr();
  const int result = parse_arguments(static_cast<int>(argv.size()), argv.data(), config);
  return {result, testing::internal::GetCapturedStderr()};
}

std::string expected_invalid_value(const std::string& option, const std::string& value, const std::string& reason) {
  return Messages::error_prefix() + Messages::error_invalid_value(option, value, reason);
}

}  // namespace

TEST(ConfigTest, ParsesEquivalentLongAndShortOptions) {
  struct ValidAliasCase {
    const char* label;
    std::vector<std::string> arguments;
  };

  const ValidAliasCase cases[] = {
      {"long",
       {"program", "--benchmark", "--buffer-size", "1024", "--iterations", "500", "--count", "3", "--threads", "1",
        "--latency-samples", "17", "--output", "results.json", "--non-cacheable"}},
      {"short",
       {"program", "-B", "-b", "1024", "-i", "500", "-r", "3", "-t", "1", "-n", "17", "-o", "results.json", "-u"}},
  };

  for (const ValidAliasCase& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);

    ASSERT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(config.run_benchmark);
    EXPECT_FALSE(config.run_patterns);
    EXPECT_EQ(config.buffer_size_mb, 1024u);
    EXPECT_EQ(config.iterations, 500);
    EXPECT_EQ(config.loop_count, 3);
    EXPECT_EQ(config.num_threads, 1);
    EXPECT_TRUE(config.user_specified_threads);
    EXPECT_EQ(config.latency_sample_count, 17);
    EXPECT_TRUE(config.user_specified_latency_samples);
    EXPECT_EQ(config.output_file, "results.json");
    EXPECT_TRUE(config.use_non_cacheable);
  }
}

TEST(ConfigTest, ParseSweepValid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "--benchmark", "--output",         "sweep.json", "--sweep", "buffer-size=128,256",
                        "--sweep", "threads=1,2", "--sweep-max-runs", "4"};
  int argc = 10;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_sweep);
  ASSERT_EQ(config.sweep_specs.size(), 2u);
  EXPECT_EQ(config.sweep_specs[0].parameter, SweepParameter::BufferSizeMb);
  EXPECT_EQ(config.sweep_specs[0].values[0].integer_value, 128);
  EXPECT_EQ(config.sweep_specs[0].values[1].integer_value, 256);
  EXPECT_EQ(config.sweep_specs[1].parameter, SweepParameter::Threads);
  EXPECT_EQ(config.sweep_specs[1].values[0].integer_value, 1);
  EXPECT_EQ(config.sweep_specs[1].values[1].integer_value, 2);
  EXPECT_EQ(config.sweep_max_runs, 4u);
}

TEST(ConfigTest, ParseFlagShapedOutputTargetsRemainOpaqueAcrossGeneralModes) {
  struct ModeCase {
    const char* option;
    bool patterns;
  };

  const char* output_options[] = {"-o", "--output"};
  const char* output_targets[] = {"-", "./-", "-T", "--analyze-tlb", "-k", "--cache-size"};
  const ModeCase modes[] = {{"--benchmark", false}, {"--patterns", true}};

  size_t case_index = 0;
  for (const char* output_option : output_options) {
    for (const char* output_target : output_targets) {
      for (const ModeCase& mode : modes) {
        std::vector<std::string> arguments = {"program"};
        const bool output_first = (case_index++ / 2) % 2 == 0;
        if (output_first) {
          arguments.insert(arguments.end(), {output_option, output_target, "--non-cacheable", mode.option});
        } else {
          arguments.insert(arguments.end(), {mode.option, output_option, output_target, "--non-cacheable"});
        }
        SCOPED_TRACE(testing::PrintToString(arguments));

        BenchmarkConfig config;
        const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);

        EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
        if (parsed.result != EXIT_SUCCESS) {
          continue;
        }
        EXPECT_EQ(config.output_file, output_target);
        EXPECT_EQ(config.run_benchmark, !mode.patterns);
        EXPECT_EQ(config.run_patterns, mode.patterns);
        EXPECT_FALSE(config.analyze_tlb);
        EXPECT_FALSE(config.run_sweep);
        EXPECT_TRUE(config.use_non_cacheable);
        EXPECT_EQ(config.custom_cache_size_kb_ll, -1);
        EXPECT_FALSE(config.use_custom_cache_size);
        EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
      }
    }
  }
}

TEST(ConfigTest, ParseOutputPrescanSkipsExactlyOneFlagShapedValue) {
  struct BoundaryCase {
    const char* label;
    std::vector<std::string> arguments;
    const char* output_target;
    bool analyze_tlb;
    bool benchmark;
    long long cache_size_kb;
  };

  const BoundaryCase cases[] = {
      {"real TLB option after output value", {"program", "--output", "-T", "--analyze-tlb"}, "-T", true, false,
       -1},
      {"real TLB option before output value", {"program", "-T", "--output", "--analyze-tlb"}, "--analyze-tlb",
       true, false, -1},
      {"real cache option after output value",
       {"program", "--output", "-k", "--cache-size", "256", "--benchmark"}, "-k", false, true, 256},
      {"real cache option before output value",
       {"program", "-k", "256", "--benchmark", "--output", "--cache-size"}, "--cache-size", false, true, 256},
  };

  for (const BoundaryCase& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);

    EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    if (parsed.result != EXIT_SUCCESS) {
      continue;
    }
    EXPECT_EQ(config.output_file, test_case.output_target);
    EXPECT_EQ(config.analyze_tlb, test_case.analyze_tlb);
    EXPECT_EQ(config.run_benchmark, test_case.benchmark);
    EXPECT_FALSE(config.run_patterns);
    EXPECT_EQ(config.custom_cache_size_kb_ll, test_case.cache_size_kb);
    EXPECT_EQ(config.use_custom_cache_size, test_case.cache_size_kb >= 0);
    EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  }
}

TEST(ConfigTest, ParseMissingValuesUseNormalParserDiagnostics) {
  for (const char* option : {"--output", "--cache-size", "--iterations"}) {
    SCOPED_TRACE(option);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", "--benchmark", option}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output.find(Messages::error_prefix() + Messages::error_missing_value(option) + "\n"), 0u)
        << parsed.stderr_output;
  }
}

TEST(ConfigTest, RejectsMalformedNumericTokensWithCentralizedErrors) {
  struct InvalidNumericCase {
    std::vector<std::string> arguments;
    const char* option;
    const char* value;
    const char* reason;
  };

  constexpr const char* kInvalidSignedReason =
      "must be an integer without whitespace, a plus sign, or trailing characters";
  constexpr const char* kInvalidUnsignedReason =
      "must be an unsigned 64-bit integer without whitespace, a sign, or trailing characters";
  const InvalidNumericCase cases[] = {
      {{"program", "--iterations", "5x"}, "--iterations", "5x", kInvalidSignedReason},
      {{"program", "--buffer-size", "1junk"}, "--buffer-size", "1junk", kInvalidSignedReason},
      {{"program", "--count", "2x"}, "--count", "2x", kInvalidSignedReason},
      {{"program", "--latency-samples", "3x"}, "--latency-samples", "3x", kInvalidSignedReason},
      {{"program", "--latency-stride-bytes", "64x"}, "--latency-stride-bytes", "64x", kInvalidSignedReason},
      {{"program", "--latency-tlb-locality-kb", "16x"}, "--latency-tlb-locality-kb", "16x", kInvalidSignedReason},
      {{"program", "--cache-size", "16x"}, "--cache-size", "16x", kInvalidSignedReason},
      {{"program", "--threads", "1x"}, "--threads", "1x", kInvalidSignedReason},
      {{"program", "--sweep-max-runs", "4x"}, "--sweep-max-runs", "4x", kInvalidSignedReason},
      {{"program", "--benchmark", "--seed", "9x"}, "--seed", "9x", kInvalidUnsignedReason},
      {{"program", "--iterations", " 5"}, "--iterations", " 5", kInvalidSignedReason},
      {{"program", "--iterations", "5 "}, "--iterations", "5 ", kInvalidSignedReason},
      {{"program", "--iterations", "+5"}, "--iterations", "+5", kInvalidSignedReason},
      {{"program", "--benchmark", "--seed", "+5"}, "--seed", "+5", kInvalidUnsignedReason},
      {{"program", "--iterations", "9223372036854775808"}, "--iterations", "9223372036854775808", "out of range"},
      {{"program", "--benchmark", "--seed", "18446744073709551616"},
       "--seed",
       "18446744073709551616",
       "out of range for an unsigned 64-bit integer"},
      {{"program", "--analyze-tlb", "--latency-stride-bytes", "64x"},
       "--latency-stride-bytes",
       "64x",
       kInvalidSignedReason},
      {{"program", "--analyze-tlb", "--sweep-max-runs", "4x"}, "--sweep-max-runs", "4x", kInvalidSignedReason},
      {{"program", "--analyze-tlb", "--seed", "+5"}, "--seed", "+5", kInvalidUnsignedReason},
      {{"program", "--analyze-tlb", "--seed", "-1"}, "--seed", "-1", kInvalidUnsignedReason},
  };

  for (const InvalidNumericCase& test_case : cases) {
    SCOPED_TRACE(test_case.option);
    SCOPED_TRACE(test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_NE(parsed.stderr_output.find(expected_invalid_value(test_case.option, test_case.value, test_case.reason)),
              std::string::npos);
  }
}

TEST(ConfigTest, RejectsMalformedSweepListsAndNumericValues) {
  struct InvalidSweepCase {
    const char* specification;
    const char* reason;
    bool analyze_tlb;
  };

  constexpr const char* kInvalidSignedReason =
      "must be an integer without whitespace, a plus sign, or trailing characters";
  const InvalidSweepCase cases[] = {
      {"latency-samples=100,200", "unsupported sweep parameter: latency-samples", false},
      {"buffer-size=1,,2", "sweep value list cannot contain empty values", false},
      {"buffer-size=1x", kInvalidSignedReason, false},
      {"cache-size=16x", kInvalidSignedReason, false},
      {"threads=1x", kInvalidSignedReason, false},
      {"latency-tlb-locality-kb=16x", kInvalidSignedReason, false},
      {"latency-stride-bytes=64x", kInvalidSignedReason, false},
      {"latency-stride-bytes=64x", kInvalidSignedReason, true},
  };

  for (const InvalidSweepCase& test_case : cases) {
    SCOPED_TRACE(test_case.specification);
    std::vector<std::string> arguments = {"program"};
    arguments.push_back(test_case.analyze_tlb ? "--analyze-tlb" : "--benchmark");
    arguments.push_back("--sweep");
    arguments.push_back(test_case.specification);

    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_NE(parsed.stderr_output.find(expected_invalid_value("--sweep", test_case.specification, test_case.reason)),
              std::string::npos);
  }
}

TEST(ConfigTest, RejectsSweepValuesOutsideSemanticDomains) {
  struct InvalidSweepCase {
    std::string specification;
    std::string reason;
    bool analyze_tlb;
  };

  const long long max_locality_kb =
      static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
  const InvalidSweepCase cases[] = {
      {"buffer-size=-1", Messages::error_buffersize_invalid(-1, std::numeric_limits<unsigned long>::max()), false},
      {"cache-size=8",
       Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB,
                                          Constants::MAX_CACHE_SIZE_KB / 1024),
       false},
      {"threads=0", Messages::error_threads_invalid(0, 1, std::numeric_limits<int>::max()), false},
      {"latency-tlb-locality-kb=-1", Messages::error_latency_tlb_locality_invalid(-1, max_locality_kb), false},
      {"latency-stride-bytes=0", Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max()),
       false},
      {"latency-chain-mode=invalid", Messages::error_latency_chain_mode_invalid(), false},
      {"tlb-density=invalid", "must be one of: low, medium, high", true},
  };

  for (const InvalidSweepCase& test_case : cases) {
    SCOPED_TRACE(test_case.specification);
    std::vector<std::string> arguments = {"program", test_case.analyze_tlb ? "--analyze-tlb" : "--benchmark", "--sweep",
                                          test_case.specification};

    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_NE(parsed.stderr_output.find(expected_invalid_value("--sweep", test_case.specification, test_case.reason)),
              std::string::npos);
  }
}

TEST(ConfigTest, ValidateSweepRejectsInvalidConfigurationsWithExactDiagnostics) {
  struct SweepCase {
    const char* name;
    std::function<void(BenchmarkConfig&)> configure;
    std::string diagnostic;
  };
  const SweepCase cases[] = {
      {"run limit",
       [](BenchmarkConfig& config) {
         config.run_benchmark = true;
         config.sweep_max_runs = 3;
         config.sweep_specs = {{SweepParameter::BufferSizeMb, "buffer-size", {{"128"}, {"256"}}},
                               {SweepParameter::Threads, "threads", {{"1"}, {"2"}}}};
       },
       Messages::error_sweep_too_many_runs(4, 3)},
      {"pattern cache",
       [](BenchmarkConfig& config) {
         config.run_patterns = true;
         config.sweep_specs = {{SweepParameter::CacheSizeKb, "cache-size", {{"1024"}}}};
       },
       Messages::error_sweep_parameter_not_allowed("cache-size", "--patterns")},
      {"missing output",
       [](BenchmarkConfig& config) {
         config.run_benchmark = true;
         config.output_file.clear();
         config.sweep_specs = {{SweepParameter::BufferSizeMb, "buffer-size", {{"128"}}}};
       },
       Messages::error_sweep_requires_output()},
      {"TLB global random",
       [](BenchmarkConfig& config) {
         config.analyze_tlb = true;
         SweepValue value;
         value.raw_value = "global-random";
         value.latency_chain_mode = LatencyChainMode::GlobalRandom;
         config.sweep_specs = {{SweepParameter::LatencyChainMode, "latency-chain-mode", {value}}};
       },
       Messages::error_analyze_tlb_global_random_unsupported()},
      {"duplicate stride",
       [](BenchmarkConfig& config) {
         config.analyze_tlb = true;
         SweepValue first, second;
         first.integer_value = 64;
         second.integer_value = 128;
         config.sweep_specs = {{SweepParameter::LatencyStrideBytes, "latency-stride-bytes", {first}},
                               {SweepParameter::LatencyStrideBytes, "latency-stride-bytes", {second}}};
       },
       Messages::error_duplicate_sweep_parameter("latency-stride-bytes")},
  };
  for (const SweepCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    BenchmarkConfig config;
    config.run_sweep = true;
    config.output_file = "sweep.json";
    test_case.configure(config);
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(diagnostic, Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, ValidateAnalyzeTlbStrideBoundsForDirectAndSweepModes) {
  const size_t page_size = scoped_config_test_hooks.page_size_bytes();
  ASSERT_EQ(136u % sizeof(uintptr_t), 0u);
  ASSERT_NE(page_size % 136u, 0u);
  for (const size_t stride : {size_t{136}, page_size * 2}) {
    for (bool sweep : {false, true}) {
      SCOPED_TRACE(stride);
      SCOPED_TRACE(sweep);
      BenchmarkConfig config;
      config.analyze_tlb = true;
      config.run_sweep = sweep;
      if (sweep) {
        config.output_file = "sweep.json";
        SweepValue value;
        value.integer_value = static_cast<long long>(stride);
        config.sweep_specs = {{SweepParameter::LatencyStrideBytes, "latency-stride-bytes", {value}}};
      } else {
        config.latency_stride_bytes = stride;
      }
      testing::internal::CaptureStderr();
      const int result = validate_config(config);
      const std::string diagnostic = testing::internal::GetCapturedStderr();
      EXPECT_EQ(result, stride <= page_size ? EXIT_SUCCESS : EXIT_FAILURE);
      EXPECT_EQ(diagnostic, stride <= page_size
                                ? ""
                                : Messages::error_prefix() +
                                      Messages::error_analyze_tlb_stride_exceeds_page(stride, page_size) + "\n");
    }
  }
}

TEST(ConfigTest, ParseSemanticErrorsHaveExactDiagnostics) {
  struct InvalidCase {
    const char* option;
    const char* value;
    std::string reason;
  };
  const InvalidCase cases[] = {
      {"--cache-size", "8",
       Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB,
                                          Constants::MAX_CACHE_SIZE_KB / 1024)},
      {"--cache-size", "1100000",
       Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB,
                                          Constants::MAX_CACHE_SIZE_KB / 1024)},
      {"--latency-stride-bytes", "0",
       Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max())},
      {"--latency-chain-mode", "unknown-mode", Messages::error_latency_chain_mode_invalid()},
      {"--latency-tlb-locality-kb", "-1",
       Messages::error_latency_tlb_locality_invalid(
           -1, static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB))},
      {"--threads", "0", Messages::error_threads_invalid(0, 1, std::numeric_limits<int>::max())},
      {"--latency-samples", "0", Messages::error_latency_samples_invalid(0, 1, std::numeric_limits<int>::max())},
      {"-unknown", "value", ""},
  };
  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(test_case.option);
    SCOPED_TRACE(test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", test_case.option, test_case.value}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    const std::string diagnostic = test_case.reason.empty()
                                       ? Messages::error_prefix() + Messages::error_unknown_option(test_case.option)
                                       : expected_invalid_value(test_case.option, test_case.value, test_case.reason);
    EXPECT_EQ(parsed.stderr_output.find(diagnostic + "\n"), 0u) << parsed.stderr_output;
  }
}

// Test parsing cache size zero (validated later; allowed only with --only-latency)
TEST(ConfigTest, ParseExplicitMemoryAndLatencyOptions) {
  struct ValidCase {
    const char* option;
    const char* value;
    std::function<void(const BenchmarkConfig&)> verify;
  };
  const ValidCase cases[] = {
      {"--cache-size", "0",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.custom_cache_size_kb_ll, 0);
         EXPECT_TRUE(config.use_custom_cache_size);
       }},
      {"--buffer-size", "0", [](const BenchmarkConfig& config) { EXPECT_EQ(config.buffer_size_mb, 0u); }},
      {"--latency-tlb-locality-kb", "16",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_tlb_locality_bytes, 16u * Constants::BYTES_PER_KB);
       }},
      {"--latency-tlb-locality-kb", "0",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.latency_tlb_locality_bytes, 0u); }},
      {"--latency-stride-bytes", "64",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.latency_stride_bytes, 64u); }},
      {"--latency-chain-mode", "same-random-in-box",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::SameRandomInBoxIncreasingBox);
       }},
  };
  for (const ValidCase& test_case : cases) {
    SCOPED_TRACE(std::string(test_case.option) + " " + test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", test_case.option, test_case.value}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    test_case.verify(config);
  }
}

TEST(ConfigTest, ParseAnalyzeTlbStandaloneAppliesSafeDefaultsAndGeneratedSeed) {
  for (const char* option : {"-T", "--analyze-tlb"}) {
    SCOPED_TRACE(option);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", option}, config);

    EXPECT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(config.analyze_tlb);
    EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Medium);
    EXPECT_EQ(config.sweep_max_runs, Constants::DEFAULT_ANALYZE_TLB_SWEEP_MAX_RUNS);
    EXPECT_EQ(config.tlb_seed, scoped_config_test_hooks.generated_seed());
    EXPECT_FALSE(config.user_specified_tlb_seed);
    EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  }
}

TEST(ConfigTest, ParseAnalyzeTlbExplicitOptions) {
  struct ValidCase {
    const char* option;
    const char* value;
    std::function<void(const BenchmarkConfig&)> verify;
  };
  const ValidCase cases[] = {
      {"--sweep-max-runs", "24", [](const BenchmarkConfig& config) { EXPECT_EQ(config.sweep_max_runs, 24u); }},
      {"--seed", "18446744073709551615",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.tlb_seed, std::numeric_limits<uint64_t>::max());
         EXPECT_TRUE(config.user_specified_tlb_seed);
       }},
      {"--latency-stride-bytes", "128",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.latency_stride_bytes, 128u); }},
      {"--latency-chain-mode", "random-box",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::RandomInBoxRandomBox);
       }},
      {"--tlb-density", "low",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Low); }},
      {"--tlb-density", "medium",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Medium); }},
      {"--tlb-density", "high",
       [](const BenchmarkConfig& config) { EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::High); }},
  };
  for (const ValidCase& test_case : cases) {
    SCOPED_TRACE(std::string(test_case.option) + " " + test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--analyze-tlb", test_case.option, test_case.value}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.analyze_tlb);
    test_case.verify(config);
  }
}

TEST(ConfigTest, ParseAnalyzeTlbRejectsInvalidOptionsWithExactDiagnostics) {
  struct InvalidCase {
    std::vector<std::string> options;
    std::string diagnostic;
  };
  const InvalidCase cases[] = {
      {{"--buffer-size", "512"}, Messages::error_analyze_tlb_must_be_used_alone()},
      {{"--seed", "42", "--seed", "43"}, Messages::error_duplicate_option("--seed")},
      {{"--latency-chain-mode", "global-random"}, Messages::error_analyze_tlb_global_random_unsupported()},
      {{"--tlb-density", "ultra"},
       Messages::error_invalid_value("--tlb-density", "ultra", "must be one of: low, medium, high")},
      {{"--latency-stride-bytes", "0"},
       Messages::error_invalid_value(
           "--latency-stride-bytes", "0",
           Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max()))},
      {{"--latency-stride-bytes", "65"}, Messages::error_latency_stride_alignment(65, sizeof(void*))},
      {{"--output"}, Messages::error_missing_value("--output")},
  };
  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.options));
    std::vector<std::string> arguments = {"program", "--analyze-tlb"};
    arguments.insert(arguments.end(), test_case.options.begin(), test_case.options.end());
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output.find(Messages::error_prefix() + test_case.diagnostic + "\n"), 0u)
        << parsed.stderr_output;
  }
}

TEST(ConfigTest, ParseStandardAndPatternSeedsAndThreadCounts) {
  for (bool patterns : {false, true}) {
    for (bool explicit_seed : {false, true}) {
      for (bool explicit_threads : {false, true}) {
        SCOPED_TRACE(patterns);
        SCOPED_TRACE(explicit_seed);
        SCOPED_TRACE(explicit_threads);
        std::vector<std::string> arguments = {"program", patterns ? "--patterns" : "--benchmark"};
        if (explicit_seed) arguments.insert(arguments.end(), {"--seed", "18446744073709551615"});
        if (explicit_threads) arguments.insert(arguments.end(), {"--threads", "1"});
        BenchmarkConfig config;
        const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);
        ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
        EXPECT_TRUE(parsed.stderr_output.empty());
        EXPECT_EQ(patterns ? config.pattern_seed : config.benchmark_seed,
                  explicit_seed ? std::numeric_limits<uint64_t>::max() : scoped_config_test_hooks.generated_seed());
        EXPECT_EQ(config.user_specified_pattern_seed, patterns && explicit_seed);
        EXPECT_EQ(config.user_specified_benchmark_seed, !patterns && explicit_seed);
        EXPECT_EQ(config.num_threads, explicit_threads ? 1 : scoped_config_test_hooks.total_logical_cores());
        EXPECT_EQ(config.user_specified_threads, explicit_threads);
      }
    }
  }
}

TEST(ConfigTest, ParseDuplicateValueOptionsRejected) {
  struct DuplicateOptionCase {
    const char* option;
    const char* first_value;
    const char* second_value;
  };

  const DuplicateOptionCase cases[] = {
      {"--iterations", "1", "2"},
      {"--buffer-size", "1", "2"},
      {"--count", "1", "2"},
      {"--latency-samples", "1", "2"},
      {"--latency-stride-bytes", "64", "128"},
      {"--latency-chain-mode", "auto", "global-random"},
      {"--latency-tlb-locality-kb", "16", "32"},
      {"--threads", "1", "2"},
      {"--output", "-T", "second.json"},
      {"--cache-size", "256", "512"},
      {"--seed", "42", "43"},
  };

  for (const DuplicateOptionCase& test_case : cases) {
    for (const char* mode : {"--benchmark", "--patterns"}) {
      SCOPED_TRACE(test_case.option);
      SCOPED_TRACE(mode);
      BenchmarkConfig config;
      const CapturedParseResult parsed = parse_capturing_stderr(
          {"program", mode, test_case.option, test_case.first_value, test_case.option, test_case.second_value}, config);
      EXPECT_EQ(parsed.result, EXIT_FAILURE);
      EXPECT_EQ(parsed.stderr_output.find(Messages::error_prefix() +
                                          Messages::error_duplicate_option(test_case.option) + "\n"),
                0u)
          << parsed.stderr_output;
    }
  }
}

// Test parsing help flag
TEST(ConfigTest, ParseHelpFlag) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-h"};
  int argc = 2;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);  // Help returns SUCCESS
  EXPECT_TRUE(config.help_printed);
}

TEST(ConfigTest, ValidateParsedIncompatibleFlagsWithExactDiagnostics) {
  struct InvalidCase {
    std::vector<std::string> arguments;
    bool BenchmarkConfig::* first_flag;
    bool BenchmarkConfig::* second_flag;
    std::string diagnostic;
  };
  const InvalidCase cases[] = {
      {{"program", "--benchmark", "--only-bandwidth", "--latency-samples", "10"},
       &BenchmarkConfig::only_bandwidth,
       &BenchmarkConfig::user_specified_latency_samples,
       Messages::error_only_bandwidth_with_latency_samples()},
      {{"program", "--benchmark", "--only-bandwidth", "--cache-size", "256"},
       &BenchmarkConfig::only_bandwidth,
       &BenchmarkConfig::use_custom_cache_size,
       Messages::error_only_bandwidth_with_cache_size()},
      {{"program", "--benchmark", "--only-latency", "--iterations", "10"},
       &BenchmarkConfig::only_latency,
       &BenchmarkConfig::user_specified_iterations,
       Messages::error_only_latency_with_iterations()},
      {{"program", "--benchmark", "--only-bandwidth", "--only-latency"},
       &BenchmarkConfig::only_bandwidth,
       &BenchmarkConfig::only_latency,
       Messages::error_incompatible_flags()},
      {{"program", "--patterns", "--only-bandwidth"},
       &BenchmarkConfig::run_patterns,
       &BenchmarkConfig::only_bandwidth,
       Messages::error_only_flags_with_patterns()},
  };
  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(config.*test_case.first_flag);
    EXPECT_TRUE(config.*test_case.second_flag);
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(diagnostic, Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, CalculateBufferSizesMatchesExactStandardAndCustomCases) {
  struct BufferSizeCase {
    bool use_custom_cache_size;
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t custom_cache_size;
    size_t expected_l1_buffer_size;
    size_t expected_l2_buffer_size;
    size_t expected_custom_buffer_size;
  };

  constexpr size_t kL1Size = 128 * Constants::BYTES_PER_KB;
  constexpr size_t kL2Size = 4 * Constants::BYTES_PER_MB;
  constexpr size_t kCustomSize = 256 * Constants::BYTES_PER_KB;
  const BufferSizeCase cases[] = {
      {false, kL1Size, kL2Size, 0, kL1Size, kL2Size, 0},
      {true, 0, 0, kCustomSize, 0, 0, kCustomSize},
  };

  for (const BufferSizeCase& test_case : cases) {
    SCOPED_TRACE(test_case.use_custom_cache_size ? "custom" : "standard");
    BenchmarkConfig config;
    config.use_custom_cache_size = test_case.use_custom_cache_size;
    config.l1_cache_size = test_case.l1_cache_size;
    config.l2_cache_size = test_case.l2_cache_size;
    config.custom_cache_size_bytes = test_case.custom_cache_size;

    calculate_buffer_sizes(config);

    EXPECT_EQ(config.l1_buffer_size, test_case.expected_l1_buffer_size);
    EXPECT_EQ(config.l2_buffer_size, test_case.expected_l2_buffer_size);
    EXPECT_EQ(config.custom_buffer_size, test_case.expected_custom_buffer_size);
  }
}

TEST(ConfigTest, CalculateAccessCountsUsesExactLinearScaling) {
  struct AccessCountCase {
    unsigned long buffer_size_mb;
    size_t expected_accesses;
  };

  const AccessCountCase cases[] = {
      {256, Constants::BASE_LATENCY_ACCESSES / 2},
      {Constants::DEFAULT_BUFFER_SIZE_MB, Constants::BASE_LATENCY_ACCESSES},
      {1024, Constants::BASE_LATENCY_ACCESSES * 2},
  };

  for (const AccessCountCase& test_case : cases) {
    SCOPED_TRACE(test_case.buffer_size_mb);
    BenchmarkConfig config;
    config.buffer_size_mb = test_case.buffer_size_mb;

    calculate_access_counts(config);

    EXPECT_EQ(config.lat_num_accesses, test_case.expected_accesses);
  }
}

// Test validate_config rejects mutually exclusive flags

// Test mode-aware per-buffer capping based on required main buffer count
TEST(ConfigTest, ValidateConfigModeAwareBufferCap) {
  auto expected_cap = [](const BenchmarkConfig& cfg, unsigned long required_main_buffers) {
    unsigned long cap = cfg.max_total_allowed_mb / required_main_buffers;
    if (cap < Constants::MINIMUM_LIMIT_MB_PER_BUFFER) {
      cap = Constants::MINIMUM_LIMIT_MB_PER_BUFFER;
    }
    return cap;
  };

  BenchmarkConfig full;
  full.run_benchmark = true;
  full.buffer_size_mb = std::numeric_limits<unsigned long>::max();
  EXPECT_EQ(validate_config(full), EXIT_SUCCESS);
  EXPECT_EQ(full.buffer_size_mb, expected_cap(full, 2));

  BenchmarkConfig bw_only;
  bw_only.run_benchmark = true;
  bw_only.only_bandwidth = true;
  bw_only.buffer_size_mb = std::numeric_limits<unsigned long>::max();
  EXPECT_EQ(validate_config(bw_only), EXIT_SUCCESS);
  EXPECT_EQ(bw_only.buffer_size_mb, expected_cap(bw_only, 2));

  BenchmarkConfig lat_only;
  lat_only.run_benchmark = true;
  lat_only.only_latency = true;
  lat_only.buffer_size_mb = std::numeric_limits<unsigned long>::max();
  EXPECT_EQ(validate_config(lat_only), EXIT_SUCCESS);
  EXPECT_EQ(lat_only.buffer_size_mb, expected_cap(lat_only, 1));
}

TEST(ConfigTest, ValidateConfigLatencyTargetCombinations) {
  struct TargetCase {
    bool only_latency;
    unsigned long buffer_mb;
    long long cache_kb;
    std::string diagnostic;
  };
  const TargetCase cases[] = {
      {false, 0, -1, Messages::error_buffersize_zero_requires_only_latency()},
      {false, 512, 0, Messages::error_cache_size_zero_requires_only_latency()},
      {true, 0, 8096, ""},
      {true, 16, 0, ""},
      {true, 0, 0, Messages::error_only_latency_requires_latency_target()},
  };
  for (const TargetCase& test_case : cases) {
    SCOPED_TRACE(test_case.only_latency);
    SCOPED_TRACE(test_case.buffer_mb);
    SCOPED_TRACE(test_case.cache_kb);
    BenchmarkConfig config;
    config.run_benchmark = true;
    config.only_latency = test_case.only_latency;
    config.buffer_size_mb = test_case.buffer_mb;
    config.custom_cache_size_kb_ll = test_case.cache_kb;
    config.use_custom_cache_size = test_case.cache_kb >= 0;
    config.custom_cache_size_bytes =
        test_case.cache_kb < 0 ? 0 : static_cast<size_t>(test_case.cache_kb) * Constants::BYTES_PER_KB;
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, test_case.diagnostic.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
    EXPECT_EQ(diagnostic, test_case.diagnostic.empty() ? "" : Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, ValidateConfigLatencyLocalityAndStrideConstraints) {
  struct LocalityCase {
    size_t stride;
    size_t locality;
    LatencyChainMode mode;
    std::string diagnostic;
  };
  const size_t page = scoped_config_test_hooks.page_size_bytes();
  const LocalityCase cases[] = {
      {Constants::LATENCY_STRIDE_BYTES, page + Constants::BYTES_PER_KB, LatencyChainMode::Auto,
       Messages::error_latency_tlb_locality_page_multiple(page / Constants::BYTES_PER_KB + 1,
                                                          page / Constants::BYTES_PER_KB)},
      {sizeof(uintptr_t) + 1, page, LatencyChainMode::Auto,
       Messages::error_latency_stride_alignment(sizeof(uintptr_t) + 1, sizeof(uintptr_t))},
      {page, page, LatencyChainMode::Auto, Messages::error_latency_tlb_locality_too_small_for_stride(page, page)},
      {page, page * 2, LatencyChainMode::Auto, ""},
      {Constants::LATENCY_STRIDE_BYTES, 0, LatencyChainMode::SameRandomInBoxIncreasingBox,
       Messages::error_latency_chain_mode_requires_locality("same-random-in-box")},
      {Constants::LATENCY_STRIDE_BYTES, 0, LatencyChainMode::GlobalRandom, ""},
  };
  for (const LocalityCase& test_case : cases) {
    SCOPED_TRACE(test_case.stride);
    SCOPED_TRACE(test_case.locality);
    SCOPED_TRACE(static_cast<int>(test_case.mode));
    BenchmarkConfig config;
    config.latency_stride_bytes = test_case.stride;
    config.latency_tlb_locality_bytes = test_case.locality;
    config.latency_chain_mode = test_case.mode;
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, test_case.diagnostic.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
    EXPECT_EQ(diagnostic, test_case.diagnostic.empty() ? "" : Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, ValidateConfigAnalyzeTlbSkipsUnrelatedStandardModeRules) {
  BenchmarkConfig config;
  config.analyze_tlb = true;
  config.only_bandwidth = true;
  config.only_latency = true;
  config.latency_stride_bytes = Constants::LATENCY_STRIDE_BYTES;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}
