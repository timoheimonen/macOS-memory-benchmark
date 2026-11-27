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

  const std::string& cpu_name() const { return hooks_.cpu_name; }
  const std::string& macos_version() const { return hooks_.macos_version; }
  int performance_cores() const { return hooks_.performance_cores; }
  int efficiency_cores() const { return hooks_.efficiency_cores; }
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

struct CapturedStreamsParseResult {
  int result = EXIT_FAILURE;
  std::string stdout_output;
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

CapturedStreamsParseResult parse_capturing_streams(const std::vector<std::string>& arguments,
                                                   BenchmarkConfig& config) {
  std::vector<std::string> mutable_arguments = arguments;
  std::vector<char*> argv;
  argv.reserve(mutable_arguments.size());
  for (std::string& argument : mutable_arguments) {
    argv.push_back(argument.data());
  }

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int result = parse_arguments(static_cast<int>(argv.size()), argv.data(), config);
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();
  return {result, stdout_output, stderr_output};
}

std::string expected_invalid_value(const std::string& option, const std::string& value, const std::string& reason) {
  return Messages::error_prefix() + Messages::error_invalid_value(option, value, reason);
}

}  // namespace

TEST(ConfigTest, ParsesAliasesExplicitOptionsAndHelpFlag) {
  struct ValidAliasCase {
    const char* label;
    std::vector<std::string> arguments;
  };
  const ValidAliasCase alias_cases[] = {
      {"long",
       {"program", "--buffer-size", "1024", "--iterations", "500", "--count", "3", "--threads", "1",
        "--output", "results.json", "--non-cacheable"}},
      {"short",
       {"program", "-b", "1024", "-i", "500", "-r", "3", "-t", "1", "-o", "results.json", "-u"}},
  };
  for (const ValidAliasCase& test_case : alias_cases) {
    SCOPED_TRACE(test_case.label);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_FALSE(config.analyze_tlb);
    EXPECT_EQ(config.buffer_size_mb, 1024u);
    EXPECT_EQ(config.iterations, 500);
    EXPECT_EQ(config.loop_count, 3);
    EXPECT_EQ(config.num_threads, 1);
    EXPECT_TRUE(config.user_specified_threads);
    EXPECT_EQ(config.output_file, "results.json");
    EXPECT_TRUE(config.use_non_cacheable);
  }

  struct ExplicitCase {
    const char* option;
    const char* value;
    std::function<void(const BenchmarkConfig&)> verify;
  };
  const ExplicitCase explicit_cases[] = {
      {"--cache-size", "0",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.custom_cache_size_kb_ll, 0);
         EXPECT_TRUE(config.use_custom_cache_size);
       }},
      {"--buffer-size", "0",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.buffer_size_mb, 0u);
         EXPECT_TRUE(config.user_specified_buffer_size);
       }},
      {"--latency-tlb-locality-kb", "16",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_tlb_locality_bytes, 16u * Constants::BYTES_PER_KB);
         EXPECT_TRUE(config.user_specified_latency_tlb_locality);
       }},
      {"--latency-tlb-locality-kb", "0",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_tlb_locality_bytes, 0u);
         EXPECT_TRUE(config.user_specified_latency_tlb_locality);
       }},
      {"--latency-stride-bytes", "64",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_stride_bytes, 64u);
         EXPECT_TRUE(config.user_specified_latency_stride);
       }},
      {"--latency-chain-mode", "same-random-in-box",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::SameRandomInBoxIncreasingBox);
         EXPECT_TRUE(config.user_specified_latency_chain_mode);
       }},
  };
  for (const ExplicitCase& test_case : explicit_cases) {
    SCOPED_TRACE(std::string(test_case.option) + " " + test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", test_case.option, test_case.value}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    test_case.verify(config);
  }

  {
    SCOPED_TRACE("help-flag");
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed = parse_capturing_streams({"program", "-h"}, config);
    EXPECT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(config.help_printed);
    EXPECT_NE(parsed.stdout_output.find("Usage:"), std::string::npos);
    EXPECT_TRUE(parsed.stderr_output.empty());
  }
}

TEST(ConfigTest, ParseSweepValid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "--output", "sweep.json", "--sweep", "buffer-size=128,256",
                        "--sweep", "threads=1,2", "--sweep-max-runs", "4"};
  int argc = 9;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_sweep);
  ASSERT_EQ(config.sweep_specs.size(), 2u);
  EXPECT_EQ(config.sweep_specs[0].parameter, SweepParameter::BufferSizeMb);
  EXPECT_EQ(config.sweep_specs[0].values[0].unsigned_value, 128u);
  EXPECT_EQ(config.sweep_specs[0].values[0].integer_value, 128);
  EXPECT_EQ(config.sweep_specs[0].values[1].unsigned_value, 256u);
  EXPECT_EQ(config.sweep_specs[0].values[1].integer_value, 256);
  EXPECT_EQ(config.sweep_specs[1].parameter, SweepParameter::Threads);
  EXPECT_EQ(config.sweep_specs[1].values[0].integer_value, 1);
  EXPECT_EQ(config.sweep_specs[1].values[1].integer_value, 2);
  EXPECT_EQ(config.sweep_max_runs, 4u);
}

TEST(ConfigTest, ParseFlagShapedOutputTargetsAndPrescanBoundaries) {
  const char* output_options[] = {"-o", "--output"};
  const char* output_targets[] = {"-", "./-", "-T", "--analyze-tlb", "-k", "--cache-size"};

  size_t case_index = 0;
  for (const char* output_option : output_options) {
    for (const char* output_target : output_targets) {
      std::vector<std::string> arguments = {"program"};
      const bool output_first = (case_index++ / 2) % 2 == 0;
      if (output_first) {
        arguments.insert(arguments.end(), {output_option, output_target, "--non-cacheable"});
      } else {
        arguments.insert(arguments.end(), {"--non-cacheable", output_option, output_target});
      }
      SCOPED_TRACE(testing::PrintToString(arguments));

      BenchmarkConfig config;
      const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);

      EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
      if (parsed.result != EXIT_SUCCESS) {
        continue;
      }
      EXPECT_EQ(config.output_file, output_target);
      EXPECT_FALSE(config.analyze_tlb);
      EXPECT_FALSE(config.run_sweep);
      EXPECT_TRUE(config.use_non_cacheable);
      EXPECT_EQ(config.custom_cache_size_kb_ll, -1);
      EXPECT_FALSE(config.use_custom_cache_size);
      EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
    }
  }

  struct BoundaryCase {
    const char* label;
    std::vector<std::string> arguments;
    const char* output_target;
    bool analyze_tlb;
    long long cache_size_kb;
  };

  const BoundaryCase cases[] = {
      {"real TLB option after output value", {"program", "--output", "-T", "--analyze-tlb"}, "-T", true, -1},
      {"real TLB option before output value", {"program", "-T", "--output", "--analyze-tlb"}, "--analyze-tlb", true,
       -1},
      {"real cache option after output value", {"program", "--output", "-k", "--cache-size", "256"}, "-k", false, 256},
      {"real cache option before output value", {"program", "-k", "256", "--output", "--cache-size"}, "--cache-size",
       false, 256},
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
    EXPECT_EQ(config.custom_cache_size_kb_ll, test_case.cache_size_kb);
    EXPECT_EQ(config.use_custom_cache_size, test_case.cache_size_kb >= 0);
    EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  }
}

TEST(ConfigTest, RejectsMissingDuplicateAndSemanticOptionErrors) {
  for (const char* option : {"--output", "--cache-size", "--iterations"}) {
    SCOPED_TRACE(std::string("missing ") + option);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", option}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output, Messages::error_prefix() + Messages::error_missing_value(option) + "\n");
  }

  struct DuplicateOptionCase {
    const char* option;
    const char* first_value;
    const char* second_value;
  };
  const DuplicateOptionCase duplicate_cases[] = {
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
  for (const DuplicateOptionCase& test_case : duplicate_cases) {
    SCOPED_TRACE(test_case.option);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(
        {"program", test_case.option, test_case.first_value, test_case.option, test_case.second_value}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output,
              Messages::error_prefix() + Messages::error_duplicate_option(test_case.option) + "\n");
  }

  struct SemanticInvalidCase {
    const char* option;
    const char* value;
    std::string reason;
  };
  const SemanticInvalidCase semantic_cases[] = {
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
      {"--sweep-max-runs", "0", Messages::reason_positive_integer()},
      {"-unknown", "value", ""},
  };
  for (const SemanticInvalidCase& test_case : semantic_cases) {
    SCOPED_TRACE(test_case.option);
    SCOPED_TRACE(test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", test_case.option, test_case.value}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    const std::string diagnostic = test_case.reason.empty()
                                       ? Messages::error_prefix() + Messages::error_unknown_option(test_case.option)
                                       : expected_invalid_value(test_case.option, test_case.value, test_case.reason);
    EXPECT_EQ(parsed.stderr_output, diagnostic + "\n");
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
      {{"program", "--seed", "9x"}, "--seed", "9x", kInvalidUnsignedReason},
      {{"program", "--iterations", " 5"}, "--iterations", " 5", kInvalidSignedReason},
      {{"program", "--iterations", "5 "}, "--iterations", "5 ", kInvalidSignedReason},
      {{"program", "--iterations", "+5"}, "--iterations", "+5", kInvalidSignedReason},
      {{"program", "--seed", "+5"}, "--seed", "+5", kInvalidUnsignedReason},
      {{"program", "--iterations", "9223372036854775808"}, "--iterations", "9223372036854775808", "out of range"},
      {{"program", "--seed", "18446744073709551616"},
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

TEST(ConfigTest, RejectsMalformedAndOutOfDomainSweepValues) {
  struct InvalidSweepCase {
    std::string specification;
    std::string reason;
    bool analyze_tlb;
  };

  constexpr const char* kInvalidSignedReason =
      "must be an integer without whitespace, a plus sign, or trailing characters";
  const long long max_locality_kb =
      static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
  const InvalidSweepCase cases[] = {
      {"latency-samples=100,200", "unsupported sweep parameter: latency-samples", false},
      {"buffer-size=1,,2", "sweep value list cannot contain empty values", false},
      {"buffer-size=1x", kInvalidSignedReason, false},
      {"cache-size=16x", kInvalidSignedReason, false},
      {"threads=1x", kInvalidSignedReason, false},
      {"latency-tlb-locality-kb=16x", kInvalidSignedReason, false},
      {"latency-stride-bytes=64x", kInvalidSignedReason, false},
      {"latency-stride-bytes=64x", kInvalidSignedReason, true},
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
    std::vector<std::string> arguments = {"program"};
    if (test_case.analyze_tlb) {
      arguments.push_back("--analyze-tlb");
    }
    arguments.push_back("--sweep");
    arguments.push_back(test_case.specification);
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
         config.analyze_tlb = true;
         config.sweep_max_runs = 3;
         SweepValue d1, d2, s1, s2;
         d1.tlb_sweep_density = TlbSweepDensity::Low;
         d2.tlb_sweep_density = TlbSweepDensity::High;
         s1.integer_value = 64;
         s2.integer_value = 128;
         config.sweep_specs = {{SweepParameter::TlbDensity, "tlb-density", {d1, d2}},
                               {SweepParameter::LatencyStrideBytes, "latency-stride-bytes", {s1, s2}}};
       },
       Messages::error_sweep_too_many_runs(4, 3)},
      {"tlb cache",
       [](BenchmarkConfig& config) {
         config.analyze_tlb = true;
         config.sweep_specs = {{SweepParameter::CacheSizeKb, "cache-size", {{"1024"}}}};
       },
       Messages::error_sweep_parameter_not_allowed("cache-size", "--analyze-tlb")},
      {"missing output",
       [](BenchmarkConfig& config) {
         config.analyze_tlb = true;
         config.output_file.clear();
         SweepValue d1;
         d1.tlb_sweep_density = TlbSweepDensity::Low;
         config.sweep_specs = {{SweepParameter::TlbDensity, "tlb-density", {d1}}};
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

TEST(ConfigTest, ParseAnalyzeTlbDefaultsExplicitAndInvalidOptions) {
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
    EXPECT_EQ(config.cpu_name, scoped_config_test_hooks.cpu_name());
    EXPECT_EQ(config.macos_version, scoped_config_test_hooks.macos_version());
    EXPECT_EQ(config.perf_cores, scoped_config_test_hooks.performance_cores());
    EXPECT_EQ(config.eff_cores, scoped_config_test_hooks.efficiency_cores());
    EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  }

  struct ValidCase {
    const char* option;
    const char* value;
    std::function<void(const BenchmarkConfig&)> verify;
  };
  const ValidCase valid_cases[] = {
      {"--sweep-max-runs", "24", [](const BenchmarkConfig& config) { EXPECT_EQ(config.sweep_max_runs, 24u); }},
      {"--seed", "18446744073709551615",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.tlb_seed, std::numeric_limits<uint64_t>::max());
         EXPECT_TRUE(config.user_specified_tlb_seed);
       }},
      {"--latency-stride-bytes", "128",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_stride_bytes, 128u);
         EXPECT_TRUE(config.user_specified_latency_stride);
       }},
      {"--latency-chain-mode", "random-box",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::RandomInBoxRandomBox);
         EXPECT_TRUE(config.user_specified_latency_chain_mode);
       }},
      {"--tlb-density", "low",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Low);
         EXPECT_TRUE(config.user_specified_tlb_density);
       }},
      {"--tlb-density", "medium",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Medium);
         EXPECT_TRUE(config.user_specified_tlb_density);
       }},
      {"--tlb-density", "high",
       [](const BenchmarkConfig& config) {
         EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::High);
         EXPECT_TRUE(config.user_specified_tlb_density);
       }},
  };
  for (const ValidCase& test_case : valid_cases) {
    SCOPED_TRACE(std::string(test_case.option) + " " + test_case.value);
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--analyze-tlb", test_case.option, test_case.value}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.analyze_tlb);
    test_case.verify(config);
  }

  struct InvalidCase {
    std::vector<std::string> options;
    std::string diagnostic;
  };
  const InvalidCase invalid_cases[] = {
      {{"--buffer-size", "512"},
       Messages::error_option_not_allowed_in_mode("--buffer-size", Messages::error_analyze_tlb_must_be_used_alone())},
      {{"--seed", "42", "--seed", "43"}, Messages::error_duplicate_option("--seed")},
      {{"--latency-chain-mode", "global-random"}, Messages::error_analyze_tlb_global_random_unsupported()},
      {{"--tlb-density", "ultra"},
       Messages::error_invalid_value("--tlb-density", "ultra", Messages::reason_tlb_density_invalid())},
      {{"--latency-stride-bytes", "0"},
       Messages::error_invalid_value(
           "--latency-stride-bytes", "0",
           Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max()))},
      {{"--latency-stride-bytes", "65"}, Messages::error_latency_stride_alignment(65, sizeof(void*))},
      {{"--sweep-max-runs", "0"},
       Messages::error_invalid_value("--sweep-max-runs", "0", Messages::reason_positive_integer())},
      {{"--output"}, Messages::error_missing_value("--output")},
  };
  for (const InvalidCase& test_case : invalid_cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.options));
    std::vector<std::string> arguments = {"program", "--analyze-tlb"};
    arguments.insert(arguments.end(), test_case.options.begin(), test_case.options.end());
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output, Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, CalculateBufferSizesAndAccessCounts) {
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
  const BufferSizeCase buffer_cases[] = {
      {false, kL1Size, kL2Size, 0, kL1Size, kL2Size, 0},
      {true, 0, 0, kCustomSize, 0, 0, kCustomSize},
  };
  for (const BufferSizeCase& test_case : buffer_cases) {
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

  struct AccessCountCase {
    unsigned long buffer_size_mb;
    size_t expected_accesses;
  };
  const AccessCountCase access_cases[] = {
      {256, Constants::BASE_LATENCY_ACCESSES / 2},
      {Constants::DEFAULT_BUFFER_SIZE_MB, Constants::BASE_LATENCY_ACCESSES},
      {1024, Constants::BASE_LATENCY_ACCESSES * 2},
  };
  for (const AccessCountCase& test_case : access_cases) {
    SCOPED_TRACE(test_case.buffer_size_mb);
    BenchmarkConfig config;
    config.buffer_size_mb = test_case.buffer_size_mb;
    calculate_access_counts(config);
    EXPECT_EQ(config.lat_num_accesses, test_case.expected_accesses);
  }
}

TEST(ConfigTest, ValidateAnalyzeTlbWorksWithZeroBufferSize) {
  BenchmarkConfig config;
  config.analyze_tlb = true;
  config.buffer_size_mb = 0;
  config.latency_stride_bytes = Constants::LATENCY_STRIDE_BYTES;
  EXPECT_EQ(validate_config(config), EXIT_SUCCESS);
}

TEST(ConfigTest, SupportsInlineEqualsSyntaxAndRejectsUnexpectedValues) {
  {
    SCOPED_TRACE("general-inline-options");
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(
        {"program", "--buffer-size=512", "--iterations=25", "--count=2", "--threads=2",
         "--cache-size=256", "--latency-samples=12", "--latency-stride-bytes=128", "--latency-chain-mode=random-box",
         "--latency-tlb-locality-kb=32", "--seed=99", "--output=-", "--sweep=buffer-size=128,256",
         "--sweep-max-runs=6"},
        config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_FALSE(config.analyze_tlb);
    EXPECT_EQ(config.buffer_size_mb, 512u);
    EXPECT_TRUE(config.user_specified_buffer_size);
    EXPECT_EQ(config.iterations, 25);
    EXPECT_TRUE(config.user_specified_iterations);
    EXPECT_EQ(config.loop_count, 2);
    EXPECT_EQ(config.num_threads, 2);
    EXPECT_TRUE(config.user_specified_threads);
    EXPECT_EQ(config.custom_cache_size_kb_ll, 256);
    EXPECT_TRUE(config.use_custom_cache_size);
    EXPECT_EQ(config.latency_sample_count, 12);
    EXPECT_TRUE(config.user_specified_latency_samples);
    EXPECT_EQ(config.latency_stride_bytes, 128u);
    EXPECT_TRUE(config.user_specified_latency_stride);
    EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::RandomInBoxRandomBox);
    EXPECT_TRUE(config.user_specified_latency_chain_mode);
    EXPECT_EQ(config.latency_tlb_locality_bytes, 32u * Constants::BYTES_PER_KB);
    EXPECT_TRUE(config.user_specified_latency_tlb_locality);
    EXPECT_TRUE(config.user_specified_seed);
    EXPECT_EQ(config.output_file, "-");
    EXPECT_TRUE(config.run_sweep);
    ASSERT_EQ(config.sweep_specs.size(), 1u);
    EXPECT_EQ(config.sweep_specs[0].parameter, SweepParameter::BufferSizeMb);
    ASSERT_EQ(config.sweep_specs[0].values.size(), 2u);
    EXPECT_EQ(config.sweep_specs[0].values[0].unsigned_value, 128u);
    EXPECT_EQ(config.sweep_specs[0].values[1].unsigned_value, 256u);
    EXPECT_EQ(config.sweep_max_runs, 6u);
  }

  {
    SCOPED_TRACE("general-empty-inline-output");
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", "--output="}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.output_file.empty());
  }

  {
    SCOPED_TRACE("tlb-inline-options");
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(
        {"program", "--analyze-tlb", "--output=-", "--latency-stride-bytes=128",
         "--latency-chain-mode=same-random-in-box", "--tlb-density=high", "--seed=77",
         "--sweep=tlb-density=low,high", "--sweep-max-runs=12"},
        config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.analyze_tlb);
    EXPECT_EQ(config.output_file, "-");
    EXPECT_EQ(config.latency_stride_bytes, 128u);
    EXPECT_TRUE(config.user_specified_latency_stride);
    EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::SameRandomInBoxIncreasingBox);
    EXPECT_TRUE(config.user_specified_latency_chain_mode);
    EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::High);
    EXPECT_TRUE(config.user_specified_tlb_density);
    EXPECT_EQ(config.tlb_seed, 77u);
    EXPECT_TRUE(config.user_specified_tlb_seed);
    EXPECT_TRUE(config.run_sweep);
    ASSERT_EQ(config.sweep_specs.size(), 1u);
    EXPECT_EQ(config.sweep_specs[0].parameter, SweepParameter::TlbDensity);
    EXPECT_EQ(config.sweep_max_runs, 12u);
  }

  {
    SCOPED_TRACE("tlb-empty-inline-output");
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", "--analyze-tlb", "--output="}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.output_file.empty());
  }

  struct UnexpectedValueCase {
    std::vector<std::string> arguments;
    const char* canonical_option;
  };
  const UnexpectedValueCase unexpected_cases[] = {
      {{"program", "--analyze-tlb=1"}, "--analyze-tlb"},
      {{"program", "--non-cacheable=1"}, "--non-cacheable"},
      {{"program", "--help=1"}, "--help"},
      {{"program", "--analyze-tlb", "--help=1"}, "--help"},
  };
  for (const UnexpectedValueCase& test_case : unexpected_cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output,
              Messages::error_prefix() + Messages::error_option_does_not_take_value(test_case.canonical_option) +
                  "\n");
  }
}

TEST(ConfigTest, RejectsDuplicatePrimaryAndBooleanFlags) {
  struct DuplicateFlagCase {
    std::vector<std::string> arguments;
    const char* canonical_option;
  };
  const DuplicateFlagCase cases[] = {
      {{"program", "-T", "-T"}, "--analyze-tlb"},
      {{"program", "-T", "--analyze-tlb"}, "--analyze-tlb"},
      {{"program", "-u", "-u"}, "--non-cacheable"},
      {{"program", "-h", "--help"}, "--help"},
      {{"program", "--analyze-tlb", "--help", "-h"}, "--help"},
  };
  for (const DuplicateFlagCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed = parse_capturing_streams(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output,
              Messages::error_prefix() + Messages::error_duplicate_option(test_case.canonical_option) + "\n");
  }
}

TEST(ConfigTest, HelpValidationOrderAndSideEffectIsolationAcrossGeneralAndTlbModes) {
  {
    SCOPED_TRACE("analyze-tlb-help-skips-os-and-seed");
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed = parse_capturing_streams({"program", "--analyze-tlb", "--help"}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(config.analyze_tlb);
    EXPECT_TRUE(config.help_printed);
    EXPECT_EQ(config.tlb_seed, 0u);
    EXPECT_TRUE(config.cpu_name.empty());
    EXPECT_TRUE(config.macos_version.empty());
    EXPECT_EQ(config.perf_cores, 0);
    EXPECT_EQ(config.eff_cores, 0);
    EXPECT_NE(parsed.stdout_output.find("Usage:"), std::string::npos);
    EXPECT_TRUE(parsed.stderr_output.empty());
  }

  {
    SCOPED_TRACE("analyze-tlb-help-rejects-unknown-option");
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed =
        parse_capturing_streams({"program", "--analyze-tlb", "--help", "--unknown"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_FALSE(config.help_printed);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output,
              Messages::error_prefix() +
                  Messages::error_option_not_allowed_in_mode("--unknown",
                                                             Messages::error_analyze_tlb_must_be_used_alone()) +
                  "\n");
  }

  {
    SCOPED_TRACE("general-help-skips-os-cache-and-seed");
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed = parse_capturing_streams({"program", "--help"}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_FALSE(config.analyze_tlb);
    EXPECT_TRUE(config.help_printed);
    EXPECT_FALSE(config.user_specified_seed);
    EXPECT_TRUE(config.cpu_name.empty());
    EXPECT_TRUE(config.macos_version.empty());
    EXPECT_EQ(config.l1_cache_size, 0u);
    EXPECT_EQ(config.l2_cache_size, 0u);
    EXPECT_EQ(config.num_threads, 0);
    EXPECT_NE(parsed.stdout_output.find("Usage:"), std::string::npos);
    EXPECT_TRUE(parsed.stderr_output.empty());
  }

  for (const char* bad_option : {"--cache-size", "--iterations"}) {
    SCOPED_TRACE(bad_option);
    BenchmarkConfig config;
    const CapturedStreamsParseResult parsed =
        parse_capturing_streams({"program", "--help", bad_option, "bad"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output,
              expected_invalid_value(bad_option, "bad", Messages::reason_signed_integer_invalid()) + "\n");
  }
}

TEST(ConfigTest, SinglePassParsesFlagShapedChainModeValueWithoutPrescanCollision) {
  BenchmarkConfig config;
  const CapturedParseResult parsed =
      parse_capturing_stderr({"program", "--analyze-tlb", "--latency-chain-mode", "-k"}, config);
  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_EQ(parsed.stderr_output,
            expected_invalid_value("--latency-chain-mode", "-k", Messages::error_latency_chain_mode_invalid()) + "\n");
}

TEST(ConfigTest, ParsesFullUnsignedLongRangeForBufferSizeAndSweep) {
  const unsigned long llong_max_plus_one =
      static_cast<unsigned long>(std::numeric_limits<long long>::max()) + 1UL;
  const unsigned long ulong_max = std::numeric_limits<unsigned long>::max();
  const std::string llong_max_plus_one_str = std::to_string(llong_max_plus_one);
  const std::string ulong_max_str = std::to_string(ulong_max);

  {
    SCOPED_TRACE("direct-llong-max-plus-one");
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--buffer-size", llong_max_plus_one_str}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_EQ(config.buffer_size_mb, llong_max_plus_one);
    EXPECT_TRUE(config.user_specified_buffer_size);
  }

  {
    SCOPED_TRACE("direct-ulong-max");
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--buffer-size=" + ulong_max_str}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_EQ(config.buffer_size_mb, ulong_max);
    EXPECT_TRUE(config.user_specified_buffer_size);
  }

  {
    SCOPED_TRACE("direct-ulong-max-plus-one");
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--buffer-size", "18446744073709551616"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output,
              expected_invalid_value("--buffer-size", "18446744073709551616",
                                     Messages::reason_unsigned_integer_out_of_range()) +
                  "\n");
  }

  {
    SCOPED_TRACE("direct-negative");
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--buffer-size", "-1"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output,
              expected_invalid_value("--buffer-size", "-1", Messages::error_buffersize_invalid(-1, ulong_max)) + "\n");
  }

  {
    SCOPED_TRACE("sweep-full-ulong-range");
    BenchmarkConfig config;
    const std::string spec = "buffer-size=" + llong_max_plus_one_str + "," + ulong_max_str;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--sweep", spec, "--output", "out.json"}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    ASSERT_EQ(config.sweep_specs.size(), 1u);
    ASSERT_EQ(config.sweep_specs[0].values.size(), 2u);
    EXPECT_EQ(config.sweep_specs[0].values[0].unsigned_value, llong_max_plus_one);
    EXPECT_EQ(config.sweep_specs[0].values[1].unsigned_value, ulong_max);
  }

  {
    SCOPED_TRACE("sweep-ulong-overflow");
    BenchmarkConfig config;
    const CapturedParseResult parsed =
        parse_capturing_stderr({"program", "--sweep", "buffer-size=18446744073709551616"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output,
              expected_invalid_value("--sweep", "buffer-size=18446744073709551616",
                                     Messages::reason_unsigned_integer_out_of_range()) +
                  "\n");
  }
}

TEST(ConfigTest, ParseArgumentsResetsStaleConfigStateOnReuse) {
  BenchmarkConfig config;
  const CapturedParseResult first = parse_capturing_stderr(
      {"program", "--buffer-size", "128", "--threads", "2", "--non-cacheable", "--cache-size", "256",
       "--latency-samples", "5", "--latency-stride-bytes", "128", "--latency-chain-mode", "random-box",
       "--latency-tlb-locality-kb", "32", "--seed", "42", "--output", "first.json", "--sweep", "threads=1,2"},
      config);
  ASSERT_EQ(first.result, EXIT_SUCCESS) << first.stderr_output;

  const CapturedParseResult second = parse_capturing_stderr({"program", "--analyze-tlb"}, config);
  ASSERT_EQ(second.result, EXIT_SUCCESS) << second.stderr_output;
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.buffer_size_mb, Constants::DEFAULT_BUFFER_SIZE_MB);
  EXPECT_FALSE(config.user_specified_buffer_size);
  EXPECT_FALSE(config.use_non_cacheable);
  EXPECT_FALSE(config.use_custom_cache_size);
  EXPECT_EQ(config.custom_cache_size_kb_ll, -1);
  EXPECT_FALSE(config.user_specified_latency_samples);
  EXPECT_FALSE(config.user_specified_latency_stride);
  EXPECT_FALSE(config.user_specified_latency_chain_mode);
  EXPECT_FALSE(config.user_specified_latency_tlb_locality);
  EXPECT_FALSE(config.user_specified_threads);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_TRUE(config.output_file.empty());
  EXPECT_FALSE(config.run_sweep);
  EXPECT_TRUE(config.sweep_specs.empty());
  EXPECT_EQ(config.cpu_name, scoped_config_test_hooks.cpu_name());
  EXPECT_EQ(config.macos_version, scoped_config_test_hooks.macos_version());
  EXPECT_EQ(config.perf_cores, scoped_config_test_hooks.performance_cores());
  EXPECT_EQ(config.eff_cores, scoped_config_test_hooks.efficiency_cores());

  const CapturedParseResult third = parse_capturing_stderr({"program"}, config);
  ASSERT_EQ(third.result, EXIT_SUCCESS) << third.stderr_output;
  EXPECT_FALSE(config.analyze_tlb);
  EXPECT_EQ(config.tlb_seed, 0u);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_EQ(config.num_threads, scoped_config_test_hooks.total_logical_cores());
}

TEST(ConfigTest, ValidateRejectsNoPrimaryModeAndLegacyFlagsWithExactDiagnostics) {
  for (const char* legacy_option : {"--benchmark", "-W", "--only-bandwidth", "--only-latency"}) {
    SCOPED_TRACE(legacy_option);
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr({"program", legacy_option}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(parsed.stderr_output, Messages::error_prefix() + Messages::error_unknown_option(legacy_option) + "\n");
  }

  struct NoModeCase {
    std::vector<std::string> arguments;
    std::string diagnostic;
  };
  const NoModeCase cases[] = {
      {{"program", "--seed", "42"}, Messages::error_seed_requires_supported_mode()},
      {{"program", "--threads", "1"}, Messages::error_missing_primary_mode()},
      {{"program", "--buffer-size", "64"}, Messages::error_missing_primary_mode()},
      {{"program", "--sweep", "buffer-size=1,2", "--output", "sweep.json"}, Messages::error_missing_primary_mode()},
  };
  for (const NoModeCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(diagnostic, Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(ConfigTest, ValidateRejectsDirectAndSweepOptionConflictsInTlbMode) {
  struct ConflictCase {
    std::vector<std::string> arguments;
    const char* direct_option;
    const char* sweep_key;
  };
  const ConflictCase cases[] = {
      {{"program", "--analyze-tlb", "--latency-stride-bytes", "64", "--sweep", "latency-stride-bytes=64,128",
        "--output", "out.json"},
       "--latency-stride-bytes", "latency-stride-bytes"},
      {{"program", "--analyze-tlb", "--latency-chain-mode", "auto", "--sweep", "latency-chain-mode=auto,random-box",
        "--output", "out.json"},
       "--latency-chain-mode", "latency-chain-mode"},
      {{"program", "--analyze-tlb", "--tlb-density", "low", "--sweep", "tlb-density=low,high", "--output", "out.json"},
       "--tlb-density", "tlb-density"},
  };
  for (const ConflictCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    BenchmarkConfig config;
    const CapturedParseResult parsed = parse_capturing_stderr(test_case.arguments, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    testing::internal::CaptureStderr();
    const int result = validate_config(config);
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(diagnostic,
              Messages::error_prefix() +
                  Messages::error_sweep_option_conflict(test_case.direct_option, test_case.sweep_key) + "\n");
  }
}
