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
#include "core/config/config.h"
#include "core/config/constants.h"
#include <cstdlib>
#include <limits>
#include <cstdint>
#include <unistd.h>  // getpagesize

// Test default configuration values
TEST(ConfigTest, DefaultValues) {
  BenchmarkConfig config;
  EXPECT_EQ(config.buffer_size_mb, Constants::DEFAULT_BUFFER_SIZE_MB);
  EXPECT_EQ(config.iterations, Constants::DEFAULT_ITERATIONS);
  EXPECT_EQ(config.loop_count, Constants::DEFAULT_LOOP_COUNT);
  EXPECT_EQ(config.latency_stride_bytes, static_cast<size_t>(Constants::LATENCY_STRIDE_BYTES));
  EXPECT_EQ(config.latency_tlb_locality_bytes,
            Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB * Constants::BYTES_PER_KB);
  EXPECT_EQ(config.custom_cache_size_kb_ll, -1);
  EXPECT_FALSE(config.use_custom_cache_size);
}

// Test parsing valid arguments
TEST(ConfigTest, ParseValidArguments) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-iterations", "500", "-buffersize", "1024", "-count", "3"};
  int argc = 7;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.iterations, 500);
  EXPECT_EQ(config.buffer_size_mb, 1024u);
  EXPECT_EQ(config.loop_count, 3);
}

// Test parsing custom cache size
// The code parses -cache-size in a first pass, then skips it in the second pass
// (since it was already parsed). This allows -cache-size to work correctly.
TEST(ConfigTest, ParseCustomCacheSize) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-cache-size", "256"};
  int argc = 3;
  
  // The code parses -cache-size in the first pass and sets the value,
  // then in the second pass it skips it (since it was already parsed)
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.custom_cache_size_kb_ll, 256);
  EXPECT_TRUE(config.use_custom_cache_size);
}

// Test parsing invalid cache size (too small)
TEST(ConfigTest, ParseInvalidCacheSizeTooSmall) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-cache-size", "8"};  // Below minimum of 16 KB
  int argc = 3;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test parsing invalid cache size (too large)
TEST(ConfigTest, ParseInvalidCacheSizeTooLarge) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-cache-size", "1100000"};  // Above maximum of 1048576 KB (1 GB)
  int argc = 3;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test parsing cache size zero (validated later; allowed only with -only-latency)
TEST(ConfigTest, ParseCacheSizeZero) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-cache-size", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.custom_cache_size_kb_ll, 0);
  EXPECT_TRUE(config.use_custom_cache_size);
}

// Test parsing buffer size zero (validated later; allowed only with -only-latency)
TEST(ConfigTest, ParseBufferSizeZero) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-buffersize", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.buffer_size_mb, 0u);
}

TEST(ConfigTest, ParseLatencyTlbLocalityValid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-tlb-locality-kb", "16"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.latency_tlb_locality_bytes, static_cast<size_t>(16) * Constants::BYTES_PER_KB);
}

TEST(ConfigTest, ParseLatencyStrideValid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-stride-bytes", "64"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.latency_stride_bytes, 64u);
}

TEST(ConfigTest, ParseLatencyStrideInvalidZero) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-stride-bytes", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseLatencyChainModeValid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-chain-mode", "same-random-in-box"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::SameRandomInBoxIncreasingBox);
}

TEST(ConfigTest, ParseLatencyChainModeInvalid) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-chain-mode", "unknown-mode"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseLatencyTlbLocalityZeroDisables) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-tlb-locality-kb", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(config.latency_tlb_locality_bytes, 0u);
}

TEST(ConfigTest, ParseLatencyTlbLocalityInvalidNegative) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-tlb-locality-kb", "-1"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseAnalyzeTlbStandalone) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb"};
  int argc = 2;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
}

TEST(ConfigTest, ParseAnalyzeTlbWithOtherArgumentsFails) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-buffersize", "512"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseAnalyzeTlbWithOutputSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-output", "tlb.json"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.output_file, "tlb.json");
}

TEST(ConfigTest, ParseAnalyzeTlbWithOutputFirstSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-output", "tlb.json", "-analyze-tlb"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.output_file, "tlb.json");
}

TEST(ConfigTest, ParseAnalyzeTlbWithLatencyStrideSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-latency-stride-bytes", "128"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.latency_stride_bytes, 128u);
}

TEST(ConfigTest, ParseAnalyzeTlbWithLatencyStrideAndOutputSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {
      "program", "-analyze-tlb", "-latency-stride-bytes", "64", "-output", "tlb.json"};
  int argc = 6;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.latency_stride_bytes, 64u);
  EXPECT_EQ(config.output_file, "tlb.json");
}

TEST(ConfigTest, ParseAnalyzeTlbWithLatencyChainModeSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-latency-chain-mode", "random-box"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.latency_chain_mode, LatencyChainMode::RandomInBoxRandomBox);
}

TEST(ConfigTest, ParseAnalyzeTlbWithTlbDensityLowSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-tlb-density", "low"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Low);
}

TEST(ConfigTest, ParseAnalyzeTlbWithTlbDensityMediumSucceeds) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-tlb-density", "medium"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.analyze_tlb);
  EXPECT_EQ(config.tlb_sweep_density, TlbSweepDensity::Medium);
}

TEST(ConfigTest, ParseAnalyzeTlbWithInvalidTlbDensityFails) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-tlb-density", "ultra"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseAnalyzeTlbWithInvalidLatencyStrideFails) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-latency-stride-bytes", "0"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseAnalyzeTlbWithUnalignedLatencyStrideFails) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-latency-stride-bytes", "65"};
  int argc = 4;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseAnalyzeTlbWithMissingOutputValueFails) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-analyze-tlb", "-output"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseNormalModeUserOptions) {
  BenchmarkConfig config;
  const char* argv[] = {
      "program", "-benchmark", "-threads", "1", "-latency-samples", "17", "-output", "results.json",
      "-non-cacheable"};
  int argc = 9;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_benchmark);
  EXPECT_EQ(config.num_threads, 1);
  EXPECT_TRUE(config.user_specified_threads);
  EXPECT_EQ(config.latency_sample_count, 17);
  EXPECT_TRUE(config.user_specified_latency_samples);
  EXPECT_EQ(config.output_file, "results.json");
  EXPECT_TRUE(config.use_non_cacheable);
}

TEST(ConfigTest, ParseThreadsInvalidZero) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-threads", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseLatencySamplesInvalidZero) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-latency-samples", "0"};
  int argc = 3;

  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ParseDuplicateValueOptionsRejected) {
  struct DuplicateOptionCase {
    const char* option;
    const char* first_value;
    const char* second_value;
  };

  const DuplicateOptionCase cases[] = {
      {"-iterations", "1", "2"},
      {"-buffersize", "1", "2"},
      {"-count", "1", "2"},
      {"-latency-samples", "1", "2"},
      {"-latency-stride-bytes", "64", "128"},
      {"-latency-chain-mode", "auto", "global-random"},
      {"-latency-tlb-locality-kb", "16", "32"},
      {"-threads", "1", "2"},
      {"-output", "first.json", "second.json"},
      {"-cache-size", "256", "512"},
  };

  for (const DuplicateOptionCase& test_case : cases) {
    SCOPED_TRACE(test_case.option);
    BenchmarkConfig config;
    const char* argv[] = {
        "program", test_case.option, test_case.first_value, test_case.option, test_case.second_value};
    int argc = 5;

    int result = parse_arguments(argc, const_cast<char**>(argv), config);
    EXPECT_EQ(result, EXIT_FAILURE);
  }
}

// Test parsing missing value for option
TEST(ConfigTest, ParseMissingValue) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-iterations"};
  int argc = 2;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test parsing unknown option
TEST(ConfigTest, ParseUnknownOption) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-unknown", "value"};
  int argc = 3;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test parsing help flag
TEST(ConfigTest, ParseHelpFlag) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-h"};
  int argc = 2;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);  // Help returns SUCCESS
}

// Test parsing -benchmark flag
TEST(ConfigTest, ParseBenchmarkFlag) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark"};
  int argc = 2;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_benchmark);
  EXPECT_FALSE(config.run_patterns);
}

// Test -benchmark with other modifier flags
TEST(ConfigTest, ParseBenchmarkWithModifiers) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark", "-only-latency", "-cache-size", "256"};
  int argc = 5;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_benchmark);
  EXPECT_TRUE(config.only_latency);
  EXPECT_EQ(config.custom_cache_size_kb_ll, 256);
}

TEST(ConfigTest, ValidateParsedOnlyBandwidthRejectsLatencySamples) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark", "-only-bandwidth", "-latency-samples", "10"};
  int argc = 5;

  ASSERT_EQ(parse_arguments(argc, const_cast<char**>(argv), config), EXIT_SUCCESS);
  EXPECT_TRUE(config.only_bandwidth);
  EXPECT_TRUE(config.user_specified_latency_samples);

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateParsedOnlyBandwidthRejectsCacheSize) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark", "-only-bandwidth", "-cache-size", "256"};
  int argc = 5;

  ASSERT_EQ(parse_arguments(argc, const_cast<char**>(argv), config), EXIT_SUCCESS);
  EXPECT_TRUE(config.only_bandwidth);
  EXPECT_TRUE(config.use_custom_cache_size);

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateParsedOnlyLatencyRejectsIterations) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark", "-only-latency", "-iterations", "10"};
  int argc = 5;

  ASSERT_EQ(parse_arguments(argc, const_cast<char**>(argv), config), EXIT_SUCCESS);
  EXPECT_TRUE(config.only_latency);
  EXPECT_TRUE(config.user_specified_iterations);

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test -benchmark and -patterns are mutually exclusive
TEST(ConfigTest, ParseBenchmarkAndPatternsMutuallyExclusive) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-benchmark", "-patterns"};
  int argc = 3;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test -benchmark and -patterns in reverse order
TEST(ConfigTest, ParsePatternsAndBenchmarkMutuallyExclusive) {
  BenchmarkConfig config;
  const char* argv[] = {"program", "-patterns", "-benchmark"};
  int argc = 3;
  
  int result = parse_arguments(argc, const_cast<char**>(argv), config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test buffer size calculation
TEST(ConfigTest, CalculateBufferSizes) {
  BenchmarkConfig config;
  config.l1_cache_size = 128 * 1024;  // 128 KB
  config.l2_cache_size = 4 * 1024 * 1024;  // 4 MB
  config.use_custom_cache_size = false;
  
  calculate_buffer_sizes(config);
  
  // L1 should be 100% of cache size, rounded to stride
  EXPECT_GT(config.l1_buffer_size, 0u);
  EXPECT_LE(config.l1_buffer_size, config.l1_cache_size);
  
  // L2 should be 100% of cache size, rounded to stride
  EXPECT_GT(config.l2_buffer_size, 0u);
  EXPECT_LE(config.l2_buffer_size, config.l2_cache_size);
}

// Test custom cache size buffer calculation
TEST(ConfigTest, CalculateCustomBufferSize) {
  BenchmarkConfig config;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = 256 * 1024;  // 256 KB
  
  calculate_buffer_sizes(config);
  
  EXPECT_GT(config.custom_buffer_size, 0u);
  EXPECT_LE(config.custom_buffer_size, config.custom_cache_size_bytes);
  EXPECT_EQ(config.custom_buffer_size % Constants::LATENCY_STRIDE_BYTES, 0u);
}

// Test access count calculation
TEST(ConfigTest, CalculateAccessCounts) {
  BenchmarkConfig config;
  config.buffer_size_mb = 512;  // Default size
  
  calculate_access_counts(config);
  
  // Should scale based on buffer size
  EXPECT_GT(config.lat_num_accesses, 0u);
  
  // With default buffer size, should be close to base
  size_t expected_min = Constants::BASE_LATENCY_ACCESSES / 2;
  size_t expected_max = Constants::BASE_LATENCY_ACCESSES * 2;
  EXPECT_GE(config.lat_num_accesses, expected_min);
  EXPECT_LE(config.lat_num_accesses, expected_max);
}

// Test access count scaling with different buffer sizes
TEST(ConfigTest, AccessCountScaling) {
  BenchmarkConfig config1, config2;
  
  config1.buffer_size_mb = 256;
  config2.buffer_size_mb = 1024;
  
  calculate_access_counts(config1);
  calculate_access_counts(config2);
  
  // Larger buffer should have more accesses
  EXPECT_LT(config1.lat_num_accesses, config2.lat_num_accesses);
}

// Test validate_config rejects mutually exclusive flags
TEST(ConfigTest, ValidateConfigRejectsOnlyBandwidthAndOnlyLatency) {
  BenchmarkConfig config;
  config.only_bandwidth = true;
  config.only_latency = true;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

// Test validate_config rejects only-flags with pattern mode
TEST(ConfigTest, ValidateConfigRejectsOnlyFlagsWithPatterns) {
  BenchmarkConfig config;
  config.run_patterns = true;
  config.only_bandwidth = true;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

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

TEST(ConfigTest, ValidateConfigRejectsBufferSizeZeroWithoutOnlyLatency) {
  BenchmarkConfig config;
  config.buffer_size_mb = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigRejectsCacheSizeZeroWithoutOnlyLatency) {
  BenchmarkConfig config;
  config.custom_cache_size_kb_ll = 0;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigAllowsCacheOnlyLatencyMode) {
  BenchmarkConfig config;
  config.run_benchmark = true;
  config.only_latency = true;
  config.buffer_size_mb = 0;
  config.custom_cache_size_kb_ll = 8096;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = static_cast<size_t>(8096) * Constants::BYTES_PER_KB;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST(ConfigTest, ValidateConfigAllowsMainOnlyLatencyMode) {
  BenchmarkConfig config;
  config.run_benchmark = true;
  config.only_latency = true;
  config.buffer_size_mb = 16;
  config.custom_cache_size_kb_ll = 0;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST(ConfigTest, ValidateConfigRejectsOnlyLatencyWithNoTargets) {
  BenchmarkConfig config;
  config.run_benchmark = true;
  config.only_latency = true;
  config.buffer_size_mb = 0;
  config.custom_cache_size_kb_ll = 0;
  config.use_custom_cache_size = true;
  config.custom_cache_size_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigRejectsLatencyTlbLocalityNotPageMultiple) {
  BenchmarkConfig config;
  const size_t page_size = static_cast<size_t>(getpagesize());
  config.latency_tlb_locality_bytes = page_size + Constants::BYTES_PER_KB;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigRejectsLatencyStrideNotPointerAligned) {
  BenchmarkConfig config;
  config.latency_stride_bytes = sizeof(uintptr_t) + 1;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigRejectsLatencyTlbLocalityTooSmallForStride) {
  BenchmarkConfig config;
  const size_t page_size = static_cast<size_t>(getpagesize());
  config.latency_stride_bytes = page_size;
  config.latency_tlb_locality_bytes = page_size;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigAllowsLatencyTlbLocalityForStride) {
  BenchmarkConfig config;
  const size_t page_size = static_cast<size_t>(getpagesize());
  config.latency_stride_bytes = page_size;
  config.latency_tlb_locality_bytes = page_size * 2;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST(ConfigTest, ValidateConfigAllowsLatencyTlbLocalityPageMultiple) {
  BenchmarkConfig config;
  const size_t page_size = static_cast<size_t>(getpagesize());
  config.latency_tlb_locality_bytes = page_size * 2;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST(ConfigTest, ValidateConfigRejectsLatencyChainModeWithoutLocality) {
  BenchmarkConfig config;
  config.latency_chain_mode = LatencyChainMode::SameRandomInBoxIncreasingBox;
  config.latency_tlb_locality_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_FAILURE);
}

TEST(ConfigTest, ValidateConfigAllowsGlobalLatencyChainModeWithoutLocality) {
  BenchmarkConfig config;
  config.latency_chain_mode = LatencyChainMode::GlobalRandom;
  config.latency_tlb_locality_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST(ConfigTest, ValidateConfigAnalyzeTlbBypassesRegularValidation) {
  BenchmarkConfig config;
  config.analyze_tlb = true;
  config.only_bandwidth = true;
  config.only_latency = true;
  config.latency_stride_bytes = 0;

  int result = validate_config(config);
  EXPECT_EQ(result, EXIT_SUCCESS);
}
