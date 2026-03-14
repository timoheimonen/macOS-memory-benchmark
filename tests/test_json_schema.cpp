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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "benchmark/core_to_core_latency_json.h"
#include "benchmark/tlb_analysis_json.h"
#include "benchmark/benchmark_runner.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/json/json_output.h"
#include "pattern_benchmark/pattern_benchmark.h"

namespace {

std::filesystem::path make_temp_json_path(const char* stem) {
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  const std::string file_name =
      std::string(stem) + "_" + std::to_string(::getpid()) + "_" + std::to_string(std::rand()) + ".json";
  return tmp_dir / file_name;
}

nlohmann::json read_json_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  nlohmann::json parsed;
  in >> parsed;
  return parsed;
}

}  // namespace

TEST(JsonSchemaTest, LatencySamplesUseMetricContainerShape) {
  nlohmann::json json_obj;
  add_latency_results(json_obj, {15.0, 16.0}, {15.1, 15.2, 15.3});

  ASSERT_TRUE(json_obj.contains(JsonKeys::LATENCY));
  ASSERT_TRUE(json_obj[JsonKeys::LATENCY].contains(JsonKeys::SAMPLES_NS));
  const nlohmann::json samples_json = json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS];
  ASSERT_TRUE(samples_json.is_object());
  EXPECT_TRUE(samples_json.contains(JsonKeys::VALUES));
  EXPECT_TRUE(samples_json.contains(JsonKeys::STATISTICS));
}

TEST(JsonSchemaTest, BenchmarkExporterIncludesBenchmarkModeAndOmitsEmptySections) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("benchmark_schema").string();
  config.only_bandwidth = true;
  config.only_latency = true;

  BenchmarkStatistics stats;
  ASSERT_EQ(save_results_to_json(config, stats, 1.0), EXIT_SUCCESS);

  const nlohmann::json output_json = read_json_file(config.output_file);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::BENCHMARK_JSON_MODE_NAME);
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::LATENCY_CHAIN_MODE));
  EXPECT_FALSE(output_json.contains(JsonKeys::MAIN_MEMORY));
  EXPECT_FALSE(output_json.contains(JsonKeys::CACHE));

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, PatternExporterIncludesPatternsMode) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("patterns_schema").string();

  PatternStatistics stats;
  ASSERT_EQ(save_pattern_results_to_json(config, stats, 2.0), EXIT_SUCCESS);

  const nlohmann::json output_json = read_json_file(config.output_file);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::PATTERNS_JSON_MODE_NAME);

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, TlbAnalysisExporterIncludesModeAndCoreCounts) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("tlb_schema").string();

  const std::string cpu_name = "test-cpu";
  const std::vector<size_t> localities_bytes = {16 * Constants::BYTES_PER_KB};
  const std::vector<std::vector<double>> sweep_loop_latencies_ns = {{15.0, 15.1}};
  const std::vector<double> p50_latency_ns = {15.0};
  const std::vector<double> page_walk_comparison_loop_latencies_ns = {95.0, 96.0};
  const TlbBoundaryDetection l1_boundary;
  const TlbBoundaryDetection l2_boundary;

  const TlbAnalysisJsonContext context = {
      config,
      cpu_name,
      4,
      6,
      16384,
      131072,
      1024 * 1024,
      64,
      2,
      1000,
      16 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_MB,
      1024,
      true,
      localities_bytes,
      sweep_loop_latencies_ns,
      p50_latency_ns,
      l1_boundary,
      l2_boundary,
      256,
      2048,
      true,
      page_walk_comparison_loop_latencies_ns,
      95.5,
      15.0,
      80.5,
      3.0,
  };

  ASSERT_EQ(save_tlb_analysis_to_json(context), EXIT_SUCCESS);
  const nlohmann::json output_json = read_json_file(config.output_file);

  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::TLB_ANALYSIS_JSON_MODE_NAME);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::PERFORMANCE_CORES], 4);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::EFFICIENCY_CORES], 6);
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::LATENCY_CHAIN_MODE));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::LATENCY_CHAIN_MODE], "random-box");

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, CoreToCoreExporterUsesSharedModeKeyAndSamplesContainer) {
  CoreToCoreLatencyConfig config;
  config.output_file = make_temp_json_path("core2core_schema").string();
  config.loop_count = 1;
  config.latency_sample_count = 2;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {
      {
          Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
          {10.0, 11.0},
          {10.2, 10.4},
          {},
          {},
      },
  };

  const CoreToCoreLatencyJsonContext context = {
      config,
      cpu_name,
      4,
      6,
      100,
      200,
      20,
      scenarios,
      4.5,
  };

  ASSERT_EQ(save_core_to_core_latency_to_json(context), EXIT_SUCCESS);
  const nlohmann::json output_json = read_json_file(config.output_file);

  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::CORE_TO_CORE_JSON_MODE_NAME);
  ASSERT_TRUE(output_json["core_to_core_latency"]["scenarios"][0].contains(JsonKeys::SAMPLES_NS));
  const nlohmann::json samples_json = output_json["core_to_core_latency"]["scenarios"][0][JsonKeys::SAMPLES_NS];
  EXPECT_TRUE(samples_json.contains(JsonKeys::VALUES));
  EXPECT_TRUE(samples_json.contains(JsonKeys::STATISTICS));

  std::filesystem::remove(config.output_file);
}
