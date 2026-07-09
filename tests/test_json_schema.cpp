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

#include <algorithm>
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
#include "output/json/json_output/json_output_api.h"
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

TlbMeasurementRecord make_paired_tlb_record(TlbMeasurementPass pass,
                                            size_t point_index,
                                            size_t locality_bytes,
                                            size_t round_index,
                                            size_t order_index,
                                            uint64_t task_seed,
                                            double spread_latency_ns) {
  TlbMeasurementRecord record{pass,
                              point_index,
                              locality_bytes,
                              round_index,
                              order_index,
                              task_seed,
                              spread_latency_ns};
  const size_t page_size = 16384;
  const size_t requested_pages = locality_bytes / page_size;
  record.paired.available = true;
  record.paired.spread_measured_first = (round_index % 2) == 0;
  record.paired.spread.seed = task_seed + 1;
  record.paired.spread.latency_ns = spread_latency_ns;
  record.paired.spread.diagnostics = {
      TlbChainLayout::Spread,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      requested_pages,
      requested_pages,
      requested_pages,
      requested_pages,
      1,
      locality_bytes,
      page_size,
      64,
      64,
      task_seed + 1,
      true,
  };
  record.paired.packed.seed = task_seed + 2;
  record.paired.packed.latency_ns = spread_latency_ns - 5.0;
  record.paired.packed.diagnostics = {
      TlbChainLayout::Packed,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      requested_pages,
      std::max<size_t>(1, (requested_pages + 255) / 256),
      requested_pages,
      requested_pages,
      std::min<size_t>(requested_pages, 256),
      requested_pages * 64,
      page_size,
      64,
      64,
      task_seed + 2,
      true,
  };
  record.paired.translation_delta_ns = 5.0;
  return record;
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
  config.tlb_sweep_density = TlbSweepDensity::High;
  config.tlb_seed = 12345;
  config.user_specified_tlb_seed = true;

  const std::string cpu_name = "test-cpu";
  const std::vector<size_t> localities_bytes = {16 * Constants::BYTES_PER_KB};
  const std::vector<std::vector<double>> sweep_loop_latencies_ns = {{15.0, 15.1}};
  const std::vector<double> p50_latency_ns = {15.0};
  const std::vector<double> page_walk_comparison_loop_latencies_ns = {95.0, 96.0};
  const std::vector<TlbSweepPoint> sweep_points = {{
      0, 1, 1, 16 * Constants::BYTES_PER_KB, 64, 256, "base",
      16 * Constants::BYTES_PER_KB, 16 * Constants::BYTES_PER_KB}};
  const std::vector<TlbMeasurementRecord> measurement_records = {
      make_paired_tlb_record(TlbMeasurementPass::Base,
                             0,
                             16 * Constants::BYTES_PER_KB,
                             0,
                             0,
                             100,
                             15.0),
      make_paired_tlb_record(TlbMeasurementPass::Base,
                             0,
                             16 * Constants::BYTES_PER_KB,
                             1,
                             0,
                             101,
                             15.1),
      make_paired_tlb_record(TlbMeasurementPass::Validation,
                             2,
                             16 * Constants::BYTES_PER_KB,
                             0,
                             0,
                             300,
                             14.9),
      make_paired_tlb_record(TlbMeasurementPass::Validation,
                             2,
                             16 * Constants::BYTES_PER_KB,
                             1,
                             0,
                             301,
                             15.0),
      make_paired_tlb_record(TlbMeasurementPass::LargeLocality,
                             1,
                             512 * Constants::BYTES_PER_MB,
                             0,
                             0,
                             200,
                             95.0),
      make_paired_tlb_record(TlbMeasurementPass::LargeLocality,
                             1,
                             512 * Constants::BYTES_PER_MB,
                             1,
                             0,
                             201,
                             96.0),
  };
  TlbBoundaryDetection l1_boundary;
  l1_boundary.detected = true;
  l1_boundary.boundary_index = 0;
  l1_boundary.boundary_locality_bytes = 16 * Constants::BYTES_PER_KB;
  l1_boundary.bracket_lower_bytes = 8 * Constants::BYTES_PER_KB;
  l1_boundary.bracket_upper_bytes = 16 * Constants::BYTES_PER_KB;
  l1_boundary.overlaps_private_cache_knee = true;
  l1_boundary.confidence = "Medium";
  l1_boundary.discovery.available = true;
  l1_boundary.discovery.passed = true;
  l1_boundary.discovery.effect_ns = 2.0;
  l1_boundary.discovery.minimum_effect_ns = 0.5;
  l1_boundary.discovery.noise_floor_ns = 0.1;
  l1_boundary.discovery.effect_ci = {1.5, 2.5, 0.95, 30, 2000};
  l1_boundary.discovery.persistence_points_passed = 2;
  l1_boundary.validation = l1_boundary.discovery;
  TlbBoundaryCandidate accepted_candidate;
  accepted_candidate.accepted = true;
  accepted_candidate.boundary_index = 0;
  accepted_candidate.boundary_locality_bytes = 16 * Constants::BYTES_PER_KB;
  accepted_candidate.bracket_lower_bytes = 8 * Constants::BYTES_PER_KB;
  accepted_candidate.bracket_upper_bytes = 16 * Constants::BYTES_PER_KB;
  accepted_candidate.discovery = l1_boundary.discovery;
  accepted_candidate.validation = l1_boundary.validation;
  l1_boundary.candidates.push_back(accepted_candidate);
  TlbBoundaryDetection l2_boundary;
  TlbBoundaryCandidate rejected_candidate;
  rejected_candidate.boundary_index = 1;
  rejected_candidate.boundary_locality_bytes = 32 * Constants::BYTES_PER_KB;
  rejected_candidate.bracket_lower_bytes = 16 * Constants::BYTES_PER_KB;
  rejected_candidate.bracket_upper_bytes = 32 * Constants::BYTES_PER_KB;
  rejected_candidate.discovery.available = true;
  rejected_candidate.discovery.effect_ns = 1.0;
  rejected_candidate.discovery.minimum_effect_ns = 0.5;
  rejected_candidate.discovery.rejection_reason =
      "persistence-not-confirmed";
  rejected_candidate.validation.rejection_reason =
      "discovery-not-accepted";
  l2_boundary.candidates.push_back(rejected_candidate);
  const PrivateCacheKneeDetection private_cache_knee;

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
      "complete",
      1,
      1,
      1,
      1,
      true,
      true,
      sweep_points,
      measurement_records,
      localities_bytes,
      sweep_loop_latencies_ns,
      p50_latency_ns,
      l1_boundary,
      l2_boundary,
      private_cache_knee,
      248,
      2048,
      240,
      256,
      2000,
      2048,
      6,
      true,
      4 * Constants::BYTES_PER_MB,
      256,
      true,
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
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["schema_version"], 4);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["methodology_version"],
            "page-native-paired-validated-v4");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["boundary_signal"],
            "translation_delta_ns");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["seed"], 12345);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["seed_source"], "user");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["schedule_policy"],
            "seeded-cyclic-latin");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["latency_chain_mode_requested"], "auto");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::LATENCY_CHAIN_MODE));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::LATENCY_CHAIN_MODE], "random-box");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::TLB_DENSITY));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::TLB_DENSITY], "high");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains("fine_sweep_added_points"));
  EXPECT_TRUE(output_json["tlb_analysis"].contains("private_cache_knee"));
  EXPECT_EQ(output_json["tlb_analysis"]["l1_tlb_detection"]["inferred_entries"], 248);
  EXPECT_EQ(output_json["tlb_analysis"]["l1_tlb_detection"]["inferred_entries_method"],
            "validated-bracket-range-midpoint-estimate");
  EXPECT_TRUE(output_json["tlb_analysis"]["l1_tlb_detection"]["discovery"]["passed"]);
  EXPECT_TRUE(output_json["tlb_analysis"]["l1_tlb_detection"]["validation"]["passed"]);
  EXPECT_TRUE(output_json["tlb_analysis"]["l1_tlb_detection"]["candidates"][0]["accepted"]);
  EXPECT_FALSE(output_json["tlb_analysis"]["l2_tlb_detection"]["detected"]);
  EXPECT_EQ(output_json["tlb_analysis"]["l2_tlb_detection"]["candidates"].size(), 1u);
  EXPECT_EQ(output_json["tlb_analysis"]["l2_tlb_detection"]["candidates"][0]
                       ["discovery"]["rejection_reason"],
            "persistence-not-confirmed");
  EXPECT_TRUE(output_json["tlb_analysis"]["l1_tlb_detection"]["overlaps_private_cache_knee"]);
  EXPECT_EQ(output_json["tlb_analysis"]["status"], "complete");
  EXPECT_EQ(output_json["tlb_analysis"]["planned_points"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["measured_points"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["validation_planned_points"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["validation_measured_points"], 1);
  EXPECT_TRUE(output_json["tlb_analysis"]["validation_complete"]);
  EXPECT_EQ(output_json["tlb_analysis"]["planned_measurements"], 4);
  EXPECT_EQ(output_json["tlb_analysis"]["completed_measurements"], 4);
  EXPECT_EQ(output_json["tlb_analysis"]["planned_measurement_pairs"], 4);
  EXPECT_EQ(output_json["tlb_analysis"]["completed_measurement_pairs"], 4);
  EXPECT_EQ(output_json["tlb_analysis"]["planned_raw_measurements"], 8);
  EXPECT_EQ(output_json["tlb_analysis"]["completed_raw_measurements"], 8);
  EXPECT_TRUE(output_json["tlb_analysis"]["conclusions_valid"]);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"].size(), 6u);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]["pass"], "base");
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][2]["pass"],
            "validation");
  EXPECT_TRUE(output_json["tlb_analysis"]["measurement_records"][0]
                         ["paired_control"]["available"]);
  EXPECT_DOUBLE_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                              ["paired_control"]["translation_delta_ns"],
                   5.0);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["requested_pages"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["actual_pages"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["actual_node_count"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["actual_unique_cache_lines"], 1);
  EXPECT_DOUBLE_EQ(output_json["tlb_analysis"]["sweep"][0]
                              ["translation_delta_p50_ns"],
                   5.0);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["measurements"].size(), 4u);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["measurements"][1]["round_index"], 1);
  EXPECT_TRUE(output_json["tlb_analysis"]["sweep"][0].contains(
      "validation_translation_deltas_ns"));
  EXPECT_TRUE(output_json["tlb_analysis"].contains("large_locality_latency_delta"));
  EXPECT_DOUBLE_EQ(output_json["tlb_analysis"]["large_locality_latency_delta"]["delta_ns"], 80.5);
  EXPECT_EQ(output_json["tlb_analysis"]["large_locality_latency_delta"]["measurements"].size(), 2u);
  EXPECT_TRUE(output_json["tlb_analysis"]["page_walk_penalty"]["deprecated"]);
  EXPECT_EQ(output_json["tlb_analysis"]["page_walk_penalty"]["replacement"],
            "large_locality_latency_delta");

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, TlbAnalysisExporterOmitsPageWalkPenaltyWhenComparisonIncomplete) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("tlb_page_walk_incomplete").string();

  const std::string cpu_name = "test-cpu";
  const std::vector<size_t> localities_bytes = {16 * Constants::BYTES_PER_KB};
  const std::vector<std::vector<double>> sweep_loop_latencies_ns = {{15.0, 15.1}};
  const std::vector<double> p50_latency_ns = {15.0};
  const std::vector<double> page_walk_comparison_loop_latencies_ns;
  const std::vector<TlbSweepPoint> sweep_points = {{
      0, 1, 1, 16 * Constants::BYTES_PER_KB, 64, 256, "base",
      16 * Constants::BYTES_PER_KB, 16 * Constants::BYTES_PER_KB}};
  const std::vector<TlbMeasurementRecord> measurement_records = {
      make_paired_tlb_record(TlbMeasurementPass::Base,
                             0,
                             16 * Constants::BYTES_PER_KB,
                             0,
                             0,
                             100,
                             15.0),
  };
  const TlbBoundaryDetection l1_boundary;
  const TlbBoundaryDetection l2_boundary;
  const PrivateCacheKneeDetection private_cache_knee;

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
      "interrupted",
      2,
      1,
      1,
      0,
      false,
      false,
      sweep_points,
      measurement_records,
      localities_bytes,
      sweep_loop_latencies_ns,
      p50_latency_ns,
      l1_boundary,
      l2_boundary,
      private_cache_knee,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      false,
      0,
      0,
      true,
      false,
      page_walk_comparison_loop_latencies_ns,
      0.0,
      15.0,
      0.0,
      3.0,
  };

  ASSERT_EQ(save_tlb_analysis_to_json(context), EXIT_SUCCESS);
  const nlohmann::json output_json = read_json_file(config.output_file);
  const nlohmann::json page_walk_json = output_json["tlb_analysis"]["page_walk_penalty"];

  EXPECT_EQ(output_json["tlb_analysis"]["status"], "interrupted");
  EXPECT_FALSE(output_json["tlb_analysis"]["conclusions_valid"]);
  EXPECT_FALSE(output_json["tlb_analysis"]["l1_tlb_detection"]["detected"]);
  EXPECT_NE(output_json["tlb_analysis"]["l1_tlb_detection"]["reason"]
                .get<std::string>()
                .find("suppressed"),
            std::string::npos);
  EXPECT_FALSE(page_walk_json["available"]);
  EXPECT_FALSE(page_walk_json.contains("comparison_p50_ns"));
  EXPECT_FALSE(page_walk_json.contains("penalty_ns"));
  EXPECT_NE(page_walk_json["reason"].get<std::string>().find("incomplete"), std::string::npos);

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

TEST(JsonSchemaTest, CoreToCoreJsonBuilderReturnsInMemoryPayload) {
  CoreToCoreLatencyConfig config;
  config.output_file.clear();
  config.loop_count = 1;
  config.latency_sample_count = 2;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {
      {
          Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
          {10.0},
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

  const nlohmann::json output_json = build_core_to_core_latency_json(context);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::CORE_TO_CORE_JSON_MODE_NAME);
  EXPECT_TRUE(output_json.contains("core_to_core_latency"));
  EXPECT_EQ(output_json["core_to_core_latency"]["scenarios"][0]["name"],
            Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY);
}

TEST(JsonSchemaTest, CoreToCoreExporterReturnsSuccessWhenOutputPathIsEmpty) {
  CoreToCoreLatencyConfig config;
  config.output_file.clear();
  config.loop_count = 1;
  config.latency_sample_count = 1;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios;
  const CoreToCoreLatencyJsonContext context = {
      config,
      cpu_name,
      4,
      6,
      100,
      200,
      20,
      scenarios,
      1.0,
  };

  EXPECT_EQ(save_core_to_core_latency_to_json(context), EXIT_SUCCESS);
}

TEST(JsonSchemaTest, CoreToCoreExporterSerializesOneWayValuesAndThreadHints) {
  CoreToCoreLatencyConfig config;
  config.output_file = make_temp_json_path("core2core_values").string();
  config.loop_count = 2;
  config.latency_sample_count = 2;

  const std::string cpu_name = "test-cpu";
  ThreadHintStatus initiator_hint;
  initiator_hint.qos_applied = true;
  initiator_hint.qos_code = 0;
  initiator_hint.affinity_requested = true;
  initiator_hint.affinity_applied = false;
  initiator_hint.affinity_code = 5;
  initiator_hint.affinity_tag = 1;

  ThreadHintStatus responder_hint;
  responder_hint.qos_applied = false;
  responder_hint.qos_code = 6;
  responder_hint.affinity_requested = true;
  responder_hint.affinity_applied = true;
  responder_hint.affinity_code = 0;
  responder_hint.affinity_tag = 2;

  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {
      {
          Constants::CORE_TO_CORE_SCENARIO_SAME_AFFINITY,
          {12.0, 18.0},
          {15.0, 17.0},
          initiator_hint,
          responder_hint,
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

  const nlohmann::json scenario_json = output_json["core_to_core_latency"]["scenarios"][0];
  ASSERT_EQ(scenario_json["one_way_estimate_ns"][JsonKeys::VALUES].size(), 2u);
  EXPECT_DOUBLE_EQ(scenario_json["one_way_estimate_ns"][JsonKeys::VALUES][0].get<double>(), 6.0);
  EXPECT_DOUBLE_EQ(scenario_json["one_way_estimate_ns"][JsonKeys::VALUES][1].get<double>(), 9.0);

  ASSERT_TRUE(scenario_json["thread_hints"].contains("initiator"));
  ASSERT_TRUE(scenario_json["thread_hints"].contains("responder"));
  EXPECT_EQ(scenario_json["thread_hints"]["initiator"]["affinity_tag"], 1);
  EXPECT_EQ(scenario_json["thread_hints"]["responder"]["qos_code"], 6);
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("qos_applied"));
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("qos_code"));
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("affinity_requested"));
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("affinity_applied"));
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("affinity_code"));
  EXPECT_TRUE(scenario_json["thread_hints"]["initiator"].contains("affinity_tag"));

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, CoreToCoreExporterOmitsStatisticsWhenOnlySingleValueExists) {
  CoreToCoreLatencyConfig config;
  config.output_file = make_temp_json_path("core2core_single").string();
  config.loop_count = 1;
  config.latency_sample_count = 1;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {
      {
          Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
          {10.0},
          {10.5},
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

  const nlohmann::json scenario_json = output_json["core_to_core_latency"]["scenarios"][0];
  EXPECT_FALSE(scenario_json["round_trip_ns"].contains(JsonKeys::STATISTICS));
  EXPECT_FALSE(scenario_json["one_way_estimate_ns"].contains(JsonKeys::STATISTICS));
  EXPECT_FALSE(scenario_json[JsonKeys::SAMPLES_NS].contains(JsonKeys::STATISTICS));

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, CoreToCoreExporterReturnsFailureForInvalidOutputPath) {
  CoreToCoreLatencyConfig config;
  config.output_file = "/dev/null/core2core.json";
  config.loop_count = 1;
  config.latency_sample_count = 1;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {
      {
          Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
          {10.0},
          {10.5},
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

  testing::internal::CaptureStderr();
  const int result = save_core_to_core_latency_to_json(context);
  const std::string error_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_FALSE(error_output.empty());
}
