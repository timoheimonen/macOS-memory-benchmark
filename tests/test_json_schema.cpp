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
#include <utility>
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
  record.paired.spread.pilot_access_count = 4096;
  record.paired.spread.pilot_duration_ns = 40960.0;
  record.paired.spread.access_count = 1000000;
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
  record.paired.packed.pilot_access_count = 4096;
  record.paired.packed.pilot_duration_ns = 32768.0;
  record.paired.packed.access_count = 1250000;
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

TEST(JsonSchemaTest, BenchmarkSchemaV2IncludesCompletionAndNullableMeasurements) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("benchmark_v2").string();
  config.run_benchmark = true;
  config.only_bandwidth = true;
  config.buffer_size = 4096;
  config.buffer_size_mb = 1;
  config.benchmark_seed = 18446744073709551615ULL;
  config.user_specified_benchmark_seed = true;

  BenchmarkResults loop;
  loop.status = BenchmarkRunStatus::Partial;
  loop.status_reason = "interrupted by user";
  loop.loop_index = 0;
  loop.planned_phases = 1;
  loop.completed_phases = 0;
  loop.planned_measurements = 3;
  loop.completed_measurements = 1;
  loop.planned_phase_order = {"main-bandwidth"};
  loop.realized_phase_order = {"main-bandwidth"};
  set_measurement_value(loop.main_read_bandwidth, 12.5, 0.150);
  loop.main_read_bandwidth.target = "main-memory";
  loop.main_read_bandwidth.operation = "read";
  loop.main_read_bandwidth.work_policy = "automatic-duration-calibration";
  loop.main_read_bandwidth.automatic_calibration = true;
  loop.main_read_bandwidth.duration_within_target = true;
  loop.main_read_bandwidth.duration_quality = "within-target-window";
  loop.main_read_bandwidth.pilot_elapsed_seconds = 0.010;
  loop.main_read_bandwidth.buffer_size_bytes = 4096;
  loop.main_read_bandwidth.passes = 16;
  loop.main_read_bandwidth.exact_payload_bytes = 65536;
  loop.main_read_bandwidth.requested_threads = 4;
  loop.main_read_bandwidth.effective_threads = 4;
  loop.main_read_bandwidth.created_workers = 4;
  loop.main_read_bandwidth.qos_outcome = "applied-to-all-workers";
  loop.main_read_bandwidth.qos_successful_workers = 4;
  set_measurement_unavailable(loop.main_write_bandwidth,
                              BenchmarkMeasurementStatus::Interrupted,
                              "interrupted during measured operation");

  BenchmarkStatistics stats;
  stats.status = BenchmarkRunStatus::Partial;
  stats.status_reason = "interrupted by user";
  stats.planned_loops = 2;
  stats.completed_loops = 0;
  stats.planned_measurements = 6;
  stats.completed_measurements = 1;
  stats.loop_results.push_back(loop);

  ASSERT_EQ(save_results_to_json(config, stats, 1.0), EXIT_SUCCESS);
  const nlohmann::json output = read_json_file(config.output_file);
  EXPECT_EQ(output["configuration"]["benchmark_schema_version"], 2);
  EXPECT_EQ(output["configuration"]["methodology_version"],
            "benchmark-v2-calibrated-seeded-balanced");
  EXPECT_DOUBLE_EQ(
      output["configuration"]["latency_calibration_window_max_seconds"],
      0.300);
  EXPECT_EQ(output["configuration"]["benchmark_seed"],
            "18446744073709551615");
  EXPECT_EQ(output["status"], "partial");
  EXPECT_FALSE(output["results_complete"].get<bool>());
  EXPECT_EQ(output["planned_loops"], 2u);
  ASSERT_EQ(output["loops"].size(), 1u);
  const nlohmann::json measurements = output["loops"][0]["measurements"];
  EXPECT_EQ(measurements["main_read_bandwidth"]["status"], "measured");
  EXPECT_DOUBLE_EQ(measurements["main_read_bandwidth"]["value"].get<double>(),
                   12.5);
  EXPECT_EQ(measurements["main_read_bandwidth"]["passes"], 16u);
  EXPECT_EQ(measurements["main_read_bandwidth"]["qos_outcome"],
            "applied-to-all-workers");
  EXPECT_EQ(measurements["main_read_bandwidth"]["qos_successful_workers"],
            4u);
  EXPECT_EQ(measurements["main_read_bandwidth"]["created_workers"], 4);
  EXPECT_EQ(measurements["main_write_bandwidth"]["status"], "interrupted");
  EXPECT_TRUE(measurements["main_write_bandwidth"]["value"].is_null());
  EXPECT_EQ(output["main_memory"]["bandwidth"]["read_gb_s"]["value"], 12.5);
  EXPECT_TRUE(output["main_memory"]["bandwidth"]["write_gb_s"]["value"].is_null());
  EXPECT_FALSE(output.dump().find("page_walk_penalty_ns") != std::string::npos);

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, BenchmarkAggregateHeadlineUsesMedianAndReportsCvAndMad) {
  BenchmarkConfig config;
  config.only_bandwidth = true;
  config.buffer_size = 4096;
  BenchmarkStatistics stats;
  stats.status = BenchmarkRunStatus::Complete;
  stats.planned_loops = 3;
  stats.completed_loops = 3;
  stats.planned_measurements = 9;
  stats.completed_measurements = 9;
  for (size_t index = 0; index < 3; ++index) {
    BenchmarkResults loop;
    loop.status = BenchmarkRunStatus::Complete;
    loop.loop_index = index;
    const double value = index == 0 ? 10.0 : index == 1 ? 20.0 : 100.0;
    set_measurement_value(loop.main_read_bandwidth, value, 0.150);
    loop.main_read_bandwidth.duration_within_target = true;
    stats.loop_results.push_back(loop);
  }

  const nlohmann::json output = build_results_json(config, stats, 1.0);
  const nlohmann::json aggregate =
      output["main_memory"]["bandwidth"]["read_gb_s"];
  EXPECT_DOUBLE_EQ(aggregate["value"].get<double>(), 20.0);
  EXPECT_EQ(aggregate["headline_semantics"],
            "median-p50-across-loop-headlines");
  EXPECT_TRUE(aggregate["statistics"].contains(
      "coefficient_of_variation_pct"));
  EXPECT_TRUE(aggregate["statistics"].contains(
      "median_absolute_deviation"));
  EXPECT_DOUBLE_EQ(
      aggregate["statistics"]["median_absolute_deviation"].get<double>(),
      10.0);
}

TEST(JsonSchemaTest, BenchmarkCheckpointAtomicallyProgressesToComplete) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("benchmark_checkpoint").string();
  config.only_bandwidth = true;
  config.only_latency = true;
  BenchmarkStatistics stats;
  stats.status = BenchmarkRunStatus::Partial;
  stats.status_reason = "benchmark loops remain";
  stats.planned_loops = 2;
  stats.completed_loops = 1;

  ASSERT_EQ(save_results_to_json(config, stats, 0.5, false), EXIT_SUCCESS);
  nlohmann::json output = read_json_file(config.output_file);
  EXPECT_EQ(output["status"], "partial");
  EXPECT_FALSE(output["results_complete"].get<bool>());

  stats.status = BenchmarkRunStatus::Complete;
  stats.status_reason.clear();
  stats.completed_loops = 2;
  ASSERT_EQ(save_results_to_json(config, stats, 1.0, false), EXIT_SUCCESS);
  output = read_json_file(config.output_file);
  EXPECT_EQ(output["status"], "complete");
  EXPECT_TRUE(output["results_complete"].get<bool>());
  EXPECT_FALSE(std::filesystem::exists(config.output_file + ".tmp"));

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, PatternExporterIncludesPatternsMode) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("patterns_schema").string();
  config.pattern_seed = 42;
  config.user_specified_pattern_seed = true;

  PatternStatistics stats;
  ASSERT_EQ(save_pattern_results_to_json(config, stats, 2.0), EXIT_SUCCESS);

  const nlohmann::json output_json = read_json_file(config.output_file);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::PATTERNS_JSON_MODE_NAME);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["methodology_version"],
            "pattern-v2-phase-calibrated-seeded");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["pattern_seed"], "42");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["thread_selection_policy"],
            "detected-core-count-default");

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, PatternMeasurementsIncludeStatusAndMethodologyMetadata) {
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("patterns_metadata").string();
  config.pattern_seed = 18446744073709551615ULL;
  config.user_specified_pattern_seed = true;

  PatternResults loop;
  PatternMeasurement measured;
  measured.status = PatternMeasurementStatus::Measured;
  measured.status_reason.clear();
  measured.bandwidth_gb_s = 12.5;
  measured.elapsed_seconds = 0.150;
  measured.pilot_elapsed_seconds = 0.010;
  measured.access_size_bytes = 32;
  measured.requested_threads = 4;
  measured.effective_threads = 4;
  measured.accesses_per_pass = 100;
  measured.min_accesses_per_pass = 100;
  measured.max_accesses_per_pass = 100;
  measured.passes = 10;
  measured.total_accesses = 1000;
  measured.total_payload_bytes = 32000;
  measured.distinct_address_count = 100;
  measured.logical_working_set_bytes = 4096;
  measured.native_page_size_bytes = 16384;
  measured.automatic_calibration = true;
  measured.benchmark_loop_index = 3;
  measured.pattern_order_index = 2;
  set_pattern_measurement(loop, PatternKind::SequentialForward,
                          PatternOperation::Read, measured);

  PatternMeasurement phased = measured;
  phased.stride_bytes = Constants::PATTERN_STRIDE_CACHE_LINE;
  phased.accesses_per_pass = 2;
  phased.min_accesses_per_pass = 1;
  phased.max_accesses_per_pass = 2;
  phased.phase_period_passes = 2;
  set_pattern_measurement(loop, PatternKind::Strided64,
                          PatternOperation::Read, std::move(phased));

  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    PatternMeasurement skipped;
    skipped.status = PatternMeasurementStatus::Skipped;
    skipped.status_reason = "buffer cannot provide a valid stride transition";
    skipped.access_size_bytes = 32;
    skipped.stride_bytes = Constants::PATTERN_STRIDE_SUPERPAGE_2MB;
    skipped.requested_threads = 4;
    skipped.native_page_size_bytes = 16384;
    set_pattern_measurement(loop, PatternKind::Strided2MiB, operation,
                            std::move(skipped));
  }

  PatternMeasurement random = measured;
  random.has_seed = true;
  random.seed = config.pattern_seed;
  set_pattern_measurement(loop, PatternKind::Random, PatternOperation::Read,
                          std::move(random));

  PatternStatistics statistics;
  statistics.loop_results.push_back(loop);
  statistics.all_forward_read_bw.push_back(12.5);
  statistics.all_random_read_bw.push_back(12.5);

  ASSERT_EQ(save_pattern_results_to_json(config, statistics, 1.0), EXIT_SUCCESS);
  const nlohmann::json output = read_json_file(config.output_file);
  const nlohmann::json forward =
      output[JsonKeys::PATTERNS][JsonKeys::SEQUENTIAL_FORWARD];
  EXPECT_EQ(forward["methodology_version"],
            "pattern-v2-phase-calibrated-seeded");
  EXPECT_EQ(forward["warmup_semantics"], "steady-state-same-shape");
  const nlohmann::json read =
      forward[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S];
  EXPECT_EQ(read["status"], "measured");
  EXPECT_DOUBLE_EQ(read["value_gb_s"].get<double>(), 12.5);
  ASSERT_EQ(read["measurements"].size(), 1u);
  EXPECT_EQ(read["measurements"][0]["passes"], 10u);
  EXPECT_EQ(read["measurements"][0]["accesses_per_pass_semantics"],
            "constant-count");
  EXPECT_EQ(read["measurements"][0]["total_payload_bytes"], 32000u);
  EXPECT_EQ(read["measurements"][0]["benchmark_loop_index"], 3u);
  EXPECT_EQ(read["measurements"][0]["pattern_order_index"], 2u);
  EXPECT_EQ(output[JsonKeys::CONFIGURATION]["pattern_execution_order_policy"],
            "cyclic-latin-square-across-count-loops");
  const nlohmann::json phased_json =
      output[JsonKeys::PATTERNS][JsonKeys::STRIDED_64][JsonKeys::BANDWIDTH]
            [JsonKeys::READ_GB_S]["measurements"][0];
  EXPECT_EQ(phased_json["accesses_per_pass_semantics"], "phase-zero-count");
  EXPECT_EQ(phased_json["min_accesses_per_pass"], 1u);
  EXPECT_EQ(phased_json["max_accesses_per_pass"], 2u);
  EXPECT_EQ(phased_json["phase_period_passes"], 2u);

  const nlohmann::json skipped = output[JsonKeys::PATTERNS][JsonKeys::STRIDED_2MB]
      [JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S];
  EXPECT_EQ(skipped["status"], "skipped");
  EXPECT_TRUE(skipped["value_gb_s"].is_null());
  EXPECT_EQ(skipped["measurements"][0]["accesses_per_pass_semantics"],
            "phase-zero-count");
  EXPECT_FALSE(output[JsonKeys::PATTERNS][JsonKeys::STRIDED_2MB]
                   ["large_page_backing_verified"]
                       .get<bool>());

  const nlohmann::json random_json =
      output[JsonKeys::PATTERNS][JsonKeys::RANDOM];
  EXPECT_EQ(random_json["seed"], "18446744073709551615");

  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, TlbAnalysisExporterIncludesModeAndCoreCounts) {
  constexpr uint64_t kConfigSeed = 18446744073709551615ULL;
  constexpr uint64_t kFirstTaskSeed = 18446744073709551613ULL;
  BenchmarkConfig config;
  config.output_file = make_temp_json_path("tlb_schema").string();
  config.tlb_sweep_density = TlbSweepDensity::High;
  config.tlb_seed = kConfigSeed;
  config.user_specified_tlb_seed = true;
  config.main_thread_qos_requested = true;
  config.main_thread_qos_applied = true;
  config.main_thread_qos_code = 0;

  const std::string cpu_name = "test-cpu";
  const std::vector<size_t> localities_bytes = {16 * Constants::BYTES_PER_KB};
  const std::vector<std::vector<double>> sweep_loop_latencies_ns = {{15.0, 15.1}};
  const std::vector<double> p50_latency_ns = {15.0};
  const std::vector<double> page_walk_comparison_loop_latencies_ns = {95.0, 96.0};
  const std::vector<TlbSweepPoint> sweep_points = {{
      0, 1, 1, 16 * Constants::BYTES_PER_KB, 64, 256, "base",
      16 * Constants::BYTES_PER_KB, 16 * Constants::BYTES_PER_KB}};
  std::vector<TlbMeasurementRecord> measurement_records = {
      make_paired_tlb_record(TlbMeasurementPass::Base,
                             0,
                             16 * Constants::BYTES_PER_KB,
                             0,
                             0,
                             kFirstTaskSeed,
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
                             100.0),
      make_paired_tlb_record(TlbMeasurementPass::LargeLocality,
                             1,
                             512 * Constants::BYTES_PER_MB,
                             2,
                             0,
                             202,
                             101.0),
  };
  measurement_records[4].paired.packed.latency_ns = 94.0;
  measurement_records[4].paired.translation_delta_ns = 1.0;
  measurement_records[5].paired.packed.latency_ns = 90.0;
  measurement_records[5].paired.translation_delta_ns = 10.0;
  measurement_records[6].paired.packed.latency_ns = 100.0;
  measurement_records[6].paired.translation_delta_ns = 1.0;
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
  const TlbRuntimeProfile runtime_profile = {
      "exhaustive", 2, 4, 20 * 1000 * 1000ULL, 32, 5 * 1000 * 1000,
      0.15, 1000};
  const TlbWorkEstimate base_work_estimate = estimate_tlb_work(
      1, 1050 * Constants::BYTES_PER_MB, 1, runtime_profile);
  const std::vector<TlbPassExecutionSummary> pass_summaries = {
      {TlbMeasurementPass::Base, 1, 2, true, true},
      {TlbMeasurementPass::Validation, 1, 2, true, true},
      {TlbMeasurementPass::LargeLocality, 1, 2, true, true},
  };

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
      runtime_profile,
      4096,
      1228,
      1050 * Constants::BYTES_PER_MB,
      0,
      "",
      base_work_estimate,
      pass_summaries,
  };

  ASSERT_EQ(save_tlb_analysis_to_json(context), EXIT_SUCCESS);
  const nlohmann::json output_json = read_json_file(config.output_file);

  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::MODE], Constants::TLB_ANALYSIS_JSON_MODE_NAME);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::PERFORMANCE_CORES], 4);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::EFFICIENCY_CORES], 6);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["schema_version"], 4);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["methodology_version"],
            "page-native-paired-adaptive-validated-v4");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["boundary_signal"],
            "translation_delta_ns");
  EXPECT_NE(output_json[JsonKeys::CONFIGURATION]["latency_interpretation"]
                .get<std::string>()
                .find("not direct DRAM latency"),
            std::string::npos);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["seed"],
            "18446744073709551615");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["seed_encoding"],
            "uint64-decimal-string");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["seed_source"], "user");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["seed_derivation"].contains(
      "measurement_task"));
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["seed_derivation"].contains(
      "chain_layout"));
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["main_thread_qos"]["requested"]);
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["main_thread_qos"]["applied"]);
  EXPECT_FALSE(output_json[JsonKeys::CONFIGURATION].contains(
      "schema_compatibility"));
  EXPECT_FALSE(output_json[JsonKeys::CONFIGURATION].contains(
      JsonKeys::LATENCY_SAMPLE_COUNT));
  EXPECT_FALSE(output_json[JsonKeys::CONFIGURATION].contains(
      "buffer_locked"));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["schedule_policy"],
            "seeded-cyclic-latin");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["latency_chain_mode_requested"], "auto");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::LATENCY_CHAIN_MODE));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::LATENCY_CHAIN_MODE], "random-box");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["chain_mode_comparability"]
                  .get<std::string>()
                  .find("identical effective chain mode") != std::string::npos);
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains(JsonKeys::TLB_DENSITY));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION][JsonKeys::TLB_DENSITY], "high");
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION].contains("fine_sweep_added_points"));
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["runtime_profile"],
            "exhaustive");
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["adaptive_rounds"]["minimum"],
            2);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["adaptive_rounds"]["maximum"],
            4);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["memory_budget"]["budget_mb"],
            1228);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["buffer_lock"]["errno"], 0);
  EXPECT_EQ(output_json[JsonKeys::CONFIGURATION]["base_work_estimate"]["point_count"],
            1);
  EXPECT_TRUE(output_json[JsonKeys::CONFIGURATION]["base_work_estimate"].contains(
      "maximum_pilot_accesses_per_measurement"));
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
  EXPECT_TRUE(output_json["tlb_analysis"]["validation_required"]);
  EXPECT_EQ(output_json["tlb_analysis"]["validation_status"], "complete");
  EXPECT_TRUE(output_json["tlb_analysis"]["validation_complete"]);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["minimum_planned_base_validation_pairs"],
            4);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["maximum_planned_base_validation_pairs"],
            8);
  EXPECT_EQ(output_json["tlb_analysis"]["planned_base_validation_pairs"],
            8);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["planned_base_validation_raw_measurements"],
            16);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["completed_base_validation_measurement_records"],
            4);
  EXPECT_EQ(output_json["tlb_analysis"]["completed_base_validation_pairs"],
            4);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["completed_base_validation_raw_measurements"],
            8);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["completed_large_locality_measurement_records"],
            3);
  EXPECT_EQ(output_json["tlb_analysis"]["completed_large_locality_pairs"],
            3);
  EXPECT_EQ(output_json["tlb_analysis"]
                       ["completed_large_locality_raw_measurements"],
            6);
  EXPECT_EQ(output_json["tlb_analysis"]["total_completed_measurement_records"],
            7);
  EXPECT_EQ(output_json["tlb_analysis"]["total_completed_measurement_pairs"],
            7);
  EXPECT_EQ(output_json["tlb_analysis"]["total_completed_raw_measurements"],
            14);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_counter_scope"]
                       ["total"],
            "all serialized measurement passes");
  EXPECT_FALSE(output_json["tlb_analysis"].contains(
      "completed_measurement_pairs"));
  EXPECT_FALSE(output_json["tlb_analysis"].contains(
      "completed_raw_measurements"));
  EXPECT_TRUE(output_json["tlb_analysis"]["conclusions_valid"]);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"].size(), 7u);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]["pass"], "base");
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                       ["seed"],
            "18446744073709551613");
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                       ["paired_control"]["spread"]["seed"],
            "18446744073709551614");
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                       ["paired_control"]["packed"]["seed"],
            "18446744073709551615");
  EXPECT_FALSE(output_json["tlb_analysis"]["measurement_records"][0]
                           .contains("latency_ns"));
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][2]["pass"],
            "validation");
  EXPECT_TRUE(output_json["tlb_analysis"]["measurement_records"][0]
                         ["paired_control"]["available"]);
  EXPECT_DOUBLE_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                              ["paired_control"]["translation_delta_ns"],
                   5.0);
  EXPECT_EQ(output_json["tlb_analysis"]["measurement_records"][0]
                       ["paired_control"]["spread"]["access_count"],
            1000000);
  ASSERT_EQ(output_json["tlb_analysis"]["pass_summaries"].size(), 3u);
  EXPECT_TRUE(output_json["tlb_analysis"]["pass_summaries"][0]["converged"]);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["requested_pages"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["actual_pages"], 1);
  EXPECT_FALSE(output_json["tlb_analysis"]["sweep"][0].contains(
      "pointer_count"));
  EXPECT_FALSE(output_json["tlb_analysis"]["sweep"][0].contains(
      "actual_node_count"));
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["pointer_nodes"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]
                       ["spread_pointers_per_page_max"],
            1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]
                       ["packed_pointers_per_page_max"],
            1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["actual_unique_cache_lines"], 1);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]
                       ["active_cache_line_footprint_bytes"],
            Constants::CACHE_LINE_SIZE_BYTES);
  EXPECT_TRUE(output_json["tlb_analysis"]["sweep"][0]
                         ["short_cycle_diagnostic"]);
  EXPECT_FALSE(output_json["tlb_analysis"]["sweep"][0].contains(
      "p50_latency_ns"));
  EXPECT_FALSE(output_json["tlb_analysis"]["sweep"][0].contains(
      "loop_latencies_ns"));
  EXPECT_DOUBLE_EQ(output_json["tlb_analysis"]["sweep"][0]
                              ["translation_delta_p50_ns"],
                   5.0);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["measurements"].size(), 4u);
  EXPECT_EQ(output_json["tlb_analysis"]["sweep"][0]["measurements"][1]["round_index"], 1);
  EXPECT_TRUE(output_json["tlb_analysis"]["sweep"][0].contains(
      "validation_translation_deltas_ns"));
  const nlohmann::json spread_chain =
      output_json["tlb_analysis"]["sweep"][0]["spread_chain"];
  EXPECT_EQ(spread_chain["pointer_nodes"], 1);
  EXPECT_EQ(spread_chain["pointers_per_page_max"], 1);
  EXPECT_FALSE(spread_chain.contains("node_count"));
  EXPECT_FALSE(spread_chain.contains("max_nodes_per_page"));
  ASSERT_TRUE(output_json["tlb_analysis"].contains(
      "large_locality_paired_comparison"));
  EXPECT_FALSE(output_json["tlb_analysis"].contains(
      "large_locality_latency_delta"));
  EXPECT_FALSE(output_json["tlb_analysis"].contains("page_walk_penalty"));
  const nlohmann::json paired_large =
      output_json["tlb_analysis"]["large_locality_paired_comparison"];
  EXPECT_TRUE(paired_large["available"]);
  EXPECT_DOUBLE_EQ(paired_large["spread_p50_ns"], 100.0);
  EXPECT_DOUBLE_EQ(paired_large["packed_p50_ns"], 94.0);
  EXPECT_DOUBLE_EQ(paired_large["translation_delta_p50_ns"], 1.0);
  EXPECT_NE(paired_large["translation_delta_p50_ns"].get<double>(),
            paired_large["spread_p50_ns"].get<double>() -
                paired_large["packed_p50_ns"].get<double>());
  EXPECT_EQ(paired_large["spread_actual_pages"], 32768);
  EXPECT_EQ(paired_large["packed_actual_pages"], 128);
  EXPECT_EQ(paired_large["unique_cache_lines"], 32768);
  EXPECT_EQ(paired_large["active_cache_line_footprint_bytes"],
            2 * Constants::BYTES_PER_MB);
  EXPECT_EQ(paired_large["pointer_nodes"], 32768);
  EXPECT_FALSE(paired_large.contains("node_count"));
  EXPECT_EQ(paired_large["measurements"].size(), 3u);
  EXPECT_NE(paired_large["interpretation"].get<std::string>().find(
                "not DRAM latency"),
            std::string::npos);
  std::filesystem::remove(config.output_file);
}

TEST(JsonSchemaTest, TlbAnalysisExporterOmitsRemovedAliasesAndHandlesUnavailableComparison) {
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

  TlbAnalysisJsonContext context = {
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
  const nlohmann::json paired_large =
      output_json["tlb_analysis"]["large_locality_paired_comparison"];

  EXPECT_EQ(output_json["tlb_analysis"]["status"], "interrupted");
  EXPECT_FALSE(output_json["tlb_analysis"]["conclusions_valid"]);
  EXPECT_TRUE(output_json["tlb_analysis"]["validation_required"]);
  EXPECT_EQ(output_json["tlb_analysis"]["validation_status"], "not-run");
  EXPECT_FALSE(output_json["tlb_analysis"]["validation_complete"]);
  EXPECT_FALSE(output_json["tlb_analysis"]["l1_tlb_detection"]["detected"]);
  EXPECT_NE(output_json["tlb_analysis"]["l1_tlb_detection"]["reason"]
                .get<std::string>()
                .find("suppressed"),
            std::string::npos);
  EXPECT_FALSE(output_json["tlb_analysis"].contains("page_walk_penalty"));
  EXPECT_FALSE(output_json["tlb_analysis"].contains(
      "large_locality_latency_delta"));
  EXPECT_FALSE(paired_large["available"]);
  EXPECT_FALSE(paired_large.contains("translation_delta_p50_ns"));
  EXPECT_NE(paired_large["reason"].get<std::string>().find("incomplete"),
            std::string::npos);

  std::filesystem::remove(config.output_file);

  config.output_file = make_temp_json_path("tlb_256mb_fallback").string();
  context.analysis_status = "complete";
  context.conclusions_valid = true;
  context.validation_planned_points = 0;
  context.validation_measured_points = 0;
  context.validation_complete = true;
  context.selected_buffer_mb = 256;
  context.can_measure_page_walk_penalty = false;
  ASSERT_EQ(save_tlb_analysis_to_json(context), EXIT_SUCCESS);
  const nlohmann::json fallback_json = read_json_file(config.output_file);
  const nlohmann::json fallback_large =
      fallback_json["tlb_analysis"]["large_locality_paired_comparison"];
  EXPECT_FALSE(fallback_json["tlb_analysis"]["validation_required"]);
  EXPECT_EQ(fallback_json["tlb_analysis"]["validation_status"],
            "not-required");
  EXPECT_TRUE(fallback_json["tlb_analysis"]["validation_complete"]);
  EXPECT_FALSE(fallback_large["available"]);
  EXPECT_NE(fallback_large["reason"].get<std::string>().find("512 MiB"),
            std::string::npos);

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

TEST(JsonSchemaTest, CoreToCoreV2SerializesCalibratedBalancedAuditTrail) {
  CoreToCoreLatencyConfig config;
  config.loop_count = 1;
  config.latency_sample_count = 2;

  ThreadHintStatus initiator_hint;
  initiator_hint.qos_applied = true;
  initiator_hint.affinity_requested = true;
  initiator_hint.affinity_applied = true;
  initiator_hint.affinity_tag = 1;
  ThreadHintStatus responder_hint = initiator_hint;

  CoreToCoreLoopRecord loop_record;
  loop_record.loop_index = 0;
  loop_record.order_position = 1;
  loop_record.status = CoreToCoreMeasurementStatus::Measured;
  loop_record.round_trip_ns = 70.0;
  loop_record.headline_elapsed_seconds = 0.250;
  loop_record.duration_quality = "within-target-window";
  loop_record.sample_start_index = 0;
  loop_record.completed_sample_windows = 2;
  loop_record.initiator_hint = initiator_hint;
  loop_record.responder_hint = responder_hint;

  CoreToCoreLatencyScenarioResult scenario;
  scenario.scenario_name = Constants::CORE_TO_CORE_SCENARIO_SAME_AFFINITY;
  scenario.loop_round_trip_ns = {70.0};
  scenario.sample_round_trip_ns = {69.0, 71.0};
  scenario.initiator_hint = initiator_hint;
  scenario.responder_hint = responder_hint;
  scenario.work_plan = {true, 100000, 0.007, 70.0, 350000, 3500000, 14000};
  scenario.loop_records = {loop_record};
  scenario.status = CoreToCoreMeasurementStatus::Measured;
  scenario.planned_loops = 1;
  scenario.completed_loops = 1;

  const std::string cpu_name = "test-cpu";
  const std::vector<CoreToCoreLatencyScenarioResult> scenarios = {scenario};
  const CoreToCoreLatencyJsonContext context = {
      config, cpu_name, 4, 6, 20000, 1000000, 2000, scenarios, 1.5,
      "complete", 1, 1};

  const nlohmann::json output = build_core_to_core_latency_json(context);
  EXPECT_EQ(output[JsonKeys::CONFIGURATION]["schema_version"], 2);
  EXPECT_EQ(output[JsonKeys::CONFIGURATION]["methodology_version"],
            "core2core-v2-calibrated-balanced-auditable");
  EXPECT_EQ(output[JsonKeys::CONFIGURATION]["scenario_schedule"],
            "cyclic-latin-square-across-count-loops");
  EXPECT_EQ(output[JsonKeys::CONFIGURATION]["calibration_warmup_round_trips"],
            Constants::CORE_TO_CORE_CALIBRATION_WARMUP_ROUND_TRIPS);

  const nlohmann::json result = output["core_to_core_latency"];
  EXPECT_EQ(result["status"], "complete");
  EXPECT_TRUE(result["measurements_complete"]);
  EXPECT_TRUE(result["affinity_hint_comparison_interpretable"]);
  const nlohmann::json scenario_json = result["scenarios"][0];
  EXPECT_EQ(scenario_json["headline_round_trip_ns"], 70.0);
  EXPECT_EQ(scenario_json["work_plan"]["headline_round_trips"], 3500000u);
  EXPECT_EQ(scenario_json["loop_records"][0]["order_position"], 1u);
  EXPECT_EQ(scenario_json["loop_records"][0]["sample_window_range"]["count"], 2u);
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
