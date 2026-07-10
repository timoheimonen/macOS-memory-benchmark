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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "third_party/nlohmann/json.hpp"

namespace {

struct CliResult {
  int exit_code = -1;
  std::string output;
};

CliResult run_memory_benchmark(const std::string& args) {
  CliResult result;
  const std::string command = "./memory_benchmark " + args + " 2>&1";

  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return result;
  }

  std::array<char, 4096> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    result.output += buffer.data();
  }

  const int status = pclose(pipe);
  if (status != -1 && WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }
  return result;
}

std::string make_temp_json_path(const std::string& test_name) {
  return "/tmp/membenchmark_cli_" + std::to_string(getpid()) + "_" + test_name + ".json";
}

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

}  // namespace

TEST(ExecutableCliIntegrationTest, NoArgumentsShowsHelpAndReturnsSuccessIntegration) {
  const CliResult result = run_memory_benchmark("");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--benchmark"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, HelpFlagShowsHelpAndReturnsSuccessIntegration) {
  const CliResult result = run_memory_benchmark("-h");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--patterns"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, OptionsWithoutModeShowHelpAndReturnSuccessIntegration) {
  const CliResult result = run_memory_benchmark("--threads 1");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--threads <count>"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, InvalidStandardModeConfigReturnsFailureIntegration) {
  const CliResult result = run_memory_benchmark("--benchmark --only-bandwidth --latency-samples 1");

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  EXPECT_NE(result.output.find("--only-bandwidth cannot be used with --latency-samples"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, CoreToCoreArgumentsAreRoutedBeforeNormalParserIntegration) {
  const CliResult result = run_memory_benchmark("--analyze-core2core --buffer-size 256");

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  EXPECT_NE(result.output.find("--analyze-core2core allows only optional"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, CoreToCoreWritesCalibratedAuditJsonIntegration) {
  const std::string output_path = make_temp_json_path("core2core_v2");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark(
      "--analyze-core2core --count 1 --latency-samples 1 --output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  const nlohmann::json output = nlohmann::json::parse(read_file(output_path));
  EXPECT_EQ(output["version"], "0.58.1");
  EXPECT_EQ(output["configuration"]["schema_version"], 2);
  EXPECT_EQ(output["core_to_core_latency"]["status"], "complete");
  EXPECT_TRUE(output["core_to_core_latency"]["measurements_complete"]);
  ASSERT_EQ(output["core_to_core_latency"]["scenarios"].size(), 3u);
  EXPECT_EQ(output["core_to_core_latency"]["scenarios"][0]["loop_records"].size(), 1u);

  std::remove(output_path.c_str());
}

TEST(ExecutableCliIntegrationTest, CoreToCoreSweepWritesCompletionMetadataIntegration) {
  const std::string output_path = make_temp_json_path("core2core_sweep_v2");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark(
      "--analyze-core2core --count 1 --sweep latency-samples=1,2 "
      "--sweep-max-runs 2 --output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  const nlohmann::json output = nlohmann::json::parse(read_file(output_path));
  EXPECT_EQ(output["version"], "0.58.1");
  EXPECT_EQ(output["status"], "complete");
  EXPECT_EQ(output["planned_runs"], 2u);
  EXPECT_EQ(output["completed_runs"], 2u);
  EXPECT_TRUE(output["conclusions_valid"]);
  ASSERT_EQ(output["runs"].size(), 2u);
  EXPECT_EQ(output["runs"][0]["result"]["configuration"]["schema_version"], 2);

  std::remove(output_path.c_str());
}

TEST(ExecutableCliIntegrationTest, AnalyzeTlbInvalidStrideSweepFailsBeforeExecutionIntegration) {
  const std::string output_path = make_temp_json_path("tlb_invalid_stride_sweep");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark(
      "--analyze-tlb --sweep latency-stride-bytes=64,130 --output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  EXPECT_NE(result.output.find("must be a multiple of 8 bytes"), std::string::npos);
  EXPECT_EQ(result.output.find("Running sweep"), std::string::npos);
  EXPECT_EQ(access(output_path.c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest, StandardBenchmarkWritesJsonIntegration) {
  const std::string output_path = make_temp_json_path("standard");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark("--benchmark --only-bandwidth --buffer-size 1 --iterations 1 "
                                                "--count 1 --threads 1 --output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Running benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Results saved to:"), std::string::npos);

  const std::string json = read_file(output_path);
  EXPECT_NE(json.find("\"mode\": \"benchmark\""), std::string::npos);
  EXPECT_NE(json.find("\"main_memory\""), std::string::npos);
  EXPECT_NE(json.find("\"bandwidth\""), std::string::npos);

  std::remove(output_path.c_str());
}

TEST(ExecutableCliIntegrationTest, StandardSweepWritesCompletionMetadataIntegration) {
  const std::string output_path = make_temp_json_path("standard_sweep_status");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark(
      "--benchmark --only-bandwidth --iterations 1 --count 1 --threads 1 "
      "--sweep buffer-size=1,2 --sweep-max-runs 2 --output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  const std::string json = read_file(output_path);
  EXPECT_NE(json.find("\"status\": \"complete\""), std::string::npos);
  EXPECT_NE(json.find("\"planned_runs\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"completed_runs\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"conclusions_valid\": true"), std::string::npos);

  std::remove(output_path.c_str());
}

TEST(ExecutableCliIntegrationTest, PatternModeRunsPatternOrchestrationIntegration) {
  const CliResult result = run_memory_benchmark("--patterns --buffer-size 1 --iterations 1 --count 1");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Running Pattern Benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Sequential Forward:"), std::string::npos);
  EXPECT_EQ(result.output.find("Pattern Efficiency Analysis:"), std::string::npos);
  EXPECT_EQ(result.output.find("Prefetcher effectiveness"), std::string::npos);
  EXPECT_EQ(result.output.find("TLB pressure"), std::string::npos);
  EXPECT_NE(result.output.find("2 MiB stride"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, PatternSeedReproducesWorkloadMetadataIntegration) {
  const std::string first_path = make_temp_json_path("patterns_seed_first");
  const std::string second_path = make_temp_json_path("patterns_seed_second");
  std::remove(first_path.c_str());
  std::remove(second_path.c_str());

  const std::string common_args =
      "--patterns --buffer-size 1 --iterations 1 --count 2 --threads 2 --seed 42 --output ";
  const CliResult first_run = run_memory_benchmark(common_args + first_path);
  const CliResult second_run = run_memory_benchmark(common_args + second_path);
  ASSERT_EQ(first_run.exit_code, EXIT_SUCCESS);
  ASSERT_EQ(second_run.exit_code, EXIT_SUCCESS);

  const nlohmann::json first = nlohmann::json::parse(read_file(first_path));
  const nlohmann::json second = nlohmann::json::parse(read_file(second_path));
  EXPECT_EQ(first["configuration"]["pattern_seed"], "42");
  EXPECT_EQ(first["configuration"]["pattern_seed"],
            second["configuration"]["pattern_seed"]);

  for (const char* operation : {"read_gb_s", "write_gb_s", "copy_gb_s"}) {
    const nlohmann::json& first_samples =
        first["patterns"]["random"]["bandwidth"][operation]["measurements"];
    const nlohmann::json& second_samples =
        second["patterns"]["random"]["bandwidth"][operation]["measurements"];
    ASSERT_EQ(first_samples.size(), 2u);
    ASSERT_EQ(second_samples.size(), first_samples.size());
    for (size_t sample = 0; sample < first_samples.size(); ++sample) {
      EXPECT_EQ(first_samples[sample]["seed"], "42");
      EXPECT_EQ(first_samples[sample]["seed"], second_samples[sample]["seed"]);
      EXPECT_EQ(first_samples[sample]["accesses_per_pass"],
                second_samples[sample]["accesses_per_pass"]);
      EXPECT_EQ(first_samples[sample]["distinct_address_count"],
                second_samples[sample]["distinct_address_count"]);
      EXPECT_EQ(first_samples[sample]["logical_working_set_bytes"],
                second_samples[sample]["logical_working_set_bytes"]);
      EXPECT_EQ(first_samples[sample]["pattern_order_index"],
                second_samples[sample]["pattern_order_index"]);
    }
  }

  std::remove(first_path.c_str());
  std::remove(second_path.c_str());
}

TEST(ExecutableCliIntegrationTest, PatternAutomaticCalibrationWritesMetadataIntegration) {
  const std::string output_path = make_temp_json_path("patterns_calibration");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark(
      "--patterns --buffer-size 1 --count 1 --threads 1 --seed 42 --output " +
      output_path);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("automatic duration calibration"),
            std::string::npos);

  const nlohmann::json output = nlohmann::json::parse(read_file(output_path));
  EXPECT_EQ(output["configuration"]["pattern_pass_policy"],
            "automatic-duration-calibration");
  EXPECT_EQ(output["configuration"]["calibration_max_corrections"], 2u);
  for (const auto& pattern : output["patterns"].items()) {
    for (const auto& operation : pattern.value()["bandwidth"].items()) {
      const nlohmann::json& aggregate = operation.value();
      ASSERT_EQ(aggregate["measurements"].size(), 1u);
      const nlohmann::json& measurement = aggregate["measurements"][0];
      if (measurement["status"] == "measured") {
        EXPECT_TRUE(measurement["automatic_calibration"].get<bool>());
        EXPECT_GT(measurement["pilot_elapsed_seconds"].get<double>(), 0.0);
        EXPECT_GT(measurement["elapsed_seconds"].get<double>(), 0.0);
        EXPECT_GT(measurement["passes"].get<size_t>(), 0u);
      }
    }
  }

  std::remove(output_path.c_str());
}
