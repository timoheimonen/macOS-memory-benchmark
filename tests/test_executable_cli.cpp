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
  EXPECT_NE(result.output.find("-benchmark"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, HelpFlagShowsHelpAndReturnsSuccessIntegration) {
  const CliResult result = run_memory_benchmark("-h");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("-patterns"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, OptionsWithoutModeShowHelpAndReturnSuccessIntegration) {
  const CliResult result = run_memory_benchmark("-threads 1");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("-threads <count>"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, InvalidStandardModeConfigReturnsFailureIntegration) {
  const CliResult result = run_memory_benchmark("-benchmark -only-bandwidth -latency-samples 1");

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  EXPECT_NE(result.output.find("-only-bandwidth cannot be used with -latency-samples"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, CoreToCoreArgumentsAreRoutedBeforeNormalParserIntegration) {
  const CliResult result = run_memory_benchmark("-analyze-core2core -buffersize 256");

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  EXPECT_NE(result.output.find("-analyze-core2core allows only optional"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, StandardBenchmarkWritesJsonIntegration) {
  const std::string output_path = make_temp_json_path("standard");
  std::remove(output_path.c_str());

  const CliResult result = run_memory_benchmark("-benchmark -only-bandwidth -buffersize 1 -iterations 1 "
                                                "-count 1 -threads 1 -output " + output_path);

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Running benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Results saved to:"), std::string::npos);

  const std::string json = read_file(output_path);
  EXPECT_NE(json.find("\"mode\": \"benchmark\""), std::string::npos);
  EXPECT_NE(json.find("\"main_memory\""), std::string::npos);
  EXPECT_NE(json.find("\"bandwidth\""), std::string::npos);

  std::remove(output_path.c_str());
}

TEST(ExecutableCliIntegrationTest, PatternModeRunsPatternOrchestrationIntegration) {
  const CliResult result = run_memory_benchmark("-patterns -buffersize 1 -iterations 1 -count 1");

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("Running Pattern Benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Sequential Forward:"), std::string::npos);
  EXPECT_NE(result.output.find("Pattern Efficiency Analysis:"), std::string::npos);
}
