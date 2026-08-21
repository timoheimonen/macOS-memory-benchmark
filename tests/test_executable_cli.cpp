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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <memory>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/console/messages/messages_api.h"
#include "third_party/nlohmann/json.hpp"

extern char** environ;

namespace {

class TemporaryCliDirectory {
 public:
  TemporaryCliDirectory() {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_cli_run_" + std::to_string(::getpid()) + "_" +
             std::to_string(sequence.fetch_add(1)));
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    error.clear();
    if (!std::filesystem::create_directories(path_, error) || error) {
      throw std::runtime_error("failed to create CLI test directory: " +
                               error.message());
    }
  }

  ~TemporaryCliDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryCliDirectory(const TemporaryCliDirectory&) = delete;
  TemporaryCliDirectory& operator=(const TemporaryCliDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

struct CliResult {
  int exit_code = -1;
  int termination_signal = 0;
  int spawn_error = 0;
  int wait_error = 0;
  bool timed_out = false;
  std::string stdout_output;
  std::string stderr_output;
  // Existing file-output tests use a channel-agnostic transcript. Capture is
  // still performed independently; only the post-process test view is joined.
  std::string output;
  std::string infrastructure_error;
  std::unique_ptr<TemporaryCliDirectory> directory;
};

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void decode_process_status(int status, CliResult& result) {
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.termination_signal = WTERMSIG(status);
  }
}

bool terminate_and_reap_child(pid_t child, int& status, CliResult& result) {
  if (kill(child, SIGKILL) == -1 && errno != ESRCH &&
      result.wait_error == 0) {
    result.wait_error = errno;
  }

  pid_t wait_result = -1;
  do {
    wait_result = waitpid(child, &status, 0);
  } while (wait_result == -1 && errno == EINTR);

  if (wait_result == child) {
    return true;
  }
  if (wait_result == -1 && errno != ECHILD && result.wait_error == 0) {
    result.wait_error = errno;
  }
  return false;
}

CliResult run_memory_benchmark(
    const std::vector<std::string>& args,
    std::chrono::milliseconds timeout = std::chrono::minutes(10)) {
  CliResult result;
  try {
    result.directory = std::make_unique<TemporaryCliDirectory>();
  } catch (const std::exception& error) {
    result.infrastructure_error = error.what();
    return result;
  }

  std::error_code path_error;
  const std::filesystem::path executable_path =
      std::filesystem::absolute("memory_benchmark", path_error);
  if (path_error) {
    result.infrastructure_error =
        "failed to resolve executable path: " + path_error.message();
    return result;
  }

  const std::filesystem::path stdout_path =
      result.directory->path() / ".stdout-capture";
  const std::filesystem::path stderr_path =
      result.directory->path() / ".stderr-capture";

  posix_spawn_file_actions_t actions;
  int action_result = posix_spawn_file_actions_init(&actions);
  if (action_result != 0) {
    result.spawn_error = action_result;
    return result;
  }

  action_result = posix_spawn_file_actions_addchdir_np(
      &actions, result.directory->path().c_str());
  if (action_result == 0) {
    action_result = posix_spawn_file_actions_addopen(
        &actions, STDOUT_FILENO, stdout_path.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  }
  if (action_result == 0) {
    action_result = posix_spawn_file_actions_addopen(
        &actions, STDERR_FILENO, stderr_path.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  }
  if (action_result != 0) {
    posix_spawn_file_actions_destroy(&actions);
    result.spawn_error = action_result;
    return result;
  }

  std::vector<std::string> argument_storage;
  argument_storage.reserve(args.size() + 1);
  argument_storage.push_back("./memory_benchmark");
  argument_storage.insert(argument_storage.end(), args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(argument_storage.size() + 1);
  for (std::string& argument : argument_storage) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  pid_t child = -1;
  const int spawn_result =
      posix_spawn(&child, executable_path.c_str(), &actions, nullptr,
                  argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawn_result != 0) {
    result.spawn_error = spawn_result;
    return result;
  }

  int status = 0;
  bool child_reaped = false;
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + timeout;
  while (!child_reaped) {
    const pid_t wait_result = waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      child_reaped = true;
      break;
    }
    if (wait_result == -1) {
      if (errno == EINTR) {
        continue;
      }
      result.wait_error = errno;
      child_reaped = terminate_and_reap_child(child, status, result);
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      child_reaped = terminate_and_reap_child(child, status, result);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (child_reaped) {
    decode_process_status(status, result);
  }
  result.stdout_output = read_file(stdout_path.string());
  result.stderr_output = read_file(stderr_path.string());
  result.output = result.stdout_output + result.stderr_output;
  return result;
}

class TemporaryJsonFile {
 public:
  explicit TemporaryJsonFile(const std::string& test_name) {
    static std::atomic<unsigned long> sequence{0};
    path_ = "/tmp/membenchmark_cli_" + std::to_string(getpid()) + "_" +
            std::to_string(sequence.fetch_add(1)) + "_" + test_name +
            ".json";
    std::remove(path_.c_str());
    std::remove((path_ + ".tmp").c_str());
  }

  ~TemporaryJsonFile() {
    std::remove(path_.c_str());
    std::remove((path_ + ".tmp").c_str());
  }

  TemporaryJsonFile(const TemporaryJsonFile&) = delete;
  TemporaryJsonFile& operator=(const TemporaryJsonFile&) = delete;

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

size_t count_occurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t position = 0;
  while (!needle.empty() &&
         (position = text.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

void expect_single_runtime_banner(const CliResult& result) {
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::config_header(SOFTVERSION)),
            1u)
      << result.output;
  EXPECT_EQ(count_occurrences(result.output, Messages::config_copyright()),
            1u)
      << result.output;
  EXPECT_EQ(count_occurrences(result.output, Messages::config_license()), 1u)
      << result.output;
}

void expect_no_runtime_banner(const CliResult& result) {
  EXPECT_EQ(result.output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.output;
}

void expect_process_completed(const CliResult& result) {
  EXPECT_TRUE(result.infrastructure_error.empty())
      << result.infrastructure_error;
  EXPECT_EQ(result.spawn_error, 0)
      << std::strerror(result.spawn_error);
  EXPECT_EQ(result.wait_error, 0)
      << std::strerror(result.wait_error);
  EXPECT_FALSE(result.timed_out);
  EXPECT_EQ(result.termination_signal, 0);
}

void expect_no_dash_transport_artifacts(const CliResult& result) {
  ASSERT_NE(result.directory, nullptr);
  EXPECT_FALSE(std::filesystem::exists(result.directory->path() / "-"));
  EXPECT_FALSE(std::filesystem::exists(result.directory->path() / "-.tmp"));
}

nlohmann::json parse_single_stdout_json(const CliResult& result) {
  const std::string& text = result.stdout_output;
  if (text.empty()) {
    ADD_FAILURE() << "stdout was empty; stderr:\n" << result.stderr_output;
    return nlohmann::json::object();
  }

  EXPECT_EQ(text.back(), '\n') << text;
  const size_t last_non_whitespace = text.find_last_not_of(" \t\r\n");
  if (last_non_whitespace == std::string::npos) {
    ADD_FAILURE() << "stdout contained only whitespace";
    return nlohmann::json::object();
  }
  EXPECT_EQ(last_non_whitespace + 2, text.size())
      << "stdout must have exactly one trailing newline";
  EXPECT_EQ(text.find('\r'), std::string::npos);
  if (!nlohmann::json::accept(text)) {
    ADD_FAILURE() << "stdout was not exactly one JSON document:\n" << text
                  << "stderr:\n"
                  << result.stderr_output;
    return nlohmann::json::object();
  }
  return nlohmann::json::parse(text);
}

}  // namespace

TEST(ExecutableCliIntegrationTest, NoArgumentsShowsHelpAndReturnsSuccessIntegration) {
  const CliResult result = run_memory_benchmark({});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--benchmark"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, HelpFlagShowsHelpAndReturnsSuccessIntegration) {
  const CliResult result = run_memory_benchmark({"-h"});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--patterns"), std::string::npos);
  EXPECT_NE(result.output.find("--gpu-bandwidth"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest,
     HelpWithStdoutTargetRemainsHumanInEitherOrderIntegration) {
  for (const std::vector<std::string>& arguments : {
           std::vector<std::string>{"--benchmark", "--output", "-",
                                    "--help"},
           std::vector<std::string>{"--benchmark", "--help", "--output",
                                    "-"},
       }) {
    SCOPED_TRACE(testing::PrintToString(arguments));
    const CliResult result = run_memory_benchmark(arguments);

    expect_process_completed(result);
    EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
    expect_no_runtime_banner(result);
    EXPECT_NE(result.stdout_output.find("Usage:"), std::string::npos);
    EXPECT_NE(result.stdout_output.find("--benchmark"), std::string::npos);
    EXPECT_TRUE(result.stderr_output.empty()) << result.stderr_output;
    EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
    expect_no_dash_transport_artifacts(result);
  }
}

TEST(ExecutableCliIntegrationTest, GpuHelpUsesDedicatedStandaloneParserIntegration) {
  const CliResult result =
      run_memory_benchmark({"--gpu-bandwidth", "--help"});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find(
                "Usage: ./memory_benchmark --gpu-bandwidth [options]"),
            std::string::npos);
  EXPECT_NE(result.output.find("minimum: 64 MB"), std::string::npos);
  EXPECT_NE(result.output.find("default: 3"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest,
     GpuHelpWithStdoutTargetRemainsHumanInEitherOrderIntegration) {
  for (const std::vector<std::string>& arguments : {
           std::vector<std::string>{"--gpu-bandwidth", "--output", "-",
                                    "--help"},
           std::vector<std::string>{"--gpu-bandwidth", "--help",
                                    "--output", "-"},
       }) {
    SCOPED_TRACE(testing::PrintToString(arguments));
    const CliResult result = run_memory_benchmark(arguments);

    expect_process_completed(result);
    EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
    expect_no_runtime_banner(result);
    EXPECT_NE(result.stdout_output.find(
                  "Usage: ./memory_benchmark --gpu-bandwidth [options]"),
              std::string::npos);
    EXPECT_NE(result.stdout_output.find("minimum: 64 MB"),
              std::string::npos);
    EXPECT_TRUE(result.stderr_output.empty()) << result.stderr_output;
    EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
    expect_no_dash_transport_artifacts(result);
  }
}

TEST(ExecutableCliIntegrationTest, GpuModeConflictIsOrderIndependentIntegration) {
  for (const std::vector<std::string>& arguments : {
           std::vector<std::string>{"--gpu-bandwidth",
                                    "--analyze-core2core"},
           std::vector<std::string>{"--analyze-core2core",
                                    "--gpu-bandwidth"}}) {
    const CliResult result = run_memory_benchmark(arguments);
    EXPECT_EQ(result.exit_code, EXIT_FAILURE);
    expect_no_runtime_banner(result);
    EXPECT_NE(result.output.find("mutually exclusive"), std::string::npos);
    EXPECT_EQ(result.output.find("Running GPU memory bandwidth"),
              std::string::npos);
  }
}

TEST(ExecutableCliIntegrationTest, GpuMinimumFailsBeforeOutputIntegration) {
  const TemporaryJsonFile output("gpu_below_minimum");
  const CliResult result = run_memory_benchmark(
      {"--gpu-bandwidth", "--buffer-size", "63", "--output",
       output.path()});

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("at least 64 MB"), std::string::npos);
  EXPECT_EQ(access(output.path().c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest,
     InvalidGpuStdoutTargetLeavesStdoutEmptyIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--gpu-bandwidth", "--buffer-size", "63", "--output", "-"});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  EXPECT_EQ(result.stderr_output,
            Messages::error_prefix() +
                Messages::error_gpu_buffer_size_below_minimum(
                    63, Constants::GPU_MIN_BUFFER_SIZE_MB) +
                "\n");
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     GpuStdoutEmitsOneSchemaV1ResultOnSupportedOrUnsupportedMetalIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--gpu-bandwidth", "--buffer-size", "64", "--iterations", "1",
       "--count", "3", "--seed", "42", "--output", "-"});

  expect_process_completed(result);
  const nlohmann::json json = parse_single_stdout_json(result);
  ASSERT_TRUE(json.is_object()) << result.stdout_output;
  ASSERT_TRUE(json.contains("status"));
  ASSERT_TRUE(json["status"].is_string());
  EXPECT_EQ(json["software_version"], SOFTVERSION);
  EXPECT_EQ(json["schema_version"], Constants::GPU_JSON_SCHEMA_VERSION);
  EXPECT_EQ(json["mode"], "gpu_bandwidth");
  ASSERT_TRUE(json.contains("configuration"));
  EXPECT_EQ(json["configuration"]["output_file"], "-");
  EXPECT_EQ(json["configuration"]["base_seed_uint64_decimal"], "42");
  EXPECT_EQ(json["configuration"]["loop_count"], 3u);
  EXPECT_EQ(json["configuration"]["iterations"], 1u);
  ASSERT_TRUE(json.contains("counters"));
  EXPECT_EQ(json["counters"]["planned_loops"], 3u);
  EXPECT_EQ(json["counters"]["planned_measurements"], 9u);
  ASSERT_TRUE(json.contains("measurements"));
  EXPECT_EQ(json["measurements"].size(), 9u);

  expect_single_runtime_banner(result);
  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(Messages::msg_running_gpu_bandwidth()),
            std::string::npos);
  EXPECT_EQ(count_occurrences(result.stderr_output,
                              Messages::config_header(SOFTVERSION)),
            1u)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.stderr_output,
                              Messages::msg_running_gpu_bandwidth()),
            1u)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u)
      << result.output;
  expect_no_dash_transport_artifacts(result);

  const std::string status = json["status"].get<std::string>();
  if (status == "complete") {
    EXPECT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    EXPECT_TRUE(json["results_complete"].get<bool>());
    EXPECT_TRUE(json["conclusions_valid"].get<bool>());
    EXPECT_EQ(json["counters"]["completed_loops"], 3u);
    EXPECT_EQ(json["counters"]["validated_measurements"], 9u);
  } else if (status == "unsupported") {
    EXPECT_EQ(result.exit_code, EXIT_FAILURE) << result.stderr_output;
    EXPECT_FALSE(json["results_complete"].get<bool>());
    EXPECT_FALSE(json["conclusions_valid"].get<bool>());
    ASSERT_FALSE(json["reason_code"].get<std::string>().empty());
    EXPECT_EQ(json["backend"]["initialization_status"], "unsupported");
    EXPECT_EQ(json["backend"]["reason_code"], json["reason_code"]);
  } else {
    FAIL() << "GPU stdout integration returned unexpected status: "
           << status << '\n'
           << result.stderr_output;
  }
}

TEST(ExecutableCliIntegrationTest, GpuWritesValidatedSchemaV1Integration) {
  const TemporaryJsonFile output("gpu_schema_v1");
  const CliResult result = run_memory_benchmark(
      {"--gpu-bandwidth", "--buffer-size", "64", "--iterations", "1",
       "--count", "3", "--seed", "42", "--output", output.path()});

  expect_single_runtime_banner(result);
  ASSERT_EQ(access(output.path().c_str(), F_OK), 0);
  const nlohmann::json json =
      nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["configuration"]["output_file"], output.path());
  EXPECT_EQ(count_occurrences(
                result.stdout_output,
                Messages::msg_results_saved_to(output.path())),
            1u)
      << result.output;
  EXPECT_EQ(result.stderr_output.find(
                Messages::msg_results_saved_to(output.path())),
            std::string::npos)
      << result.output;
  EXPECT_EQ(access((output.path() + ".tmp").c_str(), F_OK), -1);
  if (json["status"] == "unsupported") {
    GTEST_SKIP() << json["reason_code"].get<std::string>();
  }
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  EXPECT_EQ(json["software_version"], SOFTVERSION);
  EXPECT_EQ(json["schema_version"], 1);
  EXPECT_EQ(json["mode"], "gpu_bandwidth");
  EXPECT_EQ(json["status"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_TRUE(json["conclusions_valid"].get<bool>());
  EXPECT_EQ(json["counters"]["planned_loops"], 3u);
  EXPECT_EQ(json["counters"]["completed_loops"], 3u);
  EXPECT_EQ(json["counters"]["planned_measurements"], 9u);
  EXPECT_EQ(json["counters"]["validated_measurements"], 9u);
  EXPECT_EQ(json["configuration"]["base_seed_uint64_decimal"], "42");
  ASSERT_EQ(json["measurements"].size(), 9u);
  for (const nlohmann::json& measurement : json["measurements"]) {
    EXPECT_EQ(measurement["status"], "measured");
    EXPECT_GT(measurement["value_gb_s"].get<double>(), 0.0);
    EXPECT_EQ(measurement["timed"]["command_buffer_count"], 1u);
    EXPECT_EQ(measurement["timed"]["compute_encoder_count"], 1u);
    EXPECT_EQ(measurement["validation"]["validation_status"], "passed");
  }
  EXPECT_EQ(json["work_plans"][2]["exact_payload_bytes"], "134217728");
  EXPECT_EQ(json["backend"]["allocation"]["buffer_a"]["storage_mode"],
            "private");
  EXPECT_EQ(json["backend"]["allocation"]["buffer_a"]
                ["hazard_tracking_mode"],
            "tracked");
  EXPECT_EQ(json["backend"]["compilation"]["kernel_source_sha256"]
                .get<std::string>()
                .size(),
            64u);
}

TEST(ExecutableCliIntegrationTest, GpuAutomaticCalibrationFreezesPlansIntegration) {
  const TemporaryJsonFile output("gpu_automatic_calibration");
  const CliResult result = run_memory_benchmark(
      {"--gpu-bandwidth", "--buffer-size", "64", "--count", "1",
       "--seed", "42", "--output", output.path()});

  ASSERT_EQ(access(output.path().c_str(), F_OK), 0);
  const nlohmann::json json =
      nlohmann::json::parse(read_file(output.path()));
  if (json["status"] == "unsupported") {
    GTEST_SKIP() << json["reason_code"].get<std::string>();
  }
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  for (const char* operation : {"read", "write", "copy"}) {
    const nlohmann::json& attempts =
        json["excluded_calibration_attempts"][operation];
    ASSERT_GE(attempts.size(), 2u);
    EXPECT_EQ(attempts[0]["purpose"], "pilot");
    EXPECT_EQ(attempts[1]["purpose"], "duration-trial");
    EXPECT_TRUE(attempts.back()["valid"].get<bool>());
  }
  for (const nlohmann::json& measurement : json["measurements"]) {
    EXPECT_EQ(measurement["status"], "measured");
    const std::string quality =
        measurement["duration_quality"].get<std::string>();
    EXPECT_TRUE(quality == "within-target-window" ||
                quality == "dispatch-cap-below-target" ||
                quality == "payload-cap-below-target" ||
                quality == "single-pass-exceeds-window")
        << quality;
  }
}

TEST(ExecutableCliIntegrationTest, OptionsWithoutModeShowHelpAndReturnSuccessIntegration) {
  const CliResult result = run_memory_benchmark({"--threads", "1"});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("Usage:"), std::string::npos);
  EXPECT_NE(result.output.find("--threads <count>"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest, InvalidStandardModeConfigReturnsFailureIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--benchmark", "--only-bandwidth", "--latency-samples", "1"});

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("--only-bandwidth cannot be used with --latency-samples"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest,
     InvalidStdoutTargetCommandLeavesStdoutEmptyIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--benchmark", "--only-bandwidth", "--latency-samples", "1",
       "--output", "-"});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::error_only_bandwidth_with_latency_samples()),
            std::string::npos)
      << result.stderr_output;
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     ParseErrorWithStdoutTargetLeavesStdoutEmptyIntegration) {
  const std::string invalid_option = "--not-a-real-option";
  const CliResult result = run_memory_benchmark(
      {"--benchmark", "--output", "-", invalid_option});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::error_unknown_option(invalid_option)),
            std::string::npos)
      << result.stderr_output;
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, CoreToCoreArgumentsAreRoutedBeforeNormalParserIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--analyze-core2core", "--buffer-size", "256"});

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("--analyze-core2core allows only optional"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest,
     InvalidCoreToCoreStdoutTargetLeavesStdoutEmptyIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--analyze-core2core", "--output", "-", "--buffer-size", "256"});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::error_analyze_core_to_core_must_be_used_alone()),
            std::string::npos)
      << result.stderr_output;
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     CoreToCoreHelpWithStdoutTargetRemainsHumanInEitherOrderIntegration) {
  for (const std::vector<std::string>& arguments : {
           std::vector<std::string>{"--analyze-core2core", "--output", "-",
                                    "--help"},
           std::vector<std::string>{"--analyze-core2core", "--help",
                                    "--output", "-"},
       }) {
    SCOPED_TRACE(testing::PrintToString(arguments));
    const CliResult result = run_memory_benchmark(arguments);

    expect_process_completed(result);
    EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
    expect_no_runtime_banner(result);
    EXPECT_NE(result.stdout_output.find("Usage:"), std::string::npos);
    EXPECT_NE(result.stdout_output.find("--analyze-core2core"),
              std::string::npos);
    EXPECT_TRUE(result.stderr_output.empty()) << result.stderr_output;
    EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
    expect_no_dash_transport_artifacts(result);
  }
}

TEST(ExecutableCliIntegrationTest, CoreToCoreWritesCalibratedAuditJsonIntegration) {
  const TemporaryJsonFile output("core2core_v2");

  const CliResult result = run_memory_benchmark({
      "--analyze-core2core", "--count", "2", "--latency-samples", "1",
      "--output", output.path()});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  const nlohmann::json json = nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["version"], SOFTVERSION);
  EXPECT_EQ(json["configuration"]["schema_version"], 2);
  EXPECT_EQ(json["core_to_core_latency"]["status"], "complete");
  EXPECT_TRUE(json["core_to_core_latency"]["measurements_complete"]);
  ASSERT_EQ(json["core_to_core_latency"]["scenarios"].size(), 3u);
  EXPECT_EQ(json["core_to_core_latency"]["scenarios"][0]["loop_records"].size(), 2u);
}

TEST(ExecutableCliIntegrationTest,
     CoreToCoreWritesSingleJsonDocumentToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--analyze-core2core", "--count", "1", "--latency-samples", "1",
      "--output", "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"],
            Constants::CORE_TO_CORE_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["schema_version"], 2);
  EXPECT_EQ(json["core_to_core_latency"]["status"], "complete");
  EXPECT_TRUE(json["core_to_core_latency"]["measurements_complete"]);

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(
                Messages::msg_running_core_to_core_analysis()),
            std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::msg_running_core_to_core_analysis()),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, CoreToCoreSweepWritesCompletionMetadataIntegration) {
  const TemporaryJsonFile output("core2core_sweep_v2");

  const CliResult result = run_memory_benchmark({
      "--analyze-core2core", "--count", "1", "--sweep",
      "latency-samples=1,2,3", "--sweep-max-runs", "3", "--output",
      output.path()});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  const nlohmann::json json = nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["version"], SOFTVERSION);
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 3u);
  EXPECT_EQ(json["completed_runs"], 3u);
  EXPECT_TRUE(json["conclusions_valid"]);
  ASSERT_EQ(json["runs"].size(), 3u);
  EXPECT_EQ(json["runs"][0]["result"]["configuration"]["schema_version"], 2);
  EXPECT_EQ(count_occurrences(
                result.output,
                Messages::msg_results_saved_to(output.path())),
            1u)
      << result.output;
  EXPECT_EQ(access((output.path() + ".tmp").c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest,
     CoreToCoreSweepWritesSingleTerminalJsonToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--analyze-core2core", "--count", "1", "--sweep",
      "latency-samples=1,2", "--sweep-max-runs", "2", "--output",
      "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"],
            Constants::SWEEP_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["base_mode"],
            Constants::CORE_TO_CORE_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 2u);
  EXPECT_EQ(json["attempted_runs"], 2u);
  EXPECT_EQ(json["completed_runs"], 2u);
  EXPECT_TRUE(json["conclusions_valid"].get<bool>());
  ASSERT_EQ(json["runs"].size(), 2u);
  for (const nlohmann::json& run : json["runs"]) {
    EXPECT_EQ(run["status"], "complete");
    EXPECT_EQ(run["result"]["configuration"]["schema_version"], 2);
    EXPECT_EQ(run["result"]["core_to_core_latency"]["status"],
              "complete");
    EXPECT_TRUE(run["result"]["core_to_core_latency"]
                       ["measurements_complete"]
                           .get<bool>());
  }

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(Messages::msg_running_sweep(2)),
            std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(Messages::msg_running_sweep(2)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::msg_running_core_to_core_analysis()),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, AnalyzeTlbInvalidStrideSweepFailsBeforeExecutionIntegration) {
  const TemporaryJsonFile output("tlb_invalid_stride_sweep");

  const CliResult result = run_memory_benchmark({
      "--analyze-tlb", "--sweep", "latency-stride-bytes=64,130",
      "--output", output.path()});

  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_NE(result.output.find("must be a multiple of 8 bytes"), std::string::npos);
  EXPECT_EQ(result.output.find("Running sweep"), std::string::npos);
  EXPECT_EQ(access(output.path().c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest,
     AnalyzeTlbInvalidPreflightWithStdoutTargetLeavesStdoutEmptyIntegration) {
  const CliResult result = run_memory_benchmark({
      "--analyze-tlb", "--latency-stride-bytes", "32768", "--output",
      "-"});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  EXPECT_NE(result.stderr_output.find(
                Messages::error_analyze_tlb_stride_exceeds_page(
                    32768, static_cast<size_t>(page_size))),
            std::string::npos)
      << result.stderr_output;
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     AnalyzeTlbWritesSingleJsonDocumentToStdoutIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--analyze-tlb", "--tlb-density", "low", "--seed", "42",
       "--output", "-"},
      std::chrono::minutes(15));

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"],
            Constants::TLB_ANALYSIS_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["schema_version"], 4);
  EXPECT_EQ(json["tlb_analysis"]["status"], "complete");
  EXPECT_TRUE(json["tlb_analysis"]["conclusions_valid"]);
  EXPECT_FALSE(json["tlb_analysis"].contains("status_reason"));

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(Messages::msg_running_tlb_analysis()),
            std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(Messages::msg_running_tlb_analysis()),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, StandardBenchmarkWritesJsonIntegration) {
  const TemporaryJsonFile output("standard output with spaces");

  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--buffer-size", "1",
      "--iterations", "1", "--count", "3", "--threads", "1", "--output",
      output.path()});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  EXPECT_NE(result.output.find("Running benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Results saved to:"), std::string::npos);
  EXPECT_EQ(result.output.find('\r'), std::string::npos);

  const nlohmann::json json = nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["configuration"]["mode"], "benchmark");
  EXPECT_EQ(json["planned_loops"], 3u);
  EXPECT_EQ(json["completed_loops"], 3u);
  EXPECT_EQ(json["loops"].size(), 3u);
  EXPECT_TRUE(json.contains("main_memory"));
  EXPECT_TRUE(json["main_memory"].contains("bandwidth"));
}

TEST(ExecutableCliIntegrationTest,
     StandardBenchmarkWritesSingleJsonDocumentToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--buffer-size", "1",
      "--iterations", "1", "--count", "1", "--threads", "1",
      "--output", "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"], "benchmark");
  EXPECT_EQ(json["configuration"]["benchmark_schema_version"], 2);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_EQ(json["planned_loops"], 1u);
  EXPECT_EQ(json["completed_loops"], 1u);

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(Messages::msg_running_benchmarks()),
            std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(Messages::msg_running_benchmarks()),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     ExplicitDotDashOutputRemainsOrdinaryFileTargetIntegration) {
  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--buffer-size", "1",
      "--iterations", "1", "--count", "1", "--threads", "1",
      "--output", "./-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  ASSERT_NE(result.directory, nullptr);
  const std::filesystem::path output_path = result.directory->path() / "-";
  ASSERT_TRUE(std::filesystem::exists(output_path));
  EXPECT_FALSE(std::filesystem::exists(result.directory->path() / "-.tmp"));
  const nlohmann::json json =
      nlohmann::json::parse(read_file(output_path.string()));
  EXPECT_EQ(json["configuration"]["mode"], "benchmark");
  EXPECT_EQ(json["configuration"]["benchmark_schema_version"], 2);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_NE(result.stdout_output.find(Messages::msg_running_benchmarks()),
            std::string::npos)
      << result.stdout_output;
  EXPECT_NE(result.stdout_output.find("Results saved to:"),
            std::string::npos)
      << result.stdout_output;
  EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
}

TEST(ExecutableCliIntegrationTest,
     FlagShapedOutputValueRemainsStandardFileTargetIntegration) {
  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--buffer-size", "1",
      "--iterations", "1", "--count", "1", "--threads", "1",
      "--output", "-G"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  ASSERT_NE(result.directory, nullptr);
  const std::filesystem::path output_path = result.directory->path() / "-G";
  ASSERT_TRUE(std::filesystem::is_regular_file(output_path));
  EXPECT_FALSE(
      std::filesystem::exists(result.directory->path() / "-G.tmp"));

  const nlohmann::json json =
      nlohmann::json::parse(read_file(output_path.string()));
  EXPECT_EQ(json["configuration"]["mode"], "benchmark");
  EXPECT_EQ(json["configuration"]["benchmark_schema_version"], 2);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_EQ(result.output.find("mutually exclusive"), std::string::npos)
      << result.output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            1u)
      << result.output;
}

TEST(ExecutableCliIntegrationTest, StandardSweepWritesCompletionMetadataIntegration) {
  const TemporaryJsonFile output("standard_sweep_status");

  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--iterations", "1", "--count",
      "1", "--threads", "1", "--sweep", "buffer-size=1,2,3",
      "--sweep-max-runs", "3", "--output", output.path()});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  const nlohmann::json json = nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 3u);
  EXPECT_EQ(json["completed_runs"], 3u);
  EXPECT_TRUE(json["conclusions_valid"].get<bool>());
  EXPECT_EQ(count_occurrences(
                result.output,
                Messages::msg_results_saved_to(output.path())),
            1u)
      << result.output;
  EXPECT_EQ(access((output.path() + ".tmp").c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest,
     StandardSweepWritesSingleTerminalJsonToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--benchmark", "--only-bandwidth", "--iterations", "1", "--count",
      "1", "--threads", "1", "--sweep", "buffer-size=1,2",
      "--sweep-max-runs", "2", "--output", "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"],
            Constants::SWEEP_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["base_mode"],
            Constants::BENCHMARK_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 2u);
  EXPECT_EQ(json["attempted_runs"], 2u);
  EXPECT_EQ(json["completed_runs"], 2u);
  EXPECT_TRUE(json["conclusions_valid"].get<bool>());
  ASSERT_EQ(json["runs"].size(), 2u);
  for (const nlohmann::json& run : json["runs"]) {
    EXPECT_EQ(run["status"], "complete");
    EXPECT_EQ(run["result"]["configuration"]
                 ["benchmark_schema_version"],
              2);
    EXPECT_EQ(run["result"]["status"], "complete");
    EXPECT_TRUE(run["result"]["results_complete"].get<bool>());
  }

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(result.stdout_output.find(Messages::msg_running_sweep(2)),
            std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(Messages::msg_running_sweep(2)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(result.stderr_output.find(Messages::msg_sweep_run_progress(1, 2)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest,
     InvalidTlbSweepWithStdoutTargetLeavesStdoutEmptyIntegration) {
  const CliResult result = run_memory_benchmark({
      "--analyze-tlb", "--sweep", "latency-stride-bytes=64,130",
      "--output", "-"});

  expect_process_completed(result);
  EXPECT_EQ(result.exit_code, EXIT_FAILURE);
  expect_no_runtime_banner(result);
  EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
  EXPECT_NE(result.stderr_output.find(
                Messages::error_latency_stride_alignment(
                    130, sizeof(uintptr_t))),
            std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(result.stderr_output.find(Messages::msg_running_sweep(2)),
            std::string::npos);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, PatternSweepPrintsOneBannerAcrossNestedLoopsIntegration) {
  const TemporaryJsonFile output("pattern_sweep_banner");

  const CliResult result = run_memory_benchmark({
      "--patterns", "--iterations", "1", "--count", "2", "--threads",
      "1", "--seed", "42", "--sweep", "buffer-size=1,2,3",
      "--sweep-max-runs", "3", "--output", output.path()});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  const nlohmann::json json = nlohmann::json::parse(read_file(output.path()));
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 3u);
  EXPECT_EQ(json["completed_runs"], 3u);
  ASSERT_EQ(json["runs"].size(), 3u);
  for (const nlohmann::json& run : json["runs"]) {
    EXPECT_EQ(run["status"], "complete");
    EXPECT_EQ(run["result"]["planned_loops"], 2u);
    EXPECT_EQ(run["result"]["completed_loops"], 2u);
  }
  EXPECT_EQ(count_occurrences(
                result.output,
                Messages::msg_results_saved_to(output.path())),
            1u)
      << result.output;
  EXPECT_EQ(access((output.path() + ".tmp").c_str(), F_OK), -1);
}

TEST(ExecutableCliIntegrationTest,
     PatternSweepWritesSingleTerminalJsonToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--patterns", "--iterations", "1", "--count", "1", "--threads",
      "1", "--seed", "42", "--sweep", "buffer-size=1,2",
      "--sweep-max-runs", "2", "--output", "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"],
            Constants::SWEEP_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["base_mode"],
            Constants::PATTERNS_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["sweep_schema_version"], 1);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["planned_runs"], 2u);
  EXPECT_EQ(json["attempted_runs"], 2u);
  EXPECT_EQ(json["completed_runs"], 2u);
  EXPECT_TRUE(json["conclusions_valid"].get<bool>());
  ASSERT_EQ(json["runs"].size(), 2u);
  for (const nlohmann::json& run : json["runs"]) {
    EXPECT_EQ(run["status"], "complete");
    EXPECT_EQ(run["result"]["configuration"]["pattern_schema_version"],
              3);
    EXPECT_EQ(run["result"]["status"], "complete");
    EXPECT_TRUE(run["result"]["results_complete"].get<bool>());
  }

  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, PatternModeRunsPatternOrchestrationIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--patterns", "--buffer-size", "1", "--iterations", "1", "--count", "1"});

  EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(result);
  EXPECT_NE(result.output.find("Running Pattern Benchmarks"), std::string::npos);
  EXPECT_NE(result.output.find("Sequential Forward:"), std::string::npos);
  EXPECT_EQ(result.output.find('\r'), std::string::npos);
  EXPECT_EQ(result.output.find("Pattern Efficiency Analysis:"), std::string::npos);
  EXPECT_EQ(result.output.find("Prefetcher effectiveness"), std::string::npos);
  EXPECT_EQ(result.output.find("TLB pressure"), std::string::npos);
  EXPECT_NE(result.output.find("2 MiB stride"), std::string::npos);
}

TEST(ExecutableCliIntegrationTest,
     PatternBenchmarkWritesSingleJsonDocumentToStdoutIntegration) {
  const CliResult result = run_memory_benchmark({
      "--patterns", "--buffer-size", "1", "--iterations", "1", "--count",
      "1", "--threads", "1", "--seed", "42", "--output", "-"});

  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = parse_single_stdout_json(result);
  EXPECT_EQ(json["configuration"]["mode"], "patterns");
  EXPECT_EQ(json["configuration"]["pattern_schema_version"], 3);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_EQ(json["planned_loops"], 1u);
  EXPECT_EQ(json["completed_loops"], 1u);

  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_EQ(
      result.stdout_output.find(Messages::msg_running_pattern_benchmarks()),
      std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << result.stderr_output;
  EXPECT_NE(
      result.stderr_output.find(Messages::msg_running_pattern_benchmarks()),
      std::string::npos)
      << result.stderr_output;
  EXPECT_EQ(count_occurrences(result.output,
                              Messages::msg_results_saved_to("")),
            0u);
  expect_no_dash_transport_artifacts(result);
}

TEST(ExecutableCliIntegrationTest, PatternSeedReproducesWorkloadMetadataIntegration) {
  const TemporaryJsonFile first_output("patterns_seed_first");
  const TemporaryJsonFile second_output("patterns_seed_second");

  const std::vector<std::string> common_args = {
      "--patterns", "--buffer-size", "1", "--iterations", "1", "--count",
      "3", "--threads", "2", "--seed", "42", "--output"};
  std::vector<std::string> first_args = common_args;
  first_args.push_back(first_output.path());
  std::vector<std::string> second_args = common_args;
  second_args.push_back(second_output.path());
  const CliResult first_run = run_memory_benchmark(first_args);
  const CliResult second_run = run_memory_benchmark(second_args);
  ASSERT_EQ(first_run.exit_code, EXIT_SUCCESS);
  ASSERT_EQ(second_run.exit_code, EXIT_SUCCESS);
  expect_single_runtime_banner(first_run);
  expect_single_runtime_banner(second_run);

  const nlohmann::json first = nlohmann::json::parse(read_file(first_output.path()));
  const nlohmann::json second = nlohmann::json::parse(read_file(second_output.path()));
  EXPECT_EQ(first["configuration"]["pattern_seed"], "42");
  EXPECT_EQ(first["configuration"]["pattern_seed"],
            second["configuration"]["pattern_seed"]);

  for (const char* operation : {"read_gb_s", "write_gb_s", "copy_gb_s"}) {
    const nlohmann::json& first_samples =
        first["patterns"]["random"]["bandwidth"][operation]["measurements"];
    const nlohmann::json& second_samples =
        second["patterns"]["random"]["bandwidth"][operation]["measurements"];
    ASSERT_EQ(first_samples.size(), 3u);
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
}

TEST(ExecutableCliIntegrationTest, PatternAutomaticCalibrationWritesMetadataIntegration) {
  const TemporaryJsonFile output_file("patterns_calibration");

  const CliResult result = run_memory_benchmark({
      "--patterns", "--buffer-size", "1", "--count", "1", "--threads",
      "1", "--seed", "42", "--output", output_file.path()});
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS);
  EXPECT_NE(result.output.find("automatic duration calibration"),
            std::string::npos);

  const nlohmann::json output = nlohmann::json::parse(read_file(output_file.path()));
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
}
