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
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "llm_memory/llm_backend.h"
#include "output/console/messages/messages_api.h"
#include "third_party/nlohmann/json.hpp"

extern char** environ;

namespace {

class TemporaryCliDirectory {
 public:
  TemporaryCliDirectory() {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_cli_run_" + std::to_string(::getpid()) + "_" + std::to_string(sequence.fetch_add(1)));
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    error.clear();
    if (!std::filesystem::create_directories(path_, error) || error) {
      throw std::runtime_error("failed to create CLI test directory: " + error.message());
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

class TemporaryJsonFile {
 public:
  explicit TemporaryJsonFile(const std::string& test_name) {
    static std::atomic<unsigned long> sequence{0};
    path_ = "/tmp/membenchmark_cli_" + std::to_string(::getpid()) + "_" + std::to_string(sequence.fetch_add(1)) + "_" +
            test_name + ".json";
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

struct CliResult {
  int exit_code = -1;
  int termination_signal = 0;
  int spawn_error = 0;
  int wait_error = 0;
  bool timed_out = false;
  std::string stdout_output;
  std::string stderr_output;
  // Channel-agnostic transcript for file-output checks; capture remains split.
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

bool terminate_and_reap_child(pid_t child, int& status, CliResult& result) {
  if (kill(child, SIGKILL) == -1 && errno != ESRCH && result.wait_error == 0) result.wait_error = errno;
  pid_t wait_result = -1;
  do {
    wait_result = waitpid(child, &status, 0);
  } while (wait_result == -1 && errno == EINTR);
  if (wait_result == child) return true;
  if (wait_result == -1 && errno != ECHILD && result.wait_error == 0) result.wait_error = errno;
  return false;
}

CliResult run_memory_benchmark(const std::vector<std::string>& args,
                               std::chrono::milliseconds timeout = std::chrono::minutes(10)) {
  CliResult result;
  try {
    result.directory = std::make_unique<TemporaryCliDirectory>();
  } catch (const std::exception& error) {
    result.infrastructure_error = error.what();
    return result;
  }

  std::error_code path_error;
  const std::filesystem::path executable_path = std::filesystem::absolute("memory_benchmark", path_error);
  if (path_error) {
    result.infrastructure_error = "failed to resolve executable path: " + path_error.message();
    return result;
  }

  const std::filesystem::path stdout_path = result.directory->path() / ".stdout-capture";
  const std::filesystem::path stderr_path = result.directory->path() / ".stderr-capture";
  posix_spawn_file_actions_t actions;
  int action_result = posix_spawn_file_actions_init(&actions);
  if (action_result != 0) {
    result.spawn_error = action_result;
    return result;
  }
  action_result = posix_spawn_file_actions_addchdir(&actions, result.directory->path().c_str());
  if (action_result == 0) {
    action_result = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, stdout_path.c_str(),
                                                     O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  }
  if (action_result == 0) {
    action_result = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, stderr_path.c_str(),
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
  for (std::string& argument : argument_storage) argv.push_back(argument.data());
  argv.push_back(nullptr);

  pid_t child = -1;
  const int spawn_result = posix_spawn(&child, executable_path.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawn_result != 0) {
    result.spawn_error = spawn_result;
    return result;
  }

  int status = 0;
  bool child_reaped = false;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!child_reaped) {
    const pid_t wait_result = waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      child_reaped = true;
      break;
    }
    if (wait_result == -1) {
      if (errno == EINTR) continue;
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
    if (WIFEXITED(status)) {
      result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      result.termination_signal = WTERMSIG(status);
    }
  }
  result.stdout_output = read_file(stdout_path.string());
  result.stderr_output = read_file(stderr_path.string());
  result.output = result.stdout_output + result.stderr_output;
  return result;
}

size_t count_occurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t position = 0;
  while (!needle.empty() && (position = text.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

void expect_single_runtime_banner(const CliResult& result) {
  EXPECT_EQ(count_occurrences(result.output, Messages::config_header(SOFTVERSION)), 1u) << result.output;
  EXPECT_EQ(count_occurrences(result.output, Messages::config_copyright()), 1u) << result.output;
  EXPECT_EQ(count_occurrences(result.output, Messages::config_license()), 1u) << result.output;
}

void expect_no_runtime_banner(const CliResult& result) {
  EXPECT_EQ(result.output.find(Messages::config_header(SOFTVERSION)), std::string::npos) << result.output;
}

void expect_process_completed(const CliResult& result) {
  EXPECT_TRUE(result.infrastructure_error.empty()) << result.infrastructure_error;
  EXPECT_EQ(result.spawn_error, 0) << std::strerror(result.spawn_error);
  EXPECT_EQ(result.wait_error, 0) << std::strerror(result.wait_error);
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
  EXPECT_EQ(last_non_whitespace + 2, text.size()) << "stdout must have exactly one trailing newline";
  EXPECT_EQ(text.find('\r'), std::string::npos);
  if (!nlohmann::json::accept(text)) {
    ADD_FAILURE() << "stdout was not exactly one JSON document:\n" << text << "stderr:\n" << result.stderr_output;
    return nlohmann::json::object();
  }
  return nlohmann::json::parse(text);
}

nlohmann::json expect_stdout_json_transport(const CliResult& result, const std::string& stderr_marker = "") {
  const nlohmann::json json = parse_single_stdout_json(result);
  expect_single_runtime_banner(result);
  EXPECT_EQ(result.stdout_output.find(Messages::config_header(SOFTVERSION)), std::string::npos);
  EXPECT_NE(result.stderr_output.find(Messages::config_header(SOFTVERSION)), std::string::npos) << result.stderr_output;
  if (!stderr_marker.empty()) {
    EXPECT_EQ(result.stdout_output.find(stderr_marker), std::string::npos);
    EXPECT_NE(result.stderr_output.find(stderr_marker), std::string::npos) << result.stderr_output;
  }
  EXPECT_EQ(count_occurrences(result.output, Messages::msg_results_saved_to("")), 0u) << result.output;
  expect_no_dash_transport_artifacts(result);
  return json;
}

nlohmann::json expect_atomic_file_json(const CliResult& result, const std::filesystem::path& file_path,
                                       bool expect_exact_saved_path = true) {
  expect_single_runtime_banner(result);
  EXPECT_TRUE(std::filesystem::is_regular_file(file_path)) << result.output;
  EXPECT_FALSE(std::filesystem::exists(file_path.string() + ".tmp"));
  if (expect_exact_saved_path) {
    EXPECT_EQ(count_occurrences(result.stdout_output, Messages::msg_results_saved_to(file_path.string())), 1u)
        << result.output;
    EXPECT_EQ(result.stderr_output.find(Messages::msg_results_saved_to(file_path.string())), std::string::npos)
        << result.output;
  }
  return nlohmann::json::parse(read_file(file_path.string()));
}

void expect_complete_sweep_envelope(const nlohmann::json& envelope, const char* expected_base_mode,
                                    size_t expected_runs) {
  EXPECT_EQ(envelope["configuration"]["mode"], Constants::SWEEP_JSON_MODE_NAME);
  EXPECT_EQ(envelope["configuration"]["base_mode"], expected_base_mode);
  EXPECT_EQ(envelope["configuration"]["sweep_schema_version"], Constants::SWEEP_JSON_SCHEMA_VERSION);
  EXPECT_EQ(envelope["status"], "complete");
  EXPECT_EQ(envelope["planned_runs"], expected_runs);
  EXPECT_EQ(envelope["attempted_runs"], expected_runs);
  EXPECT_EQ(envelope["completed_runs"], expected_runs);
  EXPECT_TRUE(envelope["conclusions_valid"].get<bool>());
  ASSERT_EQ(envelope["runs"].size(), expected_runs);
}

const nlohmann::json& canonical_llm_plan(const nlohmann::json& document, const nlohmann::json& measurement) {
  return document.at("resolved_plan").at("scenario_plans").at(measurement.at("plan_ref").get<size_t>());
}

void expect_complete_llm_named_checks(const nlohmann::json& document, const nlohmann::json& measurement) {
  const bool cpu = document.at("backend") == "cpu";
  const bool prefill = document.at("phase") == "prefill";
  const bool paged = document.at("kv_layout") == "paged";
  const bool writes = measurement.at("scenario") != "weights_only";
  const bool padding = paged && document.at("resolved_plan").at("resources").at("k_layout_padding_bytes") != "0";
  const std::array<const char*, 3> kinds = {
      "post-validation-structure",
      prefill ? "kv-prefill-final-samples" : (cpu && !writes ? "kv-append-unchanged" : "kv-append-final"),
      "kv-padding-canary"};
  const std::array<bool, 3> applicability = {
      cpu ? paged || prefill || writes : writes,
      writes || (cpu && paged && !prefill),
      padding && (cpu || writes)};
  const auto& checks = measurement.at("execution").at("validation").at("checks");
  ASSERT_TRUE(checks.is_array());
  ASSERT_EQ(checks.size(), kinds.size());
  for (size_t index = 0; index < kinds.size(); ++index) {
    SCOPED_TRACE(kinds[index]);
    const auto& check = checks.at(index);
    EXPECT_EQ(check.at("kind"), kinds[index]);
    EXPECT_EQ(check.at("applicable"), applicability[index]);
    if (applicability[index]) {
      EXPECT_EQ(check.at("evaluated"), true);
      EXPECT_EQ(check.at("valid"), true);
      EXPECT_EQ(check.at("reason_code"), "valid");
    } else {
      EXPECT_TRUE(check.at("evaluated").is_null());
      EXPECT_TRUE(check.at("valid").is_null());
      EXPECT_EQ(check.at("reason_code"), "not-applicable");
    }
  }
}

void expect_complete_llm_checkpoint_lifecycle(const nlohmann::json& json) {
  ASSERT_TRUE(json.contains("checkpoint_lifecycle"));
  const auto& lifecycle = json["checkpoint_lifecycle"];
  EXPECT_FALSE(lifecycle["checkpoint_failed"].get<bool>());
  const size_t count = json["configuration"]["loop_count"].get<size_t>();
  const size_t interval = std::max(size_t{1}, count / 8 + (count % 8 != 0));
  const bool file = json["configuration"]["output_file"] != "-";
  EXPECT_EQ(lifecycle["prior_file_writer_attempts"], file ? count / interval : 0);
  EXPECT_EQ(lifecycle["prior_successful_file_writes"], file ? count / interval : 0);
  EXPECT_EQ(lifecycle["observation_point"], "before-current-snapshot-preparation");
  EXPECT_EQ(lifecycle["current_request"], "terminal");
  EXPECT_EQ(lifecycle["snapshot_interval_loops"], interval);
  EXPECT_EQ(lifecycle["checkpoint_policy"], "bounded-loop-snapshots");
  EXPECT_TRUE(lifecycle["current_persistence_success"].is_null());
}

void expect_complete_cpu_llm_document(const nlohmann::json& json, const char* expected_phase,
                                      const char* expected_kv_layout, const char* expected_methodology,
                                      const char* expected_work_unit_kind) {
  ASSERT_TRUE(json.is_object());
  EXPECT_EQ(json["schema_version"], Constants::LLM_JSON_SCHEMA_VERSION);
  EXPECT_EQ(json["mode"], Constants::LLM_JSON_MODE_NAME);
  EXPECT_EQ(json["backend"], "cpu");
  EXPECT_EQ(json["phase"], expected_phase);
  EXPECT_EQ(json["kv_layout"], expected_kv_layout);
  EXPECT_EQ(json["methodology_version"], expected_methodology);
  EXPECT_EQ(json["status"], "complete");
  EXPECT_EQ(json["reason_code"], "complete");
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_TRUE(json["run_accepted"].get<bool>());
  EXPECT_TRUE(json["scenario_order_balance_complete"].get<bool>());
  EXPECT_EQ(json["resolved_plan"]["work_unit_kind"], expected_work_unit_kind);
  EXPECT_TRUE(json["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(json["backend_evidence"]["metal"].is_null());
}

void expect_complete_or_unsupported_metal_result(const CliResult& result, const nlohmann::json& json,
                                                 const char* expected_kv_layout = "contiguous",
                                                 const char* expected_methodology =
                                                     "llm-memory-v2-metal-decode-contiguous",
                                                 const char* expected_phase = "decode",
                                                 const char* expected_work_unit_kind = "decode_step") {
  ASSERT_TRUE(json.is_object());
  EXPECT_EQ(json["schema_version"], Constants::LLM_JSON_SCHEMA_VERSION);
  EXPECT_EQ(json["mode"], Constants::LLM_JSON_MODE_NAME);
  EXPECT_EQ(json["backend"], "metal");
  EXPECT_EQ(json["phase"], expected_phase);
  EXPECT_EQ(json["kv_layout"], expected_kv_layout);
  EXPECT_EQ(json["methodology_version"], expected_methodology);
  ASSERT_TRUE(json["resolved_plan"].is_object());
  EXPECT_EQ(json["resolved_plan"]["phase"], expected_phase);
  EXPECT_EQ(json["resolved_plan"]["kv_layout"], expected_kv_layout);
  EXPECT_EQ(json["resolved_plan"]["work_unit_kind"], expected_work_unit_kind);
  EXPECT_TRUE(json["backend_evidence"]["cpu"].is_null());
  ASSERT_TRUE(json["backend_evidence"]["metal"].is_object());
  EXPECT_FALSE(json["backend_evidence"]["metal"]["workers_applicable"].get<bool>());
  EXPECT_FALSE(json["backend_evidence"]["metal"]["worker_qos_applicable"].get<bool>());

  const std::string status = json["status"].get<std::string>();
  if (status == "unsupported") {
    EXPECT_EQ(result.exit_code, EXIT_FAILURE) << result.output;
    EXPECT_FALSE(json["results_complete"].get<bool>());
    EXPECT_FALSE(json["run_accepted"].get<bool>());
    const std::string reason_code = json["reason_code"].get<std::string>();
    EXPECT_TRUE(reason_code == LlmBackendReason::METAL_DEVICE_UNAVAILABLE ||
                reason_code == LlmBackendReason::UNIFIED_MEMORY_REQUIRED ||
                reason_code == LlmBackendReason::APPLE7_FAMILY_REQUIRED ||
                reason_code == LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED ||
                reason_code == LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY)
        << reason_code;
    EXPECT_EQ(json["backend_evidence"]["metal"]["lifecycle"]["initialization"]["status"], "unsupported");
    EXPECT_EQ(json["backend_evidence"]["metal"]["lifecycle"]["initialization"]["reason_code"], json["reason_code"]);
    EXPECT_EQ(json["checkpoint_lifecycle"]["prior_file_writer_attempts"], 0u);
    EXPECT_EQ(json["checkpoint_lifecycle"]["prior_successful_file_writes"], 0u);
    EXPECT_TRUE(json["checkpoint_lifecycle"]["current_persistence_success"].is_null());
    return;
  }

  ASSERT_EQ(status, "complete") << json.dump(2);
  EXPECT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  EXPECT_TRUE(json["results_complete"].get<bool>());
  EXPECT_TRUE(json["run_accepted"].get<bool>());
  EXPECT_EQ(json["reason_code"], "complete");
  EXPECT_EQ(json["backend_evidence"]["metal"]["lifecycle"]["initialization"]["status"], "ready");
  EXPECT_TRUE(json["backend_evidence"]["metal"]["timed_results_available"].get<bool>());
  ASSERT_EQ(json["measurements"].size(), 9u);
  for (const nlohmann::json& measurement : json["measurements"]) {
    EXPECT_EQ(measurement["status"], "measured");
    EXPECT_TRUE(measurement["requested_workers"].is_null());
    EXPECT_TRUE(measurement["effective_workers"].is_null());
    ASSERT_TRUE(measurement["execution"]["metal"].is_object());
    const nlohmann::json& task = measurement["execution"]["metal"];
    for (const char* cmd : {"reset_command_buffers", "timed_command_buffers", "post_validation_command_buffers",
                            "timed_compute_encoders", "timed_workload_dispatches"}) {
      EXPECT_EQ(task["commands"][cmd], 1u) << cmd;
    }
    EXPECT_TRUE(task["timing"]["gpu_start_seconds"].is_number());
    EXPECT_TRUE(task["timing"]["gpu_end_seconds"].is_number());
    EXPECT_TRUE(measurement["elapsed_seconds"].is_number());
    EXPECT_TRUE(task["timing"]["queue_delay_seconds"].is_null());
    EXPECT_TRUE(measurement["checksum"]["checksum_valid"].get<bool>());
    const auto& canonical = canonical_llm_plan(json, measurement);
    EXPECT_EQ(canonical["scenario"], measurement["scenario"]);
    EXPECT_EQ(canonical["expected_checksum"]["status"], "available");
    EXPECT_EQ(canonical["expected_checksum"]["expected_run_checksum"], measurement["checksum"]["actual_run_checksum"]);
    expect_complete_llm_named_checks(json, measurement);
  }
  expect_complete_llm_checkpoint_lifecycle(json);
}

void expect_bounded_metal_prefill_result(const CliResult& result, const nlohmann::json& json, bool paged) {
  expect_complete_or_unsupported_metal_result(
      result, json, paged ? "paged" : "contiguous",
      paged ? "llm-memory-v2-metal-prefill-paged" : "llm-memory-v2-metal-prefill-contiguous", "prefill",
      "prefill_operation");
  const nlohmann::json& plan = json["resolved_plan"];
  EXPECT_EQ(plan["methodology"]["context_policy"], "full-prompt-population-with-tiled-causal-prefix-scans");
  EXPECT_EQ(plan["component_identities"]["write_pattern_version"],
            paged ? "llm-metal-prefill-paged-full-prompt-affine32-v1"
                  : "llm-metal-prefill-contiguous-full-prompt-affine32-v1");
  EXPECT_EQ(plan["component_identities"]["checksum_pattern_version"],
            paged ? "llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1" : "llm-metal-dual-mod32-v1");

  const nlohmann::json& geometry = plan["geometry"];
  EXPECT_TRUE(geometry["decode"].is_null());
  ASSERT_TRUE(geometry["prefill"].is_object());
  EXPECT_EQ(geometry["prefill"]["prompt_tokens"], 5u);
  EXPECT_EQ(geometry["prefill"]["attention_query_tile_tokens"], 2u);
  EXPECT_EQ(geometry["prefill"]["tile_count"], "3");
  EXPECT_EQ(geometry["prefill"]["attention_prefix_token_visits_per_sequence"], "11");
  EXPECT_EQ(geometry["kv_only_effective_model_payload_bytes_per_work_unit"], "256");
  EXPECT_EQ(geometry["mixed_effective_model_payload_bytes_per_work_unit"], "1048832");

  if (paged) {
    EXPECT_EQ(json["configuration"]["kv_block_tokens"], 4u);
    EXPECT_EQ(plan["layout"]["kv_layout"], "paged");
    EXPECT_EQ(plan["resources"]["block_table_bytes"], "8");
    if (json["status"] == "complete") {
      EXPECT_EQ(plan["layout"]["blocks_per_sequence"], "2");
      EXPECT_EQ(plan["layout"]["last_block_tokens"], "1");
      EXPECT_EQ(plan["layout"]["last_block_valid_bytes"], "8");
    }
  }
  if (json["status"] == "complete") {
    for (const nlohmann::json& measurement : json["measurements"]) {
      const std::string scenario = measurement["scenario"];
      const bool weights_only = scenario == "weights_only";
      const bool kv_only = scenario == "kv_only";
      const auto& canonical = canonical_llm_plan(json, measurement);
      EXPECT_EQ(canonical["kv_write_kind"], weights_only ? "none" : "full_prompt_population");
      EXPECT_EQ(canonical["effective_model_payload_bytes_per_work_unit"],
                weights_only ? "1048576" : (kv_only ? "256" : "1048832"));
      const nlohmann::json& grid = measurement["execution"]["metal"]["grid"];
      EXPECT_EQ(grid["cost_unit"].is_null(), !(weights_only || paged));
      if (paged) {
        EXPECT_EQ(canonical["accounted_bytes_per_work_unit"],
                  weights_only ? "1048576" : (kv_only ? "296" : "1048872"));
        EXPECT_EQ(grid["serial_range_visits_per_lane"], weights_only ? "1" : "0");
      }
    }
  }
  EXPECT_EQ(result.output.find(" tokens/s"), std::string::npos) << result.output;
}

std::vector<std::string> bounded_llm_arguments(const std::string& output_target, bool metal = false,
                                               bool prefill = false, bool paged = false, size_t loop_count = 3) {
  std::vector<std::string> args = {"--llm-memory"};
  if (metal) args.insert(args.end(), {"--llm-memory-backend", "metal"});
  args.insert(args.end(), {"--weight-size-mb", "1", "--layers", "1", "--query-heads", "1", "--kv-heads", "1",
                           "--head-dim", "8", "--kv-element-bytes", "1"});
  if (prefill) {
    args.insert(args.end(), {"--phase", "prefill", "--prompt-tokens", "5", "--attention-query-tile-tokens", "2"});
  } else {
    args.insert(args.end(), {"--context-tokens", "2"});
  }
  args.insert(args.end(), {"--batch-size", "1"});
  if (!metal) args.insert(args.end(), {"--threads", "1"});
  if (paged) args.insert(args.end(), {"--kv-layout", "paged", "--kv-block-tokens", prefill ? "4" : "2"});
  args.insert(args.end(), {"--iterations", "1", "--count", std::to_string(loop_count), "--seed", "42", "--output",
                           output_target});
  return args;
}

}  // namespace

TEST(ExecutableCliIntegrationTest, HelpRemainsHumanAcrossModesAndOutputOrderingIntegration) {
  struct HelpCase {
    std::vector<std::string> arguments;
    const char* heading;
    const char* detail;
  };
  std::vector<HelpCase> cases = {
      {{}, "Usage:", "--cpu-bandwidth"},
      {{"--help"}, "Usage:", "--cpu-bandwidth"},
  };
  const HelpCase modes[] = {
      {{"--cpu-bandwidth"}, "Usage:", "--cpu-bandwidth"},
      {{"--cpu-latency"}, "Usage:", "--cpu-latency"},
      {{"--patterns"}, "Usage:", "--patterns"},
      {{"--analyze-tlb"}, "Usage:", "--analyze-tlb"},
      {{"--gpu-bandwidth"}, "Usage: ./memory_benchmark --gpu-bandwidth [options]", "minimum: 64 MB"},
      {{"--llm-memory"}, "Usage: ./memory_benchmark --llm-memory [options]", "schema 2"},
      {{"--llm-memory", "--llm-memory-backend=metal"},
       "Usage: ./memory_benchmark --llm-memory [options]",
       "schema 2"},
      {{"--analyze-core2core"}, "Usage:", "--analyze-core2core"},
  };
  for (const HelpCase& mode : modes) {
    for (bool help_first : {false, true}) {
      HelpCase test_case = mode;
      if (help_first) {
        test_case.arguments.insert(test_case.arguments.begin() + 1, "--help");
        test_case.arguments.insert(test_case.arguments.end(), {"--output", "-"});
      } else {
        test_case.arguments.insert(test_case.arguments.end(), {"--output=-", "--help"});
      }
      cases.push_back(std::move(test_case));
    }
  }
  for (const HelpCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    const CliResult result = run_memory_benchmark(test_case.arguments);
    expect_process_completed(result);
    EXPECT_EQ(result.exit_code, EXIT_SUCCESS);
    expect_no_runtime_banner(result);
    EXPECT_NE(result.stdout_output.find(test_case.heading), std::string::npos);
    EXPECT_NE(result.stdout_output.find(test_case.detail), std::string::npos);
    EXPECT_TRUE(result.stderr_output.empty()) << result.stderr_output;
    EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
    expect_no_dash_transport_artifacts(result);
  }
}

TEST(ExecutableCliIntegrationTest, InvalidCommandsFailBeforeOutputAcrossModesIntegration) {
  const TemporaryJsonFile output("tlb_invalid_stride_sweep");
  const long page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  struct InvalidCase {
    std::vector<std::string> arguments;
    std::string diagnostic;
    bool diagnostic_is_complete = false;
  };
  const InvalidCase cases[] = {
      {{"--llm-memory", "--gpu-bandwidth"},
       Messages::error_mutually_exclusive_modes("--llm-memory", "--gpu-bandwidth"),
       true},
      {{"--gpu-bandwidth", "--llm-memory"},
       Messages::error_mutually_exclusive_modes("--gpu-bandwidth", "--llm-memory"),
       true},
      {{"--llm-memory", "--weight-size-mb", "1", "--layers", "1", "--query-heads",
        std::to_string(Constants::LLM_JSON_MAX_SAFE_INTEGER + 1), "--kv-heads", "1", "--head-dim", "8",
        "--kv-element-bytes", "1", "--context-tokens", "2", "--batch-size", "1", "--threads", "1", "--iterations", "1",
        "--count", "1", "--output", "-"},
       Messages::error_llm_memory_config_invalid("json-integer-out-of-range"),
       true},
      {{"--cpu-bandwidth", "--cpu-latency"},
       Messages::error_mutually_exclusive_modes("--cpu-bandwidth", "--cpu-latency"),
       true},
      {{"--benchmark"}, Messages::error_unknown_option("--benchmark"), true},
      {{"-W"}, Messages::error_unknown_option("-W"), true},
      {{"--only-bandwidth"}, Messages::error_unknown_option("--only-bandwidth"), true},
      {{"--only-latency"}, Messages::error_unknown_option("--only-latency"), true},
      {{"--patterns", "--cache-size=256", "--output=-"},
       Messages::error_option_not_allowed_in_mode(
           "--cache-size=256",
           "-P/--patterns, -b/--buffer-size, -t/--threads, -i/--iterations, "
           "-r/--count, --seed, -u/--non-cacheable, -o/--output, -S/--sweep, "
           "-X/--sweep-max-runs, -h/--help"),
       true},
      {{"--patterns", "--buffer-size=128", "--sweep=buffer-size=128,256", "--output=-"},
       Messages::error_sweep_option_conflict("--buffer-size", "buffer-size"),
       true},
      {{"--patterns", "--buffer-size", "-G"},
       Messages::error_invalid_value("--buffer-size", "-G", Messages::reason_unsigned_integer_invalid()),
       true},
      {{"--patterns=1"}, Messages::error_option_does_not_take_value("--patterns"), true},
      {{"--patterns", "--patterns"}, Messages::error_duplicate_option("--patterns"), true},
      {{"--threads", "1"}, Messages::error_missing_primary_mode(), true},
      {{"--seed", "42"}, Messages::error_seed_requires_supported_mode(), true},
      {{"--llm-memory", "--weight-size-mb", "1", "--output", "-"},
       Messages::error_llm_memory_missing_required_option("--layers"),
       true},
      {{"--gpu-bandwidth", "--buffer-size", "63", "--output", "-"},
       Messages::error_gpu_buffer_size_below_minimum(63, Constants::GPU_MIN_BUFFER_SIZE_MB),
       true},
      {{"--patterns", "--output", "-", "--not-a-real-option"},
       Messages::error_option_not_allowed_in_mode(
           "--not-a-real-option",
           "-P/--patterns, -b/--buffer-size, -t/--threads, -i/--iterations, "
           "-r/--count, --seed, -u/--non-cacheable, -o/--output, -S/--sweep, "
           "-X/--sweep-max-runs, -h/--help"),
       true},
      {{"--analyze-core2core", "--output", "-", "--buffer-size", "256"},
       Messages::error_option_not_allowed_in_mode("--buffer-size",
                                                  Messages::error_analyze_core_to_core_must_be_used_alone()),
       true},
      {{"--analyze-tlb", "--sweep", "latency-stride-bytes=64,130", "--output", output.path()},
       Messages::error_latency_stride_alignment(130, sizeof(uintptr_t)),
       true},
      {{"--analyze-tlb", "--latency-stride-bytes", "32768", "--output", "-"},
       Messages::error_analyze_tlb_stride_exceeds_page(32768, static_cast<size_t>(page_size)),
       true},
      {{"--analyze-tlb", "--sweep", "latency-stride-bytes=64,130", "--output", "-"},
       Messages::error_latency_stride_alignment(130, sizeof(uintptr_t)),
       true},
  };
  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(testing::PrintToString(test_case.arguments));
    const CliResult result = run_memory_benchmark(test_case.arguments);
    expect_process_completed(result);
    EXPECT_EQ(result.exit_code, EXIT_FAILURE);
    expect_no_runtime_banner(result);
    EXPECT_TRUE(result.stdout_output.empty()) << result.stdout_output;
    const std::string diagnostic = Messages::error_prefix() + test_case.diagnostic + "\n";
    EXPECT_EQ(result.stderr_output.find(diagnostic), 0u) << result.stderr_output;
    if (test_case.diagnostic_is_complete) EXPECT_EQ(result.stderr_output, diagnostic);
    EXPECT_EQ(result.stderr_output.find(Messages::msg_running_sweep(2)), std::string::npos);
    EXPECT_EQ(access(output.path().c_str(), F_OK), -1);
    expect_no_dash_transport_artifacts(result);
  }
}

TEST(ExecutableCliIntegrationTest, CpuBandwidthAndLatencyFileStdoutAndFlagShapedTargetIntegration) {
  const TemporaryJsonFile bw_file_output("cpu bw output with spaces");
  const TemporaryJsonFile lat_file_output("cpu lat output with spaces");
  struct TargetCase {
    const char* name;
    std::vector<std::string> base_args;
    const char* expected_mode;
    const char* stderr_marker;
    std::string target;
    bool is_stdout;
    bool is_flag_shaped;
  };
  const TargetCase cases[] = {
      {"bw_file_with_spaces",
       {"--cpu-bandwidth", "--buffer-size", "1", "--cache-size", "0", "--iterations", "1", "--count", "2", "--threads",
        "1"},
       "cpu_bandwidth", "CPU Bandwidth Mode", bw_file_output.path(), false, false},
      {"bw_stdout",
       {"--cpu-bandwidth", "--buffer-size", "1", "--cache-size", "0", "--iterations", "1", "--count", "1", "--threads",
        "1"},
       "cpu_bandwidth", "CPU Bandwidth Mode", "-", true, false},
      {"bw_flag_shaped_target",
       {"--cpu-bandwidth", "--buffer-size", "1", "--cache-size", "0", "--iterations", "1", "--count", "1", "--threads",
        "1"},
       "cpu_bandwidth", "CPU Bandwidth Mode", "-T", false, true},
      {"lat_file_with_spaces",
       {"--cpu-latency", "--buffer-size", "1", "--cache-size", "0", "--iterations", "1", "--count", "2",
        "--latency-samples", "0"},
       "cpu_latency", "CPU Latency Mode", lat_file_output.path(), false, false},
      {"lat_stdout",
       {"--cpu-latency", "--buffer-size", "1", "--cache-size", "0", "--iterations", "1", "--count", "1",
        "--latency-samples", "0"},
       "cpu_latency", "CPU Latency Mode", "-", true, false},
  };
  for (const TargetCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    std::vector<std::string> args = test_case.base_args;
    args.push_back("--output");
    args.push_back(test_case.target);
    const CliResult result = run_memory_benchmark(args, std::chrono::minutes(2));
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;

    nlohmann::json json;
    if (test_case.is_stdout) {
      json = expect_stdout_json_transport(result, test_case.stderr_marker);
    } else if (test_case.is_flag_shaped) {
      ASSERT_NE(result.directory, nullptr);
      json = expect_atomic_file_json(result, result.directory->path() / test_case.target, false);
      EXPECT_EQ(result.output.find(Messages::error_analyze_tlb_must_be_used_alone()), std::string::npos);
      EXPECT_EQ(result.output.find("mutually exclusive"), std::string::npos);
      EXPECT_NE(result.stdout_output.find(test_case.stderr_marker), std::string::npos);
      EXPECT_FALSE(nlohmann::json::accept(result.stdout_output));
      EXPECT_EQ(count_occurrences(result.output, Messages::msg_results_saved_to("")), 1u) << result.output;
    } else {
      EXPECT_NE(result.stdout_output.find(test_case.stderr_marker), std::string::npos);
      EXPECT_EQ(result.output.find('\r'), std::string::npos);
      json = expect_atomic_file_json(result, test_case.target);
    }

    EXPECT_EQ(json["schema_version"], 1);
    EXPECT_EQ(json["mode"], test_case.expected_mode);
    EXPECT_EQ(json["configuration"]["output_file"], test_case.target);
    EXPECT_EQ(json["status"], "complete");
    EXPECT_TRUE(json["results_complete"].get<bool>());
    EXPECT_TRUE(json["run_accepted"].get<bool>());
  }
}

TEST(ExecutableCliIntegrationTest, PatternAndCpuSweepFileAndStdoutIntegration) {
  const TemporaryJsonFile cpu_bw_file("cpu_bw_sweep_status");
  const TemporaryJsonFile pattern_file("pattern_sweep_banner");
  struct SweepCase {
    const char* name;
    std::vector<std::string> args;
    std::string output_target;
    const char* base_mode;
    size_t runs;
    size_t loops;
    bool is_stdout;
    bool is_pattern;
  };
  const SweepCase cases[] = {
      {"cpu_bw_file",
       {"--cpu-bandwidth", "--cache-size", "0", "--iterations", "1", "--count", "1", "--threads", "1", "--sweep",
        "buffer-size=1,2", "--sweep-max-runs", "2", "--output", cpu_bw_file.path()},
       cpu_bw_file.path(), "cpu_bandwidth", 2u, 1u, false, false},
      {"cpu_lat_stdout",
       {"--cpu-latency", "--cache-size", "0", "--iterations", "1", "--count", "1", "--latency-samples", "0", "--sweep",
        "buffer-size=1,2", "--sweep-max-runs", "2", "--output", "-"},
       "-", "cpu_latency", 2u, 1u, true, false},
      {"pattern_file_nested_loops",
       {"--patterns", "--iterations", "1", "--count", "2", "--threads", "1", "--seed", "42", "--sweep",
        "buffer-size=1,2,3", "--sweep-max-runs", "3", "--output", pattern_file.path()},
       pattern_file.path(), Constants::PATTERNS_JSON_MODE_NAME, 3u, 2u, false, true},
  };
  for (const SweepCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const CliResult result = run_memory_benchmark(test_case.args);
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;

    const nlohmann::json json =
        test_case.is_stdout ? expect_stdout_json_transport(result, Messages::msg_running_sweep(test_case.runs))
                            : expect_atomic_file_json(result, test_case.output_target);
    if (test_case.is_stdout) {
      EXPECT_NE(result.stderr_output.find(Messages::msg_sweep_run_progress(1, test_case.runs)), std::string::npos)
          << result.stderr_output;
    }
    expect_complete_sweep_envelope(json, test_case.base_mode, test_case.runs);
    for (const nlohmann::json& run : json["runs"]) {
      EXPECT_EQ(run["status"], "complete");
      const nlohmann::json& nested = run["result"];
      EXPECT_EQ(nested["status"], "complete");
      EXPECT_TRUE(nested["results_complete"].get<bool>());
      EXPECT_EQ(nested["schema_version"], 1);
      EXPECT_EQ(nested["mode"], test_case.base_mode);
      EXPECT_EQ(nested["configuration"]["output_file"], "");
      EXPECT_TRUE(nested["run_accepted"].get<bool>());
      if (test_case.is_pattern) {
        EXPECT_EQ(nested["counters"]["planned_loops"], test_case.loops);
        EXPECT_EQ(nested["counters"]["completed_loops"], test_case.loops);
      }
    }
  }
}

TEST(ExecutableCliIntegrationTest, PatternBenchmarkWritesSingleJsonDocumentToStdoutIntegration) {
  for (bool automatic_calibration : {false, true}) {
    SCOPED_TRACE(automatic_calibration ? "automatic_calibration" : "explicit_iterations");
    std::vector<std::string> args = {"--patterns", "--buffer-size", "1", "--count", "1", "--threads", "1", "--seed",
                                     "42", "--output", "-"};
    if (!automatic_calibration) args.insert(args.begin() + 3, {"--iterations", "1"});
    const CliResult result = run_memory_benchmark(args);
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    const nlohmann::json json = expect_stdout_json_transport(result, "Patterns Mode (schema v1)");
    EXPECT_EQ(json["schema_version"], 1);
    EXPECT_EQ(json["mode"], Constants::PATTERNS_JSON_MODE_NAME);
    EXPECT_EQ(json["methodology_version"], "patterns-v1-arm64-phase-calibrated-seeded-balanced");
    EXPECT_EQ(json["status"], "complete");
    EXPECT_TRUE(json["results_complete"].get<bool>());
    EXPECT_TRUE(json["run_accepted"].get<bool>());
    EXPECT_EQ(json["counters"]["planned_loops"], 1u);
    EXPECT_EQ(json["counters"]["completed_loops"], 1u);
    EXPECT_EQ(json["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped");
    EXPECT_EQ(json["resolved_plan"]["patterns"]["strided_2mb"]["reason_code"],
              "buffer-too-small-for-stride-transition");
    EXPECT_FALSE(json["calibration"]["attempts"].contains("strided_2mb"));
    if (automatic_calibration) {
      EXPECT_NE(result.stderr_output.find("automatic_calibration"), std::string::npos);
      EXPECT_EQ(json["configuration"]["work_policy"], "automatic_calibration");
      ASSERT_FALSE(json["calibration"]["attempts"].empty());
      for (const auto& row : json["measurements"]) {
        EXPECT_EQ(row["status"], "measured");
        EXPECT_TRUE(row["execution"]["timing"]["valid"].get<bool>());
        EXPECT_TRUE(row["execution"]["timing"]["cpu_raw"].is_object());
        EXPECT_GT(row["execution"]["timing"]["elapsed_seconds"].get<double>(), 0.0);
      }
    }
  }
}

TEST(ExecutableCliIntegrationTest, AnalyzeTlbWritesSingleJsonDocumentToStdoutIntegration) {
  const CliResult result = run_memory_benchmark(
      {"--analyze-tlb", "--tlb-density", "low", "--seed", "42", "--output", "-"}, std::chrono::minutes(15));
  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
  const nlohmann::json json = expect_stdout_json_transport(result, Messages::msg_running_tlb_analysis());
  EXPECT_EQ(json["configuration"]["mode"], Constants::TLB_ANALYSIS_JSON_MODE_NAME);
  EXPECT_EQ(json["configuration"]["schema_version"], 4);
  EXPECT_EQ(json["tlb_analysis"]["status"], "complete");
  EXPECT_TRUE(json["tlb_analysis"]["conclusions_valid"].get<bool>());
  EXPECT_FALSE(json["tlb_analysis"].contains("status_reason"));
}

TEST(ExecutableCliIntegrationTest, CoreToCoreDirectFileAndStdoutIntegration) {
  const TemporaryJsonFile file_output("core2core_v2");
  for (bool is_stdout : {false, true}) {
    SCOPED_TRACE(is_stdout ? "stdout" : "file");
    const size_t count = is_stdout ? 1u : 2u;
    const CliResult result = run_memory_benchmark(
        {"--analyze-core2core", "--count", std::to_string(count), "--latency-samples", "1", "--output",
         is_stdout ? "-" : file_output.path()});
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;

    const nlohmann::json json =
        is_stdout ? expect_stdout_json_transport(result, Messages::msg_running_core_to_core_analysis())
                  : expect_atomic_file_json(result, file_output.path());
    EXPECT_EQ(json["version"], SOFTVERSION);
    EXPECT_EQ(json["configuration"]["mode"], Constants::CORE_TO_CORE_JSON_MODE_NAME);
    EXPECT_EQ(json["configuration"]["schema_version"], Constants::CORE_TO_CORE_JSON_SCHEMA_VERSION);
    EXPECT_EQ(json["core_to_core_latency"]["status"], "complete");
    EXPECT_TRUE(json["core_to_core_latency"]["measurements_complete"].get<bool>());
    ASSERT_EQ(json["core_to_core_latency"]["scenarios"].size(), 3u);
    EXPECT_EQ(json["core_to_core_latency"]["scenarios"][0]["loop_records"].size(), count);
  }
}

TEST(ExecutableCliIntegrationTest, CoreToCoreSweepFileAndStdoutIntegration) {
  const TemporaryJsonFile file_output("core2core_sweep_v2");
  for (bool is_stdout : {false, true}) {
    SCOPED_TRACE(is_stdout ? "stdout" : "file");
    const size_t runs = is_stdout ? 2u : 3u;
    const CliResult result = run_memory_benchmark(
        {"--analyze-core2core", "--count", "1", "--sweep", is_stdout ? "latency-samples=1,2" : "latency-samples=1,2,3",
         "--sweep-max-runs", std::to_string(runs), "--output", is_stdout ? "-" : file_output.path()});
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;

    const nlohmann::json json = is_stdout ? expect_stdout_json_transport(result, Messages::msg_running_sweep(runs))
                                          : expect_atomic_file_json(result, file_output.path());
    if (is_stdout) {
      EXPECT_NE(result.stderr_output.find(Messages::msg_running_core_to_core_analysis()), std::string::npos)
          << result.stderr_output;
    }
    EXPECT_EQ(json["version"], SOFTVERSION);
    expect_complete_sweep_envelope(json, Constants::CORE_TO_CORE_JSON_MODE_NAME, runs);
    for (const nlohmann::json& run : json["runs"]) {
      EXPECT_EQ(run["status"], "complete");
      EXPECT_EQ(run["result"]["configuration"]["schema_version"], Constants::CORE_TO_CORE_JSON_SCHEMA_VERSION);
      EXPECT_EQ(run["result"]["core_to_core_latency"]["status"], "complete");
      EXPECT_TRUE(run["result"]["core_to_core_latency"]["measurements_complete"].get<bool>());
    }
  }
}

TEST(ExecutableCliIntegrationTest, GpuBandwidthFileAndStdoutSchemaV1Integration) {
  const TemporaryJsonFile file_output("gpu_schema_v1");
  for (bool is_stdout : {true, false}) {
    SCOPED_TRACE(is_stdout ? "stdout_explicit" : "file_automatic_calibration");
    const std::string target = is_stdout ? "-" : file_output.path();
    const size_t loop_count = is_stdout ? 3u : 1u;
    const size_t measurement_count = loop_count * 3u;
    std::vector<std::string> args = {"--gpu-bandwidth", "--buffer-size", "64", "--count", std::to_string(loop_count),
                                     "--seed", "42", "--output", target};
    if (is_stdout) args.insert(args.begin() + 3, {"--iterations", "1"});
    const CliResult result = run_memory_benchmark(args);
    expect_process_completed(result);

    const nlohmann::json json = is_stdout ? expect_stdout_json_transport(result, Messages::msg_running_gpu_bandwidth())
                                          : expect_atomic_file_json(result, file_output.path());
    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json["software_version"], SOFTVERSION);
    EXPECT_EQ(json["schema_version"], Constants::GPU_JSON_SCHEMA_VERSION);
    EXPECT_EQ(json["mode"], Constants::GPU_JSON_MODE_NAME);
    EXPECT_EQ(json["configuration"]["output_file"], target);
    EXPECT_EQ(json["configuration"]["base_seed_uint64_decimal"], "42");
    EXPECT_EQ(json["configuration"]["loop_count"], loop_count);
    if (is_stdout) {
      EXPECT_EQ(json["configuration"]["iterations"], 1u);
      EXPECT_EQ(json["configuration"]["work_policy"], "explicit-fixed-work");
    } else {
      EXPECT_TRUE(json["configuration"]["iterations"].is_null());
      EXPECT_EQ(json["configuration"]["work_policy"], "automatic-calibration");
    }
    EXPECT_EQ(json["counters"]["planned_loops"], loop_count);
    EXPECT_EQ(json["counters"]["planned_measurements"], measurement_count);
    ASSERT_EQ(json["measurements"].size(), measurement_count);

    const std::string status = json["status"].get<std::string>();
    if (status == "complete") {
      EXPECT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
      EXPECT_TRUE(json["results_complete"].get<bool>());
      EXPECT_TRUE(json["conclusions_valid"].get<bool>());
      EXPECT_EQ(json["counters"]["completed_loops"], loop_count);
      EXPECT_EQ(json["counters"]["validated_measurements"], measurement_count);
      for (const nlohmann::json& measurement : json["measurements"]) {
        EXPECT_EQ(measurement["status"], "measured");
        EXPECT_GT(measurement["value_gb_s"].get<double>(), 0.0);
        EXPECT_EQ(measurement["timed"]["command_buffer_count"], 1u);
        EXPECT_EQ(measurement["timed"]["compute_encoder_count"], 1u);
        EXPECT_EQ(measurement["validation"]["validation_status"], "passed");
        if (!is_stdout) {
          const std::string quality = measurement["duration_quality"].get<std::string>();
          EXPECT_TRUE(quality == "within-target-window" || quality == "below-target-window" ||
                      quality == "above-target-window" || quality == "dispatch-cap-below-target" ||
                      quality == "payload-cap-below-target" || quality == "single-pass-exceeds-window")
              << quality;
        }
      }
      if (is_stdout) {
        EXPECT_EQ(json["work_plans"][2]["exact_payload_bytes"], "134217728");
      } else {
        for (const char* operation : {"read", "write", "copy"}) {
          const nlohmann::json& attempts = json["excluded_calibration_attempts"][operation];
          ASSERT_GE(attempts.size(), 2u);
          EXPECT_EQ(attempts[0]["purpose"], "pilot");
          EXPECT_EQ(attempts[1]["purpose"], "duration-trial");
          EXPECT_TRUE(attempts.back()["valid"].get<bool>());
        }
      }
      EXPECT_EQ(json["backend"]["allocation"]["buffer_a"]["storage_mode"], "private");
      EXPECT_EQ(json["backend"]["allocation"]["buffer_a"]["hazard_tracking_mode"], "tracked");
      EXPECT_EQ(json["backend"]["compilation"]["kernel_source_sha256"].get<std::string>().size(), 64u);
    } else if (status == "unsupported") {
      EXPECT_EQ(result.exit_code, EXIT_FAILURE) << result.output;
      EXPECT_FALSE(json["results_complete"].get<bool>());
      EXPECT_FALSE(json["conclusions_valid"].get<bool>());
      ASSERT_FALSE(json["reason_code"].get<std::string>().empty());
      EXPECT_EQ(json["backend"]["initialization_status"], "unsupported");
      EXPECT_EQ(json["backend"]["reason_code"], json["reason_code"]);
    } else {
      FAIL() << "GPU integration returned unexpected status: " << status << '\n' << result.stderr_output;
    }
  }
}

TEST(ExecutableCliIntegrationTest, LlmMetalDirectFileAndStdoutIntegration) {
  for (bool prefill : {false, true}) {
    for (bool paged : {false, true}) {
      SCOPED_TRACE(std::string(prefill ? "prefill_" : "decode_") + (paged ? "paged" : "contiguous"));
      const CliResult result = run_memory_benchmark(bounded_llm_arguments("-", /*metal=*/true, prefill, paged));
      expect_process_completed(result);
      const nlohmann::json json = expect_stdout_json_transport(result);
      if (prefill) {
        expect_bounded_metal_prefill_result(result, json, paged);
      } else {
        expect_complete_or_unsupported_metal_result(
            result, json, paged ? "paged" : "contiguous",
            paged ? "llm-memory-v2-metal-decode-paged" : "llm-memory-v2-metal-decode-contiguous");
        if (paged) EXPECT_EQ(json["configuration"]["kv_block_tokens"], 2u);
      }
    }
  }

  const TemporaryJsonFile file_output("llm_metal_schema_v2");
  const CliResult result = run_memory_benchmark(bounded_llm_arguments(file_output.path(), /*metal=*/true));
  expect_process_completed(result);
  const nlohmann::json json = expect_atomic_file_json(result, file_output.path());
  expect_complete_or_unsupported_metal_result(result, json);
  EXPECT_EQ(json["configuration"]["output_file"], file_output.path());
}

TEST(ExecutableCliIntegrationTest, LlmCpuDecodeFileAndStdoutSchemaV2Integration) {
  nlohmann::json first_stdout;
  for (int repetition = 0; repetition < 2; ++repetition) {
    SCOPED_TRACE(repetition);
    const CliResult result = run_memory_benchmark(bounded_llm_arguments("-"));
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    const nlohmann::json json = expect_stdout_json_transport(
        result, Messages::report_llm_memory_header("cpu", "decode", "decode_step", "contiguous"));
    expect_complete_cpu_llm_document(json, "decode", "contiguous",
                                     Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION, "decode_step");
    EXPECT_EQ(json["software"]["version"], SOFTVERSION);
    EXPECT_TRUE(json["software"]["timestamp"].is_string());
    EXPECT_EQ(json["configuration"]["output_file"], "-");
    EXPECT_EQ(json["configuration"]["base_seed_uint64_decimal"], "42");
    EXPECT_EQ(json["configuration"]["iterations"], 1u);
    EXPECT_EQ(json["configuration"]["loop_count"], 3u);
    EXPECT_EQ(json["configuration"]["resolved_sources"]["backend"], "default");
    EXPECT_EQ(json["configuration"]["resolved_sources"]["phase"], "default");
    EXPECT_EQ(json["configuration"]["resolved_sources"]["kv_layout"], "default");
    EXPECT_EQ(json["counters"]["planned_measurements"], 9u);
    EXPECT_EQ(json["counters"]["measured_measurements"], 9u);
    ASSERT_EQ(json["measurements"].size(), 9u);
    for (const nlohmann::json& measurement : json["measurements"]) {
      EXPECT_EQ(measurement["status"], "measured");
      EXPECT_EQ(canonical_llm_plan(json, measurement)["work_unit_kind"], "decode_step");
      EXPECT_EQ(canonical_llm_plan(json, measurement)["work_units"], 1u);
      EXPECT_EQ(measurement["completed_work_units"], 1u);
      expect_complete_llm_named_checks(json, measurement);
      EXPECT_TRUE(measurement["checksum"]["checksum_valid"].get<bool>());
      EXPECT_TRUE(measurement["synthetic_work_unit_latency_seconds"].is_number());
      EXPECT_TRUE(measurement["synthetic_memory_work_units_per_second"].is_number());
      EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_number());
    }
    expect_complete_llm_checkpoint_lifecycle(json);

    if (repetition == 0) {
      first_stdout = json;
      continue;
    }
    EXPECT_EQ(first_stdout["configuration"], json["configuration"]);
    EXPECT_EQ(first_stdout["seeds"], json["seeds"]);
    for (const char* field : {"methodology", "geometry", "model_work_plan", "scenario_plans"}) {
      EXPECT_EQ(first_stdout["resolved_plan"][field], json["resolved_plan"][field]) << field;
    }
    ASSERT_EQ(first_stdout["measurements"].size(), json["measurements"].size());
    for (size_t index = 0; index < first_stdout["measurements"].size(); ++index) {
      EXPECT_EQ(first_stdout["measurements"][index]["plan_ref"], json["measurements"][index]["plan_ref"]);
      EXPECT_EQ(first_stdout["measurements"][index]["checksum"], json["measurements"][index]["checksum"]);
    }
  }

  const TemporaryJsonFile file_output("llm_schema_v2");
  const CliResult result = run_memory_benchmark(bounded_llm_arguments(file_output.path()));
  expect_process_completed(result);
  ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.output;
  const nlohmann::json json = expect_atomic_file_json(result, file_output.path());
  expect_complete_cpu_llm_document(json, "decode", "contiguous",
                                   Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION, "decode_step");
  EXPECT_EQ(json["configuration"]["output_file"], file_output.path());
  EXPECT_FALSE(json["resolved_plan"]["plan_identity"].get<std::string>().empty());
  for (const auto& plan : json["resolved_plan"]["scenario_plans"]) EXPECT_EQ(plan["model_ref"], "resolved_plan");
  expect_complete_llm_checkpoint_lifecycle(json);

  {
    SCOPED_TRACE("checkpoint_write_failed_directory_target");
    TemporaryCliDirectory output_parent;
    const std::filesystem::path output_target = output_parent.path() / "existing-output-directory";
    std::error_code create_error;
    ASSERT_TRUE(std::filesystem::create_directory(output_target, create_error)) << create_error.message();
    const CliResult failed = run_memory_benchmark(bounded_llm_arguments(output_target.string()));
    expect_process_completed(failed);
    EXPECT_EQ(failed.exit_code, EXIT_FAILURE);
    EXPECT_TRUE(std::filesystem::is_directory(output_target));
    EXPECT_FALSE(std::filesystem::exists(output_target.string() + ".tmp"));
    EXPECT_EQ(count_occurrences(failed.stderr_output,
                                Messages::error_llm_memory_run_failed("checkpoint-write-failed")),
              1u)
        << failed.stderr_output;
  }
}

TEST(ExecutableCliIntegrationTest, LlmCpuPrefillContiguousAndPagedSchemaV2Integration) {
  for (bool paged : {false, true}) {
    SCOPED_TRACE(paged ? "paged" : "contiguous");
    const CliResult result =
        run_memory_benchmark(bounded_llm_arguments("-", /*metal=*/false, /*prefill=*/true, paged));
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    const nlohmann::json json = expect_stdout_json_transport(
        result, Messages::report_llm_memory_header("cpu", "prefill", "prefill_operation",
                                                   paged ? "paged" : "contiguous"));
    expect_complete_cpu_llm_document(
        json, "prefill", paged ? "paged" : "contiguous",
        paged ? Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION
              : Constants::LLM_CPU_PREFILL_CONTIGUOUS_METHODOLOGY_VERSION,
        "prefill_operation");

    const nlohmann::json& config = json["configuration"];
    EXPECT_TRUE(config["visible_context_tokens"].is_null());
    EXPECT_EQ(config["prompt_tokens"], 5u);
    EXPECT_EQ(config["attention_query_tile_tokens"], 2u);
    EXPECT_EQ(config["resolved_sources"]["phase"], "explicit");
    if (paged) {
      EXPECT_EQ(config["kv_block_tokens"], 4u);
      EXPECT_EQ(config["resolved_sources"]["kv_layout"], "explicit");
      EXPECT_EQ(config["resolved_sources"]["kv_block_tokens"], "explicit");
    }

    const nlohmann::json& plan = json["resolved_plan"];
    EXPECT_TRUE(plan["geometry"]["prefill"].is_object());
    EXPECT_EQ(plan["layout"]["kv_layout"], paged ? "paged" : "contiguous");
    if (paged) EXPECT_EQ(plan["layout"]["kv_block_tokens"], 4u);
    EXPECT_TRUE(json["backend_evidence"]["cpu"]["prefill"].is_object());
    EXPECT_EQ(json["backend_evidence"]["cpu"]["paged"].is_object(), paged);
    ASSERT_EQ(plan["scenario_plans"].size(), 3u);
    EXPECT_EQ(plan["scenario_plans"][0]["effective_model_payload_bytes_per_work_unit"], "1048576");
    EXPECT_EQ(plan["scenario_plans"][1]["effective_model_payload_bytes_per_work_unit"], "256");
    EXPECT_EQ(plan["scenario_plans"][2]["effective_model_payload_bytes_per_work_unit"], "1048832");
    EXPECT_EQ(json["counters"]["completed_work_units"], "9");
    EXPECT_EQ(json["counters"]["completed_effective_model_payload_bytes"], "6292992");

    ASSERT_EQ(json["measurements"].size(), 9u);
    for (const nlohmann::json& measurement : json["measurements"]) {
      EXPECT_EQ(measurement["status"], "measured");
      EXPECT_EQ(measurement["execution"]["status"], "valid");
      EXPECT_TRUE(measurement["checksum"]["checksum_valid"].get<bool>());
      EXPECT_EQ(canonical_llm_plan(json, measurement)["expected_checksum"]["expected_run_checksum"],
                measurement["checksum"]["actual_run_checksum"]);
      expect_complete_llm_named_checks(json, measurement);
    }
    expect_complete_llm_checkpoint_lifecycle(json);
    EXPECT_NE(result.stderr_output.find("176"), std::string::npos);
    EXPECT_NE(result.stderr_output.find("80"), std::string::npos);
    EXPECT_EQ(result.stderr_output.find("Crossover:"), std::string::npos);
    EXPECT_EQ(result.stderr_output.find(" tokens/s"), std::string::npos);
  }
}

TEST(ExecutableCliIntegrationTest, LlmMemorySweepsFileStdoutAndPagedPrefillRemainderIntegration) {
  {
    SCOPED_TRACE("paged_decode_sweep_file");
    const TemporaryJsonFile output("llm_memory_paged_sweep");
    const CliResult result = run_memory_benchmark(
        {"--llm-memory", "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4", "--kv-heads", "2",
         "--head-dim", "16", "--kv-layout", "paged", "--sweep", "context-tokens=16,32", "--sweep",
         "kv-block-tokens=8,16", "--sweep-max-runs", "4", "--threads", "1", "--iterations", "1", "--count", "1",
         "--seed", "42", "--output", output.path()});
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    EXPECT_NE(result.output.find(Messages::msg_running_sweep(4)), std::string::npos);
    const nlohmann::json envelope = expect_atomic_file_json(result, output.path());
    expect_complete_sweep_envelope(envelope, Constants::LLM_JSON_MODE_NAME, 4u);
    EXPECT_EQ(envelope["configuration"]["run_count"], 4u);
    const std::array<std::pair<size_t, size_t>, 4> expected_pairs = {{{16u, 8u}, {16u, 16u}, {32u, 8u}, {32u, 16u}}};
    for (size_t i = 0; i < expected_pairs.size(); ++i) {
      const nlohmann::json& run = envelope["runs"][i];
      EXPECT_EQ(run["index"], i);
      EXPECT_EQ(run["status"], "complete");
      EXPECT_EQ(run["parameters"]["context-tokens"], expected_pairs[i].first);
      EXPECT_EQ(run["parameters"]["kv-block-tokens"], expected_pairs[i].second);
      const nlohmann::json& nested = run["result"];
      EXPECT_EQ(nested["schema_version"], Constants::LLM_JSON_SCHEMA_VERSION);
      EXPECT_EQ(nested["mode"], Constants::LLM_JSON_MODE_NAME);
      EXPECT_EQ(nested["status"], "complete");
      EXPECT_TRUE(nested["results_complete"].get<bool>());
      EXPECT_TRUE(nested["run_accepted"].get<bool>());
      EXPECT_EQ(nested["configuration"]["visible_context_tokens"], expected_pairs[i].first);
      EXPECT_EQ(nested["configuration"]["kv_block_tokens"], expected_pairs[i].second);
      EXPECT_EQ(nested["configuration"]["base_seed_uint64_decimal"], "42");
      const nlohmann::json& json_peak = nested["backend_evidence"]["cpu"]["resources"]["json_output_peak_estimate"];
      EXPECT_TRUE(json_peak["enabled"].get<bool>());
      EXPECT_GT(std::stoull(json_peak["total_bytes"].get<std::string>()), 0u);
    }
  }

  {
    SCOPED_TRACE("prefill_sweep_stdout");
    const CliResult result = run_memory_benchmark(
        {"--llm-memory", "--phase", "prefill", "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4",
         "--kv-heads", "2", "--head-dim", "16", "--sweep", "prompt-tokens=16,32", "--sweep",
         "attention-query-tile-tokens=8,16", "--threads", "1", "--iterations", "1", "--count", "1", "--seed", "42",
         "--output", "-"});
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    const nlohmann::json envelope = expect_stdout_json_transport(result, Messages::msg_running_sweep(4));
    expect_complete_sweep_envelope(envelope, Constants::LLM_JSON_MODE_NAME, 4u);
    EXPECT_EQ(envelope["runs"][0]["parameters"]["prompt-tokens"], 16u);
    EXPECT_EQ(envelope["runs"][0]["parameters"]["attention-query-tile-tokens"], 8u);
    EXPECT_EQ(envelope["runs"][3]["parameters"]["prompt-tokens"], 32u);
    EXPECT_EQ(envelope["runs"][3]["parameters"]["attention-query-tile-tokens"], 16u);
    EXPECT_EQ(envelope["runs"][3]["result"]["configuration"]["prompt_tokens"], 32u);
    EXPECT_EQ(envelope["runs"][3]["result"]["configuration"]["attention_query_tile_tokens"], 16u);
  }

  struct RemainderSweepCase {
    const char* name;
    std::vector<std::string> args;
    size_t expected_runs;
    bool is_metal;
  };
  const RemainderSweepCase remainder_cases[] = {
      {"cpu_paged_prefill_remainder_sweep",
       {"--llm-memory", "--phase", "prefill", "--kv-layout", "paged", "--weight-size-mb", "1", "--layers", "2",
        "--query-heads", "4", "--kv-heads", "2", "--head-dim", "16", "--attention-query-tile-tokens", "8", "--sweep",
        "prompt-tokens=16,32", "--sweep", "kv-block-tokens=8,16", "--threads", "1", "--iterations", "1", "--count", "1",
        "--seed", "42", "--output", "-"},
       4u, false},
      {"metal_paged_prefill_remainder_sweep",
       {"--llm-memory", "--llm-memory-backend", "metal", "--phase", "prefill", "--kv-layout", "paged",
        "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4", "--kv-heads", "2", "--head-dim", "16",
        "--attention-query-tile-tokens", "8", "--kv-block-tokens", "16", "--sweep", "prompt-tokens=16,32",
        "--iterations", "1", "--count", "1", "--seed", "42", "--output", "-"},
       2u, true},
  };
  for (const RemainderSweepCase& test_case : remainder_cases) {
    SCOPED_TRACE(test_case.name);
    const CliResult result = run_memory_benchmark(test_case.args);
    expect_process_completed(result);
    const nlohmann::json envelope = parse_single_stdout_json(result);
    expect_no_dash_transport_artifacts(result);
    if (test_case.is_metal && result.exit_code != EXIT_SUCCESS && envelope.contains("runs") &&
        !envelope["runs"].empty() && envelope["runs"][0]["result"].is_object() &&
        envelope["runs"][0]["result"]["status"] == "unsupported") {
      GTEST_SKIP() << "Metal backend unsupported on this host: " << envelope["runs"][0]["result"]["reason_code"];
    }
    ASSERT_EQ(result.exit_code, EXIT_SUCCESS) << result.stderr_output;
    expect_complete_sweep_envelope(envelope, Constants::LLM_JSON_MODE_NAME, test_case.expected_runs);
    for (const nlohmann::json& run : envelope["runs"]) {
      EXPECT_EQ(run["status"], "complete");
      EXPECT_TRUE(run["result"]["run_accepted"].get<bool>());
      EXPECT_TRUE(run["result"]["resolved_plan"]["layout"]["decode_append_offset_in_last_block"].is_null());
    }
  }

  {
    SCOPED_TRACE("preinit_json_peak_overflow_sweep_failure");
    const CliResult result = run_memory_benchmark(
        {"--llm-memory", "--weight-size-mb", "1", "--layers", "1", "--query-heads", "1", "--kv-heads", "1",
         "--head-dim", "8", "--kv-element-bytes", "1", "--sweep", "context-tokens=2,4", "--threads", "1",
         "--iterations", "1", "--count", std::to_string(Constants::LLM_JSON_MAX_SAFE_INTEGER / 3), "--seed", "42",
         "--output", "-"});
    expect_process_completed(result);
    ASSERT_EQ(result.exit_code, EXIT_FAILURE) << result.stderr_output;
    const nlohmann::json envelope = parse_single_stdout_json(result);
    EXPECT_EQ(envelope["status"], "failed");
    EXPECT_EQ(envelope["status_reason"], "json-output-peak-bytes-overflow");
    EXPECT_FALSE(envelope["conclusions_valid"].get<bool>());
    ASSERT_EQ(envelope["runs"].size(), 1u);
    EXPECT_EQ(envelope["runs"][0]["status"], "failed");
    EXPECT_EQ(envelope["runs"][0]["status_reason"], "json-output-peak-bytes-overflow");
    EXPECT_TRUE(envelope["runs"][0]["result"].is_null());
    expect_no_dash_transport_artifacts(result);
  }
}
