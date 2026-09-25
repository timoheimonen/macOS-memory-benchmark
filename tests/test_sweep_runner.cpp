// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

namespace {

using Json = nlohmann::ordered_json;



std::vector<Json> make_parameters(size_t count) {
  std::vector<Json> parameters;
  parameters.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    parameters.push_back({{"value", index + 1}});
  }
  return parameters;
}

Json make_standard_result(const std::string& status, bool results_complete, const std::string& reason = "") {
  return {{"configuration",
           {{"mode", Constants::BENCHMARK_JSON_MODE_NAME},
            {"benchmark_schema_version",
             Constants::BENCHMARK_JSON_SCHEMA_VERSION},
            {"output_file", ""}}},
          {"status", status},
          {"status_reason", reason},
          {"results_complete", results_complete},
          {"conclusions_valid", results_complete}};
}

Json make_tlb_result(const std::string& status, bool conclusions_valid) {
  return {{"tlb_analysis", {{"status", status}, {"conclusions_valid", conclusions_valid}}}};
}

Json make_pattern_result(const std::string& status, bool results_complete, const std::string& reason = "") {
  return {{"status", status}, {"status_reason", reason}, {"results_complete", results_complete}};
}

Json make_core_to_core_result(const std::string& status, bool measurements_complete) {
  return {{"core_to_core_latency", {{"status", status}, {"measurements_complete", measurements_complete}}}};
}

SweepExecutionHooks make_hooks(const std::vector<SweepRunOutcome>& outcomes, std::vector<Json>& checkpoints,
                               std::vector<bool>& announce_flags, size_t& executed_runs) {
  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    ++executed_runs;
    return outcomes.at(run_index);
  };
  hooks.stop_requested = []() { return false; };
  hooks.elapsed_seconds = []() { return 1.25; };
  hooks.utc_timestamp = []() { return "2026-01-01T00:00:00Z"; };
  hooks.write_checkpoint = [&](const Json& output, bool announce_success) {
    checkpoints.push_back(output);
    announce_flags.push_back(announce_success);
    return EXIT_SUCCESS;
  };
  return hooks;
}

}  // namespace

TEST(SweepRunnerTest, StandardNestedCompletionRequiresCurrentSchema3Contract) {
  Json schema_3_complete = make_standard_result("complete", true);
  Json schema_3_partial = make_standard_result(
      "partial", false, "benchmark loops remain");
  Json schema_3_partial_without_reason = make_standard_result("partial", false);
  Json schema_3_interrupted = make_standard_result(
      "interrupted", false, "stop requested");
  Json schema_3_interrupted_without_reason =
      make_standard_result("interrupted", false);
  Json schema_3_failed = make_standard_result(
      "failed", false, "benchmark timer failed");
  Json schema_3_failed_without_reason = make_standard_result("failed", false);
  Json schema_3_results_false = schema_3_complete;
  schema_3_results_false["results_complete"] = false;
  Json schema_3_conclusions_false = schema_3_complete;
  schema_3_conclusions_false["conclusions_valid"] = false;
  Json schema_3_incomplete_with_reason = schema_3_results_false;
  schema_3_incomplete_with_reason["status_reason"] =
      "completion metadata disagrees";
  Json schema_3_missing_status = schema_3_complete;
  schema_3_missing_status.erase("status");
  Json schema_3_invalid_status = schema_3_complete;
  schema_3_invalid_status["status"] = true;
  Json schema_3_invalid_reason = schema_3_partial_without_reason;
  schema_3_invalid_reason["status_reason"] = true;
  Json schema_3_missing_conclusions = schema_3_complete;
  schema_3_missing_conclusions.erase("conclusions_valid");
  Json schema_3_invalid_conclusions = schema_3_complete;
  schema_3_invalid_conclusions["conclusions_valid"] = 1;
  Json schema_3_missing_output = schema_3_complete;
  schema_3_missing_output["configuration"].erase("output_file");
  Json schema_3_invalid_output = schema_3_complete;
  schema_3_invalid_output["configuration"]["output_file"] = nullptr;
  Json schema_3_interrupted_missing_output = schema_3_interrupted;
  schema_3_interrupted_missing_output["configuration"].erase("output_file");
  Json schema_3_invalid_results = schema_3_complete;
  schema_3_invalid_results["results_complete"] = 1;
  Json schema_3_failed_invalid_results = schema_3_failed;
  schema_3_failed_invalid_results["results_complete"] = "false";
  Json schema_3_missing_results = schema_3_complete;
  schema_3_missing_results.erase("results_complete");
  Json schema_3_missing_mode = schema_3_complete;
  schema_3_missing_mode["configuration"].erase("mode");
  Json schema_3_invalid_mode = schema_3_complete;
  schema_3_invalid_mode["configuration"]["mode"] = true;
  Json schema_3_wrong_mode = schema_3_complete;
  schema_3_wrong_mode["configuration"]["mode"] =
      Constants::PATTERNS_JSON_MODE_NAME;
  Json missing_configuration = schema_3_complete;
  missing_configuration.erase("configuration");
  Json invalid_configuration = schema_3_complete;
  invalid_configuration["configuration"] = Json::array();
  Json missing_schema = schema_3_complete;
  missing_schema["configuration"].erase("benchmark_schema_version");
  Json schema_2 = schema_3_complete;
  schema_2["configuration"]["benchmark_schema_version"] = 2;
  Json invalid_schema_type = schema_3_complete;
  invalid_schema_type["configuration"]["benchmark_schema_version"] = "3";
  Json unsupported_schema_1 = schema_3_complete;
  unsupported_schema_1["configuration"]["benchmark_schema_version"] = 1;
  Json unsupported_schema_4 = schema_3_complete;
  unsupported_schema_4["configuration"]["benchmark_schema_version"] = 4;
  Json top_level_gpu_identity = {
      {"mode", Constants::GPU_JSON_MODE_NAME},
      {"schema_version", 1},
  };
  Json mixed_top_level_gpu_identity = schema_3_complete;
  mixed_top_level_gpu_identity["mode"] = Constants::GPU_JSON_MODE_NAME;
  mixed_top_level_gpu_identity["schema_version"] = 1;
  Json mixed_pattern_identity = schema_3_complete;
  mixed_pattern_identity["configuration"]["pattern_schema_version"] = 3;
  Json mixed_mode_schema_identity = schema_3_complete;
  mixed_mode_schema_identity["configuration"]["schema_version"] = 4;
  Json mixed_sweep_identity = schema_3_complete;
  mixed_sweep_identity["configuration"]["sweep_schema_version"] = 1;

  struct CompletionCase {
    const char* name;
    Json result;
    SweepAttemptStatus expected_status;
    const char* expected_reason;
  };
  const std::vector<CompletionCase> cases = {
      {"schema 3 complete with nested empty output", schema_3_complete,
       SweepAttemptStatus::Complete, ""},
      {"schema 3 partial with reason", schema_3_partial,
       SweepAttemptStatus::Partial, "benchmark loops remain"},
      {"schema 3 partial without reason", schema_3_partial_without_reason,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 interrupted with reason", schema_3_interrupted,
       SweepAttemptStatus::Interrupted, "stop requested"},
      {"schema 3 interrupted without reason",
       schema_3_interrupted_without_reason, SweepAttemptStatus::Interrupted,
       "nested-run-interrupted"},
      {"schema 3 failed with reason", schema_3_failed,
       SweepAttemptStatus::Failed, "benchmark timer failed"},
      {"schema 3 failed without reason", schema_3_failed_without_reason,
       SweepAttemptStatus::Failed, "nested-run-failed"},
      {"schema 3 with results false", schema_3_results_false,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 with conclusions false", schema_3_conclusions_false,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 incomplete preserves reason", schema_3_incomplete_with_reason,
       SweepAttemptStatus::Partial, "completion metadata disagrees"},
      {"schema 3 without status", schema_3_missing_status,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 with invalid status type", schema_3_invalid_status,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 with invalid reason type", schema_3_invalid_reason,
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 without conclusions", schema_3_missing_conclusions,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid conclusions type",
       schema_3_invalid_conclusions, SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 without output", schema_3_missing_output,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid output type", schema_3_invalid_output,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 interrupted without output uses contract reason",
       schema_3_interrupted_missing_output, SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 with invalid results type", schema_3_invalid_results,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 failed with invalid results uses contract reason",
       schema_3_failed_invalid_results, SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 without results", schema_3_missing_results,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 without mode", schema_3_missing_mode,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid mode type", schema_3_invalid_mode,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with wrong mode", schema_3_wrong_mode,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"missing configuration", missing_configuration,
       SweepAttemptStatus::Partial, "missing-standard-schema-version"},
      {"invalid configuration type", invalid_configuration,
       SweepAttemptStatus::Partial, "missing-standard-schema-version"},
      {"missing schema identity", missing_schema, SweepAttemptStatus::Partial,
       "missing-standard-schema-version"},
      {"released schema 2", schema_2, SweepAttemptStatus::Partial,
       "unsupported-standard-schema-version"},
      {"invalid schema identity type", invalid_schema_type,
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"unsupported explicit schema 1", unsupported_schema_1,
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"unsupported explicit schema 4", unsupported_schema_4,
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"top-level GPU identity", top_level_gpu_identity,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed top-level GPU identity", mixed_top_level_gpu_identity,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed pattern identity", mixed_pattern_identity,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed TLB or core-to-core identity", mixed_mode_schema_identity,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed sweep identity", mixed_sweep_identity,
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
  };

  for (const CompletionCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const SweepNestedCompletion completion = classify_sweep_nested_completion(
        SweepNestedMode::Standard, test_case.result);
    EXPECT_EQ(completion.status, test_case.expected_status);
    EXPECT_EQ(completion.reason, test_case.expected_reason);
  }
}

TEST(SweepRunnerTest, NestedCompletionIsModeAware) {
  const SweepNestedCompletion standard =
      classify_sweep_nested_completion(SweepNestedMode::Standard, make_standard_result("complete", true));
  EXPECT_EQ(standard.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("complete", true));
  EXPECT_EQ(tlb.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion pattern =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, make_pattern_result("complete", true));
  EXPECT_EQ(pattern.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion core_to_core =
      classify_sweep_nested_completion(SweepNestedMode::CoreToCore, make_core_to_core_result("complete", true));
  EXPECT_EQ(core_to_core.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion incomplete_core_to_core =
      classify_sweep_nested_completion(SweepNestedMode::CoreToCore, make_core_to_core_result("complete", false));
  EXPECT_EQ(incomplete_core_to_core.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(incomplete_core_to_core.reason, "nested-core-to-core-result-incomplete");

  const SweepNestedCompletion interrupted_core_to_core =
      classify_sweep_nested_completion(SweepNestedMode::CoreToCore, make_core_to_core_result("interrupted", false));
  EXPECT_EQ(interrupted_core_to_core.status, SweepAttemptStatus::Interrupted);
  EXPECT_EQ(interrupted_core_to_core.reason, "nested-core-to-core-run-interrupted");

  const SweepNestedCompletion wrong_core_to_core_shape =
      classify_sweep_nested_completion(SweepNestedMode::CoreToCore, make_standard_result("complete", true));
  EXPECT_EQ(wrong_core_to_core_shape.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(wrong_core_to_core_shape.reason, "missing-core-to-core-result");

  const SweepNestedCompletion unvalidated_tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("complete", false));
  EXPECT_EQ(unvalidated_tlb.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(unvalidated_tlb.reason, "nested-tlb-result-incomplete");

  const SweepNestedCompletion partial_tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("partial", false));
  EXPECT_EQ(partial_tlb.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(partial_tlb.reason, "nested-tlb-result-incomplete");

  const SweepNestedCompletion interrupted_tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("interrupted", false));
  EXPECT_EQ(interrupted_tlb.status, SweepAttemptStatus::Interrupted);
  EXPECT_EQ(interrupted_tlb.reason, "nested-tlb-run-interrupted");

  const Json error_tlb_result = make_tlb_result("error", false);
  const SweepNestedCompletion error_tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, error_tlb_result);
  EXPECT_EQ(error_tlb.status, SweepAttemptStatus::Failed);
  EXPECT_EQ(error_tlb.reason, "nested-tlb-run-error");
  EXPECT_FALSE(error_tlb_result["tlb_analysis"].contains("status_reason"));

  const SweepNestedCompletion incomplete_pattern = classify_sweep_nested_completion(
      SweepNestedMode::Patterns, make_pattern_result("partial", false, "pattern loops remain"));
  EXPECT_EQ(incomplete_pattern.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(incomplete_pattern.reason, "pattern loops remain");

  const SweepNestedCompletion inconsistent_complete =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, make_pattern_result("complete", false));
  EXPECT_EQ(inconsistent_complete.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(inconsistent_complete.reason, "nested-pattern-result-incomplete");

  const SweepNestedCompletion interrupted_pattern = classify_sweep_nested_completion(
      SweepNestedMode::Patterns, make_pattern_result("interrupted", false, "stop requested"));
  EXPECT_EQ(interrupted_pattern.status, SweepAttemptStatus::Interrupted);
  EXPECT_EQ(interrupted_pattern.reason, "stop requested");

  const SweepNestedCompletion failed_pattern = classify_sweep_nested_completion(
      SweepNestedMode::Patterns, make_pattern_result("failed", false, "allocation failed"));
  EXPECT_EQ(failed_pattern.status, SweepAttemptStatus::Failed);
  EXPECT_EQ(failed_pattern.reason, "allocation failed");

  const SweepNestedCompletion malformed_pattern =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, {{"status", "complete"}});
  EXPECT_EQ(malformed_pattern.status, SweepAttemptStatus::Partial);
  EXPECT_EQ(malformed_pattern.reason, "missing-pattern-completion-metadata");
}

TEST(SweepRunnerTest, CompleteSweepCheckpointsEveryRunAndValidatesConclusions) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::Standard, make_parameters(2), Json::object(),
                         make_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 2u);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0]["status"], "partial");
  EXPECT_EQ(checkpoints[0]["attempted_runs"], 1u);
  EXPECT_EQ(checkpoints[0]["completed_runs"], 1u);
  EXPECT_FALSE(checkpoints[0]["conclusions_valid"]);
  EXPECT_EQ(checkpoints[1]["status"], "complete");
  EXPECT_EQ(execution.output_json["planned_runs"], 2u);
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 2u);
  EXPECT_TRUE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "complete");
  EXPECT_EQ(announce_flags, (std::vector<bool>{false, true}));
}

TEST(SweepRunnerTest, TypedAndUnknownExecutorExceptionsRetainPriorEvidenceAndCheckpointOnce) {
  for (bool typed_exception : {true, false}) {
    SCOPED_TRACE(typed_exception);
    const size_t prior_runs = typed_exception ? 1 : 0;
    std::vector<SweepRunOutcome> outcomes;
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    SweepExecutionHooks hooks = make_hooks(outcomes, checkpoints, announce_flags, executed_runs);
    hooks.execute_run = [&](size_t run_index) -> SweepRunOutcome {
      ++executed_runs;
      if (run_index < prior_runs) return {EXIT_SUCCESS, make_standard_result("complete", true), ""};
      if (typed_exception) throw std::runtime_error("injected nested failure");
      throw 7;
    };
    SweepExecutionResult execution;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(execution = execute_sweep_plan(SweepNestedMode::Standard, make_parameters(typed_exception ? 3 : 1),
                                                   Json::object(), hooks));
    const std::string error = testing::internal::GetCapturedStderr();
    const std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
    EXPECT_EQ(executed_runs, prior_runs + 1);
    EXPECT_EQ(checkpoints.size(), executed_runs);
    EXPECT_EQ(announce_flags, std::vector<bool>(executed_runs, false));
    EXPECT_EQ(execution.output_json["status"], "failed");
    EXPECT_EQ(execution.output_json["status_reason"], "nested-run-execution-exception");
    EXPECT_EQ(execution.output_json["planned_runs"], typed_exception ? 3u : 1u);
    EXPECT_EQ(execution.output_json["attempted_runs"], executed_runs);
    EXPECT_EQ(execution.output_json["completed_runs"], prior_runs);
    EXPECT_FALSE(execution.output_json["conclusions_valid"].get<bool>());
    ASSERT_EQ(execution.output_json["runs"].size(), executed_runs);
    if (prior_runs != 0) EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
    const Json& failed = execution.output_json["runs"].back();
    EXPECT_EQ(failed["status"], "failed");
    EXPECT_EQ(failed["status_reason"], "nested-run-execution-exception");
    EXPECT_TRUE(failed["result"].is_null());
    EXPECT_TRUE(output.empty());
    EXPECT_EQ(error, Messages::error_prefix() +
                         Messages::error_sweep_nested_run_exception(typed_exception ? "injected nested failure" : "") +
                         "\n");
  }
}

TEST(SweepRunnerTest, PartialNestedRunStopsAndPreservesPriorCompleteRun) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_SUCCESS, make_standard_result("partial", false, "benchmark loops remain"), ""},
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::Standard, make_parameters(3), Json::object(),
                         make_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 2u);
  EXPECT_EQ(execution.output_json["status"], "partial");
  EXPECT_EQ(execution.output_json["status_reason"], "benchmark loops remain");
  EXPECT_EQ(execution.output_json["planned_runs"], 3u);
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  ASSERT_EQ(execution.output_json["runs"].size(), 2u);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "partial");
  EXPECT_EQ(checkpoints.size(), 2u);
}

TEST(SweepRunnerTest, InterruptedNestedRunIsAttemptedButNotCompleted) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_tlb_result("complete", true), ""},
      {EXIT_SUCCESS, make_tlb_result("interrupted", false), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::TlbAnalysis, make_parameters(2), Json::object(),
                         make_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(execution.output_json["status"], "interrupted");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "interrupted");
  EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], "nested-tlb-run-interrupted");
  EXPECT_FALSE(execution.output_json["runs"][1]["result"]["tlb_analysis"].contains("status_reason"));
}

TEST(SweepRunnerTest, FailureClassificationRetainsPayloadAndHonorsExplicitReason) {
  const Json diagnostic = {{"diagnostic", "runner failed after setup"}};
  struct FailureCase {
    const char* name;
    SweepNestedMode mode;
    Json payload;
    const char* override_reason;
    const char* expected_reason;
    bool prior_success;
  };
  for (const FailureCase& entry :
       {FailureCase{"override", SweepNestedMode::Standard, diagnostic, "simulated-execution-failure",
                    "simulated-execution-failure", true},
        FailureCase{"classifier", SweepNestedMode::Standard, diagnostic, "", "missing-standard-schema-version", false},
        FailureCase{"mode reason", SweepNestedMode::Standard,
                    make_standard_result("failed", false, "benchmark timer failed"), "", "benchmark timer failed",
                    false},
        FailureCase{"TLB error", SweepNestedMode::TlbAnalysis, make_tlb_result("error", false), "",
                    "nested-tlb-run-error", false},
        FailureCase{"empty", SweepNestedMode::Standard, Json::object(), "simulated-execution-failure",
                    "simulated-execution-failure", false}}) {
    SCOPED_TRACE(entry.name);
    std::vector<SweepRunOutcome> outcomes;
    if (entry.prior_success) outcomes.push_back({EXIT_SUCCESS, make_standard_result("complete", true), ""});
    outcomes.push_back({EXIT_FAILURE, entry.payload, entry.override_reason});
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    const SweepExecutionResult execution =
        execute_sweep_plan(entry.mode, make_parameters(outcomes.size()), Json::object(),
                           make_hooks(outcomes, checkpoints, announce_flags, executed_runs));
    ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
    EXPECT_EQ(execution.output_json["status"], "failed");
    EXPECT_EQ(execution.output_json["attempted_runs"], outcomes.size());
    EXPECT_EQ(execution.output_json["completed_runs"], entry.prior_success ? 1u : 0u);
    EXPECT_FALSE(execution.output_json["conclusions_valid"]);
    ASSERT_EQ(execution.output_json["runs"].size(), outcomes.size());
    const Json& failed = execution.output_json["runs"].back();
    EXPECT_EQ(failed["status"], "failed");
    EXPECT_EQ(failed["status_reason"], entry.expected_reason);
    EXPECT_EQ(failed["result"], entry.payload.empty() ? Json(nullptr) : entry.payload);
    if (entry.prior_success) EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
    EXPECT_EQ(checkpoints.size(), outcomes.size());
  }
}

TEST(SweepRunnerTest, CheckpointWriteFailureStopsFurtherRunsAndInvalidatesSweep) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
  };
  std::vector<Json> attempted_checkpoints;
  size_t executed_runs = 0;
  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    ++executed_runs;
    return outcomes.at(run_index);
  };
  hooks.stop_requested = []() { return false; };
  hooks.elapsed_seconds = []() { return 2.0; };
  hooks.utc_timestamp = []() { return "2026-01-01T00:00:00Z"; };
  hooks.write_checkpoint = [&](const Json& output, bool) {
    attempted_checkpoints.push_back(output);
    return attempted_checkpoints.size() == 2 ? EXIT_FAILURE : EXIT_SUCCESS;
  };

  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::Standard, make_parameters(3), Json::object(), hooks);

  ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
  EXPECT_EQ(executed_runs, 2u);
  ASSERT_EQ(attempted_checkpoints.size(), 2u);
  EXPECT_EQ(attempted_checkpoints[0]["attempted_runs"], 1u);
  EXPECT_EQ(attempted_checkpoints[0]["completed_runs"], 1u);
  EXPECT_EQ(execution.output_json["status"], "failed");
  EXPECT_EQ(execution.output_json["status_reason"], "checkpoint-write-failed");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 2u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"].size(), 2u);
}

TEST(SweepRunnerTest, InterruptionBeforeOrAfterFirstRunPreservesCompletedEvidence) {
  for (bool before_first : {true, false}) {
    SCOPED_TRACE(before_first);
    const std::vector<SweepRunOutcome> outcomes = {
        {EXIT_SUCCESS, make_standard_result("complete", true), ""},
        {EXIT_SUCCESS, make_standard_result("complete", true), ""},
    };
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    size_t stop_checks = 0;
    SweepExecutionHooks hooks = make_hooks(outcomes, checkpoints, announce_flags, executed_runs);
    hooks.stop_requested = [&]() { return ++stop_checks >= (before_first ? 1u : 2u); };
    const SweepExecutionResult execution =
        execute_sweep_plan(SweepNestedMode::Standard, make_parameters(2), Json::object(), hooks);
    ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(executed_runs, before_first ? 0u : 1u);
    EXPECT_EQ(execution.output_json["status"], "interrupted");
    EXPECT_EQ(execution.output_json["status_reason"],
              before_first ? "interruption-requested-before-run" : "interruption-requested-after-complete-run");
    EXPECT_EQ(execution.output_json["planned_runs"], 2u);
    EXPECT_EQ(execution.output_json["attempted_runs"], executed_runs);
    EXPECT_EQ(execution.output_json["completed_runs"], executed_runs);
    EXPECT_FALSE(execution.output_json["conclusions_valid"]);
    ASSERT_EQ(execution.output_json["runs"].size(), executed_runs);
    if (!before_first) EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
    ASSERT_EQ(checkpoints.size(), 1u);
    ASSERT_EQ(announce_flags.size(), 1u);
    EXPECT_TRUE(announce_flags[0]);
  }
}
