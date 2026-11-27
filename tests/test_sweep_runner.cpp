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

Json make_tlb_result(const std::string& status, bool conclusions_valid) {
  return {{"tlb_analysis", {{"status", status}, {"conclusions_valid", conclusions_valid}}}};
}

Json make_cpu_mode_result(const std::string& mode, const std::string& status,
                          bool results_complete, bool run_accepted,
                          const std::string& reason = "") {
  return {
      {"schema_version", 1},
      {"mode", mode},
      {"status", status},
      {"reason_code", reason},
      {"results_complete", results_complete},
      {"run_accepted", run_accepted},
      {"measurements", Json::array({{{"measurement_id", 0}, {"status", "measured"}}})},
  };
}

Json make_pattern_result(const std::string& status, bool results_complete,
                         const std::string& reason = "",
                         bool run_accepted = false) {
  const bool accepted = run_accepted || (status == "complete" && results_complete);
  return make_cpu_mode_result(Constants::PATTERNS_JSON_MODE_NAME, status,
                              results_complete, accepted, reason);
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

TEST(SweepRunnerTest, NestedCompletionIsModeAware) {
  struct ModeCase {
    const char* name;
    SweepNestedMode mode;
    Json payload;
    SweepAttemptStatus expected_status;
    const char* expected_reason;
  };
  const Json error_tlb_result = make_tlb_result("error", false);
  EXPECT_FALSE(error_tlb_result["tlb_analysis"].contains("status_reason"));
  const ModeCase cases[] = {
      {"cpu-bandwidth-complete", SweepNestedMode::CpuBandwidth,
       make_cpu_mode_result("cpu_bandwidth", "complete", true, true, "complete"),
       SweepAttemptStatus::Complete, ""},
      {"cpu-latency-complete", SweepNestedMode::CpuLatency,
       make_cpu_mode_result("cpu_latency", "complete", true, true, "complete"),
       SweepAttemptStatus::Complete, ""},
      {"cpu-bandwidth-failed", SweepNestedMode::CpuBandwidth,
       make_cpu_mode_result("cpu_bandwidth", "failed", false, false, "checksum-mismatch"),
       SweepAttemptStatus::Failed, "checksum-mismatch"},
      {"cpu-latency-interrupted", SweepNestedMode::CpuLatency,
       make_cpu_mode_result("cpu_latency", "interrupted", false, false, "interruption-requested"),
       SweepAttemptStatus::Interrupted, "interruption-requested"},
      {"tlb-complete", SweepNestedMode::TlbAnalysis, make_tlb_result("complete", true),
       SweepAttemptStatus::Complete, ""},
      {"pattern-complete", SweepNestedMode::Patterns, make_pattern_result("complete", true),
       SweepAttemptStatus::Complete, ""},
      {"c2c-complete", SweepNestedMode::CoreToCore, make_core_to_core_result("complete", true),
       SweepAttemptStatus::Complete, ""},
      {"c2c-incomplete", SweepNestedMode::CoreToCore, make_core_to_core_result("complete", false),
       SweepAttemptStatus::Partial, "nested-core-to-core-result-incomplete"},
      {"c2c-interrupted", SweepNestedMode::CoreToCore, make_core_to_core_result("interrupted", false),
       SweepAttemptStatus::Interrupted, "nested-core-to-core-run-interrupted"},
      {"c2c-wrong-shape", SweepNestedMode::CoreToCore, make_pattern_result("complete", true),
       SweepAttemptStatus::Partial, "missing-core-to-core-result"},
      {"tlb-unvalidated", SweepNestedMode::TlbAnalysis, make_tlb_result("complete", false),
       SweepAttemptStatus::Partial, "nested-tlb-result-incomplete"},
      {"tlb-partial", SweepNestedMode::TlbAnalysis, make_tlb_result("partial", false),
       SweepAttemptStatus::Partial, "nested-tlb-result-incomplete"},
      {"tlb-interrupted", SweepNestedMode::TlbAnalysis, make_tlb_result("interrupted", false),
       SweepAttemptStatus::Interrupted, "nested-tlb-run-interrupted"},
      {"tlb-error", SweepNestedMode::TlbAnalysis, error_tlb_result, SweepAttemptStatus::Failed, "nested-tlb-run-error"},
      {"pattern-partial", SweepNestedMode::Patterns, make_pattern_result("partial", false, "pattern loops remain"),
       SweepAttemptStatus::Partial, "pattern loops remain"},
      {"pattern-inconsistent-complete", SweepNestedMode::Patterns, make_pattern_result("complete", false),
       SweepAttemptStatus::Partial, "nested-patterns-result-incomplete"},
      {"pattern-interrupted", SweepNestedMode::Patterns, make_pattern_result("interrupted", false, "stop requested"),
       SweepAttemptStatus::Interrupted, "stop requested"},
      {"pattern-failed", SweepNestedMode::Patterns, make_pattern_result("failed", false, "allocation failed"),
       SweepAttemptStatus::Failed, "allocation failed"},
      {"pattern-malformed", SweepNestedMode::Patterns, {{"status", "complete"}}, SweepAttemptStatus::Partial,
       "missing-patterns-schema-version"},
  };

  for (const ModeCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const SweepNestedCompletion completion = classify_sweep_nested_completion(entry.mode, entry.payload);
    EXPECT_EQ(completion.status, entry.expected_status);
    EXPECT_EQ(completion.reason, entry.expected_reason);
  }
}

TEST(SweepRunnerTest, SweepCheckpointingValidatesCompleteAndStopsOnWriteFailure) {
  {
    SCOPED_TRACE("complete-checkpoints");
    const std::vector<SweepRunOutcome> outcomes = {
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
    };
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    const SweepExecutionResult execution =
        execute_sweep_plan(SweepNestedMode::Patterns, make_parameters(2), Json::object(),
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

  {
    SCOPED_TRACE("checkpoint-write-failure");
    const std::vector<SweepRunOutcome> outcomes = {
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
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
        execute_sweep_plan(SweepNestedMode::Patterns, make_parameters(3), Json::object(), hooks);

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
      if (run_index < prior_runs) return {EXIT_SUCCESS, make_pattern_result("complete", true), ""};
      if (typed_exception) throw std::runtime_error("injected nested failure");
      throw 7;
    };
    SweepExecutionResult execution;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(execution = execute_sweep_plan(SweepNestedMode::Patterns, make_parameters(typed_exception ? 3 : 1),
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

TEST(SweepRunnerTest, PartialAndInterruptedNestedRunsStopAndPreservePriorEvidence) {
  struct IncompleteRunCase {
    const char* name;
    SweepNestedMode mode;
    std::vector<SweepRunOutcome> outcomes;
    const char* expected_status;
    const char* expected_reason;
    bool check_tlb_no_reason_field;
  };
  const IncompleteRunCase cases[] = {
      {"partial-patterns",
       SweepNestedMode::Patterns,
       {{EXIT_SUCCESS, make_pattern_result("complete", true), ""},
        {EXIT_SUCCESS, make_pattern_result("partial", false, "pattern loops remain"), ""},
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""}},
       "partial",
       "pattern loops remain",
       false},
      {"interrupted-tlb",
       SweepNestedMode::TlbAnalysis,
       {{EXIT_SUCCESS, make_tlb_result("complete", true), ""},
        {EXIT_SUCCESS, make_tlb_result("interrupted", false), ""}},
       "interrupted",
       "nested-tlb-run-interrupted",
       true},
  };

  for (const IncompleteRunCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    const SweepExecutionResult execution =
        execute_sweep_plan(entry.mode, make_parameters(entry.outcomes.size()), Json::object(),
                           make_hooks(entry.outcomes, checkpoints, announce_flags, executed_runs));

    ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
    EXPECT_EQ(executed_runs, 2u);
    EXPECT_EQ(execution.output_json["status"], entry.expected_status);
    EXPECT_EQ(execution.output_json["status_reason"], entry.expected_reason);
    EXPECT_EQ(execution.output_json["planned_runs"], entry.outcomes.size());
    EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
    EXPECT_EQ(execution.output_json["completed_runs"], 1u);
    EXPECT_FALSE(execution.output_json["conclusions_valid"]);
    ASSERT_EQ(execution.output_json["runs"].size(), 2u);
    EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
    EXPECT_EQ(execution.output_json["runs"][1]["status"], entry.expected_status);
    EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], entry.expected_reason);
    EXPECT_EQ(checkpoints.size(), 2u);
    if (entry.check_tlb_no_reason_field) {
      EXPECT_FALSE(execution.output_json["runs"][1]["result"]["tlb_analysis"].contains("status_reason"));
    }
  }
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
       {FailureCase{"override", SweepNestedMode::Patterns, diagnostic, "simulated-execution-failure",
                    "simulated-execution-failure", true},
        FailureCase{"classifier", SweepNestedMode::Patterns, diagnostic, "", "missing-patterns-schema-version", false},
        FailureCase{"mode reason", SweepNestedMode::Patterns,
                    make_pattern_result("failed", false, "pattern allocation failed"), "", "pattern allocation failed",
                    false},
        FailureCase{"TLB error", SweepNestedMode::TlbAnalysis, make_tlb_result("error", false), "",
                    "nested-tlb-run-error", false},
        FailureCase{"empty", SweepNestedMode::Patterns, Json::object(), "simulated-execution-failure",
                    "simulated-execution-failure", false}}) {
    SCOPED_TRACE(entry.name);
    std::vector<SweepRunOutcome> outcomes;
    if (entry.prior_success) outcomes.push_back({EXIT_SUCCESS, make_pattern_result("complete", true), ""});
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

TEST(SweepRunnerTest, InterruptionBeforeOrAfterFirstRunPreservesCompletedEvidence) {
  for (bool before_first : {true, false}) {
    SCOPED_TRACE(before_first);
    const std::vector<SweepRunOutcome> outcomes = {
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
        {EXIT_SUCCESS, make_pattern_result("complete", true), ""},
    };
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    size_t stop_checks = 0;
    SweepExecutionHooks hooks = make_hooks(outcomes, checkpoints, announce_flags, executed_runs);
    hooks.stop_requested = [&]() { return ++stop_checks >= (before_first ? 1u : 2u); };
    const SweepExecutionResult execution =
        execute_sweep_plan(SweepNestedMode::Patterns, make_parameters(2), Json::object(), hooks);
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

TEST(SweepRunnerTest, LlmMemoryNestedCompletionAndPreInitOutcomesPreserveSchemaAndReason) {
  auto make_llm_result = [](const std::string& status, bool results_complete, bool run_accepted,
                            const std::string& reason = "") {
    return Json{
        {"schema_version", Constants::LLM_JSON_SCHEMA_VERSION},
        {"mode", Constants::LLM_JSON_MODE_NAME},
        {"status", status},
        {"reason_code", reason},
        {"results_complete", results_complete},
        {"run_accepted", run_accepted},
        {"measurements", Json::array({{{"measurement_id", 0}, {"status", "measured"}}})},
    };
  };

  Json complete = make_llm_result("complete", true, true, "complete");
  Json interrupted = make_llm_result("interrupted", false, false, "interruption-requested");
  Json interrupted_no_reason = make_llm_result("interrupted", false, false);
  Json unsupported = make_llm_result("unsupported", false, false, "metal-device-unavailable");
  Json unsupported_no_reason = make_llm_result("unsupported", false, false);
  Json failed = make_llm_result("failed", false, false, "checksum-mismatch");
  Json failed_no_reason = make_llm_result("failed", false, false);
  Json partial = make_llm_result("partial", false, false, "partial-results");
  Json accepted_false = make_llm_result("complete", true, false, "checkpoint-write-failed");
  Json empty_measurements = complete;
  empty_measurements["measurements"] = Json::array();
  Json unmeasured_row = complete;
  unmeasured_row["measurements"][0]["status"] = "invalid";
  Json wrong_schema = complete;
  wrong_schema["schema_version"] = 1;
  Json bool_schema = complete;
  bool_schema["schema_version"] = true;
  Json wrong_mode = complete;
  wrong_mode["mode"] = "patterns";
  Json missing_schema = complete;
  missing_schema.erase("schema_version");

  struct Case {
    const char* name;
    Json payload;
    SweepAttemptStatus expected_status;
    const char* expected_reason;
  };
  const std::vector<Case> cases = {
      {"complete schema 2", complete, SweepAttemptStatus::Complete, ""},
      {"interrupted with reason", interrupted, SweepAttemptStatus::Interrupted, "interruption-requested"},
      {"interrupted without reason", interrupted_no_reason, SweepAttemptStatus::Interrupted,
       "nested-llm-run-interrupted"},
      {"unsupported with reason", unsupported, SweepAttemptStatus::Failed, "metal-device-unavailable"},
      {"unsupported without reason", unsupported_no_reason, SweepAttemptStatus::Failed, "nested-llm-run-unsupported"},
      {"failed with reason", failed, SweepAttemptStatus::Failed, "checksum-mismatch"},
      {"failed without reason", failed_no_reason, SweepAttemptStatus::Failed, "nested-llm-run-failed"},
      {"partial with reason", partial, SweepAttemptStatus::Partial, "partial-results"},
      {"run_accepted false", accepted_false, SweepAttemptStatus::Partial, "checkpoint-write-failed"},
      {"empty measurements", empty_measurements, SweepAttemptStatus::Partial, "invalid-llm-schema-contract"},
      {"unmeasured row", unmeasured_row, SweepAttemptStatus::Partial, "invalid-llm-schema-contract"},
      {"wrong schema", wrong_schema, SweepAttemptStatus::Partial, "unsupported-llm-schema-version"},
      {"bool schema", bool_schema, SweepAttemptStatus::Partial, "unsupported-llm-schema-version"},
      {"wrong mode", wrong_mode, SweepAttemptStatus::Partial, "invalid-llm-schema-contract"},
      {"missing schema", missing_schema, SweepAttemptStatus::Partial, "missing-llm-schema-version"},
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const SweepNestedCompletion completion =
        classify_sweep_nested_completion(SweepNestedMode::LlmMemory, test_case.payload);
    EXPECT_EQ(completion.status, test_case.expected_status);
    EXPECT_EQ(completion.reason, test_case.expected_reason);
  }

  struct LlmSecondRunCase {
    const char* name;
    SweepRunOutcome second_outcome;
    int expected_exit_code;
    const char* expected_status;
    const char* expected_reason;
    bool expect_null_result;
  };
  const LlmSecondRunCase sweep_cases[] = {
      {"unsupported-nested", {EXIT_FAILURE, unsupported, ""}, EXIT_FAILURE, "failed", "metal-device-unavailable",
       false},
      {"preinit-budget-exceeded", {EXIT_FAILURE, Json::object(), "memory-budget-exceeded", false}, EXIT_FAILURE,
       "failed", "memory-budget-exceeded", true},
      {"preinit-interrupted-success", {EXIT_SUCCESS, Json::object(), "preparation-interrupted", true}, EXIT_SUCCESS,
       "interrupted", "preparation-interrupted", true},
      {"preinit-interrupted-failure", {EXIT_FAILURE, Json::object(), "preparation-interrupted", true}, EXIT_FAILURE,
       "failed", "preparation-interrupted", true},
  };

  for (const LlmSecondRunCase& entry : sweep_cases) {
    SCOPED_TRACE(entry.name);
    std::vector<Json> checkpoints;
    std::vector<bool> announce_flags;
    size_t executed_runs = 0;
    const std::vector<SweepRunOutcome> outcomes = {{EXIT_SUCCESS, complete, "", false}, entry.second_outcome};
    const SweepExecutionResult execution =
        execute_sweep_plan(SweepNestedMode::LlmMemory, make_parameters(2), Json::object(),
                           make_hooks(outcomes, checkpoints, announce_flags, executed_runs));
    EXPECT_EQ(execution.exit_code, entry.expected_exit_code);
    EXPECT_EQ(execution.output_json["status"], entry.expected_status);
    EXPECT_EQ(execution.output_json["status_reason"], entry.expected_reason);
    EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
    EXPECT_EQ(execution.output_json["completed_runs"], 1u);
    EXPECT_FALSE(execution.output_json["conclusions_valid"].get<bool>());
    EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
    EXPECT_EQ(execution.output_json["runs"][1]["status"], entry.expected_status);
    EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], entry.expected_reason);
    EXPECT_EQ(execution.output_json["runs"][1]["result"].is_null(), entry.expect_null_result);
  }
}
