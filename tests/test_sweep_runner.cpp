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
  const Json complete = make_standard_result("complete", true);
  const Json partial = make_standard_result("partial", false, "benchmark loops remain");
  const Json partial_no_reason = make_standard_result("partial", false);
  const Json interrupted = make_standard_result("interrupted", false, "stop requested");
  const Json failed = make_standard_result("failed", false, "benchmark timer failed");
  auto with = [](Json doc, const auto& mutate) {
    mutate(doc);
    return doc;
  };

  struct CompletionCase {
    const char* name;
    Json result;
    SweepAttemptStatus expected_status;
    const char* expected_reason;
  };
  const std::vector<CompletionCase> cases = {
      {"schema 3 complete with nested empty output", complete, SweepAttemptStatus::Complete, ""},
      {"schema 3 partial with reason", partial, SweepAttemptStatus::Partial, "benchmark loops remain"},
      {"schema 3 partial without reason", partial_no_reason, SweepAttemptStatus::Partial,
       "nested-standard-result-incomplete"},
      {"schema 3 interrupted with reason", interrupted, SweepAttemptStatus::Interrupted, "stop requested"},
      {"schema 3 interrupted without reason", make_standard_result("interrupted", false),
       SweepAttemptStatus::Interrupted, "nested-run-interrupted"},
      {"schema 3 failed with reason", failed, SweepAttemptStatus::Failed, "benchmark timer failed"},
      {"schema 3 failed without reason", make_standard_result("failed", false), SweepAttemptStatus::Failed,
       "nested-run-failed"},
      {"schema 3 with results false", with(complete, [](Json& j) { j["results_complete"] = false; }),
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 with conclusions false", with(complete, [](Json& j) { j["conclusions_valid"] = false; }),
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 incomplete preserves reason",
       with(complete,
            [](Json& j) {
              j["results_complete"] = false;
              j["status_reason"] = "completion metadata disagrees";
            }),
       SweepAttemptStatus::Partial, "completion metadata disagrees"},
      {"schema 3 without status", with(complete, [](Json& j) { j.erase("status"); }), SweepAttemptStatus::Partial,
       "nested-standard-result-incomplete"},
      {"schema 3 with invalid status type", with(complete, [](Json& j) { j["status"] = true; }),
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 with invalid reason type", with(partial_no_reason, [](Json& j) { j["status_reason"] = true; }),
       SweepAttemptStatus::Partial, "nested-standard-result-incomplete"},
      {"schema 3 without conclusions", with(complete, [](Json& j) { j.erase("conclusions_valid"); }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid conclusions type", with(complete, [](Json& j) { j["conclusions_valid"] = 1; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 without output", with(complete, [](Json& j) { j["configuration"].erase("output_file"); }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid output type",
       with(complete, [](Json& j) { j["configuration"]["output_file"] = nullptr; }), SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 interrupted without output uses contract reason",
       with(interrupted, [](Json& j) { j["configuration"].erase("output_file"); }), SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 with invalid results type", with(complete, [](Json& j) { j["results_complete"] = 1; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 failed with invalid results uses contract reason",
       with(failed, [](Json& j) { j["results_complete"] = "false"; }), SweepAttemptStatus::Partial,
       "invalid-standard-schema-contract"},
      {"schema 3 without results", with(complete, [](Json& j) { j.erase("results_complete"); }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 without mode", with(complete, [](Json& j) { j["configuration"].erase("mode"); }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with invalid mode type", with(complete, [](Json& j) { j["configuration"]["mode"] = true; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"schema 3 with wrong mode",
       with(complete, [](Json& j) { j["configuration"]["mode"] = Constants::PATTERNS_JSON_MODE_NAME; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"missing configuration", with(complete, [](Json& j) { j.erase("configuration"); }), SweepAttemptStatus::Partial,
       "missing-standard-schema-version"},
      {"invalid configuration type", with(complete, [](Json& j) { j["configuration"] = Json::array(); }),
       SweepAttemptStatus::Partial, "missing-standard-schema-version"},
      {"missing schema identity", with(complete, [](Json& j) { j["configuration"].erase("benchmark_schema_version"); }),
       SweepAttemptStatus::Partial, "missing-standard-schema-version"},
      {"released schema 2", with(complete, [](Json& j) { j["configuration"]["benchmark_schema_version"] = 2; }),
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"invalid schema identity type",
       with(complete, [](Json& j) { j["configuration"]["benchmark_schema_version"] = "3"; }),
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"unsupported explicit schema 1",
       with(complete, [](Json& j) { j["configuration"]["benchmark_schema_version"] = 1; }),
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"unsupported explicit schema 4",
       with(complete, [](Json& j) { j["configuration"]["benchmark_schema_version"] = 4; }),
       SweepAttemptStatus::Partial, "unsupported-standard-schema-version"},
      {"top-level GPU identity", {{"mode", Constants::GPU_JSON_MODE_NAME}, {"schema_version", 1}},
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed top-level GPU identity",
       with(complete,
            [](Json& j) {
              j["mode"] = Constants::GPU_JSON_MODE_NAME;
              j["schema_version"] = 1;
            }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed pattern identity", with(complete, [](Json& j) { j["configuration"]["pattern_schema_version"] = 3; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed TLB or core-to-core identity", with(complete, [](Json& j) { j["configuration"]["schema_version"] = 4; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
      {"mixed sweep identity", with(complete, [](Json& j) { j["configuration"]["sweep_schema_version"] = 1; }),
       SweepAttemptStatus::Partial, "invalid-standard-schema-contract"},
  };

  for (const CompletionCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const SweepNestedCompletion completion =
        classify_sweep_nested_completion(SweepNestedMode::Standard, test_case.result);
    EXPECT_EQ(completion.status, test_case.expected_status);
    EXPECT_EQ(completion.reason, test_case.expected_reason);
  }
}

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
      {"standard-complete", SweepNestedMode::Standard, make_standard_result("complete", true),
       SweepAttemptStatus::Complete, ""},
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
      {"c2c-wrong-shape", SweepNestedMode::CoreToCore, make_standard_result("complete", true),
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
       SweepAttemptStatus::Partial, "nested-pattern-result-incomplete"},
      {"pattern-interrupted", SweepNestedMode::Patterns, make_pattern_result("interrupted", false, "stop requested"),
       SweepAttemptStatus::Interrupted, "stop requested"},
      {"pattern-failed", SweepNestedMode::Patterns, make_pattern_result("failed", false, "allocation failed"),
       SweepAttemptStatus::Failed, "allocation failed"},
      {"pattern-malformed", SweepNestedMode::Patterns, {{"status", "complete"}}, SweepAttemptStatus::Partial,
       "missing-pattern-completion-metadata"},
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

  {
    SCOPED_TRACE("checkpoint-write-failure");
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
      {"partial-standard",
       SweepNestedMode::Standard,
       {{EXIT_SUCCESS, make_standard_result("complete", true), ""},
        {EXIT_SUCCESS, make_standard_result("partial", false, "benchmark loops remain"), ""},
        {EXIT_SUCCESS, make_standard_result("complete", true), ""}},
       "partial",
       "benchmark loops remain",
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
  wrong_mode["mode"] = "benchmark";
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
