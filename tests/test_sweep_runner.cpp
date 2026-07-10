// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "pattern_benchmark/pattern_benchmark.h"

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
  return {{"status", status}, {"status_reason", reason}, {"results_complete", results_complete}};
}

Json make_tlb_result(const std::string& status, bool conclusions_valid, const std::string& reason = "") {
  return {{"tlb_analysis", {{"status", status}, {"status_reason", reason}, {"conclusions_valid", conclusions_valid}}}};
}

Json make_complete_pattern_result(size_t loop_count) {
  Json patterns = Json::object();
  for (size_t pattern_index = 0; pattern_index < static_cast<size_t>(PatternKind::Count); ++pattern_index) {
    Json bandwidth = Json::object();
    for (size_t operation_index = 0; operation_index < static_cast<size_t>(PatternOperation::Count);
         ++operation_index) {
      Json measurements = Json::array();
      for (size_t loop_index = 0; loop_index < loop_count; ++loop_index) {
        measurements.push_back({{"status", "measured"}, {"reason", ""}});
      }
      bandwidth["operation_" + std::to_string(operation_index)] = {{"measurements", std::move(measurements)}};
    }
    patterns["pattern_" + std::to_string(pattern_index)] = {{"bandwidth", std::move(bandwidth)}};
  }
  return {{"configuration", {{"loop_count", loop_count}}}, {"patterns", std::move(patterns)}};
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
  const SweepNestedCompletion standard =
      classify_sweep_nested_completion(SweepNestedMode::Standard, make_standard_result("complete", true));
  EXPECT_EQ(standard.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("complete", true));
  EXPECT_EQ(tlb.status, SweepAttemptStatus::Complete);

  Json patterns = make_complete_pattern_result(2);
  patterns["patterns"]["pattern_5"]["bandwidth"]["operation_2"]["measurements"][0]["status"] = "skipped";
  const SweepNestedCompletion pattern = classify_sweep_nested_completion(SweepNestedMode::Patterns, patterns);
  EXPECT_EQ(pattern.status, SweepAttemptStatus::Complete);

  const SweepNestedCompletion unvalidated_tlb =
      classify_sweep_nested_completion(SweepNestedMode::TlbAnalysis, make_tlb_result("complete", false));
  EXPECT_EQ(unvalidated_tlb.status, SweepAttemptStatus::Partial);

  patterns["patterns"].erase("pattern_6");
  const SweepNestedCompletion incomplete_pattern =
      classify_sweep_nested_completion(SweepNestedMode::Patterns, patterns);
  EXPECT_EQ(incomplete_pattern.status, SweepAttemptStatus::Partial);
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
      {EXIT_SUCCESS, make_tlb_result("interrupted", false, "stop requested"), ""},
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
  EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], "stop requested");
}

TEST(SweepRunnerTest, ExecutionFailureIsRecordedAndPriorResultSurvives) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_FAILURE, {{"diagnostic", "runner failed after setup"}}, "simulated-execution-failure"},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::Standard, make_parameters(2), Json::object(),
                         make_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
  EXPECT_EQ(execution.output_json["status"], "failed");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  ASSERT_EQ(execution.output_json["runs"].size(), 2u);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "failed");
  EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], "simulated-execution-failure");
  EXPECT_EQ(execution.output_json["runs"][1]["result"]["diagnostic"], "runner failed after setup");
  EXPECT_EQ(checkpoints.size(), 2u);
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

TEST(SweepRunnerTest, InterruptionAfterCompleteRunKeepsRunCompleted) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
      {EXIT_SUCCESS, make_standard_result("complete", true), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  size_t stop_checks = 0;
  SweepExecutionHooks hooks = make_hooks(outcomes, checkpoints, announce_flags, executed_runs);
  hooks.stop_requested = [&]() {
    ++stop_checks;
    return stop_checks >= 2;
  };

  const SweepExecutionResult execution =
      execute_sweep_plan(SweepNestedMode::Standard, make_parameters(2), Json::object(), hooks);

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 1u);
  EXPECT_EQ(execution.output_json["status"], "interrupted");
  EXPECT_EQ(execution.output_json["status_reason"], "interruption-requested-after-complete-run");
  EXPECT_EQ(execution.output_json["planned_runs"], 2u);
  EXPECT_EQ(execution.output_json["attempted_runs"], 1u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
  ASSERT_EQ(checkpoints.size(), 1u);
  ASSERT_EQ(announce_flags.size(), 1u);
  EXPECT_TRUE(announce_flags[0]);
}
