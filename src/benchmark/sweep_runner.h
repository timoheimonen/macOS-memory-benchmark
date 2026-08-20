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
/**
 * @file sweep_runner.h
 * @brief Multi-configuration benchmark sweep runner.
 */

#ifndef SWEEP_RUNNER_H
#define SWEEP_RUNNER_H

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "third_party/nlohmann/json.hpp"

struct BenchmarkConfig;
class JsonOutputSession;

/** Nested result schema used to decide whether one sweep run completed. */
enum class SweepNestedMode {
  Standard = 0,
  Patterns,
  TlbAnalysis,
  CoreToCore,
};

/** Auditable terminal state of one attempted nested sweep run. */
enum class SweepAttemptStatus {
  Complete = 0,
  Partial,
  Interrupted,
  Failed,
};

struct SweepNestedCompletion {
  SweepAttemptStatus status = SweepAttemptStatus::Partial;
  std::string reason;
};

/** Result returned by an injected or production nested-run executor. */
struct SweepRunOutcome {
  int exit_code = EXIT_FAILURE;
  nlohmann::ordered_json result_json;
  std::string failure_reason;
};

/** Side-effect seams used by the deterministic sweep coordinator. */
struct SweepExecutionHooks {
  std::function<SweepRunOutcome(size_t)> execute_run;
  std::function<bool()> stop_requested;
  std::function<double()> elapsed_seconds;
  std::function<std::string()> utc_timestamp;
  std::function<int(const nlohmann::ordered_json&, bool)> write_checkpoint;
};

struct SweepExecutionResult {
  int exit_code = EXIT_FAILURE;
  nlohmann::ordered_json output_json;
};

/** Determine nested completion from the mode-specific result schema. */
SweepNestedCompletion classify_sweep_nested_completion(SweepNestedMode mode, const nlohmann::ordered_json& result_json);

/**
 * Execute an already planned sweep through injected run/stop/write seams.
 *
 * Every attempted run is appended before its checkpoint. `completed_runs`
 * counts only nested results classified as complete for the selected mode.
 * Both sweep producers receive `configuration.sweep_schema_version` here
 * before the first checkpoint; a missing or non-object configuration member
 * is normalized to an object.
 *
 * @param mode Nested result schema used for completion classification.
 * @param run_parameters Immutable parameter object for every planned run.
 * @param initial_output Initial envelope metadata, moved into the result.
 * @param hooks Required executor/checkpoint seams and optional clock/stop seams.
 * @return Terminal status and the latest complete in-memory envelope.
 * @note Hooks are called synchronously on the invoking thread. Exceptions from
 *       `execute_run` are converted to a failed attempted run with a null
 *       result, then checkpointed normally. Other hook exceptions propagate.
 */
SweepExecutionResult execute_sweep_plan(SweepNestedMode mode, const std::vector<nlohmann::ordered_json>& run_parameters,
                                        nlohmann::ordered_json initial_output, const SweepExecutionHooks& hooks);

/**
 * Execute a parameter sweep through one command-owned JSON output session.
 *
 * The session must outlive this call and be installed before any benchmark
 * workers start. Every logical checkpoint is offered to the session; file
 * targets persist it atomically, while stdout checkpoints remain lazy no-ops.
 * The returned terminal envelope is not written a second time for file
 * targets. The command boundary is responsible for the one terminal stdout
 * write after this function has emitted its final human message.
 *
 * @param base_config Parsed and validated base configuration.
 * @param output_session Single-owner output session for this command.
 * @return Terminal process status and the latest in-memory sweep envelope.
 * @throws May propagate setup or nested-run exceptions. The command boundary
 *         must convert them to its return-code error path.
 * @note Called synchronously and not thread-safe. The session must not overlap
 *       another output session.
 */
SweepExecutionResult run_sweep_mode(const BenchmarkConfig& base_config,
                                    JsonOutputSession& output_session);

#endif  // SWEEP_RUNNER_H
