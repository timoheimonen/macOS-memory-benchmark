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
 * @file llm_sweep_runner.h
 * @brief Standalone LLM memory-profile parameter sweep runner.
 */

#ifndef LLM_SWEEP_RUNNER_H
#define LLM_SWEEP_RUNNER_H

#include <cstddef>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "third_party/nlohmann/json.hpp"

struct LlmMemoryConfig;
class JsonOutputSession;

/** Calculate the total Cartesian run count for an LLM sweep configuration. */
size_t calculate_llm_sweep_run_count(const LlmMemoryConfig& config);

/**
 * Expand a validated sweep base configuration into concrete per-run configs.
 *
 * Each generated config clears sweep fields and `output_file`, marks
 * `collect_json_for_sweep = true` so memory admission accounts for live JSON
 * serialization, and sets the swept field plus its `user_specified_*` flag.
 */
std::vector<LlmMemoryConfig> build_llm_sweep_run_configs(
    const LlmMemoryConfig& base_config);

/**
 * Execute one concrete LLM memory profile run and collect its schema-2 JSON.
 *
 * Resources and backend state are created and released within this call so
 * consecutive sweep points do not accumulate memory mappings or Metal buffers.
 *
 * @param config Concrete single-run configuration (`run_sweep == false`).
 * @param result_json Output schema-2 payload when a representable result is
 *        initialized; left empty on pre-initialization setup failures or
 *        pre-initialization interruption.
 * @param failure_reason Optional stable reason code populated when execution
 *        fails or is interrupted before result initialization.
 * @param preparation_interrupted Optional output set to `true` when stop is
 *        observed during paged table or backend resource preparation before
 *        measurement execution begins.
 * @return `EXIT_SUCCESS` on complete or gracefully interrupted execution;
 *         `EXIT_FAILURE` otherwise.
 */
int run_llm_memory_collect(const LlmMemoryConfig& config,
                           nlohmann::ordered_json& result_json,
                           std::string* failure_reason = nullptr,
                           bool* preparation_interrupted = nullptr);

/**
 * Execute an LLM parameter sweep through one command-owned output session.
 *
 * Every attempted run is checkpointed to file targets atomically; stdout
 * checkpoints are lazy no-ops so the command boundary emits one terminal
 * schema-1 sweep envelope to stdout.
 *
 * @param base_config Parsed and preflight-validated LLM sweep configuration.
 * @param output_session Session that must outlive this synchronous call.
 * @return Terminal process status and latest complete in-memory envelope.
 */
SweepExecutionResult run_llm_memory_sweep(
    const LlmMemoryConfig& base_config,
    JsonOutputSession& output_session);

#endif  // LLM_SWEEP_RUNNER_H
