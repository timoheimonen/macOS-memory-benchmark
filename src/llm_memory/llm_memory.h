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

/**
 * @file llm_memory.h
 * @brief Cold-path configuration and status model for the CPU LLM memory profile
 */

#ifndef LLM_MEMORY_H
#define LLM_MEMORY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/config/constants.h"

/** The three independently calibrated synthetic decode-memory scenarios. */
enum class LlmScenario : uint8_t {
  WeightsOnly = 0,
  KvOnly,
  Mixed,
};

constexpr size_t kLlmScenarioCount = 3;

/** Metadata classification derived from query-head and KV-head geometry. */
enum class LlmAttentionKind : uint8_t {
  Mha = 0,
  Gqa,
  Mqa,
};

/** Terminal state of one scenario measurement. */
enum class LlmMeasurementStatus : uint8_t {
  NotRun = 0,
  Measured,
  Interrupted,
  Invalid,
  Failed,
};

/** Command-level lifecycle state for the standalone LLM memory profile. */
enum class LlmRunStatus : uint8_t {
  NotStarted = 0,
  Complete,
  Partial,
  Interrupted,
  Failed,
};

/** Stable machine-readable config validation reasons. */
namespace LlmMemoryConfigReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* WEIGHT_SIZE_REQUIRED = "weight-size-required";
inline constexpr const char* LAYER_COUNT_REQUIRED = "layer-count-required";
inline constexpr const char* QUERY_HEAD_COUNT_REQUIRED =
    "query-head-count-required";
inline constexpr const char* KV_HEAD_COUNT_REQUIRED =
    "kv-head-count-required";
inline constexpr const char* HEAD_DIMENSION_REQUIRED =
    "head-dimension-required";
inline constexpr const char* INVALID_KV_ELEMENT_BYTES =
    "invalid-kv-element-bytes";
inline constexpr const char* CONTEXT_TOKENS_REQUIRED =
    "context-tokens-required";
inline constexpr const char* BATCH_SIZE_REQUIRED = "batch-size-required";
inline constexpr const char* WORKER_COUNT_REQUIRED = "worker-count-required";
inline constexpr const char* LOOP_COUNT_REQUIRED = "loop-count-required";
inline constexpr const char* QUERY_HEADS_BELOW_KV_HEADS =
    "query-heads-below-kv-heads";
inline constexpr const char* QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS =
    "query-heads-not-divisible-by-kv-heads";
inline constexpr const char* EXPLICIT_ITERATIONS_REQUIRED =
    "explicit-iterations-required";
inline constexpr const char* AUTOMATIC_ITERATIONS_MUST_BE_ZERO =
    "automatic-iterations-must-be-zero";
inline constexpr const char* WEIGHT_SIZE_BYTES_OVERFLOW =
    "weight-size-bytes-overflow";
}  // namespace LlmMemoryConfigReason

/** Parsed and resolved standalone `--llm-memory` command configuration. */
struct LlmMemoryConfig {
  size_t weight_size_mb = 0;
  size_t layer_count = 0;
  size_t query_head_count = 0;
  size_t kv_head_count = 0;
  size_t head_dimension = 0;
  size_t kv_element_bytes = Constants::LLM_DEFAULT_KV_ELEMENT_BYTES;
  size_t visible_context_tokens = 0;
  size_t batch_size = Constants::LLM_DEFAULT_BATCH_SIZE;
  size_t requested_workers = 0;
  size_t iterations = 0;
  size_t loop_count = Constants::LLM_DEFAULT_LOOP_COUNT;
  uint64_t seed = 0;
  bool user_specified_iterations = false;
  bool user_specified_seed = false;
  bool help_printed = false;
  std::string output_file;
  std::vector<std::string> argv;
};

/** Pure validation outcome used before any mapping or worker creation. */
struct LlmMemoryConfigValidation {
  bool valid = false;
  std::string reason_code = LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED;
  size_t active_weight_bytes = 0;
};

/** Foundational status-bearing record for one future runner measurement. */
struct LlmMeasurementState {
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmMeasurementStatus status = LlmMeasurementStatus::NotRun;
  std::string reason_code = "not-run";
  std::string diagnostic;
  size_t loop_index = 0;
  size_t order_position = 0;
  size_t planned_steps = 0;
  size_t completed_steps = 0;
  size_t planned_exact_payload_bytes = 0;
  size_t completed_exact_payload_bytes = 0;
  std::optional<double> elapsed_seconds;
  std::optional<double> effective_payload_gb_s;
  bool checksum_valid = false;
};

/** Exact lifecycle counters retained by partial and terminal run results. */
struct LlmRunCounters {
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t measured_measurements = 0;
  size_t planned_synthetic_steps = 0;
  size_t completed_synthetic_steps = 0;
  size_t planned_exact_payload_bytes = 0;
  size_t completed_exact_payload_bytes = 0;
};

/** Cold-path command result foundation; orchestration fields are added later. */
struct LlmMemoryResult {
  LlmRunStatus status = LlmRunStatus::NotStarted;
  std::string reason_code = "not-started";
  std::string diagnostic;
  bool interruption_requested = false;
  bool results_complete = false;
  bool conclusions_valid = false;
  bool scenario_order_balance_complete = false;
  LlmRunCounters counters;
  std::vector<LlmMeasurementState> measurements;
};

/** Validate resolved config fields and checked MiB-to-byte conversion. */
LlmMemoryConfigValidation validate_llm_memory_config(
    const LlmMemoryConfig& config);

/** Return the stable schema token for a scenario, or `unknown`. */
const char* llm_scenario_to_string(LlmScenario scenario);

/** Return the stable schema token for an attention classification. */
const char* llm_attention_kind_to_string(LlmAttentionKind kind);

/** Return the frozen measurement-status token. */
const char* llm_measurement_status_to_string(LlmMeasurementStatus status);

/** Return the frozen run-status token. */
const char* llm_run_status_to_string(LlmRunStatus status);

#endif  // LLM_MEMORY_H
