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
 * @brief Backend-neutral configuration vocabulary and LLM run statuses
 */

#ifndef LLM_MEMORY_H
#define LLM_MEMORY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/config/constants.h"

/** Execution backend selected for an LLM memory profile. */
enum class LlmMemoryBackend : uint8_t {
  Cpu = 0,
  Metal,
};

/** Logical LLM workload phase represented by one work unit. */
enum class LlmPhase : uint8_t {
  Decode = 0,
  Prefill,
};

/** Logical-to-physical KV storage layout. */
enum class LlmKvLayout : uint8_t {
  Contiguous = 0,
  Paged,
};

/** Stable unit used for iteration counts, calibration, and rates. */
enum class LlmWorkUnitKind : uint8_t {
  DecodeStep = 0,
  PrefillOperation,
};

/** Logical KV write performed by one scenario work unit. */
enum class LlmKvWriteKind : uint8_t {
  None = 0,
  CurrentTokenAppend,
  FullPromptPopulation,
};

/** The three independently calibrated synthetic memory scenarios. */
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
  Unsupported,
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
inline constexpr const char* AVAILABLE_WORKER_COUNT_REQUIRED =
    "available-worker-count-required";
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
inline constexpr const char* JSON_INTEGER_OUT_OF_RANGE =
    "json-integer-out-of-range";
}  // namespace LlmMemoryConfigReason

/** Parsed and resolved standalone `--llm-memory` command configuration. */
struct LlmMemoryConfig {
  LlmMemoryBackend backend = LlmMemoryBackend::Cpu;
  LlmPhase phase = LlmPhase::Decode;
  LlmKvLayout kv_layout = LlmKvLayout::Contiguous;
  size_t weight_size_mb = 0;
  size_t layer_count = 0;
  size_t query_head_count = 0;
  size_t kv_head_count = 0;
  size_t head_dimension = 0;
  size_t kv_element_bytes = Constants::LLM_DEFAULT_KV_ELEMENT_BYTES;
  size_t visible_context_tokens = 0;
  size_t batch_size = Constants::LLM_DEFAULT_BATCH_SIZE;
  size_t requested_workers = 0;
  size_t available_workers = 0;
  size_t iterations = 0;
  size_t loop_count = Constants::LLM_DEFAULT_LOOP_COUNT;
  uint64_t seed = 0;
  bool user_specified_iterations = false;
  bool user_specified_seed = false;
  bool user_specified_workers = false;
  bool help_printed = false;
  std::string output_file;
  std::vector<std::string> argv;
};

/** Deterministic platform and entropy inputs copied for parser unit tests. */
struct LlmMemoryParserTestHooks {
  size_t detected_workers = 1;
  uint64_t generated_seed = 1;
};

/**
 * Install or clear deterministic LLM parser inputs.
 *
 * @param hooks Values copied immediately, or `nullptr` to restore production
 *        providers. The caller retains no lifetime obligation.
 * @warning This process-global test seam is not thread-safe and must not be
 *          changed concurrently with parsing.
 */
void set_llm_memory_parser_test_hooks(
    const LlmMemoryParserTestHooks* hooks);

/**
 * Parse and preflight the standalone `--llm-memory` option whitelist.
 *
 * Syntax, required inputs, checked geometry, worker availability, seed
 * resolution, and explicit per-scenario work limits are resolved before any
 * output session, mapping, or worker can be created. Human help returns before
 * platform detection and seed generation.
 *
 * @param argc Number of entries in @p argv.
 * @param argv Command arguments; entries are copied without retaining pointers.
 * @param config Destination reset before parsing and populated with the exact
 *        argv plus all successfully resolved fields, including on failure.
 * @return `EXIT_SUCCESS` for valid configuration or isolated help;
 *         `EXIT_FAILURE` after a centralized diagnostic otherwise.
 * @throws std::exception Allocation or caller-configured stream failures may
 *         propagate to the production command boundary, which contains them.
 * @note Parsing is single-threaded; installed test hooks are process-global.
 */
int parse_llm_memory_arguments(int argc, char* argv[],
                               LlmMemoryConfig& config);

/**
 * Own the standalone LLM command boundary.
 *
 * The boundary parses first, returns immediately for help, then installs the
 * command-scoped JSON transport before runtime output, QoS preparation, and
 * signal masking. It selects a command-owned backend, admits the exact
 * three-mapping memory plan, prepares and pre-touches the working set through
 * that backend, executes whole tasks through the backend-independent runner,
 * and emits console plus schema-v1 checkpoint or final-output evidence
 * according to the selected transport.
 *
 * @param argc Number of entries in @p argv.
 * @param argv Command arguments, valid for the duration of this call.
 * @return `EXIT_SUCCESS` for isolated help, complete execution, or graceful
 *         task-boundary interruption; `EXIT_FAILURE` for parse, planning,
 *         allocation, execution, validation, or output failure.
 * @post No exception escapes this boundary, and any installed output routing
 *       and signal mask are restored before return.
 */
int run_llm_memory_mode(int argc, char* argv[]);

/** Pure validation outcome used before any mapping or worker creation. */
struct LlmMemoryConfigValidation {
  bool valid = false;
  std::string reason_code = LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED;
  size_t active_weight_bytes = 0;
};

/** Validate resolved config fields and checked MiB-to-byte conversion. */
LlmMemoryConfigValidation validate_llm_memory_config(
    const LlmMemoryConfig& config);

/** Return the stable schema token for a scenario, or `unknown`. */
const char* llm_scenario_to_string(LlmScenario scenario);

/** Return the stable schema token for an execution backend. */
const char* llm_memory_backend_to_string(LlmMemoryBackend backend);

/** Return the stable schema token for an LLM phase. */
const char* llm_phase_to_string(LlmPhase phase);

/** Return the stable schema token for a KV layout. */
const char* llm_kv_layout_to_string(LlmKvLayout layout);

/** Return the work-unit kind implied by @p phase. */
LlmWorkUnitKind llm_work_unit_kind_for_phase(LlmPhase phase);

/** Return the stable schema token for a work-unit kind. */
const char* llm_work_unit_kind_to_string(LlmWorkUnitKind kind);

/** Return the KV write kind implied by @p phase and @p scenario. */
LlmKvWriteKind llm_kv_write_kind_for(LlmPhase phase,
                                     LlmScenario scenario);

/** Return the stable schema token for a KV write kind. */
const char* llm_kv_write_kind_to_string(LlmKvWriteKind kind);

/** Return the stable schema token for an attention classification. */
const char* llm_attention_kind_to_string(LlmAttentionKind kind);

/** Return the frozen measurement-status token. */
const char* llm_measurement_status_to_string(LlmMeasurementStatus status);

/** Return the frozen run-status token. */
const char* llm_run_status_to_string(LlmRunStatus status);

#endif  // LLM_MEMORY_H
