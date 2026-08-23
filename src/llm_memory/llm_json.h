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
 * @file llm_json.h
 * @brief Auditable backend-neutral JSON schema v1 for LLM memory profiles
 */

#ifndef LLM_JSON_H
#define LLM_JSON_H

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "core/system/benchmark_qos.h"
#include "llm_memory/llm_environment.h"
#include "llm_memory/llm_runner.h"
#include "third_party/nlohmann/json.hpp"

/** Stable machine-readable reasons for schema/transport peak estimation. */
namespace LlmJsonReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* INVALID_MODEL_WORK_PLAN = "invalid-model-work-plan";
inline constexpr const char* PEAK_BYTES_OVERFLOW = "json-output-peak-bytes-overflow";
}  // namespace LlmJsonReason

/** Conservative memory reserve for one live schema DOM and transport output. */
struct LlmJsonPeakEstimate {
  bool valid = false;
  std::string_view reason_code = LlmJsonReason::PEAK_BYTES_OVERFLOW;
  size_t fixed_schema_bytes = 0;
  size_t input_string_bytes = 0;
  size_t measurement_record_bytes = 0;
  size_t worker_checksum_bytes = 0;
  size_t total_bytes = 0;
};

/** Cold-path host and command metadata shared by every checkpoint snapshot. */
struct LlmResultMetadata {
  std::string timestamp;
  std::string processor_name;
  std::string macos_version;
  int performance_core_count = 0;
  int efficiency_core_count = 0;
  int logical_core_count = 0;
  size_t page_size_bytes = 0;
  size_t l1_data_cache_bytes = 0;
  size_t l2_data_cache_bytes = 0;
  size_t available_memory_bytes = 0;
  std::string available_memory_source = "mach-free-plus-inactive-rounded-mib";
  LlmJsonPeakEstimate json_peak_estimate;
  MainThreadQosResult main_thread_qos;
  LlmHostEnvironmentSnapshot environment_start;
  LlmHostEnvironmentSnapshot environment_end;
};

/**
 * Build one complete or partial top-level LLM schema-v1 document.
 *
 * Potentially large exact counts, payload bytes, seeds, and checksum integers
 * are emitted as canonical decimal strings. Work-unit counts and validated
 * control geometry use JSON integers. Phase-, layout-, and backend-specific
 * evidence uses object-or-null applicability, while an unobserved derived
 * metric is JSON null. The builder is synchronous and retains no references.
 *
 * @param config Parsed command configuration, including exact argv/output.
 * @param model_plan Immutable full-size model geometry and layout plan.
 * @param backend_evidence Backend snapshot supplying the selected tagged
 *        lifecycle, preparation, allocation, and task evidence.
 * @param metadata Command-scoped system and environment snapshots.
 * @param result Runner snapshot at a logical checkpoint or command terminal.
 * @return An independently owned ordered JSON document.
 * @throws std::exception on allocation or JSON construction failure. Transport
 *         boundaries contain such failures and apply checkpoint precedence.
 * @note The inputs must not be mutated concurrently during this call.
 */
nlohmann::ordered_json build_llm_memory_json(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                             const LlmBackendEvidence& backend_evidence,
                                             const LlmResultMetadata& metadata, const LlmMemoryResult& result);

/** CPU-only compatibility adapter retained for focused serializer tests. */
nlohmann::ordered_json build_llm_memory_json(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                             const LlmResourcePreparationResult& preparation,
                                             const LlmResultMetadata& metadata, const LlmMemoryResult& result);

/** Return the versioned exact-byte traffic classification token. */
const char* classify_llm_traffic_payload(const LlmGeometry& geometry) noexcept;

/**
 * Estimate a conservative schema-v1 DOM plus transport serialization peak.
 *
 * A disabled output target returns a valid zero estimate. File and stdout
 * targets reserve fixed schema/input storage, all variable-length component,
 * layout, scenario, and ownership identity evidence, every planned
 * measurement, both expected/actual CPU worker checksum trees, and each
 * bounded Metal measurement/calibration task-evidence tree before the final
 * work-plan memory admission. The estimate allocates no memory and retains no
 * references.
 *
 * @return A valid byte estimate, or a stable reason-bearing invalid estimate
 *         after invalid-plan or checked-arithmetic failure.
 */
LlmJsonPeakEstimate calculate_llm_json_peak_estimate(const LlmMemoryConfig& config,
                                                     const LlmMemoryWorkPlan& model_plan) noexcept;

/** Calculate the same conservative JSON peak from a pre-table view. */
LlmJsonPeakEstimate calculate_llm_json_peak_estimate(const LlmMemoryConfig& config,
                                                     const LlmAuxiliaryPreflightView& preflight) noexcept;

/**
 * Collect the canonical console/JSON quality-warning tokens.
 *
 * Runner-owned warnings retain their order. Environment, QoS, cache-working-
 * set, and duration warnings are then appended in the schema-v1 order with
 * first-occurrence deduplication. The helper performs no I/O.
 *
 * @param model_plan Immutable geometry used for cache-working-set checks.
 * @param metadata Command-scoped environment and main-thread QoS evidence.
 * @param result Runner warnings, worker QoS outcomes, and duration evidence.
 * @return An independently owned, ordered, deduplicated token list.
 * @throws std::bad_alloc if result storage cannot be allocated.
 * @note The inputs must not be mutated concurrently during this call.
 */
std::vector<std::string> collect_llm_quality_warning_tokens(const LlmMemoryWorkPlan& model_plan,
                                                            const LlmResultMetadata& metadata,
                                                            const LlmMemoryResult& result);

#endif  // LLM_JSON_H
