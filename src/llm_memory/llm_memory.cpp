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
 * @file llm_memory.cpp
 * @brief Pure configuration validation and status token conversion
 */

#include "llm_memory/llm_memory.h"

#include "utils/numeric_utils.h"

LlmMemoryConfigValidation validate_llm_memory_config(
    const LlmMemoryConfig& config) {
  LlmMemoryConfigValidation validation;
  if (config.weight_size_mb == 0) {
    validation.reason_code = LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED;
    return validation;
  }
  if (config.layer_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::LAYER_COUNT_REQUIRED;
    return validation;
  }
  if (config.query_head_count == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEAD_COUNT_REQUIRED;
    return validation;
  }
  if (config.kv_head_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::KV_HEAD_COUNT_REQUIRED;
    return validation;
  }
  if (config.head_dimension == 0) {
    validation.reason_code = LlmMemoryConfigReason::HEAD_DIMENSION_REQUIRED;
    return validation;
  }
  if (config.kv_element_bytes != 1 && config.kv_element_bytes != 2 &&
      config.kv_element_bytes != 4) {
    validation.reason_code =
        LlmMemoryConfigReason::INVALID_KV_ELEMENT_BYTES;
    return validation;
  }
  if (config.visible_context_tokens == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED;
    return validation;
  }
  if (config.batch_size == 0) {
    validation.reason_code = LlmMemoryConfigReason::BATCH_SIZE_REQUIRED;
    return validation;
  }
  if (config.requested_workers == 0) {
    validation.reason_code = LlmMemoryConfigReason::WORKER_COUNT_REQUIRED;
    return validation;
  }
  if (config.loop_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::LOOP_COUNT_REQUIRED;
    return validation;
  }
  if (config.query_head_count < config.kv_head_count) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEADS_BELOW_KV_HEADS;
    return validation;
  }
  if (config.query_head_count % config.kv_head_count != 0) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS;
    return validation;
  }
  if (config.user_specified_iterations && config.iterations == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::EXPLICIT_ITERATIONS_REQUIRED;
    return validation;
  }
  if (!config.user_specified_iterations && config.iterations != 0) {
    validation.reason_code =
        LlmMemoryConfigReason::AUTOMATIC_ITERATIONS_MUST_BE_ZERO;
    return validation;
  }
  if (!NumericUtils::checked_multiply(config.weight_size_mb,
                                      Constants::BYTES_PER_MB,
                                      validation.active_weight_bytes)) {
    validation.reason_code =
        LlmMemoryConfigReason::WEIGHT_SIZE_BYTES_OVERFLOW;
    return validation;
  }

  validation.valid = true;
  validation.reason_code = LlmMemoryConfigReason::VALID;
  return validation;
}

const char* llm_scenario_to_string(LlmScenario scenario) {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return "weights_only";
    case LlmScenario::KvOnly:
      return "kv_only";
    case LlmScenario::Mixed:
      return "mixed";
  }
  return "unknown";
}

const char* llm_attention_kind_to_string(LlmAttentionKind kind) {
  switch (kind) {
    case LlmAttentionKind::Mha:
      return "mha";
    case LlmAttentionKind::Gqa:
      return "gqa";
    case LlmAttentionKind::Mqa:
      return "mqa";
  }
  return "unknown";
}

const char* llm_measurement_status_to_string(LlmMeasurementStatus status) {
  switch (status) {
    case LlmMeasurementStatus::NotRun:
      return "not_run";
    case LlmMeasurementStatus::Measured:
      return "measured";
    case LlmMeasurementStatus::Interrupted:
      return "interrupted";
    case LlmMeasurementStatus::Invalid:
      return "invalid";
    case LlmMeasurementStatus::Failed:
      return "failed";
  }
  return "invalid";
}

const char* llm_run_status_to_string(LlmRunStatus status) {
  switch (status) {
    case LlmRunStatus::NotStarted:
      return "not_started";
    case LlmRunStatus::Complete:
      return "complete";
    case LlmRunStatus::Partial:
      return "partial";
    case LlmRunStatus::Interrupted:
      return "interrupted";
    case LlmRunStatus::Failed:
      return "failed";
  }
  return "failed";
}
