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

#include <gtest/gtest.h>

#include <limits>
#include <string>

#include "llm_memory/llm_memory.h"

namespace {

LlmMemoryConfig valid_config() {
  LlmMemoryConfig config;
  config.weight_size_mb = 4096;
  config.layer_count = 32;
  config.query_head_count = 32;
  config.kv_head_count = 8;
  config.head_dimension = 128;
  config.visible_context_tokens = 8192;
  config.requested_workers = 8;
  return config;
}

void expect_invalid(const LlmMemoryConfig& config,
                    const std::string& expected_reason) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.reason_code, expected_reason);
  EXPECT_EQ(validation.active_weight_bytes, 0u);
}

}  // namespace

TEST(LlmMemoryConfigTest, DefaultsMatchFrozenStandaloneContract) {
  const LlmMemoryConfig config;
  EXPECT_EQ(config.weight_size_mb, 0u);
  EXPECT_EQ(config.layer_count, 0u);
  EXPECT_EQ(config.query_head_count, 0u);
  EXPECT_EQ(config.kv_head_count, 0u);
  EXPECT_EQ(config.head_dimension, 0u);
  EXPECT_EQ(config.kv_element_bytes, 2u);
  EXPECT_EQ(config.visible_context_tokens, 0u);
  EXPECT_EQ(config.batch_size, 1u);
  EXPECT_EQ(config.requested_workers, 0u);
  EXPECT_EQ(config.iterations, 0u);
  EXPECT_EQ(config.loop_count, 3u);
  EXPECT_EQ(config.seed, 0u);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_FALSE(config.help_printed);
  EXPECT_TRUE(config.output_file.empty());
  EXPECT_TRUE(config.argv.empty());

  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.reason_code,
            LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED);
}

TEST(LlmMemoryConfigTest, StableScenarioAttentionAndStatusTokens) {
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::WeightsOnly),
               "weights_only");
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::KvOnly), "kv_only");
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::Mixed), "mixed");
  EXPECT_STREQ(llm_scenario_to_string(static_cast<LlmScenario>(99)),
               "unknown");

  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Mha), "mha");
  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Gqa), "gqa");
  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Mqa), "mqa");
  EXPECT_STREQ(
      llm_attention_kind_to_string(static_cast<LlmAttentionKind>(99)),
      "unknown");

  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::NotRun),
      "not_run");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Measured),
      "measured");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Interrupted),
      "interrupted");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Invalid),
      "invalid");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Failed),
      "failed");
  EXPECT_STREQ(llm_measurement_status_to_string(
                   static_cast<LlmMeasurementStatus>(99)),
               "invalid");

  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::NotStarted),
               "not_started");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Complete), "complete");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Partial), "partial");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Interrupted),
               "interrupted");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Failed), "failed");
  EXPECT_STREQ(llm_run_status_to_string(static_cast<LlmRunStatus>(99)),
               "failed");
}

TEST(LlmMemoryConfigTest, ValidatesPositiveGeometryAndHeadSharing) {
  LlmMemoryConfig config = valid_config();
  LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid) << validation.reason_code;
  EXPECT_EQ(validation.reason_code, LlmMemoryConfigReason::VALID);
  EXPECT_EQ(validation.active_weight_bytes, 4ULL * 1024ULL * 1024ULL * 1024ULL);

  config = valid_config();
  config.weight_size_mb = 0;
  expect_invalid(config, LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED);
  config = valid_config();
  config.layer_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::LAYER_COUNT_REQUIRED);
  config = valid_config();
  config.query_head_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::QUERY_HEAD_COUNT_REQUIRED);
  config = valid_config();
  config.kv_head_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::KV_HEAD_COUNT_REQUIRED);
  config = valid_config();
  config.head_dimension = 0;
  expect_invalid(config, LlmMemoryConfigReason::HEAD_DIMENSION_REQUIRED);
  for (size_t invalid_width : {0u, 3u, 5u, 8u}) {
    config = valid_config();
    config.kv_element_bytes = invalid_width;
    expect_invalid(config, LlmMemoryConfigReason::INVALID_KV_ELEMENT_BYTES);
  }
  config = valid_config();
  config.visible_context_tokens = 0;
  expect_invalid(config, LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED);
  config = valid_config();
  config.batch_size = 0;
  expect_invalid(config, LlmMemoryConfigReason::BATCH_SIZE_REQUIRED);
  config = valid_config();
  config.requested_workers = 0;
  expect_invalid(config, LlmMemoryConfigReason::WORKER_COUNT_REQUIRED);
  config = valid_config();
  config.loop_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::LOOP_COUNT_REQUIRED);
  config = valid_config();
  config.query_head_count = 4;
  config.kv_head_count = 8;
  expect_invalid(config, LlmMemoryConfigReason::QUERY_HEADS_BELOW_KV_HEADS);
  config = valid_config();
  config.query_head_count = 10;
  config.kv_head_count = 4;
  expect_invalid(
      config,
      LlmMemoryConfigReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS);
  config = valid_config();
  config.user_specified_iterations = true;
  config.iterations = 0;
  expect_invalid(config, LlmMemoryConfigReason::EXPLICIT_ITERATIONS_REQUIRED);
  config = valid_config();
  config.iterations = 4;
  expect_invalid(
      config, LlmMemoryConfigReason::AUTOMATIC_ITERATIONS_MUST_BE_ZERO);

  config = valid_config();
  config.user_specified_iterations = true;
  config.iterations = 4;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  config = valid_config();
  config.query_head_count = config.kv_head_count;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  config = valid_config();
  config.query_head_count = 8;
  config.kv_head_count = 1;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
}

TEST(LlmMemoryConfigTest, ConvertsWeightMiBWithCheckedArithmetic) {
  LlmMemoryConfig config = valid_config();
  config.weight_size_mb = 4096;
  LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid);
  EXPECT_EQ(validation.active_weight_bytes, 4294967296ULL);

  const size_t maximum_mb =
      std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB;
  config.weight_size_mb = maximum_mb;
  validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid);
  EXPECT_EQ(validation.active_weight_bytes,
            maximum_mb * Constants::BYTES_PER_MB);

  config.weight_size_mb = maximum_mb + 1;
  expect_invalid(config, LlmMemoryConfigReason::WEIGHT_SIZE_BYTES_OVERFLOW);
}

TEST(LlmMemoryConfigTest, ResultFoundationKeepsUnavailableValuesAbsent) {
  const LlmMeasurementState measurement;
  EXPECT_EQ(measurement.status, LlmMeasurementStatus::NotRun);
  EXPECT_EQ(measurement.reason_code, "not-run");
  EXPECT_EQ(measurement.planned_steps, 0u);
  EXPECT_EQ(measurement.completed_steps, 0u);
  EXPECT_FALSE(measurement.elapsed_seconds.has_value());
  EXPECT_FALSE(measurement.effective_payload_gb_s.has_value());
  EXPECT_FALSE(measurement.checksum_valid);

  const LlmMemoryResult result;
  EXPECT_EQ(result.status, LlmRunStatus::NotStarted);
  EXPECT_EQ(result.reason_code, "not-started");
  EXPECT_FALSE(result.interruption_requested);
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_FALSE(result.scenario_order_balance_complete);
  EXPECT_TRUE(result.measurements.empty());
  EXPECT_EQ(result.counters.planned_measurements, 0u);
  EXPECT_EQ(result.counters.completed_exact_payload_bytes, 0u);
}
