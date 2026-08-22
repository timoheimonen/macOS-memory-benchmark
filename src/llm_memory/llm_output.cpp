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
 * @file llm_output.cpp
 * @brief Human-readable output for the CPU LLM memory profile
 */

#include "llm_memory/llm_output.h"

#include <iostream>
#include <string>
#include <string_view>

#include "core/config/constants.h"
#include "llm_memory/llm_json.h"
#include "output/console/messages/messages_api.h"

namespace {

constexpr size_t scenario_index(LlmScenario scenario) noexcept { return static_cast<size_t>(scenario); }

const LlmScenarioAggregate& aggregate_for(const LlmMemoryResult& result, LlmScenario scenario) {
  return result.aggregates[scenario_index(scenario)];
}

void emit_quality_warning(std::string_view token, const LlmMemoryWorkPlan& model_plan,
                          const LlmResultMetadata& metadata, const LlmMemoryResult& result) {
  if (token == "weights_only-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::WeightsOnly);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::WeightsOnly),
                     aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "kv_only-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::KvOnly);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::KvOnly),
                     aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "mixed-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::Mixed);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::Mixed),
                     aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "scenario-order-not-balanced") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_order_not_balanced() << std::endl;
    return;
  }
  if (token == "environment-not-nominal") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_environment_not_nominal() << std::endl;
    return;
  }
  if (token == "main-thread-qos-not-applied") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_main_thread_qos_not_applied(metadata.main_thread_qos.code) << std::endl;
    return;
  }
  if (token == "worker-qos-not-applied") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_worker_qos_not_applied() << std::endl;
    return;
  }
  if (token == "weight-working-set-cache-dominant") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_weight_cache_dominant(model_plan.geometry.active_weight_bytes_per_step,
                                                                    metadata.l2_data_cache_bytes)
              << std::endl;
    return;
  }
  if (token == "kv-working-set-cache-dominant") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_kv_cache_dominant(model_plan.geometry.kv_capacity_bytes,
                                                                metadata.l2_data_cache_bytes)
              << std::endl;
    return;
  }

  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const std::string scenario_token = llm_scenario_to_string(scenario);
    const std::string prefix = scenario_token + "-duration-";
    if (token.substr(0, prefix.size()) != prefix) {
      continue;
    }
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_duration_quality(scenario_token, std::string(token.substr(prefix.size())))
              << std::endl;
    return;
  }
}

void print_headline(const LlmScenarioAggregate& aggregate) {
  if (!aggregate.step_latency_seconds.headline.has_value() || !aggregate.effective_payload_gb_s.headline.has_value()) {
    return;
  }

  const bool include_steps_per_second = aggregate.scenario == LlmScenario::Mixed;
  if (include_steps_per_second && !aggregate.synthetic_memory_steps_per_second.headline.has_value()) {
    return;
  }
  const double steps_per_second = aggregate.synthetic_memory_steps_per_second.headline.value_or(0.0);
  const std::string scenario_name =
      Messages::report_llm_memory_scenario_name(llm_scenario_to_string(aggregate.scenario));
  std::cout << Messages::report_llm_memory_scenario_headline(
                   scenario_name, *aggregate.step_latency_seconds.headline * 1000.0, steps_per_second,
                   *aggregate.effective_payload_gb_s.headline, include_steps_per_second)
            << std::endl;
}

}  // namespace

void print_llm_memory_console_report(const LlmMemoryWorkPlan& model_plan, const LlmResultMetadata& metadata,
                                     const LlmMemoryResult& result) {
  std::cout << Messages::report_llm_memory_header() << std::endl;
  std::cout << Messages::report_llm_memory_payload(model_plan.geometry.active_weight_bytes_per_step,
                                                   model_plan.geometry.kv_read_bytes_per_step,
                                                   model_plan.geometry.kv_append_write_bytes_per_step)
            << std::endl;
  std::cout << Messages::report_llm_memory_crossover(model_plan.geometry.traffic_crossover_context_tokens) << "\n\n";
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    print_headline(aggregate);
  }
  std::cout << Messages::report_llm_memory_interpretation_note() << std::endl;
  std::cout.flush();

  for (const std::string& token : collect_llm_quality_warning_tokens(model_plan, metadata, result)) {
    emit_quality_warning(token, model_plan, metadata, result);
  }
}
