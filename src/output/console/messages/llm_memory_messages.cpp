// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file llm_memory_messages.cpp
 * @brief Centralized synthetic LLM decode-memory CLI text
 */

#include "output/console/messages/messages_api.h"

#include <iomanip>
#include <sstream>

#include "core/config/constants.h"

namespace Messages {

const std::string& llm_memory_command_name() {
  static const std::string name = "LLM memory profile";
  return name;
}

const std::string& error_llm_memory_must_be_used_alone() {
  static const std::string message =
      "--llm-memory requires --weight-size-mb <MiB>, --layers <count>, "
      "--query-heads <count>, --kv-heads <count>, --head-dim <count>, and "
      "--context-tokens <count>; it allows only optional "
      "--kv-element-bytes <1|2|4>, --batch-size <count>, "
      "-t/--threads <count>, -i/--iterations <count>, "
      "-r/--count <count>, --seed <uint64>, -o/--output <target>, and "
      "-h/--help (no other options allowed)";
  return message;
}

std::string error_llm_memory_missing_required_option(
    const std::string& option) {
  return "Missing required --llm-memory option: " + option;
}

std::string error_llm_memory_config_invalid(
    const std::string& reason_code) {
  return "Invalid --llm-memory configuration (reason_code=" + reason_code +
         ")";
}

std::string error_llm_memory_iterations_exceed_limit(size_t requested,
                                                     size_t maximum) {
  std::ostringstream message;
  message << "LLM memory iterations exceed the exact-work guardrail "
          << "(requested " << requested << ", maximum " << maximum << ")";
  return message.str();
}

std::string error_llm_memory_run_failed(const std::string& reason_code) {
  return "Synthetic LLM decode memory profile failed (reason_code=" +
         reason_code + ")";
}

const std::string& llm_memory_reason_positive_integer() {
  static const std::string reason = "must be a positive integer";
  return reason;
}

const std::string& llm_memory_reason_kv_element_bytes() {
  static const std::string reason = "must be exactly 1, 2, or 4";
  return reason;
}

const std::string& llm_memory_reason_platform_size_range() {
  static const std::string reason =
      "out of range for a platform size";
  return reason;
}

std::string llm_memory_usage_options(const std::string& prog_name) {
  std::ostringstream usage;
  usage
      << "Usage: " << prog_name << " --llm-memory [options]\n"
      << "Options for standalone CPU synthetic LLM decode-memory mode:\n"
      << "  -M, --llm-memory       Select the fixed-context memory-only decode profile.\n"
      << "      --weight-size-mb <MiB>\n"
      << "                          Required active weight bytes per synthetic step, in MiB.\n"
      << "      --layers <count>    Required transformer layer count.\n"
      << "      --query-heads <count>\n"
      << "                          Required query-head count; must be at least as large as KV heads\n"
      << "                          and divisible by them.\n"
      << "      --kv-heads <count> Required physical KV-head count.\n"
      << "      --head-dim <count> Required elements per K or V head vector.\n"
      << "      --kv-element-bytes <1|2|4>\n"
      << "                          KV element width (default: "
      << Constants::LLM_DEFAULT_KV_ELEMENT_BYTES << " bytes).\n"
      << "      --context-tokens <count>\n"
      << "                          Required fixed visible context, including the current token.\n"
      << "      --batch-size <count>\n"
      << "                          Batch sequences per synthetic step (default: "
      << Constants::LLM_DEFAULT_BATCH_SIZE << ").\n"
      << "  -t, --threads <count>  Requested CPU workers; detected workers are used when omitted.\n"
      << "  -i, --iterations <count>\n"
      << "                          Exact steps per scenario measurement. When omitted, each\n"
      << "                          scenario calibrates toward 150 ms in a\n"
      << "                          100-250 ms window.\n"
      << "  -r, --count <count>    Cyclic weights/KV/mixed loops (default: "
      << Constants::LLM_DEFAULT_LOOP_COUNT << ").\n"
      << "      --seed <uint64>    Reproducible base seed; generated once when omitted.\n"
      << "  -o, --output <target>  JSON schema 1 target; exact - writes one final document to\n"
      << "                          stdout and routes human output to stderr. Every other non-empty\n"
      << "                          target is a file with atomic scenario and terminal checkpoints.\n"
      << "                          An empty value disables JSON for this direct command.\n"
      << "  -h, --help             Show this LLM-mode help and exit.\n"
      << "This profile models CPU memory traffic only: it performs no Transformer math and\n"
      << "does not report inference tokens/s. Effective payload is not physical DRAM traffic.\n";
  return usage.str();
}

const std::string& report_llm_memory_header() {
  static const std::string message =
      "Synthetic LLM decode memory profile (CPU, fixed context, warm/cacheable)";
  return message;
}

std::string report_llm_memory_payload(size_t active_weight_bytes_per_step,
                                      size_t kv_read_bytes_per_step,
                                      size_t kv_append_bytes_per_step) {
  std::ostringstream report;
  report << "  Active weight bytes / step: " << active_weight_bytes_per_step
         << "\n"
         << "  KV read bytes / step:       " << kv_read_bytes_per_step << "\n"
         << "  KV append bytes / step:     " << kv_append_bytes_per_step;
  return report.str();
}

std::string report_llm_memory_crossover(
    double traffic_crossover_context_tokens) {
  std::ostringstream report;
  report << "  Traffic crossover:          " << std::fixed
         << std::setprecision(2) << traffic_crossover_context_tokens
         << " visible context tokens";
  return report.str();
}

std::string report_llm_memory_scenario_headline(
    const std::string& scenario_name, double step_latency_ms,
    double synthetic_memory_steps_per_second, double effective_payload_gb_s,
    bool include_steps_per_second) {
  std::ostringstream report;
  report << "  " << std::left << std::setw(14) << (scenario_name + ":")
         << std::right << std::fixed << std::setprecision(3)
         << step_latency_ms << " ms/step, ";
  if (include_steps_per_second) {
    report << std::setprecision(2) << synthetic_memory_steps_per_second
           << " synthetic memory steps/s, ";
  }
  report << std::setprecision(2) << effective_payload_gb_s
         << " GB/s effective payload";
  return report.str();
}

std::string report_llm_memory_scenario_name(
    const std::string& scenario_token) {
  if (scenario_token == "weights_only") {
    return "Weights only";
  }
  if (scenario_token == "kv_only") {
    return "KV only";
  }
  if (scenario_token == "mixed") {
    return "Mixed";
  }
  return "Unknown";
}

const std::string& report_llm_memory_interpretation_note() {
  static const std::string message =
      "  Interpretation: each step is synthetic memory-only work, not an inference token; "
      "effective payload is logical, not physical DRAM-counter traffic.\n"
      "  Context/layout: the fixed visible context includes the current-token slot; KV uses "
      "contiguous layer/batch/token/head/dimension layout.\n"
      "  Crossover: logical weight/KV-read payload equality is not a proven hardware "
      "bottleneck transition.\n"
      "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
      "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence.";
  return message;
}

std::string warning_llm_memory_high_cv(const std::string& scenario_token,
                                       double cv_pct,
                                       double threshold_pct) {
  std::ostringstream warning;
  warning << "LLM " << report_llm_memory_scenario_name(scenario_token)
          << " repeatability CV " << std::fixed << std::setprecision(2)
          << cv_pct << "% exceeds " << threshold_pct << "%";
  return warning.str();
}

const std::string& warning_llm_memory_order_not_balanced() {
  static const std::string message =
      "LLM scenario order is not fully balanced across completed loops";
  return message;
}

std::string warning_llm_memory_duration_quality(
    const std::string& scenario_token, const std::string& quality) {
  return "LLM " + report_llm_memory_scenario_name(scenario_token) +
         " duration quality is " + quality;
}

const std::string& warning_llm_memory_environment_not_nominal() {
  static const std::string message =
      "LLM result environment is not reference-eligible (thermal state or Low Power Mode)";
  return message;
}

std::string warning_llm_memory_main_thread_qos_not_applied(int code) {
  std::ostringstream warning;
  warning << "LLM main-thread QoS request was not applied (code: " << code
          << ")";
  return warning.str();
}

const std::string& warning_llm_memory_worker_qos_not_applied() {
  static const std::string message =
      "One or more LLM worker QoS requests were not applied";
  return message;
}

namespace {

std::string warning_llm_memory_cache_dominant(
    const char* working_set_name, size_t working_set_bytes,
    size_t l2_cache_bytes) {
  std::ostringstream warning;
  warning << "LLM " << working_set_name << " working set ("
          << working_set_bytes << " bytes) does not exceed reported L2 cache ("
          << l2_cache_bytes << " bytes); the result may be cache-dominant";
  return warning.str();
}

}  // namespace

std::string warning_llm_memory_weight_cache_dominant(
    size_t working_set_bytes, size_t l2_cache_bytes) {
  return warning_llm_memory_cache_dominant("weight", working_set_bytes,
                                           l2_cache_bytes);
}

std::string warning_llm_memory_kv_cache_dominant(
    size_t working_set_bytes, size_t l2_cache_bytes) {
  return warning_llm_memory_cache_dominant("KV", working_set_bytes,
                                           l2_cache_bytes);
}

}  // namespace Messages
