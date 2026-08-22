// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file llm_memory_messages.cpp
 * @brief Centralized synthetic LLM memory-profile CLI text
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
      "phase-specific token geometry; it allows only optional "
      "--phase <decode|prefill>, --context-tokens <count>, "
      "--prompt-tokens <count>, --attention-query-tile-tokens <count>, "
      "--kv-element-bytes <1|2|4>, --kv-layout <contiguous|paged>, "
      "--kv-block-tokens <count>, --batch-size <count>, "
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
  return "Synthetic LLM memory profile failed (reason_code=" + reason_code +
         ")";
}

const std::string& error_llm_paged_table_protection_failed() {
  static const std::string message =
      "Failed to make the paged KV block table read-only";
  return message;
}

const std::string& llm_memory_reason_positive_integer() {
  static const std::string reason = "must be a positive integer";
  return reason;
}

const std::string& llm_memory_reason_kv_element_bytes() {
  static const std::string reason = "must be exactly 1, 2, or 4";
  return reason;
}

const std::string& llm_memory_reason_phase() {
  static const std::string reason = "must be exactly decode or prefill";
  return reason;
}

const std::string& llm_memory_reason_kv_layout() {
  static const std::string reason =
      "must be exactly contiguous or paged";
  return reason;
}

const std::string& llm_memory_reason_platform_size_range() {
  static const std::string reason =
      "out of range for a platform size";
  return reason;
}

std::string llm_memory_usage_options(const std::string& prog_name) {
  std::ostringstream usage;
  usage << "Usage: " << prog_name << " --llm-memory [options]\n"
        << "Options for standalone CPU synthetic LLM memory mode:\n"
        << "  -M, --llm-memory       Select the memory-only LLM profile.\n"
        << "      --phase <decode|prefill>\n"
        << "                          Workload phase (default: decode). CPU prefill currently\n"
        << "                          supports contiguous KV only.\n"
        << "      --weight-size-mb <MiB>\n"
        << "                          Required active weight bytes per work unit, in MiB.\n"
        << "      --layers <count>    Required transformer layer count.\n"
        << "      --query-heads <count>\n"
        << "                          Required query-head count; must be at least as large as KV heads\n"
        << "                          and divisible by them.\n"
        << "      --kv-heads <count> Required physical KV-head count.\n"
        << "      --head-dim <count> Required elements per K or V head vector.\n"
        << "      --kv-element-bytes <1|2|4>\n"
        << "                          KV element width (default: " << Constants::LLM_DEFAULT_KV_ELEMENT_BYTES
        << " bytes).\n"
        << "      --context-tokens <count>\n"
        << "                          Required only for decode; fixed visible context including\n"
        << "                          the current token. Rejected for prefill.\n"
        << "      --prompt-tokens <count>\n"
        << "                          Required only for prefill; full prompt length P, P >= 1.\n"
        << "                          Rejected for decode.\n"
        << "      --attention-query-tile-tokens <count>\n"
        << "                          Required only for prefill; query tile Q, 1 <= Q <= P.\n"
        << "                          Rejected for decode.\n"
        << "      --kv-layout <contiguous|paged>\n"
        << "                          KV storage layout (default: contiguous).\n"
        << "      --kv-block-tokens <count>\n"
        << "                          Required only for paged KV; must be a positive power of two\n"
        << "                          no greater than UINT32_MAX; it may exceed the phase sequence length.\n"
        << "                          Rejected for contiguous KV. CPU prefill+paged is not yet\n"
        << "                          supported (reason_code=cpu-prefill-paged-not-yet-supported).\n"
        << "      --batch-size <count>\n"
        << "                          Batch sequences per work unit (default: " << Constants::LLM_DEFAULT_BATCH_SIZE
        << ").\n"
        << "  -t, --threads <count>  Requested CPU workers; detected workers are used when omitted.\n"
        << "  -i, --iterations <count>\n"
        << "                          Exact work units per scenario measurement. A work unit is one\n"
        << "                          decode step or full-prompt prefill operation. When omitted, each\n"
        << "                          scenario calibrates toward 150 ms in a\n"
        << "                          100-250 ms window.\n"
        << "  -r, --count <count>    Cyclic weights/KV/mixed loops (default: " << Constants::LLM_DEFAULT_LOOP_COUNT
        << ").\n"
        << "      --seed <uint64>    Reproducible base seed; generated once when omitted.\n"
        << "  -o, --output <target>  JSON schema 1 target; exact - writes one final document to\n"
        << "                          stdout and routes human output to stderr. Every other non-empty\n"
        << "                          target is a file with atomic scenario and terminal checkpoints.\n"
        << "                          An empty value disables JSON for this direct command.\n"
        << "  -h, --help             Show this LLM-mode help and exit.\n"
        << "This profile models CPU memory traffic only: it performs no Transformer math and\n"
        << "does not report inference tokens/s. Effective model payload is not physical DRAM traffic.\n";
  return usage.str();
}

std::string report_llm_memory_header(const std::string& backend,
                                     const std::string& phase,
                                     const std::string& work_unit_kind,
                                     const std::string& kv_layout) {
  return "Synthetic LLM memory profile (backend=" + backend +
         ", phase=" + phase + ", work_unit=" + work_unit_kind +
         ", kv_layout=" + kv_layout + ", warm/cacheable)";
}

std::string report_llm_memory_work_unit_name(
    const std::string& work_unit_kind, bool plural) {
  if (work_unit_kind == "decode_step") {
    return plural ? "decode steps" : "decode step";
  }
  if (work_unit_kind == "prefill_operation") {
    return plural ? "prefill operations" : "prefill operation";
  }
  return plural ? "work units" : "work unit";
}

std::string report_llm_memory_payload(
    const std::string& work_unit_name,
    size_t active_weight_bytes_per_work_unit,
    size_t kv_read_bytes_per_work_unit,
    size_t kv_write_bytes_per_work_unit) {
  std::ostringstream report;
  report << "  Active weight bytes / " << work_unit_name << ": "
         << active_weight_bytes_per_work_unit << "\n"
         << "  KV read bytes / " << work_unit_name << ":       "
         << kv_read_bytes_per_work_unit << "\n"
         << "  KV write bytes / " << work_unit_name << ":      "
         << kv_write_bytes_per_work_unit;
  return report.str();
}

std::string report_llm_memory_decode_geometry(
    size_t visible_context_tokens, double traffic_crossover_context_tokens) {
  std::ostringstream report;
  report << "  Visible context tokens:     " << visible_context_tokens << "\n"
         << "  Traffic crossover:          " << std::fixed
         << std::setprecision(2) << traffic_crossover_context_tokens
         << " visible context tokens";
  return report.str();
}

std::string report_llm_memory_prefill_geometry(size_t prompt_tokens, size_t attention_query_tile_tokens,
                                               size_t tile_count, size_t attention_prefix_token_visits_per_sequence,
                                               size_t causal_token_pairs_per_sequence, size_t logical_attention_pairs,
                                               size_t logical_attention_fma_terms) {
  std::ostringstream report;
  report << "  Prompt tokens (P):                 " << prompt_tokens << "\n"
         << "  Attention query tile tokens (Q):  " << attention_query_tile_tokens << "\n"
         << "  Attention query tiles (C):        " << tile_count << "\n"
         << "  Prefix token visits / sequence:   " << attention_prefix_token_visits_per_sequence << "\n"
         << "  Causal token pairs / sequence:    " << causal_token_pairs_per_sequence << "\n"
         << "  Logical attention pairs:          " << logical_attention_pairs << "\n"
         << "  Logical attention FMA terms:      " << logical_attention_fma_terms;
  return report.str();
}

std::string report_llm_memory_paged_layout(
    const LlmPagedLayoutReportValues& values) {
  std::ostringstream report;
  report << "  Paged KV block tokens (G): " << values.block_tokens << "\n"
         << "  Blocks per sequence (N):   " << values.blocks_per_sequence
         << "\n"
         << "  Physical blocks/layer (P_b): "
         << values.physical_blocks_per_layer << "\n"
         << "  Physical block geometry: total_blocks="
         << values.total_physical_blocks << ", block_bytes="
         << values.block_bytes << "\n"
         << "  Terminal block: tokens=" << values.terminal_block_tokens
         << ", valid_bytes=" << values.terminal_valid_bytes << "\n"
         << "  K bytes (logical/physical/padding): "
         << values.k_logical_bytes << "/" << values.k_physical_bytes << "/"
         << values.k_padding_bytes << "\n"
         << "  V bytes (logical/physical/padding): "
         << values.v_logical_bytes << "/" << values.v_physical_bytes << "/"
         << values.v_padding_bytes << "\n"
         << "  Block table: " << values.block_table_entries
         << " uint32 entries, " << values.block_table_bytes << " bytes, "
         << values.block_table_page_rounded_bytes << " page-rounded bytes\n"
         << "  Permutation: version=" << values.permutation_version
         << ", seed=" << values.permutation_seed
         << ", sha256=" << values.permutation_sha256 << "\n"
         << "  Permutation identity: " << values.permutation_identity << "\n"
         << "  Timed block-table metadata / KV-active "
         << values.work_unit_name << ": "
         << values.metadata_lookups_per_work_unit << " lookups, "
         << values.metadata_bytes_per_work_unit << " bytes\n"
         << "  Accounted bytes / KV-active " << values.work_unit_name << ": "
         << values.accounted_bytes_per_work_unit << "\n"
         << "  Effective model payload excludes timed block-table metadata "
            "bytes.";
  return report.str();
}

std::string report_llm_memory_scenario_headline(
    const std::string& scenario_name, const std::string& work_unit_name,
    const std::string& plural_work_unit_name, double work_unit_latency_ms,
    double synthetic_memory_work_units_per_second,
    double effective_model_payload_gb_s,
    bool include_work_units_per_second) {
  std::ostringstream report;
  report << "  " << std::left << std::setw(14) << (scenario_name + ":")
         << std::right << std::fixed << std::setprecision(3)
         << work_unit_latency_ms << " ms/" << work_unit_name << ", ";
  if (include_work_units_per_second) {
    report << std::setprecision(2) << synthetic_memory_work_units_per_second
           << " synthetic " << plural_work_unit_name << "/s, ";
  }
  report << std::setprecision(2) << effective_model_payload_gb_s
         << " GB/s effective model payload";
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

std::string report_llm_memory_interpretation_note(
    const std::string& phase, const std::string& kv_layout,
    const std::string& work_unit_name) {
  std::ostringstream message;
  message << "  Interpretation: each " << work_unit_name
          << " is synthetic memory-only work, not an inference token; effective model payload "
             "is logical, not physical DRAM-counter traffic.\n"
          << "  Phase/layout: phase=" << phase << ", kv_layout=" << kv_layout;
  if (phase == "decode") {
    message << "; the visible context includes the current-token slot";
  } else if (phase == "prefill") {
    message << "; prefill performs no Transformer compute and does not "
               "predict TTFT";
  }
  if (kv_layout == "contiguous") {
    message << "; KV uses layer/batch/token/head/dimension order";
  }
  message << ".\n";
  if (phase == "decode") {
    message << "  Crossover: logical weight/KV-read payload equality is not a proven hardware "
               "bottleneck transition.\n";
  }
  message << "  Comparability: small weight or KV working sets can be cache-dominant; order imbalance, "
             "high CV, non-nominal environment, QoS failures, or off-target duration reduce confidence.";
  return message.str();
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
