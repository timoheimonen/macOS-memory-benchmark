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

#include <sstream>

#include "core/config/constants.h"

namespace Messages {

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
      << "                          Exact steps per future scenario measurement. When omitted,\n"
      << "                          the future executor will calibrate toward 150 ms in a\n"
      << "                          100-250 ms window.\n"
      << "  -r, --count <count>    Planned cyclic weights/KV/mixed loops (default: "
      << Constants::LLM_DEFAULT_LOOP_COUNT << ").\n"
      << "      --seed <uint64>    Reproducible base seed; generated once when omitted.\n"
      << "  -o, --output <target>  Future result target; exact - is final stdout, an empty value\n"
      << "                          disables JSON, and every other non-empty value is a file.\n"
      << "  -h, --help             Show this LLM-mode help and exit.\n"
      << "This profile models CPU memory traffic only: it performs no Transformer math and\n"
      << "does not report inference tokens/s. Effective payload is not physical DRAM traffic.\n"
      << "This build validates the command boundary only: measurement execution and JSON\n"
      << "result creation are unavailable, so valid non-help runs fail without a JSON or file result.\n";
  return usage.str();
}

}  // namespace Messages
