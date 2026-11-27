// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file mode_selector.cpp
 * @brief Primary benchmark mode pre-scan implementation
 */

#include "core/config/mode_selector.h"

#include "core/config/cli_parser.h"

#include <array>
#include <string_view>

namespace {

struct ModeOption {
  PrimaryBenchmarkMode mode;
  const char* short_option;
  const char* long_option;
};

constexpr std::array<ModeOption, 7> kModeOptions{{
    {PrimaryBenchmarkMode::CpuBandwidth, "-B", "--cpu-bandwidth"},
    {PrimaryBenchmarkMode::CpuLatency, "-L", "--cpu-latency"},
    {PrimaryBenchmarkMode::Patterns, "-P", "--patterns"},
    {PrimaryBenchmarkMode::AnalyzeTlb, "-T", "--analyze-tlb"},
    {PrimaryBenchmarkMode::AnalyzeCoreToCore, "-C", "--analyze-core2core"},
    {PrimaryBenchmarkMode::GpuBandwidth, "-G", "--gpu-bandwidth"},
    {PrimaryBenchmarkMode::LlmMemory, "-M", "--llm-memory"},
}};

}  // namespace

PrimaryModeSelection select_primary_benchmark_mode(int argc, char* argv[]) {
  PrimaryModeSelection selection;
  if (argc <= 1 || argv == nullptr) {
    return selection;
  }

  std::array<bool, kModeOptions.size()> seen{};

  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    if (argv[argument_index] == nullptr) {
      continue;
    }

    const std::string_view token(argv[argument_index]);
    // Space-separated value-taking options consume one raw token that may
    // look like a primary-mode flag (e.g. `-o -G`, `--buffer-size -G`,
    // `--sweep -T`). Skip that value; the owning parser validates it.
    if (token.find('=') == std::string_view::npos && is_known_cli_value_option(token)) {
      if (argument_index + 1 < argc) {
        ++argument_index;
      }
      continue;
    }

    const std::string_view long_option_name = extract_long_option_name(token);
    for (size_t mode_index = 0; mode_index < kModeOptions.size(); ++mode_index) {
      const ModeOption& option = kModeOptions[mode_index];
      if (token != option.short_option && long_option_name != option.long_option) {
        continue;
      }
      if (!seen[mode_index]) {
        seen[mode_index] = true;
        selection.selected_options.emplace_back(option.long_option);
        if (selection.selected_options.size() == 1) {
          selection.mode = option.mode;
        } else {
          selection.mode = PrimaryBenchmarkMode::Conflict;
        }
      }
      break;
    }
  }

  return selection;
}
