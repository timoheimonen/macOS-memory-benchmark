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

#include <array>

namespace {

struct ModeOption {
  PrimaryBenchmarkMode mode;
  const char* short_option;
  const char* long_option;
};

constexpr std::array<ModeOption, 5> kModeOptions{{
    {PrimaryBenchmarkMode::Standard, "-B", "--benchmark"},
    {PrimaryBenchmarkMode::Patterns, "-P", "--patterns"},
    {PrimaryBenchmarkMode::AnalyzeTlb, "-T", "--analyze-tlb"},
    {PrimaryBenchmarkMode::AnalyzeCoreToCore, "-C", "--analyze-core2core"},
    {PrimaryBenchmarkMode::GpuBandwidth, "-G", "--gpu-bandwidth"},
}};

}  // namespace

PrimaryModeSelection select_primary_benchmark_mode(int argc, char* argv[]) {
  PrimaryModeSelection selection;
  std::array<bool, kModeOptions.size()> seen{};

  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    for (size_t mode_index = 0; mode_index < kModeOptions.size(); ++mode_index) {
      const ModeOption& option = kModeOptions[mode_index];
      if (argument != option.short_option && argument != option.long_option) {
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
