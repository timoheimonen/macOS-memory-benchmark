// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file mode_selector.h
 * @brief Side-effect-free primary benchmark mode pre-scan
 */

#ifndef MODE_SELECTOR_H
#define MODE_SELECTOR_H

#include <string>
#include <vector>

/** Primary command modes recognized before mode-specific parsing begins. */
enum class PrimaryBenchmarkMode {
  None = 0,
  Standard,
  Patterns,
  AnalyzeTlb,
  AnalyzeCoreToCore,
  GpuBandwidth,
  Conflict,
};

/** Result of scanning argv for distinct primary mode selectors. */
struct PrimaryModeSelection {
  PrimaryBenchmarkMode mode = PrimaryBenchmarkMode::None;
  std::vector<std::string> selected_options;
};

/**
 * @brief Discover all distinct primary mode selectors in one argv scan.
 *
 * Repeated occurrences of one selector remain one selected mode so that the
 * owning parser can report its normal duplicate-option error. Two or more
 * distinct selectors produce `Conflict`, independent of argv order.
 */
PrimaryModeSelection select_primary_benchmark_mode(int argc, char* argv[]);

#endif  // MODE_SELECTOR_H
