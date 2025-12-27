// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
//
/**
 * @file benchmark.h
 * @brief Convenience header for all benchmark utilities
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header is a convenience wrapper that includes all benchmark-related
 * headers. For better compilation performance and clearer dependencies, consider
 * including specific headers directly:
 * - core/timing/timer.h - High-resolution timer
 * - core/system/system_info.h - System information functions
 * - core/memory/memory_utils.h - Memory utility functions
 * - benchmark/benchmark_tests.h - Benchmark test functions
 * - utils/utils.h - Utility functions
 * - output/console/output_printer.h - Output/printing functions
 * - output/console/statistics.h - Statistics functions
 * - asm/asm_functions.h - Assembly function declarations
 * - core/config/version.h - Version information
 */
#ifndef BENCHMARK_H
#define BENCHMARK_H

// Include all modular headers
#include "core/timing/timer.h"
#include "core/system/system_info.h"
#include "core/memory/memory_utils.h"
#include "benchmark/benchmark_tests.h"
#include "utils/utils.h"
#include "output/console/output_printer.h"
#include "output/console/statistics.h"
#include "asm/asm_functions.h"
#include "core/config/version.h"

// Forward declarations (needed for function signatures in other headers)
struct BenchmarkConfig;
struct BenchmarkStatistics;

// --- Warmup Functions (warmup/) ---
#include "warmup/warmup.h"

#endif // BENCHMARK_H