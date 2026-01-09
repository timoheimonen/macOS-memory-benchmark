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
//

/**
 * @file program_messages.cpp
 * @brief Main program messages
 *
 * Provides implementations for main program message generation functions including
 * benchmark execution status messages, program usage/help text, and version information.
 */

#include "messages.h"
#include "core/config/constants.h"  // For default values
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Main Program Messages ---
const std::string& msg_running_benchmarks() {
  static const std::string msg = "\nRunning benchmarks...";
  return msg;
}

std::string msg_done_total_time(double total_time_sec) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::TIME_PRECISION);
  oss << "\nDone. Total execution time: " << total_time_sec << " s";
  return oss.str();
}

const std::string& msg_running_pattern_benchmarks() {
  static const std::string msg = "\nRunning Pattern Benchmarks...\n";
  return msg;
}

// --- Usage/Help Messages ---
std::string usage_header(const std::string& version) {
  std::ostringstream oss;
  oss << "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>\n"
      << "Version: " << version << " by Timo Heimonen <timo.heimonen@proton.me>\n"
      << "License: GNU GPL v3. See <https://www.gnu.org/licenses/>\n"
      << "This program is free software: you can redistribute it and/or modify\n"
      << "it under the terms of the GNU General Public License as published by\n"
      << "the Free Software Foundation, either version 3 of the License, or\n"
      << "(at your option) any later version.\n"
      << "This program is distributed in the hope that it will be useful,\n"
      << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
      << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
      << "Link: https://github.com/timoheimonen/macOS-memory-benchmark\n\n";
  return oss.str();
}

std::string usage_options(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Usage: " << prog_name << " [options]\n"
      << "Options:\n"
      << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: " << Constants::DEFAULT_ITERATIONS << ")\n"
      << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: " << Constants::DEFAULT_BUFFER_SIZE_MB << ").\n"
      << "                        The maximum allowed <size_mb> is automatically determined such that\n"
      << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
      << "  -count <count>        Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ").\n"
      << "                        When count > 1, statistics include percentiles (P50/P90/P95/P99) and stddev.\n"
      << "  -latency-samples <count> Number of latency samples to collect per test (default: " << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ")\n"
      << "  -threads <count>      Number of threads to use for benchmarks (default: detected\n"
      << "                        CPU core count). Applies to all benchmarks including cache tests.\n"
      << "                        If specified value exceeds available cores, it will be capped to\n"
      << "                        the maximum number of cores with a warning.\n"
      << "  -cache-size <size_kb> Custom cache size in Kilobytes (KB) as integer (16 KB to 524288 KB).\n"
      << "                        Minimum is 16 KB (system page size). When set, skips automatic\n"
      << "                        L1/L2 cache size detection and only performs bandwidth and latency\n"
      << "                        tests for the custom cache size.\n"
      << "  -patterns             Run pattern benchmarks (sequential forward/reverse, strided,\n"
      << "                        and random access patterns). When set, only pattern benchmarks\n"
      << "                        are executed, skipping standard bandwidth and latency tests.\n"
      << "                        use with -buffersize <size_mb> to set the buffer size for the pattern benchmarks.\n"
      << "  -only-bandwidth       Run only bandwidth tests (read/write/copy for main memory and cache).\n"
      << "                        Skips all latency tests. Cannot be used with -patterns, -cache-size,\n"
      << "                        or -latency-samples.\n"
      << "  -only-latency         Run only latency tests (main memory and cache latency).\n"
      << "                        Skips all bandwidth tests. Cannot be used with -patterns or -iterations.\n"
      << "  -non-cacheable        Apply cache-discouraging hints to src/dst buffers.\n"
      << "                        Uses madvise() hints to discourage caching, but does NOT provide\n"
      << "                        true non-cacheable memory (user-space cannot modify page tables).\n"
      << "                        Best-effort approach that may reduce but not eliminate caching.\n"
      << "  -output <file>        Save benchmark results to JSON file. If path is relative,\n"
      << "                        file is saved in current working directory.\n"
      << "  -h, --help            Show this help message and exit\n\n";
  return oss.str();
}

std::string usage_example(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Example: " << prog_name << " -iterations 2000 -buffersize 1024 -output results.json\n";
  return oss.str();
}

} // namespace Messages

