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
#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>  // size_t
#include <string>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Configuration structure containing all benchmark settings
struct BenchmarkConfig {
  // User-provided settings
  unsigned long buffer_size_mb = 512;           // Default buffer size (MB)
  int iterations = 1000;                        // Default test iterations
  int loop_count = 1;                           // Default benchmark loops
  long long custom_cache_size_kb_ll = -1;       // User requested custom cache size in KB (-1 = none)
  
  // Calculated sizes
  size_t buffer_size = 0;                       // Final buffer size in bytes
  size_t l1_buffer_size = 0;                   // L1 cache buffer size
  size_t l2_buffer_size = 0;                   // L2 cache buffer size
  size_t custom_buffer_size = 0;                // Custom cache buffer size
  
  // Access counts
  size_t lat_num_accesses = 0;                  // Main memory latency test access count
  size_t l1_num_accesses = 100 * 1000 * 1000;  // L1 cache latency test access count
  size_t l2_num_accesses = 50 * 1000 * 1000;   // L2 cache latency test access count
  size_t custom_num_accesses = 100 * 1000 * 1000;  // Custom cache latency test access count
  
  // System info
  std::string cpu_name;
  int perf_cores = 0;
  int eff_cores = 0;
  int num_threads = 0;
  size_t l1_cache_size = 0;
  size_t l2_cache_size = 0;
  size_t custom_cache_size_bytes = 0;
  
  // Flags
  bool use_custom_cache_size = false;
};

// Parse command line arguments and populate config
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error (and prints error message)
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config);

// Validate configuration values
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int validate_config(BenchmarkConfig& config);

// Calculate cache buffer sizes based on cache sizes and constraints
void calculate_buffer_sizes(BenchmarkConfig& config);

// Calculate latency test access counts based on buffer sizes
void calculate_access_counts(BenchmarkConfig& config);

#endif // CONFIG_H

