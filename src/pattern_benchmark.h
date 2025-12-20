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
#ifndef PATTERN_BENCHMARK_H
#define PATTERN_BENCHMARK_H

#include <cstddef>  // size_t
#include <vector>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkBuffers;
struct HighResTimer;

// Structure to hold pattern benchmark results
struct PatternResults {
  // Sequential Forward (baseline)
  double forward_read_bw = 0.0;
  double forward_write_bw = 0.0;
  double forward_copy_bw = 0.0;
  
  // Sequential Reverse
  double reverse_read_bw = 0.0;
  double reverse_write_bw = 0.0;
  double reverse_copy_bw = 0.0;
  
  // Strided (Cache Line - 64B)
  double strided_64_read_bw = 0.0;
  double strided_64_write_bw = 0.0;
  double strided_64_copy_bw = 0.0;
  
  // Strided (Page - 4096B)
  double strided_4096_read_bw = 0.0;
  double strided_4096_write_bw = 0.0;
  double strided_4096_copy_bw = 0.0;
  
  // Random Uniform
  double random_read_bw = 0.0;
  double random_write_bw = 0.0;
  double random_copy_bw = 0.0;
};

// Run pattern benchmarks
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternResults& results);

// Print pattern benchmark results
void print_pattern_results(const PatternResults& results);

#endif // PATTERN_BENCHMARK_H

