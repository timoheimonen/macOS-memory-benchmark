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
#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include <vector>
#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations to avoid including headers
struct BenchmarkConfig;
struct BenchmarkBuffers;

// Structure containing results from a single benchmark loop
struct BenchmarkResults {
  // Main memory bandwidth results
  double read_bw_gb_s = 0.0;
  double write_bw_gb_s = 0.0;
  double copy_bw_gb_s = 0.0;
  double total_read_time = 0.0;
  double total_write_time = 0.0;
  double total_copy_time = 0.0;
  
  // Main memory latency results
  double average_latency_ns = 0.0;
  double total_lat_time_ns = 0.0;
  
  // Cache latency results
  double l1_latency_ns = 0.0;
  double l2_latency_ns = 0.0;
  double custom_latency_ns = 0.0;
  
  // Cache bandwidth results
  double l1_read_bw_gb_s = 0.0;
  double l1_write_bw_gb_s = 0.0;
  double l1_copy_bw_gb_s = 0.0;
  double l2_read_bw_gb_s = 0.0;
  double l2_write_bw_gb_s = 0.0;
  double l2_copy_bw_gb_s = 0.0;
  double custom_read_bw_gb_s = 0.0;
  double custom_write_bw_gb_s = 0.0;
  double custom_copy_bw_gb_s = 0.0;
};

// Structure containing aggregated statistics across all benchmark loops
struct BenchmarkStatistics {
  // Vectors storing results from each loop
  std::vector<double> all_read_bw_gb_s;
  std::vector<double> all_write_bw_gb_s;
  std::vector<double> all_copy_bw_gb_s;
  std::vector<double> all_l1_latency_ns;
  std::vector<double> all_l2_latency_ns;
  std::vector<double> all_average_latency_ns;
  std::vector<double> all_l1_read_bw_gb_s;
  std::vector<double> all_l1_write_bw_gb_s;
  std::vector<double> all_l1_copy_bw_gb_s;
  std::vector<double> all_l2_read_bw_gb_s;
  std::vector<double> all_l2_write_bw_gb_s;
  std::vector<double> all_l2_copy_bw_gb_s;
  std::vector<double> all_custom_latency_ns;
  std::vector<double> all_custom_read_bw_gb_s;
  std::vector<double> all_custom_write_bw_gb_s;
  std::vector<double> all_custom_copy_bw_gb_s;
};

// Run all benchmark loops and collect statistics
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int run_all_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, BenchmarkStatistics& stats);

#endif // BENCHMARK_RUNNER_H

