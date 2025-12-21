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
#ifndef BENCHMARK_EXECUTOR_H
#define BENCHMARK_EXECUTOR_H

#include <cstddef>  // size_t
#include <vector>   // std::vector
#include <atomic>
#include <cstdint>

// Forward declarations
struct BenchmarkBuffers;
struct BenchmarkConfig;
struct BenchmarkResults;
struct HighResTimer;

// Structure to hold timing results during benchmark execution
struct TimingResults {
  double total_read_time = 0.0;
  double total_write_time = 0.0;
  double total_copy_time = 0.0;
  double total_lat_time_ns = 0.0;
  double l1_lat_time_ns = 0.0;
  double l2_lat_time_ns = 0.0;
  double custom_lat_time_ns = 0.0;
  double l1_read_time = 0.0;
  double l1_write_time = 0.0;
  double l1_copy_time = 0.0;
  double l2_read_time = 0.0;
  double l2_write_time = 0.0;
  double l2_copy_time = 0.0;
  double custom_read_time = 0.0;
  double custom_write_time = 0.0;
  double custom_copy_time = 0.0;
  std::atomic<uint64_t> total_read_checksum{0};
  std::atomic<uint64_t> l1_read_checksum{0};
  std::atomic<uint64_t> l2_read_checksum{0};
  std::atomic<uint64_t> custom_read_checksum{0};
};

// Run main memory bandwidth tests (read, write, copy)
void run_main_memory_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, 
                                     TimingResults& timings, HighResTimer& test_timer);

// Helper function to run a single cache bandwidth test (read, write, copy)
void run_single_cache_bandwidth_test(void* src_buffer, void* dst_buffer, size_t buffer_size,
                                     int cache_iterations, HighResTimer& test_timer,
                                     double& read_time, double& write_time, double& copy_time,
                                     std::atomic<uint64_t>& read_checksum);

// Run cache bandwidth tests (L1, L2, or custom)
void run_cache_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                               TimingResults& timings, HighResTimer& test_timer);

// Helper function to run a single cache latency test
void run_single_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses,
                                   HighResTimer& test_timer, double& lat_time_ns, double& latency_ns,
                                   std::vector<double>* latency_samples = nullptr, int sample_count = 0);

// Run cache latency tests (L1, L2, or custom)
void run_cache_latency_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                             TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer);

// Run main memory latency test
void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                  TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer);

#endif // BENCHMARK_EXECUTOR_H
