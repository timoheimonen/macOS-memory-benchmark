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
#include "benchmark_executor.h"
#include "buffer_manager.h"  // BenchmarkBuffers
#include "config.h"           // BenchmarkConfig
#include "benchmark.h"        // All benchmark functions and print functions
#include "benchmark_runner.h" // BenchmarkResults
#include "constants.h"
#include <atomic>

// Run main memory bandwidth tests (read, write, copy)
void run_main_memory_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, 
                                     TimingResults& timings, HighResTimer& test_timer) {
  show_progress();
  std::atomic<uint64_t> warmup_read_checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, warmup_read_checksum);
  timings.total_read_time = run_read_test(buffers.src_buffer(), config.buffer_size, config.iterations, 
                                          config.num_threads, timings.total_read_checksum, test_timer);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  timings.total_write_time = run_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations, 
                                            config.num_threads, test_timer);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  timings.total_copy_time = run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, 
                                          config.iterations, config.num_threads, test_timer);
}

// Helper function to run a single cache bandwidth test (read, write, copy)
void run_single_cache_bandwidth_test(void* src_buffer, void* dst_buffer, size_t buffer_size,
                                     int cache_iterations, HighResTimer& test_timer,
                                     double& read_time, double& write_time, double& copy_time,
                                     std::atomic<uint64_t>& read_checksum) {
  show_progress();
  std::atomic<uint64_t> warmup_read_checksum{0};
  warmup_cache_read(src_buffer, buffer_size, Constants::SINGLE_THREAD, warmup_read_checksum);
  read_time = run_read_test(src_buffer, buffer_size, cache_iterations, 
                           Constants::SINGLE_THREAD, read_checksum, test_timer);
  
  warmup_cache_write(dst_buffer, buffer_size, Constants::SINGLE_THREAD);
  write_time = run_write_test(dst_buffer, buffer_size, cache_iterations, 
                             Constants::SINGLE_THREAD, test_timer);
  
  warmup_cache_copy(dst_buffer, src_buffer, buffer_size, Constants::SINGLE_THREAD);
  copy_time = run_copy_test(dst_buffer, src_buffer, buffer_size, 
                           cache_iterations, Constants::SINGLE_THREAD, test_timer);
}

// Run cache bandwidth tests (L1, L2, or custom)
void run_cache_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                               TimingResults& timings, HighResTimer& test_timer) {
  int cache_iterations = config.iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_bw_src() != nullptr && buffers.custom_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.custom_bw_src(), buffers.custom_bw_dst(), config.custom_buffer_size,
                                      cache_iterations, test_timer,
                                      timings.custom_read_time, timings.custom_write_time, timings.custom_copy_time,
                                      timings.custom_read_checksum);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_bw_src() != nullptr && buffers.l1_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size,
                                      cache_iterations, test_timer,
                                      timings.l1_read_time, timings.l1_write_time, timings.l1_copy_time,
                                      timings.l1_read_checksum);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_bw_src() != nullptr && buffers.l2_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size,
                                      cache_iterations, test_timer,
                                      timings.l2_read_time, timings.l2_write_time, timings.l2_copy_time,
                                      timings.l2_read_checksum);
    }
  }
}

// Helper function to run a single cache latency test
void run_single_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses,
                                   HighResTimer& test_timer, double& lat_time_ns, double& latency_ns,
                                   std::vector<double>* latency_samples, int sample_count) {
  show_progress();
  warmup_cache_latency(buffer, buffer_size);
  lat_time_ns = run_cache_latency_test(buffer, buffer_size, num_accesses, test_timer, latency_samples, sample_count);
  if (num_accesses > 0) {
    latency_ns = lat_time_ns / static_cast<double>(num_accesses);
  } else {
    latency_ns = 0.0;  // Avoid division by zero
  }
}

// Run cache latency tests (L1, L2, or custom)
void run_cache_latency_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                             TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_buffer() != nullptr && config.custom_num_accesses > 0) {
      run_single_cache_latency_test(buffers.custom_buffer(), config.custom_buffer_size, config.custom_num_accesses,
                                    test_timer, timings.custom_lat_time_ns, results.custom_latency_ns,
                                    &results.custom_latency_samples, config.latency_sample_count);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_buffer() != nullptr && config.l1_num_accesses > 0) {
      run_single_cache_latency_test(buffers.l1_buffer(), config.l1_buffer_size, config.l1_num_accesses,
                                    test_timer, timings.l1_lat_time_ns, results.l1_latency_ns,
                                    &results.l1_latency_samples, config.latency_sample_count);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_buffer() != nullptr && config.l2_num_accesses > 0) {
      run_single_cache_latency_test(buffers.l2_buffer(), config.l2_buffer_size, config.l2_num_accesses,
                                    test_timer, timings.l2_lat_time_ns, results.l2_latency_ns,
                                    &results.l2_latency_samples, config.latency_sample_count);
    }
  }
}

// Run main memory latency test
void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                  TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  show_progress();
  warmup_latency(buffers.lat_buffer(), config.buffer_size);
  timings.total_lat_time_ns = run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer,
                                                nullptr, 0);
}
