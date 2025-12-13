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
#include "benchmark_runner.h"
#include "buffer_manager.h"  // BenchmarkBuffers
#include "config.h"           // BenchmarkConfig
#include "benchmark.h"        // All benchmark functions and print functions
#include "constants.h"
#include <atomic>
#include <iostream>
#include <stdexcept>

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
static void run_main_memory_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, 
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

// Run cache bandwidth tests (L1, L2, or custom)
static void run_cache_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                      TimingResults& timings, HighResTimer& test_timer) {
  int cache_iterations = config.iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_bw_src() != nullptr && buffers.custom_bw_dst() != nullptr) {
      show_progress();
      std::atomic<uint64_t> custom_warmup_read_checksum{0};
      warmup_cache_read(buffers.custom_bw_src(), config.custom_buffer_size, Constants::SINGLE_THREAD, custom_warmup_read_checksum);
      timings.custom_read_time = run_read_test(buffers.custom_bw_src(), config.custom_buffer_size, cache_iterations, 
                                               Constants::SINGLE_THREAD, timings.custom_read_checksum, test_timer);
      warmup_cache_write(buffers.custom_bw_dst(), config.custom_buffer_size, Constants::SINGLE_THREAD);
      timings.custom_write_time = run_write_test(buffers.custom_bw_dst(), config.custom_buffer_size, cache_iterations, 
                                                 Constants::SINGLE_THREAD, test_timer);
      warmup_cache_copy(buffers.custom_bw_dst(), buffers.custom_bw_src(), config.custom_buffer_size, Constants::SINGLE_THREAD);
      timings.custom_copy_time = run_copy_test(buffers.custom_bw_dst(), buffers.custom_bw_src(), config.custom_buffer_size, 
                                              cache_iterations, Constants::SINGLE_THREAD, test_timer);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_bw_src() != nullptr && buffers.l1_bw_dst() != nullptr) {
      show_progress();
      std::atomic<uint64_t> l1_warmup_read_checksum{0};
      warmup_cache_read(buffers.l1_bw_src(), config.l1_buffer_size, Constants::SINGLE_THREAD, l1_warmup_read_checksum);
      timings.l1_read_time = run_read_test(buffers.l1_bw_src(), config.l1_buffer_size, cache_iterations, 
                                           Constants::SINGLE_THREAD, timings.l1_read_checksum, test_timer);
      warmup_cache_write(buffers.l1_bw_dst(), config.l1_buffer_size, Constants::SINGLE_THREAD);
      timings.l1_write_time = run_write_test(buffers.l1_bw_dst(), config.l1_buffer_size, cache_iterations, 
                                             Constants::SINGLE_THREAD, test_timer);
      warmup_cache_copy(buffers.l1_bw_dst(), buffers.l1_bw_src(), config.l1_buffer_size, Constants::SINGLE_THREAD);
      timings.l1_copy_time = run_copy_test(buffers.l1_bw_dst(), buffers.l1_bw_src(), config.l1_buffer_size, 
                                           cache_iterations, Constants::SINGLE_THREAD, test_timer);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_bw_src() != nullptr && buffers.l2_bw_dst() != nullptr) {
      show_progress();
      std::atomic<uint64_t> l2_warmup_read_checksum{0};
      warmup_cache_read(buffers.l2_bw_src(), config.l2_buffer_size, Constants::SINGLE_THREAD, l2_warmup_read_checksum);
      timings.l2_read_time = run_read_test(buffers.l2_bw_src(), config.l2_buffer_size, cache_iterations, 
                                           Constants::SINGLE_THREAD, timings.l2_read_checksum, test_timer);
      warmup_cache_write(buffers.l2_bw_dst(), config.l2_buffer_size, Constants::SINGLE_THREAD);
      timings.l2_write_time = run_write_test(buffers.l2_bw_dst(), config.l2_buffer_size, cache_iterations, 
                                             Constants::SINGLE_THREAD, test_timer);
      warmup_cache_copy(buffers.l2_bw_dst(), buffers.l2_bw_src(), config.l2_buffer_size, Constants::SINGLE_THREAD);
      timings.l2_copy_time = run_copy_test(buffers.l2_bw_dst(), buffers.l2_bw_src(), config.l2_buffer_size, 
                                           cache_iterations, Constants::SINGLE_THREAD, test_timer);
    }
  }
}

// Run cache latency tests (L1, L2, or custom)
static void run_cache_latency_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                     TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_buffer() != nullptr) {
      show_progress();
      warmup_cache_latency(buffers.custom_buffer(), config.custom_buffer_size);
      timings.custom_lat_time_ns = run_cache_latency_test(buffers.custom_buffer(), config.custom_buffer_size, 
                                                          config.custom_num_accesses, test_timer);
      results.custom_latency_ns = timings.custom_lat_time_ns / static_cast<double>(config.custom_num_accesses);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_buffer() != nullptr) {
      show_progress();
      warmup_cache_latency(buffers.l1_buffer(), config.l1_buffer_size);
      timings.l1_lat_time_ns = run_cache_latency_test(buffers.l1_buffer(), config.l1_buffer_size, 
                                                      config.l1_num_accesses, test_timer);
      results.l1_latency_ns = timings.l1_lat_time_ns / static_cast<double>(config.l1_num_accesses);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_buffer() != nullptr) {
      show_progress();
      warmup_cache_latency(buffers.l2_buffer(), config.l2_buffer_size);
      timings.l2_lat_time_ns = run_cache_latency_test(buffers.l2_buffer(), config.l2_buffer_size, 
                                                      config.l2_num_accesses, test_timer);
      results.l2_latency_ns = timings.l2_lat_time_ns / static_cast<double>(config.l2_num_accesses);
    }
  }
}

// Run main memory latency test
static void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                          TimingResults& timings, HighResTimer& test_timer) {
  show_progress();
  warmup_latency(buffers.lat_buffer(), config.buffer_size);
  timings.total_lat_time_ns = run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer);
}

// Calculate bandwidth results from timing data
static void calculate_bandwidth_results(const BenchmarkConfig& config, const TimingResults& timings, 
                                         BenchmarkResults& results) {
  // Main memory bandwidth calculations
  size_t total_bytes_read = static_cast<size_t>(config.iterations) * config.buffer_size;
  size_t total_bytes_written = static_cast<size_t>(config.iterations) * config.buffer_size;
  size_t total_bytes_copied_op = static_cast<size_t>(config.iterations) * config.buffer_size;
  
  if (timings.total_read_time > 0) {
    results.read_bw_gb_s = static_cast<double>(total_bytes_read) / timings.total_read_time / Constants::NANOSECONDS_PER_SECOND;
  }
  if (timings.total_write_time > 0) {
    results.write_bw_gb_s = static_cast<double>(total_bytes_written) / timings.total_write_time / Constants::NANOSECONDS_PER_SECOND;
  }
  if (timings.total_copy_time > 0) {
    results.copy_bw_gb_s = static_cast<double>(total_bytes_copied_op * Constants::COPY_OPERATION_MULTIPLIER) / 
                          timings.total_copy_time / Constants::NANOSECONDS_PER_SECOND;
  }
  
  // Cache bandwidth calculations
  int cache_iterations = config.iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      size_t custom_total_bytes_read = static_cast<size_t>(cache_iterations) * config.custom_buffer_size;
      size_t custom_total_bytes_written = static_cast<size_t>(cache_iterations) * config.custom_buffer_size;
      size_t custom_total_bytes_copied_op = static_cast<size_t>(cache_iterations) * config.custom_buffer_size;
      if (timings.custom_read_time > 0) {
        results.custom_read_bw_gb_s = static_cast<double>(custom_total_bytes_read) / timings.custom_read_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.custom_write_time > 0) {
        results.custom_write_bw_gb_s = static_cast<double>(custom_total_bytes_written) / timings.custom_write_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.custom_copy_time > 0) {
        results.custom_copy_bw_gb_s = static_cast<double>(custom_total_bytes_copied_op * Constants::COPY_OPERATION_MULTIPLIER) / 
                                      timings.custom_copy_time / Constants::NANOSECONDS_PER_SECOND;
      }
    }
  } else {
    if (config.l1_buffer_size > 0) {
      size_t l1_total_bytes_read = static_cast<size_t>(cache_iterations) * config.l1_buffer_size;
      size_t l1_total_bytes_written = static_cast<size_t>(cache_iterations) * config.l1_buffer_size;
      size_t l1_total_bytes_copied_op = static_cast<size_t>(cache_iterations) * config.l1_buffer_size;
      if (timings.l1_read_time > 0) {
        results.l1_read_bw_gb_s = static_cast<double>(l1_total_bytes_read) / timings.l1_read_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.l1_write_time > 0) {
        results.l1_write_bw_gb_s = static_cast<double>(l1_total_bytes_written) / timings.l1_write_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.l1_copy_time > 0) {
        results.l1_copy_bw_gb_s = static_cast<double>(l1_total_bytes_copied_op * Constants::COPY_OPERATION_MULTIPLIER) / 
                                  timings.l1_copy_time / Constants::NANOSECONDS_PER_SECOND;
      }
    }
    if (config.l2_buffer_size > 0) {
      size_t l2_total_bytes_read = static_cast<size_t>(cache_iterations) * config.l2_buffer_size;
      size_t l2_total_bytes_written = static_cast<size_t>(cache_iterations) * config.l2_buffer_size;
      size_t l2_total_bytes_copied_op = static_cast<size_t>(cache_iterations) * config.l2_buffer_size;
      if (timings.l2_read_time > 0) {
        results.l2_read_bw_gb_s = static_cast<double>(l2_total_bytes_read) / timings.l2_read_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.l2_write_time > 0) {
        results.l2_write_bw_gb_s = static_cast<double>(l2_total_bytes_written) / timings.l2_write_time / Constants::NANOSECONDS_PER_SECOND;
      }
      if (timings.l2_copy_time > 0) {
        results.l2_copy_bw_gb_s = static_cast<double>(l2_total_bytes_copied_op * Constants::COPY_OPERATION_MULTIPLIER) / 
                                  timings.l2_copy_time / Constants::NANOSECONDS_PER_SECOND;
      }
    }
  }
}

// Run a single benchmark loop and return results
static BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, int loop, HighResTimer& test_timer) {
  BenchmarkResults results;
  TimingResults timings;

  try {
    // Run all benchmark tests
    run_main_memory_bandwidth_tests(buffers, config, timings, test_timer);
    run_cache_bandwidth_tests(buffers, config, timings, test_timer);
    run_cache_latency_tests(buffers, config, timings, results, test_timer);
    run_main_memory_latency_test(buffers, config, timings, test_timer);
  } catch (const std::exception &e) {
    std::cerr << "Error during benchmark tests: " << e.what() << std::endl;
    throw;  // Re-throw to be handled by caller
  }

  // Calculate all results from timing data
  calculate_bandwidth_results(config, timings, results);
  
  // Store timing results
  results.total_read_time = timings.total_read_time;
  results.total_write_time = timings.total_write_time;
  results.total_copy_time = timings.total_copy_time;
  results.total_lat_time_ns = timings.total_lat_time_ns;
  
  // Calculate main memory latency
  if (config.lat_num_accesses > 0) {
    results.average_latency_ns = timings.total_lat_time_ns / static_cast<double>(config.lat_num_accesses);
  }

  return results;
}

int run_all_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, BenchmarkStatistics& stats) {
  // Initialize result vectors
  stats.all_read_bw_gb_s.clear();
  stats.all_write_bw_gb_s.clear();
  stats.all_copy_bw_gb_s.clear();
  stats.all_l1_latency_ns.clear();
  stats.all_l2_latency_ns.clear();
  stats.all_average_latency_ns.clear();
  stats.all_l1_read_bw_gb_s.clear();
  stats.all_l1_write_bw_gb_s.clear();
  stats.all_l1_copy_bw_gb_s.clear();
  stats.all_l2_read_bw_gb_s.clear();
  stats.all_l2_write_bw_gb_s.clear();
  stats.all_l2_copy_bw_gb_s.clear();
  stats.all_custom_latency_ns.clear();
  stats.all_custom_read_bw_gb_s.clear();
  stats.all_custom_write_bw_gb_s.clear();
  stats.all_custom_copy_bw_gb_s.clear();

  // Pre-allocate vector space if needed
  if (config.loop_count > 0) {
    stats.all_read_bw_gb_s.reserve(config.loop_count);
    stats.all_write_bw_gb_s.reserve(config.loop_count);
    stats.all_copy_bw_gb_s.reserve(config.loop_count);
    stats.all_average_latency_ns.reserve(config.loop_count);
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        stats.all_custom_latency_ns.reserve(config.loop_count);
        stats.all_custom_read_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_write_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_copy_bw_gb_s.reserve(config.loop_count);
      }
    } else {
      stats.all_l1_latency_ns.reserve(config.loop_count);
      stats.all_l2_latency_ns.reserve(config.loop_count);
      if (config.l1_buffer_size > 0) {
        stats.all_l1_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_copy_bw_gb_s.reserve(config.loop_count);
      }
      if (config.l2_buffer_size > 0) {
        stats.all_l2_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_copy_bw_gb_s.reserve(config.loop_count);
      }
    }
  }

  HighResTimer test_timer;  // Timer for individual tests

  // Main benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    try {
      BenchmarkResults loop_results = run_single_benchmark_loop(buffers, config, loop, test_timer);

      // Store results for this loop
      stats.all_read_bw_gb_s.push_back(loop_results.read_bw_gb_s);
      stats.all_write_bw_gb_s.push_back(loop_results.write_bw_gb_s);
      stats.all_copy_bw_gb_s.push_back(loop_results.copy_bw_gb_s);
      if (config.use_custom_cache_size) {
        if (config.custom_buffer_size > 0) {
          stats.all_custom_latency_ns.push_back(loop_results.custom_latency_ns);
          stats.all_custom_read_bw_gb_s.push_back(loop_results.custom_read_bw_gb_s);
          stats.all_custom_write_bw_gb_s.push_back(loop_results.custom_write_bw_gb_s);
          stats.all_custom_copy_bw_gb_s.push_back(loop_results.custom_copy_bw_gb_s);
        }
      } else {
        if (config.l1_buffer_size > 0) {
          stats.all_l1_latency_ns.push_back(loop_results.l1_latency_ns);
          stats.all_l1_read_bw_gb_s.push_back(loop_results.l1_read_bw_gb_s);
          stats.all_l1_write_bw_gb_s.push_back(loop_results.l1_write_bw_gb_s);
          stats.all_l1_copy_bw_gb_s.push_back(loop_results.l1_copy_bw_gb_s);
        }
        if (config.l2_buffer_size > 0) {
          stats.all_l2_latency_ns.push_back(loop_results.l2_latency_ns);
          stats.all_l2_read_bw_gb_s.push_back(loop_results.l2_read_bw_gb_s);
          stats.all_l2_write_bw_gb_s.push_back(loop_results.l2_write_bw_gb_s);
          stats.all_l2_copy_bw_gb_s.push_back(loop_results.l2_copy_bw_gb_s);
        }
      }
      stats.all_average_latency_ns.push_back(loop_results.average_latency_ns);

      // Print results for this loop
      std::cout << '\r' << std::flush;  // Clear progress indicator
      print_results(loop, config.buffer_size, config.buffer_size_mb, config.iterations, config.num_threads, 
                    loop_results.read_bw_gb_s, loop_results.total_read_time,
                    loop_results.write_bw_gb_s, loop_results.total_write_time, 
                    loop_results.copy_bw_gb_s, loop_results.total_copy_time,
                    loop_results.l1_latency_ns, loop_results.l2_latency_ns,
                    config.l1_buffer_size, config.l2_buffer_size,
                    loop_results.l1_read_bw_gb_s, loop_results.l1_write_bw_gb_s, loop_results.l1_copy_bw_gb_s,
                    loop_results.l2_read_bw_gb_s, loop_results.l2_write_bw_gb_s, loop_results.l2_copy_bw_gb_s,
                    loop_results.average_latency_ns, loop_results.total_lat_time_ns,
                    config.use_custom_cache_size, loop_results.custom_latency_ns, config.custom_buffer_size,
                    loop_results.custom_read_bw_gb_s, loop_results.custom_write_bw_gb_s, loop_results.custom_copy_bw_gb_s);
    } catch (const std::exception &e) {
      std::cerr << "Error during benchmark loop " << loop << ": " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

