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
#include "pattern_benchmark.h"
#include "benchmark.h"
#include "buffer_manager.h"
#include "config.h"
#include "constants.h"
#include "messages.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>
#include <atomic>

// Helper function to run a pattern read test
static double run_pattern_read_test(void* buffer, size_t size, int iterations, 
                                     uint64_t (*read_func)(const void*, size_t),
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = read_func(buffer, size);
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_relaxed);
  return elapsed;
}

// Helper function to run a pattern write test
static double run_pattern_write_test(void* buffer, size_t size, int iterations,
                                      void (*write_func)(void*, size_t),
                                      HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    write_func(buffer, size);
  }
  return timer.stop();
}

// Helper function to run a pattern copy test
static double run_pattern_copy_test(void* dst, void* src, size_t size, int iterations,
                                     void (*copy_func)(void*, const void*, size_t),
                                     HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    copy_func(dst, src, size);
  }
  return timer.stop();
}

// Helper function to run a strided pattern test
static double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                             std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = memory_read_strided_loop_asm(buffer, size, stride);
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_relaxed);
  return elapsed;
}

static double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                              HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_write_strided_loop_asm(buffer, size, stride);
  }
  return timer.stop();
}

static double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                             HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_copy_strided_loop_asm(dst, src, size, stride);
  }
  return timer.stop();
}

// Helper function to generate random indices for random access pattern
static std::vector<size_t> generate_random_indices(size_t buffer_size, size_t num_accesses) {
  std::vector<size_t> indices;
  indices.reserve(num_accesses);
  
  // Generate random offsets that are 32-byte aligned (for 32B loads)
  // Each access loads 32 bytes, so we need offset + 32 <= buffer_size
  // Maximum valid offset is buffer_size - 32 (if buffer_size >= 32)
  if (buffer_size < 32) {
    // Buffer too small for 32-byte access
    return indices;  // Return empty vector
  }
  
  // Calculate maximum valid 32-byte aligned offset
  // We need: offset + 32 <= buffer_size, so offset <= buffer_size - 32
  // Round down (buffer_size - 32) to 32-byte boundary
  size_t max_offset = ((buffer_size - 32) / 32) * 32;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, max_offset / 32);  // Index in 32-byte units
  
  for (size_t i = 0; i < num_accesses; ++i) {
    size_t offset = dis(gen) * 32;  // Convert to byte offset
    indices.push_back(offset);
  }
  
  return indices;
}

// Helper function to run a random pattern test
static double run_pattern_read_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                            std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = memory_read_random_loop_asm(buffer, indices.data(), indices.size());
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_relaxed);
  return elapsed;
}

static double run_pattern_write_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                             HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_write_random_loop_asm(buffer, indices.data(), indices.size());
  }
  return timer.stop();
}

static double run_pattern_copy_random_test(void* dst, void* src, const std::vector<size_t>& indices, int iterations,
                                            HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_copy_random_loop_asm(dst, src, indices.data(), indices.size());
  }
  return timer.stop();
}

// Helper function to calculate bandwidth from time and data size
static double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns) {
  // Avoid division by zero - use minimum time of 1 nanosecond if time is too small
  // This handles cases where very fast operations complete in < 1ns due to timer resolution
  double effective_time_ns = (elapsed_time_ns < 1e-9) ? 1e-9 : elapsed_time_ns;
  return (static_cast<double>(data_size) * iterations) / (effective_time_ns * 1e9);
}

// Run forward pattern benchmarks (baseline sequential access)
static void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                            PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  double read_time = run_read_test(buffers.src_buffer(), config.buffer_size, config.iterations,
                                    config.num_threads, checksum, timer);
  results.forward_read_bw = calculate_bandwidth(config.buffer_size, config.iterations, read_time);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  double write_time = run_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations,
                                      config.num_threads, timer);
  results.forward_write_bw = calculate_bandwidth(config.buffer_size, config.iterations, write_time);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  double copy_time = run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                   config.iterations, config.num_threads, timer);
  results.forward_copy_bw = calculate_bandwidth(config.buffer_size, config.iterations, copy_time);
}

// Run reverse pattern benchmarks (backward sequential access)
static void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                           PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  double read_time = run_pattern_read_test(buffers.src_buffer(), config.buffer_size, config.iterations,
                                            memory_read_reverse_loop_asm, checksum, timer);
  results.reverse_read_bw = calculate_bandwidth(config.buffer_size, config.iterations, read_time);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  double write_time = run_pattern_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations,
                                             memory_write_reverse_loop_asm, timer);
  results.reverse_write_bw = calculate_bandwidth(config.buffer_size, config.iterations, write_time);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  double copy_time = run_pattern_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                            config.iterations, memory_copy_reverse_loop_asm, timer);
  results.reverse_copy_bw = calculate_bandwidth(config.buffer_size, config.iterations, copy_time);
}

// Run strided pattern benchmarks (access with specified stride)
static void run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                           size_t stride, double& read_bw, double& write_bw, double& copy_bw,
                                           HighResTimer& timer) {
  // Validate stride
  if (stride < 32) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return;
  }
  if (stride > config.buffer_size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, config.buffer_size) << std::endl;
    return;
  }
  
  // Ensure buffer is large enough for 32-byte accesses
  if (config.buffer_size < 32) {
    std::cerr << Messages::error_prefix() << "Buffer too small for strided access (minimum 32 bytes)" << std::endl;
    return;
  }
  
  // The strided assembly functions use modulo arithmetic: addr = base + (offset % byteCount)
  // Each access loads/stores 32 bytes, so we need to ensure addr + 32 <= buffer_size
  // Therefore, we pass (buffer_size - 32) as the effective buffer size for modulo calculation
  // This ensures addresses are always <= buffer_size - 32, and accessing 32 bytes stays within bounds
  size_t effective_buffer_size = config.buffer_size - 32;
  
  // Ensure effective buffer size is at least as large as stride
  // If not, the strided pattern won't work correctly
  if (effective_buffer_size < stride) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, config.buffer_size) << std::endl;
    return;
  }
  
  // Calculate actual data accessed for strided pattern
  // The strided loop accesses 32 bytes per iteration, advancing by stride each time
  // Number of iterations = ceil(effective_buffer_size / stride)
  // Actual data accessed = iterations * 32 bytes
  size_t num_iterations = (effective_buffer_size + stride - 1) / stride;  // Ceiling division
  size_t bytes_per_access = 32;  // Each iteration accesses 32 bytes (one cache line)
  size_t actual_data_accessed = num_iterations * bytes_per_access;
  
  // Ensure we have at least one iteration
  if (num_iterations == 0) {
    std::cerr << Messages::error_prefix() << "No iterations possible for strided pattern (buffer too small)" << std::endl;
    return;
  }
  
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_strided(buffers.src_buffer(), effective_buffer_size, stride, config.num_threads, checksum);
  double read_time = run_pattern_read_strided_test(buffers.src_buffer(), effective_buffer_size, stride,
                                                     config.iterations, checksum, timer);
  // For read: actual_data_accessed bytes are read per iteration
  read_bw = calculate_bandwidth(actual_data_accessed, config.iterations, read_time);
  
  show_progress();
  warmup_write_strided(buffers.dst_buffer(), effective_buffer_size, stride, config.num_threads);
  double write_time = run_pattern_write_strided_test(buffers.dst_buffer(), effective_buffer_size, stride,
                                                      config.iterations, timer);
  // For write: actual_data_accessed bytes are written per iteration
  write_bw = calculate_bandwidth(actual_data_accessed, config.iterations, write_time);
  
  show_progress();
  warmup_copy_strided(buffers.dst_buffer(), buffers.src_buffer(), effective_buffer_size, stride, config.num_threads);
  double copy_time = run_pattern_copy_strided_test(buffers.dst_buffer(), buffers.src_buffer(),
                                                    effective_buffer_size, stride, config.iterations, timer);
  // For copy: actual_data_accessed bytes are read + actual_data_accessed bytes are written per iteration
  copy_bw = calculate_bandwidth(actual_data_accessed * 2, config.iterations, copy_time);
}

// Run random pattern benchmarks (uniform random access)
static void run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                           const std::vector<size_t>& random_indices, size_t num_accesses,
                                           PatternResults& results, HighResTimer& timer) {
  // Validate indices
  if (random_indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return;
  }
  
  // Validate that indices are within buffer bounds and properly aligned
  // Each access loads/stores 32 bytes, so we need index + 32 <= buffer_size
  for (size_t i = 0; i < random_indices.size() && i < 100; ++i) {
    if (random_indices[i] + 32 > config.buffer_size) {
      std::cerr << Messages::error_prefix() << Messages::error_index_out_of_bounds(i, random_indices[i], config.buffer_size) << std::endl;
      return;
    }
    if (random_indices[i] % 32 != 0) {
      std::cerr << Messages::error_prefix() << Messages::error_index_not_aligned(i, random_indices[i]) << std::endl;
      return;
    }
  }
  
  // Use first min(10000, indices.size() / 10) indices for warmup
  size_t warmup_indices_count = std::min(static_cast<size_t>(10000), random_indices.size() / 10);
  if (warmup_indices_count == 0) warmup_indices_count = 1;  // At least one index
  std::vector<size_t> warmup_indices(random_indices.begin(), random_indices.begin() + warmup_indices_count);
  
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_random(buffers.src_buffer(), warmup_indices, config.num_threads, checksum);
  double read_time = run_pattern_read_random_test(buffers.src_buffer(), random_indices,
                                                   config.iterations, checksum, timer);
  // For random, we use num_accesses * 32 instead of buffer_size for bandwidth calculation
  results.random_read_bw = calculate_bandwidth(num_accesses * 32, config.iterations, read_time);
  
  show_progress();
  warmup_write_random(buffers.dst_buffer(), warmup_indices, config.num_threads);
  double write_time = run_pattern_write_random_test(buffers.dst_buffer(), random_indices,
                                                      config.iterations, timer);
  results.random_write_bw = calculate_bandwidth(num_accesses * 32, config.iterations, write_time);
  
  show_progress();
  warmup_copy_random(buffers.dst_buffer(), buffers.src_buffer(), warmup_indices, config.num_threads);
  double copy_time = run_pattern_copy_random_test(buffers.dst_buffer(), buffers.src_buffer(), random_indices,
                                                   config.iterations, timer);
  results.random_copy_bw = calculate_bandwidth(num_accesses * 32, config.iterations, copy_time);
}

int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternResults& results) {
  HighResTimer timer;
  
  // Calculate number of accesses for random pattern (use buffer_size / 32 to get reasonable number)
  size_t num_random_accesses = config.buffer_size / 32;
  if (num_random_accesses < 1000) num_random_accesses = 1000;
  if (num_random_accesses > 1000000) num_random_accesses = 1000000;
  
  // Generate random indices once
  std::vector<size_t> random_indices = generate_random_indices(config.buffer_size, num_random_accesses);
  
  std::cout << "\nRunning Pattern Benchmarks...\n" << std::flush;
  
  // Sequential Forward (baseline)
  run_forward_pattern_benchmarks(buffers, config, results, timer);
  
  // Sequential Reverse
  run_reverse_pattern_benchmarks(buffers, config, results, timer);
  
  // Strided (Cache Line - 64B)
  run_strided_pattern_benchmarks(buffers, config, 64, results.strided_64_read_bw,
                                  results.strided_64_write_bw, results.strided_64_copy_bw, timer);
  
  // Strided (Page - 4096B)
  run_strided_pattern_benchmarks(buffers, config, 4096, results.strided_4096_read_bw,
                                  results.strided_4096_write_bw, results.strided_4096_copy_bw, timer);
  
  // Random Uniform
  run_random_pattern_benchmarks(buffers, config, random_indices, num_random_accesses, results, timer);
  
  return EXIT_SUCCESS;
}

void print_pattern_results(const PatternResults& results) {
  std::cout << "\n================================\n\n";
  
  // Helper function to format percentage
  auto format_pct = [](double baseline, double value) -> std::string {
    if (baseline == 0.0) return "N/A";
    double pct = ((value - baseline) / baseline) * 100.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (pct >= 0) {
      oss << " (+" << pct << "%)";
    } else {
      oss << " (" << pct << "%)";
    }
    return oss.str();
  };
  
  // Sequential Forward (baseline)
  std::cout << "Sequential Forward:\n";
  std::cout << "  Read : " << std::fixed << std::setprecision(3) << results.forward_read_bw << " GB/s\n";
  std::cout << "  Write: " << results.forward_write_bw << " GB/s\n";
  std::cout << "  Copy : " << results.forward_copy_bw << " GB/s\n\n";
  
  // Sequential Reverse
  std::cout << "Sequential Reverse:\n";
  std::cout << "  Read : " << results.reverse_read_bw << " GB/s" 
            << format_pct(results.forward_read_bw, results.reverse_read_bw) << "\n";
  std::cout << "  Write: " << results.reverse_write_bw << " GB/s"
            << format_pct(results.forward_write_bw, results.reverse_write_bw) << "\n";
  std::cout << "  Copy : " << results.reverse_copy_bw << " GB/s"
            << format_pct(results.forward_copy_bw, results.reverse_copy_bw) << "\n\n";
  
  // Strided (Cache Line - 64B)
  std::cout << "Strided (Cache Line - 64B):\n";
  std::cout << "  Read : " << results.strided_64_read_bw << " GB/s"
            << format_pct(results.forward_read_bw, results.strided_64_read_bw) << "\n";
  std::cout << "  Write: " << results.strided_64_write_bw << " GB/s"
            << format_pct(results.forward_write_bw, results.strided_64_write_bw) << "\n";
  std::cout << "  Copy : " << results.strided_64_copy_bw << " GB/s"
            << format_pct(results.forward_copy_bw, results.strided_64_copy_bw) << "\n\n";
  
  // Strided (Page - 4096B)
  std::cout << "Strided (Page - 4096B):\n";
  std::cout << "  Read : " << results.strided_4096_read_bw << " GB/s"
            << format_pct(results.forward_read_bw, results.strided_4096_read_bw) << "\n";
  std::cout << "  Write: " << results.strided_4096_write_bw << " GB/s"
            << format_pct(results.forward_write_bw, results.strided_4096_write_bw) << "\n";
  std::cout << "  Copy : " << results.strided_4096_copy_bw << " GB/s"
            << format_pct(results.forward_copy_bw, results.strided_4096_copy_bw) << "\n\n";
  
  // Random Uniform
  std::cout << "Random Uniform:\n";
  std::cout << "  Read : " << results.random_read_bw << " GB/s"
            << format_pct(results.forward_read_bw, results.random_read_bw) << "\n";
  std::cout << "  Write: " << results.random_write_bw << " GB/s"
            << format_pct(results.forward_write_bw, results.random_write_bw) << "\n";
  std::cout << "  Copy : " << results.random_copy_bw << " GB/s"
            << format_pct(results.forward_copy_bw, results.random_copy_bw) << "\n\n";
  
  // Pattern Efficiency Analysis
  std::cout << "Pattern Efficiency Analysis:\n";
  
  // Sequential coherence: ratio of reverse to forward
  double seq_coherence = ((results.reverse_read_bw + results.reverse_write_bw + results.reverse_copy_bw) /
                          (results.forward_read_bw + results.forward_write_bw + results.forward_copy_bw)) * 100.0;
  std::cout << "- Sequential coherence: " << std::fixed << std::setprecision(1) << seq_coherence << "%\n";
  
  // Prefetcher effectiveness: ratio of strided 64B to forward (cache line stride should be well-prefetched)
  double prefetch_effectiveness = ((results.strided_64_read_bw + results.strided_64_write_bw + results.strided_64_copy_bw) /
                                    (results.forward_read_bw + results.forward_write_bw + results.forward_copy_bw)) * 100.0;
  std::cout << "- Prefetcher effectiveness: " << prefetch_effectiveness << "%\n";
  
  // Cache thrashing potential: based on strided 4096B performance (page stride causes more misses)
  double cache_thrashing = ((results.strided_4096_read_bw + results.strided_4096_write_bw + results.strided_4096_copy_bw) /
                             (results.forward_read_bw + results.forward_write_bw + results.forward_copy_bw)) * 100.0;
  std::string thrashing_level = (cache_thrashing > 70) ? "Low" : (cache_thrashing > 40) ? "Medium" : "High";
  std::cout << "- Cache thrashing potential: " << thrashing_level << "\n";
  
  // TLB pressure: based on random vs strided 4096B (random has more TLB misses)
  double tlb_pressure = ((results.random_read_bw + results.random_write_bw + results.random_copy_bw) /
                          (results.strided_4096_read_bw + results.strided_4096_write_bw + results.strided_4096_copy_bw)) * 100.0;
  std::string tlb_level = (tlb_pressure > 50) ? "Minimal" : (tlb_pressure > 20) ? "Moderate" : "High";
  std::cout << "- TLB pressure: " << tlb_level << "\n";
  
  std::cout << "\n";
}

