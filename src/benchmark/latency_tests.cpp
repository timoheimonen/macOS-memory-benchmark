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
 * @file latency_tests.cpp
 * @brief Latency test implementations
 *
 * Implements single-threaded latency benchmark functions for main memory and cache.
 * Uses pointer-chasing methodology to measure true memory access latency by defeating
 * hardware prefetchers through randomized access patterns.
 *
 * Key features:
 * - Single-threaded execution (latency is serial by nature)
 * - Pointer-chasing through randomized linked list
 * - Optional sample collection for statistical analysis
 * - Nanosecond-precision timing via HighResTimer
 *
 * Methodology:
 * - Buffer contains circular linked list with random ordering
 * - Each access depends on previous result (defeats prefetching)
 * - Latency = total_time / number_of_accesses
 */

#include <vector>                // For std::vector
#include <cstddef>               // For size_t

#include "benchmark/benchmark_tests.h"  // Function declarations
#include "core/timing/timer.h"  // HighResTimer
#include "asm/asm_functions.h"  // Assembly function declarations

/**
 * @brief Executes the single-threaded memory latency benchmark.
 *
 * Measures memory access latency using pointer-chasing through a randomized circular
 * linked list. Each access depends on the previous result, preventing hardware prefetching
 * and ensuring accurate latency measurement.
 *
 * Two modes of operation:
 * 1. Single measurement: One continuous pointer chase, returns total time
 * 2. Sample collection: Multiple smaller chases, collects per-sample latencies
 *
 * Sample collection benefits:
 * - Enables statistical analysis (mean, median, percentiles)
 * - Identifies outliers and variance
 * - Useful for understanding latency distribution
 *
 * @param[in]     buffer           Pointer to the buffer containing the pointer chain.
 *                                 Must be non-null and initialized by setup_latency_chain().
 * @param[in]     num_accesses     Total number of pointer dereferences to perform.
 * @param[in,out] timer            High-resolution timer for measuring execution time.
 * @param[out]    latency_samples  Optional output vector to store per-sample latencies.
 *                                 If nullptr or sample_count is 0, uses single measurement.
 * @param[in]     sample_count     Number of samples to collect (if 0, uses single measurement).
 *
 * @return Total duration in nanoseconds
 *
 * @note Returns 0.0 if num_accesses is 0 (early validation).
 * @note Uses assembly function memory_latency_chase_asm() for pointer chasing.
 * @note Buffer must be initialized with setup_latency_chain() before calling.
 * @note Average latency = total_duration / num_accesses.
 *
 * @see run_cache_latency_test() for cache-specific latency measurement
 * @see memory_latency_chase_asm() for the low-level pointer chase implementation
 * @see setup_latency_chain() for buffer initialization
 */
double run_latency_test(void *buffer, size_t num_accesses, HighResTimer &timer,
                        std::vector<double> *latency_samples, int sample_count) {
  // Early validation: return 0 if no accesses requested
  if (num_accesses == 0) {
    return 0.0;
  }
  
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    // Calculate accesses per sample, ensuring at least 1 access per sample
    size_t accesses_per_sample = (num_accesses >= static_cast<size_t>(sample_count)) 
                                  ? (num_accesses / static_cast<size_t>(sample_count))
                                  : 1;
    
    double total_duration_ns = 0.0;
    
    for (int i = 0; i < sample_count; ++i) {
      timer.start();
      memory_latency_chase_asm(lat_start_ptr, accesses_per_sample);
      double sample_duration_ns = timer.stop_ns();
      double sample_latency_ns = sample_duration_ns / static_cast<double>(accesses_per_sample);
      latency_samples->push_back(sample_latency_ns);
      total_duration_ns += sample_duration_ns;
    }
    
    return total_duration_ns;
  } else {
    // Original single measurement behavior
    timer.start();  // Start timing.
    // Call external assembly function to chase the pointer chain.
    memory_latency_chase_asm(lat_start_ptr, num_accesses);
    // Stop timing, getting result in nanoseconds for latency.
    double duration_ns = timer.stop_ns();
    return duration_ns;  // Return total time elapsed in nanoseconds.
  }
}

/**
 * @brief Executes the single-threaded cache latency benchmark for a specific cache level.
 *
 * Measures cache access latency using the same pointer-chasing methodology as the main
 * memory latency test. The buffer size should be chosen to fit within the target cache
 * level (L1, L2, or custom) for accurate cache-specific latency measurement.
 *
 * Cache-specific considerations:
 * - Buffer size determines which cache level is measured
 * - L1 buffer: Should fit in L1 data cache (typically 32-64KB)
 * - L2 buffer: Should fit in L2 cache but exceed L1 (typically 256KB-1MB)
 * - Random access pattern ensures cache hits/misses are measured accurately
 *
 * Identical methodology to run_latency_test():
 * - Same pointer-chasing algorithm
 * - Same sample collection options
 * - Uses same assembly function
 *
 * @param[in]     buffer           Pointer to the buffer containing the pointer chain.
 *                                 Should fit in target cache level.
 * @param[in]     buffer_size      Size of the buffer in bytes (for validation/future use).
 * @param[in]     num_accesses     Total number of pointer dereferences to perform.
 * @param[in,out] timer            High-resolution timer for measuring execution time.
 * @param[out]    latency_samples  Optional output vector to store per-sample latencies.
 * @param[in]     sample_count     Number of samples to collect (if 0, uses single measurement).
 *
 * @return Total duration in nanoseconds
 *
 * @note Returns 0.0 if num_accesses is 0 (early validation).
 * @note Buffer must be initialized with setup_latency_chain() before calling.
 * @note Buffer size should match target cache level for accurate results.
 * @note Uses the same memory_latency_chase_asm() as main memory latency test.
 *
 * @see run_latency_test() for main memory latency measurement
 * @see memory_latency_chase_asm() for the low-level pointer chase implementation
 * @see setup_latency_chain() for buffer initialization
 */
double run_cache_latency_test(void *buffer, size_t buffer_size, size_t num_accesses, HighResTimer &timer,
                              std::vector<double> *latency_samples, int sample_count) {
  // Early validation: return 0 if no accesses requested
  if (num_accesses == 0) {
    return 0.0;
  }
  
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    // Calculate accesses per sample, ensuring at least 1 access per sample
    size_t accesses_per_sample = (num_accesses >= static_cast<size_t>(sample_count)) 
                                  ? (num_accesses / static_cast<size_t>(sample_count))
                                  : 1;
    
    double total_duration_ns = 0.0;
    
    for (int i = 0; i < sample_count; ++i) {
      timer.start();
      memory_latency_chase_asm(lat_start_ptr, accesses_per_sample);
      double sample_duration_ns = timer.stop_ns();
      double sample_latency_ns = sample_duration_ns / static_cast<double>(accesses_per_sample);
      latency_samples->push_back(sample_latency_ns);
      total_duration_ns += sample_duration_ns;
    }
    
    return total_duration_ns;
  } else {
    // Original single measurement behavior
    timer.start();  // Start timing.
    // Call external assembly function to chase the pointer chain (same as main latency test).
    memory_latency_chase_asm(lat_start_ptr, num_accesses);
    // Stop timing, getting result in nanoseconds for latency.
    double duration_ns = timer.stop_ns();
    return duration_ns;  // Return total time elapsed in nanoseconds.
  }
}

