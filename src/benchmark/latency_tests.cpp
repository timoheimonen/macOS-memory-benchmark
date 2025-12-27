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
#include <vector>                // For std::vector
#include <cstddef>               // For size_t

#include "benchmark/benchmark_tests.h"  // Function declarations
#include "core/timing/timer.h"  // HighResTimer
#include "asm/asm_functions.h"  // Assembly function declarations

// Executes the single-threaded memory latency benchmark.
// 'buffer': Pointer to the buffer containing the pointer chain.
// 'num_accesses': Total number of pointer dereferences to perform.
// 'timer': High-resolution timer for measuring execution time.
// 'latency_samples': Optional output vector to store per-sample latencies (if provided and sample_count > 0).
// 'sample_count': Number of samples to collect (if 0, uses single measurement).
// Returns: Total duration in nanoseconds.
double run_latency_test(void *buffer, size_t num_accesses, HighResTimer &timer,
                        std::vector<double> *latency_samples, int sample_count) {
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    size_t accesses_per_sample = num_accesses / sample_count;
    if (accesses_per_sample == 0) accesses_per_sample = 1;  // Ensure at least 1 access per sample
    
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

// Executes the single-threaded cache latency benchmark for a specific cache level.
// Uses the same pointer chasing methodology as the main memory latency test.
// 'buffer': Pointer to the buffer containing the pointer chain (should fit in target cache).
// 'buffer_size': Size of the buffer in bytes (for validation/future use).
// 'num_accesses': Total number of pointer dereferences to perform.
// 'timer': High-resolution timer for measuring execution time.
// 'latency_samples': Optional output vector to store per-sample latencies (if provided and sample_count > 0).
// 'sample_count': Number of samples to collect (if 0, uses single measurement).
// Returns: Total duration in nanoseconds.
double run_cache_latency_test(void *buffer, size_t buffer_size, size_t num_accesses, HighResTimer &timer,
                              std::vector<double> *latency_samples, int sample_count) {
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    size_t accesses_per_sample = num_accesses / sample_count;
    if (accesses_per_sample == 0) accesses_per_sample = 1;  // Ensure at least 1 access per sample
    
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

