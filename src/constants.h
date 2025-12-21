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
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstddef>  // size_t

// Memory and buffer constants
namespace Constants {
  // Buffer size factors
  constexpr double L1_BUFFER_SIZE_FACTOR = 0.75;  // Use 75% of L1 cache size
  constexpr double L2_BUFFER_SIZE_FACTOR = 0.10;  // Use 10% of L2 cache size
  constexpr double CUSTOM_BUFFER_SIZE_FACTOR = 1.0;  // Use 100% of custom cache size
  
  // Memory limit constants
  constexpr double MEMORY_LIMIT_FACTOR = 0.80;  // Use 80% of available memory
  constexpr unsigned long FALLBACK_TOTAL_LIMIT_MB = 2048;  // Fallback if memory detection fails
  constexpr unsigned long MINIMUM_LIMIT_MB_PER_BUFFER = 64;  // Minimum buffer size limit
  
  // Cache size limits (in KB)
  constexpr long long MIN_CACHE_SIZE_KB = 16;  // Minimum cache size (page size on macOS)
  constexpr long long MAX_CACHE_SIZE_KB = 524288;  // Maximum cache size (512 MB)
  
  // Size conversion constants
  constexpr size_t BYTES_PER_KB = 1024;
  constexpr size_t BYTES_PER_MB = 1024 * 1024;
  
  // Latency test constants
  constexpr size_t LATENCY_STRIDE_BYTES = 128;  // Latency test access stride
  constexpr size_t MIN_LATENCY_BUFFER_SIZE = LATENCY_STRIDE_BYTES * 2;  // Minimum size (2 pointers worth)
  
  // Default buffer size (MB) - used for scaling latency accesses
  constexpr unsigned long DEFAULT_BUFFER_SIZE_MB = 512;
  
  // Latency access count constants
  constexpr size_t BASE_LATENCY_ACCESSES = 200 * 1000 * 1000;  // Base accesses for 512MB buffer
  constexpr size_t L1_LATENCY_ACCESSES = 100 * 1000 * 1000;  // L1 cache latency test accesses
  constexpr size_t L2_LATENCY_ACCESSES = 50 * 1000 * 1000;  // L2 cache latency test accesses
  constexpr size_t CUSTOM_LATENCY_ACCESSES = 100 * 1000 * 1000;  // Custom cache latency test accesses
  
  // Benchmark execution constants
  constexpr int CACHE_ITERATIONS_MULTIPLIER = 10;  // Cache tests use 10x iterations for accuracy
  constexpr int SINGLE_THREAD = 1;  // Single-threaded execution for cache tests
  
  // Default configuration values
  constexpr int DEFAULT_ITERATIONS = 1000;  // Default number of iterations for R/W/Copy tests
  constexpr int DEFAULT_LOOP_COUNT = 1;     // Default number of full benchmark loops
  constexpr int DEFAULT_LATENCY_SAMPLE_COUNT = 1000;  // Default number of latency samples per test
  
  // Bandwidth calculation constants
  constexpr double NANOSECONDS_PER_SECOND = 1e9;  // Conversion factor for GB/s calculations
  constexpr int COPY_OPERATION_MULTIPLIER = 2;  // Copy = read + write
  
  // Output formatting precision constants
  constexpr int BANDWIDTH_PRECISION = 5;  // Decimal places for bandwidth values (GB/s)
  constexpr int TIME_PRECISION = 5;  // Decimal places for time values (seconds)
  
  // Pattern benchmark constants
  constexpr size_t PATTERN_ACCESS_SIZE_BYTES = 32;  // Bytes per access in pattern benchmarks (cache line alignment)
  constexpr size_t PATTERN_MIN_BUFFER_SIZE_BYTES = 32;  // Minimum buffer size for pattern benchmarks
  constexpr size_t PATTERN_STRIDE_CACHE_LINE = 64;  // Cache line stride (bytes)
  constexpr size_t PATTERN_STRIDE_PAGE = 4096;  // Page stride (bytes)
  constexpr size_t PATTERN_RANDOM_ACCESS_MIN = 1000;  // Minimum number of random accesses
  constexpr size_t PATTERN_RANDOM_ACCESS_MAX = 1000000;  // Maximum number of random accesses
  constexpr size_t PATTERN_WARMUP_INDICES_MAX = 10000;  // Maximum indices to use for warmup
  constexpr size_t PATTERN_WARMUP_INDICES_FRACTION = 10;  // Use 1/N of indices for warmup
  constexpr size_t PATTERN_VALIDATION_INDICES_LIMIT = 100;  // Maximum indices to validate
  constexpr double PATTERN_MIN_TIME_NS = 1e-9;  // Minimum time for bandwidth calculation (nanoseconds)
  constexpr double PATTERN_CACHE_THRASHING_HIGH_THRESHOLD = 70.0;  // Cache thrashing threshold (percentage)
  constexpr double PATTERN_CACHE_THRASHING_MEDIUM_THRESHOLD = 40.0;  // Cache thrashing medium threshold (percentage)
  constexpr double PATTERN_TLB_PRESSURE_MINIMAL_THRESHOLD = 50.0;  // TLB pressure minimal threshold (percentage)
  constexpr double PATTERN_TLB_PRESSURE_MODERATE_THRESHOLD = 20.0;  // TLB pressure moderate threshold (percentage)
  constexpr int PATTERN_PERCENTAGE_PRECISION = 1;  // Decimal places for percentage values
  constexpr int PATTERN_BANDWIDTH_PRECISION = 3;  // Decimal places for bandwidth values in pattern results
}

#endif // CONSTANTS_H

