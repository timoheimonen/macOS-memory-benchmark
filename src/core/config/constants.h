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
 * @file constants.h
 * @brief Central configuration constants for benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header defines all configuration constants used throughout the benchmark
 * suite, organized into logical categories for maintainability and clarity.
 *
 * Constant categories:
 * - Buffer size factors: Scaling factors for cache-level buffer sizing
 * - Memory limits: System memory usage constraints and fallbacks
 * - Cache size limits: Validation ranges for cache size specifications
 * - Size conversions: Unit conversion factors (KB, MB, cache lines)
 * - Latency test parameters: Stride values and buffer size requirements
 * - Access count defaults: Number of operations for latency measurements
 * - Execution parameters: Thread counts and iteration multipliers
 * - Default values: User-configurable parameter defaults
 * - Bandwidth calculations: Constants for throughput computation
 * - Output formatting: Display precision for various metric types
 * - Pattern benchmarks: Parameters for memory access pattern tests
 * - Cache fallback sizes: Default cache sizes when detection fails
 * - UI constants: Progress indicator settings
 *
 * All constants are defined within the Constants namespace to avoid naming
 * conflicts and provide clear semantic grouping.
 */
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstddef>  // size_t
#include <cstdint>  // uint32_t

/**
 * @namespace Constants
 * @brief Namespace containing all benchmark configuration constants
 *
 * This namespace provides centralized access to configuration constants used
 * throughout the benchmark suite. Constants are organized by functional area
 * and include comprehensive inline documentation explaining their purpose and
 * usage context.
 */
namespace Constants {
  // Buffer size factors
  constexpr double L1_BUFFER_SIZE_FACTOR = 1.0;  // Use 100% of L1 cache size
  constexpr double L2_BUFFER_SIZE_FACTOR = 1.0;  // Use 100% of L2 cache size
  
  // Memory limit constants
  constexpr double MEMORY_LIMIT_FACTOR = 0.80;  // Use 80% of available memory
  constexpr unsigned long FALLBACK_TOTAL_LIMIT_MB = 2048;  // Fallback if memory detection fails
  constexpr unsigned long MINIMUM_LIMIT_MB_PER_BUFFER = 64;  // Minimum buffer size limit
  
  // Cache size limits (in KB)
  constexpr long long MIN_CACHE_SIZE_KB = 16;  // Minimum cache size (page size on macOS)
  constexpr long long MAX_CACHE_SIZE_KB = 1048576;  // Maximum cache size (1 GB)
  
  // Size conversion constants
  constexpr size_t BYTES_PER_KB = 1024;
  constexpr size_t BYTES_PER_MB = 1024 * 1024;
  
  // Cache line alignment constant
  constexpr size_t CACHE_LINE_SIZE_BYTES = 64;  // Cache line size for alignment (64 bytes on Apple Silicon)
  
  // Latency test constants
  constexpr size_t LATENCY_STRIDE_BYTES = 256;  // Latency test access stride (8-byte aligned, DRAM-oriented default)
  constexpr size_t DEFAULT_LATENCY_TLB_LOCALITY_KB = 1024;  // Default locality window for latency chains (1 MB)
  static_assert((LATENCY_STRIDE_BYTES % sizeof(void*)) == 0,
                "LATENCY_STRIDE_BYTES must be pointer-size aligned for latency chain");
  
  // Default buffer size (MB) - used for scaling latency accesses
  constexpr unsigned long DEFAULT_BUFFER_SIZE_MB = 512;
  
  // Latency access count constants
  constexpr size_t BASE_LATENCY_ACCESSES = 200 * 1000 * 1000;  // Base accesses for 512MB buffer
  constexpr size_t L1_LATENCY_ACCESSES = 100 * 1000 * 1000;  // L1 cache latency test accesses
  constexpr size_t L2_LATENCY_ACCESSES = 50 * 1000 * 1000;  // L2 cache latency test accesses
  constexpr size_t CUSTOM_LATENCY_ACCESSES = 100 * 1000 * 1000;  // Custom cache latency test accesses
  
  // Benchmark execution constants
  constexpr int SINGLE_THREAD = 1;  // Single-threaded execution for cache tests
  
  // Default configuration values
  constexpr int DEFAULT_ITERATIONS = 1000;  // Initial calibration pilot/fallback; fixed only with --iterations
  constexpr int DEFAULT_LOOP_COUNT = 1;     // Default number of full benchmark loops
  constexpr int DEFAULT_LATENCY_SAMPLE_COUNT = 1000;  // Default number of latency samples per test
  // Shared automatic bandwidth/pass calibration policy. Mode-specific aliases
  // below keep ownership explicit at call sites and in serialized metadata.
  constexpr double BANDWIDTH_CALIBRATION_TARGET_SECONDS = 0.150;
  constexpr double BANDWIDTH_CALIBRATION_MIN_SECONDS = 0.100;
  constexpr double BANDWIDTH_CALIBRATION_MAX_SECONDS = 0.250;
  constexpr size_t BANDWIDTH_CALIBRATION_MAX_CORRECTIONS = 2;
  constexpr size_t BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES = 8 * BYTES_PER_MB;
  constexpr size_t BANDWIDTH_CALIBRATION_MAX_PASSES = 1000000000;
  constexpr double BENCHMARK_CALIBRATION_TARGET_SECONDS =
      BANDWIDTH_CALIBRATION_TARGET_SECONDS;
  constexpr double BENCHMARK_CALIBRATION_MIN_SECONDS =
      BANDWIDTH_CALIBRATION_MIN_SECONDS;
  constexpr double BENCHMARK_CALIBRATION_MAX_SECONDS =
      BANDWIDTH_CALIBRATION_MAX_SECONDS;
  constexpr size_t BENCHMARK_CALIBRATION_MAX_CORRECTIONS =
      BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;
  constexpr size_t BENCHMARK_CALIBRATION_MIN_PILOT_BYTES =
      BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES;
  constexpr size_t BENCHMARK_CALIBRATION_MAX_PASSES =
      BANDWIDTH_CALIBRATION_MAX_PASSES;

  // Standalone GPU memory-bandwidth mode. The calibration values are aliases
  // of the shared bandwidth policy; the dispatch, payload, and grid limits are
  // methodology guardrails serialized by GPU schema v1.
  constexpr unsigned long GPU_DEFAULT_BUFFER_SIZE_MB =
      DEFAULT_BUFFER_SIZE_MB;
  constexpr unsigned long GPU_MIN_BUFFER_SIZE_MB = 64;
  constexpr size_t GPU_DEFAULT_LOOP_COUNT = 3;
  constexpr double GPU_CALIBRATION_TARGET_SECONDS =
      BANDWIDTH_CALIBRATION_TARGET_SECONDS;
  constexpr double GPU_CALIBRATION_MIN_SECONDS =
      BANDWIDTH_CALIBRATION_MIN_SECONDS;
  constexpr double GPU_CALIBRATION_MAX_SECONDS =
      BANDWIDTH_CALIBRATION_MAX_SECONDS;
  constexpr size_t GPU_CALIBRATION_MAX_CORRECTIONS =
      BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;
  constexpr size_t GPU_CALIBRATION_MIN_PILOT_BYTES =
      BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES;
  constexpr size_t GPU_MAX_DISPATCHES_PER_MEASUREMENT = 16384;
  constexpr size_t GPU_MAX_EXACT_PAYLOAD_BYTES =
      64ULL * 1024ULL * BYTES_PER_MB;
  constexpr size_t GPU_VECTOR_WIDTH_BYTES = 16;
  constexpr size_t GPU_THREADS_PER_THREADGROUP_CAP = 256;
  constexpr size_t GPU_MAX_THREADGROUPS_PER_GRID = 8192;
  constexpr size_t GPU_AUXILIARY_BUFFER_BYTES = 4096;
  constexpr double GPU_STREAMING_CV_WARNING_PCT = 5.0;
  constexpr int GPU_JSON_SCHEMA_VERSION = 1;
  constexpr const char* GPU_JSON_MODE_NAME = "gpu_bandwidth";
  constexpr const char* GPU_METHODOLOGY_VERSION =
      "gpu-bandwidth-v1-private-runtime-single-cmdbuf-calibrated-balanced";
  constexpr const char* GPU_WORK_PLAN_IDENTITY_VERSION =
      "gpu-work-plan-v1";

  constexpr double BENCHMARK_LATENCY_TARGET_SECONDS = 0.250;
  constexpr double BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS = 0.100;
  constexpr double BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS = 0.300;
  constexpr size_t BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES = 16;
  constexpr size_t BENCHMARK_LATENCY_MAX_ACCESSES = 4000000000ULL;
  constexpr double BENCHMARK_CV_WARNING_PCT = 7.5;
  constexpr int BENCHMARK_JSON_SCHEMA_VERSION = 2;
  constexpr const char* BENCHMARK_METHODOLOGY_VERSION =
      "benchmark-v2-calibrated-seeded-balanced";
  constexpr size_t DEFAULT_SWEEP_MAX_RUNS = 256;  // Default guardrail for generated sweep combinations
  constexpr size_t DEFAULT_ANALYZE_TLB_SWEEP_MAX_RUNS = 16;  // Safer standalone TLB Cartesian sweep limit

  // Core-to-core standalone mode constants
  constexpr int CORE_TO_CORE_DEFAULT_LOOP_COUNT = 3;  // Default repeats provide a median and repeatability diagnostics
  constexpr int CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT = DEFAULT_LATENCY_SAMPLE_COUNT;  // Default sample count per loop for core-to-core mode
  constexpr size_t CORE_TO_CORE_WARMUP_ROUND_TRIPS = 20 * 1000;  // Minimum warmup handoff count
  constexpr size_t CORE_TO_CORE_HEADLINE_ROUND_TRIPS = 1 * 1000 * 1000;  // Minimum headline handoff count
  constexpr size_t CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS = 2 * 1000;  // Minimum handoffs per sample window
  constexpr size_t CORE_TO_CORE_CALIBRATION_WARMUP_ROUND_TRIPS = 1 * 1000 * 1000;  // Excluded pilot-pair warmup
  constexpr size_t CORE_TO_CORE_CALIBRATION_ROUND_TRIPS = 100 * 1000;  // Excluded pilot handoff count
  constexpr size_t CORE_TO_CORE_MAX_ROUND_TRIPS = 100 * 1000 * 1000;  // Calibration arithmetic guardrail
  constexpr double CORE_TO_CORE_WARMUP_TARGET_SECONDS = 0.025;  // Per-pair steady-state warmup target
  constexpr double CORE_TO_CORE_HEADLINE_TARGET_SECONDS = 0.250;  // Continuous headline target
  constexpr double CORE_TO_CORE_HEADLINE_MIN_SECONDS = 0.100;  // Intended headline duration lower bound
  constexpr double CORE_TO_CORE_HEADLINE_MAX_SECONDS = 0.300;  // Intended headline duration upper bound
  constexpr double CORE_TO_CORE_SAMPLE_TARGET_SECONDS = 0.001;  // Comparable sampled-window duration target
  constexpr double CORE_TO_CORE_CV_WARNING_PCT = 7.5;  // Repeatability diagnostic threshold
  constexpr int CORE_TO_CORE_JSON_SCHEMA_VERSION = 2;
  constexpr const char* CORE_TO_CORE_METHODOLOGY_VERSION =
      "core2core-v2-calibrated-balanced-auditable";
  constexpr int CORE_TO_CORE_READY_THREADS_TARGET = 2;  // Required thread count before benchmark start signal
  constexpr uint32_t CORE_TO_CORE_INITIATOR_TURN_VALUE = 0;  // Turn token value for initiator thread ownership
  constexpr uint32_t CORE_TO_CORE_RESPONDER_TURN_VALUE = 1;  // Turn token value for responder thread ownership
  constexpr int CORE_TO_CORE_AFFINITY_TAG_NONE = 0;  // No affinity tag requested for thread policy hint
  constexpr int CORE_TO_CORE_AFFINITY_TAG_PRIMARY = 1;  // Primary affinity tag used in hinted scenarios
  constexpr int CORE_TO_CORE_AFFINITY_TAG_SECONDARY = 2;  // Secondary affinity tag used for split-tag scenario
  constexpr bool CORE_TO_CORE_AFFINITY_HINT_DISABLED = false;  // Scenario does not request affinity hint
  constexpr bool CORE_TO_CORE_AFFINITY_HINT_ENABLED = true;  // Scenario requests affinity hint
  constexpr const char CORE_TO_CORE_SCENARIO_NO_AFFINITY[] = "no_affinity_hint";  // Scenario name for no affinity policy hints
  constexpr const char CORE_TO_CORE_SCENARIO_SAME_AFFINITY[] = "same_affinity_tag";  // Scenario name for matching affinity tags
  constexpr const char CORE_TO_CORE_SCENARIO_DIFFERENT_AFFINITY[] = "different_affinity_tags";  // Scenario name for different affinity tags
  constexpr const char BENCHMARK_JSON_MODE_NAME[] = "benchmark";  // Serialized mode identifier for standard benchmark JSON output
  constexpr const char PATTERNS_JSON_MODE_NAME[] = "patterns";  // Serialized mode identifier for pattern benchmark JSON output
  constexpr const char TLB_ANALYSIS_JSON_MODE_NAME[] = "analyze_tlb";  // Serialized mode identifier for standalone TLB analysis JSON output
  constexpr const char CORE_TO_CORE_JSON_MODE_NAME[] = "analyze_core2core";  // Serialized mode identifier in JSON output
  constexpr const char SWEEP_JSON_MODE_NAME[] = "sweep";  // Serialized mode identifier for sweep JSON output
  constexpr bool CORE_TO_CORE_JSON_HARD_PINNING_SUPPORTED = false;  // User-space hard core pinning is not available on macOS
  constexpr bool CORE_TO_CORE_JSON_AFFINITY_TAGS_ARE_HINTS = true;  // Affinity tags are scheduler hints, not strict binding
  
  // Bandwidth calculation constants
  constexpr double NANOSECONDS_PER_SECOND = 1e9;  // Conversion factor for GB/s calculations
  constexpr int COPY_OPERATION_MULTIPLIER = 2;  // Copy = read + write
  
  // Output formatting precision constants
  constexpr int BANDWIDTH_PRECISION = 5;  // Decimal places for bandwidth values (GB/s)
  constexpr int TIME_PRECISION = 5;  // Decimal places for time values (seconds)
  constexpr int LATENCY_PRECISION = 2;    // Decimal places for latency values (nanoseconds)
  
  // Pattern benchmark constants
  constexpr size_t PATTERN_ACCESS_SIZE_BYTES = 32;  // Bytes per access in pattern benchmarks (cache line alignment)
  constexpr size_t PATTERN_MIN_BUFFER_SIZE_BYTES = 32;  // Minimum buffer size for pattern benchmarks
  constexpr size_t PATTERN_STRIDE_CACHE_LINE = 64;  // Cache line stride (bytes)
  constexpr size_t PATTERN_STRIDE_PAGE = 4096;  // Page stride (bytes)
  constexpr size_t PATTERN_STRIDE_PAGE_16K = 16 * 1024;  // Apple Silicon page-size stride candidate (bytes)
  constexpr size_t PATTERN_STRIDE_SUPERPAGE_2MB = 2 * 1024 * 1024;  // 2 MiB virtual-address stride (bytes)
  constexpr double PATTERN_CALIBRATION_TARGET_SECONDS =
      BANDWIDTH_CALIBRATION_TARGET_SECONDS;
  constexpr double PATTERN_CALIBRATION_MIN_SECONDS =
      BANDWIDTH_CALIBRATION_MIN_SECONDS;
  constexpr double PATTERN_CALIBRATION_MAX_SECONDS =
      BANDWIDTH_CALIBRATION_MAX_SECONDS;
  constexpr size_t PATTERN_CALIBRATION_MAX_CORRECTIONS =
      BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;
  constexpr int PATTERN_JSON_SCHEMA_VERSION = 3;
  constexpr const char* PATTERN_METHODOLOGY_VERSION =
      "pattern-v2-phase-calibrated-seeded";
  constexpr size_t PATTERN_CALIBRATION_MIN_PILOT_BYTES =
      BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES;
  constexpr size_t PATTERN_CALIBRATION_MAX_PASSES =
      BANDWIDTH_CALIBRATION_MAX_PASSES;
  constexpr size_t PATTERN_RANDOM_ACCESS_MIN = 1000;  // Minimum number of random accesses
  constexpr size_t PATTERN_RANDOM_ACCESS_MAX = 1000000;  // Maximum number of random accesses
  constexpr double PATTERN_MIN_TIME_NS = 1e-9;  // Minimum time for bandwidth calculation (nanoseconds)
  constexpr int PATTERN_PERCENTAGE_PRECISION = 1;  // Decimal places for percentage values
  constexpr int PATTERN_BANDWIDTH_PRECISION = 3;  // Decimal places for bandwidth values in pattern results
  constexpr double PATTERN_STREAMING_CV_WARNING_PCT = 5.0;
  constexpr double PATTERN_SPARSE_CV_WARNING_PCT = 10.0;
  
  // Cache fallback sizes (bytes) - used when cache detection fails
  constexpr size_t L1_CACHE_FALLBACK_SIZE_BYTES = 128 * 1024;  // 128 KB - typical Apple Silicon P-core L1 size
  constexpr size_t L2_CACHE_M1_FALLBACK_SIZE_BYTES = 12 * 1024 * 1024;  // 12 MB - M1 L2 cache per P-core cluster
  constexpr size_t L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES = 16 * 1024 * 1024;  // 16 MB - M2/M3/M4/M5 L2 cache per P-core cluster
  constexpr size_t L2_CACHE_GENERIC_FALLBACK_SIZE_BYTES = 16 * 1024 * 1024;  // 16 MB - generic fallback L2 cache size
}

#endif // CONSTANTS_H
