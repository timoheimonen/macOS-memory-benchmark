// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file buffer_allocator.h
 * @brief Buffer allocation for benchmark memory allocations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides pattern-buffer allocation and peak-memory accounting.
 */
#ifndef BUFFER_ALLOCATOR_H
#define BUFFER_ALLOCATOR_H

#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations to avoid including headers in header
struct BenchmarkConfig;
struct PatternBuffers;

/**
 * @brief Allocate the two mappings shared by all pattern benchmark loops.
 * @param config Pattern benchmark configuration.
 * @param[in,out] buffers Destination owner, replaced only after both mappings
 *                       have been allocated successfully.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on validation, overflow,
 *         memory-limit, or mapping failure.
 *
 * Allocation is atomic from the caller's perspective: if either mapping
 * fails, any newly created mapping is released and `buffers` is unchanged.
 * `config.use_non_cacheable` selects the best-effort MADV_RANDOM path.
 */
int allocate_pattern_buffers(const BenchmarkConfig& config,
                             PatternBuffers& buffers);

/**
 * @brief Calculate peak concurrent bytes required based on configuration
 * @param config Reference to benchmark configuration
 * @param[out] total_memory_bytes Peak required bytes on success
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on validation/overflow/limit error
 *
 * Uses mode-aware accounting rules for phased benchmark execution and reports
 * the highest concurrent memory footprint expected during a run.
 */
int calculate_total_allocation_bytes(const BenchmarkConfig& config, size_t& total_memory_bytes);

#endif // BUFFER_ALLOCATOR_H
