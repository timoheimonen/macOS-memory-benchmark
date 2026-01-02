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
 * This header provides functions to allocate all benchmark buffers based on
 * configuration settings.
 */
#ifndef BUFFER_ALLOCATOR_H
#define BUFFER_ALLOCATOR_H

#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations to avoid including headers in header
struct BenchmarkConfig;
struct BenchmarkBuffers;

/**
 * @brief Allocate all buffers based on configuration
 * @param config Reference to benchmark configuration
 * @param[out] buffers Reference to BenchmarkBuffers structure to populate
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Allocates all required buffers for main memory, cache latency, and cache
 * bandwidth tests based on the configuration settings. Performs validation
 * and overflow checks before allocation.
 */
int allocate_all_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers);

#endif // BUFFER_ALLOCATOR_H

