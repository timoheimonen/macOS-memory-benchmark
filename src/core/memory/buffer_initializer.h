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
 * @file buffer_initializer.h
 * @brief Buffer initialization for benchmark memory allocations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides functions to initialize all benchmark buffers by
 * filling them with data and setting up latency chains.
 */
#ifndef BUFFER_INITIALIZER_H
#define BUFFER_INITIALIZER_H

#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations to avoid including headers in header
struct BenchmarkConfig;
struct BenchmarkBuffers;

/**
 * @brief Initialize all buffers (fill data and setup latency chains)
 * @param buffers Reference to BenchmarkBuffers structure
 * @param config Reference to benchmark configuration
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Initializes buffers by filling them with data and setting up pointer-chasing
 * chains for latency tests.
 */
int initialize_all_buffers(BenchmarkBuffers& buffers, const BenchmarkConfig& config);

#endif // BUFFER_INITIALIZER_H

