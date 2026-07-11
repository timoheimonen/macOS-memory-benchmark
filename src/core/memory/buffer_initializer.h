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
 * This header provides deterministic initialization for pattern mappings.
 */
#ifndef BUFFER_INITIALIZER_H
#define BUFFER_INITIALIZER_H

#include <cstddef>

struct PatternBuffers;

/**
 * @brief Initialize the pattern source and destination mappings.
 * @param buffers Allocated pattern mappings.
 * @param buffer_size Size of each mapping in bytes.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE if either mapping or the size
 *         is invalid.
 *
 * Fills the source with the deterministic byte pattern and zeroes the
 * destination. The function does not modify configuration state.
 */
int initialize_pattern_buffers(const PatternBuffers& buffers,
                               size_t buffer_size);

#endif // BUFFER_INITIALIZER_H
