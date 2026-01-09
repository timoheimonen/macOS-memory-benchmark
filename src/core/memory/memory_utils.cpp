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
 * @file memory_utils.cpp
 * @brief Memory utility implementations
 *
 * Provides implementations for memory buffer initialization and latency chain setup.
 * Includes functions for creating pointer-chasing chains for latency measurements
 * and initializing source/destination buffers with test patterns.
 */

#include "core/memory/memory_utils.h"
#include "output/console/messages.h"
#include <vector>
#include <numeric>   // Needed for std::iota
#include <random>    // Needed for std::random_device, std::mt19937_64
#include <algorithm> // Needed for std::shuffle
#include <cstring>   // Needed for memset
#include <iostream>  // Needed for std::cout, std::cerr
#include <cstdlib>   // Needed for EXIT_SUCCESS, EXIT_FAILURE

/**
 * @brief Sets up a randomly shuffled pointer chain within the buffer for latency measurement.
 *
 * Creates a circular linked list of pointers within the buffer where each pointer points to
 * the next element in a randomly shuffled sequence. This ensures unpredictable memory access
 * patterns that defeat hardware prefetchers and measure true memory latency.
 *
 * The function:
 * 1. Calculates how many pointers can fit in the buffer based on the stride
 * 2. Creates a shuffled sequence of indices using std::shuffle
 * 3. Forms a circular chain by making each element point to the next in the shuffled sequence
 * 4. Performs bounds checking to prevent buffer overruns
 *
 * @param[in,out] buffer       The memory area to set up the chain in. Must be non-null.
 * @param[in]     buffer_size  Total size of the buffer in bytes. Must be >= stride * 2.
 * @param[in]     stride       The distance between consecutive pointer locations in bytes.
 *                             Must be >= sizeof(uintptr_t) and non-zero.
 *
 * @return EXIT_SUCCESS (0) on success
 * @return EXIT_FAILURE (1) if buffer is null, stride is zero, or buffer is too small
 *
 * @note The buffer must be large enough to hold at least 2 pointers at the given stride.
 * @note Uses std::random_device and std::mt19937_64 for high-quality randomization.
 * @note All pointer locations are bounds-checked to prevent buffer overruns.
 *
 * @see initialize_buffers() for buffer initialization
 */
int setup_latency_chain(void *buffer, size_t buffer_size, size_t stride)
{
    // Validate input parameters
    if (buffer == nullptr) {
        std::cerr << Messages::error_prefix() << Messages::error_buffer_pointer_null_latency_chain() << std::endl;
        return EXIT_FAILURE;
    }
    
    if (stride == 0) {
        std::cerr << Messages::error_prefix() << Messages::error_stride_zero_latency_chain() << std::endl;
        return EXIT_FAILURE;
    }
    
    // Calculate how many pointers fit in the buffer with the given stride.
    size_t num_pointers = buffer_size / stride;
    // Need at least two pointers to form a chain.
    if (num_pointers < 2)
    {
        std::cerr << Messages::error_prefix() << Messages::error_buffer_stride_invalid_latency_chain(num_pointers, buffer_size, stride) << std::endl;
        return EXIT_FAILURE;
    }

    // Create a vector to store the indices of pointer locations.
    std::vector<size_t> indices(num_pointers);
    // Fill indices vector with 0, 1, 2, ..., num_pointers - 1.
    std::iota(indices.begin(), indices.end(), 0);

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937_64 g(rd());
    // Randomly shuffle the indices.
    std::shuffle(indices.begin(), indices.end(), g);

    // Get a base pointer to the buffer.
    char *base_ptr = static_cast<char *>(buffer);
    
    // Calculate maximum valid offset to prevent buffer overrun
    size_t max_valid_offset = buffer_size - sizeof(uintptr_t);
    if (max_valid_offset > buffer_size) {  // Check for underflow
        std::cerr << Messages::error_prefix() << Messages::error_buffer_too_small_for_pointers() << std::endl;
        return EXIT_FAILURE;
    }

    // Create the linked list by linking shuffled indices.
    for (size_t i = 0; i < num_pointers; ++i)
    {
        // Calculate the memory address where the current pointer value will be stored.
        size_t current_offset = indices[i] * stride;
        
        // Bounds check: ensure we don't write beyond buffer
        if (current_offset + sizeof(uintptr_t) > buffer_size) {
            std::cerr << Messages::error_prefix() << Messages::error_offset_exceeds_bounds(current_offset, max_valid_offset) << std::endl;
            return EXIT_FAILURE;
        }
        
        uintptr_t *current_loc = (uintptr_t *)(base_ptr + current_offset);
        
        // Calculate the memory address of the *next* element in the shuffled sequence.
        // The modulo operator ensures the last element points back to the first.
        size_t next_index = (i + 1) % num_pointers;
        size_t next_offset = indices[next_index] * stride;
        
        // Bounds check: ensure next pointer is within buffer
        if (next_offset + sizeof(uintptr_t) > buffer_size) {
            std::cerr << Messages::error_prefix() << Messages::error_next_pointer_offset_exceeds_bounds(next_offset, max_valid_offset) << std::endl;
            return EXIT_FAILURE;
        }
        
        uintptr_t next_addr = (uintptr_t)(base_ptr + next_offset);
        
        // Write the address of the next element into the current location.
        *current_loc = next_addr;
    }
    
    return EXIT_SUCCESS;
}

/**
 * @brief Fills source and destination buffers with initial data.
 *
 * Initializes two buffers for bandwidth testing:
 * - Source buffer: Filled with a repeating pattern (0-255) to provide deterministic data
 * - Destination buffer: Zeroed using memset to ensure clean starting state
 *
 * This initialization ensures that bandwidth tests start with known buffer states and
 * can detect errors in memory operations by comparing expected vs actual values.
 *
 * @param[out] src_buffer   Pointer to the source buffer. Must be non-null.
 *                          Will be filled with pattern (i % 256) for each byte i.
 * @param[out] dst_buffer   Pointer to the destination buffer. Must be non-null.
 *                          Will be zeroed.
 * @param[in]  buffer_size  Size of each buffer in bytes. Must be non-zero.
 *
 * @return EXIT_SUCCESS (0) on success
 * @return EXIT_FAILURE (1) if either buffer is null or buffer_size is zero
 *
 * @note Both buffers must be pre-allocated to at least buffer_size bytes.
 * @note The source pattern (i % 256) creates a repeating 0-255 byte sequence.
 *
 * @see setup_latency_chain() for latency buffer setup
 */
int initialize_buffers(void *src_buffer, void *dst_buffer, size_t buffer_size)
{
    // Validate input parameters
    if (src_buffer == nullptr) {
        std::cerr << Messages::error_prefix() << Messages::error_source_buffer_null() << std::endl;
        return EXIT_FAILURE;
    }
    
    if (dst_buffer == nullptr) {
        std::cerr << Messages::error_prefix() << Messages::error_destination_buffer_null() << std::endl;
        return EXIT_FAILURE;
    }
    
    if (buffer_size == 0) {
        std::cerr << Messages::error_prefix() << Messages::error_buffer_size_zero_generic() << std::endl;
        return EXIT_FAILURE;
    }
    
    // Fill the source buffer with a repeating byte pattern (0-255).
    for (size_t i = 0; i < buffer_size; ++i)
    {
        static_cast<char *>(src_buffer)[i] = (char)(i % 256);
    }
    // Fill the destination buffer with zeros using memset.
    memset(dst_buffer, 0, buffer_size);
    
    return EXIT_SUCCESS;
}