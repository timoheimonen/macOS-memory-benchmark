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
#include <string>
#include <numeric>   // Needed for std::iota
#include <random>    // Needed for std::random_device, std::mt19937_64
#include <algorithm> // Needed for std::shuffle
#include <cstring>   // Needed for memset
#include <cctype>
#include <iostream>  // Needed for std::cout, std::cerr
#include <cstdlib>   // Needed for EXIT_SUCCESS, EXIT_FAILURE
#include <unistd.h>  // getpagesize

namespace {

std::string normalize_mode_token(const std::string& value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char c : value) {
    if (c == '-' || c == ' ') {
      normalized.push_back('_');
      continue;
    }
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return normalized;
}

void append_same_random_in_box(std::vector<size_t>& out_indices,
                               size_t locality_id,
                               size_t locality_pointer_span,
                               size_t num_pointers,
                               const std::vector<size_t>& in_box_permutation) {
  const size_t start = locality_id * locality_pointer_span;
  const size_t end = std::min(start + locality_pointer_span, num_pointers);
  for (size_t offset : in_box_permutation) {
    const size_t index = start + offset;
    if (index < end) {
      out_indices.push_back(index);
    }
  }
}

void append_diff_random_in_box(std::vector<size_t>& out_indices,
                               size_t locality_id,
                               size_t locality_pointer_span,
                               size_t num_pointers,
                               std::mt19937_64& rng) {
  const size_t start = locality_id * locality_pointer_span;
  const size_t end = std::min(start + locality_pointer_span, num_pointers);
  std::vector<size_t> locality_indices(end - start);
  std::iota(locality_indices.begin(), locality_indices.end(), start);
  std::shuffle(locality_indices.begin(), locality_indices.end(), rng);
  out_indices.insert(out_indices.end(), locality_indices.begin(), locality_indices.end());
}

void reorder_indices_with_mode(std::vector<size_t>& indices,
                               size_t locality_pointer_span,
                               LatencyChainMode mode,
                               std::mt19937_64& rng) {
  const size_t num_pointers = indices.size();
  const size_t locality_count = (num_pointers + locality_pointer_span - 1) / locality_pointer_span;

  std::vector<size_t> locality_order(locality_count);
  std::iota(locality_order.begin(), locality_order.end(), 0);

  if (mode == LatencyChainMode::RandomInBoxRandomBox) {
    std::shuffle(locality_order.begin(), locality_order.end(), rng);
  }

  std::vector<size_t> reordered_indices;
  reordered_indices.reserve(num_pointers);

  std::vector<size_t> shared_permutation;
  if (mode == LatencyChainMode::SameRandomInBoxIncreasingBox) {
    shared_permutation.resize(locality_pointer_span);
    std::iota(shared_permutation.begin(), shared_permutation.end(), 0);
    std::shuffle(shared_permutation.begin(), shared_permutation.end(), rng);
  }

  for (size_t locality_id : locality_order) {
    if (mode == LatencyChainMode::SameRandomInBoxIncreasingBox) {
      append_same_random_in_box(reordered_indices,
                                locality_id,
                                locality_pointer_span,
                                num_pointers,
                                shared_permutation);
      continue;
    }

    append_diff_random_in_box(reordered_indices,
                              locality_id,
                              locality_pointer_span,
                              num_pointers,
                              rng);
  }

  indices.swap(reordered_indices);
}

}  // namespace

const char* latency_chain_mode_to_string(LatencyChainMode mode) {
  switch (mode) {
    case LatencyChainMode::Auto:
      return "auto";
    case LatencyChainMode::GlobalRandom:
      return "global-random";
    case LatencyChainMode::RandomInBoxRandomBox:
      return "random-box";
    case LatencyChainMode::SameRandomInBoxIncreasingBox:
      return "same-random-in-box";
    case LatencyChainMode::DiffRandomInBoxIncreasingBox:
      return "diff-random-in-box";
  }

  return "auto";
}

bool latency_chain_mode_from_string(const std::string& mode_value, LatencyChainMode& out_mode) {
  const std::string normalized = normalize_mode_token(mode_value);

  if (normalized == "auto") {
    out_mode = LatencyChainMode::Auto;
    return true;
  }
  if (normalized == "global" || normalized == "global_random") {
    out_mode = LatencyChainMode::GlobalRandom;
    return true;
  }
  if (normalized == "random_box" || normalized == "random_in_box_random_box") {
    out_mode = LatencyChainMode::RandomInBoxRandomBox;
    return true;
  }
  if (normalized == "same_random_in_box" ||
      normalized == "same_random_in_box_increasing_box") {
    out_mode = LatencyChainMode::SameRandomInBoxIncreasingBox;
    return true;
  }
  if (normalized == "diff_random_in_box" ||
      normalized == "diff_random_in_box_increasing_box") {
    out_mode = LatencyChainMode::DiffRandomInBoxIncreasingBox;
    return true;
  }

  return false;
}

LatencyChainMode resolve_latency_chain_mode(LatencyChainMode mode, size_t tlb_locality_bytes) {
  if (mode != LatencyChainMode::Auto) {
    return mode;
  }

  return (tlb_locality_bytes == 0) ? LatencyChainMode::GlobalRandom
                                   : LatencyChainMode::RandomInBoxRandomBox;
}

bool latency_chain_mode_uses_locality(LatencyChainMode mode) {
  switch (mode) {
    case LatencyChainMode::GlobalRandom:
      return false;
    case LatencyChainMode::Auto:
    case LatencyChainMode::RandomInBoxRandomBox:
    case LatencyChainMode::SameRandomInBoxIncreasingBox:
    case LatencyChainMode::DiffRandomInBoxIncreasingBox:
      return true;
  }

  return true;
}

/**
 * @brief Sets up a randomly shuffled pointer chain within the buffer for latency measurement.
 *
 * Creates a circular linked list of pointers within the buffer where each pointer points to
 * the next element in a randomly shuffled sequence. This ensures unpredictable memory access
 * patterns that defeat hardware prefetchers and measure true memory latency.
 *
 * The function:
 * 1. Calculates how many pointers can fit in the buffer based on the stride
 * 2. Creates a shuffled sequence of indices using std::shuffle (globally or TLB-locality aware)
 * 3. Forms a circular chain by making each element point to the next in the shuffled sequence
 * 4. Performs bounds checking to prevent buffer overruns
 *
 * @param[in,out] buffer       The memory area to set up the chain in. Must be non-null.
 * @param[in]     buffer_size  Total size of the buffer in bytes. Must be >= stride * 2.
 * @param[in]     stride       The distance between consecutive pointer locations in bytes.
 *                             Must be >= sizeof(uintptr_t) and non-zero.
 * @param[in]     tlb_locality_bytes Optional locality window in bytes.
 *                             0 = global random chain. >0 = random within local windows,
 *                             then randomized window order.
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
int setup_latency_chain(void *buffer, size_t buffer_size, size_t stride,
                        size_t tlb_locality_bytes,
                        LatencyChainDiagnostics* diagnostics,
                        LatencyChainMode mode)
{
    if (diagnostics != nullptr) {
        diagnostics->pointer_count = 0;
        diagnostics->unique_pages_touched = 0;
        diagnostics->page_size_bytes = 0;
        diagnostics->stride_bytes = stride;
    }

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

    const int page_size_raw = getpagesize();
    const size_t page_size = (page_size_raw > 0) ? static_cast<size_t>(page_size_raw) : 0;
    const bool collect_page_diagnostics = (diagnostics != nullptr && page_size > 0);
    std::vector<uint8_t> page_seen;
    size_t unique_pages_touched = 0;
    size_t base_page_offset = 0;
    if (collect_page_diagnostics) {
        base_page_offset = reinterpret_cast<uintptr_t>(buffer) % page_size;
        const size_t span_with_offset = base_page_offset + buffer_size;
        const size_t page_count = (span_with_offset + page_size - 1) / page_size;
        page_seen.assign(page_count, 0);
    }

    const LatencyChainMode effective_mode = resolve_latency_chain_mode(mode, tlb_locality_bytes);

    if (latency_chain_mode_uses_locality(effective_mode) && tlb_locality_bytes == 0) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_latency_chain_mode_requires_locality(
                         latency_chain_mode_to_string(effective_mode))
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Create a vector to store the indices of pointer locations.
    std::vector<size_t> indices(num_pointers);
    // Fill indices vector with 0, 1, 2, ..., num_pointers - 1.
    std::iota(indices.begin(), indices.end(), 0);

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937_64 g(rd());
    if (effective_mode == LatencyChainMode::GlobalRandom) {
        // Randomly shuffle the full index space.
        std::shuffle(indices.begin(), indices.end(), g);
    } else {
        const size_t locality_pointer_span = tlb_locality_bytes / stride;
        if (locality_pointer_span < 2) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_buffer_stride_invalid_latency_chain(locality_pointer_span, tlb_locality_bytes, stride)
                      << std::endl;
            return EXIT_FAILURE;
        }

        reorder_indices_with_mode(indices, locality_pointer_span, effective_mode, g);
    }

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

        if (collect_page_diagnostics) {
            const size_t page_index = (base_page_offset + current_offset) / page_size;
            if (page_index >= page_seen.size()) {
                std::cerr << Messages::error_prefix() << Messages::error_offset_exceeds_bounds(page_index, page_seen.size() - 1) << std::endl;
                return EXIT_FAILURE;
            }
            if (page_seen[page_index] == 0) {
                page_seen[page_index] = 1;
                ++unique_pages_touched;
            }
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

    if (diagnostics != nullptr) {
        diagnostics->pointer_count = num_pointers;
        diagnostics->unique_pages_touched = unique_pages_touched;
        diagnostics->page_size_bytes = page_size;
        diagnostics->stride_bytes = stride;
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
