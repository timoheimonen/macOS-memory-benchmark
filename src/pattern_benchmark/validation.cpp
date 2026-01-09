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
 * @file validation.cpp
 * @brief Validation functions for pattern benchmark parameters
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file provides validation functions for pattern benchmark parameters
 * including stride values and random access index arrays. Ensures that all
 * benchmark parameters meet safety and correctness requirements before
 * execution begins.
 *
 * Validates:
 * - Stride values are within acceptable bounds
 * - Random access indices are within buffer bounds
 * - Indices are properly aligned to access size requirements
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <iostream>
#include <algorithm>

// ============================================================================
// Validation Functions
// ============================================================================

// Validate stride parameters
bool validate_stride(size_t stride, size_t buffer_size) {
  using namespace Constants;
  if (stride < PATTERN_MIN_BUFFER_SIZE_BYTES) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return false;
  }
  if (stride > buffer_size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, buffer_size) << std::endl;
    return false;
  }
  if (buffer_size < PATTERN_MIN_BUFFER_SIZE_BYTES) {
    std::cerr << Messages::error_prefix() 
              << Messages::error_buffer_too_small_strided(PATTERN_MIN_BUFFER_SIZE_BYTES) << std::endl;
    return false;
  }
  return true;
}

// Validate random indices
bool validate_random_indices(const std::vector<size_t>& indices, size_t buffer_size) {
  using namespace Constants;
  if (indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return false;
  }
  
  // Validate that indices are within buffer bounds and properly aligned
  // Each access loads/stores PATTERN_ACCESS_SIZE_BYTES bytes
  size_t validation_limit = std::min(indices.size(), 
                                     static_cast<size_t>(PATTERN_VALIDATION_INDICES_LIMIT));
  for (size_t i = 0; i < validation_limit; ++i) {
    if (indices[i] + PATTERN_ACCESS_SIZE_BYTES > buffer_size) {
      std::cerr << Messages::error_prefix() 
                << Messages::error_index_out_of_bounds(i, indices[i], buffer_size) << std::endl;
      return false;
    }
    if (indices[i] % PATTERN_ACCESS_SIZE_BYTES != 0) {
      std::cerr << Messages::error_prefix() 
                << Messages::error_index_not_aligned(i, indices[i]) << std::endl;
      return false;
    }
  }
  return true;
}

