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
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/constants.h"
#include <random>
#include <vector>
#include <algorithm>

// ============================================================================
// Utility Functions
// ============================================================================

// Helper function to calculate bandwidth from time and data size
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns) {
  // Avoid division by zero - use minimum time if time is too small
  // This handles cases where very fast operations complete in < 1ns due to timer resolution
  double effective_time_ns = (elapsed_time_ns < Constants::PATTERN_MIN_TIME_NS) 
                              ? Constants::PATTERN_MIN_TIME_NS 
                              : elapsed_time_ns;
  return (static_cast<double>(data_size) * iterations) / 
         (effective_time_ns * Constants::NANOSECONDS_PER_SECOND);
}

// Helper function to calculate maximum valid aligned offset
static size_t calculate_max_aligned_offset(size_t buffer_size) {
  using namespace Constants;
  if (buffer_size < PATTERN_MIN_BUFFER_SIZE_BYTES) {
    return 0;
  }
  // Round down (buffer_size - PATTERN_ACCESS_SIZE_BYTES) to alignment boundary
  return ((buffer_size - PATTERN_ACCESS_SIZE_BYTES) / PATTERN_ACCESS_SIZE_BYTES) * PATTERN_ACCESS_SIZE_BYTES;
}

// Helper function to generate random indices for random access pattern
std::vector<size_t> generate_random_indices(size_t buffer_size, size_t num_accesses) {
  using namespace Constants;
  std::vector<size_t> indices;
  indices.reserve(num_accesses);
  
  // Generate random offsets that are aligned (for aligned loads)
  // Each access loads PATTERN_ACCESS_SIZE_BYTES bytes, so we need offset + PATTERN_ACCESS_SIZE_BYTES <= buffer_size
  size_t max_offset = calculate_max_aligned_offset(buffer_size);
  if (max_offset == 0) {
    return indices;  // Return empty vector if buffer too small
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, max_offset / PATTERN_ACCESS_SIZE_BYTES);  // Index in units
  
  for (size_t i = 0; i < num_accesses; ++i) {
    size_t offset = dis(gen) * PATTERN_ACCESS_SIZE_BYTES;  // Convert to byte offset
    indices.push_back(offset);
  }
  
  return indices;
}

// Helper function to calculate number of random accesses based on buffer size
size_t calculate_num_random_accesses(size_t buffer_size) {
  using namespace Constants;
  size_t num_accesses = buffer_size / PATTERN_ACCESS_SIZE_BYTES;
  if (num_accesses < PATTERN_RANDOM_ACCESS_MIN) {
    num_accesses = PATTERN_RANDOM_ACCESS_MIN;
  }
  if (num_accesses > PATTERN_RANDOM_ACCESS_MAX) {
    num_accesses = PATTERN_RANDOM_ACCESS_MAX;
  }
  return num_accesses;
}

