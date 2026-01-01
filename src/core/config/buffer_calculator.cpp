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
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <iostream>
#include <unistd.h>  // getpagesize
#include <limits>
#include <cmath>

void calculate_buffer_sizes(BenchmarkConfig& config) {
  size_t page_size_check = getpagesize();
  
  if (config.use_custom_cache_size) {
    // Use 100% of custom cache size
    config.custom_buffer_size = config.custom_cache_size_bytes;
    
    // Validate that stride rounding won't cause issues
    // Note: Division truncation is intentional here - we round down to nearest stride multiple
    // This ensures the buffer fits within the cache size while maintaining stride alignment
    if (config.custom_buffer_size >= Constants::LATENCY_STRIDE_BYTES) {
      config.custom_buffer_size = ((config.custom_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    } else {
      // If buffer is smaller than stride, set to minimum size
      config.custom_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.custom_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) {
      config.custom_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    
    // Ensure buffer size is at least page size aligned
    if (config.custom_buffer_size < page_size_check) {
      size_t original_size_kb = config.custom_cache_size_bytes / Constants::BYTES_PER_KB;
      size_t rounded_size_kb = page_size_check / Constants::BYTES_PER_KB;
      if (original_size_kb < rounded_size_kb) {
        std::cout << Messages::info_custom_cache_rounded_up(original_size_kb, rounded_size_kb) << std::endl;
      }
      config.custom_buffer_size = page_size_check;
    }
    
    // Final validation: ensure buffer size is not zero after all calculations
    if (config.custom_buffer_size == 0) {
      std::cerr << Messages::error_prefix() << Messages::error_calculated_custom_buffer_size_zero() << std::endl;
      config.custom_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
  } else {
    // Use configured factors for L1 and L2 to ensure fits within target level
    // Check for overflow before multiplication
    if (config.l1_cache_size > 0 && 
        config.l1_cache_size > std::numeric_limits<size_t>::max() / Constants::L1_BUFFER_SIZE_FACTOR) {
      std::cerr << Messages::error_prefix() << Messages::error_l1_cache_size_overflow() << std::endl;
      config.l1_buffer_size = std::numeric_limits<size_t>::max();
    } else {
      config.l1_buffer_size = static_cast<size_t>(config.l1_cache_size * Constants::L1_BUFFER_SIZE_FACTOR);
    }
    
    if (config.l2_cache_size > 0 && 
        config.l2_cache_size > std::numeric_limits<size_t>::max() / Constants::L2_BUFFER_SIZE_FACTOR) {
      std::cerr << Messages::error_prefix() << Messages::error_l2_cache_size_overflow() << std::endl;
      config.l2_buffer_size = std::numeric_limits<size_t>::max();
    } else {
      config.l2_buffer_size = static_cast<size_t>(config.l2_cache_size * Constants::L2_BUFFER_SIZE_FACTOR);
    }
    
    // Ensure buffer sizes are multiples of stride (with validation)
    // Note: Division truncation is intentional - rounds down to nearest stride multiple
    if (config.l1_buffer_size >= Constants::LATENCY_STRIDE_BYTES) {
      config.l1_buffer_size = ((config.l1_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    } else {
      config.l1_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    
    if (config.l2_buffer_size >= Constants::LATENCY_STRIDE_BYTES) {
      config.l2_buffer_size = ((config.l2_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    } else {
      config.l2_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.l1_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) {
      config.l1_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    if (config.l2_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) {
      config.l2_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    
    // Ensure buffer sizes are at least page size aligned
    if (config.l1_buffer_size < page_size_check) {
      config.l1_buffer_size = page_size_check;
    }
    if (config.l2_buffer_size < page_size_check) {
      config.l2_buffer_size = page_size_check;
    }
    
    // Final validation: ensure buffer sizes are not zero after all calculations
    if (config.l1_buffer_size == 0) {
      std::cerr << Messages::error_prefix() << Messages::error_calculated_l1_buffer_size_zero() << std::endl;
      config.l1_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
    if (config.l2_buffer_size == 0) {
      std::cerr << Messages::error_prefix() << Messages::error_calculated_l2_buffer_size_zero() << std::endl;
      config.l2_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    }
  }
}

void calculate_access_counts(BenchmarkConfig& config) {
  // Scale latency accesses proportionally to buffer size
  // Use floating-point arithmetic for precision, but validate result fits in size_t
  double scale_factor = static_cast<double>(config.buffer_size_mb) / Constants::DEFAULT_BUFFER_SIZE_MB;
  double scaled_accesses = Constants::BASE_LATENCY_ACCESSES * scale_factor;
  
  // Check for overflow before casting
  if (scaled_accesses > static_cast<double>(std::numeric_limits<size_t>::max())) {
    std::cerr << Messages::error_prefix() << Messages::error_latency_access_count_overflow() << std::endl;
    config.lat_num_accesses = std::numeric_limits<size_t>::max();
  } else if (scaled_accesses < 0) {
    std::cerr << Messages::error_prefix() << Messages::error_latency_access_count_negative() << std::endl;
    config.lat_num_accesses = Constants::BASE_LATENCY_ACCESSES;  // Use default
  } else {
    config.lat_num_accesses = static_cast<size_t>(scaled_accesses);
  }
  
  // Ensure minimum access count
  if (config.lat_num_accesses == 0) {
    config.lat_num_accesses = Constants::BASE_LATENCY_ACCESSES;
  }
  
  // Cache latency test access counts are already set in struct defaults
  // (l1_num_accesses, l2_num_accesses, custom_num_accesses)
}

