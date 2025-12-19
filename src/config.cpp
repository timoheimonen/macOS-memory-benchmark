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
#include "config.h"
#include "benchmark.h"
#include "constants.h"
#include "messages.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>  // getpagesize
#include <sstream>

int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)
  
  // First pass: parse -cache-size early (needed for cache size detection)
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-cache-size") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll < Constants::MIN_CACHE_SIZE_KB || val_ll > Constants::MAX_CACHE_SIZE_KB)
            throw std::out_of_range(Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB / 1024));
          config.custom_cache_size_kb_ll = val_ll;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
      }
    } catch (const std::invalid_argument &e) {
      std::cerr << Messages::error_prefix() << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    } catch (const std::out_of_range &e) {
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // Get system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  
  // Determine if custom cache size is being used
  config.use_custom_cache_size = (config.custom_cache_size_kb_ll != -1);
  
  // Get cache sizes
  if (config.use_custom_cache_size) {
    config.custom_cache_size_bytes = static_cast<size_t>(config.custom_cache_size_kb_ll) * Constants::BYTES_PER_KB;
  } else {
    config.l1_cache_size = get_l1_cache_size();
    config.l2_cache_size = get_l2_cache_size();
  }
  
  // Set default access counts from constants
  config.l1_num_accesses = Constants::L1_LATENCY_ACCESSES;
  config.l2_num_accesses = Constants::L2_LATENCY_ACCESSES;
  config.custom_num_accesses = Constants::CUSTOM_LATENCY_ACCESSES;

  // Second pass: parse all other arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-iterations") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_iterations_invalid());
          config.iterations = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument(Messages::error_missing_value("-iterations"));
      } else if (arg == "-buffersize") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range(Messages::error_buffersize_invalid());
          requested_buffer_size_mb_ll = val_ll;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-buffersize"));
      } else if (arg == "-count") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_count_invalid());
          config.loop_count = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument(Messages::error_missing_value("-count"));
      } else if (arg == "-latency-samples") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_latency_samples_invalid());
          config.latency_sample_count = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument(Messages::error_missing_value("-latency-samples"));
      } else if (arg == "-cache-size") {
        if (++i < argc) {
          // Already parsed in first pass, just skip the value
        } else
          throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
      } else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;  // Special return value for help
      } else {
        throw std::invalid_argument(Messages::error_unknown_option(arg));
      }
    } catch (const std::invalid_argument &e) {
      std::cerr << Messages::error_prefix() << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    } catch (const std::out_of_range &e) {
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // Set buffer size from user request or default
  if (requested_buffer_size_mb_ll != -1) {
    config.buffer_size_mb = static_cast<unsigned long>(requested_buffer_size_mb_ll);
  }
  // Otherwise use default (already set in struct)

  return EXIT_SUCCESS;
}

int validate_config(BenchmarkConfig& config) {
  // Calculate memory limit
  unsigned long available_mem_mb = get_available_memory_mb();
  unsigned long max_allowed_mb_per_buffer = 0;

  if (available_mem_mb > 0) {
    unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * Constants::MEMORY_LIMIT_FACTOR);
    max_allowed_mb_per_buffer = max_total_allowed_mb / 3;
  } else {
    std::cerr << Messages::warning_cannot_get_memory() << std::endl;
    max_allowed_mb_per_buffer = Constants::FALLBACK_TOTAL_LIMIT_MB / 3;
    std::cout << Messages::info_setting_max_fallback(max_allowed_mb_per_buffer) << std::endl;
  }
  
  if (max_allowed_mb_per_buffer < Constants::MINIMUM_LIMIT_MB_PER_BUFFER) {
    std::cout << Messages::info_calculated_max_less_than_min(max_allowed_mb_per_buffer, Constants::MINIMUM_LIMIT_MB_PER_BUFFER) << std::endl;
    max_allowed_mb_per_buffer = Constants::MINIMUM_LIMIT_MB_PER_BUFFER;
  }

  // Validate and cap buffer size
  if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
    std::cerr << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
    config.buffer_size_mb = max_allowed_mb_per_buffer;
  }

  // Calculate final buffer size in bytes
  config.buffer_size = static_cast<size_t>(config.buffer_size_mb) * Constants::BYTES_PER_MB;

  // Sanity checks
  size_t page_size = getpagesize();
  
  if (config.buffer_size_mb > 0 && (config.buffer_size == 0 || config.buffer_size / Constants::BYTES_PER_MB != config.buffer_size_mb)) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_calculation(config.buffer_size_mb) << std::endl;
    return EXIT_FAILURE;
  }
  
  if (config.buffer_size < page_size || config.buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_too_small(config.buffer_size) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void calculate_buffer_sizes(BenchmarkConfig& config) {
  size_t page_size_check = getpagesize();
  
  if (config.use_custom_cache_size) {
    // Use 100% of custom cache size
    config.custom_buffer_size = config.custom_cache_size_bytes;
    
    // Ensure buffer size is multiple of stride
    config.custom_buffer_size = ((config.custom_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.custom_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) config.custom_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    
    // Ensure buffer size is at least page size aligned
    if (config.custom_buffer_size < page_size_check) {
      size_t original_size_kb = config.custom_cache_size_bytes / Constants::BYTES_PER_KB;
      size_t rounded_size_kb = page_size_check / Constants::BYTES_PER_KB;
      if (original_size_kb < rounded_size_kb) {
        std::cout << Messages::info_custom_cache_rounded_up(original_size_kb, rounded_size_kb) << std::endl;
      }
      config.custom_buffer_size = page_size_check;
    }
  } else {
    // Use configured factors for L1 and L2 to ensure fits within target level
    config.l1_buffer_size = static_cast<size_t>(config.l1_cache_size * Constants::L1_BUFFER_SIZE_FACTOR);
    config.l2_buffer_size = static_cast<size_t>(config.l2_cache_size * Constants::L2_BUFFER_SIZE_FACTOR);
    
    // Ensure buffer sizes are multiples of stride
    config.l1_buffer_size = ((config.l1_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    config.l2_buffer_size = ((config.l2_buffer_size / Constants::LATENCY_STRIDE_BYTES) * Constants::LATENCY_STRIDE_BYTES);
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.l1_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) config.l1_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    if (config.l2_buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) config.l2_buffer_size = Constants::MIN_LATENCY_BUFFER_SIZE;
    
    // Ensure buffer sizes are at least page size aligned
    if (config.l1_buffer_size < page_size_check) config.l1_buffer_size = page_size_check;
    if (config.l2_buffer_size < page_size_check) config.l2_buffer_size = page_size_check;
  }
}

void calculate_access_counts(BenchmarkConfig& config) {
  // Scale latency accesses proportionally to buffer size
  config.lat_num_accesses = static_cast<size_t>(Constants::BASE_LATENCY_ACCESSES * (static_cast<double>(config.buffer_size_mb) / Constants::DEFAULT_BUFFER_SIZE_MB));
  
  // Cache latency test access counts are already set in struct defaults
  // (l1_num_accesses, l2_num_accesses, custom_num_accesses)
}

