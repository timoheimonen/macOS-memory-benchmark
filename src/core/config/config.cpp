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
#include "core/config/config.h"
#include "utils/benchmark.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>  // getpagesize
#include <sstream>
#include <cmath>     // std::numeric_limits for overflow checks

int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)
  long long requested_threads_ll = -1;  // User requested threads (-1 = none)
  
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
  int max_cores = get_total_logical_cores();
  config.num_threads = max_cores;  // Default: use all available cores
  
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
            throw std::out_of_range(Messages::error_iterations_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.iterations = static_cast<int>(val_ll);
          config.user_specified_iterations = true;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-iterations"));
      } else if (arg == "-buffersize") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range(Messages::error_buffersize_invalid(val_ll, std::numeric_limits<unsigned long>::max()));
          requested_buffer_size_mb_ll = val_ll;
          config.user_specified_buffersize = true;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-buffersize"));
      } else if (arg == "-count") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_count_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.loop_count = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument(Messages::error_missing_value("-count"));
      } else if (arg == "-latency-samples") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_latency_samples_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.latency_sample_count = static_cast<int>(val_ll);
          config.user_specified_latency_samples = true;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-latency-samples"));
      } else if (arg == "-cache-size") {
        // Already parsed in first pass, skip it and its value in second pass
        if (config.custom_cache_size_kb_ll != -1) {
          // Skip the value argument (already validated in first pass)
          if (++i >= argc) {
            // This shouldn't happen (first pass should have validated it), but handle defensively
            throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
          }
          // Silently skip - already parsed and validated in first pass
          continue;
        } else {
          // This shouldn't happen (first pass should have set it), but handle defensively
          if (++i < argc) {
            // Try to parse it now (fallback case)
            long long val_ll = std::stoll(argv[i]);
            if (val_ll <= 0 || val_ll < Constants::MIN_CACHE_SIZE_KB || val_ll > Constants::MAX_CACHE_SIZE_KB)
              throw std::out_of_range(Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB / 1024));
            config.custom_cache_size_kb_ll = val_ll;
          } else
            throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
        }
      } else if (arg == "-output") {
        if (++i < argc) {
          config.output_file = argv[i];
        } else
          throw std::invalid_argument(Messages::error_missing_value("-output"));
      } else if (arg == "-patterns") {
        config.run_patterns = true;
      } else if (arg == "-non-cacheable") {
        config.use_non_cacheable = true;
      } else if (arg == "-only-bandwidth") {
        config.only_bandwidth = true;
      } else if (arg == "-only-latency") {
        config.only_latency = true;
      } else if (arg == "-threads") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_threads_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          requested_threads_ll = val_ll;
        } else
          throw std::invalid_argument(Messages::error_missing_value("-threads"));
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

  // Set threads from user request or default
  if (requested_threads_ll != -1) {
    int requested_threads = static_cast<int>(requested_threads_ll);
    if (requested_threads > max_cores) {
      std::cerr << Messages::warning_prefix() << Messages::warning_threads_capped(requested_threads, max_cores) << std::endl;
      config.num_threads = max_cores;
    } else {
      config.num_threads = requested_threads;
    }
    config.user_specified_threads = true;
  }

  return EXIT_SUCCESS;
}

int validate_config(BenchmarkConfig& config) {
  // Validate mutually exclusive flags
  if (config.only_bandwidth && config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_incompatible_flags() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Validate flags with -patterns
  if (config.run_patterns && (config.only_bandwidth || config.only_latency)) {
    std::cerr << Messages::error_prefix() << Messages::error_only_flags_with_patterns() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Validate -only-bandwidth incompatibilities
  if (config.only_bandwidth) {
    if (config.use_custom_cache_size) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_cache_size() << std::endl;
      return EXIT_FAILURE;
    }
    if (config.user_specified_latency_samples) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_latency_samples() << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  // Validate -only-latency incompatibilities
  if (config.only_latency) {
    if (config.user_specified_iterations) {
      std::cerr << Messages::error_prefix() << Messages::error_only_latency_with_iterations() << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  // Calculate memory limit
  unsigned long available_mem_mb = get_available_memory_mb();
  unsigned long max_allowed_mb_per_buffer = 0;

  if (available_mem_mb > 0) {
    unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * Constants::MEMORY_LIMIT_FACTOR);
    max_allowed_mb_per_buffer = max_total_allowed_mb / 3;
  } else {
    std::cerr << Messages::warning_prefix() << Messages::warning_cannot_get_memory() << std::endl;
    max_allowed_mb_per_buffer = Constants::FALLBACK_TOTAL_LIMIT_MB / 3;
    std::cout << Messages::info_setting_max_fallback(max_allowed_mb_per_buffer) << std::endl;
  }
  
  if (max_allowed_mb_per_buffer < Constants::MINIMUM_LIMIT_MB_PER_BUFFER) {
    std::cout << Messages::info_calculated_max_less_than_min(max_allowed_mb_per_buffer, Constants::MINIMUM_LIMIT_MB_PER_BUFFER) << std::endl;
    max_allowed_mb_per_buffer = Constants::MINIMUM_LIMIT_MB_PER_BUFFER;
  }

  // Validate and cap buffer size (only if not latency-only, or if latency-only but buffer_size is needed)
  // For latency-only mode, we still need buffer_size for main memory latency test, but we use default
  if (!config.only_latency) {
    // For bandwidth tests, validate and cap buffer size
    if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
      std::cerr << Messages::warning_prefix() << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
      config.buffer_size_mb = max_allowed_mb_per_buffer;
    }
  } else {
    // For latency-only, we still need a buffer for main memory latency test
    // Use default if not set, but don't cap it (latency test uses less memory)
    if (config.buffer_size_mb == 0) {
      config.buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
    }
    if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
      std::cerr << Messages::warning_prefix() << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
      config.buffer_size_mb = max_allowed_mb_per_buffer;
    }
  }

  // Calculate final buffer size in bytes
  config.buffer_size = static_cast<size_t>(config.buffer_size_mb) * Constants::BYTES_PER_MB;

  // Sanity checks
  size_t page_size = getpagesize();
  
  if (config.buffer_size_mb > 0 && (config.buffer_size == 0 || config.buffer_size / Constants::BYTES_PER_MB != config.buffer_size_mb)) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_calculation(config.buffer_size_mb) << std::endl;
    return EXIT_FAILURE;
  }
  
  // For latency-only, we still need buffer_size for main memory latency test
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

