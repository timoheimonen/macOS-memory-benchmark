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
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>  // getpagesize

int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)
  
  // First pass: parse -cache-size early (needed for cache size detection)
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-cache-size") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          const long long min_cache_size_kb = 16;  // Minimum is page size (16 KB on macOS)
          if (val_ll <= 0 || val_ll < min_cache_size_kb || val_ll > 524288)
            throw std::out_of_range("cache-size invalid (must be between " + std::to_string(min_cache_size_kb) + " KB and 524288 KB (512 MB))");
          config.custom_cache_size_kb_ll = val_ll;
        } else
          throw std::invalid_argument("Missing value for -cache-size");
      }
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    } catch (const std::out_of_range &e) {
      std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
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
    config.custom_cache_size_bytes = static_cast<size_t>(config.custom_cache_size_kb_ll) * 1024;
  } else {
    config.l1_cache_size = get_l1_cache_size();
    config.l2_cache_size = get_l2_cache_size();
  }

  // Second pass: parse all other arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-iterations") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range("iterations invalid");
          config.iterations = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument("Missing value for -iterations");
      } else if (arg == "-buffersize") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range("buffersize invalid");
          requested_buffer_size_mb_ll = val_ll;
        } else
          throw std::invalid_argument("Missing value for -buffersize");
      } else if (arg == "-count") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range("count invalid");
          config.loop_count = static_cast<int>(val_ll);
        } else
          throw std::invalid_argument("Missing value for -count");
      } else if (arg == "-cache-size") {
        if (++i < argc) {
          // Already parsed in first pass, just skip the value
        } else
          throw std::invalid_argument("Missing value for -cache-size");
      } else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;  // Special return value for help
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    } catch (const std::out_of_range &e) {
      std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << " (" << e.what() << ")" << std::endl;
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
  const double memory_limit_factor = 0.80;
  const unsigned long fallback_total_limit_mb = 2048;
  const unsigned long minimum_limit_mb_per_buffer = 64;

  if (available_mem_mb > 0) {
    unsigned long max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * memory_limit_factor);
    max_allowed_mb_per_buffer = max_total_allowed_mb / 3;
  } else {
    std::cerr << "Warning: Cannot get available memory. Using fallback limit." << std::endl;
    max_allowed_mb_per_buffer = fallback_total_limit_mb / 3;
    std::cout << "Info: Setting max per buffer to fallback: " << max_allowed_mb_per_buffer << " MB." << std::endl;
  }
  
  if (max_allowed_mb_per_buffer < minimum_limit_mb_per_buffer) {
    std::cout << "Info: Calculated max (" << max_allowed_mb_per_buffer << " MB) < min (" << minimum_limit_mb_per_buffer
              << " MB). Using min." << std::endl;
    max_allowed_mb_per_buffer = minimum_limit_mb_per_buffer;
  }

  // Validate and cap buffer size
  if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
    std::cerr << "Warning: Requested buffer size (" << config.buffer_size_mb << " MB) > limit ("
              << max_allowed_mb_per_buffer << " MB). Using limit." << std::endl;
    config.buffer_size_mb = max_allowed_mb_per_buffer;
  }

  // Calculate final buffer size in bytes
  const size_t bytes_per_mb = 1024 * 1024;
  config.buffer_size = static_cast<size_t>(config.buffer_size_mb) * bytes_per_mb;

  // Sanity checks
  size_t page_size = getpagesize();
  const size_t lat_stride = 128;  // Latency test access stride (bytes)
  
  if (config.buffer_size_mb > 0 && (config.buffer_size == 0 || config.buffer_size / bytes_per_mb != config.buffer_size_mb)) {
    std::cerr << "Error: Buffer size calculation error (" << config.buffer_size_mb << " MB)." << std::endl;
    return EXIT_FAILURE;
  }
  
  if (config.buffer_size < page_size || config.buffer_size < lat_stride * 2) {
    std::cerr << "Error: Final buffer size (" << config.buffer_size << " bytes) is too small." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

void calculate_buffer_sizes(BenchmarkConfig& config) {
  const size_t lat_stride = 128;  // Latency test access stride (bytes)
  size_t page_size_check = getpagesize();
  
  if (config.use_custom_cache_size) {
    // Use 100% of custom cache size
    config.custom_buffer_size = config.custom_cache_size_bytes;
    
    // Ensure buffer size is multiple of stride (128 bytes)
    config.custom_buffer_size = ((config.custom_buffer_size / lat_stride) * lat_stride);
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.custom_buffer_size < lat_stride * 2) config.custom_buffer_size = lat_stride * 2;
    
    // Ensure buffer size is at least page size aligned
    if (config.custom_buffer_size < page_size_check) {
      size_t original_size_kb = config.custom_cache_size_bytes / 1024;
      size_t rounded_size_kb = page_size_check / 1024;
      if (original_size_kb < rounded_size_kb) {
        std::cout << "Info: Custom cache size (" << original_size_kb << " KB) rounded up to " 
                  << rounded_size_kb << " KB (system page size)" << std::endl;
      }
      config.custom_buffer_size = page_size_check;
    }
  } else {
    // Use 75% of cache size for L1 and 10% for L2 to ensure fits within target level
    config.l1_buffer_size = static_cast<size_t>(config.l1_cache_size * 0.75);
    config.l2_buffer_size = static_cast<size_t>(config.l2_cache_size * 0.10);
    
    // Ensure buffer sizes are multiples of stride (128 bytes)
    config.l1_buffer_size = ((config.l1_buffer_size / lat_stride) * lat_stride);
    config.l2_buffer_size = ((config.l2_buffer_size / lat_stride) * lat_stride);
    
    // Ensure minimum size (at least 2 pointers worth)
    if (config.l1_buffer_size < lat_stride * 2) config.l1_buffer_size = lat_stride * 2;
    if (config.l2_buffer_size < lat_stride * 2) config.l2_buffer_size = lat_stride * 2;
    
    // Ensure buffer sizes are at least page size aligned
    if (config.l1_buffer_size < page_size_check) config.l1_buffer_size = page_size_check;
    if (config.l2_buffer_size < page_size_check) config.l2_buffer_size = page_size_check;
  }
}

void calculate_access_counts(BenchmarkConfig& config) {
  // Scale latency accesses proportionally to buffer size (e.g., ~200M for 512MB default)
  config.lat_num_accesses = static_cast<size_t>(200 * 1000 * 1000 * (static_cast<double>(config.buffer_size_mb) / 512.0));
  
  // Cache latency test access counts are already set in struct defaults
  // (l1_num_accesses, l2_num_accesses, custom_num_accesses)
}

