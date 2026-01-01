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
#include "core/system/system_info.h"
#include "output/console/messages.h"
#include "utils/benchmark.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdlib>

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

