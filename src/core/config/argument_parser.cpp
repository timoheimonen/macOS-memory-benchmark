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
 * @file argument_parser.cpp
 * @brief Command-line argument parsing implementation
 *
 * This file implements the argument parser for the memory benchmark application.
 * It handles parsing and validation of all command-line options including:
 * - Buffer size configuration (-buffersize)
 * - Iteration counts (-iterations, -count)
 * - Latency pointer-chain stride (-latency-stride-bytes)
 * - Cache size specification (-cache-size)
 * - Thread count configuration (-threads)
 * - Test mode selection (-patterns, -only-bandwidth, -only-latency)
 * - Output options (-output)
 * - Help display (-h, --help)
 *
 * The parser uses a two-pass approach: first parsing cache size (needed for
 * system detection), then parsing all other arguments. It employs exception
 * handling internally for validation, converting exceptions to return codes
 * at the function boundary.
 *
 * @note Uses exception handling internally but presents a return code interface
 * @note Performs comprehensive range validation for all numeric parameters
 */

#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"
#include "utils/benchmark.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdlib>

/**
 * @brief Error Handling Strategy for this module:
 * 
 * This function uses EXCEPTIONS internally, which are caught and converted to RETURN CODES.
 * 
 * Rationale:
 * - Argument parsing involves multiple validation points with complex logic
 * - Standard library functions (std::stoll) throw exceptions naturally
 * - Exceptions allow early termination without deeply nested conditionals
 * - Cleaner code flow for validation logic
 * - BUT: Must convert to return codes at boundary (called from main())
 * 
 * Exception types used:
 * - std::invalid_argument: Missing values, unknown options, malformed arguments
 * - std::out_of_range: Value validation errors (out of valid range)
 * 
 * Error conversion:
 * - All exceptions are caught and converted to EXIT_FAILURE
 * - Error messages are printed to std::cerr
 * - Usage information is printed before returning
 * - Returns EXIT_SUCCESS on success, EXIT_FAILURE on any error
 * 
 * Note: This is a "boundary function" - it converts exceptions to return codes
 *       to integrate with the main program flow (which uses exit codes).
 */
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)
  long long requested_threads_ll = -1;  // User requested threads (-1 = none)
  long long requested_latency_tlb_locality_kb_ll = -1;  // User requested TLB-locality window in KB (-1 = none)

  bool analyze_tlb_present = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-analyze-tlb") {
      analyze_tlb_present = true;
      break;
    }
  }

  if (analyze_tlb_present) {
    config.analyze_tlb = true;
    bool output_seen = false;
    bool latency_stride_seen = false;
    bool latency_chain_mode_seen = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-analyze-tlb") {
        continue;
      }

      if (arg == "-output") {
        if (output_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option("-output")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value("-output")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        config.output_file = argv[i];
        output_seen = true;
        continue;
      }

      if (arg == "-latency-stride-bytes") {
        if (latency_stride_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option("-latency-stride-bytes")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value("-latency-stride-bytes")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string stride_value = argv[i];
        long long val_ll = 0;
        try {
          val_ll = std::stoll(stride_value);
        } catch (const std::invalid_argument&) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, stride_value, "must be an integer")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        } catch (const std::out_of_range&) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, stride_value, "out of range")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        if (val_ll <= 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           arg,
                           stride_value,
                           Messages::error_latency_stride_invalid(
                               val_ll, 1, std::numeric_limits<long long>::max()))
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.latency_stride_bytes = static_cast<size_t>(val_ll);
        if ((config.latency_stride_bytes % sizeof(void*)) != 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_latency_stride_alignment(
                           config.latency_stride_bytes,
                           sizeof(void*))
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.user_specified_latency_stride = true;
        latency_stride_seen = true;
        continue;
      }

      if (arg == "-latency-chain-mode") {
        if (latency_chain_mode_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option("-latency-chain-mode")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value("-latency-chain-mode")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string mode_value = argv[i];
        LatencyChainMode parsed_mode = LatencyChainMode::Auto;
        if (!latency_chain_mode_from_string(mode_value, parsed_mode)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, mode_value,
                                                     Messages::error_latency_chain_mode_invalid())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.latency_chain_mode = parsed_mode;
        config.user_specified_latency_chain_mode = true;
        latency_chain_mode_seen = true;
        continue;
      }

      std::cerr << Messages::error_prefix()
                << Messages::error_analyze_tlb_must_be_used_alone()
                << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }
  
  // First pass: parse -cache-size early (needed for cache size detection)
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-cache-size") {
        if (++i < argc) {
          // std::stoll() throws std::invalid_argument if conversion fails
          // std::stoll() throws std::out_of_range if value out of range
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          // Note: 0 is accepted here and validated later (allowed only with -only-latency)
          if (val_ll < 0 || val_ll > Constants::MAX_CACHE_SIZE_KB ||
              (val_ll > 0 && val_ll < Constants::MIN_CACHE_SIZE_KB))
            throw std::out_of_range(Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB / 1024));
          config.custom_cache_size_kb_ll = val_ll;
        } else
          // Error: Missing required value for option
          throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
      }
    } catch (const std::invalid_argument &e) {
      // Exception caught: Convert to return code and print error
      std::cerr << Messages::error_prefix() << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    } catch (const std::out_of_range &e) {
      // Exception caught: Convert to return code and print error
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    }
  }

  // Get system info
  config.cpu_name = get_processor_name();
  config.macos_version = get_macos_version();
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
  // Error handling: Uses try-catch to convert exceptions to return codes
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (arg == "-iterations") {
        if (++i < argc) {
          // Error: std::stoll() may throw std::invalid_argument or std::out_of_range
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_iterations_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.iterations = static_cast<int>(val_ll);
          config.user_specified_iterations = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-iterations"));
      } else if (arg == "-buffersize") {
        if (++i < argc) {
          // Error: std::stoll() may throw exceptions
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          // Note: 0 is accepted here and validated later (allowed only with -only-latency)
          if (val_ll < 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range(Messages::error_buffersize_invalid(val_ll, std::numeric_limits<unsigned long>::max()));
          requested_buffer_size_mb_ll = val_ll;
          config.user_specified_buffersize = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-buffersize"));
      } else if (arg == "-count") {
        if (++i < argc) {
          // Error: std::stoll() may throw exceptions
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_count_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.loop_count = static_cast<int>(val_ll);
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-count"));
      } else if (arg == "-latency-samples") {
        if (++i < argc) {
          // Error: std::stoll() may throw exceptions
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_latency_samples_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.latency_sample_count = static_cast<int>(val_ll);
          config.user_specified_latency_samples = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-latency-samples"));
      } else if (arg == "-latency-stride-bytes") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          if (val_ll <= 0) {
            throw std::out_of_range(Messages::error_latency_stride_invalid(
                val_ll, 1, std::numeric_limits<long long>::max()));
          }
          config.latency_stride_bytes = static_cast<size_t>(val_ll);
          config.user_specified_latency_stride = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value("-latency-stride-bytes"));
        }
      } else if (arg == "-latency-chain-mode") {
        if (++i < argc) {
          LatencyChainMode parsed_mode = LatencyChainMode::Auto;
          if (!latency_chain_mode_from_string(argv[i], parsed_mode)) {
            throw std::out_of_range(Messages::error_latency_chain_mode_invalid());
          }
          config.latency_chain_mode = parsed_mode;
          config.user_specified_latency_chain_mode = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value("-latency-chain-mode"));
        }
      } else if (arg == "-latency-tlb-locality-kb") {
        if (++i < argc) {
          long long val_ll = std::stoll(argv[i]);
          const long long max_locality_kb =
              static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
          if (val_ll < 0 || val_ll > max_locality_kb) {
            throw std::out_of_range(Messages::error_latency_tlb_locality_invalid(val_ll, max_locality_kb));
          }
          requested_latency_tlb_locality_kb_ll = val_ll;
          config.user_specified_latency_tlb_locality = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value("-latency-tlb-locality-kb"));
        }
      } else if (arg == "-cache-size") {
        // Already parsed in first pass, skip it and its value in second pass
        if (config.custom_cache_size_kb_ll != -1) {
          // Skip the value argument (already validated in first pass)
          if (++i >= argc) {
            // Error: This shouldn't happen (first pass should have validated it), but handle defensively
            throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
          }
          // Silently skip - already parsed and validated in first pass
          continue;
        } else {
          // This shouldn't happen (first pass should have set it), but handle defensively
          if (++i < argc) {
            // Error: Try to parse it now (fallback case) - std::stoll() may throw
            long long val_ll = std::stoll(argv[i]);
            // Error: Value validation - out of valid range
            if (val_ll < 0 || val_ll > Constants::MAX_CACHE_SIZE_KB ||
                (val_ll > 0 && val_ll < Constants::MIN_CACHE_SIZE_KB))
              throw std::out_of_range(Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB, Constants::MAX_CACHE_SIZE_KB / 1024));
            config.custom_cache_size_kb_ll = val_ll;
          } else
            // Error: Missing required value
            throw std::invalid_argument(Messages::error_missing_value("-cache-size"));
        }
      } else if (arg == "-output") {
        if (++i < argc) {
          config.output_file = argv[i];
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-output"));
      } else if (arg == "-benchmark") {
        config.run_benchmark = true;
        if (config.run_patterns) {
          throw std::invalid_argument(Messages::error_mutually_exclusive_modes("-benchmark", "-patterns"));
        }
      } else if (arg == "-patterns") {
        config.run_patterns = true;
        if (config.run_benchmark) {
          throw std::invalid_argument(Messages::error_mutually_exclusive_modes("-benchmark", "-patterns"));
        }
      } else if (arg == "-non-cacheable") {
        config.use_non_cacheable = true;
      } else if (arg == "-only-bandwidth") {
        config.only_bandwidth = true;
      } else if (arg == "-only-latency") {
        config.only_latency = true;
      } else if (arg == "-threads") {
        if (++i < argc) {
          // Error: std::stoll() may throw exceptions
          long long val_ll = std::stoll(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_threads_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          requested_threads_ll = val_ll;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value("-threads"));
      } else if (arg == "-h" || arg == "--help") {
        config.help_printed = true;
        print_usage(argv[0]);
        return EXIT_SUCCESS;  // Special return value for help (not an error)
      } else {
        // Error: Unknown option
        throw std::invalid_argument(Messages::error_unknown_option(arg));
      }
    } catch (const std::invalid_argument &e) {
      // Exception caught: Convert to return code
      // Error: Invalid argument (missing value, unknown option, etc.)
      std::cerr << Messages::error_prefix() << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    } catch (const std::out_of_range &e) {
      // Exception caught: Convert to return code
      // Error: Value out of valid range
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
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

  if (requested_latency_tlb_locality_kb_ll != -1) {
    config.latency_tlb_locality_bytes =
        static_cast<size_t>(requested_latency_tlb_locality_kb_ll) * Constants::BYTES_PER_KB;
  }

  return EXIT_SUCCESS;  // All arguments parsed successfully
}
