// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_bandwidth.h
 * @brief Standalone Metal GPU memory-bandwidth command configuration
 */

#ifndef GPU_BANDWIDTH_H
#define GPU_BANDWIDTH_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/config/constants.h"

/** Parsed configuration for the standalone `--gpu-bandwidth` command. */
struct GpuBandwidthConfig {
  unsigned long buffer_size_mb = Constants::GPU_DEFAULT_BUFFER_SIZE_MB;
  size_t buffer_size_bytes = 0;
  size_t iterations = 0;
  size_t loop_count = Constants::GPU_DEFAULT_LOOP_COUNT;
  /**
   * Raw JSON output target: empty disables JSON, exact `-` selects stdout, and
   * every other spelling is preserved as a file path, including `./-`.
   */
  std::string output_file;
  uint64_t seed = 0;
  bool user_specified_iterations = false;
  bool user_specified_seed = false;
  bool help_printed = false;
  std::vector<std::string> argv;
};

/** Deterministic seed input used only by GPU parser unit tests. */
struct GpuBandwidthParserTestHooks {
  uint64_t generated_seed = 0;
};

void set_gpu_bandwidth_parser_test_hooks(
    const GpuBandwidthParserTestHooks* hooks);

/**
 * @brief Parse and validate the standalone GPU option whitelist.
 *
 * This boundary performs all syntax/methodology validation that must fail
 * before a Metal device or output checkpoint is created. It never calls the
 * general `BenchmarkConfig` parser.
 */
int parse_gpu_bandwidth_arguments(int argc, char* argv[],
                                  GpuBandwidthConfig& config);

/**
 * @brief Run the complete standalone GPU-bandwidth command.
 *
 * Parsing, validation, and help handling precede output-session creation. For
 * an exact `--output -` target, post-parse human output is routed to stderr and
 * one initialized terminal schema-v1 result is emitted to stdout. Other output
 * values retain file checkpoint behavior and raw path spelling.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array retained in the GPU JSON payload.
 * @return `EXIT_SUCCESS` for help, complete execution, or graceful
 *         interruption; `EXIT_FAILURE` for invalid input, unsupported or
 *         failed execution, or output failure.
 * @note Unexpected command-boundary exceptions are converted to diagnostics
 *       and return codes; they do not propagate to `main()`.
 */
int run_gpu_bandwidth_mode(int argc, char* argv[]);

#endif  // GPU_BANDWIDTH_H
