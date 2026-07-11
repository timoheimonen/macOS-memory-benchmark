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

/** Run the complete standalone GPU-bandwidth command. */
int run_gpu_bandwidth_mode(int argc, char* argv[]);

#endif  // GPU_BANDWIDTH_H
