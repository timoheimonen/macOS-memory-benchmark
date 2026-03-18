#ifndef TEST_CONFIG_HELPERS_H
#define TEST_CONFIG_HELPERS_H

#include <gtest/gtest.h>

#include <cstdlib>

#include "core/config/config.h"
#include "core/memory/buffer_manager.h"
#include "utils/benchmark.h"

inline void initialize_system_info(BenchmarkConfig& config) {
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
}

inline ::testing::AssertionResult allocate_and_initialize_buffers(BenchmarkConfig& config,
                                                                  BenchmarkBuffers& buffers) {
  const int alloc_result = allocate_all_buffers(config, buffers);
  if (alloc_result != EXIT_SUCCESS) {
    return ::testing::AssertionFailure()
           << "allocate_all_buffers(config, buffers) failed with code " << alloc_result;
  }

  const int init_result = initialize_all_buffers(buffers, config);
  if (init_result != EXIT_SUCCESS) {
    return ::testing::AssertionFailure()
           << "initialize_all_buffers(buffers, config) failed with code " << init_result;
  }

  return ::testing::AssertionSuccess();
}

#endif  // TEST_CONFIG_HELPERS_H
