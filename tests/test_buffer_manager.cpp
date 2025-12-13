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
#include <gtest/gtest.h>
#include "buffer_manager.h"
#include "config.h"
#include "constants.h"
#include "benchmark.h"  // Declares system_info functions
#include <cstdlib>

// Test buffer allocation with valid config
TEST(BufferManagerTest, AllocateAllBuffersValid) {
  BenchmarkConfig config;
  config.buffer_size = 1024 * 1024;  // 1 MB
  config.l1_buffer_size = 64 * 1024;  // 64 KB
  config.l2_buffer_size = 512 * 1024;  // 512 KB
  config.use_custom_cache_size = false;
  
  // Initialize system info (needed for allocation)
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int result = allocate_all_buffers(config, buffers);
  
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_NE(buffers.src_buffer(), nullptr);
  EXPECT_NE(buffers.dst_buffer(), nullptr);
  EXPECT_NE(buffers.lat_buffer(), nullptr);
  EXPECT_NE(buffers.l1_buffer(), nullptr);
  EXPECT_NE(buffers.l2_buffer(), nullptr);
}

// Test buffer allocation with custom cache size
TEST(BufferManagerTest, AllocateAllBuffersCustomCache) {
  BenchmarkConfig config;
  config.buffer_size = 1024 * 1024;  // 1 MB
  config.use_custom_cache_size = true;
  config.custom_buffer_size = 128 * 1024;  // 128 KB
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.custom_cache_size_bytes = config.custom_buffer_size;
  
  BenchmarkBuffers buffers;
  int result = allocate_all_buffers(config, buffers);
  
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_NE(buffers.src_buffer(), nullptr);
  EXPECT_NE(buffers.dst_buffer(), nullptr);
  EXPECT_NE(buffers.lat_buffer(), nullptr);
  EXPECT_NE(buffers.custom_buffer(), nullptr);
}

// Test buffer helper methods return valid pointers
TEST(BufferManagerTest, BufferHelperMethods) {
  BenchmarkConfig config;
  config.buffer_size = 1024 * 1024;  // 1 MB
  config.l1_buffer_size = 64 * 1024;  // 64 KB
  config.l2_buffer_size = 512 * 1024;  // 512 KB
  config.use_custom_cache_size = false;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Test helper methods
  EXPECT_EQ(buffers.src_buffer(), buffers.src_buffer_ptr.get());
  EXPECT_EQ(buffers.dst_buffer(), buffers.dst_buffer_ptr.get());
  EXPECT_EQ(buffers.lat_buffer(), buffers.lat_buffer_ptr.get());
  EXPECT_EQ(buffers.l1_buffer(), buffers.l1_buffer_ptr.get());
  EXPECT_EQ(buffers.l2_buffer(), buffers.l2_buffer_ptr.get());
}

// Test buffer initialization
TEST(BufferManagerTest, InitializeAllBuffers) {
  BenchmarkConfig config;
  config.buffer_size = 1024 * 1024;  // 1 MB
  config.l1_buffer_size = 64 * 1024;  // 64 KB
  config.l2_buffer_size = 512 * 1024;  // 512 KB
  config.use_custom_cache_size = false;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  EXPECT_EQ(init_result, EXIT_SUCCESS);
}

