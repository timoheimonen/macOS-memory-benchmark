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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>

#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/memory/buffer_manager.h"
#include "output/console/messages/messages_api.h"
#include "test_config_helpers.h"
#include "test_memory_system_calls.h"

class BufferManagerTest : public FakeMemorySystemCallsTest {};

namespace {

BenchmarkConfig make_bandwidth_config() {
  BenchmarkConfig config;
  config.buffer_size = 512;
  config.only_bandwidth = true;
  config.only_latency = false;
  config.max_total_allowed_mb = 0;
  return config;
}

}  // namespace

TEST_F(BufferManagerTest, DefaultCacheAllocationHasExactModeShape) {
  BenchmarkConfig config = make_bandwidth_config();
  config.l1_buffer_size = 64;
  config.l2_buffer_size = 128;

  {
    BenchmarkBuffers buffers;
    ASSERT_EQ(allocate_all_buffers(config, buffers), EXIT_SUCCESS);

    EXPECT_NE(buffers.src_buffer(), nullptr);
    EXPECT_NE(buffers.dst_buffer(), nullptr);
    EXPECT_EQ(buffers.lat_buffer(), nullptr);
    EXPECT_EQ(buffers.l1_buffer(), nullptr);
    EXPECT_EQ(buffers.l2_buffer(), nullptr);
    EXPECT_NE(buffers.l1_bw_src(), nullptr);
    EXPECT_NE(buffers.l1_bw_dst(), nullptr);
    EXPECT_NE(buffers.l2_bw_src(), nullptr);
    EXPECT_NE(buffers.l2_bw_dst(), nullptr);
    EXPECT_EQ(buffers.custom_buffer(), nullptr);
    EXPECT_EQ(state.map_calls, 6u);
    EXPECT_EQ(state.advise_calls, 6u);
    EXPECT_EQ(state.last_advice, MADV_WILLNEED);
  }
  EXPECT_EQ(state.unmap_calls, 6u);
}

TEST_F(BufferManagerTest, PatternNonCacheableModeRequestsOnlyTwoRandomHintedBuffers) {
  BenchmarkConfig config = make_bandwidth_config();
  config.run_patterns = true;
  config.use_non_cacheable = true;
  config.l1_buffer_size = 128;
  config.l2_buffer_size = 256;

  {
    BenchmarkBuffers buffers;
    ASSERT_EQ(allocate_all_buffers(config, buffers), EXIT_SUCCESS);
    EXPECT_NE(buffers.src_buffer(), nullptr);
    EXPECT_NE(buffers.dst_buffer(), nullptr);
    EXPECT_EQ(buffers.l1_bw_src(), nullptr);
    EXPECT_EQ(buffers.l2_bw_src(), nullptr);
    EXPECT_EQ(state.map_calls, 2u);
    EXPECT_EQ(state.advise_calls, 2u);
    EXPECT_EQ(state.last_advice, MADV_RANDOM);
  }
  EXPECT_EQ(state.unmap_calls, 2u);
}

TEST_F(BufferManagerTest, NthAllocationFailureKeepsOwnershipUntilPartialStateDestruction) {
  BenchmarkConfig config = make_bandwidth_config();
  config.run_patterns = true;
  state.fail_map_on_call = 2;

  {
    BenchmarkBuffers buffers;
    testing::internal::CaptureStderr();
    EXPECT_EQ(allocate_all_buffers(config, buffers), EXIT_FAILURE);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_NE(buffers.src_buffer(), nullptr);
    EXPECT_EQ(buffers.dst_buffer(), nullptr);
    EXPECT_EQ(state.map_calls, 2u);
    EXPECT_EQ(state.unmap_calls, 0u);
    EXPECT_NE(error.find(Messages::error_mmap_failed("dst_buffer")),
              std::string::npos);
  }
  EXPECT_EQ(state.unmap_calls, 1u);
}

TEST_F(BufferManagerTest, InitializationWritesExactSourcePatternAndZeroDestination) {
  BenchmarkConfig config = make_bandwidth_config();
  config.run_patterns = true;

  {
    BenchmarkBuffers buffers;
    ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
    const auto* source = static_cast<const unsigned char*>(buffers.src_buffer());
    const auto* destination =
        static_cast<const unsigned char*>(buffers.dst_buffer());
    ASSERT_NE(source, nullptr);
    ASSERT_NE(destination, nullptr);
    for (size_t index = 0; index < config.buffer_size; ++index) {
      EXPECT_EQ(source[index], static_cast<unsigned char>(index & 0xff));
      EXPECT_EQ(destination[index], 0u);
    }
  }
  EXPECT_EQ(state.unmap_calls, 2u);
}

TEST_F(BufferManagerTest, ZeroMainBufferFailsBeforeAllocationWithExactReason) {
  BenchmarkConfig config = make_bandwidth_config();
  config.buffer_size = 0;

  testing::internal::CaptureStderr();
  BenchmarkBuffers buffers;
  const int result = allocate_all_buffers(config, buffers);
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_EQ(error, Messages::error_prefix() +
                       Messages::error_main_buffer_size_zero() + "\n");
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_EQ(buffers.src_buffer(), nullptr);
  EXPECT_EQ(buffers.dst_buffer(), nullptr);
}

TEST_F(BufferManagerTest, MainLatencyAdditionOverflowIsRejectedBeforeAllocation) {
  BenchmarkConfig config;
  config.buffer_size = std::numeric_limits<size_t>::max() / 2;
  config.only_latency = false;
  config.only_bandwidth = false;

  testing::internal::CaptureStderr();
  BenchmarkBuffers buffers;
  const int result = allocate_all_buffers(config, buffers);
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error.find(Messages::error_total_memory_overflow()),
            std::string::npos);
  EXPECT_EQ(state.map_calls, 0u);
}

TEST_F(BufferManagerTest, PeakAccountingUsesLargerCachePhaseInsteadOfMainPhase) {
  BenchmarkConfig config;
  config.buffer_size = Constants::BYTES_PER_MB;
  config.l1_buffer_size = Constants::BYTES_PER_MB;
  config.l2_buffer_size = 2 * Constants::BYTES_PER_MB;
  config.max_total_allowed_mb = 0;

  size_t total_memory_bytes = 0;
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes),
            EXIT_SUCCESS);
  EXPECT_EQ(total_memory_bytes, 6 * Constants::BYTES_PER_MB);
}

TEST_F(BufferManagerTest, PeakAccountingHandlesLatencyOnlyCustomCacheWithMainDisabled) {
  BenchmarkConfig config;
  config.buffer_size = 0;
  config.only_latency = true;
  config.use_custom_cache_size = true;
  config.custom_buffer_size = 43 * Constants::BYTES_PER_MB;
  config.max_total_allowed_mb = 0;

  size_t total_memory_bytes = 0;
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes),
            EXIT_SUCCESS);
  EXPECT_EQ(total_memory_bytes, 43 * Constants::BYTES_PER_MB);
}

TEST_F(BufferManagerTest, PeakAccountingPatternsSkipLatencyAndCacheBuffers) {
  BenchmarkConfig config;
  config.buffer_size = Constants::BYTES_PER_MB;
  config.run_patterns = true;
  config.use_custom_cache_size = true;
  config.custom_buffer_size = 43 * Constants::BYTES_PER_MB;
  config.max_total_allowed_mb = 0;

  size_t total_memory_bytes = 0;
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes),
            EXIT_SUCCESS);
  EXPECT_EQ(total_memory_bytes, 2 * Constants::BYTES_PER_MB);
}
