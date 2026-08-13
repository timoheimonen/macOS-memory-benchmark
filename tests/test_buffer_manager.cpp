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

BenchmarkConfig make_pattern_config() {
  BenchmarkConfig config;
  config.buffer_size = 512;
  config.only_bandwidth = true;
  config.only_latency = false;
  config.run_patterns = true;
  config.max_total_allowed_mb = 0;
  return config;
}

}  // namespace

TEST_F(BufferManagerTest, PatternAllocationUsesModeSpecificAdviceForBothOwnedMappings) {
  for (const bool use_non_cacheable : {false, true}) {
    SCOPED_TRACE(use_non_cacheable ? "non-cacheable" : "regular");
    BenchmarkConfig config = make_pattern_config();
    config.use_non_cacheable = use_non_cacheable;

    const size_t map_calls_before = state.map_calls;
    const size_t advise_calls_before = state.advise_calls;
    const size_t unmap_calls_before = state.unmap_calls;
    {
      PatternBuffers buffers;
      ASSERT_EQ(allocate_pattern_buffers(config, buffers), EXIT_SUCCESS);
      EXPECT_NE(buffers.src_buffer(), nullptr);
      EXPECT_NE(buffers.dst_buffer(), nullptr);
      EXPECT_EQ(state.map_calls - map_calls_before, 2u);
      EXPECT_EQ(state.advise_calls - advise_calls_before, 2u);
      EXPECT_EQ(state.last_map_size, config.buffer_size);
      EXPECT_EQ(state.last_advise_size, config.buffer_size);
      EXPECT_EQ(state.last_advice, use_non_cacheable ? MADV_RANDOM : MADV_WILLNEED);
      EXPECT_EQ(state.unmap_calls, unmap_calls_before);
    }
    EXPECT_EQ(state.unmap_calls - unmap_calls_before, 2u);
    EXPECT_EQ(state.last_unmapped_size, config.buffer_size);
  }
}

TEST_F(BufferManagerTest, SecondPatternMappingFailureCleansCandidateAtomically) {
  const BenchmarkConfig config = make_pattern_config();

  {
    PatternBuffers buffers;
    ASSERT_EQ(allocate_pattern_buffers(config, buffers), EXIT_SUCCESS);
    void* original_source = buffers.src_buffer();
    void* original_destination = buffers.dst_buffer();
    state.fail_map_on_call = 4;

    testing::internal::CaptureStderr();
    EXPECT_EQ(allocate_pattern_buffers(config, buffers), EXIT_FAILURE);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_EQ(buffers.src_buffer(), original_source);
    EXPECT_EQ(buffers.dst_buffer(), original_destination);
    EXPECT_EQ(state.map_calls, 4u);
    EXPECT_EQ(state.unmap_calls, 1u);
    EXPECT_NE(error.find(Messages::error_mmap_failed("dst_buffer")), std::string::npos);
  }
  EXPECT_EQ(state.unmap_calls, 3u);
}

TEST_F(BufferManagerTest, InitializationWritesExactSourcePatternAndZeroDestination) {
  const BenchmarkConfig config = make_pattern_config();

  {
    PatternBuffers buffers;
    ASSERT_TRUE(allocate_and_initialize_pattern_buffers(config, buffers));
    const auto* source = static_cast<const unsigned char*>(buffers.src_buffer());
    const auto* destination = static_cast<const unsigned char*>(buffers.dst_buffer());
    ASSERT_NE(source, nullptr);
    ASSERT_NE(destination, nullptr);
    for (size_t index = 0; index < config.buffer_size; ++index) {
      EXPECT_EQ(source[index], static_cast<unsigned char>(index & 0xff));
      EXPECT_EQ(destination[index], 0u);
    }
  }
  EXPECT_EQ(state.unmap_calls, 2u);
}

TEST_F(BufferManagerTest, InvalidPatternAllocationRequestsFailBeforeSystemCalls) {
  struct RejectionCase {
    const char* name;
    size_t buffer_size;
    unsigned long max_total_allowed_mb;
    std::string expected_reason;
  };
  const RejectionCase cases[] = {
      {"zero main buffer", 0, 0, Messages::error_main_buffer_size_zero()},
      {"pair overflow", std::numeric_limits<size_t>::max() / 2 + 1, 0,
       Messages::error_buffer_size_overflow_calculation()},
      {"memory cap", Constants::BYTES_PER_MB, 1, Messages::error_total_memory_exceeds_limit(2, 1)},
  };

  for (const RejectionCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    BenchmarkConfig config = make_pattern_config();
    config.buffer_size = test_case.buffer_size;
    config.max_total_allowed_mb = test_case.max_total_allowed_mb;
    const size_t map_calls_before = state.map_calls;
    const size_t advise_calls_before = state.advise_calls;
    const size_t unmap_calls_before = state.unmap_calls;

    testing::internal::CaptureStderr();
    PatternBuffers buffers;
    const int result = allocate_pattern_buffers(config, buffers);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(error, Messages::error_prefix() + test_case.expected_reason + "\n");
    EXPECT_EQ(state.map_calls, map_calls_before);
    EXPECT_EQ(state.advise_calls, advise_calls_before);
    EXPECT_EQ(state.unmap_calls, unmap_calls_before);
    EXPECT_EQ(buffers.src_buffer(), nullptr);
    EXPECT_EQ(buffers.dst_buffer(), nullptr);
  }
}

TEST_F(BufferManagerTest, PatternInitializationRejectsInvalidInputs) {
  struct InvalidInputCase {
    const char* name;
    bool allocate_mappings;
    size_t buffer_size;
    std::string expected_reason;
  };
  const InvalidInputCase cases[] = {
      {"missing mappings", false, 512, Messages::error_main_buffers_not_allocated()},
      {"zero size", true, 0, Messages::error_buffer_size_zero_generic()},
  };

  for (const InvalidInputCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    PatternBuffers buffers;
    if (test_case.allocate_mappings) {
      const BenchmarkConfig config = make_pattern_config();
      ASSERT_EQ(allocate_pattern_buffers(config, buffers), EXIT_SUCCESS);
    }

    testing::internal::CaptureStderr();
    const int result = initialize_pattern_buffers(buffers, test_case.buffer_size);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(error, Messages::error_prefix() + test_case.expected_reason + "\n");
  }
}

TEST_F(BufferManagerTest, PeakAccountingUsesLargerCachePhaseInsteadOfMainPhase) {
  BenchmarkConfig config;
  config.buffer_size = Constants::BYTES_PER_MB;
  config.l1_buffer_size = Constants::BYTES_PER_MB;
  config.l2_buffer_size = 2 * Constants::BYTES_PER_MB;
  config.max_total_allowed_mb = 0;

  size_t total_memory_bytes = 0;
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes), EXIT_SUCCESS);
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
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes), EXIT_SUCCESS);
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
  ASSERT_EQ(calculate_total_allocation_bytes(config, total_memory_bytes), EXIT_SUCCESS);
  EXPECT_EQ(total_memory_bytes, 2 * Constants::BYTES_PER_MB);
}
