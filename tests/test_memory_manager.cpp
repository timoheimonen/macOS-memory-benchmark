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

#include <cerrno>
#include <cstring>
#include <string>
#include <sys/mman.h>

#include "core/memory/memory_manager.h"
#include "output/console/messages/messages_api.h"
#include "test_memory_system_calls.h"

class MemoryManagerTest : public FakeMemorySystemCallsTest {};

TEST_F(MemoryManagerTest, RegularAllocationRequestsWillNeedAndReleasesExactMapping) {
  void* mapped_pointer = nullptr;
  {
    MmapPtr buffer = allocate_buffer(128, "regular");
    ASSERT_NE(buffer.get(), nullptr);
    mapped_pointer = buffer.get();
    EXPECT_EQ(state.map_calls, 1u);
    EXPECT_EQ(state.advise_calls, 1u);
    EXPECT_EQ(state.last_map_size, 128u);
    EXPECT_EQ(state.last_advise_size, 128u);
    EXPECT_EQ(state.last_protection, PROT_READ | PROT_WRITE);
    EXPECT_EQ(state.last_flags, MAP_PRIVATE | MAP_ANONYMOUS);
    EXPECT_EQ(state.last_advice, MADV_WILLNEED);
    EXPECT_EQ(state.unmap_calls, 0u);
  }

  EXPECT_EQ(state.unmap_calls, 1u);
  EXPECT_EQ(state.last_unmapped_pointer, mapped_pointer);
  EXPECT_EQ(state.last_unmapped_size, 128u);
}

TEST_F(MemoryManagerTest, NonCacheableAllocationRequestsRandomHintAndReleasesMapping) {
  {
    MmapPtr buffer = allocate_buffer_non_cacheable(256, "cache-discouraged");
    ASSERT_NE(buffer.get(), nullptr);
    EXPECT_EQ(state.map_calls, 1u);
    EXPECT_EQ(state.advise_calls, 1u);
    EXPECT_EQ(state.last_advice, MADV_RANDOM);
    EXPECT_EQ(state.last_map_size, 256u);
    EXPECT_EQ(state.last_advise_size, 256u);
  }
  EXPECT_EQ(state.unmap_calls, 1u);
  EXPECT_EQ(state.last_unmapped_size, 256u);
}

TEST_F(MemoryManagerTest, MappingFailureReturnsNullWithoutAdviceOrUnmap) {
  state.fail_map_on_call = 1;
  testing::internal::CaptureStderr();
  MmapPtr buffer = allocate_buffer(64, "failed-map");
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(buffer.get(), nullptr);
  EXPECT_EQ(state.map_calls, 1u);
  EXPECT_EQ(state.advise_calls, 0u);
  EXPECT_EQ(state.unmap_calls, 0u);
  EXPECT_EQ(error, Messages::error_prefix() +
                       Messages::error_mmap_failed("failed-map") + ": " +
                       std::strerror(ENOMEM) + "\n");
}

TEST_F(MemoryManagerTest, AdviceFailureIsReportedButAllocationRemainsOwned) {
  state.advise_result = -1;
  state.advise_errno = EINVAL;
  bool allocation_succeeded = false;
  testing::internal::CaptureStderr();
  {
    MmapPtr buffer = allocate_buffer_non_cacheable(64, "advice-failure");
    allocation_succeeded = buffer != nullptr;
  }
  const std::string error = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(allocation_succeeded);
  EXPECT_EQ(state.map_calls, 1u);
  EXPECT_EQ(state.advise_calls, 1u);
  EXPECT_EQ(state.unmap_calls, 1u);
  EXPECT_EQ(error, Messages::warning_prefix() +
                       Messages::warning_madvise_random_failed(
                           "advice-failure", std::strerror(EINVAL)) +
                       "\n");
}

TEST_F(MemoryManagerTest, UnmapFailureIsReportedAfterOwnedBufferDestruction) {
  state.unmap_result = -1;
  state.unmap_errno = EINVAL;

  bool allocation_succeeded = false;
  testing::internal::CaptureStderr();
  {
    MmapPtr buffer = allocate_buffer(64, "unmap-failure");
    allocation_succeeded = buffer != nullptr;
  }
  const std::string error = testing::internal::GetCapturedStderr();

  ASSERT_TRUE(allocation_succeeded);
  EXPECT_EQ(state.unmap_calls, 1u);
  EXPECT_EQ(error, Messages::error_prefix() + Messages::error_munmap_failed() +
                       ": " + std::strerror(EINVAL) + "\n");
}

TEST_F(MemoryManagerTest, ZeroSizeFailsBeforeAnySystemCallWithExactMessage) {
  for (const bool non_cacheable : {false, true}) {
    const char* name = non_cacheable ? "zero-non-cacheable" : "zero-regular";
    testing::internal::CaptureStderr();
    MmapPtr buffer = non_cacheable ? allocate_buffer_non_cacheable(0, name)
                                   : allocate_buffer(0, name);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(error, Messages::error_prefix() +
                         Messages::error_buffer_size_zero(name) + "\n");
  }
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_EQ(state.advise_calls, 0u);
  EXPECT_EQ(state.unmap_calls, 0u);
}
