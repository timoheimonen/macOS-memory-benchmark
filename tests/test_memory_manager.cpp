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
#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "core/memory/memory_manager.h"
#include "core/system/page_size.h"
#include "output/console/messages/messages_api.h"
#include "test_memory_system_calls.h"
#include "utils/numeric_utils.h"

class MemoryManagerTest : public FakeMemorySystemCallsTest {};

TEST_F(MemoryManagerTest, RegularAllocationUsesWillNeedAndOwnedMappingsReportUnmapFailures) {
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
  EXPECT_EQ(state.unmap_calls, 2u);
  EXPECT_EQ(state.last_unmapped_size, 64u);
  EXPECT_EQ(error, Messages::error_prefix() + Messages::error_munmap_failed() + ": " + std::strerror(EINVAL) + "\n");
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
  EXPECT_EQ(error,
            Messages::error_prefix() + Messages::error_mmap_failed("failed-map") + ": " + std::strerror(ENOMEM) + "\n");
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
                       Messages::warning_madvise_random_failed("advice-failure", std::strerror(EINVAL)) + "\n");
}

TEST_F(MemoryManagerTest, ReadOnlyProtectionUsesExactMappingRangeAndReportsFailure) {
  MmapPtr buffer = allocate_buffer(4096, "protected");
  ASSERT_NE(buffer, nullptr);

  EXPECT_TRUE(protect_buffer_read_only(buffer.get(), 4096));
  EXPECT_EQ(state.protect_calls, 1u);
  EXPECT_EQ(state.last_protect_size, 4096u);
  EXPECT_EQ(state.last_protect_flags, PROT_READ);

  state.protect_result = -1;
  EXPECT_FALSE(protect_buffer_read_only(buffer.get(), 4096));
  EXPECT_EQ(state.protect_calls, 2u);
  EXPECT_FALSE(protect_buffer_read_only(nullptr, 4096));
  EXPECT_FALSE(protect_buffer_read_only(buffer.get(), 0));
  EXPECT_EQ(state.protect_calls, 2u);
}

TEST_F(MemoryManagerTest, AllocateBufferWithEvidenceRecordsMappingAndAdviseState) {
  const size_t page_size = get_system_page_size_bytes();
  size_t expected_mapped = 128;
  if (page_size > 0) {
    ASSERT_TRUE(NumericUtils::checked_round_up(128, page_size, expected_mapped));
  }

  AllocatedBufferEvidence regular =
      allocate_buffer_with_evidence(128, false, "evidence-regular");
  ASSERT_NE(regular.ptr.get(), nullptr);
  EXPECT_EQ(regular.requested_bytes, 128u);
  EXPECT_EQ(regular.mapped_bytes, expected_mapped);
  EXPECT_FALSE(regular.non_cacheable_requested);
  EXPECT_EQ(regular.madvise_advice, MADV_WILLNEED);
  EXPECT_TRUE(regular.madvise_applied);
  EXPECT_EQ(regular.madvise_errno, 0);
  EXPECT_FALSE(regular.read_only_protected);
  EXPECT_EQ(regular.mprotect_errno, 0);

  AllocatedBufferEvidence non_cacheable =
      allocate_buffer_with_evidence(256, true, "evidence-non-cacheable");
  ASSERT_NE(non_cacheable.ptr.get(), nullptr);
  EXPECT_EQ(non_cacheable.requested_bytes, 256u);
  EXPECT_EQ(non_cacheable.mapped_bytes, expected_mapped);
  EXPECT_TRUE(non_cacheable.non_cacheable_requested);
  EXPECT_EQ(non_cacheable.madvise_advice, MADV_RANDOM);
  EXPECT_TRUE(non_cacheable.madvise_applied);
  EXPECT_EQ(non_cacheable.madvise_errno, 0);
}

TEST_F(MemoryManagerTest, AllocateBufferWithEvidenceHandlesInvalidAndFailurePaths) {
  testing::internal::CaptureStderr();
  AllocatedBufferEvidence zero = allocate_buffer_with_evidence(0, false, "zero");
  const std::string zero_err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(zero.ptr.get(), nullptr);
  EXPECT_EQ(zero.requested_bytes, 0u);
  EXPECT_EQ(zero.mapped_bytes, 0u);
  EXPECT_FALSE(zero.madvise_applied);
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_FALSE(zero_err.empty());

  state.fail_map_on_call = 1;
  testing::internal::CaptureStderr();
  AllocatedBufferEvidence map_fail =
      allocate_buffer_with_evidence(64, false, "map-fail");
  testing::internal::GetCapturedStderr();
  EXPECT_EQ(map_fail.ptr.get(), nullptr);
  EXPECT_EQ(map_fail.requested_bytes, 64u);
  EXPECT_EQ(map_fail.mapped_bytes, 0u);
  EXPECT_FALSE(map_fail.madvise_applied);

  state.fail_map_on_call = 0;
  state.advise_result = -1;
  state.advise_errno = EINVAL;
  testing::internal::CaptureStderr();
  AllocatedBufferEvidence advise_fail =
      allocate_buffer_with_evidence(64, false, "advise-fail");
  testing::internal::GetCapturedStderr();
  ASSERT_NE(advise_fail.ptr.get(), nullptr);
  EXPECT_FALSE(advise_fail.madvise_applied);
  EXPECT_EQ(advise_fail.madvise_errno, EINVAL);
}

TEST_F(MemoryManagerTest, ProtectBufferReadOnlyWithEvidenceRecordsStatusAndErrno) {
  AllocatedBufferEvidence evidence =
      allocate_buffer_with_evidence(4096, false, "protect-evidence");
  ASSERT_NE(evidence.ptr.get(), nullptr);

  EXPECT_TRUE(protect_buffer_read_only_with_evidence(evidence, 4096));
  EXPECT_TRUE(evidence.read_only_protected);
  EXPECT_EQ(evidence.mprotect_errno, 0);
  EXPECT_EQ(state.protect_calls, 1u);

  state.protect_result = -1;
  state.protect_errno = EACCES;
  EXPECT_FALSE(protect_buffer_read_only_with_evidence(evidence, 4096));
  EXPECT_FALSE(evidence.read_only_protected);
  EXPECT_EQ(evidence.mprotect_errno, EACCES);
  EXPECT_EQ(state.protect_calls, 2u);

  EXPECT_FALSE(protect_buffer_read_only_with_evidence(evidence, 0));
  EXPECT_FALSE(evidence.read_only_protected);
  EXPECT_EQ(evidence.mprotect_errno, EINVAL);
  EXPECT_EQ(state.protect_calls, 2u);

  AllocatedBufferEvidence empty_evidence;
  EXPECT_FALSE(protect_buffer_read_only_with_evidence(empty_evidence, 4096));
  EXPECT_FALSE(empty_evidence.read_only_protected);
  EXPECT_EQ(empty_evidence.mprotect_errno, EINVAL);
  EXPECT_EQ(state.protect_calls, 2u);
}
