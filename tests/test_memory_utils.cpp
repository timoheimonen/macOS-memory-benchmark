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
#include <gtest/gtest.h>
#include "core/memory/memory_utils.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace {

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t byte_count)
      : storage_((byte_count + sizeof(uintptr_t) - 1) / sizeof(uintptr_t), 0) {}

  void* data() { return storage_.data(); }

 private:
  std::vector<uintptr_t> storage_;
};

class MemoryUtilsTest : public testing::Test {
 protected:
  void SetUp() override {
    hooks_.page_size_bytes = page_size_bytes;
    hooks_.generated_seed = 0x123456789abcdef0ULL;
    set_memory_utils_test_hooks(&hooks_);
  }

  void TearDown() override { set_memory_utils_test_hooks(nullptr); }

  static constexpr size_t page_size_bytes = 16 * Constants::BYTES_PER_KB;

 private:
  MemoryUtilsTestHooks hooks_;
};

}  // namespace

// Test setup_latency_chain with null buffer - should fail with error message
TEST_F(MemoryUtilsTest, SetupLatencyChainNullBuffer) {
  using namespace Constants;
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(nullptr, LATENCY_STRIDE_BYTES * 2, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message contains expected content
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
  EXPECT_NE(error_output.find("null"), std::string::npos);
}

// Test setup_latency_chain with zero stride - should fail with error message
TEST_F(MemoryUtilsTest, SetupLatencyChainZeroStride) {
  using namespace Constants;
  
  // Allocate a buffer for testing
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;
  AlignedBuffer buffer(buffer_size);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, 0);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message contains expected content
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
  EXPECT_NE(error_output.find("zero"), std::string::npos);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainRejectsTooSmallAndUnalignedStrides) {
  std::vector<uintptr_t> storage(16, 0);
  const size_t invalid_strides[] = {
      1, sizeof(uintptr_t) - 1, sizeof(uintptr_t) + 1};

  for (const size_t stride : invalid_strides) {
    SCOPED_TRACE(stride);
    testing::internal::CaptureStderr();
    const int result = setup_latency_chain(
        storage.data(), storage.size() * sizeof(uintptr_t), stride);
    const std::string error_output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_NE(error_output.find(
                  Messages::error_prefix() +
                  Messages::error_latency_stride_alignment(
                      stride, sizeof(uintptr_t))),
              std::string::npos);
  }
}

TEST_F(MemoryUtilsTest, SetupLatencyChainAcceptsPointerSizedStride) {
  std::vector<uintptr_t> storage(2, 0);
  ASSERT_EQ(setup_latency_chain(storage.data(), storage.size() * sizeof(uintptr_t),
                                sizeof(uintptr_t)),
            EXIT_SUCCESS);

  const uintptr_t first = reinterpret_cast<uintptr_t>(storage.data());
  const uintptr_t second = first + sizeof(uintptr_t);
  EXPECT_EQ(storage[0], second);
  EXPECT_EQ(storage[1], first);
}

// Test setup_latency_chain with buffer smaller than stride - should fail
TEST_F(MemoryUtilsTest, SetupLatencyChainBufferSmallerThanStride) {
  using namespace Constants;
  
  // Allocate buffer smaller than stride
  size_t buffer_size = LATENCY_STRIDE_BYTES - 1;  // < stride
  AlignedBuffer buffer(buffer_size);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail (num_pointers = buffer_size / stride = 0, need at least 2)
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with buffer equal to stride - should fail (num_pointers == 1)
TEST_F(MemoryUtilsTest, SetupLatencyChainBufferEqualToStride) {
  using namespace Constants;
  
  // Allocate buffer equal to stride
  size_t buffer_size = LATENCY_STRIDE_BYTES;  // == stride
  AlignedBuffer buffer(buffer_size);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail (num_pointers = stride / stride = 1, need at least 2)
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with buffer just larger than stride - boundary case
TEST_F(MemoryUtilsTest, SetupLatencyChainBufferJustLargerThanStride) {
  using namespace Constants;
  
  // Allocate buffer just larger than stride
  // Need at least 2 pointers, so buffer_size >= stride * 2
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;  // minimum valid
  AlignedBuffer buffer(buffer_size);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed (num_pointers = 2, minimum valid)
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // No error should be logged
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with num_pointers == 3 (valid case)
TEST_F(MemoryUtilsTest, SetupLatencyChainThreePointers) {
  using namespace Constants;
  
  // Allocate buffer for 3 pointers
  size_t buffer_size = LATENCY_STRIDE_BYTES * 3;
  AlignedBuffer buffer(buffer_size);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // No error should be logged
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test that setup_latency_chain creates a valid linked list
TEST_F(MemoryUtilsTest, SetupLatencyChainCreatesValidChain) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES * 4;  // 4 pointers
  AlignedBuffer buffer(buffer_size);
  
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES);
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  const size_t pointer_count = buffer_size / LATENCY_STRIDE_BYTES;
  const uintptr_t buffer_start = reinterpret_cast<uintptr_t>(buffer.data());
  const uintptr_t buffer_end = buffer_start + buffer_size;
  std::vector<bool> visited(pointer_count, false);

  uintptr_t current = buffer_start;
  for (size_t step = 0; step < pointer_count; ++step) {
    ASSERT_GE(current, buffer_start);
    ASSERT_LT(current, buffer_end);
    const size_t offset = static_cast<size_t>(current - buffer_start);
    ASSERT_EQ(offset % LATENCY_STRIDE_BYTES, 0u);
    const size_t index = offset / LATENCY_STRIDE_BYTES;
    ASSERT_LT(index, pointer_count);
    EXPECT_FALSE(visited[index]) << "chain repeated node at step " << step;
    visited[index] = true;
    current = *reinterpret_cast<const uintptr_t*>(current);
  }

  EXPECT_EQ(current, buffer_start);
  for (size_t index = 0; index < pointer_count; ++index) {
    EXPECT_TRUE(visited[index]) << "chain omitted node " << index;
  }
}

TEST_F(MemoryUtilsTest, SetupLatencyChainWithTlbLocality) {
  using namespace Constants;

  size_t buffer_size = LATENCY_STRIDE_BYTES * 128;
  AlignedBuffer buffer(buffer_size);

  const size_t locality_bytes = page_size_bytes;
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES, locality_bytes);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainCollectsDiagnostics) {
  using namespace Constants;

  const size_t page_size = page_size_bytes;
  const size_t buffer_size = page_size * 4;
  const size_t stride = sizeof(uintptr_t) * 8;
  AlignedBuffer buffer(buffer_size);

  LatencyChainDiagnostics diagnostics;
  int result = setup_latency_chain(buffer.data(), buffer_size, stride, 0, &diagnostics);
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(diagnostics.pointer_count, buffer_size / stride);
  EXPECT_GT(diagnostics.unique_pages_touched, 0u);
  EXPECT_LE(diagnostics.unique_pages_touched, 4u);
  EXPECT_EQ(diagnostics.page_size_bytes, page_size);
  EXPECT_EQ(diagnostics.stride_bytes, stride);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainWithTooSmallTlbLocalityFails) {
  using namespace Constants;

  size_t buffer_size = LATENCY_STRIDE_BYTES * 16;
  AlignedBuffer buffer(buffer_size);

  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainWithSameRandomInBoxMode) {
  using namespace Constants;

  size_t buffer_size = LATENCY_STRIDE_BYTES * 128;
  AlignedBuffer buffer(buffer_size);

  const size_t locality_bytes = page_size_bytes;
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES,
                                   locality_bytes, nullptr,
                                   LatencyChainMode::SameRandomInBoxIncreasingBox);
  EXPECT_EQ(result, EXIT_SUCCESS);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainWithBoxModeAndZeroLocalityFails) {
  using namespace Constants;

  size_t buffer_size = LATENCY_STRIDE_BYTES * 128;
  AlignedBuffer buffer(buffer_size);

  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES,
                                   0, nullptr,
                                   LatencyChainMode::DiffRandomInBoxIncreasingBox);
  std::string error_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error_output.find("latency-chain-mode"), std::string::npos);
}

TEST_F(MemoryUtilsTest, SetupLatencyChainExplicitSeedIsReproducible) {
  const size_t stride = 64;
  const size_t page_size = page_size_bytes;
  const size_t buffer_size = 4 * page_size;
  AlignedBuffer buffer(buffer_size);

  auto snapshot_next_indices = [&]() {
    std::vector<size_t> next_indices;
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.data());
    for (size_t offset = 0; offset < buffer_size; offset += stride) {
      const uintptr_t next = *reinterpret_cast<uintptr_t*>(
          static_cast<char*>(buffer.data()) + offset);
      next_indices.push_back((next - base) / stride);
    }
    return next_indices;
  };

  ASSERT_EQ(setup_latency_chain(buffer.data(),
                                buffer_size,
                                stride,
                                page_size,
                                nullptr,
                                LatencyChainMode::RandomInBoxRandomBox,
                                uint64_t{12345}),
            EXIT_SUCCESS);
  const std::vector<size_t> first = snapshot_next_indices();

  ASSERT_EQ(setup_latency_chain(buffer.data(),
                                buffer_size,
                                stride,
                                page_size,
                                nullptr,
                                LatencyChainMode::RandomInBoxRandomBox,
                                uint64_t{12345}),
            EXIT_SUCCESS);
  const std::vector<size_t> second = snapshot_next_indices();

  ASSERT_EQ(setup_latency_chain(buffer.data(),
                                buffer_size,
                                stride,
                                page_size,
                                nullptr,
                                LatencyChainMode::RandomInBoxRandomBox,
                                uint64_t{54321}),
            EXIT_SUCCESS);
  const std::vector<size_t> different = snapshot_next_indices();

  EXPECT_EQ(first, second);
  EXPECT_NE(first, different);
}
