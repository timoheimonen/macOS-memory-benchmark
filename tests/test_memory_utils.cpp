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

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "core/memory/memory_utils.h"
#include "output/console/messages/messages_api.h"

namespace {

class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t byte_count) : storage_((byte_count + sizeof(uintptr_t) - 1) / sizeof(uintptr_t), 0) {}

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

std::vector<size_t> snapshot_next_indices(void* buffer, size_t buffer_size, size_t stride) {
  std::vector<size_t> next_indices;
  const uintptr_t base = reinterpret_cast<uintptr_t>(buffer);
  for (size_t offset = 0; offset < buffer_size; offset += stride) {
    const uintptr_t next = *reinterpret_cast<const uintptr_t*>(static_cast<const char*>(buffer) + offset);
    next_indices.push_back((next - base) / stride);
  }
  return next_indices;
}

void expect_same_permutation_per_box(const std::vector<size_t>& next_indices, size_t pointers_per_box) {
  ASSERT_GT(pointers_per_box, 0u);
  ASSERT_EQ(next_indices.size() % pointers_per_box, 0u);
  const size_t box_count = next_indices.size() / pointers_per_box;
  ASSERT_GT(box_count, 1u);

  std::vector<size_t> expected_permutation;
  for (size_t box = 0; box < box_count; ++box) {
    size_t entry = next_indices.size();
    size_t incoming_cross_box_edges = 0;
    for (size_t source = 0; source < next_indices.size(); ++source) {
      const size_t target = next_indices[source];
      ASSERT_LT(target, next_indices.size());
      if (target / pointers_per_box == box && source / pointers_per_box != box) {
        entry = target;
        ++incoming_cross_box_edges;
      }
    }
    ASSERT_EQ(incoming_cross_box_edges, 1u) << "box " << box;

    std::vector<size_t> permutation;
    std::vector<bool> visited(pointers_per_box, false);
    size_t current = entry;
    for (size_t position = 0; position < pointers_per_box; ++position) {
      ASSERT_EQ(current / pointers_per_box, box) << "box " << box << " position " << position;
      const size_t offset = current % pointers_per_box;
      EXPECT_FALSE(visited[offset]) << "box " << box << " repeated offset " << offset;
      visited[offset] = true;
      permutation.push_back(offset);
      current = next_indices[current];
    }
    EXPECT_EQ(current / pointers_per_box, (box + 1) % box_count);

    if (box == 0) {
      expected_permutation = permutation;
    } else {
      EXPECT_EQ(permutation, expected_permutation) << "box " << box;
    }
  }
}

}  // namespace

TEST_F(MemoryUtilsTest, SetupLatencyChainRejectsInvalidInputsWithExactReasons) {
  using namespace Constants;

  std::vector<uintptr_t> storage(16, 0);
  struct InvalidInputCase {
    const char* name;
    bool null_buffer;
    size_t buffer_size;
    size_t stride;
    std::string expected_reason;
  };
  const InvalidInputCase cases[] = {
      {"null buffer", true, LATENCY_STRIDE_BYTES * 2, LATENCY_STRIDE_BYTES,
       Messages::error_buffer_pointer_null_latency_chain()},
      {"zero stride", false, LATENCY_STRIDE_BYTES * 2, 0, Messages::error_stride_zero_latency_chain()},
      {"one byte stride", false, storage.size() * sizeof(uintptr_t), 1,
       Messages::error_latency_stride_alignment(1, sizeof(uintptr_t))},
      {"sub-pointer stride", false, storage.size() * sizeof(uintptr_t), sizeof(uintptr_t) - 1,
       Messages::error_latency_stride_alignment(sizeof(uintptr_t) - 1, sizeof(uintptr_t))},
      {"unaligned stride", false, storage.size() * sizeof(uintptr_t), sizeof(uintptr_t) + 1,
       Messages::error_latency_stride_alignment(sizeof(uintptr_t) + 1, sizeof(uintptr_t))},
      {"one-node buffer", false, LATENCY_STRIDE_BYTES, LATENCY_STRIDE_BYTES,
       Messages::error_buffer_stride_invalid_latency_chain(1, LATENCY_STRIDE_BYTES, LATENCY_STRIDE_BYTES)},
  };

  for (const InvalidInputCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    testing::internal::CaptureStderr();
    const int result =
        setup_latency_chain(test_case.null_buffer ? nullptr : storage.data(), test_case.buffer_size, test_case.stride);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(error, Messages::error_prefix() + test_case.expected_reason + "\n");
  }
}

TEST_F(MemoryUtilsTest, SetupLatencyChainAcceptsMinimumTwoNodeChainsAtSupportedStrides) {
  using namespace Constants;

  for (const size_t stride : {sizeof(uintptr_t), LATENCY_STRIDE_BYTES}) {
    SCOPED_TRACE(stride);
    const size_t buffer_size = stride * 2;
    AlignedBuffer buffer(buffer_size);
    ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, stride), EXIT_SUCCESS);

    const uintptr_t first = reinterpret_cast<uintptr_t>(buffer.data());
    const uintptr_t second = first + stride;
    EXPECT_EQ(*reinterpret_cast<const uintptr_t*>(buffer.data()), second);
    EXPECT_EQ(*reinterpret_cast<const uintptr_t*>(static_cast<const char*>(buffer.data()) + stride), first);
  }
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
  const uintptr_t first_node = reinterpret_cast<uintptr_t>(buffer.data());
  const uintptr_t last_node = first_node + (diagnostics.pointer_count - 1) * stride;
  const size_t expected_pages = last_node / page_size - first_node / page_size + 1;
  EXPECT_EQ(diagnostics.unique_pages_touched, expected_pages);
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

TEST_F(MemoryUtilsTest, SetupLatencyChainWithBoxModeAndZeroLocalityFails) {
  using namespace Constants;

  size_t buffer_size = LATENCY_STRIDE_BYTES * 128;
  AlignedBuffer buffer(buffer_size);

  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.data(), buffer_size, LATENCY_STRIDE_BYTES, 0, nullptr,
                                   LatencyChainMode::DiffRandomInBoxIncreasingBox);
  std::string error_output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error_output.find("latency-chain-mode"), std::string::npos);
}

TEST_F(MemoryUtilsTest, ExplicitSeedsAreReproducibleAndSameRandomModeReusesBoxPermutation) {
  const size_t stride = 64;
  const size_t page_size = page_size_bytes;
  const size_t buffer_size = 4 * page_size;
  AlignedBuffer buffer(buffer_size);

  ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, stride, page_size, nullptr,
                                LatencyChainMode::RandomInBoxRandomBox, uint64_t{12345}),
            EXIT_SUCCESS);
  const std::vector<size_t> first = snapshot_next_indices(buffer.data(), buffer_size, stride);

  ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, stride, page_size, nullptr,
                                LatencyChainMode::RandomInBoxRandomBox, uint64_t{12345}),
            EXIT_SUCCESS);
  const std::vector<size_t> second = snapshot_next_indices(buffer.data(), buffer_size, stride);

  ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, stride, page_size, nullptr,
                                LatencyChainMode::RandomInBoxRandomBox, uint64_t{54321}),
            EXIT_SUCCESS);
  const std::vector<size_t> different = snapshot_next_indices(buffer.data(), buffer_size, stride);

  EXPECT_EQ(first, second);
  EXPECT_NE(first, different);

  ASSERT_EQ(setup_latency_chain(buffer.data(), buffer_size, stride, page_size, nullptr,
                                LatencyChainMode::SameRandomInBoxIncreasingBox, uint64_t{12345}),
            EXIT_SUCCESS);
  expect_same_permutation_per_box(snapshot_next_indices(buffer.data(), buffer_size, stride), page_size / stride);
}
