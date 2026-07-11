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

#include <limits>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include "pattern_benchmark/pattern_benchmark.h"

namespace {

template <typename Callable>
std::string capture_stderr(Callable&& callable) {
  testing::internal::CaptureStderr();
  callable();
  return testing::internal::GetCapturedStderr();
}

}  // namespace

TEST(PatternValidationTest, StrideAcceptsSupportedBounds) {
  using namespace Constants;
  const struct {
    size_t stride;
    size_t buffer_size;
  } cases[] = {
      {PATTERN_MIN_BUFFER_SIZE_BYTES, PATTERN_MIN_BUFFER_SIZE_BYTES},
      {PATTERN_STRIDE_CACHE_LINE, PATTERN_STRIDE_CACHE_LINE},
      {PATTERN_STRIDE_PAGE, PATTERN_STRIDE_PAGE + PATTERN_ACCESS_SIZE_BYTES},
  };

  for (const auto& test_case : cases) {
    EXPECT_TRUE(validate_stride(test_case.stride, test_case.buffer_size));
  }
}

TEST(PatternValidationTest, StrideRejectsTooSmallValueWithCentralizedReason) {
  bool result = true;
  const std::string output = capture_stderr([&] {
    result = validate_stride(Constants::PATTERN_MIN_BUFFER_SIZE_BYTES - 1,
                             Constants::PATTERN_MIN_BUFFER_SIZE_BYTES);
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(output, Messages::error_prefix() + Messages::error_stride_too_small() + "\n");
}

TEST(PatternValidationTest, StrideRejectsValueBeyondBufferWithoutBenchmarkExecution) {
  EXPECT_FALSE(validate_stride(Constants::PATTERN_STRIDE_PAGE,
                               Constants::PATTERN_STRIDE_PAGE - 1));
}

TEST(PatternValidationTest, RandomIndicesAcceptEveryAlignedExactlyFittingAccess) {
  constexpr size_t kAccess = Constants::PATTERN_ACCESS_SIZE_BYTES;
  EXPECT_TRUE(validate_random_indices({0, kAccess, 2 * kAccess}, 3 * kAccess));
}

TEST(PatternValidationTest, RandomIndicesRejectEmptyInputWithCentralizedReason) {
  bool result = true;
  const std::string output = capture_stderr([&] {
    result = validate_random_indices({}, 1024);
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(output, Messages::error_prefix() + Messages::error_indices_empty() + "\n");
}

TEST(PatternValidationTest, RandomIndicesRejectMisalignedOffsetWithExactIndex) {
  bool result = true;
  const std::string output = capture_stderr([&] {
    result = validate_random_indices({0, 1}, 1024);
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(output, Messages::error_prefix() + Messages::error_index_not_aligned(1, 1) + "\n");
}

TEST(PatternValidationTest, RandomIndicesRejectExactlyPastEndAndSizeMaxWithoutOverflow) {
  constexpr size_t kBufferSize = 64;
  for (const size_t offset : {kBufferSize, std::numeric_limits<size_t>::max()}) {
    bool result = true;
    const std::string output = capture_stderr([&] {
      result = validate_random_indices({offset}, kBufferSize);
    });

    EXPECT_FALSE(result);
    EXPECT_EQ(output, Messages::error_prefix() +
                          Messages::error_index_out_of_bounds(0, offset, kBufferSize) + "\n");
  }
}

TEST(PatternValidationTest, RandomIndicesValidateItemsBeyondFormerSamplingLimit) {
  constexpr size_t kFormerSamplingLimit = 100;
  std::vector<size_t> indices(kFormerSamplingLimit + 1, 0);
  indices.back() = std::numeric_limits<size_t>::max();

  bool result = true;
  const std::string output = capture_stderr([&] {
    result = validate_random_indices(indices, 1024);
  });

  EXPECT_FALSE(result);
  const size_t invalid_index = indices.size() - 1;
  EXPECT_EQ(output, Messages::error_prefix() +
                        Messages::error_index_out_of_bounds(
                            invalid_index, indices.back(), 1024) +
                        "\n");
}
