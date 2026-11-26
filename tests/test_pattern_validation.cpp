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

TEST(PatternValidationTest, StrideValidatesBufferBounds) {
  using namespace Constants;
  const struct {
    size_t stride;
    size_t buffer_size;
    bool valid;
  } cases[] = {
      {PATTERN_MIN_BUFFER_SIZE_BYTES, PATTERN_MIN_BUFFER_SIZE_BYTES, true},
      {PATTERN_STRIDE_CACHE_LINE, PATTERN_STRIDE_CACHE_LINE, true},
      {PATTERN_STRIDE_PAGE, PATTERN_STRIDE_PAGE + PATTERN_ACCESS_SIZE_BYTES, true},
      {PATTERN_STRIDE_PAGE, PATTERN_STRIDE_PAGE - 1, false},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.buffer_size);
    EXPECT_EQ(validate_stride(test_case.stride, test_case.buffer_size), test_case.valid);
  }
}

TEST(PatternValidationTest, RandomIndicesAcceptEveryAlignedExactlyFittingAccess) {
  constexpr size_t kAccess = Constants::PATTERN_ACCESS_SIZE_BYTES;
  EXPECT_TRUE(validate_random_indices({0, kAccess, 2 * kAccess}, 3 * kAccess));
}

TEST(PatternValidationTest, RandomIndicesRejectEmptyMisalignedOutOfBoundsAndLateErrors) {
  constexpr size_t kSmallBuffer = 64;
  constexpr size_t kFormerSamplingLimit = 100;
  constexpr size_t kSizeMax = std::numeric_limits<size_t>::max();
  std::vector<size_t> late_invalid(kFormerSamplingLimit + 1, 0);
  late_invalid.back() = kSizeMax;

  struct RejectionCase {
    const char* name;
    std::vector<size_t> indices;
    size_t buffer_size;
    std::string expected_message;
  };
  const std::vector<RejectionCase> cases = {
      {"empty input", {}, 1024, Messages::error_indices_empty()},
      {"misaligned offset", {0, 1}, 1024, Messages::error_index_not_aligned(1, 1)},
      {"exactly past end", {kSmallBuffer}, kSmallBuffer,
       Messages::error_index_out_of_bounds(0, kSmallBuffer, kSmallBuffer)},
      {"size_t max overflow", {kSizeMax}, kSmallBuffer, Messages::error_index_out_of_bounds(0, kSizeMax, kSmallBuffer)},
      {"beyond former sampling limit", late_invalid, 1024,
       Messages::error_index_out_of_bounds(late_invalid.size() - 1, kSizeMax, 1024)},
  };

  for (const RejectionCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    bool result = true;
    const std::string output = capture_stderr([&] {
      result = validate_random_indices(test_case.indices, test_case.buffer_size);
    });

    EXPECT_FALSE(result);
    EXPECT_EQ(output, Messages::error_prefix() + test_case.expected_message + "\n");
  }
}
