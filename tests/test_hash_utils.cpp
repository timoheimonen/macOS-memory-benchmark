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

/**
 * @file test_hash_utils.cpp
 * @brief Deterministic standard-vector tests for SHA-256 helpers
 */

#include <gtest/gtest.h>

#include "gpu_bandwidth/gpu_backend.h"
#include "utils/hash_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

TEST(HashUtilsTest, AllocatingAndNoallocHelpersMatchKnownSha256Vectors) {
  struct Vector {
    std::string input;
    const char* digest;
  };
  const Vector vectors[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {std::string{"prefix\0middle\xffsuffix", 20}, "fa5e64c9d422596327e5747f3a0b502b4bcf9f8ae8ab984652d48e88f501190c"},
  };
  for (const Vector& entry : vectors) {
    SCOPED_TRACE(::testing::PrintToString(entry.input));
    EXPECT_EQ(HashUtils::sha256_hex(entry.input), entry.digest);
    std::array<char, 64> digest{};
    ASSERT_TRUE(HashUtils::sha256_hex_noalloc(entry.input, digest));
    EXPECT_EQ(std::string_view(digest.data(), digest.size()), entry.digest);
  }
}

TEST(HashUtilsTest, IncrementalAndOneShotResultsMatchForEverySplitWithEmbeddedNulls) {
  const std::string payload{"prefix\0middle\xffsuffix", 20};
  const std::string expected = HashUtils::sha256_hex(payload);

  for (size_t split = 0; split <= payload.size(); ++split) {
    SCOPED_TRACE(split);
    HashUtils::Sha256Hasher hasher;
    hasher.update(std::string_view(payload).substr(0, split));
    hasher.update(nullptr, 0);
    hasher.update(payload.data() + split, payload.size() - split);
    EXPECT_EQ(hasher.finalize_hex(), expected);
  }
}

TEST(HashUtilsTest, HasherIsMoveOnlyAndMovedStateContinuesTheDigest) {
  static_assert(!std::is_copy_constructible_v<HashUtils::Sha256Hasher>);
  static_assert(!std::is_copy_assignable_v<HashUtils::Sha256Hasher>);
  static_assert(std::is_nothrow_move_constructible_v<HashUtils::Sha256Hasher>);
  static_assert(std::is_nothrow_move_assignable_v<HashUtils::Sha256Hasher>);

  HashUtils::Sha256Hasher source;
  source.update("a");
  HashUtils::Sha256Hasher moved(std::move(source));
  EXPECT_THROW(source.update("ignored"), std::logic_error);
  EXPECT_THROW(static_cast<void>(source.finalize_hex()), std::logic_error);

  HashUtils::Sha256Hasher assigned;
  assigned.update("discarded state");
  assigned = std::move(moved);
  EXPECT_THROW(moved.update("ignored"), std::logic_error);
  assigned.update("bc");
  EXPECT_EQ(assigned.finalize_hex(), HashUtils::sha256_hex("abc"));
}

TEST(HashUtilsTest, FinalizationAndInvalidInputHaveExplicitLifecycleErrors) {
  HashUtils::Sha256Hasher hasher;
  EXPECT_THROW(hasher.update(nullptr, 1), std::invalid_argument);
  hasher.update(std::string_view{});
  EXPECT_EQ(hasher.finalize_hex(), HashUtils::sha256_hex(""));
  EXPECT_THROW(hasher.update("late"), std::logic_error);
  EXPECT_THROW(static_cast<void>(hasher.finalize_hex()), std::logic_error);
}

TEST(HashUtilsTest, CanonicalGpuSourceBytesMatchLockedRevisionHash) {
  EXPECT_EQ(canonical_gpu_kernel_source_sha256(),
            "21def2d75d3545dba31aa4897ea57ec2f"
            "d0e4481cd86ce21725338ab0f322ac5");
}
