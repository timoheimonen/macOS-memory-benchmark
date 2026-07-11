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

TEST(HashUtilsTest, EmptyInputMatchesStandardSha256Vector) {
  EXPECT_EQ(HashUtils::sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
}

TEST(HashUtilsTest, AbcMatchesStandardSha256Vector) {
  EXPECT_EQ(HashUtils::sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
}

TEST(HashUtilsTest, CanonicalGpuSourceBytesMatchLockedRevisionHash) {
  EXPECT_EQ(canonical_gpu_kernel_source_sha256(),
            "b9a242d2b959c9c11f6f130a52afd66f"
            "111d6761be2193beec1f051baa094296");
}
