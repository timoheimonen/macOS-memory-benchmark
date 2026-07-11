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
 * @file hash_utils.h
 * @brief SHA-256 helpers backed by the macOS CommonCrypto API
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef HASH_UTILS_H
#define HASH_UTILS_H

#include <string>
#include <string_view>

namespace HashUtils {

/**
 * @brief Hash an exact byte sequence and encode its SHA-256 digest as lowercase hexadecimal.
 *
 * The function hashes all bytes represented by `input`, including embedded NUL bytes. It performs no text encoding,
 * normalization, or terminator handling, so callers can hash canonical UTF-8 source bytes without transformation.
 * Empty input is valid. The implementation has no shared mutable state and is safe for concurrent calls.
 *
 * @param input Exact bytes to hash.
 * @return A canonical 64-character lowercase hexadecimal SHA-256 digest.
 * @throws std::length_error If the input length cannot be represented by CommonCrypto's one-shot API.
 * @throws std::runtime_error If CommonCrypto unexpectedly fails to produce a digest.
 */
std::string sha256_hex(std::string_view input);

}  // namespace HashUtils

#endif  // HASH_UTILS_H
