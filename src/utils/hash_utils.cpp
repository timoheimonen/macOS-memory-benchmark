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
 * @file hash_utils.cpp
 * @brief CommonCrypto-backed SHA-256 helper implementation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "utils/hash_utils.h"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <limits>
#include <stdexcept>

namespace HashUtils {

std::string sha256_hex(std::string_view input) {
  if (input.size() > static_cast<size_t>(std::numeric_limits<CC_LONG>::max())) {
    throw std::length_error("SHA-256 input exceeds CommonCrypto's one-shot length limit");
  }

  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  const char* bytes = input.empty() ? "" : input.data();
  if (CC_SHA256(bytes, static_cast<CC_LONG>(input.size()), digest.data()) == nullptr) {
    throw std::runtime_error("CommonCrypto failed to produce a SHA-256 digest");
  }

  constexpr char kLowercaseHex[] = "0123456789abcdef";
  std::string encoded(digest.size() * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    const unsigned char value = digest[index];
    encoded[index * 2] = kLowercaseHex[value >> 4U];
    encoded[index * 2 + 1] = kLowercaseHex[value & 0x0fU];
  }
  return encoded;
}

}  // namespace HashUtils
