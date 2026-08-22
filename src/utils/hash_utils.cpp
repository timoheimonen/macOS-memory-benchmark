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

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace HashUtils {
namespace {

std::string encode_digest(
    const std::array<unsigned char, CC_SHA256_DIGEST_LENGTH>& digest) {
  constexpr char kLowercaseHex[] = "0123456789abcdef";
  std::string encoded(digest.size() * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    const unsigned char value = digest[index];
    encoded[index * 2] = kLowercaseHex[value >> 4U];
    encoded[index * 2 + 1] = kLowercaseHex[value & 0x0fU];
  }
  return encoded;
}

}  // namespace

struct Sha256Hasher::State {
  CC_SHA256_CTX context{};
  bool active = false;

  State() {
    if (CC_SHA256_Init(&context) != 1) {
      throw std::runtime_error("CommonCrypto failed to initialize SHA-256");
    }
    active = true;
  }
};

Sha256Hasher::Sha256Hasher() : state_(std::make_unique<State>()) {}

Sha256Hasher::~Sha256Hasher() = default;

Sha256Hasher::Sha256Hasher(Sha256Hasher&&) noexcept = default;

Sha256Hasher& Sha256Hasher::operator=(Sha256Hasher&&) noexcept = default;

void Sha256Hasher::update(const void* data, size_t size) {
  if (!state_ || !state_->active) {
    throw std::logic_error("SHA-256 hasher is not active");
  }
  if (size == 0) {
    return;
  }
  if (data == nullptr) {
    throw std::invalid_argument("SHA-256 input is null with a nonzero size");
  }

  const auto* cursor = static_cast<const unsigned char*>(data);
  constexpr size_t kMaximumUpdateBytes =
      static_cast<size_t>(std::numeric_limits<CC_LONG>::max());
  while (size > 0) {
    const size_t chunk_size = std::min(size, kMaximumUpdateBytes);
    if (CC_SHA256_Update(&state_->context, cursor,
                         static_cast<CC_LONG>(chunk_size)) != 1) {
      state_->active = false;
      throw std::runtime_error("CommonCrypto failed to update SHA-256");
    }
    cursor += chunk_size;
    size -= chunk_size;
  }
}

void Sha256Hasher::update(std::string_view input) {
  update(input.data(), input.size());
}

std::string Sha256Hasher::finalize_hex() {
  if (!state_ || !state_->active) {
    throw std::logic_error("SHA-256 hasher is not active");
  }

  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  state_->active = false;
  if (CC_SHA256_Final(digest.data(), &state_->context) != 1) {
    throw std::runtime_error("CommonCrypto failed to finalize SHA-256");
  }
  return encode_digest(digest);
}

std::string sha256_hex(std::string_view input) {
  Sha256Hasher hasher;
  hasher.update(input);
  return hasher.finalize_hex();
}

}  // namespace HashUtils
