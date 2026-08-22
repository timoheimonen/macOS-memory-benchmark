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

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace HashUtils {

/**
 * @brief Single-owner incremental SHA-256 calculation.
 *
 * Updates consume exact bytes without retaining or copying the complete input. One update may exceed CommonCrypto's
 * per-call `CC_LONG` limit; the implementation submits it as bounded internal chunks. Instances are move-only and are
 * not safe for concurrent use. Finalization consumes the active state, after which further updates or finalization
 * throw `std::logic_error`. A moved-from instance has the same inactive behavior.
 */
class Sha256Hasher {
 public:
  Sha256Hasher();
  ~Sha256Hasher();

  Sha256Hasher(const Sha256Hasher&) = delete;
  Sha256Hasher& operator=(const Sha256Hasher&) = delete;
  Sha256Hasher(Sha256Hasher&&) noexcept;
  Sha256Hasher& operator=(Sha256Hasher&&) noexcept;

  /**
   * @brief Add an exact byte range to the digest.
   *
   * A null pointer is accepted only when `size` is zero. Empty updates are valid and leave the active digest state
   * unchanged.
   *
   * @param data First byte, or null for an empty update.
   * @param size Number of bytes to consume.
   * @throws std::invalid_argument If `data` is null and `size` is nonzero.
   * @throws std::logic_error If the hasher is finalized or moved from.
   * @throws std::runtime_error If CommonCrypto rejects an update.
   */
  void update(const void* data, size_t size);

  /**
   * @brief Add all exact bytes represented by a string view.
   *
   * @param input Exact bytes to consume, including embedded NUL bytes.
   * @throws std::logic_error If the hasher is finalized or moved from.
   * @throws std::runtime_error If CommonCrypto rejects an update.
   */
  void update(std::string_view input);

  /**
   * @brief Finalize the digest and encode it as lowercase hexadecimal.
   *
   * @return A canonical 64-character lowercase hexadecimal SHA-256 digest.
   * @throws std::logic_error If the hasher is already finalized or moved from.
   * @throws std::runtime_error If CommonCrypto rejects finalization.
   */
  [[nodiscard]] std::string finalize_hex();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

/**
 * @brief Hash an exact byte sequence and encode its SHA-256 digest as lowercase hexadecimal.
 *
 * The function hashes all bytes represented by `input`, including embedded NUL bytes. It performs no text encoding,
 * normalization, or terminator handling, so callers can hash canonical UTF-8 source bytes without transformation.
 * Empty input is valid. The implementation has no shared mutable state and is safe for concurrent calls.
 *
 * @param input Exact bytes to hash.
 * @return A canonical 64-character lowercase hexadecimal SHA-256 digest.
 * @throws std::runtime_error If CommonCrypto unexpectedly fails to produce a digest.
 */
std::string sha256_hex(std::string_view input);

}  // namespace HashUtils

#endif  // HASH_UTILS_H
