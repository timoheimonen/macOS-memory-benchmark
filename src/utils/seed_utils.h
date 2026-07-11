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
 * @file seed_utils.h
 * @brief Deterministic seed-mixing primitives
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef SEED_UTILS_H
#define SEED_UTILS_H

#include <cstdint>
#include <functional>

namespace SeedUtils {

/**
 * @brief Optional deterministic source used by tests and mode-specific parsers.
 *
 * A provider returning zero is treated as unavailable so generated seeds retain
 * the production non-zero contract.
 */
using SeedGenerationProvider = std::function<uint64_t()>;

/**
 * @brief Generate one non-zero seed for a benchmark command.
 *
 * Production generation combines two `std::random_device` values and falls
 * back to a monotonic-clock value when entropy is unavailable. A supplied
 * provider is invoked exactly once and its non-zero result is returned without
 * modification.
 *
 * @param provider Optional deterministic seed provider.
 * @return A non-zero 64-bit seed.
 * @throws Any exception raised by an injected provider. Production entropy
 *         failures are caught and converted to the monotonic-clock fallback.
 */
uint64_t generate_seed(const SeedGenerationProvider& provider = {});

/**
 * @brief Apply one stateless SplitMix64 output transform.
 *
 * Unsigned wraparound, including the initial Weyl-sequence increment, is
 * intentional and forms part of the deterministic seed contract.
 *
 * @param value Input value to mix.
 * @return Deterministically mixed 64-bit value.
 */
uint64_t splitmix64(uint64_t value) noexcept;

}  // namespace SeedUtils

#endif  // SEED_UTILS_H
