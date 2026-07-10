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

/**
 * @brief Apply one stateless SplitMix64 output transform.
 *
 * Unsigned wraparound, including the initial Weyl-sequence increment, is
 * intentional and forms part of the deterministic seed contract.
 *
 * @param value Input value to mix
 * @return Deterministically mixed 64-bit value
 */
namespace SeedUtils {

uint64_t splitmix64(uint64_t value) noexcept;

}  // namespace SeedUtils

#endif  // SEED_UTILS_H
