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
 * @file seed_utils.cpp
 * @brief Deterministic seed-mixing primitive implementations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "utils/seed_utils.h"

#include <chrono>
#include <random>

namespace SeedUtils {

uint64_t generate_seed(const SeedGenerationProvider& provider) {
  if (provider) {
    const uint64_t provided_seed = provider();
    if (provided_seed != 0) {
      return provided_seed;
    }
  }

  try {
    std::random_device random_device;
    const uint64_t high = static_cast<uint64_t>(random_device()) << 32U;
    const uint64_t seed = high ^ static_cast<uint64_t>(random_device());
    if (seed != 0) {
      return seed;
    }
  } catch (...) {
    // Fall through to a local monotonic-clock seed if random_device is unavailable.
  }

  const uint64_t clock_seed = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  return clock_seed != 0 ? clock_seed : 0x9e3779b97f4a7c15ULL;
}

uint64_t splitmix64(uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

}  // namespace SeedUtils
