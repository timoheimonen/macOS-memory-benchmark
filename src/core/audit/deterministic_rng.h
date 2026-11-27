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
 * @file deterministic_rng.h
 * @brief Shared SplitMix64 and rejection-sampled Fisher-Yates primitives
 */

#ifndef CORE_AUDIT_DETERMINISTIC_RNG_H
#define CORE_AUDIT_DETERMINISTIC_RNG_H

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

inline constexpr const char* kSplitMix64FisherYatesRejectionVersion =
    "splitmix64-fisher-yates-rejection-v1";

inline uint64_t splitmix64_next(uint64_t& state) noexcept {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

inline uint64_t splitmix64_mix(uint64_t x) noexcept {
  return splitmix64_next(x);
}

inline uint64_t splitmix64_bounded_draw(uint64_t& state,
                                        uint64_t bound) noexcept {
  if (bound <= 1) {
    return 0;
  }
  const uint64_t threshold = (uint64_t{0} - bound) % bound;
  uint64_t draw = 0;
  do {
    draw = splitmix64_next(state);
  } while (draw < threshold);
  return draw % bound;
}

template <typename T>
inline void fisher_yates_rejection_shuffle_with_state(std::vector<T>& items,
                                                      uint64_t& state) {
  for (size_t index = items.size(); index > 1; --index) {
    const size_t swap_index = static_cast<size_t>(
        splitmix64_bounded_draw(state, static_cast<uint64_t>(index)));
    std::swap(items[index - 1], items[swap_index]);
  }
}

template <typename T>
inline void fisher_yates_rejection_shuffle(std::vector<T>& items,
                                           uint64_t seed) {
  uint64_t state = seed;
  fisher_yates_rejection_shuffle_with_state(items, state);
}

#endif  // CORE_AUDIT_DETERMINISTIC_RNG_H
