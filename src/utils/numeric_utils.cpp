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
//

/**
 * @file numeric_utils.cpp
 * @brief Overflow-safe size arithmetic and duration calibration primitives
 */

#include "utils/numeric_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace NumericUtils {

bool checked_add(size_t lhs, size_t rhs, size_t& out) noexcept {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  out = lhs + rhs;
  return true;
}

bool checked_multiply(size_t lhs, size_t rhs, size_t& out) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  out = lhs * rhs;
  return true;
}

bool checked_round_up(size_t value, size_t quantum, size_t& out) noexcept {
  if (quantum == 0) {
    return false;
  }
  const size_t remainder = value % quantum;
  if (remainder == 0) {
    out = value;
    return true;
  }
  return checked_add(value, quantum - remainder, out);
}

size_t saturating_add(size_t lhs, size_t rhs) noexcept {
  size_t result = 0;
  return checked_add(lhs, rhs, result) ? result
                                       : std::numeric_limits<size_t>::max();
}

size_t saturating_multiply(size_t lhs, size_t rhs) noexcept {
  size_t result = 0;
  return checked_multiply(lhs, rhs, result)
             ? result
             : std::numeric_limits<size_t>::max();
}

size_t saturating_round_up(size_t value, size_t quantum) noexcept {
  if (quantum == 0) {
    return 0;
  }
  size_t result = 0;
  return checked_round_up(value, quantum, result)
             ? result
             : std::numeric_limits<size_t>::max();
}

size_t calculate_minimum_pilot_count(size_t work_per_count,
                                     size_t minimum_pilot_work,
                                     size_t maximum_count) noexcept {
  if (work_per_count == 0 || maximum_count == 0) {
    return 0;
  }
  const size_t quotient = minimum_pilot_work / work_per_count;
  const size_t remainder = minimum_pilot_work % work_per_count;
  const size_t required = quotient + (remainder != 0 ? 1 : 0);
  return std::min(maximum_count, std::max<size_t>(1, required));
}

size_t calculate_duration_scaled_count(double pilot_duration_seconds,
                                       size_t pilot_count,
                                       double target_duration_seconds,
                                       size_t minimum_count,
                                       size_t maximum_count,
                                       size_t quantum) noexcept {
  if (!std::isfinite(pilot_duration_seconds) ||
      pilot_duration_seconds <= 0.0 || pilot_count == 0 ||
      !std::isfinite(target_duration_seconds) ||
      target_duration_seconds <= 0.0 || minimum_count == 0 ||
      maximum_count < minimum_count || quantum == 0) {
    return 0;
  }

  const long double scaled =
      static_cast<long double>(pilot_count) * target_duration_seconds /
      pilot_duration_seconds;
  size_t count = scaled >= static_cast<long double>(maximum_count)
                     ? maximum_count
                     : static_cast<size_t>(std::ceil(scaled));
  count = std::max(count, minimum_count);

  size_t quantized_count = 0;
  if (!checked_round_up(count, quantum, quantized_count) ||
      quantized_count > maximum_count) {
    const size_t maximum_quantized =
        maximum_count - maximum_count % quantum;
    return maximum_quantized >= minimum_count ? maximum_quantized : 0;
  }
  return quantized_count;
}

}  // namespace NumericUtils
