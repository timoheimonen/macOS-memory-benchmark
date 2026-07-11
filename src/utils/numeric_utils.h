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
 * @file numeric_utils.h
 * @brief Overflow-safe size arithmetic and duration calibration primitives
 */

#ifndef NUMERIC_UTILS_H
#define NUMERIC_UTILS_H

#include <cstddef>

namespace NumericUtils {

/**
 * @brief Add two sizes without overflow.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param out Exact sum on success; unchanged on failure.
 * @return true when the exact sum fits in `size_t`.
 */
bool checked_add(size_t lhs, size_t rhs, size_t& out) noexcept;

/**
 * @brief Multiply two sizes without overflow.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @param out Exact product on success; unchanged on failure.
 * @return true when the exact product fits in `size_t`.
 */
bool checked_multiply(size_t lhs, size_t rhs, size_t& out) noexcept;

/**
 * @brief Round a size upward to a positive quantum without overflow.
 * @param value Value to round.
 * @param quantum Required positive multiple.
 * @param out Rounded value on success; unchanged on failure.
 * @return false when `quantum` is zero or the rounded value would overflow.
 */
bool checked_round_up(size_t value, size_t quantum, size_t& out) noexcept;

/** @brief Add two sizes, returning `SIZE_MAX` on overflow. */
size_t saturating_add(size_t lhs, size_t rhs) noexcept;

/** @brief Multiply two sizes, returning `SIZE_MAX` on overflow. */
size_t saturating_multiply(size_t lhs, size_t rhs) noexcept;

/**
 * @brief Round upward to a quantum, returning zero for a zero quantum and
 *        `SIZE_MAX` on overflow.
 */
size_t saturating_round_up(size_t value, size_t quantum) noexcept;

/**
 * @brief Choose the smallest bounded pilot count that meets a work floor.
 *
 * A valid result is at least one even when `minimum_pilot_work` is zero.
 * Zero signals an invalid zero work unit or zero maximum count.
 */
size_t calculate_minimum_pilot_count(size_t work_per_count,
                                     size_t minimum_pilot_work,
                                     size_t maximum_count) noexcept;

/**
 * @brief Scale a measured pilot count toward a target duration.
 *
 * The result is rounded upward, clamped to the inclusive count bounds, and
 * quantized without exceeding `maximum_count`. Zero signals invalid inputs or
 * the absence of a valid in-range quantum multiple.
 */
size_t calculate_duration_scaled_count(double pilot_duration_seconds,
                                       size_t pilot_count,
                                       double target_duration_seconds,
                                       size_t minimum_count,
                                       size_t maximum_count,
                                       size_t quantum = 1) noexcept;

}  // namespace NumericUtils

#endif  // NUMERIC_UTILS_H
