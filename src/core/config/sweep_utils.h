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
 * @file sweep_utils.h
 * @brief Shared structural parsing and Cartesian run counting for sweeps.
 */

#ifndef SWEEP_UTILS_H
#define SWEEP_UTILS_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "utils/numeric_utils.h"

/** Structurally parsed `key=value1,value2` text without mode-specific validation. */
struct ParsedSweepText {
  std::string key;
  std::vector<std::string> values;
};

/**
 * Parse the common structure of one sweep specification.
 *
 * Tokens are preserved exactly: the first equals sign separates the key, commas
 * separate values, and no whitespace is trimmed. Parameter names and typed
 * values remain the responsibility of the mode-specific parser.
 *
 * @param specification Raw `key=value1,value2` text.
 * @return The unvalidated key and non-empty raw value tokens.
 * @throws std::invalid_argument if the structure is malformed or any value is empty.
 */
ParsedSweepText parse_sweep_text(const std::string& specification);

/**
 * Calculate a Cartesian product size from dimension cardinalities.
 *
 * @param dimension_sizes Number of candidates in each Cartesian dimension.
 * @return One for no dimensions, zero for an empty dimension, or `SIZE_MAX` if
 *         multiplication overflows.
 */
size_t calculate_cartesian_run_count(const std::vector<size_t>& dimension_sizes) noexcept;

namespace SweepUtilsDetail {

template <typename DimensionRange, typename CardinalityFunction>
size_t calculate_cartesian_run_count(const DimensionRange& dimensions,
                                     CardinalityFunction cardinality) noexcept {
  size_t run_count = 1;
  for (const auto& dimension : dimensions) {
    const size_t dimension_size = cardinality(dimension);
    if (dimension_size == 0) {
      return 0;
    }
    run_count = NumericUtils::saturating_multiply(run_count, dimension_size);
    if (run_count == std::numeric_limits<size_t>::max()) {
      return run_count;
    }
  }
  return run_count;
}

}  // namespace SweepUtilsDetail

/**
 * Calculate the run count for sweep-spec types that expose a `values` member.
 *
 * This adapter keeps the mode-specific spec types independent while routing
 * their shared Cartesian-counting rule through one implementation.
 */
template <typename SweepSpecType>
size_t calculate_sweep_run_count_from_specs(const std::vector<SweepSpecType>& specs) noexcept {
  return SweepUtilsDetail::calculate_cartesian_run_count(
      specs, [](const SweepSpecType& spec) { return spec.values.size(); });
}

#endif  // SWEEP_UTILS_H
