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
 * @file sweep_utils.cpp
 * @brief Shared structural parsing and Cartesian run counting for sweeps.
 */

#include "core/config/sweep_utils.h"

#include <stdexcept>

ParsedSweepText parse_sweep_text(const std::string& specification) {
  const size_t equals_pos = specification.find('=');
  if (equals_pos == std::string::npos || equals_pos == 0 ||
      equals_pos == specification.size() - 1) {
    throw std::invalid_argument("sweep must use key=value1,value2 syntax");
  }

  ParsedSweepText parsed;
  parsed.key = specification.substr(0, equals_pos);
  const std::string value_text = specification.substr(equals_pos + 1);

  size_t value_start = 0;
  while (value_start <= value_text.size()) {
    const size_t comma = value_text.find(',', value_start);
    if (comma == std::string::npos) {
      parsed.values.push_back(value_text.substr(value_start));
      break;
    }
    parsed.values.push_back(value_text.substr(value_start, comma - value_start));
    value_start = comma + 1;
  }

  for (const std::string& value : parsed.values) {
    if (value.empty()) {
      throw std::invalid_argument("sweep value list cannot contain empty values");
    }
  }

  return parsed;
}

size_t calculate_cartesian_run_count(const std::vector<size_t>& dimension_sizes) noexcept {
  return SweepUtilsDetail::calculate_cartesian_run_count(
      dimension_sizes, [](size_t dimension_size) { return dimension_size; });
}
