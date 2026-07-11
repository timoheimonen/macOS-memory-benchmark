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
 * @file cyclic_order.cpp
 * @brief Shared deterministic cyclic-order implementation
 */

#include "utils/cyclic_order.h"

std::vector<size_t> build_cyclic_order(size_t item_count,
                                       size_t rotation_index) {
  std::vector<size_t> order;
  order.reserve(item_count);
  if (item_count == 0) {
    return order;
  }

  const size_t start = rotation_index % item_count;
  const size_t suffix_count = item_count - start;
  for (size_t position = 0; position < item_count; ++position) {
    order.push_back(position < suffix_count ? start + position
                                            : position - suffix_count);
  }
  return order;
}
