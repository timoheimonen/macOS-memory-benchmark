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
 * @file cyclic_order.h
 * @brief Deterministic cyclic rotation shared by benchmark schedulers
 */

#ifndef CYCLIC_ORDER_H
#define CYCLIC_ORDER_H

#include <cstddef>
#include <vector>

/**
 * @brief Build one zero-based cyclic rotation of `item_count` indexes.
 *
 * The returned order starts at `rotation_index % item_count` and contains
 * every index exactly once. An empty item set produces an empty order without
 * evaluating the modulo expression.
 */
std::vector<size_t> build_cyclic_order(size_t item_count,
                                       size_t rotation_index);

#endif  // CYCLIC_ORDER_H
