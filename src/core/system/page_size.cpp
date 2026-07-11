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
 * @file page_size.cpp
 * @brief System page-size query implementation
 */

#include "core/system/page_size.h"

#include <unistd.h>

size_t get_system_page_size_bytes() noexcept {
  const int page_size = getpagesize();
  return page_size > 0 ? static_cast<size_t>(page_size) : 0;
}
