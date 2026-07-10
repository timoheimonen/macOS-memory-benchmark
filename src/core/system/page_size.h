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
 * @file page_size.h
 * @brief System page-size query boundary
 */

#ifndef PAGE_SIZE_H
#define PAGE_SIZE_H

#include <cstddef>

/**
 * @brief Query the native system page size.
 *
 * This is the shared production boundary for page-size-dependent accounting,
 * validation, metadata, and page-prefault operations. The function is
 * thread-safe and does not cache the platform result.
 *
 * @return Native page size in bytes, or zero when the platform reports an
 *         invalid value.
 */
size_t get_system_page_size_bytes() noexcept;

#endif  // PAGE_SIZE_H
