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
 * @file buffer_initializer.cpp
 * @brief Deterministic pattern-buffer initialization.
 */

#include "core/memory/buffer_initializer.h"

#include "core/memory/buffer_manager.h"
#include "core/memory/memory_utils.h"
#include "output/console/messages/messages_api.h"

#include <cstdlib>
#include <iostream>

int initialize_pattern_buffers(const PatternBuffers& buffers,
                               size_t buffer_size) {
  if (buffers.src_buffer() == nullptr || buffers.dst_buffer() == nullptr) {
    std::cerr << Messages::error_prefix()
              << Messages::error_main_buffers_not_allocated() << std::endl;
    return EXIT_FAILURE;
  }
  return initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(),
                            buffer_size);
}
