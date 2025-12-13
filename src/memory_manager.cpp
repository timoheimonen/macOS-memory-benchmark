// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
#include "memory_manager.h"
#include <cstring>  // strlen, strcpy

MmapPtr allocate_buffer(size_t size, const char* buffer_name) {
  // Allocate memory using mmap
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (ptr == MAP_FAILED) {
    // Create specific error message
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "mmap failed for %s", buffer_name);
    perror(error_msg);
    // Return empty unique_ptr (nullptr) on failure
    return MmapPtr(nullptr, MmapDeleter{0});
  }
  
  // Create MmapPtr with custom deleter
  MmapPtr buffer_ptr(ptr, MmapDeleter{size});
  
  // Advise the kernel that we will need this memory (prefault optimization)
  if (madvise(ptr, size, MADV_WILLNEED) == -1) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "madvise failed for %s", buffer_name);
    perror(error_msg);
    // Non-fatal error, continue anyway
  }
  
  return buffer_ptr;
}

