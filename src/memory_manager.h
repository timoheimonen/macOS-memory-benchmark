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
#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cstddef>  // size_t
#include <memory>   // std::unique_ptr
#include <cstring>  // strerror
#include <iostream> // std::cerr
#include <cerrno>   // errno
#include <sys/mman.h>  // mmap, munmap, MAP_FAILED, madvise, MADV_WILLNEED
#include "messages.h"  // For Messages namespace

// Custom deleter for memory allocated with mmap
struct MmapDeleter {
  size_t allocation_size;  // Store the size needed for munmap

  // This function call operator is invoked by unique_ptr upon destruction
  void operator()(void *ptr) const {
    if (ptr && ptr != MAP_FAILED) {
      if (munmap(ptr, allocation_size) == -1) {
        // Log error if munmap fails, but don't throw from destructor
        std::cerr << Messages::error_prefix() << Messages::error_munmap_failed() 
                  << ": " << strerror(errno) << std::endl;
      }
    }
  }
};

// Define a type alias for convenience
using MmapPtr = std::unique_ptr<void, MmapDeleter>;

// Allocate a buffer using mmap with proper error handling and madvise hints
// Returns a MmapPtr that will automatically free the memory on destruction
// Returns nullptr (empty unique_ptr) if allocation fails
// buffer_name is used in error messages for clarity
MmapPtr allocate_buffer(size_t size, const char* buffer_name = "buffer");

// Allocate a buffer with cache-discouraging hints (best-effort, not true non-cacheable)
// On macOS ARM64, applies madvise() hints to discourage caching, but cannot achieve
// true non-cacheable memory (user-space cannot modify page table attributes).
// Returns a MmapPtr that will automatically free the memory on destruction
// Returns nullptr (empty unique_ptr) if allocation fails
// buffer_name is used in error messages for clarity
MmapPtr allocate_buffer_non_cacheable(size_t size, const char* buffer_name = "buffer");

#endif // MEMORY_MANAGER_H

