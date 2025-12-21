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
#include "messages.h"
#include <cstring>  // strlen, strcpy, strerror
#include <cstdio>   // perror
#include <iostream> // std::cerr
#include <cerrno>   // errno

MmapPtr allocate_buffer(size_t size, const char* buffer_name) {
  // Allocate memory using mmap
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (ptr == MAP_FAILED) {
    // Create specific error message
    perror(Messages::error_mmap_failed(buffer_name).c_str());
    // Return empty unique_ptr (nullptr) on failure
    return MmapPtr(nullptr, MmapDeleter{0});
  }
  
  // Create MmapPtr with custom deleter
  MmapPtr buffer_ptr(ptr, MmapDeleter{size});
  
  // Advise the kernel that we will need this memory (prefault optimization)
  if (madvise(ptr, size, MADV_WILLNEED) == -1) {
    perror(Messages::error_madvise_failed(buffer_name).c_str());
    // Non-fatal error, continue anyway
  }
  
  return buffer_ptr;
}

MmapPtr allocate_buffer_non_cacheable(size_t size, const char* buffer_name) {
  // Allocate memory using mmap (same as regular allocation)
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (ptr == MAP_FAILED) {
    // Create specific error message
    perror(Messages::error_mmap_failed(buffer_name).c_str());
    // Return empty unique_ptr (nullptr) on failure
    return MmapPtr(nullptr, MmapDeleter{0});
  }
  
  // Create MmapPtr with custom deleter
  MmapPtr buffer_ptr(ptr, MmapDeleter{size});
  
  // Apply cache-discouraging hints (best-effort approach)
  // Note: User-space on macOS cannot achieve true non-cacheable memory because:
  // - Cannot modify page table attributes (requires kernel privileges)
  // - Cannot set MAIR (Memory Attribute Indirection Register)
  // - Cannot create truly uncached mappings
  // These hints may reduce but not eliminate caching behavior.
  
  // Use MADV_RANDOM to hint that access pattern is random, discouraging aggressive caching
  // This is a best-effort hint; it does not guarantee non-cacheable behavior
  if (madvise(ptr, size, MADV_RANDOM) == -1) {
    std::cerr << Messages::warning_madvise_random_failed(buffer_name, strerror(errno)) << std::endl;
    // Non-fatal error, continue anyway
  }
  
  return buffer_ptr;
}

