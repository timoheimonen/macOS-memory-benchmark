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
/**
 * @file memory_manager.h
 * @brief Memory allocation and management using mmap
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides memory allocation functions using mmap with automatic
 * cleanup via RAII. Supports both normal and cache-discouraging memory allocation.
 */
#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cstddef>  // size_t
#include <memory>   // std::unique_ptr
#include <cstring>  // strerror
#include <iostream> // std::cerr
#include <cerrno>   // errno
#include <sys/mman.h>  // mmap, munmap, MAP_FAILED, madvise, MADV_WILLNEED
#include "output/console/messages.h"  // For Messages namespace

/**
 * @struct MmapDeleter
 * @brief Custom deleter for memory allocated with mmap
 *
 * This deleter is used with std::unique_ptr to automatically unmap memory
 * allocated via mmap when the unique_ptr is destroyed. This provides RAII
 * (Resource Acquisition Is Initialization) guarantees:
 * - Memory is automatically freed when the unique_ptr goes out of scope
 * - Memory is freed even if exceptions are thrown
 * - No manual memory management required
 * - Prevents memory leaks on all code paths
 *
 * @note The allocation_size must match the size passed to mmap() when the
 *       memory was allocated. Incorrect sizes can cause munmap() to fail
 *       or corrupt memory. The size is stored at MmapPtr creation time.
 */
struct MmapDeleter {
  size_t allocation_size;  ///< Store the size needed for munmap (must match mmap size)

  /**
   * @brief Function call operator invoked by unique_ptr upon destruction
   * @param ptr Pointer to the memory to unmap
   *
   * Automatically unmaps the memory using munmap. Logs errors but does not
   * throw exceptions (as destructors should not throw).
   * 
   * @note This function is safe to call with nullptr or MAP_FAILED, which
   *       are handled gracefully without attempting to unmap.
   */
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

/**
 * @typedef MmapPtr
 * @brief Type alias for unique_ptr managing mmap-allocated memory
 *
 * Convenience type for managing memory allocated with mmap. Automatically
 * unmaps the memory when the pointer goes out of scope.
 */
using MmapPtr = std::unique_ptr<void, MmapDeleter>;

/**
 * @brief Allocate a buffer using mmap with proper error handling and madvise hints
 * @param size Size of the buffer to allocate in bytes (must be > 0)
 * @param buffer_name Name of the buffer (used in error messages for clarity)
 * @return MmapPtr that will automatically free the memory on destruction (RAII)
 * @return nullptr (empty unique_ptr) if allocation fails or size is 0
 *
 * Allocates memory using mmap with MAP_PRIVATE | MAP_ANONYMOUS flags and
 * applies MADV_WILLNEED hint to encourage the OS to prefault pages.
 * 
 * @note The returned MmapPtr uses RAII - memory is automatically freed when
 *       the pointer goes out of scope, even if exceptions are thrown.
 * @note If madvise() fails, the buffer is still returned (non-fatal error).
 */
MmapPtr allocate_buffer(size_t size, const char* buffer_name = "buffer");

/**
 * @brief Allocate a buffer with cache-discouraging hints (best-effort, not true non-cacheable)
 * @param size Size of the buffer to allocate in bytes (must be > 0)
 * @param buffer_name Name of the buffer (used in error messages for clarity)
 * @return MmapPtr that will automatically free the memory on destruction (RAII)
 * @return nullptr (empty unique_ptr) if allocation fails or size is 0
 *
 * On macOS ARM64, applies madvise() hints to discourage caching, but cannot achieve
 * true non-cacheable memory (user-space cannot modify page table attributes).
 * This is a best-effort approach to reduce cache effects.
 * 
 * @note The returned MmapPtr uses RAII - memory is automatically freed when
 *       the pointer goes out of scope, even if exceptions are thrown.
 * @note If madvise() fails, the buffer is still returned (non-fatal error).
 */
MmapPtr allocate_buffer_non_cacheable(size_t size, const char* buffer_name = "buffer");

#endif // MEMORY_MANAGER_H

