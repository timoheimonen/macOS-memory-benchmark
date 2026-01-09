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
 * @file memory_manager.cpp
 * @brief Memory allocation with mmap
 *
 * Provides low-level memory allocation functions using mmap for benchmark buffers.
 * Supports both regular cached allocation and best-effort non-cacheable allocation
 * with automatic memory cleanup via RAII smart pointers.
 *
 * Key features:
 * - Uses mmap for large, page-aligned allocations
 * - Optional madvise hints for prefaulting (MADV_WILLNEED) and cache control (MADV_RANDOM)
 * - Custom MmapDeleter for automatic munmap cleanup
 * - Comprehensive error handling with errno details
 */

#include "core/memory/memory_manager.h"
#include "output/console/messages.h"
#include <cstring>  // strlen, strcpy, strerror
#include <iostream> // std::cerr
#include <cerrno>   // errno

/**
 * @brief Allocates a memory buffer using mmap with prefaulting hints.
 *
 * Allocates a buffer using mmap with MAP_ANONYMOUS and MAP_PRIVATE flags,
 * then applies MADV_WILLNEED to prefault pages and potentially improve
 * initial access performance.
 *
 * Error Handling Strategy:
 * - Uses NULL POINTER RETURNS for error handling
 * - Validation errors: Returns null pointer with error message logged
 * - System call errors (mmap): Returns null pointer with errno-based error message
 * - Non-fatal errors (madvise): Logs warning but continues (doesn't fail allocation)
 *
 * Rationale:
 * - Memory allocation is a common operation that may fail frequently
 * - Null checks are lightweight and don't require exception handling overhead
 * - Fits naturally with smart pointer patterns (std::unique_ptr)
 * - Allows callers to decide how to handle failures (check null or propagate)
 * - C-style API (mmap) returns MAP_FAILED, which maps naturally to null pointers
 *
 * @param[in] size         Size of the buffer to allocate in bytes. Must be non-zero.
 * @param[in] buffer_name  Descriptive name for the buffer (used in error messages).
 *                         Must be a valid null-terminated string.
 *
 * @return MmapPtr smart pointer managing the allocated memory
 * @return nullptr if allocation fails (size is 0 or mmap fails)
 *
 * @note The returned pointer is managed by MmapPtr and will be automatically
 *       freed via munmap when it goes out of scope.
 * @note madvise(MADV_WILLNEED) failure is non-fatal and logged as a warning.
 * @note Callers should check for null pointer before using the returned MmapPtr.
 *
 * @see allocate_buffer_non_cacheable() for non-cached allocation
 * @see buffer_allocator.cpp for example usage with null pointer checking
 */
MmapPtr allocate_buffer(size_t size, const char* buffer_name) {
  // Error: Validate size before allocation - zero size is invalid
  if (size == 0) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_zero(buffer_name) << std::endl;
    return MmapPtr(nullptr, MmapDeleter{0});  // Return null pointer on validation error
  }
  
  // Allocate memory using mmap (C-style API returns MAP_FAILED on error)
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  // Error: System call failed - mmap returns MAP_FAILED on error
  if (ptr == MAP_FAILED) {
    // Create specific error message with errno details
    std::cerr << Messages::error_prefix() << Messages::error_mmap_failed(buffer_name) 
              << ": " << strerror(errno) << std::endl;
    return MmapPtr(nullptr, MmapDeleter{0});  // Return null pointer on allocation failure
  }
  
  // Create MmapPtr with custom deleter
  MmapPtr buffer_ptr(ptr, MmapDeleter{size});
  
  // Advise the kernel that we will need this memory (prefault optimization)
  // Note: Non-fatal error - madvise failure doesn't prevent memory from being usable
  if (madvise(ptr, size, MADV_WILLNEED) == -1) {
    std::cerr << Messages::error_prefix() << Messages::error_madvise_failed(buffer_name) 
              << ": " << strerror(errno) << std::endl;
    // Non-fatal error, continue anyway - allocation succeeded
  }
  
  return buffer_ptr;  // Return valid pointer on success
}

/**
 * @brief Allocates a memory buffer with hints to discourage CPU caching (best-effort).
 *
 * Allocates a buffer using mmap and applies MADV_RANDOM to hint that access patterns
 * are random and unpredictable, which may discourage aggressive prefetching and caching.
 *
 * IMPORTANT LIMITATIONS:
 * User-space code on macOS cannot create truly non-cacheable memory because:
 * - Cannot modify page table attributes (requires kernel privileges)
 * - Cannot set MAIR (Memory Attribute Indirection Register)
 * - Cannot create truly uncached mappings
 *
 * This function provides best-effort cache discouragement, but the CPU may still
 * cache the data. For true non-cacheable memory, kernel-level support is required.
 *
 * @param[in] size         Size of the buffer to allocate in bytes. Must be non-zero.
 * @param[in] buffer_name  Descriptive name for the buffer (used in error messages).
 *                         Must be a valid null-terminated string.
 *
 * @return MmapPtr smart pointer managing the allocated memory
 * @return nullptr if allocation fails (size is 0 or mmap fails)
 *
 * @note The returned pointer is managed by MmapPtr and will be automatically
 *       freed via munmap when it goes out of scope.
 * @note MADV_RANDOM is a hint to the kernel and does not guarantee non-cached behavior.
 * @note madvise(MADV_RANDOM) failure is non-fatal and logged as a warning.
 * @note This is primarily useful for testing memory subsystem behavior under
 *       different access patterns, not for guaranteeing uncached access.
 *
 * @see allocate_buffer() for regular cached allocation
 */
MmapPtr allocate_buffer_non_cacheable(size_t size, const char* buffer_name) {
  // Error: Validate size before allocation - zero size is invalid
  if (size == 0) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_zero(buffer_name) << std::endl;
    return MmapPtr(nullptr, MmapDeleter{0});  // Return null pointer on validation error
  }
  
  // Allocate memory using mmap (same as regular allocation)
  // Error: System call failed - mmap returns MAP_FAILED on error
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (ptr == MAP_FAILED) {
    // Create specific error message with errno details
    std::cerr << Messages::error_prefix() << Messages::error_mmap_failed(buffer_name) 
              << ": " << strerror(errno) << std::endl;
    return MmapPtr(nullptr, MmapDeleter{0});  // Return null pointer on allocation failure
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
  // Note: Non-fatal error - madvise failure doesn't prevent memory from being usable
  if (madvise(ptr, size, MADV_RANDOM) == -1) {
    std::cerr << Messages::warning_prefix() << Messages::warning_madvise_random_failed(buffer_name, strerror(errno)) << std::endl;
    // Non-fatal error, continue anyway - allocation succeeded
  }
  
  return buffer_ptr;  // Return valid pointer on success
}

