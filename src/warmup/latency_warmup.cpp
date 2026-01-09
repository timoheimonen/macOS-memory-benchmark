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
 * @file latency_warmup.cpp
 * @brief Latency test warmup functions
 */

#include <unistd.h>  // For getpagesize()

#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"

/**
 * @brief Warms up memory for latency test by page prefaulting (single thread).
 *
 * This ensures pages are mapped and page faults are removed, but does not
 * build/run the pointer chain, allowing for more accurate "cold-ish" latency measurements.
 *
 * @param buffer Memory region to warm up
 * @param buffer_size Total size of the buffer in bytes
 */
void warmup_latency(void* buffer, size_t buffer_size) {
  auto latency_op = [buffer, buffer_size]() {
    // Only perform warmup if buffer is valid.
    if (buffer == nullptr || buffer_size == 0) {
      return;
    }
    
    // Get actual system page size (4KB on most systems, 16KB on some Apple Silicon)
    const size_t page_size = static_cast<size_t>(getpagesize());
    char* buf = static_cast<char*>(buffer);
    
    // Touch each page with a 1-byte read/write to ensure it's mapped
    // This removes page faults without building the pointer chain
    for (size_t offset = 0; offset < buffer_size; offset += page_size) {
      // Read one byte to trigger page mapping
      volatile char dummy = buf[offset];
      (void)dummy;  // Prevent optimization
      // Write one byte to ensure write access is established
      *static_cast<volatile char*>(&buf[offset]) = buf[offset];
    }
  };
  warmup_single(latency_op);
}

/**
 * @brief Warms up memory for cache latency test by page prefaulting (single thread).
 *
 * This ensures pages are mapped and page faults are removed, but does not
 * build/run the pointer chain, allowing for more accurate "cold-ish" cache latency measurements.
 *
 * @param buffer Memory region to warm up
 * @param buffer_size Total size of the buffer in bytes
 */
void warmup_cache_latency(void* buffer, size_t buffer_size) {
  // Use the same implementation as warmup_latency since they're identical.
  warmup_latency(buffer, buffer_size);
}

