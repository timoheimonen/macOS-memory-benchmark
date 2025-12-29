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
#include "pattern_benchmark/pattern_benchmark.h"
#include "utils/benchmark.h"
#include <atomic>

// ============================================================================
// Helper Functions for Pattern Tests
// ============================================================================

// Helper function to run a pattern read test
double run_pattern_read_test(void* buffer, size_t size, int iterations, 
                             uint64_t (*read_func)(const void*, size_t),
                             std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = read_func(buffer, size);
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_release);
  return elapsed;
}

// Helper function to run a pattern write test
double run_pattern_write_test(void* buffer, size_t size, int iterations,
                              void (*write_func)(void*, size_t),
                              HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    write_func(buffer, size);
  }
  return timer.stop();
}

// Helper function to run a pattern copy test
double run_pattern_copy_test(void* dst, void* src, size_t size, int iterations,
                             void (*copy_func)(void*, const void*, size_t),
                             HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    copy_func(dst, src, size);
  }
  return timer.stop();
}

// Helper function to run a strided pattern test
double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = memory_read_strided_loop_asm(buffer, size, stride);
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_release);
  return elapsed;
}

double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                       HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_write_strided_loop_asm(buffer, size, stride);
  }
  return timer.stop();
}

double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_copy_strided_loop_asm(dst, src, size, stride);
  }
  return timer.stop();
}

// Helper function to run a random pattern test
double run_pattern_read_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                    std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  timer.start();
  uint64_t total_checksum = 0;
  for (int i = 0; i < iterations; ++i) {
    uint64_t result = memory_read_random_loop_asm(buffer, indices.data(), indices.size());
    total_checksum ^= result;
  }
  double elapsed = timer.stop();
  checksum.fetch_xor(total_checksum, std::memory_order_release);
  return elapsed;
}

double run_pattern_write_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                     HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_write_random_loop_asm(buffer, indices.data(), indices.size());
  }
  return timer.stop();
}

double run_pattern_copy_random_test(void* dst, void* src, const std::vector<size_t>& indices, int iterations,
                                    HighResTimer& timer) {
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    memory_copy_random_loop_asm(dst, src, indices.data(), indices.size());
  }
  return timer.stop();
}

