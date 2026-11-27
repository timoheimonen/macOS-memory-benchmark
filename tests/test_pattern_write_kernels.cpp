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

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "asm/asm_functions.h"

extern "C" uint64_t verify_pattern_callee_saved_registers_asm(
    uintptr_t function_address, uintptr_t arg0, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);

namespace {

using PatternWriteKernel = void (*)(void*, size_t, uint64_t);

class PatternGuardedMapping {
 public:
  explicit PatternGuardedMapping(size_t payload_size)
      : page_size_(static_cast<size_t>(getpagesize())),
        mapping_size_(page_size_ * 3) {
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return;
    }
    if (mprotect(mapping_, page_size_, PROT_NONE) != 0 ||
        mprotect(static_cast<unsigned char*>(mapping_) + 2 * page_size_,
                 page_size_, PROT_NONE) != 0) {
      munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      return;
    }
    payload_ = static_cast<unsigned char*>(mapping_) + 2 * page_size_ -
               payload_size;
  }

  ~PatternGuardedMapping() {
    if (mapping_ != nullptr) {
      munmap(mapping_, mapping_size_);
    }
  }

  PatternGuardedMapping(const PatternGuardedMapping&) = delete;
  PatternGuardedMapping& operator=(const PatternGuardedMapping&) = delete;

  unsigned char* payload() const { return payload_; }
  unsigned char* accessible_begin() const {
    return static_cast<unsigned char*>(mapping_) + page_size_;
  }
  size_t prefix_size() const {
    return static_cast<size_t>(payload_ - accessible_begin());
  }
  bool valid() const { return mapping_ != nullptr; }

 private:
  void* mapping_ = nullptr;
  unsigned char* payload_ = nullptr;
  size_t page_size_ = 0;
  size_t mapping_size_ = 0;
};

constexpr std::array<size_t, 19> kKernelSizes = {
    0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 511, 512, 513, 1024, 1031, 4095, 4096};

constexpr std::array<uint64_t, 5> kTestPatterns = {
    0x0123456789ABCDEFULL,
    0xA55A5AA5F00FF00FULL,
    0x0000000000000000ULL,
    0xFFFFFFFFFFFFFFFFULL,
    0x8000000000000001ULL,
};

uint8_t expected_pattern_byte(uint64_t pattern64, size_t offset) {
  return static_cast<uint8_t>((pattern64 >> ((offset & 7U) * 8U)) & 0xFFU);
}

void verify_pattern_kernel_on_guarded_sizes(PatternWriteKernel kernel,
                                            const char* name) {
  SCOPED_TRACE(name);
  for (size_t size : kKernelSizes) {
    SCOPED_TRACE(size);
    PatternGuardedMapping mapping(size);
    ASSERT_TRUE(mapping.valid());

    for (uint64_t pattern : kTestPatterns) {
      SCOPED_TRACE(pattern);
      std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
      if (size > 0) {
        std::memset(mapping.payload(), 0x5A, size);
      }

      kernel(mapping.payload(), size, pattern);

      for (size_t i = 0; i < mapping.prefix_size(); ++i) {
        ASSERT_EQ(mapping.accessible_begin()[i], 0xD3)
            << "prefix byte modified at " << i;
      }
      for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(mapping.payload()[i], expected_pattern_byte(pattern, i))
            << "payload byte mismatch at " << i;
      }
    }
  }
}

void verify_pattern_kernel_on_unaligned_offsets(PatternWriteKernel kernel,
                                                const char* name) {
  SCOPED_TRACE(name);
  constexpr uint64_t kPattern = 0x1122334455667788ULL;
  constexpr size_t kHeadPad = 16;
  constexpr size_t kTailPad = 16;
  for (size_t offset : {size_t{1}, size_t{3}, size_t{7}, size_t{8}}) {
    for (size_t size : {size_t{0}, size_t{1}, size_t{7}, size_t{15},
                        size_t{63}, size_t{519}, size_t{1031}}) {
      std::vector<unsigned char> buffer(kHeadPad + offset + size + kTailPad, 0xCC);
      unsigned char* dst = buffer.data() + kHeadPad + offset;
      kernel(dst, size, kPattern);

      for (size_t i = 0; i < kHeadPad + offset; ++i) {
        ASSERT_EQ(buffer[i], 0xCC);
      }
      for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(dst[i], expected_pattern_byte(kPattern, i));
      }
      for (size_t i = kHeadPad + offset + size; i < buffer.size(); ++i) {
        ASSERT_EQ(buffer[i], 0xCC);
      }
    }
  }
}

void verify_pattern_kernel_callee_saved_registers(PatternWriteKernel kernel,
                                                  const char* name) {
  SCOPED_TRACE(name);
  constexpr uint64_t kPattern = 0xFEDCBA9876543210ULL;
  std::vector<unsigned char> buffer(2048, 0);
  const uintptr_t fn_addr = reinterpret_cast<uintptr_t>(kernel);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(buffer.data());

  for (size_t size : {size_t{0}, size_t{1}, size_t{7}, size_t{31},
                      size_t{512}, size_t{513}, size_t{1031}}) {
    const uint64_t ok = verify_pattern_callee_saved_registers_asm(
        fn_addr, dst_addr, size, static_cast<uintptr_t>(kPattern), 0, 0, 0);
    EXPECT_EQ(ok, 1u) << "callee-saved violation at size=" << size;
  }
}

}  // namespace

TEST(PatternWriteKernelTest, NonTemporalPatternWriteExactBytesAndGuards) {
  verify_pattern_kernel_on_guarded_sizes(memory_write_pattern_loop_asm,
                                         "memory_write_pattern_loop_asm");
}

TEST(PatternWriteKernelTest, CachePatternWriteExactBytesAndGuards) {
  verify_pattern_kernel_on_guarded_sizes(memory_write_cache_pattern_loop_asm,
                                         "memory_write_cache_pattern_loop_asm");
}

TEST(PatternWriteKernelTest, UnalignedDestinationsPreserveSurroundingSentinels) {
  verify_pattern_kernel_on_unaligned_offsets(memory_write_pattern_loop_asm,
                                             "memory_write_pattern_loop_asm");
  verify_pattern_kernel_on_unaligned_offsets(memory_write_cache_pattern_loop_asm,
                                             "memory_write_cache_pattern_loop_asm");
}

TEST(PatternWriteKernelTest, PreservesAapcs64CalleeSavedRegisters) {
  verify_pattern_kernel_callee_saved_registers(memory_write_pattern_loop_asm,
                                               "memory_write_pattern_loop_asm");
  verify_pattern_kernel_callee_saved_registers(memory_write_cache_pattern_loop_asm,
                                               "memory_write_cache_pattern_loop_asm");
}
