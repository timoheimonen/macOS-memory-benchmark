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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "asm/asm_functions.h"

extern "C" uint64_t verify_pattern_callee_saved_registers_asm(
    uintptr_t function_address, uintptr_t arg0, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);

// Test-only AAPCS64 probe. It seeds x19-x29 and the preserved low 64 bits of
// d8-d15, calls a pattern kernel with up to six integer arguments, and returns
// one only when every callee-saved value survives.
__asm__(R"ASM(
.text
.p2align 4
.global _verify_pattern_callee_saved_registers_asm
_verify_pattern_callee_saved_registers_asm:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]

    mov x16, x0
    mov x0, x1
    mov x1, x2
    mov x2, x3
    mov x3, x4
    mov x4, x5
    mov x5, x6

    mov x19, #0x1919
    mov x20, #0x2020
    mov x21, #0x2121
    mov x22, #0x2222
    mov x23, #0x2323
    mov x24, #0x2424
    mov x25, #0x2525
    mov x26, #0x2626
    mov x27, #0x2727
    mov x28, #0x2828
    mov x9, #0xd8
    fmov d8, x9
    mov x9, #0xd9
    fmov d9, x9
    mov x9, #0xda
    fmov d10, x9
    mov x9, #0xdb
    fmov d11, x9
    mov x9, #0xdc
    fmov d12, x9
    mov x9, #0xdd
    fmov d13, x9
    mov x9, #0xde
    fmov d14, x9
    mov x9, #0xdf
    fmov d15, x9

    blr x16

    mov x17, #1
    mov x9, sp
    cmp x29, x9
    csel x17, x17, xzr, eq
    mov x9, #0x1919
    cmp x19, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2020
    cmp x20, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2121
    cmp x21, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2222
    cmp x22, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2323
    cmp x23, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2424
    cmp x24, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2525
    cmp x25, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2626
    cmp x26, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2727
    cmp x27, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2828
    cmp x28, x9
    csel x17, x17, xzr, eq
    mov x9, #0xd8
    fmov x10, d8
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xd9
    fmov x10, d9
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xda
    fmov x10, d10
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdb
    fmov x10, d11
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdc
    fmov x10, d12
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdd
    fmov x10, d13
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xde
    fmov x10, d14
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdf
    fmov x10, d15
    cmp x10, x9
    csel x17, x17, xzr, eq

    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #160
    mov x0, x17
    ret
)ASM");

namespace {

using PatternWriteKernel = void (*)(void*, size_t, uint64_t);

class PatternGuardedMapping {
 public:
  explicit PatternGuardedMapping(size_t payload_size)
      : page_size_(static_cast<size_t>(getpagesize())),
        accessible_pages_(std::max<size_t>(
            1, (payload_size + page_size_ - 1) / page_size_)),
        mapping_size_(page_size_ * (accessible_pages_ + 2)) {
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return;
    }
    if (mprotect(mapping_, page_size_, PROT_NONE) != 0 ||
        mprotect(static_cast<unsigned char*>(mapping_) +
                     (accessible_pages_ + 1) * page_size_,
                 page_size_, PROT_NONE) != 0) {
      munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      return;
    }
    payload_ = static_cast<unsigned char*>(mapping_) +
               (accessible_pages_ + 1) * page_size_ - payload_size;
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
  size_t accessible_pages_ = 0;
  size_t mapping_size_ = 0;
};

class PatternFrontGuardedMapping {
 public:
  explicit PatternFrontGuardedMapping(size_t payload_size)
      : page_size_(static_cast<size_t>(getpagesize())),
        accessible_pages_(std::max<size_t>(
            1, (payload_size + page_size_ - 1) / page_size_)),
        mapping_size_(page_size_ * (accessible_pages_ + 2)),
        payload_size_(payload_size) {
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return;
    }
    if (mprotect(mapping_, page_size_, PROT_NONE) != 0 ||
        mprotect(static_cast<unsigned char*>(mapping_) +
                     (accessible_pages_ + 1) * page_size_,
                 page_size_, PROT_NONE) != 0) {
      munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      return;
    }
    payload_ = static_cast<unsigned char*>(mapping_) + page_size_;
  }

  ~PatternFrontGuardedMapping() {
    if (mapping_ != nullptr) {
      munmap(mapping_, mapping_size_);
    }
  }

  PatternFrontGuardedMapping(const PatternFrontGuardedMapping&) = delete;
  PatternFrontGuardedMapping& operator=(const PatternFrontGuardedMapping&) =
      delete;

  unsigned char* payload() const { return payload_; }
  unsigned char* suffix_begin() const { return payload_ + payload_size_; }
  size_t suffix_size() const {
    return accessible_pages_ * page_size_ - payload_size_;
  }
  bool valid() const { return mapping_ != nullptr; }

 private:
  void* mapping_ = nullptr;
  unsigned char* payload_ = nullptr;
  size_t page_size_ = 0;
  size_t accessible_pages_ = 0;
  size_t mapping_size_ = 0;
  size_t payload_size_ = 0;
};

constexpr std::array<size_t, 21> kKernelSizes = {
    0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 35, 64, 511, 512, 513, 515, 1024, 1031, 4095, 4096};

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

TEST(PatternWriteKernelTest, ReversePatternWriteExactBytesAndGuards) {
  verify_pattern_kernel_on_guarded_sizes(
      memory_write_reverse_pattern_loop_asm,
      "memory_write_reverse_pattern_loop_asm");

  for (size_t size : kKernelSizes) {
    SCOPED_TRACE(size);
    PatternFrontGuardedMapping mapping(size);
    ASSERT_TRUE(mapping.valid());

    for (uint64_t pattern : kTestPatterns) {
      SCOPED_TRACE(pattern);
      if (size > 0) {
        std::memset(mapping.payload(), 0x5A, size);
      }
      std::memset(mapping.suffix_begin(), 0xD3, mapping.suffix_size());

      memory_write_reverse_pattern_loop_asm(mapping.payload(), size, pattern);

      for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(mapping.payload()[i], expected_pattern_byte(pattern, i))
            << "payload byte mismatch at " << i;
      }
      for (size_t i = 0; i < mapping.suffix_size(); ++i) {
        ASSERT_EQ(mapping.suffix_begin()[i], 0xD3)
            << "suffix byte modified at " << i;
      }
    }
  }
}

TEST(PatternWriteKernelTest, UnalignedDestinationsPreserveSurroundingSentinels) {
  verify_pattern_kernel_on_unaligned_offsets(memory_write_pattern_loop_asm,
                                             "memory_write_pattern_loop_asm");
  verify_pattern_kernel_on_unaligned_offsets(memory_write_cache_pattern_loop_asm,
                                             "memory_write_cache_pattern_loop_asm");
  verify_pattern_kernel_on_unaligned_offsets(
      memory_write_reverse_pattern_loop_asm,
      "memory_write_reverse_pattern_loop_asm");
}

TEST(PatternWriteKernelTest, PreservesAapcs64CalleeSavedRegisters) {
  verify_pattern_kernel_callee_saved_registers(memory_write_pattern_loop_asm,
                                               "memory_write_pattern_loop_asm");
  verify_pattern_kernel_callee_saved_registers(memory_write_cache_pattern_loop_asm,
                                               "memory_write_cache_pattern_loop_asm");
  verify_pattern_kernel_callee_saved_registers(
      memory_write_reverse_pattern_loop_asm,
      "memory_write_reverse_pattern_loop_asm");
}

TEST(PatternWriteKernelTest, StridedPhasedPatternWriteGuardsEdgeInputs) {
  constexpr size_t kSpan = 256;
  PatternGuardedMapping mapping(kSpan);
  ASSERT_TRUE(mapping.valid());

  auto reset_and_verify_untouched = [&](size_t span, size_t stride,
                                        size_t passes, size_t initial_phase) {
    std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
    std::memset(mapping.payload(), 0x5A, kSpan);

    memory_write_strided_phased_pattern_loop_asm(
        mapping.payload(), span, stride, passes, initial_phase,
        0x0123456789ABCDEFULL);

    for (size_t i = 0; i < mapping.prefix_size(); ++i) {
      ASSERT_EQ(mapping.accessible_begin()[i], 0xD3);
    }
    for (size_t i = 0; i < kSpan; ++i) {
      ASSERT_EQ(mapping.payload()[i], 0x5A);
    }
  };

  reset_and_verify_untouched(kSpan, 64, 0, 0);    // passes == 0
  reset_and_verify_untouched(kSpan, 0, 1, 0);     // stride == 0
  reset_and_verify_untouched(0, 64, 1, 0);        // byteCount == 0
  reset_and_verify_untouched(16, 64, 1, 0);       // byteCount < 32
  reset_and_verify_untouched(31, 64, 1, 0);       // byteCount == 31
  reset_and_verify_untouched(kSpan, 64, 1, 64);   // initial_phase == stride
  reset_and_verify_untouched(kSpan, 64, 1, 96);   // initial_phase > stride
}

TEST(PatternWriteKernelTest, StridedPhasedPatternWriteExactBytesAndIsolation) {
  struct StrideCase {
    size_t stride;
    size_t span;
    size_t passes;
    size_t initial_phase;
  };

  const std::vector<StrideCase> cases = {
      // Stride 64 (Q = 2): 1 pass, partial, full cycle (Q), > Q, mid/wrap phase, non-32B tail
      {64, 64 + 32, 1, 0},
      {64, 2 * 64 + 16, 1, 0},
      {64, 2 * 64 + 32, 1, 32},
      {64, 2 * 64 + 32, 2, 0},
      {64, 2 * 64 + 32, 3, 32},
      // Stride 4096 (Q = 128): 1 pass, partial < Q, full == Q, > Q, mid/wrap phase, tail
      {4096, 4096 + 32, 1, 0},
      {4096, 2 * 4096 + 16, 3, 2048},
      {4096, 2 * 4096 + 32, 128, 0},
      {4096, 2 * 4096 + 32, 129, 4096 - 32},
      // Stride 16384 (Q = 512): 1 pass, partial < Q, full == Q, > Q, mid/wrap phase, tail
      {16384, 16384 + 32, 1, 0},
      {16384, 2 * 16384 + 16, 3, 8192},
      {16384, 2 * 16384 + 32, 512, 0},
      {16384, 2 * 16384 + 32, 513, 16384 - 32},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(testing::Message()
                 << "stride=" << tc.stride << " span=" << tc.span
                 << " passes=" << tc.passes
                 << " initial_phase=" << tc.initial_phase);
    PatternGuardedMapping mapping(tc.span);
    ASSERT_TRUE(mapping.valid());

    for (uint64_t pattern : {kTestPatterns[0], kTestPatterns[1], kTestPatterns[4]}) {
      std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
      std::memset(mapping.payload(), 0x5A, tc.span);

      memory_write_strided_phased_pattern_loop_asm(
          mapping.payload(), tc.span, tc.stride, tc.passes, tc.initial_phase,
          pattern);

      for (size_t i = 0; i < mapping.prefix_size(); ++i) {
        ASSERT_EQ(mapping.accessible_begin()[i], 0xD3);
      }

      std::vector<uint8_t> touched(tc.span, 0);
      size_t phase = tc.initial_phase;
      for (size_t p = 0; p < tc.passes; ++p) {
        for (size_t off = phase; off + 32 <= tc.span; off += tc.stride) {
          std::fill(touched.begin() + static_cast<std::ptrdiff_t>(off),
                    touched.begin() + static_cast<std::ptrdiff_t>(off + 32), 1);
        }
        phase += 32;
        if (phase >= tc.stride) {
          phase -= tc.stride;
        }
      }

      for (size_t i = 0; i < tc.span; ++i) {
        if (touched[i]) {
          ASSERT_EQ(mapping.payload()[i], expected_pattern_byte(pattern, i))
              << "strided byte mismatch at " << i;
        } else {
          ASSERT_EQ(mapping.payload()[i], 0x5A)
              << "untouched strided byte overwritten at " << i;
        }
      }
    }
  }
}

TEST(PatternWriteKernelTest, Strided2MiBPhasedPatternWriteAndPhaseWrapIsolation) {
  constexpr size_t kStride2MiB = 2097152;
  constexpr size_t kSpan2MiBPlus32 = kStride2MiB + 32;
  constexpr size_t kInitialPhase = kStride2MiB - 32;
  constexpr size_t kPasses = 2;

  PatternGuardedMapping mapping(kSpan2MiBPlus32);
  ASSERT_TRUE(mapping.valid());

  for (uint64_t pattern : {kTestPatterns[0], kTestPatterns[4]}) {
    std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
    std::memset(mapping.payload(), 0x5A, kSpan2MiBPlus32);

    memory_write_strided_phased_pattern_loop_asm(
        mapping.payload(), kSpan2MiBPlus32, kStride2MiB, kPasses,
        kInitialPhase, pattern);

    for (size_t i = 0; i < mapping.prefix_size(); ++i) {
      ASSERT_EQ(mapping.accessible_begin()[i], 0xD3);
    }

    // Pass 0 (phase = 2 MiB - 32) writes [2 MiB - 32, 2 MiB).
    // Phase wraps across kStride2MiB to 0; Pass 1 writes [0, 32) and [2 MiB, 2 MiB + 32).
    for (size_t i = 0; i < kSpan2MiBPlus32; ++i) {
      const bool touched = (i < 32) || (i >= kInitialPhase && i < kSpan2MiBPlus32);
      if (touched) {
        ASSERT_EQ(mapping.payload()[i], expected_pattern_byte(pattern, i))
            << "strided 2MiB byte mismatch at " << i;
      } else {
        ASSERT_EQ(mapping.payload()[i], 0x5A)
            << "untouched strided 2MiB byte overwritten at " << i;
      }
    }
  }

  std::vector<unsigned char> scratch(kSpan2MiBPlus32, 0);
  const uint64_t ok = verify_pattern_callee_saved_registers_asm(
      reinterpret_cast<uintptr_t>(memory_write_strided_phased_pattern_loop_asm),
      reinterpret_cast<uintptr_t>(scratch.data()), kSpan2MiBPlus32, kStride2MiB,
      kPasses, kInitialPhase, static_cast<uintptr_t>(kTestPatterns[0]));
  EXPECT_EQ(ok, 1u);
}

TEST(PatternWriteKernelTest, StridedPhasedPatternWriteUnalignedAndCalleeSaved) {
  constexpr uint64_t kPattern = 0x1122334455667788ULL;
  constexpr size_t kHeadPad = 16;
  constexpr size_t kTailPad = 16;
  constexpr size_t kStride = 64;
  constexpr size_t kSpan = 2 * kStride + 16;

  for (size_t offset : {size_t{1}, size_t{3}, size_t{7}}) {
    std::vector<unsigned char> buffer(kHeadPad + offset + kSpan + kTailPad,
                                      0xCC);
    unsigned char* dst = buffer.data() + kHeadPad + offset;
    memory_write_strided_phased_pattern_loop_asm(dst, kSpan, kStride, 1, 32,
                                                 kPattern);

    for (size_t i = 0; i < kHeadPad + offset; ++i) {
      ASSERT_EQ(buffer[i], 0xCC);
    }
    for (size_t i = 0; i < kSpan; ++i) {
      const bool in_slot = (i >= 32 && i < 64) || (i >= 96 && i < 128);
      if (in_slot) {
        ASSERT_EQ(dst[i], expected_pattern_byte(kPattern, i));
      } else {
        ASSERT_EQ(dst[i], 0xCC);
      }
    }
    for (size_t i = kHeadPad + offset + kSpan; i < buffer.size(); ++i) {
      ASSERT_EQ(buffer[i], 0xCC);
    }
  }

  const uintptr_t fn_addr =
      reinterpret_cast<uintptr_t>(memory_write_strided_phased_pattern_loop_asm);
  std::vector<unsigned char> scratch(36864, 0);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(scratch.data());

  for (size_t stride : {size_t{64}, size_t{4096}, size_t{16384}}) {
    const size_t span = 2 * stride + 32;
    for (size_t passes : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
      for (size_t phase : {size_t{0}, size_t{32}, stride - 32}) {
        const uint64_t ok = verify_pattern_callee_saved_registers_asm(
            fn_addr, dst_addr, span, stride, passes, phase,
            static_cast<uintptr_t>(kPattern));
        EXPECT_EQ(ok, 1u) << "callee-saved violation stride=" << stride
                          << " passes=" << passes << " phase=" << phase;
      }
    }
  }
}

TEST(PatternWriteKernelTest, RandomPatternWriteGuardsAndExactBytes) {
  constexpr size_t kPayloadSize = 256;
  PatternGuardedMapping mapping(kPayloadSize);
  ASSERT_TRUE(mapping.valid());

  // Zero accesses with nullptr indices and valid indices must not write or fault.
  std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
  std::memset(mapping.payload(), 0x5A, kPayloadSize);
  memory_write_random_pattern_loop_asm(mapping.payload(), nullptr, 0,
                                       kTestPatterns[0]);
  for (size_t i = 0; i < mapping.prefix_size(); ++i) {
    ASSERT_EQ(mapping.accessible_begin()[i], 0xD3);
  }
  for (size_t i = 0; i < kPayloadSize; ++i) {
    ASSERT_EQ(mapping.payload()[i], 0x5A);
  }

  const std::vector<size_t> dummy_indices = {0, 32};
  memory_write_random_pattern_loop_asm(mapping.payload(), dummy_indices.data(),
                                       0, kTestPatterns[1]);
  for (size_t i = 0; i < kPayloadSize; ++i) {
    ASSERT_EQ(mapping.payload()[i], 0x5A);
  }

  const std::vector<std::vector<size_t>> index_cases = {
      {224},                 // single access abutting trailing PROT_NONE guard page
      {128, 0, 64},          // out-of-order with untouched interior gaps
      {64, 0, 64, 224, 0},   // repeated indices
      {0, 32, 64},           // adjacent 32B slots
  };

  for (const auto& indices : index_cases) {
    for (uint64_t pattern : kTestPatterns) {
      std::memset(mapping.accessible_begin(), 0xD3, mapping.prefix_size());
      std::memset(mapping.payload(), 0x5A, kPayloadSize);

      memory_write_random_pattern_loop_asm(mapping.payload(), indices.data(),
                                           indices.size(), pattern);

      for (size_t i = 0; i < mapping.prefix_size(); ++i) {
        ASSERT_EQ(mapping.accessible_begin()[i], 0xD3);
      }

      std::vector<uint8_t> touched(kPayloadSize, 0);
      for (size_t idx : indices) {
        std::fill(touched.begin() + static_cast<std::ptrdiff_t>(idx),
                  touched.begin() + static_cast<std::ptrdiff_t>(idx + 32), 1);
      }
      for (size_t i = 0; i < kPayloadSize; ++i) {
        if (touched[i]) {
          ASSERT_EQ(mapping.payload()[i], expected_pattern_byte(pattern, i))
              << "random byte mismatch at " << i;
        } else {
          ASSERT_EQ(mapping.payload()[i], 0x5A)
              << "untouched random byte overwritten at " << i;
        }
      }
    }
  }
}

TEST(PatternWriteKernelTest, RandomPatternWriteUnalignedAndCalleeSaved) {
  constexpr uint64_t kPattern = 0x1122334455667788ULL;
  constexpr size_t kHeadPad = 16;
  constexpr size_t kTailPad = 16;
  constexpr size_t kPayloadSize = 160;
  const std::vector<size_t> indices = {128, 0, 64};

  for (size_t offset : {size_t{1}, size_t{3}, size_t{7}}) {
    std::vector<unsigned char> buffer(
        kHeadPad + offset + kPayloadSize + kTailPad, 0xCC);
    unsigned char* dst = buffer.data() + kHeadPad + offset;

    memory_write_random_pattern_loop_asm(dst, indices.data(), indices.size(),
                                         kPattern);

    for (size_t i = 0; i < kHeadPad + offset; ++i) {
      ASSERT_EQ(buffer[i], 0xCC);
    }
    for (size_t i = 0; i < kPayloadSize; ++i) {
      const bool in_slot = (i < 32) || (i >= 64 && i < 96) || (i >= 128);
      if (in_slot) {
        ASSERT_EQ(dst[i], expected_pattern_byte(kPattern, i));
      } else {
        ASSERT_EQ(dst[i], 0xCC);
      }
    }
    for (size_t i = kHeadPad + offset + kPayloadSize; i < buffer.size(); ++i) {
      ASSERT_EQ(buffer[i], 0xCC);
    }
  }

  const uintptr_t fn_addr =
      reinterpret_cast<uintptr_t>(memory_write_random_pattern_loop_asm);
  std::vector<unsigned char> scratch(512, 0);
  const std::vector<size_t> probe_indices = {128, 0, 64, 32, 96};
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(scratch.data());
  const uintptr_t idx_addr =
      reinterpret_cast<uintptr_t>(probe_indices.data());

  for (size_t count : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
    const uint64_t ok = verify_pattern_callee_saved_registers_asm(
        fn_addr, dst_addr, idx_addr, count, static_cast<uintptr_t>(kPattern),
        0, 0);
    EXPECT_EQ(ok, 1u) << "callee-saved violation count=" << count;
  }
}
