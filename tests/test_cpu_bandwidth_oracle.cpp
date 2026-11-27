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

#include <cstdint>
#include <cstring>
#include <vector>

#include "asm/asm_functions.h"
#include "core/audit/deterministic_rng.h"
#include "core/memory/memory_manager.h"
#include "cpu_bandwidth/cpu_bandwidth_executor.h"
#include "cpu_bandwidth/cpu_bandwidth_oracle.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"

namespace {

uint64_t brute_force_pass_xor(uint64_t seed, size_t offset, size_t span) {
  if (span == 0) {
    return 0;
  }
  const CpuBandwidthPatternTable table = build_cpu_bandwidth_pattern_table(seed);
  const size_t total_end = offset + span;
  const size_t total_words = (total_end + 7u) / 8u;
  std::vector<uint8_t> buf(total_words * 8u, 0);
  for (size_t wi = 0; wi < total_words; ++wi) {
    const uint64_t w = cpu_bandwidth_buffer_word(table, wi);
    std::memcpy(buf.data() + wi * 8u, &w, sizeof(w));
  }
  const size_t vec_bytes = span - (span % 32u);
  uint64_t vec_xor = 0;
  for (size_t pos = offset; pos < offset + vec_bytes; pos += 8u) {
    uint64_t w = 0;
    std::memcpy(&w, buf.data() + pos, sizeof(w));
    vec_xor ^= w;
  }
  uint64_t tail_xor = 0;
  for (size_t pos = offset + vec_bytes; pos < offset + span; ++pos) {
    tail_xor ^= static_cast<uint64_t>(buf[pos]);
  }
  return vec_xor ^ tail_xor;
}

TEST(CpuBandwidthOracleTest, O1PassXorMatchesBruteForceAcrossOffsetsAndTails) {
  const uint64_t seed =
      derive_cpu_bandwidth_target_seed(424242ULL, CpuBandwidthTarget::MainMemory);
  const CpuBandwidthPatternTable table = build_cpu_bandwidth_pattern_table(seed);
  const std::vector<size_t> offsets = {0, 64, 128, 256};
  const std::vector<size_t> spans = {0,   1,   7,   8,   9,   15,  16,  31,
                                     32,  33,  63,  64,  65,  255, 256, 257,
                                     511, 512, 513, 1023, 2065};
  for (size_t off : offsets) {
    for (size_t sp : spans) {
      EXPECT_EQ(compute_cpu_bandwidth_pass_xor(table, off, sp),
                brute_force_pass_xor(seed, off, sp))
          << "off=" << off << " span=" << sp;
    }
  }
}

TEST(CpuBandwidthOracleTest, ClosedFormReadChecksumMatchesIterativePasses) {
  const uint64_t target_seed =
      derive_cpu_bandwidth_target_seed(99ULL, CpuBandwidthTarget::Custom);
  const uint64_t op_seed = 0x123456789ABCDEF0ULL;
  const std::vector<CpuBandwidthWorkerSpan> spans = {{0, 4096}, {4096, 4115}};
  const CpuBandwidthPatternTable table =
      build_cpu_bandwidth_pattern_table(target_seed);

  for (uint64_t passes : {1ULL, 2ULL, 3ULL, 10ULL, 100ULL, 10000ULL}) {
    const auto exp = compute_cpu_bandwidth_expected_read_checksum(
        target_seed, op_seed, passes, spans);
    ASSERT_TRUE(exp.available);
    std::vector<CpuBandwidthExpectedWorkerChecksum> iter_workers;
    for (size_t w = 0; w < spans.size(); ++w) {
      const uint64_t px = compute_cpu_bandwidth_pass_xor(
          table, spans[w].offset_bytes, spans[w].span_bytes);
      uint64_t sa = 0;
      uint64_t sb = 0;
      initial_cpu_bandwidth_worker_checksum_state(op_seed, w, sa, sb);
      for (uint64_t p = 1; p <= passes; ++p) {
        sa += px + kCpuBandwidthChecksumA * p;
        sb += (px ^ kCpuBandwidthChecksumB) * p + kCpuBandwidthChecksumC;
      }
      EXPECT_EQ(exp.workers[w].state_a, sa);
      EXPECT_EQ(exp.workers[w].state_b, sb);
      iter_workers.push_back(
          {w, sa, sb, passes * spans[w].span_bytes, passes});
    }
    const auto folded = fold_cpu_bandwidth_worker_checksums(iter_workers);
    EXPECT_EQ(exp.run_checksum.state_a, folded.state_a);
    EXPECT_EQ(exp.run_checksum.state_b, folded.state_b);
  }

  // Overflow check for passes = 1,000,000,000 (passes * (passes + 1) > 2^59,
  // safe in mod 2^64 arithmetic via even-half decomposition).
  const auto large_exp = compute_cpu_bandwidth_expected_read_checksum(
      target_seed, op_seed, 1000000000ULL, spans);
  EXPECT_TRUE(large_exp.available);
  EXPECT_EQ(large_exp.workers[0].pass_count, 1000000000ULL);
}

TEST(CpuBandwidthOracleIntegrationTest, Arm64ReadKernelsMatchO1PassXorBitForBit) {
  const size_t total_bytes = 4096;
  const uint64_t seed = 0xCAFEBABE12345678ULL;
  const CpuBandwidthPatternTable table = build_cpu_bandwidth_pattern_table(seed);
  MmapPtr buf = allocate_buffer(total_bytes, "oracle_read_test");
  ASSERT_NE(buf, nullptr);
  initialize_cpu_bandwidth_src_buffer(buf.get(), total_bytes, seed, {{0, total_bytes}});

  const auto* bytes = static_cast<const uint8_t*>(buf.get());
  const std::vector<size_t> spans = {0, 1, 7, 15, 31, 32, 63, 64, 511, 512, 513, 2065};
  for (size_t off : {size_t{0}, size_t{64}, size_t{256}}) {
    for (size_t sp : spans) {
      if (off + sp > total_bytes) {
        continue;
      }
      const uint64_t expected =
          compute_cpu_bandwidth_pass_xor(table, off, sp);
      EXPECT_EQ(memory_read_loop_asm(bytes + off, sp), expected)
          << "main read off=" << off << " sp=" << sp;
      EXPECT_EQ(memory_read_cache_loop_asm(bytes + off, sp), expected)
          << "cache read off=" << off << " sp=" << sp;
    }
  }
}

TEST(CpuBandwidthOracleIntegrationTest, WriteAndCopyPostValidationCatchSingleBitFlips) {
  const size_t total_bytes = 16384;
  const std::vector<CpuBandwidthWorkerSpan> spans =
      partition_cpu_bandwidth_64b(total_bytes, 2);

  MmapPtr src = allocate_buffer(total_bytes, "oracle_src");
  MmapPtr dst = allocate_buffer(total_bytes, "oracle_dst");
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);

  const uint64_t src_seed = 0x1122334455667788ULL;
  initialize_cpu_bandwidth_src_buffer(src.get(), total_bytes, src_seed, spans);

  const uint64_t write_pat = derive_cpu_bandwidth_write_pattern(777ULL);
  EXPECT_NE(write_pat, 0ULL);
  EXPECT_EQ(write_pat & 1ULL, 1ULL);

  for (const auto& sp : spans) {
    memory_write_pattern_loop_asm(
        static_cast<uint8_t*>(dst.get()) + sp.offset_bytes, sp.span_bytes,
        write_pat);
  }
  EXPECT_TRUE(
      validate_cpu_bandwidth_write_buffer(dst.get(), total_bytes, write_pat, spans));

  // Flip 1 bit in last byte -> must detect corruption.
  auto* dst_u8 = static_cast<uint8_t*>(dst.get());
  dst_u8[total_bytes - 1] ^= 0x01U;
  EXPECT_FALSE(
      validate_cpu_bandwidth_write_buffer(dst.get(), total_bytes, write_pat, spans));

  // Sentinel before copy, then copy, verify, and flip 1 bit in first byte.
  initialize_cpu_bandwidth_dst_sentinel(dst.get(), total_bytes, 0xA5A5A5A5A5A5A5A5ULL,
                                        false, spans);
  EXPECT_FALSE(validate_cpu_bandwidth_copy_buffer(dst.get(), src.get(),
                                                  total_bytes, spans));
  for (const auto& sp : spans) {
    memory_copy_loop_asm(static_cast<uint8_t*>(dst.get()) + sp.offset_bytes,
                         static_cast<const uint8_t*>(src.get()) + sp.offset_bytes,
                         sp.span_bytes);
  }
  EXPECT_TRUE(validate_cpu_bandwidth_copy_buffer(dst.get(), src.get(),
                                                 total_bytes, spans));
  dst_u8[0] ^= 0x80U;
  EXPECT_FALSE(validate_cpu_bandwidth_copy_buffer(dst.get(), src.get(),
                                                  total_bytes, spans));
}

}  // namespace
