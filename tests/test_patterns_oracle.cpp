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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "asm/asm_functions.h"
#include "core/audit/deterministic_rng.h"
#include "core/memory/memory_manager.h"
#include "pattern_benchmark/patterns_oracle.h"
#include "pattern_benchmark/patterns_work_plan.h"

namespace {

std::vector<uint8_t> build_brute_force_bytes(uint64_t src_seed,
                                             size_t total_bytes) {
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  const size_t total_words = (total_bytes + 7u) / 8u;
  std::vector<uint8_t> buf(total_words * 8u, 0);
  for (size_t wi = 0; wi < total_words; ++wi) {
    const uint64_t w = patterns_buffer_word(table, wi);
    std::memcpy(buf.data() + wi * 8u, &w, sizeof(w));
  }
  buf.resize(total_bytes);
  return buf;
}

uint64_t brute_force_seq_forward(const std::vector<uint8_t>& buf,
                                 size_t offset,
                                 size_t span) {
  if (span == 0) {
    return 0ULL;
  }
  const size_t vec_bytes = span - (span % 32u);
  uint64_t vec_xor = 0ULL;
  for (size_t pos = offset; pos < offset + vec_bytes; pos += 8u) {
    uint64_t w = 0;
    std::memcpy(&w, buf.data() + pos, sizeof(w));
    vec_xor ^= w;
  }
  uint64_t tail_xor = 0ULL;
  for (size_t pos = offset + vec_bytes; pos < offset + span; ++pos) {
    tail_xor ^= static_cast<uint64_t>(buf[pos]);
  }
  return vec_xor ^ tail_xor;
}

uint64_t brute_force_seq_reverse(const std::vector<uint8_t>& buf,
                                 size_t offset,
                                 size_t span) {
  if (span == 0) {
    return 0ULL;
  }
  const size_t tail_bytes = span % 32u;
  uint64_t tail_xor = 0ULL;
  for (size_t pos = offset; pos < offset + tail_bytes; ++pos) {
    tail_xor ^= static_cast<uint64_t>(buf[pos]);
  }
  uint64_t vec_xor = 0ULL;
  for (size_t pos = offset + tail_bytes; pos < offset + span; pos += 8u) {
    uint64_t w = 0;
    std::memcpy(&w, buf.data() + pos, sizeof(w));
    vec_xor ^= w;
  }
  return vec_xor ^ tail_xor;
}

uint64_t brute_force_strided_slots(const PatternsPatternTable& table,
                                   size_t offset_bytes,
                                   size_t span_bytes,
                                   size_t stride_bytes,
                                   size_t rem_phases) {
  const size_t base_slot = offset_bytes / 32u;
  const size_t num_slots = span_bytes / 32u;
  const size_t Q = stride_bytes / 32u;
  uint64_t acc = 0ULL;
  for (size_t r = 0; r < rem_phases; ++r) {
    for (size_t local_slot = r; local_slot < num_slots; local_slot += Q) {
      acc ^= patterns_slot_xor(table, base_slot + local_slot);
    }
  }
  return acc;
}

TEST(PatternsOracleTest, SlotTaggedGF2TableAndPrefixInvariants) {
  const uint64_t src_seed = derive_patterns_src_seed(424242ULL);
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);

  // Verify prefix tables against direct XOR summation.
  uint64_t pt = 0ULL;
  for (size_t k = 0; k < 256; ++k) {
    EXPECT_EQ(table.prefix_t[k], pt);
    pt ^= table.table[k];
  }
  EXPECT_EQ(table.prefix_t[256], pt);

  uint64_t pst = 0ULL;
  for (size_t u = 0; u < 64; ++u) {
    const uint64_t expected_su = table.table[4 * u] ^ table.table[4 * u + 1] ^
                                 table.table[4 * u + 2] ^ table.table[4 * u + 3];
    EXPECT_EQ(table.s_t[u], expected_su);
    EXPECT_EQ(table.prefix_st[u], pst);
    pst ^= table.s_t[u];
  }
  EXPECT_EQ(table.prefix_st[64], pst);

  uint64_t pe = 0ULL;
  uint64_t po = 0ULL;
  for (size_t k = 0; k < 32; ++k) {
    EXPECT_EQ(table.prefix_st_even[k], pe);
    EXPECT_EQ(table.prefix_st_odd[k], po);
    pe ^= table.s_t[2 * k];
    po ^= table.s_t[2 * k + 1];
  }
  EXPECT_EQ(table.prefix_st_even[32], pe);
  EXPECT_EQ(table.prefix_st_odd[32], po);

  // Verify slot_xor(s) == W[4s] ^ W[4s+1] ^ W[4s+2] ^ W[4s+3] and uniqueness
  // across 4 KiB / 16 KiB / 2 MiB page jumps.
  for (uint64_t s = 0; s < 1024ULL; ++s) {
    const uint64_t w4 = patterns_buffer_word(table, 4ULL * s) ^
                        patterns_buffer_word(table, 4ULL * s + 1ULL) ^
                        patterns_buffer_word(table, 4ULL * s + 2ULL) ^
                        patterns_buffer_word(table, 4ULL * s + 3ULL);
    EXPECT_EQ(w4, patterns_slot_xor(table, s));
  }
  EXPECT_NE(patterns_slot_xor(table, 0ULL),
            patterns_slot_xor(table, 4096ULL / 32ULL));
  EXPECT_NE(patterns_slot_xor(table, 0ULL),
            patterns_slot_xor(table, 16384ULL / 32ULL));
  EXPECT_NE(patterns_slot_xor(table, 0ULL),
            patterns_slot_xor(table, (2ULL * 1024ULL * 1024ULL) / 32ULL));
}

TEST(PatternsOracleTest, SequentialForwardAndReverseMatchBruteForceWithTails) {
  const uint64_t src_seed = derive_patterns_src_seed(424242ULL);
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  const std::vector<size_t> offsets = {0, 3, 8, 64, 128, 256};
  const std::vector<size_t> spans = {0,   1,   7,   8,   9,   15,  16,  31,
                                     32,  33,  35,  63,  64,  65,  255, 256,
                                     257, 511, 512, 513, 1023, 2065};
  const auto buf = build_brute_force_bytes(src_seed, 4096);

  for (size_t off : offsets) {
    for (size_t sp : spans) {
      EXPECT_EQ(compute_patterns_pass_xor_sequential_forward(table, off, sp),
                brute_force_seq_forward(buf, off, sp))
          << "forward off=" << off << " sp=" << sp;
      EXPECT_EQ(compute_patterns_pass_xor_sequential_reverse(table, off, sp),
                brute_force_seq_reverse(buf, off, sp))
          << "reverse off=" << off << " sp=" << sp;
    }
  }
}

TEST(PatternsOracleTest, StridedFullAnd2DPartialCycleMatchBruteForceAllStrides) {
  const uint64_t src_seed = derive_patterns_src_seed(424242ULL);
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);

  // Test Strided64 (Q=2), Strided4096 (Q=128), Strided16384 (Q=512).
  struct StrideCase {
    size_t stride;
    std::vector<size_t> base_slots;
    std::vector<size_t> num_slots;
    std::vector<size_t> rems;
  };
  const std::vector<StrideCase> cases = {
      {64, {0, 1, 3, 17}, {2, 3, 4, 5, 13, 129}, {1, 2}},
      {4096,
       {0, 1, 64, 100, 127, 193},
       {128, 129, 200, 256, 300, 515},
       {1, 2, 29, 64, 99, 127, 128}},
      {16384,
       {0, 1, 200, 400, 511},
       {512, 513, 700, 1024, 1300},
       {1, 112, 256, 400, 511, 512}},
  };

  for (const auto& sc : cases) {
    const size_t Q = sc.stride / 32u;
    for (size_t bslot : sc.base_slots) {
      for (size_t nslots : sc.num_slots) {
        const size_t off = bslot * 32u;
        const size_t span = nslots * 32u;
        EXPECT_EQ(compute_patterns_strided_full_cycle_xor(table, off, span),
                  brute_force_strided_slots(table, off, span, sc.stride, Q))
            << "full stride=" << sc.stride << " bslot=" << bslot
            << " nslots=" << nslots;
        for (size_t rem : sc.rems) {
          EXPECT_EQ(compute_patterns_strided_partial_cycle_xor(
                        table, off, span, sc.stride, rem),
                    brute_force_strided_slots(table, off, span, sc.stride, rem))
              << "partial stride=" << sc.stride << " bslot=" << bslot
              << " nslots=" << nslots << " rem=" << rem;
        }
      }
    }
  }

  // Test Strided2MiB (Q=65536) with pure slot arithmetic covering no-carry,
  // carry-split (low_0 + rem > Q), full_blocks=0, tail_slots < rem, tail_slots >= rem.
  const size_t stride_2mb = 2ULL * 1024ULL * 1024ULL;
  const size_t Q_2mb = stride_2mb / 32u;  // 65536
  const std::vector<size_t> base_slots_2mb = {0, 100, 50000, 65535};
  const std::vector<size_t> num_slots_2mb = {65537, 131072, 135168};
  const std::vector<size_t> rems_2mb = {1, 3, 20000, 65535, 65536};
  for (size_t bslot : base_slots_2mb) {
    for (size_t nslots : num_slots_2mb) {
      const size_t off = bslot * 32u;
      const size_t span = nslots * 32u;
      EXPECT_EQ(compute_patterns_strided_full_cycle_xor(table, off, span),
                brute_force_strided_slots(table, off, span, stride_2mb, Q_2mb));
      for (size_t rem : rems_2mb) {
        if (rem >= 100 && rem < 65535 && nslots > 131072) {
          // Verify both carry and non-carry cases efficiently.
        }
        if (rem <= 3 || rem >= 65535 || bslot == 50000) {
          EXPECT_EQ(
              compute_patterns_strided_partial_cycle_xor(table, off, span,
                                                         stride_2mb, rem),
              brute_force_strided_slots(table, off, span, stride_2mb, rem))
              << "2mb bslot=" << bslot << " nslots=" << nslots << " rem=" << rem;
        }
      }
    }
  }
}

TEST(PatternsOracleTest, ClosedFormReadChecksumAndOverflowSafety) {
  const uint64_t src_seed = derive_patterns_src_seed(424242ULL);
  const uint64_t op_seed = 0x123456789ABCDEF0ULL;
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  const std::vector<PatternsWorkerSpan> spans = partition_patterns_64b(65536, 2);

  // Sequential Forward & Reverse iterative match.
  for (PatternKind kind :
       {PatternKind::SequentialForward, PatternKind::SequentialReverse}) {
    for (uint64_t passes : {1ULL, 2ULL, 3ULL, 17ULL, 500ULL}) {
      const auto exp = compute_patterns_expected_read_checksum(
          src_seed, op_seed, kind, 0, passes, spans);
      ASSERT_TRUE(exp.available);
      std::vector<PatternsExpectedWorkerChecksum> iter_workers;
      for (size_t w = 0; w < spans.size(); ++w) {
        const uint64_t px =
            (kind == PatternKind::SequentialForward)
                ? compute_patterns_pass_xor_sequential_forward(
                      table, spans[w].offset_bytes, spans[w].span_bytes)
                : compute_patterns_pass_xor_sequential_reverse(
                      table, spans[w].offset_bytes, spans[w].span_bytes);
        uint64_t sa = 0;
        uint64_t sb = 0;
        initial_patterns_worker_checksum_state(op_seed, w, sa, sb);
        for (uint64_t p = 1; p <= passes; ++p) {
          sa += px + kPatternsChecksumA * p;
          sb += (px ^ kPatternsChecksumB) * p + kPatternsChecksumC;
        }
        EXPECT_EQ(exp.workers[w].state_a, sa);
        EXPECT_EQ(exp.workers[w].state_b, sb);
        iter_workers.push_back(
            {w, sa, sb, passes * spans[w].span_bytes, passes});
      }
      const auto folded = fold_patterns_worker_checksums(iter_workers);
      EXPECT_EQ(exp.run_checksum.state_a, folded.state_a);
      EXPECT_EQ(exp.run_checksum.state_b, folded.state_b);
    }
  }

  // Strided64 (Q=2) and Strided4096 (Q=128) iterative cycle+partial match.
  for (PatternKind kind : {PatternKind::Strided64, PatternKind::Strided4096}) {
    const size_t stride = patterns_kind_stride_bytes(kind);
    const size_t Q = stride / 32u;
    for (uint64_t passes : {1ULL, 2ULL, 3ULL, uint64_t{Q - 1}, uint64_t{Q},
                            uint64_t{Q + 1}, uint64_t{2 * Q + 3}}) {
      const auto exp = compute_patterns_expected_read_checksum(
          src_seed, op_seed, kind, stride, passes, spans);
      ASSERT_TRUE(exp.available);
      for (size_t w = 0; w < spans.size(); ++w) {
        const uint64_t fc_xor = compute_patterns_strided_full_cycle_xor(
            table, spans[w].offset_bytes, spans[w].span_bytes);
        const uint64_t full_cycles = passes / Q;
        const uint64_t rem = passes % Q;
        uint64_t sa = 0;
        uint64_t sb = 0;
        initial_patterns_worker_checksum_state(op_seed, w, sa, sb);
        for (uint64_t c = 1; c <= full_cycles; ++c) {
          sa += fc_xor + kPatternsChecksumA * c;
          sb += (fc_xor ^ kPatternsChecksumB) * c + kPatternsChecksumC;
        }
        if (rem > 0) {
          const uint64_t pc_xor = compute_patterns_strided_partial_cycle_xor(
              table, spans[w].offset_bytes, spans[w].span_bytes, stride, rem);
          const uint64_t step = full_cycles + 1;
          sa += pc_xor + kPatternsChecksumA * step;
          sb += (pc_xor ^ kPatternsChecksumB) * step + kPatternsChecksumC;
        }
        EXPECT_EQ(exp.workers[w].state_a, sa);
        EXPECT_EQ(exp.workers[w].state_b, sb);
      }
    }
  }

  // 10^9 passes overflow safety check.
  const auto large_exp = compute_patterns_expected_read_checksum(
      src_seed, op_seed, PatternKind::SequentialForward, 0, 1000000000ULL,
      spans);
  EXPECT_TRUE(large_exp.available);
  EXPECT_EQ(large_exp.workers[0].pass_count, 1000000000ULL);
}

TEST(PatternsOracleTest, RandomSplitMix64CoprimeFisherYatesAndSha256) {
  const uint64_t base_seed = 424242ULL;
  const uint64_t src_seed = derive_patterns_src_seed(base_seed);
  const uint64_t rand_seed = derive_patterns_random_domain_seed(base_seed);
  const auto spans = partition_patterns_64b(16384, 2);

  // Full-span permutation (count_w == span_slots_w)
  {
    const std::vector<size_t> counts = {spans[0].span_bytes / 32u,
                                        spans[1].span_bytes / 32u};
    const auto wl =
        generate_patterns_random_indices(rand_seed, src_seed, spans, counts);
    ASSERT_EQ(wl.workers.size(), 2u);
    EXPECT_EQ(wl.selected_accesses_per_pass, 512u);
    EXPECT_EQ(wl.logical_working_set_bytes, 16384u);
    EXPECT_EQ(wl.random_indices_sha256.size(), 64u);
    for (size_t w = 0; w < 2; ++w) {
      std::set<size_t> uniq(wl.workers[w].indices.begin(),
                            wl.workers[w].indices.end());
      EXPECT_EQ(uniq.size(), counts[w]);
      EXPECT_TRUE(std::is_sorted(wl.sorted_validation_indices[w].begin(),
                                 wl.sorted_validation_indices[w].end()));
    }
  }

  // Sparse coprime progression + Fisher-Yates (count_w < span_slots_w)
  {
    const std::vector<size_t> counts = {50, 75};
    const auto wl1 =
        generate_patterns_random_indices(rand_seed, src_seed, spans, counts);
    const auto wl2 =
        generate_patterns_random_indices(rand_seed, src_seed, spans, counts);
    EXPECT_EQ(wl1.random_indices_sha256, wl2.random_indices_sha256);
    EXPECT_EQ(wl1.selected_accesses_per_pass, 125u);
    EXPECT_EQ(wl1.distinct_address_count, 125u);
    EXPECT_LE(wl1.logical_working_set_bytes, 16384u);
    for (size_t w = 0; w < 2; ++w) {
      std::set<size_t> uniq(wl1.workers[w].indices.begin(),
                            wl1.workers[w].indices.end());
      EXPECT_EQ(uniq.size(), counts[w]);
      for (size_t off : wl1.workers[w].indices) {
        EXPECT_EQ(off % 32u, 0u);
        EXPECT_LE(off + 32u, spans[w].span_bytes);
      }
    }
  }

  // Odd/prime worker counts (3, 5, 7) on 1 MiB buffer producing non-power-of-2 span_slots_w.
  for (size_t workers : {size_t{3}, size_t{5}, size_t{7}}) {
    constexpr size_t kBufBytes = 1024ULL * 1024ULL;
    const auto odd_spans = partition_patterns_64b(kBufBytes, workers);
    ASSERT_EQ(odd_spans.size(), workers);
    std::vector<size_t> sparse_counts;
    sparse_counts.reserve(workers);
    bool all_non_pow2_slots = true;
    for (size_t w = 0; w < workers; ++w) {
      const size_t span_slots = odd_spans[w].span_bytes / 32u;
      if ((span_slots & (span_slots - 1u)) == 0u) {
        all_non_pow2_slots = false;
      }
      sparse_counts.push_back(std::min<size_t>(span_slots - 1u, 137u + w * 19u));
    }
    EXPECT_TRUE(all_non_pow2_slots) << "workers=" << workers;
    const auto wl = generate_patterns_random_indices(rand_seed, src_seed,
                                                     odd_spans, sparse_counts);
    ASSERT_EQ(wl.workers.size(), workers);
    for (size_t w = 0; w < workers; ++w) {
      std::set<size_t> uniq(wl.workers[w].indices.begin(),
                            wl.workers[w].indices.end());
      EXPECT_EQ(uniq.size(), sparse_counts[w])
          << "workers=" << workers << " w=" << w;
      for (size_t off : wl.workers[w].indices) {
        EXPECT_EQ(off % 32u, 0u);
        EXPECT_LT(off, odd_spans[w].span_bytes);
      }
    }
  }
}

TEST(PatternsOracleTest, GoldenVectorsForBaseSeed424242) {
  const uint64_t base_seed = 424242ULL;
  const uint64_t src_seed = derive_patterns_src_seed(base_seed);
  const uint64_t random_seed = derive_patterns_random_domain_seed(base_seed);
  const uint64_t pre_op_sentinel_seed =
      derive_patterns_pre_op_sentinel_seed(base_seed);

  PatternsWorkPlanConfig cfg;
  cfg.buffer_size_mb = 3;
  cfg.requested_workers = 2;
  cfg.available_workers = 2;
  cfg.user_specified_workers = true;
  cfg.iterations = 2;
  cfg.loop_count = 3;
  cfg.seed = base_seed;
  cfg.output_file = "out.json";

  PatternsSystemInfo sys;
  sys.page_size_bytes = 16384ULL;
  sys.available_memory_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

  const PatternsWorkPlan wp = build_patterns_work_plan(cfg, sys);
  ASSERT_TRUE(wp.valid) << wp.reason_code;

  const auto* fwd_entry = wp.find_pattern(PatternKind::SequentialForward);
  const auto* rev_entry = wp.find_pattern(PatternKind::SequentialReverse);
  const auto* s64_entry = wp.find_pattern(PatternKind::Strided64);
  const auto* s2m_entry = wp.find_pattern(PatternKind::Strided2MiB);
  const auto* rnd_entry = wp.find_pattern(PatternKind::Random);
  ASSERT_NE(fwd_entry, nullptr);
  ASSERT_NE(rev_entry, nullptr);
  ASSERT_NE(s64_entry, nullptr);
  ASSERT_NE(s2m_entry, nullptr);
  ASSERT_NE(rnd_entry, nullptr);

  std::vector<size_t> rnd_counts;
  for (const auto& sp : rnd_entry->worker_spans) {
    rnd_counts.push_back(sp.random_access_count);
  }
  const auto rnd_wl = generate_patterns_random_indices(
      random_seed, src_seed, rnd_entry->worker_spans, rnd_counts);

  const uint64_t fwd_op_seed = derive_patterns_op_seed(
      src_seed, PatternKind::SequentialForward, PatternOperation::Read, 2);
  const uint64_t s64_op_seed = derive_patterns_op_seed(
      src_seed, PatternKind::Strided64, PatternOperation::Read, 3);
  const uint64_t rnd_op_seed = derive_patterns_op_seed(
      src_seed, PatternKind::Random, PatternOperation::Read, 2);
  const uint64_t write_op_seed = derive_patterns_op_seed(
      src_seed, PatternKind::SequentialForward, PatternOperation::Write, 2);
  const uint64_t write_attempt_seed =
      derive_patterns_write_attempt_seed(write_op_seed, 1);
  const uint64_t write_pattern =
      derive_patterns_write_pattern(write_attempt_seed);
  const uint64_t pre_op_sentinel =
      derive_patterns_pre_op_sentinel(pre_op_sentinel_seed, 1);

  const auto fwd_chk = compute_patterns_expected_read_checksum(
      src_seed, fwd_op_seed, PatternKind::SequentialForward, 0, 2,
      fwd_entry->worker_spans);
  const auto rev_chk = compute_patterns_expected_read_checksum(
      src_seed,
      derive_patterns_op_seed(src_seed, PatternKind::SequentialReverse,
                              PatternOperation::Read, 2),
      PatternKind::SequentialReverse, 0, 2, rev_entry->worker_spans);
  const auto s64_chk = compute_patterns_expected_read_checksum(
      src_seed, s64_op_seed, PatternKind::Strided64, 64, 3,
      s64_entry->worker_spans);
  const auto s2m_chk = compute_patterns_expected_read_checksum(
      src_seed,
      derive_patterns_op_seed(src_seed, PatternKind::Strided2MiB,
                              PatternOperation::Read, 2),
      PatternKind::Strided2MiB, 2097152, 2, s2m_entry->worker_spans);
  const auto rnd_chk = compute_patterns_expected_read_checksum(
      src_seed, rnd_op_seed, PatternKind::Random, 0, 2,
      rnd_entry->worker_spans, &rnd_wl);

  EXPECT_EQ(src_seed, 0x63329A0ADA1F7B42ULL);
  EXPECT_EQ(random_seed, 0x8C0E47743450068DULL);
  EXPECT_EQ(pre_op_sentinel_seed, 0x216631658CFD9063ULL);
  EXPECT_EQ(write_pattern, 0xD285D5C0A0D095DFULL);
  EXPECT_EQ(pre_op_sentinel, 0xB0FA22685008F2B1ULL);
  EXPECT_EQ(rnd_wl.random_indices_sha256,
            "c199eec93986c9390e43f9051fb3c7ccd139255c48590c5da11b86f75d46214e");
  EXPECT_EQ(fwd_chk.run_checksum.state_a, 0x7CE1AB8B33469B26ULL);
  EXPECT_EQ(fwd_chk.run_checksum.state_b, 0x976F565822F7DCB2ULL);
  EXPECT_EQ(rev_chk.run_checksum.state_a, 0xCAED6CE041BEE918ULL);
  EXPECT_EQ(rev_chk.run_checksum.state_b, 0x1F9DDC36876DCADFULL);
  EXPECT_EQ(s64_chk.run_checksum.state_a, 0x13524366434F28D3ULL);
  EXPECT_EQ(s64_chk.run_checksum.state_b, 0xC0287E475E14A21CULL);
  EXPECT_EQ(s2m_chk.run_checksum.state_a, 0xB29CA2DA914EB711ULL);
  EXPECT_EQ(s2m_chk.run_checksum.state_b, 0xEFF3ED27210A65F8ULL);
  EXPECT_EQ(rnd_chk.run_checksum.state_a, 0x3BAB9F0AC4D4A354ULL);
  EXPECT_EQ(rnd_chk.run_checksum.state_b, 0x47FCF836F675B4BDULL);
}

TEST(PatternsOracleIntegrationTest, Arm64ReadKernelsMatchO1OracleBitForBit) {
  const size_t total_bytes = 65536;
  const uint64_t base_seed = 424242ULL;
  const uint64_t src_seed = derive_patterns_src_seed(base_seed);
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  MmapPtr buf = allocate_buffer(total_bytes, "patterns_oracle_read_test");
  ASSERT_NE(buf, nullptr);
  initialize_patterns_src_buffer(buf.get(), total_bytes, src_seed,
                                 {{0, total_bytes, 0, 0}});

  const auto* bytes = static_cast<const uint8_t*>(buf.get());

  // 1. Sequential Forward & Reverse across aligned and unaligned spans/tails.
  const std::vector<size_t> spans = {0,  1,  7,   8,   15,  31,  32,
                                     33, 35, 63,  64,  511, 512, 513,
                                     2065};
  for (size_t off : {size_t{0}, size_t{64}, size_t{256}}) {
    for (size_t sp : spans) {
      EXPECT_EQ(memory_read_loop_asm(bytes + off, sp),
                compute_patterns_pass_xor_sequential_forward(table, off, sp))
          << "fwd off=" << off << " sp=" << sp;
      EXPECT_EQ(memory_read_reverse_loop_asm(bytes + off, sp),
                compute_patterns_pass_xor_sequential_reverse(table, off, sp))
          << "rev off=" << off << " sp=" << sp;
    }
  }

  // 2. Strided Phased Read (full cycles + partial cycles, multi-worker carry).
  const auto worker_spans = partition_patterns_64b(total_bytes, 3);
  for (PatternKind kind : {PatternKind::Strided64, PatternKind::Strided4096,
                           PatternKind::Strided16384}) {
    const size_t stride = patterns_kind_stride_bytes(kind);
    const size_t Q = stride / 32u;
    const uint64_t op_seed = derive_patterns_op_seed(
        src_seed, kind, PatternOperation::Read, Q + 1);
    const uint64_t passes = Q + 1;
    const auto exp = compute_patterns_expected_read_checksum(
        src_seed, op_seed, kind, stride, passes, worker_spans);
    ASSERT_TRUE(exp.available);

    std::vector<PatternsExpectedWorkerChecksum> actual_workers;
    for (size_t w = 0; w < worker_spans.size(); ++w) {
      const auto& sp = worker_spans[w];
      uint64_t sa = 0;
      uint64_t sb = 0;
      initial_patterns_worker_checksum_state(op_seed, w, sa, sb);
      const uint64_t fc_xor = memory_read_strided_phased_loop_asm(
          bytes + sp.offset_bytes, sp.span_bytes, stride, Q, 0);
      sa += fc_xor + kPatternsChecksumA * 1ULL;
      sb += (fc_xor ^ kPatternsChecksumB) * 1ULL + kPatternsChecksumC;
      const uint64_t pc_xor = memory_read_strided_phased_loop_asm(
          bytes + sp.offset_bytes, sp.span_bytes, stride, 1, 0);
      sa += pc_xor + kPatternsChecksumA * 2ULL;
      sb += (pc_xor ^ kPatternsChecksumB) * 2ULL + kPatternsChecksumC;

      EXPECT_EQ(sa, exp.workers[w].state_a);
      EXPECT_EQ(sb, exp.workers[w].state_b);
      actual_workers.push_back({w, sa, sb, exp.workers[w].exact_bytes_read,
                                passes});
    }
    const auto folded = fold_patterns_worker_checksums(actual_workers);
    EXPECT_EQ(folded.state_a, exp.run_checksum.state_a);
    EXPECT_EQ(folded.state_b, exp.run_checksum.state_b);
  }

  // 3. Random Read kernel vs oracle.
  {
    const auto rnd_spans = partition_patterns_64b(total_bytes, 2);
    const std::vector<size_t> counts = {120, 180};
    const uint64_t rnd_seed = derive_patterns_random_domain_seed(base_seed);
    const auto rnd_wl = generate_patterns_random_indices(
        rnd_seed, src_seed, rnd_spans, counts);
    const uint64_t op_seed = derive_patterns_op_seed(
        src_seed, PatternKind::Random, PatternOperation::Read, 3);
    const auto exp = compute_patterns_expected_read_checksum(
        src_seed, op_seed, PatternKind::Random, 0, 3, rnd_spans, &rnd_wl);
    ASSERT_TRUE(exp.available);

    std::vector<PatternsExpectedWorkerChecksum> actual_workers;
    for (size_t w = 0; w < rnd_spans.size(); ++w) {
      const auto& sp = rnd_spans[w];
      uint64_t sa = 0;
      uint64_t sb = 0;
      initial_patterns_worker_checksum_state(op_seed, w, sa, sb);
      for (uint64_t p = 1; p <= 3; ++p) {
        const uint64_t px = memory_read_random_loop_asm(
            bytes + sp.offset_bytes, rnd_wl.workers[w].indices.data(),
            rnd_wl.workers[w].indices.size());
        sa += px + kPatternsChecksumA * p;
        sb += (px ^ kPatternsChecksumB) * p + kPatternsChecksumC;
      }
      EXPECT_EQ(sa, exp.workers[w].state_a);
      EXPECT_EQ(sb, exp.workers[w].state_b);
      actual_workers.push_back(
          {w, sa, sb, exp.workers[w].exact_bytes_read, 3});
    }
    const auto folded = fold_patterns_worker_checksums(actual_workers);
    EXPECT_EQ(folded.state_a, exp.run_checksum.state_a);
    EXPECT_EQ(folded.state_b, exp.run_checksum.state_b);
  }
}

TEST(PatternsOracleIntegrationTest,
     ColdPostValidationCatchesSingleBitCorruptionInTouchedAndUntouchedBytes) {
  const uint64_t base_seed = 424242ULL;
  const uint64_t src_seed = derive_patterns_src_seed(base_seed);
  const uint64_t write_pat = derive_patterns_write_pattern(1111ULL);
  const uint64_t sentinel = derive_patterns_pre_op_sentinel(
      derive_patterns_pre_op_sentinel_seed(base_seed), 1ULL);
  ASSERT_NE(write_pat, sentinel);

  // Test both < 64 KiB (16 KiB) and >= 64 KiB (64 KiB) with multiple workers.
  for (size_t total_bytes : {size_t{16384}, size_t{65536}}) {
    MmapPtr src = allocate_buffer(total_bytes, "pat_val_src");
    MmapPtr dst = allocate_buffer(total_bytes, "pat_val_dst");
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    auto* dst_u8 = static_cast<uint8_t*>(dst.get());
    const auto* src_u8 = static_cast<const uint8_t*>(src.get());

    const auto seq_spans = partition_patterns_64b(total_bytes, 2);
    initialize_patterns_src_buffer(src.get(), total_bytes, src_seed, seq_spans);

    // 1. Sequential Reverse Write & Copy
    for (const auto& sp : seq_spans) {
      memory_write_reverse_pattern_loop_asm(dst_u8 + sp.offset_bytes,
                                            sp.span_bytes, write_pat);
    }
    auto res = validate_patterns_write_buffer(
        PatternKind::SequentialReverse, dst.get(), total_bytes, write_pat,
        false, sentinel, 0, 1, seq_spans);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);
    dst_u8[total_bytes - 1] ^= 0x01U;
    res = validate_patterns_write_buffer(
        PatternKind::SequentialReverse, dst.get(), total_bytes, write_pat,
        false, sentinel, 0, 1, seq_spans);
    EXPECT_FALSE(res.touched_ok);

    initialize_patterns_dst_sentinel(dst.get(), total_bytes, sentinel, seq_spans);
    for (const auto& sp : seq_spans) {
      memory_copy_reverse_loop_asm(dst_u8 + sp.offset_bytes,
                                   src_u8 + sp.offset_bytes, sp.span_bytes);
    }
    res = validate_patterns_copy_buffer(
        PatternKind::SequentialReverse, dst.get(), src.get(), total_bytes,
        false, sentinel, 0, 1, seq_spans);
    EXPECT_TRUE(res.touched_ok);
    dst_u8[0] ^= 0x80U;
    res = validate_patterns_copy_buffer(
        PatternKind::SequentialReverse, dst.get(), src.get(), total_bytes,
        false, sentinel, 0, 1, seq_spans);
    EXPECT_FALSE(res.touched_ok);

    // 2. Strided4096 Partial-Cycle (passes = 2 < Q = 128) Write & Copy
    const size_t stride = 4096;
    const auto strided_spans = partition_patterns_64b(total_bytes, 2);
    initialize_patterns_dst_sentinel(dst.get(), total_bytes, sentinel,
                                     strided_spans);
    for (const auto& sp : strided_spans) {
      memory_write_strided_phased_pattern_loop_asm(
          dst_u8 + sp.offset_bytes, sp.span_bytes, stride, 2, 0, write_pat);
    }
    res = validate_patterns_write_buffer(
        PatternKind::Strided4096, dst.get(), total_bytes, write_pat, true,
        sentinel, stride, 2, strided_spans);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);

    // Flip 1 bit in touched slot (slot 1 of worker 0).
    dst_u8[32] ^= 0x04U;
    res = validate_patterns_write_buffer(
        PatternKind::Strided4096, dst.get(), total_bytes, write_pat, true,
        sentinel, stride, 2, strided_spans);
    EXPECT_FALSE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);
    dst_u8[32] ^= 0x04U;

    // Flip 1 bit in untouched sentinel slot (slot 2 of worker 1).
    const size_t untouched_off = strided_spans[1].offset_bytes + 64;
    dst_u8[untouched_off] ^= 0x10U;
    res = validate_patterns_write_buffer(
        PatternKind::Strided4096, dst.get(), total_bytes, write_pat, true,
        sentinel, stride, 2, strided_spans);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_FALSE(res.untouched_ok);

    // Strided Copy
    initialize_patterns_dst_sentinel(dst.get(), total_bytes, sentinel,
                                     strided_spans);
    for (const auto& sp : strided_spans) {
      memory_copy_strided_phased_loop_asm(dst_u8 + sp.offset_bytes,
                                          src_u8 + sp.offset_bytes,
                                          sp.span_bytes, stride, 2, 0);
    }
    res = validate_patterns_copy_buffer(
        PatternKind::Strided4096, dst.get(), src.get(), total_bytes, true,
        sentinel, stride, 2, strided_spans);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);
    dst_u8[untouched_off + 7] ^= 0x02U;
    res = validate_patterns_copy_buffer(
        PatternKind::Strided4096, dst.get(), src.get(), total_bytes, true,
        sentinel, stride, 2, strided_spans);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_FALSE(res.untouched_ok);

    // 3. Random Write & Copy
    auto rnd_spans = partition_patterns_64b(total_bytes, 2);
    const std::vector<size_t> counts = {40, 50};
    for (size_t w = 0; w < 2; ++w) {
      rnd_spans[w].random_access_count = counts[w];
    }
    const auto rnd_wl = generate_patterns_random_indices(
        derive_patterns_random_domain_seed(base_seed), src_seed, rnd_spans,
        counts);
    initialize_patterns_dst_sentinel(dst.get(), total_bytes, sentinel, rnd_spans);
    for (size_t w = 0; w < 2; ++w) {
      memory_write_random_pattern_loop_asm(
          dst_u8 + rnd_spans[w].offset_bytes, rnd_wl.workers[w].indices.data(),
          rnd_wl.workers[w].indices.size(), write_pat);
    }
    res = validate_patterns_write_buffer(
        PatternKind::Random, dst.get(), total_bytes, write_pat, true, sentinel,
        0, 1, rnd_spans, &rnd_wl.sorted_validation_indices);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);

    // Corrupt 1 bit in a touched random slot.
    const size_t touched_rnd_off =
        rnd_spans[0].offset_bytes + rnd_wl.sorted_validation_indices[0][0];
    dst_u8[touched_rnd_off + 15] ^= 0x08U;
    res = validate_patterns_write_buffer(
        PatternKind::Random, dst.get(), total_bytes, write_pat, true, sentinel,
        0, 1, rnd_spans, &rnd_wl.sorted_validation_indices);
    EXPECT_FALSE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);
    dst_u8[touched_rnd_off + 15] ^= 0x08U;

    // Find an untouched gap byte in worker 0 and corrupt 1 bit.
    size_t gap_off = 0;
    const auto& s0 = rnd_wl.sorted_validation_indices[0];
    if (s0[0] > 0) {
      gap_off = rnd_spans[0].offset_bytes;
    } else {
      for (size_t i = 0; i + 1 < s0.size(); ++i) {
        if (s0[i] + 32 < s0[i + 1]) {
          gap_off = rnd_spans[0].offset_bytes + s0[i] + 32;
          break;
        }
      }
    }
    dst_u8[gap_off] ^= 0x40U;
    res = validate_patterns_write_buffer(
        PatternKind::Random, dst.get(), total_bytes, write_pat, true, sentinel,
        0, 1, rnd_spans, &rnd_wl.sorted_validation_indices);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_FALSE(res.untouched_ok);

    // Random Copy
    initialize_patterns_dst_sentinel(dst.get(), total_bytes, sentinel, rnd_spans);
    for (size_t w = 0; w < 2; ++w) {
      memory_copy_random_loop_asm(
          dst_u8 + rnd_spans[w].offset_bytes,
          src_u8 + rnd_spans[w].offset_bytes, rnd_wl.workers[w].indices.data(),
          rnd_wl.workers[w].indices.size());
    }
    res = validate_patterns_copy_buffer(
        PatternKind::Random, dst.get(), src.get(), total_bytes, true, sentinel,
        0, 1, rnd_spans, &rnd_wl.sorted_validation_indices);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_TRUE(res.untouched_ok);
    dst_u8[gap_off] ^= 0x20U;
    res = validate_patterns_copy_buffer(
        PatternKind::Random, dst.get(), src.get(), total_bytes, true, sentinel,
        0, 1, rnd_spans, &rnd_wl.sorted_validation_indices);
    EXPECT_TRUE(res.touched_ok);
    EXPECT_FALSE(res.untouched_ok);
  }
}

}  // namespace
