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

/**
 * @file patterns_oracle.cpp
 * @brief Slot-tagged GF(2) src pattern, O(1) read checksum oracles, random generator, and cold validation
 */

#include "pattern_benchmark/patterns_oracle.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>

#include "asm/asm_functions.h"
#include "core/audit/deterministic_rng.h"
#include "utils/hash_utils.h"

namespace {

uint64_t prefix_xor_0_to_n_minus_1(uint64_t n) noexcept {
  if (n == 0ULL) {
    return 0ULL;
  }
  const uint64_t m = n - 1ULL;
  switch (m & 3ULL) {
    case 0ULL:
      return m;
    case 1ULL:
      return 1ULL;
    case 2ULL:
      return m + 1ULL;
    default:
      return 0ULL;
  }
}

uint64_t even_half_s1(uint64_t passes) noexcept {
  return ((passes & 1ULL) == 0ULL)
             ? ((passes >> 1U) * (passes + 1ULL))
             : (passes * ((passes + 1ULL) >> 1U));
}

uint64_t prefix_t_mod256(const PatternsPatternTable& table, uint64_t n) noexcept {
  return (((n >> 8U) & 1ULL) != 0ULL ? table.prefix_t[256] : 0ULL) ^
         table.prefix_t[n & 255ULL];
}

uint64_t prefix_st_mod64(const PatternsPatternTable& table, uint64_t n) noexcept {
  return (((n >> 6U) & 1ULL) != 0ULL ? table.prefix_st[64] : 0ULL) ^
         table.prefix_st[n & 63ULL];
}

uint64_t prefix_st_even_mod32(const PatternsPatternTable& table, uint64_t n) noexcept {
  return (((n >> 5U) & 1ULL) != 0ULL ? table.prefix_st_even[32] : 0ULL) ^
         table.prefix_st_even[n & 31ULL];
}

uint64_t prefix_st_odd_mod32(const PatternsPatternTable& table, uint64_t n) noexcept {
  return (((n >> 5U) & 1ULL) != 0ULL ? table.prefix_st_odd[32] : 0ULL) ^
         table.prefix_st_odd[n & 31ULL];
}

uint64_t xor_words_range(const PatternsPatternTable& table,
                         uint64_t w_start,
                         uint64_t w_end) noexcept {
  if (w_end <= w_start) {
    return 0ULL;
  }
  const uint64_t range_t =
      prefix_t_mod256(table, w_end) ^ prefix_t_mod256(table, w_start);
  const uint64_t lx =
      prefix_xor_0_to_n_minus_1(w_end) ^ prefix_xor_0_to_n_minus_1(w_start);
  const uint64_t range_lane =
      lx ^ patterns_rotl64(lx, 7U) ^ patterns_rotl64(lx, 33U);
  const uint64_t s_start = (w_start + 3ULL) >> 2U;
  const uint64_t s_end = (w_end + 3ULL) >> 2U;
  const uint64_t sx =
      prefix_xor_0_to_n_minus_1(s_end) ^ prefix_xor_0_to_n_minus_1(s_start);
  const uint64_t range_slot = patterns_slot_mix(sx);
  return range_t ^ range_lane ^ range_slot;
}

uint64_t xor_slots_range(const PatternsPatternTable& table,
                         uint64_t s_start,
                         uint64_t s_end) noexcept {
  if (s_end <= s_start) {
    return 0ULL;
  }
  const uint64_t range_st =
      prefix_st_mod64(table, s_end) ^ prefix_st_mod64(table, s_start);
  const uint64_t sx =
      prefix_xor_0_to_n_minus_1(s_end) ^ prefix_xor_0_to_n_minus_1(s_start);
  return range_st ^ patterns_slot_mix(sx);
}

uint64_t xor_vec_bytes_32_aligned(const PatternsPatternTable& table,
                                  size_t byte_start,
                                  size_t vec_bytes) noexcept {
  if (vec_bytes == 0) {
    return 0ULL;
  }
  const uint64_t k = static_cast<uint64_t>(vec_bytes >> 3U);
  const uint64_t w0 = static_cast<uint64_t>(byte_start >> 3U);
  const unsigned int r = static_cast<unsigned int>(byte_start & 7U);
  const uint64_t x0 = xor_words_range(table, w0, w0 + k);
  if (r == 0U) {
    return x0;
  }
  const uint64_t x1 =
      x0 ^ patterns_buffer_word(table, w0) ^ patterns_buffer_word(table, w0 + k);
  return (x0 >> (8U * r)) | (x1 << (64U - 8U * r));
}

void fill_src_range(uint8_t* dst, size_t offset, size_t span,
                    const PatternsPatternTable& table) noexcept {
  if (span == 0) {
    return;
  }
  size_t pos = offset;
  const size_t end = offset + span;
  while (pos < end && (pos & 7U) != 0U) {
    const uint64_t w = patterns_buffer_word(table, pos >> 3U);
    dst[pos] = static_cast<uint8_t>((w >> ((pos & 7U) * 8U)) & 0xFFULL);
    ++pos;
  }
  while (pos + 8U <= end) {
    const uint64_t w = patterns_buffer_word(table, pos >> 3U);
    std::memcpy(dst + pos, &w, sizeof(w));
    pos += 8U;
  }
  while (pos < end) {
    const uint64_t w = patterns_buffer_word(table, pos >> 3U);
    dst[pos] = static_cast<uint8_t>((w >> ((pos & 7U) * 8U)) & 0xFFULL);
    ++pos;
  }
}

bool check_write_range(const uint8_t* dst, size_t offset, size_t span,
                       uint64_t pattern) noexcept {
  size_t pos = offset;
  const size_t end = offset + span;
  while (pos < end && (pos & 7U) != 0U) {
    const uint8_t expected =
        static_cast<uint8_t>((pattern >> ((pos & 7U) * 8U)) & 0xFFULL);
    if (dst[pos] != expected) {
      return false;
    }
    ++pos;
  }
  while (pos + 8U <= end) {
    uint64_t word = 0;
    std::memcpy(&word, dst + pos, sizeof(word));
    if (word != pattern) {
      return false;
    }
    pos += 8U;
  }
  while (pos < end) {
    const uint8_t expected =
        static_cast<uint8_t>((pattern >> ((pos & 7U) * 8U)) & 0xFFULL);
    if (dst[pos] != expected) {
      return false;
    }
    ++pos;
  }
  return true;
}

bool spans_cover_buffer_contiguously(
    size_t total_bytes,
    const std::vector<PatternsWorkerSpan>& worker_spans) noexcept {
  if (worker_spans.empty()) {
    return false;
  }
  size_t cursor = 0;
  for (const auto& sp : worker_spans) {
    if (sp.offset_bytes != cursor) {
      return false;
    }
    if (sp.span_bytes > total_bytes - cursor) {
      return false;
    }
    cursor += sp.span_bytes;
  }
  return cursor == total_bytes;
}

template <typename Fn>
void run_parallel_cold_worker_spans(
    size_t total_bytes,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    Fn fn) {
  if (worker_spans.size() <= 1 || total_bytes < 64ULL * 1024ULL) {
    for (size_t idx = 0; idx < worker_spans.size(); ++idx) {
      const auto& sp = worker_spans[idx];
      if (sp.span_bytes == 0) {
        continue;
      }
      fn(sp.offset_bytes, sp.span_bytes, idx);
    }
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(worker_spans.size());
  for (size_t idx = 0; idx < worker_spans.size(); ++idx) {
    const auto span = worker_spans[idx];
    if (span.span_bytes == 0) {
      continue;
    }
    threads.emplace_back([&, span, idx]() {
      fn(span.offset_bytes, span.span_bytes, idx);
    });
  }
  for (std::thread& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void append_uint64_le(HashUtils::Sha256Hasher& hasher, uint64_t value) {
  uint8_t bytes[8]{};
  for (size_t b = 0; b < 8; ++b) {
    bytes[b] = static_cast<uint8_t>((value >> (b * 8U)) & 0xFFULL);
  }
  hasher.update(bytes, sizeof(bytes));
}

template <typename TouchedChecker>
PatternsPostValidationResult validate_patterns_buffer_impl(
    PatternKind kind,
    const uint8_t* dst_bytes,
    size_t bytes,
    bool check_untouched,
    uint64_t pre_op_sentinel,
    size_t stride_bytes,
    size_t passes,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    const std::vector<std::vector<size_t>>* sorted_random_indices,
    TouchedChecker check_touched_range) {
  if (dst_bytes == nullptr || bytes == 0 ||
      !spans_cover_buffer_contiguously(bytes, worker_spans)) {
    return {false, false};
  }
  if (patterns_kind_is_strided(kind)) {
    if (passes == 0 || stride_bytes < 64 ||
        (stride_bytes % kPatternsAccessSizeBytes) != 0) {
      return {false, false};
    }
    for (const auto& sp : worker_spans) {
      if ((sp.offset_bytes % kPatternsAccessSizeBytes) != 0 ||
          sp.span_bytes < stride_bytes + kPatternsAccessSizeBytes) {
        return {false, false};
      }
    }
  } else if (kind == PatternKind::Random) {
    if (sorted_random_indices == nullptr ||
        sorted_random_indices->size() != worker_spans.size()) {
      return {false, false};
    }
    for (size_t w = 0; w < worker_spans.size(); ++w) {
      if ((worker_spans[w].offset_bytes % kPatternsAccessSizeBytes) != 0) {
        return {false, false};
      }
      if (worker_spans[w].random_access_count != 0 &&
          (*sorted_random_indices)[w].size() !=
              worker_spans[w].random_access_count) {
        return {false, false};
      }
    }
  }

  std::atomic<bool> touched_ok{true};
  std::atomic<bool> untouched_ok{true};

  run_parallel_cold_worker_spans(
      bytes, worker_spans, [&](size_t offset, size_t span, size_t w_idx) {
        bool local_touched_ok = true;
        bool local_untouched_ok = true;

        if (kind == PatternKind::SequentialForward ||
            kind == PatternKind::SequentialReverse) {
          if (!check_touched_range(offset, span)) {
            local_touched_ok = false;
          }
        } else if (patterns_kind_is_strided(kind)) {
          const size_t q_slots = stride_bytes / kPatternsAccessSizeBytes;
          const size_t rem = std::min(passes, q_slots);
          const size_t num_slots = span / kPatternsAccessSizeBytes;
          const size_t full_blocks = num_slots / q_slots;
          const size_t tail_slots = num_slots % q_slots;

          if (rem == q_slots) {
            const size_t aligned_bytes = num_slots * kPatternsAccessSizeBytes;
            if (aligned_bytes > 0 &&
                !check_touched_range(offset, aligned_bytes)) {
              local_touched_ok = false;
            }
            if (check_untouched && aligned_bytes < span &&
                !check_write_range(dst_bytes, offset + aligned_bytes,
                                   span - aligned_bytes, pre_op_sentinel)) {
              local_untouched_ok = false;
            }
          } else {
            const size_t touched_block_bytes = rem * kPatternsAccessSizeBytes;
            const size_t untouched_block_bytes =
                (q_slots - rem) * kPatternsAccessSizeBytes;
            for (size_t b = 0; b < full_blocks; ++b) {
              const size_t block_off = offset + b * stride_bytes;
              if (local_touched_ok &&
                  !check_touched_range(block_off, touched_block_bytes)) {
                local_touched_ok = false;
              }
              if (check_untouched && local_untouched_ok &&
                  !check_write_range(dst_bytes, block_off + touched_block_bytes,
                                     untouched_block_bytes, pre_op_sentinel)) {
                local_untouched_ok = false;
              }
              if (!local_touched_ok && (!check_untouched || !local_untouched_ok)) {
                break;
              }
            }

            const size_t tail_base = offset + full_blocks * stride_bytes;
            const size_t touched_tail_slots = std::min(tail_slots, rem);
            const size_t touched_tail_bytes =
                touched_tail_slots * kPatternsAccessSizeBytes;
            if (touched_tail_bytes > 0 && local_touched_ok &&
                !check_touched_range(tail_base, touched_tail_bytes)) {
              local_touched_ok = false;
            }
            const size_t untouched_tail_off = tail_base + touched_tail_bytes;
            const size_t span_end = offset + span;
            if (check_untouched && untouched_tail_off < span_end &&
                local_untouched_ok &&
                !check_write_range(dst_bytes, untouched_tail_off,
                                   span_end - untouched_tail_off,
                                   pre_op_sentinel)) {
              local_untouched_ok = false;
            }
          }
        } else {
          // PatternKind::Random
          const std::vector<size_t>& sorted_idx =
              (*sorted_random_indices)[w_idx];
          size_t pos = 0;
          for (size_t idx : sorted_idx) {
            if (idx < pos || idx > span ||
                kPatternsAccessSizeBytes > span - idx ||
                (idx % kPatternsAccessSizeBytes) != 0) {
              local_touched_ok = false;
              break;
            }
            if (check_untouched && idx > pos && local_untouched_ok &&
                !check_write_range(dst_bytes, offset + pos, idx - pos,
                                   pre_op_sentinel)) {
              local_untouched_ok = false;
            }
            if (local_touched_ok &&
                !check_touched_range(offset + idx, kPatternsAccessSizeBytes)) {
              local_touched_ok = false;
            }
            pos = idx + kPatternsAccessSizeBytes;
          }
          if (check_untouched && pos < span && local_untouched_ok &&
              !check_write_range(dst_bytes, offset + pos, span - pos,
                                 pre_op_sentinel)) {
            local_untouched_ok = false;
          }
        }

        if (!local_touched_ok) {
          touched_ok.store(false, std::memory_order_relaxed);
        }
        if (!local_untouched_ok) {
          untouched_ok.store(false, std::memory_order_relaxed);
        }
      });

  return {touched_ok.load(std::memory_order_relaxed),
          untouched_ok.load(std::memory_order_relaxed)};
}

}  // namespace

PatternsPatternTable build_patterns_pattern_table(uint64_t src_seed) noexcept {
  PatternsPatternTable result;
  result.prefix_t[0] = 0ULL;
  for (size_t k = 0; k < 256; ++k) {
    result.table[k] = splitmix64_mix(src_seed ^ static_cast<uint64_t>(k));
    result.prefix_t[k + 1] = result.prefix_t[k] ^ result.table[k];
  }
  result.prefix_st[0] = 0ULL;
  for (size_t u = 0; u < 64; ++u) {
    result.s_t[u] = result.table[4 * u] ^ result.table[4 * u + 1] ^
                    result.table[4 * u + 2] ^ result.table[4 * u + 3];
    result.prefix_st[u + 1] = result.prefix_st[u] ^ result.s_t[u];
  }
  result.prefix_st_even[0] = 0ULL;
  result.prefix_st_odd[0] = 0ULL;
  for (size_t k = 0; k < 32; ++k) {
    result.prefix_st_even[k + 1] =
        result.prefix_st_even[k] ^ result.s_t[2 * k];
    result.prefix_st_odd[k + 1] =
        result.prefix_st_odd[k] ^ result.s_t[2 * k + 1];
  }
  return result;
}

void initialize_patterns_src_buffer(
    void* src, size_t bytes, uint64_t src_seed,
    const std::vector<PatternsWorkerSpan>& worker_spans) {
  if (src == nullptr || bytes == 0) {
    return;
  }
  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  auto* dst = static_cast<uint8_t*>(src);
  if (spans_cover_buffer_contiguously(bytes, worker_spans)) {
    run_parallel_cold_worker_spans(
        bytes, worker_spans, [dst, &table](size_t offset, size_t span, size_t) {
          fill_src_range(dst, offset, span, table);
        });
  } else {
    fill_src_range(dst, 0, bytes, table);
  }
}

uint64_t compute_patterns_pass_xor_sequential_forward(
    const PatternsPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes) noexcept {
  if (span_bytes == 0) {
    return 0ULL;
  }
  const size_t vec_bytes = span_bytes - (span_bytes % 32U);
  const uint64_t vec_xor =
      xor_vec_bytes_32_aligned(table, offset_bytes, vec_bytes);

  uint64_t tail_xor = 0ULL;
  const size_t tail_start = offset_bytes + vec_bytes;
  const size_t tail_end = offset_bytes + span_bytes;
  for (size_t pos = tail_start; pos < tail_end; ++pos) {
    const uint64_t w_idx = static_cast<uint64_t>(pos >> 3U);
    const uint64_t w_val = patterns_buffer_word(table, w_idx);
    tail_xor ^= (w_val >> (8U * (pos & 7U))) & 0xFFULL;
  }
  return vec_xor ^ tail_xor;
}

uint64_t compute_patterns_pass_xor_sequential_reverse(
    const PatternsPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes) noexcept {
  if (span_bytes == 0) {
    return 0ULL;
  }
  const size_t tail_bytes = span_bytes % 32U;
  const size_t vec_bytes = span_bytes - tail_bytes;

  uint64_t tail_xor = 0ULL;
  const size_t tail_end = offset_bytes + tail_bytes;
  for (size_t pos = offset_bytes; pos < tail_end; ++pos) {
    const uint64_t w_idx = static_cast<uint64_t>(pos >> 3U);
    const uint64_t w_val = patterns_buffer_word(table, w_idx);
    tail_xor ^= (w_val >> (8U * (pos & 7U))) & 0xFFULL;
  }

  const uint64_t vec_xor =
      xor_vec_bytes_32_aligned(table, tail_end, vec_bytes);
  return vec_xor ^ tail_xor;
}

uint64_t compute_patterns_strided_full_cycle_xor(
    const PatternsPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes) noexcept {
  if (span_bytes < kPatternsAccessSizeBytes ||
      (offset_bytes % kPatternsAccessSizeBytes) != 0) {
    return 0ULL;
  }
  const uint64_t base_slot = static_cast<uint64_t>(offset_bytes >> 5U);
  const uint64_t num_slots = static_cast<uint64_t>(span_bytes >> 5U);
  return xor_slots_range(table, base_slot, base_slot + num_slots);
}

uint64_t compute_patterns_strided_partial_cycle_xor(
    const PatternsPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes,
    size_t stride_bytes,
    size_t rem_phases) noexcept {
  if (span_bytes < kPatternsAccessSizeBytes ||
      (offset_bytes % kPatternsAccessSizeBytes) != 0 ||
      stride_bytes < 64 || (stride_bytes % kPatternsAccessSizeBytes) != 0 ||
      rem_phases == 0) {
    return 0ULL;
  }
  const uint64_t Q = static_cast<uint64_t>(stride_bytes >> 5U);
  if (rem_phases >= Q) {
    return compute_patterns_strided_full_cycle_xor(table, offset_bytes,
                                                   span_bytes);
  }
  if ((Q & (Q - 1ULL)) != 0ULL) {
    return 0ULL;
  }
  const unsigned int q = static_cast<unsigned int>(__builtin_ctzll(Q));
  const uint64_t base_slot = static_cast<uint64_t>(offset_bytes >> 5U);
  const uint64_t num_slots = static_cast<uint64_t>(span_bytes >> 5U);
  const uint64_t full_blocks = num_slots >> q;
  const uint64_t tail_slots = num_slots & (Q - 1ULL);
  const uint64_t rem = static_cast<uint64_t>(rem_phases);

  const uint64_t high_0 = base_slot >> q;
  const uint64_t low_0 = base_slot & (Q - 1ULL);
  const uint64_t rem_1 = std::min(rem, Q - low_0);
  const uint64_t rem_2 = rem - rem_1;

  const auto xor_rect = [&](uint64_t h_start, uint64_t h_end,
                            uint64_t l_start, uint64_t l_end) -> uint64_t {
    if (h_end <= h_start || l_end <= l_start) {
      return 0ULL;
    }
    const uint64_t n_h = h_end - h_start;
    const uint64_t n_l = l_end - l_start;
    uint64_t out = 0ULL;

    if ((n_l & 1ULL) != 0ULL) {
      const uint64_t hx =
          prefix_xor_0_to_n_minus_1(h_end) ^ prefix_xor_0_to_n_minus_1(h_start);
      out ^= patterns_slot_mix(hx << q);
    }
    if ((n_h & 1ULL) != 0ULL) {
      const uint64_t lx =
          prefix_xor_0_to_n_minus_1(l_end) ^ prefix_xor_0_to_n_minus_1(l_start);
      out ^= patterns_slot_mix(lx);
    }

    if (q >= 6U) {
      if ((n_h & 1ULL) != 0ULL) {
        out ^= prefix_st_mod64(table, l_end) ^ prefix_st_mod64(table, l_start);
      }
    } else if (q == 1U) {
      // Q == 2 and 1 <= rem < 2 forces n_l == 1 and l_start in {0, 1}.
      if (l_start == 0ULL) {
        out ^= prefix_st_even_mod32(table, h_end) ^
               prefix_st_even_mod32(table, h_start);
      } else {
        out ^= prefix_st_odd_mod32(table, h_end) ^
               prefix_st_odd_mod32(table, h_start);
      }
    } else {
      // Non-canonical small power-of-2 stride (4, 8, 16, 32 slots).
      for (uint64_t h = h_start; h < h_end; ++h) {
        for (uint64_t l = l_start; l < l_end; ++l) {
          out ^= table.s_t[((h << q) | l) & 63ULL];
        }
      }
    }
    return out;
  };

  uint64_t acc = 0ULL;
  if (full_blocks > 0ULL) {
    acc ^= xor_rect(high_0, high_0 + full_blocks, low_0, low_0 + rem_1);
    if (rem_2 > 0ULL) {
      acc ^= xor_rect(high_0 + 1ULL, high_0 + 1ULL + full_blocks, 0ULL, rem_2);
    }
  }
  const uint64_t tail_count = std::min(tail_slots, rem);
  if (tail_count > 0ULL) {
    const uint64_t tail_base = base_slot + (full_blocks << q);
    acc ^= xor_slots_range(table, tail_base, tail_base + tail_count);
  }
  return acc;
}

uint64_t compute_patterns_random_pass_xor(
    const PatternsPatternTable& table,
    size_t offset_bytes,
    const std::vector<size_t>& relative_byte_indices) noexcept {
  const uint64_t base_slot = static_cast<uint64_t>(offset_bytes >> 5U);
  uint64_t acc = 0ULL;
  for (size_t rel_off : relative_byte_indices) {
    acc ^= patterns_slot_xor(
        table, base_slot + static_cast<uint64_t>(rel_off >> 5U));
  }
  return acc;
}

std::string compute_patterns_random_indices_sha256(
    const std::vector<PatternsRandomWorkerIndices>& workers) {
  HashUtils::Sha256Hasher hasher;
  for (size_t w = 0; w < workers.size(); ++w) {
    append_uint64_le(hasher, static_cast<uint64_t>(w));
    append_uint64_le(hasher, static_cast<uint64_t>(workers[w].indices.size()));
    for (size_t off : workers[w].indices) {
      append_uint64_le(hasher, static_cast<uint64_t>(off));
    }
  }
  return hasher.finalize_hex();
}

PatternsRandomWorkload generate_patterns_random_indices(
    uint64_t random_domain_seed,
    uint64_t src_seed,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    const std::vector<size_t>& worker_access_counts) {
  PatternsRandomWorkload workload;
  workload.random_seed = random_domain_seed;
  if (worker_spans.empty() ||
      worker_spans.size() != worker_access_counts.size()) {
    return workload;
  }

  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  const size_t num_workers = worker_spans.size();
  workload.workers.resize(num_workers);
  workload.sorted_validation_indices.resize(num_workers);
  workload.worker_pass_xors.resize(num_workers, 0ULL);

  bool has_any_access = false;
  size_t min_global_offset = std::numeric_limits<size_t>::max();
  size_t max_global_offset = 0;

  for (size_t w = 0; w < num_workers; ++w) {
    const auto& sp = worker_spans[w];
    const size_t count_w = worker_access_counts[w];
    const uint64_t span_slots_w =
        static_cast<uint64_t>(sp.span_bytes / kPatternsAccessSizeBytes);
    if ((sp.offset_bytes % kPatternsAccessSizeBytes) != 0 ||
        count_w > span_slots_w) {
      return PatternsRandomWorkload{};
    }

    PatternsRandomWorkerIndices& w_out = workload.workers[w];
    w_out.offset_bytes = sp.offset_bytes;
    w_out.span_bytes = sp.span_bytes;

    uint64_t state =
        splitmix64_mix(random_domain_seed ^ static_cast<uint64_t>(w));

    if (count_w > 0) {
      std::vector<uint64_t> local_slots(count_w);
      if (count_w == span_slots_w) {
        for (size_t k = 0; k < count_w; ++k) {
          local_slots[k] = static_cast<uint64_t>(k);
        }
      } else {
        const uint64_t start = splitmix64_bounded_draw(state, span_slots_w);
        uint64_t step =
            splitmix64_bounded_draw(state, span_slots_w - 1ULL) + 1ULL;
        while (std::gcd(step, span_slots_w) != 1ULL) {
          ++step;
          if (step >= span_slots_w) {
            step = 1ULL;
          }
        }
        for (size_t k = 0; k < count_w; ++k) {
          local_slots[k] = static_cast<uint64_t>(
              (static_cast<unsigned __int128>(k) * step + start) %
              span_slots_w);
        }
      }
      fisher_yates_rejection_shuffle_with_state(local_slots, state);

      w_out.indices.resize(count_w);
      for (size_t k = 0; k < count_w; ++k) {
        const size_t rel_off =
            static_cast<size_t>(local_slots[k]) * kPatternsAccessSizeBytes;
        w_out.indices[k] = rel_off;
        const size_t global_off = sp.offset_bytes + rel_off;
        min_global_offset = std::min(min_global_offset, global_off);
        max_global_offset = std::max(max_global_offset, global_off);
        has_any_access = true;
      }
    }

    workload.sorted_validation_indices[w] = w_out.indices;
    std::sort(workload.sorted_validation_indices[w].begin(),
              workload.sorted_validation_indices[w].end());
    workload.worker_pass_xors[w] = compute_patterns_random_pass_xor(
        table, sp.offset_bytes, w_out.indices);
    workload.selected_accesses_per_pass += count_w;
  }

  workload.distinct_address_count = workload.selected_accesses_per_pass;
  workload.logical_working_set_bytes =
      has_any_access
          ? (max_global_offset - min_global_offset + kPatternsAccessSizeBytes)
          : 0;
  workload.random_indices_sha256 =
      compute_patterns_random_indices_sha256(workload.workers);
  return workload;
}

void initial_patterns_worker_checksum_state(
    uint64_t op_seed, size_t worker_index,
    uint64_t& out_state_a, uint64_t& out_state_b) noexcept {
  out_state_a = splitmix64_mix(
      op_seed ^ (0xA100000000000000ULL | static_cast<uint64_t>(worker_index)));
  out_state_b = splitmix64_mix(
      op_seed ^ (0xB200000000000000ULL | static_cast<uint64_t>(worker_index)));
}

PatternsRunChecksum fold_patterns_worker_checksums(
    const std::vector<PatternsExpectedWorkerChecksum>& workers) noexcept {
  uint64_t a = kPatternsFoldInitA;
  uint64_t b = kPatternsFoldInitB;
  for (size_t ordinal = 0; ordinal < workers.size(); ++ordinal) {
    const auto& w = workers[ordinal];
    const uint64_t tag_a =
        w.exact_bytes_read + kPatternsChecksumA * (ordinal + 1ULL);
    const uint64_t tag_b =
        w.pass_count + kPatternsChecksumB * (ordinal + 1ULL);
    a = splitmix64_mix(a ^ w.state_a ^ tag_a);
    b = splitmix64_mix(b ^ w.state_b ^ tag_b);
  }
  return {a, b};
}

PatternsExpectedChecksum compute_patterns_expected_read_checksum(
    uint64_t src_seed,
    uint64_t op_seed,
    PatternKind kind,
    size_t stride_bytes,
    uint64_t passes,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    const PatternsRandomWorkload* random_workload) {
  PatternsExpectedChecksum expected;
  expected.applicable = true;

  if (passes == 0 || worker_spans.empty()) {
    expected.available = false;
    expected.reason_code = "invalid-parameters";
    return expected;
  }

  if (kind == PatternKind::Random) {
    if (random_workload == nullptr ||
        random_workload->workers.size() != worker_spans.size() ||
        random_workload->worker_pass_xors.size() != worker_spans.size()) {
      expected.available = false;
      expected.reason_code = "missing-random-workload";
      return expected;
    }
  }

  uint64_t Q = 0;
  if (patterns_kind_is_strided(kind)) {
    if (stride_bytes < 64 || (stride_bytes % kPatternsAccessSizeBytes) != 0) {
      expected.available = false;
      expected.reason_code = "invalid-strided-geometry";
      return expected;
    }
    Q = static_cast<uint64_t>(stride_bytes / kPatternsAccessSizeBytes);
    if ((Q & (Q - 1ULL)) != 0ULL) {
      expected.available = false;
      expected.reason_code = "invalid-strided-geometry";
      return expected;
    }
  }

  const PatternsPatternTable table = build_patterns_pattern_table(src_seed);
  expected.workers.reserve(worker_spans.size());

  const uint64_t s1_passes = even_half_s1(passes);

  for (size_t w = 0; w < worker_spans.size(); ++w) {
    const auto& span = worker_spans[w];
    uint64_t init_a = 0;
    uint64_t init_b = 0;
    initial_patterns_worker_checksum_state(op_seed, w, init_a, init_b);

    PatternsExpectedWorkerChecksum row;
    row.worker_index = w;
    row.pass_count = passes;

    if (kind == PatternKind::SequentialForward ||
        kind == PatternKind::SequentialReverse) {
      const uint64_t px =
          (kind == PatternKind::SequentialForward)
              ? compute_patterns_pass_xor_sequential_forward(
                    table, span.offset_bytes, span.span_bytes)
              : compute_patterns_pass_xor_sequential_reverse(
                    table, span.offset_bytes, span.span_bytes);
      row.state_a = init_a + passes * px + kPatternsChecksumA * s1_passes;
      row.state_b = init_b + (px ^ kPatternsChecksumB) * s1_passes +
                    kPatternsChecksumC * passes;
      row.exact_bytes_read = passes * static_cast<uint64_t>(span.span_bytes);
    } else if (patterns_kind_is_strided(kind)) {
      if ((span.offset_bytes % kPatternsAccessSizeBytes) != 0 ||
          span.span_bytes < stride_bytes + kPatternsAccessSizeBytes) {
        expected.available = false;
        expected.reason_code = "invalid-strided-geometry";
        expected.workers.clear();
        return expected;
      }
      const uint64_t full_cycles = passes / Q;
      const uint64_t rem_phases = passes % Q;
      const uint64_t fc_xor = compute_patterns_strided_full_cycle_xor(
          table, span.offset_bytes, span.span_bytes);
      const uint64_t s1_fc = even_half_s1(full_cycles);

      uint64_t state_a =
          init_a + full_cycles * fc_xor + kPatternsChecksumA * s1_fc;
      uint64_t state_b = init_b + (fc_xor ^ kPatternsChecksumB) * s1_fc +
                         kPatternsChecksumC * full_cycles;

      if (rem_phases > 0ULL) {
        const uint64_t pc_xor = compute_patterns_strided_partial_cycle_xor(
            table, span.offset_bytes, span.span_bytes, stride_bytes,
            static_cast<size_t>(rem_phases));
        const uint64_t step = full_cycles + 1ULL;
        state_a += pc_xor + kPatternsChecksumA * step;
        state_b += (pc_xor ^ kPatternsChecksumB) * step + kPatternsChecksumC;
      }

      const uint64_t num_slots =
          static_cast<uint64_t>(span.span_bytes / kPatternsAccessSizeBytes);
      const uint64_t full_blocks = num_slots / Q;
      const uint64_t tail_slots = num_slots % Q;
      const uint64_t total_slots_read =
          full_cycles * num_slots + full_blocks * rem_phases +
          std::min(tail_slots, rem_phases);

      row.state_a = state_a;
      row.state_b = state_b;
      row.exact_bytes_read = total_slots_read * kPatternsAccessSizeBytes;
    } else {
      // PatternKind::Random
      const uint64_t px = random_workload->worker_pass_xors[w];
      const uint64_t count_w =
          static_cast<uint64_t>(random_workload->workers[w].indices.size());
      row.state_a = init_a + passes * px + kPatternsChecksumA * s1_passes;
      row.state_b = init_b + (px ^ kPatternsChecksumB) * s1_passes +
                    kPatternsChecksumC * passes;
      row.exact_bytes_read = passes * count_w * kPatternsAccessSizeBytes;
    }

    expected.workers.push_back(row);
  }

  expected.run_checksum = fold_patterns_worker_checksums(expected.workers);
  expected.available = true;
  expected.reason_code = PatternsWorkPlanReason::VALID;
  return expected;
}

uint64_t derive_patterns_write_attempt_seed(
    uint64_t op_seed, uint64_t write_ordinal) noexcept {
  return splitmix64_mix(op_seed ^ (0x5752495400000000ULL | write_ordinal));
}

uint64_t derive_patterns_write_pattern(uint64_t write_attempt_seed) noexcept {
  return splitmix64_mix(write_attempt_seed) | 1ULL;
}

uint64_t derive_patterns_pre_op_sentinel(
    uint64_t pre_op_sentinel_seed, uint64_t sentinel_ordinal) noexcept {
  return splitmix64_mix(pre_op_sentinel_seed ^ sentinel_ordinal) | 1ULL;
}

void initialize_patterns_dst_sentinel(
    void* dst, size_t bytes, uint64_t sentinel_pattern,
    const std::vector<PatternsWorkerSpan>& worker_spans) {
  if (dst == nullptr || bytes == 0) {
    return;
  }
  auto* base = static_cast<uint8_t*>(dst);
  if (spans_cover_buffer_contiguously(bytes, worker_spans)) {
    run_parallel_cold_worker_spans(
        bytes, worker_spans,
        [base, sentinel_pattern](size_t offset, size_t span, size_t) {
          memory_write_pattern_loop_asm(base + offset, span, sentinel_pattern);
        });
  } else {
    memory_write_pattern_loop_asm(base, bytes, sentinel_pattern);
  }
}

PatternsPostValidationResult validate_patterns_write_buffer(
    PatternKind kind,
    const void* dst,
    size_t bytes,
    uint64_t write_pattern,
    bool check_untouched,
    uint64_t pre_op_sentinel,
    size_t stride_bytes,
    size_t passes,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    const std::vector<std::vector<size_t>>* sorted_random_indices) {
  const auto* dst_bytes = static_cast<const uint8_t*>(dst);
  return validate_patterns_buffer_impl(
      kind, dst_bytes, bytes, check_untouched, pre_op_sentinel, stride_bytes,
      passes, worker_spans, sorted_random_indices,
      [dst_bytes, write_pattern](size_t offset, size_t span) noexcept {
        return check_write_range(dst_bytes, offset, span, write_pattern);
      });
}

PatternsPostValidationResult validate_patterns_copy_buffer(
    PatternKind kind,
    const void* dst,
    const void* src,
    size_t bytes,
    bool check_untouched,
    uint64_t pre_op_sentinel,
    size_t stride_bytes,
    size_t passes,
    const std::vector<PatternsWorkerSpan>& worker_spans,
    const std::vector<std::vector<size_t>>* sorted_random_indices) {
  if (src == nullptr) {
    return {false, false};
  }
  const auto* dst_bytes = static_cast<const uint8_t*>(dst);
  const auto* src_bytes = static_cast<const uint8_t*>(src);
  return validate_patterns_buffer_impl(
      kind, dst_bytes, bytes, check_untouched, pre_op_sentinel, stride_bytes,
      passes, worker_spans, sorted_random_indices,
      [dst_bytes, src_bytes](size_t offset, size_t span) noexcept {
        return std::memcmp(dst_bytes + offset, src_bytes + offset, span) == 0;
      });
}
