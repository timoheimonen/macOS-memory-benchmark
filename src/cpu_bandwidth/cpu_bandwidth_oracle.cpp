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
 * @file cpu_bandwidth_oracle.cpp
 * @brief O(1) GF(2) src pattern, per-pass read checksum oracle, and cold post-validation
 */

#include "cpu_bandwidth/cpu_bandwidth_oracle.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

#include "asm/asm_functions.h"
#include "core/audit/deterministic_rng.h"

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

uint64_t prefix_t(const CpuBandwidthPatternTable& table, uint64_t n) noexcept {
  return (((n >> 8U) & 1ULL) != 0ULL ? table.prefix[256] : 0ULL) ^
         table.prefix[n & 255ULL];
}

void fill_src_range(uint8_t* dst, size_t offset, size_t span,
                    const CpuBandwidthPatternTable& table) noexcept {
  if (span == 0) {
    return;
  }
  size_t pos = offset;
  const size_t end = offset + span;
  while (pos < end && (pos & 7U) != 0U) {
    const uint64_t w = cpu_bandwidth_buffer_word(table, pos >> 3U);
    dst[pos] = static_cast<uint8_t>((w >> ((pos & 7U) * 8U)) & 0xFFULL);
    ++pos;
  }
  while (pos + 8U <= end) {
    const uint64_t w = cpu_bandwidth_buffer_word(table, pos >> 3U);
    std::memcpy(dst + pos, &w, sizeof(w));
    pos += 8U;
  }
  while (pos < end) {
    const uint64_t w = cpu_bandwidth_buffer_word(table, pos >> 3U);
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

template <typename Fn>
void run_parallel_cold_ranges(
    size_t total_bytes,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans,
    Fn fn) {
  if (worker_spans.size() <= 1 || total_bytes < 64ULL * 1024ULL) {
    fn(0, total_bytes, 0);
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

}  // namespace

CpuBandwidthPatternTable build_cpu_bandwidth_pattern_table(
    uint64_t target_src_seed) noexcept {
  CpuBandwidthPatternTable result;
  result.prefix[0] = 0ULL;
  for (size_t k = 0; k < 256; ++k) {
    result.table[k] =
        splitmix64_mix(target_src_seed ^ static_cast<uint64_t>(k));
    result.prefix[k + 1] = result.prefix[k] ^ result.table[k];
  }
  return result;
}

void initialize_cpu_bandwidth_src_buffer(
    void* src, size_t bytes, uint64_t target_src_seed,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans) {
  if (src == nullptr || bytes == 0) {
    return;
  }
  const CpuBandwidthPatternTable table =
      build_cpu_bandwidth_pattern_table(target_src_seed);
  auto* dst = static_cast<uint8_t*>(src);
  run_parallel_cold_ranges(bytes, worker_spans,
                           [dst, &table](size_t offset, size_t span, size_t) {
                             fill_src_range(dst, offset, span, table);
                           });
}

uint64_t compute_cpu_bandwidth_pass_xor(
    const CpuBandwidthPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes) noexcept {
  if (span_bytes == 0) {
    return 0ULL;
  }
  const size_t vec_bytes = span_bytes - (span_bytes % 32U);
  uint64_t vec_xor = 0ULL;
  if (vec_bytes > 0) {
    const uint64_t start_word = static_cast<uint64_t>(offset_bytes >> 3U);
    const uint64_t end_word = start_word + static_cast<uint64_t>(vec_bytes >> 3U);
    const uint64_t range_t = prefix_t(table, end_word) ^ prefix_t(table, start_word);
    const uint64_t x = prefix_xor_0_to_n_minus_1(end_word) ^
                       prefix_xor_0_to_n_minus_1(start_word);
    const uint64_t range_mix =
        x ^ cpu_bandwidth_rotl64(x, 19U) ^ cpu_bandwidth_rotl64(x, 41U);
    vec_xor = range_t ^ range_mix;
  }

  uint64_t tail_xor = 0ULL;
  const size_t tail_start = offset_bytes + vec_bytes;
  const size_t tail_end = offset_bytes + span_bytes;
  for (size_t pos = tail_start; pos < tail_end; ++pos) {
    const uint64_t w_idx = static_cast<uint64_t>(pos >> 3U);
    const uint64_t w_val = cpu_bandwidth_buffer_word(table, w_idx);
    tail_xor ^= (w_val >> (8U * (pos & 7U))) & 0xFFULL;
  }
  return vec_xor ^ tail_xor;
}

void initial_cpu_bandwidth_worker_checksum_state(
    uint64_t op_seed, size_t worker_index,
    uint64_t& out_state_a, uint64_t& out_state_b) noexcept {
  out_state_a = splitmix64_mix(
      op_seed ^ (0xA100000000000000ULL | static_cast<uint64_t>(worker_index)));
  out_state_b = splitmix64_mix(
      op_seed ^ (0xB200000000000000ULL | static_cast<uint64_t>(worker_index)));
}

CpuBandwidthRunChecksum fold_cpu_bandwidth_worker_checksums(
    const std::vector<CpuBandwidthExpectedWorkerChecksum>& workers) noexcept {
  uint64_t a = kCpuBandwidthFoldInitA;
  uint64_t b = kCpuBandwidthFoldInitB;
  for (size_t ordinal = 0; ordinal < workers.size(); ++ordinal) {
    const auto& w = workers[ordinal];
    const uint64_t tag_a =
        w.exact_bytes_read + kCpuBandwidthChecksumA * (ordinal + 1ULL);
    const uint64_t tag_b =
        w.pass_count + kCpuBandwidthChecksumB * (ordinal + 1ULL);
    a = splitmix64_mix(a ^ w.state_a ^ tag_a);
    b = splitmix64_mix(b ^ w.state_b ^ tag_b);
  }
  return {a, b};
}

CpuBandwidthExpectedChecksum compute_cpu_bandwidth_expected_read_checksum(
    uint64_t target_src_seed,
    uint64_t op_seed,
    uint64_t passes,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans) {
  CpuBandwidthExpectedChecksum expected;
  expected.applicable = true;
  expected.available = true;
  expected.reason_code = "valid";
  expected.workers.reserve(worker_spans.size());

  const CpuBandwidthPatternTable table =
      build_cpu_bandwidth_pattern_table(target_src_seed);
  const uint64_t s1 =
      ((passes & 1ULL) == 0ULL)
          ? ((passes >> 1U) * (passes + 1ULL))
          : (passes * ((passes + 1ULL) >> 1U));

  for (size_t w = 0; w < worker_spans.size(); ++w) {
    const auto& span = worker_spans[w];
    const uint64_t px = compute_cpu_bandwidth_pass_xor(
        table, span.offset_bytes, span.span_bytes);
    uint64_t init_a = 0;
    uint64_t init_b = 0;
    initial_cpu_bandwidth_worker_checksum_state(op_seed, w, init_a, init_b);
    const uint64_t state_a =
        init_a + passes * px + kCpuBandwidthChecksumA * s1;
    const uint64_t state_b =
        init_b + (px ^ kCpuBandwidthChecksumB) * s1 +
        kCpuBandwidthChecksumC * passes;

    CpuBandwidthExpectedWorkerChecksum row;
    row.worker_index = w;
    row.state_a = state_a;
    row.state_b = state_b;
    row.exact_bytes_read = passes * static_cast<uint64_t>(span.span_bytes);
    row.pass_count = passes;
    expected.workers.push_back(row);
  }
  expected.run_checksum = fold_cpu_bandwidth_worker_checksums(expected.workers);
  return expected;
}

uint64_t derive_cpu_bandwidth_write_pattern(uint64_t attempt_seed) noexcept {
  return splitmix64_mix(attempt_seed) | 1ULL;
}

uint64_t derive_cpu_bandwidth_write_attempt_seed(
    uint64_t op_seed, uint64_t attempt_ordinal) noexcept {
  return splitmix64_mix(op_seed ^ (0x5752495400000000ULL | attempt_ordinal));
}

uint64_t derive_cpu_bandwidth_pre_copy_sentinel_pattern(
    uint64_t pre_copy_sentinel_seed, uint64_t copy_ordinal) noexcept {
  return splitmix64_mix(pre_copy_sentinel_seed ^ copy_ordinal) | 1ULL;
}

void initialize_cpu_bandwidth_dst_sentinel(
    void* dst, size_t bytes, uint64_t sentinel_pattern,
    bool cache_target,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans) {
  if (dst == nullptr || bytes == 0) {
    return;
  }
  auto* base = static_cast<uint8_t*>(dst);
  run_parallel_cold_ranges(
      bytes, worker_spans,
      [base, sentinel_pattern, cache_target](size_t offset, size_t span,
                                             size_t) {
        if (cache_target) {
          memory_write_cache_pattern_loop_asm(base + offset, span,
                                              sentinel_pattern);
        } else {
          memory_write_pattern_loop_asm(base + offset, span, sentinel_pattern);
        }
      });
}

bool validate_cpu_bandwidth_write_buffer(
    const void* dst, size_t bytes, uint64_t write_pattern,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans) {
  if (dst == nullptr || bytes == 0) {
    return false;
  }
  const auto* base = static_cast<const uint8_t*>(dst);
  std::atomic<bool> all_valid{true};
  run_parallel_cold_ranges(
      bytes, worker_spans,
      [base, write_pattern, &all_valid](size_t offset, size_t span, size_t) {
        if (!check_write_range(base, offset, span, write_pattern)) {
          all_valid.store(false, std::memory_order_relaxed);
        }
      });
  return all_valid.load(std::memory_order_relaxed);
}

bool validate_cpu_bandwidth_copy_buffer(
    const void* dst, const void* src, size_t bytes,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans) {
  if (dst == nullptr || src == nullptr || bytes == 0) {
    return false;
  }
  const auto* dst_bytes = static_cast<const uint8_t*>(dst);
  const auto* src_bytes = static_cast<const uint8_t*>(src);
  std::atomic<bool> all_valid{true};
  run_parallel_cold_ranges(
      bytes, worker_spans,
      [dst_bytes, src_bytes, &all_valid](size_t offset, size_t span, size_t) {
        if (std::memcmp(dst_bytes + offset, src_bytes + offset, span) != 0) {
          all_valid.store(false, std::memory_order_relaxed);
        }
      });
  return all_valid.load(std::memory_order_relaxed);
}
