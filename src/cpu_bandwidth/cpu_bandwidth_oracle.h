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
 * @file cpu_bandwidth_oracle.h
 * @brief O(1) GF(2) src pattern, per-pass read checksum oracle, and cold post-validation
 */

#ifndef CPU_BANDWIDTH_ORACLE_H
#define CPU_BANDWIDTH_ORACLE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"

inline constexpr uint64_t kCpuBandwidthChecksumA = 0x9E3779B97F4A7C15ULL;
inline constexpr uint64_t kCpuBandwidthChecksumB = 0xBF58476D1CE4E5B9ULL;
inline constexpr uint64_t kCpuBandwidthChecksumC = 0x94D049BB133111EBULL;
inline constexpr uint64_t kCpuBandwidthFoldInitA = 0x6A09E667F3BCC909ULL;
inline constexpr uint64_t kCpuBandwidthFoldInitB = 0xBB67AE8584CAA73BULL;

/** Precomputed 256-word GF(2) pattern table and 257-word prefix XOR table. */
struct CpuBandwidthPatternTable {
  std::array<uint64_t, 256> table{};
  std::array<uint64_t, 257> prefix{};
};

/** 64-bit left circular rotation. */
inline uint64_t cpu_bandwidth_rotl64(uint64_t value, unsigned int shift) noexcept {
  shift &= 63U;
  if (shift == 0U) {
    return value;
  }
  return (value << shift) | (value >> (64U - shift));
}

/** Build `T[0..255]` and `prefix[0..256]` for `target_src_seed`. */
CpuBandwidthPatternTable build_cpu_bandwidth_pattern_table(
    uint64_t target_src_seed) noexcept;

/** Compute 64-bit little-endian word `W[word_index]` for `cpu-bw-buffer-pattern-v1`. */
inline uint64_t cpu_bandwidth_buffer_word(
    const CpuBandwidthPatternTable& table, uint64_t word_index) noexcept {
  const uint64_t mix =
      word_index ^ cpu_bandwidth_rotl64(word_index, 19U) ^
      cpu_bandwidth_rotl64(word_index, 41U);
  return table.table[word_index & 255ULL] ^ mix;
}

/** Initialize `src[0..bytes)` deterministically using `cpu-bw-buffer-pattern-v1`. */
void initialize_cpu_bandwidth_src_buffer(
    void* src, size_t bytes, uint64_t target_src_seed,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans);

/** Exact O(1) `pass_xor` matching ARM64 `memory_read*_loop_asm` over `[offset, offset+span)`. */
uint64_t compute_cpu_bandwidth_pass_xor(
    const CpuBandwidthPatternTable& table,
    size_t offset_bytes,
    size_t span_bytes) noexcept;

/** Compute initial `(state_a, state_b)` for one worker in `cpu-bw-read-checksum-v1`. */
void initial_cpu_bandwidth_worker_checksum_state(
    uint64_t op_seed, size_t worker_index,
    uint64_t& out_state_a, uint64_t& out_state_b) noexcept;

/** Fold per-worker checksum tuples deterministically into `CpuBandwidthRunChecksum`. */
CpuBandwidthRunChecksum fold_cpu_bandwidth_worker_checksums(
    const std::vector<CpuBandwidthExpectedWorkerChecksum>& workers) noexcept;

/** Compute closed-form O(1) expected read checksum for all worker spans. */
CpuBandwidthExpectedChecksum compute_cpu_bandwidth_expected_read_checksum(
    uint64_t target_src_seed,
    uint64_t op_seed,
    uint64_t passes,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans);

/** Derive unique non-zero odd `write_pattern = splitmix64_mix(attempt_seed) | 1ULL`. */
uint64_t derive_cpu_bandwidth_write_pattern(uint64_t attempt_seed) noexcept;

/** Derive `attempt_seed` deterministically for a write attempt. */
uint64_t derive_cpu_bandwidth_write_attempt_seed(
    uint64_t op_seed, uint64_t attempt_ordinal) noexcept;

/** Derive odd non-zero `pre_copy_sentinel` pattern before a copy attempt. */
uint64_t derive_cpu_bandwidth_pre_copy_sentinel_pattern(
    uint64_t pre_copy_sentinel_seed, uint64_t copy_ordinal) noexcept;

/** Initialize `dst[0..bytes)` with `sentinel_pattern` across `worker_spans`. */
void initialize_cpu_bandwidth_dst_sentinel(
    void* dst, size_t bytes, uint64_t sentinel_pattern,
    bool cache_target,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans);

/** Cold parallel byte-exact check that `dst[0..bytes)` matches `write_pattern`. */
bool validate_cpu_bandwidth_write_buffer(
    const void* dst, size_t bytes, uint64_t write_pattern,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans);

/** Cold parallel byte-exact check that `dst[0..bytes) == src[0..bytes)`. */
bool validate_cpu_bandwidth_copy_buffer(
    const void* dst, const void* src, size_t bytes,
    const std::vector<CpuBandwidthWorkerSpan>& worker_spans);

#endif  // CPU_BANDWIDTH_ORACLE_H
