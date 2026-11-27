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
 * @file cpu_latency_chain.h
 * @brief Deterministic SplitMix64 pointer-chasing chain generator and validator
 */

#ifndef CPU_LATENCY_CHAIN_H
#define CPU_LATENCY_CHAIN_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/memory/memory_utils.h"

inline constexpr const char* kCpuLatencyChainGeneratorVersion =
    "splitmix64-fisher-yates-rejection-v1";
inline constexpr const char* kCpuLatencyChainVersion = "cpu-latency-chain-v1";

/** Canonicalized deterministic pointer-chain permutation and SHA-256 identity. */
struct CpuLatencyChainPermutation {
  bool valid = false;
  std::string reason_code = "chain-invalid";
  uint64_t chain_seed = 0;
  size_t node_count = 0;
  size_t stride_bytes = 0;
  LatencyChainMode effective_mode = LatencyChainMode::GlobalRandom;
  size_t tlb_locality_kb = 0;
  size_t locality_nodes = 0;
  std::vector<uint64_t> indices;  ///< Cyclically rotated so `indices[0] == 0`.
  std::string permutation_sha256;
};

/**
 * Generate a deterministic canonicalized pointer-chain permutation matching
 * `script-examples/verify_cpu_latency_result.py::build_latency_chain`.
 */
CpuLatencyChainPermutation generate_cpu_latency_chain_permutation(
    uint64_t base_seed,
    uint64_t domain_id,
    size_t target_bytes,
    size_t stride_bytes,
    LatencyChainMode effective_mode,
    size_t tlb_locality_kb);

/**
 * Write linked pointers `slot[indices[i]] = &slot[indices[(i + 1) % N]]`
 * into @p buffer.
 */
bool populate_cpu_latency_chain_buffer(void* buffer,
                                       size_t buffer_bytes,
                                       const CpuLatencyChainPermutation& perm,
                                       std::string& out_reason);

/**
 * Walk `node_count` steps from `buffer` and verify a single Hamiltonian cycle
 * visiting every node in `[0, node_count)` once and returning to `buffer`.
 */
bool validate_cpu_latency_hamiltonian_cycle(const void* buffer,
                                            size_t buffer_bytes,
                                            size_t stride_bytes,
                                            size_t node_count);

/**
 * Map a pointer back to its node index in `[0, node_count)` if aligned and
 * in-bounds, or `std::nullopt` otherwise.
 */
std::optional<size_t> pointer_to_chain_node_index(
    const void* base_ptr,
    size_t buffer_bytes,
    size_t stride_bytes,
    size_t node_count,
    const uintptr_t* ptr) noexcept;

/** Return the slot pointer for @p node_index in @p base_ptr. */
inline uintptr_t* chain_node_to_pointer(void* base_ptr,
                                        size_t stride_bytes,
                                        size_t node_index) noexcept {
  return reinterpret_cast<uintptr_t*>(
      static_cast<uint8_t*>(base_ptr) + node_index * stride_bytes);
}

#endif  // CPU_LATENCY_CHAIN_H
