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
 * @file cpu_latency_chain.cpp
 * @brief Deterministic SplitMix64 pointer-chasing chain generator and validator
 */

#include "cpu_latency/cpu_latency_chain.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

#include "core/audit/deterministic_rng.h"
#include "core/config/constants.h"
#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"

namespace {

constexpr size_t kShaChunkEntries = 4096;

void append_uint64_le(uint64_t v, uint8_t* dst) noexcept {
  dst[0] = static_cast<uint8_t>(v & 0xFFU);
  dst[1] = static_cast<uint8_t>((v >> 8U) & 0xFFU);
  dst[2] = static_cast<uint8_t>((v >> 16U) & 0xFFU);
  dst[3] = static_cast<uint8_t>((v >> 24U) & 0xFFU);
  dst[4] = static_cast<uint8_t>((v >> 32U) & 0xFFU);
  dst[5] = static_cast<uint8_t>((v >> 40U) & 0xFFU);
  dst[6] = static_cast<uint8_t>((v >> 48U) & 0xFFU);
  dst[7] = static_cast<uint8_t>((v >> 56U) & 0xFFU);
}

std::string compute_indices_sha256(const std::vector<uint64_t>& indices) {
  HashUtils::Sha256Hasher hasher;
  std::array<uint8_t, kShaChunkEntries * sizeof(uint64_t)> buf{};
  size_t offset = 0;
  while (offset < indices.size()) {
    const size_t count = std::min(kShaChunkEntries, indices.size() - offset);
    for (size_t i = 0; i < count; ++i) {
      append_uint64_le(indices[offset + i], &buf[i * sizeof(uint64_t)]);
    }
    hasher.update(buf.data(), count * sizeof(uint64_t));
    offset += count;
  }
  return hasher.finalize_hex();
}

}  // namespace

CpuLatencyChainPermutation generate_cpu_latency_chain_permutation(
    uint64_t base_seed,
    uint64_t domain_id,
    size_t target_bytes,
    size_t stride_bytes,
    LatencyChainMode effective_mode,
    size_t tlb_locality_kb) {
  CpuLatencyChainPermutation perm;
  perm.stride_bytes = stride_bytes;
  perm.tlb_locality_kb = tlb_locality_kb;

  if (stride_bytes < sizeof(uintptr_t) ||
      (stride_bytes % sizeof(uintptr_t)) != 0 || target_bytes < stride_bytes) {
    perm.reason_code = "chain-geometry-invalid";
    return perm;
  }

  size_t locality_bytes = 0;
  if (!NumericUtils::checked_multiply(tlb_locality_kb, Constants::BYTES_PER_KB,
                                      locality_bytes)) {
    perm.reason_code = "chain-geometry-invalid";
    return perm;
  }

  if (effective_mode == LatencyChainMode::Auto) {
    effective_mode =
        resolve_latency_chain_mode(LatencyChainMode::Auto, locality_bytes);
  }
  perm.effective_mode = effective_mode;

  const size_t node_count = target_bytes / stride_bytes;
  if (node_count < 2) {
    perm.reason_code = "chain-geometry-invalid";
    return perm;
  }
  perm.node_count = node_count;

  size_t locality_nodes = 0;
  if (effective_mode != LatencyChainMode::GlobalRandom) {
    if (locality_bytes == 0) {
      perm.reason_code = "chain-mode-requires-locality";
      return perm;
    }
    locality_nodes = locality_bytes / stride_bytes;
    if (locality_nodes < 2) {
      perm.reason_code = "chain-geometry-invalid";
      return perm;
    }
  }
  perm.locality_nodes = locality_nodes;

  const uint64_t chain_seed = splitmix64_mix(base_seed ^ domain_id);
  perm.chain_seed = chain_seed;
  uint64_t state = chain_seed;

  std::vector<uint64_t> indices;
  try {
    if (effective_mode == LatencyChainMode::GlobalRandom) {
      indices.resize(node_count);
      std::iota(indices.begin(), indices.end(), uint64_t{0});
      fisher_yates_rejection_shuffle_with_state(indices, state);
    } else {
      const size_t box_count =
          (node_count + locality_nodes - 1) / locality_nodes;
      std::vector<size_t> box_order(box_count);
      std::iota(box_order.begin(), box_order.end(), size_t{0});
      if (effective_mode == LatencyChainMode::RandomInBoxRandomBox) {
        fisher_yates_rejection_shuffle_with_state(box_order, state);
      }

      indices.reserve(node_count);
      if (effective_mode == LatencyChainMode::SameRandomInBoxIncreasingBox) {
        std::vector<size_t> shared(locality_nodes);
        std::iota(shared.begin(), shared.end(), size_t{0});
        fisher_yates_rejection_shuffle_with_state(shared, state);
        for (size_t box_id : box_order) {
          const size_t start = box_id * locality_nodes;
          const size_t end = std::min(start + locality_nodes, node_count);
          for (size_t off : shared) {
            const size_t node = start + off;
            if (node < end) {
              indices.push_back(static_cast<uint64_t>(node));
            }
          }
        }
      } else {
        for (size_t box_id : box_order) {
          const size_t start = box_id * locality_nodes;
          const size_t end = std::min(start + locality_nodes, node_count);
          std::vector<uint64_t> box(end - start);
          std::iota(box.begin(), box.end(), static_cast<uint64_t>(start));
          fisher_yates_rejection_shuffle_with_state(box, state);
          indices.insert(indices.end(), box.begin(), box.end());
        }
      }
    }
  } catch (...) {
    perm.reason_code = "chain-allocation-failed";
    return perm;
  }

  if (indices.size() != node_count) {
    perm.reason_code = "chain-hamiltonian-invalid";
    return perm;
  }
  const auto zero_it = std::find(indices.begin(), indices.end(), uint64_t{0});
  if (zero_it == indices.end()) {
    perm.reason_code = "chain-hamiltonian-invalid";
    return perm;
  }
  std::rotate(indices.begin(), zero_it, indices.end());
  if (indices[0] != 0) {
    perm.reason_code = "chain-hamiltonian-invalid";
    return perm;
  }

  perm.permutation_sha256 = compute_indices_sha256(indices);
  perm.indices = std::move(indices);
  perm.valid = true;
  perm.reason_code = "valid";
  return perm;
}

bool populate_cpu_latency_chain_buffer(void* buffer,
                                       size_t buffer_bytes,
                                       const CpuLatencyChainPermutation& perm,
                                       std::string& out_reason) {
  if (buffer == nullptr || !perm.valid || perm.node_count < 2 ||
      perm.stride_bytes < sizeof(uintptr_t) ||
      (perm.stride_bytes % sizeof(uintptr_t)) != 0 ||
      perm.indices.size() != perm.node_count || perm.indices[0] != 0) {
    out_reason = "chain-geometry-invalid";
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(buffer) % sizeof(uintptr_t)) != 0) {
    out_reason = "chain-alignment-invalid";
    return false;
  }
  size_t required_bytes = 0;
  if (!NumericUtils::checked_multiply(perm.node_count, perm.stride_bytes,
                                      required_bytes) ||
      required_bytes > buffer_bytes) {
    out_reason = "chain-buffer-too-small";
    return false;
  }

  uint8_t* const base = static_cast<uint8_t*>(buffer);
  const size_t n = perm.node_count;
  for (size_t i = 0; i < n; ++i) {
    const uint64_t cur_node = perm.indices[i];
    const uint64_t next_node = perm.indices[(i + 1) % n];
    if (cur_node >= n || next_node >= n) {
      out_reason = "chain-hamiltonian-invalid";
      return false;
    }
    uintptr_t* const slot = reinterpret_cast<uintptr_t*>(
        base + static_cast<size_t>(cur_node) * perm.stride_bytes);
    const uintptr_t next_addr = reinterpret_cast<uintptr_t>(
        base + static_cast<size_t>(next_node) * perm.stride_bytes);
    *slot = next_addr;
  }

  out_reason = "valid";
  return true;
}

bool validate_cpu_latency_hamiltonian_cycle(const void* buffer,
                                            size_t buffer_bytes,
                                            size_t stride_bytes,
                                            size_t node_count) {
  if (buffer == nullptr || node_count < 2 || stride_bytes < sizeof(uintptr_t) ||
      (stride_bytes % sizeof(uintptr_t)) != 0) {
    return false;
  }
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(buffer);
  if ((base_addr % sizeof(uintptr_t)) != 0) {
    return false;
  }
  size_t required_bytes = 0;
  if (!NumericUtils::checked_multiply(node_count, stride_bytes, required_bytes) ||
      required_bytes > buffer_bytes) {
    return false;
  }

  std::vector<uint8_t> visited(node_count, 0);
  uintptr_t cur_addr = base_addr;
  for (size_t step = 0; step < node_count; ++step) {
    if (cur_addr < base_addr) {
      return false;
    }
    const uintptr_t offset = cur_addr - base_addr;
    if ((offset % stride_bytes) != 0) {
      return false;
    }
    const size_t node = static_cast<size_t>(offset / stride_bytes);
    if (node >= node_count || offset + sizeof(uintptr_t) > buffer_bytes) {
      return false;
    }
    if (visited[node] != 0) {
      return false;
    }
    visited[node] = 1;
    cur_addr = *reinterpret_cast<const uintptr_t*>(cur_addr);
  }
  return cur_addr == base_addr;
}

std::optional<size_t> pointer_to_chain_node_index(
    const void* base_ptr,
    size_t buffer_bytes,
    size_t stride_bytes,
    size_t node_count,
    const uintptr_t* ptr) noexcept {
  if (base_ptr == nullptr || ptr == nullptr || node_count == 0 ||
      stride_bytes < sizeof(uintptr_t) ||
      (stride_bytes % sizeof(uintptr_t)) != 0) {
    return std::nullopt;
  }
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base_ptr);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  if (addr < base_addr) {
    return std::nullopt;
  }
  const uintptr_t offset = addr - base_addr;
  if ((offset % stride_bytes) != 0 || offset + sizeof(uintptr_t) > buffer_bytes) {
    return std::nullopt;
  }
  const size_t node = static_cast<size_t>(offset / stride_bytes);
  if (node >= node_count) {
    return std::nullopt;
  }
  return node;
}
