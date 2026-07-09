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
 * @file tlb_chain.h
 * @brief Page-native pointer chains for standalone TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef TLB_CHAIN_H
#define TLB_CHAIN_H

#include <cstddef>
#include <cstdint>

enum class TlbChainLayout {
  Spread = 0,
  Packed,
};

enum class TlbChainTraversalPolicy {
  RandomPagesRandomOffsets = 0,
  IncreasingPagesSharedOffset,
  IncreasingPagesRandomOffsets,
};

enum class TlbChainBuildStatus {
  Success = 0,
  InvalidArgument,
  InsufficientBuffer,
  IntegrityFailure,
};

enum class TlbChainValidationStatus {
  Valid = 0,
  InvalidArgument,
  NodeOutOfBounds,
  NodeMisaligned,
  DuplicateNode,
  EarlyCycle,
  DoesNotReturnToHead,
  CacheLineReuse,
  PageCountMismatch,
};

/** Verified physical properties of one page-native pointer chain. */
struct TlbChainDiagnostics {
  TlbChainLayout layout = TlbChainLayout::Spread;
  TlbChainTraversalPolicy traversal_policy =
      TlbChainTraversalPolicy::RandomPagesRandomOffsets;
  size_t requested_pages = 0;
  size_t actual_pages = 0;
  size_t node_count = 0;
  size_t unique_cache_lines = 0;
  size_t max_nodes_per_page = 0;
  size_t byte_span = 0;
  size_t page_size_bytes = 0;
  size_t requested_stride_bytes = 0;
  size_t effective_node_spacing_bytes = 0;
  uint64_t seed = 0;
  bool integrity_verified = false;
};

struct TlbChainBuildResult {
  TlbChainBuildStatus status = TlbChainBuildStatus::InvalidArgument;
  TlbChainValidationStatus validation_status =
      TlbChainValidationStatus::InvalidArgument;
  void* chain_head = nullptr;
  TlbChainDiagnostics diagnostics;
};

const char* tlb_chain_layout_to_string(TlbChainLayout layout);
const char* tlb_chain_traversal_policy_to_string(
    TlbChainTraversalPolicy policy);
const char* tlb_chain_build_status_to_string(TlbChainBuildStatus status);
const char* tlb_chain_validation_status_to_string(
    TlbChainValidationStatus status);

/** Derive a stable, layout-specific seed from one scheduled task seed. */
uint64_t derive_tlb_chain_layout_seed(uint64_t task_seed,
                                      TlbChainLayout layout);

/**
 * Build a circular chain with one logical node per requested spread page.
 *
 * Spread places exactly one node on every requested page. Packed places the
 * same number of nodes on consecutive distinct cache lines, minimizing the
 * physical page count without changing data-cache line count.
 * Pointer values are written in physical-slot order after traversal order has
 * been planned, so setup writes do not reveal the measured traversal sequence.
 */
TlbChainBuildResult build_tlb_chain(
    void* buffer,
    size_t buffer_size_bytes,
    size_t requested_pages,
    size_t page_size_bytes,
    size_t requested_stride_bytes,
    TlbChainLayout layout,
    TlbChainTraversalPolicy traversal_policy,
    uint64_t seed);

/** Traverse and independently verify one complete chain cycle. */
TlbChainValidationStatus validate_tlb_chain(
    void* buffer,
    size_t buffer_size_bytes,
    void* chain_head,
    const TlbChainDiagnostics& expected,
    TlbChainDiagnostics* observed = nullptr);

#endif  // TLB_CHAIN_H
