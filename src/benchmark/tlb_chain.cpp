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
 * @file tlb_chain.cpp
 * @brief Page-native pointer chains for standalone TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "benchmark/tlb_chain.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/config/constants.h"

namespace {

uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

bool add_would_overflow(size_t left, size_t right) {
  return left > std::numeric_limits<size_t>::max() - right;
}

bool calculate_effective_spacing(size_t requested_stride_bytes,
                                 size_t& effective_spacing_bytes) {
  const size_t cache_line_bytes = Constants::CACHE_LINE_SIZE_BYTES;
  const size_t minimum_spacing =
      std::max(requested_stride_bytes, cache_line_bytes);
  if (add_would_overflow(minimum_spacing, cache_line_bytes - 1)) {
    return false;
  }
  effective_spacing_bytes =
      ((minimum_spacing + cache_line_bytes - 1) / cache_line_bytes) *
      cache_line_bytes;
  return effective_spacing_bytes >= cache_line_bytes;
}

bool pointer_in_buffer(uintptr_t address,
                       uintptr_t buffer_start,
                       uintptr_t buffer_end) {
  return address >= buffer_start && address <= buffer_end - sizeof(uintptr_t);
}

bool valid_layout(TlbChainLayout layout) {
  return layout == TlbChainLayout::Spread || layout == TlbChainLayout::Packed;
}

bool valid_traversal_policy(TlbChainTraversalPolicy policy) {
  return policy == TlbChainTraversalPolicy::RandomPagesRandomOffsets ||
         policy == TlbChainTraversalPolicy::IncreasingPagesSharedOffset ||
         policy == TlbChainTraversalPolicy::IncreasingPagesRandomOffsets;
}

}  // namespace

const char* tlb_chain_layout_to_string(TlbChainLayout layout) {
  switch (layout) {
    case TlbChainLayout::Spread:
      return "spread";
    case TlbChainLayout::Packed:
      return "packed";
  }
  return "spread";
}

const char* tlb_chain_traversal_policy_to_string(
    TlbChainTraversalPolicy policy) {
  switch (policy) {
    case TlbChainTraversalPolicy::RandomPagesRandomOffsets:
      return "random-pages-random-offsets";
    case TlbChainTraversalPolicy::IncreasingPagesSharedOffset:
      return "increasing-pages-shared-offset";
    case TlbChainTraversalPolicy::IncreasingPagesRandomOffsets:
      return "increasing-pages-random-offsets";
  }
  return "random-pages-random-offsets";
}

const char* tlb_chain_build_status_to_string(TlbChainBuildStatus status) {
  switch (status) {
    case TlbChainBuildStatus::Success:
      return "success";
    case TlbChainBuildStatus::InvalidArgument:
      return "invalid-argument";
    case TlbChainBuildStatus::InsufficientBuffer:
      return "insufficient-buffer";
    case TlbChainBuildStatus::IntegrityFailure:
      return "integrity-failure";
  }
  return "invalid-argument";
}

const char* tlb_chain_validation_status_to_string(
    TlbChainValidationStatus status) {
  switch (status) {
    case TlbChainValidationStatus::Valid:
      return "valid";
    case TlbChainValidationStatus::InvalidArgument:
      return "invalid-argument";
    case TlbChainValidationStatus::NodeOutOfBounds:
      return "node-out-of-bounds";
    case TlbChainValidationStatus::NodeMisaligned:
      return "node-misaligned";
    case TlbChainValidationStatus::DuplicateNode:
      return "duplicate-node";
    case TlbChainValidationStatus::EarlyCycle:
      return "early-cycle";
    case TlbChainValidationStatus::DoesNotReturnToHead:
      return "does-not-return-to-head";
    case TlbChainValidationStatus::CacheLineReuse:
      return "cache-line-reuse";
    case TlbChainValidationStatus::PageCountMismatch:
      return "page-count-mismatch";
  }
  return "invalid-argument";
}

uint64_t derive_tlb_chain_layout_seed(uint64_t task_seed,
                                      TlbChainLayout layout) {
  constexpr uint64_t kSpreadSalt = 0x535052454144ULL;
  constexpr uint64_t kPackedSalt = 0x5041434b4544ULL;
  return splitmix64(task_seed ^
                    (layout == TlbChainLayout::Spread ? kSpreadSalt
                                                       : kPackedSalt));
}

TlbChainValidationStatus validate_tlb_chain_with_scratch(
    void* buffer,
    size_t buffer_size_bytes,
    void* chain_head,
    const TlbChainDiagnostics& expected,
    TlbChainDiagnostics* observed,
    TlbChainScratch& scratch) {
  const size_t cache_line_bytes = Constants::CACHE_LINE_SIZE_BYTES;
  if (buffer == nullptr || chain_head == nullptr ||
      buffer_size_bytes < sizeof(uintptr_t) || expected.node_count == 0 ||
      expected.page_size_bytes == 0 || !valid_layout(expected.layout) ||
      !valid_traversal_policy(expected.traversal_policy) ||
      (expected.page_size_bytes % cache_line_bytes) != 0) {
    return TlbChainValidationStatus::InvalidArgument;
  }

  const uintptr_t buffer_start = reinterpret_cast<uintptr_t>(buffer);
  if (buffer_start >
      std::numeric_limits<uintptr_t>::max() - buffer_size_bytes) {
    return TlbChainValidationStatus::InvalidArgument;
  }
  const uintptr_t buffer_end = buffer_start + buffer_size_bytes;
  if ((buffer_start % expected.page_size_bytes) != 0 ||
      !pointer_in_buffer(reinterpret_cast<uintptr_t>(chain_head),
                         buffer_start,
                         buffer_end)) {
    return TlbChainValidationStatus::InvalidArgument;
  }

  auto& visited_nodes = scratch.visited_nodes;
  auto& visited_pages = scratch.visited_pages;
  auto& visited_cache_lines = scratch.visited_cache_lines;
  auto& nodes_per_page = scratch.nodes_per_page;
  visited_nodes.clear();
  visited_pages.clear();
  visited_cache_lines.clear();
  nodes_per_page.clear();
  if (visited_nodes.bucket_count() < expected.node_count) {
    visited_nodes.reserve(expected.node_count);
  }
  if (visited_pages.bucket_count() < expected.node_count) {
    visited_pages.reserve(expected.node_count);
  }
  if (visited_cache_lines.bucket_count() < expected.node_count) {
    visited_cache_lines.reserve(expected.node_count);
  }
  if (nodes_per_page.bucket_count() < expected.node_count) {
    nodes_per_page.reserve(expected.node_count);
  }

  uintptr_t current = reinterpret_cast<uintptr_t>(chain_head);
  uintptr_t minimum_address = current;
  uintptr_t maximum_address = current;
  size_t max_nodes_per_page = 0;

  for (size_t node_index = 0; node_index < expected.node_count;
       ++node_index) {
    if (!pointer_in_buffer(current, buffer_start, buffer_end)) {
      return TlbChainValidationStatus::NodeOutOfBounds;
    }
    if ((current % alignof(uintptr_t)) != 0) {
      return TlbChainValidationStatus::NodeMisaligned;
    }
    if (!visited_nodes.insert(current).second) {
      return node_index + 1 < expected.node_count
                 ? TlbChainValidationStatus::EarlyCycle
                 : TlbChainValidationStatus::DuplicateNode;
    }

    const size_t relative_offset =
        static_cast<size_t>(current - buffer_start);
    const size_t page_index = relative_offset / expected.page_size_bytes;
    const size_t cache_line_index = relative_offset / cache_line_bytes;
    visited_pages.insert(page_index);
    visited_cache_lines.insert(cache_line_index);
    const size_t page_node_count = ++nodes_per_page[page_index];
    max_nodes_per_page = std::max(max_nodes_per_page, page_node_count);
    minimum_address = std::min(minimum_address, current);
    maximum_address = std::max(maximum_address, current);

    uintptr_t next = 0;
    std::memcpy(&next, reinterpret_cast<const void*>(current), sizeof(next));
    if (next == reinterpret_cast<uintptr_t>(chain_head) &&
        node_index + 1 < expected.node_count) {
      return TlbChainValidationStatus::EarlyCycle;
    }
    current = next;
  }

  if (current != reinterpret_cast<uintptr_t>(chain_head)) {
    return TlbChainValidationStatus::DoesNotReturnToHead;
  }
  if (visited_cache_lines.size() != expected.node_count) {
    return TlbChainValidationStatus::CacheLineReuse;
  }
  if (expected.layout == TlbChainLayout::Spread &&
      visited_pages.size() != expected.requested_pages) {
    return TlbChainValidationStatus::PageCountMismatch;
  }

  if (observed != nullptr) {
    *observed = expected;
    observed->actual_pages = visited_pages.size();
    observed->node_count = visited_nodes.size();
    observed->unique_cache_lines = visited_cache_lines.size();
    observed->max_nodes_per_page = max_nodes_per_page;
    observed->byte_span =
        static_cast<size_t>(maximum_address - minimum_address) +
        sizeof(uintptr_t);
    observed->integrity_verified = true;
  }
  return TlbChainValidationStatus::Valid;
}

TlbChainValidationStatus validate_tlb_chain(
    void* buffer,
    size_t buffer_size_bytes,
    void* chain_head,
    const TlbChainDiagnostics& expected,
    TlbChainDiagnostics* observed) {
  TlbChainScratch scratch;
  return validate_tlb_chain_with_scratch(buffer,
                                         buffer_size_bytes,
                                         chain_head,
                                         expected,
                                         observed,
                                         scratch);
}

TlbChainBuildResult build_tlb_chain(
    void* buffer,
    size_t buffer_size_bytes,
    size_t requested_pages,
    size_t page_size_bytes,
    size_t requested_stride_bytes,
    TlbChainLayout layout,
    TlbChainTraversalPolicy traversal_policy,
    uint64_t seed) {
  TlbChainScratch scratch;
  return build_tlb_chain(buffer,
                         buffer_size_bytes,
                         requested_pages,
                         page_size_bytes,
                         requested_stride_bytes,
                         layout,
                         traversal_policy,
                         seed,
                         scratch);
}

TlbChainBuildResult build_tlb_chain(
    void* buffer,
    size_t buffer_size_bytes,
    size_t requested_pages,
    size_t page_size_bytes,
    size_t requested_stride_bytes,
    TlbChainLayout layout,
    TlbChainTraversalPolicy traversal_policy,
    uint64_t seed,
    TlbChainScratch& scratch) {
  TlbChainBuildResult result;
  result.diagnostics.layout = layout;
  result.diagnostics.traversal_policy = traversal_policy;
  result.diagnostics.requested_pages = requested_pages;
  result.diagnostics.node_count = requested_pages;
  result.diagnostics.page_size_bytes = page_size_bytes;
  result.diagnostics.requested_stride_bytes = requested_stride_bytes;
  result.diagnostics.seed = seed;

  const size_t cache_line_bytes = Constants::CACHE_LINE_SIZE_BYTES;
  size_t spread_spacing_bytes = 0;
  if (buffer == nullptr || requested_pages == 0 || !valid_layout(layout) ||
      !valid_traversal_policy(traversal_policy) ||
      page_size_bytes < cache_line_bytes ||
      (page_size_bytes % cache_line_bytes) != 0 ||
      requested_stride_bytes == 0 ||
      (requested_stride_bytes % alignof(uintptr_t)) != 0 ||
      requested_stride_bytes > page_size_bytes ||
      buffer_size_bytes < sizeof(uintptr_t) ||
      (reinterpret_cast<uintptr_t>(buffer) % page_size_bytes) != 0 ||
      !calculate_effective_spacing(requested_stride_bytes,
                                   spread_spacing_bytes) ||
      spread_spacing_bytes > page_size_bytes) {
    return result;
  }
  const size_t effective_spacing_bytes =
      layout == TlbChainLayout::Packed ? cache_line_bytes
                                       : spread_spacing_bytes;
  result.diagnostics.effective_node_spacing_bytes =
      effective_spacing_bytes;

  if (layout == TlbChainLayout::Spread) {
    if (requested_pages > buffer_size_bytes / page_size_bytes) {
      result.status = TlbChainBuildStatus::InsufficientBuffer;
      return result;
    }
  } else {
    const size_t packed_capacity =
        1 + (buffer_size_bytes - sizeof(uintptr_t)) /
                effective_spacing_bytes;
    if (requested_pages > packed_capacity) {
      result.status = TlbChainBuildStatus::InsufficientBuffer;
      return result;
    }
  }

  try {
    auto& physical_offsets = scratch.physical_offsets;
    physical_offsets.resize(requested_pages);
    std::mt19937_64 offset_rng(splitmix64(seed ^ 0x4f464653455453ULL));

    if (layout == TlbChainLayout::Spread) {
      const size_t slot_count =
          1 + (page_size_bytes - cache_line_bytes) /
                  effective_spacing_bytes;
      std::uniform_int_distribution<size_t> slot_distribution(0,
                                                               slot_count - 1);
      size_t shared_slot = 0;
      if (traversal_policy ==
          TlbChainTraversalPolicy::IncreasingPagesSharedOffset) {
        shared_slot = slot_distribution(offset_rng);
      }
      for (size_t page_index = 0; page_index < requested_pages;
           ++page_index) {
        const size_t slot =
            traversal_policy ==
                    TlbChainTraversalPolicy::IncreasingPagesSharedOffset
                ? shared_slot
                : slot_distribution(offset_rng);
        physical_offsets[page_index] =
            page_index * page_size_bytes + slot * effective_spacing_bytes;
      }
    } else {
      for (size_t node_index = 0; node_index < requested_pages;
           ++node_index) {
        physical_offsets[node_index] = node_index * effective_spacing_bytes;
      }
    }

    auto& traversal = scratch.traversal;
    traversal.resize(requested_pages);
    std::iota(traversal.begin(), traversal.end(), 0);
    if (traversal_policy ==
        TlbChainTraversalPolicy::RandomPagesRandomOffsets) {
      std::mt19937_64 traversal_rng(
          splitmix64(seed ^ 0x5452415645525345ULL));
      std::shuffle(traversal.begin(), traversal.end(), traversal_rng);
    }

    auto& physical_writes = scratch.physical_writes;
    physical_writes.clear();
    physical_writes.reserve(requested_pages);
    for (size_t traversal_index = 0; traversal_index < requested_pages;
         ++traversal_index) {
      const size_t current_node = traversal[traversal_index];
      const size_t next_node =
          traversal[(traversal_index + 1) % requested_pages];
      physical_writes.emplace_back(physical_offsets[current_node],
                                   physical_offsets[next_node]);
    }
    std::sort(physical_writes.begin(), physical_writes.end());

    auto* bytes = static_cast<unsigned char*>(buffer);
    for (const auto& write : physical_writes) {
      const uintptr_t next_address =
          reinterpret_cast<uintptr_t>(bytes + write.second);
      std::memcpy(bytes + write.first, &next_address, sizeof(next_address));
    }

    result.chain_head = bytes + physical_offsets[traversal.front()];
    TlbChainDiagnostics observed;
    result.validation_status = validate_tlb_chain_with_scratch(
        buffer,
        buffer_size_bytes,
        result.chain_head,
        result.diagnostics,
        &observed,
        scratch);
    if (result.validation_status != TlbChainValidationStatus::Valid) {
      result.status = TlbChainBuildStatus::IntegrityFailure;
      result.chain_head = nullptr;
      return result;
    }
    result.diagnostics = observed;
    result.status = TlbChainBuildStatus::Success;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = TlbChainBuildStatus::InsufficientBuffer;
    result.chain_head = nullptr;
    return result;
  }
}
