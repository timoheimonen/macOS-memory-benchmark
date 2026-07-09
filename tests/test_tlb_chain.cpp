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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "benchmark/benchmark_tests.h"
#include "benchmark/tlb_chain.h"
#include "core/timing/timer.h"

namespace {

class PageBuffer {
 public:
  explicit PageBuffer(size_t size_bytes) : size_bytes_(size_bytes) {
    pointer_ = mmap(nullptr,
                    size_bytes_,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0);
    if (pointer_ == MAP_FAILED) {
      pointer_ = nullptr;
    }
  }

  ~PageBuffer() {
    if (pointer_ != nullptr) {
      (void)munmap(pointer_, size_bytes_);
    }
  }

  void* get() const { return pointer_; }
  size_t size() const { return size_bytes_; }

 private:
  void* pointer_ = nullptr;
  size_t size_bytes_ = 0;
};

std::vector<size_t> traversal_offsets(void* buffer,
                                      void* head,
                                      size_t node_count) {
  std::vector<size_t> offsets;
  offsets.reserve(node_count);
  uintptr_t current = reinterpret_cast<uintptr_t>(head);
  const uintptr_t base = reinterpret_cast<uintptr_t>(buffer);
  for (size_t i = 0; i < node_count; ++i) {
    offsets.push_back(static_cast<size_t>(current - base));
    uintptr_t next = 0;
    std::memcpy(&next, reinterpret_cast<void*>(current), sizeof(next));
    current = next;
  }
  return offsets;
}

}  // namespace

TEST(TlbChainTest, SpreadVisitsExactlyEveryRequestedPage) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 17;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  const TlbChainBuildResult result = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      kRequestedPages,
      page_size,
      256,
      TlbChainLayout::Spread,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      1234);

  ASSERT_EQ(result.status, TlbChainBuildStatus::Success);
  EXPECT_NE(result.chain_head, nullptr);
  EXPECT_TRUE(result.diagnostics.integrity_verified);
  EXPECT_EQ(result.diagnostics.requested_pages, kRequestedPages);
  EXPECT_EQ(result.diagnostics.actual_pages, kRequestedPages);
  EXPECT_EQ(result.diagnostics.node_count, kRequestedPages);
  EXPECT_EQ(result.diagnostics.unique_cache_lines, kRequestedPages);
  EXPECT_EQ(result.diagnostics.max_nodes_per_page, 1U);
  EXPECT_EQ(validate_tlb_chain(buffer.get(),
                               buffer.size(),
                               result.chain_head,
                               result.diagnostics),
            TlbChainValidationStatus::Valid);
}

TEST(TlbChainTest, PackedControlPreservesNodeAndCacheLineCounts) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 65;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  const TlbChainBuildResult spread = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      kRequestedPages,
      page_size,
      256,
      TlbChainLayout::Spread,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      111);
  ASSERT_EQ(spread.status, TlbChainBuildStatus::Success);

  const TlbChainBuildResult packed = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      kRequestedPages,
      page_size,
      256,
      TlbChainLayout::Packed,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      222);
  ASSERT_EQ(packed.status, TlbChainBuildStatus::Success);

  EXPECT_EQ(packed.diagnostics.node_count, spread.diagnostics.node_count);
  EXPECT_EQ(packed.diagnostics.unique_cache_lines,
            spread.diagnostics.unique_cache_lines);
  EXPECT_LT(packed.diagnostics.actual_pages,
            spread.diagnostics.actual_pages);
  EXPECT_GT(packed.diagnostics.max_nodes_per_page, 1U);
  EXPECT_EQ(packed.diagnostics.effective_node_spacing_bytes, 64U);
  EXPECT_TRUE(packed.diagnostics.integrity_verified);
}

TEST(TlbChainTest, PackedControlStaysDenseWithPageSizedRequestedStride) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 32;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  const TlbChainBuildResult result = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      kRequestedPages,
      page_size,
      page_size,
      TlbChainLayout::Packed,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      321);

  ASSERT_EQ(result.status, TlbChainBuildStatus::Success);
  EXPECT_EQ(result.diagnostics.node_count, kRequestedPages);
  EXPECT_EQ(result.diagnostics.unique_cache_lines, kRequestedPages);
  EXPECT_EQ(result.diagnostics.actual_pages, 1U);
  EXPECT_EQ(result.diagnostics.effective_node_spacing_bytes, 64U);
}

TEST(TlbChainTest, SameSeedReproducesTraversalAndDifferentSeedChangesIt) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 32;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  auto build_offsets = [&](uint64_t seed) {
    const TlbChainBuildResult result = build_tlb_chain(
        buffer.get(),
        buffer.size(),
        kRequestedPages,
        page_size,
        128,
        TlbChainLayout::Spread,
        TlbChainTraversalPolicy::RandomPagesRandomOffsets,
        seed);
    EXPECT_EQ(result.status, TlbChainBuildStatus::Success);
    return traversal_offsets(buffer.get(), result.chain_head, kRequestedPages);
  };

  const std::vector<size_t> first = build_offsets(9001);
  const std::vector<size_t> repeated = build_offsets(9001);
  const std::vector<size_t> different = build_offsets(9002);
  EXPECT_EQ(first, repeated);
  EXPECT_NE(first, different);
}

TEST(TlbChainTest, ValidationRejectsPointerOutsideBuffer) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 8;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  const TlbChainBuildResult result = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      kRequestedPages,
      page_size,
      64,
      TlbChainLayout::Spread,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      55);
  ASSERT_EQ(result.status, TlbChainBuildStatus::Success);

  const uintptr_t outside =
      reinterpret_cast<uintptr_t>(buffer.get()) + buffer.size();
  std::memcpy(result.chain_head, &outside, sizeof(outside));
  EXPECT_EQ(validate_tlb_chain(buffer.get(),
                               buffer.size(),
                               result.chain_head,
                               result.diagnostics),
            TlbChainValidationStatus::NodeOutOfBounds);
}

TEST(TlbChainTest, RejectsInvalidStrideAndInsufficientSpreadBuffer) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  PageBuffer buffer(4 * page_size);
  ASSERT_NE(buffer.get(), nullptr);

  EXPECT_EQ(build_tlb_chain(buffer.get(),
                            buffer.size(),
                            4,
                            page_size,
                            10,
                            TlbChainLayout::Spread,
                            TlbChainTraversalPolicy::RandomPagesRandomOffsets,
                            1)
                .status,
            TlbChainBuildStatus::InvalidArgument);
  EXPECT_EQ(build_tlb_chain(buffer.get(),
                            buffer.size(),
                            4,
                            page_size,
                            64,
                            static_cast<TlbChainLayout>(99),
                            TlbChainTraversalPolicy::RandomPagesRandomOffsets,
                            1)
                .status,
            TlbChainBuildStatus::InvalidArgument);
  EXPECT_EQ(build_tlb_chain(buffer.get(),
                            buffer.size(),
                            5,
                            page_size,
                            64,
                            TlbChainLayout::Spread,
                            TlbChainTraversalPolicy::RandomPagesRandomOffsets,
                            1)
                .status,
            TlbChainBuildStatus::InsufficientBuffer);
}

TEST(TlbChainTest, AcceptsAlignedStrideThatDoesNotDividePage) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  PageBuffer buffer(8 * page_size);
  ASSERT_NE(buffer.get(), nullptr);
  ASSERT_EQ(136U % alignof(uintptr_t), 0U);
  ASSERT_NE(page_size % 136U, 0U);

  const TlbChainBuildResult result = build_tlb_chain(
      buffer.get(),
      buffer.size(),
      8,
      page_size,
      136,
      TlbChainLayout::Spread,
      TlbChainTraversalPolicy::RandomPagesRandomOffsets,
      123);
  ASSERT_EQ(result.status, TlbChainBuildStatus::Success);
  EXPECT_EQ(result.diagnostics.actual_pages, 8U);
  EXPECT_EQ(result.diagnostics.effective_node_spacing_bytes, 192U);
}

TEST(TlbChainTest, LayoutSeedsAreStableAndDistinct) {
  EXPECT_EQ(derive_tlb_chain_layout_seed(42, TlbChainLayout::Spread),
            derive_tlb_chain_layout_seed(42, TlbChainLayout::Spread));
  EXPECT_NE(derive_tlb_chain_layout_seed(42, TlbChainLayout::Spread),
            derive_tlb_chain_layout_seed(42, TlbChainLayout::Packed));
  EXPECT_NE(derive_tlb_chain_layout_seed(42, TlbChainLayout::Spread),
            derive_tlb_chain_layout_seed(43, TlbChainLayout::Spread));
}

TEST(TlbChainTest, SpreadAndPackedRunThroughLatencyKernelIntegration) {
  const size_t page_size = static_cast<size_t>(getpagesize());
  constexpr size_t kRequestedPages = 32;
  PageBuffer buffer(kRequestedPages * page_size);
  ASSERT_NE(buffer.get(), nullptr);
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  for (TlbChainLayout layout :
       {TlbChainLayout::Spread, TlbChainLayout::Packed}) {
    const TlbChainBuildResult result = build_tlb_chain(
        buffer.get(),
        buffer.size(),
        kRequestedPages,
        page_size,
        256,
        layout,
        TlbChainTraversalPolicy::RandomPagesRandomOffsets,
        1234);
    ASSERT_EQ(result.status, TlbChainBuildStatus::Success);
    EXPECT_GT(run_latency_test(result.chain_head,
                               10000,
                               *timer,
                               nullptr,
                               0),
              0.0);
  }
}
