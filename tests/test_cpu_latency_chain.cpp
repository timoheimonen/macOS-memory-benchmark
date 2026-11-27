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
#include <string>
#include <vector>

#include "cpu_latency/cpu_latency_chain.h"
#include "cpu_latency/cpu_latency_work_plan.h"

namespace {

TEST(CpuLatencyChainTest, FixtureGoldenSeedsAndDigestsMatchPythonOracle) {
  // Exact values from tests/fixtures/cpu-latency-schema-v1-complete-current.json:
  // base_seed = 424242, stride = 256, mode = random-box, locality = 1024 KB.
  const auto custom_perm = generate_cpu_latency_chain_permutation(
      424242ULL, kCpuLatencyDomainCustom, 16384ULL, 256ULL,
      LatencyChainMode::RandomInBoxRandomBox, 1024ULL);
  ASSERT_TRUE(custom_perm.valid);
  EXPECT_EQ(custom_perm.chain_seed, 7436701275638583618ULL);
  EXPECT_EQ(custom_perm.node_count, 64u);
  EXPECT_EQ(custom_perm.indices[0], 0u);
  EXPECT_EQ(custom_perm.permutation_sha256,
            "968dc7e052a2c3f576982910159bcde7225ba76697bc3eb7a5cbb999ee159ea9");

  const auto main_perm = generate_cpu_latency_chain_permutation(
      424242ULL, kCpuLatencyDomainMainMemory, 1048576ULL, 256ULL,
      LatencyChainMode::RandomInBoxRandomBox, 1024ULL);
  ASSERT_TRUE(main_perm.valid);
  EXPECT_EQ(main_perm.chain_seed, 5661157505406129473ULL);
  EXPECT_EQ(main_perm.node_count, 4096u);
  EXPECT_EQ(main_perm.indices[0], 0u);
  EXPECT_EQ(main_perm.permutation_sha256,
            "7b2c27666a8b73e9af9989fd5e860feed9cd4943296534ea095ed25ed0eafb94");
}

TEST(CpuLatencyChainTest, AllFourChainModesProduceValidCanonicalPermutations) {
  const std::vector<std::pair<LatencyChainMode, size_t>> modes = {
      {LatencyChainMode::GlobalRandom, 0},
      {LatencyChainMode::RandomInBoxRandomBox, 16},
      {LatencyChainMode::SameRandomInBoxIncreasingBox, 16},
      {LatencyChainMode::DiffRandomInBoxIncreasingBox, 16},
  };

  for (const auto& [mode, loc_kb] : modes) {
    // 64 KiB buffer / 256 B stride = 256 nodes; 16 KiB locality = 64 nodes (4 boxes).
    const auto perm = generate_cpu_latency_chain_permutation(
        424242ULL, kCpuLatencyDomainCustom, 65536ULL, 256ULL, mode, loc_kb);
    ASSERT_TRUE(perm.valid);
    EXPECT_EQ(perm.indices.size(), 256u);
    EXPECT_EQ(perm.indices[0], 0u);
    EXPECT_EQ(perm.permutation_sha256.size(), 64u);

    std::vector<uint64_t> sorted = perm.indices;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size(); ++i) {
      EXPECT_EQ(sorted[i], i);
    }

    std::vector<uint8_t> buf(65536ULL, 0);
    std::string pop_reason;
    ASSERT_TRUE(
        populate_cpu_latency_chain_buffer(buf.data(), buf.size(), perm, pop_reason))
        << pop_reason;
    EXPECT_TRUE(validate_cpu_latency_hamiltonian_cycle(buf.data(), buf.size(),
                                                       256ULL, 256u));
  }
}

TEST(CpuLatencyChainTest, PartialLastBoxAndHamiltonianCorruptionDetection) {
  // Partial last box: 20480 bytes / 64 B = 320 nodes; 16 KiB / 64 B = 256 locality nodes
  // (1 full box of 256 + partial 2nd box of 64).
  for (LatencyChainMode mode :
       {LatencyChainMode::RandomInBoxRandomBox,
        LatencyChainMode::SameRandomInBoxIncreasingBox,
        LatencyChainMode::DiffRandomInBoxIncreasingBox}) {
    const auto perm = generate_cpu_latency_chain_permutation(
        98765ULL, kCpuLatencyDomainMainMemory, 20480ULL, 64ULL, mode, 16ULL);
    ASSERT_TRUE(perm.valid);
    EXPECT_EQ(perm.node_count, 320u);
    EXPECT_EQ(perm.indices[0], 0u);

    std::vector<uint8_t> buf(20480ULL, 0);
    std::string reason;
    ASSERT_TRUE(
        populate_cpu_latency_chain_buffer(buf.data(), buf.size(), perm, reason));
    ASSERT_TRUE(validate_cpu_latency_hamiltonian_cycle(buf.data(), buf.size(),
                                                       64ULL, 320u));

    // Corrupt node 0 to point to itself (short 1-node cycle).
    uintptr_t* slot0 = chain_node_to_pointer(buf.data(), 64ULL, 0);
    *slot0 = reinterpret_cast<uintptr_t>(slot0);
    EXPECT_FALSE(validate_cpu_latency_hamiltonian_cycle(buf.data(), buf.size(),
                                                        64ULL, 320u));
  }
}

TEST(CpuLatencyChainTest, PointerToNodeIndexHandlesBoundsAndMisalignment) {
  std::vector<uint8_t> buf(4096, 0);
  const auto node0 = pointer_to_chain_node_index(
      buf.data(), buf.size(), 256, 16, reinterpret_cast<uintptr_t*>(buf.data()));
  ASSERT_TRUE(node0.has_value());
  EXPECT_EQ(*node0, 0u);

  const auto node5 = pointer_to_chain_node_index(
      buf.data(), buf.size(), 256, 16,
      reinterpret_cast<uintptr_t*>(buf.data() + 5 * 256));
  ASSERT_TRUE(node5.has_value());
  EXPECT_EQ(*node5, 5u);

  // Misaligned pointer
  EXPECT_FALSE(pointer_to_chain_node_index(
                   buf.data(), buf.size(), 256, 16,
                   reinterpret_cast<uintptr_t*>(buf.data() + 8))
                   .has_value());
  // Out-of-bounds pointer
  EXPECT_FALSE(pointer_to_chain_node_index(
                   buf.data(), buf.size(), 256, 16,
                   reinterpret_cast<uintptr_t*>(buf.data() + buf.size()))
                   .has_value());
}

}  // namespace
