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
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/config/constants.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_executor.h"
#include "llm_memory/llm_work_plan.h"

extern "C" uint64_t verify_llm_kernel_callee_saved_registers_asm(uintptr_t function_address,
                                                                 const uintptr_t* arguments);

// Test-only direct-call probe. Unlike the older six-argument pattern probe,
// this passes all seven LLM kernel arguments and checks the complete 128-bit
// q8-q15 values in addition to x19-x28. The probe restores the caller's full
// vector state even though base AAPCS64 requires only d8-d15 preservation.
__asm__(R"ASM(
.text
.p2align 4
.global _verify_llm_kernel_callee_saved_registers_asm
_verify_llm_kernel_callee_saved_registers_asm:
    stp x29, x30, [sp, #-224]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp q8, q9, [sp, #96]
    stp q10, q11, [sp, #128]
    stp q12, q13, [sp, #160]
    stp q14, q15, [sp, #192]

    mov x16, x0
    mov x17, x1
    ldp x0, x1, [x17, #0]
    ldp x2, x3, [x17, #16]
    ldp x4, x5, [x17, #32]
    ldr x6, [x17, #48]

    mov x19, #0x1919
    mov x20, #0x2020
    mov x21, #0x2121
    mov x22, #0x2222
    mov x23, #0x2323
    mov x24, #0x2424
    mov x25, #0x2525
    mov x26, #0x2626
    mov x27, #0x2727
    mov x28, #0x2828
    movi v8.16b, #0x88
    movi v9.16b, #0x99
    movi v10.16b, #0xaa
    movi v11.16b, #0xbb
    movi v12.16b, #0xcc
    movi v13.16b, #0xdd
    movi v14.16b, #0xee
    movi v15.16b, #0xff

    blr x16

    mov x17, #1
    mov x9, #0x1919
    cmp x19, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2020
    cmp x20, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2121
    cmp x21, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2222
    cmp x22, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2323
    cmp x23, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2424
    cmp x24, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2525
    cmp x25, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2626
    cmp x26, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2727
    cmp x27, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2828
    cmp x28, x9
    csel x17, x17, xzr, eq

    movi v0.16b, #0x88
    cmeq v1.16b, v8.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0x99
    cmeq v1.16b, v9.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xaa
    cmeq v1.16b, v10.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xbb
    cmeq v1.16b, v11.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xcc
    cmeq v1.16b, v12.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xdd
    cmeq v1.16b, v13.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xee
    cmeq v1.16b, v14.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq
    movi v0.16b, #0xff
    cmeq v1.16b, v15.16b, v0.16b
    uminv b1, v1.16b
    umov w9, v1.b[0]
    cmp w9, #0xff
    csel x17, x17, xzr, eq

    ldp q14, q15, [sp, #192]
    ldp q12, q13, [sp, #160]
    ldp q10, q11, [sp, #128]
    ldp q8, q9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #224
    mov x0, x17
    ret
)ASM");

namespace {

constexpr uint64_t kStepMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kLayerMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kBatchMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kWordMultiplier = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kAppendKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kAppendVDomain = 0x5656565656565656ULL;
constexpr uint64_t kInitialA = 0x243F6A8885A308D3ULL;
constexpr uint64_t kInitialB = 0x13198A2E03707344ULL;
constexpr uint64_t kWeightDomain = 0x5745494748545F31ULL;
constexpr uint64_t kReadKDomain = 0x4B5F524541445F31ULL;
constexpr uint64_t kReadVDomain = 0x565F524541445F31ULL;
constexpr std::array<size_t, 8> kExactTailSizes = {0, 1, 31, 32, 33, 511, 512, 513};

class GuardedMapping {
 public:
  explicit GuardedMapping(size_t payload_size)
      : page_size_(static_cast<size_t>(getpagesize())), mapping_size_(page_size_ * 3) {
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return;
    }
    uint8_t* bytes = static_cast<uint8_t*>(mapping_);
    if (mprotect(bytes, page_size_, PROT_NONE) != 0 || mprotect(bytes + 2 * page_size_, page_size_, PROT_NONE) != 0) {
      munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      return;
    }
    accessible_begin_ = bytes + page_size_;
    payload_ = accessible_begin_ + page_size_ - payload_size;
    std::memset(accessible_begin_, 0xD3, page_size_);
  }

  ~GuardedMapping() {
    if (mapping_ != nullptr) {
      munmap(mapping_, mapping_size_);
    }
  }

  GuardedMapping(const GuardedMapping&) = delete;
  GuardedMapping& operator=(const GuardedMapping&) = delete;

  bool valid() const { return mapping_ != nullptr; }
  uint8_t* payload() const { return payload_; }
  uint8_t* accessible_begin() const { return accessible_begin_; }
  size_t prefix_size() const { return static_cast<size_t>(payload_ - accessible_begin_); }

 private:
  void* mapping_ = nullptr;
  uint8_t* accessible_begin_ = nullptr;
  uint8_t* payload_ = nullptr;
  size_t page_size_ = 0;
  size_t mapping_size_ = 0;
};

uint64_t rotate_left(uint64_t value, unsigned int amount) { return (value << amount) | (value >> (64 - amount)); }

uint64_t component_domain(LlmChecksumComponent component) {
  switch (component) {
    case LlmChecksumComponent::Weight:
      return kWeightDomain;
    case LlmChecksumComponent::K:
      return kReadKDomain;
    case LlmChecksumComponent::V:
      return kReadVDomain;
  }
  return 0;
}

LlmReadChecksumComponent oracle_initial(LlmChecksumComponent component) {
  const uint64_t domain = component_domain(component);
  return {kInitialA ^ domain, kInitialB + domain, 0, 0};
}

LlmWorkerChecksum oracle_initial_worker() {
  return {oracle_initial(LlmChecksumComponent::Weight), oracle_initial(LlmChecksumComponent::K),
          oracle_initial(LlmChecksumComponent::V)};
}

uint64_t load_partial_little_endian(const uint8_t* bytes, size_t count) {
  uint64_t value = 0;
  for (size_t index = 0; index < count; ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
  }
  return value;
}

void oracle_absorb(LlmReadChecksumComponent& state, const uint8_t* bytes, size_t count) {
  if (count == 0) {
    return;
  }
  uint64_t span_even = 0;
  uint64_t span_odd = 0;
  for (size_t offset = 0, word_index = 0; offset < count; offset += 8, ++word_index) {
    const size_t word_bytes = std::min<size_t>(8, count - offset);
    const uint64_t word = load_partial_little_endian(bytes + offset, word_bytes);
    if ((word_index & 1U) == 0U) {
      span_even += word;
    } else {
      span_odd += word;
    }
  }
  const uint64_t ordinal = state.span_count;
  state.state_a = rotate_left(state.state_a + span_even + kStepMultiplier * (ordinal + 1), 17);
  state.state_b =
      rotate_left(state.state_b + span_odd + static_cast<uint64_t>(count) + kWordMultiplier * (ordinal + 1), 29);
  state.exact_bytes_read += count;
  ++state.span_count;
}

uint64_t oracle_append_word(uint64_t seed, uint64_t step, uint64_t layer, uint64_t batch, uint64_t word_index,
                            LlmChecksumComponent component) {
  const uint64_t domain = component == LlmChecksumComponent::K ? kAppendKDomain : kAppendVDomain;
  return seed + kStepMultiplier * (step + 1) + kLayerMultiplier * (layer + 1) + kBatchMultiplier * (batch + 1) +
         kWordMultiplier * (word_index + 1) + domain;
}

void oracle_apply_append(std::vector<uint8_t>& visible, size_t local_append_offset, size_t append_bytes,
                         size_t record_byte_offset, uint64_t seed, uint64_t step, uint64_t layer, uint64_t batch,
                         LlmChecksumComponent component) {
  for (size_t byte = 0; byte < append_bytes; ++byte) {
    const size_t canonical_byte = record_byte_offset + byte;
    const uint64_t word = oracle_append_word(seed, step, layer, batch, canonical_byte / 8, component);
    visible[local_append_offset + byte] = static_cast<uint8_t>(word >> (8 * (canonical_byte % 8)));
  }
}

struct OracleWorkerRun {
  LlmWorkerChecksum checksum = oracle_initial_worker();
  std::vector<std::vector<uint8_t>> final_k;
  std::vector<std::vector<uint8_t>> final_v;
};

OracleWorkerRun oracle_worker_run(const LlmLayerDescriptor* layers, size_t layer_count,
                                  const LlmKvSequenceDescriptor* sequences, size_t work_units, uint64_t scenario_flags,
                                  uint64_t scenario_seed) {
  OracleWorkerRun run;
  size_t sequence_count = 0;
  if ((scenario_flags & kLlmScenarioFlagKv) != 0 && sequences != nullptr) {
    for (size_t layer = 0; layer < layer_count; ++layer) {
      sequence_count =
          std::max<size_t>(sequence_count, layers[layer].first_sequence_index + layers[layer].sequence_count);
    }
  }
  run.final_k.resize(sequence_count);
  run.final_v.resize(sequence_count);
  for (size_t sequence = 0; sequence < sequence_count; ++sequence) {
    const LlmKvSequenceDescriptor& descriptor = sequences[sequence];
    if (descriptor.k_visible_bytes != 0) {
      run.final_k[sequence].assign(descriptor.k_visible_ptr, descriptor.k_visible_ptr + descriptor.k_visible_bytes);
    }
    if (descriptor.v_visible_bytes != 0) {
      run.final_v[sequence].assign(descriptor.v_visible_ptr, descriptor.v_visible_ptr + descriptor.v_visible_bytes);
    }
  }

  for (size_t step = 0; step < work_units; ++step) {
    for (size_t layer = 0; layer < layer_count; ++layer) {
      const LlmLayerDescriptor& layer_descriptor = layers[layer];
      if ((scenario_flags & kLlmScenarioFlagWeight) != 0) {
        oracle_absorb(run.checksum.weight, layer_descriptor.weight_ptr, layer_descriptor.weight_bytes);
      }
      if ((scenario_flags & kLlmScenarioFlagKv) == 0) {
        continue;
      }
      for (size_t local = 0; local < layer_descriptor.sequence_count; ++local) {
        const size_t sequence_index = layer_descriptor.first_sequence_index + local;
        const LlmKvSequenceDescriptor& descriptor = sequences[sequence_index];
        const size_t k_append_offset = descriptor.k_append_bytes == 0
                                           ? 0
                                           : static_cast<size_t>(descriptor.k_append_ptr - descriptor.k_visible_ptr);
        const size_t v_append_offset = descriptor.v_append_bytes == 0
                                           ? 0
                                           : static_cast<size_t>(descriptor.v_append_ptr - descriptor.v_visible_ptr);
        oracle_apply_append(run.final_k[sequence_index], k_append_offset, descriptor.k_append_bytes,
                            descriptor.append_record_byte_offset, scenario_seed, step, layer_descriptor.layer_index,
                            descriptor.batch_sequence_index, LlmChecksumComponent::K);
        oracle_apply_append(run.final_v[sequence_index], v_append_offset, descriptor.v_append_bytes,
                            descriptor.append_record_byte_offset, scenario_seed, step, layer_descriptor.layer_index,
                            descriptor.batch_sequence_index, LlmChecksumComponent::V);
        oracle_absorb(run.checksum.k, run.final_k[sequence_index].data(), run.final_k[sequence_index].size());
        oracle_absorb(run.checksum.v, run.final_v[sequence_index].data(), run.final_v[sequence_index].size());
      }
    }
  }
  return run;
}

void expect_component_equal(const LlmReadChecksumComponent& actual, const LlmReadChecksumComponent& expected) {
  EXPECT_EQ(actual.state_a, expected.state_a);
  EXPECT_EQ(actual.state_b, expected.state_b);
  EXPECT_EQ(actual.exact_bytes_read, expected.exact_bytes_read);
  EXPECT_EQ(actual.span_count, expected.span_count);
}

void expect_worker_equal(const LlmWorkerChecksum& actual, const LlmWorkerChecksum& expected) {
  expect_component_equal(actual.weight, expected.weight);
  expect_component_equal(actual.k, expected.k);
  expect_component_equal(actual.v, expected.v);
}

LlmMemoryWorkPlan build_real_executor_plan(const LlmGeometryRequest& geometry, size_t workers) {
  LlmMemoryWorkPlanRequest request;
  request.geometry = geometry;
  request.requested_workers = workers;
  request.available_workers = workers;
  request.available_memory_bytes = 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = static_cast<size_t>(getpagesize());
  request.base_seed = 0x0123456789ABCDEFULL;
  LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  if (!preliminary.valid) {
    return preliminary;
  }
  const LlmExecutorAuxiliaryEstimate auxiliary = calculate_llm_executor_auxiliary_estimate(preliminary);
  if (!auxiliary.valid) {
    return preliminary;
  }
  request.checksum_auxiliary_bytes = auxiliary.checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = auxiliary.orchestration_auxiliary_bytes;
  return build_llm_memory_work_plan(request);
}

struct ManualDescriptorFixture {
  std::array<std::array<uint8_t, 64>, 2> weights{};
  std::array<std::array<uint8_t, 64>, 4> k{};
  std::array<std::array<uint8_t, 64>, 4> v{};
  std::array<LlmLayerDescriptor, 2> layers{};
  std::array<LlmKvSequenceDescriptor, 4> sequences{};

  ManualDescriptorFixture() {
    for (size_t layer = 0; layer < weights.size(); ++layer) {
      for (size_t byte = 0; byte < weights[layer].size(); ++byte) {
        weights[layer][byte] = static_cast<uint8_t>((byte * 17 + layer * 31 + 3) & 0xFF);
      }
    }
    for (size_t sequence = 0; sequence < k.size(); ++sequence) {
      for (size_t byte = 0; byte < k[sequence].size(); ++byte) {
        k[sequence][byte] = static_cast<uint8_t>((byte * 19 + sequence * 23 + 5) & 0xFF);
        v[sequence][byte] = static_cast<uint8_t>((byte * 29 + sequence * 11 + 7) & 0xFF);
      }
    }

    layers[0] = {weights[0].data() + 1, 33, 0, 2, 0, 0};
    layers[1] = {weights[1].data() + 3, 47, 2, 2, 1, 0};
    constexpr std::array<size_t, 4> kVisible = {37, 33, 19, 41};
    constexpr std::array<size_t, 4> kAppendOffset = {20, 16, 8, 27};
    constexpr std::array<size_t, 4> kAppendBytes = {17, 17, 11, 14};
    constexpr std::array<size_t, 4> kRecordOffsets = {0, 3, 5, 1};
    for (size_t index = 0; index < sequences.size(); ++index) {
      sequences[index] = {
          k[index].data(),
          kVisible[index],
          v[index].data(),
          kVisible[index],
          k[index].data() + kAppendOffset[index],
          kAppendBytes[index],
          v[index].data() + kAppendOffset[index],
          kAppendBytes[index],
          index % 2,
          kRecordOffsets[index],
      };
    }
  }
};

}  // namespace

TEST(LlmMemoryKernelIntegrationTest, SafeZeroNullAndInvalidTopLevelBoundariesReturnInitialStates) {
  // A null output is the sole case where no initialized result can be stored.
  llm_decode_memory_asm(nullptr, nullptr, 0, 0, 0, 0, nullptr);

  alignas(16) std::array<uint8_t, 32> data{};
  LlmLayerDescriptor layer{data.data(), data.size(), 0, 1, 0, 0};
  LlmKvSequenceDescriptor sequence{data.data(), data.size(), data.data(), data.size(), data.data(),
                                   0,           data.data(), 0,           0,           0};
  const LlmWorkerChecksum initial = oracle_initial_worker();
  const std::array<std::array<uint64_t, 5>, 6> cases = {{
      {0, 0, 0, 0, 0},
      {reinterpret_cast<uintptr_t>(&layer), 0, 1, 0, 1},
      {0, reinterpret_cast<uintptr_t>(&sequence), 1, 1, 1},
      {reinterpret_cast<uintptr_t>(&layer), 0, 1, 1, 2},
      {reinterpret_cast<uintptr_t>(&layer), reinterpret_cast<uintptr_t>(&sequence), 1, 1, 0},
      {reinterpret_cast<uintptr_t>(&layer), reinterpret_cast<uintptr_t>(&sequence), 1, 1, 4},
  }};
  for (const auto& test_case : cases) {
    LlmWorkerChecksum actual;
    std::memset(&actual, 0xA5, sizeof(actual));
    llm_decode_memory_asm(reinterpret_cast<const LlmLayerDescriptor*>(test_case[0]),
                          reinterpret_cast<const LlmKvSequenceDescriptor*>(test_case[1]), test_case[2], test_case[3],
                          test_case[4], 123, &actual);
    expect_worker_equal(actual, initial);
  }
}

TEST(LlmMemoryKernelIntegrationTest, GuardedExactReadAppendTailsAndMisalignedGoldenBytes) {
  constexpr std::array<uint8_t, 15> kAppendPrefix = {0x97, 0xBC, 0x05, 0x61, 0xE5, 0x54, 0x94, 0x14,
                                                     0x2A, 0xBA, 0x5F, 0xC7, 0x9D, 0x53, 0x7D};
  constexpr std::array<uint8_t, 8> kVAppendFirstWord = {0xA2, 0xC7, 0x10, 0x6C, 0xF0, 0x5F, 0x9F, 0x1F};

  for (size_t size : kExactTailSizes) {
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    GuardedMapping weight(size);
    GuardedMapping k(size);
    GuardedMapping v(size);
    ASSERT_TRUE(weight.valid());
    ASSERT_TRUE(k.valid());
    ASSERT_TRUE(v.valid());
    for (size_t byte = 0; byte < size; ++byte) {
      weight.payload()[byte] = static_cast<uint8_t>((byte * 17 + 3) & 0xFF);
      k.payload()[byte] = static_cast<uint8_t>((byte * 19 + 5) & 0xFF);
      v.payload()[byte] = static_cast<uint8_t>((byte * 23 + 7) & 0xFF);
    }
    const std::vector<uint8_t> weight_prefix(weight.accessible_begin(), weight.payload());
    const std::vector<uint8_t> k_prefix(k.accessible_begin(), k.payload());
    const std::vector<uint8_t> v_prefix(v.accessible_begin(), v.payload());
    const std::vector<uint8_t> weight_before(weight.payload(), weight.payload() + size);

    const uint8_t* weight_pointer = size == 0 ? nullptr : weight.payload();
    uint8_t* k_pointer = size == 0 ? nullptr : k.payload();
    uint8_t* v_pointer = size == 0 ? nullptr : v.payload();
    LlmLayerDescriptor layer{weight_pointer, size, 0, 1, 0, 0};
    LlmKvSequenceDescriptor sequence{k_pointer, size, v_pointer, size, k_pointer, size, v_pointer, size, 0, 0};
    const OracleWorkerRun expected = oracle_worker_run(&layer, 1, &sequence, 1, kLlmScenarioFlagMixed, 0);
    LlmWorkerChecksum actual{};
    llm_decode_memory_asm(&layer, &sequence, 1, 1, kLlmScenarioFlagMixed, 0, &actual);

    expect_worker_equal(actual, expected.checksum);
    EXPECT_TRUE(std::equal(weight_prefix.begin(), weight_prefix.end(), weight.accessible_begin()));
    EXPECT_TRUE(std::equal(k_prefix.begin(), k_prefix.end(), k.accessible_begin()));
    EXPECT_TRUE(std::equal(v_prefix.begin(), v_prefix.end(), v.accessible_begin()));
    EXPECT_TRUE(std::equal(weight_before.begin(), weight_before.end(), weight.payload()));
    if (size != 0) {
      EXPECT_TRUE(std::equal(expected.final_k[0].begin(), expected.final_k[0].end(), k.payload()));
      EXPECT_TRUE(std::equal(expected.final_v[0].begin(), expected.final_v[0].end(), v.payload()));
    }
    if (size >= kAppendPrefix.size()) {
      EXPECT_TRUE(std::equal(kAppendPrefix.begin(), kAppendPrefix.end(), k.payload()));
      EXPECT_TRUE(std::equal(kVAppendFirstWord.begin(), kVAppendFirstWord.end(), v.payload()));
    }
  }

  // This range begins at canonical byte three and is physically misaligned at
  // the end of a writable page. Prefix canaries detect any leading RMW/store.
  GuardedMapping k(11);
  GuardedMapping v(11);
  ASSERT_TRUE(k.valid());
  ASSERT_TRUE(v.valid());
  const std::vector<uint8_t> k_prefix(k.accessible_begin(), k.payload());
  const std::vector<uint8_t> v_prefix(v.accessible_begin(), v.payload());
  LlmLayerDescriptor layer{nullptr, 0, 0, 1, 0, 0};
  LlmKvSequenceDescriptor sequence{k.payload(), 11, v.payload(), 11, k.payload(), 11, v.payload(), 11, 0, 3};
  LlmWorkerChecksum actual{};
  llm_decode_memory_asm(&layer, &sequence, 1, 1, kLlmScenarioFlagKv, 0, &actual);
  constexpr std::array<uint8_t, 11> kMisalignedGolden = {0x61, 0xE5, 0x54, 0x94, 0x14, 0x2A,
                                                         0xBA, 0x5F, 0xC7, 0x9D, 0x53};
  EXPECT_TRUE(std::equal(kMisalignedGolden.begin(), kMisalignedGolden.end(), k.payload()));
  EXPECT_TRUE(std::equal(k_prefix.begin(), k_prefix.end(), k.accessible_begin()));
  EXPECT_TRUE(std::equal(v_prefix.begin(), v_prefix.end(), v.accessible_begin()));
  LlmWorkerChecksum expected = oracle_initial_worker();
  oracle_absorb(expected.k, k.payload(), 11);
  oracle_absorb(expected.v, v.payload(), 11);
  expect_worker_equal(actual, expected);
}

TEST(LlmMemoryKernelIntegrationTest, MultiLayerBatchStepScenariosMatchIndependentOracleAndPreserveHistory) {
  constexpr uint64_t kSeed = 0x0123456789ABCDEFULL;
  for (uint64_t flags : {kLlmScenarioFlagWeight, kLlmScenarioFlagKv, kLlmScenarioFlagMixed}) {
    SCOPED_TRACE(::testing::Message() << "flags=" << flags);
    ManualDescriptorFixture fixture;
    const auto weight_before = fixture.weights;
    const auto k_before = fixture.k;
    const auto v_before = fixture.v;
    const OracleWorkerRun expected =
        oracle_worker_run(fixture.layers.data(), fixture.layers.size(), fixture.sequences.data(), 3, flags, kSeed);
    LlmWorkerChecksum actual{};
    llm_decode_memory_asm(fixture.layers.data(), fixture.sequences.data(), fixture.layers.size(), 3, flags, kSeed,
                          &actual);
    expect_worker_equal(actual, expected.checksum);
    EXPECT_EQ(fixture.weights, weight_before);

    for (size_t sequence = 0; sequence < fixture.sequences.size(); ++sequence) {
      const size_t visible = fixture.sequences[sequence].k_visible_bytes;
      if ((flags & kLlmScenarioFlagKv) != 0) {
        EXPECT_TRUE(std::equal(expected.final_k[sequence].begin(), expected.final_k[sequence].end(),
                               fixture.k[sequence].begin()));
        EXPECT_TRUE(std::equal(expected.final_v[sequence].begin(), expected.final_v[sequence].end(),
                               fixture.v[sequence].begin()));
      } else {
        EXPECT_EQ(fixture.k[sequence], k_before[sequence]);
        EXPECT_EQ(fixture.v[sequence], v_before[sequence]);
      }
      EXPECT_TRUE(std::equal(k_before[sequence].begin() + visible, k_before[sequence].end(),
                             fixture.k[sequence].begin() + visible));
      EXPECT_TRUE(std::equal(v_before[sequence].begin() + visible, v_before[sequence].end(),
                             fixture.v[sequence].begin() + visible));
    }
  }
}

TEST(LlmMemoryKernelIntegrationTest, PreservesIntegerAndFullVectorCalleeSavedRegisters) {
  alignas(64) std::array<uint8_t, 513> weight{};
  alignas(64) std::array<uint8_t, 513> k{};
  alignas(64) std::array<uint8_t, 513> v{};
  for (size_t byte = 0; byte < weight.size(); ++byte) {
    weight[byte] = static_cast<uint8_t>((byte * 17 + 3) & 0xFF);
    k[byte] = static_cast<uint8_t>((byte * 19 + 5) & 0xFF);
    v[byte] = static_cast<uint8_t>((byte * 23 + 7) & 0xFF);
  }
  LlmLayerDescriptor layer{weight.data(), weight.size(), 0, 1, 0, 0};
  LlmKvSequenceDescriptor sequence{k.data(), k.size(), v.data(), v.size(), k.data(),
                                   k.size(), v.data(), v.size(), 0,        0};
  LlmWorkerChecksum output{};
  const std::array<uintptr_t, 7> arguments = {
      reinterpret_cast<uintptr_t>(&layer),  reinterpret_cast<uintptr_t>(&sequence), 1, 1, kLlmScenarioFlagMixed, 0,
      reinterpret_cast<uintptr_t>(&output),
  };
  EXPECT_EQ(verify_llm_kernel_callee_saved_registers_asm(reinterpret_cast<uintptr_t>(&llm_decode_memory_asm),
                                                         arguments.data()),
            1u);
  const OracleWorkerRun expected = oracle_worker_run(&layer, 1, &sequence, 1, kLlmScenarioFlagMixed, 0);
  expect_worker_equal(output, expected.checksum);
}

TEST(LlmMemoryKernelIntegrationTest, OneWorkerProductionExecutorRealAsmSmokeMatchesIndependentOracle) {
  const LlmMemoryWorkPlan plan = build_real_executor_plan({257, 2, 4, 2, 7, 1, 3, 2}, 1);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.effective_workers, 1u);
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared = prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const OracleWorkerRun independent =
      oracle_worker_run(resources.worker_layers(0), plan.layer_descriptors_per_worker, resources.worker_sequences(0),
                        scenario.work_units, kLlmScenarioFlagMixed, scenario.scenario_seed);
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  const LlmExecutorResult result =
      execute_llm_scenario(plan, scenario, resources, *timer, production_llm_kernel_adapter());
  ASSERT_TRUE(result.valid) << result.reason_code;
  EXPECT_EQ(result.reason_code, LlmExecutorReason::VALID);
  EXPECT_TRUE(result.kernel_succeeded);
  EXPECT_TRUE(result.checksum_valid);
  EXPECT_TRUE(std::isfinite(result.elapsed_seconds));
  EXPECT_GT(result.elapsed_seconds, 0.0);
  EXPECT_EQ(result.created_workers, 1u);
  EXPECT_EQ(result.completed_workers, 1u);
  ASSERT_EQ(result.actual_checksums.size(), 1u);
  expect_worker_equal(result.actual_checksums[0], independent.checksum);
  EXPECT_EQ(result.actual_checksums[0].weight.exact_bytes_read, scenario.weight_read_bytes);
  EXPECT_EQ(result.actual_checksums[0].k.exact_bytes_read + result.actual_checksums[0].v.exact_bytes_read,
            scenario.kv_read_bytes);
}

TEST(LlmMemoryKernelIntegrationTest, MultiWorkerProductionExecutorRealAsmCoversExactPlannerRanges) {
  const LlmMemoryWorkPlan plan = build_real_executor_plan({513, 2, 4, 2, 7, 1, 3, 2}, 3);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.effective_workers, 3u);
  LlmExecutionResources resources;
  const LlmResourcePreparationResult prepared = prepare_llm_execution_resources(plan, resources);
  ASSERT_TRUE(prepared.valid) << prepared.reason_code;
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;

  std::vector<OracleWorkerRun> independent;
  independent.reserve(plan.effective_workers);
  for (size_t worker = 0; worker < plan.effective_workers; ++worker) {
    independent.push_back(oracle_worker_run(
        resources.worker_layers(worker), plan.layer_descriptors_per_worker,
        resources.worker_sequences(worker), scenario.work_units,
        kLlmScenarioFlagMixed, scenario.scenario_seed));
  }
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  const LlmExecutorResult result =
      execute_llm_scenario(plan, scenario, resources, *timer, production_llm_kernel_adapter());
  ASSERT_TRUE(result.valid) << result.reason_code;
  EXPECT_TRUE(result.kernel_succeeded);
  EXPECT_TRUE(result.checksum_valid);
  EXPECT_EQ(result.created_workers, plan.effective_workers);
  EXPECT_EQ(result.completed_workers, plan.effective_workers);
  ASSERT_EQ(result.actual_checksums.size(), independent.size());

  uint64_t weight_bytes = 0;
  uint64_t kv_read_bytes = 0;
  for (size_t worker = 0; worker < independent.size(); ++worker) {
    SCOPED_TRACE(::testing::Message() << "worker=" << worker);
    expect_worker_equal(result.actual_checksums[worker], independent[worker].checksum);
    weight_bytes += result.actual_checksums[worker].weight.exact_bytes_read;
    kv_read_bytes +=
        result.actual_checksums[worker].k.exact_bytes_read + result.actual_checksums[worker].v.exact_bytes_read;
  }
  EXPECT_EQ(weight_bytes, scenario.weight_read_bytes);
  EXPECT_EQ(kv_read_bytes, scenario.kv_read_bytes);
}
