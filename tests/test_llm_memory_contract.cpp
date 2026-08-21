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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

constexpr uint64_t kAppendStepMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kAppendLayerMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kAppendBatchMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kAppendWordMultiplier = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kAppendKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kAppendVDomain = 0x5656565656565656ULL;

constexpr uint64_t kChecksumInitialA = 0x243F6A8885A308D3ULL;
constexpr uint64_t kChecksumInitialB = 0x13198A2E03707344ULL;
constexpr uint64_t kChecksumWeightDomain = 0x5745494748545F31ULL;
constexpr uint64_t kChecksumKDomain = 0x4B5F524541445F31ULL;
constexpr uint64_t kChecksumVDomain = 0x565F524541445F31ULL;
constexpr uint64_t kRunInitialA = 0x6A09E667F3BCC909ULL;
constexpr uint64_t kRunInitialB = 0xBB67AE8584CAA73BULL;

enum class ChecksumComponent {
  Weight,
  K,
  V,
};

struct PayloadContract {
  uint64_t weight_read_per_step = 0;
  uint64_t kv_read_per_step = 0;
  uint64_t kv_append_per_step = 0;
  uint64_t kv_only_per_step = 0;
  uint64_t mixed_per_step = 0;
  uint64_t weight_read_total = 0;
  uint64_t kv_read_total = 0;
  uint64_t kv_append_total = 0;
  uint64_t kv_only_total = 0;
  uint64_t mixed_total = 0;
};

struct ReadChecksum {
  uint64_t state_a = 0;
  uint64_t state_b = 0;
  uint64_t exact_bytes_read = 0;
  uint64_t span_count = 0;
};

struct RunChecksum {
  uint64_t state_a = kRunInitialA;
  uint64_t state_b = kRunInitialB;
};

struct ContractIdentity {
  std::string_view mode;
  std::string_view backend;
  int schema_version;
  std::string_view methodology;
  std::string_view descriptor_abi;
  std::string_view append_identity;
  std::string_view checksum_identity;
};

/**
 * @brief Test-side scalar oracle for the frozen Phase 0 payload formulas.
 *
 * Inputs are already resolved exact byte counts. Production work-plan code is
 * intentionally not present in Phase 0; later production tests must compare
 * against the hard-coded values in this file instead of sharing this oracle.
 */
PayloadContract resolve_payload_contract(uint64_t weight_bytes,
                                         uint64_t kv_bytes_per_token,
                                         uint64_t visible_context_tokens,
                                         uint64_t batch_size,
                                         uint64_t steps) {
  PayloadContract result;
  result.weight_read_per_step = weight_bytes;
  result.kv_read_per_step =
      batch_size * visible_context_tokens * kv_bytes_per_token;
  result.kv_append_per_step = batch_size * kv_bytes_per_token;
  result.kv_only_per_step =
      result.kv_read_per_step + result.kv_append_per_step;
  result.mixed_per_step =
      result.weight_read_per_step + result.kv_only_per_step;
  result.weight_read_total = result.weight_read_per_step * steps;
  result.kv_read_total = result.kv_read_per_step * steps;
  result.kv_append_total = result.kv_append_per_step * steps;
  result.kv_only_total = result.kv_only_per_step * steps;
  result.mixed_total = result.mixed_per_step * steps;
  return result;
}

uint64_t rotate_left(uint64_t value, unsigned int shift) {
  return (value << shift) | (value >> (64 - shift));
}

uint64_t append_word(uint64_t base_seed,
                     uint64_t task_local_step,
                     uint64_t layer,
                     uint64_t batch_sequence,
                     uint64_t record_word_index,
                     uint64_t buffer_domain) {
  return base_seed + kAppendStepMultiplier * (task_local_step + 1) +
         kAppendLayerMultiplier * (layer + 1) +
         kAppendBatchMultiplier * (batch_sequence + 1) +
         kAppendWordMultiplier * (record_word_index + 1) + buffer_domain;
}

std::vector<uint8_t> append_byte_range(uint64_t base_seed,
                                       uint64_t task_local_step,
                                       uint64_t layer,
                                       uint64_t batch_sequence,
                                       uint64_t buffer_domain,
                                       size_t record_byte_offset,
                                       size_t byte_count) {
  std::vector<uint8_t> bytes;
  bytes.reserve(byte_count);
  for (size_t byte = 0; byte < byte_count; ++byte) {
    const size_t canonical_byte = record_byte_offset + byte;
    const uint64_t word = append_word(
        base_seed, task_local_step, layer, batch_sequence,
        static_cast<uint64_t>(canonical_byte / sizeof(uint64_t)),
        buffer_domain);
    bytes.push_back(static_cast<uint8_t>(
        word >> (8 * (canonical_byte % sizeof(uint64_t)))));
  }
  return bytes;
}

uint64_t checksum_domain(ChecksumComponent component) {
  switch (component) {
    case ChecksumComponent::Weight:
      return kChecksumWeightDomain;
    case ChecksumComponent::K:
      return kChecksumKDomain;
    case ChecksumComponent::V:
      return kChecksumVDomain;
  }
  return 0;
}

ReadChecksum initial_checksum(ChecksumComponent component) {
  const uint64_t domain = checksum_domain(component);
  return {kChecksumInitialA ^ domain, kChecksumInitialB + domain, 0, 0};
}

uint64_t load_partial_little_endian(const uint8_t* bytes,
                                    size_t byte_count) {
  uint64_t value = 0;
  for (size_t index = 0; index < byte_count; ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
  }
  return value;
}

void absorb_span(ReadChecksum& checksum, const std::vector<uint8_t>& span) {
  if (span.empty()) {
    return;
  }
  uint64_t sum_even_words = 0;
  uint64_t sum_odd_words = 0;
  size_t offset = 0;
  size_t word_index = 0;
  while (offset < span.size()) {
    const size_t remaining = span.size() - offset;
    const size_t word_bytes =
        remaining < sizeof(uint64_t) ? remaining : sizeof(uint64_t);
    const uint64_t word =
        load_partial_little_endian(span.data() + offset, word_bytes);
    if (word_index % 2 == 0) {
      sum_even_words += word;
    } else {
      sum_odd_words += word;
    }
    offset += word_bytes;
    ++word_index;
  }

  const uint64_t ordinal = checksum.span_count;
  checksum.state_a = rotate_left(
      checksum.state_a + sum_even_words +
          kAppendStepMultiplier * (ordinal + 1),
      17);
  checksum.state_b = rotate_left(
      checksum.state_b + sum_odd_words +
          static_cast<uint64_t>(span.size()) +
          kAppendWordMultiplier * (ordinal + 1),
      29);
  checksum.exact_bytes_read += static_cast<uint64_t>(span.size());
  ++checksum.span_count;
}

RunChecksum fold_components(
    const std::vector<ReadChecksum>& canonical_components) {
  RunChecksum run;
  for (size_t ordinal = 0; ordinal < canonical_components.size();
       ++ordinal) {
    const ReadChecksum& component = canonical_components[ordinal];
    run.state_a = rotate_left(
        run.state_a + component.state_a +
            kAppendStepMultiplier * (static_cast<uint64_t>(ordinal) + 1),
        23);
    run.state_b = rotate_left(
        run.state_b + component.state_b + component.exact_bytes_read +
            kAppendWordMultiplier * component.span_count +
            kAppendBatchMultiplier *
                (static_cast<uint64_t>(ordinal) + 1),
        41);
  }
  return run;
}

struct alignas(16) LayerDescriptorAbiV1 {
  const uint8_t* weight_ptr;
  uint64_t weight_bytes;
  uint64_t first_sequence_index;
  uint64_t sequence_count;
  uint64_t layer_index;
  uint64_t reserved_zero;
};

struct alignas(16) KvSequenceDescriptorAbiV1 {
  const uint8_t* k_visible_ptr;
  uint64_t k_visible_bytes;
  const uint8_t* v_visible_ptr;
  uint64_t v_visible_bytes;
  uint8_t* k_append_ptr;
  uint64_t k_append_bytes;
  uint8_t* v_append_ptr;
  uint64_t v_append_bytes;
  uint64_t batch_sequence_index;
  uint64_t append_record_byte_offset;
};

bool matches_frozen_identity(const ContractIdentity& identity) {
  return identity.mode == "llm_memory" && identity.backend == "cpu" &&
         identity.schema_version == 1 &&
         identity.methodology ==
             "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved" &&
         identity.descriptor_abi == "llm-memory-descriptor-abi-v1" &&
         identity.append_identity == "llm-kv-append-affine64-v1" &&
         identity.checksum_identity == "llm-read-checksum-v1";
}

bool accepted_llm_result(std::string_view mode,
                         int schema_version,
                         std::string_view status,
                         bool results_complete,
                         bool conclusions_valid) {
  return mode == "llm_memory" && schema_version == 1 &&
         status == "complete" && results_complete && conclusions_valid;
}

}  // namespace

TEST(LlmMemoryContractTest, ResolvedTrafficFormulaGoldenVectors) {
  constexpr uint64_t gibibyte = 1024ULL * 1024ULL * 1024ULL;
  constexpr uint64_t weight_bytes = 4 * gibibyte;
  constexpr uint64_t layer_count = 32;
  constexpr uint64_t kv_head_count = 8;
  constexpr uint64_t head_dimension = 128;
  constexpr uint64_t kv_element_bytes = 2;
  constexpr uint64_t batch_size = 1;
  constexpr uint64_t visible_context_tokens = 8192;

  const uint64_t kv_vector_bytes = head_dimension * kv_element_bytes;
  const uint64_t kv_record_bytes_per_layer =
      2 * kv_head_count * kv_vector_bytes;
  const uint64_t kv_bytes_per_token =
      layer_count * kv_record_bytes_per_layer;
  const PayloadContract large = resolve_payload_contract(
      weight_bytes, kv_bytes_per_token, visible_context_tokens, batch_size,
      1);

  EXPECT_EQ(kv_vector_bytes, 256u);
  EXPECT_EQ(kv_record_bytes_per_layer, 4096u);
  EXPECT_EQ(kv_bytes_per_token, 131072u);
  EXPECT_EQ(large.kv_read_per_step, gibibyte);
  EXPECT_EQ(large.kv_append_per_step, 131072u);
  EXPECT_EQ(large.kv_only_per_step, 1073872896u);
  EXPECT_EQ(large.mixed_per_step, 5368840192u);
  EXPECT_EQ(weight_bytes, 4294967296u);
  EXPECT_EQ(batch_size * kv_bytes_per_token, 131072u);
  EXPECT_EQ(weight_bytes / (batch_size * kv_bytes_per_token), 32768u);

  const PayloadContract crossover = resolve_payload_contract(
      weight_bytes, kv_bytes_per_token, 32768, batch_size, 1);
  EXPECT_EQ(crossover.kv_read_per_step, weight_bytes);

  const PayloadContract small =
      resolve_payload_contract(1024, 128, 3, 1, 4);
  EXPECT_EQ(small.weight_read_per_step, 1024u);
  EXPECT_EQ(small.kv_read_per_step, 384u);
  EXPECT_EQ(small.kv_append_per_step, 128u);
  EXPECT_EQ(small.kv_only_per_step, 512u);
  EXPECT_EQ(small.mixed_per_step, 1536u);
  EXPECT_EQ(small.weight_read_total, 4096u);
  EXPECT_EQ(small.kv_read_total, 1536u);
  EXPECT_EQ(small.kv_append_total, 512u);
  EXPECT_EQ(small.kv_only_total, 2048u);
  EXPECT_EQ(small.mixed_total, 6144u);

  const PayloadContract batched =
      resolve_payload_contract(1024, 128, 3, 2, 4);
  EXPECT_EQ(batched.weight_read_per_step, 1024u);
  EXPECT_EQ(batched.kv_read_per_step, 768u);
  EXPECT_EQ(batched.kv_append_per_step, 256u);
  EXPECT_EQ(batched.mixed_total, 8192u);
}

TEST(LlmMemoryContractTest,
     AppendAffine64GoldenWordsDomainsWrappingAndTailBytes) {
  EXPECT_EQ(append_word(0, 0, 0, 0, 0, kAppendKDomain),
            0x149454E56105BC97ULL);
  EXPECT_EQ(append_word(0, 0, 0, 0, 0, kAppendVDomain),
            0x1F9F5FF06C10C7A2ULL);
  EXPECT_EQ(append_word(0, 0, 0, 0, 1, kAppendKDomain),
            0xEB7D539DC75FBA2AULL);
  EXPECT_EQ(append_word(std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0,
                        kAppendKDomain),
            0x149454E56105BC96ULL);
  EXPECT_EQ(append_word(0x0123456789ABCDEFULL, 2, 3, 4, 5,
                        kAppendKDomain),
            0x15FD848D8C7B6F66ULL);
  EXPECT_EQ(append_word(0x0123456789ABCDEFULL, 2, 3, 4, 5,
                        kAppendVDomain),
            0x21088F9897867A71ULL);
  EXPECT_EQ(append_word(0, 1, 0, 0, 0, kAppendKDomain),
            0xB2CBCE9EE05038ACULL);
  EXPECT_EQ(append_word(0, 0, 1, 0, 0, kAppendKDomain),
            0xD3EC9C527DEAA250ULL);
  EXPECT_EQ(append_word(0, 0, 0, 1, 0, kAppendKDomain),
            0xA9649EA07436CE82ULL);

  const std::array<uint8_t, 15> expected_prefix = {
      0x97, 0xBC, 0x05, 0x61, 0xE5, 0x54, 0x94, 0x14,
      0x2A, 0xBA, 0x5F, 0xC7, 0x9D, 0x53, 0x7D};
  for (size_t length = 9; length <= expected_prefix.size(); ++length) {
    const std::vector<uint8_t> expected(expected_prefix.begin(),
                                        expected_prefix.begin() + length);
    EXPECT_EQ(append_byte_range(0, 0, 0, 0, kAppendKDomain, 0, length),
              expected)
        << "length=" << length;
  }

  EXPECT_EQ(append_byte_range(0, 0, 0, 0, kAppendKDomain, 3, 11),
            (std::vector<uint8_t>{0x61, 0xE5, 0x54, 0x94, 0x14, 0x2A,
                                  0xBA, 0x5F, 0xC7, 0x9D, 0x53}));
}

TEST(LlmMemoryContractTest, ReadChecksumGoldenStatesAndSpanParity) {
  std::vector<uint8_t> span(19);
  for (size_t index = 0; index < span.size(); ++index) {
    span[index] = static_cast<uint8_t>(index);
  }

  ReadChecksum weight = initial_checksum(ChecksumComponent::Weight);
  EXPECT_EQ(weight.state_a, 0x737A23CFCDF757E2ULL);
  EXPECT_EQ(weight.state_b, 0x6A5ED3754BC4D275ULL);
  absorb_span(weight, {});
  EXPECT_EQ(weight.state_a, 0x737A23CFCDF757E2ULL);
  EXPECT_EQ(weight.state_b, 0x6A5ED3754BC4D275ULL);
  EXPECT_EQ(weight.exact_bytes_read, 0u);
  EXPECT_EQ(weight.span_count, 0u);
  absorb_span(weight, span);
  EXPECT_EQ(weight.state_a, 0x451AA0ABCC0E316FULL);
  EXPECT_EQ(weight.state_b, 0x37A51B246A0ABBE7ULL);
  EXPECT_EQ(weight.exact_bytes_read, 19u);
  EXPECT_EQ(weight.span_count, 1u);

  ReadChecksum k = initial_checksum(ChecksumComponent::K);
  EXPECT_EQ(k.state_a, 0x6F6038CDC4E757E2ULL);
  EXPECT_EQ(k.state_b, 0x5E78DC7344B4D275ULL);
  absorb_span(k, span);
  EXPECT_EQ(k.state_a, 0x6F168E8BCC0E293BULL);
  EXPECT_EQ(k.state_b, 0xF6C31B24688DFD06ULL);

  ReadChecksum v = initial_checksum(ChecksumComponent::V);
  EXPECT_EQ(v.state_a, 0x726038CDC4E757E2ULL);
  EXPECT_EQ(v.state_b, 0x6978DC7344B4D275ULL);
  absorb_span(v, span);
  EXPECT_EQ(v.state_a, 0x6F168E8BCC0E2F3BULL);
  EXPECT_EQ(v.state_b, 0xF6C31B2469EDFD06ULL);

  const std::vector<uint8_t> second_span = {
      0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF};
  absorb_span(weight, second_span);
  EXPECT_EQ(weight.state_a, 0x24378D3C47230311ULL);
  EXPECT_EQ(weight.state_b, 0xA6D7D6E2BCAEE312ULL);
  EXPECT_EQ(weight.exact_bytes_read, 27u);
  EXPECT_EQ(weight.span_count, 2u);
}

TEST(LlmMemoryContractTest, ReadChecksumRepeatedSpansDoNotCancel) {
  const std::vector<uint8_t> span = {1, 2, 3, 4, 5, 6, 7, 8};
  ReadChecksum checksum = initial_checksum(ChecksumComponent::Weight);
  absorb_span(checksum, span);
  EXPECT_EQ(checksum.state_a, 0x471CA289ABF03371ULL);
  EXPECT_EQ(checksum.state_b, 0xB643DA020828FA45ULL);
  absorb_span(checksum, span);
  EXPECT_EQ(checksum.state_a, 0x38035D105B391725ULL);
  EXPECT_EQ(checksum.state_b, 0x5A9B9EAE6C82BAEEULL);
  EXPECT_EQ(checksum.exact_bytes_read, 16u);
  EXPECT_EQ(checksum.span_count, 2u);
}

TEST(LlmMemoryContractTest,
     RunChecksumFoldIncludesEmptyComponentsInCanonicalOrder) {
  std::vector<uint8_t> span(19);
  for (size_t index = 0; index < span.size(); ++index) {
    span[index] = static_cast<uint8_t>(index);
  }
  ReadChecksum weight = initial_checksum(ChecksumComponent::Weight);
  absorb_span(weight, span);
  const ReadChecksum empty_k = initial_checksum(ChecksumComponent::K);
  const ReadChecksum empty_v = initial_checksum(ChecksumComponent::V);

  const RunChecksum folded = fold_components({weight, empty_k, empty_v});
  EXPECT_EQ(folded.state_a, 0xBCA46801BE6585DBULL);
  EXPECT_EQ(folded.state_b, 0xEAAC493C97E06CF7ULL);

  const RunChecksum omitted_empty_components = fold_components({weight});
  EXPECT_NE(omitted_empty_components.state_a, folded.state_a);
  EXPECT_NE(omitted_empty_components.state_b, folded.state_b);
  const RunChecksum wrong_order = fold_components({empty_k, weight, empty_v});
  EXPECT_NE(wrong_order.state_a, folded.state_a);
  EXPECT_NE(wrong_order.state_b, folded.state_b);
}

TEST(LlmMemoryContractTest, DescriptorAbiLayoutGolden) {
  static_assert(sizeof(void*) == 8, "LLM descriptor ABI requires ARM64 pointers");
  static_assert(std::is_standard_layout_v<LayerDescriptorAbiV1>);
  static_assert(std::is_standard_layout_v<KvSequenceDescriptorAbiV1>);

  EXPECT_EQ(alignof(LayerDescriptorAbiV1), 16u);
  EXPECT_EQ(sizeof(LayerDescriptorAbiV1), 48u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, weight_ptr), 0u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, weight_bytes), 8u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, first_sequence_index), 16u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, sequence_count), 24u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, layer_index), 32u);
  EXPECT_EQ(offsetof(LayerDescriptorAbiV1, reserved_zero), 40u);

  EXPECT_EQ(alignof(KvSequenceDescriptorAbiV1), 16u);
  EXPECT_EQ(sizeof(KvSequenceDescriptorAbiV1), 80u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, k_visible_ptr), 0u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, k_visible_bytes), 8u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, v_visible_ptr), 16u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, v_visible_bytes), 24u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, k_append_ptr), 32u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, k_append_bytes), 40u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, v_append_ptr), 48u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, v_append_bytes), 56u);
  EXPECT_EQ(offsetof(KvSequenceDescriptorAbiV1, batch_sequence_index), 64u);
  EXPECT_EQ(
      offsetof(KvSequenceDescriptorAbiV1, append_record_byte_offset), 72u);
}

TEST(LlmMemoryContractTest, IdentityAndCompletionPredicateTruthTable) {
  const ContractIdentity frozen = {
      "llm_memory",
      "cpu",
      1,
      "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved",
      "llm-memory-descriptor-abi-v1",
      "llm-kv-append-affine64-v1",
      "llm-read-checksum-v1",
  };
  EXPECT_TRUE(matches_frozen_identity(frozen));

  ContractIdentity candidate = frozen;
  candidate.mode = "benchmark";
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.backend = "gpu";
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.schema_version = 2;
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.methodology = "llm-memory-v2";
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.descriptor_abi = "llm-memory-descriptor-abi-v2";
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.append_identity = "llm-kv-append-affine64-v2";
  EXPECT_FALSE(matches_frozen_identity(candidate));
  candidate = frozen;
  candidate.checksum_identity = "llm-read-checksum-v2";
  EXPECT_FALSE(matches_frozen_identity(candidate));

  EXPECT_TRUE(accepted_llm_result("llm_memory", 1, "complete", true, true));
  EXPECT_FALSE(
      accepted_llm_result("benchmark", 1, "complete", true, true));
  EXPECT_FALSE(accepted_llm_result("llm_memory", 2, "complete", true,
                                   true));
  EXPECT_FALSE(accepted_llm_result("llm_memory", 1, "partial", true,
                                   true));
  EXPECT_FALSE(accepted_llm_result("llm_memory", 1, "complete", false,
                                   true));
  EXPECT_FALSE(accepted_llm_result("llm_memory", 1, "complete", true,
                                   false));
}
