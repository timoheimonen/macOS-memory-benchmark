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

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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

constexpr uint64_t kKvBlockPermutationDomain = 0x4C4C4D4B56504731ULL;
constexpr uint64_t kCanonicalSegmentCapacityBytes = 256ULL * 1024ULL * 1024ULL;
constexpr size_t kCanonicalSegmentSlotsPerPool = 256;
constexpr uint64_t kCanonicalPoolCapacityBytes =
    kCanonicalSegmentCapacityBytes * kCanonicalSegmentSlotsPerPool;

enum class ContractKvLayout {
  Contiguous,
  Paged,
};

enum class ContractScenario {
  WeightsOnly,
  KvOnly,
  Mixed,
};

struct PrefillPayloadContract {
  std::vector<uint64_t> tile_ends;
  uint64_t attention_prefix_token_visits = 0;
  uint64_t weight_passes = 0;
  uint64_t weight_read_bytes = 0;
  uint64_t kv_read_bytes = 0;
  uint64_t kv_write_bytes = 0;
  uint64_t kv_only_bytes = 0;
  uint64_t mixed_bytes = 0;
};

struct PagedGeometryContract {
  uint64_t blocks_per_sequence = 0;
  uint64_t physical_blocks_per_layer = 0;
  uint64_t block_bytes = 0;
  uint64_t last_block_tokens = 0;
  uint64_t last_block_valid_bytes = 0;
  uint64_t k_logical_bytes = 0;
  uint64_t k_physical_bytes = 0;
  uint64_t k_layout_padding_bytes = 0;
  uint64_t block_table_entries = 0;
  uint64_t block_table_bytes = 0;
  uint64_t decode_lookup_count = 0;
  uint64_t decode_layout_metadata_bytes = 0;
};

struct PrefillClosedFormContract {
  uint64_t tile_count = 0;
  uint64_t attention_prefix_token_visits = 0;
  uint64_t blocks_per_sequence = 0;
  uint64_t prefix_block_visits = 0;
  uint64_t lookups_per_layer_sequence = 0;
};

struct PrefillBlockContract {
  uint64_t valid_tokens = 0;
  uint64_t data_visits = 0;
  uint64_t semantic_lookups = 0;
  uint64_t model_payload_bytes = 0;
  uint64_t layout_metadata_bytes = 0;
  uint64_t accounted_bytes = 0;
};

struct ComponentIdentityContract {
  std::string_view logical_profile_version;
  std::string_view kv_layout_version;
  std::optional<std::string_view> permutation_version;
  std::string_view backend_executor_version;
  std::string_view resource_abi_version;
  std::string_view schedule_version;
  std::string_view timer_policy_version;
  std::string_view buffer_pattern_version;
  std::string_view write_pattern_version;
  std::string_view checksum_pattern_version;
  std::optional<std::string_view> msl_revision;
  std::optional<std::string_view> msl_source_sha256;
};

struct ScenarioAccountingContract {
  uint64_t model_payload_bytes = 0;
  uint64_t layout_metadata_lookups = 0;
  uint64_t layout_metadata_bytes = 0;
  uint64_t accounted_bytes = 0;
};

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


struct GenericResultIdentity {
  std::string_view mode;
  int schema_version;
  std::string_view requested_backend;
  std::string_view backend;
  std::string_view requested_phase;
  std::string_view phase;
  std::string_view requested_kv_layout;
  std::string_view kv_layout;
  std::string_view methodology_version;
  std::string_view status;
  bool results_complete;
  bool run_accepted;
  bool all_planned_measurements_measured;
};

/**
 * @brief Test-side scalar oracle for the original decode payload contract.
 *
 * Inputs are already resolved exact byte counts. Expansion production code
 * must compare against the hard-coded values in this file instead of sharing
 * this oracle.
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

uint64_t ceil_divide_small(uint64_t value, uint64_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

uint64_t triangular_small(uint64_t value) {
  return value % 2 == 0 ? (value / 2) * (value + 1)
                        : value * (value / 2 + 1);
}

std::vector<uint64_t> enumerate_tile_ends(uint64_t prompt_tokens,
                                          uint64_t query_tile_tokens) {
  std::vector<uint64_t> ends;
  if (query_tile_tokens == 0) {
    return ends;
  }
  uint64_t current_end = 0;
  while (current_end < prompt_tokens) {
    current_end +=
        std::min(query_tile_tokens, prompt_tokens - current_end);
    ends.push_back(current_end);
  }
  return ends;
}

PrefillPayloadContract resolve_prefill_payload_contract(
    uint64_t weight_bytes,
    uint64_t kv_bytes_per_token,
    uint64_t batch_size,
    uint64_t prompt_tokens,
    uint64_t query_tile_tokens) {
  PrefillPayloadContract result;
  result.tile_ends =
      enumerate_tile_ends(prompt_tokens, query_tile_tokens);
  result.attention_prefix_token_visits =
      std::accumulate(result.tile_ends.begin(), result.tile_ends.end(),
                      uint64_t{0});
  result.weight_passes = 1;
  result.weight_read_bytes = weight_bytes;
  result.kv_read_bytes = batch_size *
                         result.attention_prefix_token_visits *
                         kv_bytes_per_token;
  result.kv_write_bytes =
      batch_size * prompt_tokens * kv_bytes_per_token;
  result.kv_only_bytes = result.kv_read_bytes + result.kv_write_bytes;
  result.mixed_bytes = result.weight_read_bytes + result.kv_only_bytes;
  return result;
}

PagedGeometryContract resolve_paged_geometry_contract(
    uint64_t sequence_tokens,
    uint64_t block_tokens,
    uint64_t layer_count,
    uint64_t batch_size,
    uint64_t k_or_v_record_bytes_per_layer) {
  PagedGeometryContract result;
  result.blocks_per_sequence =
      ceil_divide_small(sequence_tokens, block_tokens);
  result.physical_blocks_per_layer =
      batch_size * result.blocks_per_sequence;
  result.block_bytes = block_tokens * k_or_v_record_bytes_per_layer;
  result.last_block_tokens =
      sequence_tokens - (result.blocks_per_sequence - 1) * block_tokens;
  result.last_block_valid_bytes =
      result.last_block_tokens * k_or_v_record_bytes_per_layer;
  result.k_logical_bytes = layer_count * batch_size * sequence_tokens *
                           k_or_v_record_bytes_per_layer;
  result.k_physical_bytes = layer_count * result.physical_blocks_per_layer *
                            result.block_bytes;
  result.k_layout_padding_bytes =
      result.k_physical_bytes - result.k_logical_bytes;
  result.block_table_entries = result.physical_blocks_per_layer;
  result.block_table_bytes = result.block_table_entries * sizeof(uint32_t);
  result.decode_lookup_count = layer_count * batch_size *
                               (2 * result.blocks_per_sequence + 1);
  result.decode_layout_metadata_bytes =
      result.decode_lookup_count * sizeof(uint32_t);
  return result;
}

ScenarioAccountingContract resolve_scenario_accounting_contract(
    ContractKvLayout layout,
    ContractScenario scenario,
    uint64_t weight_payload_bytes,
    uint64_t kv_payload_bytes,
    uint64_t paged_layout_metadata_lookups) {
  ScenarioAccountingContract result;
  if (scenario == ContractScenario::WeightsOnly) {
    result.model_payload_bytes = weight_payload_bytes;
  } else {
    result.model_payload_bytes = kv_payload_bytes;
    if (scenario == ContractScenario::Mixed) {
      result.model_payload_bytes += weight_payload_bytes;
    }
    if (layout == ContractKvLayout::Paged) {
      result.layout_metadata_lookups = paged_layout_metadata_lookups;
    }
  }
  result.layout_metadata_bytes =
      result.layout_metadata_lookups * sizeof(uint32_t);
  result.accounted_bytes =
      result.model_payload_bytes + result.layout_metadata_bytes;
  return result;
}

std::vector<uint64_t> enumerate_prefix_block_counts(
    uint64_t prompt_tokens,
    uint64_t query_tile_tokens,
    uint64_t block_tokens) {
  std::vector<uint64_t> counts;
  for (uint64_t end :
       enumerate_tile_ends(prompt_tokens, query_tile_tokens)) {
    counts.push_back(ceil_divide_small(end, block_tokens));
  }
  return counts;
}

uint64_t floor_sum_small(uint64_t count,
                         uint64_t denominator,
                         uint64_t slope,
                         uint64_t intercept) {
  uint64_t result = 0;
  for (uint64_t index = 0; index < count; ++index) {
    result += (slope * index + intercept) / denominator;
  }
  return result;
}

PrefillClosedFormContract resolve_prefill_closed_form_contract(
    uint64_t prompt_tokens,
    uint64_t query_tile_tokens,
    uint64_t block_tokens) {
  PrefillClosedFormContract result;
  const uint64_t full_tile_count = prompt_tokens / query_tile_tokens;
  const uint64_t remainder = prompt_tokens % query_tile_tokens;
  result.tile_count =
      full_tile_count + (remainder != 0 ? 1 : 0);
  result.attention_prefix_token_visits =
      query_tile_tokens * triangular_small(full_tile_count) +
      (remainder != 0 ? prompt_tokens : 0);
  result.blocks_per_sequence =
      ceil_divide_small(prompt_tokens, block_tokens);
  const uint64_t full_tile_block_visits =
      full_tile_count +
      floor_sum_small(full_tile_count, block_tokens, query_tile_tokens,
                      query_tile_tokens - 1);
  result.prefix_block_visits =
      full_tile_block_visits +
      (remainder != 0 ? result.blocks_per_sequence : 0);
  result.lookups_per_layer_sequence =
      result.blocks_per_sequence + 2 * result.prefix_block_visits;
  return result;
}

std::vector<PrefillBlockContract> enumerate_prefill_block_contracts(
    uint64_t prompt_tokens,
    uint64_t query_tile_tokens,
    uint64_t block_tokens,
    uint64_t k_or_v_record_bytes_per_layer) {
  const std::vector<uint64_t> tile_ends =
      enumerate_tile_ends(prompt_tokens, query_tile_tokens);
  const uint64_t block_count =
      ceil_divide_small(prompt_tokens, block_tokens);
  std::vector<PrefillBlockContract> blocks;
  blocks.reserve(static_cast<size_t>(block_count));
  for (uint64_t block = 0; block < block_count; ++block) {
    PrefillBlockContract contract;
    const uint64_t block_start = block * block_tokens;
    contract.valid_tokens =
        std::min(block_tokens, prompt_tokens - block_start);
    contract.data_visits = contract.valid_tokens;
    uint64_t read_visits = 0;
    for (uint64_t token = block_start;
         token < block_start + contract.valid_tokens; ++token) {
      for (uint64_t end : tile_ends) {
        if (token < end) {
          ++contract.data_visits;
        }
      }
    }
    for (uint64_t end : tile_ends) {
      if (block_start < end) {
        ++read_visits;
      }
    }
    contract.semantic_lookups = 1 + 2 * read_visits;
    contract.model_payload_bytes =
        2 * k_or_v_record_bytes_per_layer * contract.data_visits;
    contract.layout_metadata_bytes =
        sizeof(uint32_t) * contract.semantic_lookups;
    contract.accounted_bytes = contract.model_payload_bytes +
                               contract.layout_metadata_bytes;
    blocks.push_back(contract);
  }
  return blocks;
}

size_t choose_two_way_cost_boundary(
    const std::vector<PrefillBlockContract>& blocks) {
  if (blocks.size() < 2) {
    return blocks.size();
  }
  uint64_t total = 0;
  for (const PrefillBlockContract& block : blocks) {
    total += block.accounted_bytes;
  }

  size_t best_boundary = 1;
  uint64_t prefix = 0;
  uint64_t best_scaled_distance = std::numeric_limits<uint64_t>::max();
  for (size_t boundary = 1; boundary < blocks.size(); ++boundary) {
    prefix += blocks[boundary - 1].accounted_bytes;
    const uint64_t doubled_prefix = 2 * prefix;
    const uint64_t distance = doubled_prefix > total
                                  ? doubled_prefix - total
                                  : total - doubled_prefix;
    if (distance < best_scaled_distance) {
      best_scaled_distance = distance;
      best_boundary = boundary;
    }
  }
  return best_boundary;
}

std::vector<std::vector<uint64_t>> enumerate_terminal_visit_tokens(
    uint64_t prompt_tokens,
    uint64_t query_tile_tokens,
    uint64_t block_tokens) {
  const uint64_t block_count =
      ceil_divide_small(prompt_tokens, block_tokens);
  std::vector<std::vector<uint64_t>> visits;
  for (uint64_t end :
       enumerate_tile_ends(prompt_tokens, query_tile_tokens)) {
    std::vector<uint64_t> tile_visits;
    tile_visits.reserve(static_cast<size_t>(block_count));
    for (uint64_t block = 0; block < block_count; ++block) {
      const uint64_t block_start = block * block_tokens;
      const uint64_t visit_tokens =
          block_start >= end
              ? 0
              : std::min(block_tokens, end - block_start);
      tile_visits.push_back(visit_tokens);
    }
    visits.push_back(std::move(tile_visits));
  }
  return visits;
}

uint64_t splitmix64_next(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

uint64_t derive_permutation_seed(uint64_t base_seed) {
  uint64_t state = base_seed ^ kKvBlockPermutationDomain;
  return splitmix64_next(state);
}

std::vector<uint32_t> materialize_permutation(size_t entry_count,
                                              uint64_t stream_state) {
  std::vector<uint32_t> entries(entry_count);
  std::iota(entries.begin(), entries.end(), uint32_t{0});
  for (size_t index = entry_count; index > 1; --index) {
    const uint64_t bound = static_cast<uint64_t>(index);
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t draw = 0;
    do {
      draw = splitmix64_next(stream_state);
    } while (draw < threshold);
    const size_t swap_index = static_cast<size_t>(draw % bound);
    std::swap(entries[index - 1], entries[swap_index]);
  }
  return entries;
}

std::string sha256_little_endian_entries(
    const std::vector<uint32_t>& entries,
    size_t entries_per_chunk) {
  if (entries_per_chunk == 0) {
    return {};
  }

  CC_SHA256_CTX context;
  if (CC_SHA256_Init(&context) != 1) {
    return {};
  }
  constexpr size_t kEntriesPerUpdateCap = 1024;
  std::array<unsigned char,
             kEntriesPerUpdateCap * sizeof(uint32_t)>
      bytes{};
  size_t offset = 0;
  while (offset < entries.size()) {
    const size_t count =
        std::min({entries_per_chunk, entries.size() - offset,
                  kEntriesPerUpdateCap});
    for (size_t index = 0; index < count; ++index) {
      const uint32_t value = entries[offset + index];
      for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
        bytes[index * sizeof(uint32_t) + byte] =
            static_cast<unsigned char>(value >> (8 * byte));
      }
    }
    const size_t byte_count = count * sizeof(uint32_t);
    if (CC_SHA256_Update(&context, bytes.data(),
                         static_cast<CC_LONG>(byte_count)) != 1) {
      return {};
    }
    offset += count;
  }

  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (CC_SHA256_Final(digest.data(), &context) != 1) {
    return {};
  }
  constexpr char kLowercaseHex[] = "0123456789abcdef";
  std::string encoded(digest.size() * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    encoded[index * 2] = kLowercaseHex[digest[index] >> 4U];
    encoded[index * 2 + 1] =
        kLowercaseHex[digest[index] & 0x0fU];
  }
  return encoded;
}

std::vector<uint64_t> contiguous_segment_lengths(uint64_t logical_bytes) {
  std::vector<uint64_t> lengths;
  uint64_t remaining = logical_bytes;
  while (remaining > 0) {
    const uint64_t length =
        std::min(remaining, kCanonicalSegmentCapacityBytes);
    lengths.push_back(length);
    remaining -= length;
  }
  return lengths;
}

bool paged_block_fits_canonical_segment(uint64_t block_bytes) {
  return block_bytes <= kCanonicalSegmentCapacityBytes;
}

bool segment_count_fits_pool_slots(uint64_t segment_count) {
  return segment_count <= kCanonicalSegmentSlotsPerPool;
}

void append_identity_component(std::string& identity,
                               std::string_view key,
                               std::optional<std::string_view> value) {
  identity += '|';
  identity += key;
  identity += '=';
  if (!value.has_value()) {
    identity += "null";
    return;
  }
  identity += std::to_string(value->size());
  identity += ':';
  identity += *value;
}

void append_identity_component(std::string& identity,
                               std::string_view key,
                               std::string_view value) {
  identity += '|';
  identity += key;
  identity += '=';
  identity += std::to_string(value.size());
  identity += ':';
  identity += value;
}

std::string serialize_component_identity(
    const ComponentIdentityContract& components) {
  std::string identity = "llm-memory-components-v1";
  append_identity_component(identity, "logical_profile_version",
                            components.logical_profile_version);
  append_identity_component(identity, "kv_layout_version",
                            components.kv_layout_version);
  append_identity_component(identity, "permutation_version",
                            components.permutation_version);
  append_identity_component(identity, "backend_executor_version",
                            components.backend_executor_version);
  append_identity_component(identity, "resource_abi_version",
                            components.resource_abi_version);
  append_identity_component(identity, "schedule_version",
                            components.schedule_version);
  append_identity_component(identity, "timer_policy_version",
                            components.timer_policy_version);
  append_identity_component(identity, "buffer_pattern_version",
                            components.buffer_pattern_version);
  append_identity_component(identity, "write_pattern_version",
                            components.write_pattern_version);
  append_identity_component(identity, "checksum_pattern_version",
                            components.checksum_pattern_version);
  append_identity_component(identity, "msl_revision",
                            components.msl_revision);
  append_identity_component(identity, "msl_source_sha256",
                            components.msl_source_sha256);
  return identity;
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

bool accepted_generic_result(const GenericResultIdentity& identity) {
  const bool backend_is_known =
      identity.backend == "cpu" || identity.backend == "metal";
  const bool phase_is_known =
      identity.phase == "decode" || identity.phase == "prefill";
  const bool layout_is_known = identity.kv_layout == "contiguous" ||
                               identity.kv_layout == "paged";
  const std::string expected_methodology =
      "llm-memory-v2-" + std::string(identity.backend) + "-" +
      std::string(identity.phase) + "-" + std::string(identity.kv_layout);
  return identity.mode == "llm_memory" && identity.schema_version == 2 &&
         backend_is_known && phase_is_known && layout_is_known &&
         identity.backend == identity.requested_backend &&
         identity.phase == identity.requested_phase &&
         identity.kv_layout == identity.requested_kv_layout &&
         identity.methodology_version == expected_methodology &&
         identity.status == "complete" && identity.results_complete &&
         identity.run_accepted &&
         identity.all_planned_measurements_measured;
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

TEST(LlmMemoryContractTest, GenericV2AcceptancePredicateIsExact) {
  const GenericResultIdentity accepted = {
      "llm_memory",
      2,
      "metal",
      "metal",
      "prefill",
      "prefill",
      "paged",
      "paged",
      "llm-memory-v2-metal-prefill-paged",
      "complete",
      true,
      true,
      true,
  };
  EXPECT_TRUE(accepted_generic_result(accepted));

  GenericResultIdentity candidate = accepted;
  candidate.mode = "benchmark";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.schema_version = 1;
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.backend = "cpu";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.phase = "decode";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.kv_layout = "contiguous";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.requested_backend = "gpu";
  candidate.backend = "gpu";
  candidate.methodology_version = "llm-memory-v2-gpu-prefill-paged";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.requested_phase = "train";
  candidate.phase = "train";
  candidate.methodology_version = "llm-memory-v2-metal-train-paged";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.requested_kv_layout = "sparse";
  candidate.kv_layout = "sparse";
  candidate.methodology_version = "llm-memory-v2-metal-prefill-sparse";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.methodology_version =
      "llm-memory-v2-cpu-fixed-context-warm-layer-interleaved";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.status = "partial";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.results_complete = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.run_accepted = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.all_planned_measurements_measured = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
}

TEST(LlmMemoryContractTest,
     PrefillPayloadGoldenCoversTilesBoundsAndSingleWeightPass) {
  const PrefillPayloadContract golden =
      resolve_prefill_payload_contract(1024, 128, 2, 5, 2);
  EXPECT_EQ(golden.tile_ends,
            (std::vector<uint64_t>{2, 4, 5}));
  EXPECT_EQ(golden.attention_prefix_token_visits, 11u);
  EXPECT_EQ(golden.weight_passes, 1u);
  EXPECT_EQ(golden.weight_read_bytes, 1024u);
  EXPECT_EQ(golden.kv_read_bytes, 2816u);
  EXPECT_EQ(golden.kv_write_bytes, 1280u);
  EXPECT_EQ(golden.kv_only_bytes, 4096u);
  EXPECT_EQ(golden.mixed_bytes, 5120u);

  const PrefillPayloadContract token_tiled =
      resolve_prefill_payload_contract(1024, 128, 2, 5, 1);
  EXPECT_EQ(token_tiled.tile_ends,
            (std::vector<uint64_t>{1, 2, 3, 4, 5}));
  EXPECT_EQ(token_tiled.attention_prefix_token_visits, 15u);
  EXPECT_EQ(token_tiled.kv_read_bytes, 3840u);
  EXPECT_EQ(token_tiled.kv_write_bytes, 1280u);
  EXPECT_EQ(token_tiled.kv_only_bytes, 5120u);
  EXPECT_EQ(token_tiled.mixed_bytes, 6144u);

  const PrefillPayloadContract full_prompt_tiled =
      resolve_prefill_payload_contract(1024, 128, 2, 5, 5);
  EXPECT_EQ(full_prompt_tiled.tile_ends,
            (std::vector<uint64_t>{5}));
  EXPECT_EQ(full_prompt_tiled.attention_prefix_token_visits, 5u);
  EXPECT_EQ(full_prompt_tiled.kv_read_bytes, 1280u);
  EXPECT_EQ(full_prompt_tiled.kv_write_bytes, 1280u);
  EXPECT_EQ(full_prompt_tiled.kv_only_bytes, 2560u);
  EXPECT_EQ(full_prompt_tiled.mixed_bytes, 3584u);

  struct WeightPassCase {
    uint64_t prompt_tokens;
    uint64_t query_tile_tokens;
    uint64_t batch_size;
  };
  constexpr std::array<WeightPassCase, 5> cases = {{{1, 1, 1},
                                                     {5, 1, 1},
                                                     {5, 5, 4},
                                                     {17, 4, 2},
                                                     {31, 8, 7}}};
  for (const WeightPassCase& test_case : cases) {
    const PrefillPayloadContract result = resolve_prefill_payload_contract(
        987, 64, test_case.batch_size, test_case.prompt_tokens,
        test_case.query_tile_tokens);
    EXPECT_EQ(result.weight_passes, 1u);
    EXPECT_EQ(result.weight_read_bytes, 987u);
  }
}

TEST(LlmMemoryContractTest,
     PagedGeometryDecodeLookupAndMetadataGoldenIsExact) {
  const PagedGeometryContract paged = resolve_paged_geometry_contract(
      35, 16, 2, 2, 32);
  EXPECT_EQ(paged.blocks_per_sequence, 3u);
  EXPECT_EQ(paged.physical_blocks_per_layer, 6u);
  EXPECT_EQ(paged.block_bytes, 512u);
  EXPECT_EQ(paged.last_block_tokens, 3u);
  EXPECT_EQ(paged.last_block_valid_bytes, 96u);
  EXPECT_EQ(paged.k_logical_bytes, 4480u);
  EXPECT_EQ(paged.k_physical_bytes, 6144u);
  EXPECT_EQ(paged.k_layout_padding_bytes, 1664u);
  EXPECT_EQ(paged.block_table_entries, 6u);
  EXPECT_EQ(paged.block_table_bytes, 24u);
  EXPECT_EQ(paged.decode_lookup_count, 28u);
  EXPECT_EQ(paged.decode_layout_metadata_bytes, 112u);

  constexpr uint64_t weight_read_bytes = 1024;
  constexpr uint64_t kv_bytes_per_token = 2 * 2 * 32;
  constexpr uint64_t kv_read_bytes = 2 * 35 * kv_bytes_per_token;
  constexpr uint64_t kv_write_bytes = 2 * kv_bytes_per_token;
  constexpr uint64_t kv_only_model_payload =
      kv_read_bytes + kv_write_bytes;
  constexpr uint64_t mixed_model_payload =
      weight_read_bytes + kv_only_model_payload;
  EXPECT_EQ(kv_bytes_per_token, 128u);
  EXPECT_EQ(kv_read_bytes, 8960u);
  EXPECT_EQ(kv_write_bytes, 256u);
  EXPECT_EQ(kv_only_model_payload, 9216u);
  EXPECT_EQ(mixed_model_payload, 10240u);

  const ScenarioAccountingContract contiguous_weights =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Contiguous, ContractScenario::WeightsOnly,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);
  const ScenarioAccountingContract contiguous_kv =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Contiguous, ContractScenario::KvOnly,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);
  const ScenarioAccountingContract contiguous_mixed =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Contiguous, ContractScenario::Mixed,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);
  const ScenarioAccountingContract paged_weights =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Paged, ContractScenario::WeightsOnly,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);
  const ScenarioAccountingContract paged_kv =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Paged, ContractScenario::KvOnly,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);
  const ScenarioAccountingContract paged_mixed =
      resolve_scenario_accounting_contract(
          ContractKvLayout::Paged, ContractScenario::Mixed,
          weight_read_bytes, kv_only_model_payload,
          paged.decode_lookup_count);

  EXPECT_EQ(contiguous_weights.model_payload_bytes, 1024u);
  EXPECT_EQ(contiguous_weights.layout_metadata_lookups, 0u);
  EXPECT_EQ(contiguous_weights.layout_metadata_bytes, 0u);
  EXPECT_EQ(contiguous_weights.accounted_bytes, 1024u);
  EXPECT_EQ(contiguous_kv.model_payload_bytes, 9216u);
  EXPECT_EQ(contiguous_kv.layout_metadata_bytes, 0u);
  EXPECT_EQ(contiguous_kv.accounted_bytes, 9216u);
  EXPECT_EQ(contiguous_mixed.model_payload_bytes, 10240u);
  EXPECT_EQ(contiguous_mixed.layout_metadata_bytes, 0u);
  EXPECT_EQ(contiguous_mixed.accounted_bytes, 10240u);
  EXPECT_EQ(paged_weights.model_payload_bytes, 1024u);
  EXPECT_EQ(paged_weights.layout_metadata_lookups, 0u);
  EXPECT_EQ(paged_weights.layout_metadata_bytes, 0u);
  EXPECT_EQ(paged_weights.accounted_bytes, 1024u);
  EXPECT_EQ(paged_kv.model_payload_bytes, 9216u);
  EXPECT_EQ(paged_kv.layout_metadata_lookups, 28u);
  EXPECT_EQ(paged_kv.layout_metadata_bytes, 112u);
  EXPECT_EQ(paged_kv.accounted_bytes, 9328u);
  EXPECT_EQ(paged_mixed.model_payload_bytes, 10240u);
  EXPECT_EQ(paged_mixed.layout_metadata_lookups, 28u);
  EXPECT_EQ(paged_mixed.layout_metadata_bytes, 112u);
  EXPECT_EQ(paged_mixed.accounted_bytes, 10352u);

  constexpr uint64_t guardrail_bytes =
      64ULL * 1024ULL * 1024ULL * 1024ULL;
  EXPECT_EQ(guardrail_bytes / contiguous_weights.accounted_bytes,
            67108864u);
  EXPECT_EQ(guardrail_bytes / contiguous_kv.accounted_bytes, 7456540u);
  EXPECT_EQ(guardrail_bytes / paged_kv.accounted_bytes, 7367010u);
  EXPECT_EQ(guardrail_bytes / contiguous_mixed.accounted_bytes, 6710886u);
  EXPECT_EQ(guardrail_bytes / paged_mixed.accounted_bytes, 6638280u);
  EXPECT_NE(paged.block_table_bytes,
            paged.decode_layout_metadata_bytes);
}

TEST(LlmMemoryContractTest,
     BlockPermutationGoldenVectorsAndLittleEndianHashesAreFrozen) {
  const std::vector<uint32_t> direct = materialize_permutation(8, 0);
  EXPECT_EQ(direct,
            (std::vector<uint32_t>{2, 5, 0, 3, 4, 6, 1, 7}));
  constexpr std::string_view direct_hash =
      "9d1cfab79005723a285fec9a5716b53baa7a6c0501e3d17434bfb31ea88935d1";
  EXPECT_EQ(sha256_little_endian_entries(direct, 1), direct_hash);
  EXPECT_EQ(sha256_little_endian_entries(direct, 3), direct_hash);
  EXPECT_EQ(sha256_little_endian_entries(direct, 8), direct_hash);
  EXPECT_EQ(sha256_little_endian_entries(
                direct, std::numeric_limits<size_t>::max()),
            direct_hash);

  EXPECT_EQ(kKvBlockPermutationDomain, 0x4C4C4D4B56504731ULL);
  EXPECT_EQ(derive_permutation_seed(42), 8109369757063363730ULL);
  const std::vector<uint32_t> derived =
      materialize_permutation(8, derive_permutation_seed(42));
  EXPECT_EQ(derived,
            (std::vector<uint32_t>{0, 6, 2, 3, 7, 1, 5, 4}));
  constexpr std::string_view derived_hash =
      "4032b29a855010d82199c15c3f3e2b94582b86e67b3add8cb86bebc425f9c2b4";
  EXPECT_EQ(sha256_little_endian_entries(derived, 1), derived_hash);
  EXPECT_EQ(sha256_little_endian_entries(derived, 3), derived_hash);
  EXPECT_EQ(sha256_little_endian_entries(derived, 8), derived_hash);
  EXPECT_EQ(sha256_little_endian_entries(
                derived, std::numeric_limits<size_t>::max()),
            derived_hash);
}

TEST(LlmMemoryContractTest,
     PrefillPagedLookupAndTerminalVisitGoldensAreExact) {
  const PrefillClosedFormContract aligned =
      resolve_prefill_closed_form_contract(5, 2, 2);
  EXPECT_EQ(enumerate_tile_ends(5, 2),
            (std::vector<uint64_t>{2, 4, 5}));
  EXPECT_EQ(enumerate_prefix_block_counts(5, 2, 2),
            (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(aligned.tile_count, 3u);
  EXPECT_EQ(aligned.attention_prefix_token_visits, 11u);
  EXPECT_EQ(aligned.blocks_per_sequence, 3u);
  EXPECT_EQ(aligned.prefix_block_visits, 6u);
  EXPECT_EQ(aligned.lookups_per_layer_sequence, 15u);
  const std::vector<PrefillBlockContract> aligned_blocks =
      enumerate_prefill_block_contracts(5, 2, 2, 32);
  ASSERT_EQ(aligned_blocks.size(), 3u);
  EXPECT_EQ(aligned_blocks[0].model_payload_bytes, 512u);
  EXPECT_EQ(aligned_blocks[0].layout_metadata_bytes, 28u);
  EXPECT_EQ(aligned_blocks[0].accounted_bytes, 540u);
  EXPECT_EQ(aligned_blocks[1].model_payload_bytes, 384u);
  EXPECT_EQ(aligned_blocks[1].layout_metadata_bytes, 20u);
  EXPECT_EQ(aligned_blocks[1].accounted_bytes, 404u);
  EXPECT_EQ(aligned_blocks[2].model_payload_bytes, 128u);
  EXPECT_EQ(aligned_blocks[2].layout_metadata_bytes, 12u);
  EXPECT_EQ(aligned_blocks[2].accounted_bytes, 140u);
  const uint64_t aligned_model_payload = std::accumulate(
      aligned_blocks.begin(), aligned_blocks.end(), uint64_t{0},
      [](uint64_t total, const PrefillBlockContract& block) {
        return total + block.model_payload_bytes;
      });
  const uint64_t aligned_metadata = std::accumulate(
      aligned_blocks.begin(), aligned_blocks.end(), uint64_t{0},
      [](uint64_t total, const PrefillBlockContract& block) {
        return total + block.layout_metadata_bytes;
      });
  EXPECT_EQ(aligned_model_payload * 2 * 2, 4096u);
  EXPECT_EQ(aligned_metadata * 2 * 2, 240u);
  EXPECT_EQ((aligned_model_payload + aligned_metadata) * 2 * 2, 4336u);
  ASSERT_EQ(choose_two_way_cost_boundary(aligned_blocks), 1u);
  EXPECT_EQ(aligned_blocks[0].accounted_bytes, 540u);
  EXPECT_EQ(aligned_blocks[1].accounted_bytes +
                aligned_blocks[2].accounted_bytes,
            544u);

  const PrefillClosedFormContract unaligned =
      resolve_prefill_closed_form_contract(7, 3, 2);
  EXPECT_EQ(enumerate_tile_ends(7, 3),
            (std::vector<uint64_t>{3, 6, 7}));
  EXPECT_EQ(enumerate_prefix_block_counts(7, 3, 2),
            (std::vector<uint64_t>{2, 3, 4}));
  EXPECT_EQ(unaligned.tile_count, 3u);
  EXPECT_EQ(unaligned.attention_prefix_token_visits, 16u);
  EXPECT_EQ(unaligned.blocks_per_sequence, 4u);
  EXPECT_EQ(unaligned.prefix_block_visits, 9u);
  EXPECT_EQ(unaligned.lookups_per_layer_sequence, 22u);
  const std::vector<PrefillBlockContract> unaligned_blocks =
      enumerate_prefill_block_contracts(7, 3, 2, 32);
  ASSERT_EQ(unaligned_blocks.size(), 4u);
  EXPECT_EQ(unaligned_blocks[0].accounted_bytes, 540u);
  EXPECT_EQ(unaligned_blocks[1].accounted_bytes, 476u);
  EXPECT_EQ(unaligned_blocks[2].accounted_bytes, 404u);
  EXPECT_EQ(unaligned_blocks[3].accounted_bytes, 140u);
  EXPECT_EQ(choose_two_way_cost_boundary(unaligned_blocks), 2u);

  std::vector<PrefillBlockContract> tie_break_blocks(3);
  tie_break_blocks[0].accounted_bytes = 5;
  tie_break_blocks[1].accounted_bytes = 10;
  tie_break_blocks[2].accounted_bytes = 5;
  EXPECT_EQ(choose_two_way_cost_boundary(tie_break_blocks), 1u);

  const PrefillClosedFormContract larger_than_prompt =
      resolve_prefill_closed_form_contract(5, 2, 8);
  EXPECT_EQ(larger_than_prompt.blocks_per_sequence, 1u);
  EXPECT_EQ(larger_than_prompt.prefix_block_visits,
            larger_than_prompt.tile_count);
  EXPECT_EQ(larger_than_prompt.lookups_per_layer_sequence,
            1 + 2 * larger_than_prompt.tile_count);

  const std::vector<std::vector<uint64_t>> terminal_visits =
      enumerate_terminal_visit_tokens(6, 2, 4);
  EXPECT_EQ(terminal_visits,
            (std::vector<std::vector<uint64_t>>{{2, 0}, {4, 0}, {4, 2}}));
  std::vector<std::vector<uint64_t>> terminal_visit_bytes = terminal_visits;
  for (std::vector<uint64_t>& tile : terminal_visit_bytes) {
    for (uint64_t& bytes : tile) {
      bytes *= 32;
    }
  }
  EXPECT_EQ(terminal_visit_bytes,
            (std::vector<std::vector<uint64_t>>{{64, 0},
                                                {128, 0},
                                                {128, 64}}));
}

TEST(LlmMemoryContractTest,
     SmallPrefillDomainEnumerationMatchesClosedFormsAndPartitionSums) {
  constexpr uint64_t layer_count = 2;
  constexpr uint64_t batch_size = 3;
  constexpr uint64_t k_or_v_record_bytes_per_layer = 5;
  constexpr uint64_t k_and_v_record_bytes_per_layer =
      2 * k_or_v_record_bytes_per_layer;
  constexpr uint64_t kv_bytes_per_token =
      layer_count * k_and_v_record_bytes_per_layer;
  constexpr uint64_t weight_bytes = 37;

  for (uint64_t prompt_tokens = 1; prompt_tokens <= 12; ++prompt_tokens) {
    for (uint64_t query_tile_tokens = 1;
         query_tile_tokens <= prompt_tokens; ++query_tile_tokens) {
      for (uint64_t block_tokens = 1; block_tokens <= 8;
           ++block_tokens) {
        const PrefillClosedFormContract closed =
            resolve_prefill_closed_form_contract(
                prompt_tokens, query_tile_tokens, block_tokens);
        const std::vector<uint64_t> tile_ends = enumerate_tile_ends(
            prompt_tokens, query_tile_tokens);
        const std::vector<uint64_t> prefix_block_counts =
            enumerate_prefix_block_counts(
                prompt_tokens, query_tile_tokens, block_tokens);
        const uint64_t enumerated_prefix_token_visits =
            std::accumulate(tile_ends.begin(), tile_ends.end(),
                            uint64_t{0});
        const uint64_t enumerated_prefix_block_visits =
            std::accumulate(prefix_block_counts.begin(),
                            prefix_block_counts.end(), uint64_t{0});
        EXPECT_EQ(tile_ends.size(), closed.tile_count)
            << "P=" << prompt_tokens << " Q=" << query_tile_tokens
            << " G=" << block_tokens;
        EXPECT_EQ(enumerated_prefix_token_visits,
                  closed.attention_prefix_token_visits)
            << "P=" << prompt_tokens << " Q=" << query_tile_tokens
            << " G=" << block_tokens;
        EXPECT_EQ(enumerated_prefix_block_visits,
                  closed.prefix_block_visits)
            << "P=" << prompt_tokens << " Q=" << query_tile_tokens
            << " G=" << block_tokens;
        EXPECT_EQ(closed.lookups_per_layer_sequence,
                  closed.blocks_per_sequence +
                      2 * enumerated_prefix_block_visits);

        const PrefillPayloadContract payload =
            resolve_prefill_payload_contract(
                weight_bytes, kv_bytes_per_token, batch_size,
                prompt_tokens, query_tile_tokens);
        const uint64_t expected_model_payload =
            layer_count * batch_size *
            k_and_v_record_bytes_per_layer *
            (prompt_tokens + enumerated_prefix_token_visits);
        EXPECT_EQ(payload.weight_passes, 1u);
        EXPECT_EQ(payload.weight_read_bytes, weight_bytes);
        EXPECT_EQ(payload.kv_only_bytes, expected_model_payload);

        const std::vector<PrefillBlockContract> blocks =
            enumerate_prefill_block_contracts(
                prompt_tokens, query_tile_tokens, block_tokens,
                k_or_v_record_bytes_per_layer);
        uint64_t per_layer_sequence_model_payload = 0;
        uint64_t per_layer_sequence_metadata = 0;
        uint64_t per_layer_sequence_accounted = 0;
        uint64_t per_layer_sequence_lookups = 0;
        for (const PrefillBlockContract& block : blocks) {
          per_layer_sequence_model_payload += block.model_payload_bytes;
          per_layer_sequence_metadata += block.layout_metadata_bytes;
          per_layer_sequence_accounted += block.accounted_bytes;
          per_layer_sequence_lookups += block.semantic_lookups;
        }
        EXPECT_EQ(per_layer_sequence_model_payload * layer_count * batch_size,
                  expected_model_payload);
        EXPECT_EQ(per_layer_sequence_lookups,
                  closed.lookups_per_layer_sequence);
        EXPECT_EQ(per_layer_sequence_metadata * layer_count * batch_size,
                  sizeof(uint32_t) * layer_count * batch_size *
                      closed.lookups_per_layer_sequence);
        EXPECT_EQ(per_layer_sequence_accounted * layer_count * batch_size,
                  expected_model_payload +
                      per_layer_sequence_metadata * layer_count *
                          batch_size);

        if (blocks.size() >= 2) {
          const size_t boundary = choose_two_way_cost_boundary(blocks);
          ASSERT_GT(boundary, 0u);
          ASSERT_LT(boundary, blocks.size());
          uint64_t left = 0;
          uint64_t right = 0;
          for (size_t block = 0; block < blocks.size(); ++block) {
            if (block < boundary) {
              left += blocks[block].accounted_bytes;
            } else {
              right += blocks[block].accounted_bytes;
            }
          }
          EXPECT_EQ(left + right, per_layer_sequence_accounted)
              << "P=" << prompt_tokens << " Q=" << query_tile_tokens
              << " G=" << block_tokens << " boundary=" << boundary;
          const uint64_t chosen_doubled = 2 * left;
          const uint64_t chosen_distance =
              chosen_doubled > per_layer_sequence_accounted
                  ? chosen_doubled - per_layer_sequence_accounted
                  : per_layer_sequence_accounted - chosen_doubled;
          uint64_t candidate_prefix = 0;
          for (size_t candidate = 1; candidate < blocks.size();
               ++candidate) {
            candidate_prefix += blocks[candidate - 1].accounted_bytes;
            const uint64_t candidate_doubled = 2 * candidate_prefix;
            const uint64_t candidate_distance =
                candidate_doubled > per_layer_sequence_accounted
                    ? candidate_doubled - per_layer_sequence_accounted
                    : per_layer_sequence_accounted - candidate_doubled;
            EXPECT_LE(chosen_distance, candidate_distance);
            if (chosen_distance == candidate_distance) {
              EXPECT_LE(boundary, candidate);
            }
          }
        }
      }
    }
  }
}

TEST(LlmMemoryContractTest,
     MetalSegmentCapacityAndPerPoolSlotCapAreFrozen) {
  EXPECT_EQ(kCanonicalSegmentCapacityBytes, 268435456u);
  EXPECT_EQ(kCanonicalSegmentSlotsPerPool, 256u);
  EXPECT_EQ(kCanonicalPoolCapacityBytes, 68719476736ULL);
  EXPECT_EQ(kCanonicalSegmentCapacityBytes / sizeof(uint32_t),
            67108864u);
  constexpr std::array<std::string_view, 4> segmented_pools = {
      "weights", "k", "v", "block_table"};
  constexpr std::array<size_t, 4> slots_per_pool = {256, 256, 256, 256};
  EXPECT_EQ(segmented_pools,
            (std::array<std::string_view, 4>{
                "weights", "k", "v", "block_table"}));
  for (size_t slots : slots_per_pool) {
    EXPECT_EQ(slots, kCanonicalSegmentSlotsPerPool);
  }

  EXPECT_EQ(contiguous_segment_lengths(kCanonicalSegmentCapacityBytes - 1),
            (std::vector<uint64_t>{kCanonicalSegmentCapacityBytes - 1}));
  EXPECT_EQ(contiguous_segment_lengths(kCanonicalSegmentCapacityBytes),
            (std::vector<uint64_t>{kCanonicalSegmentCapacityBytes}));
  EXPECT_EQ(contiguous_segment_lengths(kCanonicalSegmentCapacityBytes + 1),
            (std::vector<uint64_t>{kCanonicalSegmentCapacityBytes, 1}));

  const std::vector<uint64_t> maximum_pool =
      contiguous_segment_lengths(kCanonicalPoolCapacityBytes);
  ASSERT_EQ(maximum_pool.size(), kCanonicalSegmentSlotsPerPool);
  EXPECT_TRUE(std::all_of(
      maximum_pool.begin(), maximum_pool.end(), [](uint64_t length) {
        return length == kCanonicalSegmentCapacityBytes;
      }));
  EXPECT_EQ(
      contiguous_segment_lengths(kCanonicalPoolCapacityBytes + 1).size(),
      kCanonicalSegmentSlotsPerPool + 1);
  EXPECT_TRUE(segment_count_fits_pool_slots(maximum_pool.size()));
  EXPECT_FALSE(segment_count_fits_pool_slots(
      contiguous_segment_lengths(kCanonicalPoolCapacityBytes + 1).size()));

  constexpr uint64_t block_bytes = 512;
  constexpr uint64_t blocks_per_segment =
      kCanonicalSegmentCapacityBytes / block_bytes;
  constexpr uint64_t maximum_addressable_blocks =
      kCanonicalSegmentSlotsPerPool * blocks_per_segment;
  EXPECT_EQ(blocks_per_segment, 524288u);
  EXPECT_EQ(maximum_addressable_blocks, 134217728u);
  EXPECT_EQ(maximum_addressable_blocks * block_bytes,
            kCanonicalPoolCapacityBytes);
  EXPECT_EQ(ceil_divide_small(maximum_addressable_blocks,
                             blocks_per_segment),
            kCanonicalSegmentSlotsPerPool);
  EXPECT_EQ(ceil_divide_small(maximum_addressable_blocks + 1,
                             blocks_per_segment),
            kCanonicalSegmentSlotsPerPool + 1);
  EXPECT_TRUE(segment_count_fits_pool_slots(ceil_divide_small(
      maximum_addressable_blocks, blocks_per_segment)));
  EXPECT_FALSE(segment_count_fits_pool_slots(ceil_divide_small(
      maximum_addressable_blocks + 1, blocks_per_segment)));

  constexpr uint64_t non_dividing_block_bytes = 96;
  constexpr uint64_t non_dividing_blocks_per_segment =
      kCanonicalSegmentCapacityBytes / non_dividing_block_bytes;
  constexpr uint64_t non_dividing_maximum_pool =
      kCanonicalSegmentSlotsPerPool * non_dividing_blocks_per_segment *
      non_dividing_block_bytes;
  EXPECT_LT(non_dividing_maximum_pool, kCanonicalPoolCapacityBytes);
  EXPECT_EQ(kCanonicalPoolCapacityBytes - non_dividing_maximum_pool,
            kCanonicalSegmentSlotsPerPool *
                (kCanonicalSegmentCapacityBytes %
                 non_dividing_block_bytes));
  EXPECT_TRUE(
      paged_block_fits_canonical_segment(kCanonicalSegmentCapacityBytes));
  EXPECT_FALSE(paged_block_fits_canonical_segment(
      kCanonicalSegmentCapacityBytes + 1));
}

TEST(LlmMemoryContractTest,
     ComponentIdentityCanonicalSerializationUsesFixedFieldOrder) {
  static_assert(std::is_same_v<
                decltype(ComponentIdentityContract::logical_profile_version),
                std::string_view>);
  static_assert(std::is_same_v<
                decltype(ComponentIdentityContract::backend_executor_version),
                std::string_view>);
  static_assert(std::is_same_v<
                decltype(ComponentIdentityContract::permutation_version),
                std::optional<std::string_view>>);
  static_assert(std::is_same_v<
                decltype(ComponentIdentityContract::msl_revision),
                std::optional<std::string_view>>);
  const ComponentIdentityContract components = {
      "l|=v",
      "k=v",
      "p|q",
      "b",
      "r",
      "s",
      "t",
      "buf",
      "write",
      "sum",
      "msl",
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  };
  EXPECT_EQ(
      serialize_component_identity(components),
      "llm-memory-components-v1"
      "|logical_profile_version=4:l|=v"
      "|kv_layout_version=3:k=v"
      "|permutation_version=3:p|q"
      "|backend_executor_version=1:b"
      "|resource_abi_version=1:r"
      "|schedule_version=1:s"
      "|timer_policy_version=1:t"
      "|buffer_pattern_version=3:buf"
      "|write_pattern_version=5:write"
      "|checksum_pattern_version=3:sum"
      "|msl_revision=3:msl"
      "|msl_source_sha256=64:"
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");

  ComponentIdentityContract cpu_components = components;
  cpu_components.msl_revision.reset();
  cpu_components.msl_source_sha256.reset();
  const std::string cpu_identity =
      serialize_component_identity(cpu_components);
  const size_t msl_suffix = cpu_identity.find("|msl_revision=");
  ASSERT_NE(msl_suffix, std::string::npos);
  EXPECT_EQ(cpu_identity.substr(msl_suffix),
            "|msl_revision=null|msl_source_sha256=null");
}

namespace {

constexpr uint64_t kPrefillContractPhaseDomain =
    0x50524546494C4C31ULL;
constexpr uint64_t kPrefillContractOperationMultiplier =
    0x9E3779B97F4A7C15ULL;
constexpr uint64_t kPrefillContractLayerMultiplier =
    0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kPrefillContractBatchMultiplier =
    0x94D049BB133111EBULL;
constexpr uint64_t kPrefillContractWordMultiplier =
    0xD6E8FEB86659FD93ULL;
constexpr uint64_t kPrefillContractKDomain =
    0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kPrefillContractVDomain =
    0x5656565656565656ULL;

enum class PrefillContractDomain {
  K,
  V,
};

enum class PrefillContractAccess {
  Write,
  Read,
};

struct PrefillContractTaskChecksum {
  size_t exact_word_count = 0;
  size_t even_word_count = 0;
  size_t odd_word_count = 0;
  uint64_t even_word_sum = 0;
  uint64_t odd_word_sum = 0;
};

struct PrefillContractEvent {
  PrefillContractAccess access;
  PrefillContractDomain domain;
  size_t tile_index;
  size_t tile_end;
  size_t block_index;
  size_t visit_tokens;
};

uint64_t contract_prefill_affine_word(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    PrefillContractDomain domain, uint64_t logical_word_index) {
  const uint64_t domain_term =
      domain == PrefillContractDomain::K ? kPrefillContractKDomain
                                        : kPrefillContractVDomain;
  return scenario_seed + kPrefillContractPhaseDomain +
         kPrefillContractOperationMultiplier * (operation_ordinal + 1) +
         kPrefillContractLayerMultiplier * (layer_index + 1) +
         kPrefillContractBatchMultiplier *
             (batch_sequence_index + 1) +
         domain_term +
         kPrefillContractWordMultiplier * (logical_word_index + 1);
}

PrefillContractTaskChecksum enumerate_contract_prefill_task_checksum(
    uint64_t scenario_seed, size_t operation_count,
    uint64_t layer_index, uint64_t batch_sequence_index,
    PrefillContractDomain domain, size_t first_word, size_t word_count) {
  PrefillContractTaskChecksum checksum;
  for (size_t operation = 0; operation < operation_count; ++operation) {
    for (size_t offset = 0; offset < word_count; ++offset) {
      const size_t logical_word = first_word + offset;
      const uint64_t value = contract_prefill_affine_word(
          scenario_seed, operation, layer_index, batch_sequence_index,
          domain, logical_word);
      ++checksum.exact_word_count;
      if ((logical_word & 1U) == 0) {
        ++checksum.even_word_count;
        checksum.even_word_sum += value;
      } else {
        ++checksum.odd_word_count;
        checksum.odd_word_sum += value;
      }
    }
  }
  return checksum;
}

std::vector<PrefillContractEvent> enumerate_contract_prefill_block_trace(
    size_t prompt_tokens, size_t query_tile_tokens,
    size_t block_tokens, size_t block_count) {
  std::vector<PrefillContractEvent> events;
  for (size_t block = 0; block < block_count; ++block) {
    const size_t block_start = block * block_tokens;
    const size_t valid_tokens =
        std::min(block_tokens, prompt_tokens - block_start);
    events.push_back({PrefillContractAccess::Write,
                      PrefillContractDomain::K, 0, prompt_tokens,
                      block, valid_tokens});
    events.push_back({PrefillContractAccess::Write,
                      PrefillContractDomain::V, 0, prompt_tokens,
                      block, valid_tokens});
  }

  size_t tile_index = 0;
  size_t tile_end = 0;
  while (tile_end < prompt_tokens) {
    tile_end += std::min(query_tile_tokens, prompt_tokens - tile_end);
    for (const PrefillContractDomain domain :
         {PrefillContractDomain::K, PrefillContractDomain::V}) {
      for (size_t block = 0; block < block_count; ++block) {
        const size_t block_start = block * block_tokens;
        if (block_start >= tile_end) {
          break;
        }
        events.push_back({PrefillContractAccess::Read, domain,
                          tile_index, tile_end, block,
                          std::min(block_tokens,
                                   tile_end - block_start)});
      }
    }
    ++tile_index;
  }
  return events;
}

}  // namespace

TEST(LlmMemoryContractTest,
     PrefillAffine64TwoOperationChecksumAndFinalOrdinalAreFrozen) {
  constexpr uint64_t kScenarioSeed = 0x0123456789ABCDEFULL;
  const PrefillContractTaskChecksum checksum =
      enumerate_contract_prefill_task_checksum(
          kScenarioSeed, 2, 2, 3, PrefillContractDomain::K, 1, 5);
  EXPECT_EQ(checksum.exact_word_count, 10u);
  EXPECT_EQ(checksum.even_word_count, 4u);
  EXPECT_EQ(checksum.odd_word_count, 6u);
  EXPECT_EQ(checksum.even_word_sum, 0xDC08129268383AB6ULL);
  EXPECT_EQ(checksum.odd_word_sum, 0xCA0C1BDB9C545811ULL);

  EXPECT_EQ(contract_prefill_affine_word(
                kScenarioSeed, 0, 2, 3, PrefillContractDomain::K, 1),
            0x7A144A570DB4D57DULL);
  const uint64_t final_word = contract_prefill_affine_word(
      kScenarioSeed, 1, 2, 3, PrefillContractDomain::K, 1);
  EXPECT_EQ(final_word, 0x184BC4108CFF5192ULL);
  constexpr std::array<uint8_t, 8> kFinalLittleEndianBytes = {
      0x92, 0x51, 0xFF, 0x8C, 0x10, 0xC4, 0x4B, 0x18};
  for (size_t byte = 0; byte < kFinalLittleEndianBytes.size(); ++byte) {
    EXPECT_EQ(static_cast<uint8_t>(final_word >> (byte * 8)),
              kFinalLittleEndianBytes[byte]);
  }
}

TEST(LlmMemoryContractTest,
     PrefillTwoBlockTwoTileOwnerLocalSemanticTraceIsFrozen) {
  const std::vector<PrefillContractEvent> trace =
      enumerate_contract_prefill_block_trace(4, 2, 2, 2);
  const std::array<PrefillContractEvent, 10> expected = {{
      {PrefillContractAccess::Write, PrefillContractDomain::K, 0, 4, 0, 2},
      {PrefillContractAccess::Write, PrefillContractDomain::V, 0, 4, 0, 2},
      {PrefillContractAccess::Write, PrefillContractDomain::K, 0, 4, 1, 2},
      {PrefillContractAccess::Write, PrefillContractDomain::V, 0, 4, 1, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::K, 0, 2, 0, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::V, 0, 2, 0, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::K, 1, 4, 0, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::K, 1, 4, 1, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::V, 1, 4, 0, 2},
      {PrefillContractAccess::Read, PrefillContractDomain::V, 1, 4, 1, 2},
  }};
  ASSERT_EQ(trace.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(trace[index].access, expected[index].access);
    EXPECT_EQ(trace[index].domain, expected[index].domain);
    EXPECT_EQ(trace[index].tile_index, expected[index].tile_index);
    EXPECT_EQ(trace[index].tile_end, expected[index].tile_end);
    EXPECT_EQ(trace[index].block_index, expected[index].block_index);
    EXPECT_EQ(trace[index].visit_tokens,
              expected[index].visit_tokens);
  }
}
