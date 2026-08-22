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
#include <type_traits>
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

enum class ContractBackend {
  Cpu,
  Metal,
};

enum class ContractPhase {
  Decode,
  Prefill,
};

enum class ContractKvLayout {
  Contiguous,
  Paged,
};

enum class ContractWorkUnitKind {
  DecodeStep,
  PrefillOperation,
};

enum class ContractKvWriteKind {
  None,
  CurrentTokenAppend,
  FullPromptPopulation,
};

enum class ContractScenario {
  WeightsOnly,
  KvOnly,
  Mixed,
};

enum class ContractRunStatus {
  NotStarted,
  Complete,
  Partial,
  Interrupted,
  Unsupported,
  Failed,
};

enum class ContractMeasurementStatus {
  NotRun,
  Measured,
  Interrupted,
  Invalid,
  Failed,
};

enum class SchemaValueKind {
  Integer,
  IntegerOrNull,
  DecimalString,
  DecimalStringOrNull,
  Boolean,
  String,
  StringOrNull,
  FiniteNumberOrNull,
  Object,
  ObjectOrNull,
  Array,
};

struct SchemaFieldContract {
  std::string_view section;
  std::string_view name;
  SchemaValueKind kind;
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

struct ProfileActivationContract {
  ContractBackend backend;
  ContractPhase phase;
  ContractKvLayout layout;
  std::optional<int> activation_phase;
  bool preexisting_at_phase_zero;
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

struct PreExpansionDecodeIdentity {
  std::string_view mode;
  std::string_view backend;
  int schema_version;
  std::string_view methodology;
  std::string_view descriptor_abi;
  std::string_view append_identity;
  std::string_view checksum_identity;
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
  bool conclusions_valid;
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

std::string_view contract_token(ContractBackend backend) {
  switch (backend) {
    case ContractBackend::Cpu:
      return "cpu";
    case ContractBackend::Metal:
      return "metal";
  }
  return {};
}

std::string_view contract_token(ContractPhase phase) {
  switch (phase) {
    case ContractPhase::Decode:
      return "decode";
    case ContractPhase::Prefill:
      return "prefill";
  }
  return {};
}

std::string_view contract_token(ContractKvLayout layout) {
  switch (layout) {
    case ContractKvLayout::Contiguous:
      return "contiguous";
    case ContractKvLayout::Paged:
      return "paged";
  }
  return {};
}

std::string_view contract_token(ContractWorkUnitKind kind) {
  switch (kind) {
    case ContractWorkUnitKind::DecodeStep:
      return "decode_step";
    case ContractWorkUnitKind::PrefillOperation:
      return "prefill_operation";
  }
  return {};
}

std::string_view contract_token(ContractKvWriteKind kind) {
  switch (kind) {
    case ContractKvWriteKind::None:
      return "none";
    case ContractKvWriteKind::CurrentTokenAppend:
      return "current_token_append";
    case ContractKvWriteKind::FullPromptPopulation:
      return "full_prompt_population";
  }
  return {};
}

std::string_view contract_token(ContractScenario scenario) {
  switch (scenario) {
    case ContractScenario::WeightsOnly:
      return "weights_only";
    case ContractScenario::KvOnly:
      return "kv_only";
    case ContractScenario::Mixed:
      return "mixed";
  }
  return {};
}

std::string_view contract_token(ContractRunStatus status) {
  switch (status) {
    case ContractRunStatus::NotStarted:
      return "not_started";
    case ContractRunStatus::Complete:
      return "complete";
    case ContractRunStatus::Partial:
      return "partial";
    case ContractRunStatus::Interrupted:
      return "interrupted";
    case ContractRunStatus::Unsupported:
      return "unsupported";
    case ContractRunStatus::Failed:
      return "failed";
  }
  return {};
}

std::string_view contract_token(ContractMeasurementStatus status) {
  switch (status) {
    case ContractMeasurementStatus::NotRun:
      return "not_run";
    case ContractMeasurementStatus::Measured:
      return "measured";
    case ContractMeasurementStatus::Interrupted:
      return "interrupted";
    case ContractMeasurementStatus::Invalid:
      return "invalid";
    case ContractMeasurementStatus::Failed:
      return "failed";
  }
  return {};
}

std::string_view schema_kind_token(SchemaValueKind kind) {
  switch (kind) {
    case SchemaValueKind::Integer:
      return "integer";
    case SchemaValueKind::IntegerOrNull:
      return "integer_or_null";
    case SchemaValueKind::DecimalString:
      return "decimal_string";
    case SchemaValueKind::DecimalStringOrNull:
      return "decimal_string_or_null";
    case SchemaValueKind::Boolean:
      return "boolean";
    case SchemaValueKind::String:
      return "string";
    case SchemaValueKind::StringOrNull:
      return "string_or_null";
    case SchemaValueKind::FiniteNumberOrNull:
      return "finite_number_or_null";
    case SchemaValueKind::Object:
      return "object";
    case SchemaValueKind::ObjectOrNull:
      return "object_or_null";
    case SchemaValueKind::Array:
      return "array";
  }
  return {};
}

std::string methodology_token(ContractBackend backend,
                              ContractPhase phase,
                              ContractKvLayout layout) {
  return "llm-memory-v1-" + std::string(contract_token(backend)) + "-" +
         std::string(contract_token(phase)) + "-" +
         std::string(contract_token(layout));
}

ContractWorkUnitKind work_unit_kind_for_phase(ContractPhase phase) {
  return phase == ContractPhase::Decode
             ? ContractWorkUnitKind::DecodeStep
             : ContractWorkUnitKind::PrefillOperation;
}

ContractKvWriteKind kv_write_kind_for(ContractPhase phase,
                                      ContractScenario scenario) {
  if (scenario == ContractScenario::WeightsOnly) {
    return ContractKvWriteKind::None;
  }
  return phase == ContractPhase::Decode
             ? ContractKvWriteKind::CurrentTokenAppend
             : ContractKvWriteKind::FullPromptPopulation;
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

std::string sha256_text(std::string_view input) {
  CC_SHA256_CTX context;
  if (CC_SHA256_Init(&context) != 1) {
    return {};
  }
  constexpr size_t kBytesPerUpdateCap = 4096;
  size_t offset = 0;
  while (offset < input.size()) {
    const size_t byte_count =
        std::min(kBytesPerUpdateCap, input.size() - offset);
    if (CC_SHA256_Update(&context, input.data() + offset,
                         static_cast<CC_LONG>(byte_count)) != 1) {
      return {};
    }
    offset += byte_count;
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

std::vector<SchemaFieldContract> minimum_generic_schema_v1_vocabulary() {
  return {
      {"top_level", "schema_version", SchemaValueKind::Integer},
      {"top_level", "mode", SchemaValueKind::String},
      {"top_level", "backend", SchemaValueKind::String},
      {"top_level", "phase", SchemaValueKind::String},
      {"top_level", "kv_layout", SchemaValueKind::String},
      {"top_level", "methodology_version", SchemaValueKind::String},
      {"top_level", "software", SchemaValueKind::Object},
      {"top_level", "configuration", SchemaValueKind::Object},
      {"top_level", "resolved_plan", SchemaValueKind::Object},
      {"top_level", "backend_evidence", SchemaValueKind::Object},
      {"top_level", "memory_budget", SchemaValueKind::Object},
      {"top_level", "calibration", SchemaValueKind::Object},
      {"top_level", "measurements", SchemaValueKind::Array},
      {"top_level", "aggregates", SchemaValueKind::Object},
      {"top_level", "status", SchemaValueKind::String},
      {"top_level", "reason_code", SchemaValueKind::String},
      {"top_level", "results_complete", SchemaValueKind::Boolean},
      {"top_level", "conclusions_valid", SchemaValueKind::Boolean},
      {"top_level", "interpretation", SchemaValueKind::Object},

      {"configuration", "argv", SchemaValueKind::Array},
      {"configuration", "resolved_sources", SchemaValueKind::Object},

      {"resolved_plan", "geometry", SchemaValueKind::Object},
      {"resolved_plan", "layout", SchemaValueKind::Object},
      {"resolved_plan", "resources", SchemaValueKind::Object},
      {"resolved_plan", "component_identities", SchemaValueKind::Object},
      {"resolved_plan.geometry", "decode", SchemaValueKind::ObjectOrNull},
      {"resolved_plan.geometry", "prefill", SchemaValueKind::ObjectOrNull},

      {"resolved_plan.geometry.decode", "visible_context_tokens",
       SchemaValueKind::Integer},
      {"resolved_plan.geometry.prefill", "prompt_tokens",
       SchemaValueKind::Integer},
      {"resolved_plan.geometry.prefill", "attention_query_tile_tokens",
       SchemaValueKind::Integer},
      {"resolved_plan.geometry.prefill", "tile_count",
       SchemaValueKind::DecimalString},
      {"resolved_plan.geometry.prefill",
       "attention_prefix_token_visits_per_sequence",
       SchemaValueKind::DecimalString},
      {"resolved_plan.geometry.prefill", "causal_token_pairs_per_sequence",
       SchemaValueKind::DecimalString},
      {"resolved_plan.geometry.prefill", "logical_attention_pairs",
       SchemaValueKind::DecimalString},
      {"resolved_plan.geometry.prefill", "logical_attention_fma_terms",
       SchemaValueKind::DecimalString},

      {"resolved_plan.layout", "kv_layout", SchemaValueKind::String},
      {"resolved_plan.layout", "kv_block_tokens",
       SchemaValueKind::IntegerOrNull},
      {"resolved_plan.layout", "blocks_per_sequence",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "physical_blocks_per_layer",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "last_block_tokens",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "last_block_valid_bytes",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "block_table_entries",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "block_table_bytes",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "permutation_domain_uint64_hex",
       SchemaValueKind::StringOrNull},
      {"resolved_plan.layout", "permutation_seed_uint64_decimal",
       SchemaValueKind::DecimalStringOrNull},
      {"resolved_plan.layout", "permutation_algorithm_version",
       SchemaValueKind::StringOrNull},
      {"resolved_plan.layout", "permutation_sha256",
       SchemaValueKind::StringOrNull},

      {"resolved_plan.resources", "weight_logical_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "k_logical_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "v_logical_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "k_physical_length_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "v_physical_length_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "k_layout_padding_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "v_layout_padding_bytes",
       SchemaValueKind::DecimalString},
      {"resolved_plan.resources", "block_table_bytes",
       SchemaValueKind::DecimalStringOrNull},

      {"memory_budget", "resource_rounding_bytes",
       SchemaValueKind::DecimalString},
      {"memory_budget", "transient_peak_bytes",
       SchemaValueKind::DecimalString},
      {"memory_budget", "known_owned_peak_bytes",
       SchemaValueKind::DecimalString},
      {"memory_budget", "admitted_budget_bytes",
       SchemaValueKind::DecimalString},

      {"resolved_plan.component_identities", "logical_profile_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "kv_layout_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "permutation_version",
       SchemaValueKind::StringOrNull},
      {"resolved_plan.component_identities", "backend_executor_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "resource_abi_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "schedule_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "timer_policy_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "buffer_pattern_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "write_pattern_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "checksum_pattern_version",
       SchemaValueKind::String},
      {"resolved_plan.component_identities", "msl_revision",
       SchemaValueKind::StringOrNull},
      {"resolved_plan.component_identities", "msl_source_sha256",
       SchemaValueKind::StringOrNull},

      {"measurements.*", "work_unit_kind", SchemaValueKind::String},
      {"measurements.*", "planned_work_units", SchemaValueKind::Integer},
      {"measurements.*", "completed_work_units",
       SchemaValueKind::Integer},
      {"measurements.*", "weight_read_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "kv_read_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "kv_write_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "kv_write_kind", SchemaValueKind::String},
      {"measurements.*", "effective_model_payload_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "layout_metadata_lookup_count_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "layout_metadata_read_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "accounted_bytes_per_work_unit",
       SchemaValueKind::DecimalString},
      {"measurements.*", "planned_effective_model_payload_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "completed_effective_model_payload_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "planned_layout_metadata_lookup_count",
       SchemaValueKind::DecimalString},
      {"measurements.*", "completed_layout_metadata_lookup_count",
       SchemaValueKind::DecimalString},
      {"measurements.*", "planned_layout_metadata_read_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "completed_layout_metadata_read_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "planned_task_accounted_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "completed_task_accounted_bytes",
       SchemaValueKind::DecimalString},
      {"measurements.*", "synthetic_work_unit_latency_seconds",
       SchemaValueKind::FiniteNumberOrNull},
      {"measurements.*", "synthetic_memory_work_units_per_second",
       SchemaValueKind::FiniteNumberOrNull},
      {"measurements.*", "effective_model_payload_gb_s",
       SchemaValueKind::FiniteNumberOrNull},

      {"backend_evidence", "cpu", SchemaValueKind::ObjectOrNull},
      {"backend_evidence", "metal", SchemaValueKind::ObjectOrNull},
  };
}

std::string serialize_schema_vocabulary(
    const std::vector<SchemaFieldContract>& vocabulary) {
  std::string result = "llm-memory-schema-v1";
  for (const SchemaFieldContract& field : vocabulary) {
    result += '|';
    result += field.section;
    result += '.';
    result += field.name;
    result += ':';
    result += schema_kind_token(field.kind);
  }
  return result;
}

const SchemaFieldContract* find_schema_field(
    const std::vector<SchemaFieldContract>& vocabulary,
    std::string_view section,
    std::string_view name) {
  const auto found = std::find_if(
      vocabulary.begin(), vocabulary.end(),
      [section, name](const SchemaFieldContract& field) {
        return field.section == section && field.name == name;
      });
  return found == vocabulary.end() ? nullptr : &*found;
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

bool matches_pre_expansion_decode_identity(
    const PreExpansionDecodeIdentity& identity) {
  return identity.mode == "llm_memory" && identity.backend == "cpu" &&
         identity.schema_version == 1 &&
         identity.methodology ==
             "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved" &&
         identity.descriptor_abi == "llm-memory-descriptor-abi-v1" &&
         identity.append_identity == "llm-kv-append-affine64-v1" &&
         identity.checksum_identity == "llm-read-checksum-v1";
}

bool accepted_pre_expansion_completion(std::string_view mode,
                                       int schema_version,
                                       std::string_view status,
                                       bool results_complete,
                                       bool conclusions_valid) {
  return mode == "llm_memory" && schema_version == 1 &&
         status == "complete" && results_complete && conclusions_valid;
}

bool accepted_generic_result(const GenericResultIdentity& identity) {
  const bool backend_is_known =
      identity.backend == "cpu" || identity.backend == "metal";
  const bool phase_is_known =
      identity.phase == "decode" || identity.phase == "prefill";
  const bool layout_is_known = identity.kv_layout == "contiguous" ||
                               identity.kv_layout == "paged";
  const std::string expected_methodology =
      "llm-memory-v1-" + std::string(identity.backend) + "-" +
      std::string(identity.phase) + "-" + std::string(identity.kv_layout);
  return identity.mode == "llm_memory" && identity.schema_version == 1 &&
         backend_is_known && phase_is_known && layout_is_known &&
         identity.backend == identity.requested_backend &&
         identity.phase == identity.requested_phase &&
         identity.kv_layout == identity.requested_kv_layout &&
         identity.methodology_version == expected_methodology &&
         identity.status == "complete" && identity.results_complete &&
         identity.conclusions_valid &&
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

TEST(LlmMemoryContractTest,
     PreExpansionDecodeIdentityAndCompletionRemainRegressionGoldens) {
  const PreExpansionDecodeIdentity frozen = {
      "llm_memory",
      "cpu",
      1,
      "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved",
      "llm-memory-descriptor-abi-v1",
      "llm-kv-append-affine64-v1",
      "llm-read-checksum-v1",
  };
  EXPECT_TRUE(matches_pre_expansion_decode_identity(frozen));

  PreExpansionDecodeIdentity candidate = frozen;
  candidate.mode = "benchmark";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.backend = "gpu";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.schema_version = 2;
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.methodology = "llm-memory-v2";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.descriptor_abi = "llm-memory-descriptor-abi-v2";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.append_identity = "llm-kv-append-affine64-v2";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));
  candidate = frozen;
  candidate.checksum_identity = "llm-read-checksum-v2";
  EXPECT_FALSE(matches_pre_expansion_decode_identity(candidate));

  EXPECT_TRUE(accepted_pre_expansion_completion(
      "llm_memory", 1, "complete", true, true));
  EXPECT_FALSE(
      accepted_pre_expansion_completion("benchmark", 1, "complete", true,
                                        true));
  EXPECT_FALSE(accepted_pre_expansion_completion(
      "llm_memory", 2, "complete", true, true));
  EXPECT_FALSE(accepted_pre_expansion_completion(
      "llm_memory", 1, "partial", true, true));
  EXPECT_FALSE(accepted_pre_expansion_completion(
      "llm_memory", 1, "complete", false, true));
  EXPECT_FALSE(accepted_pre_expansion_completion(
      "llm_memory", 1, "complete", true, false));
}

TEST(LlmMemoryContractTest, GenericV1AcceptancePredicateIsExact) {
  const GenericResultIdentity accepted = {
      "llm_memory",
      1,
      "metal",
      "metal",
      "prefill",
      "prefill",
      "paged",
      "paged",
      "llm-memory-v1-metal-prefill-paged",
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
  candidate.schema_version = 2;
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
  candidate.methodology_version = "llm-memory-v1-gpu-prefill-paged";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.requested_phase = "train";
  candidate.phase = "train";
  candidate.methodology_version = "llm-memory-v1-metal-train-paged";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.requested_kv_layout = "sparse";
  candidate.kv_layout = "sparse";
  candidate.methodology_version = "llm-memory-v1-metal-prefill-sparse";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.methodology_version =
      "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.status = "partial";
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.results_complete = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.conclusions_valid = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
  candidate = accepted;
  candidate.all_planned_measurements_measured = false;
  EXPECT_FALSE(accepted_generic_result(candidate));
}

TEST(LlmMemoryContractTest,
     GenericVocabularyAndMethodologyTokensAreCanonical) {
  EXPECT_EQ(
      (std::array<std::string_view, 2>{
          contract_token(ContractBackend::Cpu),
          contract_token(ContractBackend::Metal),
      }),
      (std::array<std::string_view, 2>{"cpu", "metal"}));
  EXPECT_EQ(
      (std::array<std::string_view, 2>{
          contract_token(ContractPhase::Decode),
          contract_token(ContractPhase::Prefill),
      }),
      (std::array<std::string_view, 2>{"decode", "prefill"}));
  EXPECT_EQ(
      (std::array<std::string_view, 2>{
          contract_token(ContractKvLayout::Contiguous),
          contract_token(ContractKvLayout::Paged),
      }),
      (std::array<std::string_view, 2>{"contiguous", "paged"}));
  EXPECT_EQ(
      (std::array<std::string_view, 2>{
          contract_token(ContractWorkUnitKind::DecodeStep),
          contract_token(ContractWorkUnitKind::PrefillOperation),
      }),
      (std::array<std::string_view, 2>{"decode_step",
                                       "prefill_operation"}));
  EXPECT_EQ(
      (std::array<std::string_view, 3>{
          contract_token(ContractKvWriteKind::None),
          contract_token(ContractKvWriteKind::CurrentTokenAppend),
          contract_token(ContractKvWriteKind::FullPromptPopulation),
      }),
      (std::array<std::string_view, 3>{"none", "current_token_append",
                                       "full_prompt_population"}));
  EXPECT_EQ(
      (std::array<std::string_view, 3>{
          contract_token(ContractScenario::WeightsOnly),
          contract_token(ContractScenario::KvOnly),
          contract_token(ContractScenario::Mixed),
      }),
      (std::array<std::string_view, 3>{"weights_only", "kv_only",
                                       "mixed"}));
  EXPECT_EQ(
      (std::array<std::string_view, 6>{
          contract_token(ContractRunStatus::NotStarted),
          contract_token(ContractRunStatus::Complete),
          contract_token(ContractRunStatus::Partial),
          contract_token(ContractRunStatus::Interrupted),
          contract_token(ContractRunStatus::Unsupported),
          contract_token(ContractRunStatus::Failed),
      }),
      (std::array<std::string_view, 6>{
          "not_started", "complete", "partial", "interrupted",
          "unsupported", "failed"}));
  EXPECT_EQ(
      (std::array<std::string_view, 5>{
          contract_token(ContractMeasurementStatus::NotRun),
          contract_token(ContractMeasurementStatus::Measured),
          contract_token(ContractMeasurementStatus::Interrupted),
          contract_token(ContractMeasurementStatus::Invalid),
          contract_token(ContractMeasurementStatus::Failed),
      }),
      (std::array<std::string_view, 5>{
          "not_run", "measured", "interrupted", "invalid", "failed"}));

  EXPECT_EQ(work_unit_kind_for_phase(ContractPhase::Decode),
            ContractWorkUnitKind::DecodeStep);
  EXPECT_EQ(work_unit_kind_for_phase(ContractPhase::Prefill),
            ContractWorkUnitKind::PrefillOperation);
  for (ContractPhase phase :
       {ContractPhase::Decode, ContractPhase::Prefill}) {
    EXPECT_EQ(kv_write_kind_for(phase, ContractScenario::WeightsOnly),
              ContractKvWriteKind::None);
  }
  for (ContractScenario scenario :
       {ContractScenario::KvOnly, ContractScenario::Mixed}) {
    EXPECT_EQ(kv_write_kind_for(ContractPhase::Decode, scenario),
              ContractKvWriteKind::CurrentTokenAppend);
    EXPECT_EQ(kv_write_kind_for(ContractPhase::Prefill, scenario),
              ContractKvWriteKind::FullPromptPopulation);
  }

  const std::array<ContractBackend, 2> backends = {
      ContractBackend::Cpu, ContractBackend::Metal};
  const std::array<ContractPhase, 2> phases = {
      ContractPhase::Decode, ContractPhase::Prefill};
  const std::array<ContractKvLayout, 2> layouts = {
      ContractKvLayout::Contiguous, ContractKvLayout::Paged};
  const std::array<std::string_view, 8> expected_methodologies = {
      "llm-memory-v1-cpu-decode-contiguous",
      "llm-memory-v1-cpu-decode-paged",
      "llm-memory-v1-cpu-prefill-contiguous",
      "llm-memory-v1-cpu-prefill-paged",
      "llm-memory-v1-metal-decode-contiguous",
      "llm-memory-v1-metal-decode-paged",
      "llm-memory-v1-metal-prefill-contiguous",
      "llm-memory-v1-metal-prefill-paged",
  };
  size_t methodology_index = 0;
  for (ContractBackend backend : backends) {
    for (ContractPhase phase : phases) {
      for (ContractKvLayout layout : layouts) {
        EXPECT_EQ(methodology_token(backend, phase, layout),
                  expected_methodologies[methodology_index]);
        ++methodology_index;
      }
    }
  }
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

TEST(LlmMemoryContractTest,
     MinimumGenericSchemaV1FieldVocabularyAndTypesAreFrozen) {
  const std::vector<SchemaFieldContract> vocabulary =
      minimum_generic_schema_v1_vocabulary();
  ASSERT_EQ(vocabulary.size(), 95u);

  const auto names_in_section =
      [&vocabulary](std::string_view section) {
        std::vector<std::string_view> names;
        for (const SchemaFieldContract& field : vocabulary) {
          if (field.section == section) {
            names.push_back(field.name);
          }
        }
        return names;
      };
  EXPECT_EQ(
      names_in_section("top_level"),
      (std::vector<std::string_view>{
          "schema_version", "mode", "backend", "phase", "kv_layout",
          "methodology_version", "software", "configuration",
          "resolved_plan", "backend_evidence", "memory_budget",
          "calibration", "measurements", "aggregates", "status",
          "reason_code", "results_complete", "conclusions_valid",
          "interpretation"}));
  EXPECT_EQ(names_in_section("configuration"),
            (std::vector<std::string_view>{"argv", "resolved_sources"}));
  EXPECT_EQ(names_in_section("resolved_plan"),
            (std::vector<std::string_view>{
                "geometry", "layout", "resources",
                "component_identities"}));
  EXPECT_EQ(names_in_section("resolved_plan.geometry"),
            (std::vector<std::string_view>{"decode", "prefill"}));
  EXPECT_EQ(names_in_section("resolved_plan.geometry.decode"),
            (std::vector<std::string_view>{"visible_context_tokens"}));
  EXPECT_EQ(
      names_in_section("resolved_plan.geometry.prefill"),
      (std::vector<std::string_view>{
          "prompt_tokens", "attention_query_tile_tokens", "tile_count",
          "attention_prefix_token_visits_per_sequence",
          "causal_token_pairs_per_sequence", "logical_attention_pairs",
          "logical_attention_fma_terms"}));
  EXPECT_EQ(
      names_in_section("resolved_plan.layout"),
      (std::vector<std::string_view>{
          "kv_layout", "kv_block_tokens", "blocks_per_sequence",
          "physical_blocks_per_layer", "last_block_tokens",
          "last_block_valid_bytes", "block_table_entries",
          "block_table_bytes", "permutation_domain_uint64_hex",
          "permutation_seed_uint64_decimal",
          "permutation_algorithm_version", "permutation_sha256"}));
  EXPECT_EQ(
      names_in_section("resolved_plan.resources"),
      (std::vector<std::string_view>{
          "weight_logical_bytes", "k_logical_bytes", "v_logical_bytes",
          "k_physical_length_bytes", "v_physical_length_bytes",
          "k_layout_padding_bytes", "v_layout_padding_bytes",
          "block_table_bytes"}));
  EXPECT_EQ(names_in_section("memory_budget"),
            (std::vector<std::string_view>{
                "resource_rounding_bytes", "transient_peak_bytes",
                "known_owned_peak_bytes", "admitted_budget_bytes"}));
  EXPECT_EQ(
      names_in_section("resolved_plan.component_identities"),
      (std::vector<std::string_view>{
          "logical_profile_version", "kv_layout_version",
          "permutation_version", "backend_executor_version",
          "resource_abi_version", "schedule_version",
          "timer_policy_version", "buffer_pattern_version",
          "write_pattern_version", "checksum_pattern_version",
          "msl_revision", "msl_source_sha256"}));
  EXPECT_EQ(
      names_in_section("measurements.*"),
      (std::vector<std::string_view>{
          "work_unit_kind", "planned_work_units", "completed_work_units",
          "weight_read_bytes_per_work_unit",
          "kv_read_bytes_per_work_unit", "kv_write_bytes_per_work_unit",
          "kv_write_kind", "effective_model_payload_bytes_per_work_unit",
          "layout_metadata_lookup_count_per_work_unit",
          "layout_metadata_read_bytes_per_work_unit",
          "accounted_bytes_per_work_unit",
          "planned_effective_model_payload_bytes",
          "completed_effective_model_payload_bytes",
          "planned_layout_metadata_lookup_count",
          "completed_layout_metadata_lookup_count",
          "planned_layout_metadata_read_bytes",
          "completed_layout_metadata_read_bytes",
          "planned_task_accounted_bytes", "completed_task_accounted_bytes",
          "synthetic_work_unit_latency_seconds",
          "synthetic_memory_work_units_per_second",
          "effective_model_payload_gb_s"}));
  EXPECT_EQ(names_in_section("backend_evidence"),
            (std::vector<std::string_view>{"cpu", "metal"}));

  for (size_t left = 0; left < vocabulary.size(); ++left) {
    for (size_t right = left + 1; right < vocabulary.size(); ++right) {
      EXPECT_FALSE(vocabulary[left].section == vocabulary[right].section &&
                   vocabulary[left].name == vocabulary[right].name);
    }
  }

  ASSERT_NE(find_schema_field(vocabulary, "resolved_plan.geometry.prefill",
                              "prompt_tokens"),
            nullptr);
  EXPECT_EQ(find_schema_field(vocabulary, "resolved_plan.geometry.prefill",
                              "prompt_tokens")
                ->kind,
            SchemaValueKind::Integer);
  EXPECT_EQ(find_schema_field(vocabulary, "resolved_plan.geometry.prefill",
                              "attention_prefix_token_visits_per_sequence")
                ->kind,
            SchemaValueKind::DecimalString);
  EXPECT_EQ(find_schema_field(vocabulary, "resolved_plan.layout",
                              "kv_block_tokens")
                ->kind,
            SchemaValueKind::IntegerOrNull);
  EXPECT_EQ(find_schema_field(vocabulary, "resolved_plan.layout",
                              "permutation_seed_uint64_decimal")
                ->kind,
            SchemaValueKind::DecimalStringOrNull);
  EXPECT_EQ(find_schema_field(vocabulary, "measurements.*",
                              "layout_metadata_lookup_count_per_work_unit")
                ->kind,
            SchemaValueKind::DecimalString);
  EXPECT_EQ(find_schema_field(vocabulary, "measurements.*",
                              "effective_model_payload_gb_s")
                ->kind,
            SchemaValueKind::FiniteNumberOrNull);
  EXPECT_EQ(find_schema_field(vocabulary, "backend_evidence", "cpu")
                ->kind,
            SchemaValueKind::ObjectOrNull);
  EXPECT_EQ(find_schema_field(vocabulary, "memory_budget",
                              "known_owned_peak_bytes")
                ->kind,
            SchemaValueKind::DecimalString);
  EXPECT_EQ(find_schema_field(vocabulary, "memory_budget",
                              "resource_rounding_bytes")
                ->kind,
            SchemaValueKind::DecimalString);
  EXPECT_EQ(find_schema_field(vocabulary, "top_level", "reason_code")
                ->kind,
            SchemaValueKind::String);
  EXPECT_EQ(find_schema_field(vocabulary, "top_level",
                              "conclusions_valid")
                ->kind,
            SchemaValueKind::Boolean);

  const std::string serialized = serialize_schema_vocabulary(vocabulary);
  EXPECT_EQ(serialized.substr(0, 20), "llm-memory-schema-v1");
  EXPECT_EQ(sha256_text(serialized),
            "77fbcf8e13b7399cff685c41854d5d93e36b18cb4103cb490cf52e52829b2442");
}

TEST(LlmMemoryContractTest,
     FinalSupportMatrixAndPublicActivationOrderAreFrozen) {
  constexpr std::array<ProfileActivationContract, 8> profiles = {{
      {ContractBackend::Cpu, ContractPhase::Decode,
       ContractKvLayout::Contiguous, std::nullopt, true},
      {ContractBackend::Cpu, ContractPhase::Decode,
       ContractKvLayout::Paged, 4, false},
      {ContractBackend::Cpu, ContractPhase::Prefill,
       ContractKvLayout::Contiguous, 6, false},
      {ContractBackend::Cpu, ContractPhase::Prefill,
       ContractKvLayout::Paged, 7, false},
      {ContractBackend::Metal, ContractPhase::Decode,
       ContractKvLayout::Contiguous, 9, false},
      {ContractBackend::Metal, ContractPhase::Decode,
       ContractKvLayout::Paged, 10, false},
      {ContractBackend::Metal, ContractPhase::Prefill,
       ContractKvLayout::Contiguous, 11, false},
      {ContractBackend::Metal, ContractPhase::Prefill,
       ContractKvLayout::Paged, 12, false},
  }};
  constexpr std::array<std::string_view, 8> expected_profiles = {
      "cpu/decode/contiguous", "cpu/decode/paged",
      "cpu/prefill/contiguous", "cpu/prefill/paged",
      "metal/decode/contiguous", "metal/decode/paged",
      "metal/prefill/contiguous", "metal/prefill/paged",
  };
  constexpr std::array<std::optional<int>, 8> expected_activation_phases = {
      std::nullopt, 4, 6, 7, 9, 10, 11, 12};

  size_t scenario_profile_count = 0;
  for (size_t index = 0; index < profiles.size(); ++index) {
    const ProfileActivationContract& profile = profiles[index];
    const std::string serialized =
        std::string(contract_token(profile.backend)) + "/" +
        std::string(contract_token(profile.phase)) + "/" +
        std::string(contract_token(profile.layout));
    EXPECT_EQ(serialized, expected_profiles[index]);
    EXPECT_EQ(profile.activation_phase, expected_activation_phases[index]);
    EXPECT_EQ(profile.preexisting_at_phase_zero, index == 0);
    scenario_profile_count += 3;
    for (size_t other = index + 1; other < profiles.size(); ++other) {
      EXPECT_FALSE(profile.backend == profiles[other].backend &&
                   profile.phase == profiles[other].phase &&
                   profile.layout == profiles[other].layout);
    }
  }
  EXPECT_EQ(scenario_profile_count, 24u);

  constexpr std::array<std::pair<int, std::string_view>, 7>
      expected_new_profile_activation_order = {{
          {4, "cpu/decode/paged"},
          {6, "cpu/prefill/contiguous"},
          {7, "cpu/prefill/paged"},
          {9, "metal/decode/contiguous"},
          {10, "metal/decode/paged"},
          {11, "metal/prefill/contiguous"},
          {12, "metal/prefill/paged"},
      }};
  size_t activation_index = 0;
  for (const ProfileActivationContract& profile : profiles) {
    if (!profile.activation_phase.has_value()) {
      continue;
    }
    ASSERT_LT(activation_index,
              expected_new_profile_activation_order.size());
    const std::string profile_name =
        std::string(contract_token(profile.backend)) + "/" +
        std::string(contract_token(profile.phase)) + "/" +
        std::string(contract_token(profile.layout));
    EXPECT_EQ(*profile.activation_phase,
              expected_new_profile_activation_order[activation_index].first);
    EXPECT_EQ(profile_name,
              expected_new_profile_activation_order[activation_index].second);
    ++activation_index;
  }
  EXPECT_EQ(activation_index, expected_new_profile_activation_order.size());

  constexpr int generic_schema_rename_phase = 1;
  constexpr int capability_validation_pending_phase = 12;
  constexpr int capability_production_supported_phase = 13;
  EXPECT_EQ(generic_schema_rename_phase, 1);
  EXPECT_EQ(capability_validation_pending_phase, 12);
  EXPECT_EQ(capability_production_supported_phase, 13);
  EXPECT_LT(capability_validation_pending_phase,
            capability_production_supported_phase);

  constexpr std::array<int, 7> phases_without_new_profile_activation = {
      0, 1, 2, 3, 5, 8, 13};
  for (int phase : phases_without_new_profile_activation) {
    EXPECT_TRUE(std::none_of(
        profiles.begin(), profiles.end(), [phase](const auto& profile) {
          return profile.activation_phase.has_value() &&
                 *profile.activation_phase == phase;
        }));
  }
}
