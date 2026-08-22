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
 * @file llm_kv_layout.cpp
 * @brief Pure checked paged-KV layout planning implementation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "llm_memory/llm_kv_layout.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "core/config/constants.h"
#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

constexpr uint64_t kKvBlockPermutationDomain = 0x4C4C4D4B56504731ULL;
constexpr size_t kHashSerializationBufferEntries = 1024;

bool checked_ceil_divide(size_t value, size_t divisor, size_t& out) {
  if (divisor == 0) {
    return false;
  }
  const size_t quotient = value / divisor;
  const size_t increment = value % divisor != 0 ? 1 : 0;
  return NumericUtils::checked_add(quotient, increment, out);
}

bool checked_subtract(size_t lhs, size_t rhs, size_t& out) {
  if (rhs > lhs) {
    return false;
  }
  out = lhs - rhs;
  return true;
}

bool valid_scenario(LlmScenario scenario) {
  return static_cast<size_t>(scenario) < kLlmScenarioCount;
}

bool scenario_uses_kv(LlmScenario scenario) {
  return scenario == LlmScenario::KvOnly || scenario == LlmScenario::Mixed;
}

bool is_lowercase_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool has_identity_version(std::string_view identity,
                          std::string_view version) {
  return identity.size() > version.size() &&
         identity.compare(0, version.size(), version) == 0 &&
         identity[version.size()] == '|';
}

void append_identity_field(std::string& identity, const char* name,
                           std::string_view value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += std::to_string(value.size());
  identity += ':';
  identity.append(value.data(), value.size());
}

void append_identity_field(std::string& identity, const char* name,
                           const std::string& value) {
  append_identity_field(identity, name, std::string_view(value));
}

void append_identity_field(std::string& identity, const char* name,
                           const char* value) {
  append_identity_field(identity, name, std::string_view(value));
}

template <typename Integer>
void append_identity_field(std::string& identity, const char* name,
                           Integer value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += std::to_string(value);
}

/** Match one canonical length-prefixed identity without constructing text. */
class IdentityMatcher {
 public:
  explicit IdentityMatcher(std::string_view retained) noexcept
      : retained_(retained) {}

  bool literal(std::string_view expected) noexcept {
    if (!valid_ || expected.size() > retained_.size() - cursor_ ||
        retained_.compare(cursor_, expected.size(), expected) != 0) {
      valid_ = false;
      return false;
    }
    cursor_ += expected.size();
    return true;
  }

  bool string_field(std::string_view name, std::string_view value) noexcept {
    return literal("|") && literal(name) && literal("=") &&
           integer(value.size()) && literal(":") && literal(value);
  }

  template <typename Integer>
  bool integer(Integer value) noexcept {
    std::array<char, std::numeric_limits<Integer>::digits10 + 3> encoded{};
    const auto result =
        std::to_chars(encoded.data(), encoded.data() + encoded.size(), value);
    return result.ec == std::errc{} &&
           literal(std::string_view(
               encoded.data(),
               static_cast<size_t>(result.ptr - encoded.data())));
  }

  template <typename Integer>
  bool integer_field(std::string_view name, Integer value) noexcept {
    return literal("|") && literal(name) && literal("=") && integer(value);
  }

  bool complete() const noexcept {
    return valid_ && cursor_ == retained_.size();
  }

 private:
  std::string_view retained_;
  size_t cursor_ = 0;
  bool valid_ = true;
};

bool match_permutation_identity_noalloc(
    const LlmKvPermutationIdentity& permutation) noexcept {
  if (permutation.algorithm_version !=
          Constants::LLM_KV_BLOCK_PERMUTATION_VERSION ||
      permutation.domain != kKvBlockPermutationDomain ||
      permutation.domain_uint64_hex != "0x4c4c4d4b56504731" ||
      !is_lowercase_sha256(permutation.sha256) ||
      permutation.entry_count == 0) {
    return false;
  }
  IdentityMatcher identity(permutation.identity);
  return identity.literal(Constants::LLM_KV_BLOCK_PERMUTATION_VERSION) &&
         identity.integer_field("domain", permutation.domain) &&
         identity.string_field("domain_uint64_hex",
                               permutation.domain_uint64_hex) &&
         identity.integer_field("resolved_seed", permutation.resolved_seed) &&
         identity.integer_field("entry_count", permutation.entry_count) &&
         identity.string_field("sha256", permutation.sha256) &&
         identity.complete();
}

uint64_t splitmix64_next(uint64_t& state) noexcept {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

bool should_poll(size_t processed) {
  return processed != 0 &&
         processed % Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES ==
             0;
}

bool preparation_interrupted(const LlmKvStopRequested& stop_requested) {
  return stop_requested && stop_requested();
}

void discard_table(LlmKvBlockTable& table) {
  std::vector<uint32_t>().swap(table.entries);
  table.valid = false;
}

void initialize_permutation_identity(
    LlmKvPermutationIdentity& permutation,
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed) {
  permutation.algorithm_version =
      Constants::LLM_KV_BLOCK_PERMUTATION_VERSION;
  permutation.domain = kKvBlockPermutationDomain;
  permutation.domain_uint64_hex = "0x4c4c4d4b56504731";
  permutation.resolved_seed = resolved_seed;
  permutation.entry_count = layout.block_table_entries;
}

/** Fill the identity permutation while retaining bounded stop latency. */
template <typename StoreEntry>
bool initialize_block_table_entries(
    size_t entry_count, StoreEntry&& store_entry,
    const LlmKvStopRequested& stop_requested) {
  for (size_t entry = 0; entry < entry_count; ++entry) {
    store_entry(entry, static_cast<uint32_t>(entry));
    if (should_poll(entry + 1) &&
        preparation_interrupted(stop_requested)) {
      return false;
    }
  }
  return !preparation_interrupted(stop_requested);
}

/** Apply the frozen rejection-sampled Fisher-Yates permutation in place. */
bool shuffle_block_table_entries(
    uint32_t* entries, size_t entry_count, uint64_t resolved_seed,
    const LlmKvStopRequested& stop_requested) {
  uint64_t stream_state = resolved_seed;
  size_t permutation_iterations = 0;
  for (size_t index = entry_count; index > 1; --index) {
    const uint64_t bound = static_cast<uint64_t>(index);
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t draw = 0;
    do {
      draw = splitmix64_next(stream_state);
    } while (draw < threshold);
    const size_t swap_index = static_cast<size_t>(draw % bound);
    std::swap(entries[index - 1], entries[swap_index]);
    ++permutation_iterations;
    if (should_poll(permutation_iterations) &&
        preparation_interrupted(stop_requested)) {
      return false;
    }
  }
  return !preparation_interrupted(stop_requested);
}

LlmKvBlockTableValidation validate_llm_kv_block_table_storage(
    size_t expected_entries, const uint32_t* entries, size_t entry_count,
    const LlmKvStopRequested& stop_requested) {
  LlmKvBlockTableValidation result;
  result.expected_entries = expected_entries;
  if (expected_entries == 0 ||
      expected_entries >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      entry_count != expected_entries) {
    result.reason_code = LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH;
    return result;
  }
  if (entries == nullptr) {
    result.reason_code = LlmKvLayoutReason::TABLE_OUTPUT_NULL;
    return result;
  }
  if (!checked_ceil_divide(expected_entries, 8,
                           result.validation_bitset_bytes)) {
    result.reason_code = LlmKvLayoutReason::TRANSIENT_BYTES_OVERFLOW;
    return result;
  }
  if (preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  std::vector<uint8_t> seen;
  try {
    seen.reserve(result.validation_bitset_bytes);
  } catch (const std::bad_alloc&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return result;
  } catch (const std::length_error&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return result;
  }
  const size_t bitset_poll_bytes =
      std::max<size_t>(
          1,
          Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES / 8);
  while (seen.size() < result.validation_bitset_bytes) {
    const size_t chunk_bytes =
        std::min(bitset_poll_bytes,
                 result.validation_bitset_bytes - seen.size());
    try {
      seen.insert(seen.end(), chunk_bytes, uint8_t{0});
    } catch (const std::bad_alloc&) {
      result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
      return result;
    } catch (const std::length_error&) {
      result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
      return result;
    }
    if (preparation_interrupted(stop_requested)) {
      result.interrupted = true;
      result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      return result;
    }
  }

  for (size_t index = 0; index < expected_entries; ++index) {
    const uint32_t value = entries[index];
    result.examined_entries = index + 1;
    if (value == std::numeric_limits<uint32_t>::max()) {
      result.reason_code = LlmKvLayoutReason::TABLE_INVALID_SENTINEL;
      return result;
    }
    if (static_cast<size_t>(value) >= expected_entries) {
      result.reason_code = LlmKvLayoutReason::TABLE_ID_OUT_OF_RANGE;
      return result;
    }
    const size_t byte_index = static_cast<size_t>(value) / 8;
    const uint8_t bit =
        static_cast<uint8_t>(1U << (static_cast<size_t>(value) % 8));
    if ((seen[byte_index] & bit) != 0) {
      result.reason_code = LlmKvLayoutReason::TABLE_DUPLICATE_ID;
      return result;
    }
    seen[byte_index] = static_cast<uint8_t>(seen[byte_index] | bit);
    if (should_poll(result.examined_entries) &&
        preparation_interrupted(stop_requested)) {
      result.interrupted = true;
      result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      return result;
    }
  }

  if (preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  result.valid = true;
  result.reason_code = LlmKvLayoutReason::VALID;
  return result;
}

struct LlmKvTableHashResult {
  bool valid = false;
  bool interrupted = false;
  std::string reason_code = LlmKvLayoutReason::HASH_FAILED;
  std::string sha256;
};

LlmKvTableHashResult hash_block_table_little_endian(
    const uint32_t* entries, size_t total_entries,
    size_t requested_chunk_entries,
    const LlmKvStopRequested& stop_requested) {
  LlmKvTableHashResult result;
  if (requested_chunk_entries == 0) {
    result.reason_code = LlmKvLayoutReason::HASH_CHUNK_ENTRIES_ZERO;
    return result;
  }
  if (preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  std::unique_ptr<HashUtils::Sha256Hasher> hasher;
  try {
    hasher = std::make_unique<HashUtils::Sha256Hasher>();
  } catch (const std::bad_alloc&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.reason_code = LlmKvLayoutReason::HASH_FAILED;
    return result;
  }

  std::array<unsigned char,
             kHashSerializationBufferEntries * sizeof(uint32_t)>
      bytes{};
  size_t offset = 0;
  size_t entries_since_poll = 0;
  while (offset < total_entries) {
    const size_t entries_until_poll =
        Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES -
        entries_since_poll;
    const size_t chunk_entries =
        std::min({requested_chunk_entries, total_entries - offset,
                  kHashSerializationBufferEntries, entries_until_poll});
    for (size_t index = 0; index < chunk_entries; ++index) {
      const uint32_t value = entries[offset + index];
      for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
        bytes[index * sizeof(uint32_t) + byte] =
            static_cast<unsigned char>(value >> (byte * 8U));
      }
    }
    try {
      hasher->update(bytes.data(), chunk_entries * sizeof(uint32_t));
    } catch (const std::bad_alloc&) {
      result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
      return result;
    } catch (...) {
      result.reason_code = LlmKvLayoutReason::HASH_FAILED;
      return result;
    }
    offset += chunk_entries;
    entries_since_poll += chunk_entries;
    if (entries_since_poll ==
        Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES) {
      if (preparation_interrupted(stop_requested)) {
        result.interrupted = true;
        result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
        return result;
      }
      entries_since_poll = 0;
    }
  }
  if (entries_since_poll != 0 &&
      preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }
  try {
    result.sha256 = hasher->finalize_hex();
  } catch (const std::bad_alloc&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.reason_code = LlmKvLayoutReason::HASH_FAILED;
    return result;
  }
  result.valid = true;
  result.reason_code = LlmKvLayoutReason::VALID;
  return result;
}

std::string build_permutation_identity(
    const LlmKvPermutationIdentity& permutation) {
  std::string identity = Constants::LLM_KV_BLOCK_PERMUTATION_VERSION;
  append_identity_field(identity, "domain", permutation.domain);
  append_identity_field(identity, "domain_uint64_hex",
                        permutation.domain_uint64_hex);
  append_identity_field(identity, "resolved_seed",
                        permutation.resolved_seed);
  append_identity_field(identity, "entry_count", permutation.entry_count);
  append_identity_field(identity, "sha256", permutation.sha256);
  return identity;
}

/** Validate and identify a completed table without taking storage ownership. */
template <typename MaterializationResult>
void complete_block_table_materialization(
    MaterializationResult& result, const uint32_t* entries,
    size_t entry_count, size_t hash_chunk_entries,
    const LlmKvStopRequested& stop_requested) {
  result.validation = validate_llm_kv_block_table_storage(
      entry_count, entries, entry_count, stop_requested);
  if (!result.validation.valid) {
    result.interrupted = result.validation.interrupted;
    result.reason_code = result.validation.reason_code;
    return;
  }

  const LlmKvTableHashResult hash = hash_block_table_little_endian(
      entries, entry_count, hash_chunk_entries, stop_requested);
  if (!hash.valid) {
    result.interrupted = hash.interrupted;
    result.reason_code = hash.reason_code;
    return;
  }
  try {
    result.permutation.sha256 = hash.sha256;
    result.permutation.identity =
        build_permutation_identity(result.permutation);
  } catch (const std::bad_alloc&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return;
  } catch (const std::length_error&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return;
  }

  result.valid = true;
  result.reason_code = LlmKvLayoutReason::VALID;
}

size_t rotated_worker(size_t ordinal, size_t rank, size_t worker_count) {
  const size_t start = ordinal % worker_count;
  const size_t distance_to_wrap = worker_count - start;
  return rank >= distance_to_wrap ? rank - distance_to_wrap : start + rank;
}

/** Choose the nearest exact decode-cost prefix; lower boundaries win ties. */
size_t closest_decode_cost_boundary(size_t layer_sequence_cost,
                                    size_t regular_block_cost,
                                    size_t boundary_rank,
                                    size_t active_workers,
                                    size_t minimum_boundary,
                                    size_t maximum_boundary) {
  using WideInteger = unsigned __int128;
  const WideInteger target =
      static_cast<WideInteger>(layer_sequence_cost) * boundary_rank;
  const WideInteger denominator =
      static_cast<WideInteger>(regular_block_cost) * active_workers;
  const WideInteger quotient = target / denominator;

  const size_t lower_unclamped =
      quotient > static_cast<WideInteger>(maximum_boundary)
          ? maximum_boundary
          : static_cast<size_t>(quotient);
  const size_t upper_unclamped =
      lower_unclamped == maximum_boundary ? maximum_boundary
                                          : lower_unclamped + 1;
  const size_t lower =
      std::max(minimum_boundary,
               std::min(maximum_boundary, lower_unclamped));
  const size_t upper =
      std::max(minimum_boundary,
               std::min(maximum_boundary, upper_unclamped));
  const auto distance = [&](size_t boundary) {
    const WideInteger scaled_prefix =
        static_cast<WideInteger>(boundary) * regular_block_cost *
        active_workers;
    return scaled_prefix >= target ? scaled_prefix - target
                                   : target - scaled_prefix;
  };
  return distance(lower) <= distance(upper) ? lower : upper;
}

}  // namespace

LlmKvLayoutPlan build_llm_kv_layout_plan(
    const LlmKvLayoutRequest& request) {
  LlmKvLayoutPlan plan;
  plan.sequence_tokens = request.sequence_tokens;
  plan.kv_block_tokens = request.kv_block_tokens;
  plan.layer_count = request.layer_count;
  plan.batch_size = request.batch_size;
  plan.k_or_v_record_bytes_per_layer =
      request.k_or_v_record_bytes_per_layer;

  if (request.sequence_tokens == 0) {
    plan.reason_code = LlmKvLayoutReason::SEQUENCE_TOKENS_ZERO;
    return plan;
  }
  if (request.kv_block_tokens == 0) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_TOKENS_ZERO;
    return plan;
  }
  if (request.kv_block_tokens >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_TOKENS_EXCEEDS_UINT32;
    return plan;
  }
  if ((request.kv_block_tokens & (request.kv_block_tokens - 1)) != 0) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_TOKENS_NOT_POWER_OF_TWO;
    return plan;
  }
  if (request.layer_count == 0) {
    plan.reason_code = LlmKvLayoutReason::LAYER_COUNT_ZERO;
    return plan;
  }
  if (request.batch_size == 0) {
    plan.reason_code = LlmKvLayoutReason::BATCH_SIZE_ZERO;
    return plan;
  }
  if (request.k_or_v_record_bytes_per_layer == 0) {
    plan.reason_code = LlmKvLayoutReason::RECORD_BYTES_ZERO;
    return plan;
  }

  if (!checked_ceil_divide(request.sequence_tokens,
                           request.kv_block_tokens,
                           plan.blocks_per_sequence) ||
      !NumericUtils::checked_multiply(
          request.batch_size, plan.blocks_per_sequence,
          plan.physical_blocks_per_layer)) {
    plan.reason_code = LlmKvLayoutReason::PHYSICAL_BLOCK_COUNT_OVERFLOW;
    return plan;
  }
  if (plan.physical_blocks_per_layer >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_ID_RANGE_EXCEEDED;
    return plan;
  }
  plan.block_table_entries = plan.physical_blocks_per_layer;

  if (!NumericUtils::checked_multiply(
          request.kv_block_tokens,
          request.k_or_v_record_bytes_per_layer, plan.block_bytes)) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_BYTES_OVERFLOW;
    return plan;
  }
  size_t last_block_start_token = 0;
  if (!NumericUtils::checked_multiply(
          plan.blocks_per_sequence - 1, request.kv_block_tokens,
          last_block_start_token)) {
    plan.reason_code = LlmKvLayoutReason::LAST_BLOCK_OFFSET_OVERFLOW;
    return plan;
  }
  if (!checked_subtract(request.sequence_tokens, last_block_start_token,
                        plan.last_block_tokens) ||
      plan.last_block_tokens == 0 ||
      plan.last_block_tokens > request.kv_block_tokens) {
    plan.reason_code = LlmKvLayoutReason::LAST_BLOCK_TOKENS_INVALID;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          plan.last_block_tokens,
          request.k_or_v_record_bytes_per_layer,
          plan.last_block_valid_bytes)) {
    plan.reason_code = LlmKvLayoutReason::LAST_BLOCK_BYTES_OVERFLOW;
    return plan;
  }
  const size_t append_token_in_last_block =
      (request.sequence_tokens - 1) % request.kv_block_tokens;
  if (!NumericUtils::checked_multiply(
          append_token_in_last_block,
          request.k_or_v_record_bytes_per_layer,
          plan.decode_append_offset_in_last_block)) {
    plan.reason_code = LlmKvLayoutReason::APPEND_OFFSET_OVERFLOW;
    return plan;
  }

  size_t logical_layer_sequences = 0;
  size_t logical_token_records = 0;
  if (!NumericUtils::checked_multiply(
          request.layer_count, request.batch_size,
          logical_layer_sequences) ||
      !NumericUtils::checked_multiply(
          logical_layer_sequences, request.sequence_tokens,
          logical_token_records) ||
      !NumericUtils::checked_multiply(
          logical_token_records,
          request.k_or_v_record_bytes_per_layer,
          plan.memory.k_logical_bytes)) {
    plan.reason_code = LlmKvLayoutReason::LOGICAL_BYTES_OVERFLOW;
    return plan;
  }
  plan.memory.v_logical_bytes = plan.memory.k_logical_bytes;

  if (!NumericUtils::checked_multiply(
          request.layer_count, plan.physical_blocks_per_layer,
          plan.total_physical_blocks)) {
    plan.reason_code = LlmKvLayoutReason::TOTAL_BLOCKS_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          plan.total_physical_blocks, plan.block_bytes,
          plan.memory.k_physical_bytes)) {
    plan.reason_code = LlmKvLayoutReason::PHYSICAL_BYTES_OVERFLOW;
    return plan;
  }
  plan.memory.v_physical_bytes = plan.memory.k_physical_bytes;
  if (!checked_subtract(plan.memory.k_physical_bytes,
                        plan.memory.k_logical_bytes,
                        plan.memory.k_layout_padding_bytes)) {
    plan.reason_code = LlmKvLayoutReason::LAYOUT_PADDING_UNDERFLOW;
    return plan;
  }
  plan.memory.v_layout_padding_bytes =
      plan.memory.k_layout_padding_bytes;

  if (!NumericUtils::checked_multiply(
          plan.block_table_entries,
          Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
          plan.memory.block_table_bytes)) {
    plan.reason_code = LlmKvLayoutReason::BLOCK_TABLE_BYTES_OVERFLOW;
    return plan;
  }
  if (!checked_ceil_divide(plan.block_table_entries, 8,
                           plan.memory.validation_bitset_bytes) ||
      !NumericUtils::checked_add(
          plan.memory.block_table_bytes,
          plan.memory.validation_bitset_bytes,
          plan.memory.transient_peak_bytes)) {
    plan.reason_code = LlmKvLayoutReason::TRANSIENT_BYTES_OVERFLOW;
    return plan;
  }

  size_t physical_kv_bytes = 0;
  if (!NumericUtils::checked_add(plan.memory.k_physical_bytes,
                                 plan.memory.v_physical_bytes,
                                 physical_kv_bytes) ||
      !NumericUtils::checked_add(physical_kv_bytes,
                                 plan.memory.block_table_bytes,
                                 plan.memory.resident_layout_bytes) ||
      !NumericUtils::checked_add(
          plan.memory.resident_layout_bytes,
          plan.memory.validation_bitset_bytes,
          plan.memory.known_owned_peak_bytes)) {
    plan.reason_code = LlmKvLayoutReason::MEMORY_BUDGET_OVERFLOW;
    return plan;
  }

  plan.permutation_iterations = plan.block_table_entries - 1;
  plan.validation_entries = plan.block_table_entries;
  plan.hash_entries = plan.block_table_entries;
  plan.upload_bytes = plan.memory.block_table_bytes;

  plan.geometry_identity =
      Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION;
  append_identity_field(plan.geometry_identity, "kv_layout_version",
                        Constants::LLM_PAGED_KV_LAYOUT_VERSION);
  append_identity_field(plan.geometry_identity, "sequence_tokens",
                        plan.sequence_tokens);
  append_identity_field(plan.geometry_identity, "kv_block_tokens",
                        plan.kv_block_tokens);
  append_identity_field(plan.geometry_identity, "layer_count",
                        plan.layer_count);
  append_identity_field(plan.geometry_identity, "batch_size",
                        plan.batch_size);
  append_identity_field(plan.geometry_identity,
                        "k_or_v_record_bytes_per_layer",
                        plan.k_or_v_record_bytes_per_layer);
  append_identity_field(plan.geometry_identity, "blocks_per_sequence",
                        plan.blocks_per_sequence);
  append_identity_field(plan.geometry_identity,
                        "physical_blocks_per_layer",
                        plan.physical_blocks_per_layer);
  append_identity_field(plan.geometry_identity, "total_physical_blocks",
                        plan.total_physical_blocks);
  append_identity_field(plan.geometry_identity, "block_bytes",
                        plan.block_bytes);
  append_identity_field(plan.geometry_identity, "last_block_tokens",
                        plan.last_block_tokens);
  append_identity_field(plan.geometry_identity, "last_block_valid_bytes",
                        plan.last_block_valid_bytes);
  append_identity_field(plan.geometry_identity,
                        "decode_append_offset_in_last_block",
                        plan.decode_append_offset_in_last_block);
  append_identity_field(plan.geometry_identity, "k_logical_bytes",
                        plan.memory.k_logical_bytes);
  append_identity_field(plan.geometry_identity, "v_logical_bytes",
                        plan.memory.v_logical_bytes);
  append_identity_field(plan.geometry_identity, "k_physical_bytes",
                        plan.memory.k_physical_bytes);
  append_identity_field(plan.geometry_identity, "v_physical_bytes",
                        plan.memory.v_physical_bytes);
  append_identity_field(plan.geometry_identity, "k_layout_padding_bytes",
                        plan.memory.k_layout_padding_bytes);
  append_identity_field(plan.geometry_identity, "v_layout_padding_bytes",
                        plan.memory.v_layout_padding_bytes);
  append_identity_field(plan.geometry_identity, "block_table_entries",
                        plan.block_table_entries);
  append_identity_field(plan.geometry_identity, "block_table_bytes",
                        plan.memory.block_table_bytes);
  append_identity_field(plan.geometry_identity, "validation_bitset_bytes",
                        plan.memory.validation_bitset_bytes);
  append_identity_field(plan.geometry_identity, "transient_peak_bytes",
                        plan.memory.transient_peak_bytes);
  append_identity_field(plan.geometry_identity, "resident_layout_bytes",
                        plan.memory.resident_layout_bytes);
  append_identity_field(plan.geometry_identity, "known_owned_peak_bytes",
                        plan.memory.known_owned_peak_bytes);
  append_identity_field(plan.geometry_identity, "permutation_iterations",
                        plan.permutation_iterations);
  append_identity_field(plan.geometry_identity, "validation_entries",
                        plan.validation_entries);
  append_identity_field(plan.geometry_identity, "hash_entries",
                        plan.hash_entries);
  append_identity_field(plan.geometry_identity, "upload_bytes",
                        plan.upload_bytes);
  plan.valid = true;
  plan.reason_code = LlmKvLayoutReason::VALID;
  return plan;
}

uint64_t derive_llm_kv_permutation_seed(uint64_t base_seed) noexcept {
  return SeedUtils::splitmix64(base_seed ^ kKvBlockPermutationDomain);
}

LlmKvPermutationIdentity build_llm_kv_permutation_identity(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    std::string sha256) {
  LlmKvPermutationIdentity permutation;
  if (!layout.valid ||
      layout.block_table_entries == 0 ||
      !is_lowercase_sha256(sha256)) {
    return permutation;
  }
  initialize_permutation_identity(permutation, layout, resolved_seed);
  permutation.sha256 = std::move(sha256);
  permutation.identity = build_permutation_identity(permutation);
  return permutation;
}

LlmKvBlockTableValidation validate_llm_kv_block_table(
    size_t expected_entries, const std::vector<uint32_t>& entries,
    const LlmKvStopRequested& stop_requested) {
  return validate_llm_kv_block_table_storage(
      expected_entries, entries.data(), entries.size(), stop_requested);
}

LlmKvBlockTable materialize_llm_kv_block_table(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    size_t hash_chunk_entries,
    const LlmKvStopRequested& stop_requested) {
  LlmKvBlockTable result;
  initialize_permutation_identity(result.permutation, layout,
                                  resolved_seed);
  if (!layout.valid || layout.block_table_entries == 0) {
    result.reason_code = layout.reason_code;
    return result;
  }
  if (hash_chunk_entries == 0) {
    result.reason_code = LlmKvLayoutReason::HASH_CHUNK_ENTRIES_ZERO;
    return result;
  }
  if (preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  try {
    result.entries.reserve(layout.block_table_entries);
  } catch (const std::bad_alloc&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    discard_table(result);
    return result;
  } catch (const std::length_error&) {
    result.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    discard_table(result);
    return result;
  }
  if (!initialize_block_table_entries(
          layout.block_table_entries,
          [&result](size_t, uint32_t value) {
            result.entries.push_back(value);
          },
          stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    discard_table(result);
    return result;
  }

  if (!shuffle_block_table_entries(
          result.entries.data(), layout.block_table_entries,
          resolved_seed, stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    discard_table(result);
    return result;
  }

  complete_block_table_materialization(
      result, result.entries.data(), layout.block_table_entries,
      hash_chunk_entries, stop_requested);
  if (!result.valid) {
    discard_table(result);
  }
  return result;
}

LlmKvInPlaceBlockTableMaterialization
materialize_llm_kv_block_table_in_place(
    const LlmKvLayoutPlan& layout, uint64_t resolved_seed,
    uint32_t* entries, size_t entry_count, size_t hash_chunk_entries,
    const LlmKvStopRequested& stop_requested) {
  LlmKvInPlaceBlockTableMaterialization result;
  initialize_permutation_identity(result.permutation, layout,
                                  resolved_seed);
  if (!layout.valid || layout.block_table_entries == 0) {
    result.reason_code = layout.reason_code;
    return result;
  }
  if (hash_chunk_entries == 0) {
    result.reason_code = LlmKvLayoutReason::HASH_CHUNK_ENTRIES_ZERO;
    return result;
  }
  if (entry_count != layout.block_table_entries) {
    result.reason_code = LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH;
    return result;
  }
  if (entries == nullptr) {
    result.reason_code = LlmKvLayoutReason::TABLE_OUTPUT_NULL;
    return result;
  }
  if (preparation_interrupted(stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  if (!initialize_block_table_entries(
          entry_count,
          [entries](size_t entry, uint32_t value) {
            entries[entry] = value;
          },
          stop_requested) ||
      !shuffle_block_table_entries(entries, entry_count, resolved_seed,
                                   stop_requested)) {
    result.interrupted = true;
    result.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return result;
  }

  complete_block_table_materialization(
      result, entries, entry_count, hash_chunk_entries, stop_requested);
  return result;
}

LlmPagedDecodeWorkloadPlan build_llm_paged_decode_workload_plan(
    const LlmKvLayoutPlan& layout, LlmScenario scenario, size_t work_units,
    size_t effective_model_payload_bytes_per_work_unit,
    const LlmKvPermutationIdentity& permutation) {
  LlmPagedDecodeWorkloadPlan plan;
  plan.scenario = scenario;
  plan.work_units = work_units;
  plan.effective_model_payload_bytes_per_work_unit =
      effective_model_payload_bytes_per_work_unit;
  plan.maximum_work_units_by_work_unit_cap =
      Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT;
  if (!layout.valid) {
    plan.reason_code = layout.reason_code;
    return plan;
  }
  plan.layout_geometry_identity = layout.geometry_identity;
  plan.layout_identity =
      serialize_llm_kv_layout_identity(layout, permutation);
  if (plan.layout_identity.empty()) {
    plan.reason_code = LlmKvLayoutReason::INVALID_LAYOUT_IDENTITY;
    return plan;
  }
  if (!valid_scenario(scenario)) {
    plan.reason_code = LlmKvLayoutReason::INVALID_SCENARIO;
    return plan;
  }
  if (work_units == 0) {
    plan.reason_code = LlmKvLayoutReason::WORK_UNIT_COUNT_ZERO;
    return plan;
  }
  if (work_units > plan.maximum_work_units_by_work_unit_cap) {
    plan.reason_code = LlmKvLayoutReason::WORK_UNIT_CAP_EXCEEDED;
    return plan;
  }

  if (scenario_uses_kv(scenario)) {
    size_t doubled_blocks = 0;
    if (!NumericUtils::checked_multiply(layout.blocks_per_sequence, 2,
                                        doubled_blocks) ||
        !NumericUtils::checked_add(
            doubled_blocks, 1,
            plan.layout_metadata_lookup_count_per_layer_sequence)) {
      plan.reason_code = LlmKvLayoutReason::LOOKUP_COUNT_OVERFLOW;
      return plan;
    }
    size_t layer_sequences = 0;
    if (!NumericUtils::checked_multiply(layout.layer_count,
                                        layout.batch_size,
                                        layer_sequences) ||
        !NumericUtils::checked_multiply(
            layer_sequences,
            plan.layout_metadata_lookup_count_per_layer_sequence,
            plan.layout_metadata_lookup_count_per_work_unit)) {
      plan.reason_code = LlmKvLayoutReason::LOOKUP_COUNT_OVERFLOW;
      return plan;
    }
    if (!NumericUtils::checked_multiply(
            plan.layout_metadata_lookup_count_per_work_unit,
            Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
            plan.layout_metadata_read_bytes_per_work_unit)) {
      plan.reason_code = LlmKvLayoutReason::LOOKUP_BYTES_OVERFLOW;
      return plan;
    }
  }

  if (!NumericUtils::checked_add(
          plan.effective_model_payload_bytes_per_work_unit,
          plan.layout_metadata_read_bytes_per_work_unit,
          plan.accounted_bytes_per_work_unit)) {
    plan.reason_code = LlmKvLayoutReason::TASK_TOTAL_OVERFLOW;
    return plan;
  }
  if (plan.accounted_bytes_per_work_unit == 0) {
    plan.reason_code = LlmKvLayoutReason::ACCOUNTED_BYTES_ZERO;
    return plan;
  }
  plan.maximum_work_units_by_guardrail =
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
      plan.accounted_bytes_per_work_unit;
  plan.effective_maximum_work_units =
      std::min(plan.maximum_work_units_by_work_unit_cap,
               plan.maximum_work_units_by_guardrail);
  if (plan.effective_maximum_work_units == 0) {
    plan.reason_code = LlmKvLayoutReason::GUARDRAIL_BELOW_ONE_WORK_UNIT;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          plan.effective_model_payload_bytes_per_work_unit, work_units,
          plan.effective_model_payload_bytes) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_lookup_count_per_work_unit, work_units,
          plan.layout_metadata_lookup_count) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_read_bytes_per_work_unit, work_units,
          plan.layout_metadata_read_bytes) ||
      !NumericUtils::checked_multiply(plan.accounted_bytes_per_work_unit,
                                      work_units,
                                      plan.task_accounted_bytes)) {
    plan.reason_code = LlmKvLayoutReason::TASK_TOTAL_OVERFLOW;
    return plan;
  }
  if (plan.task_accounted_bytes >
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK) {
    plan.reason_code =
        LlmKvLayoutReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED;
    return plan;
  }

  plan.identity = Constants::LLM_PAGED_DECODE_WORKLOAD_IDENTITY_VERSION;
  append_identity_field(plan.identity, "layout_identity",
                        plan.layout_identity);
  append_identity_field(plan.identity, "scenario",
                        llm_scenario_to_string(plan.scenario));
  append_identity_field(plan.identity, "work_units", plan.work_units);
  append_identity_field(
      plan.identity, "effective_model_payload_bytes_per_work_unit",
      plan.effective_model_payload_bytes_per_work_unit);
  append_identity_field(
      plan.identity, "layout_metadata_lookup_count_per_layer_sequence",
      plan.layout_metadata_lookup_count_per_layer_sequence);
  append_identity_field(
      plan.identity, "layout_metadata_lookup_count_per_work_unit",
      plan.layout_metadata_lookup_count_per_work_unit);
  append_identity_field(
      plan.identity, "layout_metadata_read_bytes_per_work_unit",
      plan.layout_metadata_read_bytes_per_work_unit);
  append_identity_field(plan.identity, "accounted_bytes_per_work_unit",
                        plan.accounted_bytes_per_work_unit);
  append_identity_field(plan.identity,
                        "effective_model_payload_bytes",
                        plan.effective_model_payload_bytes);
  append_identity_field(plan.identity,
                        "layout_metadata_lookup_count",
                        plan.layout_metadata_lookup_count);
  append_identity_field(plan.identity,
                        "layout_metadata_read_bytes",
                        plan.layout_metadata_read_bytes);
  append_identity_field(plan.identity, "task_accounted_bytes",
                        plan.task_accounted_bytes);
  append_identity_field(plan.identity, "maximum_work_units_by_guardrail",
                        plan.maximum_work_units_by_guardrail);
  append_identity_field(plan.identity,
                        "maximum_work_units_by_work_unit_cap",
                        plan.maximum_work_units_by_work_unit_cap);
  append_identity_field(plan.identity, "effective_maximum_work_units",
                        plan.effective_maximum_work_units);
  plan.valid = true;
  plan.reason_code = LlmKvLayoutReason::VALID;
  return plan;
}

LlmKvCpuOwnershipPlan build_llm_paged_decode_kv_cpu_ownership_plan(
    const LlmKvLayoutPlan& layout, size_t worker_count,
    const LlmKvStopRequested& stop_requested) {
  LlmKvCpuOwnershipPlan plan;
  plan.worker_count = worker_count;
  if (!layout.valid) {
    plan.reason_code = layout.reason_code;
    return plan;
  }
  plan.layout_geometry_identity = layout.geometry_identity;
  if (worker_count == 0) {
    plan.reason_code = LlmKvLayoutReason::WORKER_COUNT_ZERO;
    return plan;
  }
  if (preparation_interrupted(stop_requested)) {
    plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
    return plan;
  }

  if (!NumericUtils::checked_multiply(layout.layer_count,
                                      layout.batch_size,
                                      plan.layer_sequence_count) ||
      !NumericUtils::checked_multiply(
          plan.layer_sequence_count, layout.blocks_per_sequence,
          plan.total_owned_blocks)) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
    return plan;
  }
  const size_t active_workers =
      std::min(worker_count, layout.blocks_per_sequence);
  size_t assignment_count = 0;
  if (!NumericUtils::checked_multiply(plan.layer_sequence_count,
                                      active_workers,
                                      assignment_count)) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
    return plan;
  }

  size_t kv_record_bytes = 0;
  size_t regular_model_payload = 0;
  size_t regular_block_cost = 0;
  size_t last_model_token_visits = 0;
  size_t last_model_payload = 0;
  size_t last_block_cost = 0;
  size_t regular_block_total = 0;
  size_t layer_sequence_cost = 0;
  if (!NumericUtils::checked_multiply(
          layout.k_or_v_record_bytes_per_layer, 2, kv_record_bytes) ||
      !NumericUtils::checked_multiply(layout.kv_block_tokens,
                                      kv_record_bytes,
                                      regular_model_payload) ||
      !NumericUtils::checked_add(regular_model_payload, 2 * sizeof(uint32_t),
                                 regular_block_cost) ||
      !NumericUtils::checked_add(layout.last_block_tokens, 1,
                                 last_model_token_visits) ||
      !NumericUtils::checked_multiply(last_model_token_visits,
                                      kv_record_bytes,
                                      last_model_payload) ||
      !NumericUtils::checked_add(last_model_payload, 3 * sizeof(uint32_t),
                                 last_block_cost) ||
      !NumericUtils::checked_multiply(layout.blocks_per_sequence - 1,
                                      regular_block_cost,
                                      regular_block_total) ||
      !NumericUtils::checked_add(regular_block_total, last_block_cost,
                                 layer_sequence_cost)) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
    return plan;
  }

  std::vector<LlmKvCpuBlockAssignment> assignments;
  std::vector<size_t> worker_costs;
  size_t total_model_payload = 0;
  size_t total_lookups = 0;
  size_t total_metadata_bytes = 0;
  size_t total_accounted_bytes = 0;
  try {
    assignments.reserve(assignment_count);
    worker_costs.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
      worker_costs.push_back(0);
      if (should_poll(worker + 1) &&
          preparation_interrupted(stop_requested)) {
        plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
        return plan;
      }
    }
    if (preparation_interrupted(stop_requested)) {
      plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      return plan;
    }
    std::vector<size_t> boundaries;
    boundaries.reserve(active_workers + 1);
    boundaries.push_back(0);
    for (size_t rank = 1; rank < active_workers; ++rank) {
      boundaries.push_back(closest_decode_cost_boundary(
          layer_sequence_cost, regular_block_cost, rank, active_workers,
          boundaries[rank - 1] + 1,
          layout.blocks_per_sequence - (active_workers - rank)));
      if (should_poll(rank) && preparation_interrupted(stop_requested)) {
        plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
        return plan;
      }
    }
    boundaries.push_back(layout.blocks_per_sequence);
    if (preparation_interrupted(stop_requested)) {
      plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      return plan;
    }

    size_t processed_assignments = 0;
    for (size_t ordinal = 0; ordinal < plan.layer_sequence_count;
         ++ordinal) {
      const size_t layer = ordinal / layout.batch_size;
      const size_t batch = ordinal % layout.batch_size;
      size_t sequence_accounted_bytes = 0;
      for (size_t rank = 0; rank < active_workers; ++rank) {
        LlmKvCpuBlockAssignment assignment;
        assignment.layer_index = layer;
        assignment.batch_sequence_index = batch;
        assignment.worker_index =
            rotated_worker(ordinal, rank, worker_count);
        assignment.first_logical_block = boundaries[rank];
        assignment.block_count = boundaries[rank + 1] - boundaries[rank];

        const bool owns_last =
            boundaries[rank + 1] == layout.blocks_per_sequence;
        if (!NumericUtils::checked_multiply(
                assignment.block_count, layout.kv_block_tokens,
                assignment.valid_token_count)) {
          plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
          return plan;
        }
        if (owns_last) {
          const size_t terminal_padding_tokens =
              layout.kv_block_tokens - layout.last_block_tokens;
          if (!checked_subtract(assignment.valid_token_count,
                                terminal_padding_tokens,
                                assignment.valid_token_count)) {
            plan.reason_code =
                LlmKvLayoutReason::OWNERSHIP_ACCOUNTING_MISMATCH;
            return plan;
          }
        }

        size_t model_token_visits = assignment.valid_token_count;
        if ((owns_last &&
             !NumericUtils::checked_add(model_token_visits, 1,
                                        model_token_visits)) ||
            !NumericUtils::checked_multiply(
                model_token_visits, kv_record_bytes,
                assignment.model_payload_bytes_per_work_unit) ||
            !NumericUtils::checked_multiply(
                assignment.block_count, 2,
                assignment.layout_metadata_lookup_count_per_work_unit) ||
            (owns_last &&
             !NumericUtils::checked_add(
                 assignment.layout_metadata_lookup_count_per_work_unit, 1,
                 assignment.layout_metadata_lookup_count_per_work_unit)) ||
            !NumericUtils::checked_multiply(
                assignment.layout_metadata_lookup_count_per_work_unit,
                Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
                assignment.layout_metadata_read_bytes_per_work_unit) ||
            !NumericUtils::checked_add(
                assignment.model_payload_bytes_per_work_unit,
                assignment.layout_metadata_read_bytes_per_work_unit,
                assignment.accounted_bytes_per_work_unit) ||
            !NumericUtils::checked_add(
                worker_costs[assignment.worker_index],
                assignment.accounted_bytes_per_work_unit,
                worker_costs[assignment.worker_index]) ||
            !NumericUtils::checked_add(
                sequence_accounted_bytes,
                assignment.accounted_bytes_per_work_unit,
                sequence_accounted_bytes) ||
            !NumericUtils::checked_add(
                total_model_payload,
                assignment.model_payload_bytes_per_work_unit,
                total_model_payload) ||
            !NumericUtils::checked_add(
                total_lookups,
                assignment.layout_metadata_lookup_count_per_work_unit,
                total_lookups) ||
            !NumericUtils::checked_add(
                total_metadata_bytes,
                assignment.layout_metadata_read_bytes_per_work_unit,
                total_metadata_bytes) ||
            !NumericUtils::checked_add(
                total_accounted_bytes,
                assignment.accounted_bytes_per_work_unit,
                total_accounted_bytes)) {
          plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
          return plan;
        }
        assignments.push_back(assignment);
        ++processed_assignments;
        if (should_poll(processed_assignments) &&
            preparation_interrupted(stop_requested)) {
          plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
          return plan;
        }
      }
      if (sequence_accounted_bytes != layer_sequence_cost) {
        plan.reason_code =
            LlmKvLayoutReason::OWNERSHIP_ACCOUNTING_MISMATCH;
        return plan;
      }
    }
    if (preparation_interrupted(stop_requested)) {
      plan.reason_code = LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      return plan;
    }
  } catch (const std::bad_alloc&) {
    plan.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    plan.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  size_t expected_lookups_per_layer_sequence = 0;
  size_t expected_lookups = 0;
  size_t expected_model_token_visits = 0;
  size_t expected_model_payload = 0;
  size_t expected_metadata_bytes = 0;
  size_t expected_accounted_bytes = 0;
  if (!NumericUtils::checked_multiply(layout.blocks_per_sequence, 2,
                                      expected_lookups_per_layer_sequence) ||
      !NumericUtils::checked_add(expected_lookups_per_layer_sequence, 1,
                                 expected_lookups_per_layer_sequence) ||
      !NumericUtils::checked_multiply(
          plan.layer_sequence_count, expected_lookups_per_layer_sequence,
          expected_lookups) ||
      !NumericUtils::checked_add(layout.sequence_tokens, 1,
                                 expected_model_token_visits) ||
      !NumericUtils::checked_multiply(expected_model_token_visits,
                                      kv_record_bytes,
                                      expected_model_payload) ||
      !NumericUtils::checked_multiply(expected_model_payload,
                                      plan.layer_sequence_count,
                                      expected_model_payload) ||
      !NumericUtils::checked_multiply(
          expected_lookups, Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
          expected_metadata_bytes) ||
      !NumericUtils::checked_multiply(layer_sequence_cost,
                                      plan.layer_sequence_count,
                                      expected_accounted_bytes)) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_COUNT_OVERFLOW;
    return plan;
  }
  if (total_lookups != expected_lookups ||
      total_model_payload != expected_model_payload ||
      total_accounted_bytes != expected_accounted_bytes ||
      total_metadata_bytes != expected_metadata_bytes ||
      assignments.size() != assignment_count) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_ACCOUNTING_MISMATCH;
    return plan;
  }

  const auto worker_cost_range =
      std::minmax_element(worker_costs.begin(), worker_costs.end());
  plan.minimum_worker_accounted_bytes_per_work_unit =
      *worker_cost_range.first;
  plan.maximum_worker_accounted_bytes_per_work_unit =
      *worker_cost_range.second;
  if (!checked_subtract(
          plan.maximum_worker_accounted_bytes_per_work_unit,
          plan.minimum_worker_accounted_bytes_per_work_unit,
          plan.worker_accounted_imbalance_bytes_per_work_unit)) {
    plan.reason_code = LlmKvLayoutReason::OWNERSHIP_ACCOUNTING_MISMATCH;
    return plan;
  }
  plan.total_model_payload_bytes_per_work_unit = total_model_payload;
  plan.total_layout_metadata_lookup_count_per_work_unit = total_lookups;
  plan.total_layout_metadata_read_bytes_per_work_unit =
      total_metadata_bytes;
  plan.total_accounted_bytes_per_work_unit = total_accounted_bytes;
  plan.assignments = std::move(assignments);
  plan.worker_accounted_bytes_per_work_unit = std::move(worker_costs);

  plan.identity = Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION;
  append_identity_field(plan.identity, "layout_geometry_identity",
                        layout.geometry_identity);
  append_identity_field(plan.identity, "schedule_version",
                        Constants::LLM_PAGED_CPU_SCHEDULE_VERSION);
  append_identity_field(plan.identity, "worker_count", plan.worker_count);
  append_identity_field(plan.identity, "layer_sequence_count",
                        plan.layer_sequence_count);
  append_identity_field(plan.identity, "total_owned_blocks",
                        plan.total_owned_blocks);
  append_identity_field(plan.identity, "assignment_count",
                        plan.assignments.size());
  append_identity_field(
      plan.identity, "total_model_payload_bytes_per_work_unit",
      plan.total_model_payload_bytes_per_work_unit);
  append_identity_field(
      plan.identity, "total_layout_metadata_lookup_count_per_work_unit",
      plan.total_layout_metadata_lookup_count_per_work_unit);
  append_identity_field(
      plan.identity, "total_layout_metadata_read_bytes_per_work_unit",
      plan.total_layout_metadata_read_bytes_per_work_unit);
  append_identity_field(plan.identity,
                        "total_accounted_bytes_per_work_unit",
                        plan.total_accounted_bytes_per_work_unit);
  append_identity_field(
      plan.identity, "minimum_worker_accounted_bytes_per_work_unit",
      plan.minimum_worker_accounted_bytes_per_work_unit);
  append_identity_field(
      plan.identity, "maximum_worker_accounted_bytes_per_work_unit",
      plan.maximum_worker_accounted_bytes_per_work_unit);
  append_identity_field(
      plan.identity, "worker_accounted_imbalance_bytes_per_work_unit",
      plan.worker_accounted_imbalance_bytes_per_work_unit);
  plan.valid = true;
  plan.reason_code = LlmKvLayoutReason::VALID;
  return plan;
}

LlmKvSegmentPlan build_llm_kv_segment_plan(
    size_t element_count, size_t element_bytes,
    const LlmKvMetalSegmentLimits& limits) {
  LlmKvSegmentPlan plan;
  plan.element_count = element_count;
  plan.element_bytes = element_bytes;
  plan.segment_capacity_bytes = limits.segment_capacity_bytes;
  plan.segment_slot_cap = limits.segment_slot_cap;
  if (limits.segment_capacity_bytes == 0) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_CAPACITY_ZERO;
    return plan;
  }
  if (limits.segment_slot_cap == 0) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_SLOT_CAP_ZERO;
    return plan;
  }
  if (element_bytes == 0) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_ELEMENT_BYTES_ZERO;
    return plan;
  }
  if (element_count == 0) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_ELEMENT_COUNT_ZERO;
    return plan;
  }
  if (element_bytes > limits.segment_capacity_bytes) {
    plan.reason_code =
        LlmKvLayoutReason::SEGMENT_ELEMENT_EXCEEDS_CAPACITY;
    return plan;
  }
  plan.elements_per_segment =
      limits.segment_capacity_bytes / element_bytes;
  if (plan.elements_per_segment == 0) {
    plan.reason_code =
        LlmKvLayoutReason::SEGMENT_ELEMENT_EXCEEDS_CAPACITY;
    return plan;
  }
  if (!checked_ceil_divide(element_count, plan.elements_per_segment,
                           plan.segment_count)) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_ARITHMETIC_OVERFLOW;
    return plan;
  }
  if (plan.segment_count > limits.segment_slot_cap) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP;
    return plan;
  }

  size_t nominal_capacity = 0;
  if (!NumericUtils::checked_multiply(
          limits.segment_slot_cap, plan.elements_per_segment,
          plan.maximum_addressable_elements) ||
      !NumericUtils::checked_multiply(
          plan.maximum_addressable_elements, element_bytes,
          plan.maximum_addressable_bytes) ||
      !NumericUtils::checked_multiply(limits.segment_slot_cap,
                                      limits.segment_capacity_bytes,
                                      nominal_capacity) ||
      !checked_subtract(nominal_capacity,
                        plan.maximum_addressable_bytes,
                        plan.unused_nominal_segment_capacity_bytes) ||
      !NumericUtils::checked_multiply(element_count, element_bytes,
                                      plan.total_length_bytes)) {
    plan.reason_code = LlmKvLayoutReason::SEGMENT_ARITHMETIC_OVERFLOW;
    return plan;
  }

  try {
    plan.segment_lengths.reserve(plan.segment_count);
    size_t remaining_elements = element_count;
    size_t accumulated_length = 0;
    while (remaining_elements != 0) {
      const size_t current_elements =
          std::min(remaining_elements, plan.elements_per_segment);
      size_t current_length = 0;
      if (!NumericUtils::checked_multiply(current_elements, element_bytes,
                                          current_length) ||
          !NumericUtils::checked_add(accumulated_length, current_length,
                                     accumulated_length)) {
        plan.reason_code =
            LlmKvLayoutReason::SEGMENT_ARITHMETIC_OVERFLOW;
        return plan;
      }
      plan.segment_lengths.push_back(current_length);
      remaining_elements -= current_elements;
    }
    if (accumulated_length != plan.total_length_bytes ||
        plan.segment_lengths.size() != plan.segment_count) {
      plan.reason_code =
          LlmKvLayoutReason::SEGMENT_ARITHMETIC_OVERFLOW;
      return plan;
    }
  } catch (const std::bad_alloc&) {
    plan.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    std::vector<size_t>().swap(plan.segment_lengths);
    return plan;
  } catch (const std::length_error&) {
    plan.reason_code = LlmKvLayoutReason::PLANNER_ALLOCATION_FAILED;
    std::vector<size_t>().swap(plan.segment_lengths);
    return plan;
  }

  plan.identity = Constants::LLM_METAL_SEGMENT_LAYOUT_VERSION;
  append_identity_field(plan.identity, "element_count", plan.element_count);
  append_identity_field(plan.identity, "element_bytes", plan.element_bytes);
  append_identity_field(plan.identity, "segment_capacity_bytes",
                        plan.segment_capacity_bytes);
  append_identity_field(plan.identity, "segment_slot_cap",
                        plan.segment_slot_cap);
  append_identity_field(plan.identity, "elements_per_segment",
                        plan.elements_per_segment);
  append_identity_field(plan.identity, "segment_count", plan.segment_count);
  append_identity_field(plan.identity, "maximum_addressable_elements",
                        plan.maximum_addressable_elements);
  append_identity_field(plan.identity, "maximum_addressable_bytes",
                        plan.maximum_addressable_bytes);
  append_identity_field(plan.identity,
                        "unused_nominal_segment_capacity_bytes",
                        plan.unused_nominal_segment_capacity_bytes);
  append_identity_field(plan.identity, "total_length_bytes",
                        plan.total_length_bytes);
  plan.valid = true;
  plan.reason_code = LlmKvLayoutReason::VALID;
  return plan;
}

LlmKvMetalSegmentPlan build_llm_kv_metal_segment_plan(
    const LlmKvLayoutPlan& layout,
    const LlmKvMetalSegmentLimits& limits) {
  LlmKvMetalSegmentPlan plan;
  plan.limits = limits;
  if (!layout.valid) {
    plan.reason_code = layout.reason_code;
    return plan;
  }
  plan.layout_geometry_identity = layout.geometry_identity;
  plan.k_or_v_pool = build_llm_kv_segment_plan(
      layout.total_physical_blocks, layout.block_bytes, limits);
  if (!plan.k_or_v_pool.valid) {
    plan.reason_code = plan.k_or_v_pool.reason_code;
    return plan;
  }
  plan.block_table = build_llm_kv_segment_plan(
      layout.block_table_entries,
      Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES, limits);
  if (!plan.block_table.valid) {
    plan.reason_code = plan.block_table.reason_code;
    return plan;
  }
  plan.identity = Constants::LLM_PAGED_METAL_EXECUTION_IDENTITY_VERSION;
  append_identity_field(plan.identity, "layout_geometry_identity",
                        layout.geometry_identity);
  append_identity_field(plan.identity, "segment_layout_version",
                        Constants::LLM_METAL_SEGMENT_LAYOUT_VERSION);
  append_identity_field(plan.identity, "k_or_v_pool_identity",
                        plan.k_or_v_pool.identity);
  append_identity_field(plan.identity, "block_table_identity",
                        plan.block_table.identity);
  plan.valid = true;
  plan.reason_code = LlmKvLayoutReason::VALID;
  return plan;
}

std::string serialize_llm_kv_layout_identity(
    const LlmKvLayoutPlan& layout,
    const LlmKvPermutationIdentity& permutation) {
  if (!layout.valid ||
      !has_identity_version(
          layout.geometry_identity,
          Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION) ||
      permutation.algorithm_version !=
          Constants::LLM_KV_BLOCK_PERMUTATION_VERSION ||
      permutation.domain != kKvBlockPermutationDomain ||
      permutation.domain_uint64_hex != "0x4c4c4d4b56504731" ||
      !is_lowercase_sha256(permutation.sha256) ||
      permutation.entry_count != layout.block_table_entries ||
      !match_permutation_identity_noalloc(permutation)) {
    return {};
  }
  std::string identity = Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "geometry_identity",
                        layout.geometry_identity);
  append_identity_field(identity, "permutation_algorithm_version",
                        permutation.algorithm_version);
  append_identity_field(identity, "permutation_domain", permutation.domain);
  append_identity_field(identity, "permutation_domain_uint64_hex",
                        permutation.domain_uint64_hex);
  append_identity_field(identity, "permutation_seed",
                        permutation.resolved_seed);
  append_identity_field(identity, "permutation_entry_count",
                        permutation.entry_count);
  append_identity_field(identity, "permutation_sha256",
                        permutation.sha256);
  return identity;
}

bool validate_llm_kv_layout_identity(
    const LlmKvLayoutPlan& layout,
    const LlmKvPermutationIdentity& permutation,
    std::string_view identity) noexcept {
  if (!layout.valid ||
      !has_identity_version(
          layout.geometry_identity,
          Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION) ||
      permutation.entry_count != layout.block_table_entries ||
      !match_permutation_identity_noalloc(permutation)) {
    return false;
  }
  IdentityMatcher matcher(identity);
  return matcher.literal(Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION) &&
         matcher.string_field("geometry_identity",
                              layout.geometry_identity) &&
         matcher.string_field("permutation_algorithm_version",
                              permutation.algorithm_version) &&
         matcher.integer_field("permutation_domain", permutation.domain) &&
         matcher.string_field("permutation_domain_uint64_hex",
                              permutation.domain_uint64_hex) &&
         matcher.integer_field("permutation_seed",
                               permutation.resolved_seed) &&
         matcher.integer_field("permutation_entry_count",
                               permutation.entry_count) &&
         matcher.string_field("permutation_sha256", permutation.sha256) &&
         matcher.complete();
}

std::string serialize_llm_kv_cpu_execution_identity(
    const LlmPagedDecodeWorkloadPlan& workload,
    const LlmKvCpuOwnershipPlan& ownership) {
  if (!workload.valid || !ownership.valid ||
      workload.layout_geometry_identity.empty() ||
      workload.layout_geometry_identity !=
          ownership.layout_geometry_identity ||
      !has_identity_version(
          workload.layout_geometry_identity,
          Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION) ||
      !has_identity_version(
          workload.layout_identity,
          Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION) ||
      !has_identity_version(
          workload.identity,
          Constants::LLM_PAGED_DECODE_WORKLOAD_IDENTITY_VERSION) ||
      !has_identity_version(
          ownership.identity,
          Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION)) {
    return {};
  }
  std::string identity =
      Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION;
  append_identity_field(identity, "workload_identity", workload.identity);
  append_identity_field(identity, "ownership_identity",
                        ownership.identity);
  return identity;
}

std::string serialize_llm_kv_metal_execution_identity(
    const LlmPagedDecodeWorkloadPlan& workload,
    const LlmKvMetalSegmentPlan& segments) {
  if (!workload.valid || !segments.valid ||
      workload.layout_geometry_identity.empty() ||
      workload.layout_geometry_identity !=
          segments.layout_geometry_identity ||
      !has_identity_version(
          workload.layout_geometry_identity,
          Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION) ||
      !has_identity_version(
          workload.layout_identity,
          Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION) ||
      !has_identity_version(
          workload.identity,
          Constants::LLM_PAGED_DECODE_WORKLOAD_IDENTITY_VERSION) ||
      !has_identity_version(
          segments.identity,
          Constants::LLM_PAGED_METAL_EXECUTION_IDENTITY_VERSION)) {
    return {};
  }
  std::string identity =
      Constants::LLM_PAGED_METAL_EXECUTION_IDENTITY_VERSION;
  append_identity_field(identity, "workload_identity", workload.identity);
  append_identity_field(identity, "segment_identity", segments.identity);
  return identity;
}
