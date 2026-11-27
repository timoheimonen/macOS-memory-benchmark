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
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "core/audit/build_manifest.h"
#include "core/audit/checkpoint_policy.h"
#include "core/audit/deterministic_rng.h"
#include "core/audit/host_environment.h"
#include "llm_memory/llm_environment.h"
#include "utils/seed_utils.h"

#if defined(__has_include)
#if __has_include("../.build-provenance.h")
#include "../.build-provenance.h"
#endif
#endif

namespace {

bool is_lowercase_hex64(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char ch) {
           return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
         });
}

}  // namespace

TEST(AuditPrimitivesTest, HostEnvironmentSnapshotDefaultsAndAliasesMatch) {
  static_assert(std::is_same_v<LlmHostEnvironmentSnapshot, AuditHostEnvironmentSnapshot>);

  const AuditHostEnvironmentSnapshot defaults;
  EXPECT_EQ(defaults.thermal_state, "unavailable");
  EXPECT_FALSE(defaults.low_power_mode_available);
  EXPECT_FALSE(defaults.low_power_mode_enabled);
  EXPECT_EQ(defaults.physical_memory_bytes, 0u);

  const AuditHostEnvironmentSnapshot env = capture_host_environment();
  const LlmHostEnvironmentSnapshot llm_env = capture_llm_host_environment();
  const std::set<std::string> valid_thermal_tokens = {
      "nominal", "fair", "serious", "critical", "unavailable"};
  EXPECT_EQ(valid_thermal_tokens.count(env.thermal_state), 1u);
  EXPECT_EQ(valid_thermal_tokens.count(llm_env.thermal_state), 1u);
  EXPECT_GT(env.physical_memory_bytes, 0u);
  EXPECT_EQ(env.physical_memory_bytes, llm_env.physical_memory_bytes);
  EXPECT_EQ(env.low_power_mode_available, llm_env.low_power_mode_available);

  const std::string sha256 = capture_binary_sha256();
  const std::string llm_sha256 = capture_llm_binary_sha256();
  EXPECT_TRUE(is_lowercase_hex64(sha256));
  EXPECT_EQ(sha256, llm_sha256);
}

TEST(AuditPrimitivesTest, BuildManifestResolvesProvenanceAndFallbacks) {
  const std::string fake_digest(64, 'a');

  const nlohmann::ordered_json unavailable = unavailable_build_manifest_json();
  const std::vector<std::string> expected_keys = {
      "manifest_version", "status",      "reason_code",  "binary_sha256",
      "git_commit",       "git_dirty",   "compiler",     "compile_flags",
      "link_flags",       "target_arch", "sdk",          "min_os"};
  std::vector<std::string> unavailable_keys;
  for (auto it = unavailable.begin(); it != unavailable.end(); ++it) {
    unavailable_keys.push_back(it.key());
  }
  EXPECT_EQ(unavailable_keys, expected_keys);
  EXPECT_EQ(unavailable["manifest_version"], 1);
  EXPECT_EQ(unavailable["status"], "unavailable");
  EXPECT_EQ(unavailable["reason_code"], "build-provenance-not-provided");

  const nlohmann::ordered_json runtime_with_hash =
      build_runtime_provenance_manifest_json(fake_digest);
  EXPECT_EQ(runtime_with_hash["manifest_version"], 1);
  EXPECT_NE(runtime_with_hash["status"], "unavailable");
  EXPECT_EQ(runtime_with_hash["binary_sha256"], fake_digest);

  const nlohmann::ordered_json runtime_without_hash =
      build_runtime_provenance_manifest_json("");
  EXPECT_EQ(runtime_without_hash["status"], "partial");
  EXPECT_EQ(runtime_without_hash["reason_code"], "binary-hash-unavailable");
  EXPECT_TRUE(runtime_without_hash["binary_sha256"].is_null());

  const nlohmann::ordered_json empty_empty = build_provenance_manifest_json("", "");
  EXPECT_EQ(empty_empty["status"], "unavailable");
  EXPECT_EQ(empty_empty["reason_code"], "build-provenance-not-provided");
  EXPECT_TRUE(empty_empty["binary_sha256"].is_null());

  const nlohmann::ordered_json empty_with_hash =
      build_provenance_manifest_json("", fake_digest);
  EXPECT_EQ(empty_with_hash["status"], "partial");
  EXPECT_EQ(empty_with_hash["reason_code"], "build-provenance-not-provided");
  EXPECT_EQ(empty_with_hash["binary_sha256"], fake_digest);

  const nlohmann::ordered_json malformed =
      build_provenance_manifest_json("{bad-json", "");
  EXPECT_EQ(malformed["status"], "unavailable");
  EXPECT_EQ(malformed["reason_code"], "build-provenance-not-provided");

  const nlohmann::ordered_json non_object =
      build_provenance_manifest_json("[1,2,3]", fake_digest);
  EXPECT_EQ(non_object["status"], "partial");
  EXPECT_EQ(non_object["reason_code"], "build-provenance-not-provided");
  EXPECT_EQ(non_object["binary_sha256"], fake_digest);

#if defined(BUILD_MANIFEST_JSON) && defined(LLM_BUILD_MANIFEST_JSON)
  EXPECT_EQ(nlohmann::ordered_json::parse(BUILD_MANIFEST_JSON),
            nlohmann::ordered_json::parse(LLM_BUILD_MANIFEST_JSON));
#endif
}

TEST(AuditPrimitivesTest, CheckpointPolicyIntervalAndLifecycleSchema) {
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(0), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(1), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(8), 1u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(9), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(16), 2u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(17), 3u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(64), 8u);
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(65), 9u);
  constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
  EXPECT_EQ(calculate_bounded_snapshot_interval_loops(kMaxSize), kMaxSize / 8 + 1);

  const CheckpointLifecycleState state{true, 3, 2, "late-command-error-correction", 4};
  const nlohmann::ordered_json json = build_checkpoint_lifecycle_json(state);
  const std::vector<std::string> expected_keys = {
      "checkpoint_failed",
      "observation_point",
      "prior_file_writer_attempts",
      "prior_successful_file_writes",
      "current_request",
      "current_persistence_success",
      "snapshot_interval_loops",
      "checkpoint_policy",
      "file_checkpoint_failure_is_terminal_and_not_retried",
      "stdout_intermediate_checkpoints_are_lazy",
  };
  std::vector<std::string> actual_keys;
  for (auto it = json.begin(); it != json.end(); ++it) {
    actual_keys.push_back(it.key());
  }
  EXPECT_EQ(actual_keys, expected_keys);
  EXPECT_EQ(json["checkpoint_failed"], true);
  EXPECT_EQ(json["observation_point"], "before-current-snapshot-preparation");
  EXPECT_EQ(json["prior_file_writer_attempts"], 3u);
  EXPECT_EQ(json["prior_successful_file_writes"], 2u);
  EXPECT_EQ(json["current_request"], "late-command-error-correction");
  EXPECT_TRUE(json["current_persistence_success"].is_null());
  EXPECT_EQ(json["snapshot_interval_loops"], 4u);
  EXPECT_EQ(json["checkpoint_policy"], "bounded-loop-snapshots");
  EXPECT_EQ(json["file_checkpoint_failure_is_terminal_and_not_retried"], true);
  EXPECT_EQ(json["stdout_intermediate_checkpoints_are_lazy"], true);
}

TEST(AuditPrimitivesTest, DeterministicRngSplitMix64AndFisherYatesGoldens) {
  EXPECT_STREQ(kSplitMix64FisherYatesRejectionVersion,
               "splitmix64-fisher-yates-rejection-v1");

  const std::array<uint64_t, 5> inputs = {
      0ULL, 1ULL, 42ULL, 0x9E3779B97F4A7C15ULL, std::numeric_limits<uint64_t>::max()};
  for (uint64_t value : inputs) {
    EXPECT_EQ(splitmix64_mix(value), SeedUtils::splitmix64(value));
  }
  EXPECT_EQ(splitmix64_mix(0ULL), 0xe220a8397b1dcdafULL);
  EXPECT_EQ(splitmix64_mix(1ULL), 0x910a2dec89025cc1ULL);
  EXPECT_EQ(splitmix64_mix(std::numeric_limits<uint64_t>::max()),
            0xe4d971771b652c20ULL);

  uint64_t state = 12345ULL;
  const uint64_t first = splitmix64_next(state);
  EXPECT_EQ(state, 12345ULL + 0x9E3779B97F4A7C15ULL);
  EXPECT_EQ(first, splitmix64_mix(12345ULL));

  // Empty and singleton shuffles must be no-ops and preserve state.
  std::vector<uint32_t> empty;
  uint64_t untouched_state = 777ULL;
  fisher_yates_rejection_shuffle_with_state(empty, untouched_state);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(untouched_state, 777ULL);

  std::vector<uint32_t> single = {99u};
  fisher_yates_rejection_shuffle_with_state(single, untouched_state);
  EXPECT_EQ(single, (std::vector<uint32_t>{99u}));
  EXPECT_EQ(untouched_state, 777ULL);

  // Known goldens matching LlmMemoryWorkPlanTest.PagedBlockTableMaterializationOwnedAndInPlaceMatchGoldens.
  std::vector<uint32_t> items0(8);
  std::iota(items0.begin(), items0.end(), 0u);
  fisher_yates_rejection_shuffle(items0, 0ULL);
  EXPECT_EQ(items0, (std::vector<uint32_t>{2, 5, 0, 3, 4, 6, 1, 7}));

  std::vector<uint32_t> items42(8);
  std::iota(items42.begin(), items42.end(), 0u);
  fisher_yates_rejection_shuffle(items42, 8109369757063363730ULL);
  EXPECT_EQ(items42, (std::vector<uint32_t>{0, 6, 2, 3, 7, 1, 5, 4}));

  // Determinism and bijection for non-power-of-two sizes.
  for (size_t n : {size_t{17}, size_t{257}}) {
    std::vector<uint32_t> a(n);
    std::vector<uint32_t> b(n);
    std::iota(a.begin(), a.end(), 0u);
    std::iota(b.begin(), b.end(), 0u);
    fisher_yates_rejection_shuffle(a, 0xCAFEBABE12345678ULL);
    fisher_yates_rejection_shuffle(b, 0xCAFEBABE12345678ULL);
    EXPECT_EQ(a, b);
    std::sort(a.begin(), a.end());
    for (size_t i = 0; i < n; ++i) {
      EXPECT_EQ(a[i], static_cast<uint32_t>(i));
    }
  }

  // Bounded draw handles bound <= 1 and rejection-sampling bounds.
  uint64_t bound_state = 0x1234ULL;
  EXPECT_EQ(splitmix64_bounded_draw(bound_state, 0ULL), 0ULL);
  EXPECT_EQ(splitmix64_bounded_draw(bound_state, 1ULL), 0ULL);
  EXPECT_EQ(bound_state, 0x1234ULL);

  for (int i = 0; i < 64; ++i) {
    const uint64_t draw3 = splitmix64_bounded_draw(bound_state, 3ULL);
    EXPECT_LT(draw3, 3ULL);
    constexpr uint64_t kLargeBound = (1ULL << 63) + 1ULL;
    const uint64_t draw_large = splitmix64_bounded_draw(bound_state, kLargeBound);
    EXPECT_LT(draw_large, kLargeBound);
  }
}
