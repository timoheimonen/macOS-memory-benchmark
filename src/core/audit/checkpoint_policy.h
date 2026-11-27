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
 * @file checkpoint_policy.h
 * @brief Shared bounded checkpoint interval calculation and lifecycle JSON builder
 */

#ifndef CORE_AUDIT_CHECKPOINT_POLICY_H
#define CORE_AUDIT_CHECKPOINT_POLICY_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

#include "third_party/nlohmann/json.hpp"

/** Compute bounded checkpoint loop interval K = max(1, ceil(N / 8)) without overflow. */
inline constexpr size_t calculate_bounded_snapshot_interval_loops(
    size_t planned_loops) noexcept {
  return std::max<size_t>(
      1, planned_loops / 8 + (planned_loops % 8 != 0 ? size_t{1} : size_t{0}));
}

/** Shared checkpoint lifecycle state exported into auditable JSON schemas. */
struct CheckpointLifecycleState {
  bool checkpoint_failed = false;
  size_t prior_file_writer_attempts = 0;
  size_t prior_successful_file_writes = 0;
  std::string_view current_request = "none";
  size_t snapshot_interval_loops = 1;
};

/** Build the 10-key checkpoint_lifecycle JSON object in canonical key order. */
inline nlohmann::ordered_json build_checkpoint_lifecycle_json(
    const CheckpointLifecycleState& state) {
  nlohmann::ordered_json output;
  output["checkpoint_failed"] = state.checkpoint_failed;
  output["observation_point"] = "before-current-snapshot-preparation";
  output["prior_file_writer_attempts"] = state.prior_file_writer_attempts;
  output["prior_successful_file_writes"] = state.prior_successful_file_writes;
  output["current_request"] = std::string(state.current_request);
  output["current_persistence_success"] = nullptr;
  output["snapshot_interval_loops"] = state.snapshot_interval_loops;
  output["checkpoint_policy"] = "bounded-loop-snapshots";
  output["file_checkpoint_failure_is_terminal_and_not_retried"] = true;
  output["stdout_intermediate_checkpoints_are_lazy"] = true;
  return output;
}

#endif  // CORE_AUDIT_CHECKPOINT_POLICY_H
