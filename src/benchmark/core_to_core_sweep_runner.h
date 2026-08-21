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
//
/**
 * @file core_to_core_sweep_runner.h
 * @brief Core-to-core parameter sweep runner.
 */

#ifndef CORE_TO_CORE_SWEEP_RUNNER_H
#define CORE_TO_CORE_SWEEP_RUNNER_H

#include <cstddef>

#include "benchmark/sweep_runner.h"

struct CoreToCoreLatencyConfig;
class JsonOutputSession;

size_t calculate_core_to_core_sweep_run_count(const CoreToCoreLatencyConfig& config);

/**
 * Execute a core-to-core sweep through one command-owned output session.
 *
 * Every logical checkpoint is offered to the session. File targets persist
 * checkpoints atomically; stdout checkpoints are lazy no-ops, leaving the
 * command boundary to emit the returned terminal envelope exactly once after
 * this function has printed its final human message.
 *
 * @param base_config Parsed and validated base configuration.
 * @param output_session Session that must outlive this synchronous call.
 * @return Terminal process status and latest complete in-memory envelope.
 * @throws std::exception If setup or envelope construction fails outside the
 *         contained nested-run executor boundary. The command boundary must
 *         convert propagated exceptions to its return-code error path.
 * @note Nested-run executor exceptions become failed, checkpointed attempts in
 *       the returned envelope rather than propagating from this function.
 * @note Not thread-safe. The session must be installed before worker startup
 *       and must not overlap another output session.
 */
SweepExecutionResult run_core_to_core_latency_sweep(
    const CoreToCoreLatencyConfig& base_config,
    JsonOutputSession& output_session);

#endif  // CORE_TO_CORE_SWEEP_RUNNER_H
