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
#include <cstdlib>

struct CoreToCoreLatencyConfig;

size_t calculate_core_to_core_sweep_run_count(const CoreToCoreLatencyConfig& config);

int run_core_to_core_latency_sweep(const CoreToCoreLatencyConfig& base_config);

#endif  // CORE_TO_CORE_SWEEP_RUNNER_H
