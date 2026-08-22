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
 * @file llm_environment.h
 * @brief Process environment snapshot for the LLM memory profile
 */

#ifndef LLM_ENVIRONMENT_H
#define LLM_ENVIRONMENT_H

#include <cstdint>
#include <string>

/** Cold-path thermal, power, and physical-memory evidence. */
struct LlmHostEnvironmentSnapshot {
  std::string thermal_state = "unavailable";
  bool low_power_mode_available = false;
  bool low_power_mode_enabled = false;
  uint64_t physical_memory_bytes = 0;
};

/**
 * Capture process-wide host state without throwing across the command boundary.
 *
 * The values are observational metadata only and do not change benchmark work.
 * Thermal state and Low Power Mode come from `NSProcessInfo`; unavailable
 * platform fields retain explicit availability state rather than fabricated
 * nominal values.
 */
LlmHostEnvironmentSnapshot capture_llm_host_environment() noexcept;

#endif  // LLM_ENVIRONMENT_H
