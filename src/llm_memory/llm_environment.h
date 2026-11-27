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

#include "core/audit/host_environment.h"

/** Cold-path thermal, power, and physical-memory evidence. */
using LlmHostEnvironmentSnapshot = AuditHostEnvironmentSnapshot;

/**
 * Capture process-wide host state without throwing across the command boundary.
 */
inline LlmHostEnvironmentSnapshot capture_llm_host_environment() noexcept {
  return capture_host_environment();
}

/** Stream SHA-256 of the executable path once on the cold command path. */
inline std::string capture_llm_binary_sha256() noexcept {
  return capture_binary_sha256();
}

#endif  // LLM_ENVIRONMENT_H
