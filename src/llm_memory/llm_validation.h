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

/** @file llm_validation.h
 * @brief Fixed, allocation-free observations from existing cold validation walks.
 */
#ifndef LLM_VALIDATION_H
#define LLM_VALIDATION_H

#include <array>
#include <cstddef>
#include <string_view>

/** Bounded coverage kinds; these do not certify untimed or timed access traces. */
enum class LlmColdCheckKind {
  PostValidationStructure,
  KvAppendFinal,
  KvPrefillFinalSamples,
  KvAppendUnchanged,
  KvPaddingCanary
};

/** A verdict is meaningful only when applicable and evaluated. Reasons reference
 * static storage. Partial walks without a mismatch remain unresolved. */
struct LlmColdCheckResult {
  LlmColdCheckKind kind = LlmColdCheckKind::PostValidationStructure;
  bool applicable = false;
  bool evaluated = false;
  bool valid = false;
  std::string_view reason_code = "not-evaluated";
};

using LlmColdChecks = std::array<LlmColdCheckResult, 3>;

/** Construct structure/write/padding slots before any execution can fail. */
inline LlmColdChecks make_llm_cold_checks(bool structure, LlmColdCheckKind write_kind,
                                          bool write, bool padding) noexcept {
  return {{{LlmColdCheckKind::PostValidationStructure, structure},
           {write_kind, write}, {LlmColdCheckKind::KvPaddingCanary, padding}}};
}

/** Resolve one observed set, leaving other sets unchanged. Requires slot < 3 and
 * a reason with static storage lifetime. No allocation or throw. */
inline bool resolve_llm_cold_check(LlmColdChecks& checks, size_t slot, bool valid,
                                   std::string_view reason = "valid") noexcept {
  if (checks[slot].applicable) {
    checks[slot].evaluated = true;
    checks[slot].valid = valid;
    checks[slot].reason_code = reason;
  }
  return valid;
}

#endif
