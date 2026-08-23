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
 * @file llm_environment.mm
 * @brief Foundation-backed host environment snapshot for LLM result metadata
 */

#import <Foundation/Foundation.h>

#include "llm_memory/llm_environment.h"

namespace {

const char* thermal_state_token(NSProcessInfoThermalState state) noexcept {
  switch (state) {
    case NSProcessInfoThermalStateNominal:
      return "nominal";
    case NSProcessInfoThermalStateFair:
      return "fair";
    case NSProcessInfoThermalStateSerious:
      return "serious";
    case NSProcessInfoThermalStateCritical:
      return "critical";
  }
  return "unavailable";
}

}  // namespace

LlmHostEnvironmentSnapshot capture_llm_host_environment() noexcept {
  try {
    @autoreleasepool {
      @try {
        LlmHostEnvironmentSnapshot snapshot;
        NSProcessInfo* process_info = NSProcessInfo.processInfo;
        if (process_info == nil) {
          return snapshot;
        }
        snapshot.thermal_state =
            thermal_state_token(process_info.thermalState);
        snapshot.physical_memory_bytes =
            static_cast<uint64_t>(process_info.physicalMemory);
        if (@available(macOS 12.0, *)) {
          snapshot.low_power_mode_available = true;
          snapshot.low_power_mode_enabled = process_info.lowPowerModeEnabled;
        }
        return snapshot;
      } @catch (NSException* exception) {
        static_cast<void>(exception);
        return LlmHostEnvironmentSnapshot{};
      }
    }
  } catch (...) {
    return LlmHostEnvironmentSnapshot{};
  }
}
