// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file info_messages.cpp
 * @brief Info message implementations
 *
 * Provides implementations for informational message generation functions used
 * throughout the benchmark application. Includes formatted info messages for
 * configuration adjustments and runtime calculations.
 */

#include "messages.h"
#include <sstream>

namespace Messages {

// --- Info Messages ---
std::string info_setting_max_fallback(unsigned long max_mb) {
  std::ostringstream oss;
  oss << "Info: Setting max per buffer to fallback: " << max_mb << " MB.";
  return oss.str();
}

std::string info_calculated_max_less_than_min(unsigned long max_mb, unsigned long min_mb) {
  std::ostringstream oss;
  oss << "Info: Calculated max (" << max_mb << " MB) < min (" << min_mb
      << " MB). Using min.";
  return oss.str();
}

std::string info_custom_cache_rounded_up(unsigned long original_kb, unsigned long rounded_kb) {
  std::ostringstream oss;
  oss << "Info: Custom cache size (" << original_kb << " KB) rounded up to "
      << rounded_kb << " KB (system page size)";
  return oss.str();
}

} // namespace Messages

