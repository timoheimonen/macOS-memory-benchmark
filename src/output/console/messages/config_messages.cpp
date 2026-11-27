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
 * @file config_messages.cpp
 * @brief Configuration output messages
 *
 * Provides implementations for configuration display message generation functions.
 * Includes formatted messages for displaying benchmark configuration parameters,
 * system information, and CPU details.
 */

#include "messages_api.h"
#include <sstream>

namespace Messages {

// --- Configuration Output Messages ---
std::string config_header(const std::string& version) {
  std::ostringstream oss;
  oss << "----- macOS-memory-benchmark v" << version << " -----";
  return oss.str();
}

std::string config_copyright() {
  return "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>";
}

std::string config_license() {
  return "This program is free software: you can redistribute it and/or modify\n"
         "it under the terms of the GNU General Public License as published by\n"
         "the Free Software Foundation, either version 3 of the License, or\n"
         "(at your option) any later version.\n"
         "This program is distributed in the hope that it will be useful,\n"
         "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
         "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
         "See <https://www.gnu.org/licenses/> for more details.\n";
}

} // namespace Messages
