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
 * @file output_printer.cpp
 * @brief Console output formatting
 */

#include <iostream>   // Required for std::cout

#include "core/config/version.h"  // SOFTVERSION
#include "output/console/messages/messages_api.h"   // Include centralized messages
#include "output/console/output_printer.h"  // Function declarations

/**
 * @brief Displays help text to standard output (for -h/--help).
 *
 * @param prog_name The name of the executable (typically argv[0])
 */
void print_help(const char *prog_name) {
  const char* safe_prog_name =
      (prog_name != nullptr && prog_name[0] != '\0') ? prog_name : "memory_benchmark";
  std::cout << Messages::usage_header(SOFTVERSION)
            << Messages::usage_options(safe_prog_name)
            << Messages::usage_example(safe_prog_name);
}

void print_runtime_banner() {
  std::cout << Messages::config_header(SOFTVERSION) << std::endl;
  std::cout << Messages::config_copyright() << std::endl;
  std::cout << Messages::config_license() << std::endl;
}
