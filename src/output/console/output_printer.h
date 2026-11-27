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
 * @file output_printer.h
 * @brief Console output and printing functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides functions for printing general CLI help and the shared
 * runtime banner to the console.
 */
#ifndef OUTPUT_PRINTER_H
#define OUTPUT_PRINTER_H

// --- Output/Printing Functions ---
/**
 * @brief Print help text to stdout (for -h/--help)
 * @param prog_name Program name (typically argv[0])
 */
void print_help(const char* prog_name);

/**
 * @brief Print the shared runtime version, copyright, and GPL banner.
 */
void print_runtime_banner();

#endif // OUTPUT_PRINTER_H
