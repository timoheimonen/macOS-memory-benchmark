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
 * @file system_info.h
 * @brief System information query functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions to query system information such as CPU cores,
 * processor name, memory, and cache sizes.
 */
#ifndef SYSTEM_INFO_H
#define SYSTEM_INFO_H

#include <cstddef>  // size_t
#include <string>

// --- System Info Functions ---
/**
 * @brief Get number of performance cores
 * @return Number of performance cores
 */
int get_performance_cores();

/**
 * @brief Get number of efficiency cores
 * @return Number of efficiency cores
 */
int get_efficiency_cores();

/**
 * @brief Get total logical core count
 * @return Total number of logical cores
 */
int get_total_logical_cores();

/**
 * @brief Get processor model name
 * @return Processor model name as string
 */
std::string get_processor_name();

/**
 * @brief Get available system memory in MB
 * @return Available system memory in megabytes
 */
unsigned long get_available_memory_mb();

/**
 * @brief Get L1 data cache size for performance cores
 * @return L1 data cache size in bytes
 */
size_t get_l1_cache_size();

/**
 * @brief Get L2 cache size for performance cores
 * @return L2 cache size in bytes
 */
size_t get_l2_cache_size();

#endif // SYSTEM_INFO_H

