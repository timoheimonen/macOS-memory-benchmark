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
 * @file warmup.h
 * @brief Memory warmup functions for benchmark preparation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides page-prefaulting warmup functions before running
 * latency benchmarks.
 */
#ifndef WARMUP_H
#define WARMUP_H

#include <cstddef>

// --- Latency Warmup Functions ---
/**
 * @brief Warms up memory for latency test by page prefaulting (single thread).
 * @param buffer Pointer to the buffer to warm up
 * @param buffer_size Size of the buffer in bytes
 * @note Uses single-threaded page prefaulting to ensure pages are resident in memory
 */
void warmup_latency(void* buffer, size_t buffer_size);

#endif // WARMUP_H
