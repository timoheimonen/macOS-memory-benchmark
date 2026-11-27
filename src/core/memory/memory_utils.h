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
 * @file memory_utils.h
 * @brief Memory utility functions for latency pointer-chain construction
 *
 * This file provides utility functions for pointer-chasing latency chains:
 * - LatencyChainMode conversion, parsing, and resolution helpers
 * - Deterministic test hooks for latency-chain generation unit tests
 * - Pointer-chasing latency chain setup (unseeded and seeded overloads)
 */

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <cstddef>  // size_t
#include <cstdint>  // uint64_t
#include <string>

/**
 * @enum LatencyChainMode
 * @brief Pointer-chain construction policy for latency benchmarks
 */
enum class LatencyChainMode {
  Auto = 0,
  GlobalRandom,
  RandomInBoxRandomBox,
  SameRandomInBoxIncreasingBox,
  DiffRandomInBoxIncreasingBox,
};

/**
 * @brief Convert latency-chain mode enum to canonical CLI/JSON string.
 */
const char* latency_chain_mode_to_string(LatencyChainMode mode);

/**
 * @brief Parse latency-chain mode string.
 * @return true when parsing succeeds, false otherwise.
 */
bool latency_chain_mode_from_string(const std::string& mode_value, LatencyChainMode& out_mode);

/**
 * @brief Resolve effective chain mode from user mode and locality setting.
 */
LatencyChainMode resolve_latency_chain_mode(LatencyChainMode mode, size_t tlb_locality_bytes);

/**
 * @brief Whether a chain mode requires a non-zero locality window.
 */
bool latency_chain_mode_uses_locality(LatencyChainMode mode);

/** Deterministic random input used only by memory utility unit tests. */
struct MemoryUtilsTestHooks {
  uint64_t generated_seed = 0;
};

void set_memory_utils_test_hooks(const MemoryUtilsTestHooks* hooks);

// --- Memory Utility Functions ---
/**
 * @brief Prepare buffer for latency test by setting up pointer-chasing chain
 * @param buffer Pointer to the buffer to initialize
 * @param buffer_size Size of the buffer in bytes
 * @param stride Stride size in bytes between linked pointers
 * @param tlb_locality_bytes Optional TLB-locality window in bytes (0 = global random chain)
 * @param mode Pointer-chain construction policy
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Creates a linked list structure in memory where each pointer points to the
 * next location, enabling pointer-chasing latency measurements.
 */
int setup_latency_chain(void* buffer, size_t buffer_size, size_t stride,
                        size_t tlb_locality_bytes = 0,
                        LatencyChainMode mode = LatencyChainMode::Auto);

/**
 * @brief Seeded overload of setup_latency_chain for reproducible analysis runs.
 */
int setup_latency_chain(void* buffer, size_t buffer_size, size_t stride,
                        size_t tlb_locality_bytes,
                        LatencyChainMode mode,
                        uint64_t deterministic_seed);

#endif // MEMORY_UTILS_H
