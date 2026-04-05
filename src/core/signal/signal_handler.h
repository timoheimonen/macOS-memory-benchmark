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
 * @file signal_handler.h
 * @brief Graceful Ctrl+C signal handling for benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Provides signal handling infrastructure for graceful shutdown on SIGINT/SIGTERM.
 * The design blocks signals in worker threads so that only the main thread receives
 * interruptions. The main thread can then check for pending signals between benchmark
 * loops and break out cleanly.
 *
 * Architecture:
 * - install_signal_handlers() registers the handler (call once in main())
 * - block_benchmark_signals() must be called before spawning any worker threads
 *   so they inherit the blocked signal mask
 * - signal_received() checks for pending signals (non-blocking, async-signal-safe)
 * - restore_signal_mask() unblocks signals after benchmarks complete
 *
 * @note Assembly kernels cannot be interrupted mid-flight — signal checks happen
 *       between test phases and between benchmark loops only.
 * @note The signal handler sets an atomic flag only — no async-signal-unsafe operations.
 */
#ifndef SIGNAL_HANDLER_H
#define SIGNAL_HANDLER_H

/**
 * @brief Register SIGINT and SIGTERM handlers
 *
 * Installs a signal handler that sets an internal atomic flag when SIGINT (Ctrl+C)
 * or SIGTERM is received. The handler is minimal — it only records that a signal
 * was received.
 *
 * @note Must be called once early in main(), before any benchmark execution.
 */
void install_signal_handlers();

/**
 * @brief Block SIGINT and SIGTERM in the calling thread
 *
 * Must be called before spawning worker threads so they inherit the blocked mask.
 * This ensures only the main thread can receive Ctrl+C, preventing signal delivery
 * to worker threads in the middle of assembly kernels.
 *
 * @note Call after install_signal_handlers() and before any thread creation.
 */
void block_benchmark_signals();

/**
 * @brief Restore original signal mask (unblock SIGINT/SIGTERM)
 *
 * Called after benchmark execution is complete to restore normal signal behavior.
 *
 * @note Safe to call even if block_benchmark_signals() was not called.
 */
void restore_signal_mask();

/**
 * @brief Check if Ctrl+C (or SIGTERM) was received
 *
 * Non-blocking check combining the atomic handler flag and sigpending().
 * Safe to call between benchmark loops or test phases.
 *
 * @return true if a signal was received and the program should stop
 */
bool signal_received();

#endif // SIGNAL_HANDLER_H
