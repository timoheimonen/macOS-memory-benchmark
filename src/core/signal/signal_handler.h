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
 * The coordinator blocks signals before creating workers, so the workers inherit the
 * blocked mask. The coordinator then checks pending signals between benchmark loops
 * and breaks out cleanly before restoring its exact previous mask.
 *
 * Architecture:
 * - install_signal_handlers() registers the handler (call once in main())
 * - BenchmarkSignalMaskGuard blocks benchmark signals while workers exist and
 *   restores the calling thread's exact previous mask at scope exit
 * - signal_received() checks for pending signals between benchmark phases
 *
 * @note Assembly kernels cannot be interrupted mid-flight — signal checks happen
 *       between test phases and between benchmark loops only.
 * @note The signal handler sets an atomic flag only — no async-signal-unsafe operations.
 */
#ifndef SIGNAL_HANDLER_H
#define SIGNAL_HANDLER_H

#include <signal.h>

/**
 * @brief Scope-bound SIGINT/SIGTERM blocking for benchmark worker creation
 *
 * The constructor adds the benchmark interruption signals to the calling
 * thread's mask and saves the complete previous mask. The destructor restores
 * that exact mask, which makes nested guards safe and preserves signal state
 * owned by callers.
 *
 * @note Construct and destroy the guard on the same thread. Threads created
 *       while the guard is alive inherit the blocked benchmark signals.
 */
class BenchmarkSignalMaskGuard {
 public:
  BenchmarkSignalMaskGuard() noexcept;
  ~BenchmarkSignalMaskGuard() noexcept;

  BenchmarkSignalMaskGuard(const BenchmarkSignalMaskGuard&) = delete;
  BenchmarkSignalMaskGuard& operator=(const BenchmarkSignalMaskGuard&) = delete;
  BenchmarkSignalMaskGuard(BenchmarkSignalMaskGuard&&) = delete;
  BenchmarkSignalMaskGuard& operator=(BenchmarkSignalMaskGuard&&) = delete;

 private:
  sigset_t previous_mask_{};
  bool mask_changed_ = false;
};

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
 * @brief Check if Ctrl+C (or SIGTERM) was received
 *
 * Non-blocking check combining the atomic handler flag and sigpending().
 * Safe to call between benchmark loops or test phases.
 *
 * @return true if a signal was received and the program should stop
 */
bool signal_received();

#endif  // SIGNAL_HANDLER_H
