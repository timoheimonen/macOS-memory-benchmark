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
 * @file signal_handler.cpp
 * @brief Graceful Ctrl+C signal handling implementation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Implements signal handling for graceful benchmark shutdown. The coordinator blocks
 * SIGINT/SIGTERM before worker creation, workers inherit that mask, and the coordinator
 * uses sigpending() to detect pending signals between test phases.
 *
 * The handler itself only sets an atomic flag — no async-signal-unsafe operations.
 * signal_received() checks both the flag and sigpending() for completeness.
 */

#include "core/signal/signal_handler.h"

#include <atomic>
#include <csignal>
#include <pthread.h>

namespace {

// Atomic flag set by the signal handler
std::atomic<bool> g_signal_received{false};

// Signal handler — only sets atomic flag, nothing async-signal-unsafe
void signal_handler(int /* sig */) {
  g_signal_received.store(true, std::memory_order_relaxed);
}

}  // namespace

BenchmarkSignalMaskGuard::BenchmarkSignalMaskGuard() noexcept {
  sigset_t benchmark_signals;
  sigemptyset(&benchmark_signals);
  sigaddset(&benchmark_signals, SIGINT);
  sigaddset(&benchmark_signals, SIGTERM);

  mask_changed_ =
      pthread_sigmask(SIG_BLOCK, &benchmark_signals, &previous_mask_) == 0;
}

BenchmarkSignalMaskGuard::~BenchmarkSignalMaskGuard() noexcept {
  if (mask_changed_) {
    pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr);
  }
}

void install_signal_handlers() {
  struct sigaction sa{};
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
}

bool signal_received() {
  // Check the atomic flag first (fast path)
  if (g_signal_received.load(std::memory_order_relaxed)) {
    return true;
  }

  // Also check sigpending in case the handler hasn't run yet
  sigset_t pending;
  sigpending(&pending);

  if (sigismember(&pending, SIGINT) || sigismember(&pending, SIGTERM)) {
    g_signal_received.store(true, std::memory_order_relaxed);
    return true;
  }

  return false;
}
