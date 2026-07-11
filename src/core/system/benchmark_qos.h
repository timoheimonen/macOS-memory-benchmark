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
 * @file benchmark_qos.h
 * @brief Shared best-effort main-thread QoS preparation
 */

#ifndef BENCHMARK_QOS_H
#define BENCHMARK_QOS_H

#include <functional>

/** @brief Observable outcome of requesting benchmark main-thread QoS. */
struct MainThreadQosResult {
  bool requested = false;  ///< Whether the USER_INTERACTIVE request was attempted.
  bool applied = false;    ///< Whether the platform accepted the request.
  int code = 0;            ///< Exact platform return code.
};

/**
 * @brief Injectable boundary for `pthread_set_qos_class_self_np()`.
 * @return Platform-compatible return code; zero indicates success.
 */
using MainThreadQosSetter = std::function<int()>;

/**
 * @brief Request USER_INTERACTIVE QoS for the calling benchmark thread.
 *
 * Failure is non-fatal and emits the existing centralized warning. The result
 * remains available for console and JSON audit metadata.
 *
 * @param setter Optional deterministic setter used by tests.
 * @return Requested/applied state and the platform return code.
 * @throws Any exception raised by an injected setter. The production system
 *         call path does not throw.
 * @note Changes the QoS class of the calling thread when no setter is supplied.
 * @note Concurrent failure warnings may interleave on the process
 *       standard-error stream; the QoS request itself applies to each caller.
 */
MainThreadQosResult prepare_main_thread_benchmark_qos(
    const MainThreadQosSetter& setter = {});

#endif  // BENCHMARK_QOS_H
