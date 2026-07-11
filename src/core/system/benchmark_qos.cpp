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
 * @file benchmark_qos.cpp
 * @brief Shared best-effort main-thread QoS preparation implementation
 */

#include "core/system/benchmark_qos.h"

#include "output/console/messages/messages_api.h"

#include <mach/mach.h>
#include <pthread/qos.h>

#include <iostream>

MainThreadQosResult prepare_main_thread_benchmark_qos(
    const MainThreadQosSetter& setter) {
  MainThreadQosResult result;
  result.requested = true;
  result.code = setter
                    ? setter()
                    : pthread_set_qos_class_self_np(
                          QOS_CLASS_USER_INTERACTIVE, 0);
  result.applied = result.code == KERN_SUCCESS;
  if (!result.applied) {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_qos_failed(result.code) << std::endl;
  }
  return result;
}
