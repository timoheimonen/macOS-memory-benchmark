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

/**
 * @file llm_cpu_backend.h
 * @brief CPU adapter for the generic LLM memory backend boundary
 */

#ifndef LLM_CPU_BACKEND_H
#define LLM_CPU_BACKEND_H

#include <memory>

#include "llm_memory/llm_backend.h"

/**
 * Convert unchanged CPU executor evidence into one generic task result.
 *
 * Worker, QoS, CPU timer, and checksum-vector invariants are checked here so
 * the logical runner needs only backend-neutral identity, timing, completion,
 * and validation fields. The returned identity views refer to @p model_plan,
 * @p scenario_plan, and @p context and are intended for synchronous runner
 * validation only.
 *
 * @param model_plan Valid CPU execution plan used by the executor.
 * @param scenario_plan Frozen exact work submitted to the executor.
 * @param context Common task identity supplied by the runner.
 * @param executor_result Unchanged CPU evidence, moved behind the CPU tag.
 * @return A generic success, invalid, or failed task result.
 * @throws std::bad_alloc if retained generic evidence cannot be allocated.
 */
LlmTaskExecutionResult adapt_llm_cpu_executor_result(const LlmMemoryWorkPlan& model_plan,
                                                     const LlmScenarioWorkPlan& scenario_plan,
                                                     const LlmRunnerTaskContext& context,
                                                     LlmExecutorResult executor_result);

/** Create an uninitialized CPU adapter owning its timer and resources. */
std::unique_ptr<LlmBackend> create_llm_cpu_backend();

#endif  // LLM_CPU_BACKEND_H
