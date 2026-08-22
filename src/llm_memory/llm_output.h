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
 * @file llm_output.h
 * @brief Human-readable output for the CPU LLM memory profile
 */

#ifndef LLM_OUTPUT_H
#define LLM_OUTPUT_H

#include "llm_memory/llm_runner.h"
#include "llm_memory/llm_work_plan.h"

struct LlmResultMetadata;

/**
 * Print exact model-payload geometry, available aggregate headlines, interpretation
 * limits, and evidence-backed quality warnings for one initialized LLM run.
 *
 * Human report lines use `std::cout` and warnings use `std::cerr`, allowing the
 * command-scoped JSON stdout transport to route all human text away from its
 * terminal document. The function retains no references and does not mutate
 * its inputs.
 *
 * @param model_plan Immutable geometry and exact-byte work plan.
 * @param metadata Command environment, cache, and main-thread QoS evidence.
 * @param result Complete or partial runner evidence and measured aggregates.
 */
void print_llm_memory_console_report(const LlmMemoryWorkPlan& model_plan, const LlmResultMetadata& metadata,
                                     const LlmMemoryResult& result);

#endif  // LLM_OUTPUT_H
