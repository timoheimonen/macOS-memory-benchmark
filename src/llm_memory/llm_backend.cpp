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
 * @file llm_backend.cpp
 * @brief Shared LLM backend status and factory implementation
 */

#include "llm_memory/llm_backend.h"

#include "llm_memory/llm_cpu_backend.h"

std::unique_ptr<LlmBackend> create_llm_backend(LlmMemoryBackend backend) {
  switch (backend) {
    case LlmMemoryBackend::Cpu:
      return create_llm_cpu_backend();
    case LlmMemoryBackend::Metal:
      return nullptr;
  }
  return nullptr;
}

const LlmResourcePreparationResult* get_llm_cpu_preparation(const LlmBackendEvidence& evidence) noexcept {
  const auto* cpu = std::get_if<LlmCpuBackendEvidence>(&evidence.backend_evidence);
  return cpu == nullptr ? nullptr : &cpu->preparation;
}

const LlmExecutorResult* get_llm_cpu_task_evidence(const LlmTaskExecutionResult& result) noexcept {
  const auto* cpu = std::get_if<LlmCpuTaskEvidence>(&result.backend_evidence);
  return cpu == nullptr ? nullptr : &cpu->executor;
}

LlmExecutorResult* get_llm_cpu_task_evidence(LlmTaskExecutionResult& result) noexcept {
  auto* cpu = std::get_if<LlmCpuTaskEvidence>(&result.backend_evidence);
  return cpu == nullptr ? nullptr : &cpu->executor;
}

const char* llm_backend_status_to_string(LlmBackendStatus status) noexcept {
  switch (status) {
    case LlmBackendStatus::NotStarted:
      return "not_started";
    case LlmBackendStatus::Ready:
      return "ready";
    case LlmBackendStatus::Unsupported:
      return "unsupported";
    case LlmBackendStatus::Failed:
      return "failed";
  }
  return "failed";
}

const char* llm_task_execution_status_to_string(LlmTaskExecutionStatus status) noexcept {
  switch (status) {
    case LlmTaskExecutionStatus::NotStarted:
      return "not_started";
    case LlmTaskExecutionStatus::Complete:
      return "complete";
    case LlmTaskExecutionStatus::Invalid:
      return "invalid";
    case LlmTaskExecutionStatus::Unsupported:
      return "unsupported";
    case LlmTaskExecutionStatus::Failed:
      return "failed";
  }
  return "failed";
}
