// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_backend.cpp
 * @brief Pure C++ GPU backend vocabulary helpers
 */

#include "gpu_bandwidth/gpu_backend.h"

const char* gpu_backend_status_to_string(GpuBackendStatus status) {
  switch (status) {
    case GpuBackendStatus::NotRun:
      return "not-run";
    case GpuBackendStatus::Success:
      return "success";
    case GpuBackendStatus::Failed:
      return "failed";
    case GpuBackendStatus::Unsupported:
      return "unsupported";
  }
  return "failed";
}

const char* gpu_command_status_to_string(GpuCommandStatus status) {
  switch (status) {
    case GpuCommandStatus::NotRun:
      return "not-run";
    case GpuCommandStatus::Completed:
      return "completed";
    case GpuCommandStatus::Error:
      return "error";
  }
  return "error";
}

const char* gpu_validation_status_to_string(GpuValidationStatus status) {
  switch (status) {
    case GpuValidationStatus::NotRun:
      return "not-run";
    case GpuValidationStatus::Passed:
      return "passed";
    case GpuValidationStatus::Mismatch:
      return "mismatch";
    case GpuValidationStatus::Error:
      return "error";
    case GpuValidationStatus::NotRunTimerInvalid:
      return "not-run-timer-invalid";
  }
  return "error";
}
