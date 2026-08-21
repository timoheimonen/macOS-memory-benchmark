// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_json.h
 * @brief GPU bandwidth schema v1 builder and atomic writer
 */

#ifndef GPU_JSON_H
#define GPU_JSON_H

#include "gpu_bandwidth/gpu_bandwidth.h"
#include "gpu_bandwidth/gpu_runner.h"
#include "third_party/nlohmann/json.hpp"

/**
 * Build one complete or checkpoint GPU schema-v1 document in memory.
 *
 * The builder reflects the supplied terminal or intermediate state without
 * performing I/O. If @p result has no timestamp, the builder supplies the
 * current UTC timestamp.
 *
 * @param config Validated command configuration captured in the document.
 * @param result Immutable GPU execution-state snapshot to serialize.
 * @return An ordered schema-v1 document with the result's status, counters,
 *         completeness flags, measurements, and methodology metadata.
 * @throws std::exception If timestamp, string, or JSON construction fails.
 * @note The function retains no references and is safe to call concurrently
 *       when each caller keeps its inputs immutable for the call duration.
 */
nlohmann::ordered_json build_gpu_bandwidth_json(
    const GpuBandwidthConfig& config, const GpuRunResult& result);

/**
 * Build and atomically replace a file with one GPU schema-v1 checkpoint.
 *
 * This is the legacy file-only adapter. Command code that accepts the stdout
 * sentinel must classify the target first and use `JsonOutputSession` instead.
 * An empty `config.output_file` is a successful no-op.
 *
 * @param config Validated configuration whose `output_file` names a file.
 * @param result Immutable GPU execution-state snapshot to persist.
 * @param announce_success Whether a successful file replacement prints the
 *        centralized save confirmation.
 * @return `EXIT_SUCCESS` for a disabled target or successful atomic replace;
 *         `EXIT_FAILURE` for a contained file-output failure.
 * @throws std::exception If payload construction fails before file writing.
 * @pre `config.output_file`, when non-empty, is a file target rather than the
 *      exact stdout sentinel `-`.
 * @note The function retains no references and is synchronous. Callers must
 *       not mutate the inputs or target path concurrently.
 */
int save_gpu_bandwidth_json(const GpuBandwidthConfig& config,
                            const GpuRunResult& result,
                            bool announce_success = false);

#endif  // GPU_JSON_H
