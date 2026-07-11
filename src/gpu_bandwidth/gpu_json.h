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

/** Build one complete or checkpoint GPU schema v1 document. */
nlohmann::ordered_json build_gpu_bandwidth_json(
    const GpuBandwidthConfig& config, const GpuRunResult& result);

/** Atomically write one GPU schema v1 checkpoint. */
int save_gpu_bandwidth_json(const GpuBandwidthConfig& config,
                            const GpuRunResult& result,
                            bool announce_success = false);

#endif  // GPU_JSON_H
