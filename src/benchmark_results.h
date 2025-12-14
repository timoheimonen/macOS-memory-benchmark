// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
#ifndef BENCHMARK_RESULTS_H
#define BENCHMARK_RESULTS_H

#include <cstddef>  // size_t

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkResults;
struct TimingResults;

// Helper function to calculate bandwidth for a single cache level
void calculate_single_bandwidth(size_t buffer_size, int iterations,
                               double read_time, double write_time, double copy_time,
                               double& read_bw_gb_s, double& write_bw_gb_s, double& copy_bw_gb_s);

// Calculate bandwidth results from timing data
void calculate_bandwidth_results(const BenchmarkConfig& config, const TimingResults& timings, 
                                 BenchmarkResults& results);

#endif // BENCHMARK_RESULTS_H
