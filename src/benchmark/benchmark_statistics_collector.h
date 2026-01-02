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
 * @file benchmark_statistics_collector.h
 * @brief Statistics collection and aggregation for benchmark results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides functions to initialize and collect statistics
 * from multiple benchmark loop executions.
 */
#ifndef BENCHMARK_STATISTICS_COLLECTOR_H
#define BENCHMARK_STATISTICS_COLLECTOR_H

// Forward declarations
struct BenchmarkStatistics;
struct BenchmarkConfig;
struct BenchmarkResults;

/**
 * @brief Initialize statistics structure by clearing and pre-allocating vectors
 * @param stats Reference to BenchmarkStatistics structure to initialize
 * @param config Reference to benchmark configuration
 *
 * Clears all result vectors and pre-allocates space based on loop_count
 * to improve performance during statistics collection.
 */
void initialize_statistics(BenchmarkStatistics& stats, const BenchmarkConfig& config);

/**
 * @brief Collect results from a single benchmark loop into statistics
 * @param stats Reference to BenchmarkStatistics structure to update
 * @param loop_results Reference to BenchmarkResults from a single loop
 * @param config Reference to benchmark configuration
 *
 * Aggregates results from a single benchmark loop execution into the
 * statistics structure for later statistical analysis.
 */
void collect_loop_results(BenchmarkStatistics& stats, const BenchmarkResults& loop_results, const BenchmarkConfig& config);

#endif // BENCHMARK_STATISTICS_COLLECTOR_H

