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
/**
 * @file statistics.h
 * @brief Statistics calculation and printing functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions for calculating and printing summary statistics
 * across multiple benchmark loops.
 */
#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>  // std::vector

// --- Statistics Functions ---
/**
 * @brief Print summary statistics after all loops
 * @param loop_count Number of loops executed
 * @param all_read_bw Vector of read bandwidth measurements
 * @param all_write_bw Vector of write bandwidth measurements
 * @param all_copy_bw Vector of copy bandwidth measurements
 * @param all_l1_latency Vector of L1 cache latency measurements
 * @param all_l2_latency Vector of L2 cache latency measurements
 * @param all_l1_read_bw Vector of L1 read bandwidth measurements
 * @param all_l1_write_bw Vector of L1 write bandwidth measurements
 * @param all_l1_copy_bw Vector of L1 copy bandwidth measurements
 * @param all_l2_read_bw Vector of L2 read bandwidth measurements
 * @param all_l2_write_bw Vector of L2 write bandwidth measurements
 * @param all_l2_copy_bw Vector of L2 copy bandwidth measurements
 * @param all_main_mem_latency Vector of main memory latency measurements
 * @param use_custom_cache_size Whether custom cache size is used
 * @param all_custom_latency Vector of custom cache latency measurements
 * @param all_custom_read_bw Vector of custom cache read bandwidth measurements
 * @param all_custom_write_bw Vector of custom cache write bandwidth measurements
 * @param all_custom_copy_bw Vector of custom cache copy bandwidth measurements
 * @param all_main_mem_latency_samples Vector of main memory latency samples
 * @param all_l1_latency_samples Vector of L1 cache latency samples
 * @param all_l2_latency_samples Vector of L2 cache latency samples
 * @param all_custom_latency_samples Vector of custom cache latency samples
 */
void print_statistics(int loop_count,
                      const std::vector<double>& all_read_bw,
                      const std::vector<double>& all_write_bw,
                      const std::vector<double>& all_copy_bw,
                      const std::vector<double>& all_l1_latency,
                      const std::vector<double>& all_l2_latency,
                      const std::vector<double>& all_l1_read_bw,
                      const std::vector<double>& all_l1_write_bw,
                      const std::vector<double>& all_l1_copy_bw,
                      const std::vector<double>& all_l2_read_bw,
                      const std::vector<double>& all_l2_write_bw,
                      const std::vector<double>& all_l2_copy_bw,
                      const std::vector<double>& all_main_mem_latency,
                      bool use_custom_cache_size,
                      const std::vector<double>& all_custom_latency,
                      const std::vector<double>& all_custom_read_bw,
                      const std::vector<double>& all_custom_write_bw,
                      const std::vector<double>& all_custom_copy_bw,
                      const std::vector<double>& all_main_mem_latency_samples,
                      const std::vector<double>& all_l1_latency_samples,
                      const std::vector<double>& all_l2_latency_samples,
                      const std::vector<double>& all_custom_latency_samples);

#endif // STATISTICS_H

