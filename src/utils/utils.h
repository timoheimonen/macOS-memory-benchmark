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
 * @file utils.h
 * @brief General utility functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides general utility functions for thread management and
 * progress indication.
 */
#ifndef UTILS_H
#define UTILS_H

#include <vector>  // std::vector
#include <thread>   // std::thread

// --- Utility Functions ---
/**
 * @brief Join all threads in vector and clear it
 * @param threads Reference to vector of threads to join
 */
void join_threads(std::vector<std::thread>& threads);

/**
 * @brief Show progress indicator (spinner)
 *
 * Displays a simple spinner animation to indicate progress.
 */
void show_progress();

#endif // UTILS_H

