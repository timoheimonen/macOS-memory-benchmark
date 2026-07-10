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

#include <cstddef>
#include <iosfwd>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Terminal progress spinner with injectable output and enablement.
 *
 * The default constructor writes to stderr only when stderr is a terminal.
 * The injectable constructor is intended for deterministic rendering tests
 * and callers that already know whether progress output is appropriate.
 * Calls must be serialized by the caller.
 */
class ProgressSpinner {
 public:
  ProgressSpinner();
  ProgressSpinner(std::ostream& output, bool enabled);
  ~ProgressSpinner();

  ProgressSpinner(const ProgressSpinner&) = delete;
  ProgressSpinner& operator=(const ProgressSpinner&) = delete;

  /**
   * @brief Render the next spinner frame and message on the current line.
   * @param message Progress message without the spinner frame.
   */
  void tick(const std::string& message);

  /** @brief Erase the rendered line, if any. Safe to call repeatedly. */
  void clear();

 private:
  std::ostream& output_;
  bool enabled_ = false;
  size_t frame_index_ = 0;
  size_t rendered_width_ = 0;
};

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

/** @brief Clear the shared progress indicator, if it has been rendered. */
void clear_progress();

#endif  // UTILS_H
