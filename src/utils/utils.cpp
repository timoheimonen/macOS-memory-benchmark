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
 * @file utils.cpp
 * @brief General utility function implementations
 *
 * Provides implementations for general-purpose utility functions including
 * progress indicators and thread management utilities used throughout the
 * benchmark application.
 */

#include "utils/utils.h"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <unistd.h>

namespace {

ProgressSpinner& shared_progress_spinner() {
  static ProgressSpinner spinner;
  return spinner;
}

}  // namespace

ProgressSpinner::ProgressSpinner()
    : ProgressSpinner(std::cerr, isatty(STDERR_FILENO) == 1) {}

ProgressSpinner::ProgressSpinner(std::ostream& output, bool enabled)
    : output_(output), enabled_(enabled) {}

ProgressSpinner::~ProgressSpinner() {
  clear();
}

void ProgressSpinner::tick(const std::string& message) {
  if (!enabled_) {
    return;
  }

  static constexpr char kFrames[] = {'|', '/', '-', '\\'};
  std::string text;
  text.reserve(message.size() + 2);
  text.push_back(kFrames[frame_index_ % (sizeof(kFrames) / sizeof(kFrames[0]))]);
  text.push_back(' ');
  text.append(message);
  ++frame_index_;

  const size_t width = std::max(rendered_width_, text.size());
  text.append(width - text.size(), ' ');
  output_ << '\r' << text << std::flush;
  rendered_width_ = width;
}

void ProgressSpinner::clear() {
  if (!enabled_ || rendered_width_ == 0) {
    return;
  }

  output_ << '\r' << std::string(rendered_width_, ' ') << '\r' << std::flush;
  rendered_width_ = 0;
}

void show_progress() {
  shared_progress_spinner().tick("Running tests...");
}

void clear_progress() {
  shared_progress_spinner().clear();
}

// --- Thread Utility Functions ---
// Joins all threads in the provided vector and clears the vector.
// 'threads': Vector of thread objects to join.
void join_threads(std::vector<std::thread>& threads) {
  for (auto& t : threads) {
    if (t.joinable()) {  // Check if thread is joinable
      t.join();          // Wait for thread completion
    }
  }
  threads.clear();  // Remove thread objects after joining
}
