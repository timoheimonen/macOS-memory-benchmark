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
#include <atomic>     // Required for std::atomic (progress indicator)
#include <iostream>   // Required for std::cout
#include <thread>     // Required for std::thread
#include <vector>     // Required for std::vector

#include "utils/benchmark.h"  // Include common definitions/constants

// --- Progress Indicator ---
// Simple spinner for showing progress without affecting performance
// Uses a static counter to cycle through spinner characters
static std::atomic<int> spinner_counter{0};
static const char spinner_chars[] = {'|', '/', '-', '\\'};

void show_progress() {
  int idx = spinner_counter.fetch_add(1, std::memory_order_relaxed) % (sizeof(spinner_chars) / sizeof(spinner_chars[0]));
  std::cout << '\r' << spinner_chars[idx] << " Running tests... " << std::flush;
}

// --- Thread Utility Functions ---
// Joins all threads in the provided vector and clears the vector.
// 'threads': Vector of thread objects to join.
void join_threads(std::vector<std::thread> &threads) {
  for (auto &t : threads) {
    if (t.joinable()) {  // Check if thread is joinable
      t.join();          // Wait for thread completion
    }
  }
  threads.clear();  // Remove thread objects after joining
}