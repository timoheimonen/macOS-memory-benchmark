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
#include <mach/mach_error.h>  // For mach_error_string
#include <mach/mach_time.h>   // For mach_absolute_time, mach_timebase_info

#include <cstdlib>  // For exit, EXIT_FAILURE
#include <iostream> // For std::cerr

#include "core/timing/timer.h"
#include "output/console/messages.h"

// Constructor: Initializes the timer by getting the timebase info.
HighResTimer::HighResTimer() {
  // Get the timebase info for converting ticks to nanoseconds.
  kern_return_t kern_ret = mach_timebase_info(&timebase_info);
  if (kern_ret != KERN_SUCCESS) {
    std::cerr << Messages::error_prefix() 
              << Messages::error_mach_timebase_info_failed(mach_error_string(kern_ret)) 
              << std::endl;
    // Could also throw an exception here instead of exiting.
    exit(EXIT_FAILURE);
  }
  // Validate timebase denominator to prevent division by zero
  if (timebase_info.denom == 0) {
    std::cerr << Messages::error_prefix() 
              << "timebase denominator is zero (invalid timebase)" 
              << std::endl;
    exit(EXIT_FAILURE);
  }
  start_ticks = 0;  // Ensure start_ticks is initialized.
}

// start: Records the current time in ticks.
void HighResTimer::start() { start_ticks = mach_absolute_time(); }

// stop: Calculates elapsed time since start() in seconds.
double HighResTimer::stop() {
  uint64_t end = mach_absolute_time();  // Get current time.
  // Calculate elapsed ticks, handling potential timer wrap-around.
  uint64_t elapsed_ticks = (end >= start_ticks) ? (end - start_ticks) : (UINT64_MAX - start_ticks + end + 1);
  // Convert ticks to nanoseconds using the timebase info.
  // Defensive check: ensure denom is not zero (should never happen after constructor validation)
  if (timebase_info.denom == 0) {
    std::cerr << Messages::error_prefix() 
              << "timebase denominator is zero in stop()" 
              << std::endl;
    return 0.0;  // Return 0 to avoid division by zero
  }
  double elapsed_nanos = static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
  // Convert nanoseconds to seconds.
  return elapsed_nanos / 1e9;
}

// stop_ns: Calculates elapsed time since start() in nanoseconds.
double HighResTimer::stop_ns() {
  uint64_t end = mach_absolute_time();  // Get current time.
  // Calculate elapsed ticks, handling potential timer wrap-around.
  uint64_t elapsed_ticks = (end >= start_ticks) ? (end - start_ticks) : (UINT64_MAX - start_ticks + end + 1);
  // Convert ticks to nanoseconds and return.
  // Defensive check: ensure denom is not zero (should never happen after constructor validation)
  if (timebase_info.denom == 0) {
    std::cerr << Messages::error_prefix() 
              << "timebase denominator is zero in stop_ns()" 
              << std::endl;
    return 0.0;  // Return 0 to avoid division by zero
  }
  return static_cast<double>(elapsed_ticks) * timebase_info.numer / timebase_info.denom;
}