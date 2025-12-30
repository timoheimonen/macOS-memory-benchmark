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
 * @file timer.h
 * @brief High-resolution timer for benchmark measurements
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides the HighResTimer struct for nanosecond-precision timing
 * using the macOS mach timing API.
 */
#ifndef TIMER_H
#define TIMER_H

#include <cstdint>
#include <optional>
// macOS specific: High-resolution timer
#include <mach/mach_time.h>

// --- High-resolution timer helper ---
/**
 * @struct HighResTimer
 * @brief High-resolution timer using macOS mach_absolute_time()
 *
 * Provides nanosecond-precision timing using the macOS mach timing API.
 * Automatically handles timebase conversion for accurate measurements.
 */
struct HighResTimer {
    uint64_t start_ticks = 0;           ///< Ticks at timer start
    mach_timebase_info_data_t timebase_info; ///< Timebase info for conversion

    /**
     * @brief Create a HighResTimer with validated timebase
     * @return Optional timer; nullopt if timebase initialization fails
     */
    static std::optional<HighResTimer> create();

    /**
     * @brief Start the timer
     */
    void start();

    /**
     * @brief Stop timer and return elapsed time in seconds
     * @return Elapsed time in seconds
     */
    double stop();

    /**
     * @brief Stop timer and return elapsed time in nanoseconds
     * @return Elapsed time in nanoseconds
     */
    double stop_ns();

private:
    /**
     * @brief Private constructor - use create() factory method instead
     */
    HighResTimer();
};

#endif // TIMER_H

