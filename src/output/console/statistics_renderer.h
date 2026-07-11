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
 * @file statistics_renderer.h
 * @brief Reusable console rendering for descriptive-statistics summaries
 */

#ifndef STATISTICS_RENDERER_H
#define STATISTICS_RENDERER_H

#include "utils/descriptive_statistics.h"

#include <cstddef>
#include <iosfwd>
#include <string>

/**
 * @brief Formatting controls for a descriptive-statistics summary.
 *
 * Message helpers provide the two-space field indentation. The prefixes add
 * any context-specific indentation outside those messages. When
 * `median_from_samples` is true, the sample-aware median message is emitted
 * without a prefix because that message owns its full indentation.
 */
struct StatisticsSummaryRenderOptions {
  int precision = 3;
  int cv_precision = 1;
  std::string line_prefix;
  std::string variability_prefix;
  bool median_from_samples = false;
  size_t sample_count = 0;
  std::string inline_diagnostic;
};

/**
 * @brief Render a complete statistics summary in the canonical field order.
 *
 * @param output Destination stream. Stream ownership remains with the caller.
 * @param statistics Calculated statistics to render.
 * @param options Precision, indentation, sample-median, and diagnostic options.
 */
void render_statistics_summary(
    std::ostream& output, const DescriptiveStatistics& statistics,
    const StatisticsSummaryRenderOptions& options = {});

#endif  // STATISTICS_RENDERER_H
