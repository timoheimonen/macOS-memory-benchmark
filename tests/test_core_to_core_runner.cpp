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
 * @file test_core_to_core_runner.cpp
 * @brief Unit tests for standalone core-to-core measurement loop
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include <gtest/gtest.h>

#include <cmath>

#include "benchmark/core_to_core_latency_runner_internal.h"
#include "core/config/constants.h"

TEST(CoreToCoreRunnerTest, ExecuteSingleScenarioProducesHeadlineAndSamples) {
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
  };
  ScenarioMeasurement measurement;

  const bool ok = execute_single_scenario(scenario, 1, measurement);

  ASSERT_TRUE(ok);
  EXPECT_GT(measurement.round_trip_ns, 0.0);
  ASSERT_EQ(measurement.samples_ns.size(), 1u);
  EXPECT_GT(measurement.samples_ns[0], 0.0);
  EXPECT_FALSE(std::isnan(measurement.round_trip_ns));
  EXPECT_FALSE(std::isinf(measurement.round_trip_ns));
}

TEST(CoreToCoreRunnerTest, ExecuteSingleScenarioSupportsZeroSamples) {
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_DIFFERENT_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY,
      Constants::CORE_TO_CORE_AFFINITY_TAG_SECONDARY,
  };
  ScenarioMeasurement measurement;

  const bool ok = execute_single_scenario(scenario, 0, measurement);

  ASSERT_TRUE(ok);
  EXPECT_GT(measurement.round_trip_ns, 0.0);
  EXPECT_TRUE(measurement.samples_ns.empty());
  EXPECT_EQ(measurement.initiator_hint.affinity_requested, scenario.apply_affinity);
  EXPECT_EQ(measurement.responder_hint.affinity_requested, scenario.apply_affinity);
  EXPECT_EQ(measurement.initiator_hint.affinity_tag, scenario.initiator_affinity_tag);
  EXPECT_EQ(measurement.responder_hint.affinity_tag, scenario.responder_affinity_tag);
}
