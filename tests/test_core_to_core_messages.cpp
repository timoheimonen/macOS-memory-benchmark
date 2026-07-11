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

#include <gtest/gtest.h>

#include <string>

#include "output/console/messages/messages_api.h"

TEST(CoreToCoreMessagesTest, UsageMentionsStandaloneMode) {
  const std::string usage = Messages::usage_options("memory_benchmark");
  EXPECT_NE(usage.find("--analyze-core2core"), std::string::npos);
  EXPECT_NE(usage.find("acquire/release token-handoff"), std::string::npos);
  EXPECT_NE(usage.find("protocol, coherence, and scheduler effects"), std::string::npos);
  EXPECT_NE(usage.find("core-to-core schema 2"), std::string::npos);
  EXPECT_NE(usage.find("target 250 ms"), std::string::npos);
  EXPECT_NE(usage.find("Defaults to 3 loops"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, ReportLabelsMeasurementAsTokenHandoffProtocol) {
  EXPECT_NE(Messages::msg_running_core_to_core_analysis().find("token-handoff protocol"), std::string::npos);
  EXPECT_NE(Messages::report_core_to_core_header().find("Token-Handoff Protocol"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, StandaloneModeErrorMessageExists) {
  const std::string& msg = Messages::error_analyze_core_to_core_must_be_used_alone();
  EXPECT_NE(msg.find("--analyze-core2core"), std::string::npos);
  EXPECT_NE(msg.find("--output"), std::string::npos);
  EXPECT_NE(msg.find("--count"), std::string::npos);
  EXPECT_NE(msg.find("--latency-samples"), std::string::npos);
  EXPECT_NE(msg.find("--sweep"), std::string::npos);
  EXPECT_NE(msg.find("--help"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, HintStatusMessageContainsRoleAndCodes) {
  const std::string msg = Messages::report_core_to_core_hint_status("Initiator", false, 100, true, false, 46, 1);

  EXPECT_NE(msg.find("Initiator"), std::string::npos);
  EXPECT_NE(msg.find("failed(100)"), std::string::npos);
  EXPECT_NE(msg.find("tag=1"), std::string::npos);
  EXPECT_NE(msg.find("failed(46)"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, CalibratedReportMessagesDescribeAuditState) {
  const std::string config = Messages::report_core_to_core_loop_config(3, 1000, 0.250, 0.100, 0.300, 0.001);
  EXPECT_NE(config.find("250"), std::string::npos);
  EXPECT_NE(config.find("100-300"), std::string::npos);

  const std::string status =
      Messages::report_core_to_core_measurement_status("interrupted", "command-incomplete", 2, 3);
  EXPECT_NE(status.find("interrupted"), std::string::npos);
  EXPECT_NE(status.find("2/3"), std::string::npos);

  const std::string plan = Messages::report_core_to_core_work_plan(100000, 70.0, 350000, 3500000, 14000);
  EXPECT_NE(plan.find("pilot=100000"), std::string::npos);
  EXPECT_NE(plan.find("headline=3500000"), std::string::npos);
}
