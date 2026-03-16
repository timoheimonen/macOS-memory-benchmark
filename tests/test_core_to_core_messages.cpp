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
  EXPECT_NE(usage.find("-analyze-core2core"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, StandaloneModeErrorMessageExists) {
  const std::string& msg = Messages::error_analyze_core_to_core_must_be_used_alone();
  EXPECT_NE(msg.find("-analyze-core2core"), std::string::npos);
  EXPECT_NE(msg.find("-output"), std::string::npos);
  EXPECT_NE(msg.find("-count"), std::string::npos);
  EXPECT_NE(msg.find("-latency-samples"), std::string::npos);
}

TEST(CoreToCoreMessagesTest, HintStatusMessageContainsRoleAndCodes) {
  const std::string msg = Messages::report_core_to_core_hint_status(
      "Initiator",
      false,
      100,
      true,
      false,
      46,
      1);

  EXPECT_NE(msg.find("Initiator"), std::string::npos);
  EXPECT_NE(msg.find("failed(100)"), std::string::npos);
  EXPECT_NE(msg.find("tag=1"), std::string::npos);
  EXPECT_NE(msg.find("failed(46)"), std::string::npos);
}
