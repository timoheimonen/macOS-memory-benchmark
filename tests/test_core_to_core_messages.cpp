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
#include <vector>

#include "output/console/messages/messages_api.h"

TEST(CoreToCoreMessagesTest, HelpersHaveExactOutput) {
  struct MessageCase {
    const char* name;
    std::string actual;
    std::string expected;
  };
  const std::vector<MessageCase> cases = {
      {"running label", Messages::msg_running_core_to_core_analysis(),
       "\nRunning standalone core-to-core token-handoff protocol analysis..."},
      {"report header", Messages::report_core_to_core_header(), "--- Core-to-Core Token-Handoff Protocol Report ---"},
      {"standalone options", Messages::error_analyze_core_to_core_must_be_used_alone(),
       "--analyze-core2core allows only optional -o/--output <file>, -r/--count <count>, and "
       "-n/--latency-samples <count>; sweep mode additionally allows -S/--sweep count=..., "
       "-S/--sweep latency-samples=..., and -X/--sweep-max-runs <count>; -h/--help prints help"},
      {"successful hints", Messages::report_core_to_core_hint_status("Responder", true, 0, true, true, 0, 2),
       "  Responder hints: qos=ok, affinity(tag=2)=ok"},
      {"failed hints", Messages::report_core_to_core_hint_status("Initiator", false, 100, true, false, 46, 1),
       "  Initiator hints: qos=failed(100), affinity(tag=1)=failed(46)"},
      {"affinity not requested", Messages::report_core_to_core_hint_status("Initiator", true, 0, false, false, 0, 0),
       "  Initiator hints: qos=ok, affinity=not requested"},
      {"loop config", Messages::report_core_to_core_loop_config(3, 1000, 0.250, 0.100, 0.300, 0.001),
       "Config: loops=3, samples/loop=1000, calibrated headline target=250 ms (window "
       "100-300 ms), sample-window target=1 ms"},
      {"measurement status",
       Messages::report_core_to_core_measurement_status("interrupted", "command-incomplete", 2, 3),
       "  Status: interrupted (2/3 loops measured), reason=command-incomplete"},
      {"work plan", Messages::report_core_to_core_work_plan(100000, 70.0, 350000, 3500000, 14000),
       "  Calibrated work: pilot=100000 round trips (70.00 ns/round trip), warmup=350000, "
       "headline=3500000, sample window=14000"},
      {"measurement failure", Messages::error_core_to_core_measurement_failed("invalid-headline-elapsed"),
       "Core-to-core measurement failed: invalid-headline-elapsed"},
      {"round trip", Messages::report_core_to_core_round_trip(70.0), "  Median headline round-trip latency: 70.00 ns"},
      {"headline statistics", Messages::report_core_to_core_headline_statistics(3),
       "  Continuous headline repeatability (3 loops):"},
      {"sample statistics", Messages::report_core_to_core_sample_statistics(1000),
       "  Pooled separate sample-window distribution (1000 windows):"},
  };

  for (const MessageCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(test_case.actual, test_case.expected);
  }
}
