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
 * @file test_cli_parser.cpp
 * @brief Unit tests for shared descriptor-driven CLI parser and strict numeric helpers
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "output/console/messages/messages_api.h"

namespace {

enum TestOptionId {
  OPT_CPU_BANDWIDTH = 1,
  OPT_BUFFER_SIZE,
  OPT_OUTPUT,
  OPT_SWEEP,
  OPT_HELP,
};

constexpr std::array<CliOptionSpec, 5> kTestSpecs{{
    {OPT_CPU_BANDWIDTH, "-B", "--cpu-bandwidth", CliOptionValueKind::None, CliOptionArity::Single},
    {OPT_BUFFER_SIZE, "-b", "--buffer-size", CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_OUTPUT, "-o", "--output", CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_SWEEP, "-S", "--sweep", CliOptionValueKind::RequiredValue, CliOptionArity::Repeatable},
    {OPT_HELP, "-h", "--help", CliOptionValueKind::None, CliOptionArity::Single},
}};

template <size_t N>
CliParseResult parse_with_specs(const std::vector<std::string>& args,
                                const std::array<CliOptionSpec, N>& specs) {
  std::vector<std::string> storage = args;
  std::vector<char*> argv;
  argv.reserve(storage.size());
  for (std::string& token : storage) {
    argv.push_back(token.data());
  }
  return parse_cli_options(static_cast<int>(argv.size()), argv.data(), specs);
}

}  // namespace

TEST(CliParserTest, ProgramNameUsesArgvZeroOrFallsBackSafely) {
  std::string prog = "./custom_bench";
  char* valid_argv[] = {prog.data()};
  EXPECT_STREQ(cli_program_name(1, valid_argv), "./custom_bench");

  std::string empty;
  char* empty_argv[] = {empty.data()};
  EXPECT_STREQ(cli_program_name(1, empty_argv), "memory_benchmark");

  char* null_first_argv[] = {nullptr};
  EXPECT_STREQ(cli_program_name(1, null_first_argv), "memory_benchmark");
  EXPECT_STREQ(cli_program_name(0, valid_argv), "memory_benchmark");
  EXPECT_STREQ(cli_program_name(-1, valid_argv), "memory_benchmark");
  EXPECT_STREQ(cli_program_name(1, nullptr), "memory_benchmark");
}

TEST(CliParserTest, ParsesShortLongInlineOpaqueAndRepeatableOptions) {
  const CliParseResult result = parse_with_specs(
      {"program", "-B", "--buffer-size=512", "-o", "-G", "-S", "threads=1,2", "--sweep=buffer-size=64,128"},
      kTestSpecs);

  ASSERT_TRUE(result.ok);
  EXPECT_EQ(result.error_kind, CliParseErrorKind::None);
  ASSERT_EQ(result.options.size(), 5u);

  EXPECT_EQ(result.options[0].id, OPT_CPU_BANDWIDTH);
  EXPECT_EQ(result.options[0].matched_name, "-B");
  EXPECT_STREQ(result.options[0].canonical_name, "--cpu-bandwidth");
  EXPECT_FALSE(result.options[0].has_value);
  EXPECT_TRUE(result.options[0].value.empty());

  EXPECT_EQ(result.options[1].id, OPT_BUFFER_SIZE);
  EXPECT_EQ(result.options[1].matched_name, "--buffer-size");
  EXPECT_STREQ(result.options[1].canonical_name, "--buffer-size");
  EXPECT_TRUE(result.options[1].has_value);
  EXPECT_EQ(result.options[1].value, "512");

  EXPECT_EQ(result.options[2].id, OPT_OUTPUT);
  EXPECT_EQ(result.options[2].matched_name, "-o");
  EXPECT_STREQ(result.options[2].canonical_name, "--output");
  EXPECT_TRUE(result.options[2].has_value);
  EXPECT_EQ(result.options[2].value, "-G");

  EXPECT_EQ(result.options[3].id, OPT_SWEEP);
  EXPECT_EQ(result.options[3].matched_name, "-S");
  EXPECT_STREQ(result.options[3].canonical_name, "--sweep");
  EXPECT_TRUE(result.options[3].has_value);
  EXPECT_EQ(result.options[3].value, "threads=1,2");

  EXPECT_EQ(result.options[4].id, OPT_SWEEP);
  EXPECT_EQ(result.options[4].matched_name, "--sweep");
  EXPECT_STREQ(result.options[4].canonical_name, "--sweep");
  EXPECT_TRUE(result.options[4].has_value);
  EXPECT_EQ(result.options[4].value, "buffer-size=64,128");
}

TEST(CliParserTest, PreservesEmptyAndFlagShapedInlineAndSpaceValues) {
  const CliParseResult empty_inline = parse_with_specs({"program", "--output="}, kTestSpecs);
  ASSERT_TRUE(empty_inline.ok);
  ASSERT_EQ(empty_inline.options.size(), 1u);
  EXPECT_EQ(empty_inline.options[0].matched_name, "--output");
  EXPECT_TRUE(empty_inline.options[0].has_value);
  EXPECT_EQ(empty_inline.options[0].value, "");

  const CliParseResult flag_inline = parse_with_specs({"program", "--output=-G"}, kTestSpecs);
  ASSERT_TRUE(flag_inline.ok);
  ASSERT_EQ(flag_inline.options.size(), 1u);
  EXPECT_EQ(flag_inline.options[0].value, "-G");

  const CliParseResult empty_space = parse_with_specs({"program", "--output", ""}, kTestSpecs);
  ASSERT_TRUE(empty_space.ok);
  ASSERT_EQ(empty_space.options.size(), 1u);
  EXPECT_TRUE(empty_space.options[0].has_value);
  EXPECT_EQ(empty_space.options[0].value, "");

  const CliParseResult stdout_dash = parse_with_specs({"program", "-o", "-"}, kTestSpecs);
  ASSERT_TRUE(stdout_dash.ok);
  ASSERT_EQ(stdout_dash.options.size(), 1u);
  EXPECT_EQ(stdout_dash.options[0].value, "-");
}

TEST(CliParserTest, ReportsDuplicateSingleOptionsAcrossSpellingsAndInlineForms) {
  const std::vector<std::vector<std::string>> duplicate_cases = {
      {"program", "-B", "--cpu-bandwidth"},
      {"program", "-b", "128", "--buffer-size", "256"},
      {"program", "--buffer-size=128", "-b", "256"},
      {"program", "-b", "128", "--buffer-size=256"},
      {"program", "-b", "128", "-b"},
  };

  for (const std::vector<std::string>& args : duplicate_cases) {
    SCOPED_TRACE(testing::PrintToString(args));
    const CliParseResult result = parse_with_specs(args, kTestSpecs);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.error_kind, CliParseErrorKind::DuplicateOption);
    EXPECT_EQ(result.canonical_option, args[1] == "-B" ? "--cpu-bandwidth" : "--buffer-size");
  }
}

TEST(CliParserTest, ReportsMissingValueForTrailingOrNullArgument) {
  for (const std::string& option : {"-b", "--buffer-size", "-o", "--output", "-S", "--sweep"}) {
    SCOPED_TRACE(option);
    const CliParseResult result = parse_with_specs({"program", option}, kTestSpecs);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.error_kind, CliParseErrorKind::MissingValue);
    EXPECT_EQ(result.offending_token, option);
  }

  std::string prog = "program";
  std::string output = "--output";
  char* argv_with_null_value[] = {prog.data(), output.data(), nullptr};
  const CliParseResult null_value = parse_cli_options(3, argv_with_null_value, kTestSpecs);
  EXPECT_FALSE(null_value.ok);
  EXPECT_EQ(null_value.error_kind, CliParseErrorKind::MissingValue);
  EXPECT_EQ(null_value.canonical_option, "--output");
}

TEST(CliParserTest, ReportsUnexpectedValueOnFlagsWithInlineEquals) {
  for (const std::string& token : {"--cpu-bandwidth=1", "--cpu-bandwidth=", "--help=true"}) {
    SCOPED_TRACE(token);
    const CliParseResult result = parse_with_specs({"program", token}, kTestSpecs);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.error_kind, CliParseErrorKind::UnexpectedValue);
    EXPECT_EQ(result.offending_token, token);
    EXPECT_EQ(result.canonical_option, token.rfind("--help", 0) == 0 ? "--help" : "--cpu-bandwidth");
  }
}

TEST(CliParserTest, RejectsUnknownOptionsAndSingleDashInlineForms) {
  for (const std::string& token : {"-b=512", "-buffersize", "--unknown", "--unknown=1", "--=val", "-", "--", ""}) {
    SCOPED_TRACE(token);
    const CliParseResult result = parse_with_specs({"program", token}, kTestSpecs);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.error_kind, CliParseErrorKind::UnknownOption);
    EXPECT_EQ(result.offending_token, token);
  }
}

TEST(CliParserTest, HandlesNullAndEmptyArgvSafely) {
  const CliParseResult empty_argc = parse_cli_options(0, nullptr, kTestSpecs);
  EXPECT_TRUE(empty_argc.ok);
  EXPECT_TRUE(empty_argc.options.empty());

  const CliParseResult single_argc = parse_cli_options(1, nullptr, kTestSpecs);
  EXPECT_TRUE(single_argc.ok);
  EXPECT_TRUE(single_argc.options.empty());

  const CliParseResult null_argv = parse_cli_options(3, nullptr, kTestSpecs);
  EXPECT_TRUE(null_argv.ok);
  EXPECT_TRUE(null_argv.options.empty());

  std::string prog = "program";
  std::string flag = "-h";
  char* argv_with_null_slot[] = {prog.data(), nullptr, flag.data()};
  const CliParseResult skipped_null = parse_cli_options(3, argv_with_null_slot, kTestSpecs);
  ASSERT_TRUE(skipped_null.ok);
  ASSERT_EQ(skipped_null.options.size(), 1u);
  EXPECT_EQ(skipped_null.options[0].id, OPT_HELP);

  char* argv_with_unknown[] = {prog.data(), flag.data()};
  const CliParseResult null_specs = parse_cli_options(2, argv_with_unknown, nullptr, 0);
  EXPECT_FALSE(null_specs.ok);
  EXPECT_EQ(null_specs.error_kind, CliParseErrorKind::UnknownOption);
  EXPECT_EQ(null_specs.offending_token, "-h");
}

TEST(CliParserTest, FormatsAllParseErrorKindsWithAndWithoutModeHint) {
  CliParseResult ok_result;
  EXPECT_TRUE(format_cli_parse_error(ok_result).empty());

  const CliParseResult missing = parse_with_specs({"program", "-b"}, kTestSpecs);
  EXPECT_EQ(format_cli_parse_error(missing), Messages::error_missing_value("--buffer-size"));

  const CliParseResult duplicate = parse_with_specs({"program", "-b", "1", "--buffer-size", "2"}, kTestSpecs);
  EXPECT_EQ(format_cli_parse_error(duplicate), Messages::error_duplicate_option("--buffer-size"));

  const CliParseResult unexpected = parse_with_specs({"program", "--cpu-bandwidth=1"}, kTestSpecs);
  EXPECT_EQ(format_cli_parse_error(unexpected), Messages::error_option_does_not_take_value("--cpu-bandwidth"));

  const CliParseResult unknown = parse_with_specs({"program", "--threads"}, kTestSpecs);
  EXPECT_EQ(format_cli_parse_error(unknown), Messages::error_unknown_option("--threads"));

  const std::string empty_hint;
  EXPECT_EQ(format_cli_parse_error(unknown, &empty_hint), Messages::error_unknown_option("--threads"));

  const std::string mode_hint = Messages::error_gpu_bandwidth_must_be_used_alone();
  EXPECT_EQ(format_cli_parse_error(unknown, &mode_hint),
            Messages::error_option_not_allowed_in_mode("--threads", mode_hint));
}

TEST(CliParserTest, ExtractsLongOptionNameOnlyFromDoubleDashTokens) {
  EXPECT_EQ(extract_long_option_name("--buffer-size=512"), "--buffer-size");
  EXPECT_EQ(extract_long_option_name("--sweep=buffer-size=16,32"), "--sweep");
  EXPECT_EQ(extract_long_option_name("--output="), "--output");
  EXPECT_EQ(extract_long_option_name("--gpu-bandwidth=1"), "--gpu-bandwidth");
  EXPECT_EQ(extract_long_option_name("--cpu-bandwidth"), "--cpu-bandwidth");
  EXPECT_EQ(extract_long_option_name("-b=512"), "-b=512");
  EXPECT_EQ(extract_long_option_name("-B"), "-B");
  EXPECT_EQ(extract_long_option_name("--=value"), "--");
  EXPECT_EQ(extract_long_option_name("--"), "--");
  EXPECT_EQ(extract_long_option_name(""), "");
}

TEST(CliParserTest, RecognizesAllKnownValueOptionsAcrossAllModes) {
  const std::array<std::string_view, 41> expected_value_options{{
      "-b",
      "--buffer-size",
      "-i",
      "--iterations",
      "-r",
      "--count",
      "-t",
      "--threads",
      "-k",
      "--cache-size",
      "-n",
      "--latency-samples",
      "-s",
      "--latency-stride-bytes",
      "-m",
      "--latency-chain-mode",
      "-l",
      "--latency-tlb-locality-kb",
      "-D",
      "--tlb-density",
      "--seed",
      "-o",
      "--output",
      "-S",
      "--sweep",
      "-X",
      "--sweep-max-runs",
      "--llm-memory-backend",
      "--weight-size-mb",
      "--layers",
      "--query-heads",
      "--kv-heads",
      "--head-dim",
      "--kv-element-bytes",
      "--phase",
      "--context-tokens",
      "--prompt-tokens",
      "--attention-query-tile-tokens",
      "--kv-layout",
      "--kv-block-tokens",
      "--batch-size",
  }};
  for (const std::string_view option : expected_value_options) {
    SCOPED_TRACE(std::string(option));
    EXPECT_TRUE(is_known_cli_value_option(option));
  }

  const std::array<std::string_view, 17> non_value_tokens{{
      "-B",
      "--cpu-bandwidth",
      "-L",
      "--cpu-latency",
      "-P",
      "--patterns",
      "-T",
      "--analyze-tlb",
      "-C",
      "--analyze-core2core",
      "-G",
      "--gpu-bandwidth",
      "-M",
      "--llm-memory",
      "--compare-locality",
      "--buffer-size=512",
      "--output=-G",
  }};
  for (const std::string_view token : non_value_tokens) {
    SCOPED_TRACE(std::string(token));
    EXPECT_FALSE(is_known_cli_value_option(token));
  }
}

TEST(CliParserTest, ParsesStrictNonNegativeUlongAcrossFullUnsignedRange) {
  unsigned long out_value = 99;
  long long out_negative = 99;

  EXPECT_EQ(parse_strict_nonnegative_ulong("0", out_value, out_negative),
            StrictNonNegativeUlongStatus::Success);
  EXPECT_EQ(out_value, 0UL);

  EXPECT_EQ(parse_strict_nonnegative_ulong("9223372036854775807", out_value, out_negative),
            StrictNonNegativeUlongStatus::Success);
  EXPECT_EQ(out_value, static_cast<unsigned long>(std::numeric_limits<long long>::max()));

  EXPECT_EQ(parse_strict_nonnegative_ulong("9223372036854775808", out_value, out_negative),
            StrictNonNegativeUlongStatus::Success);
  EXPECT_EQ(out_value, static_cast<unsigned long>(std::numeric_limits<long long>::max()) + 1UL);

  EXPECT_EQ(parse_strict_nonnegative_ulong("18446744073709551615", out_value, out_negative),
            StrictNonNegativeUlongStatus::Success);
  EXPECT_EQ(out_value, std::numeric_limits<unsigned long>::max());

  EXPECT_EQ(parse_strict_nonnegative_ulong("18446744073709551616", out_value, out_negative),
            StrictNonNegativeUlongStatus::OutOfRange);

  EXPECT_EQ(parse_strict_nonnegative_ulong("-1", out_value, out_negative),
            StrictNonNegativeUlongStatus::Negative);
  EXPECT_EQ(out_negative, -1LL);

  EXPECT_EQ(parse_strict_nonnegative_ulong("-9223372036854775808", out_value, out_negative),
            StrictNonNegativeUlongStatus::Negative);
  EXPECT_EQ(out_negative, std::numeric_limits<long long>::min());

  EXPECT_EQ(parse_strict_nonnegative_ulong("-9223372036854775809", out_value, out_negative),
            StrictNonNegativeUlongStatus::OutOfRange);

  for (const std::string& invalid : {"", "-", "-0", "-1x", "+1", "1x", " 1", "1 "}) {
    SCOPED_TRACE(invalid);
    EXPECT_EQ(parse_strict_nonnegative_ulong(invalid, out_value, out_negative),
              StrictNonNegativeUlongStatus::Invalid);
  }
}

TEST(CliParserTest, StrictDecimalErrorReasonsMatchCentralizedMessages) {
  EXPECT_STREQ(strict_signed_decimal_error_reason(StrictIntegerParseStatus::Invalid),
               Messages::reason_signed_integer_invalid().c_str());
  EXPECT_STREQ(strict_signed_decimal_error_reason(StrictIntegerParseStatus::OutOfRange),
               Messages::reason_signed_integer_out_of_range().c_str());
  EXPECT_STREQ(strict_unsigned_decimal_error_reason(StrictIntegerParseStatus::Invalid),
               Messages::reason_unsigned_integer_invalid().c_str());
  EXPECT_STREQ(strict_unsigned_decimal_error_reason(StrictIntegerParseStatus::OutOfRange),
               Messages::reason_unsigned_integer_out_of_range().c_str());
}
