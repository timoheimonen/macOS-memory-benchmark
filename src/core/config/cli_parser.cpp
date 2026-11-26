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
 * @file cli_parser.cpp
 * @brief Implementation of shared descriptor-driven CLI option parsing helpers
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "core/config/cli_parser.h"

#include "core/config/config.h"
#include "output/console/messages/messages_api.h"

#include <limits>
#include <utility>

namespace {

constexpr const char* kDefaultProgramName = "memory_benchmark";

constexpr std::array<std::string_view, 41> kKnownCliValueOptions{{
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

const char* canonical_option_name(const CliOptionSpec& spec) noexcept {
  if (spec.long_name != nullptr) {
    return spec.long_name;
  }
  if (spec.short_name != nullptr) {
    return spec.short_name;
  }
  return "";
}

CliParseResult make_parse_error(CliParseErrorKind kind,
                                std::string offending_token,
                                std::string canonical_option = {}) {
  CliParseResult result;
  result.ok = false;
  result.error_kind = kind;
  result.offending_token = std::move(offending_token);
  result.canonical_option = std::move(canonical_option);
  return result;
}

bool find_long_spec(const CliOptionSpec* specs,
                    size_t spec_count,
                    std::string_view name,
                    size_t& out_index) noexcept {
  for (size_t index = 0; index < spec_count; ++index) {
    if (specs[index].long_name != nullptr && name == specs[index].long_name) {
      out_index = index;
      return true;
    }
  }
  return false;
}

bool find_exact_spec(const CliOptionSpec* specs,
                     size_t spec_count,
                     std::string_view token,
                     size_t& out_index) noexcept {
  for (size_t index = 0; index < spec_count; ++index) {
    if ((specs[index].short_name != nullptr && token == specs[index].short_name) ||
        (specs[index].long_name != nullptr && token == specs[index].long_name)) {
      out_index = index;
      return true;
    }
  }
  return false;
}

}  // namespace

const char* cli_program_name(int argc, char* argv[]) noexcept {
  if (argc > 0 && argv != nullptr && argv[0] != nullptr && argv[0][0] != '\0') {
    return argv[0];
  }
  return kDefaultProgramName;
}

std::string_view extract_long_option_name(std::string_view token) noexcept {
  if (token.size() >= 2 && token[0] == '-' && token[1] == '-') {
    const size_t equals_pos = token.find('=');
    if (equals_pos != std::string_view::npos) {
      return token.substr(0, equals_pos);
    }
  }
  return token;
}

bool is_known_cli_value_option(std::string_view token) noexcept {
  for (const std::string_view known : kKnownCliValueOptions) {
    if (token == known) {
      return true;
    }
  }
  return false;
}

CliParseResult parse_cli_options(int argc,
                                 char* argv[],
                                 const CliOptionSpec* specs,
                                 size_t spec_count) {
  CliParseResult result;
  if (argc <= 1 || argv == nullptr) {
    return result;
  }
  if (specs == nullptr) {
    spec_count = 0;
  }

  std::vector<bool> seen(spec_count, false);

  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) {
      continue;
    }

    const std::string_view token(argv[i]);
    const bool is_long_prefix = token.size() >= 2 && token[0] == '-' && token[1] == '-';
    const size_t equals_pos = is_long_prefix ? token.find('=') : std::string_view::npos;

    if (equals_pos != std::string_view::npos) {
      const std::string_view option_name = token.substr(0, equals_pos);
      const std::string_view inline_value = token.substr(equals_pos + 1);
      size_t spec_index = 0;
      if (!find_long_spec(specs, spec_count, option_name, spec_index)) {
        return make_parse_error(CliParseErrorKind::UnknownOption, std::string(token));
      }

      const CliOptionSpec& spec = specs[spec_index];
      const char* const canonical = canonical_option_name(spec);
      if (spec.value_kind == CliOptionValueKind::None) {
        return make_parse_error(CliParseErrorKind::UnexpectedValue,
                                std::string(token),
                                canonical);
      }
      if (spec.arity == CliOptionArity::Single && seen[spec_index]) {
        return make_parse_error(CliParseErrorKind::DuplicateOption,
                                std::string(token),
                                canonical);
      }

      seen[spec_index] = true;
      CliParsedOption parsed;
      parsed.id = spec.id;
      parsed.matched_name = std::string(option_name);
      parsed.canonical_name = canonical;
      parsed.value = std::string(inline_value);
      parsed.has_value = true;
      result.options.push_back(std::move(parsed));
      continue;
    }

    size_t spec_index = 0;
    if (!find_exact_spec(specs, spec_count, token, spec_index)) {
      return make_parse_error(CliParseErrorKind::UnknownOption, std::string(token));
    }

    const CliOptionSpec& spec = specs[spec_index];
    const char* const canonical = canonical_option_name(spec);
    if (spec.arity == CliOptionArity::Single && seen[spec_index]) {
      return make_parse_error(CliParseErrorKind::DuplicateOption,
                              std::string(token),
                              canonical);
    }

    CliParsedOption parsed;
    parsed.id = spec.id;
    parsed.matched_name = std::string(token);
    parsed.canonical_name = canonical;

    if (spec.value_kind == CliOptionValueKind::RequiredValue) {
      ++i;
      if (i >= argc || argv[i] == nullptr) {
        return make_parse_error(CliParseErrorKind::MissingValue,
                                std::string(token),
                                canonical);
      }
      parsed.value = argv[i];
      parsed.has_value = true;
    }

    seen[spec_index] = true;
    result.options.push_back(std::move(parsed));
  }

  return result;
}

std::string format_cli_parse_error(const CliParseResult& result,
                                   const std::string* mode_allowed_hint) {
  switch (result.error_kind) {
    case CliParseErrorKind::None:
      return {};
    case CliParseErrorKind::MissingValue:
      return Messages::error_missing_value(result.canonical_option);
    case CliParseErrorKind::DuplicateOption:
      return Messages::error_duplicate_option(result.canonical_option);
    case CliParseErrorKind::UnexpectedValue:
      return Messages::error_option_does_not_take_value(result.canonical_option);
    case CliParseErrorKind::UnknownOption:
      if (mode_allowed_hint != nullptr && !mode_allowed_hint->empty()) {
        return Messages::error_option_not_allowed_in_mode(result.offending_token,
                                                          *mode_allowed_hint);
      }
      return Messages::error_unknown_option(result.offending_token);
  }
  return {};
}

StrictNonNegativeUlongStatus parse_strict_nonnegative_ulong(const std::string& value,
                                                            unsigned long& out_value,
                                                            long long& out_negative_value) {
  if (!value.empty() && value[0] == '-') {
    long long signed_value = 0;
    const StrictIntegerParseStatus status = parse_strict_signed_decimal(value, signed_value);
    if (status == StrictIntegerParseStatus::Success && signed_value < 0) {
      out_negative_value = signed_value;
      return StrictNonNegativeUlongStatus::Negative;
    }
    if (status == StrictIntegerParseStatus::OutOfRange) {
      return StrictNonNegativeUlongStatus::OutOfRange;
    }
    return StrictNonNegativeUlongStatus::Invalid;
  }

  uint64_t unsigned_value = 0;
  const StrictIntegerParseStatus status = parse_strict_unsigned_decimal(value, unsigned_value);
  if (status == StrictIntegerParseStatus::OutOfRange) {
    return StrictNonNegativeUlongStatus::OutOfRange;
  }
  if (status != StrictIntegerParseStatus::Success) {
    return StrictNonNegativeUlongStatus::Invalid;
  }
  if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<unsigned long>::max())) {
    return StrictNonNegativeUlongStatus::OutOfRange;
  }

  out_value = static_cast<unsigned long>(unsigned_value);
  return StrictNonNegativeUlongStatus::Success;
}
