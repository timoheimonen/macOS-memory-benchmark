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
 * @file cli_parser.h
 * @brief Shared descriptor-driven CLI option parser and numeric parsing helpers
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Provides a uniform option scanner supporting short flags, long flags,
 * `--option=value` inline values, opaque space-separated values, single vs.
 * repeatable option arity, centralized error formatting, and LP64-safe
 * non-negative `unsigned long` parsing.
 */

#ifndef CLI_PARSER_H
#define CLI_PARSER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Return a non-empty program name for usage and diagnostics.
 *
 * Returns `argv[0]` when `argc > 0`, `argv != nullptr`, `argv[0] != nullptr`,
 * and `argv[0][0] != '\0'`; otherwise returns `"memory_benchmark"`.
 */
const char* cli_program_name(int argc, char* argv[]) noexcept;

/** Value-taking behavior of one CLI option descriptor. */
enum class CliOptionValueKind : uint8_t {
  None = 0,
  RequiredValue,
};

/** Permitted multiplicity of one CLI option descriptor. */
enum class CliOptionArity : uint8_t {
  Single = 0,
  Repeatable,
};

/** Declarative specification for one recognized CLI option. */
struct CliOptionSpec {
  int id = 0;
  const char* short_name = nullptr;
  const char* long_name = nullptr;
  CliOptionValueKind value_kind = CliOptionValueKind::None;
  CliOptionArity arity = CliOptionArity::Single;
};

/** One parsed CLI option occurrence in original argv order. */
struct CliParsedOption {
  int id = 0;
  std::string matched_name;
  const char* canonical_name = nullptr;
  std::string value;
  bool has_value = false;
};

/** Machine-readable syntax failure category from `parse_cli_options`. */
enum class CliParseErrorKind : uint8_t {
  None = 0,
  UnknownOption,
  MissingValue,
  DuplicateOption,
  UnexpectedValue,
};

/** Result of parsing argv against a mode's CLI option descriptor table. */
struct CliParseResult {
  bool ok = true;
  CliParseErrorKind error_kind = CliParseErrorKind::None;
  std::string offending_token;
  std::string canonical_option;
  std::vector<CliParsedOption> options;
};

/**
 * @brief Parse `argv[1..argc-1]` against a descriptor table.
 *
 * Supports space-separated short/long options and inline `--long=value` forms.
 * Single-dash inline forms (such as `-b=512`) are rejected as `UnknownOption`.
 * Space-separated values are consumed verbatim without inspecting their prefix,
 * so empty or flag-shaped values (for example `-o -G` or `-o -`) remain opaque.
 * Safe when `argc <= 1`, `argv == nullptr`, `specs == nullptr`, or any `argv[i]`
 * is `nullptr`.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param specs Pointer to option descriptors (may be null when `spec_count == 0`).
 * @param spec_count Number of descriptors in `specs`.
 * @return Parsed option sequence on success, or the first syntax error encountered.
 */
CliParseResult parse_cli_options(int argc,
                                 char* argv[],
                                 const CliOptionSpec* specs,
                                 size_t spec_count);

/** Convenience overload accepting a fixed-size `std::array` of descriptors. */
template <size_t N>
inline CliParseResult parse_cli_options(int argc,
                                        char* argv[],
                                        const std::array<CliOptionSpec, N>& specs) {
  return parse_cli_options(argc, argv, specs.data(), specs.size());
}

/**
 * @brief Format a user-facing error message for a failed `CliParseResult`.
 *
 * When `result.error_kind == CliParseErrorKind::UnknownOption` and
 * `mode_allowed_hint` is non-null and non-empty, returns
 * `Messages::error_option_not_allowed_in_mode`; otherwise formats the standard
 * parser diagnostic via `Messages::`.
 *
 * @param result Parse result to format.
 * @param mode_allowed_hint Optional mode-specific allowed-options summary.
 * @return Formatted diagnostic without error prefix, or empty string when `result.error_kind == None`.
 */
std::string format_cli_parse_error(const CliParseResult& result,
                                   const std::string* mode_allowed_hint = nullptr);

/**
 * @brief Extract the long option name from a token, stripping any `=value` suffix.
 *
 * For tokens starting with `"--"`, returns the substring before the first `'='`
 * (or the full token if no `'='` is present). For all other tokens, returns
 * `token` unchanged.
 */
std::string_view extract_long_option_name(std::string_view token) noexcept;

/**
 * @brief Report whether `token` is a known space-separated value-taking option.
 *
 * Covers every short and long option that consumes a following argv token across
 * all six primary modes. Inline `--option=value` forms return `false`.
 */
bool is_known_cli_value_option(std::string_view token) noexcept;

/** Result of parsing a strict non-negative base-10 `unsigned long` token. */
enum class StrictNonNegativeUlongStatus : uint8_t {
  Success = 0,
  Invalid,
  OutOfRange,
  Negative,
};

/**
 * @brief Parse a strict base-10 non-negative `unsigned long` across `[0, ULONG_MAX]`.
 *
 * Avoids LP64 `long long` truncation for values in `(LLONG_MAX, ULONG_MAX]`.
 * Tokens beginning with `'-'` are parsed with `parse_strict_signed_decimal`:
 * valid negative values populate `out_negative_value` and return `Negative`,
 * signed overflows return `OutOfRange`, and all other leading-`'-'` tokens
 * (including `"-0"`, `"-"`, and trailing garbage) return `Invalid`.
 * Non-negative tokens are parsed with `parse_strict_unsigned_decimal` and
 * range-checked against `std::numeric_limits<unsigned long>::max()`.
 *
 * @param value Raw CLI value token.
 * @param[out] out_value Populated on `Success`.
 * @param[out] out_negative_value Populated on `Negative`.
 * @return Parse status distinguishing valid values, negative values, overflow, and syntax errors.
 */
StrictNonNegativeUlongStatus parse_strict_nonnegative_ulong(const std::string& value,
                                                            unsigned long& out_value,
                                                            long long& out_negative_value);

#endif  // CLI_PARSER_H
