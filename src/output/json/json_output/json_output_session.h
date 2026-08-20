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

/**
 * @file json_output_session.h
 * @brief Command-boundary JSON output target and stdout routing.
 */

#ifndef JSON_OUTPUT_JSON_OUTPUT_SESSION_H
#define JSON_OUTPUT_JSON_OUTPUT_SESSION_H

#include <filesystem>
#include <functional>
#include <streambuf>
#include <string>
#include <string_view>

#include "third_party/nlohmann/json.hpp"

/** Transport selected by the raw `--output` option value. */
enum class JsonOutputKind {
  Disabled = 0,
  File,
  Stdout,
};

/** File-path treatment used after the exact stdout sentinel is classified. */
enum class JsonFilePathPolicy {
  ResolveAgainstCurrentDirectory = 0,
  PreserveRaw,
};

/** Raw and effective JSON output target retained for one command. */
struct JsonOutputTarget {
  JsonOutputKind kind = JsonOutputKind::Disabled;
  std::string raw_value;
  std::filesystem::path file_path;
};

/**
 * Classify a raw output option before performing any path resolution.
 *
 * An empty value disables JSON output and exactly `-` selects stdout. Every
 * other value is a file target, including `./-` and names containing dashes.
 * ResolveAgainstCurrentDirectory makes relative file targets absolute without
 * otherwise normalizing their spelling; PreserveRaw retains the path spelling.
 *
 * @param raw_value Raw value supplied to `--output`.
 * @param path_policy File-path treatment for non-sentinel values.
 * @return Classified target containing both the raw value and effective path.
 * @throws std::filesystem::filesystem_error if current-directory resolution
 *         fails. Command boundaries must convert this pre-execution failure to
 *         their normal return-code error path.
 */
JsonOutputTarget make_json_output_target(
    std::string_view raw_value,
    JsonFilePathPolicy path_policy =
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory);

/**
 * Own one command's JSON transport and optional stdout stream routing.
 *
 * For stdout targets, ordinary `std::cout` output is routed to the current
 * `std::cerr` buffer for the session lifetime. Final JSON bypasses that routing
 * through a separate stream backed by the retained original stdout buffer.
 * Sessions are single-owner, must not overlap, and must be installed before
 * worker threads start.
 *
 * File checkpoints retain the existing atomic writer. Stdout and disabled
 * checkpoints are lazy successful no-ops, so their payload builder is not
 * invoked. Stream write, flush, serialization, and payload-builder exceptions
 * are converted to `EXIT_FAILURE`.
 */
class JsonOutputSession {
 public:
  explicit JsonOutputSession(JsonOutputTarget target);
  ~JsonOutputSession() noexcept;

  JsonOutputSession(const JsonOutputSession&) = delete;
  JsonOutputSession& operator=(const JsonOutputSession&) = delete;
  JsonOutputSession(JsonOutputSession&&) = delete;
  JsonOutputSession& operator=(JsonOutputSession&&) = delete;

  JsonOutputKind kind() const noexcept;
  bool persists_checkpoints() const noexcept;

  int checkpoint(
      const std::function<nlohmann::ordered_json()>& build_payload,
      bool announce_success = false);
  int write_final(const nlohmann::ordered_json& payload,
                  bool announce_success = true);

 private:
  int write_stdout(const nlohmann::ordered_json& payload) noexcept;

  JsonOutputTarget target_;
  std::streambuf* original_stdout_buffer_ = nullptr;
  bool stdout_routing_installed_ = false;
};

#endif  // JSON_OUTPUT_JSON_OUTPUT_SESSION_H
