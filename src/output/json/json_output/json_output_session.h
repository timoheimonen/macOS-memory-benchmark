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
 * other non-empty value is a file target, including `./-` and names containing
 * dashes.
 * ResolveAgainstCurrentDirectory makes relative file targets absolute without
 * otherwise normalizing their spelling; PreserveRaw retains the path spelling.
 *
 * @param raw_value Raw value supplied to `--output`.
 * @param path_policy File-path treatment for classified file targets.
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
  /**
   * Take ownership of one classified command target and install routing.
   *
   * @param target Target value moved into the session. For stdout, the current
   *        `std::cout` buffer is retained and `std::cout` is routed to the
   *        current `std::cerr` buffer until destruction.
   * @throws std::ios_base::failure If caller-configured stream exception state
   *         prevents stdout routing from being installed.
   * @post File and disabled targets leave the standard streams unchanged.
   */
  explicit JsonOutputSession(JsonOutputTarget target);

  /** Restore the exact stdout buffer retained by the constructor. */
  ~JsonOutputSession() noexcept;

  JsonOutputSession(const JsonOutputSession&) = delete;
  JsonOutputSession& operator=(const JsonOutputSession&) = delete;
  JsonOutputSession(JsonOutputSession&&) = delete;
  JsonOutputSession& operator=(JsonOutputSession&&) = delete;

  /** @return The immutable transport kind owned by this session. */
  JsonOutputKind kind() const noexcept;

  /** @return `true` only when logical checkpoints must be written to a file. */
  bool persists_checkpoints() const noexcept;

  /**
   * Offer one logical checkpoint to the selected transport.
   *
   * File targets invoke @p build_payload exactly once and atomically replace
   * the target. Stdout and disabled targets return success without invoking
   * the callback, while the command still observes the logical boundary.
   *
   * @param build_payload Lazy builder valid for this synchronous call.
   * @param announce_success Whether a successful file write prints the
   *        centralized save confirmation; ignored by other transports.
   * @return `EXIT_SUCCESS` for a skipped or persisted checkpoint;
   *         `EXIT_FAILURE` after a contained builder or file-output failure.
   * @note The callback and its captures are not retained after return.
   */
  int checkpoint(
      const std::function<nlohmann::ordered_json()>& build_payload,
      bool announce_success = false);

  /**
   * Persist one prebuilt terminal document through the selected transport.
   *
   * File targets atomically replace the target. Stdout targets write one
   * two-space-indented JSON value and one trailing newline through the retained
   * original stdout buffer. Disabled targets are successful no-ops.
   *
   * @param payload Immutable terminal payload; no reference is retained.
   * @param announce_success Whether a successful file write prints the
   *        centralized save confirmation; ignored by stdout and disabled.
   * @return `EXIT_SUCCESS` on success or when disabled; `EXIT_FAILURE` after a
   *         contained serialization, write, flush, or file-output failure.
   * @warning A machine-output command boundary must call this at most once for
   *          a stdout target. Ordinary `std::cout` remains routed to stderr
   *          until session destruction, including messages emitted afterward.
   */
  int write_final(const nlohmann::ordered_json& payload,
                  bool announce_success = true);

 private:
  int write_stdout(const nlohmann::ordered_json& payload) noexcept;

  JsonOutputTarget target_;
  std::streambuf* original_stdout_buffer_ = nullptr;
  bool stdout_routing_installed_ = false;
};

#endif  // JSON_OUTPUT_JSON_OUTPUT_SESSION_H
