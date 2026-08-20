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
 * @file json_output_session.cpp
 * @brief Command-boundary JSON output target and stdout routing.
 */

#include "output/json/json_output/json_output_session.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"

namespace {

void report_stdout_write_failure(const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_json_stdout_write_failed(details) << '\n';
  } catch (...) {
    // Error reporting must not let an output exception escape the boundary.
  }
}

void report_payload_build_failure(const std::filesystem::path& path,
                                  const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_file_write_failed(
                     path.string(),
                     Messages::error_json_payload_construction_failed(details))
              << '\n';
  } catch (...) {
    // Error reporting must not let a secondary formatting failure escape.
  }
}

}  // namespace

JsonOutputTarget make_json_output_target(std::string_view raw_value,
                                         JsonFilePathPolicy path_policy) {
  JsonOutputTarget target;
  target.raw_value = std::string(raw_value);

  if (raw_value.empty()) {
    return target;
  }
  if (raw_value == "-") {
    target.kind = JsonOutputKind::Stdout;
    return target;
  }

  target.kind = JsonOutputKind::File;
  target.file_path = std::filesystem::path(target.raw_value);
  if (path_policy == JsonFilePathPolicy::ResolveAgainstCurrentDirectory &&
      target.file_path.is_relative()) {
    target.file_path = std::filesystem::current_path() / target.file_path;
  }
  return target;
}

JsonOutputSession::JsonOutputSession(JsonOutputTarget target)
    : target_(std::move(target)) {
  if (target_.kind != JsonOutputKind::Stdout) {
    return;
  }

  original_stdout_buffer_ = std::cout.rdbuf();
  std::cout.rdbuf(std::cerr.rdbuf());
  stdout_routing_installed_ = true;
}

JsonOutputSession::~JsonOutputSession() noexcept {
  if (!stdout_routing_installed_) {
    return;
  }

  try {
    std::cout.rdbuf(original_stdout_buffer_);
  } catch (...) {
    // basic_ios::rdbuf is not expected to throw, but the destructor boundary
    // must remain noexcept even with a non-standard stream implementation.
  }
}

JsonOutputKind JsonOutputSession::kind() const noexcept { return target_.kind; }

bool JsonOutputSession::persists_checkpoints() const noexcept {
  return target_.kind == JsonOutputKind::File;
}

int JsonOutputSession::checkpoint(
    const std::function<nlohmann::ordered_json()>& build_payload,
    bool announce_success) {
  if (!persists_checkpoints()) {
    return EXIT_SUCCESS;
  }

  try {
    const nlohmann::ordered_json payload = build_payload();
    return write_json_to_file(target_.file_path, payload, announce_success);
  } catch (const std::exception& error) {
    report_payload_build_failure(target_.file_path, error.what());
  } catch (...) {
    report_payload_build_failure(target_.file_path, "");
  }
  return EXIT_FAILURE;
}

int JsonOutputSession::write_final(const nlohmann::ordered_json& payload,
                                   bool announce_success) {
  switch (target_.kind) {
    case JsonOutputKind::Disabled:
      return EXIT_SUCCESS;
    case JsonOutputKind::File:
      return write_json_to_file(target_.file_path, payload, announce_success);
    case JsonOutputKind::Stdout:
      // A file-style save announcement is intentionally suppressed. Human
      // output already routes to stderr, while stdout remains JSON-only.
      return write_stdout(payload);
  }
  return EXIT_FAILURE;
}

int JsonOutputSession::write_stdout(
    const nlohmann::ordered_json& payload) noexcept {
  if (!stdout_routing_installed_ || original_stdout_buffer_ == nullptr) {
    report_stdout_write_failure(Messages::json_stdout_reason_stream_unavailable());
    return EXIT_FAILURE;
  }

  try {
    const std::string serialized = payload.dump(2);
    if (serialized.size() >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      report_stdout_write_failure(
          Messages::json_stdout_reason_size_limit_exceeded());
      return EXIT_FAILURE;
    }

    std::ostream output(original_stdout_buffer_);
    output.write(serialized.data(),
                 static_cast<std::streamsize>(serialized.size()));
    output.put('\n');
    if (!output.good()) {
      report_stdout_write_failure(Messages::json_stdout_reason_write_failed());
      return EXIT_FAILURE;
    }

    output.flush();
    if (!output.good()) {
      report_stdout_write_failure(Messages::json_stdout_reason_flush_failed());
      return EXIT_FAILURE;
    }
  } catch (const std::exception& error) {
    report_stdout_write_failure(error.what());
    return EXIT_FAILURE;
  } catch (...) {
    report_stdout_write_failure("");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
