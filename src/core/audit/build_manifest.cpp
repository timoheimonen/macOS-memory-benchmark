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
 * @file build_manifest.cpp
 * @brief Auditable build provenance manifest construction and fallback helpers
 */

#include "core/audit/build_manifest.h"

#include <utility>

#if defined(__has_include)
#if __has_include("../../../.build-provenance.h")
#include "../../../.build-provenance.h"
#endif
#endif

nlohmann::ordered_json unavailable_build_manifest_json() {
  return nlohmann::ordered_json{
      {"manifest_version", 1},
      {"status", "unavailable"},
      {"reason_code", "build-provenance-not-provided"},
      {"binary_sha256", nullptr},
      {"git_commit", nullptr},
      {"git_dirty", nullptr},
      {"compiler", nullptr},
      {"compile_flags", nullptr},
      {"link_flags", nullptr},
      {"target_arch", nullptr},
      {"sdk", nullptr},
      {"min_os", nullptr},
  };
}

nlohmann::ordered_json build_provenance_manifest_json(
    std::string_view compiled_manifest_json,
    const std::string& binary_sha256) {
  nlohmann::ordered_json manifest;
  bool parsed_valid = false;
  if (!compiled_manifest_json.empty()) {
    try {
      nlohmann::ordered_json parsed =
          nlohmann::ordered_json::parse(compiled_manifest_json);
      if (parsed.is_object()) {
        manifest = std::move(parsed);
        parsed_valid = true;
      }
    } catch (...) {
      parsed_valid = false;
    }
  }
  if (!parsed_valid) {
    manifest = unavailable_build_manifest_json();
    if (!binary_sha256.empty()) {
      manifest["binary_sha256"] = binary_sha256;
      manifest["status"] = "partial";
    }
    return manifest;
  }
  if (!binary_sha256.empty()) {
    manifest["binary_sha256"] = binary_sha256;
  } else {
    manifest["status"] = "partial";
    manifest["reason_code"] = "binary-hash-unavailable";
  }
  return manifest;
}

nlohmann::ordered_json build_runtime_provenance_manifest_json(
    const std::string& binary_sha256) {
#if defined(BUILD_MANIFEST_JSON)
  return build_provenance_manifest_json(BUILD_MANIFEST_JSON, binary_sha256);
#elif defined(LLM_BUILD_MANIFEST_JSON)
  return build_provenance_manifest_json(LLM_BUILD_MANIFEST_JSON, binary_sha256);
#else
  return build_provenance_manifest_json("", binary_sha256);
#endif
}
