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
 * @file build_manifest.h
 * @brief Auditable build provenance manifest construction and fallback helpers
 */

#ifndef CORE_AUDIT_BUILD_MANIFEST_H
#define CORE_AUDIT_BUILD_MANIFEST_H

#include <string>
#include <string_view>

#include "third_party/nlohmann/json.hpp"

/** Return the canonical unavailable build manifest schema object. */
nlohmann::ordered_json unavailable_build_manifest_json();

/**
 * Parse a compiled provenance JSON string and attach runtime binary SHA-256.
 * Falls back cleanly to unavailable/partial manifests when provenance or hash
 * inputs are missing or malformed.
 */
nlohmann::ordered_json build_provenance_manifest_json(
    std::string_view compiled_manifest_json,
    const std::string& binary_sha256);

/**
 * Build the runtime provenance manifest using `.build-provenance.h` inputs
 * (if available) and the supplied runtime binary SHA-256 digest.
 */
nlohmann::ordered_json build_runtime_provenance_manifest_json(
    const std::string& binary_sha256);

#endif  // CORE_AUDIT_BUILD_MANIFEST_H
