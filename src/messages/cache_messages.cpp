// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
#include "messages/messages.h"
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Cache Info Messages ---
std::string cache_info_header() {
  return "\nDetected Cache Sizes:";
}

std::string cache_size_custom(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  Custom Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB";
  }
  return oss.str();
}

std::string cache_size_l1(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  L1 Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B (per P-core)";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB (per P-core)";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB (per P-core)";
  }
  return oss.str();
}

std::string cache_size_l2(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  L2 Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B (per P-core cluster)";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB (per P-core cluster)";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB (per P-core cluster)";
  }
  return oss.str();
}

std::string cache_size_per_pcore() {
  return " (per P-core)";
}

std::string cache_size_per_pcore_cluster() {
  return " (per P-core cluster)";
}

} // namespace Messages

