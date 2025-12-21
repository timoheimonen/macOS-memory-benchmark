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
#include "messages.h"

namespace Messages {

// --- Pattern Benchmark Messages ---
const std::string& pattern_na() {
  static const std::string msg = "N/A";
  return msg;
}

const std::string& pattern_sequential_forward() {
  static const std::string msg = "Sequential Forward:";
  return msg;
}

const std::string& pattern_sequential_reverse() {
  static const std::string msg = "Sequential Reverse:";
  return msg;
}

std::string pattern_strided(const std::string& stride_name) {
  return "Strided (" + stride_name + "):";
}

const std::string& pattern_random_uniform() {
  static const std::string msg = "Random Uniform:";
  return msg;
}

const std::string& pattern_cache_line_64b() {
  static const std::string msg = "Cache Line - 64B";
  return msg;
}

const std::string& pattern_page_4096b() {
  static const std::string msg = "Page - 4096B";
  return msg;
}

const std::string& pattern_efficiency_analysis() {
  static const std::string msg = "Pattern Efficiency Analysis:";
  return msg;
}

const std::string& pattern_sequential_coherence() {
  static const std::string msg = "Sequential coherence:";
  return msg;
}

const std::string& pattern_prefetcher_effectiveness() {
  static const std::string msg = "Prefetcher effectiveness:";
  return msg;
}

const std::string& pattern_cache_thrashing_potential() {
  static const std::string msg = "Cache thrashing potential:";
  return msg;
}

const std::string& pattern_tlb_pressure() {
  static const std::string msg = "TLB pressure:";
  return msg;
}

const std::string& pattern_cache_thrashing_low() {
  static const std::string msg = "Low";
  return msg;
}

const std::string& pattern_cache_thrashing_medium() {
  static const std::string msg = "Medium";
  return msg;
}

const std::string& pattern_cache_thrashing_high() {
  static const std::string msg = "High";
  return msg;
}

const std::string& pattern_tlb_pressure_minimal() {
  static const std::string msg = "Minimal";
  return msg;
}

const std::string& pattern_tlb_pressure_moderate() {
  static const std::string msg = "Moderate";
  return msg;
}

const std::string& pattern_tlb_pressure_high() {
  static const std::string msg = "High";
  return msg;
}

const std::string& pattern_separator() {
  static const std::string msg = "\n================================\n\n";
  return msg;
}

const std::string& pattern_read_label() {
  static const std::string msg = "  Read : ";
  return msg;
}

const std::string& pattern_write_label() {
  static const std::string msg = "  Write: ";
  return msg;
}

const std::string& pattern_copy_label() {
  static const std::string msg = "  Copy : ";
  return msg;
}

const std::string& pattern_bandwidth_unit() {
  static const std::string msg = " GB/s";
  return msg;
}

const std::string& pattern_bandwidth_unit_newline() {
  static const std::string msg = " GB/s\n";
  return msg;
}

} // namespace Messages

