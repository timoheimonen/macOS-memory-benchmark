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

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/json_utils.h"

namespace {

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::string& stem) {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_" + stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(sequence.fetch_add(1)));
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
    std::filesystem::create_directories(path_);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(output.is_open());
  output << text;
  ASSERT_TRUE(output.good());
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  nlohmann::json parsed;
  input >> parsed;
  return parsed;
}

}  // namespace

TEST(JsonFileWriterTest, CreatesParentsAtomicallyReplacesAndAnnouncesOnlyWhenRequested) {
  TemporaryDirectory temporary("writer_replace");
  const std::filesystem::path target = temporary.path() / "nested" / "result.json";

  testing::internal::CaptureStdout();
  const int first_result = write_json_to_file(target, {{"generation", 1}}, false);
  const std::string first_output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(first_result, EXIT_SUCCESS);
  EXPECT_TRUE(first_output.empty());
  EXPECT_EQ(read_json(target)["generation"], 1);
  EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));

  testing::internal::CaptureStdout();
  const int second_result = write_json_to_file(target, {{"generation", 2}}, true);
  const std::string second_output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(second_result, EXIT_SUCCESS);
  EXPECT_EQ(second_output, Messages::msg_results_saved_to(target.string()) + "\n");
  EXPECT_EQ(read_json(target)["generation"], 2);
  EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));
}

TEST(JsonFileWriterTest, WriteFailuresCleanTemporaryFilesPreserveDestinationAndStayContained) {
  {
    SCOPED_TRACE("failed temporary open cleans stale temp path");
    TemporaryDirectory temporary("writer_open_failure");
    const std::filesystem::path target = temporary.path() / "result.json";
    const std::filesystem::path stale_temp = target.string() + ".tmp";
    ASSERT_TRUE(std::filesystem::create_directory(stale_temp));

    testing::internal::CaptureStderr();
    EXPECT_EQ(write_json_to_file(target, {{"ok", true}}, false), EXIT_FAILURE);
    const std::string error = testing::internal::GetCapturedStderr();

    const std::string expected_reason = "Failed to open temporary file: " + std::string(std::strerror(EISDIR));
    EXPECT_EQ(error, Messages::error_prefix() +
                         Messages::error_file_write_failed(stale_temp.string(), expected_reason) + "\n");
    EXPECT_FALSE(std::filesystem::exists(stale_temp));
    EXPECT_FALSE(std::filesystem::exists(target));
  }

  {
    SCOPED_TRACE("rename failure preserves destination and cleans temporary file");
    TemporaryDirectory temporary("writer_rename_failure");
    const std::filesystem::path target = temporary.path() / "result.json";
    ASSERT_TRUE(std::filesystem::create_directory(target));
    write_text(target / "keep", "sentinel");

    testing::internal::CaptureStderr();
    EXPECT_EQ(write_json_to_file(target, {{"ok", true}}, false), EXIT_FAILURE);
    const std::string error = testing::internal::GetCapturedStderr();

    EXPECT_NE(error.find(Messages::error_prefix() + "Failed to write file \"" + target.string() +
                         "\": Failed to rename temporary file: "),
              std::string::npos);
    EXPECT_TRUE(std::filesystem::is_directory(target));
    EXPECT_TRUE(std::filesystem::exists(target / "keep"));
    EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));
  }

  SCOPED_TRACE("parent path filesystem error cannot escape boundary");
  const std::filesystem::path target = std::filesystem::path("/tmp") / std::string(5000, 'x') / "result.json";

  testing::internal::CaptureStderr();
  int result = EXIT_SUCCESS;
  EXPECT_NO_THROW(result = write_json_to_file(target, {{"ok", true}}, false));
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error.find(Messages::error_prefix()), std::string::npos);
  EXPECT_NE(error.find("Failed to write file"), std::string::npos);
}

TEST(JsonUtilsTest, UtcTimestampFormatsFixedTimeIndependentOfLocalTimezone) {
  const auto fixed_time = std::chrono::system_clock::from_time_t(1577934245);

  EXPECT_EQ(build_utc_timestamp(fixed_time), "2020-01-02T03:04:05Z");
}

TEST(JsonUtilsTest, StatisticsComputeSampleDeviationPercentilesMadAndCv) {
  struct StatsCase {
    const char* name; std::vector<double> values;
    double average, min, max, median, p90, p95, p99, stddev;
    bool cv_defined; double cv_pct, mad;
  };
  const std::vector<StatsCase> cases = {
      {"multi-sample distribution", {10.0, 20.0, 30.0, 40.0}, 25.0, 10.0, 40.0, 25.0, 37.0, 38.5, 39.7,
       12.909944487358056, true, 51.63977794943222, 10.0},
      {"single zero sample", {0.0}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0.0},
  };
  for (const StatsCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const nlohmann::json statistics = calculate_json_statistics(entry.values);
    EXPECT_DOUBLE_EQ(statistics["average"], entry.average);
    EXPECT_DOUBLE_EQ(statistics["min"], entry.min);
    EXPECT_DOUBLE_EQ(statistics["max"], entry.max);
    EXPECT_DOUBLE_EQ(statistics["median"], entry.median);
    EXPECT_DOUBLE_EQ(statistics["p90"], entry.p90);
    EXPECT_DOUBLE_EQ(statistics["p95"], entry.p95);
    EXPECT_NEAR(statistics["p99"].get<double>(), entry.p99, 1e-12);
    EXPECT_NEAR(statistics["stddev"].get<double>(), entry.stddev, 1e-12);
    if (entry.cv_defined) {
      EXPECT_NEAR(statistics["coefficient_of_variation_pct"].get<double>(), entry.cv_pct, 1e-12);
    } else {
      EXPECT_TRUE(statistics["coefficient_of_variation_pct"].is_null());
    }
    EXPECT_DOUBLE_EQ(statistics["median_absolute_deviation"], entry.mad);
  }
}
