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

#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/json_utils.h"

namespace {

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::string& stem) {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_" + stem + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(sequence.fetch_add(1)));
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

TEST(JsonFileWriterTest, CreatesParentsAtomicallyReplacesAndSuppressesAnnouncement) {
  TemporaryDirectory temporary("writer_replace");
  const std::filesystem::path target = temporary.path() / "nested" / "result.json";

  testing::internal::CaptureStdout();
  const int first_result =
      write_json_to_file(target, {{"generation", 1}}, false);
  const std::string first_output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(first_result, EXIT_SUCCESS);
  EXPECT_TRUE(first_output.empty());
  EXPECT_EQ(read_json(target)["generation"], 1);
  EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));

  ASSERT_EQ(write_json_to_file(target, {{"generation", 2}}, false), EXIT_SUCCESS);
  EXPECT_EQ(read_json(target)["generation"], 2);
  EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));
}

TEST(JsonFileWriterTest, AnnouncesSuccessfulFinalPathOnlyWhenRequested) {
  TemporaryDirectory temporary("writer_announce");
  const std::filesystem::path target = temporary.path() / "result.json";

  testing::internal::CaptureStdout();
  const int result = write_json_to_file(target, {{"ok", true}}, true);
  const std::string output = testing::internal::GetCapturedStdout();

  ASSERT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(output, Messages::msg_results_saved_to(target.string()) + "\n");
}

TEST(JsonFileWriterTest, FailedTemporaryOpenCleansStaleTemporaryPath) {
  TemporaryDirectory temporary("writer_open_failure");
  const std::filesystem::path target = temporary.path() / "result.json";
  const std::filesystem::path stale_temp = target.string() + ".tmp";
  ASSERT_TRUE(std::filesystem::create_directory(stale_temp));

  testing::internal::CaptureStderr();
  EXPECT_EQ(write_json_to_file(target, {{"ok", true}}, false), EXIT_FAILURE);
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_EQ(error, Messages::error_prefix() +
                       Messages::error_file_write_failed(
                           stale_temp.string(),
                           "Failed to open temporary file: " +
                               std::string(std::strerror(EISDIR))) +
                       "\n");
  EXPECT_FALSE(std::filesystem::exists(stale_temp));
  EXPECT_FALSE(std::filesystem::exists(target));
}

TEST(JsonFileWriterTest, RenameFailurePreservesDestinationAndCleansTemporaryFile) {
  TemporaryDirectory temporary("writer_rename_failure");
  const std::filesystem::path target = temporary.path() / "result.json";
  ASSERT_TRUE(std::filesystem::create_directory(target));
  write_text(target / "keep", "sentinel");

  testing::internal::CaptureStderr();
  EXPECT_EQ(write_json_to_file(target, {{"ok", true}}, false), EXIT_FAILURE);
  const std::string error = testing::internal::GetCapturedStderr();

  EXPECT_NE(error.find(Messages::error_prefix() +
                       "Failed to write file \"" + target.string() +
                       "\": Failed to rename temporary file: "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::is_directory(target));
  EXPECT_TRUE(std::filesystem::exists(target / "keep"));
  EXPECT_FALSE(std::filesystem::exists(target.string() + ".tmp"));
}

TEST(JsonUtilsTest, ParseStringRejectsEmptyAndMalformedAndAcceptsValidJson) {
  nlohmann::json parsed;
  std::string error;

  EXPECT_FALSE(parse_json_from_string("", parsed, error));
  EXPECT_EQ(error, "Empty JSON string");

  error.clear();
  EXPECT_FALSE(parse_json_from_string("{broken", parsed, error));
  EXPECT_NE(error.find("JSON parse error at position"), std::string::npos);

  EXPECT_TRUE(parse_json_from_string("{\"value\":42}", parsed, error));
  EXPECT_EQ(parsed["value"], 42);
  EXPECT_TRUE(error.empty());
}

TEST(JsonUtilsTest, StatisticsNeverFabricateZeroesForAnEmptyPopulation) {
  EXPECT_TRUE(calculate_json_statistics({}).is_null());
}

TEST(JsonUtilsTest, StatisticsUseSampleDeviationInterpolationCvAndMad) {
  const nlohmann::json statistics =
      calculate_json_statistics({10.0, 20.0, 30.0, 40.0});

  EXPECT_DOUBLE_EQ(statistics["average"], 25.0);
  EXPECT_DOUBLE_EQ(statistics["min"], 10.0);
  EXPECT_DOUBLE_EQ(statistics["max"], 40.0);
  EXPECT_DOUBLE_EQ(statistics["median"], 25.0);
  EXPECT_DOUBLE_EQ(statistics["p90"], 37.0);
  EXPECT_DOUBLE_EQ(statistics["p95"], 38.5);
  EXPECT_NEAR(statistics["p99"].get<double>(), 39.7, 1e-12);
  EXPECT_NEAR(statistics["stddev"].get<double>(), 12.909944487358056,
              1e-12);
  EXPECT_NEAR(statistics["coefficient_of_variation_pct"].get<double>(),
              51.63977794943222, 1e-12);
  EXPECT_DOUBLE_EQ(statistics["median_absolute_deviation"], 10.0);
}

TEST(JsonUtilsTest, SingleZeroHasNoVariationAndUndefinedCv) {
  const nlohmann::json statistics = calculate_json_statistics({0.0});

  EXPECT_DOUBLE_EQ(statistics["average"], 0.0);
  EXPECT_DOUBLE_EQ(statistics["median"], 0.0);
  EXPECT_DOUBLE_EQ(statistics["stddev"], 0.0);
  EXPECT_TRUE(statistics["coefficient_of_variation_pct"].is_null());
  EXPECT_DOUBLE_EQ(statistics["median_absolute_deviation"], 0.0);
}

TEST(JsonUtilsTest, ParseFileDistinguishesMissingDirectoryEmptyMalformedAndValid) {
  TemporaryDirectory temporary("parse_file");
  nlohmann::json parsed;
  std::string error;

  const std::filesystem::path missing = temporary.path() / "missing.json";
  EXPECT_FALSE(parse_json_from_file(missing.string(), parsed, error));
  EXPECT_EQ(error, "File does not exist: " + missing.string());

  error.clear();
  EXPECT_FALSE(parse_json_from_file(temporary.path().string(), parsed, error));
  EXPECT_EQ(error, "Path is not a regular file: " + temporary.path().string());

  const std::filesystem::path empty = temporary.path() / "empty.json";
  write_text(empty, "");
  error.clear();
  EXPECT_FALSE(parse_json_from_file(empty.string(), parsed, error));
  EXPECT_EQ(error, "File is empty: " + empty.string());

  const std::filesystem::path malformed = temporary.path() / "malformed.json";
  write_text(malformed, "[invalid]");
  error.clear();
  EXPECT_FALSE(parse_json_from_file(malformed.string(), parsed, error));
  EXPECT_NE(error.find("JSON parse error at position"), std::string::npos);

  const std::filesystem::path valid = temporary.path() / "valid.json";
  write_text(valid, "{\"status\":\"complete\"}");
  error.clear();
  EXPECT_TRUE(parse_json_from_file(valid.string(), parsed, error));
  EXPECT_EQ(parsed["status"], "complete");
  EXPECT_TRUE(error.empty());
}
