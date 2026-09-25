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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <type_traits>

#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_session.h"

namespace {

class ScopedStreamBuffers {
 public:
  ScopedStreamBuffers(std::streambuf* stdout_buffer,
                      std::streambuf* stderr_buffer)
      : original_stdout_buffer_(std::cout.rdbuf(stdout_buffer)),
        original_stderr_buffer_(std::cerr.rdbuf(stderr_buffer)) {}

  ~ScopedStreamBuffers() {
    std::cout.rdbuf(original_stdout_buffer_);
    std::cerr.rdbuf(original_stderr_buffer_);
  }

  ScopedStreamBuffers(const ScopedStreamBuffers&) = delete;
  ScopedStreamBuffers& operator=(const ScopedStreamBuffers&) = delete;

 private:
  std::streambuf* original_stdout_buffer_;
  std::streambuf* original_stderr_buffer_;
};

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::string& stem) {
    static std::atomic<unsigned long> sequence{0};
    path_ = std::filesystem::path("/tmp") /
            ("membenchmark_json_session_" + stem + "_" +
             std::to_string(::getpid()) + "_" +
             std::to_string(sequence.fetch_add(1)));
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

class WriteFailingStreamBuffer : public std::streambuf {
 protected:
  std::streamsize xsputn(const char*, std::streamsize) override { return 0; }

  int_type overflow(int_type) override { return traits_type::eof(); }
};

class FlushFailingStreamBuffer : public std::stringbuf {
 protected:
  int sync() override { return -1; }
};



std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

static_assert(!std::is_copy_constructible_v<JsonOutputSession>);
static_assert(!std::is_copy_assignable_v<JsonOutputSession>);
static_assert(!std::is_move_constructible_v<JsonOutputSession>);
static_assert(!std::is_move_assignable_v<JsonOutputSession>);

}  // namespace

TEST(JsonOutputTargetTest, ClassifiesDisabledStdoutAndLiteralFileTargetsBeforePathResolution) {
  const auto directory = std::filesystem::current_path();
  struct TargetCase {
    const char* raw;
    JsonOutputKind kind;
  };
  for (const TargetCase& entry :
       {TargetCase{"", JsonOutputKind::Disabled}, TargetCase{"-", JsonOutputKind::Stdout},
        TargetCase{"./-", JsonOutputKind::File}, TargetCase{"nested/result-with-dashes.json", JsonOutputKind::File}}) {
    for (JsonFilePathPolicy policy :
         {JsonFilePathPolicy::ResolveAgainstCurrentDirectory, JsonFilePathPolicy::PreserveRaw}) {
      SCOPED_TRACE(entry.raw);
      SCOPED_TRACE(static_cast<int>(policy));
      const JsonOutputTarget target = make_json_output_target(entry.raw, policy);
      EXPECT_EQ(target.kind, entry.kind);
      EXPECT_EQ(target.raw_value, entry.raw);
      const std::filesystem::path expected = entry.kind != JsonOutputKind::File ? std::filesystem::path{}
                                             : policy == JsonFilePathPolicy::PreserveRaw
                                                 ? std::filesystem::path(entry.raw)
                                                 : directory / entry.raw;
      EXPECT_EQ(target.file_path, expected);
    }
  }
}

TEST(JsonOutputTargetTest, FilePathPoliciesPreserveTheirExistingSemantics) {
  const std::string relative = "nested/../result.json";
  const JsonOutputTarget resolved = make_json_output_target(
      relative, JsonFilePathPolicy::ResolveAgainstCurrentDirectory);
  EXPECT_EQ(resolved.kind, JsonOutputKind::File);
  EXPECT_EQ(resolved.raw_value, relative);
  EXPECT_EQ(resolved.file_path,
            std::filesystem::current_path() / std::filesystem::path(relative));

  const JsonOutputTarget preserved = make_json_output_target(
      relative, JsonFilePathPolicy::PreserveRaw);
  EXPECT_EQ(preserved.kind, JsonOutputKind::File);
  EXPECT_EQ(preserved.raw_value, relative);
  EXPECT_EQ(preserved.file_path, std::filesystem::path(relative));

  const std::filesystem::path absolute = "/tmp/result.json";
  for (JsonFilePathPolicy policy : {
           JsonFilePathPolicy::ResolveAgainstCurrentDirectory,
           JsonFilePathPolicy::PreserveRaw,
       }) {
    EXPECT_EQ(make_json_output_target(absolute.string(), policy).file_path,
              absolute);
  }
}

TEST(JsonOutputSessionTest, DisabledAndStdoutCheckpointsAreLazyWithoutFilePersistence) {
  for (const std::string& target : {std::string{}, std::string{"-"}}) {
    SCOPED_TRACE(target);
    std::ostringstream stdout_capture;
    std::ostringstream stderr_capture;
    size_t builder_calls = 0;
    int checkpoint_result = EXIT_FAILURE;
    int final_result = EXIT_FAILURE;
    bool buffer_unchanged = false;

    {
      ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
      std::streambuf* installed_stdout = std::cout.rdbuf();
      JsonOutputSession session(make_json_output_target(target));
      buffer_unchanged = std::cout.rdbuf() == installed_stdout;
      checkpoint_result = session.checkpoint([&]() {
        ++builder_calls;
        return nlohmann::ordered_json{{"unused", true}};
      });
      EXPECT_EQ(session.file_writer_attempts(), 0u);
      EXPECT_EQ(session.successful_file_writes(), 0u);
      if (target.empty()) {
        final_result = session.write_final({{"unused", true}});
        std::cout << "ordinary stdout\n";
      }
    }

    EXPECT_EQ(buffer_unchanged, target.empty());
    EXPECT_EQ(checkpoint_result, EXIT_SUCCESS);
    if (target.empty()) EXPECT_EQ(final_result, EXIT_SUCCESS);
    EXPECT_EQ(builder_calls, 0u);
    EXPECT_EQ(stdout_capture.str(), target.empty() ? "ordinary stdout\n" : "");
    EXPECT_TRUE(stderr_capture.str().empty());
  }
}

TEST(JsonOutputSessionTest,
     StdoutCheckpointIsLazyAndFinalJsonBypassesHumanRoutingOnce) {
  const nlohmann::ordered_json payload = {
      {"status", "complete"}, {"nested", {{"value", 42}}}};
  std::ostringstream stdout_capture;
  std::ostringstream stderr_capture;
  size_t builder_calls = 0;
  int checkpoint_result = EXIT_FAILURE;
  int final_result = EXIT_FAILURE;
  JsonOutputKind kind = JsonOutputKind::Disabled;
  bool persists_checkpoints = true;
  bool routing_installed = false;
  bool routing_restored = false;

  {
    ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
    std::streambuf* installed_stdout = std::cout.rdbuf();
    {
      JsonOutputSession session(make_json_output_target("-"));
      kind = session.kind();
      persists_checkpoints = session.persists_checkpoints();
      routing_installed = std::cout.rdbuf() == std::cerr.rdbuf();
      std::cout << "human transcript\n";
      checkpoint_result = session.checkpoint([&]() -> nlohmann::ordered_json {
        ++builder_calls;
        throw std::runtime_error("stdout checkpoint builder must stay lazy");
      });
      final_result = session.write_final(payload, true);
    }
    routing_restored = std::cout.rdbuf() == installed_stdout;
  }

  EXPECT_EQ(kind, JsonOutputKind::Stdout);
  EXPECT_FALSE(persists_checkpoints);
  EXPECT_TRUE(routing_installed);
  EXPECT_TRUE(routing_restored);
  EXPECT_EQ(checkpoint_result, EXIT_SUCCESS);
  EXPECT_EQ(final_result, EXIT_SUCCESS);
  EXPECT_EQ(builder_calls, 0u);
  EXPECT_EQ(stdout_capture.str(), payload.dump(2) + "\n");
  EXPECT_EQ(nlohmann::ordered_json::parse(stdout_capture.str()), payload);
  EXPECT_EQ(stderr_capture.str(), "human transcript\n");
  EXPECT_EQ(stdout_capture.str().find("Results saved"), std::string::npos);
}

TEST(JsonOutputSessionTest, RoutingRestoresOriginalStdoutDuringExceptionUnwind) {
  std::ostringstream stdout_capture;
  std::ostringstream stderr_capture;
  bool caught = false;
  bool routing_restored = false;

  {
    ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
    std::streambuf* installed_stdout = std::cout.rdbuf();
    try {
      JsonOutputSession session(make_json_output_target("-"));
      std::cout << "human before exception\n";
      throw std::runtime_error("injected command exception");
    } catch (const std::runtime_error&) {
      caught = true;
    }
    routing_restored = std::cout.rdbuf() == installed_stdout;
  }

  EXPECT_TRUE(caught);
  EXPECT_TRUE(routing_restored);
  EXPECT_TRUE(stdout_capture.str().empty());
  EXPECT_EQ(stderr_capture.str(), "human before exception\n");
}

TEST(JsonOutputSessionTest, FileCheckpointsUseAtomicWriterAndRetainCadence) {
  TemporaryDirectory temporary("file_checkpoint");
  const std::filesystem::path target_path =
      temporary.path() / "nested" / "result.json";
  const nlohmann::ordered_json partial = {
      {"status", "partial"}, {"completed", 1}};
  const nlohmann::ordered_json complete = {
      {"status", "complete"}, {"completed", 2}};
  std::ostringstream stdout_capture;
  std::ostringstream stderr_capture;
  size_t builder_calls = 0;
  int checkpoint_result = EXIT_FAILURE;
  int final_result = EXIT_FAILURE;
  bool buffer_unchanged = false;
  bool persists_checkpoints = false;

  {
    ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
    std::streambuf* installed_stdout = std::cout.rdbuf();
    JsonOutputSession session(make_json_output_target(
        target_path.string(), JsonFilePathPolicy::PreserveRaw));
    buffer_unchanged = std::cout.rdbuf() == installed_stdout;
    persists_checkpoints = session.persists_checkpoints();
    checkpoint_result = session.checkpoint(
        [&]() {
          ++builder_calls;
          return partial;
        },
        false);
    final_result = session.write_final(complete, false);
    std::cout << "file target human output\n";
  }

  EXPECT_TRUE(buffer_unchanged);
  EXPECT_TRUE(persists_checkpoints);
  EXPECT_EQ(checkpoint_result, EXIT_SUCCESS);
  EXPECT_EQ(final_result, EXIT_SUCCESS);
  EXPECT_EQ(builder_calls, 1u);
  EXPECT_EQ(read_text_file(target_path), complete.dump(2) + "\n");
  EXPECT_FALSE(std::filesystem::exists(target_path.string() + ".tmp"));
  EXPECT_EQ(stdout_capture.str(), "file target human output\n");
  EXPECT_TRUE(stderr_capture.str().empty());
}

TEST(JsonOutputSessionTest, FailedStdoutWriteOrFlushReturnsExactFailureWithoutThrowing) {
  WriteFailingStreamBuffer write_failure;
  FlushFailingStreamBuffer flush_failure;
  struct FailureCase {
    const char* name;
    std::streambuf* buffer;
    std::string reason;
  };
  for (const FailureCase& entry : {FailureCase{"write", &write_failure, Messages::json_stdout_reason_write_failed()},
                                   FailureCase{"flush", &flush_failure, Messages::json_stdout_reason_flush_failed()}}) {
    SCOPED_TRACE(entry.name);
    std::ostringstream stderr_capture;
    int result = EXIT_SUCCESS;
    {
      ScopedStreamBuffers capture(entry.buffer, stderr_capture.rdbuf());
      JsonOutputSession session(make_json_output_target("-"));
      EXPECT_NO_THROW(result = session.write_final({{"status", "complete"}}));
    }
    EXPECT_EQ(result, EXIT_FAILURE);
    EXPECT_EQ(stderr_capture.str(),
              Messages::error_prefix() + Messages::error_json_stdout_write_failed(entry.reason) + "\n");
  }
}

TEST(JsonOutputSessionTest, SerializationExceptionCannotEscapeBoundary) {
  nlohmann::ordered_json payload;
  payload["invalid_utf8"] = std::string(1, static_cast<char>(0xff));
  std::string expected_details;
  try {
    (void)payload.dump(2);
  } catch (const std::exception& error) {
    expected_details = error.what();
  }
  ASSERT_FALSE(expected_details.empty());

  std::ostringstream stdout_capture;
  std::ostringstream stderr_capture;
  int result = EXIT_SUCCESS;

  {
    ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
    JsonOutputSession session(make_json_output_target("-"));
    EXPECT_NO_THROW(result = session.write_final(payload));
  }

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_TRUE(stdout_capture.str().empty());
  EXPECT_EQ(stderr_capture.str(),
            Messages::error_prefix() +
                Messages::error_json_stdout_write_failed(expected_details) +
                "\n");
}

TEST(JsonOutputSessionTest, FileCheckpointBuilderExceptionIsContained) {
  TemporaryDirectory temporary("builder_exception");
  const std::filesystem::path target_path = temporary.path() / "result.json";
  std::ostringstream stdout_capture;
  std::ostringstream stderr_capture;
  int result = EXIT_SUCCESS;

  {
    ScopedStreamBuffers capture(stdout_capture.rdbuf(), stderr_capture.rdbuf());
    JsonOutputSession session(make_json_output_target(
        target_path.string(), JsonFilePathPolicy::PreserveRaw));
    EXPECT_NO_THROW(result = session.checkpoint(
                        []() -> nlohmann::ordered_json {
                          throw std::runtime_error("injected builder exception");
                        },
                        false));
  }

  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_FALSE(std::filesystem::exists(target_path));
  EXPECT_TRUE(stdout_capture.str().empty());
  EXPECT_EQ(
      stderr_capture.str(),
      Messages::error_prefix() +
          Messages::error_file_write_failed(
              target_path.string(),
              Messages::error_json_payload_construction_failed(
                  "injected builder exception")) +
          "\n");
}

TEST(JsonOutputSessionTest, FileObservationsCountWriterEntryAfterBuilderAndOnlySuccessfulReturns) {
  TemporaryDirectory temporary("writer_observations");
  JsonOutputSession session(make_json_output_target((temporary.path() / "result.json").string()));
  EXPECT_EQ(session.file_writer_attempts(), 0u);
  EXPECT_EQ(session.successful_file_writes(), 0u);
  EXPECT_EQ(session.checkpoint([]() -> nlohmann::ordered_json { throw std::runtime_error("builder failure"); }),
            EXIT_FAILURE);
  EXPECT_EQ(session.file_writer_attempts(), 0u);
  EXPECT_EQ(session.successful_file_writes(), 0u);
  EXPECT_EQ(session.checkpoint([&]() {
    EXPECT_EQ(session.file_writer_attempts(), 0u);
    return nlohmann::ordered_json{{"prior_attempts", session.file_writer_attempts()},
                                  {"prior_successes", session.successful_file_writes()}};
  }), EXIT_SUCCESS);
  EXPECT_EQ(session.file_writer_attempts(), 1u);
  EXPECT_EQ(session.successful_file_writes(), 1u);
  std::ifstream input(temporary.path() / "result.json");
  const auto document = nlohmann::ordered_json::parse(input);
  EXPECT_EQ(document["prior_attempts"], 0);
  EXPECT_EQ(document["prior_successes"], 0);
  JsonOutputSession failing(make_json_output_target(temporary.path().string()));
  EXPECT_EQ(failing.checkpoint([]() { return nlohmann::ordered_json::object(); }), EXIT_FAILURE);
  EXPECT_EQ(failing.file_writer_attempts(), 1u);
  EXPECT_EQ(failing.successful_file_writes(), 0u);
}
