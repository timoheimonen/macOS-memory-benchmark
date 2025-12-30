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
#include <gtest/gtest.h>
#include "core/memory/memory_utils.h"
#include "core/memory/memory_manager.h"
#include "core/config/constants.h"
#include <cstdlib>
#include <cstdint>

// Test setup_latency_chain with null buffer - should fail with error message
TEST(MemoryUtilsTest, SetupLatencyChainNullBuffer) {
  using namespace Constants;
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(nullptr, LATENCY_STRIDE_BYTES * 2, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message contains expected content
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
  EXPECT_NE(error_output.find("null"), std::string::npos);
}

// Test setup_latency_chain with zero stride - should fail with error message
TEST(MemoryUtilsTest, SetupLatencyChainZeroStride) {
  using namespace Constants;
  
  // Allocate a buffer for testing
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, 0);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message contains expected content
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
  EXPECT_NE(error_output.find("zero"), std::string::npos);
}

// Test setup_latency_chain with buffer smaller than stride - should fail
TEST(MemoryUtilsTest, SetupLatencyChainBufferSmallerThanStride) {
  using namespace Constants;
  
  // Allocate buffer smaller than stride
  size_t buffer_size = LATENCY_STRIDE_BYTES - 1;  // 127 bytes < 128 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail (num_pointers = buffer_size / stride = 0, need at least 2)
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with buffer equal to stride - should fail (num_pointers == 1)
TEST(MemoryUtilsTest, SetupLatencyChainBufferEqualToStride) {
  using namespace Constants;
  
  // Allocate buffer equal to stride
  size_t buffer_size = LATENCY_STRIDE_BYTES;  // 128 bytes == 128 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail (num_pointers = 128 / 128 = 1, need at least 2)
  EXPECT_EQ(result, EXIT_FAILURE);
  
  // Verify error message
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with buffer just larger than stride - boundary case
TEST(MemoryUtilsTest, SetupLatencyChainBufferJustLargerThanStride) {
  using namespace Constants;
  
  // Allocate buffer just larger than stride
  // Need at least 2 pointers, so buffer_size >= stride * 2
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;  // 256 bytes (minimum valid)
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed (num_pointers = 256 / 128 = 2, minimum valid)
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // No error should be logged
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with minimum valid case (num_pointers == 2)
TEST(MemoryUtilsTest, SetupLatencyChainMinimumValid) {
  using namespace Constants;
  
  // Allocate buffer for exactly 2 pointers
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;  // 256 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed (num_pointers == 2, minimum valid)
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // No error should be logged
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test setup_latency_chain with num_pointers == 3 (valid case)
TEST(MemoryUtilsTest, SetupLatencyChainThreePointers) {
  using namespace Constants;
  
  // Allocate buffer for 3 pointers
  size_t buffer_size = LATENCY_STRIDE_BYTES * 3;  // 384 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // No error should be logged
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test buffer size progression: buffer_size < LATENCY_STRIDE_BYTES
TEST(MemoryUtilsTest, BufferSizeProgressionLessThanStride) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES - 1;  // 127 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail
  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test buffer size progression: buffer_size == LATENCY_STRIDE_BYTES
TEST(MemoryUtilsTest, BufferSizeProgressionEqualToStride) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES;  // 128 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should fail (num_pointers == 1, need at least 2)
  EXPECT_EQ(result, EXIT_FAILURE);
  EXPECT_NE(error_output.find("Error: "), std::string::npos);
}

// Test buffer size progression: buffer_size == LATENCY_STRIDE_BYTES * 2 (minimum valid)
TEST(MemoryUtilsTest, BufferSizeProgressionMinimumValid) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES * 2;  // 256 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed (num_pointers == 2, minimum valid)
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test buffer size progression: buffer_size == LATENCY_STRIDE_BYTES * 3
TEST(MemoryUtilsTest, BufferSizeProgressionThreePointers) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES * 3;  // 384 bytes
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  testing::internal::CaptureStderr();
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  std::string error_output = testing::internal::GetCapturedStderr();
  
  // Should succeed (num_pointers == 3)
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(error_output.find("Error: "), std::string::npos);
}

// Test that setup_latency_chain creates a valid linked list
TEST(MemoryUtilsTest, SetupLatencyChainCreatesValidChain) {
  using namespace Constants;
  
  size_t buffer_size = LATENCY_STRIDE_BYTES * 4;  // 512 bytes (4 pointers)
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  ASSERT_NE(buffer.get(), nullptr);
  
  int result = setup_latency_chain(buffer.get(), buffer_size, LATENCY_STRIDE_BYTES);
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Verify that the chain was created by checking that pointers are set
  // The chain should have pointers linking to different locations
  uintptr_t* ptr1 = reinterpret_cast<uintptr_t*>(static_cast<char*>(buffer.get()) + 0 * LATENCY_STRIDE_BYTES);
  uintptr_t* ptr2 = reinterpret_cast<uintptr_t*>(static_cast<char*>(buffer.get()) + 1 * LATENCY_STRIDE_BYTES);
  
  // Pointers should be set (not zero)
  EXPECT_NE(*ptr1, 0u);
  EXPECT_NE(*ptr2, 0u);
  
  // Pointers should be within buffer bounds
  uintptr_t buffer_start = reinterpret_cast<uintptr_t>(buffer.get());
  uintptr_t buffer_end = buffer_start + buffer_size;
  EXPECT_GE(*ptr1, buffer_start);
  EXPECT_LT(*ptr1, buffer_end);
  EXPECT_GE(*ptr2, buffer_start);
  EXPECT_LT(*ptr2, buffer_end);
}

