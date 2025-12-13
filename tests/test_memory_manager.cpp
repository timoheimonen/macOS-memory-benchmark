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
#include "memory_manager.h"
#include <cstring>
#include <unistd.h>  // getpagesize

// Test successful buffer allocation
TEST(MemoryManagerTest, AllocateBufferSuccess) {
  size_t buffer_size = 1024 * 1024;  // 1 MB
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  
  EXPECT_NE(buffer.get(), nullptr);
  EXPECT_NE(buffer.get(), MAP_FAILED);
}

// Test buffer allocation with page-aligned size
TEST(MemoryManagerTest, AllocateBufferPageAligned) {
  size_t page_size = getpagesize();
  size_t buffer_size = page_size * 4;  // 4 pages
  MmapPtr buffer = allocate_buffer(buffer_size, "test_buffer");
  
  EXPECT_NE(buffer.get(), nullptr);
  EXPECT_NE(buffer.get(), MAP_FAILED);
}

// Test buffer allocation with default buffer name
TEST(MemoryManagerTest, AllocateBufferDefaultName) {
  size_t buffer_size = 64 * 1024;  // 64 KB
  MmapPtr buffer = allocate_buffer(buffer_size);
  
  EXPECT_NE(buffer.get(), nullptr);
  EXPECT_NE(buffer.get(), MAP_FAILED);
}

// Test that allocated buffer can be written to
TEST(MemoryManagerTest, AllocateBufferWritable) {
  size_t buffer_size = 1024;  // 1 KB
  MmapPtr buffer = allocate_buffer(buffer_size, "writable_buffer");
  
  ASSERT_NE(buffer.get(), nullptr);
  
  // Write to buffer
  char* ptr = static_cast<char*>(buffer.get());
  strcpy(ptr, "test data");
  
  // Read back
  EXPECT_STREQ(ptr, "test data");
}

// Test that allocated buffer can be read from
TEST(MemoryManagerTest, AllocateBufferReadable) {
  size_t buffer_size = 1024;  // 1 KB
  MmapPtr buffer = allocate_buffer(buffer_size, "readable_buffer");
  
  ASSERT_NE(buffer.get(), nullptr);
  
  // Write pattern
  unsigned char* ptr = static_cast<unsigned char*>(buffer.get());
  for (size_t i = 0; i < buffer_size; ++i) {
    ptr[i] = static_cast<unsigned char>(i % 256);
  }
  
  // Read back and verify
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ptr[i], static_cast<unsigned char>(i % 256));
  }
}

// Test multiple buffer allocations
TEST(MemoryManagerTest, AllocateMultipleBuffers) {
  size_t buffer_size = 64 * 1024;  // 64 KB
  
  MmapPtr buffer1 = allocate_buffer(buffer_size, "buffer1");
  MmapPtr buffer2 = allocate_buffer(buffer_size, "buffer2");
  MmapPtr buffer3 = allocate_buffer(buffer_size, "buffer3");
  
  EXPECT_NE(buffer1.get(), nullptr);
  EXPECT_NE(buffer2.get(), nullptr);
  EXPECT_NE(buffer3.get(), nullptr);
  
  // Verify they are different memory locations
  EXPECT_NE(buffer1.get(), buffer2.get());
  EXPECT_NE(buffer1.get(), buffer3.get());
  EXPECT_NE(buffer2.get(), buffer3.get());
}

// Test buffer with minimum size (page size)
TEST(MemoryManagerTest, AllocateBufferMinimumSize) {
  size_t page_size = getpagesize();
  MmapPtr buffer = allocate_buffer(page_size, "min_buffer");
  
  EXPECT_NE(buffer.get(), nullptr);
  EXPECT_NE(buffer.get(), MAP_FAILED);
}

// Test that buffer is automatically freed when going out of scope
TEST(MemoryManagerTest, BufferAutoCleanup) {
  {
    size_t buffer_size = 1024 * 1024;  // 1 MB
    MmapPtr buffer = allocate_buffer(buffer_size, "auto_cleanup_buffer");
    EXPECT_NE(buffer.get(), nullptr);
    // Buffer should be automatically freed here when going out of scope
  }
  // If we get here without crashing, cleanup worked
  SUCCEED();
}

// Test large buffer allocation
TEST(MemoryManagerTest, AllocateLargeBuffer) {
  size_t buffer_size = 10 * 1024 * 1024;  // 10 MB
  MmapPtr buffer = allocate_buffer(buffer_size, "large_buffer");
  
  EXPECT_NE(buffer.get(), nullptr);
  EXPECT_NE(buffer.get(), MAP_FAILED);
}

