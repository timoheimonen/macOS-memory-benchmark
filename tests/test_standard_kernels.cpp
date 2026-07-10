// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include "asm/asm_functions.h"
#include "benchmark/benchmark_tests.h"
#include "benchmark/parallel_test_framework.h"
#include "benchmark/benchmark_work_plan.h"
#include "core/memory/memory_utils.h"
#include "core/timing/timer.h"

extern "C" uint64_t verify_pattern_callee_saved_registers_asm(
    uintptr_t function_address, uintptr_t arg0, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);

namespace {

constexpr std::array<size_t, 6> kTailSizes = {1, 31, 32, 511, 512, 513};

class GuardedMapping {
 public:
  explicit GuardedMapping(size_t payload_size)
      : page_size_(static_cast<size_t>(getpagesize())),
        mapping_size_(page_size_ * 3) {
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return;
    }
    if (mprotect(mapping_, page_size_, PROT_NONE) != 0 ||
        mprotect(static_cast<unsigned char*>(mapping_) + 2 * page_size_,
                 page_size_, PROT_NONE) != 0) {
      munmap(mapping_, mapping_size_);
      mapping_ = nullptr;
      return;
    }
    payload_ = static_cast<unsigned char*>(mapping_) + 2 * page_size_ -
               payload_size;
  }

  ~GuardedMapping() {
    if (mapping_ != nullptr) munmap(mapping_, mapping_size_);
  }

  GuardedMapping(const GuardedMapping&) = delete;
  GuardedMapping& operator=(const GuardedMapping&) = delete;

  unsigned char* payload() const { return payload_; }
  bool valid() const { return mapping_ != nullptr; }

 private:
  void* mapping_ = nullptr;
  unsigned char* payload_ = nullptr;
  size_t page_size_ = 0;
  size_t mapping_size_ = 0;
};

uint64_t expected_streaming_checksum(const unsigned char* data, size_t size) {
  const size_t vector_bytes = size - size % 32;
  uint64_t checksum = 0;
  for (size_t index = 0; index < vector_bytes; ++index) {
    checksum ^= static_cast<uint64_t>(data[index]) << ((index % 8) * 8);
  }
  uint64_t byte_tail_checksum = 0;
  for (size_t index = vector_bytes; index < size; ++index) {
    byte_tail_checksum ^= data[index];
  }
  return checksum ^ byte_tail_checksum;
}

using ReadKernel = uint64_t (*)(const void*, size_t);
using WriteKernel = void (*)(void*, size_t);
using CopyKernel = void (*)(void*, const void*, size_t);

std::atomic<size_t> fake_read_bytes{0};
std::atomic<size_t> fake_read_calls{0};
std::atomic<size_t> fake_copy_bytes{0};
std::atomic<size_t> fake_copy_calls{0};

uint64_t fake_read_kernel(const void*, size_t size) {
  fake_read_bytes.fetch_add(size, std::memory_order_relaxed);
  fake_read_calls.fetch_add(1, std::memory_order_relaxed);
  return static_cast<uint64_t>(size);
}

void fake_copy_kernel(void* destination, const void* source, size_t size) {
  std::memcpy(destination, source, size);
  fake_copy_bytes.fetch_add(size, std::memory_order_relaxed);
  fake_copy_calls.fetch_add(1, std::memory_order_relaxed);
}

void verify_read_kernel_boundaries(ReadKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping source(size);
    ASSERT_TRUE(source.valid());
    for (size_t index = 0; index < size; ++index) {
      source.payload()[index] =
          static_cast<unsigned char>((index * 37 + 11) & 0xff);
    }
    const std::vector<unsigned char> before(source.payload(),
                                            source.payload() + size);
    EXPECT_EQ(kernel(source.payload(), size),
              expected_streaming_checksum(source.payload(), size))
        << "size=" << size;
    EXPECT_TRUE(std::equal(before.begin(), before.end(), source.payload()))
        << "size=" << size;
  }
}

void verify_write_kernel_boundaries(WriteKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping destination(size);
    ASSERT_TRUE(destination.valid());
    std::memset(destination.payload(), 0xa5, size);
    kernel(destination.payload(), size);
    for (size_t index = 0; index < size; ++index) {
      EXPECT_EQ(destination.payload()[index], 0u)
          << "size=" << size << " index=" << index;
    }
  }
}

void verify_copy_kernel_boundaries(CopyKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping source(size);
    GuardedMapping destination(size);
    ASSERT_TRUE(source.valid());
    ASSERT_TRUE(destination.valid());
    for (size_t index = 0; index < size; ++index) {
      source.payload()[index] =
          static_cast<unsigned char>((index * 19 + 7) & 0xff);
    }
    std::memset(destination.payload(), 0xa5, size);
    const std::vector<unsigned char> source_before(source.payload(),
                                                   source.payload() + size);
    kernel(destination.payload(), source.payload(), size);
    EXPECT_TRUE(std::equal(source_before.begin(), source_before.end(),
                           source.payload()))
        << "size=" << size;
    EXPECT_TRUE(std::equal(source_before.begin(), source_before.end(),
                           destination.payload()))
        << "size=" << size;
  }
}

}  // namespace

TEST(StandardKernelIntegrationTest, MainReadHonorsTailsAndChecksum) {
  verify_read_kernel_boundaries(memory_read_loop_asm);
}

TEST(StandardKernelIntegrationTest, CacheReadHonorsTailsAndChecksum) {
  verify_read_kernel_boundaries(memory_read_cache_loop_asm);
}

TEST(StandardKernelIntegrationTest, CacheReadChecksumIncludesUpperVectorLane) {
  alignas(64) std::array<unsigned char, 32> data{};
  data[8] = 0x5a;
  EXPECT_EQ(memory_read_cache_loop_asm(data.data(), data.size()), 0x5aULL);
}

TEST(StandardKernelIntegrationTest, MainAndCacheWritesHonorExactBoundaries) {
  verify_write_kernel_boundaries(memory_write_loop_asm);
  verify_write_kernel_boundaries(memory_write_cache_loop_asm);
}

TEST(StandardKernelIntegrationTest, MainAndCacheCopiesHonorExactBoundaries) {
  verify_copy_kernel_boundaries(memory_copy_loop_asm);
  verify_copy_kernel_boundaries(memory_copy_cache_loop_asm);
}

TEST(StandardKernelIntegrationTest, StandardKernelsPreserveCalleeSavedRegisters) {
  alignas(64) std::array<unsigned char, 1024> source{};
  alignas(64) std::array<unsigned char, 1024> destination{};
  for (size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<unsigned char>(index & 0xff);
  }
  constexpr size_t kSize = 513;

  for (ReadKernel kernel : {memory_read_loop_asm,
                            memory_read_cache_loop_asm}) {
    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(kernel),
                  reinterpret_cast<uintptr_t>(source.data()), kSize, 0, 0, 0,
                  0),
              1u);
  }
  for (WriteKernel kernel : {memory_write_loop_asm,
                             memory_write_cache_loop_asm}) {
    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(kernel),
                  reinterpret_cast<uintptr_t>(destination.data()), kSize, 0,
                  0, 0, 0),
              1u);
  }
  for (CopyKernel kernel : {memory_copy_loop_asm,
                            memory_copy_cache_loop_asm}) {
    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(kernel),
                  reinterpret_cast<uintptr_t>(destination.data()),
                  reinterpret_cast<uintptr_t>(source.data()), kSize, 0, 0, 0),
              1u);
  }

  ASSERT_EQ(setup_latency_chain(source.data(), source.size(), 256, 0, nullptr,
                                LatencyChainMode::GlobalRandom, 12345),
            EXIT_SUCCESS);
  EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                reinterpret_cast<uintptr_t>(memory_latency_chase_asm),
                reinterpret_cast<uintptr_t>(source.data()), 16, 0, 0, 0, 0),
            1u);
}

TEST(StandardKernelIntegrationTest, ExecutorConsumesPlannerAccountingExactly) {
  constexpr size_t kSize = 513;
  constexpr size_t kPasses = 3;
  alignas(64) std::array<unsigned char, kSize> source{};
  alignas(64) std::array<unsigned char, kSize> destination{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  BenchmarkWorkPlan read_plan = build_benchmark_bandwidth_work_plan(
      kSize, 4, kPasses, BenchmarkTarget::MainMemory,
      BenchmarkOperation::Read);
  ASSERT_EQ(read_plan.status, BenchmarkMeasurementStatus::Measured);
  fake_read_bytes.store(0, std::memory_order_relaxed);
  fake_read_calls.store(0, std::memory_order_relaxed);
  uint64_t checksum = 0;
  ParallelExecutionMetadata read_metadata;
  EXPECT_GT(run_read_test_with_plan(source.data(), read_plan, checksum, *timer,
                                    fake_read_kernel, &read_metadata),
            0.0);
  EXPECT_EQ(fake_read_bytes.load(std::memory_order_relaxed),
            read_plan.total_payload_bytes);
  EXPECT_EQ(fake_read_calls.load(std::memory_order_relaxed) / kPasses,
            static_cast<size_t>(read_plan.effective_threads));
  EXPECT_EQ(read_metadata.created_workers, read_plan.effective_threads);
  EXPECT_EQ(read_metadata.qos_successful_workers +
                read_metadata.qos_failed_workers,
            static_cast<size_t>(read_plan.effective_threads));
  EXPECT_FALSE(read_metadata.worker_startup_failed);

  BenchmarkWorkPlan copy_plan = build_benchmark_bandwidth_work_plan(
      kSize, 4, kPasses, BenchmarkTarget::MainMemory,
      BenchmarkOperation::Copy);
  ASSERT_EQ(copy_plan.status, BenchmarkMeasurementStatus::Measured);
  fake_copy_bytes.store(0, std::memory_order_relaxed);
  fake_copy_calls.store(0, std::memory_order_relaxed);
  ParallelExecutionMetadata copy_metadata;
  EXPECT_GT(run_copy_test_with_plan(destination.data(), source.data(), copy_plan,
                                    *timer, fake_copy_kernel, &copy_metadata),
            0.0);
  EXPECT_EQ(fake_copy_bytes.load(std::memory_order_relaxed) * 2,
            copy_plan.total_payload_bytes);
  EXPECT_EQ(fake_copy_calls.load(std::memory_order_relaxed) / kPasses,
            static_cast<size_t>(copy_plan.effective_threads));
  EXPECT_EQ(copy_metadata.created_workers, copy_plan.effective_threads);
}

TEST(StandardKernelIntegrationTest, WorkerStartupFailureIsDeterministicAndNotMeasured) {
  alignas(64) std::array<unsigned char, 4096> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  ParallelExecutionMetadata metadata;
  ParallelExecutionTestControl control;
  control.fail_before_worker_index = 1;
  auto make_work = [](size_t, size_t, int, size_t) {
    return [] {};
  };

  const double elapsed = run_parallel_test_common(
      buffer.data(), buffer.size(), 1, 2, *timer, "startup-injection",
      make_work, nullptr, &metadata, &control);

  EXPECT_EQ(elapsed, 0.0);
  EXPECT_TRUE(metadata.worker_startup_failed);
  EXPECT_EQ(metadata.requested_workers, 2);
  EXPECT_EQ(metadata.created_workers, 1);
  EXPECT_EQ(metadata.qos_successful_workers + metadata.qos_failed_workers, 1u);
}
