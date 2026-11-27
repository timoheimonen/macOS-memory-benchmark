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
#include "core/config/constants.h"
#include "core/memory/memory_utils.h"
#include "core/timing/timer.h"
#include "test_timer_system_calls.h"

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
  unsigned char* accessible_begin() const {
    return static_cast<unsigned char*>(mapping_) + page_size_;
  }
  size_t prefix_size() const {
    return static_cast<size_t>(payload_ - accessible_begin());
  }
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

uint64_t expected_reverse_checksum(const unsigned char* data, size_t size) {
  const size_t byte_prefix = size % 32;
  uint64_t checksum = 0;
  for (size_t index = byte_prefix; index < size; ++index) {
    checksum ^= static_cast<uint64_t>(data[index])
                << (((index - byte_prefix) % 8) * 8);
  }
  uint64_t byte_prefix_checksum = 0;
  for (size_t index = 0; index < byte_prefix; ++index) {
    byte_prefix_checksum ^= data[index];
  }
  return checksum ^ byte_prefix_checksum;
}

uint64_t expected_random_checksum(const unsigned char* data,
                                  const std::vector<size_t>& indices) {
  uint64_t checksum = 0;
  for (size_t offset : indices) {
    for (size_t lane = 0; lane < 4; ++lane) {
      uint64_t value = 0;
      std::memcpy(&value, data + offset + lane * sizeof(value), sizeof(value));
      checksum ^= value;
    }
  }
  return checksum;
}

using ReadKernel = uint64_t (*)(const void*, size_t);
using WriteKernel = void (*)(void*, size_t);
using CopyKernel = void (*)(void*, const void*, size_t);

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

void verify_reverse_read_kernel_boundaries(ReadKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping source(size);
    ASSERT_TRUE(source.valid());
    std::memset(source.accessible_begin(), 0xd3, source.prefix_size());
    for (size_t index = 0; index < size; ++index) {
      source.payload()[index] =
          static_cast<unsigned char>((index * 41 + 13) & 0xff);
    }
    const std::vector<unsigned char> prefix_before(
        source.accessible_begin(), source.payload());
    const std::vector<unsigned char> payload_before(source.payload(),
                                                    source.payload() + size);

    EXPECT_EQ(kernel(source.payload(), size),
              expected_reverse_checksum(source.payload(), size))
        << "size=" << size;
    EXPECT_TRUE(std::equal(prefix_before.begin(), prefix_before.end(),
                           source.accessible_begin()))
        << "size=" << size;
    EXPECT_TRUE(std::equal(payload_before.begin(), payload_before.end(),
                           source.payload()))
        << "size=" << size;
  }
}

void verify_reverse_write_kernel_boundaries(WriteKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping destination(size);
    ASSERT_TRUE(destination.valid());
    std::memset(destination.accessible_begin(), 0xd3,
                destination.prefix_size());
    std::memset(destination.payload(), 0xa5, size);
    const std::vector<unsigned char> prefix_before(
        destination.accessible_begin(), destination.payload());

    kernel(destination.payload(), size);

    EXPECT_TRUE(std::equal(prefix_before.begin(), prefix_before.end(),
                           destination.accessible_begin()))
        << "size=" << size;
    for (size_t index = 0; index < size; ++index) {
      EXPECT_EQ(destination.payload()[index], 0u)
          << "size=" << size << " index=" << index;
    }
  }
}

void verify_reverse_copy_kernel_boundaries(CopyKernel kernel) {
  for (size_t size : kTailSizes) {
    GuardedMapping source(size);
    GuardedMapping destination(size);
    ASSERT_TRUE(source.valid());
    ASSERT_TRUE(destination.valid());
    std::memset(source.accessible_begin(), 0xc7, source.prefix_size());
    std::memset(destination.accessible_begin(), 0xd3,
                destination.prefix_size());
    for (size_t index = 0; index < size; ++index) {
      source.payload()[index] =
          static_cast<unsigned char>((index * 23 + 5) & 0xff);
    }
    std::memset(destination.payload(), 0xa5, size);
    const std::vector<unsigned char> source_prefix_before(
        source.accessible_begin(), source.payload());
    const std::vector<unsigned char> destination_prefix_before(
        destination.accessible_begin(), destination.payload());
    const std::vector<unsigned char> source_before(source.payload(),
                                                   source.payload() + size);

    kernel(destination.payload(), source.payload(), size);

    EXPECT_TRUE(std::equal(source_prefix_before.begin(),
                           source_prefix_before.end(),
                           source.accessible_begin()))
        << "size=" << size;
    EXPECT_TRUE(std::equal(destination_prefix_before.begin(),
                           destination_prefix_before.end(),
                           destination.accessible_begin()))
        << "size=" << size;
    EXPECT_TRUE(std::equal(source_before.begin(), source_before.end(),
                           source.payload()))
        << "size=" << size;
    EXPECT_TRUE(std::equal(source_before.begin(), source_before.end(),
                           destination.payload()))
        << "size=" << size;
  }
}

}  // namespace

TEST(StandardKernelIntegrationTest, MainAndCacheReadsHonorTailsAndChecksum) {
  for (ReadKernel kernel : {memory_read_loop_asm, memory_read_cache_loop_asm}) {
    SCOPED_TRACE(kernel == memory_read_loop_asm ? "main" : "cache");
    verify_read_kernel_boundaries(kernel);
  }
}

TEST(PatternKernelIntegrationTest, ReverseReadHonorsTailsGuardsAndChecksum) {
  verify_reverse_read_kernel_boundaries(memory_read_reverse_loop_asm);
}

TEST(PatternKernelIntegrationTest, ReverseWriteHonorsExactGuardedBoundaries) {
  verify_reverse_write_kernel_boundaries(memory_write_reverse_loop_asm);
}

TEST(PatternKernelIntegrationTest, ReverseCopyHonorsExactGuardedBoundaries) {
  verify_reverse_copy_kernel_boundaries(memory_copy_reverse_loop_asm);
}

TEST(PatternKernelIntegrationTest, RandomReadHonorsIndicesLastAccessAndChecksum) {
  constexpr size_t kPayloadSize = 160;
  const std::vector<size_t> indices = {0, 64, 128};
  GuardedMapping source(kPayloadSize);
  ASSERT_TRUE(source.valid());
  std::memset(source.accessible_begin(), 0xd3, source.prefix_size());
  for (size_t index = 0; index < kPayloadSize; ++index) {
    source.payload()[index] =
        static_cast<unsigned char>((index * 29 + 17) & 0xff);
  }
  const std::vector<unsigned char> prefix_before(source.accessible_begin(),
                                                 source.payload());
  const std::vector<unsigned char> payload_before(
      source.payload(), source.payload() + kPayloadSize);

  EXPECT_EQ(memory_read_random_loop_asm(source.payload(), indices.data(),
                                        indices.size()),
            expected_random_checksum(source.payload(), indices));
  EXPECT_TRUE(std::equal(prefix_before.begin(), prefix_before.end(),
                         source.accessible_begin()));
  EXPECT_TRUE(std::equal(payload_before.begin(), payload_before.end(),
                         source.payload()));
  EXPECT_EQ(memory_read_random_loop_asm(source.payload(), indices.data(), 0),
            0u);
}

TEST(PatternKernelIntegrationTest, RandomWriteTouchesOnlySelectedAccesses) {
  constexpr size_t kPayloadSize = 160;
  const std::vector<size_t> indices = {0, 64, 128};
  GuardedMapping destination(kPayloadSize);
  ASSERT_TRUE(destination.valid());
  std::memset(destination.accessible_begin(), 0xd3,
              destination.prefix_size());
  std::memset(destination.payload(), 0xa5, kPayloadSize);
  const std::vector<unsigned char> prefix_before(
      destination.accessible_begin(), destination.payload());

  memory_write_random_loop_asm(destination.payload(), indices.data(),
                               indices.size());

  EXPECT_TRUE(std::equal(prefix_before.begin(), prefix_before.end(),
                         destination.accessible_begin()));
  for (size_t index = 0; index < kPayloadSize; ++index) {
    const bool selected = index < 32 || (index >= 64 && index < 96) ||
                          (index >= 128 && index < 160);
    EXPECT_EQ(destination.payload()[index], selected ? 0u : 0xa5u)
        << "index=" << index;
  }
}

TEST(PatternKernelIntegrationTest, RandomCopyTouchesOnlySelectedAccesses) {
  constexpr size_t kPayloadSize = 160;
  const std::vector<size_t> indices = {0, 64, 128};
  GuardedMapping source(kPayloadSize);
  GuardedMapping destination(kPayloadSize);
  ASSERT_TRUE(source.valid());
  ASSERT_TRUE(destination.valid());
  std::memset(source.accessible_begin(), 0xc7, source.prefix_size());
  std::memset(destination.accessible_begin(), 0xd3,
              destination.prefix_size());
  for (size_t index = 0; index < kPayloadSize; ++index) {
    source.payload()[index] =
        static_cast<unsigned char>((index * 31 + 7) & 0xff);
  }
  std::memset(destination.payload(), 0xa5, kPayloadSize);
  const std::vector<unsigned char> source_prefix_before(
      source.accessible_begin(), source.payload());
  const std::vector<unsigned char> destination_prefix_before(
      destination.accessible_begin(), destination.payload());
  const std::vector<unsigned char> source_before(
      source.payload(), source.payload() + kPayloadSize);

  memory_copy_random_loop_asm(destination.payload(), source.payload(),
                              indices.data(), indices.size());

  EXPECT_TRUE(std::equal(source_prefix_before.begin(),
                         source_prefix_before.end(),
                         source.accessible_begin()));
  EXPECT_TRUE(std::equal(destination_prefix_before.begin(),
                         destination_prefix_before.end(),
                         destination.accessible_begin()));
  EXPECT_TRUE(std::equal(source_before.begin(), source_before.end(),
                         source.payload()));
  for (size_t index = 0; index < kPayloadSize; ++index) {
    const bool selected = index < 32 || (index >= 64 && index < 96) ||
                          (index >= 128 && index < 160);
    EXPECT_EQ(destination.payload()[index],
              selected ? source.payload()[index] : 0xa5u)
        << "index=" << index;
  }
}

TEST(StandardKernelIntegrationTest, MainAndCacheWritesHonorExactBoundaries) {
  verify_write_kernel_boundaries(memory_write_loop_asm);
  verify_write_kernel_boundaries(memory_write_cache_loop_asm);
}

TEST(StandardKernelIntegrationTest, MainAndCacheCopiesHonorExactBoundaries) {
  verify_copy_kernel_boundaries(memory_copy_loop_asm);
  verify_copy_kernel_boundaries(memory_copy_cache_loop_asm);
}

TEST(StandardKernelIntegrationTest, StandardAndPatternKernelsPreserveCalleeSavedRegisters) {
  std::vector<unsigned char> source(Constants::PATTERN_STRIDE_SUPERPAGE_2MB + Constants::PATTERN_ACCESS_SIZE_BYTES);
  std::vector<unsigned char> destination(source.size());
  for (size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<unsigned char>(index & 0xff);
  }
  constexpr size_t kSize = 513;

  const std::array<size_t, 3> indices = {0, 64, 128};
  const uintptr_t src = reinterpret_cast<uintptr_t>(source.data());
  const uintptr_t dst = reinterpret_cast<uintptr_t>(destination.data());
  const uintptr_t index_data = reinterpret_cast<uintptr_t>(indices.data());
  struct KernelCase {
    const char* name;
    uintptr_t function;
    std::array<uintptr_t, 6> arguments;
  };
  std::vector<KernelCase> cases = {
      {"read", reinterpret_cast<uintptr_t>(memory_read_loop_asm), {src, kSize, 0, 0, 0, 0}},
      {"write", reinterpret_cast<uintptr_t>(memory_write_loop_asm), {dst, kSize, 0, 0, 0, 0}},
      {"copy", reinterpret_cast<uintptr_t>(memory_copy_loop_asm), {dst, src, kSize, 0, 0, 0}},
      {"read_cache", reinterpret_cast<uintptr_t>(memory_read_cache_loop_asm), {src, kSize, 0, 0, 0, 0}},
      {"write_cache", reinterpret_cast<uintptr_t>(memory_write_cache_loop_asm), {dst, kSize, 0, 0, 0, 0}},
      {"copy_cache", reinterpret_cast<uintptr_t>(memory_copy_cache_loop_asm), {dst, src, kSize, 0, 0, 0}},
      {"read_reverse", reinterpret_cast<uintptr_t>(memory_read_reverse_loop_asm), {src, kSize, 0, 0, 0, 0}},
      {"write_reverse", reinterpret_cast<uintptr_t>(memory_write_reverse_loop_asm), {dst, kSize, 0, 0, 0, 0}},
      {"copy_reverse", reinterpret_cast<uintptr_t>(memory_copy_reverse_loop_asm), {dst, src, kSize, 0, 0, 0}},
      {"read_random",
       reinterpret_cast<uintptr_t>(memory_read_random_loop_asm),
       {src, index_data, indices.size(), 0, 0, 0}},
      {"write_random",
       reinterpret_cast<uintptr_t>(memory_write_random_loop_asm),
       {dst, index_data, indices.size(), 0, 0, 0}},
      {"copy_random",
       reinterpret_cast<uintptr_t>(memory_copy_random_loop_asm),
       {dst, src, index_data, indices.size(), 0, 0}},
  };
  for (size_t stride : {Constants::PATTERN_STRIDE_CACHE_LINE, Constants::PATTERN_STRIDE_PAGE,
                        Constants::PATTERN_STRIDE_PAGE_16K, Constants::PATTERN_STRIDE_SUPERPAGE_2MB}) {
    const size_t span = stride + Constants::PATTERN_ACCESS_SIZE_BYTES;
    cases.push_back({"read_strided",
                     reinterpret_cast<uintptr_t>(memory_read_strided_phased_loop_asm),
                     {src, span, stride, 2, 0, 0}});
    cases.push_back({"write_strided",
                     reinterpret_cast<uintptr_t>(memory_write_strided_phased_loop_asm),
                     {dst, span, stride, 2, 0, 0}});
    cases.push_back({"copy_strided",
                     reinterpret_cast<uintptr_t>(memory_copy_strided_phased_loop_asm),
                     {dst, src, span, stride, 2, 0}});
  }
  for (const KernelCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    SCOPED_TRACE(::testing::PrintToString(entry.arguments));
    const auto& args = entry.arguments;
    EXPECT_EQ(
        verify_pattern_callee_saved_registers_asm(entry.function, args[0], args[1], args[2], args[3], args[4], args[5]),
        1u);
  }
  ASSERT_EQ(setup_latency_chain(source.data(), 1024, 256, 0, LatencyChainMode::GlobalRandom, 12345), EXIT_SUCCESS);
  EXPECT_EQ(verify_pattern_callee_saved_registers_asm(reinterpret_cast<uintptr_t>(memory_latency_chase_asm), src, 16, 0,
                                                      0, 0, 0),
            1u);
}

namespace {

uint64_t deterministic_executor_timer_tick = 0;

uint64_t deterministic_executor_timer_ticks() {
  deterministic_executor_timer_tick += 100;
  return deterministic_executor_timer_tick;
}

void reset_deterministic_executor_timer_ticks() {
  deterministic_executor_timer_tick = 0;
}

struct ThreadExitTickProbe {
  std::atomic<uint64_t>* observed_tick;

  ~ThreadExitTickProbe() {
    observed_tick->store(deterministic_executor_timer_tick,
                         std::memory_order_relaxed);
  }
};

using ScopedDeterministicExecutorTimerSystemCalls =
    test_timer_system_calls::ScopedTimerSystemCalls<
        deterministic_executor_timer_ticks,
        reset_deterministic_executor_timer_ticks>;

}  // namespace

TEST(StandardKernelIntegrationTest,
     ParallelExecutorConsumesAlignedUnevenChunksExactly) {
  const ScopedDeterministicExecutorTimerSystemCalls timer_system_calls;
  alignas(Constants::CACHE_LINE_SIZE_BYTES)
      std::array<unsigned char, 2049> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  std::array<size_t, 10> observed_offsets{};
  std::array<size_t, 10> observed_sizes{};
  std::array<std::atomic<bool>, 10> executed;
  for (std::atomic<bool>& flag : executed) {
    flag.store(false, std::memory_order_relaxed);
  }
  auto make_work = [&](size_t offset, size_t size, int iterations,
                       size_t worker_index) {
    observed_offsets[worker_index] = offset;
    observed_sizes[worker_index] = size;
    EXPECT_EQ(iterations, 1);
    return [&, worker_index] {
      executed[worker_index].store(true, std::memory_order_relaxed);
    };
  };
  ParallelExecutionMetadata metadata;
  const double elapsed = run_parallel_test_common(
      buffer.data(), buffer.size(), 1, 10, *timer, "partition-test", make_work,
      nullptr, &metadata);

  EXPECT_DOUBLE_EQ(elapsed, 100.0 / 1e9);
  EXPECT_EQ(metadata.created_workers, 10);
  EXPECT_FALSE(metadata.worker_startup_failed);

  size_t covered = 0;
  for (size_t index = 0; index < observed_offsets.size(); ++index) {
    EXPECT_EQ(observed_offsets[index], covered);
    EXPECT_GT(observed_sizes[index], 0u);
    EXPECT_TRUE(executed[index].load(std::memory_order_relaxed));
    covered += observed_sizes[index];
    if (index + 1 < observed_offsets.size()) {
      const uintptr_t boundary_address =
          reinterpret_cast<uintptr_t>(buffer.data()) + covered;
      EXPECT_EQ(boundary_address % Constants::CACHE_LINE_SIZE_BYTES, 0u);
    }
  }
  EXPECT_EQ(covered, buffer.size());
}

TEST(StandardKernelIntegrationTest, ParallelTimingStopsBeforeWorkerTeardown) {
  const ScopedDeterministicExecutorTimerSystemCalls timer_system_calls;
  std::array<unsigned char, 4096> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  uint64_t tick_during_work = 0;
  std::atomic<uint64_t> tick_at_worker_teardown{0};
  auto make_work = [&](size_t, size_t, int, size_t) {
    return [&] {
      thread_local ThreadExitTickProbe probe{&tick_at_worker_teardown};
      static_cast<void>(probe);
      tick_during_work = deterministic_executor_timer_tick;
    };
  };

  const double measured_duration = run_parallel_test_common(
      buffer.data(), buffer.size(), 1, 1, *timer, "timing_test", make_work);

  EXPECT_DOUBLE_EQ(measured_duration, 100.0 / 1e9);
  EXPECT_EQ(tick_during_work, 100u);
  EXPECT_EQ(tick_at_worker_teardown.load(std::memory_order_relaxed), 200u);
  EXPECT_EQ(deterministic_executor_timer_tick, 200u);
}

TEST(StandardKernelIntegrationTest,
     LatencySamplingClampsAndContinuesFromPriorTerminalPointer) {
  const ScopedDeterministicExecutorTimerSystemCalls timer_system_calls;
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  auto verify_sampling = [&](int requested_samples,
                             const std::vector<size_t>& expected_access_counts,
                             const std::vector<double>& expected_samples) {
    reset_deterministic_executor_timer_ticks();
    std::array<uintptr_t, 8> nodes{};
    std::vector<uintptr_t*> observed_starts;
    std::vector<size_t> observed_access_counts;
    size_t chase_calls = 0;
    LatencyMeasurementTestHooks hooks;
    hooks.chase = [&](uintptr_t* start, size_t access_count) {
      observed_starts.push_back(start);
      observed_access_counts.push_back(access_count);
      ++chase_calls;
      return &nodes[chase_calls];
    };

    std::vector<double> samples;
    const double total_duration_ns = run_latency_test(
        &nodes[0], 7, *timer, &samples, requested_samples, &hooks);

    std::vector<uintptr_t*> expected_starts;
    for (size_t index = 0; index < expected_access_counts.size(); ++index) {
      expected_starts.push_back(&nodes[index]);
    }
    EXPECT_DOUBLE_EQ(
        total_duration_ns,
        100.0 * static_cast<double>(expected_access_counts.size()));
    EXPECT_EQ(samples, expected_samples);
    EXPECT_EQ(observed_starts, expected_starts);
    EXPECT_EQ(observed_access_counts, expected_access_counts);
  };

  verify_sampling(3, {3, 2, 2}, {100.0 / 3.0, 50.0, 50.0});
  verify_sampling(100, {1, 1, 1, 1, 1, 1, 1},
                  {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0});
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
  EXPECT_FALSE(metadata.timing_snapshot.has_value());
}

namespace {
std::atomic<uint64_t> deterministic_kernel_ticks{0};

void reset_deterministic_kernel_ticks() {
  deterministic_kernel_ticks.store(0, std::memory_order_relaxed);
}

uint64_t next_deterministic_kernel_ticks() {
  const uint64_t call =
      deterministic_kernel_ticks.fetch_add(1, std::memory_order_relaxed);
  return call == 0 ? 1000ULL : 2500ULL;
}
}  // namespace

TEST(StandardKernelIntegrationTest, ParallelExecutionMetadataPropagatesMachTimingSnapshot) {
  test_timer_system_calls::ScopedTimerSystemCalls<
      next_deterministic_kernel_ticks, reset_deterministic_kernel_ticks>
      scoped_timer;

  alignas(64) std::array<unsigned char, 4096> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());

  ParallelExecutionMetadata metadata;
  auto make_work = [](size_t, size_t, int, size_t) {
    return [] {};
  };

  const double elapsed = run_parallel_test_common(
      buffer.data(), buffer.size(), 1, 2, *timer, "timing-snapshot",
      make_work, nullptr, &metadata);

  EXPECT_GT(elapsed, 0.0);
  EXPECT_FALSE(metadata.worker_startup_failed);
  ASSERT_TRUE(metadata.timing_snapshot.has_value());
  ASSERT_TRUE(timer->last_snapshot.has_value());
  EXPECT_EQ(metadata.timing_snapshot->start_ticks, 1000ULL);
  EXPECT_EQ(metadata.timing_snapshot->stop_ticks, 2500ULL);
  EXPECT_EQ(metadata.timing_snapshot->delta_ticks, 1500ULL);
  EXPECT_EQ(metadata.timing_snapshot->numer, 1u);
  EXPECT_EQ(metadata.timing_snapshot->denom, 1u);
}

TEST(StandardKernelIntegrationTest, ParallelExecutionEarlyReturnsLeaveTimingSnapshotNullopt) {
  alignas(64) std::array<unsigned char, 4096> buffer{};
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  auto make_work = [](size_t, size_t, int, size_t) {
    return [] {};
  };

  ParallelExecutionMetadata metadata;
  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_common(buffer.data(), 0, 1, 2, *timer, "zero-size",
                                     make_work, nullptr, &metadata),
            0.0);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());

  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_common(buffer.data(), buffer.size(), 1, 0, *timer,
                                     "zero-threads", make_work, nullptr, &metadata),
            0.0);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());

  const std::vector<size_t> invalid_boundaries = {0, 100};
  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_common(buffer.data(), buffer.size(), 1, 1, *timer,
                                     "bad-boundaries", make_work,
                                     &invalid_boundaries, &metadata),
            0.0);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());

  ParallelExecutionTestControl fail_first;
  fail_first.fail_before_worker_index = 0;
  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_common(buffer.data(), buffer.size(), 1, 2, *timer,
                                     "fail-first", make_work, nullptr,
                                     &metadata, &fail_first),
            0.0);
  EXPECT_TRUE(metadata.worker_startup_failed);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());

  const std::vector<size_t> single_boundary = {0};
  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_indexed_with_boundaries(
                buffer.data(), buffer.size(), 1, *timer, single_boundary,
                [](void*, size_t, int, size_t) {}, "bad-indexed", &metadata),
            0.0);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());

  metadata.timing_snapshot = MachTimingSnapshot{1, 2, 1, 1, 1};
  EXPECT_EQ(run_parallel_test_copy_indexed_with_boundaries(
                buffer.data(), buffer.data(), buffer.size(), 1, *timer,
                single_boundary,
                [](void*, void*, size_t, int, size_t) {}, "bad-copy-indexed",
                &metadata),
            0.0);
  EXPECT_FALSE(metadata.timing_snapshot.has_value());
}
