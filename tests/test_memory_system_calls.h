// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// Deterministic mmap-family provider shared by memory allocation unit tests.
#ifndef TEST_MEMORY_SYSTEM_CALLS_H
#define TEST_MEMORY_SYSTEM_CALLS_H

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <sys/mman.h>

#include "core/memory/memory_manager.h"

struct FakeMemorySystemCallState {
  static constexpr size_t kSlotSize = 4096;
  static constexpr size_t kSlotCount = 32;

  alignas(4096) std::array<std::byte, kSlotSize * kSlotCount> storage{};
  size_t map_calls = 0;
  size_t advise_calls = 0;
  size_t unmap_calls = 0;
  size_t fail_map_on_call = 0;
  int advise_result = 0;
  int advise_errno = EINVAL;
  int unmap_result = 0;
  int unmap_errno = EINVAL;
  size_t last_map_size = 0;
  size_t last_advise_size = 0;
  int last_protection = 0;
  int last_flags = 0;
  int last_advice = 0;
  void* last_unmapped_pointer = nullptr;
  size_t last_unmapped_size = 0;
};

inline FakeMemorySystemCallState* active_fake_memory_state = nullptr;

inline void* fake_memory_map(void*, size_t size, int protection, int flags,
                             int, off_t) {
  FakeMemorySystemCallState& state = *active_fake_memory_state;
  ++state.map_calls;
  state.last_map_size = size;
  state.last_protection = protection;
  state.last_flags = flags;
  if (state.fail_map_on_call == state.map_calls ||
      state.map_calls > FakeMemorySystemCallState::kSlotCount ||
      size > FakeMemorySystemCallState::kSlotSize) {
    errno = ENOMEM;
    return MAP_FAILED;
  }
  return state.storage.data() +
         (state.map_calls - 1) * FakeMemorySystemCallState::kSlotSize;
}

inline int fake_memory_advise(void*, size_t size, int advice) {
  FakeMemorySystemCallState& state = *active_fake_memory_state;
  ++state.advise_calls;
  state.last_advise_size = size;
  state.last_advice = advice;
  if (state.advise_result != 0) {
    errno = state.advise_errno;
  }
  return state.advise_result;
}

inline int fake_memory_unmap(void* pointer, size_t size) {
  FakeMemorySystemCallState& state = *active_fake_memory_state;
  ++state.unmap_calls;
  state.last_unmapped_pointer = pointer;
  state.last_unmapped_size = size;
  if (state.unmap_result != 0) {
    errno = state.unmap_errno;
  }
  return state.unmap_result;
}

class FakeMemorySystemCallsTest : public testing::Test {
 protected:
  void SetUp() override {
    active_fake_memory_state = &state;
    set_memory_system_calls_for_testing(
        {fake_memory_map, fake_memory_advise, fake_memory_unmap});
  }

  void TearDown() override {
    reset_memory_system_calls_for_testing();
    active_fake_memory_state = nullptr;
  }

  FakeMemorySystemCallState state;
};

#endif  // TEST_MEMORY_SYSTEM_CALLS_H
