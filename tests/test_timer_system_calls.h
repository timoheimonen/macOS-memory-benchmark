#ifndef TEST_TIMER_SYSTEM_CALLS_H
#define TEST_TIMER_SYSTEM_CALLS_H

#include <cstdint>

#include "core/timing/timer.h"

namespace test_timer_system_calls {

inline kern_return_t unit_timebase_info(mach_timebase_info_t info) {
  info->numer = 1;
  info->denom = 1;
  return KERN_SUCCESS;
}

/** Restores the production Mach clock hooks when a deterministic test exits. */
template <uint64_t (*AbsoluteTime)(), void (*BeforeInstall)() = nullptr>
class ScopedTimerSystemCalls {
 public:
  ScopedTimerSystemCalls() {
    if constexpr (BeforeInstall != nullptr) {
      BeforeInstall();
    }
    set_timer_system_calls_for_testing({AbsoluteTime, unit_timebase_info});
  }

  ScopedTimerSystemCalls(const ScopedTimerSystemCalls&) = delete;
  ScopedTimerSystemCalls& operator=(const ScopedTimerSystemCalls&) = delete;

  ~ScopedTimerSystemCalls() { reset_timer_system_calls_for_testing(); }
};

}  // namespace test_timer_system_calls

#endif  // TEST_TIMER_SYSTEM_CALLS_H
