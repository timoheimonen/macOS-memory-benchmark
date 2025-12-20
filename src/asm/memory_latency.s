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
// -----------------------------------------------------------------------------
// memory_latency_chase_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Purpose:
//   Measure load‑to‑use latency via dependent pointer chasing. Each load feeds
//   the address of the next, forming a serialized chain. Loop is unrolled by 8
//   to reduce branch impact while preserving dependency.
// Arguments:
//   x0 = start_pointer (uintptr_t*)
//   x1 = count (number of pointer dereferences)
// Returns:
//   x0 = final pointer value (acts as a sink to prevent DCE)
// Clobbers:
//   x4‑x5 (temporaries for iteration counts)
// Implementation Notes:
//   * Uses barriers (dsb/isb/dmb) to isolate measurement window and reduce
//     speculative side effects across the unrolled section.
//   * No prefetching: intentional to keep strictly serialized accesses.
// -----------------------------------------------------------------------------

.global _memory_latency_chase_asm
.align 4
_memory_latency_chase_asm:
    // Pre-touch to ensure TLB entries are loaded before timing measurement.
    // This prevents TLB miss overhead from contaminating latency results.
    ldr x4, [x0]                // Prime the first access but don't use result yet
    // DSB ISH: Ensure TLB preload completes before timing measurement.
    // Prevents speculative execution from contaminating latency results.
    dsb ish                     // Data synchronization barrier (inner shareable)
    isb                         // Instruction synchronization barrier

    // Compute unrolled iterations and remainder for exact count handling
    lsr x4, x1, #3              // x4 = count / 8 (unrolled iterations)
    and x5, x1, #7              // x5 = count % 8 (remainder)

    cbz x4, latency_remainder   // Skip unrolled loop if no full groups

    // DMB SY: Full system memory barrier to prevent reordering before measurement.
    // Ensures all prior memory operations complete and prevents speculative loads
    // from affecting the timing window.
    dmb sy                      // Memory barrier to prevent reordering before measurement

latency_loop_unrolled:          // 8 dependent loads per iteration
    // Use pointer chasing with 8x unrolling to minimize branch overhead while
    // maintaining dependency chain. Each load depends on the previous result,
    // creating a true latency measurement rather than throughput.
    // Expected: ~1 cycle per load on L1 hit, ~4-10 cycles on L2 hit,
    // 50+ cycles on DRAM access. Unrolling hides some latency.
    ldr x0, [x0]                // Load 1: Start dependency chain (true latency measurement)
    ldr x0, [x0]                // Load 2: x0 = *x0 (dependent on load 1)
    ldr x0, [x0]                // Load 3: x0 = *x0 (dependent on load 2)
    ldr x0, [x0]                // Load 4: x0 = *x0 (dependent on load 3)
    ldr x0, [x0]                // Load 5: x0 = *x0 (dependent on load 4)
    ldr x0, [x0]                // Load 6: x0 = *x0 (dependent on load 5)
    ldr x0, [x0]                // Load 7: x0 = *x0 (dependent on load 6)
    ldr x0, [x0]                // Load 8: x0 = *x0 (dependent on load 7)
    
    subs x4, x4, #1             // Decrement unrolled iteration count
    b.gt latency_loop_unrolled  // Loop if more groups remain

    // DMB SY: Full system memory barrier after unrolled section to ensure completion.
    // Guarantees all loads in the measurement window have completed before
    // proceeding to remainder handling or return.
    dmb sy                      // Memory barrier after unrolled section to ensure completion

latency_remainder:              // Handle remaining (0-7) dereferences
    // Process remainder with single dependent loads. Maintains same dependency
    // chain characteristics as unrolled loop for consistent latency measurement.
    cbz x5, latency_end         // Skip if no remainder
latency_single:                 // Single dependent dereference loop
    ldr x0, [x0]                // Load: x0 = *x0 (dependent chain)
    subs x5, x5, #1             // Decrement remainder count
    b.gt latency_single         // Loop if more dereferences remain

latency_end:                    // Return final pointer value
    ret                         // Return final pointer to prevent optimizations