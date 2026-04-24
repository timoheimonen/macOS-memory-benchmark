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
// -----------------------------------------------------------------------------
// memory_write_reverse_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_reverse_loop_asm(void* dst, size_t byteCount);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' in reverse order (backwards) using
//   512B block stores to measure raw write bandwidth with reverse sequential pattern.
//   Uses STNP as a non‑temporal hint.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3‑x7, q0‑q7 (zero vectors, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Zero vectors materialized once (8 regs), reused for all 16 stnp pairs.
//   * Tiered tail mirrors copy/read for consistency.
//   * Accesses memory from end to start, testing reverse sequential write behavior.
//   * Main loop label is 64-byte aligned to keep the unrolled body on a single
//     I-cache line boundary for steady run-to-run timing on Apple Silicon.
// Timing Contract:
//   Caller must emit `dsb ish; isb` before reading the start-of-measurement
//   timestamp and another `dsb ish; isb` before reading the end-of-measurement
//   timestamp. This kernel emits no internal fences; barrier discipline is the
//   caller's responsibility for reproducible timing.
// -----------------------------------------------------------------------------

.global _memory_write_reverse_loop_asm
.align 4
_memory_write_reverse_loop_asm:
    mov x3, xzr             // offset = 0 (tracked as end_ptr)
    add x3, x0, x1          // end_ptr = dst + byteCount
    // 512B blocks chosen as sweet spot: large enough for high throughput,
    // small enough to fit in L2 cache, avoids TLB pressure on large regions
    mov x4, #512            // step = 512 bytes (optimal block size)

    // Zero out 8 data registers (caller-saved: v0-v7, avoiding v8-v15 per AAPCS64).
    // All 16 stnp pairs reuse these 8 registers (zeros are identical).
    movi v0.16b, #0
    movi v1.16b, #0
    movi v2.16b, #0
    movi v3.16b, #0
    movi v4.16b, #0
    movi v5.16b, #0
    movi v6.16b, #0
    movi v7.16b, #0


    // Align hot loop entry to 64B so the unrolled 512B body always lands on a
    // predictable I-cache line. Reduces first-iteration fetch-boundary jitter.
    .p2align 6
write_reverse_loop_start_nt512:     // Main 512B block loop (reverse direction)
    cmp x3, x0              // end_ptr <= dst?
    b.ls write_reverse_loop_end     // If done (unsigned <=), exit

    sub x5, x3, x0          // remaining = end_ptr - dst
    cmp x5, x4              // remaining < step?
    b.lo write_reverse_cleanup // If less (unsigned), handle remaining bytes

    sub x7, x3, #512        // block_start = end_ptr - 512

    // Store 512B zeros (non-temporal) (16 stnp pairs).
    // Non-temporal stores (stnp) hint to CPU that data won't be reused soon,
    // encouraging write-combining and reducing cache pollution during bandwidth tests.
    stnp q0,  q1,  [x7, #0]
    stnp q2,  q3,  [x7, #32]
    stnp q4,  q5,  [x7, #64]
    stnp q6,  q7,  [x7, #96]
    stnp q0,  q1,  [x7, #128]
    stnp q2,  q3,  [x7, #160]
    stnp q4,  q5,  [x7, #192]
    stnp q6,  q7,  [x7, #224]
    stnp q0,  q1,  [x7, #256]
    stnp q2,  q3,  [x7, #288]
    stnp q4,  q5,  [x7, #320]
    stnp q6,  q7,  [x7, #352]
    stnp q0,  q1,  [x7, #384]
    stnp q2,  q3,  [x7, #416]
    stnp q4,  q5,  [x7, #448]
    stnp q6,  q7,  [x7, #480]

    sub x3, x3, x4          // end_ptr -= step (move backwards)
    b write_reverse_loop_start_nt512 // Loop again

write_reverse_cleanup:         // Tail handling when <512B remain
    cmp x3, x0              // end_ptr <= dst?
    b.ls write_reverse_loop_end     // If done (unsigned <=), exit

    subs x5, x3, x0         // Recalc remaining (x5)
    sub x7, x3, x5          // block_start = end_ptr - remaining

    // Handle 256B chunks (8 stnp pairs)
    cmp x5, #256
    b.lo write_reverse_cleanup_128
    sub x7, x3, #256
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    stnp q0, q1, [x7, #128]
    stnp q2, q3, [x7, #160]
    stnp q4, q5, [x7, #192]
    stnp q6, q7, [x7, #224]
    sub x3, x3, #256          // Advance end_ptr backwards by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

write_reverse_cleanup_128:            // 128B chunk
    cmp x5, #128
    b.lo write_reverse_cleanup_64
    sub x7, x3, #128
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    stnp q4, q5, [x7, #64]
    stnp q6, q7, [x7, #96]
    sub x3, x3, #128
    sub x5, x5, #128

write_reverse_cleanup_64:             // 64B chunk
    cmp x5, #64
    b.lo write_reverse_cleanup_32
    sub x7, x3, #64
    stnp q0, q1, [x7, #0]
    stnp q2, q3, [x7, #32]
    sub x3, x3, #64
    sub x5, x5, #64

write_reverse_cleanup_32:             // 32B chunk
    cmp x5, #32
    b.lo write_reverse_cleanup_byte
    sub x7, x3, #32
    stnp q0, q1, [x7, #0]
    sub x3, x3, #32
    sub x5, x5, #32

write_reverse_cleanup_byte:            // Byte tail (<32B)
    cbz x5, write_reverse_loop_end
    sub x7, x3, #1
    strb wzr, [x7]
    sub x3, x3, #1
    subs x5, x5, #1
    b.ne write_reverse_cleanup_byte

write_reverse_loop_end:             // Return to caller
    ret                     // Return
