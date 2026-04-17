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
// memory_read_strided_2mb_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" uint64_t memory_read_strided_2mb_loop_asm(const void* src,
//                                                         size_t byteCount,
//                                                         size_t num_iterations);
// Purpose:
//   Read memory using a fixed 2MB superpage stride with wraparound.
// Arguments:
//   x0 = src (const void*)
//   x1 = byteCount (size_t)
//   x2 = num_iterations (size_t)
// Returns:
//   x0 = 64-bit XOR checksum
// Clobbers:
//   x3-x5, x12-x13, q0-q3 (caller-saved only)
// Implementation Notes:
//   * Per-iteration loop overhead (offset wrap + counter check) is intentional:
//     this kernel measures steady per-access cost under the fixed stride, not
//     peak streaming throughput. Do not unroll without re-baselining all
//     strided benchmark modes.
// Timing Contract:
//   Caller must emit `dsb ish; isb` before reading the start-of-measurement
//   timestamp and another `dsb ish; isb` before reading the end-of-measurement
//   timestamp. This kernel emits no internal fences; barrier discipline is the
//   caller's responsibility for reproducible timing.
// -----------------------------------------------------------------------------

.global _memory_read_strided_2mb_loop_asm
.align 4
_memory_read_strided_2mb_loop_asm:
    mov x3, xzr                 // iteration_count = 0
    mov x4, xzr                 // offset = 0
    mov x12, xzr                // checksum low-half scratch
    mov x13, xzr                // checksum high-half scratch
    eor v0.16b, v0.16b, v0.16b  // zero accumulator 0
    eor v1.16b, v1.16b, v1.16b  // zero accumulator 1

read_strided_2mb_loop:
    cmp x3, x2
    b.hs read_strided_2mb_done

    add x5, x0, x4              // addr = src + offset
    ldp q2, q3, [x5]            // load 32B
    eor v0.16b, v0.16b, v2.16b
    eor v1.16b, v1.16b, v3.16b

    add x4, x4, #0x200, lsl #12 // offset += 2MB stride
    cmp x4, x1
    b.lo read_strided_2mb_next
    sub x4, x4, x1              // wrap

read_strided_2mb_next:
    add x3, x3, #1
    b read_strided_2mb_loop

read_strided_2mb_done:
    eor v0.16b, v0.16b, v1.16b
    umov x12, v0.d[0]
    umov x13, v0.d[1]
    eor x0, x12, x13
    ret
