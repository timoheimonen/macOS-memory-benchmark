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
// memory_write_strided_16384_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_strided_16384_loop_asm(void* dst,
//                                                        size_t byteCount,
//                                                        size_t num_iterations);
// Purpose:
//   Write zeros using a fixed 16384-byte stride with wraparound.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
//   x2 = num_iterations (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x5, q0-q1 (caller-saved only)
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

.global _memory_write_strided_16384_loop_asm
.align 4
_memory_write_strided_16384_loop_asm:
    mov x3, xzr                 // iteration_count = 0
    mov x4, xzr                 // offset = 0
    movi v0.16b, #0             // zero vector 0
    movi v1.16b, #0             // zero vector 1

write_strided_16384_loop:
    cmp x3, x2
    b.hs write_strided_16384_done

    add x5, x0, x4              // addr = dst + offset
    stnp q0, q1, [x5]           // store 32B

    add x4, x4, #0x4000         // offset += 16384B stride
    cmp x4, x1
    b.lo write_strided_16384_next
    sub x4, x4, x1              // wrap

write_strided_16384_next:
    add x3, x3, #1
    b write_strided_16384_loop

write_strided_16384_done:
    ret
