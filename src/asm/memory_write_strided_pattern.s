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
// memory_write_strided_phased_pattern_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_strided_phased_pattern_loop_asm(
//       void* dst, size_t byteCount, size_t stride,
//       size_t passes, size_t initial_phase, uint64_t pattern64);
// Purpose:
//   Execute complete valid 32-byte strided pattern write passes using
//   non-temporal STNP stores and advance the starting phase by 32 bytes after
//   each pass.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
//   x2 = stride (size_t)
//   x3 = passes (size_t)
//   x4 = initial_phase (size_t)
//   x5 = pattern64 (uint64_t)
// Returns:
//   (none)
// Clobbers:
//   x6-x10, q0-q1 (caller-saved only; preserves AAPCS64 callee-saved)
// Timing Contract:
//   No internal fences; barrier discipline (dsb ish; isb) remains the caller's
//   responsibility.
// -----------------------------------------------------------------------------

.global _memory_write_strided_phased_pattern_loop_asm
.align 4
_memory_write_strided_phased_pattern_loop_asm:
    cbz x3, write_strided_phased_pattern_done       // passes == 0
    cbz x2, write_strided_phased_pattern_done       // stride == 0
    cmp x1, #32
    b.lo write_strided_phased_pattern_done          // byteCount < 32
    cmp x4, x2
    b.hs write_strided_phased_pattern_done          // initial_phase >= stride

    dup v0.2d, x5
    mov v1.16b, v0.16b

    mov x10, xzr                                    // completed passes
    mov x6, x4                                      // current phase
    sub x7, x1, #32                                 // last valid 32-byte access offset

write_strided_phased_pattern_pass:
    cmp x10, x3
    b.hs write_strided_phased_pattern_done
    mov x8, x6                                      // offset = phase

write_strided_phased_pattern_access:
    cmp x8, x7
    b.hi write_strided_phased_pattern_next_pass
    add x9, x0, x8
    stnp q0, q1, [x9]
    add x8, x8, x2
    b write_strided_phased_pattern_access

write_strided_phased_pattern_next_pass:
    add x6, x6, #32
    cmp x6, x2
    b.lo write_strided_phased_pattern_phase_ready
    sub x6, x6, x2
write_strided_phased_pattern_phase_ready:
    add x10, x10, #1
    b write_strided_phased_pattern_pass

write_strided_phased_pattern_done:
    ret
