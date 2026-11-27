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
// memory_write_random_pattern_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_random_pattern_loop_asm(
//       void* dst, const size_t* indices, size_t num_accesses, uint64_t pattern64);
// Purpose:
//   Write 32-byte non-temporal pattern blocks (STNP) at random offsets given
//   by 'indices', repeating the little-endian 64-bit 'pattern64'.
// Arguments:
//   x0 = dst (void*)
//   x1 = indices (const size_t*) - array of byte offsets into dst
//   x2 = num_accesses (size_t) - number of accesses to perform
//   x3 = pattern64 (uint64_t)
// Returns:
//   (none)
// Clobbers:
//   x4-x6, q0-q1 (caller-saved only; preserves AAPCS64 callee-saved)
// Implementation Notes:
//   * Single-iteration loop (do not unroll without re-baselining random benchmark modes).
//   * No internal barriers (dsb/isb); timing barriers remain caller's responsibility.
// -----------------------------------------------------------------------------

.global _memory_write_random_pattern_loop_asm
.align 4
_memory_write_random_pattern_loop_asm:
    cbz x2, write_random_pattern_end                // num_accesses == 0
    mov x6, xzr                                     // i = 0 (loop counter)

    dup v0.2d, x3
    mov v1.16b, v0.16b

write_random_pattern_loop:
    cmp x6, x2                                      // i >= num_accesses?
    b.hs write_random_pattern_end
    ldr x4, [x1, x6, lsl #3]                        // x4 = indices[i]
    add x5, x0, x4                                  // addr = dst + indices[i]
    stnp q0, q1, [x5]                               // store 32B pattern (non-temporal)
    add x6, x6, #1                                  // i++
    b write_random_pattern_loop

write_random_pattern_end:
    ret
