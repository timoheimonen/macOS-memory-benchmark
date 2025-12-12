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
// memory_write_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_loop_asm(void* dst, size_t byteCount);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' using 512B block stores
//   to measure raw write bandwidth. Uses STNP as a non‑temporal hint.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3‑x7, q0‑q7, q16‑q31 (zero vectors, avoiding q8‑q15 per AAPCS64)
// Implementation Notes:
//   * Zero vectors are materialized once (movi) then reused.
//   * Tiered tail mirrors copy/read for consistency.
// -----------------------------------------------------------------------------

.global _memory_write_loop_asm
.align 4
_memory_write_loop_asm:
    mov x3, xzr             // offset = 0
    mov x4, #512            // step = 512 bytes

    // Zero out data registers (using only caller-saved: v0-v7 and v16-v31, avoiding v8-v15 per AAPCS64)
    movi v0.16b, #0             // Zero vector 0
    movi v1.16b, #0             // Zero vector 1
    movi v2.16b, #0             // Zero vector 2
    movi v3.16b, #0             // Zero vector 3
    movi v4.16b, #0             // Zero vector 4
    movi v5.16b, #0             // Zero vector 5
    movi v6.16b, #0             // Zero vector 6
    movi v7.16b, #0             // Zero vector 7
    movi v16.16b, #0            // Zero vector 16
    movi v17.16b, #0            // Zero vector 17
    movi v18.16b, #0            // Zero vector 18
    movi v19.16b, #0            // Zero vector 19
    movi v20.16b, #0            // Zero vector 20
    movi v21.16b, #0            // Zero vector 21
    movi v22.16b, #0            // Zero vector 22
    movi v23.16b, #0            // Zero vector 23
    movi v24.16b, #0            // Zero vector 24
    movi v25.16b, #0            // Zero vector 25
    movi v26.16b, #0            // Zero vector 26
    movi v27.16b, #0            // Zero vector 27
    movi v28.16b, #0            // Zero vector 28
    movi v29.16b, #0            // Zero vector 29
    movi v30.16b, #0            // Zero vector 30
    movi v31.16b, #0            // Zero vector 31


write_loop_start_nt512:     // Main 512B block loop
    subs x5, x1, x3         // remaining = count - offset
    cmp x5, x4              // remaining < step?
    b.lt write_loop_cleanup // If less, handle remaining bytes

    add x7, x0, x3          // dst_addr = base + offset

    // Store 512B zeros (non-temporal) (16 stnp pairs)
    // Using only caller-saved registers: q0-q7 and q16-q31
    stnp q0,  q1,  [x7, #0]       // Store pair 0 (offset 0, non-temporal)
    stnp q2,  q3,  [x7, #32]      // Store pair 1 (offset 32, non-temporal)
    stnp q4,  q5,  [x7, #64]      // Store pair 2 (offset 64, non-temporal)
    stnp q6,  q7,  [x7, #96]      // Store pair 3 (offset 96, non-temporal)
    stnp q16, q17, [x7, #128]     // Store pair 4 (offset 128, non-temporal)
    stnp q18, q19, [x7, #160]     // Store pair 5 (offset 160, non-temporal)
    stnp q20, q21, [x7, #192]     // Store pair 6 (offset 192, non-temporal)
    stnp q22, q23, [x7, #224]     // Store pair 7 (offset 224, non-temporal)
    stnp q24, q25, [x7, #256]     // Store pair 8 (offset 256, non-temporal)
    stnp q26, q27, [x7, #288]     // Store pair 9 (offset 288, non-temporal)
    stnp q28, q29, [x7, #320]     // Store pair 10 (offset 320, non-temporal)
    stnp q30, q31, [x7, #352]     // Store pair 11 (offset 352, non-temporal)
    // Reuse q0-q7 for remaining pairs (all zeros, so safe to reuse)
    stnp q0,  q1,  [x7, #384]     // Store pair 12 (offset 384, non-temporal)
    stnp q2,  q3,  [x7, #416]     // Store pair 13 (offset 416, non-temporal)
    stnp q4,  q5,  [x7, #448]     // Store pair 14 (offset 448, non-temporal)
    stnp q6,  q7,  [x7, #480]     // Store pair 15 (offset 480, non-temporal)

    add x3, x3, x4          // offset += step
    b write_loop_start_nt512 // Loop again

write_loop_cleanup:         // Tail handling when <512B remain
    cmp x3, x1              // offset == count?
    b.ge write_loop_end     // If done, exit

    subs x5, x1, x3         // Recalc remaining (x5)
    add x7, x0, x3          // dst_addr = dst + offset

    // Handle 256B chunks (8 stnp pairs)
    cmp x5, #256              // Check if >= 256 bytes remain
    b.lt write_cleanup_128    // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal, 32B)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal, 32B)
    stnp q16, q17, [x7, #128] // Store fifth pair (non-temporal, 32B)
    stnp q18, q19, [x7, #160] // Store sixth pair (non-temporal, 32B)
    stnp q20, q21, [x7, #192] // Store seventh pair (non-temporal, 32B)
    stnp q22, q23, [x7, #224] // Store eighth pair (non-temporal, 32B)
    add x7, x7, #256          // Advance destination pointer by 256B
    sub x5, x5, #256          // Decrement remaining count by 256B

write_cleanup_128:            // 128B chunk
    cmp x5, #128              // Check if >= 128 bytes remain
    b.lt write_cleanup_64      // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    stnp q4, q5, [x7, #64]    // Store third pair (non-temporal, 32B)
    stnp q6, q7, [x7, #96]    // Store fourth pair (non-temporal, 32B)
    add x7, x7, #128          // Advance destination pointer by 128B
    sub x5, x5, #128          // Decrement remaining count by 128B

write_cleanup_64:             // 64B chunk
    cmp x5, #64               // Check if >= 64 bytes remain
    b.lt write_cleanup_32      // If less, handle smaller chunks
    stnp q0, q1, [x7, #0]     // Store first pair (non-temporal, 32B)
    stnp q2, q3, [x7, #32]    // Store second pair (non-temporal, 32B)
    add x7, x7, #64           // Advance destination pointer by 64B
    sub x5, x5, #64           // Decrement remaining count by 64B

write_cleanup_32:             // 32B chunk
    cmp x5, #32               // Check if >= 32 bytes remain
    b.lt write_cleanup_byte    // If less, handle byte tail
    stnp q0, q1, [x7, #0]     // Store pair (non-temporal, 32B)
    add x7, x7, #32           // Advance destination pointer by 32B
    sub x5, x5, #32           // Decrement remaining count by 32B

write_cleanup_byte:            // Byte tail (<32B)
    cmp x5, #0                // Check if any bytes remain
    b.le write_loop_end        // If none, exit
    strb wzr, [x7], #1         // Store zero byte, post-increment destination
    subs x5, x5, #1           // Decrement remaining count
    b.gt write_cleanup_byte    // Loop if more bytes remain

write_loop_end:             // Return to caller
    ret                     // Return