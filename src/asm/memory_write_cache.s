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
// memory_write_cache_loop_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void memory_write_cache_loop_asm(void* dst, size_t byteCount);
// Purpose:
//   Write 'byteCount' bytes of zeros to 'dst' for cache-bandwidth paths (L1/L2/custom).
//   This kernel is intentionally independent from main-memory write kernel so cache
//   tuning can evolve without coupling to DRAM path behavior.
// Arguments:
//   x0 = dst (void*)
//   x1 = byteCount (size_t)
// Returns:
//   (none)
// Clobbers:
//   x3-x7, q0-q7, q16-q31 (caller-saved only)
// Implementation Notes:
//   * Uses one zero vector (q0) and repeats it for all wide stores.
//   * Uses pointer+remaining loop state (x7/x5) for lower loop-control overhead.
//   * Processes 512B per iteration using 16 store pairs.
//   * Tail uses size-bit tests plus 16/8/4/2/1 scalar zero stores.
// -----------------------------------------------------------------------------

.global _memory_write_cache_loop_asm
.align 4
_memory_write_cache_loop_asm:
    // x7 = current dst pointer, x5 = remaining bytes.
    mov x7, x0
    mov x5, x1
    // Materialize zero vector once and reuse for all SIMD stores.
    movi v0.16b, #0

write_cache_loop_start_nt512: // Main 512B loop
    // If <512B remain, switch to tail handling.
    cmp x5, #512
    b.lo write_cache_loop_cleanup

    // 512B of zero stores (16x 32B store pairs).
    stp q0, q0, [x7, #0]
    stp q0, q0, [x7, #32]
    stp q0, q0, [x7, #64]
    stp q0, q0, [x7, #96]
    stp q0, q0, [x7, #128]
    stp q0, q0, [x7, #160]
    stp q0, q0, [x7, #192]
    stp q0, q0, [x7, #224]
    stp q0, q0, [x7, #256]
    stp q0, q0, [x7, #288]
    stp q0, q0, [x7, #320]
    stp q0, q0, [x7, #352]
    stp q0, q0, [x7, #384]
    stp q0, q0, [x7, #416]
    stp q0, q0, [x7, #448]
    stp q0, q0, [x7, #480]

    // Advance destination and remaining-byte count.
    add x7, x7, #512
    sub x5, x5, #512
    b write_cache_loop_start_nt512

write_cache_loop_cleanup:     // Tail handling when <512B remain
    cbz x5, write_cache_loop_end

    // Tiered tail: test size bits for 256/128/64/32 before scalar ladder.
    tbz x5, #8, write_cache_cleanup_128
    // 256B chunk
    stp q0, q0, [x7, #0]
    stp q0, q0, [x7, #32]
    stp q0, q0, [x7, #64]
    stp q0, q0, [x7, #96]
    stp q0, q0, [x7, #128]
    stp q0, q0, [x7, #160]
    stp q0, q0, [x7, #192]
    stp q0, q0, [x7, #224]
    add x7, x7, #256
    sub x5, x5, #256

write_cache_cleanup_128:      // Optional 128B chunk
    tbz x5, #7, write_cache_cleanup_64
    stp q0, q0, [x7, #0]
    stp q0, q0, [x7, #32]
    stp q0, q0, [x7, #64]
    stp q0, q0, [x7, #96]
    add x7, x7, #128
    sub x5, x5, #128

write_cache_cleanup_64:       // Optional 64B chunk
    tbz x5, #6, write_cache_cleanup_32
    stp q0, q0, [x7, #0]
    stp q0, q0, [x7, #32]
    add x7, x7, #64
    sub x5, x5, #64

write_cache_cleanup_32:       // Optional 32B chunk
    tbz x5, #5, write_cache_cleanup_16
    stp q0, q0, [x7, #0]
    add x7, x7, #32
    sub x5, x5, #32

write_cache_cleanup_16:       // Optional 16B scalar zero store
    tbz x5, #4, write_cache_cleanup_8
    str q0, [x7], #16
    sub x5, x5, #16

write_cache_cleanup_8:        // Optional 8B scalar zero store
    tbz x5, #3, write_cache_cleanup_4
    str xzr, [x7], #8
    sub x5, x5, #8

write_cache_cleanup_4:        // Optional 4B scalar zero store
    tbz x5, #2, write_cache_cleanup_2
    str wzr, [x7], #4
    sub x5, x5, #4

write_cache_cleanup_2:        // Optional 2B scalar zero store
    tbz x5, #1, write_cache_cleanup_1
    strh wzr, [x7], #2
    sub x5, x5, #2

write_cache_cleanup_1:        // Optional final 1B zero store
    tbz x5, #0, write_cache_loop_end
    strb wzr, [x7]

write_cache_loop_end:         // Return to caller
    ret
