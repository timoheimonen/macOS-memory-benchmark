// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)

.global _memory_copy_loop_asm // Make visible to linker
.align 4                      // Align function entry (16 bytes)

// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// AAPCS64 Args: x0=dst_base, x1=src_base, x2=byteCount
// Regs used:    x3=offset, x4=step, x5=remaining, x6=src_addr, x7=dst_addr, w5=byte_tmp, q0-q7=data

_memory_copy_loop_asm:
    mov x3, xzr     // x3 = current offset = 0
    mov x4, #128    // x4 = step size (128 bytes)

copy_loop_start_nt128:
    // Check if remaining bytes >= step size
    subs x5, x2, x3 // x5 = byteCount - offset
    cmp x5, x4      // Compare remaining with step (128)
    b.lt copy_loop_cleanup // If remaining < 128, handle remainder

    // Calculate absolute addresses for this block
    add x6, x1, x3  // x6 = src_addr = src_base + offset
    add x7, x0, x3  // x7 = dst_addr = dst_base + offset

    // Core non-temporal copy (128 bytes) using LDP/STP Non-temporal Pair
    // Minimizes cache pollution for better large copy throughput.
    ldnp q0, q1, [x6, #0]    // Load 32B NT from [src_addr + 0]
    ldnp q2, q3, [x6, #32]   // Load 32B NT from [src_addr + 32]
    ldnp q4, q5, [x6, #64]   // Load 32B NT from [src_addr + 64]
    ldnp q6, q7, [x6, #96]   // Load 32B NT from [src_addr + 96]

    stnp q0, q1, [x7, #0]    // Store 32B NT to [dst_addr + 0]
    stnp q2, q3, [x7, #32]   // Store 32B NT to [dst_addr + 32]
    stnp q4, q5, [x7, #64]   // Store 32B NT to [dst_addr + 64]
    stnp q6, q7, [x7, #96]   // Store 32B NT to [dst_addr + 96]

    // Advance to next 128-byte block
    add x3, x3, x4  // offset += 128
    b copy_loop_start_nt128

copy_loop_cleanup:
    // Handle remaining bytes (< 128) - simple byte-by-byte copy
    cmp x3, x2      // Is offset == byteCount already?
    b.ge copy_loop_end // If yes, done.

cleanup_loop_byte:
    ldrb w5, [x1, x3] // Load byte src
    strb w5, [x0, x3] // Store byte dst
    add x3, x3, #1    // offset++
    cmp x3, x2
    b.lt cleanup_loop_byte // Loop if offset < byteCount

copy_loop_end:
    ret             // Return

// --- Memory Latency Function (Pointer Chasing) ---

.global _memory_latency_chase_asm
.align 4

// C++ Signature: void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Args: x0=ptr_addr, x1=count
_memory_latency_chase_asm:
    // x0 holds address containing the next pointer. Load value -> x0 creates dependency.
latency_loop:
    ldr x0, [x0]      // Load next pointer address (the core dependent load)
    subs x1, x1, #1   // count--
    b.ne latency_loop // Loop if count != 0
ret