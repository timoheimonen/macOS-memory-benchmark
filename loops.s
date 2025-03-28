// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)

// --- Memory Copy Function (Bandwidth Test - Optimized with Non-Temporal Stores) ---

.global _memory_copy_loop_asm // Make visible to linker
.align 4                      // Align function entry to 16-byte boundary

// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// AAPCS64 Arguments:
// x0: dst (destination base address)
// x1: src (source base address)
// x2: byteCount (number of bytes)
// Registers used: x3=offset, x4=step_size, x5=temp/remaining,
//                 x6=src_addr_current_iter, x7=dst_addr_current_iter,
//                 w5=byte_temp, q0-q3=data

_memory_copy_loop_asm:
    mov x3, xzr     // x3 = current offset, start at 0
    mov x4, #64     // x4 = step size (64 bytes)

copy_loop_start_nt64:
    // Calculate remaining bytes: byteCount - offset
    subs x5, x2, x3
    // Compare remaining bytes with step size (64)
    cmp x5, x4
    b.lt copy_loop_cleanup // If remaining < 64, go to cleanup

    // Calculate absolute source and destination addresses for this iteration
    add x6, x1, x3  // x6 = src_base + offset (current src address)
    add x7, x0, x3  // x7 = dst_base + offset (current dst address)

    // Core non-temporal copy operation (64 bytes)
    // Use non-temporal pairs to minimize cache pollution for large copies.
    // Addressing mode: [calculated_base_address, #immediate_offset]
    ldnp q0, q1, [x6, #0]       // Load pair non-temporal from [src_addr + 0]
    ldnp q2, q3, [x6, #32]      // Load next pair non-temporal from [src_addr + 32]
    stnp q0, q1, [x7, #0]       // Store pair non-temporal to [dst_addr + 0]
    stnp q2, q3, [x7, #32]      // Store next pair non-temporal to [dst_addr + 32]

    // Advance offset for the next 64-byte block
    add x3, x3, x4  // offset += 64
    b copy_loop_start_nt64 // Branch back to main loop start

copy_loop_cleanup:
    // Cleanup loop for remaining bytes (less than 64).
    // This simple byte-by-byte loop uses [base, index_reg] addressing, which is valid for ldrb/strb.
    cmp x3, x2      // Check if offset == byteCount already (nothing left to copy)
    b.ge copy_loop_end // If offset >= byteCount, we are done

cleanup_loop_byte:
    ldrb w5, [x1, x3] // Load byte from src_base + offset
    strb w5, [x0, x3] // Store byte to dst_base + offset
    add x3, x3, #1    // offset++
    cmp x3, x2
    b.lt cleanup_loop_byte // Loop until offset == byteCount

copy_loop_end:
    ret             // Return to caller

// --- Memory Latency Function (Pointer Chasing) ---
// (Tämä osa pysyy täysin ennallaan)

.global _memory_latency_chase_asm
.align 4

// C++ Signature: void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// Args: x0=start_pointer_addr, x1=count
_memory_latency_chase_asm:
latency_loop:
    ldr x0, [x0]      // x0 = [x0] (Load next pointer address)
    subs x1, x1, #1   // count--
    b.ne latency_loop // If (count != 0) branch
ret