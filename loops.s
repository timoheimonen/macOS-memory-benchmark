// loops.s
// Assembly implementations for memory operations (ARM64 macOS / Apple Silicon)

// --- Memory Copy Function (Bandwidth Test) ---

.global _memory_copy_loop_asm // Make visible to linker
.align 4                      // Align function entry to 16-byte boundary

// C++ Signature: void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
// AAPCS64 Arguments:
// x0: dst (destination address)
// x1: src (source address)
// x2: byteCount (number of bytes)

_memory_copy_loop_asm:
    mov x3, xzr     // x3 = current offset, start at 0
    mov x4, #16     // x4 = step size (16 bytes via NEON q0)
                    // NOTE: Larger steps (e.g., 64/128 bytes using q0-q3/q0-q7)
                    // and loop unrolling are better for max bandwidth.

copy_loop_start:
    cmp x3, x2      // Compare current offset (x3) with total byteCount (x2)
    b.ge copy_loop_end // If offset >= byteCount, loop is done

    // Core copy operation (16 bytes)
    ldr q0, [x1, x3] // Load 16B from src + offset into NEON register q0
    str q0, [x0, x3] // Store 16B from q0 to dst + offset

    // Advance to next block
    add x3, x3, x4  // offset += step_size
    b copy_loop_start // Branch back to loop start

copy_loop_end:
    ret             // Return to caller

// --- Memory Latency Function (Pointer Chasing) ---

.global _memory_latency_chase_asm // Make visible to linker
.align 4                          // Align function entry

// C++ Signature: void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
// AAPCS64 Arguments:
// x0: Address of the first pointer in the chain (contains the address of the next element)
// x1: Number of accesses/loads to perform (chain length to traverse)

_memory_latency_chase_asm:
    // x0 initially holds the address of the *first pointer*.
    // The loop will load the value at that address (which is the *next pointer's address*)
    // back into x0, creating a dependent load chain.

latency_loop:
    // Core latency measurement: Load the next address from the address in x0.
    // The result (next address) is placed back in x0, making the next load dependent on this one.
    ldr x0, [x0]      // x0 = [x0] (Load the next pointer address)

    // Decrement loop counter and set flags.
    subs x1, x1, #1   // count--

    // Continue loop if counter is not zero.
    b.ne latency_loop // If (count != 0) branch to latency_loop

    // Loop finished
    ret               // Return to caller (x0 contains the last loaded address)