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
// llm_decode_memory_paged_asm
// -----------------------------------------------------------------------------
// C++ prototype:
//   extern "C" void llm_decode_memory_paged_asm(
//       const LlmPagedLayerDescriptor* layers,
//       const LlmPagedKvAssignmentDescriptor* assignments,
//       uint64_t layer_count,
//       uint64_t work_unit_count,
//       uint64_t scenario_flags,
//       uint64_t scenario_seed,
//       LlmWorkerChecksum* output) noexcept;
//
// Arguments:
//   x0 = paged layer descriptor array
//   x1 = paged KV assignment descriptor array
//   x2 = layer descriptor count
//   x3 = task-local decode work-unit count
//   x4 = scenario flags: bit 0 = weights, bit 1 = KV (valid values 1..3)
//   x5 = scenario-domain seed used as the append-affine base
//   x6 = 16-byte-aligned worker checksum output
//
// Frozen descriptor ABI (llm-memory-paged-descriptor-abi-v1):
//   LlmPagedLayerDescriptor, stride 48:
//      0 weight_ptr                 8 weight_bytes
//     16 first_assignment_index    24 assignment_count
//     32 layer_index               40 reserved_zero
//   LlmPagedKvAssignmentDescriptor, stride 96:
//      0 block_table_row            8 k_layer_pool
//     16 v_layer_pool              24 first_logical_block
//     32 owned_block_count         40 blocks_per_sequence
//     48 block_bytes               56 last_block_valid_bytes
//     64 decode_append_offset      72 append_record_bytes
//     80 layer_index               88 batch_sequence_index
//
// Frozen output ABI (three consecutive 32-byte components):
//   weight offsets 0/8/16/24, K offsets 32/40/48/56,
//   V offsets 64/72/80/88; fields are state_a, state_b,
//   exact_bytes_read, and span_count.
//
// Traversal and timed lookup semantics:
//   * Each work unit visits layers in descriptor order. Mixed work absorbs the
//     layer's weight span before any assignment belonging to that layer.
//   * A zero-block assignment is skipped. The assignment owning logical block
//     N-1 performs one explicit 32-bit table load for the paired K/V append.
//     The loaded physical ID is mixed into the K checksum once and only then
//     used to derive the physical K and V append addresses.
//   * K scans walk every owned logical block in increasing order, issuing one
//     explicit `ldr w` per block. V scans restart from the first owned block
//     and issue a separate `ldr w` per block. Pool addresses are never
//     calculated before their corresponding timed table load.
//   * Every non-terminal scan consumes block_bytes. Logical block N-1 consumes
//     last_block_valid_bytes, so suffix padding is neither read nor written.
//   * Every lookup mixes the flattened row-major logical index, loaded
//     physical ID, semantic visit kind, and task-local work-unit ordinal into
//     the selected checksum component before the existing span fold.
//   * The read loop consumes exact 32-byte vector bodies and exact 16/8/1-7
//     byte tails without crossing the supplied block visit. Word parity starts
//     at even for each block visit, matching the independent scalar oracle.
//   * Append stores are ordinary temporal stores. The complete K or V record
//     starts at canonical record offset zero even when its physical pointer is
//     unaligned. No read-modify-write or padding access occurs.
//   * weights_only does not dereference the assignment array, block table, or
//     K/V pools. Worker-owned ranges are disjoint, so no barrier is emitted.
//     Timing fences, task barriers, stop handling, and validation stay outside
//     this leaf hot path.
//
// AAPCS64 and register use:
//   * x19-x30 are saved and restored. q8-q15 and platform register x18 are
//     untouched. Caller-saved q0-q7 hold vector data and append values.
//   * x19-x22 = weight checksum, x23-x26 = K checksum,
//     x27-x30 = V checksum, in output-field order.
//   * x8-x17 hold descriptor/traversal state. Span and append macros clobber
//     only x0-x7 and q0-q7. The lookup mixer preserves x7 (physical ID), x8
//     (assignment descriptor), and x17 (logical block).
//   * The 160-byte frame is 16-byte aligned. Slots 96/104/112 retain the
//     top-level bases/output, slot 120 the zero-based work-unit ordinal, slot
//     128 the current assignment's exclusive logical-block end, and slots
//     136/144 the paired physical append pointers.
//   * This is a leaf function. Incoming x30 is saved before x30 becomes the V
//     checksum span counter.
//
// Safe direct-boundary behavior:
//   A null output returns immediately. With a valid output, all checksum
//   components are initialized before top-level validation. Null descriptor
//   bases, invalid scenario flags, zero layers, or zero work units return the
//   initialized result. Materialized inner descriptors are trusted executor
//   input and have already passed the C++ ABI/plan validation boundary.
// -----------------------------------------------------------------------------

// Load one exact 64-bit literal without a literal-pool memory access.
.macro LLM_PAGED_LOAD_U64 reg, half0, half1, half2, half3
    movz \reg, #\half0
    movk \reg, #\half1, lsl #16
    movk \reg, #\half2, lsl #32
    movk \reg, #\half3, lsl #48
.endm

// Absorb the non-empty span in x0/x1 into one persistent checksum component.
// x14 permanently holds 0x9E3779B97F4A7C15, the span ordinal multiplier.
.macro LLM_PAGED_ABSORB_SPAN state_a, state_b, exact_bytes, span_count
    cbz x1, Lpaged_absorb_done\@

    mov x6, x1                  // Preserve exact span bytes for the state fold.
    mov x2, xzr                 // span_even
    mov x3, xzr                 // span_odd

Lpaged_absorb_vector_loop\@:
    cmp x1, #32
    b.lo Lpaged_absorb_16\@
    ldp q0, q1, [x0], #32
    // q0=[word0,word1], q1=[word2,word3]. Unzip by checksum parity.
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x4, d4
    fmov x5, d5
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #32
    b Lpaged_absorb_vector_loop\@

Lpaged_absorb_16\@:
    tbz x1, #4, Lpaged_absorb_8\@
    ldp x4, x5, [x0], #16
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #16

Lpaged_absorb_8\@:
    tbz x1, #3, Lpaged_absorb_even_byte_tail\@
    ldr x4, [x0], #8
    add x2, x2, x4
    sub x1, x1, #8
    cbz x1, Lpaged_absorb_fold\@

    // An 8-byte remainder preceded this partial word, so it is odd.
    mov x4, xzr
    mov x5, xzr
Lpaged_absorb_odd_byte_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lpaged_absorb_odd_byte_loop\@
    add x3, x3, x4
    b Lpaged_absorb_fold\@

Lpaged_absorb_even_byte_tail\@:
    cbz x1, Lpaged_absorb_fold\@
    // No 8-byte remainder preceded this partial word, so it is even.
    mov x4, xzr
    mov x5, xzr
Lpaged_absorb_even_byte_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lpaged_absorb_even_byte_loop\@
    add x2, x2, x4

Lpaged_absorb_fold\@:
    // span_count is the old zero-based ordinal; x7 is ordinal + 1.
    add x7, \span_count, #1
    mul x0, x7, x14
    add \state_a, \state_a, x2
    add \state_a, \state_a, x0
    ror \state_a, \state_a, #47  // rotl64(..., 17)

    LLM_PAGED_LOAD_U64 x0, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    mul x0, x7, x0
    add \state_b, \state_b, x3
    add \state_b, \state_b, x6
    add \state_b, \state_b, x0
    ror \state_b, \state_b, #35  // rotl64(..., 29)

    add \exact_bytes, \exact_bytes, x6
    mov \span_count, x7

Lpaged_absorb_done\@:
.endm

// Mix one semantic table visit before deriving a physical data address.
// Inputs retained across the macro:
//   x7  = zero-extended physical block ID loaded by the caller's `ldr w`
//   x8  = current assignment descriptor
//   x17 = logical block index within the sequence
// The visit constant supplied to the macro is exactly
// (semantic_visit_kind + 1) * 0x94D049BB133111EB modulo 2^64.
.macro LLM_PAGED_MIX_LOOKUP state_a, state_b, visit0, visit1, visit2, visit3
    ldr x0, [x8, #88]           // batch_sequence_index
    ldr x1, [x8, #40]           // blocks_per_sequence
    madd x0, x0, x1, x17        // global row-major logical table index
    add x0, x0, #1
    add x1, x7, #1
    mul x0, x0, x1              // (global_logical+1)*(physical_id+1)

    LLM_PAGED_LOAD_U64 x1, \visit0, \visit1, \visit2, \visit3
    add x0, x0, x1
    ldr x2, [sp, #120]          // zero-based task-local work-unit ordinal
    add x2, x2, #1
    LLM_PAGED_LOAD_U64 x3, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x0, x2, x3, x0         // Complete exact lookup term.

    add \state_a, \state_a, x0
    add \state_a, \state_a, x14
    ror \state_a, \state_a, #51  // rotl64(..., 13)

    LLM_PAGED_LOAD_U64 x1, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    add x1, x0, x1
    eor \state_b, \state_b, x1
    ror \state_b, \state_b, #33  // rotl64(..., 31)
.endm

// Write one complete K or V append record. The caller stores the physical
// destination pointer in a frame slot and leaves the common affine base in x7.
// Canonical record byte offset is exactly zero, so the first stored word is
// llm_append_word(..., record_word_index=0, domain). The pointer itself may be
// unaligned; vector/scalar stores remain exact and never cross record_bytes.
.macro LLM_PAGED_WRITE_APPEND pointer_slot, domain0, domain1, domain2, domain3
    ldr x1, [x8, #72]           // append_record_bytes
    cbz x1, Lpaged_append_done\@
    ldr x0, [sp, #\pointer_slot]

    LLM_PAGED_LOAD_U64 x4, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    add x5, x7, x4              // record_word_index 0 contributes 1*x4.
    LLM_PAGED_LOAD_U64 x3, \domain0, \domain1, \domain2, \domain3
    add x5, x5, x3

Lpaged_append_vector_loop\@:
    cmp x1, #32
    b.lo Lpaged_append_16\@
    add x3, x5, x4
    add x6, x3, x4
    add x2, x6, x4
    ins v6.d[0], x5
    ins v6.d[1], x3
    ins v7.d[0], x6
    ins v7.d[1], x2
    stp q6, q7, [x0], #32
    add x5, x2, x4
    sub x1, x1, #32
    b Lpaged_append_vector_loop\@

Lpaged_append_16\@:
    tbz x1, #4, Lpaged_append_8\@
    add x3, x5, x4
    ins v6.d[0], x5
    ins v6.d[1], x3
    str q6, [x0], #16
    add x5, x3, x4
    sub x1, x1, #16

Lpaged_append_8\@:
    tbz x1, #3, Lpaged_append_tail\@
    str x5, [x0], #8
    add x5, x5, x4
    sub x1, x1, #8

Lpaged_append_tail\@:
    // The remaining 0-7 bytes are the exact low prefix of the next word.
    tbz x1, #2, Lpaged_append_tail_2\@
    str w5, [x0], #4
    lsr x5, x5, #32
Lpaged_append_tail_2\@:
    tbz x1, #1, Lpaged_append_tail_1\@
    strh w5, [x0], #2
    lsr x5, x5, #16
Lpaged_append_tail_1\@:
    tbz x1, #0, Lpaged_append_done\@
    strb w5, [x0]

Lpaged_append_done\@:
.endm

.text
.p2align 4
.global _llm_decode_memory_paged_asm
_llm_decode_memory_paged_asm:
    // Without an output location there is no safe result to initialize.
    cbz x6, Lpaged_return_without_frame

    // 96 bytes preserve x19-x30. The remaining 64 bytes retain top-level and
    // nested-loop values that must survive the caller-saved checksum macros.
    stp x19, x20, [sp, #-160]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    str x0, [sp, #96]           // Layer descriptor base.
    str x1, [sp, #104]          // Assignment descriptor base.
    str x6, [sp, #112]          // Checksum output.
    str xzr, [sp, #120]         // Zero-based work-unit ordinal.

    // llm-read-checksum-v1 component initial states. The paged lookup mixer
    // augments the K/V state transitions without changing output layout.
    LLM_PAGED_LOAD_U64 x19, 0x57E2, 0xCDF7, 0x23CF, 0x737A
    LLM_PAGED_LOAD_U64 x20, 0xD275, 0x4BC4, 0xD375, 0x6A5E
    mov x21, xzr
    mov x22, xzr

    LLM_PAGED_LOAD_U64 x23, 0x57E2, 0xC4E7, 0x38CD, 0x6F60
    LLM_PAGED_LOAD_U64 x24, 0xD275, 0x44B4, 0xDC73, 0x5E78
    mov x25, xzr
    mov x26, xzr

    LLM_PAGED_LOAD_U64 x27, 0x57E2, 0xC4E7, 0x38CD, 0x7260
    LLM_PAGED_LOAD_U64 x28, 0xD275, 0x44B4, 0xDC73, 0x6978
    mov x29, xzr
    mov x30, xzr

    // Preserve top-level traversal state in registers untouched by the span
    // and append macros, then reject only safe direct-boundary cases.
    mov x9, x0                   // Layer base.
    mov x10, x1                  // Assignment base until first KV layer.
    mov x12, x3                  // Work units remaining.
    mov x13, x4                  // Scenario flags.
    mov x15, x5                  // Becomes current work-unit affine base.
    cbz x9, Lpaged_store_output
    cbz x2, Lpaged_store_output
    cbz x12, Lpaged_store_output
    cbz x13, Lpaged_store_output
    cmp x13, #3
    b.hi Lpaged_store_output
    tbz x13, #1, Lpaged_top_level_valid
    cbz x10, Lpaged_store_output

Lpaged_top_level_valid:
    // Layer end = layer base + layer_count * 48. The pure/materialization
    // boundaries have already checked all inner size and count arithmetic.
    lsl x7, x2, #4
    add x7, x7, x2, lsl #5
    add x11, x9, x7

    // task_affine(work unit 0) = scenario_seed + step_multiplier * (0 + 1).
    LLM_PAGED_LOAD_U64 x14, 0x7C15, 0x7F4A, 0x79B9, 0x9E37
    add x15, x15, x14

Lpaged_work_unit_loop:
    mov x16, x9                  // Current layer descriptor.

Lpaged_layer_loop:
    cmp x16, x11
    b.eq Lpaged_work_unit_complete

    // Mixed and weights_only absorb the layer weight shard before any KV.
    tbz x13, #0, Lpaged_layer_kv
    ldp x0, x1, [x16, #0]
    LLM_PAGED_ABSORB_SPAN x19, x20, x21, x22

Lpaged_layer_kv:
    // This branch occurs before any assignment/table/pool dereference, making
    // the weights_only scenario independent of every paged KV resource.
    tbz x13, #1, Lpaged_next_layer
    ldp x0, x1, [x16, #16]      // First assignment index and count.
    cbz x1, Lpaged_next_layer

    // Assignment pointer = base + first_index * 96; end uses count * 96.
    ldr x3, [sp, #104]
    lsl x2, x0, #5
    add x2, x2, x0, lsl #6
    add x8, x3, x2
    lsl x2, x1, #5
    add x2, x2, x1, lsl #6
    add x10, x8, x2

Lpaged_assignment_loop:
    ldr x0, [x8, #32]           // owned_block_count
    cbz x0, Lpaged_next_assignment

    // Only the assignment whose range contains N-1 performs the paired K/V
    // append. Valid plans use contiguous, non-overlapping logical ranges.
    ldr x1, [x8, #40]           // blocks_per_sequence
    sub x17, x1, #1             // terminal logical block
    ldr x2, [x8, #24]           // first_logical_block
    cmp x17, x2
    b.lo Lpaged_k_scan_setup
    add x3, x2, x0              // exclusive owned logical-block end
    cmp x17, x3
    b.hs Lpaged_k_scan_setup

    // One and only one explicit table load serves the paired append. The
    // physical ID remains in x7 across the checksum mix.
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // Append visit kind = 0, so the visit term is one multiplier.
    LLM_PAGED_MIX_LOOKUP x23, x24, 0x11EB, 0x1331, 0x49BB, 0x94D0

    // Physical append addresses are derived only after the timed ID load.
    ldr x0, [x8, #48]           // block_bytes
    mul x1, x7, x0
    ldr x2, [x8, #64]           // decode_append_offset
    add x1, x1, x2
    ldr x0, [x8, #8]            // k_layer_pool
    add x0, x0, x1
    str x0, [sp, #136]
    ldr x0, [x8, #16]           // v_layer_pool
    add x0, x0, x1
    str x0, [sp, #144]

    // Common append affine base excludes K/V domain and record word index.
    ldr x0, [x8, #80]           // layer_index
    add x0, x0, #1
    LLM_PAGED_LOAD_U64 x2, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x7, x0, x2, x15
    ldr x1, [x8, #88]           // batch_sequence_index
    add x1, x1, #1
    LLM_PAGED_LOAD_U64 x2, 0x11EB, 0x1331, 0x49BB, 0x94D0
    madd x7, x1, x2, x7

    // K and V writes share the single lookup but retain separate affine
    // domains. Both records start at canonical byte offset zero.
    LLM_PAGED_WRITE_APPEND 136, 0x4B4B, 0x4B4B, 0x4B4B, 0x4B4B
    LLM_PAGED_WRITE_APPEND 144, 0x5656, 0x5656, 0x5656, 0x5656

Lpaged_k_scan_setup:
    ldr x17, [x8, #24]          // first_logical_block
    ldr x0, [x8, #32]
    add x0, x17, x0
    str x0, [sp, #128]          // exclusive logical-block end

Lpaged_k_scan_loop:
    // Each K visit performs its own explicit 32-bit timed lookup.
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // KScan visit kind = 1, hence two visit multipliers modulo 2^64.
    LLM_PAGED_MIX_LOOKUP x23, x24, 0x23D6, 0x2662, 0x9376, 0x29A0

    // Derive the K physical block address only after the lookup and mix.
    ldr x2, [x8, #48]           // block_bytes
    mul x3, x7, x2
    ldr x0, [x8, #8]            // k_layer_pool
    add x0, x0, x3
    mov x1, x2                  // default full-block visit length
    ldr x2, [x8, #40]
    sub x2, x2, #1
    cmp x17, x2
    b.ne Lpaged_k_length_ready
    ldr x1, [x8, #56]           // terminal block exact valid bytes
Lpaged_k_length_ready:
    LLM_PAGED_ABSORB_SPAN x23, x24, x25, x26

    add x17, x17, #1
    ldr x0, [sp, #128]
    cmp x17, x0
    b.lo Lpaged_k_scan_loop

    // V restarts the same logical range and must reload every table entry.
    ldr x17, [x8, #24]

Lpaged_v_scan_loop:
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // VScan visit kind = 2, hence three visit multipliers modulo 2^64.
    LLM_PAGED_MIX_LOOKUP x27, x28, 0x35C1, 0x3993, 0xDD31, 0xBE70

    // Derive the V physical block address only after the lookup and mix.
    ldr x2, [x8, #48]
    mul x3, x7, x2
    ldr x0, [x8, #16]           // v_layer_pool
    add x0, x0, x3
    mov x1, x2
    ldr x2, [x8, #40]
    sub x2, x2, #1
    cmp x17, x2
    b.ne Lpaged_v_length_ready
    ldr x1, [x8, #56]
Lpaged_v_length_ready:
    LLM_PAGED_ABSORB_SPAN x27, x28, x29, x30

    add x17, x17, #1
    ldr x0, [sp, #128]
    cmp x17, x0
    b.lo Lpaged_v_scan_loop

Lpaged_next_assignment:
    add x8, x8, #96
    cmp x8, x10
    b.lo Lpaged_assignment_loop

Lpaged_next_layer:
    add x16, x16, #48
    b Lpaged_layer_loop

Lpaged_work_unit_complete:
    // Increment affine base and separately retain the lookup work ordinal.
    add x15, x15, x14
    ldr x0, [sp, #120]
    add x0, x0, #1
    str x0, [sp, #120]
    subs x12, x12, #1
    b.ne Lpaged_work_unit_loop

Lpaged_store_output:
    ldr x6, [sp, #112]
    stp x19, x20, [x6, #0]
    stp x21, x22, [x6, #16]
    stp x23, x24, [x6, #32]
    stp x25, x26, [x6, #48]
    stp x27, x28, [x6, #64]
    stp x29, x30, [x6, #80]

    ldp x29, x30, [sp, #80]
    ldp x27, x28, [sp, #64]
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #160

Lpaged_return_without_frame:
    ret
