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
// llm_prefill_memory_paged_asm
// -----------------------------------------------------------------------------
// C++ prototype:
//   extern "C" void llm_prefill_memory_paged_asm(
//       const LlmPagedPrefillLayerDescriptor* layers,
//       const LlmPagedPrefillKvAssignmentDescriptor* assignments,
//       uint64_t layer_count,
//       uint64_t operation_count,
//       uint64_t scenario_flags,
//       uint64_t scenario_seed,
//       LlmWorkerChecksum* output) noexcept;
//
// Arguments:
//   x0 = paged-prefill layer descriptor array
//   x1 = scenario-selected paged KV assignment descriptor array
//   x2 = layer descriptor count
//   x3 = task-local prefill operation count
//   x4 = scenario flags: bit 0 = weights, bit 1 = KV (valid values 1..3)
//   x5 = scenario-domain seed for the affine prompt pattern
//   x6 = 16-byte-aligned worker checksum output
//
// Frozen descriptor ABI (llm-memory-prefill-paged-descriptor-abi-v1):
//   LlmPagedPrefillLayerDescriptor, stride 48:
//      0 weight_ptr                 8 weight_bytes
//     16 first_assignment_index    24 assignment_count
//     32 layer_index               40 reserved_zero
//   LlmPagedPrefillKvAssignmentDescriptor, stride 112:
//      0 block_table_row            8 k_layer_pool
//     16 v_layer_pool              24 first_logical_block
//     32 owned_block_count         40 blocks_per_sequence
//     48 block_tokens              56 block_bytes
//     64 last_block_valid_bytes    72 prompt_tokens
//     80 attention_query_tile_tokens
//     88 record_bytes              96 layer_index
//    104 batch_sequence_index
//
// Frozen output ABI (three consecutive 32-byte components):
//   weight offsets 0/8/16/24, K offsets 32/40/48/56,
//   V offsets 64/72/80/88; fields are state_a, state_b,
//   exact_bytes_read, and span_count.
//
// Traversal, write, lookup, and checksum semantics:
//   * Each operation visits layers in descriptor order. A weight-enabled layer
//     absorbs its shard exactly once before the layer's paged assignments.
//   * For each non-empty assignment, every owned logical block is populated in
//     increasing order before any tile read. One explicit 32-bit table load
//     serves the paired K/V write, and physical addresses are derived only
//     after that load and its non-separable lookup-checksum mix.
//   * Tile ends advance by min(Q, P - current_end). Each tile scans all owned
//     blocks intersecting its causal prefix in increasing logical order: all K
//     visits complete before V restarts the same range. K and V issue separate
//     explicit `ldr w` table loads for every semantic visit.
//   * Full blocks use block_bytes. Prompt population and prefix scans use
//     last_block_valid_bytes or the exact partial prefix bytes at the terminal
//     visit, so suffix padding is never read or written.
//   * The affine K/V pattern uses canonical logical byte positions, independent
//     of physical block IDs. K/V read lanes retain canonical even/odd word sums,
//     while every lookup additionally applies mix_llm_paged_lookup over the
//     row-major logical table index, loaded physical ID, visit kind, and
//     task-local operation ordinal. Paired writes mix into K, as on paged decode.
//   * exact_bytes_read and span_count cover non-empty K/V block reads only.
//     Prompt writes and table loads do not add payload bytes or spans.
//   * Exact 32-byte vector bodies and exact leading/trailing scalar fragments
//     support unaligned block and record sizes without crossing a visit.
//   * weights_only does not dereference assignments, the block table, or K/V
//     pools. Worker-owned blocks are disjoint, so no barrier is emitted.
//
// AAPCS64 and register use:
//   * x19-x30 are saved and restored. q8-q15 and platform register x18 are
//     untouched. Caller-saved q0-q7 hold vector data and affine values.
//   * x19-x22 = weight checksum, x23-x26 = K checksum,
//     x27-x30 = V checksum, in output-field order.
//   * x8-x17 retain descriptor and traversal state. Macros clobber only x0-x7
//     and q0-q7, apart from their explicit checksum outputs.
//   * The 224-byte frame is 16-byte aligned. Slots 96/104/112 retain top-level
//     bases/output; 120 is the operation ordinal; 128 the assignment end; 136
//     the owned-block end; 144/152 physical K/V pointers; 160 the affine base;
//     168/176 canonical block byte and exact visit bytes; 184 the tile end;
//     192 the tile-end byte offset; and 200 the exclusive visit-block end.
//   * This is a leaf function. Incoming x30 is saved before x30 is reused as the
//     V span counter. There are no calls, prefetches, barriers, or allocations.
//
// Safe direct-boundary behavior:
//   A null output returns immediately. With a valid output, all components are
//   initialized before top-level validation. Null layers, invalid flags, zero
//   layers, or zero operations return the initialized result. weights_only
//   never dereferences the assignment pointer; KV-active work requires it.
//   Materialized inner descriptors are trusted executor input.
// -----------------------------------------------------------------------------

.macro LLM_PAGED_PREFILL_LOAD_U64 reg, half0, half1, half2, half3
    movz \reg, #\half0
    movk \reg, #\half1, lsl #16
    movk \reg, #\half2, lsl #32
    movk \reg, #\half3, lsl #48
.endm

// Absorb one static weight span with the frozen llm-read-checksum-v1 fold.
// x14 permanently holds 0x9E3779B97F4A7C15, the span ordinal multiplier.
.macro LLM_PAGED_PREFILL_ABSORB_WEIGHT state_a, state_b, exact_bytes, span_count
    cbz x1, Lpaged_prefill_weight_done\@

    mov x6, x1
    mov x2, xzr
    mov x3, xzr

Lpaged_prefill_weight_vector\@:
    cmp x1, #32
    b.lo Lpaged_prefill_weight_16\@
    ldp q0, q1, [x0], #32
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x4, d4
    fmov x5, d5
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #32
    b Lpaged_prefill_weight_vector\@

Lpaged_prefill_weight_16\@:
    tbz x1, #4, Lpaged_prefill_weight_8\@
    ldp x4, x5, [x0], #16
    add x2, x2, x4
    add x3, x3, x5
    sub x1, x1, #16

Lpaged_prefill_weight_8\@:
    tbz x1, #3, Lpaged_prefill_weight_even_tail\@
    ldr x4, [x0], #8
    add x2, x2, x4
    sub x1, x1, #8
    cbz x1, Lpaged_prefill_weight_fold\@
    mov x4, xzr
    mov x5, xzr
Lpaged_prefill_weight_odd_tail_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lpaged_prefill_weight_odd_tail_loop\@
    add x3, x3, x4
    b Lpaged_prefill_weight_fold\@

Lpaged_prefill_weight_even_tail\@:
    cbz x1, Lpaged_prefill_weight_fold\@
    mov x4, xzr
    mov x5, xzr
Lpaged_prefill_weight_even_tail_loop\@:
    ldrb w7, [x0], #1
    lslv x7, x7, x5
    orr x4, x4, x7
    add x5, x5, #8
    subs x1, x1, #1
    b.ne Lpaged_prefill_weight_even_tail_loop\@
    add x2, x2, x4

Lpaged_prefill_weight_fold\@:
    add x7, \span_count, #1
    mul x0, x7, x14
    add \state_a, \state_a, x2
    add \state_a, \state_a, x0
    ror \state_a, \state_a, #47

    LLM_PAGED_PREFILL_LOAD_U64 x0, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    mul x0, x7, x0
    add \state_b, \state_b, x3
    add \state_b, \state_b, x6
    add \state_b, \state_b, x0
    ror \state_b, \state_b, #35

    add \exact_bytes, \exact_bytes, x6
    mov \span_count, x7

Lpaged_prefill_weight_done\@:
.endm

// Mix one semantic block-table visit before deriving its physical address.
// x7 retains the loaded physical ID, x8 the assignment, and x17 the logical
// block. The visit literal is (visit_kind + 1) * 0x94D049BB133111EB.
.macro LLM_PAGED_PREFILL_MIX_LOOKUP state_a, state_b, visit0, visit1, visit2, visit3
    ldr x0, [x8, #104]          // batch_sequence_index
    ldr x1, [x8, #40]           // blocks_per_sequence
    madd x0, x0, x1, x17        // global row-major logical table index
    add x0, x0, #1
    add x1, x7, #1
    mul x0, x0, x1

    LLM_PAGED_PREFILL_LOAD_U64 x1, \visit0, \visit1, \visit2, \visit3
    add x0, x0, x1
    ldr x2, [sp, #120]          // zero-based operation ordinal
    add x2, x2, #1
    LLM_PAGED_PREFILL_LOAD_U64 x3, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x0, x2, x3, x0

    add \state_a, \state_a, x0
    add \state_a, \state_a, x14
    ror \state_a, \state_a, #51

    LLM_PAGED_PREFILL_LOAD_U64 x1, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    add x1, x0, x1
    eor \state_b, \state_b, x1
    ror \state_b, \state_b, #33
.endm

// Write one exact K or V block fragment. The physical destination pointer,
// canonical logical first byte, exact length, and common affine base live in
// frame slots selected below. Canonical word parity and byte positions are
// retained across an unaligned leading fragment.
.macro LLM_PAGED_PREFILL_WRITE_BLOCK pointer_slot, domain0, domain1, domain2, domain3
    ldr x1, [sp, #176]
    cbz x1, Lpaged_prefill_write_done\@
    ldr x0, [sp, #\pointer_slot]
    ldr x2, [sp, #168]
    ldr x7, [sp, #160]
    LLM_PAGED_PREFILL_LOAD_U64 x4, 0xFD93, 0x6659, 0xFEB8, 0xD6E8
    lsr x3, x2, #3
    add x3, x3, #1
    madd x3, x3, x4, x7
    LLM_PAGED_PREFILL_LOAD_U64 x5, \domain0, \domain1, \domain2, \domain3
    add x3, x3, x5

    and x2, x2, #7
    cbz x2, Lpaged_prefill_write_vector\@
    mov x5, #8
    sub x5, x5, x2
    cmp x1, x5
    csel x5, x1, x5, lo
    add x6, x3, x4
    lsl x2, x2, #3
    lsrv x3, x3, x2
    tbz x5, #2, Lpaged_prefill_write_lead_2\@
    str w3, [x0], #4
    lsr x3, x3, #32
Lpaged_prefill_write_lead_2\@:
    tbz x5, #1, Lpaged_prefill_write_lead_1\@
    strh w3, [x0], #2
    lsr x3, x3, #16
Lpaged_prefill_write_lead_1\@:
    tbz x5, #0, Lpaged_prefill_write_lead_complete\@
    strb w3, [x0], #1
Lpaged_prefill_write_lead_complete\@:
    sub x1, x1, x5
    cbz x1, Lpaged_prefill_write_done\@
    mov x3, x6

Lpaged_prefill_write_vector\@:
    cmp x1, #32
    b.lo Lpaged_prefill_write_16\@
    add x5, x3, x4
    add x6, x5, x4
    add x7, x6, x4
    ins v6.d[0], x3
    ins v6.d[1], x5
    ins v7.d[0], x6
    ins v7.d[1], x7
    stp q6, q7, [x0], #32
    add x3, x7, x4
    sub x1, x1, #32
    b Lpaged_prefill_write_vector\@

Lpaged_prefill_write_16\@:
    tbz x1, #4, Lpaged_prefill_write_8\@
    add x5, x3, x4
    ins v6.d[0], x3
    ins v6.d[1], x5
    str q6, [x0], #16
    add x3, x5, x4
    sub x1, x1, #16

Lpaged_prefill_write_8\@:
    tbz x1, #3, Lpaged_prefill_write_tail\@
    str x3, [x0], #8
    add x3, x3, x4
    sub x1, x1, #8

Lpaged_prefill_write_tail\@:
    tbz x1, #2, Lpaged_prefill_write_tail_2\@
    str w3, [x0], #4
    lsr x3, x3, #32
Lpaged_prefill_write_tail_2\@:
    tbz x1, #1, Lpaged_prefill_write_tail_1\@
    strh w3, [x0], #2
    lsr x3, x3, #16
Lpaged_prefill_write_tail_1\@:
    tbz x1, #0, Lpaged_prefill_write_done\@
    strb w3, [x0]

Lpaged_prefill_write_done\@:
.endm

// Scan one non-empty physical block visit while preserving canonical logical
// word parity. Slot 168 contains its canonical first byte and slot 176 its exact
// length. The pointer slot is populated only after the visit's timed lookup.
.macro LLM_PAGED_PREFILL_SCAN_BLOCK pointer_slot, state_even, state_odd, exact_bytes, span_count
    ldr x1, [sp, #176]
    cbz x1, Lpaged_prefill_scan_done\@
    add \exact_bytes, \exact_bytes, x1
    add \span_count, \span_count, #1
    ldr x0, [sp, #\pointer_slot]
    ldr x2, [sp, #168]
    lsr x4, x2, #3
    and x3, x2, #7
    cbz x3, Lpaged_prefill_scan_vector\@

    mov x7, #8
    sub x7, x7, x3
    cmp x1, x7
    csel x7, x1, x7, lo
    sub x1, x1, x7
Lpaged_prefill_scan_leading_loop\@:
    ldrb w5, [x0], #1
    lsl x6, x3, #3
    lslv x5, x5, x6
    tbnz x4, #0, Lpaged_prefill_scan_leading_odd\@
    add \state_even, \state_even, x5
    b Lpaged_prefill_scan_leading_added\@
Lpaged_prefill_scan_leading_odd\@:
    add \state_odd, \state_odd, x5
Lpaged_prefill_scan_leading_added\@:
    add x3, x3, #1
    subs x7, x7, #1
    b.ne Lpaged_prefill_scan_leading_loop\@
    cbz x1, Lpaged_prefill_scan_done\@
    add x4, x4, #1

Lpaged_prefill_scan_vector\@:
    cmp x1, #32
    b.lo Lpaged_prefill_scan_16\@
    ldp q0, q1, [x0], #32
    uzp1 v2.2d, v0.2d, v1.2d
    uzp2 v3.2d, v0.2d, v1.2d
    addp d4, v2.2d
    addp d5, v3.2d
    fmov x5, d4
    fmov x6, d5
    tbnz x4, #0, Lpaged_prefill_scan_vector_odd\@
    add \state_even, \state_even, x5
    add \state_odd, \state_odd, x6
    b Lpaged_prefill_scan_vector_added\@
Lpaged_prefill_scan_vector_odd\@:
    add \state_even, \state_even, x6
    add \state_odd, \state_odd, x5
Lpaged_prefill_scan_vector_added\@:
    add x4, x4, #4
    sub x1, x1, #32
    b Lpaged_prefill_scan_vector\@

Lpaged_prefill_scan_16\@:
    tbz x1, #4, Lpaged_prefill_scan_8\@
    ldp x5, x6, [x0], #16
    tbnz x4, #0, Lpaged_prefill_scan_16_odd\@
    add \state_even, \state_even, x5
    add \state_odd, \state_odd, x6
    b Lpaged_prefill_scan_16_added\@
Lpaged_prefill_scan_16_odd\@:
    add \state_even, \state_even, x6
    add \state_odd, \state_odd, x5
Lpaged_prefill_scan_16_added\@:
    add x4, x4, #2
    sub x1, x1, #16

Lpaged_prefill_scan_8\@:
    tbz x1, #3, Lpaged_prefill_scan_tail\@
    ldr x5, [x0], #8
    tbnz x4, #0, Lpaged_prefill_scan_8_odd\@
    add \state_even, \state_even, x5
    b Lpaged_prefill_scan_8_added\@
Lpaged_prefill_scan_8_odd\@:
    add \state_odd, \state_odd, x5
Lpaged_prefill_scan_8_added\@:
    add x4, x4, #1
    sub x1, x1, #8

Lpaged_prefill_scan_tail\@:
    cbz x1, Lpaged_prefill_scan_done\@
    mov x3, xzr
Lpaged_prefill_scan_tail_loop\@:
    ldrb w5, [x0], #1
    lsl x6, x3, #3
    lslv x5, x5, x6
    tbnz x4, #0, Lpaged_prefill_scan_tail_odd\@
    add \state_even, \state_even, x5
    b Lpaged_prefill_scan_tail_added\@
Lpaged_prefill_scan_tail_odd\@:
    add \state_odd, \state_odd, x5
Lpaged_prefill_scan_tail_added\@:
    add x3, x3, #1
    subs x1, x1, #1
    b.ne Lpaged_prefill_scan_tail_loop\@

Lpaged_prefill_scan_done\@:
.endm

.text
.p2align 4
.global _llm_prefill_memory_paged_asm
_llm_prefill_memory_paged_asm:
    cbz x6, Lpaged_prefill_return_without_frame

    stp x19, x20, [sp, #-224]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    str x0, [sp, #96]
    str x1, [sp, #104]
    str x6, [sp, #112]
    str xzr, [sp, #120]

    // Weight retains llm-read-checksum-v1. Paged prefill K/V start from the raw
    // affine parity-sum state and are then transformed by every timed lookup.
    LLM_PAGED_PREFILL_LOAD_U64 x19, 0x57E2, 0xCDF7, 0x23CF, 0x737A
    LLM_PAGED_PREFILL_LOAD_U64 x20, 0xD275, 0x4BC4, 0xD375, 0x6A5E
    mov x21, xzr
    mov x22, xzr
    mov x23, xzr
    mov x24, xzr
    mov x25, xzr
    mov x26, xzr
    mov x27, xzr
    mov x28, xzr
    mov x29, xzr
    mov x30, xzr

    mov x9, x0                   // Layer base.
    mov x10, x1                  // Assignment base until the first KV layer.
    mov x12, x3                  // Operations remaining.
    mov x13, x4                  // Scenario flags.
    mov x15, x5                  // Becomes the current operation affine base.
    cbz x9, Lpaged_prefill_store_output
    cbz x2, Lpaged_prefill_store_output
    cbz x12, Lpaged_prefill_store_output
    cbz x13, Lpaged_prefill_store_output
    cmp x13, #3
    b.hi Lpaged_prefill_store_output
    tbz x13, #1, Lpaged_prefill_top_level_valid
    cbz x10, Lpaged_prefill_store_output

Lpaged_prefill_top_level_valid:
    // Layer end = layer base + layer_count * 48.
    lsl x7, x2, #4
    add x7, x7, x2, lsl #5
    add x11, x9, x7

    // affine operation 0 = scenario seed + prefill domain + multiplier * 1.
    LLM_PAGED_PREFILL_LOAD_U64 x14, 0x7C15, 0x7F4A, 0x79B9, 0x9E37
    LLM_PAGED_PREFILL_LOAD_U64 x7, 0x4C31, 0x494C, 0x4546, 0x5052
    add x15, x15, x7
    add x15, x15, x14

Lpaged_prefill_operation_loop:
    mov x16, x9

Lpaged_prefill_layer_loop:
    cmp x16, x11
    b.eq Lpaged_prefill_operation_complete

    // One weight pass per operation/layer, before any paged KV work.
    tbz x13, #0, Lpaged_prefill_layer_kv
    ldp x0, x1, [x16, #0]
    LLM_PAGED_PREFILL_ABSORB_WEIGHT x19, x20, x21, x22

Lpaged_prefill_layer_kv:
    // weights_only never dereferences assignment, table, or pool memory.
    tbz x13, #1, Lpaged_prefill_next_layer
    ldp x0, x1, [x16, #16]
    cbz x1, Lpaged_prefill_next_layer

    // Assignment pointer/end = base + index/count * 112 (128 - 16).
    ldr x3, [sp, #104]
    lsl x2, x0, #7
    sub x2, x2, x0, lsl #4
    add x8, x3, x2
    lsl x2, x1, #7
    sub x2, x2, x1, lsl #4
    add x10, x8, x2
    str x10, [sp, #128]

Lpaged_prefill_assignment_loop:
    ldr x0, [x8, #32]           // owned_block_count
    cbz x0, Lpaged_prefill_next_assignment
    ldr x1, [x8, #24]           // first_logical_block
    add x2, x1, x0
    str x2, [sp, #136]          // exclusive owned logical-block end

    // Common affine base excludes K/V domain and canonical logical word.
    ldr x0, [x8, #96]           // layer_index
    add x0, x0, #1
    LLM_PAGED_PREFILL_LOAD_U64 x2, 0xE5B9, 0x1CE4, 0x476D, 0xBF58
    madd x7, x0, x2, x15
    ldr x1, [x8, #104]          // batch_sequence_index
    add x1, x1, #1
    LLM_PAGED_PREFILL_LOAD_U64 x2, 0x11EB, 0x1331, 0x49BB, 0x94D0
    madd x7, x1, x2, x7
    str x7, [sp, #160]

    // Source-audit anchor: populate every owned block before the first tile.
Lpaged_prefill_assignment_write_phase:
    ldr x17, [x8, #24]

Lpaged_prefill_assignment_block_write_loop:
    // One explicit timed table load serves the paired K/V block write.
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // PairedWrite visit kind = 0.
    LLM_PAGED_PREFILL_MIX_LOOKUP x23, x24, 0x11EB, 0x1331, 0x49BB, 0x94D0

    // Physical addresses are derived only after the timed load and mix.
    ldr x0, [x8, #56]           // block_bytes
    mul x1, x7, x0
    ldr x2, [x8, #8]
    add x2, x2, x1
    str x2, [sp, #144]
    ldr x2, [x8, #16]
    add x2, x2, x1
    str x2, [sp, #152]

    // The affine pattern follows logical bytes, not the physical permutation.
    mul x2, x17, x0
    str x2, [sp, #168]
    mov x1, x0
    ldr x2, [x8, #40]
    sub x2, x2, #1
    cmp x17, x2
    b.ne Lpaged_prefill_write_length_ready
    ldr x1, [x8, #64]           // terminal block exact valid bytes
Lpaged_prefill_write_length_ready:
    str x1, [sp, #176]

    LLM_PAGED_PREFILL_WRITE_BLOCK 144, 0x4B4B, 0x4B4B, 0x4B4B, 0x4B4B
    LLM_PAGED_PREFILL_WRITE_BLOCK 152, 0x5656, 0x5656, 0x5656, 0x5656

    add x17, x17, #1
    ldr x0, [sp, #136]
    cmp x17, x0
    b.lo Lpaged_prefill_assignment_block_write_loop

    // Start from floor(first_owned_token / Q) * Q. The first executed tile is
    // therefore the first prefix with a non-empty assignment intersection.
    ldr x0, [x8, #24]
    ldr x1, [x8, #48]           // block_tokens
    mul x2, x0, x1
    ldr x1, [x8, #80]           // attention_query_tile_tokens
    udiv x3, x2, x1
    mul x3, x3, x1
    str x3, [sp, #184]

Lpaged_prefill_tile_loop:
    // next_end = current_end + min(Q, P - current_end), overflow-free.
    ldr x0, [x8, #72]           // prompt_tokens
    ldr x2, [sp, #184]
    sub x3, x0, x2
    ldr x1, [x8, #80]
    cmp x3, x1
    csel x3, x3, x1, lo
    add x2, x2, x3
    str x2, [sp, #184]

    // Preserve exact prefix bytes and compute ceil(tile_end / G) without
    // evaluating tile_end + G - 1.
    ldr x0, [x8, #88]           // record_bytes
    mul x3, x2, x0
    str x3, [sp, #192]
    ldr x0, [x8, #48]           // block_tokens
    udiv x4, x2, x0
    msub x5, x4, x0, x2
    cbz x5, Lpaged_prefill_prefix_block_count_ready
    add x4, x4, #1
Lpaged_prefill_prefix_block_count_ready:
    ldr x0, [sp, #136]
    cmp x4, x0
    csel x4, x4, x0, lo
    str x4, [sp, #200]          // exclusive visited logical-block end

    ldr x17, [x8, #24]
    cmp x17, x4
    b.hs Lpaged_prefill_next_tile

    // Source-audit anchor: every K block in this tile precedes every V block.
Lpaged_prefill_tile_k_scan:
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // KScan visit kind = 1.
    LLM_PAGED_PREFILL_MIX_LOOKUP x23, x24, 0x23D6, 0x2662, 0x9376, 0x29A0

    ldr x2, [x8, #56]
    mul x3, x7, x2
    ldr x0, [x8, #8]
    add x0, x0, x3
    str x0, [sp, #144]
    mul x3, x17, x2
    str x3, [sp, #168]
    ldr x0, [sp, #192]
    sub x0, x0, x3
    cmp x0, x2
    csel x0, x0, x2, lo
    str x0, [sp, #176]
    LLM_PAGED_PREFILL_SCAN_BLOCK 144, x23, x24, x25, x26

    add x17, x17, #1
    ldr x0, [sp, #200]
    cmp x17, x0
    b.lo Lpaged_prefill_tile_k_scan

    // V restarts the identical logical range and reloads every table entry.
    ldr x17, [x8, #24]

Lpaged_prefill_tile_v_scan:
    ldr x0, [x8, #0]
    ldr w7, [x0, x17, lsl #2]
    // VScan visit kind = 2.
    LLM_PAGED_PREFILL_MIX_LOOKUP x27, x28, 0x35C1, 0x3993, 0xDD31, 0xBE70

    ldr x2, [x8, #56]
    mul x3, x7, x2
    ldr x0, [x8, #16]
    add x0, x0, x3
    str x0, [sp, #152]
    mul x3, x17, x2
    str x3, [sp, #168]
    ldr x0, [sp, #192]
    sub x0, x0, x3
    cmp x0, x2
    csel x0, x0, x2, lo
    str x0, [sp, #176]
    LLM_PAGED_PREFILL_SCAN_BLOCK 152, x27, x28, x29, x30

    add x17, x17, #1
    ldr x0, [sp, #200]
    cmp x17, x0
    b.lo Lpaged_prefill_tile_v_scan

Lpaged_prefill_next_tile:
    ldr x2, [sp, #184]
    ldr x0, [x8, #72]
    cmp x2, x0
    b.lo Lpaged_prefill_tile_loop

Lpaged_prefill_next_assignment:
    add x8, x8, #112
    ldr x10, [sp, #128]
    cmp x8, x10
    b.lo Lpaged_prefill_assignment_loop

Lpaged_prefill_next_layer:
    add x16, x16, #48
    b Lpaged_prefill_layer_loop

Lpaged_prefill_operation_complete:
    add x15, x15, x14
    ldr x0, [sp, #120]
    add x0, x0, #1
    str x0, [sp, #120]
    subs x12, x12, #1
    b.ne Lpaged_prefill_operation_loop

Lpaged_prefill_store_output:
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
    ldp x19, x20, [sp], #224

Lpaged_prefill_return_without_frame:
    ret
