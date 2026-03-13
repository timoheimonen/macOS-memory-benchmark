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
// core_to_core_initiator_round_trips_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void core_to_core_initiator_round_trips_asm(
//       uint32_t* turn_ptr,
//       size_t round_trips,
//       uint32_t initiator_turn,
//       uint32_t responder_turn);
// Purpose:
//   Execute initiator side of a cache-line handoff ping-pong loop.
//   Each round trip waits for initiator ownership, hands token to responder,
//   then waits for token return.
// Arguments:
//   x0 = turn_ptr
//   x1 = round_trips
//   w2 = initiator_turn
//   w3 = responder_turn
// Returns:
//   (none)
// Clobbers:
//   w8
// -----------------------------------------------------------------------------

.global _core_to_core_initiator_round_trips_asm
.align 4
_core_to_core_initiator_round_trips_asm:
    cbz x1, c2c_initiator_end

c2c_initiator_loop:
c2c_initiator_wait_turn:
    ldar w8, [x0]                    // Acquire current token
    cmp w8, w2                       // Wait until initiator owns token
    b.ne c2c_initiator_wait_turn

    stlr w3, [x0]                    // Release token to responder

c2c_initiator_wait_return:
    ldar w8, [x0]                    // Acquire current token
    cmp w8, w2                       // Wait until responder returns token
    b.ne c2c_initiator_wait_return

    subs x1, x1, #1
    b.ne c2c_initiator_loop

c2c_initiator_end:
    ret

// -----------------------------------------------------------------------------
// core_to_core_responder_round_trips_asm
// -----------------------------------------------------------------------------
// C++ Prototype:
//   extern "C" void core_to_core_responder_round_trips_asm(
//       uint32_t* turn_ptr,
//       size_t round_trips,
//       uint32_t responder_turn,
//       uint32_t initiator_turn);
// Purpose:
//   Execute responder side of a cache-line handoff ping-pong loop.
//   Each round trip waits for responder ownership and hands token back.
// Arguments:
//   x0 = turn_ptr
//   x1 = round_trips
//   w2 = responder_turn
//   w3 = initiator_turn
// Returns:
//   (none)
// Clobbers:
//   w8
// -----------------------------------------------------------------------------

.global _core_to_core_responder_round_trips_asm
.align 4
_core_to_core_responder_round_trips_asm:
    cbz x1, c2c_responder_end

c2c_responder_loop:
c2c_responder_wait_turn:
    ldar w8, [x0]                    // Acquire current token
    cmp w8, w2                       // Wait until responder owns token
    b.ne c2c_responder_wait_turn

    stlr w3, [x0]                    // Release token to initiator

    subs x1, x1, #1
    b.ne c2c_responder_loop

c2c_responder_end:
    ret
