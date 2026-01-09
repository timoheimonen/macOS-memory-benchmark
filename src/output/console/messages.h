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
//

/**
 * @file messages.h
 * @brief Backward compatibility wrapper for the Messages namespace
 *
 * This header provides backward compatibility by forwarding to the consolidated
 * messages header located in the messages subdirectory. It allows existing code
 * to continue using the original include path while the actual implementation
 * has been reorganized.
 *
 * @deprecated This is a compatibility header. New code should include "messages/messages.h" directly.
 *
 * @see messages/messages.h for the actual Messages namespace declarations
 */

#ifndef MESSAGES_H
#define MESSAGES_H

// Backward compatibility: include the consolidated header from messages subdirectory
#include "messages/messages.h"

#endif // MESSAGES_H

