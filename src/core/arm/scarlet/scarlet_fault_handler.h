// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {

class System;

/// Makes guest stores/loads that hit rasterizer-cached pages of the fastmem arena behave like the
/// NCE backend's: the fault handler invalidates the GPU side (Memory::InvalidateNCE), unprotects
/// the page and lets the faulting instruction retry. Idempotent.
///
/// POSIX: a SIGSEGV/SIGBUS handler. Windows: a vectored exception handler (first in the chain).
void InstallScarletFaultHandler(System& system);

/// Called on the faulting thread, on the guest coroutine's stack, in place of the faulting
/// instruction when a fault is not a GPU-tracked page (a real guest fault: wild pointer, unmapped
/// address). It must not return: it parks the guest thread. context/address/is_write describe the
/// fault; address is the host address.
using GuestFaultFn = void (*)(void* context, u64 address, u64 is_write);

/// Marks the calling host thread as executing guest code (fn != nullptr) or not. Only a fault on
/// a thread that is executing guest code is redirected; any other fault is passed on untouched.
void SetGuestFaultSink(void* context, GuestFaultFn fn);

} // namespace Core
