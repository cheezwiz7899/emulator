// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>

#include "common/logging.h"
#include "core/arm/scarlet/scarlet_fault_handler.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/memory.h"

namespace Core {
namespace {

// The arena the guest runs on: guest address a is host byte g_arena_base[a].
u8* g_arena_base{};
size_t g_arena_size{};
Memory::Memory* g_memory{};

struct GuestSink {
    void* context{};
    GuestFaultFn fn{};
};
// Per host thread: a thread is "in guest code" between SetGuestFaultSink(ctx, fn) and
// SetGuestFaultSink(nullptr, nullptr). The coroutine runs on the same OS thread that calls
// RunThread, so the thread-local is valid wherever the guest is executing.
thread_local GuestSink g_sink;

[[nodiscard]] bool InArena(const void* address) {
    const auto* p = static_cast<const u8*>(address);
    return p >= g_arena_base && p < g_arena_base + g_arena_size;
}

// Handles a fault in the arena the way NCE does. True: retry the faulting instruction.
[[nodiscard]] bool TryInvalidate(const void* address) {
    const u64 vaddr =
        static_cast<u64>(static_cast<const u8*>(address) - g_arena_base) & ~u64{Memory::CITRON_PAGEMASK};
    return g_memory->InvalidateNCE(vaddr, Memory::CITRON_PAGESIZE);
}

} // namespace

void SetGuestFaultSink(void* context, GuestFaultFn fn) {
    g_sink = {context, fn};
}

} // namespace Core

// =================================================================================================
#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace Core {
namespace {

// Windows reports a protected-page access as EXCEPTION_ACCESS_VIOLATION with
// ExceptionInformation[0] = 0 (read) / 1 (write) / 8 (DEP) and [1] = the address. HostMemory
// protects the arena with VirtualProtect (PAGE_READONLY / PAGE_NOACCESS for pages the GPU caches),
// so a guest store to such a page arrives here.
//
// A vectored handler runs first, on the faulting thread's stack (the guest coroutine's;
// boost.context keeps the TEB stack bounds current across jumps), before any frame is unwound, so
// generated code needs no unwind info for this path.
LONG CALLBACK Handler(PEXCEPTION_POINTERS info) {
    const EXCEPTION_RECORD& record = *info->ExceptionRecord;
    if (record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const ULONG_PTR kind = record.ExceptionInformation[0];
    if (kind == 8) { // execute fault: never ours
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto* address = reinterpret_cast<const void*>(record.ExceptionInformation[1]);

    if (InArena(address) && TryInvalidate(address)) {
        return EXCEPTION_CONTINUE_EXECUTION; // the access is retried
    }

    if (g_sink.fn != nullptr) {
        // A real guest fault on a thread running guest code. Do not run the sink inside the
        // exception dispatcher: redirect the thread to it instead, on its own stack, below the
        // faulting frame. The guest thread is parked there and never resumed.
        CONTEXT* ctx = info->ContextRecord;
        u64 rsp = (ctx->Rsp - 0x200) & ~u64{15}; // clear of anything the faulting frame uses
        rsp -= 8;                                 // a return address: rsp % 16 == 8 at entry
        *reinterpret_cast<u64*>(rsp) = 0;
        ctx->Rsp = rsp;
        ctx->Rip = reinterpret_cast<DWORD64>(g_sink.fn);
        ctx->Rcx = reinterpret_cast<DWORD64>(g_sink.context); // Win64 argument registers
        ctx->Rdx = reinterpret_cast<DWORD64>(address);
        ctx->R8 = kind == 1 ? 1 : 0;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

} // namespace

void InstallScarletFaultHandler(System& system) {
    static std::once_flag once;
    std::call_once(once, [&] {
        auto& buffer = system.DeviceMemory().buffer;
        g_arena_base = buffer.VirtualBasePointer();
        g_arena_size = buffer.VirtualSize();
        g_memory = &system.ApplicationMemory();
        // First in the chain: ahead of any other vectored handler that might see the fault.
        if (AddVectoredExceptionHandler(1, &Handler) == nullptr) {
            LOG_CRITICAL(Core_ARM, "AddVectoredExceptionHandler failed; GPU caches will not see "
                                   "guest writes");
        }
    });
}

} // namespace Core

// =================================================================================================
#elif defined(__unix__) || defined(__APPLE__)

#include <csignal>
#include <ucontext.h>

namespace Core {
namespace {

struct sigaction g_orig_segv{};
struct sigaction g_orig_bus{};

void Forward(int sig, siginfo_t* info, void* context, const struct sigaction& orig) {
    if ((orig.sa_flags & SA_SIGINFO) != 0 && orig.sa_sigaction != nullptr) {
        orig.sa_sigaction(sig, info, context);
    } else if (orig.sa_handler != SIG_DFL && orig.sa_handler != SIG_IGN && orig.sa_handler) {
        orig.sa_handler(sig);
    } else {
        // Nobody wants it: default action, and the faulting instruction runs again and crashes.
        struct sigaction dfl{};
        dfl.sa_handler = SIG_DFL;
        sigaction(sig, &dfl, nullptr);
    }
}

// Redirects the faulting thread to the sink, on its own stack. Linux x86-64 only (the only host
// the recompiler backend targets).
bool Redirect(void* context, void* address) {
#if defined(__linux__) && defined(__x86_64__)
    auto* uc = static_cast<ucontext_t*>(context);
    auto& g = uc->uc_mcontext.gregs;
    u64 rsp = (static_cast<u64>(g[REG_RSP]) - 256) & ~u64{15}; // below the 128-byte red zone
    rsp -= 8; // a return address: rsp % 16 == 8 at entry
    *reinterpret_cast<u64*>(rsp) = 0;
    const u64 is_write = (static_cast<u64>(g[REG_ERR]) >> 1) & 1;
    g[REG_RSP] = static_cast<greg_t>(rsp);
    g[REG_RIP] = static_cast<greg_t>(reinterpret_cast<u64>(g_sink.fn));
    g[REG_RDI] = static_cast<greg_t>(reinterpret_cast<u64>(g_sink.context)); // SysV arguments
    g[REG_RSI] = static_cast<greg_t>(reinterpret_cast<u64>(address));
    g[REG_RDX] = static_cast<greg_t>(is_write);
    return true;
#else
    (void)context;
    (void)address;
    return false;
#endif
}

void Handler(int sig, siginfo_t* info, void* context) {
    if (InArena(info->si_addr) && TryInvalidate(info->si_addr)) {
        return; // the access is retried
    }
    if (g_sink.fn != nullptr && Redirect(context, info->si_addr)) {
        return; // returns into the sink, which parks the guest thread
    }
    Forward(sig, info, context, sig == SIGSEGV ? g_orig_segv : g_orig_bus);
}

} // namespace

void InstallScarletFaultHandler(System& system) {
    static std::once_flag once;
    std::call_once(once, [&] {
        auto& buffer = system.DeviceMemory().buffer;
        g_arena_base = buffer.VirtualBasePointer();
        g_arena_size = buffer.VirtualSize();
        g_memory = &system.ApplicationMemory();

        struct sigaction action{};
        action.sa_flags = SA_SIGINFO;
        action.sa_sigaction = &Handler;
        sigemptyset(&action.sa_mask);
        sigaction(SIGSEGV, &action, &g_orig_segv);
        sigaction(SIGBUS, &action, &g_orig_bus);
    });
}

} // namespace Core

#else

namespace Core {
void InstallScarletFaultHandler(System&) {
    LOG_ERROR(Core_ARM, "ScarletNX backend: no write tracking on this platform; GPU caches will "
                        "not see guest writes");
}
} // namespace Core

#endif
