// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

#include <boost/context/detail/fcontext.hpp>

#include "common/common_types.h"
#include "common/virtual_buffer.h"

namespace Core::Scarlet {

/// A stackful coroutine that runs recompiled guest code.
///
/// Recompiled guest functions are host functions, so a guest call chain is a host call chain and
/// the guest's SVC, poll and BRK hooks are called from deep inside generated code. Citron's kernel
/// wants RunThread() to *return* at each of those events (the SVC is handled outside the CPU).
/// Instead of unwinding out of generated code (setjmp/longjmp over frames that may have no unwind
/// info on Windows), the hook yields to RunThread() on this coroutine's own stack and a later
/// RunThread() resumes it: the generated frames stay intact in between.
///
/// Why not Common::Fiber: its stack is a fixed 512 KiB std::array and guest recursion depth maps to
/// host stack depth here. The stack is a lazily committed virtual buffer instead.
///
/// Guest frames hold no state that is not in the CPU struct (generated code writes registers back
/// before every runtime call), and no C++ objects with destructors. Abandoning a suspended
/// coroutine with Reset() is therefore safe.
///
/// Hooks must not cache thread_local addresses across Yield(): a suspended coroutine may be
/// resumed by a different host thread (guest threads migrate between cores).
class GuestCoroutine {
public:
    using Body = void (*)(void*);

    static constexpr std::size_t DefaultStackSize = 16 * 1024 * 1024;

    explicit GuestCoroutine(std::size_t stack_size = DefaultStackSize)
        : m_stack{stack_size}, m_stack_size{stack_size} {}

    GuestCoroutine(const GuestCoroutine&) = delete;
    GuestCoroutine& operator=(const GuestCoroutine&) = delete;

    /// Arms a fresh run of body(arg). Anything suspended before is abandoned.
    void Start(Body body, void* arg) {
        m_body = body;
        m_arg = arg;
        m_state = State::Suspended;
        m_guest = boost::context::detail::make_fcontext(m_stack.data() + m_stack_size,
                                                        m_stack_size, &GuestCoroutine::Entry);
    }

    /// Abandons a suspended run. No unwinding: the frames are simply dropped.
    void Reset() {
        m_state = State::Idle;
        m_guest = nullptr;
    }

    /// Host -> guest. Returns when the guest yields or the body returns.
    void Resume() {
        m_state = State::Running;
        auto t = boost::context::detail::jump_fcontext(m_guest, this);
        if (m_state == State::Suspended) {
            m_guest = t.fctx;
        }
    }

    /// Guest -> host. Called from inside the body (through the runtime hooks).
    [[gnu::noinline]] void Yield() {
        m_state = State::Suspended;
        auto t = boost::context::detail::jump_fcontext(m_host, nullptr);
        m_host = t.fctx;
    }

    [[nodiscard]] bool IsSuspended() const {
        return m_state == State::Suspended;
    }
    [[nodiscard]] bool IsFinished() const {
        return m_state == State::Finished;
    }

private:
    enum class State : u8 { Idle, Suspended, Running, Finished };

    static void Entry(boost::context::detail::transfer_t t) {
        auto* self = static_cast<GuestCoroutine*>(t.data);
        self->m_host = t.fctx;
        self->m_body(self->m_arg);
        self->m_state = State::Finished;
        boost::context::detail::jump_fcontext(self->m_host, nullptr);
        __builtin_unreachable();
    }

    Common::VirtualBuffer<u8> m_stack;
    std::size_t m_stack_size;
    boost::context::detail::fcontext_t m_guest{};
    boost::context::detail::fcontext_t m_host{};
    Body m_body{};
    void* m_arg{};
    State m_state{State::Idle};
};

} // namespace Core::Scarlet
