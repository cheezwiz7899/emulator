// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>

#include "common/assert.h"
#include "common/fiber.h"
#include "common/virtual_buffer.h"

#if !defined(WIN32)
#include <boost/context/detail/fcontext.hpp>
#else
#include <Windows.h>
#endif

namespace Common {

constexpr std::size_t DEFAULT_STACK_SIZE = 512 * 1024;

struct Fiber::FiberImpl {
    FiberImpl() {}

    std::array<u8, DEFAULT_STACK_SIZE> stack = {};
    std::array<u8, DEFAULT_STACK_SIZE> rewind_stack = {};

    std::mutex guard;
    std::function<void()> entry_point;
    std::function<void()> rewind_point;
    std::shared_ptr<Fiber> previous_fiber;
    bool is_thread_fiber{};
    bool released{};

    u8* stack_limit{};
    u8* rewind_stack_limit{};
#if !defined(WIN32)
    boost::context::detail::fcontext_t context{};
    boost::context::detail::fcontext_t rewind_context{};
#else
    LPVOID context{};
    LPVOID rewind_context{};
#endif
};

void Fiber::SetRewindPoint(std::function<void()>&& rewind_func) {
    impl->rewind_point = std::move(rewind_func);
}

#if !defined(WIN32)
void Fiber::Start(boost::context::detail::transfer_t& transfer) {
    ASSERT(impl->previous_fiber != nullptr);
    impl->previous_fiber->impl->context = transfer.fctx;
    impl->previous_fiber->impl->guard.unlock();
    impl->previous_fiber.reset();
    impl->entry_point();
    UNREACHABLE();
}

void Fiber::OnRewind([[maybe_unused]] boost::context::detail::transfer_t& transfer) {
    ASSERT(impl->context != nullptr);
    impl->context = impl->rewind_context;
    impl->rewind_context = nullptr;
    u8* tmp = impl->stack_limit;
    impl->stack_limit = impl->rewind_stack_limit;
    impl->rewind_stack_limit = tmp;
    impl->rewind_point();
    UNREACHABLE();
}

void Fiber::FiberStartFunc(boost::context::detail::transfer_t transfer) {
    auto* fiber = static_cast<Fiber*>(transfer.data);
    fiber->Start(transfer);
}

void Fiber::RewindStartFunc(boost::context::detail::transfer_t transfer) {
    auto* fiber = static_cast<Fiber*>(transfer.data);
    fiber->OnRewind(transfer);
}

Fiber::Fiber(std::function<void()>&& entry_point_func) : impl{std::make_unique<FiberImpl>()} {
    impl->entry_point = std::move(entry_point_func);
    impl->stack_limit = impl->stack.data();
    impl->rewind_stack_limit = impl->rewind_stack.data();
    u8* stack_base = impl->stack_limit + DEFAULT_STACK_SIZE;
    impl->context =
        boost::context::detail::make_fcontext(stack_base, impl->stack.size(), FiberStartFunc);
}
#else
VOID CALLBACK Fiber::FiberStartFunc(LPVOID param) {
    auto* fiber = static_cast<Fiber*>(param);
    ASSERT(fiber->impl->previous_fiber != nullptr);
    fiber->impl->previous_fiber->impl->guard.unlock();
    fiber->impl->previous_fiber.reset();
    fiber->impl->entry_point();
    UNREACHABLE();
}

VOID CALLBACK Fiber::RewindStartFunc(LPVOID param) {
    auto* fiber = static_cast<Fiber*>(param);
    const LPVOID old_context = fiber->impl->context;
    fiber->impl->context = fiber->impl->rewind_context;
    fiber->impl->rewind_context = nullptr;
    DeleteFiber(old_context);
    fiber->impl->rewind_point();
    UNREACHABLE();
}

Fiber::Fiber(std::function<void()>&& entry_point_func) : impl{std::make_unique<FiberImpl>()} {
    impl->entry_point = std::move(entry_point_func);
    impl->stack_limit = impl->stack.data();
    impl->rewind_stack_limit = impl->rewind_stack.data();
    impl->context = CreateFiber(impl->stack.size(), FiberStartFunc, this);
    ASSERT_MSG(impl->context != nullptr, "CreateFiber failed");
}
#endif

Fiber::Fiber() : impl{std::make_unique<FiberImpl>()} {}

Fiber::~Fiber() {
    if (impl->released) {
        return;
    }
    const bool locked = impl->guard.try_lock();
    ASSERT_MSG(locked, "Destroying a fiber that's still running");
    if (locked) {
        impl->guard.unlock();
    }
#if defined(WIN32)
    if (impl->context != nullptr && !impl->is_thread_fiber) {
        DeleteFiber(impl->context);
    }
    if (impl->rewind_context != nullptr) {
        DeleteFiber(impl->rewind_context);
    }
#endif
}

void Fiber::Exit() {
    ASSERT_MSG(impl->is_thread_fiber, "Exiting non main thread fiber");
    if (!impl->is_thread_fiber) {
        return;
    }
    impl->guard.unlock();
    impl->released = true;
}

void Fiber::Rewind() {
    ASSERT(impl->rewind_point);
    ASSERT(impl->rewind_context == nullptr);
#if !defined(WIN32)
    u8* stack_base = impl->rewind_stack_limit + DEFAULT_STACK_SIZE;
    impl->rewind_context =
        boost::context::detail::make_fcontext(stack_base, impl->stack.size(), RewindStartFunc);
    boost::context::detail::jump_fcontext(impl->rewind_context, this);
#else
    impl->rewind_context = CreateFiber(impl->rewind_stack.size(), RewindStartFunc, this);
    ASSERT_MSG(impl->rewind_context != nullptr, "CreateFiber(rewind) failed");
    SwitchToFiber(impl->rewind_context);
    DeleteFiber(impl->rewind_context);
    impl->rewind_context = nullptr;
#endif
}

void Fiber::YieldTo(std::weak_ptr<Fiber> weak_from, Fiber& to) {
    to.impl->guard.lock();
    to.impl->previous_fiber = weak_from.lock();
#if !defined(WIN32)
    auto transfer = boost::context::detail::jump_fcontext(to.impl->context, &to);
    if (auto from = weak_from.lock()) {
        if (from->impl->previous_fiber == nullptr) {
            ASSERT_MSG(false, "previous_fiber is nullptr!");
            return;
        }
        from->impl->previous_fiber->impl->context = transfer.fctx;
        from->impl->previous_fiber->impl->guard.unlock();
        from->impl->previous_fiber.reset();
    }
#else
    SwitchToFiber(to.impl->context);
    if (auto from = weak_from.lock()) {
        if (from->impl->previous_fiber == nullptr) {
            ASSERT_MSG(false, "previous_fiber is nullptr!");
            return;
        }
        from->impl->previous_fiber->impl->guard.unlock();
        from->impl->previous_fiber.reset();
    }
#endif
}

std::shared_ptr<Fiber> Fiber::ThreadToFiber() {
    std::shared_ptr<Fiber> fiber = std::shared_ptr<Fiber>{new Fiber()};
#if !defined(WIN32)
    fiber->impl->guard.lock();
    fiber->impl->is_thread_fiber = true;
#else
    if (!IsThreadAFiber()) {
        if (ConvertThreadToFiber(nullptr) == nullptr) {
            ASSERT_MSG(false, "ConvertThreadToFiber failed");
        }
    }
    fiber->impl->context = GetCurrentFiber();
    fiber->impl->guard.lock();
    fiber->impl->is_thread_fiber = true;
#endif
    return fiber;
}

} // namespace Common
