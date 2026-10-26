// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "common/common_types.h"
#include "core/arm/arm_interface.h"

namespace Core {

class System;

/// EXPERIMENTAL AArch64 backend running games recompiled ahead of time by ScarletNX's recompiler
/// library (libscarletnx-recompiler, GPL-3.0-or-later) instead of dynarmic. Select ScarletNX in
/// Configure > CPU when built with CITRON_ENABLE_SCARLET_CPU.
///
/// The recompiler's own header is a GNU C dialect that defines macros and typedefs; it is kept out
/// of this header (pimpl).
class ArmScarlet64 final : public ArmInterface {
public:
    ArmScarlet64(System& system, bool uses_wall_clock, Kernel::KProcess* process,
                 std::size_t core_index);
    ~ArmScarlet64() override;

    void Initialize() override;

    Architecture GetArchitecture() const override {
        return Architecture::AArch64;
    }

    HaltReason RunThread(Kernel::KThread* thread) override;
    HaltReason StepThread(Kernel::KThread* thread) override;

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override;
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override;
    void SetTpidrroEl0(u64 value) override;

    void GetSvcArguments(std::span<uint64_t, 8> args) const override;
    void SetSvcArguments(std::span<const uint64_t, 8> args) override;
    u32 GetSvcNumber() const override;

    void SignalInterrupt(Kernel::KThread* thread) override;
    void ClearInstructionCache() override;
    void InvalidateCacheRange(u64 addr, std::size_t size) override;

protected:
    const Kernel::DebugWatchpoint* HaltedWatchpoint() const override;
    void RewindBreakpointInstruction() override;

private:
    [[nodiscard]] bool IsInvalidationLeader() const;

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace Core
