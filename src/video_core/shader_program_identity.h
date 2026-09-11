// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "common/cityhash.h"
#include "common/common_types.h"

namespace VideoCommon {

// A Maxwell program's stable identity begins at its program header and ends
// immediately before its terminal self-branch. CFG traversal must not affect it.
inline constexpr u64 MAXWELL_SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
inline constexpr u64 MAXWELL_SELF_BRANCH_B = 0xE2400FFFFF07000FULL;

[[nodiscard]] inline std::optional<size_t> FindMaxwellProgramSize(
    std::span<const u64> program) noexcept {
    for (size_t index = 0; index < program.size(); ++index) {
        const u64 instruction = program[index];
        if (instruction == MAXWELL_SELF_BRANCH_A || instruction == MAXWELL_SELF_BRANCH_B) {
            return index * sizeof(u64);
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<u64> ComputeMaxwellProgramIdentity(
    std::span<const u64> program) noexcept {
    const std::optional<size_t> size = FindMaxwellProgramSize(program);
    if (!size) {
        return std::nullopt;
    }
    return Common::CityHash64(reinterpret_cast<const char*>(program.data()), *size);
}

// Some live callers already know the terminal offset from a bounded CFG/header
// walk. Keep that fast path, but do not let an untrusted serialized offset turn
// into an out-of-bounds hash read. A shader program is word-addressed, so a
// partial word is never a valid identity span either.
[[nodiscard]] inline std::optional<u64> ComputeMaxwellProgramIdentity(
    std::span<const u64> program, size_t size_bytes) noexcept {
    // A terminal branch at the candidate start has no executable program
    // span. Hashing zero bytes would give every such malformed candidate the
    // same identity and could make scanner bookkeeping look like a real hit.
    if (size_bytes == 0 || size_bytes % sizeof(u64) != 0 || size_bytes > program.size_bytes()) {
        return std::nullopt;
    }
    return Common::CityHash64(reinterpret_cast<const char*>(program.data()), size_bytes);
}

} // namespace VideoCommon
