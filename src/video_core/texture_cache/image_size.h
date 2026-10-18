// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <stdexcept>
#include <string>

#include "common/common_types.h"

namespace VideoCommon {

class InvalidImageSize final : public std::runtime_error {
  public:
    explicit InvalidImageSize(const std::string &reason) : std::runtime_error{reason} {}
};

// Texture-cache sizes and decoder dimensions currently have 32-bit storage. Calculate in
// 64 bits and reject unrepresentable results before narrowing, rather than wrapping the
// backing range or silently changing the guest's array, sparse or aliasing semantics.
namespace ImageSize {

[[nodiscard]] constexpr u64 Add(u64 lhs, u64 rhs) {
    if (rhs > std::numeric_limits<u64>::max() - lhs) {
        throw InvalidImageSize{"Image size addition overflow"};
    }
    return lhs + rhs;
}

[[nodiscard]] constexpr u64 Multiply(u64 lhs, u64 rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<u64>::max() / rhs) {
        throw InvalidImageSize{"Image size multiplication overflow"};
    }
    return lhs * rhs;
}

[[nodiscard]] constexpr u64 Shift(u64 value, u32 shift) {
    if (shift >= 64 || value > (std::numeric_limits<u64>::max() >> shift)) {
        throw InvalidImageSize{"Image size shift overflow"};
    }
    return value << shift;
}

[[nodiscard]] constexpr u64 AlignUp(u64 value, u64 alignment) {
    if (alignment == 0) {
        throw InvalidImageSize{"Zero image alignment"};
    }
    const u64 remainder = value % alignment;
    return remainder == 0 ? value : Add(value, alignment - remainder);
}

[[nodiscard]] constexpr u64 DivCeil(u64 value, u64 divisor) {
    if (divisor == 0) {
        throw InvalidImageSize{"Zero image divisor"};
    }
    return value / divisor + (value % divisor != 0);
}

[[nodiscard]] constexpr u32 Narrow(u64 value) {
    if (value > std::numeric_limits<u32>::max()) {
        throw InvalidImageSize{"Image size exceeds texture-cache 32-bit storage: " +
                               std::to_string(value) + " bytes/units"};
    }
    return static_cast<u32>(value);
}

[[nodiscard]] constexpr u32 AlignUp32(u64 value, u64 alignment) {
    return Narrow(AlignUp(value, alignment));
}

[[nodiscard]] constexpr u32 AlignUpLog2(u64 value, u32 shift) {
    return AlignUp32(value, Shift(1, shift));
}

[[nodiscard]] constexpr u32 DivCeil32(u64 value, u64 divisor) {
    return Narrow(DivCeil(value, divisor));
}

[[nodiscard]] constexpr u32 DivCeilLog2(u64 value, u32 shift) {
    return DivCeil32(value, Shift(1, shift));
}

} // namespace ImageSize
} // namespace VideoCommon
