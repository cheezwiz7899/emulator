// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <compare>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

namespace Shader {

struct TextureSlot {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 shift_left{};
    u32 secondary_cbuf_index{};
    u32 secondary_cbuf_offset{};
    u32 secondary_shift_left{};
    u32 count{};
    bool has_secondary{};

    auto operator<=>(const TextureSlot&) const noexcept = default;
};

struct TextureSlotHash {
    size_t operator()(const TextureSlot& slot) const noexcept {
        size_t hash{slot.cbuf_index};
        const auto combine = [&hash](u32 value) {
            hash ^= static_cast<size_t>(value) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        };
        combine(slot.cbuf_offset);
        combine(slot.shift_left);
        combine(slot.secondary_cbuf_index);
        combine(slot.secondary_cbuf_offset);
        combine(slot.secondary_shift_left);
        combine(slot.count);
        combine(slot.has_secondary ? 1U : 0U);
        return hash;
    }
};

struct TextureSlotShape {
    std::optional<TextureType> type;
    std::optional<TexturePixelFormat> pixel_format;
    std::optional<bool> is_integer;
};

using LogicalTextureSlots = std::unordered_map<TextureSlot, TextureSlotShape, TextureSlotHash>;
using LogicalTextureHandles = std::unordered_set<u32>;

} // namespace Shader
