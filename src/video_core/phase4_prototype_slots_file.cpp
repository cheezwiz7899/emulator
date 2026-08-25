// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>

#include "common/logging.h"
#include "video_core/phase4_prototype_slots_file.h"

namespace VideoCommon {

namespace {
constexpr std::array<char, 8> PHASE4_PROTOTYPE_SLOTS_MAGIC{"citrp4s"};
// v1: initial format. Bump if this file's on-disk layout ever changes -- independent of
// SPIRV_CACHE_VERSION (spirv_cache.cpp), see this header's file-level doc comment for why.
constexpr u32 PHASE4_PROTOTYPE_SLOTS_VERSION = 1;
// Sanity bound against a corrupt/truncated file claiming an absurd count -- matches the
// pattern (not the number; nowhere near SpirvCache's 4M-word bound) SpirvCache::Load already
// uses for its own word_count field. Real counts are always <= kMaxPhase4PrototypeSlots
// (currently 8); this is deliberately far more permissive than that so a future increase to
// the cap doesn't require also touching this bound.
constexpr u32 kMaxSaneCount = 1024;
} // namespace

std::vector<Shader::Phase4PrototypeSlot>
LoadPhase4PrototypeSlots(const std::filesystem::path& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return {};
        }
        file.exceptions(std::ifstream::failbit);

        std::array<char, 8> magic{};
        u32 version{};
        file.read(magic.data(), 8);
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != PHASE4_PROTOTYPE_SLOTS_MAGIC || version != PHASE4_PROTOTYPE_SLOTS_VERSION) {
            LOG_WARNING(Render_Vulkan,
                        "Phase 4 prototype slots file version mismatch, discarding "
                        "learned data for this game (defaults are unaffected)");
            return {};
        }

        u32 count{};
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (count > kMaxSaneCount) {
            LOG_WARNING(Render_Vulkan, "Bogus Phase 4 prototype slot count {}, discarding",
                        count);
            return {};
        }

        std::vector<Shader::Phase4PrototypeSlot> slot_list;
        slot_list.reserve(count);
        for (u32 i = 0; i < count; ++i) {
            Shader::Phase4PrototypeSlot slot{};
            file.read(reinterpret_cast<char*>(&slot.cbuf_index), sizeof(slot.cbuf_index));
            file.read(reinterpret_cast<char*>(&slot.cbuf_offset), sizeof(slot.cbuf_offset));
            slot_list.push_back(slot);
        }
        return slot_list;
    } catch (const std::ios_base::failure&) {
        LOG_WARNING(Render_Vulkan,
                    "Failed to read Phase 4 prototype slots file, discarding learned data "
                    "for this game (defaults are unaffected)");
        return {};
    }
}

void SavePhase4PrototypeSlots(const std::filesystem::path& filename,
                               const std::vector<Shader::Phase4PrototypeSlot>& slot_list) {
    try {
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return;
        }
        file.exceptions(std::ofstream::failbit);

        file.write(PHASE4_PROTOTYPE_SLOTS_MAGIC.data(), 8);
        file.write(reinterpret_cast<const char*>(&PHASE4_PROTOTYPE_SLOTS_VERSION),
                   sizeof(PHASE4_PROTOTYPE_SLOTS_VERSION));
        const u32 count{static_cast<u32>(slot_list.size())};
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const Shader::Phase4PrototypeSlot& slot : slot_list) {
            file.write(reinterpret_cast<const char*>(&slot.cbuf_index), sizeof(slot.cbuf_index));
            file.write(reinterpret_cast<const char*>(&slot.cbuf_offset),
                       sizeof(slot.cbuf_offset));
        }
    } catch (const std::ios_base::failure&) {
        LOG_WARNING(Render_Vulkan, "Failed to save Phase 4 prototype slots file for this game");
    }
}

} // namespace VideoCommon
