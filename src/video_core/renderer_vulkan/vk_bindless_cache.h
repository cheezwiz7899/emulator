// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <boost/container/small_vector.hpp>
#include "common/common_types.h"
#include "video_core/texture_cache/types.h"

namespace Vulkan {

// Stale SamplerIds are possible if the sampler pool is rebuilt with a different
// sampler at the same handle while the cbuf bytes don't change. Add a pool
// sequence-number check here if that ever surfaces as a visible bug.
struct BindlessCacheEntry {
    GPUVAddr key_addr{0};
    u32 key_count{0};
    u64 key_image_table_generation{};
    bool valid{false};
    boost::container::small_vector<u8, 256> last_bytes;
    boost::container::small_vector<VideoCommon::ImageViewInOut, 16> cached_views;
    boost::container::small_vector<VideoCommon::SamplerId, 16> cached_samplers;
};
constexpr size_t BINDLESS_CACHE_SIZE = 64;
using BindlessCache = std::array<BindlessCacheEntry, BINDLESS_CACHE_SIZE>;

inline BindlessCacheEntry* FindBindlessEntry(BindlessCache& cache, GPUVAddr addr, u32 count,
                                      u64 image_table_generation) {
    for (auto& entry : cache) {
        if (entry.valid && entry.key_addr == addr && entry.key_count == count &&
            entry.key_image_table_generation == image_table_generation) {
            return &entry;
        }
    }
    return nullptr;
}

inline BindlessCacheEntry& AcquireBindlessEntry(BindlessCache& cache, size_t& round_robin,
                                         GPUVAddr addr, u32 count,
                                         u64 image_table_generation) {
    if (auto* found = FindBindlessEntry(cache, addr, count, image_table_generation)) {
        return *found;
    }
    auto& slot = cache[round_robin];
    round_robin = (round_robin + 1) % BINDLESS_CACHE_SIZE;
    slot.key_addr = addr;
    slot.key_count = count;
    slot.key_image_table_generation = image_table_generation;
    slot.valid = false;
    return slot;
}

} // namespace Vulkan
