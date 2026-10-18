// SPDX-FileCopyrightText: Copyright 2026 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>

#include "common/common_types.h"

namespace Shader {

constexpr u32 AggregateDynamicDescriptorCap(std::span<const u32> counts, u64 fixed_descriptors,
                                            u64 descriptor_limit) {
    if (counts.empty() || fixed_descriptors + counts.size() > descriptor_limit) {
        return 0;
    }
    u64 actual_descriptors{fixed_descriptors};
    u32 largest_count{};
    for (const u32 count : counts) {
        actual_descriptors += count;
        largest_count = std::max(largest_count, count);
    }
    if (actual_descriptors <= descriptor_limit) {
        return largest_count;
    }

    u32 lower_bound{1};
    u32 upper_bound{largest_count};
    while (lower_bound < upper_bound) {
        const u32 candidate{lower_bound + (upper_bound - lower_bound + 1) / 2};
        u64 candidate_descriptors{fixed_descriptors};
        for (const u32 count : counts) {
            candidate_descriptors += std::min(count, candidate);
        }
        if (candidate_descriptors <= descriptor_limit) {
            lower_bound = candidate;
        } else {
            upper_bound = candidate - 1;
        }
    }
    return lower_bound;
}

} // namespace Shader
