// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

namespace VideoCommon {

struct HeapMemoryBudget {
    u64 usage{};
    u64 budget{};
};

class CacheMemoryPressure {
public:
    // Independent ratios: cache_usage/cache_target and max(heap.usage/heap.budget).
    // Driver budgets are advisory, especially on UMA, not reclaimable cache bytes.
    void Update(u64 cache_usage, u64 cache_target, std::span<const HeapMemoryBudget> heaps) {
        cache_active = AtLeast(cache_usage, cache_target, cache_active ? 65 : 75);
        bool heap_high = false;
        bool heap_critical = false;
        for (const auto& heap : heaps) {
            heap_high |= AtLeast(heap.usage, heap.budget, heap_active ? 65 : 75);
            heap_critical |= AtLeast(heap.usage, heap.budget, aggressive_active ? 80 : 90);
        }
        heap_active = heap_high;
        aggressive_active = heap_critical;
    }

    [[nodiscard]] bool IsHigh() const {
        return cache_active || heap_active;
    }

    [[nodiscard]] bool IsCritical() const {
        return aggressive_active;
    }

    [[nodiscard]] bool CanReclaim(u64 tick, u64 retirement_ticks) const {
        return IsHigh() && (!has_reclaimed || tick - last_reclaim_tick >= retirement_ticks);
    }

    void Reclaimed(u64 tick) {
        last_reclaim_tick = tick;
        has_reclaimed = true;
    }

private:
    static bool AtLeast(u64 usage, u64 target, u64 percent) {
        if (target == 0) {
            return false; // Unknown budget: use the cache ownership signal only.
        }
        // ceil(target * percent / 100), without overflowing u64.
        const u64 threshold = target / 100 * percent + (target % 100 * percent + 99) / 100;
        return usage >= threshold;
    }

    bool cache_active{};
    bool heap_active{};
    bool aggressive_active{};
    bool has_reclaimed{};
    u64 last_reclaim_tick{};
};

} // namespace VideoCommon
