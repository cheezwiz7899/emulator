// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/lru_cache.h"
#include "video_core/cache_memory_pressure.h"

using VideoCommon::CacheMemoryPressure;
using VideoCommon::HeapMemoryBudget;

TEST_CASE("Cache and heap pressure exit independently", "[video_core]") {
    CacheMemoryPressure pressure;
    std::array heaps{HeapMemoryBudget{74, 100}};
    pressure.Update(75, 100, heaps);
    REQUIRE(pressure.IsHigh());
    pressure.Update(65, 100, heaps);
    REQUIRE(pressure.IsHigh());
    pressure.Update(64, 100, heaps);
    REQUIRE_FALSE(pressure.IsHigh()); // Unlatched 74% heap cannot lock cache pressure on.

    heaps[0].usage = 75;
    pressure.Update(74, 100, heaps);
    REQUIRE(pressure.IsHigh());
    heaps[0].usage = 65;
    pressure.Update(74, 100, heaps);
    REQUIRE(pressure.IsHigh());
    heaps[0].usage = 64;
    pressure.Update(74, 100, heaps);
    REQUIRE_FALSE(pressure.IsHigh()); // Unlatched 74% cache cannot lock heap pressure on.
}

TEST_CASE("Both pressure sources retain their own hysteresis", "[video_core]") {
    CacheMemoryPressure pressure;
    std::array heaps{HeapMemoryBudget{75, 100}};
    pressure.Update(75, 100, heaps);
    heaps[0].usage = 64;
    pressure.Update(70, 100, heaps);
    REQUIRE(pressure.IsHigh());
    heaps[0].usage = 70;
    pressure.Update(64, 100, heaps);
    REQUIRE_FALSE(pressure.IsHigh());
}

TEST_CASE("Highest heap ratio controls pressure rather than summed heap budgets", "[video_core]") {
    CacheMemoryPressure pressure;
    std::array heaps{HeapMemoryBudget{90, 100}, HeapMemoryBudget{0, 10000}};
    pressure.Update(0, 100, heaps);
    REQUIRE(pressure.IsHigh());
    REQUIRE(pressure.IsCritical());
    heaps[0].usage = 80;
    pressure.Update(0, 100, heaps);
    REQUIRE(pressure.IsCritical());
    heaps[0].usage = 79;
    pressure.Update(0, 100, heaps);
    REQUIRE_FALSE(pressure.IsCritical());
    REQUIRE(pressure.IsHigh());

    // The maximum can move to another heap while maintaining a single heap latch.
    heaps[0].usage = 0;
    heaps[1] = {65, 100};
    pressure.Update(0, 100, heaps);
    REQUIRE(pressure.IsHigh());
    heaps[1].usage = 64;
    pressure.Update(0, 100, heaps);
    REQUIRE_FALSE(pressure.IsHigh());
}

TEST_CASE("UMA heap usage is independent of cache capacity and cannot impersonate cache ownership",
          "[video_core]") {
    CacheMemoryPressure pressure;
    std::array heaps{HeapMemoryBudget{800, 1600}};
    pressure.Update(10, 400, heaps);
    REQUIRE_FALSE(pressure.IsHigh());
    pressure.Update(400, 400, heaps);
    REQUIRE(pressure.IsHigh());
    REQUIRE_FALSE(pressure.IsCritical()); // Even a full cache cannot trigger aggressive GC.
}

TEST_CASE("Unknown budgets fall back to cache and integer thresholds cannot wrap", "[video_core]") {
    CacheMemoryPressure pressure;
    std::array heaps{HeapMemoryBudget{std::numeric_limits<u64>::max(), 0}};
    pressure.Update(74, 100, heaps);
    REQUIRE_FALSE(pressure.IsHigh());
    pressure.Update(75, 100, heaps);
    REQUIRE(pressure.IsHigh());
    REQUIRE_FALSE(pressure.IsCritical());
    constexpr u64 max = std::numeric_limits<u64>::max();
    heaps[0] = {max, max};
    pressure.Update(0, 0, heaps);
    REQUIRE(pressure.IsCritical());
    heaps[0] = {2, 3}; // Below 80% despite rounding a tiny budget.
    pressure.Update(0, 0, heaps);
    REQUIRE_FALSE(pressure.IsCritical());
}

TEST_CASE("GC cooldown survives pressure exit and reentry", "[video_core]") {
    CacheMemoryPressure pressure;
    pressure.Update(75, 100, {});
    REQUIRE(pressure.CanReclaim(10, 8));
    pressure.Reclaimed(10);
    REQUIRE_FALSE(pressure.CanReclaim(17, 8));
    pressure.Update(0, 100, {});
    pressure.Update(75, 100, {});
    REQUIRE_FALSE(pressure.CanReclaim(17, 8));
    REQUIRE(pressure.CanReclaim(18, 8));
}

namespace {
struct LRUTraits {
    using ObjectType = int;
    using TickType = u64;
};
} // namespace

TEST_CASE("Bounded LRU resumes beyond protected entries", "[video_core][common]") {
    Common::LeastRecentlyUsedCache<LRUTraits> cache;
    for (int i = 0; i < 6; ++i) {
        cache.Insert(i, 0);
    }
    std::vector<int> visited;
    auto visit = [&](int obj) {
        visited.push_back(obj);
        return false;
    };
    cache.ForEachItemBelow(0, 2, visit);
    REQUIRE(visited == std::vector<int>{0, 1});
    cache.ForEachItemBelow(0, 2, visit);
    REQUIRE(visited == std::vector<int>{0, 1, 2, 3});
    cache.ForEachItemBelow(0, 2, visit);
    REQUIRE(visited == std::vector<int>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("Bounded LRU cursor survives freeing touching and reusing its slot", "[video_core][common]") {
    Common::LeastRecentlyUsedCache<LRUTraits> cache;
    const auto first = cache.Insert(0, 0);
    const auto second = cache.Insert(1, 0);
    cache.Insert(2, 0);
    std::vector<int> visited;
    auto visit = [&](int obj) {
        visited.push_back(obj);
        return false;
    };
    cache.ForEachItemBelow(0, 1, visit);
    cache.Free(second);
    cache.Insert(3, 10); // Reuses the slot but must not redirect the saved cursor.
    cache.Touch(first, 11);
    cache.ForEachItemBelow(0, 4, visit);
    REQUIRE(visited == std::vector<int>{0, 2});
    cache.ForEachItemBelow(11, 4, visit);
    REQUIRE(visited == std::vector<int>{0, 2, 2, 3, 0});
}

TEST_CASE("Bounded LRU advances after callback eviction and early stop", "[video_core][common]") {
    Common::LeastRecentlyUsedCache<LRUTraits> cache;
    const auto first = cache.Insert(0, 0);
    const auto second = cache.Insert(1, 0);
    cache.Insert(2, 0);
    cache.ForEachItemBelow(0, 4, [&](int obj) {
        REQUIRE(obj == 0);
        cache.Free(first);
        return true;
    });
    cache.Touch(second, 10); // Cursor must advance to the remaining old entry.
    std::vector<int> visited;
    cache.ForEachItemBelow(0, 4, [&](int obj) {
        visited.push_back(obj);
        return false;
    });
    REQUIRE(visited == std::vector<int>{2});
}
