// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

// ---------------------------------------------------------------------------
// SpirvKey — identifies a pre-translated SPIR-V program.
//   unique_hash : CityHash64 of the raw Maxwell bytecode (matches GenericEnvironment::Analyze).
//   cbuf_key    : hash of specialised cbuf values; 0 = speculative/AOT (no specialisation).
// ---------------------------------------------------------------------------
struct SpirvKey {
    u64 unique_hash;
    u64 cbuf_key; // 0 = speculative/AOT (no cbuf specialisation)
    u64 runtime_key; // Hash of RuntimeInfo fields

    bool operator==(const SpirvKey& o) const noexcept {
        return unique_hash == o.unique_hash && cbuf_key == o.cbuf_key && runtime_key == o.runtime_key;
    }
};

struct SpirvKeyHash {
    size_t operator()(const SpirvKey& k) const noexcept {
        // Boost hash_combine-style mix of the three 64-bit fields.
        size_t h = k.unique_hash;
        h ^= k.cbuf_key + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= k.runtime_key + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// ComputeCbufKey — hash of cbuf specialisation values used during translation.
// Zero means the shader was translated without cbuf folding (conservative).
// ---------------------------------------------------------------------------
u64 ComputeCbufKey(const std::unordered_map<u64, u32>& cbuf_values);

// ---------------------------------------------------------------------------
// SpirvCache — thread-safe store of pre-translated SPIR-V programs.
//
// On-disk format (spirv_cache.bin):
//   8 bytes   magic "citrspv\0"
//   u32       version (currently 1)
//   u32       num_entries
//   per entry:
//     u64     unique_hash
//     u64     cbuf_key
//     u32     word_count
//     u32[]   spirv words (word_count × 4 bytes)
// ---------------------------------------------------------------------------
class SpirvCache {
public:
    SpirvCache() = default;
    ~SpirvCache() = default;
    SpirvCache(const SpirvCache&) = delete;
    SpirvCache& operator=(const SpirvCache&) = delete;

    // Load from disk.  No-op if the file does not exist or has a different version.
    void Load(const std::filesystem::path& path);

    // Persist all entries to disk — two-phase (snapshot under lock, write outside).
    void Save(const std::filesystem::path& path) const;

    // Throttled save — only writes to disk if >= min_new_entries have accumulated
    // since the last save, or min_interval has elapsed.
    void SaveThrottled(const std::filesystem::path& path,
                       size_t min_new_entries = 64,
                       std::chrono::seconds min_interval = std::chrono::seconds{30}) const;

    // Look up a pre-translated SPIR-V program.
    // Returns nullopt on miss. The returned vector is a copy of the stored SPIR-V words.
    [[nodiscard]] std::optional<std::vector<u32>> Lookup(const SpirvKey& key) const;

    // Returns true if the cache already contains an entry for @p key.
    // Faster than Lookup() when the SPIR-V itself is not needed — avoids the vector copy.
    [[nodiscard]] bool Contains(const SpirvKey& key) const noexcept;

    // Returns true if the cache contains ANY entry whose unique_hash matches.
    // Use this for "have we already seen this shader?" guards where cbuf_key and
    // runtime_key are not yet known.
    [[nodiscard]] bool ContainsByUniqueHash(u64 unique_hash) const noexcept;

    /// Number of successful Lookup() calls since construction.
    [[nodiscard]] size_t HitCount() const noexcept { return hit_count_.load(); }
    /// Total number of Lookup() calls (hit or miss) since construction.
    [[nodiscard]] size_t LookupCount() const noexcept { return lookup_count_.load(); }

    [[nodiscard]] size_t Size() const;

    // Insert a real runtime-compiled entry (keyed with actual cbuf values and runtime info).
    void Insert(const SpirvKey& key, std::vector<u32> spirv);
    void Insert(u64 unique_hash, const std::unordered_map<u64, u32>& cbuf_values,
                u64 runtime_key, std::vector<u32> spirv);

    // Insert a speculative/AOT entry (cbuf_key = 0).
    void InsertSpeculative(u64 unique_hash, u64 runtime_key, std::vector<u32> spirv);

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<SpirvKey, std::vector<u32>, SpirvKeyHash> entries_;
    mutable bool dirty_{false};
    mutable std::atomic<size_t> hit_count_{0};
    mutable std::atomic<size_t> lookup_count_{0};

    // Throttle state for SaveThrottled — updated under mutex_.
    mutable size_t saved_entry_count_{0};
    mutable std::chrono::steady_clock::time_point last_save_time_{};
};

} // namespace VideoCommon
