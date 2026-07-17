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
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/shader_info.h"

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
    u64 texture_key; // Hash of Texture types and formats
    bool operator==(const SpirvKey& o) const noexcept {
        return unique_hash == o.unique_hash && cbuf_key == o.cbuf_key && runtime_key == o.runtime_key && texture_key == o.texture_key;
    }
};
struct SpirvKeyHash {
    size_t operator()(const SpirvKey& k) const noexcept {
        // Boost hash_combine-style mix of the four 64-bit fields.
        size_t h = k.unique_hash;
        h ^= k.cbuf_key + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= k.runtime_key + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= k.texture_key + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
// ---------------------------------------------------------------------------
// ComputeCbufKey — hash of cbuf specialisation values used during translation.
// Zero means the shader was translated without cbuf folding (conservative).
// ---------------------------------------------------------------------------
u64 ComputeCbufKey(const std::unordered_map<u64, u32>& cbuf_values);

// ---------------------------------------------------------------------------
// ComputeTextureKey — hash of texture types and formats used during translation.
// ---------------------------------------------------------------------------
u64 ComputeTextureKey(const std::unordered_map<u32, Shader::TextureType>& texture_types,
                      const std::unordered_map<u32, Shader::TexturePixelFormat>& texture_pixel_formats);

// ---------------------------------------------------------------------------
// ComputeBindingKey — hash of the *starting* descriptor binding state (the
// Bindings accumulator as it stood immediately before EmitSPIRV ran for this
// stage). EmitSPIRV bakes absolute descriptor binding numbers into the SPIR-V
// relative to this starting point, but nothing about it was previously part
// of SpirvKey. Two calls with identical unique_hash/cbuf_key/runtime_key/
// texture_key can still legitimately have different starting bindings (e.g.
// the same fragment shader reused after two different vertex shaders that
// consume a different number of descriptor slots, or — critically — a real
// draw colliding with a speculative pre-cache entry, which always assumes a
// starting state of all-zero). Folding this in turns that collision into a
// clean cache miss instead of silently serving structurally incompatible
// SPIR-V.
// ---------------------------------------------------------------------------
u64 ComputeBindingKey(const Shader::Backend::Bindings& starting_binding);


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

    // The result of a cache hit: the SPIR-V words and the Bindings state that EmitSPIRV
    // left behind after consuming that stage's descriptor slots.  The caller must apply
    // end_binding to the running Bindings accumulator so subsequent stages start at the
    // correct slot — exactly as if EmitSPIRV had been called live.
    //
    // spirv is shared (not copied) so that Lookup() and Save()'s internal snapshot can
    // both reference the same underlying buffer without a deep copy on every access.
    struct LookupResult {
        std::shared_ptr<const std::vector<u32>> spirv;
        Shader::Backend::Bindings end_binding;
    };

    // Look up a pre-translated SPIR-V program.
    // Returns nullopt on miss.
    [[nodiscard]] std::optional<LookupResult> Lookup(const SpirvKey& key) const;

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
    // end_binding is the Bindings state immediately after EmitSPIRV returned for this stage.
    void Insert(const SpirvKey& key, std::vector<u32> spirv,
                const Shader::Backend::Bindings& end_binding);
    void Insert(u64 unique_hash, const std::unordered_map<u64, u32>& cbuf_values,
                u64 runtime_key, u64 texture_key, std::vector<u32> spirv,
                const Shader::Backend::Bindings& end_binding);

    // Insert a speculative/AOT entry (cbuf_key = 0).
    // Speculative entries store a zero end_binding — they are only valid for the prewarmer
    // path which accumulates bindings correctly across all stages itself.
    void InsertSpeculative(u64 unique_hash, u64 runtime_key, u64 texture_key, std::vector<u32> spirv);

private:
    mutable std::shared_mutex mutex_;
    struct Entry {
        // shared_ptr so Save()'s snapshot phase (which copies every Entry while
        // holding the lock) is O(entry count) — an atomic refcount bump per
        // entry — rather than O(total SPIR-V bytes in the cache). Without this,
        // the snapshot copy's lock-hold duration grows with cache size and
        // blocks the synchronous, frame-blocking CreateGraphicsPipeline() path
        // (which calls Lookup()/Insert() on every pipeline compile) for
        // increasingly long stretches as the cache fills up over a session.
        std::shared_ptr<const std::vector<u32>> spirv;
        Shader::Backend::Bindings end_binding; // binding state after EmitSPIRV for this stage
    };
    std::unordered_map<SpirvKey, Entry, SpirvKeyHash> entries_;
    mutable bool dirty_{false};
    mutable std::atomic<size_t> hit_count_{0};
    mutable std::atomic<size_t> lookup_count_{0};

    // Throttle state for SaveThrottled — updated under mutex_.
    mutable size_t saved_entry_count_{0};
    mutable std::chrono::steady_clock::time_point last_save_time_{};
};

} // namespace VideoCommon
