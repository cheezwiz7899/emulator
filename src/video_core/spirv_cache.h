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

#include <ankerl/unordered_dense.h>

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
// ComputeBindingKey — hashes the full starting Bindings state (the
// accumulator as it stood immediately before EmitSPIRV ran for this stage).
// v4 briefly narrowed this to a single leading/non-leading bit, on the
// theory that the only real risk was a real non-leading-stage draw
// colliding with a speculative entry (which always assumes an all-zero
// start). That was wrong: two different *real* non-leading-stage
// shaders/pipelines can and do collide under a single-bit scheme (both map
// to bit=1 regardless of their actual descriptor offsets), which reintroduced
// the "descriptor used by shader but not declared in the pipeline layout" /
// Ultrahand-style breakage this key exists to prevent. v5 reverted to a full
// hash of the state — see SPIRV_CACHE_VERSION's comment in the .cpp.
// ---------------------------------------------------------------------------
u64 ComputeBindingKey(const Shader::Backend::Bindings& starting_binding);

// ---------------------------------------------------------------------------
// FoldViewportTransformState / FoldBindingKey — the exact boost::hash_combine
// -style XOR fold CreateGraphicsPipeline()/CreateComputePipeline() apply to
// runtime_key before ever calling Lookup()/Insert(). Speculative insertion
// (InsertSpeculative(), called from both PipelineCache::SubmitSpeculativeShader
// and the ROM pre-cache scanner in citron/main.cpp) MUST apply the identical
// fold to the values it's guessing, or its runtime_key is in a different,
// incompatible format from every real entry's and can never match one no
// matter how accurate the guess is otherwise — this is what one of the two
// InsertSpeculative() call sites was silently skipping. Centralized here
// instead of duplicated inline at each call site specifically because that
// duplication is exactly how it drifted out of sync in the first place.
// ---------------------------------------------------------------------------
u64 FoldViewportTransformState(u64 runtime_key, u64 viewport_transform_state);
u64 FoldBindingKey(u64 runtime_key, u64 binding_key);


// ---------------------------------------------------------------------------
// SpirvCache — thread-safe store of pre-translated SPIR-V programs.
//
// On-disk format (spirv_cache.bin):
//   8 bytes   magic "citrspv\0"
//   u32       version (currently 5 — see SPIRV_CACHE_VERSION in the .cpp; a
//             mismatch discards the whole file rather than trying to
//             interpret entries in a layout they weren't written in)
//   u32       num_entries
//   per entry:
//     u64     unique_hash
//     u64     cbuf_key
//     u64     runtime_key
//     u64     texture_key
//     u32     word_count
//     u32[]   spirv words (word_count × 4 bytes)
//     <raw>   end_binding — Shader::Backend::Bindings written as a raw
//             struct (sizeof(end_binding) bytes; see Save()/Load()), not
//             field-by-field, so its exact size follows that struct's
//             layout rather than being spelled out here
// Entry::is_speculative is NOT part of this format — every entry loaded
// from disk comes back with is_speculative=false regardless of how it was
// originally inserted (see Entry::is_speculative's own doc comment).
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
        // See Entry::is_speculative — carried through so the caller can log/act on it.
        bool is_speculative = false;
    };

    // Look up a pre-translated SPIR-V program.
    // Returns nullopt on miss.
    // has_real_specialization_context: true if the caller's cbuf_key/texture_key
    // were computed from real captured data (gen_env_stage/gen_env != nullptr),
    // false if they were forced to 0 because the environment was FileEnvironment
    // (citron's older disk-cache-load path, which doesn't carry cbuf/texture
    // capture data). Used only to split the stale-miss diagnostic counters
    // precisely instead of guessing from cbuf_key==0/texture_key==0 alone, since
    // those can ALSO be the legitimate value for a shader with no specialization
    // or no textures — ambiguous without this explicit flag from the caller.
    [[nodiscard]] std::optional<LookupResult> Lookup(const SpirvKey& key,
                                                       bool has_real_specialization_context) const;

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
    /// Number of misses where the shader's unique_hash existed under a different key.
    [[nodiscard]] size_t MissWithHashPresentCount() const noexcept { return miss_with_hash_present_count_.load(); }
    /// Of those, how many had no real cbuf/texture context (FileEnvironment) vs. real context that still mismatched.
    [[nodiscard]] size_t StaleMissNoContextCount() const noexcept { return stale_miss_no_context_.load(); }
    [[nodiscard]] size_t StaleMissWithContextCount() const noexcept { return stale_miss_with_context_.load(); }
    /// Number of entries inserted via InsertSpeculative() since construction.
    [[nodiscard]] size_t SpeculativeInsertCount() const noexcept { return speculative_insert_count_.load(); }
    /// Number of entries inserted via the real (non-speculative) Insert() overloads since construction.
    [[nodiscard]] size_t RealInsertCount() const noexcept { return real_insert_count_.load(); }

    [[nodiscard]] size_t Size() const;

    // Insert a real runtime-compiled entry (keyed with actual cbuf values and runtime info).
    // end_binding is the Bindings state immediately after EmitSPIRV returned for this stage.
    // is_speculative is set internally by InsertSpeculative() — it exists purely so
    // SpeculativeInsertCount()/RealInsertCount() can attribute correctly; it does not
    // otherwise change behavior. Existing callers are unaffected by the default.
    void Insert(const SpirvKey& key, std::vector<u32> spirv,
                const Shader::Backend::Bindings& end_binding, bool is_speculative = false);
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
        // True if this entry came from InsertSpeculative() (live OnNewShaderSeen guess or
        // the ROM pre-cache scanner) rather than a real, GenericEnvironment/FileEnvironment
        // -backed compile. Surfaced through LookupResult so a caller can tell, at the exact
        // point a hit is about to be used, whether it's serving a real capture or a guess —
        // load-bearing for diagnosing whether a specific bad hit came from the scanner.
        bool is_speculative = false;
    };
    ankerl::unordered_dense::map<SpirvKey, Entry, SpirvKeyHash> entries_;
    // Secondary index: which unique_hashes have at least one entry in entries_.
    // ContainsByUniqueHash() used to be an O(N) linear scan over entries_ —
    // called on every shader the ROM pre-cache scanner considers, AND on every
    // live shader translation via OnNewShaderSeen — which got slower over the
    // course of a play session as entries_ grew, rather than staying flat.
    // This set makes that check O(1). Kept in sync in the one place entries_
    // is actually mutated (the SpirvKey overload of Insert()) and rebuilt
    // during Load().
    ankerl::unordered_dense::set<u64> unique_hashes_;
    // Maps unique_hash -> every full SpirvKey currently stored for it. Lets a
    // stale miss (see miss_with_hash_present_count_) efficiently find what
    // the stored key(s) actually look like and report which specific
    // field(s) — cbuf_key, runtime_key, or texture_key — differ from what
    // the real draw asked for, instead of just knowing that something does.
    // Capped per-hash (kMaxStoredKeysPerHashForDiagnostics) purely so a
    // pathological hash can't grow this unboundedly; diagnostic-only, never
    // consulted for correctness.
    ankerl::unordered_dense::map<u64, std::vector<SpirvKey>> keys_by_hash_;
    mutable bool dirty_{false};
    mutable std::atomic<size_t> hit_count_{0};
    mutable std::atomic<size_t> lookup_count_{0};
    // Incremented when a Lookup() misses on the full SpirvKey (unique_hash +
    // cbuf_key + runtime_key + texture_key) but unique_hashes_ shows this
    // exact shader (unique_hash alone) IS present in the cache under some
    // OTHER combination of the remaining three fields. A high rate here
    // means speculative entries the pre-cache scanner inserted (keyed with
    // guessed cbuf/texture state, since it has no real draw context) are
    // sitting in the cache unused — the shader itself was correctly
    // translated, but real draws can't find it because their actual
    // cbuf_key/texture_key don't match what was guessed. Distinguishes that
    // from the more boring "the scan just didn't cover what was played"
    // explanation, which wouldn't show up here at all.
    mutable std::atomic<size_t> miss_with_hash_present_count_{0};
    // Splits miss_with_hash_present_count_ by has_real_specialization_context.
    // stale_miss_no_context_: the caller's cbuf_key/texture_key were forced to
    // 0 because gen_env_stage/gen_env was null (FileEnvironment — citron's
    // older disk-cache-load path, no capture data available at all).
    // stale_miss_with_context_: the caller had REAL captured cbuf/texture data
    // and it still didn't match what's stored. These two need different
    // fixes: the first can only be resolved by skipping the lookup (nothing
    // to guess correctly from) or extending FileEnvironment's serialization
    // format; the second means our speculative guessing itself (fixed
    // Color2D/A8B8G8R8_UNORM texture type, cbuf_key=0) is wrong often enough
    // in practice to be worth improving.
    mutable std::atomic<size_t> stale_miss_no_context_{0};
    mutable std::atomic<size_t> stale_miss_with_context_{0};
    // Throttle for the field-level mismatch log lines emitted from Lookup()
    // (see kMaxFieldMismatchLogs in the .cpp). Capped so a long play session
    // doesn't flood the log — this is meant to establish which field(s)
    // differ, which a handful of samples is enough to answer. Split by
    // has_real_specialization_context (no_context_ / with_context_) rather
    // than sharing one counter: the disk-cache replay at boot can throw
    // dozens of no-context misses within milliseconds on the worker pool,
    // which was silently consuming the *entire* shared budget before a
    // single real-context sample — the case that actually matters for
    // live-play stutter — ever got a slot.
    mutable std::atomic<size_t> field_mismatch_logs_no_context_{0};
    mutable std::atomic<size_t> field_mismatch_logs_with_context_{0};
    // Speculative vs. real insert counts — split by which Insert() overload
    // last touched a given entry is not tracked per-entry, but the totals
    // are useful to know whether the pre-cache scanner is contributing
    // anything at all, or whether every entry is coming from the live path.
    mutable std::atomic<size_t> speculative_insert_count_{0};
    mutable std::atomic<size_t> real_insert_count_{0};

    // Throttle state for SaveThrottled — updated under mutex_.
    mutable size_t saved_entry_count_{0};
    mutable std::chrono::steady_clock::time_point last_save_time_{};

    // Baselines for the periodic windowed hit-rate log emitted by
    // SaveThrottled(). A cumulative-since-boot hit rate blends the initial
    // disk-cache warm-up (usually near 100%) with live-play behavior
    // (which is what actually matters for judging stutter reduction), so it
    // can look healthy even while live hit rate has collapsed. These track
    // "since the last log line" instead.
    mutable size_t hit_count_at_last_log_{0};
    mutable size_t lookup_count_at_last_log_{0};
    mutable size_t miss_with_hash_present_at_last_log_{0};
    mutable size_t stale_miss_no_context_at_last_log_{0};
    mutable size_t stale_miss_with_context_at_last_log_{0};

};

} // namespace VideoCommon
