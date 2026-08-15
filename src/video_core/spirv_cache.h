// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
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

// Diagnostic-only counterfactual — see the extensive doc comment on its definition in
// spirv_cache.cpp for what this tests and why it's structurally safe to compute without
// being anywhere close to a decision to actually narrow cbuf_key.
u64 ComputeCbufKeyExcludingTextureHandles(const std::unordered_map<u64, u32>& cbuf_values,
                                          const std::unordered_set<u64>& texture_handle_keys);

// ---------------------------------------------------------------------------
// ComputeTextureKey — hash of texture types and formats used during translation.
// ---------------------------------------------------------------------------
u64 ComputeTextureKey(const std::unordered_map<u32, Shader::TextureType>& texture_types,
                      const std::unordered_map<u32, Shader::TexturePixelFormat>& texture_pixel_formats);

// Phase 4 narrow prototype's texture_key fix. Same shape as
// ComputeCbufKeyExcludingTextureHandles above: excludes entries whose key (the raw resolved
// texture handle, per GraphicsEnvironment::ReadTextureType's texture_types.emplace(handle,
// result) — NOT the (cbuf_index, cbuf_offset) coordinates, which this map doesn't carry) is in
// excluded_handles, before hashing. Only filters texture_types, not texture_pixel_formats —
// this prototype's canonicalization (texture_pass.cpp) only ever touches
// ImageQueryDimensions's resolved TYPE, never pixel format, so there's nothing to exclude
// there. Falls back to plain ComputeTextureKey when excluded_handles is empty, same as the
// cbuf_key version does.
u64 ComputeTextureKeyExcludingHandles(
   const std::unordered_map<u32, Shader::TextureType>& texture_types,
   const std::unordered_map<u32, Shader::TexturePixelFormat>& texture_pixel_formats,
   const std::unordered_set<u32>& excluded_handles);

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
// ComputeWorkgroupKey — hashes the two compute-only quantities that get baked
// directly into compute SPIR-V as literals (NOT specialization constants):
//   - workgroup_size -> spv::ExecutionMode::LocalSize (emit_spirv.cpp)
//   - shared_memory_size -> the declared length of the shared-memory array
//     (spirv_emit_context.cpp, DivCeil(shared_memory_size, element_size))
// Both are read straight off ComputePipelineCacheKey/ComputeEnvironment,
// completely independent of unique_hash/cbuf_key/texture_key. Before this
// existed, CreateComputePipeline() hardcoded runtime_key=0 for every compute
// entry — meaning two dispatches of the exact same compute program with
// different workgroup/shared-memory setups (a normal thing for e.g. a
// generically-parameterized compute kernel reused at different dispatch
// sizes) could return each other's cached SPIR-V whenever cbuf_key and
// texture_key happened to coincide, silently baking in the WRONG LocalSize /
// shared-memory extent for that dispatch. That's a correctness bug, not a
// cache-efficiency one — this key closes it the same way ComputeBindingKey
// closes the equivalent gap on the graphics side.
// ---------------------------------------------------------------------------
u64 ComputeWorkgroupKey(u32 shared_memory_size, const std::array<u32, 3>& workgroup_size);

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
//   u32       version (see SPIRV_CACHE_VERSION and its version-history comment
//             in the .cpp for the current value and why each bump happened; a
//             mismatch discards the whole file rather than trying to
//             interpret entries in a layout they weren't written in — this
//             comment deliberately doesn't restate the number so it can't go
//             stale the way it just did, having still said "currently 6"
//             through both the v7 and v8 bumps)
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
//     u8      is_speculative (0 or 1 — added in v6; see Entry::is_speculative's
//             doc comment for why this needed to stop being dropped on reload)
// Entry::diag_base_runtime_hash / diag_binding_key are NOT part of this
// format — every entry loaded from disk comes back with both at 0
// regardless of what they originally were (see their own doc comment on
// Entry, and on Insert(), for why that's an acceptable, deliberate gap
// rather than an oversight to match).
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
    // diag_base_runtime_hash / diag_binding_key: the REQUESTED side's pre-fold
    // runtime_key components (see the Insert() doc comment above for what these are).
    // Only used, on a miss, to enrich the field-mismatch diagnostic — comparing them
    // against whatever pre-fold components are stored on the cached entry(ies) for this
    // unique_hash tells you whether a runtime_key mismatch is coming from the core
    // RuntimeInfo state or the binding-offset fold, which the folded runtime_key alone
    // can't distinguish. Passing 0/0 (the default) just means "no diagnostic data
    // available" — the cbuf/runtime/texture-level comparison this function already did
    // still happens unchanged.
    // diag_cbuf_key_excl_texture_handles: the REQUESTED side's counterfactual cbuf_key
    // (see ComputeCbufKeyExcludingTextureHandles in spirv_cache.cpp). 0 means either
    // "genuinely no non-texture-handle cbuf dependency" or "caller had no tagging data
    // to compute this" — Lookup() can't tell those apart from the value alone, which is
    // exactly why CbufTextureHandleNarrowingEligibleCount() exists as an explicit
    // denominator instead of inferring eligibility from this being nonzero.
    [[nodiscard]] std::optional<LookupResult> Lookup(const SpirvKey& key,
                                                       bool has_real_specialization_context,
                                                       u64 diag_base_runtime_hash = 0,
                                                       u64 diag_binding_key = 0,
                                                       u64 diag_cbuf_key_excl_texture_handles = 0) const;

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

    // Of StaleMissWithContextCount() (i.e. real, specialization-context-having stale
    // misses only — the case that actually matters for live-play stutter), how many
    // could not have matched ANY stored key for that unique_hash on the binding-offset
    // component alone / on the core-RuntimeInfo component alone. "Could not have
    // matched on X" means X differed from every stored variant, so even a
    // hypothetically-fixed guess on every OTHER field still wouldn't have produced a
    // hit — this is what actually separates "the binding fold is the dominant blocker"
    // from "the core RuntimeInfo state is the dominant blocker" (see spirv_cache.cpp's
    // Lookup() for exactly how these are computed). A shader can count toward both if
    // both components differ from every stored variant, so these are not mutually
    // exclusive and won't sum to StaleMissWithContextCount().
    [[nodiscard]] size_t RuntimeBindingComponentNeverMatchedCount() const noexcept {
        return runtime_binding_component_never_matched_with_context_.load();
    }
    [[nodiscard]] size_t RuntimeCoreComponentNeverMatchedCount() const noexcept {
        return runtime_core_component_never_matched_with_context_.load();
    }

    // Of the real-context stale misses where cbuf_key SPECIFICALLY differed from every
    // stored variant (i.e. cbuf_key, not runtime_key or texture_key, is why none of
    // them matched — see Lookup() for the exact gating), how many WOULD have matched at
    // least one stored variant on cbuf_key if the texture-handle-excluding counterfactual
    // (ComputeCbufKeyExcludingTextureHandles) had been used instead. This is the direct,
    // real-gameplay answer to "would narrowing cbuf_key to exclude texture-handle-only
    // reads actually help" — see spirv_cache.cpp's long comment on
    // ComputeCbufKeyExcludingTextureHandles for the full hypothesis and its safety
    // reasoning. Only counts misses where the comparison is meaningful (requester
    // supplied a non-zero diag_cbuf_key_excl_texture_handles, i.e. came from a live
    // GenericEnvironment that actually had tagging data — see the plumbing note at each
    // call site for why FileEnvironment/scanner-sourced comparisons don't count here).
    [[nodiscard]] size_t CbufTextureHandleNarrowingWouldHaveMatchedCount() const noexcept {
        return cbuf_texture_handle_narrowing_would_have_matched_.load();
    }
    // Denominator for the count above: real-context stale misses where cbuf_key
    // differed from every stored variant AND the requester supplied real tagging data
    // (so the counterfactual comparison was actually meaningful, not a 0-vs-0 no-op).
    [[nodiscard]] size_t CbufTextureHandleNarrowingEligibleCount() const noexcept {
        return cbuf_texture_handle_narrowing_eligible_.load();
    }

    // Resets the field-mismatch-log throttle (see kMaxFieldMismatchLogs in the .cpp) so
    // per-field diagnostic samples resume being logged. Call this once, right after the
    // boot-time disk-cache replay in LoadDiskResources() finishes — that replay alone
    // can burn through the entire budget within milliseconds on the worker pool (dozens
    // of no-context misses at once), which was silently starving live-play samples, the
    // case this diagnostic actually exists to see. Does NOT reset the underlying
    // counters (StaleMissWithContextCount(), RuntimeBindingComponentNeverMatchedCount(),
    // etc.) — only the LOG_INFO sample budget; the counters are cheap atomics meant to
    // keep counting for the whole session regardless.
    void ResetFieldMismatchLogBudget() const noexcept {
        field_mismatch_logs_no_context_ = 0;
        field_mismatch_logs_with_context_ = 0;
    }

    [[nodiscard]] size_t Size() const;

    // Insert a real runtime-compiled entry (keyed with actual cbuf values and runtime info).
    // end_binding is the Bindings state immediately after EmitSPIRV returned for this stage.
    // is_speculative is set internally by InsertSpeculative() — it exists purely so
    // SpeculativeInsertCount()/RealInsertCount() can attribute correctly; it does not
    // otherwise change behavior. Existing callers are unaffected by the default.
    //
    // diag_base_runtime_hash / diag_binding_key: the two PRE-FOLD components that were
    // XORed together (via FoldViewportTransformState / FoldBindingKey) to produce
    // key.runtime_key — i.e. the raw Shader::RuntimeInfo::Hash() (plus, for VertexB,
    // the viewport-transform-state fold) and the raw ComputeBindingKey(starting_binding)
    // value, respectively. runtime_key itself is opaque once folded, so these are stored
    // separately purely so Lookup()'s field-mismatch diagnostic can report which of the
    // two component sources is actually responsible for a runtime_key mismatch, instead
    // of only ever seeing that "runtime differs" with no way to tell whether that's the
    // fixed-function RuntimeInfo state or the starting descriptor-binding offset. NOT
    // part of SpirvKey, NOT persisted to disk (reloaded entries read back as 0/0 — see
    // Entry::diag_base_runtime_hash's own comment) — diagnostic-only, never consulted
    // for correctness. Defaulting to 0 keeps existing callers source-compatible; callers
    // that care about the diagnostic should pass the real pre-fold values they already
    // computed to build key.runtime_key in the first place.
    //
    // diag_cbuf_key_excl_texture_handles: what key.cbuf_key would be if entries only
    // ever read to resolve a bindless texture handle were excluded — see
    // ComputeCbufKeyExcludingTextureHandles's doc comment in spirv_cache.cpp. Same
    // diagnostic-only, non-persisted, safe-default-0 treatment as the two fields above.
    void Insert(const SpirvKey& key, std::vector<u32> spirv,
                const Shader::Backend::Bindings& end_binding, bool is_speculative = false,
                u64 diag_base_runtime_hash = 0, u64 diag_binding_key = 0,
                u64 diag_cbuf_key_excl_texture_handles = 0);
    void Insert(u64 unique_hash, const std::unordered_map<u64, u32>& cbuf_values,
                u64 runtime_key, u64 texture_key, std::vector<u32> spirv,
                const Shader::Backend::Bindings& end_binding,
                u64 diag_base_runtime_hash = 0, u64 diag_binding_key = 0,
                u64 diag_cbuf_key_excl_texture_handles = 0);

    // Insert a speculative/AOT entry (cbuf_key = 0).
    // Speculative entries store a zero end_binding — they are only valid for the prewarmer
    // path which accumulates bindings correctly across all stages itself.
    // diag_base_runtime_hash / diag_binding_key: see the matching Insert() doc comment
    // above — the same pre-fold components, but for the GUESSED values a speculative
    // translation used (diag_binding_key is always ComputeBindingKey(Bindings{}) — a
    // fresh, all-zero starting state — since speculative translation has no real
    // pipeline context to know a non-zero starting offset; see FoldBindingKey's doc
    // comment in this file for why that's only ever correct for a leading stage).
    // diag_cbuf_key_excl_texture_handles is not threaded through here: speculative
    // entries already use cbuf_key=0 (empty cbuf_values) unconditionally, so the
    // counterfactual is always 0 too — nothing to narrow on the guessed side.
    void InsertSpeculative(u64 unique_hash, u64 runtime_key, u64 texture_key,
                           std::vector<u32> spirv, u64 diag_base_runtime_hash = 0,
                           u64 diag_binding_key = 0);

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
        // The two pre-fold components that were combined (via FoldViewportTransformState /
        // FoldBindingKey) to produce this entry's key.runtime_key — see the Insert() doc
        // comments in this header for the full rationale. Diagnostic-only: not part of
        // SpirvKey (so they don't affect lookup/equality/hashing at all), and — UNLIKE
        // is_speculative just above (persisted since v6) — deliberately still NOT
        // persisted to spirv_cache.bin: an entry reloaded from disk always reads back
        // 0/0 here regardless of what it originally was. This one stays session-scoped
        // on purpose rather than growing the on-disk format further, since nothing
        // currently needs the pre-fold components to survive a restart — only whether
        // an entry is speculative does (see is_speculative's own comment: real vs.
        // guessed needs to keep meaning "real vs. guessed" across restarts, or any
        // future feature built on that distinction — e.g. deriving new guesses from the
        // observed distribution of real entries — would silently start treating old
        // guesses as ground truth the moment they survive one reload). Defaults to 0/0
        // for any Insert() call that doesn't pass real values (e.g. the disk-load path
        // in Load(), which has no pre-fold data to recover from the on-disk format at
        // all — the same reason it's not worth persisting these two in the first place).
        u64 diag_base_runtime_hash = 0;
        u64 diag_binding_key = 0;
        // Same non-persistence as the two fields above, but a DIFFERENT default
        // convention: every Insert() call site sets this to either the real
        // texture-handle-excluding counterfactual (when live tagging data was
        // available) or, when it wasn't (FileEnvironment, the scanner, compute without
        // gen_env), falls back to that SAME entry's own real cbuf_key — never a bare 0.
        // That makes `diag_cbuf_key_excl_texture_handles == <this entry's key.cbuf_key>`
        // the exact test for "no narrowing data was available for this entry", and
        // keeps every comparison against it meaningful (worst case, comparing against
        // an un-narrowed real value is still a valid, if narrower, test) instead of
        // risking a spurious match against an arbitrary 0 default. See
        // ComputeCbufKeyExcludingTextureHandles's doc comment in spirv_cache.cpp.
        u64 diag_cbuf_key_excl_texture_handles = 0;
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
    // See RuntimeBindingComponentNeverMatchedCount() / RuntimeCoreComponentNeverMatchedCount()
    // above for what these mean. Uncapped (unlike the LOG_INFO samples below, these are
    // just atomic increments) so they reflect the WHOLE session, not just whatever fit
    // in the log throttle's budget — this is the number that actually answers "is it the
    // binding fold or the core RuntimeInfo state" per handoff_02's step 1. Only
    // incremented for has_real_specialization_context misses (stale_miss_with_context_),
    // since a no-context [FileEnvironment] miss has no comparable requested-side
    // pre-fold data to check against — it wasn't computed for FileEnvironment's forced-0
    // case, so diag_base_runtime_hash/diag_binding_key at that call site are just 0/0
    // (meaningless), not "the shader genuinely wants an all-zero state."
    mutable std::atomic<size_t> runtime_binding_component_never_matched_with_context_{0};
    mutable std::atomic<size_t> runtime_core_component_never_matched_with_context_{0};
    // See CbufTextureHandleNarrowingEligibleCount() / ...WouldHaveMatchedCount() above
    // for what these mean and exactly how they're gated. Same uncapped-atomic treatment
    // as the pair above — whole-session totals, not throttle-limited samples, since
    // that's what actually answers "would narrowing cbuf_key help" rather than
    // whatever fraction of misses happened to get a log slot.
    mutable std::atomic<size_t> cbuf_texture_handle_narrowing_eligible_{0};
    mutable std::atomic<size_t> cbuf_texture_handle_narrowing_would_have_matched_{0};
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

    // Frequency of key.cbuf_key == 0 among REAL (has_real_specialization_context
    // == true) Lookup() calls only — i.e. actual gameplay draws, never
    // speculative/scanner lookups themselves. This is the number that settles
    // whether the speculative-entry approach has a structural ceiling: every
    // speculative entry is inserted with cbuf_key hardcoded to 0 (nothing
    // captures real cbuf content without a live draw), so it can only ever be
    // hit by a real draw whose OWN cbuf_key also happens to be 0. If this
    // stays low across real play, that's the ceiling, independent of how
    // correct the rest of the key is — see SubmitSpeculativeShader's
    // ReadCbufValue() doc comment for why cbuf can't be guessed the way
    // viewport_transform_state could.
    mutable std::atomic<size_t> real_cbuf_zero_count_{0};
    mutable std::atomic<size_t> real_cbuf_nonzero_count_{0};

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
