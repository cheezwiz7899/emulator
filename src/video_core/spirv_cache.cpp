// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <fstream>
#include <shared_mutex>
#include <thread>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/spirv_cache.h"

namespace {
constexpr std::array<char, 8> SPIRV_CACHE_MAGIC{'c', 'i', 't', 'r', 's', 'p', 'v', '\0'};
constexpr u32 SPIRV_CACHE_VERSION = 9; // v9: the Phase 4 slot table (which shaders get
                                        // texture_key's handle-exclusion treatment) switched from a fixed, compile-time
                                        // list to a runtime-published, adaptively-grown one
                                        // (Shader::ActivePhase4PrototypeSlots(), environment.h) that starts EMPTY on a
                                        // fresh profile or install now, rather than pre-loaded with the two
                                        // hand-found coordinates v8 shipped with. Same category of change as v8 below:
                                        // no on-disk format change, but which shaders get the exclusion treatment at
                                        // all now depends on what THIS game's session(s) have actually discovered so
                                        // far (persisted per-game, see phase4_prototype_slots_file.h) rather than a
                                        // fixed compile-time set every game shared — an old entry computed under v8's
                                        // fixed table isn't wrong, just silently unreachable if this session's
                                        // discovered table doesn't (yet, or ever) match it exactly, same fail-safe
                                        // story as every prior bump. Deliberately does NOT need its own bump for
                                        // every future coordinate the adaptive mechanism goes on to discover after
                                        // this point — texture_key already depends on IsPhase4PrototypeSlot, so a
                                        // newly-active coordinate naturally produces a different texture_key on its
                                        // own the moment it's learned, the same "old entry silently unreachable"
                                        // story arrived at automatically instead of through an explicit version
                                        // number — see ActivePhase4PrototypeSlots's own doc comment for the detailed
                                        // argument. This bump covers only the one-time mechanism change, fixed table
                                        // to adaptive table, not any particular slot or count of slots;
                                        // (v8: texture_key now excludes the one Phase-4
                                        // -prototype-marked slot's resolved handle for shaders that
                                        // have it (ComputeTextureKeyExcludingHandles,
                                        // vk_pipeline_cache.cpp's CreateGraphicsPipeline/
                                        // CreateComputePipeline — see the "Phase 4 narrow
                                        // prototype's texture_key fix" comments there). Same
                                        // category of change as v7's cbuf_key narrowing below,
                                        // not a coincidence: no on-disk byte format change, but
                                        // the VALUE stored under texture_key now means something
                                        // different, for the small set of shaders that touch that
                                        // one slot, than it did under v7 and earlier. Old entries
                                        // for those shaders aren't wrong under the new scheme,
                                        // just silently unreachable (a stale cross-version entry
                                        // just fails to match, same as any other cache miss) —
                                        // but left unbumped, that would muddy any future
                                        // texture_key-related diagnostics the same way mixing
                                        // cbuf_key schemes would have before v7's bump (see
                                        // handoff notes for this investigation if resuming the
                                        // thread on why that reasoning applies equally here).
                                        // Bumping forces a clean full rebuild instead, same
                                        // reasoning as v7's and v6's bundled bumps;
                                        // (v7: cbuf_key now excludes texture-handle-only
                                        // reads (ComputeCbufKeyExcludingTextureHandles,
                                        // vk_pipeline_cache.cpp's CreateGraphicsPipeline/
                                        // CreateComputePipeline — see the "PHASE 1 —
                                        // narrowing enabled" comments there for the
                                        // validation data). No on-disk byte format
                                        // change, but the VALUE stored under cbuf_key
                                        // now means something different than it did
                                        // under v6 and earlier — an old entry's cbuf_key
                                        // was computed over strictly more reads than a
                                        // v7 entry's would be for the identical shader,
                                        // so mixing them wouldn't be wrong (a stale
                                        // cross-version entry just fails to match,
                                        // same as any other cache miss) but would make
                                        // any future cbuf_key==0-rate diagnostics
                                        // ambiguous about which scheme produced a given
                                        // number. Bumping forces a clean full rebuild
                                        // instead, same reasoning as v6's bundled bump;
                                        // (v6: persists Entry::is_speculative per entry
                                        // instead of dropping it (every reloaded entry
                                        // used to silently become "real" after one
                                        // restart, which would have quietly corrupted
                                        // any future feature — e.g. scan-time guess
                                        // derivation — built on "is this entry backed by
                                        // an actual observed draw". Separately (no format
                                        // impact, but invalidates old entries just the
                                        // same, so bundled into this bump rather than
                                        // wasted on its own): graphics runtime_key now
                                        // folds RuntimeInfo::SpirvRelevantHash(stage)
                                        // instead of the whole-struct Hash(), and compute
                                        // runtime_key is ComputeWorkgroupKey(...) instead
                                        // of a hardcoded 0 — see both functions' doc
                                        // comments. Old entries aren't wrong under the
                                        // new scheme, just computed differently, so they
                                        // will not match, which is fine for a cache;
                                        // (v5: ComputeBindingKey reverted to a full-state hash
                                        // (see its doc comment for why the single-bit
                                        // narrowing in v4 was wrong — real-vs-real
                                        // collisions, not just real-vs-speculative, turned
                                        // out to be the dominant case); entries_ moved to
                                        // ankerl::unordered_dense::map and
                                        // ContainsByUniqueHash() gained an O(1) secondary
                                        // index instead of an O(N) linear scan
                                        // (v4: single-bit ComputeBindingKey; v3: runtime_key
                                        // folds in viewport_transform_state [VertexB] and
                                        // starting Bindings state; v2: adds Bindings
                                        // end_binding per entry)))
} // anonymous namespace

namespace VideoCommon {

u64 ComputeCbufKey(const std::unordered_map<u64, u32>& cbuf_values) {
    if (cbuf_values.empty()) return 0;
    std::vector<std::pair<u64,u32>> sorted(cbuf_values.begin(), cbuf_values.end());
    std::sort(sorted.begin(), sorted.end());
    return Common::CityHash64(reinterpret_cast<const char*>(sorted.data()),
                              sorted.size() * sizeof(sorted[0]));
}

// ---------------------------------------------------------------------------
// UPDATE (v7 / Phase 1): this IS now used by the real cache key —
// CreateGraphicsPipeline()/CreateComputePipeline() (vk_pipeline_cache.cpp) compute
// the real cbuf_key with this function directly, not ComputeCbufKey() above. The
// paragraph below describes why this function exists and was originally added as
// diagnostic-only; that history is still accurate, the "not used by any real cache
// key today" line specifically is not anymore and is left uncorrected inline below
// only so the reasoning that follows still reads coherently — treat this update
// note as the current status.
//
// Originally added purely to test, against real gameplay, a specific hypothesis
// about cbuf_key before anyone committed to narrowing it: most of what gets
// captured in cbuf_values (GenericEnvironment::ReadCbufValue) is read only to
// resolve a bindless texture handle (GetTextureHandle(), texture_pass.cpp) —
// whose downstream effect on codegen is ALREADY captured separately by
// texture_key (ComputeTextureKey, from ReadTextureType()/
// ReadTexturePixelFormat()'s own results). If that's true, two shaders that
// differ only in WHICH raw handle they read — but resolve to the SAME
// texture type/format, which is a real and plausible case, not an edge case
// — currently get DIFFERENT cbuf_key despite producing byte-identical
// SPIR-V, which is the same shape of over-invalidation
// RuntimeInfo::SpirvRelevantHash() already fixed on the runtime_key side.
//
// Computes what cbuf_key WOULD be if every entry tagged via
// ReadCbufValueForTextureHandle() (see its doc comment in environment.h for
// exactly what "tagged" means and why it's a structural guarantee, not an
// inference) were excluded from the hash. texture_handle_keys is expected to
// be GenericEnvironment::CapturedTextureHandleCbufKeys() — empty for a
// FileEnvironment or the scanner's SpeculativeShaderEnvironment today (see
// the plumbing note in Lookup()'s caller), which just means the
// counterfactual degrades to ComputeCbufKey() unchanged for those cases —
// never wrong, just uninformative for the ones that can't supply the tag yet.
//
// Also still used diagnostically: passed alongside the real cbuf_key it now
// computes so Lookup() can flag (currently unexplained — see the divergence
// log in Lookup()) any case where the two disagree, which per the call sites
// shouldn't be structurally possible anymore.
// ---------------------------------------------------------------------------
u64 ComputeCbufKeyExcludingTextureHandles(const std::unordered_map<u64, u32>& cbuf_values,
                                          const std::unordered_set<u64>& texture_handle_keys) {
    if (cbuf_values.empty() || texture_handle_keys.empty()) {
        return ComputeCbufKey(cbuf_values);
    }
    std::vector<std::pair<u64, u32>> sorted;
    sorted.reserve(cbuf_values.size());
    for (const auto& [key, value] : cbuf_values) {
        if (!texture_handle_keys.contains(key)) {
            sorted.emplace_back(key, value);
        }
    }
    if (sorted.empty()) {
        // Every captured read was texture-handle-purposed — the counterfactual key is
        // "no non-texture-handle cbuf dependency at all", same as an empty cbuf_values
        // map would hash to.
        return 0;
    }
    std::sort(sorted.begin(), sorted.end());
    return Common::CityHash64(reinterpret_cast<const char*>(sorted.data()),
                              sorted.size() * sizeof(sorted[0]));
}

u64 ComputeTextureKey(const std::unordered_map<u32, Shader::TextureType>& texture_types,
                      const std::unordered_map<u32, Shader::TexturePixelFormat>& texture_pixel_formats) {
    if (texture_types.empty() && texture_pixel_formats.empty()) return 0;
    std::vector<std::pair<u32, Shader::TextureType>> sorted_types(texture_types.begin(), texture_types.end());
    std::sort(sorted_types.begin(), sorted_types.end());
    std::vector<std::pair<u32, Shader::TexturePixelFormat>> sorted_formats(texture_pixel_formats.begin(), texture_pixel_formats.end());
    std::sort(sorted_formats.begin(), sorted_formats.end());
    u64 hash = 0;
    if (!sorted_types.empty()) {
        hash ^= Common::CityHash64(reinterpret_cast<const char*>(sorted_types.data()),
                                   sorted_types.size() * sizeof(sorted_types[0]));
    }
    if (!sorted_formats.empty()) {
        u64 format_hash = Common::CityHash64(reinterpret_cast<const char*>(sorted_formats.data()),
                                             sorted_formats.size() * sizeof(sorted_formats[0]));
        hash ^= format_hash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
}

u64 ComputeTextureKeyExcludingHandles(
   const std::unordered_map<u32, Shader::TextureType>& texture_types,
   const std::unordered_map<u32, Shader::TexturePixelFormat>& texture_pixel_formats,
   const std::unordered_set<u32>& excluded_handles) {
    if (texture_types.empty() || excluded_handles.empty()) {
        return ComputeTextureKey(texture_types, texture_pixel_formats);
    }
    std::unordered_map<u32, Shader::TextureType> filtered_types;
    filtered_types.reserve(texture_types.size());
    for (const auto& [handle, type] : texture_types) {
        if (!excluded_handles.contains(handle)) {
            filtered_types.emplace(handle, type);
        }
    }
    // texture_pixel_formats passed through unfiltered -- see this function's doc comment in
    // spirv_cache.h for why (this prototype never touches pixel-format resolution).
    return ComputeTextureKey(filtered_types, texture_pixel_formats);
}

u64 ComputeBindingKey(const Shader::Backend::Bindings& starting_binding) {
    // NOTE: this was briefly narrowed to a single leading/non-leading bit,
    // on the theory that the only real risk was a real non-leading-stage
    // draw colliding with a speculative entry (which always assumes an
    // all-zero start). That turned out to be wrong in practice: the ROM
    // pre-cache scanner was found to be inserting ~0 speculative entries
    // (separate, still-open bug — see the scanner investigation), so almost
    // everything in the cache during real play comes from the live path,
    // not speculative entries. Two different *real* non-leading-stage
    // shaders/pipelines can and do collide under the single-bit scheme
    // (both map to bit=1 regardless of their actual offsets), and that
    // reintroduced the exact "descriptor used by shader but not declared in
    // the pipeline layout" / Ultrahand breakage this key exists to prevent.
    // Back to hashing the full starting state until the scanner is fixed
    // and there's real hit/miss data (Tracy) to make an informed tradeoff
    // between correctness and pre-cache hit rate.
    const u32 fields[] = {
        starting_binding.unified,       starting_binding.uniform_buffer,
        starting_binding.storage_buffer, starting_binding.texture,
        starting_binding.image,         starting_binding.texture_scaling_index,
        starting_binding.image_scaling_index,
    };
    return Common::CityHash64(reinterpret_cast<const char*>(fields), sizeof(fields));
}

u64 ComputeWorkgroupKey(u32 shared_memory_size, const std::array<u32, 3>& workgroup_size) {
    const u32 fields[] = {shared_memory_size, workgroup_size[0], workgroup_size[1],
                          workgroup_size[2]};
    return Common::CityHash64(reinterpret_cast<const char*>(fields), sizeof(fields));
}

u64 FoldViewportTransformState(u64 runtime_key, u64 viewport_transform_state) {
    return runtime_key ^ (viewport_transform_state + 0x9e3779b97f4a7c15ULL +
                          (runtime_key << 6) + (runtime_key >> 2));
}

u64 FoldBindingKey(u64 runtime_key, u64 binding_key) {
    return runtime_key ^ (binding_key + 0x9e3779b97f4a7c15ULL +
                          (runtime_key << 6) + (runtime_key >> 2));
}


void SpirvCache::Load(const std::filesystem::path& path) {
    std::unique_lock lock{mutex_};
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return;
        file.exceptions(std::ifstream::failbit);

        std::array<char, 8> magic{};
        u32 version{};
        file.read(magic.data(), 8);
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != SPIRV_CACHE_MAGIC || version != SPIRV_CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "SPIR-V cache version mismatch, discarding");
            return;
        }

        u32 num_entries{};
        file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
        entries_.reserve(num_entries);
        unique_hashes_.reserve(num_entries);

        for (u32 i = 0; i < num_entries; ++i) {
            SpirvKey key{};
            u32 word_count{};
            file.read(reinterpret_cast<char*>(&key.unique_hash), sizeof(key.unique_hash));
            file.read(reinterpret_cast<char*>(&key.cbuf_key),    sizeof(key.cbuf_key));
            file.read(reinterpret_cast<char*>(&key.runtime_key), sizeof(key.runtime_key));
            file.read(reinterpret_cast<char*>(&key.texture_key), sizeof(key.texture_key));
            file.read(reinterpret_cast<char*>(&word_count),       sizeof(word_count));

            if (word_count == 0 || word_count > 4 * 1024 * 1024) {
                LOG_WARNING(Render_Vulkan, "Bogus SPIR-V word count {}, stopping load", word_count);
                break;
            }
            if (key.unique_hash == 0) {
                file.seekg(word_count * sizeof(u32), std::ios::cur);
                continue;
            }
            std::vector<u32> spirv(word_count);
            file.read(reinterpret_cast<char*>(spirv.data()), word_count * sizeof(u32));
            Shader::Backend::Bindings end_binding{};
            file.read(reinterpret_cast<char*>(&end_binding), sizeof(end_binding));
            u8 is_speculative_byte{};
            file.read(reinterpret_cast<char*>(&is_speculative_byte), sizeof(is_speculative_byte));
            // Trailing three diag_* fields: diag_base_runtime_hash/diag_binding_key stay
            // at their documented 0/0 "no data" default (unaffected by this load). The
            // third, diag_cbuf_key_excl_texture_handles, deliberately defaults to
            // key.cbuf_key (this entry's own real value) rather than a bare 0 — see its
            // doc comment on Entry: "diag_cbuf_key_excl_texture_handles ==
            // <this entry's key.cbuf_key>" is the documented signal for "no narrowing
            // data available", and a reloaded entry never has any (the tagging that
            // would produce a real value is session-scoped, not persisted). Defaulting
            // to a bare 0 here would have been silently wrong for the common case of a
            // reloaded entry with a genuinely nonzero cbuf_key.
            entries_.emplace(key, Entry{std::make_shared<const std::vector<u32>>(std::move(spirv)),
                                        end_binding, is_speculative_byte != 0, 0, 0, key.cbuf_key});
            unique_hashes_.insert(key.unique_hash);
            constexpr size_t kMaxStoredKeysPerHashForDiagnosticsLoad = 8;
            auto& stored_keys = keys_by_hash_[key.unique_hash];
            if (stored_keys.size() < kMaxStoredKeysPerHashForDiagnosticsLoad) {
                stored_keys.push_back(key);
            }
        }
        LOG_INFO(Render_Vulkan, "Loaded {} SPIR-V cache entries", entries_.size());
        saved_entry_count_ = entries_.size();
        last_save_time_    = std::chrono::steady_clock::now();
    } catch (...) {
        LOG_WARNING(Render_Vulkan, "Failed to load SPIR-V cache (corrupt or truncated)");
    }
}

void SpirvCache::Save(const std::filesystem::path& path) const {
    // Phase 1: snapshot under exclusive lock.
    std::vector<std::pair<SpirvKey, Entry>> snapshot;
    {
        std::unique_lock lock{mutex_};
        if (!dirty_) return;
        snapshot.assign(entries_.begin(), entries_.end());
        dirty_ = false;
        saved_entry_count_ = entries_.size();
    }
    // Phase 2: write to disk without holding the mutex.
    const auto tmp_path = path.parent_path() / (path.filename().string() + ".tmp");
    try {
        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            LOG_ERROR(Render_Vulkan, "Failed to open SPIR-V cache for writing");
            std::unique_lock lock{mutex_};
            dirty_ = true; // restore so next Save() retries
            return;
        }
        file.exceptions(std::ofstream::failbit);

        const u32 num_entries = static_cast<u32>(snapshot.size());
        file.write(SPIRV_CACHE_MAGIC.data(), SPIRV_CACHE_MAGIC.size());
        file.write(reinterpret_cast<const char*>(&SPIRV_CACHE_VERSION), sizeof(SPIRV_CACHE_VERSION));
        file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

        for (const auto& [key, entry] : snapshot) {
            const u32 word_count = static_cast<u32>(entry.spirv->size());
            file.write(reinterpret_cast<const char*>(&key.unique_hash), sizeof(key.unique_hash));
            file.write(reinterpret_cast<const char*>(&key.cbuf_key),    sizeof(key.cbuf_key));
            file.write(reinterpret_cast<const char*>(&key.runtime_key), sizeof(key.runtime_key));
            file.write(reinterpret_cast<const char*>(&key.texture_key), sizeof(key.texture_key));
            file.write(reinterpret_cast<const char*>(&word_count),      sizeof(word_count));
            file.write(reinterpret_cast<const char*>(entry.spirv->data()), word_count * sizeof(u32));
            file.write(reinterpret_cast<const char*>(&entry.end_binding), sizeof(entry.end_binding));
            const u8 is_speculative_byte = entry.is_speculative ? 1 : 0;
            file.write(reinterpret_cast<const char*>(&is_speculative_byte), sizeof(is_speculative_byte));
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Failed to write SPIR-V cache: {}", e.what());
        std::unique_lock lock{mutex_};
        dirty_ = true; // restore so next Save() retries
        return;
    }

    std::error_code ec;
    // std::filesystem::rename is not atomic-replace on Windows when the destination
    // already exists — it fails with an error. Remove first, then rename.
    // This is the same pattern used throughout the rest of the citron codebase.
    std::filesystem::remove(path, ec); // ignore error — file may not exist yet
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        LOG_ERROR(Render_Vulkan, "Failed to rename SPIR-V cache: {}", ec.message());
        std::filesystem::remove(tmp_path, ec);
        std::unique_lock lock{mutex_};
        dirty_ = true; // restore so next Save() retries
        return;
    }
    LOG_INFO(Render_Vulkan, "Saved {} SPIR-V cache entries", snapshot.size());
}

void SpirvCache::SaveThrottled(const std::filesystem::path& path,
                                size_t min_new_entries,
                                std::chrono::seconds min_interval) const {
    size_t hits_now{}, probes_now{}, hits_at_last{}, probes_at_last{}, entries_now{};
    size_t stale_misses_now{}, stale_misses_at_last{};
    size_t no_ctx_now{}, no_ctx_at_last{}, with_ctx_now{}, with_ctx_at_last{};
    bool should_log = false;
    {
        // Exclusive lock: we read and conditionally write last_save_time_ / dirty_.
        std::unique_lock lock{mutex_};
        if (!dirty_) return;
        const size_t new_since_save = entries_.size() - saved_entry_count_;
        const auto   time_since     = std::chrono::steady_clock::now() - last_save_time_;
        if (new_since_save < min_new_entries && time_since < min_interval) return;
        // Update the time under lock so concurrent SaveThrottled calls queued on
        // serialization_thread don't all pass the threshold check at once.
        // Save() will set saved_entry_count_ when the snapshot is taken.
        last_save_time_ = std::chrono::steady_clock::now();

        // Snapshot windowed hit-rate counters under the same lock/cadence as the
        // disk save, so this doesn't need its own timer. hit_count_/lookup_count_
        // are atomics updated from Lookup() without holding mutex_, so reading them
        // here only needs to be "recent enough," not perfectly synchronized.
        hits_now      = hit_count_.load();
        probes_now    = lookup_count_.load();
        hits_at_last  = hit_count_at_last_log_;
        probes_at_last = lookup_count_at_last_log_;
        stale_misses_now     = miss_with_hash_present_count_.load();
        stale_misses_at_last = miss_with_hash_present_at_last_log_;
        no_ctx_now      = stale_miss_no_context_.load();
        no_ctx_at_last  = stale_miss_no_context_at_last_log_;
        with_ctx_now     = stale_miss_with_context_.load();
        with_ctx_at_last = stale_miss_with_context_at_last_log_;
        entries_now   = entries_.size();
        hit_count_at_last_log_    = hits_now;
        lookup_count_at_last_log_ = probes_now;
        miss_with_hash_present_at_last_log_ = stale_misses_now;
        stale_miss_no_context_at_last_log_   = no_ctx_now;
        stale_miss_with_context_at_last_log_ = with_ctx_now;
        should_log = true;
    }
    Save(path);
    if (should_log) {
        const size_t window_hits   = hits_now   - hits_at_last;
        const size_t window_probes = probes_now - probes_at_last;
        const size_t window_stale_misses = stale_misses_now - stale_misses_at_last;
        const size_t window_no_ctx = no_ctx_now - no_ctx_at_last;
        const size_t window_with_ctx = with_ctx_now - with_ctx_at_last;
        const size_t cbuf_zero = real_cbuf_zero_count_.load();
        const size_t cbuf_nonzero = real_cbuf_nonzero_count_.load();
        const size_t cbuf_total = cbuf_zero + cbuf_nonzero;
        const int cbuf_zero_pct =
            cbuf_total > 0 ? static_cast<int>(cbuf_zero * 100 / cbuf_total) : 0;
        // See RuntimeBindingComponentNeverMatchedCount() / RuntimeCoreComponentNeverMatchedCount()'s
        // doc comments in spirv_cache.h — these split the folded runtime_key mismatch
        // (stale_miss_with_context_) into its two pre-fold components so it's clear
        // whether the binding-offset fold or the core RuntimeInfo state is the
        // dominant blocker, instead of only ever seeing "runtime differs" with no way
        // to tell which. Cumulative since boot (not windowed), since a small sample can
        // be misleading here and the whole-session total is what actually settles the
        // question.
        const size_t binding_never_matched = runtime_binding_component_never_matched_with_context_.load();
        const size_t core_never_matched = runtime_core_component_never_matched_with_context_.load();
        const size_t with_ctx_total = with_ctx_now; // cumulative stale_miss_with_context_ since boot
        // The cbuf-narrowing eligible/would-have-matched line that used to print here is
        // gone along with the counters it read (see Lookup()'s comment where that logic
        // used to live) — narrowing shipped in production (v7) after two full sessions
        // validated it at ~24-28%, and the follow-up measurement had turned into a real
        // per-Lookup() cost without answering anything that was still blocking work.
        if (window_probes > 0) {
            const int window_pct = static_cast<int>(window_hits * 100 / window_probes);
            LOG_INFO(Render_Vulkan,
                     "SPIR-V cache: {} entries ({} speculative / {} real inserted so far) — "
                     "recent window {}/{} stage hits ({}%), {} misses where the shader was "
                     "already cached under a different key ({} had no real cbuf/texture "
                     "context [FileEnvironment] / {} had real context that still mismatched). "
                     "Real-draw cbuf_key==0 so far: {}/{} ({}%) — the ceiling on how many real "
                     "draws a speculative entry could ever match, regardless of key correctness. "
                     "Of {} real-context stale misses so far, {} could not have matched on the "
                     "binding-offset component alone (vs any stored variant) / {} could not have "
                     "matched on the core RuntimeInfo component alone — whichever is closer to "
                     "{} is the dominant runtime_key blocker.",
                     entries_now, speculative_insert_count_.load(), real_insert_count_.load(),
                     window_hits, window_probes, window_pct, window_stale_misses,
                     window_no_ctx, window_with_ctx, cbuf_zero, cbuf_total, cbuf_zero_pct,
                     with_ctx_total, binding_never_matched, core_never_matched, with_ctx_total);
        } else {
            LOG_INFO(Render_Vulkan,
                     "SPIR-V cache: {} entries ({} speculative / {} real inserted so far) — "
                     "no lookups since last log. Real-draw cbuf_key==0 so far: {}/{} ({}%). "
                     "Of {} real-context stale misses so far, {} binding-component-never-matched / "
                     "{} core-component-never-matched.",
                     entries_now, speculative_insert_count_.load(), real_insert_count_.load(),
                     cbuf_zero, cbuf_total, cbuf_zero_pct,
                     with_ctx_total, binding_never_matched, core_never_matched);
        }
    }
}

bool SpirvCache::Contains(const SpirvKey& key) const noexcept {
    std::shared_lock lock{mutex_};
    return entries_.count(key) != 0;
}

bool SpirvCache::ContainsByUniqueHash(u64 unique_hash) const noexcept {
    std::shared_lock lock{mutex_};
    return unique_hashes_.contains(unique_hash);
}

std::optional<SpirvCache::LookupResult> SpirvCache::Lookup(const SpirvKey& key,
                                                             bool has_real_specialization_context,
                                                             u64 diag_base_runtime_hash,
                                                             u64 diag_binding_key,
                                                             u64 diag_cbuf_key_excl_texture_handles) const {
    std::shared_lock lock{mutex_};
    ++lookup_count_;
    if (has_real_specialization_context) {
        if (key.cbuf_key == 0) {
            ++real_cbuf_zero_count_;
        } else {
            ++real_cbuf_nonzero_count_;
        }
    }
    const auto it = entries_.find(key);
    if (it == entries_.end()) {
        // Full-key miss. Was this exact shader (by unique_hash alone) already
        // translated and sitting in the cache under a DIFFERENT cbuf_key/
        // runtime_key/texture_key? unique_hashes_ already tracks presence by
        // unique_hash alone for ContainsByUniqueHash(), so this is a cheap,
        // already-available check — no extra bookkeeping needed beyond the
        // counter itself.
        if (unique_hashes_.contains(key.unique_hash)) {
            ++miss_with_hash_present_count_;
            if (has_real_specialization_context) {
                ++stale_miss_with_context_;
            } else {
                ++stale_miss_no_context_;
            }

            // Look up every stored key for this unique_hash once — used below both
            // for the (uncapped) binding-vs-core component counters and for the
            // (throttled) per-field LOG_INFO samples.
            const auto hit_it = keys_by_hash_.find(key.unique_hash);
            const bool have_stored_keys = hit_it != keys_by_hash_.end() && !hit_it->second.empty();

            // Uncapped: for every real (has_real_specialization_context) stale miss,
            // check whether the REQUESTED side's pre-fold runtime_key components
            // (diag_base_runtime_hash / diag_binding_key) match ANY stored variant's
            // components — not just the one variant that happened to be logged. If a
            // component differs from every single stored variant, that component
            // could not possibly have produced a hit here no matter what the other
            // fields were, which is exactly the "is it the binding fold or the core
            // RuntimeInfo state" question handoff_02 asks. This runs for the whole
            // session (not throttled like the LOG_INFO samples below) since it's just
            // a couple of atomic increments — cheap enough to never need a cap, and
            // this is the number that actually matters for the aggregate answer,
            // rather than relying on however many per-field log samples fit in a
            // throttle budget.
            if (has_real_specialization_context && have_stored_keys) {
                bool binding_matched_any = false;
                bool core_matched_any = false;
                for (const SpirvKey& stored : hit_it->second) {
                    const auto stored_it = entries_.find(stored);
                    if (stored_it == entries_.end()) {
                        continue; // shouldn't happen (keys_by_hash_ mirrors entries_), but be safe
                    }
                    if (stored_it->second.diag_binding_key == diag_binding_key) {
                        binding_matched_any = true;
                    }
                    if (stored_it->second.diag_base_runtime_hash == diag_base_runtime_hash) {
                        core_matched_any = true;
                    }
                }
                if (!binding_matched_any) {
                    ++runtime_binding_component_never_matched_with_context_;
                }
                if (!core_matched_any) {
                    ++runtime_core_component_never_matched_with_context_;
                }
                // The cbuf-narrowing eligible/would-have-matched comparison that used to
                // live here (plus a throttled divergence-value log) is gone — it answered
                // the question it existed for (whether narrowing cbuf_key was worth
                // shipping; two full play sessions said yes, ~24-28%) and had turned into
                // a real, measurable per-Lookup() cost for a follow-up question that
                // wasn't blocking anything else. diag_cbuf_key_excl_texture_handles is
                // now just an alias for the real cbuf_key at every call site (see
                // CreateGraphicsPipeline()/CreateComputePipeline() in
                // vk_pipeline_cache.cpp) — still threaded through Insert()/Lookup() so
                // those signatures don't need touching again, but no longer worth
                // comparing against anything here.
            }

            // Throttled: report exactly which field(s) differ from the
            // requested key, using whatever's actually stored for this hash.
            // A handful of samples is enough to tell whether cbuf_key,
            // runtime_key, or texture_key (or some combination) is the
            // dominant mismatch source, without flooding the log over a
            // whole play session. Bumped 30 -> 500 (was exhausting itself
            // within the first ~60s of a session, almost entirely on
            // boot-time disk-cache replay, before any live-play sample ever
            // got a slot) — see also ResetFieldMismatchLogBudget(), which
            // callers should invoke once boot-time replay finishes so
            // live-play gets its own fresh budget on top of this.
            constexpr size_t kMaxFieldMismatchLogs = 500;
            std::atomic<size_t>& field_mismatch_logs =
                has_real_specialization_context ? field_mismatch_logs_with_context_
                                                 : field_mismatch_logs_no_context_;
            size_t expected = field_mismatch_logs.load();
            bool got_slot = false;
            while (expected < kMaxFieldMismatchLogs &&
                   !field_mismatch_logs.compare_exchange_weak(expected, expected + 1)) {
            }
            got_slot = expected < kMaxFieldMismatchLogs;
            if (got_slot && have_stored_keys) {
                for (const SpirvKey& stored : hit_it->second) {
                    const auto stored_it = entries_.find(stored);
                    const u64 stored_base_rt =
                        stored_it != entries_.end() ? stored_it->second.diag_base_runtime_hash : 0;
                    const u64 stored_binding_key =
                        stored_it != entries_.end() ? stored_it->second.diag_binding_key : 0;
                    // Added alongside the Phase 3 guess-refinement work: nothing above
                    // this line could previously tell you whether a given mismatch
                    // sample was against a real entry from earlier in the same session
                    // (ordinary, expected cardinality — see
                    // RecordPhase3RuntimeVariantDiagnostic's histogram) or against a
                    // scanner/live-speculative guess actually missing. Without this,
                    // "base_runtime DIFFERS" here is ambiguous between "the guess was
                    // wrong" and "this shader legitimately has 2+ real states" — two
                    // completely different next steps depending on which.
                    const bool stored_is_speculative =
                        stored_it != entries_.end() && stored_it->second.is_speculative;
                    LOG_INFO(Render_Vulkan,
                             "SPIR-V cache field mismatch [{}] (real_context={}): requested "
                             "cbuf={:016x} runtime={:016x} (base_runtime={:016x} binding_key={:016x}) "
                             "texture={:016x} vs stored[speculative={}] cbuf={:016x} runtime={:016x} "
                             "(base_runtime={:016x} binding_key={:016x}) texture={:016x} "
                             "(cbuf {}, runtime {} [base_runtime {}, binding_key {}], texture {})",
                             expected, has_real_specialization_context,
                             key.cbuf_key, key.runtime_key, diag_base_runtime_hash, diag_binding_key,
                             key.texture_key, stored_is_speculative,
                             stored.cbuf_key, stored.runtime_key, stored_base_rt, stored_binding_key,
                             stored.texture_key,
                             key.cbuf_key == stored.cbuf_key ? "matches" : "DIFFERS",
                             key.runtime_key == stored.runtime_key ? "matches" : "DIFFERS",
                             diag_base_runtime_hash == stored_base_rt ? "matches" : "DIFFERS",
                             diag_binding_key == stored_binding_key ? "matches" : "DIFFERS",
                             key.texture_key == stored.texture_key ? "matches" : "DIFFERS");
                }
            }
        }
        return std::nullopt;
    }
    ++hit_count_;
    return LookupResult{it->second.spirv, it->second.end_binding, it->second.is_speculative};
}

void SpirvCache::Insert(const SpirvKey& key, std::vector<u32> spirv,
                        const Shader::Backend::Bindings& end_binding, bool is_speculative,
                        u64 diag_base_runtime_hash, u64 diag_binding_key,
                        u64 diag_cbuf_key_excl_texture_handles) {
    std::unique_lock lock{mutex_};
    entries_.insert_or_assign(key, Entry{std::make_shared<const std::vector<u32>>(std::move(spirv)),
                                         end_binding, is_speculative,
                                         diag_base_runtime_hash, diag_binding_key,
                                         diag_cbuf_key_excl_texture_handles});
    unique_hashes_.insert(key.unique_hash);
    constexpr size_t kMaxStoredKeysPerHashForDiagnostics = 8;
    auto& stored_keys = keys_by_hash_[key.unique_hash];
    if (std::find(stored_keys.begin(), stored_keys.end(), key) == stored_keys.end() &&
        stored_keys.size() < kMaxStoredKeysPerHashForDiagnostics) {
        stored_keys.push_back(key);
    }
    dirty_ = true;
    if (is_speculative) {
        ++speculative_insert_count_;
    } else {
        ++real_insert_count_;
    }
}

size_t SpirvCache::Size() const {
    std::shared_lock lock{mutex_};
    return entries_.size();
}

void SpirvCache::Insert(u64 unique_hash, const std::unordered_map<u64, u32>& cbuf_values,
                        u64 runtime_key, u64 texture_key, std::vector<u32> spirv,
                        const Shader::Backend::Bindings& end_binding,
                        u64 diag_base_runtime_hash, u64 diag_binding_key,
                        u64 diag_cbuf_key_excl_texture_handles) {
    Insert(SpirvKey{unique_hash, ComputeCbufKey(cbuf_values), runtime_key, texture_key},
           std::move(spirv), end_binding, /*is_speculative=*/false,
           diag_base_runtime_hash, diag_binding_key, diag_cbuf_key_excl_texture_handles);
}

void SpirvCache::InsertSpeculative(u64 unique_hash, u64 runtime_key, u64 texture_key,
                                   std::vector<u32> spirv, u64 diag_base_runtime_hash,
                                   u64 diag_binding_key) {
    if (unique_hash == 0 || spirv.empty()) [[unlikely]] return;
    // Speculative entries have no valid end_binding (they are consumed by the prewarmer,
    // not by the per-stage pipeline loop).  Store a zero binding delta.
    // diag_cbuf_key_excl_texture_handles intentionally left at its default (0): every
    // speculative entry already uses cbuf_key=0 (empty cbuf_values — see the SpirvKey
    // built below), so 0 already correctly equals "this entry's own key.cbuf_key",
    // which is exactly the "no narrowing data / nothing to narrow" convention documented
    // on Entry::diag_cbuf_key_excl_texture_handles. No special-casing needed here.
    Insert(SpirvKey{unique_hash, 0, runtime_key, texture_key}, std::move(spirv), {},
           /*is_speculative=*/true, diag_base_runtime_hash, diag_binding_key);
}

} // namespace VideoCommon
