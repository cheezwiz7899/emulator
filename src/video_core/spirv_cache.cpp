// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <fstream>
#include <shared_mutex>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/spirv_cache.h"

namespace {
constexpr std::array<char, 8> SPIRV_CACHE_MAGIC{'c', 'i', 't', 'r', 's', 'p', 'v', '\0'};
constexpr u32 SPIRV_CACHE_VERSION = 5; // v5: ComputeBindingKey reverted to a full-state hash
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
                                        // end_binding per entry)
} // anonymous namespace

namespace VideoCommon {

u64 ComputeCbufKey(const std::unordered_map<u64, u32>& cbuf_values) {
    if (cbuf_values.empty()) return 0;
    std::vector<std::pair<u64,u32>> sorted(cbuf_values.begin(), cbuf_values.end());
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
            entries_.emplace(key, Entry{std::make_shared<const std::vector<u32>>(std::move(spirv)),
                                        end_binding});
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
        if (window_probes > 0) {
            const int window_pct = static_cast<int>(window_hits * 100 / window_probes);
            LOG_INFO(Render_Vulkan,
                     "SPIR-V cache: {} entries ({} speculative / {} real inserted so far) — "
                     "recent window {}/{} stage hits ({}%), {} misses where the shader was "
                     "already cached under a different key ({} had no real cbuf/texture "
                     "context [FileEnvironment] / {} had real context that still mismatched).",
                     entries_now, speculative_insert_count_.load(), real_insert_count_.load(),
                     window_hits, window_probes, window_pct, window_stale_misses,
                     window_no_ctx, window_with_ctx);
        } else {
            LOG_INFO(Render_Vulkan,
                     "SPIR-V cache: {} entries ({} speculative / {} real inserted so far) — "
                     "no lookups since last log.",
                     entries_now, speculative_insert_count_.load(), real_insert_count_.load());
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
                                                             bool has_real_specialization_context) const {
    std::shared_lock lock{mutex_};
    ++lookup_count_;
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

            // Throttled: report exactly which field(s) differ from the
            // requested key, using whatever's actually stored for this hash.
            // A handful of samples is enough to tell whether cbuf_key,
            // runtime_key, or texture_key (or some combination) is the
            // dominant mismatch source, without flooding the log over a
            // whole play session.
            constexpr size_t kMaxFieldMismatchLogs = 30;
            std::atomic<size_t>& field_mismatch_logs =
                has_real_specialization_context ? field_mismatch_logs_with_context_
                                                 : field_mismatch_logs_no_context_;
            size_t expected = field_mismatch_logs.load();
            bool got_slot = false;
            while (expected < kMaxFieldMismatchLogs &&
                   !field_mismatch_logs.compare_exchange_weak(expected, expected + 1)) {
            }
            got_slot = expected < kMaxFieldMismatchLogs;
            if (got_slot) {
                const auto hit_it = keys_by_hash_.find(key.unique_hash);
                if (hit_it != keys_by_hash_.end() && !hit_it->second.empty()) {
                    for (const SpirvKey& stored : hit_it->second) {
                        LOG_INFO(Render_Vulkan,
                                 "SPIR-V cache field mismatch [{}] (real_context={}): requested "
                                 "cbuf={:016x} runtime={:016x} texture={:016x} vs "
                                 "stored cbuf={:016x} runtime={:016x} texture={:016x} "
                                 "(cbuf {}, runtime {}, texture {})",
                                 expected, has_real_specialization_context,
                                 key.cbuf_key, key.runtime_key, key.texture_key,
                                 stored.cbuf_key, stored.runtime_key, stored.texture_key,
                                 key.cbuf_key == stored.cbuf_key ? "matches" : "DIFFERS",
                                 key.runtime_key == stored.runtime_key ? "matches" : "DIFFERS",
                                 key.texture_key == stored.texture_key ? "matches" : "DIFFERS");
                    }
                }
            }
        }
        return std::nullopt;
    }
    ++hit_count_;
    return LookupResult{it->second.spirv, it->second.end_binding, it->second.is_speculative};
}

void SpirvCache::Insert(const SpirvKey& key, std::vector<u32> spirv,
                        const Shader::Backend::Bindings& end_binding, bool is_speculative) {
    std::unique_lock lock{mutex_};
    entries_.insert_or_assign(key, Entry{std::make_shared<const std::vector<u32>>(std::move(spirv)),
                                         end_binding, is_speculative});
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
                        const Shader::Backend::Bindings& end_binding) {
    Insert(SpirvKey{unique_hash, ComputeCbufKey(cbuf_values), runtime_key, texture_key},
           std::move(spirv), end_binding);
}

void SpirvCache::InsertSpeculative(u64 unique_hash, u64 runtime_key, u64 texture_key, std::vector<u32> spirv) {
    if (unique_hash == 0 || spirv.empty()) [[unlikely]] return;
    // Speculative entries have no valid end_binding (they are consumed by the prewarmer,
    // not by the per-stage pipeline loop).  Store a zero binding delta.
    Insert(SpirvKey{unique_hash, 0, runtime_key, texture_key}, std::move(spirv), {}, /*is_speculative=*/true);
}

} // namespace VideoCommon
