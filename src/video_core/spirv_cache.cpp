// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <fstream>
#include <shared_mutex>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/spirv_cache.h"

namespace {
constexpr std::array<char, 8> SPIRV_CACHE_MAGIC{'c', 'i', 't', 'r', 's', 'p', 'v', '\0'};
constexpr u32 SPIRV_CACHE_VERSION = 2; // v2: adds Bindings end_binding per entry
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
    }
    Save(path);
}

bool SpirvCache::Contains(const SpirvKey& key) const noexcept {
    std::shared_lock lock{mutex_};
    return entries_.count(key) != 0;
}

bool SpirvCache::ContainsByUniqueHash(u64 unique_hash) const noexcept {
    std::shared_lock lock{mutex_};
    for (const auto& [key, _] : entries_) {
        if (key.unique_hash == unique_hash) {
            return true;
        }
    }
    return false;
}

std::optional<SpirvCache::LookupResult> SpirvCache::Lookup(const SpirvKey& key) const {
    std::shared_lock lock{mutex_};
    ++lookup_count_;
    const auto it = entries_.find(key);
    if (it == entries_.end()) return std::nullopt;
    ++hit_count_;
    return LookupResult{it->second.spirv, it->second.end_binding};
}

void SpirvCache::Insert(const SpirvKey& key, std::vector<u32> spirv,
                        const Shader::Backend::Bindings& end_binding) {
    std::unique_lock lock{mutex_};
    entries_.insert_or_assign(key, Entry{std::make_shared<const std::vector<u32>>(std::move(spirv)),
                                         end_binding});
    dirty_ = true;
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
    Insert(SpirvKey{unique_hash, 0, runtime_key, texture_key}, std::move(spirv), {});
}

} // namespace VideoCommon
