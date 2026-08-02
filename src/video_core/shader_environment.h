// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"
#include "shader_recompiler/environment.h"
#include "video_core/engines/maxwell_3d.h"

namespace Tegra {
class Memorymanager;
}

namespace VideoCommon {

// Canonical (cbuf_index, cbuf_offset) -> u64 packing shared by every cbuf-key consumer:
// GenericEnvironment's own cbuf_values/cbuf_replacements capture (shader_environment.cpp),
// the texture-handle tagging below, and anything reading CapturedCbufValues() /
// CapturedTextureHandleCbufKeys() from outside this class. One definition, so the two
// sides of that comparison can never silently drift out of sync with each other.
[[nodiscard]] constexpr u64 MakeCbufKey(u32 index, u32 offset) noexcept {
    return (static_cast<u64>(index) << 32) | offset;
}

class GenericEnvironment : public Shader::Environment {
public:
    explicit GenericEnvironment() = default;
    explicit GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_);

    ~GenericEnvironment() override;

    [[nodiscard]] u32 TextureBoundBuffer() const final;

    [[nodiscard]] u32 LocalMemorySize() const final;

    [[nodiscard]] u32 SharedMemorySize() const final;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const final;

    [[nodiscard]] u64 ReadInstruction(u32 address) final;

    [[nodiscard]] std::optional<u64> Analyze();

    void SetCachedSize(size_t size_bytes);

    [[nodiscard]] size_t CachedSizeWords() const noexcept;

    [[nodiscard]] size_t CachedSizeBytes() const noexcept;

    [[nodiscard]] size_t ReadSizeBytes() const noexcept;

    [[nodiscard]] bool CanBeSerialized() const noexcept;

    [[nodiscard]] u64 CalculateHash() const;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    void Serialize(std::ofstream& file) const;

    /// GPL: copy the cached Maxwell code words into @p out.
    void CopyCode(std::vector<u64>& out) const { out = code; }

    /// GPL: read-only view of cbuf values captured during translation.
    const std::unordered_map<u64, u32>& CapturedCbufValues() const noexcept {
        return cbuf_values;
    }

    /// Subset of CapturedCbufValues()'s keys (same MakeCbufKey(index, offset) packing)
    /// that were read ONLY to resolve a bindless texture handle — see
    /// ReadCbufValueForTextureHandle's doc comment in environment.h for exactly what
    /// "only" means here and why it's safe to rely on structurally rather than by
    /// inference. Diagnostic scaffolding for validating (with real per-session gameplay
    /// data, not guesswork) whether cbuf_key can be narrowed to exclude entries whose
    /// downstream effect on codegen is already fully captured by texture_key — see
    /// VideoCommon::ComputeCbufKeyExcludingTextureHandles in spirv_cache.h. Empty for
    /// any shader that doesn't sample a bindless texture at all, which is common and
    /// fine — an empty exclusion set just means the counterfactual equals cbuf_key.
    const std::unordered_set<u64>& CapturedTextureHandleCbufKeys() const noexcept {
        return texture_handle_cbuf_keys;
    }

    /// See ReadCbufValueForTextureHandle's doc comment in environment.h. GenericEnvironment
    /// is the one override point for ALL its subclasses (GraphicsEnvironment,
    /// ComputeEnvironment) — `final` stops the chain here since the tagging logic
    /// itself doesn't differ between them; it just needs to reach whichever
    /// ReadCbufValue() override is active, which happens automatically via the
    /// (still-virtual, separately overridden per subclass) call below.
    u32 ReadCbufValueForTextureHandle(u32 cbuf_index, u32 cbuf_offset) override final {
        const u32 value = ReadCbufValue(cbuf_index, cbuf_offset);
        texture_handle_cbuf_keys.insert(MakeCbufKey(cbuf_index, cbuf_offset));
        return value;
    }

    // See RecordResolvedTextureType()/RecordResolvedTexturePixelFormat()'s doc comment in
    // environment.h. GenericEnvironment is the one override point, same as
    // ReadCbufValueForTextureHandle above — UNVERIFIED, see the doc comment on these two
    // functions' definitions in shader_environment.cpp.
    void RecordResolvedTextureType(u32 cbuf_index, u32 cbuf_offset, u32 handle,
                                   Shader::TextureType type) override final;
    void RecordResolvedTexturePixelFormat(u32 cbuf_index, u32 cbuf_offset,
                                          Shader::TexturePixelFormat format) override final;

    // Phase 4 narrow prototype's texture_key fix. Populated inside RecordResolvedTextureType
    // (shader_environment.cpp) whenever a resolved handle belongs to the one hardcoded
    // (cbuf_index, cbuf_offset) slot (Shader::IsPhase4PrototypeSlot, environment.h). Passed to
    // ComputeTextureKeyExcludingHandles (spirv_cache.h) at the CreateGraphicsPipeline/
    // CreateComputePipeline call sites so the two real variants of this one slot hash to the
    // same texture_key instead of fragmenting the cache — mirrors
    // CapturedTextureHandleCbufKeys/ComputeCbufKeyExcludingTextureHandles above exactly, same
    // reasoning, different key.
    const std::unordered_set<u32>& CapturedPhase4PrototypeHandles() const noexcept {
        return phase4_prototype_handles;
    }

    // Prints (LOG_INFO, Render_Vulkan) a distinct-value-count histogram across every
    // (shader, cbuf slot) pair observed so far this session, answering
    // handoff_04_specialization_constants_investigation.md item 2. Self-throttled — cheap
    // to call from anywhere convenient during play (e.g. next to the existing
    // spirv_cache.SaveThrottled(...) call sites); most calls are a no-op. Static because the
    // data it reports is accumulated across ALL GenericEnvironment instances over the
    // session, not any one instance — see the accumulator's own doc comment in
    // shader_environment.cpp for why. UNVERIFIED, same caveat as the two functions above.
    static void LogTextureSlotVarianceReportThrottled();

    /// GPL: read-only view of texture types captured during translation.
    const std::unordered_map<u32, Shader::TextureType>& CapturedTextureTypes() const noexcept {
        return texture_types;
    }

    /// GPL: read-only view of texture pixel formats captured during translation.
    const std::unordered_map<u32, Shader::TexturePixelFormat>& CapturedTexturePixelFormats() const noexcept {
        return texture_pixel_formats;
    }

    /// No-RTTI downcast: GenericEnvironment is always a GenericEnvironment.
    VideoCommon::GenericEnvironment* AsGenericEnvironment() noexcept override {
        return this;
    }
    const VideoCommon::GenericEnvironment* AsGenericEnvironment() const noexcept override {
        return this;
    }

    bool HasHLEMacroState() const override {
        return has_hle_engine_state;
    }

protected:
    std::optional<u64> TryFindSize();

    Tegra::Texture::TICEntry ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                             bool via_header_index, u32 raw);

    Tegra::MemoryManager* gpu_memory{};
    GPUVAddr program_base{};

    std::vector<u64> code;
    std::unordered_map<u32, Shader::TextureType> texture_types;
    std::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    std::unordered_map<u64, u32> cbuf_values;
    // See ReadCbufValueForTextureHandle() and CapturedTextureHandleCbufKeys() above.
    std::unordered_set<u64> texture_handle_cbuf_keys;
    // See CapturedPhase4PrototypeHandles() above.
    std::unordered_set<u32> phase4_prototype_handles;
    // Memoized CalculateHash() for RecordResolvedTextureType()/RecordResolvedTexturePixelFormat()
    // (shader_environment.cpp) — those can fire once per texture instruction in a shader, and
    // CalculateHash() does a fresh GPU-memory read + CityHash64 over the whole shader every
    // call, so this caches it per-instance instead of recomputing per texture instruction.
    // Safe to cache: the underlying shader bytes are immutable for this instance's lifetime.
    // Computed lazily — stays unset (and free) for any translation that never touches a
    // bindless texture handle. mutable: read-only diagnostic memoization, not real state.
    mutable std::optional<u64> texture_slot_diag_hash_cache;
    std::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;
    // Cbuf sizes captured on the main thread at construction time while GPU state is live.
    std::unordered_map<u32, u32> cbuf_sizes;

    u32 local_memory_size{};
    u32 texture_bound{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    u32 read_lowest = std::numeric_limits<u32>::max();
    u32 read_highest = 0;

    u32 cached_lowest = std::numeric_limits<u32>::max();
    u32 cached_highest = 0;
    u32 initial_offset = 0;

    u32 viewport_transform_state = 1;

    bool has_unbound_instructions = false;
    bool has_hle_engine_state = false;
};

class GraphicsEnvironment final : public GenericEnvironment {
public:
    explicit GraphicsEnvironment() = default;
    explicit GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                 Tegra::MemoryManager& gpu_memory_,
                                 Tegra::Engines::Maxwell3D::Regs::ShaderType program,
                                 GPUVAddr program_base_, u32 start_address_);

    ~GraphicsEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    u32 ReadCbufSize(u32 cbuf_index) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;

private:
    Tegra::Engines::Maxwell3D* maxwell3d{};
    size_t stage_index{};
};

class ComputeEnvironment final : public GenericEnvironment {
public:
    explicit ComputeEnvironment() = default;
    explicit ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_);

    ~ComputeEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    u32 ReadCbufSize(u32 cbuf_index) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(
        [[maybe_unused]] u32 bank, [[maybe_unused]] u32 offset) override {
        return std::nullopt;
    }

private:
    Tegra::Engines::KeplerCompute* kepler_compute{};
};

class FileEnvironment final : public Shader::Environment {
public:
    FileEnvironment() = default;
    ~FileEnvironment() override = default;

    FileEnvironment& operator=(FileEnvironment&&) noexcept = default;
    FileEnvironment(FileEnvironment&&) noexcept = default;

    FileEnvironment& operator=(const FileEnvironment&) = delete;
    FileEnvironment(const FileEnvironment&) = delete;

    void Deserialize(std::ifstream& file);

    /// No-RTTI downcast: FileEnvironment is always a FileEnvironment.
    VideoCommon::FileEnvironment* AsFileEnvironment() noexcept override { return this; }
    const VideoCommon::FileEnvironment* AsFileEnvironment() const noexcept override {
        return this;
    }

    /// Real (not guessed) cbuf/texture specialization data deserialized from
    /// disk — the exact values GenericEnvironment::Serialize() wrote out when
    /// this pipeline was originally compiled live. Mirrors GenericEnvironment's
    /// accessors of the same name so SpirvCache key computation can use real
    /// specialization for the disk-replay path instead of forcing cbuf_key/
    /// texture_key to 0 for lack of anywhere to read it from — the data was
    /// deserialized into cbuf_values/texture_types/texture_pixel_formats below
    /// all along, just not previously exposed.
    const std::unordered_map<u64, u32>& CapturedCbufValues() const noexcept {
        return cbuf_values;
    }
    const std::unordered_map<u32, Shader::TextureType>& CapturedTextureTypes() const noexcept {
        return texture_types;
    }
    const std::unordered_map<u32, Shader::TexturePixelFormat>&
        CapturedTexturePixelFormats() const noexcept {
        return texture_pixel_formats;
    }

    [[nodiscard]] u64 ReadInstruction(u32 address) override;

    [[nodiscard]] bool HasValidEntryInstruction() const noexcept;

    [[nodiscard]] u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    [[nodiscard]] u32 ReadCbufSize(u32 cbuf_index) override;

    [[nodiscard]] Shader::TextureType ReadTextureType(u32 handle) override;

    [[nodiscard]] Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    [[nodiscard]] bool IsTexturePixelFormatInteger(u32 handle) override;

    [[nodiscard]] u32 ReadViewportTransformState() override;

    [[nodiscard]] u32 LocalMemorySize() const override;

    [[nodiscard]] u32 SharedMemorySize() const override;

    [[nodiscard]] u32 TextureBoundBuffer() const override;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const override;

    [[nodiscard]] std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) override;

    [[nodiscard]] bool HasHLEMacroState() const override {
        return cbuf_replacements.size() != 0;
    }

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

private:
    std::vector<u64> code;
    std::unordered_map<u32, Shader::TextureType> texture_types;
    std::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    std::unordered_map<u64, u32> cbuf_values;
    std::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;
    std::unordered_map<u32, u32> cbuf_sizes;
    std::array<u32, 3> workgroup_size{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 initial_offset{};
    u32 viewport_transform_state = 1;
};

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version);

template <typename Key, typename Envs>
void SerializePipeline(const Key& key, const Envs& envs, const std::filesystem::path& filename,
                       u32 cache_version) {
    static_assert(std::is_trivially_copyable_v<Key>);
    static_assert(std::has_unique_object_representations_v<Key>);
    SerializePipeline(std::span(reinterpret_cast<const char*>(&key), sizeof(key)),
                      std::span(envs.data(), envs.size()), filename, cache_version);
}

void LoadPipelines(
    std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics);

} // namespace VideoCommon
