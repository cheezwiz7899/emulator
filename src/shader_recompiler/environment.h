// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/shader_info.h"
#include "shader_recompiler/stage.h"

namespace Shader {

// Forward declaration so Environment can expose a no-RTTI downcast helper.
// Defined in video_core/shader_environment.h.
} // namespace Shader
namespace VideoCommon { class GenericEnvironment; }
namespace VideoCommon { class FileEnvironment; }
namespace Shader {

class Environment {
public:
    virtual ~Environment() = default;

    /// Returns a GenericEnvironment* if this object derives from GenericEnvironment,
    /// nullptr otherwise (e.g. FileEnvironment). Used in place of dynamic_cast
    /// when the build disables RTTI (-fno-rtti).
    [[nodiscard]] virtual VideoCommon::GenericEnvironment* AsGenericEnvironment() noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual const VideoCommon::GenericEnvironment*
        AsGenericEnvironment() const noexcept {
        return nullptr;
    }

    /// Returns a FileEnvironment* if this object IS a FileEnvironment, nullptr
    /// otherwise. Same no-RTTI downcast pattern as AsGenericEnvironment() above,
    /// for the one class that deliberately does NOT derive from GenericEnvironment
    /// but still deserializes real (not guessed) cbuf/texture capture data from
    /// disk — see FileEnvironment::CapturedCbufValues() and friends.
    [[nodiscard]] virtual VideoCommon::FileEnvironment* AsFileEnvironment() noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual const VideoCommon::FileEnvironment*
        AsFileEnvironment() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual u64 ReadInstruction(u32 address) = 0;

    [[nodiscard]] virtual u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) = 0;

    // Same read as ReadCbufValue, called ONLY from GetTextureHandle() (texture_pass.cpp)
    // for the two reads that resolve a bindless texture handle — never from anywhere
    // else (constant-propagation's FoldDriverConstBuffer and the indirect-branch-table
    // walk in control_flow.cpp both call plain ReadCbufValue(), deliberately not this).
    // That makes this call site itself the signal: whichever (index, offset) pairs
    // arrive here are — as far as every Environment::ReadCbufValue caller in the
    // codebase is concerned — used for NOTHING except resolving a handle that
    // ReadTextureType()/ReadTexturePixelFormat() then resolve independently and that
    // spirv_cache's texture_key already captures on its own. Default implementation
    // just forwards to ReadCbufValue() (identical behavior to before this existed) —
    // GenericEnvironment is the only override, and it ALSO records the (index, offset)
    // pair for VideoCommon::GraphicsPipelineCacheKey-adjacent diagnostics (see
    // CapturedTextureHandleCbufKeys() in shader_environment.h). Every other Environment
    // (FileEnvironment, the scanner's SpeculativeShaderEnvironment) gets the safe
    // default: no tagging, same as if this method didn't exist.
    [[nodiscard]] virtual u32 ReadCbufValueForTextureHandle(u32 cbuf_index, u32 cbuf_offset) {
        return ReadCbufValue(cbuf_index, cbuf_offset);
    }

    /// Returns the byte size of the const buffer at cbuf_index, or 0 if not bound /
    /// not knowable from this environment. Callers treat 0 as "unknown" and fall
    /// back to the original conservative default (e.g. disk-cache replay path).
    [[nodiscard]] virtual u32 ReadCbufSize([[maybe_unused]] u32 cbuf_index) {
        return 0;
    }

    [[nodiscard]] virtual TextureType ReadTextureType(u32 raw_handle) = 0;

    [[nodiscard]] virtual TexturePixelFormat ReadTexturePixelFormat(u32 raw_handle) = 0;

    // Phase 4 (specialization-constant texture-type resolution) feasibility instrumentation —
    // see handoff_04_specialization_constants_investigation.md, item 2: "how many distinct
    // (type, format) pairs does a single logical texture slot actually take on in real
    // content". Called immediately after a texture handle resolves, from the SAME call sites
    // in texture_pass.cpp that call ReadTextureType()/ReadTexturePixelFormat() for handle
    // resolution (mirrors ReadCbufValueForTextureHandle's pattern above, including the
    // reasoning for why a default no-op is safe here). Two independent hooks, not one
    // combined (type, format) callback: the two reads are NOT reliably called together at
    // the same site (e.g. the ImageQueryDimensions case only ever calls the type read; the
    // texel-fetch SNORM-workaround case only ever calls the format read), so a combined
    // signature would force a meaningless placeholder value on whichever half wasn't
    // actually resolved at that call site.
    //
    // "Slot" identity is (this shader's unique_hash, cbuf_index, cbuf_offset) — the same
    // (index, offset) pair ReadCbufValueForTextureHandle tags, plus the shader identity,
    // since the same cbuf coordinates mean different logical textures in different shaders.
    // Default no-op — only GenericEnvironment overrides these, so only real, live-gameplay
    // translations contribute (never FileEnvironment disk-replay or the scanner's
    // SpeculativeShaderEnvironment, which returns a fixed guess unconditionally and would
    // only pollute the distinct-value count with a constant that was never really observed).
    //
    // NOT YET WIRED TO ANY REPORTING CADENCE — see GenericEnvironment::
    // LogTextureSlotVarianceReportThrottled()'s doc comment in shader_environment.h for where
    // this data actually gets surfaced, and its own doc comment for why this whole mechanism
    // is unverified: written without the ability to build or run this codebase, needs a real
    // compile and a real play session before anyone trusts its output.
    virtual void RecordResolvedTextureType([[maybe_unused]] u32 cbuf_index,
                                           [[maybe_unused]] u32 cbuf_offset,
                                           [[maybe_unused]] u32 handle,
                                           [[maybe_unused]] TextureType type) {}
    virtual void RecordResolvedTexturePixelFormat([[maybe_unused]] u32 cbuf_index,
                                                  [[maybe_unused]] u32 cbuf_offset,
                                                  [[maybe_unused]] TexturePixelFormat format) {}

    [[nodiscard]] virtual bool IsTexturePixelFormatInteger(u32 raw_handle) = 0;

    [[nodiscard]] virtual u32 ReadViewportTransformState() = 0;

    [[nodiscard]] virtual u32 TextureBoundBuffer() const = 0;

    [[nodiscard]] virtual u32 LocalMemorySize() const = 0;

    [[nodiscard]] virtual u32 SharedMemorySize() const = 0;

    [[nodiscard]] virtual std::array<u32, 3> WorkgroupSize() const = 0;

    [[nodiscard]] virtual bool HasHLEMacroState() const = 0;

    [[nodiscard]] virtual std::optional<ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) = 0;

    virtual void Dump(u64 pipeline_hash, u64 shader_hash) = 0;

    [[nodiscard]] const ProgramHeader& SPH() const noexcept {
        return sph;
    }

    [[nodiscard]] const std::array<u32, 8>& GpPassthroughMask() const noexcept {
        return gp_passthrough_mask;
    }

    [[nodiscard]] Stage ShaderStage() const noexcept {
        return stage;
    }

    [[nodiscard]] u32 StartAddress() const noexcept {
        return start_address;
    }

    [[nodiscard]] bool IsProprietaryDriver() const noexcept {
        return is_proprietary_driver;
    }

protected:
    ProgramHeader sph{};
    std::array<u32, 8> gp_passthrough_mask{};
    Stage stage{};
    u32 start_address{};
    bool is_proprietary_driver{};
};

// Phase 4 narrow prototype (specialization-constant texture-type resolution). The one
// hardcoded (cbuf_index, cbuf_offset) pattern handoff_04's investigation identified (12 real
// TotK shaders, Color2D vs ColorArray2D). Shared between texture_pass.cpp (which uses it to
// canonicalize flags.type and mark TextureDescriptor::phase4_prototype_polymorphic) and
// shader_environment.cpp (which uses it to know which resolved handles to exclude from
// texture_key, so the two real variants of this slot hash to the same key instead of
// fragmenting the cache) -- one definition so the two can't independently drift out of sync.
[[nodiscard]] constexpr bool IsPhase4PrototypeSlot(u32 cbuf_index, u32 cbuf_offset) noexcept {
    constexpr u32 kPrototypeCbufIndex = 2;
    constexpr u32 kPrototypeCbufOffset = 192;
    return cbuf_index == kPrototypeCbufIndex && cbuf_offset == kPrototypeCbufOffset;
}

} // namespace Shader
