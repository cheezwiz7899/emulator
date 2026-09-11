// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include <optional>

#include "common/cityhash.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/program_header.h"
#include "video_core/shader_program_identity.h"

namespace VideoCommon {

class SpeculativeShaderEnvironment final : public Shader::Environment {
public:
    explicit SpeculativeShaderEnvironment(std::vector<u64> code_, u32 start_address_,
                                          Shader::Stage stage_, u32 local_memory_,
                                          u32 shared_memory_, std::array<u32, 3> workgroup_,
                                          u32 texture_bound_, Shader::ProgramHeader sph_,
                                          u32 code_offset_in_program_)
        : code{std::move(code_)}, local_memory_size{local_memory_},
          shared_memory_size{shared_memory_}, texture_bound{texture_bound_},
          workgroup_size{workgroup_},
          code_lowest{start_address_ + code_offset_in_program_} {
        start_address = start_address_;
        stage = stage_;
        sph = sph_;
        is_proprietary_driver = false;
    }

    // Constructor used by the pre-cache scanner. code_offset_in_program_ is 0:
    // code_ includes the SPH as its prefix, matching the live program span.
    // Program identity is calculated by shader_program_identity.h, independent
    // of this environment's CFG read bounds.
    explicit SpeculativeShaderEnvironment(std::vector<u64> code_, Shader::Stage stage_, u32 local_memory_size_, Shader::ProgramHeader sph_)
        : SpeculativeShaderEnvironment(std::move(code_), 0, stage_, local_memory_size_, 0, {1u, 1u, 1u}, 1u, sph_, 0u) {
    }

    u64 ReadInstruction(u32 address) override {
        if (address < code_lowest) {
            throw Shader::Exception("SpeculativeShaderEnvironment: ReadInstruction below code start");
        }
        const u32 i = (address - code_lowest) / 8;
        if (i >= code.size()) {
            throw Shader::Exception("SpeculativeShaderEnvironment: ReadInstruction out of bounds");
        }
        // Track read bounds for fallback hash calculation
        if (address < read_lowest) read_lowest = address;
        if (address > read_highest) read_highest = address;
        return code[i];
    }

    u32 ReadCbufValue(u32, u32) override {
        // Speculative translation has no real constant buffer data, so this can
        // only ever return a guess.  Returning a sentinel (rather than throwing)
        // is intentional and safe:
        //
        // The overwhelmingly common caller is GetTextureHandle() in
        // texture_pass.cpp, invoked for every texture sample instruction to
        // resolve a handle before calling ReadTextureType()/ReadTexturePixelFormat()
        // below — which themselves ignore the handle's value entirely and always
        // return the same guessed type.  Throwing here previously aborted
        // translation for ~100% of real shaders (anything that samples a texture).
        //
        // The rarer caller is the BRX indirect-branch-table walk in
        // control_flow.cpp, which reads `num_entries` fake jump targets from
        // here.  That loop is already bounded by an immediate baked into the
        // bytecode (an IMNMX instruction operand), not by anything read here, so
        // a sentinel value cannot cause an unbounded walk.  Any bogus branch
        // target it produces from this fake data is still caught downstream by
        // ReadInstruction()'s out-of-bounds throw or Decode()'s
        // unrecognised-instruction throw.
        unknown_cbuf_reads = true;
        return 0u;
    }
    u32 ReadCbufSize(u32 i) override {
        unknown_cbuf_sizes = true;
        return i < 18 ? 65536u : 0u;
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        unknown_texture_shape_reads = true;
        texture_types.emplace(handle, Shader::TextureType::Color2D);
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        unknown_texture_shape_reads = true;
        texture_pixel_formats.emplace(handle, Shader::TexturePixelFormat::A8B8G8R8_UNORM);
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    void RecordResolvedTextureType(const Shader::TextureSlot& slot, u32 handle,
                                   Shader::TextureType type) override {
        logical_texture_slots[slot].type = type;
        logical_texture_handles.insert(handle);
    }
    void RecordResolvedTexturePixelFormat(const Shader::TextureSlot& slot, u32 handle,
                                          Shader::TexturePixelFormat format) override {
        logical_texture_slots[slot].pixel_format = format;
        logical_texture_handles.insert(handle);
    }
    void RecordResolvedIsTexturePixelFormatInteger(const Shader::TextureSlot& slot,
                                                   bool is_integer) override {
        logical_texture_slots[slot].is_integer = is_integer;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        unknown_texture_shape_reads = true;
        return false;
    }
    u32 ReadViewportTransformState() override {
        unknown_viewport_state = true;
        return viewport_transform_state_;
    }
    // The real value is GPU register state a speculative (no live draw) translation
    // has no way to observe. Unlike cbuf content or non-leading binding state, this
    // is a genuine 2-way fork (see PositionPass() in ir_opt/position_pass.cpp) with
    // no other possible values, so callers can cheaply try both by translating twice
    // with this set differently each time, instead of betting on a single guess.
    // Defaults to 1 (the prior hardcoded behavior) so existing call sites that never
    // call this are unaffected.
    void SetViewportTransformState(u32 value) noexcept { viewport_transform_state_ = value; }
    // A raw scanner blob has no proof of the guest GPU address at which the
    // driver will place its SPH. Maxwell scheduler words depend on that address
    // modulo 32, so trying candidate alignments is useful extraction telemetry
    // but cannot certify any one resulting CFG/module for exact publication.
    // Live speculative callers receive their actual start address and never set
    // this flag.
    void MarkSchedulingAlignmentUnknown() noexcept { unknown_scheduling_alignment = true; }
    u32 TextureBoundBuffer() const override {
        // The scanner has no live bindless-texture cbuf register. This value
        // is only a fallback for translation; if lowering consumes it, final
        // module reuse needs a real binding-state contract.
        unknown_texture_binding_state = true;
        return texture_bound;
    }
    bool IsProprietaryDriver() const noexcept override {
        // This selects driver-cbuf folding in ConstantPropagationPass. It is
        // derived from live texture binding state in GenericEnvironment.
        unknown_proprietary_driver_state = true;
        return false;
    }
    const std::array<u32, 8>& GpPassthroughMask() const noexcept override {
        // Geometry passthrough is controlled by post-VTG draw state, not the
        // serialized shader program. The base value is only a fallback.
        unknown_interface_state = true;
        return gp_passthrough_mask;
    }
    u32 LocalMemorySize() const override { return local_memory_size; }
    u32 SharedMemorySize() const override {
        if (stage == Shader::Stage::Compute) {
            unknown_compute_launch_state = true;
        }
        return shared_memory_size;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        if (stage == Shader::Stage::Compute) {
            unknown_compute_launch_state = true;
        }
        return workgroup_size;
    }
    bool HasHLEMacroState() const override {
        // This controls constant-buffer replacement during finalization. A
        // scanner has no live engine state, so false is only a translation
        // fallback, never evidence that replacement cannot change the module.
        unknown_hle_macro_state = true;
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}

    const std::unordered_map<u32, Shader::TextureType>& CapturedTextureTypes() const noexcept {
        return texture_types;
    }

    const std::unordered_map<u32, Shader::TexturePixelFormat>& CapturedTexturePixelFormats() const noexcept {
        return texture_pixel_formats;
    }
    const Shader::LogicalTextureSlots& CapturedLogicalTextureSlots() const noexcept {
        return logical_texture_slots;
    }
    const Shader::LogicalTextureHandles& CapturedLogicalTextureHandles() const noexcept {
        return logical_texture_handles;
    }

    [[nodiscard]] bool HasUnknownCbufDependencies() const noexcept {
        return unknown_cbuf_reads || unknown_cbuf_sizes || unknown_proprietary_driver_state;
    }
    [[nodiscard]] bool HasUnknownTextureDependencies() const noexcept {
        return unknown_texture_shape_reads || unknown_texture_binding_state;
    }
    [[nodiscard]] bool HasUnknownViewportDependency() const noexcept {
        return unknown_viewport_state;
    }
    [[nodiscard]] bool HasUnknownComputeLaunchDependency() const noexcept {
        return unknown_compute_launch_state;
    }
    [[nodiscard]] bool HasUnknownHLEMacroDependency() const noexcept {
        return unknown_hle_macro_state;
    }
    [[nodiscard]] bool HasUnknownInterfaceDependency() const noexcept {
        return unknown_interface_state;
    }
    [[nodiscard]] bool HasUnknownSchedulingAlignmentDependency() const noexcept {
        return unknown_scheduling_alignment;
    }

    // A scanner performs CFG discovery before BuildProgramTemplate. CFG can
    // legitimately read cbuf branch data, while the frontend IR boundary must
    // only be rejected for state that BuildProgramTemplate itself consumes.
    // Keep the two observations distinct; callers restoring the snapshot before
    // final-module publication retain the stricter whole-translation contract.
    struct DependencySnapshot {
        bool cbuf_reads{};
        bool cbuf_sizes{};
        bool texture_shape_reads{};
        bool viewport{};
        bool texture_binding{};
        bool proprietary_driver{};
        bool compute_launch{};
        bool hle_macro{};
        bool interface{};
        bool scheduling_alignment{};
    };

    [[nodiscard]] DependencySnapshot TakeDependencySnapshot() noexcept {
        const DependencySnapshot snapshot{
            .cbuf_reads = unknown_cbuf_reads,
            .cbuf_sizes = unknown_cbuf_sizes,
            .texture_shape_reads = unknown_texture_shape_reads,
            .viewport = unknown_viewport_state,
            .texture_binding = unknown_texture_binding_state,
            .proprietary_driver = unknown_proprietary_driver_state,
            .compute_launch = unknown_compute_launch_state,
            .hle_macro = unknown_hle_macro_state,
            .interface = unknown_interface_state,
            .scheduling_alignment = unknown_scheduling_alignment,
        };
        unknown_cbuf_reads = false;
        unknown_cbuf_sizes = false;
        unknown_texture_shape_reads = false;
        unknown_viewport_state = false;
        unknown_texture_binding_state = false;
        unknown_proprietary_driver_state = false;
        unknown_compute_launch_state = false;
        unknown_hle_macro_state = false;
        unknown_interface_state = false;
        unknown_scheduling_alignment = false;
        return snapshot;
    }

    void MergeDependencySnapshot(const DependencySnapshot& snapshot) noexcept {
        unknown_cbuf_reads |= snapshot.cbuf_reads;
        unknown_cbuf_sizes |= snapshot.cbuf_sizes;
        unknown_texture_shape_reads |= snapshot.texture_shape_reads;
        unknown_viewport_state |= snapshot.viewport;
        unknown_texture_binding_state |= snapshot.texture_binding;
        unknown_proprietary_driver_state |= snapshot.proprietary_driver;
        unknown_compute_launch_state |= snapshot.compute_launch;
        unknown_hle_macro_state |= snapshot.hle_macro;
        unknown_interface_state |= snapshot.interface;
        unknown_scheduling_alignment |= snapshot.scheduling_alignment;
    }

    // This records whether none of the tracked synthetic dependencies were
    // consulted. It does not prove BuildProgramTemplate() is state-independent
    // or serializable: it still has code/SPH and some instruction-path reads.
    // Callers must define and prove their own artifact boundary.
    [[nodiscard]] bool HasStateIndependentTemplateContract() const noexcept {
        return !HasUnknownCbufDependencies() && !HasUnknownTextureDependencies() &&
               !HasUnknownViewportDependency() && !HasUnknownComputeLaunchDependency() &&
               !HasUnknownHLEMacroDependency() && !HasUnknownInterfaceDependency() &&
               !HasUnknownSchedulingAlignmentDependency();
    }
    // Final SPIR-V is publishable only when every scanner-synthesized input
    // consulted by lowering is known. This intentionally does not certify a
    // template artifact; it is the stricter exact-final-module boundary.
    [[nodiscard]] bool HasCompleteFinalModuleContract() const noexcept {
        return HasStateIndependentTemplateContract();
    }

    // Match GenericEnvironment::Analyze() through the shared program-identity rule.
    // Scanner CFG attempts can start at candidate offsets and visit blocks in a
    // different order from live CFG, so read_lowest/read_highest are not identity.
    u64 CalculateHash() const {
        if (const std::optional<u64> identity = ComputeMaxwellProgramIdentity(code)) {
            return *identity;
        }
        if (read_highest >= read_lowest && read_highest >= code_lowest) {
            const u32 start_i = (read_lowest - code_lowest) / sizeof(u64);
            const u32 end_i = (read_highest - code_lowest) / sizeof(u64);
            const std::span<const u64> read_span =
                std::span<const u64>{code}.subspan(start_i, end_i - start_i + 1);
            // read_span is constructed from bounded vector indices above, so this
            // identity span is known-valid. Keep the fallback defensive in case
            // this environment is ever fed malformed serialized bounds.
            if (const auto identity =
                    ComputeMaxwellProgramIdentity(read_span, read_span.size_bytes())) {
                return *identity;
            }
        }
        // No usable read range (e.g. translation never actually ran for this
        // instance) — this can'''t match a live Analyze() hash regardless of how
        // it'''s computed, so it only needs to be internally consistent for this
        // scan'''s own unique_hashes_/dedup bookkeeping.
        return Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(u64));
    }

private:
    std::vector<u64> code;
    std::unordered_map<u32, Shader::TextureType> texture_types;
    std::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    Shader::LogicalTextureSlots logical_texture_slots;
    Shader::LogicalTextureHandles logical_texture_handles;
    u32 local_memory_size;
    u32 shared_memory_size;
    u32 texture_bound;
    std::array<u32, 3> workgroup_size;
    u32 code_lowest;
    u32 read_lowest = ~0u;
    u32 read_highest = 0;
    u32 viewport_transform_state_ = 1u;
    // Scanner has no real draw state. These flags deliberately distinguish an
    // unknown zero returned by this environment from a real observed zero and
    // from no dependency at all. They are telemetry/eligibility inputs only;
    // no caller may treat zero as a certified final-module contract.
    bool unknown_cbuf_reads{};
    bool unknown_cbuf_sizes{};
    bool unknown_texture_shape_reads{};
    bool unknown_viewport_state{};
    mutable bool unknown_texture_binding_state{};
    mutable bool unknown_proprietary_driver_state{};
    mutable bool unknown_compute_launch_state{};
    mutable bool unknown_hle_macro_state{};
    mutable bool unknown_interface_state{};
    bool unknown_scheduling_alignment{};
};

} // namespace VideoCommon
