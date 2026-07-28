// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include <optional>

#include "common/cityhash.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/program_header.h"

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

    // Constructor used by pre-cache scanner. code_offset_in_program_ is 0, not
    // sizeof(ProgramHeader): the caller is expected to include the SPH as a
    // prefix of code_ (see the call site in main.cpp), matching how a live
    // GraphicsEnvironment's code[] always starts at the SPH's own address
    // (start_address IS the SPH address there — see its constructor, which
    // reads the SPH from program_base + start_address). Passing
    // sizeof(ProgramHeader) here used to silently exclude the SPH from
    // code_lowest regardless of what code_ actually contained, which is what
    // made CalculateHash() below disagree with GenericEnvironment::Analyze()
    // (the common-case live hash, which does include the SPH) even after the
    // read_lowest/read_highest fix — it was matching the rare fallback path
    // instead, since that one also excludes the SPH on the live side.
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
        return 0u;
    }
    u32 ReadCbufSize(u32 i) override { return i < 18 ? 65536u : 0u; }
    Shader::TextureType ReadTextureType(u32 handle) override {
        texture_types.emplace(handle, Shader::TextureType::Color2D);
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        texture_pixel_formats.emplace(handle, Shader::TexturePixelFormat::A8B8G8R8_UNORM);
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return viewport_transform_state_; }
    // The real value is GPU register state a speculative (no live draw) translation
    // has no way to observe. Unlike cbuf content or non-leading binding state, this
    // is a genuine 2-way fork (see PositionPass() in ir_opt/position_pass.cpp) with
    // no other possible values, so callers can cheaply try both by translating twice
    // with this set differently each time, instead of betting on a single guess.
    // Defaults to 1 (the prior hardcoded behavior) so existing call sites that never
    // call this are unaffected.
    void SetViewportTransformState(u32 value) noexcept { viewport_transform_state_ = value; }
    u32 TextureBoundBuffer() const override { return texture_bound; }
    u32 LocalMemorySize() const override { return local_memory_size; }
    u32 SharedMemorySize() const override { return shared_memory_size; }
    std::array<u32, 3> WorkgroupSize() const override { return workgroup_size; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}

    const std::unordered_map<u32, Shader::TextureType>& CapturedTextureTypes() const noexcept {
        return texture_types;
    }

    const std::unordered_map<u32, Shader::TexturePixelFormat>& CapturedTexturePixelFormats() const noexcept {
        return texture_pixel_formats;
    }

    // Must match GenericEnvironment::Analyze() EXACTLY, or unique_hash never agrees
    // with what a live lookup computes for the same shader — see the doc comment
    // on this class'''s declaration in main.cpp'''s caller for the full story. Two
    // things Analyze() does that this now mirrors:
    //  1. Hashes from the real entry point onward — read_lowest here, since that'''s
    //     the first address ReadInstruction() actually saw, i.e. wherever THIS
    //     translation attempt really started. A prior version of this function
    //     scanned code[] from index 0, which is a fixed position (code_lowest)
    //     with no relationship to where any given shader'''s real entry is.
    //  2. Stops BEFORE the self-branch terminator, not after it. Analyze() hashes
    //     `size` bytes where `size` is the self-branch'''s byte offset from
    //     start_address — i.e. the self-branch itself is excluded. read_highest,
    //     once CFG/TranslateProgram has run, IS the self-branch'''s address (it has
    //     to be read to be recognized as one), so the upper index below is
    //     exclusive rather than +1.
    u64 CalculateHash() const {
        if (read_highest >= read_lowest && read_highest >= code_lowest) {
            const u32 start_i = (read_lowest - code_lowest) / 8;
            const u32 end_i = (read_highest - code_lowest) / 8; // self-branch'''s index — excluded
            const size_t size = (end_i - start_i) * sizeof(u64);
            return Common::CityHash64(reinterpret_cast<const char*>(code.data() + start_i), size);
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
    u32 local_memory_size;
    u32 shared_memory_size;
    u32 texture_bound;
    std::array<u32, 3> workgroup_size;
    u32 code_lowest;
    u32 read_lowest = ~0u;
    u32 read_highest = 0;
    u32 viewport_transform_state_ = 1u;
};

} // namespace VideoCommon
