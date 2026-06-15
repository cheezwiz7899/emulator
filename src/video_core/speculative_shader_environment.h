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

    // Constructor used by pre-cache scanner
    explicit SpeculativeShaderEnvironment(std::vector<u64> code_, Shader::Stage stage_, u32 local_memory_size_, Shader::ProgramHeader sph_)
        : SpeculativeShaderEnvironment(std::move(code_), 0, stage_, local_memory_size_, 0, {1u, 1u, 1u}, 1u, sph_, static_cast<u32>(sizeof(Shader::ProgramHeader))) {
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
        throw Shader::Exception("SpeculativeShaderEnvironment: ReadCbufValue not supported");
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
    u32 ReadViewportTransformState() override { return 1u; }
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

    u64 CalculateHash() const {
        static constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
        static constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;

        // Try to find self-branch to match GenericEnvironment::Analyze() behavior
        for (size_t i = 0; i < code.size(); ++i) {
            if (code[i] == SELF_BRANCH_A || code[i] == SELF_BRANCH_B) {
                return Common::CityHash64(reinterpret_cast<const char*>(code.data()), (i + 1) * sizeof(u64));
            }
        }

        // Fallback to hashing read instructions
        if (read_highest >= read_lowest && read_highest >= code_lowest) {
            u32 start_i = (read_lowest - code_lowest) / 8;
            u32 end_i = (read_highest - code_lowest) / 8;
            size_t size = (end_i - start_i + 1) * sizeof(u64);
            return Common::CityHash64(reinterpret_cast<const char*>(&code[start_i]), size);
        }

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
};

} // namespace VideoCommon
