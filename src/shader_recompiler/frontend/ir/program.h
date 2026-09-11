// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>

#include "shader_recompiler/frontend/ir/abstract_syntax_list.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/shader_info.h"
#include "shader_recompiler/stage.h"

namespace Shader::IR {

enum class FrontendDependency : u32 {
    Instruction = 1U << 0,
    ConstantBuffer = 1U << 1,
    ConstantBufferSize = 1U << 2,
    TextureType = 1U << 3,
    TexturePixelFormat = 1U << 4,
    TextureIntegerFormat = 1U << 5,
    ViewportTransform = 1U << 6,
    TextureBinding = 1U << 7,
    LocalMemory = 1U << 8,
    ComputeLaunch = 1U << 9,
    HLEMacro = 1U << 10,
    ConstantBufferReplacement = 1U << 11,
    GeometryPassthrough = 1U << 12,
    ProprietaryDriver = 1U << 13,
};

struct FrontendDependencyManifest {
    u32 flags{};

    [[nodiscard]] bool Uses(FrontendDependency dependency) const noexcept {
        return (flags & static_cast<u32>(dependency)) != 0;
    }
    bool operator==(const FrontendDependencyManifest&) const noexcept = default;
};

struct Program {
    AbstractSyntaxList syntax_list;
    BlockList blocks;
    BlockList post_order_blocks;
    Info info;
    Stage stage{};
    std::array<u32, 3> workgroup_size{};
    OutputTopology output_topology{};
    u32 output_vertices{};
    u32 invocations{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    bool is_geometry_passthrough{};
    FrontendDependencyManifest frontend_dependencies{};
};

[[nodiscard]] std::string DumpProgram(const Program& program);

} // namespace Shader::IR
