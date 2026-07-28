// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "common/cityhash.h"
#include "common/common_types.h"
#include "shader_recompiler/varying_state.h"

namespace Shader {

enum class AttributeType : u8 {
    Float,
    SignedInt,
    UnsignedInt,
    SignedScaled,
    UnsignedScaled,
    Disabled,
};

enum class InputTopology {
    Points,
    Lines,
    LinesAdjacency,
    Triangles,
    TrianglesAdjacency,
};

namespace InputTopologyVertices {
    // Lookup table for vertex counts - faster than switch statement
    inline constexpr std::array<u32, 5> vertex_counts = {
        1, // Points
        2, // Lines
        4, // LinesAdjacency
        3, // Triangles
        6, // TrianglesAdjacency
    };

    // Force compile-time evaluation when possible
    inline constexpr u32 vertices(InputTopology input_topology) {
        return vertex_counts[static_cast<std::size_t>(input_topology)];
    }
}

enum class CompareFunction {
    Never,
    Less,
    Equal,
    LessThanEqual,
    Greater,
    NotEqual,
    GreaterThanEqual,
    Always,
};

enum class FragmentOutputType : u8 {
    Float,
    SignedInt,
    UnsignedInt,
};

enum class TessPrimitive {
    Isolines,
    Triangles,
    Quads,
};

enum class TessSpacing {
    Equal,
    FractionalOdd,
    FractionalEven,
};

struct TransformFeedbackVarying {
    u32 buffer{};
    u32 stride{};
    u32 offset{};
    u32 components{};
};

struct RuntimeInfo {
    std::array<AttributeType, 32> generic_input_types{};
    VaryingState previous_stage_stores;
    std::map<IR::Attribute, IR::Attribute> previous_stage_legacy_stores_mapping;

    bool convert_depth_mode{};
    bool force_early_z{};

    TessPrimitive tess_primitive{};
    TessSpacing tess_spacing{};
    bool tess_clockwise{};

    InputTopology input_topology{};

    std::optional<float> fixed_state_point_size;
    std::optional<CompareFunction> alpha_test_func;
    float alpha_test_reference{};
    bool alpha_to_coverage_enabled{};
    std::array<FragmentOutputType, 8> frag_color_types{};

    /// Static Y negate value
    bool y_negate{};
    /// Use storage buffers instead of global pointers on GLASM
    bool glasm_use_storage_buffers{};

    /// Transform feedback state for each varying
    std::array<TransformFeedbackVarying, 256> xfb_varyings{};
    u32 xfb_count{0};

    [[nodiscard]] u64 Hash() const noexcept {
        u64 hash = 0;
        auto hash_combine = [&hash](u64 val) {
            hash ^= val + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };

        hash_combine(Common::CityHash64(reinterpret_cast<const char*>(generic_input_types.data()), generic_input_types.size() * sizeof(AttributeType)));
        hash_combine(std::hash<std::bitset<512>>{}(previous_stage_stores.mask));
        for (const auto& [key, value] : previous_stage_legacy_stores_mapping) {
            hash_combine((static_cast<u64>(key) << 32) | static_cast<u64>(value));
        }
        hash_combine(convert_depth_mode);
        hash_combine(force_early_z);
        hash_combine(static_cast<u64>(tess_primitive));
        hash_combine(static_cast<u64>(tess_spacing));
        hash_combine(tess_clockwise);
        hash_combine(static_cast<u64>(input_topology));
        if (fixed_state_point_size) {
            u32 val;
            std::memcpy(&val, &*fixed_state_point_size, sizeof(val));
            hash_combine(val);
        }
        if (alpha_test_func) {
            hash_combine(static_cast<u64>(*alpha_test_func) + 1);
        }
        u32 alpha_ref;
        std::memcpy(&alpha_ref, &alpha_test_reference, sizeof(alpha_ref));
        hash_combine(alpha_ref);
        hash_combine(alpha_to_coverage_enabled);
        hash_combine(Common::CityHash64(reinterpret_cast<const char*>(frag_color_types.data()), frag_color_types.size() * sizeof(FragmentOutputType)));
        hash_combine(y_negate);
        hash_combine(glasm_use_storage_buffers);
        hash_combine(xfb_count);
        if (xfb_count > 0) {
            hash_combine(Common::CityHash64(reinterpret_cast<const char*>(xfb_varyings.data()), xfb_count * sizeof(TransformFeedbackVarying)));
        }

        return hash;
    }
};

} // namespace Shader
