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
#include "shader_recompiler/stage.h"
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

    // Same fold as Hash(), but restricted per-stage to only the fields that stage's
    // SPIR-V emission actually reads — see the citation for each block below. Used
    // exclusively as the spirv_cache "core" runtime_key component (VideoCommon::
    // ComputeBindingKey/FoldViewportTransformState/FoldBindingKey handle the rest);
    // Hash() itself is left untouched and unused elsewhere in the codebase, so this is
    // purely additive. Every exclusion here is backed by a direct grep of every
    // `runtime_info.<field>` read site under shader_recompiler/backend/spirv/ and
    // shader_recompiler/frontend/maxwell/translate_program.cpp (the only two places
    // that ever consume a RuntimeInfo) — not inferred. Where a field is provably
    // read only under one stage's case/guard, it's included only for that stage;
    // fields MakeRuntimeInfo() (vk_pipeline_cache.cpp) itself only ever populates for
    // one or two stages in the first place are already an implicit no-op everywhere
    // else, so excluding them here doesn't change what gets hashed today — it just
    // documents the dependency explicitly and removes the latent risk of a future
    // MakeRuntimeInfo change silently adding entropy nothing reads. When a read site
    // isn't stage-gated at all (y_negate) or is gated on the CURRENT shader's own
    // IR/info rather than on `stage` (previous_stage_legacy_stores_mapping is gated on
    // info.loads.Legacy(), not on stage — see translate_program.cpp), it's kept
    // unconditionally rather than guessed at.
    [[nodiscard]] u64 SpirvRelevantHash(Stage stage) const noexcept {
        u64 hash = 0;
        auto hash_combine = [&hash](u64 val) {
            hash ^= val + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };

        // generic_input_types: real per-pipeline attribute-format state for every
        // stage that has inputs at all — VertexB reads it for actual vertex-buffer
        // attribute types, every other graphics stage reads it (alongside
        // previous_stage_stores below) to wire up inputs matching whatever the
        // preceding stage declared. (spirv_emit_context.cpp DefineInputs)
        hash_combine(Common::CityHash64(reinterpret_cast<const char*>(generic_input_types.data()),
                                        generic_input_types.size() * sizeof(AttributeType)));

        // previous_stage_stores / previous_stage_legacy_stores_mapping: also read by
        // DefineInputs, but skipped for VertexB specifically — MakeRuntimeInfo sets
        // previous_stage_stores.mask to an all-ones sentinel ("no restriction")
        // whenever there's no previous program, which is always true for VertexB, so
        // it's a fixed constant there and contributes nothing. For every other stage
        // it reflects the ACTUAL preceding program's output layout and genuinely
        // varies per real pipeline.
        if (stage != Stage::VertexB) {
            hash_combine(std::hash<std::bitset<512>>{}(previous_stage_stores.mask));
            for (const auto& [key, value] : previous_stage_legacy_stores_mapping) {
                hash_combine((static_cast<u64>(key) << 32) | static_cast<u64>(value));
            }
        }

        // convert_depth_mode / fixed_state_point_size / xfb_count / xfb_varyings:
        // MakeRuntimeInfo only ever sets these for VertexB and Geometry (Fragment and
        // TessellationEval fall through to `default:` for all of them).
        if (stage == Stage::VertexB || stage == Stage::Geometry) {
            hash_combine(convert_depth_mode);
            if (fixed_state_point_size) {
                u32 val;
                std::memcpy(&val, &*fixed_state_point_size, sizeof(val));
                hash_combine(val);
            }
            hash_combine(xfb_count);
            if (xfb_count > 0) {
                hash_combine(Common::CityHash64(reinterpret_cast<const char*>(xfb_varyings.data()),
                                                xfb_count * sizeof(TransformFeedbackVarying)));
            }
        }

        // tess_primitive / tess_spacing / tess_clockwise: read only inside
        // `case Stage::TessellationEval:` in DefineEntryPoint (emit_spirv.cpp).
        if (stage == Stage::TessellationEval) {
            hash_combine(static_cast<u64>(tess_primitive));
            hash_combine(static_cast<u64>(tess_spacing));
            hash_combine(tess_clockwise);
        }

        // input_topology: read only inside `case Stage::Geometry:` — both the
        // execution-mode switch in DefineEntryPoint and the InputTopologyVertices
        // lookup in emit_spirv_context_get_set.cpp. Confirmed NOT read for VertexB,
        // TessellationControl, TessellationEval, or Fragment, despite
        // MakeRuntimeInfo computing it unconditionally for every stage from
        // key.state.topology — real per-draw variance, zero effect outside Geometry.
        if (stage == Stage::Geometry) {
            hash_combine(static_cast<u64>(input_topology));
        }

        // force_early_z: computed unconditionally by MakeRuntimeInfo from real,
        // per-pipeline GPU state (key.state.early_z), but only ever read inside
        // `case Stage::Fragment:` (the EarlyFragmentTests execution mode). Same shape
        // of gap as input_topology above.
        if (stage == Stage::Fragment) {
            hash_combine(force_early_z);
        }

        // alpha_test_func / alpha_test_reference / alpha_to_coverage_enabled /
        // frag_color_types: read only in Fragment-stage code paths
        // (emit_spirv_special.cpp, emit_spirv_context_get_set.cpp), and
        // MakeRuntimeInfo only ever sets them for Stage::Fragment in the first
        // place — like convert_depth_mode above, already-constant elsewhere rather
        // than newly-excluded live entropy.
        if (stage == Stage::Fragment) {
            if (alpha_test_func) {
                hash_combine(static_cast<u64>(*alpha_test_func) + 1);
            }
            u32 alpha_ref;
            std::memcpy(&alpha_ref, &alpha_test_reference, sizeof(alpha_ref));
            hash_combine(alpha_ref);
            hash_combine(alpha_to_coverage_enabled);
            hash_combine(Common::CityHash64(reinterpret_cast<const char*>(frag_color_types.data()),
                                            frag_color_types.size() * sizeof(FragmentOutputType)));
        }

        // y_negate: NOT stage-gated at its read site (EmitYDirection has no switch
        // on ctx.stage) — fires whenever the shader's own IR contains a Y-direction
        // query, which could in principle be any stage. Kept unconditional.
        hash_combine(y_negate);

        // glasm_use_storage_buffers deliberately omitted: not read anywhere under
        // backend/spirv/ or frontend/maxwell/ at all (it's a GLASM-backend-only
        // field, and that backend never runs on the Vulkan/SPIR-V path this hash
        // serves) — and it's never set to true anywhere in the codebase regardless,
        // so this is provably a no-op today either way.

        return hash;
    }
};

} // namespace Shader
