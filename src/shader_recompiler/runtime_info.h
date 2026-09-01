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
#include "common/settings.h"
#include "shader_recompiler/shader_info.h"
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

        // y_negate: Phase 5 converted this to a real SPIR-V spec constant
        // (Shader::kYNegateSpecId, spirv_emit_context.cpp's DefineRuntimeStateSpecConstants)
        // instead of a value baked at translation time -- EmitYDirection
        // (emit_spirv_context_get_set.cpp) no longer reads this field at all. The emitted
        // SPIR-V is now identical regardless of the real y_negate value, so it must not stay
        // in this hash: doing so would still fragment the cache on a field that no longer
        // makes two entries' SPIR-V different from each other. This is exactly the "changes
        // what the key means" category (SPIRV_CACHE_VERSION bump required, not just an
        // ordinary edit) -- see the version history below. The real per-draw value is
        // resolved separately, at pipeline-creation time (vk_graphics_pipeline.cpp), via a
        // VkSpecializationMapEntry keyed on the same SpecId.

        // glasm_use_storage_buffers deliberately omitted: not read anywhere under
        // backend/spirv/ or frontend/maxwell/ at all (it's a GLASM-backend-only
        // field, and that backend never runs on the Vulkan/SPIR-V path this hash
        // serves) — and it's never set to true anywhere in the codebase regardless,
        // so this is provably a no-op today either way.

        return hash;
    }

    // Phase 5 "free wins" (handoff_13/handoff_15): deliberate defaults for the fields
    // speculative construction (scanner in citron/main.cpp, GPL live path in
    // vk_pipeline_cache.cpp) has no real per-draw signal for. Both sites zero-init
    // RuntimeInfo and only set what they can actually derive (previous_stage_stores,
    // input_topology, frag_color_types) — everything else silently inherits C++
    // zero-init, which is often *not* the real common-case value. Called once per
    // speculative translation, after those site-specific fields are set and before
    // ConvertLegacyToGeneric/EmitSPIRV.
    //
    // Zero architecture risk either way: a wrong guess here is exactly as harmless as
    // today's accidental zero-init was — a non-matching real draw just falls through to
    // an ordinary (non-speculative) translation, same as always. The only thing this
    // changes is the odds of a speculative entry actually matching a real one.
    //
    // Confidence is graded per field below, not uniform — some of this is derived
    // directly from the real MakeRuntimeInfo()/fixed_pipeline_state.cpp path, some is a
    // documented best guess with no measurement behind it yet.
    void ApplySpeculativeDefaults(Stage stage, const Info& info) {
        // generic_input_types: Set by scanner? No / Set by GPL path? No (handoff_13's field
        // audit) -- left at zero-init (every slot AttributeType::Disabled) until now. This
        // isn't just a wrong-value guess the way the fields below are: DefineInputs
        // (spirv_emit_context.cpp) skips declaring the Input variable ENTIRELY for a
        // Disabled slot ("if (input_type == AttributeType::Disabled) continue;"), so any
        // shader that actually loads a generic attribute got a speculatively-translated
        // module MISSING that Input variable altogether -- not a different
        // SpirvRelevantHash by chance, a guaranteed-different one, unconditionally, for
        // every shader that loads any generic attribute at all. Real measurement (a real
        // two-session gameplay log, RecordGenericInputTypesCardinalityDiagnostic) puts the
        // field at 93.6%-95.9% single-observed-state per shader -- close to Phase 3's own
        // 94.5% finding for the whole core-runtime-state -- so for the large majority of
        // shaders, any correct-enough guess would match every real draw they ever see.
        // AttributeType::Float for every slot this shader's own IR actually loads
        // (info.loads.Generic(index)) is that guess: modern vertex/varying data is
        // float-typed in the large majority of real cases (position/normal/UV/color-as-
        // float all common; packed-integer formats like vertex colors as UNORM8x4 or bone
        // indices as UInt are the minority this won't catch). Also gated on
        // previous_stage_stores.Generic(index) -- already populated with real (or best-
        // available) data by both speculative call sites before this runs -- matching
        // DefineInputs' own two-part gate exactly (minus the Disabled check itself, which
        // is what this sets): guessing Float for a slot the preceding stage doesn't even
        // store would create a mismatch where zero-init's Disabled was actually correct.
        // For VertexB, previous_stage_stores is always the all-ones "no restriction"
        // sentinel (no previous program ever exists there), so this reduces to just the
        // loads check, same as the real gate does.
        //
        // A wrong concrete type is still a real mismatch, not glossed over: GetAttributeType
        // (spirv_emit_context.cpp) picks the actual SPIR-V type from this field, and unlike
        // y_negate's plain value, a spec constant can't paper over a different
        // OpTypePointer -- SPIR-V types are resolved at translation time, not
        // pipeline-creation time, so generic_input_types isn't a spec-constant candidate
        // the way y_negate was, cardinality result or not. This stops the *guaranteed* miss
        // Disabled caused, which was strictly worse: a structural absence, not just a
        // wrong-typed presence.
        for (size_t index = 0; index < generic_input_types.size(); ++index) {
            if (previous_stage_stores.Generic(index) && info.loads.Generic(index)) {
                generic_input_types[index] = AttributeType::Float;
            }
        }

        if (stage == Stage::VertexB || stage == Stage::Geometry) {
            // gl_ndc = (regs.depth_mode == DepthMode::MinusOneToOne) in the real path
            // (vk_pipeline_cache.cpp's MakeRuntimeInfo), and DepthMode::MinusOneToOne is
            // enum value 0 (maxwell_3d.h) — the same "0 is the register's power-on-reset
            // value" convention the rest of this codebase's fixed-function state already
            // follows elsewhere. Reasoned from that convention, not from a real session
            // measurement the way alpha_test_func below is — worth revisiting if that
            // assumption about NVN's own depth_mode default ever turns out wrong.
            convert_depth_mode = true;
        }
        if (stage == Stage::TessellationEval) {
            // No reset-state or Settings-driven argument to lean on here the way
            // convert_depth_mode/alpha_test_func have — a shader only reaches this stage
            // at all when a game has actively opted into tessellation, so "what does an
            // unused register default to" isn't the right question. Picked to match each
            // enum's own value-0 entry for internal consistency, not because Triangles/
            // Equal/counter-clockwise is known to be the common real choice. Lowest-
            // confidence guesses in this pass — first candidates to revise if real
            // tessellation measurement ever happens.
            tess_primitive = TessPrimitive::Triangles;
            tess_spacing = TessSpacing::Equal;
            tess_clockwise = false;
        }
        if (stage == Stage::Fragment) {
            // MakeRuntimeInfo sets alpha_test_func unconditionally — UNLESS
            // Settings::IsGPULevelLow(), which skips it entirely as a deliberate
            // accuracy/perf tradeoff — and even when alpha testing is disabled at the HW
            // level, fixed_pipeline_state.cpp packs CompareFunction::Always for it, not an
            // unset sentinel (alpha_test_enabled == 0 selects Always_GL, not "no value").
            // Leaving this nullopt — C++ zero-init — is therefore the wrong default for
            // the common (alpha test disabled) case under normal settings: it can never
            // match a real draw's RuntimeInfo, which is populated in that exact common
            // case. Mirrors the real path's own Settings check so this doesn't regress
            // under Low GPU accuracy, where the real path also leaves it nullopt.
            if (!Settings::IsGPULevelLow()) {
                alpha_test_func = CompareFunction::Always;
            }
            // alpha_test_reference's zero-init (0.0f) is left as-is: its value is moot
            // whenever func is Always or unset, and there's no real-state signal for what
            // a non-Always reference value would even be.
        }
        // force_early_z, alpha_to_coverage_enabled, fixed_state_point_size, xfb_count:
        // left at C++ zero-init deliberately, not by omission. Each reasoned as the real
        // common case from its source in fixed_pipeline_state.cpp/MakeRuntimeInfo:
        // mandated_early_z is an opt-in hint most shaders never set; alpha-to-coverage is
        // an uncommon MSAA technique; most draws are neither point-primitive
        // (fixed_state_point_size only applies when topology is Points, or a Geometry
        // shader's output_topology is PointList) nor transform-feedback (xfb_count) draws.
    }
};

} // namespace Shader
