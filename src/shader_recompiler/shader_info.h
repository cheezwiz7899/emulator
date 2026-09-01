// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <map>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/type.h"
#include "shader_recompiler/varying_state.h"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Shader {

enum class ReplaceConstant : u32 {
    BaseInstance,
    BaseVertex,
    DrawID,
};

enum class TextureType : u32 {
    Color1D,
    ColorArray1D,
    Color2D,
    ColorArray2D,
    Color3D,
    ColorCube,
    ColorArrayCube,
    Buffer,
    Color2DRect,
};
constexpr u32 NUM_TEXTURE_TYPES = 9;

enum class TexturePixelFormat {
    A8B8G8R8_UNORM,
    A8B8G8R8_SNORM,
    A8B8G8R8_SINT,
    A8B8G8R8_UINT,
    R5G6B5_UNORM,
    B5G6R5_UNORM,
    A1R5G5B5_UNORM,
    A2B10G10R10_UNORM,
    A2B10G10R10_UINT,
    A2R10G10B10_UNORM,
    A1B5G5R5_UNORM,
    A5B5G5R1_UNORM,
    R8_UNORM,
    R8_SNORM,
    R8_SINT,
    R8_UINT,
    R16G16B16A16_FLOAT,
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R16G16B16A16_SINT,
    R16G16B16A16_UINT,
    B10G11R11_FLOAT,
    R32G32B32A32_UINT,
    BC1_RGBA_UNORM,
    BC2_UNORM,
    BC3_UNORM,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC7_UNORM,
    BC6H_UFLOAT,
    BC6H_SFLOAT,
    ASTC_2D_4X4_UNORM,
    B8G8R8A8_UNORM,
    R32G32B32A32_FLOAT,
    R32G32B32A32_SINT,
    R32G32_FLOAT,
    R32G32_SINT,
    R32_FLOAT,
    R16_FLOAT,
    R16_UNORM,
    R16_SNORM,
    R16_UINT,
    R16_SINT,
    R16G16_UNORM,
    R16G16_FLOAT,
    R16G16_UINT,
    R16G16_SINT,
    R16G16_SNORM,
    R32G32B32_FLOAT,
    A8B8G8R8_SRGB,
    R8G8_UNORM,
    R8G8_SNORM,
    R8G8_SINT,
    R8G8_UINT,
    R32G32_UINT,
    R16G16B16X16_FLOAT,
    R32_UINT,
    R32_SINT,
    ASTC_2D_8X8_UNORM,
    ASTC_2D_8X5_UNORM,
    ASTC_2D_5X4_UNORM,
    B8G8R8A8_SRGB,
    BC1_RGBA_SRGB,
    BC2_SRGB,
    BC3_SRGB,
    BC7_SRGB,
    A4B4G4R4_UNORM,
    G4R4_UNORM,
    ASTC_2D_4X4_SRGB,
    ASTC_2D_8X8_SRGB,
    ASTC_2D_8X5_SRGB,
    ASTC_2D_5X4_SRGB,
    ASTC_2D_5X5_UNORM,
    ASTC_2D_5X5_SRGB,
    ASTC_2D_10X8_UNORM,
    ASTC_2D_10X8_SRGB,
    ASTC_2D_6X6_UNORM,
    ASTC_2D_6X6_SRGB,
    ASTC_2D_10X6_UNORM,
    ASTC_2D_10X6_SRGB,
    ASTC_2D_10X5_UNORM,
    ASTC_2D_10X5_SRGB,
    ASTC_2D_10X10_UNORM,
    ASTC_2D_10X10_SRGB,
    ASTC_2D_12X10_UNORM,
    ASTC_2D_12X10_SRGB,
    ASTC_2D_12X12_UNORM,
    ASTC_2D_12X12_SRGB,
    ASTC_2D_8X6_UNORM,
    ASTC_2D_8X6_SRGB,
    ASTC_2D_6X5_UNORM,
    ASTC_2D_6X5_SRGB,
    E5B9G9R9_FLOAT,
    D32_FLOAT,
    D16_UNORM,
    X8_D24_UNORM,
    S8_UINT,
    D24_UNORM_S8_UINT,
    S8_UINT_D24_UNORM,
    D32_FLOAT_S8_UINT,
};

enum class ImageFormat : u32 {
    Typeless,
    R8_UINT,
    R8_SINT,
    R16_UINT,
    R16_SINT,
    R32_UINT,
    R32_SINT,
    R32_SFLOAT,
    R32G32_UINT,
    R32G32_SINT,
    R32G32_SFLOAT,
    R32G32B32A32_UINT,
    R32G32B32A32_SINT,
    R32G32B32A32_SFLOAT,
};

enum class Interpolation {
    Smooth,
    Flat,
    NoPerspective,
};

struct ConstantBufferDescriptor {
    u32 index;
    u32 count;

    auto operator<=>(const ConstantBufferDescriptor&) const = default;
};

struct StorageBufferDescriptor {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    bool is_written;

    auto operator<=>(const StorageBufferDescriptor&) const = default;
};

struct TextureBufferDescriptor {
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;

    auto operator<=>(const TextureBufferDescriptor&) const = default;
};
using TextureBufferDescriptors = boost::container::small_vector<TextureBufferDescriptor, 6>;

struct ImageBufferDescriptor {
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;

    auto operator<=>(const ImageBufferDescriptor&) const = default;
};
using ImageBufferDescriptors = boost::container::small_vector<ImageBufferDescriptor, 2>;

struct TextureDescriptor {
    TextureType type;
    bool is_depth;
    bool is_multisample;
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;
    // Phase 4 narrow prototype (specialization-constant texture-type resolution) -- set when
    // (cbuf_index, cbuf_offset) matches a known entry in Shader::ActivePhase4PrototypeSlots()
    // (environment.h), currently just the one real, empirically-identified
    // (cbuf_index=2, cbuf_offset=192) pattern from handoff_04's investigation
    // (Color2D/ColorArray2D variance, 12 real TotK shaders). Still a hand-curated list, not
    // an automatic mechanism -- see the generalization handoffs for what's deliberately not
    // attempted yet. Both fields are left out of the operator<=>/Add() dedup predicate below:
    // phase4_prototype_slot_id is a pure function of cbuf_index/cbuf_offset, which the
    // predicate already compares, so neither field changes what makes two descriptors "the
    // same" for caching purposes -- only how this one gets emitted.
    bool phase4_prototype_polymorphic = false;
    // Index into Shader::ActivePhase4PrototypeSlots() (environment.h) -- also this slot's SpecId
    // (spirv_emit_context.cpp) and its bit position in
    // GraphicsPipelineCacheKey::phase4_prototype_needs_array_variant. Only meaningful when
    // phase4_prototype_polymorphic is true; 0 otherwise (harmless, since nothing reads it
    // unless the flag is set).
    u32 phase4_prototype_slot_id = 0;

    auto operator<=>(const TextureDescriptor&) const = default;
};
using TextureDescriptors = boost::container::small_vector<TextureDescriptor, 12>;

// Number of consecutive bindings/descriptor-slots this texture descriptor actually needs --
// 1 for an ordinary descriptor, 2 for a Phase 4 prototype polymorphic slot (the canonical
// variant plus its one alternate; still 2-way only, see the generalization handoffs for why
// N-way isn't attempted yet). Single source of truth for the three sites that each walk
// info.texture_descriptors and previously reimplemented "if polymorphic, count 2"
// independently -- correct as long as all three stayed in sync by hand, which is exactly the
// kind of fragility this mechanism's real crash history (three call sites, one of them missed
// on the first pass) argues against leaving as-is: EmitContext::DefineTextures
// (spirv_emit_context.cpp, SPIR-V binding decorations), DescriptorLayoutBuilder::Add and
// PushImageDescriptors (pipeline_helper.h, the host-side VkDescriptorSetLayout/update-template
// and per-draw descriptor-data paths). All three now consult this instead of each carrying
// their own copy of the rule.
[[nodiscard]] constexpr u32 Phase4PrototypeBindingCount(const TextureDescriptor& desc) noexcept {
    return desc.phase4_prototype_polymorphic ? 2 : 1;
}

struct ImageDescriptor {
    TextureType type;
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;

    auto operator<=>(const ImageDescriptor&) const = default;
};
using ImageDescriptors = boost::container::small_vector<ImageDescriptor, 4>;

struct Info {
    static constexpr size_t MAX_INDIRECT_CBUFS{14};
    static constexpr size_t MAX_CBUFS{18};
    static constexpr size_t MAX_SSBOS{32};

    bool uses_workgroup_id{};
    bool uses_local_invocation_id{};
    bool uses_invocation_id{};
    bool uses_invocation_info{};
    bool uses_sample_id{};
    bool uses_is_helper_invocation{};
    // Phase 5: whether this shader's IR contains a YDirection query (runtime_info.h /
    // EmitYDirection). Gates declaring the y_negate spec constant (spirv_emit_context.cpp)
    // to shaders that actually consume it.
    bool uses_y_direction{};
    bool uses_subgroup_invocation_id{};
    bool uses_subgroup_shuffles{};
    std::array<bool, 30> uses_patches{};

    std::array<Interpolation, 32> interpolation{};
    VaryingState loads;
    VaryingState stores;
    VaryingState passthrough;

    std::map<IR::Attribute, IR::Attribute> legacy_stores_mapping;

    bool loads_indexed_attributes{};

    std::array<bool, 8> stores_frag_color{};
    bool stores_sample_mask{};
    bool stores_frag_depth{};

    bool stores_tess_level_outer{};
    bool stores_tess_level_inner{};

    bool stores_indexed_attributes{};

    bool stores_global_memory{};
    bool uses_local_memory{};

    bool uses_fp16{};
    bool uses_fp64{};
    bool uses_fp16_denorms_flush{};
    bool uses_fp16_denorms_preserve{};
    bool uses_fp32_denorms_flush{};
    bool uses_fp32_denorms_preserve{};
    bool uses_int8{};
    bool uses_int16{};
    bool uses_int64{};
    bool uses_image_1d{};
    bool uses_sampled_1d{};
    bool uses_sparse_residency{};
    bool uses_demote_to_helper_invocation{};
    bool uses_subgroup_vote{};
    bool uses_subgroup_mask{};
    bool uses_fswzadd{};
    bool uses_derivatives{};
    bool uses_typeless_image_reads{};
    bool uses_typeless_image_writes{};
    bool uses_image_buffers{};
    bool uses_shared_increment{};
    bool uses_shared_decrement{};
    bool uses_global_increment{};
    bool uses_global_decrement{};
    bool uses_atomic_f32_add{};
    bool uses_atomic_f16x2_add{};
    bool uses_atomic_f16x2_min{};
    bool uses_atomic_f16x2_max{};
    bool uses_atomic_f32x2_add{};
    bool uses_atomic_f32x2_min{};
    bool uses_atomic_f32x2_max{};
    bool uses_atomic_s32_min{};
    bool uses_atomic_s32_max{};
    bool uses_int64_bit_atomics{};
    bool uses_global_memory{};
    bool uses_atomic_image_u32{};
    bool uses_shadow_lod{};
    bool uses_rescaling_uniform{};
    bool uses_cbuf_indirect{};
    bool uses_render_area{};
    bool alpha_to_coverage_enabled{};

    IR::Type used_constant_buffer_types{};
    IR::Type used_storage_buffer_types{};
    IR::Type used_indirect_cbuf_types{};

    u32 constant_buffer_mask{};
    std::array<u32, MAX_CBUFS> constant_buffer_used_sizes{};
    u32 nvn_buffer_base{};
    std::bitset<16> nvn_buffer_used{};

    bool requires_layer_emulation{};
    IR::Attribute emulated_layer{};

    u32 used_clip_distances{};

    boost::container::static_vector<ConstantBufferDescriptor, MAX_CBUFS>
        constant_buffer_descriptors;
    boost::container::static_vector<StorageBufferDescriptor, MAX_SSBOS> storage_buffers_descriptors;
    TextureBufferDescriptors texture_buffer_descriptors;
    ImageBufferDescriptors image_buffer_descriptors;
    TextureDescriptors texture_descriptors;
    ImageDescriptors image_descriptors;
};

template <typename Descriptors>
u32 NumDescriptors(const Descriptors& descriptors) {
    u32 num{};
    for (const auto& desc : descriptors) {
        num += desc.count;
    }
    return num;
}

} // namespace Shader
