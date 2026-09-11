// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fstream>

#include "common/logging.h"
#include "video_core/precache_compiler_target.h"

namespace VideoCommon {
namespace {
constexpr std::array<char, 8> kMagic{"citrpct"};
constexpr u32 kVersion = 3;

// Explicit field list: never serialize Profile/HostTranslateInfo object padding.
// clang-format off
#define PROFILE_BOOL_FIELDS(X) \
    X(unified_descriptor_binding) X(has_split_descriptor_sets) X(support_descriptor_aliasing) \
    X(support_int8) X(support_int16) X(support_int64) X(support_vertex_instance_id) \
    X(support_float_controls) X(support_separate_denorm_behavior) \
    X(support_separate_rounding_mode) X(support_fp16_denorm_preserve) \
    X(support_fp32_denorm_preserve) X(support_fp16_denorm_flush) X(support_fp32_denorm_flush) \
    X(support_fp16_signed_zero_nan_preserve) X(support_fp32_signed_zero_nan_preserve) \
    X(support_fp64_signed_zero_nan_preserve) X(support_explicit_workgroup_layout) X(support_vote) \
    X(support_viewport_index_layer_non_geometry) X(support_viewport_mask) \
    X(support_typeless_image_loads) X(support_demote_to_helper_invocation) \
    X(support_int64_atomics) X(support_derivative_control) X(support_geometry_shader_passthrough) \
    X(support_native_ndc) X(support_gl_nv_gpu_shader_5) \
    X(support_gl_amd_gpu_shader_half_float) X(support_gl_texture_shadow_lod) \
    X(support_gl_warp_intrinsics) X(support_gl_variable_aoffi) X(support_gl_sparse_textures) \
    X(support_gl_derivative_control) X(support_scaled_attributes) X(support_multi_viewport) \
    X(support_geometry_streams) X(warp_size_potentially_larger_than_guest) \
    X(lower_left_origin_mode) X(need_declared_frag_colors) X(need_fastmath_off) \
    X(force_fragment_relaxed_precision) X(need_gather_subpixel_offset) \
    X(has_broken_spirv_clamp) X(has_broken_spirv_position_input) \
    X(has_broken_unsigned_image_offsets) X(has_broken_signed_operations) \
    X(has_broken_fp16_float_controls) X(has_gl_component_indexing_bug) X(has_gl_precise_bug) \
    X(has_gl_cbuf_ftou_bug) X(has_gl_bool_ref_bug) X(ignore_nan_fp_comparisons) \
    X(has_broken_spirv_subgroup_mask_vector_extract_dynamic) X(has_broken_robust)

#define HOST_BOOL_FIELDS(X) \
    X(support_float64) X(support_float16) X(support_int64) X(needs_demote_reorder) \
    X(support_snorm_render_buffer) X(support_viewport_index_layer) \
    X(support_geometry_shader_passthrough) X(support_conditional_barrier)
// clang-format on

template <typename T>
void Write(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void Read(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
}

void WriteBool(std::ofstream& file, bool value) {
    Write(file, static_cast<u8>(value));
}

bool ReadBool(std::ifstream& file, bool& value) {
    u8 encoded{};
    Read(file, encoded);
    if (encoded > 1)
        return false;
    value = encoded != 0;
    return true;
}

void WriteTarget(std::ofstream& file, const PrecacheCompilerTarget& target) {
    Write(file, target.profile.supported_spirv);
#define WRITE_PROFILE_BOOL(field) WriteBool(file, target.profile.field);
    PROFILE_BOOL_FIELDS(WRITE_PROFILE_BOOL)
#undef WRITE_PROFILE_BOOL
    Write(file, target.profile.gl_max_compute_smem_size);
    Write(file, target.profile.min_ssbo_alignment);
    Write(file, target.profile.max_user_clip_distances);
#define WRITE_HOST_BOOL(field) WriteBool(file, target.host_info.field);
    HOST_BOOL_FIELDS(WRITE_HOST_BOOL)
#undef WRITE_HOST_BOOL
    Write(file, target.host_info.min_ssbo_alignment);
    Write(file, target.shader_recompiler_revision);
    Write(file, target.descriptor_abi_version);
    Write(file, target.specialization_schema_version);
    Write(file, target.gpu_accuracy_mode);
}

bool ReadTarget(std::ifstream& file, PrecacheCompilerTarget& target) {
    Read(file, target.profile.supported_spirv);
#define READ_PROFILE_BOOL(field)                                                                   \
    if (!ReadBool(file, target.profile.field))                                                     \
        return false;
    PROFILE_BOOL_FIELDS(READ_PROFILE_BOOL)
#undef READ_PROFILE_BOOL
    Read(file, target.profile.gl_max_compute_smem_size);
    Read(file, target.profile.min_ssbo_alignment);
    Read(file, target.profile.max_user_clip_distances);
#define READ_HOST_BOOL(field)                                                                      \
    if (!ReadBool(file, target.host_info.field))                                                   \
        return false;
    HOST_BOOL_FIELDS(READ_HOST_BOOL)
#undef READ_HOST_BOOL
    Read(file, target.host_info.min_ssbo_alignment);
    Read(file, target.shader_recompiler_revision);
    Read(file, target.descriptor_abi_version);
    Read(file, target.specialization_schema_version);
    Read(file, target.gpu_accuracy_mode);
    // Settings::GpuAccuracy is a compact enum (Low through Extreme). Do not
    // accept a corrupt/unknown setting as a valid renderer contract.
    if (target.gpu_accuracy_mode > 3)
        return false;
    return true;
}

void HashCombine(u64& hash, u64 value) noexcept {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}
} // namespace

u64 CompilerTargetFingerprint(const PrecacheCompilerTarget& target) noexcept {
    u64 hash{};
    HashCombine(hash, target.profile.supported_spirv);
#define HASH_PROFILE_BOOL(field) HashCombine(hash, target.profile.field);
    PROFILE_BOOL_FIELDS(HASH_PROFILE_BOOL)
#undef HASH_PROFILE_BOOL
    HashCombine(hash, target.profile.gl_max_compute_smem_size);
    HashCombine(hash, target.profile.min_ssbo_alignment);
    HashCombine(hash, target.profile.max_user_clip_distances);
#define HASH_HOST_BOOL(field) HashCombine(hash, target.host_info.field);
    HOST_BOOL_FIELDS(HASH_HOST_BOOL)
#undef HASH_HOST_BOOL
    HashCombine(hash, target.host_info.min_ssbo_alignment);
    HashCombine(hash, target.shader_recompiler_revision);
    HashCombine(hash, target.descriptor_abi_version);
    HashCombine(hash, target.specialization_schema_version);
    HashCombine(hash, target.gpu_accuracy_mode);
    return hash;
}

bool SavePrecacheCompilerTarget(const std::filesystem::path& filename,
                                const PrecacheCompilerTarget& target) {
    const auto temp_path = filename.parent_path() / (filename.filename().string() + ".tmp");
    const auto backup_path = filename.parent_path() / (filename.filename().string() + ".bak");
    try {
        std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
            return false;
        file.exceptions(std::ios::failbit);
        file.write(kMagic.data(), kMagic.size());
        Write(file, kVersion);
        WriteTarget(file, target);
        file.close();

        // Windows cannot replace an existing destination with rename(). Keep
        // the previous complete target until the new complete file is ready.
        std::error_code ec;
        std::filesystem::remove(backup_path, ec);
        ec.clear();
        const bool had_previous = std::filesystem::exists(filename, ec) && !ec;
        if (had_previous) {
            std::filesystem::rename(filename, backup_path, ec);
            if (ec) {
                std::filesystem::remove(temp_path, ec);
                LOG_WARNING(Render_Vulkan, "Failed to preserve pre-cache compiler target");
                return false;
            }
        }
        std::filesystem::rename(temp_path, filename, ec);
        if (ec) {
            if (had_previous) {
                std::error_code restore_ec;
                std::filesystem::rename(backup_path, filename, restore_ec);
            }
            std::filesystem::remove(temp_path, ec);
            LOG_WARNING(Render_Vulkan, "Failed to install pre-cache compiler target");
            return false;
        }
        if (had_previous) {
            std::filesystem::remove(backup_path, ec);
        }
        return true;
    } catch (const std::ios_base::failure&) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        LOG_WARNING(Render_Vulkan, "Failed to save pre-cache compiler target");
        return false;
    }
}

std::optional<PrecacheCompilerTarget> LoadPrecacheCompilerTarget(
    const std::filesystem::path& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            // SavePrecacheCompilerTarget moves the old complete contract aside
            // before installing the new complete file on Windows. Recover that
            // old contract if process termination occurred in the small gap.
            const auto backup_path =
                filename.parent_path() / (filename.filename().string() + ".bak");
            file.clear();
            file.open(backup_path, std::ios::binary);
            if (!file.is_open())
                return std::nullopt;
            LOG_WARNING(Render_Vulkan,
                        "Recovering pre-cache compiler target from interrupted save");
        }
        file.exceptions(std::ios::failbit);
        std::array<char, 8> magic{};
        u32 version{};
        file.read(magic.data(), magic.size());
        Read(file, version);
        if (magic != kMagic || version != kVersion)
            return std::nullopt;
        PrecacheCompilerTarget target{};
        if (!ReadTarget(file, target))
            return std::nullopt;
        // A target description is a complete cache-namespace contract, not a
        // prefix format.  Reject appended bytes as well as truncation: accepting
        // a valid prefix of a newer/corrupt file could make the scanner believe
        // it has the renderer's exact compiler settings when it does not.
        if (file.peek() != std::char_traits<char>::eof())
            return std::nullopt;
        return target;
    } catch (const std::ios_base::failure&) {
        LOG_WARNING(Render_Vulkan, "Failed to read pre-cache compiler target");
        return std::nullopt;
    }
}
} // namespace VideoCommon
