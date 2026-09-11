// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include <fmt/format.h>

#include "common/bit_cast.h"
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/phase4_prototype_slots_file.h"
#include "video_core/precache_compiler_target.h"
#include "video_core/precache_cfg_artifact.h"
#include "video_core/precache_frontend_artifact.h"
#include "video_core/spirv_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/speculative_shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GraphicsEnvironment;
using VideoCommon::SpirvKey;
using VideoCommon::ComputeCbufKey;
using VideoCommon::ComputeCbufKeyExcludingTextureHandles;
using VideoCommon::ComputeTextureKey;
using VideoCommon::ComputeTextureKeyExcludingHandles;
using VideoCommon::ComputeLogicalTextureKey;
using VideoCommon::HasCompleteLogicalTextureCoverage;
using VideoCommon::HasActivePhase4LogicalTextureSlot;
using VideoCommon::ComputeBindingKey;
using VideoCommon::ComputeWorkgroupKey;
using VideoCommon::FoldViewportTransformState;
using VideoCommon::FoldBindingKey;

// v21: each transferable record has a byte length. A partial final append can
// be ignored without discarding earlier records. v20 records lack framing.
// v20: logical texture slots include descriptor count as well as the source
// expression. v19 records predate this field, so decoding them as v20 shifts
// following fields and can reject a valid old record as an invalid slot.
// v18: FileEnvironment records source-level texture-slot observations so a
// scanner entry can use the same logical texture key as its real counterpart.
// v17: compute pipeline records use ShaderCache's canonical program identity.
// v16: GenericEnvironment::Serialize()/FileEnvironment::Deserialize() gained
// texture_handle_cbuf_keys (see CapturedTextureHandleCbufKeys() in
// shader_environment.h) so boot-time disk-replay entries can supply real
// cbuf/texture-handle narrowing data to the same diagnostic that live play
// already could — testing showed this gap wasn't cosmetic: sessions
// with a lot of preloaded content had an eligible-sample rate roughly 9x
// lower than a fresh-wipe session, since every stale miss sourced from a
// FileEnvironment was structurally excluded from the "would narrowing help"
// measurement. Old caches have no data for this field at all (not a partial-
// data situation — the bytes genuinely aren't there), hence the version bump
// rather than trying to read old files as if they had it.
constexpr u32 TRANSFERABLE_CACHE_VERSION = 21;
constexpr u32 VULKAN_PIPELINE_CACHE_VERSION = 14;
constexpr std::array<char, 8> CFG_TEMPLATE_CACHE_MAGIC{'c', 'i', 't', 'r', 'c', 'f', 'g', '\0'};
// v2: only templates which pass fresh in-memory reconstruction are persisted.
// v1 records may predate raw Location/owner repair, so discard them.
constexpr u32 CFG_TEMPLATE_CACHE_VERSION = 2;
constexpr u32 MAX_CFG_TEMPLATE_COUNT = 262144;
constexpr u32 MAX_CFG_TEMPLATE_BLOCKS = 65536;
constexpr u32 MAX_CFG_TEMPLATE_STACK_ENTRIES = 1024;
constexpr u32 MAX_CFG_TEMPLATE_INDIRECT_BRANCHES = 65536;
// Generic CFG-template replay stays disabled pending full equivalence proof.
constexpr bool CFG_TEMPLATE_REUSE_ENABLED = false;
// Generic CFG templates are not served. Keep their shadow codec off gameplay.
constexpr bool CFG_TEMPLATE_PERSISTENCE_ENABLED = false;
constexpr bool CFG_TEMPLATE_SHADOW_VALIDATION_ENABLED = false;
// Module shadow validated 786 scanner artifacts with no rejection. It duplicates
// frontend and emission work during gameplay, so leave it off.
constexpr bool CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED = false;
// Experimental scanner replay uses exact artifact keys and fresh fallback.
constexpr bool SCANNER_CFG_ARTIFACT_REUSE_ENABLED = true;
// First full live frontend replay (pipeline 4dd8dbde9cb3bde1) immediately preceded
// reproducible post-scan gameplay crashes. Frozen-IR structural shadow proof did not
// establish final pipeline safety. Keep artifacts loaded for diagnostics, but serve
// normal fresh frontend until a full final-output contract exists.
constexpr bool SCANNER_FRONTEND_ARTIFACT_REUSE_ENABLED = false;
// Re-enabled for controlled crash reproduction. This warms a later run's
// driver state, not first-use shaders. NVIDIA previously crashed inside this
// boot worker burst; turn it back off after the scan/no-scan comparison.
constexpr bool DISK_GRAPHICS_PIPELINE_PRELOAD_ENABLED = true;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

bool BindingsEqual(const Shader::Backend::Bindings& lhs,
                   const Shader::Backend::Bindings& rhs) noexcept {
    return lhs.unified == rhs.unified && lhs.uniform_buffer == rhs.uniform_buffer &&
           lhs.storage_buffer == rhs.storage_buffer && lhs.texture == rhs.texture &&
           lhs.image == rhs.image && lhs.texture_scaling_index == rhs.texture_scaling_index &&
           lhs.image_scaling_index == rhs.image_scaling_index;
}

bool ProgramMetadataEqual(const Shader::IR::Program& lhs,
                          const Shader::IR::Program& rhs) noexcept {
    return lhs.info == rhs.info && lhs.stage == rhs.stage &&
           lhs.workgroup_size == rhs.workgroup_size && lhs.output_topology == rhs.output_topology &&
           lhs.output_vertices == rhs.output_vertices && lhs.invocations == rhs.invocations &&
           lhs.local_memory_size == rhs.local_memory_size &&
           lhs.shared_memory_size == rhs.shared_memory_size &&
           lhs.is_geometry_passthrough == rhs.is_geometry_passthrough &&
           lhs.frontend_dependencies == rhs.frontend_dependencies;
}

Shader::OutputTopology MaxwellToOutputTopology(
    Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology) {
    switch (topology) {
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(
    Tegra::Engines::Maxwell3D::Regs::ComparisonOp comparison) {
    switch (comparison) {
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Never_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Less_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Equal_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::LessEqual_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Greater_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::NotEqual_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::GreaterEqual_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Always_D3D:
    case Tegra::Engines::Maxwell3D::Regs::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::
        UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::SNorm:
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::UNorm:
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Tegra::Engines::Maxwell3D::Regs::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::FragmentOutputType GetFragmentOutputType(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return Shader::FragmentOutputType::Float;
    }
    const auto pixel_format{VideoCore::Surface::PixelFormatFromRenderTargetFormat(format)};
    if (!VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
        return Shader::FragmentOutputType::Float;
    }
    return VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
               ? Shader::FragmentOutputType::SignedInt
               : Shader::FragmentOutputType::UnsignedInt;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{Common::BitCast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes;
                 ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Tegra::Engines::Maxwell3D::Regs::Tessellation::DomainType>(raw)) {
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Tegra::Engines::Maxwell3D::Regs::Tessellation::Spacing>(raw)) {
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Tegra::Engines::Maxwell3D::Regs::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            auto [varyings, count] =
                VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
            info.xfb_varyings = varyings;
            info.xfb_count = count;
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        std::ranges::transform(key.state.color_formats, info.frag_color_types.begin(),
                               &GetFragmentOutputType);
        // OPTIMIZED FOR LOW GPU ACCURACY - skip alpha test to reduce shader complexity
        if (!Settings::IsGPULevelLow()) {
            info.alpha_test_func = MaxwellToCompareFunction(
                key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
            info.alpha_test_reference = Common::BitCast<float>(key.state.alpha_test_ref);
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Lines:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::LineLoop:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Triangles:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::TriangleStrip:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::TriangleFan:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Quads:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::QuadStrip:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Polygon:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::LinesAdjacency:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::TrianglesAdjacency:
    case Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL);
#ifdef ANDROID
    // Leave at least a few cores free in android
    constexpr size_t free_cores = 3ULL;
    if (max_core_threads <= free_cores) {
        return 1ULL;
    }
    return max_core_threads - free_cores;
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      speculative_worker(1, "VkSpeculativeShader"),
      serialization_thread(1, "VkPipelineSerialization"),
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue()},
      workers(device.HasBrokenParallelShaderCompiling() ? 1ULL : GetTotalPipelineWorkers(),
              "VkPipelineBuilder") {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    // OPTIMIZED FOR LOW GPU ACCURACY - enable mediump in fragment shaders for better perf
    const bool low_gpu_accuracy = Settings::IsGPULevelLow();

    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .has_split_descriptor_sets = device.IsKhrPushDescriptorSupported(),
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_fastmath_off = false,
        .force_fragment_relaxed_precision = low_gpu_accuracy,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances(),
    };

    host_info = Shader::HostTranslateInfo{
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .min_ssbo_alignment = static_cast<u32>(device.GetStorageBufferAlignment()),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
    };
    // Freeze the effective title setting with the compiler contract. This is
    // deliberately GetValue(), not current_gpu_accuracy: the former includes
    // the per-game override loaded before this renderer is constructed.
    precache_compiler_target = VideoCommon::PrecacheCompilerTarget{
        .profile = profile,
        .host_info = host_info,
        .gpu_accuracy_mode = static_cast<u8>(Settings::values.gpu_accuracy.GetValue()),
    };
    speculative_worker.QueueWork([] {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    });
    serialization_thread.QueueWork([] {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    });

    if (device.GetMaxVertexInputAttributes() <
        Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(),
                    Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Tegra::Engines::Maxwell3D::Regs::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(),
                    Tegra::Engines::Maxwell3D::Regs::NumVertexArrays);
    }

    // Apply user's Extended Dynamic State setting
    const auto eds_setting = Settings::values.extended_dynamic_state.GetValue();
    const bool allow_eds1 = eds_setting >= Settings::ExtendedDynamicState::EDS1;
    const bool allow_eds2 = eds_setting >= Settings::ExtendedDynamicState::EDS2;
    const bool allow_eds3 = eds_setting >= Settings::ExtendedDynamicState::EDS3;

    dynamic_features = DynamicFeatures{
        .has_extended_dynamic_state = allow_eds1 && device.IsExtExtendedDynamicStateSupported(),
        .has_extended_dynamic_state_2 = allow_eds2 && device.IsExtExtendedDynamicState2Supported(),
        .has_extended_dynamic_state_2_extra =
            allow_eds2 && device.IsExtExtendedDynamicState2ExtrasSupported(),
        .has_extended_dynamic_state_3_blend =
            allow_eds3 && device.IsExtExtendedDynamicState3BlendingSupported(),
        .has_extended_dynamic_state_3_enables =
            allow_eds3 && device.IsExtExtendedDynamicState3EnablesSupported(),
        .has_dynamic_vertex_input = allow_eds3 && device.IsExtVertexInputDynamicStateSupported(),
        .has_transform_feedback = device.IsExtTransformFeedbackSupported(),
    };
}

PipelineCache::~PipelineCache() {
    speculative_worker.WaitForRequests();
    serialization_thread.WaitForRequests();
    if (!spirv_cache_filename.empty()) {
        spirv_cache.Save(spirv_cache_filename);
    }
    if constexpr (CFG_TEMPLATE_PERSISTENCE_ENABLED) {
        if (!cfg_templates_filename.empty()) {
            SaveCfgTemplates(cfg_templates_filename);
        }
    }
    if (!phase4_prototype_slots_filename.empty()) {
        // Merge TWO things, not just candidates: whatever LoadDiskResources already
        // published this boot (Shader::ActivePhase4PrototypeSlots() below reflects that --
        // this game's prior-session learning, or empty for a fresh profile) plus whatever
        // TakePhase4PrototypeCandidates newly found THIS session (including anything the
        // live-growth path in RecordResolvedTextureType already folded in mid-session --
        // ActivePhase4PrototypeSlots() reflects that too, so this isn't double-counting,
        // just re-confirming the same state before persisting it). Saving candidates alone
        // would silently forget every coordinate learned in an EARLIER session the moment
        // this one ends -- ActivePhase4PrototypeSlots() is what carries that forward.
        std::vector<Shader::Phase4PrototypeSlot> extra{
            Shader::ActivePhase4PrototypeSlots().begin(),
            Shader::ActivePhase4PrototypeSlots().end()};
        const std::vector<Shader::Phase4PrototypeSlot> candidates{
            VideoCommon::TakePhase4PrototypeCandidates()};
        extra.insert(extra.end(), candidates.begin(), candidates.end());
        VideoCommon::SavePhase4PrototypeSlots(phase4_prototype_slots_filename,
                                               Shader::MergePhase4PrototypeSlots(extra));
    }

    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     VULKAN_PIPELINE_CACHE_VERSION);
    }
}

void PipelineCache::EvictOldPipelines() {
    constexpr u64 FRAMES_TO_KEEP = 2000;

    const u64 current_frame = scheduler.CurrentTick();

    if (current_frame - last_memory_pressure_frame < MEMORY_PRESSURE_COOLDOWN) {
        return;
    }
    last_memory_pressure_frame = current_frame;

    const u64 evict_before_frame =
        current_frame > FRAMES_TO_KEEP ? current_frame - FRAMES_TO_KEEP : 0;

    size_t evicted_graphics = 0;
    size_t evicted_compute = 0;

    for (auto it = graphics_cache.begin(); it != graphics_cache.end();) {
        const GraphicsPipeline* pipeline = it->second.get();
        if (pipeline && pipeline != current_pipeline) {
            auto use_it = graphics_pipeline_last_use.find(pipeline);
            if (use_it == graphics_pipeline_last_use.end() || use_it->second < evict_before_frame) {
                graphics_pipeline_last_use.erase(pipeline);
                it = graphics_cache.erase(it);
                evicted_graphics++;
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }

    for (auto it = compute_cache.begin(); it != compute_cache.end();) {
        const ComputePipeline* pipeline = it->second.get();
        if (pipeline) {
            auto use_it = compute_pipeline_last_use.find(pipeline);
            if (use_it == compute_pipeline_last_use.end() || use_it->second < evict_before_frame) {
                compute_pipeline_last_use.erase(pipeline);
                it = compute_cache.erase(it);
                evicted_compute++;
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }

    if (evicted_graphics > 0 || evicted_compute > 0) {
        LOG_INFO(Render_Vulkan, "Evicted {} graphics and {} compute pipelines to free memory",
                 evicted_graphics, evicted_compute);
    }
}

// Graphics-cache lookup timing fix. The problem this solves:
// graphics_key (unique_hashes + fixed-function state) gets looked up in graphics_cache/
// current_pipeline->Next() BEFORE any Shader::Environment exists -- CurrentGraphicsPipeline()
// only reaches CreateGraphicsPipeline() (where env access and each known slot's real
// resolution already happens, see the texture_key fix above) on a cache MISS. Without this,
// phase4_prototype_needs_array_variant only ever gets a real value after the lookup its whole
// purpose depends on has already happened.
//
// Loops over Shader::ActivePhase4PrototypeSlots() (environment.h) -- was a single hardcoded
// (cbuf_index=2, cbuf_offset=192) read, widened to loop over however many known slots exist
// so a second (or third) table entry needs no further change here. For each slot the current
// fragment shader actually has (per the bitmask below), reads that slot's raw handle directly
// from GPU state and resolves its TextureType, mirroring
// GraphicsEnvironment::ReadCbufValue/ReadTextureType's actual GPU access exactly (confirmed by
// reading both, not assumed) but without needing a live GraphicsEnvironment instance --
// ResolveTextureTypeFromRawHandle (shader_environment.h/.cpp) is the shared piece both this
// and the real environment path use.
//
// Two real, deliberate simplifications, not oversights, that now apply per-slot:
// - Assumes no secondary cbuf combine for any known slot (GetTextureHandle, texture_pass.cpp,
//   can OR together two separate cbuf reads via has_secondary/secondary_cbuf_index/
//   secondary_cbuf_offset when a descriptor needs it). Not confirmed either way for any real
//   slot this prototype targets -- this session has no way to inspect those shaders' actual
//   descriptor fields directly. If a slot turns out to use a secondary combine, this function
//   silently resolves the wrong handle for that slot specifically. Flagging plainly rather
//   than guessing further.
// - Runs unconditionally for every draw with an active fragment stage, not just draws using a
//   shader with at least one known slot -- Shader::Info (which would say "this shader actually
//   has slot i") isn't available at this point any more than the environment is. Reading
//   garbage cbuf content for unrelated fragment shaders is safe (see below), but see this
//   function's use in CurrentGraphicsPipeline for why it could still theoretically add
//   spurious graphics_key entropy for shaders that don't actually care.
//
// Returns 0 (the same default every SPIR-V spec constant itself defaults to, and the same
// value the caller would compute if this whole function were a no-op) whenever no known slot's
// cbuf is enabled or in range for the currently-bound fragment shader -- exactly the same
// safe-fallback shape ReadCbufValue/ReadTextureInfo already use for their own out-of-range
// cases, so an unrelated shader reading garbage here is, at worst, exactly as safe as any other
// out-of-range cbuf read already is elsewhere in this codebase. Non-zero bits are OR'd from
// independent per-slot resolutions, so one slot's result can never overwrite another's.
u64 PipelineCache::ResolvePhase4PrototypeSpecValue() const {
    // Two different indices for two different arrays, both real, both required -- conflating
    // them is exactly what caused the freeze a real build surfaced. unique_hashes (this
    // function's first check, and the key phase4_prototype_fragment_shader_table above is
    // keyed by) is populated in ShaderCache::RefreshStages via a direct
    // static_cast<ShaderType>(index) -- confirmed by reading that function, not assumed -- so
    // it uses the RAW hardware ShaderType numbering (VertexA=0, VertexB=1, TessellationInit=2,
    // Tessellation=3, Geometry=4, Pixel=5). shader_stages (this function's second check, the
    // actual cbuf read) uses the SOFTWARE Shader::Stage numbering instead (VertexB=0, ...,
    // Fragment=4) -- confirmed against GraphicsEnvironment's own ShaderType::Pixel ->
    // stage_index=4 mapping. CreateGraphicsPipeline's own `stage_index = index - 1` (this same
    // file) is the concrete conversion between the two, and is what pins these two numbers
    // down as correct rather than assumed.
    constexpr size_t kFragmentHardwareIndex = 5;    // ShaderType::Pixel, for unique_hashes only.
    constexpr size_t kFragmentSoftwareIndex = 4;    // Shader::Stage::Fragment, for shader_stages.
    const u64 fragment_hash{graphics_key.unique_hashes[kFragmentHardwareIndex]};
    if (fragment_hash == 0) {
        return 0; // No fragment shader bound at all.
    }

    // The actual fix for the freeze: only ever do a speculative cbuf read for a slot
    // CONFIRMED (via real Shader::Info, recorded in CreateGraphicsPipeline the one time this
    // shader was actually translated) to be present on this fragment shader. A shader with
    // none of the known slots -- the overwhelming majority -- returns 0 here without touching
    // GPU memory at all, every single draw, forever, once seen once. A shader not yet in the
    // table (never translated) also returns 0 rather than guessing: reading a slot's cbuf
    // speculatively for a genuinely unknown shader is exactly the behavior that turned
    // unrelated per-draw application data into a constantly-changing graphics_key and froze
    // real gameplay -- not worth doing even once more now that it's understood.
    //
    // shared_lock, released before the GPU reads below: extract the plain u32 bitmask from the
    // iterator while the lock is held (see phase4_prototype_fragment_shader_table_mutex's doc
    // comment, vk_pipeline_cache.h, for why an unsynchronized read here was itself a real bug,
    // not just the write) -- the iterator itself would not be safe to keep using once the lock
    // releases, and the GPU reads that follow don't touch this table at all, so there's no
    // reason to hold the lock any longer than the lookup itself needs.
    u32 shader_slot_mask = 0;
    {
        std::shared_lock lock{phase4_prototype_fragment_shader_table_mutex};
        const auto it{phase4_prototype_fragment_shader_table.find(fragment_hash)};
        shader_slot_mask = it != phase4_prototype_fragment_shader_table.end() ? it->second : 0U;
    }
    if (shader_slot_mask == 0U) {
        return 0;
    }

    u64 result_mask = 0;
    const std::span<const Shader::Phase4PrototypeSlot> active_slots{
        Shader::ActivePhase4PrototypeSlots()};
    for (size_t slot_id = 0; slot_id < active_slots.size(); ++slot_id) {
        if ((shader_slot_mask & (1U << slot_id)) == 0U) {
            continue; // This shader doesn't have this particular known slot.
        }
        const Shader::Phase4PrototypeSlot& slot{active_slots[slot_id]};
        const auto& cbuf{maxwell3d->state.shader_stages[kFragmentSoftwareIndex]
                              .const_buffers[slot.cbuf_index]};
        if (!cbuf.enabled || slot.cbuf_offset >= cbuf.size) {
            continue;
        }
        const u32 handle{gpu_memory->Read<u32>(cbuf.address + slot.cbuf_offset)};
        const auto& regs{maxwell3d->regs};
        const bool via_header_index{regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding};
        const Shader::TextureType resolved{VideoCommon::ResolveTextureTypeFromRawHandle(
            *gpu_memory, regs.tex_header.Address(), regs.tex_header.limit, via_header_index,
            handle)};
        if (resolved == Shader::TextureType::ColorArray2D) {
            result_mask |= (u64{1} << slot_id);
        }
    }
    return result_mask;
}

// Runtime-variant diagnostic — see the header for what it is
// measuring and why. Diagnostic only: nothing here changes what gets cached, guessed, or
// served — this purely observes the real, already-computed values CreateGraphicsPipeline()
// passes it and reports on their shape.
void PipelineCache::RecordPhase3RuntimeVariantDiagnostic(u64 unique_hash,
                                                          u64 diag_base_runtime_hash) const {
    // Capped exactly like SpirvCache's own keys_by_hash_ (spirv_cache.cpp) and for the same
    // reason: hitting the cap IS the answer for a hash that hits it ("too high-cardinality
    // for a small scan-time guess to plausibly cover"), not a gap in the measurement.
    constexpr size_t kMaxTrackedVariantsForDiagnostics = 8;
    bool should_log = false;
    {
        std::unique_lock lock{phase3_diag_runtime_variants_mutex};
        auto& variants = phase3_diag_cbuf_zero_runtime_variants_by_hash[unique_hash];
        if (std::find(variants.begin(), variants.end(), diag_base_runtime_hash) == variants.end() &&
            variants.size() < kMaxTrackedVariantsForDiagnostics) {
            variants.push_back(diag_base_runtime_hash);
        }
        // Time-throttled the same way SpirvCache::SaveThrottled's default cadence is (30s) —
        // this runs on whatever worker thread CreateGraphicsPipeline() happens to be on, so
        // logging unconditionally on every call would repeat the exact "measurable per-call
        // cost, once caused a real hang" mistake this investigation already made once with
        // the original cbuf-narrowing diagnostic (see SPIRV_CACHE_VERSION's comment,
        // spirv_cache.cpp). The write above is cheap regardless (capped vector, at most one
        // linear scan over <=8 elements); only the histogram pass below needs throttling.
        const auto now = std::chrono::steady_clock::now();
        if (now - phase3_diag_last_log_time >= std::chrono::seconds{30}) {
            phase3_diag_last_log_time = now;
            should_log = true;
        }
    }
    if (!should_log) {
        return;
    }
    // Re-takes the lock shared/read-only for the histogram pass rather than holding the
    // exclusive lock from above across it — this is O(hashes tracked so far), typically a
    // few thousand at most for a full game, but there's no reason to block concurrent
    // CreateGraphicsPipeline() calls on other worker threads for a read-only report.
    std::array<size_t, 4> buckets{}; // [0]=1 variant, [1]=2-3, [2]=4-7, [3]=8+ (capped)
    size_t total_hashes = 0;
    {
        std::shared_lock lock{phase3_diag_runtime_variants_mutex};
        for (const auto& [hash, variants] : phase3_diag_cbuf_zero_runtime_variants_by_hash) {
            ++total_hashes;
            const size_t n = variants.size();
            if (n <= 1) {
                ++buckets[0];
            } else if (n <= 3) {
                ++buckets[1];
            } else if (n <= 7) {
                ++buckets[2];
            } else {
                ++buckets[3];
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Runtime-variant diagnostic: of {} graphics shaders seen with a real, cbuf_key==0 draw "
             "so far, {} showed exactly 1 distinct core-RuntimeInfo state, {} showed 2-3, {} "
             "showed 4-7, {} showed 8+ (capped — true count may be higher). Diagnostic only, "
             "nothing currently acts on this. Low cardinality across most hashes would say "
             "scan-time multi-variant guessing could plausibly help now that texture "
             "1 narrowed cbuf_key; a lot of 8+ hashes would say this hits the same structural "
             "ceiling the removed second viewport-transform-state guess already found once, "
             "just now confirmed with cbuf's blocking accounted for.",
             total_hashes, buckets[0], buckets[1], buckets[2], buckets[3]);
}

void PipelineCache::FlushSpirvCache() {
    if (spirv_cache_filename.empty()) {
        return;
    }
    // A normal miss queues SaveThrottled on this worker. Finish those jobs
    // first, then take one final snapshot. This closes the short-boot race:
    // starting the scanner immediately after ShutdownGame must not depend on
    // the renderer destructor eventually reaching its own Save().
    serialization_thread.WaitForRequests();
    spirv_cache.Save(spirv_cache_filename);
    if constexpr (CFG_TEMPLATE_PERSISTENCE_ENABLED) {
        if (!cfg_templates_filename.empty()) {
            SaveCfgTemplates(cfg_templates_filename);
        }
    }
}

std::optional<Shader::Maxwell::Flow::CFG::Template> PipelineCache::LookupCfgTemplate(
    CfgTemplateKey key, bool count_as_reuse) const {
    std::shared_lock lock{cfg_templates_mutex};
    const auto it{cfg_templates.find(key)};
    if (it == cfg_templates.end()) {
        return std::nullopt;
    }
    if (count_as_reuse) {
        cfg_template_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return it->second;
}

void PipelineCache::EraseCfgTemplate(CfgTemplateKey key) {
    std::unique_lock lock{cfg_templates_mutex};
    cfg_templates.erase(key);
}

void PipelineCache::RejectCfgTemplateModule(CfgTemplateKey key) {
    std::unique_lock lock{cfg_templates_mutex};
    cfg_templates.erase(key);
    cfg_template_module_rejected_keys.insert(key);
}

void PipelineCache::InsertCfgTemplate(CfgTemplateKey key,
                                      Shader::Maxwell::Flow::CFG::Template source) {
    if (!Shader::Maxwell::Flow::CFG::IsValidTemplate(source)) {
        LOG_WARNING(Render_Vulkan, "Refusing invalid CFG template for 0x{:016x}", key.unique_hash);
        return;
    }
    std::unique_lock lock{cfg_templates_mutex};
    if (cfg_template_module_rejected_keys.contains(key)) {
        return;
    }
    if (cfg_templates.size() >= MAX_CFG_TEMPLATE_COUNT) {
        return;
    }
    if (cfg_templates.try_emplace(key, std::move(source)).second) {
        cfg_template_inserts.fetch_add(1, std::memory_order_relaxed);
    }
}

void PipelineCache::ValidateAndPersistCfgTemplate(
    CfgTemplateKey key, Shader::Environment& env, const Shader::Maxwell::Flow::CFG& cfg) {
    const Shader::Maxwell::Flow::CFG::Template fresh{cfg.MakeTemplate()};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> shadow_pool(16);
    try {
        if (!Shader::Maxwell::Flow::CFG::RoundTripMatchesTemplate(env, shadow_pool, fresh)) {
            ++cfg_template_shadow_rejected;
            LOG_WARNING(Render_Vulkan, "CFG template shadow replay mismatch for 0x{:016x}",
                        key.unique_hash);
            return;
        }
    } catch (...) {
        ++cfg_template_shadow_rejected;
        LOG_WARNING(Render_Vulkan, "CFG template shadow replay threw for 0x{:016x}",
                    key.unique_hash);
        return;
    }
    ++cfg_template_shadow_verified;

    // A loaded v2 entry still cannot drive compilation. Compare it against the
    // fresh graph first; this catches a bad disk round trip or stale identity
    // before it becomes eligible for any future replay experiment.
    if (auto disk = LookupCfgTemplate(key, false)) {
        if (*disk == fresh) {
            ++cfg_template_cached_verified;
        } else {
            ++cfg_template_cached_rejected;
            LOG_WARNING(Render_Vulkan, "CFG template cached-form mismatch for 0x{:016x}",
                        key.unique_hash);
            // Keep only a form that this live CFG has validated. The bad
            // record still cannot affect compilation, but evict it so its
            // replacement is persisted at normal cache-save time.
            EraseCfgTemplate(key);
        }
    }
    InsertCfgTemplate(key, fresh);
}

bool PipelineCache::ClaimCfgTemplateModuleValidation(
    CfgTemplateKey key, const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact) {
    if (scanner_artifact) {
        std::unique_lock lock{precache_cfg_artifacts_mutex};
        if (!precache_cfg_artifact_module_claimed_keys.insert(*scanner_artifact).second) {
            return false;
        }
        ++precache_cfg_artifact_module_claims;
        return true;
    }
    std::unique_lock lock{cfg_templates_mutex};
    return cfg_template_module_claims.insert(key).second;
}

void PipelineCache::ValidateCfgTemplateGraphicsModule(
    CfgTemplateKey key, Shader::Environment& env,
    const Shader::Maxwell::Flow::CFG::Template& source,
    u32 cfg_request_address,
    const Shader::RuntimeInfo& runtime_info, const Shader::IR::Program& expected_program,
    const std::vector<u32>& expected_spirv,
    const Shader::Backend::Bindings& starting_binding,
    const Shader::Backend::Bindings& expected_end_binding,
    const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact) {
    if (!ClaimCfgTemplateModuleValidation(key, scanner_artifact)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        auto validation_source = source;
        if (scanner_artifact) {
            const auto validation_cfg_start = std::chrono::steady_clock::now();
            ShaderPools fresh_pools;
            Shader::Maxwell::Flow::CFG fresh_cfg{env, fresh_pools.flow_block,
                                                  cfg_request_address,
                                                  key.exits_to_dispatcher};
            validation_source = fresh_cfg.MakeTemplate();
            precache_cfg_artifact_validation_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - validation_cfg_start).count()),
                std::memory_order_relaxed);
        }
        ShaderPools shadow_pools;
        Shader::Maxwell::Flow::CFG shadow_cfg{env, shadow_pools.flow_block, validation_source};
        auto shadow_program{Shader::Maxwell::BuildProgramTemplate(
            shadow_pools.inst, shadow_pools.block, env, shadow_cfg, host_info)};
        Shader::Maxwell::FinalizeProgramTemplate(shadow_program, env, host_info);
        Shader::Maxwell::ConvertLegacyToGeneric(shadow_program, runtime_info);
        Shader::Backend::Bindings shadow_binding{starting_binding};
        const auto shadow_spirv{EmitSPIRV(profile, runtime_info, shadow_program, shadow_binding)};
        if (shadow_spirv == expected_spirv &&
            ProgramMetadataEqual(shadow_program, expected_program) &&
            BindingsEqual(shadow_binding, expected_end_binding)) {
            ++cfg_template_module_verified;
            ++cfg_template_module_graphics_verified;
            if (scanner_artifact) {
                ++precache_cfg_artifact_module_verified;
            }
        } else {
            ++cfg_template_module_rejected;
            if (scanner_artifact) {
                ++precache_cfg_artifact_module_rejected;
                QuarantinePrecacheCfgArtifact(*scanner_artifact);
            }
            // Graph equality is insufficient: this reconstructed form produced
            // different final output under the real environment. Keep it out
            // of the shadow-persistence corpus until a fresh graph can prove
            // the complete module contract again.
            RejectCfgTemplateModule(key);
            LOG_WARNING(Render_Vulkan,
                        "CFG template module shadow mismatch for 0x{:016x} "
                        "(SPIR-V={}, metadata={}, bindings={})",
                        key.unique_hash, shadow_spirv == expected_spirv,
                        ProgramMetadataEqual(shadow_program, expected_program),
                        BindingsEqual(shadow_binding, expected_end_binding));
        }
    } catch (...) {
        ++cfg_template_module_rejected;
        if (scanner_artifact) {
            ++precache_cfg_artifact_module_rejected;
            QuarantinePrecacheCfgArtifact(*scanner_artifact);
        }
        RejectCfgTemplateModule(key);
        LOG_WARNING(Render_Vulkan, "CFG template module shadow threw for 0x{:016x}",
                    key.unique_hash);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    cfg_template_module_us.fetch_add(static_cast<u64>(elapsed.count()), std::memory_order_relaxed);
}

void PipelineCache::ValidateCfgTemplateComputeModule(
    CfgTemplateKey key, Shader::Environment& env,
    const Shader::Maxwell::Flow::CFG::Template& source,
    u32 cfg_request_address,
    const Shader::IR::Program& expected_program,
    const std::vector<u32>& expected_spirv,
    const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact) {
    if (!ClaimCfgTemplateModuleValidation(key, scanner_artifact)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        auto validation_source = source;
        if (scanner_artifact) {
            const auto validation_cfg_start = std::chrono::steady_clock::now();
            ShaderPools fresh_pools;
            Shader::Maxwell::Flow::CFG fresh_cfg{env, fresh_pools.flow_block,
                                                  cfg_request_address};
            validation_source = fresh_cfg.MakeTemplate();
            precache_cfg_artifact_validation_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - validation_cfg_start).count()),
                std::memory_order_relaxed);
        }
        ShaderPools shadow_pools;
        Shader::Maxwell::Flow::CFG shadow_cfg{env, shadow_pools.flow_block, validation_source};
        auto shadow_program{Shader::Maxwell::BuildProgramTemplate(
            shadow_pools.inst, shadow_pools.block, env, shadow_cfg, host_info)};
        Shader::Maxwell::FinalizeProgramTemplate(shadow_program, env, host_info);
        const auto shadow_spirv{EmitSPIRV(profile, shadow_program)};
        if (shadow_spirv == expected_spirv && ProgramMetadataEqual(shadow_program, expected_program)) {
            ++cfg_template_module_verified;
            ++cfg_template_module_compute_verified;
            if (scanner_artifact) {
                ++precache_cfg_artifact_module_verified;
            }
        } else {
            ++cfg_template_module_rejected;
            if (scanner_artifact) {
                ++precache_cfg_artifact_module_rejected;
                QuarantinePrecacheCfgArtifact(*scanner_artifact);
            }
            RejectCfgTemplateModule(key);
            LOG_WARNING(Render_Vulkan,
                        "CFG template compute-module shadow mismatch for 0x{:016x} "
                        "(SPIR-V={}, metadata={})",
                        key.unique_hash, shadow_spirv == expected_spirv,
                        ProgramMetadataEqual(shadow_program, expected_program));
        }
    } catch (...) {
        ++cfg_template_module_rejected;
        if (scanner_artifact) {
            ++precache_cfg_artifact_module_rejected;
            QuarantinePrecacheCfgArtifact(*scanner_artifact);
        }
        RejectCfgTemplateModule(key);
        LOG_WARNING(Render_Vulkan, "CFG template compute-module shadow threw for 0x{:016x}",
                    key.unique_hash);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    cfg_template_module_us.fetch_add(static_cast<u64>(elapsed.count()), std::memory_order_relaxed);
}

void PipelineCache::ValidateCfgTemplateMergedVertexModule(
    CfgTemplateKey key, Shader::Environment& vertex_a_env,
    const Shader::Maxwell::Flow::CFG::Template& vertex_a_source,
    Shader::Environment& vertex_b_env,
    const Shader::Maxwell::Flow::CFG::Template& vertex_b_source,
    const Shader::RuntimeInfo& runtime_info, const Shader::IR::Program& expected_program,
    const std::vector<u32>& expected_spirv,
    const Shader::Backend::Bindings& starting_binding,
    const Shader::Backend::Bindings& expected_end_binding,
    const VideoCommon::PrecacheCfgArtifactKey* vertex_a_artifact,
    const VideoCommon::PrecacheCfgArtifactKey* vertex_b_artifact) {
    if (!ClaimCfgTemplateModuleValidation(
            key, vertex_a_artifact ? vertex_a_artifact : vertex_b_artifact)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        auto validation_source_a = vertex_a_source;
        auto validation_source_b = vertex_b_source;
        if (vertex_a_artifact) {
            const auto validation_cfg_start = std::chrono::steady_clock::now();
            ShaderPools fresh_pools;
            Shader::Maxwell::Flow::CFG fresh_cfg{
                vertex_a_env, fresh_pools.flow_block,
                static_cast<u32>(vertex_a_env.StartAddress() + sizeof(Shader::ProgramHeader)), true};
            validation_source_a = fresh_cfg.MakeTemplate();
            precache_cfg_artifact_validation_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - validation_cfg_start).count()),
                std::memory_order_relaxed);
        }
        if (vertex_b_artifact) {
            const auto validation_cfg_start = std::chrono::steady_clock::now();
            ShaderPools fresh_pools;
            Shader::Maxwell::Flow::CFG fresh_cfg{
                vertex_b_env, fresh_pools.flow_block,
                static_cast<u32>(vertex_b_env.StartAddress() + sizeof(Shader::ProgramHeader)), false};
            validation_source_b = fresh_cfg.MakeTemplate();
            precache_cfg_artifact_validation_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - validation_cfg_start).count()),
                std::memory_order_relaxed);
        }
        ShaderPools shadow_pools;
        Shader::Maxwell::Flow::CFG shadow_cfg_a{vertex_a_env, shadow_pools.flow_block,
                                                 validation_source_a};
        Shader::Maxwell::Flow::CFG shadow_cfg_b{vertex_b_env, shadow_pools.flow_block,
                                                 validation_source_b};
        auto shadow_vertex_a{Shader::Maxwell::BuildProgramTemplate(
            shadow_pools.inst, shadow_pools.block, vertex_a_env, shadow_cfg_a, host_info)};
        auto shadow_vertex_b{Shader::Maxwell::BuildProgramTemplate(
            shadow_pools.inst, shadow_pools.block, vertex_b_env, shadow_cfg_b, host_info)};
        Shader::Maxwell::FinalizeProgramTemplate(shadow_vertex_a, vertex_a_env, host_info);
        Shader::Maxwell::FinalizeProgramTemplate(shadow_vertex_b, vertex_b_env, host_info);
        auto shadow_program{Shader::Maxwell::MergeDualVertexPrograms(
            shadow_vertex_a, shadow_vertex_b, vertex_b_env)};
        Shader::Maxwell::ConvertLegacyToGeneric(shadow_program, runtime_info);
        Shader::Backend::Bindings shadow_binding{starting_binding};
        const auto shadow_spirv{EmitSPIRV(profile, runtime_info, shadow_program, shadow_binding)};
        if (shadow_spirv == expected_spirv &&
            ProgramMetadataEqual(shadow_program, expected_program) &&
            BindingsEqual(shadow_binding, expected_end_binding)) {
            ++cfg_template_module_verified;
            ++cfg_template_module_merged_vertex_verified;
            if (vertex_a_artifact || vertex_b_artifact) {
                ++precache_cfg_artifact_module_verified;
            }
        } else {
            ++cfg_template_module_rejected;
            if (vertex_a_artifact || vertex_b_artifact) {
                ++precache_cfg_artifact_module_rejected;
                if (vertex_a_artifact) QuarantinePrecacheCfgArtifact(*vertex_a_artifact);
                if (vertex_b_artifact) QuarantinePrecacheCfgArtifact(*vertex_b_artifact);
            }
            RejectCfgTemplateModule(key);
            LOG_WARNING(Render_Vulkan,
                        "CFG template merged-vertex shadow mismatch for 0x{:016x} "
                        "(SPIR-V={}, metadata={}, bindings={})",
                        key.unique_hash, shadow_spirv == expected_spirv,
                        ProgramMetadataEqual(shadow_program, expected_program),
                        BindingsEqual(shadow_binding, expected_end_binding));
        }
    } catch (...) {
        ++cfg_template_module_rejected;
        if (vertex_a_artifact || vertex_b_artifact) {
            ++precache_cfg_artifact_module_rejected;
            if (vertex_a_artifact) QuarantinePrecacheCfgArtifact(*vertex_a_artifact);
            if (vertex_b_artifact) QuarantinePrecacheCfgArtifact(*vertex_b_artifact);
        }
        RejectCfgTemplateModule(key);
        LOG_WARNING(Render_Vulkan, "CFG template merged-vertex shadow threw for 0x{:016x}",
                    key.unique_hash);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    cfg_template_module_us.fetch_add(static_cast<u64>(elapsed.count()), std::memory_order_relaxed);
}

void PipelineCache::LoadCfgTemplates(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    try {
        file.exceptions(std::ifstream::failbit);
        std::array<char, 8> magic{};
        u32 version{};
        u32 count{};
        file.read(magic.data(), magic.size());
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (magic != CFG_TEMPLATE_CACHE_MAGIC || version != CFG_TEMPLATE_CACHE_VERSION ||
            count > MAX_CFG_TEMPLATE_COUNT) {
            LOG_WARNING(Render_Vulkan, "Discarding incompatible CFG template cache");
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            return;
        }

        // Decode into a private map. A truncated or duplicate-key file must
        // never publish a partially validated cache, even though templates are
        // shadow-only today.
        decltype(cfg_templates) loaded_templates;
        for (u32 entry = 0; entry < count; ++entry) {
            CfgTemplateKey key{};
            u8 exits{};
            u32 function_count{};
            u32 block_count{};
            file.read(reinterpret_cast<char*>(&key.unique_hash), sizeof(key.unique_hash));
            file.read(reinterpret_cast<char*>(&key.start_address), sizeof(key.start_address));
            file.read(reinterpret_cast<char*>(&exits), sizeof(exits));
            file.read(reinterpret_cast<char*>(&function_count), sizeof(function_count));
            file.read(reinterpret_cast<char*>(&block_count), sizeof(block_count));
            if (key.unique_hash == 0 || exits > 1 || function_count == 0 || block_count == 0 ||
                block_count > MAX_CFG_TEMPLATE_BLOCKS || function_count > block_count) {
                throw std::ios_base::failure{"invalid CFG template header"};
            }
            key.exits_to_dispatcher = exits != 0;
            Shader::Maxwell::Flow::CFG::Template source;
            source.exits_to_dispatcher = key.exits_to_dispatcher;
            source.functions.resize(function_count);
            source.blocks.resize(block_count);
            std::vector<bool> owned_blocks(block_count);
            for (Shader::Maxwell::Flow::FunctionId function_id = 0;
                 function_id < source.functions.size(); ++function_id) {
                auto& function{source.functions[function_id]};
                u32 entrypoint{};
                u32 blocks{};
                file.read(reinterpret_cast<char*>(&entrypoint), sizeof(entrypoint));
                file.read(reinterpret_cast<char*>(&blocks), sizeof(blocks));
                if (!Shader::Maxwell::Location::IsRawOffset(entrypoint) || blocks == 0 ||
                    blocks > block_count) {
                    throw std::ios_base::failure{"invalid CFG template function"};
                }
                function.entrypoint = Shader::Maxwell::Location::FromRawOffset(entrypoint);
                function.blocks.resize(blocks);
                for (size_t& id : function.blocks) {
                    u32 value{};
                    file.read(reinterpret_cast<char*>(&value), sizeof(value));
                    if (value >= block_count || owned_blocks[value]) {
                        throw std::ios_base::failure{"invalid CFG block id"};
                    }
                    owned_blocks[value] = true;
                    source.blocks[value].owner = function_id;
                    id = value;
                }
            }
            for (auto& block : source.blocks) {
                u32 begin{}, end{}, branch_true{}, branch_false{}, function_call{}, return_block{};
                u16 flow_test{};
                u8 pred{}, negated{}, end_class{};
                u32 stack_count{};
                u64 branch_reg{};
                u32 indirect_count{};
                file.read(reinterpret_cast<char*>(&begin), sizeof(begin));
                file.read(reinterpret_cast<char*>(&end), sizeof(end));
                file.read(reinterpret_cast<char*>(&end_class), sizeof(end_class));
                file.read(reinterpret_cast<char*>(&flow_test), sizeof(flow_test));
                file.read(reinterpret_cast<char*>(&pred), sizeof(pred));
                file.read(reinterpret_cast<char*>(&negated), sizeof(negated));
                file.read(reinterpret_cast<char*>(&branch_true), sizeof(branch_true));
                file.read(reinterpret_cast<char*>(&branch_false), sizeof(branch_false));
                file.read(reinterpret_cast<char*>(&function_call), sizeof(function_call));
                file.read(reinterpret_cast<char*>(&return_block), sizeof(return_block));
                file.read(reinterpret_cast<char*>(&branch_reg), sizeof(branch_reg));
                file.read(reinterpret_cast<char*>(&block.branch_offset), sizeof(block.branch_offset));
                file.read(reinterpret_cast<char*>(&stack_count), sizeof(stack_count));
                if (end_class > static_cast<u8>(Shader::Maxwell::Flow::EndClass::Kill) || negated > 1 ||
                    stack_count > MAX_CFG_TEMPLATE_STACK_ENTRIES || function_call >= function_count) {
                    throw std::ios_base::failure{"invalid CFG template block"};
                }
                const auto decode_id = [block_count](u32 id) -> size_t {
                    if (id == std::numeric_limits<u32>::max()) return Shader::Maxwell::Flow::CFG::NoTemplateBlock;
                    if (id >= block_count) throw std::ios_base::failure{"invalid CFG block reference"};
                    return id;
                };
                if (!Shader::Maxwell::Location::IsRawOffset(begin) ||
                    !Shader::Maxwell::Location::IsRawOffset(end)) {
                    throw std::ios_base::failure{"invalid CFG location"};
                }
                block.begin = Shader::Maxwell::Location::FromRawOffset(begin);
                block.end = Shader::Maxwell::Location::FromRawOffset(end);
                block.end_class = static_cast<Shader::Maxwell::Flow::EndClass>(end_class);
                block.cond = Shader::IR::Condition{static_cast<Shader::IR::FlowTest>(flow_test),
                                                    static_cast<Shader::IR::Pred>(pred), negated != 0};
                block.branch_true = decode_id(branch_true);
                block.branch_false = decode_id(branch_false);
                block.function_call = function_call;
                block.return_block = decode_id(return_block);
                block.branch_reg = static_cast<Shader::IR::Reg>(branch_reg);
                std::vector<Shader::Maxwell::Flow::StackEntry> stack;
                stack.reserve(stack_count);
                for (u32 i = 0; i < stack_count; ++i) {
                    u8 token{};
                    u32 target{};
                    file.read(reinterpret_cast<char*>(&token), sizeof(token));
                    file.read(reinterpret_cast<char*>(&target), sizeof(target));
                    if (token > static_cast<u8>(Shader::Maxwell::Flow::Token::PLONGJMP)) {
                        throw std::ios_base::failure{"invalid CFG stack token"};
                    }
                    if (!Shader::Maxwell::Location::IsRawOffset(target)) {
                        throw std::ios_base::failure{"invalid CFG stack target"};
                    }
                    stack.push_back({static_cast<Shader::Maxwell::Flow::Token>(token),
                                     Shader::Maxwell::Location::FromRawOffset(target)});
                }
                block.stack.SetEntries(std::move(stack));
                file.read(reinterpret_cast<char*>(&indirect_count), sizeof(indirect_count));
                if (indirect_count > MAX_CFG_TEMPLATE_INDIRECT_BRANCHES) {
                    throw std::ios_base::failure{"too many CFG indirect branches"};
                }
                block.indirect_branches.resize(indirect_count);
                for (auto& [target, address] : block.indirect_branches) {
                    u32 id{};
                    file.read(reinterpret_cast<char*>(&id), sizeof(id));
                    file.read(reinterpret_cast<char*>(&address), sizeof(address));
                    target = decode_id(id);
                }
            }
            if (!Shader::Maxwell::Flow::CFG::IsValidTemplate(source)) {
                throw std::ios_base::failure{"invalid CFG template graph"};
            }
            if (!loaded_templates.try_emplace(key, std::move(source)).second) {
                throw std::ios_base::failure{"duplicate CFG template key"};
            }
        }
        {
            std::unique_lock lock{cfg_templates_mutex};
            cfg_templates = std::move(loaded_templates);
        }
        LOG_INFO(Render_Vulkan, "Loaded {} CFG templates", count);
    } catch (...) {
        LOG_WARNING(Render_Vulkan, "Discarding corrupt CFG template cache");
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
}

void PipelineCache::SaveCfgTemplates(const std::filesystem::path& path) const {
    using Template = Shader::Maxwell::Flow::CFG::Template;
    std::vector<std::pair<CfgTemplateKey, Template>> snapshot;
    {
        std::shared_lock lock{cfg_templates_mutex};
        snapshot.reserve(cfg_templates.size());
        for (const auto& entry : cfg_templates) {
            if (Shader::Maxwell::Flow::CFG::IsValidTemplate(entry.second)) {
                snapshot.push_back(entry);
            }
        }
    }
    std::ranges::sort(snapshot, [](const auto& lhs, const auto& rhs) {
        const auto& [lhs_key, lhs_template] = lhs;
        const auto& [rhs_key, rhs_template] = rhs;
        static_cast<void>(lhs_template);
        static_cast<void>(rhs_template);
        return std::tie(lhs_key.unique_hash, lhs_key.start_address, lhs_key.exits_to_dispatcher) <
               std::tie(rhs_key.unique_hash, rhs_key.start_address, rhs_key.exits_to_dispatcher);
    });
    if (snapshot.empty()) return;
    const auto tmp_path = path.parent_path() / (path.filename().string() + ".tmp");
    try {
        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ofstream::failbit);
        const u32 count = static_cast<u32>(snapshot.size());
        file.write(CFG_TEMPLATE_CACHE_MAGIC.data(), CFG_TEMPLATE_CACHE_MAGIC.size());
        file.write(reinterpret_cast<const char*>(&CFG_TEMPLATE_CACHE_VERSION), sizeof(u32));
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        const auto write_u32 = [&file](u32 value) { file.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
        for (const auto& [key, source] : snapshot) {
            file.write(reinterpret_cast<const char*>(&key.unique_hash), sizeof(key.unique_hash));
            write_u32(key.start_address);
            const u8 exits = key.exits_to_dispatcher ? 1 : 0;
            file.write(reinterpret_cast<const char*>(&exits), sizeof(exits));
            write_u32(static_cast<u32>(source.functions.size()));
            write_u32(static_cast<u32>(source.blocks.size()));
            for (const auto& function : source.functions) {
                write_u32(function.entrypoint.Offset());
                write_u32(static_cast<u32>(function.blocks.size()));
                for (const size_t id : function.blocks) write_u32(static_cast<u32>(id));
            }
            const auto encode_id = [](size_t id) { return id == Shader::Maxwell::Flow::CFG::NoTemplateBlock ? std::numeric_limits<u32>::max() : static_cast<u32>(id); };
            for (const auto& block : source.blocks) {
                write_u32(block.begin.Offset()); write_u32(block.end.Offset());
                const u8 end_class = static_cast<u8>(block.end_class);
                const auto [pred, negated] = block.cond.GetPred();
                const u16 flow_test = static_cast<u16>(block.cond.GetFlowTest());
                const u8 pred_value = static_cast<u8>(pred), negated_value = negated ? 1 : 0;
                file.write(reinterpret_cast<const char*>(&end_class), sizeof(end_class));
                file.write(reinterpret_cast<const char*>(&flow_test), sizeof(flow_test));
                file.write(reinterpret_cast<const char*>(&pred_value), sizeof(pred_value));
                file.write(reinterpret_cast<const char*>(&negated_value), sizeof(negated_value));
                write_u32(encode_id(block.branch_true)); write_u32(encode_id(block.branch_false));
                write_u32(static_cast<u32>(block.function_call)); write_u32(encode_id(block.return_block));
                const u64 branch_reg = static_cast<u64>(block.branch_reg);
                file.write(reinterpret_cast<const char*>(&branch_reg), sizeof(branch_reg));
                file.write(reinterpret_cast<const char*>(&block.branch_offset), sizeof(block.branch_offset));
                const auto& stack = block.stack.Entries();
                const u32 stack_count = static_cast<u32>(stack.size());
                file.write(reinterpret_cast<const char*>(&stack_count), sizeof(stack_count));
                for (const auto& entry : stack) {
                    const u8 token = static_cast<u8>(entry.token);
                    file.write(reinterpret_cast<const char*>(&token), sizeof(token)); write_u32(entry.target.Offset());
                }
                write_u32(static_cast<u32>(block.indirect_branches.size()));
                for (const auto& [target, address] : block.indirect_branches) { write_u32(encode_id(target)); write_u32(address); }
            }
        }
        file.close();
    } catch (...) {
        LOG_WARNING(Render_Vulkan, "Failed to save CFG templates");
        std::error_code ignored; std::filesystem::remove(tmp_path, ignored);
        return;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        LOG_WARNING(Render_Vulkan, "Failed to install CFG template cache: {}", ec.message());
        std::filesystem::remove(tmp_path, ec);
    }
}

void PipelineCache::LoadPrecacheCfgArtifactCache(const std::filesystem::path& path) {
    auto loaded = VideoCommon::LoadPrecacheCfgArtifacts(path);
    if (!loaded) {
        return;
    }
    decltype(precache_cfg_artifacts) validated;
    std::array<u64, 7> stage_counts{};
    validated.reserve(loaded->size());
    for (auto& artifact : *loaded) {
        const auto stage = artifact.key.stage;
        if (!artifact.IsValid() || !validated.try_emplace(artifact.key, std::move(artifact)).second) {
            LOG_WARNING(Render_Vulkan, "Discarding invalid scanner CFG artifact cache");
            return;
        }
        ++stage_counts[static_cast<size_t>(stage)];
    }
    {
        std::unique_lock lock{precache_cfg_artifacts_mutex};
        precache_cfg_artifacts = std::move(validated);
        precache_cfg_artifact_quarantine.clear();
        precache_cfg_artifact_module_claimed_keys.clear();
    }
    precache_cfg_artifact_loaded.store(loaded->size(), std::memory_order_relaxed);
    for (size_t index = 0; index < stage_counts.size(); ++index) {
        precache_cfg_artifact_loaded_stages[index].store(stage_counts[index],
                                                          std::memory_order_relaxed);
    }
    LOG_INFO(Render_Vulkan, "Loaded {} scanner CFG artifacts", loaded->size());
}

void PipelineCache::LoadPrecacheFrontendArtifactCache(const std::filesystem::path& path) {
    auto loaded = VideoCommon::LoadPrecacheFrontendArtifacts(path);
    if (!loaded) return;
    decltype(precache_frontend_artifacts) validated;
    validated.reserve(loaded->size());
    for (auto& record : *loaded) {
        if (!record.IsValid() ||
            !validated.try_emplace(record.key, std::move(record.artifact)).second) {
            LOG_WARNING(Render_Vulkan, "Discarding invalid scanner frontend artifact cache");
            return;
        }
    }
    {
        std::unique_lock lock{precache_frontend_artifacts_mutex};
        precache_frontend_artifacts = std::move(validated);
        precache_frontend_artifact_quarantine.clear();
        precache_frontend_artifact_shadow_validated.clear();
    }
    precache_frontend_artifact_loaded.store(loaded->size(), std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "Loaded {} scanner frontend artifacts", loaded->size());
}

std::optional<VideoCommon::PrecacheFrontendArtifact>
PipelineCache::LookupPrecacheFrontendArtifact(
    const VideoCommon::PrecacheFrontendArtifactKey& key) const {
    precache_frontend_artifact_lookups.fetch_add(1, std::memory_order_relaxed);
    try {
        std::shared_lock lock{precache_frontend_artifacts_mutex};
        if (precache_frontend_artifact_quarantine.contains(key)) {
            precache_frontend_artifact_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
        const auto it = precache_frontend_artifacts.find(key);
        if (it == precache_frontend_artifacts.end()) {
            precache_frontend_artifact_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
        precache_frontend_artifact_hits.fetch_add(1, std::memory_order_relaxed);
        return it->second;
    } catch (...) {
        precache_frontend_artifact_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

void PipelineCache::QuarantinePrecacheFrontendArtifact(
    const VideoCommon::PrecacheFrontendArtifactKey& key) const {
    try {
        std::unique_lock lock{precache_frontend_artifacts_mutex};
        if (precache_frontend_artifact_quarantine.insert(key).second) {
            precache_frontend_artifact_quarantined.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "Quarantined scanner frontend replay artifact for this session");
        }
    } catch (...) {
    }
}

void PipelineCache::QuarantinePrecacheCfgArtifact(
    const VideoCommon::PrecacheCfgArtifactKey& key) const {
    try {
        std::unique_lock lock{precache_cfg_artifacts_mutex};
        if (precache_cfg_artifact_quarantine.insert(key).second) {
            ++precache_cfg_artifact_quarantined;
            LOG_WARNING(Render_Vulkan, "Quarantined scanner CFG replay artifact for this session");
        }
    } catch (...) {
    }
}

std::optional<PipelineCache::PrecacheCfgReplay>
PipelineCache::LookupPrecacheCfgArtifact(u64 program_identity, Shader::Stage stage,
                                         u32 cfg_request_address,
                                         bool exits_to_dispatcher) const {
    try {
        const Shader::Maxwell::Location entry{cfg_request_address};
        const auto key = VideoCommon::MakePrecacheCfgArtifactKey(
            program_identity, stage, entry.Offset(), exits_to_dispatcher);
        if (!key) {
            return std::nullopt;
        }
        precache_cfg_artifact_replay_attempts.fetch_add(1, std::memory_order_relaxed);
        VideoCommon::PrecacheCfgArtifact artifact;
        {
            std::shared_lock lock{precache_cfg_artifacts_mutex};
            if (precache_cfg_artifact_quarantine.contains(*key)) {
                precache_cfg_artifact_replay_fallbacks.fetch_add(1, std::memory_order_relaxed);
                return std::nullopt;
            }
            const auto it = precache_cfg_artifacts.find(*key);
            if (it == precache_cfg_artifacts.end()) {
                precache_cfg_artifact_replay_fallbacks.fetch_add(1, std::memory_order_relaxed);
                return std::nullopt;
            }
            artifact = it->second;
        }
        auto rebased = artifact.RebaseFor(program_identity, stage, entry, exits_to_dispatcher);
        if (rebased) {
            auto decoded_instructions{artifact.decoded_instructions};
            const s64 relocation = static_cast<s64>(entry.Offset()) -
                                   static_cast<s64>(artifact.cfg.functions.front().entrypoint.Offset());
            for (auto& instruction : decoded_instructions) {
                const s64 relocated = static_cast<s64>(instruction.location) + relocation;
                if (relocated < 0 || relocated > std::numeric_limits<u32>::max()) {
                    ++precache_cfg_artifact_replay_fallbacks;
                    QuarantinePrecacheCfgArtifact(*key);
                    return std::nullopt;
                }
                instruction.location = static_cast<u32>(relocated);
            }
            if (!VideoCommon::PredecodedInstructionsMatchTemplate(*rebased,
                                                                    decoded_instructions)) {
                ++precache_cfg_artifact_replay_fallbacks;
                QuarantinePrecacheCfgArtifact(*key);
                return std::nullopt;
            }
            precache_cfg_artifact_replay_hits.fetch_add(1, std::memory_order_relaxed);
            return PrecacheCfgReplay{.cfg = std::move(*rebased),
                                     .decoded_instructions = std::move(decoded_instructions),
                                     .key = *key};
        } else {
            precache_cfg_artifact_replay_fallbacks.fetch_add(1, std::memory_order_relaxed);
            QuarantinePrecacheCfgArtifact(*key);
        }
        return std::nullopt;
    } catch (...) {
        precache_cfg_artifact_replay_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

void PipelineCache::ShadowValidatePrecacheCfgArtifact(
    u64 program_identity, Shader::Stage stage, bool exits_to_dispatcher,
    const Shader::Maxwell::Flow::CFG& fresh_cfg) const {
    Shader::Maxwell::Flow::CFG::Template fresh_template;
    try {
        fresh_template = fresh_cfg.MakeTemplate();
    } catch (...) {
        return;
    }
    if (fresh_template.functions.empty()) {
        return;
    }
    const u32 cfg_entry_address = fresh_template.functions.front().entrypoint.Offset();
    const auto key = VideoCommon::MakePrecacheCfgArtifactKey(
        program_identity, stage, cfg_entry_address, exits_to_dispatcher);
    if (!key) {
        return;
    }
    precache_cfg_artifact_probes.fetch_add(1, std::memory_order_relaxed);
    precache_cfg_artifact_probe_stages[static_cast<size_t>(stage)].fetch_add(
        1, std::memory_order_relaxed);
    VideoCommon::PrecacheCfgArtifact artifact;
    {
        std::shared_lock lock{precache_cfg_artifacts_mutex};
        const auto it = precache_cfg_artifacts.find(*key);
        if (it == precache_cfg_artifacts.end()) return;
        artifact = it->second;
    }
    precache_cfg_artifact_lookups.fetch_add(1, std::memory_order_relaxed);
    try {
        const auto rebased = artifact.RebaseFor(
            program_identity, stage, Shader::Maxwell::Location::FromRawOffset(cfg_entry_address),
            exits_to_dispatcher);
        if (rebased && *rebased == fresh_template) {
            precache_cfg_artifact_verified.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    } catch (...) {
    }
    const u64 rejection = precache_cfg_artifact_rejected.fetch_add(1, std::memory_order_relaxed);
    if (rejection < 20) {
        LOG_WARNING(Render_Vulkan, "Scanner CFG artifact graph mismatch for 0x{:016x}",
                    program_identity);
    }
}

void PipelineCache::RecordShaderCompilerWork(std::chrono::microseconds cfg_time,
                                             std::chrono::microseconds template_time,
                                             std::chrono::microseconds finalize_time,
                                             std::chrono::microseconds emit_time,
                                             bool exact_module_hit) const {
    shader_cfg_us.fetch_add(static_cast<u64>(cfg_time.count()), std::memory_order_relaxed);
    shader_template_us.fetch_add(static_cast<u64>(template_time.count()),
                                  std::memory_order_relaxed);
    shader_finalize_us.fetch_add(static_cast<u64>(finalize_time.count()),
                                  std::memory_order_relaxed);
    shader_emit_us.fetch_add(static_cast<u64>(emit_time.count()), std::memory_order_relaxed);
    shader_stage_count.fetch_add(1, std::memory_order_relaxed);
    if (exact_module_hit) {
        shader_exact_module_hits.fetch_add(1, std::memory_order_relaxed);
    }

    const auto now = std::chrono::steady_clock::now();
    std::unique_lock lock{shader_compiler_stats_mutex, std::try_to_lock};
    if (!lock || now - shader_compiler_stats_last_log < std::chrono::seconds{30}) {
        return;
    }
    shader_compiler_stats_last_log = now;
    LOG_INFO(Render_Vulkan,
             "Shader compiler work: stages={} exact-module-hits={} compute-prefrontend(disk-hits/live-candidates/live-hits)={}/{}/{} cfg-template "
             "(learned/hits)={}/{} shadow-roundtrip(ok/rejected)={}/{} "
             "cached-form(ok/rejected)={}/{} module-shadow(ok/rejected/ms)={}/{}/{} "
             "module-kind(graphics/compute/merged)={}/{}/{} "
             "scanner-cfg-shadow(loaded/probes/lookups/ok/rejected)={}/{}/{}/{}/{} "
             "scanner-cfg-module-source={} "
             "scanner-cfg-module-shadow(claimed/ok/rejected)={}/{}/{} "
             "scanner-cfg-replay(attempts/hits/fallbacks)={}/{}/{} "
             "scanner-predecoded(hits/instructions)={}/{} "
             "scanner-cfg-replay-quarantined={} "
             "scanner-frontend(full graphics loaded/pipeline-candidates/full-hits/lookups/found/stages-restored/restore-failures/fallbacks/quarantined/restore-ms)={}/{}/{}/{}/{}/{}/{}/{}/{}/{} "
             "scanner-frontend-nonlive-skipped={} "
             "scanner-frontend-shadow(stages-ok/pipelines-rejected/ms)={}/{}/{} "
             "scanner-cfg-cfg-ms(replay/fresh-fallback/fresh-reference)={}/{}/{} "
             "scanner-cfg-stages-loaded(VB/TC/TE/G/F/C/VA)={}/{}/{}/{}/{}/{}/{} "
             "scanner-cfg-stage-probes(VB/TC/TE/G/F/C/VA)={}/{}/{}/{}/{}/{}/{} "
             "cumulative ms "
             "(CFG/template/finalize/SPIR-V)={}/{}/{}/{} "
             "pipeline-enqueue(G-count/ms/max-ms C-count/ms/max-ms)={}/{}/{}/{}/{}/{} "
             "pipeline-build(G-queue-ms/max-ms driver-count/ms/max-ms wait-count/ms/max-ms "
             "C-queue-ms/max-ms driver-count/ms/max-ms wait-count/ms/max-ms)="
             "{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}",
             shader_stage_count.load(std::memory_order_relaxed),
             shader_exact_module_hits.load(std::memory_order_relaxed),
             compute_prefrontend_disk_hits.load(std::memory_order_relaxed),
             compute_prefrontend_live_candidates.load(std::memory_order_relaxed),
             compute_prefrontend_live_hits.load(std::memory_order_relaxed),
             cfg_template_inserts.load(std::memory_order_relaxed),
             cfg_template_hits.load(std::memory_order_relaxed),
             cfg_template_shadow_verified.load(std::memory_order_relaxed),
             cfg_template_shadow_rejected.load(std::memory_order_relaxed),
             cfg_template_cached_verified.load(std::memory_order_relaxed),
             cfg_template_cached_rejected.load(std::memory_order_relaxed),
             cfg_template_module_verified.load(std::memory_order_relaxed),
             cfg_template_module_rejected.load(std::memory_order_relaxed),
             cfg_template_module_us.load(std::memory_order_relaxed) / 1000,
             cfg_template_module_graphics_verified.load(std::memory_order_relaxed),
             cfg_template_module_compute_verified.load(std::memory_order_relaxed),
             cfg_template_module_merged_vertex_verified.load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded.load(std::memory_order_relaxed),
             precache_cfg_artifact_probes.load(std::memory_order_relaxed),
             precache_cfg_artifact_lookups.load(std::memory_order_relaxed),
             precache_cfg_artifact_verified.load(std::memory_order_relaxed),
             precache_cfg_artifact_rejected.load(std::memory_order_relaxed),
             precache_cfg_artifact_module_sources.load(std::memory_order_relaxed),
             precache_cfg_artifact_module_claims.load(std::memory_order_relaxed),
             precache_cfg_artifact_module_verified.load(std::memory_order_relaxed),
             precache_cfg_artifact_module_rejected.load(std::memory_order_relaxed),
             precache_cfg_artifact_replay_attempts.load(std::memory_order_relaxed),
             precache_cfg_artifact_replay_hits.load(std::memory_order_relaxed),
             precache_cfg_artifact_replay_fallbacks.load(std::memory_order_relaxed),
             precache_decoded_artifact_hits.load(std::memory_order_relaxed),
             precache_decoded_artifact_instructions.load(std::memory_order_relaxed),
             precache_cfg_artifact_quarantined.load(std::memory_order_relaxed),
             precache_frontend_artifact_loaded.load(std::memory_order_relaxed),
             precache_frontend_pipeline_candidates.load(std::memory_order_relaxed),
             precache_frontend_pipeline_full_hits.load(std::memory_order_relaxed),
             precache_frontend_artifact_lookups.load(std::memory_order_relaxed),
             precache_frontend_artifact_hits.load(std::memory_order_relaxed),
             precache_frontend_artifact_restored.load(std::memory_order_relaxed),
             precache_frontend_pipeline_restore_failures.load(std::memory_order_relaxed),
             precache_frontend_artifact_fallbacks.load(std::memory_order_relaxed),
             precache_frontend_artifact_quarantined.load(std::memory_order_relaxed),
             precache_frontend_artifact_restore_us.load(std::memory_order_relaxed) / 1000,
             precache_frontend_nonlive_skipped.load(std::memory_order_relaxed),
             precache_frontend_artifact_shadow_verified.load(std::memory_order_relaxed),
             precache_frontend_artifact_shadow_rejected.load(std::memory_order_relaxed),
             precache_frontend_artifact_shadow_us.load(std::memory_order_relaxed) / 1000,
             precache_cfg_artifact_replay_cfg_us.load(std::memory_order_relaxed) / 1000,
             precache_cfg_artifact_fresh_cfg_us.load(std::memory_order_relaxed) / 1000,
             precache_cfg_artifact_validation_cfg_us.load(std::memory_order_relaxed) / 1000,
             precache_cfg_artifact_loaded_stages[0].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[1].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[2].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[3].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[4].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[5].load(std::memory_order_relaxed),
             precache_cfg_artifact_loaded_stages[6].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[0].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[1].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[2].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[3].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[4].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[5].load(std::memory_order_relaxed),
             precache_cfg_artifact_probe_stages[6].load(std::memory_order_relaxed),
             shader_cfg_us.load(std::memory_order_relaxed) / 1000,
             shader_template_us.load(std::memory_order_relaxed) / 1000,
             shader_finalize_us.load(std::memory_order_relaxed) / 1000,
             shader_emit_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_create_count.load(std::memory_order_relaxed),
             graphics_pipeline_create_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_create_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_create_count.load(std::memory_order_relaxed),
             compute_pipeline_create_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_create_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_queue_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_queue_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_driver_count.load(std::memory_order_relaxed),
             graphics_pipeline_driver_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_driver_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_count.load(std::memory_order_relaxed),
             graphics_pipeline_wait_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_queue_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_queue_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_count.load(std::memory_order_relaxed),
             compute_pipeline_driver_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_count.load(std::memory_order_relaxed),
             compute_pipeline_wait_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_max_us.load(std::memory_order_relaxed) / 1000);
    LOG_INFO(Render_Vulkan,
             "Pipeline actual timing: G-driver boot(count/ms/max)={}/{}/{} live={}/{}/{} "
             "G-wait boot(count/ms/max)={}/{}/{} live={}/{}/{} "
             "C-driver boot(count/ms/max)={}/{}/{} live={}/{}/{} "
             "C-wait boot(count/ms/max)={}/{}/{} live={}/{}/{}",
             graphics_pipeline_driver_boot_count.load(std::memory_order_relaxed),
             graphics_pipeline_driver_boot_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_driver_boot_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_driver_live_count.load(std::memory_order_relaxed),
             graphics_pipeline_driver_live_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_driver_live_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_boot_count.load(std::memory_order_relaxed),
             graphics_pipeline_wait_boot_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_boot_max_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_live_count.load(std::memory_order_relaxed),
             graphics_pipeline_wait_live_us.load(std::memory_order_relaxed) / 1000,
             graphics_pipeline_wait_live_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_boot_count.load(std::memory_order_relaxed),
             compute_pipeline_driver_boot_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_boot_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_live_count.load(std::memory_order_relaxed),
             compute_pipeline_driver_live_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_driver_live_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_boot_count.load(std::memory_order_relaxed),
             compute_pipeline_wait_boot_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_boot_max_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_live_count.load(std::memory_order_relaxed),
             compute_pipeline_wait_live_us.load(std::memory_order_relaxed) / 1000,
             compute_pipeline_wait_live_max_us.load(std::memory_order_relaxed) / 1000);
}

void PipelineCache::RecordPipelineCreateWork(bool compute,
                                             std::chrono::microseconds create_time) const {
    auto& count = compute ? compute_pipeline_create_count : graphics_pipeline_create_count;
    auto& total = compute ? compute_pipeline_create_us : graphics_pipeline_create_us;
    auto& maximum = compute ? compute_pipeline_create_max_us : graphics_pipeline_create_max_us;
    const u64 elapsed{static_cast<u64>(create_time.count())};
    count.fetch_add(1, std::memory_order_relaxed);
    total.fetch_add(elapsed, std::memory_order_relaxed);
    u64 observed{maximum.load(std::memory_order_relaxed)};
    while (observed < elapsed &&
           !maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
    }
}

void PipelineCache::RecordPipelineBuildQueueDelay(bool compute,
                                                  std::chrono::microseconds delay) const {
    auto& total = compute ? compute_pipeline_queue_us : graphics_pipeline_queue_us;
    auto& maximum = compute ? compute_pipeline_queue_max_us : graphics_pipeline_queue_max_us;
    const u64 elapsed{static_cast<u64>(delay.count())};
    total.fetch_add(elapsed, std::memory_order_relaxed);
    u64 observed{maximum.load(std::memory_order_relaxed)};
    while (observed < elapsed &&
           !maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
    }
}

void PipelineCache::RecordPipelineDriverCreate(bool compute, bool boot_preload,
                                               std::chrono::microseconds duration) const {
    auto& count = compute ? compute_pipeline_driver_count : graphics_pipeline_driver_count;
    auto& total = compute ? compute_pipeline_driver_us : graphics_pipeline_driver_us;
    auto& maximum = compute ? compute_pipeline_driver_max_us : graphics_pipeline_driver_max_us;
    auto& origin_count = compute ? (boot_preload ? compute_pipeline_driver_boot_count
                                                 : compute_pipeline_driver_live_count)
                                 : (boot_preload ? graphics_pipeline_driver_boot_count
                                                 : graphics_pipeline_driver_live_count);
    auto& origin_total = compute ? (boot_preload ? compute_pipeline_driver_boot_us
                                                 : compute_pipeline_driver_live_us)
                                 : (boot_preload ? graphics_pipeline_driver_boot_us
                                                 : graphics_pipeline_driver_live_us);
    auto& origin_maximum = compute ? (boot_preload ? compute_pipeline_driver_boot_max_us
                                                   : compute_pipeline_driver_live_max_us)
                                   : (boot_preload ? graphics_pipeline_driver_boot_max_us
                                                   : graphics_pipeline_driver_live_max_us);
    const u64 elapsed{static_cast<u64>(duration.count())};
    count.fetch_add(1, std::memory_order_relaxed);
    total.fetch_add(elapsed, std::memory_order_relaxed);
    origin_count.fetch_add(1, std::memory_order_relaxed);
    origin_total.fetch_add(elapsed, std::memory_order_relaxed);
    u64 observed{maximum.load(std::memory_order_relaxed)};
    while (observed < elapsed &&
           !maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
    }
    observed = origin_maximum.load(std::memory_order_relaxed);
    while (observed < elapsed &&
           !origin_maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
    }
}

void PipelineCache::RecordPipelineBuildWait(bool compute, bool boot_preload,
                                            std::chrono::microseconds duration) const {
    auto& count = compute ? compute_pipeline_wait_count : graphics_pipeline_wait_count;
    auto& total = compute ? compute_pipeline_wait_us : graphics_pipeline_wait_us;
    auto& maximum = compute ? compute_pipeline_wait_max_us : graphics_pipeline_wait_max_us;
    auto& origin_count = compute ? (boot_preload ? compute_pipeline_wait_boot_count
                                                 : compute_pipeline_wait_live_count)
                                 : (boot_preload ? graphics_pipeline_wait_boot_count
                                                 : graphics_pipeline_wait_live_count);
    auto& origin_total = compute ? (boot_preload ? compute_pipeline_wait_boot_us
                                                 : compute_pipeline_wait_live_us)
                                 : (boot_preload ? graphics_pipeline_wait_boot_us
                                                 : graphics_pipeline_wait_live_us);
    auto& origin_maximum = compute ? (boot_preload ? compute_pipeline_wait_boot_max_us
                                                   : compute_pipeline_wait_live_max_us)
                                   : (boot_preload ? graphics_pipeline_wait_boot_max_us
                                                   : graphics_pipeline_wait_live_max_us);
    const u64 elapsed{static_cast<u64>(duration.count())};
    count.fetch_add(1, std::memory_order_relaxed);
    total.fetch_add(elapsed, std::memory_order_relaxed);
    origin_count.fetch_add(1, std::memory_order_relaxed);
    origin_total.fetch_add(elapsed, std::memory_order_relaxed);
    u64 observed{maximum.load(std::memory_order_relaxed)};
    while (observed < elapsed &&
           !maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
    }
    observed = origin_maximum.load(std::memory_order_relaxed);
    while (observed < elapsed &&
           !origin_maximum.compare_exchange_weak(observed, elapsed, std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
    }
}

// Runtime-state diagnostic — see the header for what it is measuring. Diagnostic only: nothing
// here changes what gets cached, guessed, or served.
void PipelineCache::RecordGenericInputTypesCardinalityDiagnostic(u64 unique_hash,
                                                                   u64 generic_input_types_hash) const {
    constexpr size_t kMaxTrackedVariantsForDiagnostics = 8;
    bool should_log = false;
    {
        std::unique_lock lock{phase5_diag_generic_input_types_mutex};
        auto& variants = phase5_diag_generic_input_types_variants_by_hash[unique_hash];
        if (std::find(variants.begin(), variants.end(), generic_input_types_hash) ==
                variants.end() &&
            variants.size() < kMaxTrackedVariantsForDiagnostics) {
            variants.push_back(generic_input_types_hash);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - phase5_diag_generic_input_types_last_log_time >= std::chrono::seconds{30}) {
            phase5_diag_generic_input_types_last_log_time = now;
            should_log = true;
        }
    }
    if (!should_log) {
        return;
    }
    std::array<size_t, 4> buckets{}; // [0]=1 variant, [1]=2-3, [2]=4-7, [3]=8+ (capped)
    size_t total_hashes = 0;
    {
        std::shared_lock lock{phase5_diag_generic_input_types_mutex};
        for (const auto& [hash, variants] : phase5_diag_generic_input_types_variants_by_hash) {
            ++total_hashes;
            const size_t n = variants.size();
            if (n <= 1) {
                ++buckets[0];
            } else if (n <= 3) {
                ++buckets[1];
            } else if (n <= 7) {
                ++buckets[2];
            } else {
                ++buckets[3];
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Runtime-state diagnostic: of {} graphics shaders seen with a real draw so far, {} "
             "showed exactly 1 distinct generic_input_types state, {} showed 2-3, {} showed "
             "4-7, {} showed 8+ (capped — true count may be higher). Diagnostic only, nothing "
             "currently acts on this. Low cardinality across most hashes would say "
             "generic_input_types is realistically chaseable the way y_negate was; a lot of "
             "8+ hashes would say it isn't -- more real attribute-format combinations than a "
             "small guess or spec-constant set could plausibly cover.",
             total_hashes, buckets[0], buckets[1], buckets[2], buckets[3]);
}

// Convert-depth diagnostic confirms or corrects the reasoned-not-measured
// default (runtime_info.h's ApplySpeculativeDefaults). Diagnostic only.
void PipelineCache::RecordConvertDepthModeDiagnostic(bool convert_depth_mode) const {
    bool should_log = false;
    u64 true_count = 0;
    u64 total_count = 0;
    {
        std::unique_lock lock{phase5_diag_convert_depth_mode_mutex};
        ++phase5_diag_convert_depth_mode_total_count;
        if (convert_depth_mode) {
            ++phase5_diag_convert_depth_mode_true_count;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - phase5_diag_convert_depth_mode_last_log_time >= std::chrono::seconds{30}) {
            phase5_diag_convert_depth_mode_last_log_time = now;
            should_log = true;
            true_count = phase5_diag_convert_depth_mode_true_count;
            total_count = phase5_diag_convert_depth_mode_total_count;
        }
    }
    if (!should_log) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Convert-depth diagnostic: of {} real VertexB/Geometry translations so far, "
             "convert_depth_mode was true (DepthMode::MinusOneToOne) for {} ({:.1f}%). The "
             "speculative default guesses true, reasoned from that enum's HW value of 0, not "
             "measured -- a real majority true confirms it, a real majority false means it "
             "should flip.",
             total_count, true_count,
             total_count > 0 ? 100.0 * static_cast<double>(true_count) /
                                   static_cast<double>(total_count)
                             : 0.0);
}

// Tessellation-state diagnostic confirms or corrects the
// defaults, the lowest-confidence guesses in that pass (runtime_info.h's
// ApplySpeculativeDefaults). Diagnostic only.
void PipelineCache::RecordTessellationStateDiagnostic(Shader::TessPrimitive primitive,
                                                        Shader::TessSpacing spacing,
                                                        bool clockwise) const {
    bool should_log = false;
    decltype(phase5_diag_tess_primitive_counts) primitive_counts;
    decltype(phase5_diag_tess_spacing_counts) spacing_counts;
    u64 clockwise_true_count = 0;
    u64 total_count = 0;
    {
        std::unique_lock lock{phase5_diag_tess_state_mutex};
        ++phase5_diag_tess_primitive_counts[primitive];
        ++phase5_diag_tess_spacing_counts[spacing];
        ++phase5_diag_tess_total_count;
        if (clockwise) {
            ++phase5_diag_tess_clockwise_true_count;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - phase5_diag_tess_state_last_log_time >= std::chrono::seconds{30}) {
            phase5_diag_tess_state_last_log_time = now;
            should_log = true;
            primitive_counts = phase5_diag_tess_primitive_counts;
            spacing_counts = phase5_diag_tess_spacing_counts;
            clockwise_true_count = phase5_diag_tess_clockwise_true_count;
            total_count = phase5_diag_tess_total_count;
        }
    }
    if (!should_log) {
        return;
    }
    const auto get = [](const auto& counts, auto key) {
        const auto it = counts.find(key);
        return it != counts.end() ? it->second : u64{0};
    };
    LOG_INFO(Render_Vulkan,
             "Tessellation-state diagnostic: of {} real TessellationEval translations so far -- "
             "tess_primitive: {} Isolines, {} Triangles, {} Quads (default guess: "
             "Triangles); tess_spacing: {} Equal, {} FractionalOdd, {} FractionalEven "
             "(default guess: Equal); tess_clockwise: {} true, {} false (default guess: "
             "false). All three defaults were picked for internal consistency (each enum's "
            "own value-0 entry), not measured -- the lowest-confidence runtime-state "
             "preset. Real gameplay data on any of these should replace the guess it "
             "disagrees with.",
             total_count, get(primitive_counts, Shader::TessPrimitive::Isolines),
             get(primitive_counts, Shader::TessPrimitive::Triangles),
             get(primitive_counts, Shader::TessPrimitive::Quads),
             get(spacing_counts, Shader::TessSpacing::Equal),
             get(spacing_counts, Shader::TessSpacing::FractionalOdd),
             get(spacing_counts, Shader::TessSpacing::FractionalEven), clockwise_true_count,
             total_count - clockwise_true_count);
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    // The graphics-cache lookup timing fix must happen here, after
    // unique_hashes/state are current but before current_pipeline->Next()/graphics_cache are
    // consulted below, since graphics_key IS the lookup key both of those use.
    graphics_key.phase4_prototype_needs_array_variant = ResolvePhase4PrototypeSpecValue();

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            if (next->IsFailed()) {
                current_pipeline = next;
                return nullptr;
            }
            current_pipeline = next;
            // Update last use frame
            graphics_pipeline_last_use[current_pipeline] = scheduler.CurrentTick();
            return BuiltPipeline(current_pipeline);
        }
    }
    GraphicsPipeline* result = CurrentGraphicsPipelineSlowPath();
    if (result) {
        graphics_pipeline_last_use[result] = scheduler.CurrentTick();
    }
    return result;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {
    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& pipeline{pair->second};
    if (!is_new && pipeline) {
        compute_pipeline_last_use[pipeline.get()] = scheduler.CurrentTick();
        return pipeline.get();
    }
    pipeline = CreateComputePipeline(key, shader);
    if (pipeline) {
        compute_pipeline_last_use[pipeline.get()] = scheduler.CurrentTick();
    }
    return pipeline.get();
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    const auto shader_dir{Common::FS::GetCitronPath(Common::FS::CitronPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename = base_dir / "vulkan.bin";

    // Scanner runs without a live Vulkan Device. Persist this exact target contract
    // after any real boot so a later ROM scan emits modules for this renderer rather
    // than its old generic guessed profile.
    if (SavePrecacheCompilerTarget(base_dir / "precache_compiler_target.bin",
                                   precache_compiler_target)) {
        LOG_INFO(Render_Vulkan, "Pre-cache compiler target saved: {:016x} (GPU accuracy {})",
                 VideoCommon::CompilerTargetFingerprint(precache_compiler_target),
                 precache_compiler_target.gpu_accuracy_mode);
    }

    // Adaptive slot learning must run before the SPIR-V
    // cache load below and before any shader translation this session, since
    // IsPhase4PrototypeSlot (environment.h), which texture_key computation depends on, reads
    // whatever table this publishes. Failure-safe: LoadPhase4PrototypeSlots returns {} on any
    // error, and MergePhase4PrototypeSlots({}) is just an empty table -- a missing/corrupt
    // file degrades to ordinary raw-key behavior for every coordinate, same as a fresh
    // profile that's never hit this path before, rather than to a crash or a stale state.
    phase4_prototype_slots_filename = base_dir / "phase4_prototype_slots.bin";
    Shader::SetActivePhase4PrototypeSlots(Shader::MergePhase4PrototypeSlots(
        VideoCommon::LoadPhase4PrototypeSlots(phase4_prototype_slots_filename)));

    // Load SPIR-V cache — feeds the GPL speculative path and AOT scanner results.
    spirv_cache_filename = base_dir / "spirv_cache.bin";
    spirv_cache.Load(spirv_cache_filename);
    precache_cfg_artifacts_filename = base_dir / fmt::format(
        "precache_cfg_artifacts_{:016x}.bin",
        VideoCommon::CompilerTargetFingerprint(precache_compiler_target));
    LoadPrecacheCfgArtifactCache(precache_cfg_artifacts_filename);
    precache_frontend_artifacts_filename = base_dir / fmt::format(
        "precache_frontend_artifacts_{:016x}.bin",
        VideoCommon::CompilerTargetFingerprint(precache_compiler_target));
    LoadPrecacheFrontendArtifactCache(precache_frontend_artifacts_filename);
    if constexpr (CFG_TEMPLATE_PERSISTENCE_ENABLED) {
        cfg_templates_filename = base_dir / "cfg_templates.bin";
        LoadCfgTemplates(cfg_templates_filename);
    }


    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename = base_dir / "vulkan_pipelines.bin";
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, VULKAN_PIPELINE_CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        size_t total_compute{};
        size_t total_graphics{};
        size_t skipped_graphics{};
        size_t invalid{};
        size_t feature_mismatch{};
    } state;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        const auto identity = env.ProgramIdentity();
        if (!env.HasValidEntryInstruction() || !identity || *identity != key.unique_hash) {
            ++state.invalid;
            return;
        }

        workers.QueueWork([this, key, env_ = std::move(env), &state, &callback]() mutable {
            CITRON_PROFILE_SCOPE("Vulkan::PipelineCacheWorker");
            ShaderPools pools;
            auto pipeline{CreateComputePipeline(pools, key, env_, state.statistics.get(), false)};
            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                compute_pipeline_last_use[pipeline.get()] = scheduler.CurrentTick();
                compute_cache.emplace(key, std::move(pipeline));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
        ++state.total_compute;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if constexpr (!DISK_GRAPHICS_PIPELINE_PRELOAD_ENABLED) {
            // File position must still advance past the key for LoadPipelines'
            // record framing. Do not construct a VkPipeline at boot; this
            // leaves live cache misses on the ordinary on-demand path.
            ++state.skipped_graphics;
            return;
        }

        const auto program_index = [](Shader::Stage stage) -> std::optional<size_t> {
            switch (stage) {
            case Shader::Stage::VertexA:
                return 0;
            case Shader::Stage::VertexB:
                return 1;
            case Shader::Stage::TessellationControl:
                return 2;
            case Shader::Stage::TessellationEval:
                return 3;
            case Shader::Stage::Geometry:
                return 4;
            case Shader::Stage::Fragment:
                return 5;
            case Shader::Stage::Compute:
                return std::nullopt;
            }
            return std::nullopt;
        };
        std::array<bool, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> seen_stages{};
        const bool valid_contract = std::ranges::all_of(envs, [&](const FileEnvironment& env) {
            const auto identity = env.ProgramIdentity();
            const auto index = program_index(env.ShaderStage());
            if (!env.HasValidEntryInstruction() || !identity || !index ||
                seen_stages[*index] || key.unique_hashes[*index] != *identity) {
                return false;
            }
            seen_stages[*index] = true;
            return true;
        });
        const bool complete_stage_chain = [&] {
            for (size_t index = 0; index < seen_stages.size(); ++index) {
                if (seen_stages[index] != (key.unique_hashes[index] != 0)) {
                    return false;
                }
            }
            return true;
        }();
        if (!valid_contract || !complete_stage_chain) {
            ++state.invalid;
            return;
        }

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_extra != 0) !=
                dynamic_features.has_extended_dynamic_state_2_extra ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.dynamic_vertex_input != 0) != dynamic_features.has_dynamic_vertex_input ||
            (key.state.xfb_enabled != 0 && !dynamic_features.has_transform_feedback)) {
            // NOTE: xfb_enabled uses a unidirectional check. It encodes both a
            // device capability AND per-pipeline runtime state. We only reject
            // the pipeline if it actively requires XFB but the host device
            // does not support it
            ++state.feature_mismatch;
            return;
        }
        workers.QueueWork([this, key, envs_ = std::move(envs), &state, &callback]() mutable {
            CITRON_PROFILE_SCOPE("Vulkan::PipelineCacheWorker");
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false)};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                // Initialize last-use frame for disk-loaded pipelines so they
                // survive EvictOldPipelines() calls before their first use.
                // Without this, pipelines loaded from disk have no entry in
                // graphics_pipeline_last_use and are immediately evicted on
                // any memory pressure event.
                graphics_pipeline_last_use[pipeline.get()] = scheduler.CurrentTick();
                graphics_cache.emplace(key, std::move(pipeline));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
        ++state.total_graphics;
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, TRANSFERABLE_CACHE_VERSION,
                               load_compute, load_graphics);

    if (state.invalid != 0) {
        LOG_WARNING(Render_Vulkan, "Skipped {} cached pipelines with invalid shader entry points",
                    state.invalid);
    }
    if (state.feature_mismatch != 0) {
        LOG_WARNING(Render_Vulkan,
                    "Skipped {} cached graphics pipelines with incompatible dynamic-state "
                    "features",
                    state.feature_mismatch);
    }
    if (state.skipped_graphics != 0) {
        LOG_INFO(Render_Vulkan,
                 "Skipped {} cached graphics pipelines during boot; live requests build on demand",
                 state.skipped_graphics);
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);

    // Pre-reserve space in caches to reduce rehashing during async builds
    {
        std::scoped_lock lock{state.mutex};
        if (state.total_compute > 0) {
            compute_cache.reserve(state.total_compute);
        }
        if (state.total_graphics > 0) {
            graphics_cache.reserve(state.total_graphics);
        }
    }
    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);

    LOG_INFO(Render_Vulkan,
             "Scanner frontend replay boot scope: {} non-live requests skipped",
             precache_frontend_nonlive_skipped.load(std::memory_order_relaxed));

    // Boot-time disk-cache replay above can throw dozens of no-context field-mismatch
    // samples at spirv_cache's diagnostic throttle within milliseconds on the worker
    // pool — resetting here gives live play (the case that actually matters for
    // judging stutter) its own fresh budget instead of starting already exhausted.
    // See ResetFieldMismatchLogBudget()'s doc comment in spirv_cache.h.
    spirv_cache.ResetFieldMismatchLogBudget();

    // Log SPIR-V cache effectiveness.
    const size_t hits   = spirv_cache.HitCount();
    const size_t probes = spirv_cache.LookupCount();
    if (probes == 0) {
        const size_t loaded = spirv_cache.Size();
        if (loaded == 0) {
            LOG_INFO(Render_Vulkan, "SPIR-V cache: empty — first boot or cache not yet populated.");
        } else {
            LOG_INFO(Render_Vulkan,
                     "SPIR-V cache: {} entries loaded but no probes during disk build "
                     "(all pipelines may have been rebuilt from disk cache).",
                     loaded);
        }
    } else {
        const int pct = static_cast<int>(hits * 100 / probes);
        LOG_INFO(Render_Vulkan,
                 "SPIR-V cache: {}/{} stage hits ({}% -- {} EmitSPIRV calls avoided).",
                 hits, probes, pct, hits);
    }


    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     VULKAN_PIPELINE_CACHE_VERSION);
    }

    if (state.statistics) {
        state.statistics->Report();
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    GraphicsPipeline* transition_source = current_pipeline;
    if (is_new) {
        pipeline = CreateGraphicsPipeline();
    }
    if (!pipeline) {
        return nullptr;
    }
    if (transition_source && transition_source != pipeline.get()) {
        transition_source->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    return BuiltPipeline(current_pipeline);
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsFailed()) {
        return nullptr;
    }
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    const auto& state = maxwell3d->draw_manager->GetDrawState();
    if (state.index_buffer.count <= 32 || state.vertex_buffer.count <= 32) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel) try {
    auto hash = key.Hash();
    LOG_INFO(Render_Vulkan, "0x{:016x}", hash);
    const u64 compiler_target_key =
        VideoCommon::CompilerTargetFingerprint(precache_compiler_target);
    size_t env_index{0};
    std::array<Shader::IR::Program, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> programs;
    std::array<std::chrono::microseconds, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        cfg_times{};
    std::array<std::chrono::microseconds, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        template_times{};
    std::array<std::chrono::microseconds, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        finalize_times{};
    // Track env pointer per shader index so the emit loop can compute cbuf keys.
    std::array<Shader::Environment*, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> stage_envs{};
    std::array<std::optional<Shader::Maxwell::Flow::CFG::Template>,
               Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        cfg_shadow_templates{};
    std::array<std::optional<VideoCommon::PrecacheCfgArtifactKey>,
               Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        cfg_shadow_scanner_artifacts{};
    std::array<std::optional<CfgTemplateKey>, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        cfg_shadow_keys{};
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

    // Restore only complete real live graphics chains. FileEnvironment is the
    // boot disk-preload path: it has a serialized historical context, not the
    // current draw state, and must never consume scanner frontend recipes.
    // A separate pool keeps a
    // failed transaction from leaving one restored stage in main pools; then
    // normal frontend rebuild runs for every stage in that pipeline. The
    // generated layer-passthrough stage has no scanner source recipe, so it
    // remains normal fallback when this device needs it.
    const bool frontend_pipeline_live =
        !envs.empty() && std::ranges::all_of(envs, [](const Shader::Environment* env) {
            return env != nullptr && env->AsGenericEnvironment() != nullptr;
        });
    if (!envs.empty() && !frontend_pipeline_live) {
        precache_frontend_nonlive_skipped.fetch_add(1, std::memory_order_relaxed);
    }
    bool frontend_pipeline_candidate = SCANNER_FRONTEND_ARTIFACT_REUSE_ENABLED &&
                                       host_info.support_viewport_index_layer &&
                                       frontend_pipeline_live;
    if (frontend_pipeline_candidate) {
        precache_frontend_pipeline_candidates.fetch_add(1, std::memory_order_relaxed);
    }
    std::array<std::optional<VideoCommon::PrecacheFrontendArtifact>,
               Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        frontend_artifacts{};
    std::array<std::optional<VideoCommon::PrecacheFrontendArtifactKey>,
               Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        frontend_artifact_keys{};
    size_t frontend_env_index{};
    if (frontend_pipeline_candidate) {
        for (size_t index = 0; index < key.unique_hashes.size(); ++index) {
            if (key.unique_hashes[index] == 0) continue;
            if (frontend_env_index >= envs.size()) {
                frontend_pipeline_candidate = false;
                break;
            }
            Shader::Environment& env{*envs[frontend_env_index++]};
            const VideoCommon::PrecacheFrontendArtifactKey artifact_key{
                .program_identity = key.unique_hashes[index],
                .compiler_target_fingerprint = compiler_target_key,
                .source_header = Common::CityHash64(
                    reinterpret_cast<const char*>(&env.SPH()), sizeof(Shader::ProgramHeader)),
                .local_memory_size = env.LocalMemorySize(),
                .stage = env.ShaderStage(),
                .scheduler_slot = static_cast<u8>(env.StartAddress() % 32),
                .exits_to_dispatcher = false,
            };
            frontend_artifact_keys[index] = artifact_key;
            frontend_artifacts[index] = LookupPrecacheFrontendArtifact(artifact_key);
            if (!frontend_artifacts[index]) {
                frontend_pipeline_candidate = false;
                break;
            }
        }
        if (frontend_env_index != envs.size()) frontend_pipeline_candidate = false;
    }
    std::optional<ShaderPools> frontend_replay_pools;
    std::array<Shader::IR::Program, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
        frontend_replay_programs;
    bool frontend_pipeline_replay = frontend_pipeline_candidate;
    if (frontend_pipeline_replay) {
        frontend_replay_pools.emplace();
        const auto restore_begin = std::chrono::steady_clock::now();
        for (size_t index = 0; index < frontend_artifacts.size(); ++index) {
            if (!frontend_artifacts[index]) continue;
            auto restored = VideoCommon::RestorePrecacheFrontendArtifact(
                *frontend_artifacts[index], frontend_replay_pools->inst, frontend_replay_pools->block);
            if (!restored) {
                frontend_pipeline_replay = false;
                precache_frontend_pipeline_restore_failures.fetch_add(1,
                                                                       std::memory_order_relaxed);
                break;
            }
            frontend_replay_programs[index] = std::move(*restored);
        }
        // Frozen frontend recipes cross a larger semantic boundary than CFG
        // recipes. Until an exact live frontend build has agreed with a recipe
        // in this session, shadow it before any restored stage reaches the
        // regular pipeline chain. This deliberately costs one fresh frontend
        // build per recipe/session; disagreement quarantines the whole atomic
        // pipeline transaction and leaves every stage on normal translation.
        bool needs_frontend_shadow{};
        if (frontend_pipeline_replay) {
            std::shared_lock lock{precache_frontend_artifacts_mutex};
            for (const auto& artifact_key : frontend_artifact_keys) {
                if (artifact_key &&
                    !precache_frontend_artifact_shadow_validated.contains(*artifact_key)) {
                    needs_frontend_shadow = true;
                    break;
                }
            }
        }
        if (needs_frontend_shadow) {
            const auto shadow_begin = std::chrono::steady_clock::now();
            bool shadow_matches{true};
            size_t shadow_env_index{};
            try {
                ShaderPools shadow_pools;
                for (size_t index = 0; index < key.unique_hashes.size(); ++index) {
                    if (!frontend_artifacts[index]) continue;
                    if (shadow_env_index >= envs.size()) {
                        shadow_matches = false;
                        break;
                    }
                    Shader::Environment& shadow_env{*envs[shadow_env_index++]};
                    Shader::Maxwell::Flow::CFG shadow_cfg{
                        shadow_env, shadow_pools.flow_block,
                        static_cast<u32>(shadow_env.StartAddress() +
                                         sizeof(Shader::ProgramHeader)),
                        index == 0};
                    auto fresh_program{Shader::Maxwell::BuildProgramTemplate(
                        shadow_pools.inst, shadow_pools.block, shadow_env, shadow_cfg, host_info)};
                    VideoCommon::PrecacheFrontendFreezeError freeze_error{};
                    const auto fresh_artifact{
                        VideoCommon::FreezePrecacheFrontendArtifact(fresh_program, freeze_error)};
                    if (!fresh_artifact || *fresh_artifact != *frontend_artifacts[index]) {
                        shadow_matches = false;
                        break;
                    }
                }
                shadow_matches = shadow_matches && shadow_env_index == envs.size();
            } catch (...) {
                shadow_matches = false;
            }
            const auto shadow_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - shadow_begin)
                    .count());
            precache_frontend_artifact_shadow_us.fetch_add(shadow_us,
                                                            std::memory_order_relaxed);
            if (shadow_matches) {
                size_t newly_validated{};
                std::unique_lock lock{precache_frontend_artifacts_mutex};
                for (const auto& artifact_key : frontend_artifact_keys) {
                    if (artifact_key) {
                        newly_validated += precache_frontend_artifact_shadow_validated
                                               .insert(*artifact_key)
                                               .second;
                    }
                }
                const u64 validated_total =
                    precache_frontend_artifact_shadow_verified.fetch_add(
                        newly_validated, std::memory_order_relaxed) +
                    newly_validated;
                if (newly_validated != 0 && validated_total <= 16) {
                    LOG_INFO(Render_Vulkan,
                             "Scanner frontend shadow checked {} graphics stage(s) in {} us for "
                             "pipeline {:016x}",
                             frontend_env_index, shadow_us, hash);
                }
            } else {
                frontend_pipeline_replay = false;
                precache_frontend_artifact_shadow_rejected.fetch_add(1,
                                                                      std::memory_order_relaxed);
                LOG_WARNING(Render_Vulkan,
                            "Scanner frontend shadow mismatch; normal frontend fallback for "
                            "pipeline {:016x}",
                            hash);
            }
        }
        if (!frontend_pipeline_replay) {
            for (const auto& artifact_key : frontend_artifact_keys) {
                if (artifact_key) QuarantinePrecacheFrontendArtifact(*artifact_key);
            }
            frontend_replay_pools.reset();
        } else {
            const auto restore_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - restore_begin)
                    .count());
            const u64 replay_number =
                precache_frontend_pipeline_full_hits.fetch_add(1, std::memory_order_relaxed) + 1;
            precache_frontend_artifact_restore_us.fetch_add(
                restore_us,
                std::memory_order_relaxed);
            precache_frontend_artifact_restored.fetch_add(
                frontend_env_index, std::memory_order_relaxed);
            if (replay_number <= 16) {
                LOG_INFO(Render_Vulkan,
                         "Scanner frontend replay #{} restored {} graphics stage(s) in {} us for "
                         "pipeline {:016x}",
                         replay_number, frontend_env_index, restore_us, hash);
            }
        }
    }

    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        const bool is_emulated_stage =
            layer_source_program != nullptr &&
            index == static_cast<u32>(Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;
        stage_envs[index] = &env;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        const auto cfg_start = std::chrono::steady_clock::now();
        const CfgTemplateKey cfg_template_key{key.unique_hashes[index], env.StartAddress(),
                                              index == 0};
        std::optional<Shader::Maxwell::Flow::CFG> cfg;
        std::optional<PrecacheCfgReplay> scanner_cfg_replay;
        if constexpr (SCANNER_CFG_ARTIFACT_REUSE_ENABLED) {
            scanner_cfg_replay = LookupPrecacheCfgArtifact(
                key.unique_hashes[index], env.ShaderStage(), cfg_offset, index == 0);
            if (scanner_cfg_replay) {
                const auto replay_cfg_start = std::chrono::steady_clock::now();
                try {
                    cfg.emplace(env, pools.flow_block, scanner_cfg_replay->cfg);
                } catch (...) {
                    ++precache_cfg_artifact_replay_fallbacks;
                    QuarantinePrecacheCfgArtifact(scanner_cfg_replay->key);
                    scanner_cfg_replay.reset();
                }
                precache_cfg_artifact_replay_cfg_us.fetch_add(
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - replay_cfg_start).count()),
                    std::memory_order_relaxed);
            }
        }
        if (!cfg) {
            const auto fresh_cfg_start = std::chrono::steady_clock::now();
            if constexpr (CFG_TEMPLATE_REUSE_ENABLED) {
                if (auto source = LookupCfgTemplate(cfg_template_key)) {
                    cfg.emplace(env, pools.flow_block, *source);
                } else {
                    cfg.emplace(env, pools.flow_block, cfg_offset, index == 0);
                }
            } else {
                cfg.emplace(env, pools.flow_block, cfg_offset, index == 0);
            }
            if (precache_cfg_artifact_loaded.load(std::memory_order_relaxed) != 0) {
                precache_cfg_artifact_fresh_cfg_us.fetch_add(
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - fresh_cfg_start).count()),
                    std::memory_order_relaxed);
            }
        }
        if (!scanner_cfg_replay) {
            ShadowValidatePrecacheCfgArtifact(
                key.unique_hashes[index], env.ShaderStage(), index == 0, *cfg);
        }
        // A CFG branch-table read makes reachability dependent on real cbuf
        // data. Shadow/persistence may retain only cbuf-independent graphs.
        if constexpr (CFG_TEMPLATE_SHADOW_VALIDATION_ENABLED ||
                      CFG_TEMPLATE_PERSISTENCE_ENABLED) {
            if (auto* generic_env = env.AsGenericEnvironment(); generic_env != nullptr &&
                generic_env->CapturedCbufValues().empty()) {
                ValidateAndPersistCfgTemplate(cfg_template_key, env, *cfg);
                if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
                    cfg_shadow_templates[index] = scanner_cfg_replay ? scanner_cfg_replay->cfg
                                                                       : cfg->MakeTemplate();
                    if (scanner_cfg_replay) {
                        ++precache_cfg_artifact_module_sources;
                        cfg_shadow_scanner_artifacts[index] = scanner_cfg_replay->key;
                    }
                    cfg_shadow_keys[index] = cfg_template_key;
                }
            }
        }
        if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
            if (precache_cfg_artifact_loaded.load(std::memory_order_relaxed) != 0 &&
                !cfg_shadow_templates[index]) {
                cfg_shadow_templates[index] = scanner_cfg_replay ? scanner_cfg_replay->cfg
                                                                   : cfg->MakeTemplate();
                cfg_shadow_keys[index] = cfg_template_key;
            }
            if (scanner_cfg_replay && !cfg_shadow_scanner_artifacts[index]) {
                ++precache_cfg_artifact_module_sources;
                cfg_shadow_scanner_artifacts[index] = scanner_cfg_replay->key;
            }
        }
        cfg_times[index] = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cfg_start);
        if (scanner_cfg_replay) {
            precache_decoded_artifact_hits.fetch_add(1, std::memory_order_relaxed);
            precache_decoded_artifact_instructions.fetch_add(
                scanner_cfg_replay->decoded_instructions.size(), std::memory_order_relaxed);
        }
        const auto template_start = std::chrono::steady_clock::now();
        if (!uses_vertex_a || index != 1) {
            if (frontend_pipeline_replay && frontend_artifacts[index]) {
                programs[index] = std::move(frontend_replay_programs[index]);
            } else {
                programs[index] = Shader::Maxwell::BuildProgramTemplate(
                    pools.inst, pools.block, env, *cfg, host_info,
                    scanner_cfg_replay
                        ? std::span<const Shader::Maxwell::PredecodedInstruction>{
                              scanner_cfg_replay->decoded_instructions}
                        : std::span<const Shader::Maxwell::PredecodedInstruction>{});
            }
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb = frontend_pipeline_replay && frontend_artifacts[index]
                                  ? std::move(frontend_replay_programs[index])
                                  : Shader::Maxwell::BuildProgramTemplate(
                                        pools.inst, pools.block, env, *cfg, host_info,
                                        scanner_cfg_replay
                                            ? std::span<const Shader::Maxwell::PredecodedInstruction>{
                                                  scanner_cfg_replay->decoded_instructions}
                                            : std::span<const Shader::Maxwell::PredecodedInstruction>{});
            template_times[index] = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - template_start);
            const auto finalize_start = std::chrono::steady_clock::now();
            Shader::Maxwell::FinalizeProgramTemplate(program_va, *stage_envs[0], host_info);
            Shader::Maxwell::FinalizeProgramTemplate(program_vb, env, host_info);
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
            finalize_times[index] = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - finalize_start);
            cfg_times[index] += cfg_times[0];
            template_times[index] += template_times[0];
            continue;
        }
        template_times[index] = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - template_start);
        // VertexA is finalized together with VertexB above: MergeDualVertexPrograms
        // requires both programs before their state-dependent passes run.
        if (uses_vertex_a && uses_vertex_b && index == 0) {
            continue;
        }
        const auto finalize_start = std::chrono::steady_clock::now();
        Shader::Maxwell::FinalizeProgramTemplate(programs[index], env, host_info);
        finalize_times[index] = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - finalize_start);

        if (Settings::values.dump_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }

    }
    std::array<const Shader::Info*, Tegra::Engines::Maxwell3D::Regs::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Tegra::Engines::Maxwell3D::Regs::MaxShaderStage> modules;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
         index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        const bool is_emulated_stage =
            layer_source_program != nullptr &&
            index == static_cast<u32>(Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        // This is the one place real Shader::Info exists for a
        // freshly-translated fragment shader (stage_index 4 == Shader::Stage::Fragment, per
        // StageFromIndex -- NOT the same as key.unique_hashes' own index 5 for this same
        // stage, ShaderType::Pixel's raw hardware position; index here is 1 less than that
        // because this loop's `index` uses the raw ShaderType numbering directly, same as
        // key.unique_hashes, while stage_index is already converted). Recording once here
        // means ResolvePhase4PrototypeSpecValue never needs to guess for a shader it's already
        // seen -- see phase4_prototype_fragment_shader_table's doc comment, vk_pipeline_cache.h.
        if (stage_index == 4) {
            // Bitmask over Shader::ActivePhase4PrototypeSlots(), not a single any_of bool -- a
            // shader can in principle have more than one known slot, each needing its own
            // bit rather than all of them collapsing into one "has some marked slot or other"
            // flag. desc.phase4_prototype_slot_id is only meaningful when
            // desc.phase4_prototype_polymorphic is true (see shader_info.h), which is
            // exactly the condition guarding its use here.
            u32 marked_slot_mask = 0;
            for (const Shader::TextureDescriptor& desc : program.info.texture_descriptors) {
                if (desc.phase4_prototype_polymorphic) {
                    marked_slot_mask |= (1U << desc.phase4_prototype_slot_id);
                }
            }
            // See phase4_prototype_fragment_shader_table_mutex's doc comment,
            // vk_pipeline_cache.h -- this function can run on a worker thread
            // (workers.QueueWork, the boot-time bulk pipeline-loading path), so this write
            // needs real synchronization, not just the table itself existing.
            std::unique_lock lock{phase4_prototype_fragment_shader_table_mutex};
            phase4_prototype_fragment_shader_table[key.unique_hashes[index]] = marked_slot_mask;
        }

        const auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage)};
        ConvertLegacyToGeneric(program, runtime_info);
        // Runtime diagnostics live here, not inside MakeRuntimeInfo:
        // MakeRuntimeInfo is a free function (no implicit `this`), these diagnostics are
        // PipelineCache members. Reading back from the just-returned runtime_info instead
        // of threading extra parameters into MakeRuntimeInfo.
        if (program.stage == Shader::Stage::VertexB || program.stage == Shader::Stage::Geometry) {
            RecordConvertDepthModeDiagnostic(runtime_info.convert_depth_mode);
        }
        if (program.stage == Shader::Stage::TessellationEval) {
            RecordTessellationStateDiagnostic(runtime_info.tess_primitive,
                                               runtime_info.tess_spacing,
                                               runtime_info.tess_clockwise);
        }
        // SPIR-V cache check
        // AsGenericEnvironment() returns nullptr for FileEnvironment (disk-load path)
        // and this* for GraphicsEnvironment (live path). Avoids dynamic_cast/-frtti.
        // FileEnvironment doesn't derive from GenericEnvironment, but it deserializes
        // the same real cbuf/texture capture data from disk (whatever the pipeline
        // was actually specialized with when it was originally compiled live and
        // serialized — see GenericEnvironment::Serialize / FileEnvironment::Deserialize).
        // AsFileEnvironment() reaches it the same no-RTTI way, so the disk-replay path
        // only falls back to the 0/"no specialization" guess when there's genuinely
        // nowhere to read real data from.
        const auto* gen_env_stage = stage_envs[index]->AsGenericEnvironment();
        const auto* file_env_stage = stage_envs[index]->AsFileEnvironment();
        const bool has_real_specialization_context =
            gen_env_stage != nullptr || file_env_stage != nullptr;
        // Cbuf narrowing is enabled. Validated across two independent full play
        // sessions (see the diagnostic below): texture-handle-only cbuf reads account
        // for ~24-28% of cbuf_key-caused stale misses where real narrowing data was
        // available — consistent between a fresh-wipe session (1126 eligible samples,
        // 28.2% would-match) and a post-precache session with far more boot-replay
        // traffic (2278 eligible samples, 21.9% would-match) once the FileEnvironment
        // gap below was closed. Not a majority fix, but real, safe (see
        // ReadCbufValueForTextureHandle's doc comment in environment.h for exactly
        // why the exclusion set can only ever contain genuinely-redundant reads), and
        // worth taking. The diagnostic further down (diag_cbuf_key_excl_texture_handles)
        // is now redundant with this — it will report eligible=0 forever going
        // forward, since it's computed with the identical formula key.cbuf_key now
        // uses. That's expected, not a bug; left in place rather than removed in the
        // same change that flips production matching behavior, so this diff stays as
        // small and reviewable as possible. Safe to clean up separately later.
        const u64 cbuf_key =
            gen_env_stage    ? ComputeCbufKeyExcludingTextureHandles(
                                   gen_env_stage->CapturedCbufValues(),
                                   gen_env_stage->CapturedTextureHandleCbufKeys())
            : file_env_stage ? ComputeCbufKeyExcludingTextureHandles(
                                   file_env_stage->CapturedCbufValues(),
                                   file_env_stage->CapturedTextureHandleCbufKeys())
                             : 0;
        // A resource handle is an allocation detail, not shader input. When every observed
        // texture query is tied to a source cbuf slot, use that logical slot instead so a
        // scanner's synthetic descriptors and a later real draw agree. A shader that actually
        // touches a polymorphic slot still owns its specialised handle exclusion and keeps the
        // established raw-key path; unrelated shaders are not penalized by that global table.
        const bool use_gen_logical_texture_key =
            gen_env_stage && !HasActivePhase4LogicalTextureSlot(
                                 gen_env_stage->CapturedLogicalTextureSlots()) &&
            HasCompleteLogicalTextureCoverage(gen_env_stage->CapturedLogicalTextureSlots(),
                                              gen_env_stage->CapturedLogicalTextureHandles(),
                                              gen_env_stage->CapturedTextureTypes(),
                                              gen_env_stage->CapturedTexturePixelFormats());
        const bool use_file_logical_texture_key =
            file_env_stage && !HasActivePhase4LogicalTextureSlot(
                                  file_env_stage->CapturedLogicalTextureSlots()) &&
            HasCompleteLogicalTextureCoverage(file_env_stage->CapturedLogicalTextureSlots(),
                                              file_env_stage->CapturedLogicalTextureHandles(),
                                              file_env_stage->CapturedTextureTypes(),
                                              file_env_stage->CapturedTexturePixelFormats());
        const u64 texture_key =
            use_gen_logical_texture_key    ? ComputeLogicalTextureKey(
                                                 gen_env_stage->CapturedLogicalTextureSlots())
            : use_file_logical_texture_key ? ComputeLogicalTextureKey(
                                                 file_env_stage->CapturedLogicalTextureSlots())
            : gen_env_stage                ? ComputeTextureKeyExcludingHandles(
                                                 gen_env_stage->CapturedTextureTypes(),
                                                 gen_env_stage->CapturedTexturePixelFormats(),
                                                 gen_env_stage->CapturedPhase4PrototypeHandles())
            : file_env_stage               ? ComputeTextureKey(file_env_stage->CapturedTextureTypes(),
                                                                file_env_stage->CapturedTexturePixelFormats())
                                         : 0;
        spirv_cache.RecordTextureKeyMode(use_gen_logical_texture_key ||
                                         use_file_logical_texture_key);
        // Was a second independent ComputeCbufKeyExcludingTextureHandles(...) call here,
        // duplicating the work cbuf_key above already did — kept as a genuinely separate
        // computation while validating whether narrowing was worth shipping at all, so
        // Lookup()'s diagnostic could tell real narrowing data apart from "no data,
        // defaulted." That question's answered now (cbuf_key above already reflects the
        // narrowed value in production) and the duplicate computation turned out to be
        // a real, measurable cost on this hot path once eligible/matched activity got
        // heavy enough — reusing cbuf_key directly instead. This also makes it
        // impossible for the two to read differently, which they were doing in
        // practice for a reason that was still open when this was cut over; see the
        // handoff notes for that thread if it's worth resuming later — it wasn't a
        // multi-threading issue (confirmed: every captured sample came from the same
        // thread id), so whatever it was, it's a separate question from whether
        // narrowing itself is safe to ship, which two full play sessions already
        // answered before this diagnostic started costing more than it was worth.
        const u64 diag_cbuf_key_excl_texture_handles = cbuf_key;
        // diag_base_runtime_hash captures runtime_info.SpirvRelevantHash(stage) BEFORE
        // any folding — for VertexB this also includes the viewport-transform-state
        // fold below, since that fold (unlike the binding fold) genuinely reflects
        // fixed-function GPU state at PositionPass()-translation time, not a
        // per-pipeline-position offset.
        // Passed through to Lookup()/Insert() purely so the field-mismatch diagnostic
        // in spirv_cache.cpp can tell whether a runtime_key mismatch traces to this
        // core RuntimeInfo state or to the binding-offset fold below — see
        // spirv_cache.h's Insert()/Lookup() doc comments.
        //
        // SpirvRelevantHash(stage), not Hash(): folds only the RuntimeInfo fields
        // THIS stage's SPIR-V emission actually reads (see its doc comment in
        // runtime_info.h for the full per-field justification) instead of hashing
        // the whole struct regardless of stage. Confirmed measurably over-broad for
        // the common case — Fragment shaders were picking up entropy from
        // input_topology and force_early_z despite neither ever being read for
        // Fragment codegen, over-invalidating cache entries for reasons that could
        // never have produced different SPIR-V bytes in the first place.
        const Shader::Stage current_stage = stage_envs[index]->ShaderStage();
        const auto diag_runtime_fields = runtime_info.SpirvRelevantFieldHashes(current_stage);
        u64 runtime_key = runtime_info.SpirvRelevantHash(current_stage);
        if (current_stage == Shader::Stage::VertexB) {
            // env.ReadViewportTransformState() is not stored in RuntimeInfo, but
            // PositionPass() (ir_opt/position_pass.cpp) branches on it directly during
            // TranslateProgram() above, before runtime_info even exists: when the real
            // GPU's viewport_scale_offset_enabled is 0, the vertex shader must do a
            // manual render-area-relative position remap; when it's 1, it must not.
            // Without folding this bit into the key, a VertexB program compiled under
            // one state (e.g. a speculative pre-cache guess, which always assumes 1)
            // can be served to a draw using the other state, silently skipping/adding
            // the remap and corrupting vertex output — most visible on screen-space
            // overlay/UI-style geometry.
            const u64 viewport_transform_state = stage_envs[index]->ReadViewportTransformState();
            runtime_key = FoldViewportTransformState(runtime_key, viewport_transform_state);
        }
        // Snapshot the diagnostic "core" component here — everything folded into
        // runtime_key so far except the binding-offset fold below. See the comment
        // above the runtime_key declaration for why viewport_transform_state belongs
        // on this side of the split rather than being treated like binding_key.
        const u64 diag_base_runtime_hash = runtime_key;
        // `binding` at this point holds the starting Bindings accumulator for THIS
        // stage. ComputeBindingKey() hashes the full state (see its doc comment in
        // spirv_cache.h) — this fold, and FoldViewportTransformState() above, are
        // exactly what a speculative InsertSpeculative() call must also apply to
        // its guessed values, or its runtime_key ends up in a different format
        // from every real entry's and can never match one.
        const Shader::Backend::Bindings starting_binding{binding};
        const u64 binding_key = ComputeBindingKey(binding);
        runtime_key = FoldBindingKey(runtime_key, binding_key);
        const SpirvKey spirv_key{key.unique_hashes[index], stage_envs[index]->ShaderStage(),
                                 compiler_target_key, cbuf_key, runtime_key, texture_key};
        std::vector<u32> code;
        std::chrono::microseconds emit_time{};
        const bool is_merged_vertex = uses_vertex_a && uses_vertex_b && index == 1;
        // Since SpirvKey now includes the runtime_key, we can safely serve cached SPIR-V
        // to both the live path and the disk-load path.
        if (!is_merged_vertex) {
            if (auto cached = spirv_cache.Lookup(spirv_key, has_real_specialization_context,
                                                 diag_base_runtime_hash, binding_key,
                                                 diag_cbuf_key_excl_texture_handles,
                                                 diag_runtime_fields)) {
                code = *cached->spirv;
                if (cached->is_speculative) {
                    // Capped like the field-mismatch diagnostic in spirv_cache.cpp — this
                    // logs unconditionally otherwise, and if speculative hits start
                    // happening at real volume during a busy scene, an uncapped LOG_INFO
                    // per hit is exactly the kind of per-frame synchronous logging cost
                    // worth avoiding once the diagnostic has said what it needs to.
                    static std::atomic<size_t> speculative_hit_logs{0};
                    if (size_t expected = speculative_hit_logs.load(); expected < 50 &&
                        speculative_hit_logs.compare_exchange_strong(expected, expected + 1)) {
                        const char* source = cached->source == VideoCommon::SpirvCacheEntrySource::PrecacheScanner
                                                 ? "scanner"
                                                 : "live";
                        LOG_INFO(Render_Vulkan,
                                 "0x{:016x} stage[{}] served from {} SPECULATIVE entry "
                                 "(unique_hash=0x{:016x})",
                                 hash, index, source, key.unique_hashes[index]);
                    }
                }
                // Restore the binding counter to where EmitSPIRV left it when this
                // SPIR-V was first compiled.  Without this, the next stage's
                // EmitSPIRV (or cache miss) would start at the wrong descriptor
                // slot, producing binding collisions and graphical corruption.
                binding = cached->end_binding;
            }
        }
        const bool exact_module_hit = !code.empty();
        if (code.empty()) {
            const auto emit_start = std::chrono::steady_clock::now();
            code = EmitSPIRV(profile, runtime_info, program, binding);
            emit_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - emit_start);
            // binding has now been advanced past this stage's slots.
            // has_real_specialization_context (not just gen_env_stage != nullptr) so a
            // disk-replay translation gets cached too, now that it's keyed with real
            // cbuf/texture data instead of a forced 0 — previously this branch's
            // EmitSPIRV work was simply thrown away every time, guaranteeing this
            // exact stage would miss and re-translate again on every future load.
            if (!is_merged_vertex && has_real_specialization_context) {
                spirv_cache.Insert(spirv_key, code, binding, /*is_speculative=*/false,
                                  diag_base_runtime_hash, binding_key,
                                  diag_cbuf_key_excl_texture_handles, diag_runtime_fields);
                // Runtime-variant diagnostic; see RecordPhase3RuntimeVariantDiagnostic.
                // comment (vk_pipeline_cache.h) for what this measures. Gated on cbuf_key
                // == 0 specifically (not just has_real_specialization_context, which the
                // Insert() above already required): that's the population a speculative
                // entry could ever match at all, since InsertSpeculative() hardcodes
                // cbuf_key=0 unconditionally — cbuf_key != 0 real inserts are already
                // structurally unreachable by any speculative entry regardless of this
                // diagnostic's outcome, so they're not part of the question being asked.
                if (cbuf_key == 0) {
                    RecordPhase3RuntimeVariantDiagnostic(key.unique_hashes[index],
                                                          diag_base_runtime_hash);
                }
                // Runtime-state diagnostic; see RecordGenericInputTypesCardinalityDiagnostic.
                // doc comment (vk_pipeline_cache.h) for what this measures. No cbuf_key
                // gate (see that same doc comment for why not) -- every real insert reaching
                // here is in scope. CityHash64 over the raw array, matching exactly how
                // SpirvRelevantHash/Hash (runtime_info.h) fold this field in, so the values
                // tracked here are the real cache-key-relevant ones, not a different
                // measure that happens to also vary with attribute formats.
                RecordGenericInputTypesCardinalityDiagnostic(
                    key.unique_hashes[index],
                    Common::CityHash64(
                        reinterpret_cast<const char*>(runtime_info.generic_input_types.data()),
                        runtime_info.generic_input_types.size() * sizeof(Shader::AttributeType)));
                if (!spirv_cache_filename.empty()) {
                    serialization_thread.QueueWork([this] { spirv_cache.SaveThrottled(spirv_cache_filename); });
                }
                // Texture-slot feasibility instrumentation; see LogTextureSlotVarianceReportThrottled.
                // doc comment in shader_environment.h. Unconditional (not gated on
                // spirv_cache_filename): unrelated to whether disk persistence is configured.
                VideoCommon::GenericEnvironment::LogTextureSlotVarianceReportThrottled();
            }
            code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
        }
        // Plain stages use one reconstructed CFG. A merged vertex module needs
        // both source CFGs reconstructed before the normal merge pass.
        if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
            if (is_merged_vertex && !is_emulated_stage &&
                !program.info.requires_layer_emulation && cfg_shadow_templates[0] &&
                cfg_shadow_keys[0] && cfg_shadow_templates[index] && cfg_shadow_keys[index] &&
                stage_envs[0] != nullptr && stage_envs[index] != nullptr &&
                stage_envs[0]->AsGenericEnvironment() != nullptr &&
                stage_envs[index]->AsGenericEnvironment() != nullptr &&
                ((cfg_shadow_scanner_artifacts[0] || cfg_shadow_scanner_artifacts[index]) ||
                 (stage_envs[0]->AsGenericEnvironment()->CapturedCbufValues().empty() &&
                  stage_envs[index]->AsGenericEnvironment()->CapturedCbufValues().empty()))) {
                const u64 merged_identity = cfg_shadow_keys[0]->unique_hash ^
                                            (cfg_shadow_keys[index]->unique_hash +
                                             0x9e3779b97f4a7c15ULL +
                                             (cfg_shadow_keys[0]->unique_hash << 6) +
                                             (cfg_shadow_keys[0]->unique_hash >> 2));
                const CfgTemplateKey merged_key{merged_identity,
                                                 cfg_shadow_keys[index]->start_address, false};
                ValidateCfgTemplateMergedVertexModule(
                    merged_key, *stage_envs[0], *cfg_shadow_templates[0], *stage_envs[index],
                    *cfg_shadow_templates[index], runtime_info, program, code, starting_binding,
                    binding,
                    cfg_shadow_scanner_artifacts[0]
                        ? &*cfg_shadow_scanner_artifacts[0]
                        : nullptr,
                    cfg_shadow_scanner_artifacts[index]
                        ? &*cfg_shadow_scanner_artifacts[index]
                        : nullptr);
            } else if (!is_merged_vertex && !is_emulated_stage &&
                       !program.info.requires_layer_emulation && cfg_shadow_templates[index] &&
                       cfg_shadow_keys[index] &&
                       stage_envs[index]->AsGenericEnvironment() != nullptr &&
                       (cfg_shadow_scanner_artifacts[index] ||
                        stage_envs[index]->AsGenericEnvironment()->CapturedCbufValues().empty())) {
                ValidateCfgTemplateGraphicsModule(*cfg_shadow_keys[index], *stage_envs[index],
                                                  *cfg_shadow_templates[index],
                                                  static_cast<u32>(stage_envs[index]->StartAddress() +
                                                                   sizeof(Shader::ProgramHeader)),
                                                  runtime_info, program,
                                                  code,
                                                  starting_binding, binding,
                                                  cfg_shadow_scanner_artifacts[index]
                                                      ? &*cfg_shadow_scanner_artifacts[index]
                                                      : nullptr);
            }
        }
        RecordShaderCompilerWork(cfg_times[index], template_times[index], finalize_times[index], emit_time,
                                 exact_module_hit);
        // Keep the state after this stage, not merely its IR output layout. A
        // later speculative translation begins after this stage in the real
        // pipeline and therefore must allocate descriptors from this exact state.
        {
            std::unique_lock real_stage_stores_lock{real_stage_stores_mutex};
            real_stage_stores_by_hash[key.unique_hashes[index]] = RealStageStoresSnapshot{
                program.info.stores, program.info.legacy_stores_mapping,
                program.info.passthrough, program.is_geometry_passthrough, binding};
        }
        device.SaveShader(code);
        modules[stage_index] = BuildShader(device, code);
        if (device.HasDebuggingToolAttached()) {
            const std::string name{fmt::format("Shader {:016x}", key.unique_hashes[index])};
            modules[stage_index].SetObjectNameEXT(name.c_str());
        }
        previous_stage = &program;
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    const auto pipeline_create_start = std::chrono::steady_clock::now();
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, thread_worker, statistics, render_pass_cache, this,
        !build_in_parallel, key,
        std::move(modules), infos);
    RecordPipelineCreateWork(false, std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pipeline_create_start));
    return pipeline;

} catch (const vk::Exception& exception) {
    if (exception.GetResult() == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        LOG_ERROR(Render_Vulkan,
                  "Out of device memory during graphics pipeline creation, attempting recovery");
        EvictOldPipelines();
        return nullptr;
    }
    throw;
} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    main_pools.ReleaseContents();
    auto pipeline{
        CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(), nullptr, true)};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        CITRON_PROFILE_SCOPE("Vulkan::PipelineCacheSerialize");
        boost::container::static_vector<const GenericEnvironment*,
                                        Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, TRANSFERABLE_CACHE_VERSION);
    });
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start};
    env.SetCachedSize(shader->size_bytes);

    main_pools.ReleaseContents();
    auto pipeline{CreateComputePipeline(main_pools, key, env, nullptr, true)};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        CITRON_PROFILE_SCOPE("Vulkan::PipelineCacheSerialize");
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, TRANSFERABLE_CACHE_VERSION);
    });
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping 0x{:016x}", hash);
        return nullptr;
    }

    LOG_INFO(Render_Vulkan, "0x{:016x}", hash);
    const u64 compiler_target_key =
        VideoCommon::CompilerTargetFingerprint(precache_compiler_target);

    // Disk-replayed compute environments carry the captured specialization
    // inputs before frontend translation. A persisted real recipe lets this
    // exact key build the host pipeline without CFG, IR, or SPIR-V emission.
    if (auto* file_env = env.AsFileEnvironment(); file_env != nullptr) {
        const u64 cbuf_key = ComputeCbufKeyExcludingTextureHandles(
            file_env->CapturedCbufValues(), file_env->CapturedTextureHandleCbufKeys());
        const bool use_logical_texture_key =
            !HasActivePhase4LogicalTextureSlot(file_env->CapturedLogicalTextureSlots()) &&
            HasCompleteLogicalTextureCoverage(file_env->CapturedLogicalTextureSlots(),
                                              file_env->CapturedLogicalTextureHandles(),
                                              file_env->CapturedTextureTypes(),
                                              file_env->CapturedTexturePixelFormats());
        const u64 texture_key = use_logical_texture_key
                                    ? ComputeLogicalTextureKey(file_env->CapturedLogicalTextureSlots())
                                    : ComputeTextureKey(file_env->CapturedTextureTypes(),
                                                        file_env->CapturedTexturePixelFormats());
        spirv_cache.RecordTextureKeyMode(use_logical_texture_key);
        const u64 workgroup_key =
            ComputeWorkgroupKey(key.shared_memory_size, key.workgroup_size);
        const SpirvKey early_key{key.unique_hash, env.ShaderStage(), compiler_target_key,
                                 cbuf_key, workgroup_key, texture_key};
        if (auto cached = spirv_cache.Lookup(early_key, true, workgroup_key, 0, cbuf_key);
            cached && !cached->is_speculative && cached->compute_info) {
            std::vector<u32> code{*cached->spirv};
            code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
            compute_prefrontend_disk_hits.fetch_add(1, std::memory_order_relaxed);
            RecordShaderCompilerWork({}, {}, {}, {}, true);
            device.SaveShader(code);
            vk::ShaderModule spv_module{BuildShader(device, code)};
            if (device.HasDebuggingToolAttached()) {
                const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
                spv_module.SetObjectNameEXT(name.c_str());
            }
            Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
            const auto pipeline_create_start = std::chrono::steady_clock::now();
            auto pipeline = std::make_unique<ComputePipeline>(
                device, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                thread_worker, statistics, &shader_notify, *cached->compute_info,
                std::move(spv_module), this, !build_in_parallel);
            RecordPipelineCreateWork(
                true, std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - pipeline_create_start));
            LOG_INFO(Render_Vulkan,
                     "0x{:016x} compute served before frontend from persisted exact entry", hash);
            return pipeline;
        }
    }
    if (auto* gen_env = env.AsGenericEnvironment(); gen_env != nullptr) {
        const u64 workgroup_key =
            ComputeWorkgroupKey(key.shared_memory_size, key.workgroup_size);
        const auto candidates =
            spirv_cache.LookupLiveComputeCandidates(key.unique_hash, compiler_target_key);
        compute_prefrontend_live_candidates.fetch_add(candidates.size(),
                                                      std::memory_order_relaxed);
        for (const auto& candidate : candidates) {
            if (candidate.key.runtime_key != workgroup_key) {
                continue;
            }
            std::unordered_map<u64, u32> cbuf_values;
            cbuf_values.reserve(candidate.cbuf_keys.size());
            for (const u64 packed_key : candidate.cbuf_keys) {
                const u32 cbuf_index = static_cast<u32>(packed_key >> 32);
                const u32 cbuf_offset = static_cast<u32>(packed_key);
                cbuf_values.emplace(packed_key, gen_env->ReadCbufValue(cbuf_index, cbuf_offset));
            }
            const SpirvKey early_key{key.unique_hash, env.ShaderStage(), compiler_target_key,
                                     ComputeCbufKey(cbuf_values), workgroup_key, 0};
            if (auto cached = spirv_cache.Lookup(early_key, true, workgroup_key, 0,
                                                 early_key.cbuf_key);
                cached && !cached->is_speculative && cached->compute_info) {
                std::vector<u32> code{*cached->spirv};
                code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
                compute_prefrontend_live_hits.fetch_add(1, std::memory_order_relaxed);
                RecordShaderCompilerWork({}, {}, {}, {}, true);
                device.SaveShader(code);
                vk::ShaderModule spv_module{BuildShader(device, code)};
                if (device.HasDebuggingToolAttached()) {
                    const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
                    spv_module.SetObjectNameEXT(name.c_str());
                }
                Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
                const auto pipeline_create_start = std::chrono::steady_clock::now();
                auto pipeline = std::make_unique<ComputePipeline>(
                device, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                thread_worker, statistics, &shader_notify, *cached->compute_info,
                std::move(spv_module), this, !build_in_parallel);
                RecordPipelineCreateWork(
                    true, std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - pipeline_create_start));
                LOG_INFO(Render_Vulkan,
                         "0x{:016x} compute served before frontend from live exact entry", hash);
                return pipeline;
            }
        }
    }

    const auto cfg_start = std::chrono::steady_clock::now();
    const CfgTemplateKey cfg_template_key{key.unique_hash, env.StartAddress(), false};
    std::optional<Shader::Maxwell::Flow::CFG::Template> cfg_shadow_template;
    std::optional<Shader::Maxwell::Flow::CFG> cfg;
    std::optional<PrecacheCfgReplay> scanner_cfg_replay;
    if constexpr (SCANNER_CFG_ARTIFACT_REUSE_ENABLED) {
        scanner_cfg_replay =
            LookupPrecacheCfgArtifact(key.unique_hash, env.ShaderStage(), env.StartAddress(), false);
        if (scanner_cfg_replay) {
            const auto replay_cfg_start = std::chrono::steady_clock::now();
            try {
                cfg.emplace(env, pools.flow_block, scanner_cfg_replay->cfg);
            } catch (...) {
                ++precache_cfg_artifact_replay_fallbacks;
                QuarantinePrecacheCfgArtifact(scanner_cfg_replay->key);
                scanner_cfg_replay.reset();
            }
            precache_cfg_artifact_replay_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - replay_cfg_start).count()),
                std::memory_order_relaxed);
        }
    }
    if (!cfg) {
        const auto fresh_cfg_start = std::chrono::steady_clock::now();
        if constexpr (CFG_TEMPLATE_REUSE_ENABLED) {
            if (auto source = LookupCfgTemplate(cfg_template_key)) {
                cfg.emplace(env, pools.flow_block, *source);
            } else {
                cfg.emplace(env, pools.flow_block, env.StartAddress());
            }
        } else {
            cfg.emplace(env, pools.flow_block, env.StartAddress());
        }
        if (precache_cfg_artifact_loaded.load(std::memory_order_relaxed) != 0) {
            precache_cfg_artifact_fresh_cfg_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - fresh_cfg_start).count()),
                std::memory_order_relaxed);
        }
    }
    if (!scanner_cfg_replay) {
        ShadowValidatePrecacheCfgArtifact(key.unique_hash, env.ShaderStage(), false, *cfg);
    }
    if constexpr (CFG_TEMPLATE_SHADOW_VALIDATION_ENABLED || CFG_TEMPLATE_PERSISTENCE_ENABLED) {
        if (auto* generic_env = env.AsGenericEnvironment(); generic_env != nullptr &&
            generic_env->CapturedCbufValues().empty()) {
            ValidateAndPersistCfgTemplate(cfg_template_key, env, *cfg);
            if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
                cfg_shadow_template = scanner_cfg_replay ? scanner_cfg_replay->cfg
                                                          : cfg->MakeTemplate();
                if (scanner_cfg_replay) {
                    ++precache_cfg_artifact_module_sources;
                }
            }
        }
    }
    if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
        const bool had_shadow_template = cfg_shadow_template.has_value();
        if (precache_cfg_artifact_loaded.load(std::memory_order_relaxed) != 0 &&
            !had_shadow_template) {
            cfg_shadow_template = scanner_cfg_replay ? scanner_cfg_replay->cfg : cfg->MakeTemplate();
        }
        if (scanner_cfg_replay && !had_shadow_template) {
            ++precache_cfg_artifact_module_sources;
        }
    }
    const auto cfg_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - cfg_start);

    // Dump it before error.
    if (Settings::values.dump_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    const auto template_start = std::chrono::steady_clock::now();
    if (scanner_cfg_replay) {
        precache_decoded_artifact_hits.fetch_add(1, std::memory_order_relaxed);
        precache_decoded_artifact_instructions.fetch_add(
            scanner_cfg_replay->decoded_instructions.size(), std::memory_order_relaxed);
    }
    auto program{Shader::Maxwell::BuildProgramTemplate(pools.inst, pools.block, env, *cfg,
                                                        host_info,
                                                        scanner_cfg_replay
                                                            ? std::span<const Shader::Maxwell::PredecodedInstruction>{
                                                                  scanner_cfg_replay->decoded_instructions}
                                                            : std::span<const Shader::Maxwell::PredecodedInstruction>{})};
    const auto template_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - template_start);
    const auto finalize_start = std::chrono::steady_clock::now();
    Shader::Maxwell::FinalizeProgramTemplate(program, env, host_info);
    const auto finalize_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - finalize_start);
    // SPIR-V cache check for compute
    // AsGenericEnvironment() returns nullptr for FileEnvironment (disk-load path).
    // AsFileEnvironment() covers that case with FileEnvironment's own real,
    // disk-deserialized cbuf/texture data — see the matching comment in
    // CreateGraphicsPipeline() above for why this is real captured data, not a
    // guess. This also closes a gap specific to this function: unlike the
    // graphics path, the Insert() below was never gated on having real context,
    // so a FileEnvironment compute miss was inserting under a forced cbuf_key=0/
    // texture_key=0 key — the same collision risk flagged for the graphics path,
    // except actually happening here rather than just possible. Real keys make
    // that insert correct instead of needing to gate it off.
    auto* gen_env = env.AsGenericEnvironment();
    auto* file_env = env.AsFileEnvironment();
    const bool has_real_specialization_context = gen_env != nullptr || file_env != nullptr;
    // Cbuf narrowing uses the same validated switch as CreateGraphicsPipeline().
    // above. See that comment for the two-session data behind it; not repeated here.
    const u64 cbuf_key_c =
        gen_env    ? ComputeCbufKeyExcludingTextureHandles(gen_env->CapturedCbufValues(),
                                                           gen_env->CapturedTextureHandleCbufKeys())
        : file_env ? ComputeCbufKeyExcludingTextureHandles(file_env->CapturedCbufValues(),
                                                           file_env->CapturedTextureHandleCbufKeys())
                   : 0;
    const bool use_gen_logical_texture_key =
        gen_env && !HasActivePhase4LogicalTextureSlot(gen_env->CapturedLogicalTextureSlots()) &&
        HasCompleteLogicalTextureCoverage(gen_env->CapturedLogicalTextureSlots(),
                                          gen_env->CapturedLogicalTextureHandles(),
                                          gen_env->CapturedTextureTypes(),
                                          gen_env->CapturedTexturePixelFormats());
    const bool use_file_logical_texture_key =
        file_env && !HasActivePhase4LogicalTextureSlot(file_env->CapturedLogicalTextureSlots()) &&
        HasCompleteLogicalTextureCoverage(file_env->CapturedLogicalTextureSlots(),
                                          file_env->CapturedLogicalTextureHandles(),
                                          file_env->CapturedTextureTypes(),
                                          file_env->CapturedTexturePixelFormats());
    const u64 texture_key_c =
        use_gen_logical_texture_key    ? ComputeLogicalTextureKey(gen_env->CapturedLogicalTextureSlots())
        : use_file_logical_texture_key ? ComputeLogicalTextureKey(file_env->CapturedLogicalTextureSlots())
        : gen_env                      ? ComputeTextureKeyExcludingHandles(
                                             gen_env->CapturedTextureTypes(),
                                             gen_env->CapturedTexturePixelFormats(),
                                             gen_env->CapturedPhase4PrototypeHandles())
        : file_env                     ? ComputeTextureKey(file_env->CapturedTextureTypes(),
                                                             file_env->CapturedTexturePixelFormats())
                                     : 0;
    spirv_cache.RecordTextureKeyMode(use_gen_logical_texture_key ||
                                     use_file_logical_texture_key);
    // Was a second independent ComputeCbufKeyExcludingTextureHandles(...) call here —
    // see the matching comment in CreateGraphicsPipeline() above for why it's gone.
    const u64 diag_cbuf_key_excl_texture_handles_c = cbuf_key_c;
    // `key.unique_hash` came from ShaderCache::MakeShaderInfo(), which uses
    // GenericEnvironment::Analyze() and the shared ProgramIdentity span rule.
    // Do not recompute this through CalculateHash(): that tracks the CFG read
    // range, which can differ from the serialized program span (and from the
    // scanner) for the same compute shader. A second identity rule here made
    // otherwise-compatible scanner entries permanently unfindable by compute.
    const u64 compute_unique_hash = key.unique_hash;
    // Was hardcoded 0 — see ComputeWorkgroupKey's doc comment in spirv_cache.h for why
    // that was a real correctness bug (not just a cache-efficiency one): workgroup_size
    // and shared_memory_size get baked into the SPIR-V as literals, and neither was
    // previously part of this key at all, so two dispatches of the same compute program
    // at different launch dimensions could silently swap SPIR-V if cbuf_key/texture_key
    // happened to coincide. diag_base_runtime_hash is set to this same value (instead of
    // the usual 0 default) so the field-mismatch diagnostic can now show real data for
    // compute misses too, instead of a 0/0 default that could spuriously read as
    // "matches" against an unrelated real-zero-state graphics entry.
    const u64 workgroup_key = ComputeWorkgroupKey(key.shared_memory_size, key.workgroup_size);
    const SpirvKey spirv_key_c{compute_unique_hash, env.ShaderStage(), compiler_target_key,
                               cbuf_key_c, workgroup_key, texture_key_c};
    std::vector<u32> code;
    std::chrono::microseconds emit_time{};
    bool exact_module_hit = false;
    if (auto cached = spirv_cache.Lookup(spirv_key_c, has_real_specialization_context,
                                         workgroup_key, 0, diag_cbuf_key_excl_texture_handles_c)) {
        code = *cached->spirv;
        exact_module_hit = true;
        if (cached->is_speculative) {
            // See the matching throttle comment in CreateGraphicsPipeline() above.
            static std::atomic<size_t> speculative_hit_logs_compute{0};
            if (size_t expected = speculative_hit_logs_compute.load(); expected < 50 &&
                speculative_hit_logs_compute.compare_exchange_strong(expected, expected + 1)) {
                const char* source = cached->source == VideoCommon::SpirvCacheEntrySource::PrecacheScanner
                                         ? "scanner"
                                         : "live";
                LOG_INFO(Render_Vulkan,
                         "0x{:016x} served from {} SPECULATIVE entry (unique_hash=0x{:016x})",
                         hash, source, compute_unique_hash);
            }
        }
        // Compute pipelines are self-contained (no preceding stage to misalign),
        // so the stored end_binding is irrelevant here and intentionally ignored.
    } else {
        const auto emit_start = std::chrono::steady_clock::now();
        code = EmitSPIRV(profile, program);
        emit_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - emit_start);
        // Compute's EmitSPIRV overload takes no Bindings parameter (it always
        // starts descriptor allocation at zero), so store a default end_binding.
        spirv_cache.Insert(spirv_key_c, code, {}, /*is_speculative=*/false, workgroup_key, 0,
                          diag_cbuf_key_excl_texture_handles_c, {}, &program.info,
                          gen_env ? &gen_env->CapturedCbufValues() : nullptr,
                          gen_env && gen_env->CapturedTextureTypes().empty() &&
                              gen_env->CapturedTexturePixelFormats().empty() &&
                              gen_env->CapturedTextureHandleCbufKeys().empty());
        // Reserve extra capacity on the local upload copy only — the stored
        // cache entry was inserted before the reserve so it stays compact.
        code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
        if (!spirv_cache_filename.empty()) {
            serialization_thread.QueueWork([this] {
                spirv_cache.SaveThrottled(spirv_cache_filename); });
        }
        // Texture-slot feasibility instrumentation; see the matching comment at the
        // CreateGraphicsPipeline call site above.
        VideoCommon::GenericEnvironment::LogTextureSlotVarianceReportThrottled();
    }
    if constexpr (CFG_TEMPLATE_MODULE_SHADOW_VALIDATION_ENABLED) {
        if (cfg_shadow_template && gen_env != nullptr &&
            (scanner_cfg_replay || gen_env->CapturedCbufValues().empty())) {
            ValidateCfgTemplateComputeModule(cfg_template_key, env, *cfg_shadow_template,
                                             env.StartAddress(), program, code,
                                             scanner_cfg_replay ? &scanner_cfg_replay->key
                                                                    : nullptr);
        }
    }
    // Ensure the upload copy has enough capacity on the cache-hit path too.
    code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
    RecordShaderCompilerWork(cfg_time, template_time, finalize_time, emit_time, exact_module_hit);
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};
    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    const auto pipeline_create_start = std::chrono::steady_clock::now();
    auto pipeline = std::make_unique<ComputePipeline>(
        device, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue, thread_worker,
        statistics, &shader_notify, program.info, std::move(spv_module), this,
        !build_in_parallel);
    RecordPipelineCreateWork(true, std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - pipeline_create_start));
    return pipeline;

} catch (const vk::Exception& exception) {
    if (exception.GetResult() == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        LOG_ERROR(Render_Vulkan,
                  "Out of device memory during compute pipeline creation, attempting recovery");
        EvictOldPipelines();
        return nullptr;
    }
    throw;
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    std::ofstream file(filename, std::ios::binary);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    size_t cache_size = 0;
    std::vector<char> cache_data;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    file.write(cache_data.data(), cache_size);

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: {}",
                 Common::FS::PathToUTF8String(filename));

        return create_pipeline_cache(cache_size, cache_data.data());

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}


// ---------------------------------------------------------------------------
// GPL: speculative Maxwell->SPIR-V translation
// ---------------------------------------------------------------------------



void PipelineCache::SubmitSpeculativeShader(
        u64 unique_hash, std::vector<u64> maxwell_code,
        Shader::Stage stage, u32 local_memory_size,
        u32 shared_memory_size, std::array<u32, 3> workgroup_size,
        u32 start_address, u32 texture_bound,
        Shader::ProgramHeader sph,
        std::optional<RealStageStoresSnapshot> previous_stage_snapshot) {
    speculative_worker.QueueWork(
        [this, unique_hash, code = std::move(maxwell_code),
         stage, local_memory_size, shared_memory_size,
         workgroup_size, start_address, texture_bound, sph,
         previous_stage_snapshot = std::move(previous_stage_snapshot)]() mutable {
        const u64 compiler_target_key =
            VideoCommon::CompilerTargetFingerprint(precache_compiler_target);
        // Reuse persistent pools to avoid per-translation VirtualAlloc churn.
        spec_pools.ReleaseContents();

        const u32 cfg_start = start_address +
            ((stage == Shader::Stage::Compute)
                 ? 0u : static_cast<u32>(sizeof(Shader::ProgramHeader)));

        // SpeculativeShaderEnvironment::ReadViewportTransformState() always guesses 1
        // (see speculative_shader_environment.h) since the real value is GPU register
        // state, not something derivable from shader bytecode. This was briefly widened
        // to translate VertexB twice (guessing both 0 and 1) on the theory that it's a
        // genuinely enumerable 2-way fork worth paying for — measured across three full
        // TotK sessions with 2000+ speculative entries sitting in the cache, it added
        // 2x translate cost for VertexB and zero additional hits: cbuf_key is what's
        // actually gating speculative hits (a speculative entry only matches a real draw
        // with zero captured cbuf specialization, which most real shaders don't have —
        // see ComputeCbufKey's doc comment), and no amount of guessing on this other axis
        // moves that ceiling. Reverted to a single guess to stop paying for the second
        // translation. If cbuf specialization coverage improves later, this is the first
        // place to reconsider re-widening.
        try {
            VideoCommon::SpeculativeShaderEnvironment env{
                std::move(code), start_address, stage, local_memory_size,
                shared_memory_size, workgroup_size, texture_bound, sph,
                /*code_offset_in_program=*/0u};

            Shader::Maxwell::Flow::CFG cfg(env, spec_pools.flow_block, cfg_start, false);
            auto program = Shader::Maxwell::TranslateProgram(
                spec_pools.inst, spec_pools.block, env, cfg, host_info);

            const Shader::Backend::Bindings starting_binding = previous_stage_snapshot
                ? previous_stage_snapshot->end_binding : Shader::Backend::Bindings{};
            Shader::Backend::Bindings binding = starting_binding;
            Shader::RuntimeInfo rt{};
            // Scanner runtime-state snapshot, GPL live-speculative path. Real previous-stage
            // data when we have it (previous_stage_snapshot — resolved in
            // OnNewShaderSeen() from real_stage_stores_by_hash, itself populated from
            // CreateGraphicsPipeline()'s own real per-stage translations earlier in
            // this same session; see that capture site's comment), the exact same
            // fields MakeRuntimeInfo() reads from a real previous_program. Falls back
            // to the conservative "no restriction" sentinel — for EVERY non-VertexB
            // stage now, not just Fragment as before this session's session: nothing
            // in this codebase's history suggested Geometry/TessControl/TessEval
            // getting an unset (all-zero, "stores nothing") default instead of the
            // same sentinel Fragment already got was ever a deliberate choice, and
            // an all-zero guess is no more likely to be right than all-ones for a
            // stage that has a real predecessor.
            if (stage != Shader::Stage::VertexB) {
                if (previous_stage_snapshot) {
                    rt.previous_stage_stores = previous_stage_snapshot->stores;
                    rt.previous_stage_legacy_stores_mapping =
                        previous_stage_snapshot->legacy_stores_mapping;
                    if (previous_stage_snapshot->is_geometry_passthrough) {
                        rt.previous_stage_stores.mask |= previous_stage_snapshot->passthrough.mask;
                    }
                } else {
                    rt.previous_stage_stores.mask.set();
                }
            }
            if (stage == Shader::Stage::Fragment) {
                rt.input_topology = Shader::InputTopology::Triangles;
            }
            // Deliberate runtime defaults (runtime_info.h) for the fields
            // this speculative path has no real per-draw signal for.
            rt.ApplySpeculativeDefaults(stage, program.info);
            if (stage != Shader::Stage::Compute) {
                Shader::Maxwell::ConvertLegacyToGeneric(program, rt);
            }
            const bool use_logical_texture_key =
                !HasActivePhase4LogicalTextureSlot(env.CapturedLogicalTextureSlots()) &&
                HasCompleteLogicalTextureCoverage(env.CapturedLogicalTextureSlots(),
                                                  env.CapturedLogicalTextureHandles(),
                                                  env.CapturedTextureTypes(),
                                                  env.CapturedTexturePixelFormats());
            const u64 texture_key = use_logical_texture_key
                                        ? ComputeLogicalTextureKey(env.CapturedLogicalTextureSlots())
                                        : ComputeTextureKey(env.CapturedTextureTypes(),
                                                            env.CapturedTexturePixelFormats());
            spirv_cache.RecordTextureKeyMode(use_logical_texture_key);
            // SpirvRelevantHash(stage), not Hash() — see the same swap and its
            // rationale in CreateGraphicsPipeline() above. Matters doubly here:
            // this is the SAME function that produces the actual translated SPIR-V
            // (rt is used for both), so restricting to stage-relevant fields also
            // means a wider set of real pipeline states can validly reuse this exact
            // guess, not just a wider match against the folded key.
            u64 runtime_key = stage == Shader::Stage::Compute
                                  ? ComputeWorkgroupKey(shared_memory_size, workgroup_size)
                                  : rt.SpirvRelevantHash(stage);
            if (stage == Shader::Stage::VertexB) {
                // Must match the fold CreateGraphicsPipeline() applies on the real
                // path — otherwise this entry's runtime_key is in a different
                // format from every real one and can never match, regardless of
                // how accurate the guess is (see FoldViewportTransformState's
                // doc comment in spirv_cache.h).
                runtime_key = FoldViewportTransformState(runtime_key, env.ReadViewportTransformState());
            }
            // See the matching diag_base_runtime_hash comment in CreateGraphicsPipeline() —
            // everything folded into runtime_key up to (not including) the binding fold
            // below, purely so a later stale miss against this entry can be attributed to
            // "core RuntimeInfo state never matched" vs "binding offset never matched".
            const u64 diag_base_runtime_hash = runtime_key;
            const auto diag_runtime_fields = rt.SpirvRelevantFieldHashes(stage);
            // `binding` is now the post-emit state; the cache key needs the
            // starting state that was baked into this module's descriptor numbers.
            const u64 binding_key = stage == Shader::Stage::Compute
                                        ? 0
                                        : ComputeBindingKey(starting_binding);
            if (stage != Shader::Stage::Compute) {
                runtime_key = FoldBindingKey(runtime_key, binding_key);
            }
            // A cache entry is only interchangeable when every component of
            // SpirvKey matches. Do not suppress this translation merely because
            // another context for the same raw shader was seen first.
            const SpirvKey speculative_key{unique_hash, stage, compiler_target_key, 0,
                                            runtime_key, texture_key};
            if (spirv_cache.Contains(speculative_key)) {
                return;
            }
            auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, rt, program, binding);
            spirv_cache.InsertSpeculative(unique_hash, stage, compiler_target_key, runtime_key,
                                          texture_key,
                                          std::move(spirv),
                                          binding, diag_base_runtime_hash, binding_key,
                                          diag_runtime_fields);
            if (!spirv_cache_filename.empty()) {
                serialization_thread.QueueWork([this] {
                    spirv_cache.SaveThrottled(spirv_cache_filename);
                });
            }
            // Texture-slot feasibility instrumentation; see the matching comment at the
            // CreateGraphicsPipeline call site above.
            VideoCommon::GenericEnvironment::LogTextureSlotVarianceReportThrottled();
        } catch (...) {}
    });
}

std::optional<PipelineCache::RealStageStoresSnapshot>
PipelineCache::ResolveRealStageStoresSnapshot(u64 previous_stage_unique_hash) const {
    if (previous_stage_unique_hash == 0) {
        return std::nullopt;
    }
    std::shared_lock lock{real_stage_stores_mutex};
    const auto it = real_stage_stores_by_hash.find(previous_stage_unique_hash);
    if (it == real_stage_stores_by_hash.end()) {
        return std::nullopt;
    }
    return it->second;
}

void PipelineCache::OnNewShaderSeen(VideoCommon::GenericEnvironment& env,
                                    u64 unique_hash, u64 previous_stage_unique_hash) {
    if (!Settings::values.use_gpl_speculative_shaders.GetValue()) return;
    if (env.ShaderStage() == Shader::Stage::VertexA) return;
    // Use CopyCode() rather than CachedSizeBytes() to obtain the shader binary.
    // When GenericEnvironment::Analyze() fails (TryFindSize returns nullopt), the
    // slow CFG path is taken and cached_lowest/cached_highest are left at their
    // sentinel defaults (UINT32_MAX / 0). Calling CachedSizeBytes() then produces
    // a wildly large value that causes std::bad_alloc when used as a vector size.
    // CopyCode() copies the `code` field directly — it is empty when Analyze() did
    // not run (fast path never set it), or populated with up to 1MB of instructions
    // when TryFindSize scanned without finding a self-branch. Either way, if the
    // code vector is empty we have nothing useful to speculatively translate.
    std::vector<u64> maxwell_code;
    env.CopyCode(maxwell_code);
    if (maxwell_code.empty()) return;

    // Sanity-check: reject unreasonably large blobs (> 256 KB of Maxwell instructions).
    // A legitimate shader is rarely over 64 KB; 256 KB gives ample headroom.
    constexpr size_t MAX_SPECULATIVE_WORDS = 256 * 1024 / sizeof(u64);
    if (maxwell_code.size() > MAX_SPECULATIVE_WORDS) return;

    SubmitSpeculativeShader(unique_hash, std::move(maxwell_code),
                            env.ShaderStage(), env.LocalMemorySize(),
                            env.SharedMemorySize(), env.WorkgroupSize(),
                            env.StartAddress(), env.TextureBoundBuffer(),
                            env.SPH(), ResolveRealStageStoresSnapshot(previous_stage_unique_hash));
}

} // namespace Vulkan
