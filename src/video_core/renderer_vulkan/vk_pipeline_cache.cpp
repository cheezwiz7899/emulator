// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <span>
#include <thread>
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
using VideoCommon::ComputeBindingKey;
using VideoCommon::ComputeWorkgroupKey;
using VideoCommon::FoldViewportTransformState;
using VideoCommon::FoldBindingKey;

// v16: GenericEnvironment::Serialize()/FileEnvironment::Deserialize() gained
// texture_handle_cbuf_keys (see CapturedTextureHandleCbufKeys() in
// shader_environment.h) so boot-time disk-replay entries can supply real
// cbuf/texture-handle narrowing data to the same diagnostic that live play
// already could — Phase 0 testing showed this gap wasn't cosmetic: sessions
// with a lot of preloaded content had an eligible-sample rate roughly 9x
// lower than a fresh-wipe session, since every stale miss sourced from a
// FileEnvironment was structurally excluded from the "would narrowing help"
// measurement. Old caches have no data for this field at all (not a partial-
// data situation — the bytes genuinely aren't there), hence the version bump
// rather than trying to read old files as if they had it.
constexpr u32 TRANSFERABLE_CACHE_VERSION = 16;
constexpr u32 VULKAN_PIPELINE_CACHE_VERSION = 14;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
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

// Phase 4 narrow prototype's graphics_cache lookup-timing fix. The problem this solves:
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

// Phase 3 groundwork — see this method's doc comment in vk_pipeline_cache.h for what it's
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
             "Phase 3 groundwork: of {} graphics shaders seen with a real, cbuf_key==0 draw "
             "so far, {} showed exactly 1 distinct core-RuntimeInfo state, {} showed 2-3, {} "
             "showed 4-7, {} showed 8+ (capped — true count may be higher). Diagnostic only, "
             "nothing currently acts on this. Low cardinality across most hashes would say "
             "scan-time multi-variant guessing (Phase 3) could plausibly help now that Phase "
             "1 narrowed cbuf_key; a lot of 8+ hashes would say this hits the same structural "
             "ceiling the removed second viewport-transform-state guess already found once, "
             "just now confirmed with cbuf's blocking accounted for.",
             total_hashes, buckets[0], buckets[1], buckets[2], buckets[3]);
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    // Phase 4 narrow prototype's graphics_cache lookup-timing fix -- must happen here, after
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

    // Phase 4 adaptive slot learning (handoff_09/handoff_10) -- must run before the SPIR-V
    // cache load below and before any shader translation this session, since
    // IsPhase4PrototypeSlot (environment.h), which texture_key computation depends on, reads
    // whatever table this publishes. Failure-safe: LoadPhase4PrototypeSlots returns {} on any
    // error, and MergePhase4PrototypeSlots({}) is just an empty table -- a missing/corrupt
    // file degrades to ordinary pre-Phase-4 behavior for every coordinate, same as a fresh
    // profile that's never hit this path before, rather than to a crash or a stale state.
    phase4_prototype_slots_filename = base_dir / "phase4_prototype_slots.bin";
    Shader::SetActivePhase4PrototypeSlots(Shader::MergePhase4PrototypeSlots(
        VideoCommon::LoadPhase4PrototypeSlots(phase4_prototype_slots_filename)));

    // Load SPIR-V cache — feeds the GPL speculative path and AOT scanner results.
    spirv_cache_filename = base_dir / "spirv_cache.bin";
    spirv_cache.Load(spirv_cache_filename);


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
        size_t invalid{};
        size_t feature_mismatch{};
    } state;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if (!env.HasValidEntryInstruction()) {
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

        if (!std::ranges::all_of(envs, &FileEnvironment::HasValidEntryInstruction)) {
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
    size_t env_index{0};
    std::array<Shader::IR::Program, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> programs;
    // Track env pointer per shader index so the emit loop can compute cbuf keys.
    std::array<Shader::Environment*, Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram> stage_envs{};
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

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
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
        }

        if (Settings::values.dump_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }

        // Phase 3 guess refinement, GPL live-speculative path: record this stage's
        // REAL translated data (same four fields MakeRuntimeInfo() below reads from a
        // real previous_program) keyed by its own unique_hash, so a LATER draw's
        // speculative guess for whichever stage follows this one (see
        // OnNewShaderSeen()/ResolveRealStageStoresSnapshot()) can use real data
        // instead of an invented sentinel. Overwrites any prior entry for this hash
        // with the most recent real observation — per Phase 3's own cardinality data
        // most shaders only ever show one real state anyway, so this is usually also
        // the only one, and even when it isn't, "most recent real state" is a better
        // starting guess than a sentinel that was never observed at all.
        {
            std::unique_lock real_stage_stores_lock{real_stage_stores_mutex};
            real_stage_stores_by_hash[key.unique_hashes[index]] = RealStageStoresSnapshot{
                programs[index].info.stores, programs[index].info.legacy_stores_mapping,
                programs[index].info.passthrough, programs[index].is_geometry_passthrough};
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

        // Phase 4 narrow prototype: this is the one place real Shader::Info exists for a
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
        // PHASE 1 — narrowing enabled. Validated across two independent full play
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
        const u64 texture_key =
            // Phase 4 narrow prototype's texture_key fix: exclude handles resolved for the
            // one hardcoded polymorphic slot, so its two real variants hash the same instead
            // of fragmenting the cache. FileEnvironment falls back to plain ComputeTextureKey
            // -- it doesn't derive from GenericEnvironment, so it has no
            // CapturedPhase4PrototypeHandles() to exclude with, same reasoning as the cbuf_key
            // narrowing diagnostic just below already applies to that path.
            gen_env_stage    ? ComputeTextureKeyExcludingHandles(
                                   gen_env_stage->CapturedTextureTypes(),
                                   gen_env_stage->CapturedTexturePixelFormats(),
                                   gen_env_stage->CapturedPhase4PrototypeHandles())
            : file_env_stage ? ComputeTextureKey(file_env_stage->CapturedTextureTypes(),
                                                 file_env_stage->CapturedTexturePixelFormats())
                             : 0;
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
        const u64 binding_key = ComputeBindingKey(binding);
        runtime_key = FoldBindingKey(runtime_key, binding_key);
        const SpirvKey spirv_key{key.unique_hashes[index], cbuf_key, runtime_key, texture_key};
        std::vector<u32> code;
        const bool is_merged_vertex = uses_vertex_a && uses_vertex_b && index == 1;
        // Since SpirvKey now includes the runtime_key, we can safely serve cached SPIR-V
        // to both the live path and the disk-load path.
        if (!is_merged_vertex) {
            if (auto cached = spirv_cache.Lookup(spirv_key, has_real_specialization_context,
                                                 diag_base_runtime_hash, binding_key,
                                                 diag_cbuf_key_excl_texture_handles)) {
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
                        LOG_INFO(Render_Vulkan, "0x{:016x} stage[{}] served from SPECULATIVE entry (unique_hash=0x{:016x})",
                                 hash, index, key.unique_hashes[index]);
                    }
                }
                // Restore the binding counter to where EmitSPIRV left it when this
                // SPIR-V was first compiled.  Without this, the next stage's
                // EmitSPIRV (or cache miss) would start at the wrong descriptor
                // slot, producing binding collisions and graphical corruption.
                binding = cached->end_binding;
            }
        }
        if (code.empty()) {
            code = EmitSPIRV(profile, runtime_info, program, binding);
            // binding has now been advanced past this stage's slots.
            // has_real_specialization_context (not just gen_env_stage != nullptr) so a
            // disk-replay translation gets cached too, now that it's keyed with real
            // cbuf/texture data instead of a forced 0 — previously this branch's
            // EmitSPIRV work was simply thrown away every time, guaranteeing this
            // exact stage would miss and re-translate again on every future load.
            if (!is_merged_vertex && has_real_specialization_context) {
                spirv_cache.Insert(spirv_key, code, binding, /*is_speculative=*/false,
                                  diag_base_runtime_hash, binding_key,
                                  diag_cbuf_key_excl_texture_handles);
                // Phase 3 groundwork — see RecordPhase3RuntimeVariantDiagnostic's doc
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
                if (!spirv_cache_filename.empty()) {
                    serialization_thread.QueueWork([this] { spirv_cache.SaveThrottled(spirv_cache_filename); });
                }
                // Phase 4 feasibility instrumentation — see LogTextureSlotVarianceReportThrottled's
                // doc comment in shader_environment.h. Unconditional (not gated on
                // spirv_cache_filename): unrelated to whether disk persistence is configured.
                VideoCommon::GenericEnvironment::LogTextureSlotVarianceReportThrottled();
            }
            code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
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
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, thread_worker, statistics, render_pass_cache, key,
        std::move(modules), infos);

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

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
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
    // PHASE 1 — narrowing enabled, same validated switch as CreateGraphicsPipeline()
    // above. See that comment for the two-session data behind it; not repeated here.
    const u64 cbuf_key_c =
        gen_env    ? ComputeCbufKeyExcludingTextureHandles(gen_env->CapturedCbufValues(),
                                                           gen_env->CapturedTextureHandleCbufKeys())
        : file_env ? ComputeCbufKeyExcludingTextureHandles(file_env->CapturedCbufValues(),
                                                           file_env->CapturedTextureHandleCbufKeys())
                   : 0;
    const u64 texture_key_c =
        // Phase 4 narrow prototype's texture_key fix — see the matching comment in
        // CreateGraphicsPipeline() above. This hardcoded slot is graphics-content-shaped
        // (a portal/particle effect), so this branch firing in practice is unlikely, but the
        // fix is applied here too for correctness rather than assuming compute never touches
        // it.
        gen_env    ? ComputeTextureKeyExcludingHandles(gen_env->CapturedTextureTypes(),
                                                       gen_env->CapturedTexturePixelFormats(),
                                                       gen_env->CapturedPhase4PrototypeHandles())
        : file_env ? ComputeTextureKey(file_env->CapturedTextureTypes(),
                                       file_env->CapturedTexturePixelFormats())
                   : 0;
    // Was a second independent ComputeCbufKeyExcludingTextureHandles(...) call here —
    // see the matching comment in CreateGraphicsPipeline() above for why it's gone.
    const u64 diag_cbuf_key_excl_texture_handles_c = cbuf_key_c;
    // Use gen_env->CalculateHash() — CalculateHash() is defined on GenericEnvironment,
    // not on the base Shader::Environment. Fall back to key.unique_hash if not available.
    const u64 compute_unique_hash = gen_env ? gen_env->CalculateHash() : key.unique_hash;
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
    const SpirvKey spirv_key_c{compute_unique_hash, cbuf_key_c, workgroup_key, texture_key_c};
    std::vector<u32> code;
    if (auto cached = spirv_cache.Lookup(spirv_key_c, has_real_specialization_context,
                                         workgroup_key, 0, diag_cbuf_key_excl_texture_handles_c)) {
        code = *cached->spirv;
        if (cached->is_speculative) {
            // See the matching throttle comment in CreateGraphicsPipeline() above.
            static std::atomic<size_t> speculative_hit_logs_compute{0};
            if (size_t expected = speculative_hit_logs_compute.load(); expected < 50 &&
                speculative_hit_logs_compute.compare_exchange_strong(expected, expected + 1)) {
                LOG_INFO(Render_Vulkan, "0x{:016x} served from SPECULATIVE entry (unique_hash=0x{:016x})",
                         hash, compute_unique_hash);
            }
        }
        // Compute pipelines are self-contained (no preceding stage to misalign),
        // so the stored end_binding is irrelevant here and intentionally ignored.
    } else {
        code = EmitSPIRV(profile, program);
        // Compute's EmitSPIRV overload takes no Bindings parameter (it always
        // starts descriptor allocation at zero), so store a default end_binding.
        spirv_cache.Insert(spirv_key_c, code, {}, /*is_speculative=*/false, workgroup_key, 0,
                          diag_cbuf_key_excl_texture_handles_c);
        // Reserve extra capacity on the local upload copy only — the stored
        // cache entry was inserted before the reserve so it stays compact.
        code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
        if (!spirv_cache_filename.empty()) {
            serialization_thread.QueueWork([this] {
                spirv_cache.SaveThrottled(spirv_cache_filename); });
        }
        // Phase 4 feasibility instrumentation — see the matching comment at the
        // CreateGraphicsPipeline call site above.
        VideoCommon::GenericEnvironment::LogTextureSlotVarianceReportThrottled();
    }
    // Ensure the upload copy has enough capacity on the cache-hit path too.
    code.reserve(std::max<size_t>(code.size(), 16 * 1024 / sizeof(u32)));
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};
    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<ComputePipeline>(device, vulkan_pipeline_cache, descriptor_pool,
                                             guest_descriptor_queue, thread_worker, statistics,
                                             &shader_notify, program.info, std::move(spv_module));

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
    if (spirv_cache.ContainsByUniqueHash(unique_hash)) return;

    speculative_worker.QueueWork(
        [this, unique_hash, code = std::move(maxwell_code),
         stage, local_memory_size, shared_memory_size,
         workgroup_size, start_address, texture_bound, sph,
         previous_stage_snapshot = std::move(previous_stage_snapshot)]() mutable {
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

            Shader::Backend::Bindings binding{};
            Shader::RuntimeInfo rt{};
            // Phase 3 guess refinement, GPL live-speculative path. Real previous-stage
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
            if (stage != Shader::Stage::Compute) {
                Shader::Maxwell::ConvertLegacyToGeneric(program, rt);
            }
            auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, rt, program, binding);
            const u64 texture_key = ComputeTextureKey(env.CapturedTextureTypes(),
                                                       env.CapturedTexturePixelFormats());
            // Phase 4 narrow prototype's texture_key fix is NOT applied here -- confirmed by
            // an actual build (not assumed, and an earlier version of this comment wrongly
            // assumed otherwise): `env` in this function is VideoCommon::
            // SpeculativeShaderEnvironment, which does not derive from GenericEnvironment and
            // has no CapturedPhase4PrototypeHandles() to exclude with. Same treatment as the
            // FileEnvironment branches elsewhere in this file -- reduced-capability path, no
            // exclusion applied, plain ComputeTextureKey.
            // SpirvRelevantHash(stage), not Hash() — see the same swap and its
            // rationale in CreateGraphicsPipeline() above. Matters doubly here:
            // this is the SAME function that produces the actual translated SPIR-V
            // (rt is used for both), so restricting to stage-relevant fields also
            // means a wider set of real pipeline states can validly reuse this exact
            // guess, not just a wider match against the folded key.
            u64 runtime_key = rt.SpirvRelevantHash(stage);
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
            // Must match the fold applied on the live path in CreateGraphicsPipeline().
            // Speculative compiles always start from an all-zero Bindings
            // accumulator (see `binding{}` above) — NOT the post-EmitSPIRV
            // `binding`, which has since been advanced past this stage's slots.
            // Using a fresh zero value here is what actually matches what was
            // baked into `spirv`, and what lets a genuinely-leading-stage real
            // draw hit this entry.
            const u64 binding_key = ComputeBindingKey(Shader::Backend::Bindings{});
            runtime_key = FoldBindingKey(runtime_key, binding_key);
            spirv_cache.InsertSpeculative(unique_hash, runtime_key, texture_key, std::move(spirv),
                                          diag_base_runtime_hash, binding_key);
            if (!spirv_cache_filename.empty()) {
                serialization_thread.QueueWork([this] {
                    spirv_cache.SaveThrottled(spirv_cache_filename);
                });
            }
            // Phase 4 feasibility instrumentation — see the matching comment at the
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
    if (spirv_cache.ContainsByUniqueHash(unique_hash)) return;

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
